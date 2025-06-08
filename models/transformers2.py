# Adapted from: https://github.com/samholt/DeepGenerativeSymbolicRegression
# Original author(s): Samuel Holt, Zhaozhi Qian, Mihaela van der Schaar

import logging
import math

import numpy as np
import torch
import torch.nn.functional
from dso.memory import Batch
from dso.prior import LengthConstraint
from dso.state_manager import TorchHierarchicalStateManager
from dso.utils import log_and_print
from torch import nn
from torch.autograd import Function

logger = logging.getLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(DEVICE)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def safe_cross_entropy(p, logq, dim=-1):
    safe_logq = torch.where(p == 0, torch.ones_like(logq).to(DEVICE), logq)
    return -torch.sum(p * safe_logq, dim)


def numpy_batch_to_tensor_batch(batch):
    if batch is None:
        return None
    else:
        return Batch(
            actions=torch.from_numpy(batch.actions).to(DEVICE),
            obs=torch.from_numpy(batch.obs).to(DEVICE),
            priors=torch.from_numpy(batch.priors).to(DEVICE),
            lengths=torch.from_numpy(batch.lengths).to(DEVICE),
            rewards=torch.from_numpy(batch.rewards).to(DEVICE),
            on_policy=torch.from_numpy(batch.on_policy).to(DEVICE),
            data_to_encode=torch.from_numpy(batch.data_to_encode).to(DEVICE),
            tgt=torch.from_numpy(batch.tgt).to(DEVICE),
        )


# pylint: disable-next=abstract-method
class GetNextObs(Function):
    @staticmethod
    # pylint: disable-next=arguments-differ
    def forward(ctx, obj, actions, obs):
        np_actions = actions.detach().cpu().numpy()
        np_obs = obs.detach().cpu().numpy()
        next_obs, next_prior = obj.task.get_next_obs(np_actions, np_obs)
        return obs.new(next_obs), obs.new(next_prior)

    @staticmethod
    # pylint: disable-next=arguments-differ
    def backward(ctx, grad_output):
        return grad_output


class PositionalEncoding(nn.Module):
    # Also works for non-even dimensions
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if (d_model % 2) == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.scale * self.pe[: x.size(0), :]  # pyright: ignore
        return self.dropout(x)

## Base model used in Symbolic Transformer 
class TransformerCustomEncoderModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden,
        cfg,
        input_pdt=None,
        output_pdt=None,
        enc_layers=3,
        dec_layers=1,
        dropout=0.1,
        input_already_encoded=True,
        max_len=1024,
        has_encoder=True,
        vocab_size = None,
        dynamics_embedding_size = None,
        dynamics_dimension = None
    ):
        super(TransformerCustomEncoderModel, self).__init__()
        self.has_encoder = has_encoder
        self.input_pdt = input_pdt
        self.output_pdt = output_pdt
        self.input_already_encoded = input_already_encoded
        self.out_dim = out_dim
        self.in_dim = in_dim
        if not self.input_already_encoded:
            self.encoder = nn.Embedding(in_dim, hidden)
        else:
            log_and_print(f"Transformer overwriting hidden size to input_dim {in_dim}")
            hidden = in_dim
        self.decoder = nn.Embedding(out_dim, hidden)
        self.pos_encoder = PositionalEncoding(in_dim, dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(hidden, dropout, max_len=max_len)

        self.dym_encoder = nn.Embedding(vocab_size, hidden)
        self.hidden = hidden
        nhead = int(np.ceil(hidden / 64))
        if (hidden % nhead) != 0:
            nhead = 1
        self.nhead = nhead
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden * 8,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(hidden, out_dim)

        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None
        cfg["num_heads"] = nhead
        cfg["dim_hidden"] = hidden
        cfg["num_features"] = 1
        cfg["linear"] = True
        cfg["bit16"] = False
        cfg["n_l_enc"] = 3
        cfg["num_inds"] = 64
        log_and_print(f"Encoder params: {cfg}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden * 8, batch_first=True)
        self.dym_embedded_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dym_reduce_dim = nn.Linear(dynamics_embedding_size, dynamics_dimension*2)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(
        self,
        inp,
        input=False,  # pylint: disable=redefined-builtin
    ):
        if input:
            return (inp == self.input_pdt).transpose(0, 1)
        else:
            return (inp == self.output_pdt).transpose(0, 1)

    def forward(self, data_src, src, tgt):
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            self.tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        src_pad_mask = None
        if self.output_pdt:
            tgt_pad_mask = self.make_len_mask(tgt)

        tgt = self.decoder(tgt)
        tgt = self.pos_decoder(tgt)

        ## Encode hierarchical information
        if src is not None:
            if not self.input_already_encoded:
                src = self.encoder(src)
            src = self.pos_encoder(src)
            memory = self.transformer.encoder(src, mask=self.src_mask, src_key_padding_mask=src_pad_mask)
        
        ## Encoder dynamics information and concat with hierarchical information as a latent vector
        if data_src is not None and data_src.nelement() != 0:
            if self.has_encoder:
                data_memory = self.dym_encoder(data_src)
                data_memory = self.pos_encoder(data_memory)
                data_memory = self.dym_embedded_encoder(data_memory)
                data_memory = (self.dym_reduce_dim(data_memory.permute(0, 2, 1))).permute(0, 2, 1)
            else:
                data_memory = torch.zeros(memory.shape[1], 1, memory.shape[2]).to(memory.device)  # pyright: ignore
            assert not torch.isnan(data_memory).any()
            if src is not None:
                memory = torch.cat([data_memory.permute(1, 0, 2), memory], axis=0)  # pyright: ignore
            else:
                memory = data_memory.permute(1, 0, 2)
        
        ## Generate probabiliy distribution over all symbolic tokens for the next token selection
        ## given latent information from encoders (hierarchical info + dynamical info)
        output = self.transformer.decoder(
            tgt,
            memory,  # pyright: ignore
            tgt_mask=self.tgt_mask,
            memory_mask=self.memory_mask,
            tgt_key_padding_mask=tgt_pad_mask,  # pyright: ignore
            memory_key_padding_mask=src_pad_mask,
        )
        output = self.fc_out(output)

        return output



# pylint: disable-next=abstract-method
class TransformerTreeEncoderController(nn.Module):
    """
    Symbolic Transformer used to generate expressions.

    Specifically, the symbolic transformer outputs a distribution over pre-order traversals of
    symbolic expression trees. It is trained using risk-seeking policy gradient with baseline.

    Parameters
    ----------
    prior : dso.prior.JointPrior
        JointPrior object used to adjust probabilities during sampling.

    state_manager: dso.state_manager.StateManager
        Object that handles the state features to be used

    summary : bool
        Write tensorboard summaries?

    debug : int
        Debug level, also used in learn(). 0: No debug. 1: logger.info shapes and
        number of parameters for each variable.

    num_layers : int
        Number of multi-head attention.

    num_units : int or list of ints
        Embedding dimension of each multi-head attention.

    initializer : str
        Initializer for the transformer parameters. Supports 'zeros' and 'var_scale'.

    optimizer : str
        Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

    learning_rate : float
        Learning rate for optimizer.

    entropy_weight : float
        Coefficient for entropy bonus.

    entropy_gamma : float or None
        Gamma in entropy decay. None (or
        equivalently, 1.0) turns off entropy decay.

    pqt : bool
        Train with priority queue training (PQT)?

    pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?

    max_length : int or None
        Maximum sequence length. This will be overridden if a LengthConstraint
        with a maximum length is part of the prior.

    batch_size: int
        Most likely 500 or 1000
    """

    def __init__(
        self,
        prior,
        library,
        task,
        cfg,
        config_state_manager=None,
        encoder_input_dim=None,
        debug=0,
        summary=False,
        # multi-head attention hyperparameters
        cell="transformer",
        num_layers=1,
        num_units=128,
        # Optimizer hyperparameters
        optimizer="adam",
        initializer="zeros",
        learning_rate=0.001,
        # Loss hyperparameters
        entropy_weight=0.005,
        entropy_gamma=1.0,
        # PQT hyperparameters
        pqt=False,
        pqt_k=10,
        pqt_batch_size=1,
        pqt_weight=200.0,
        pqt_use_pg=False,
        # Other hyperparameters
        max_length=30,
        batch_size=1000,
        n_objects=1,
        has_encoder=True,
        rl_weight=1.0,
        randomize_ce=False,
        vocab_size = None,
        dynamics_embedding_size = None,
        dynamics_dimension = None
    ):
        super(TransformerTreeEncoderController, self).__init__()
        self.encoder_input_dim = encoder_input_dim
        self.learning_rate = learning_rate
        self.rl_weight = rl_weight
        self.randomize_ce = randomize_ce

        self.prior = prior
        self.summary = summary
        self.n_objects = n_objects
        self.num_units = num_units

        lib = library

        # Find max_length from the LengthConstraint prior, if it exists
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if " "there is no LengthConstraint."
            self.max_length = max_length
            logger.info(
                "WARNING: Maximum length not constrained. Sequences will "
                "stop at {} and complete by repeating the first input "
                "variable.".format(self.max_length)
            )
        elif max_length is not None and max_length != self.max_length:
            logger.info(
                "WARNING: max_length ({}) will be overridden by value from "
                "LengthConstraint ({}).".format(max_length, self.max_length)
            )
        self.max_length *= self.n_objects
        max_length = self.max_length

        # Hyperparameters
        self.entropy_weight = entropy_weight
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg

        self.n_choices = lib.L

        self.batch_size = batch_size
        # self.baseline = torch.zeros(1)

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        # Could make this a tensor
        self.entropy_gamma_decay = torch.Tensor([entropy_gamma**t for t in range(max_length)]).to(DEVICE)

       ## Build multi-head attention parameters
        if isinstance(num_units, int):
            num_units = num_units * num_layers

        if "type" in config_state_manager:  # pyright: ignore
            del config_state_manager["type"]  # pyright: ignore
        self.state_manager = TorchHierarchicalStateManager(
            library,
            max_length,
            **config_state_manager,  # pyright: ignore
        )

        # Calculate input size
        self.input_dim_size = self.state_manager.input_dim_size

        # (input_size, hidden_size)

        self.task = task


        self.tgt_padding_token = self.n_choices
        self.sos_token = self.n_choices + 1
        out_dim = self.n_choices + 2  # EOS token at last index
        self.out_dim = out_dim

        self.model = TransformerCustomEncoderModel(
            self.input_dim_size,
            self.out_dim,
            num_units,
            cfg,
            enc_layers=3,
            dec_layers=6,
            dropout=0,
            input_already_encoded=True,
            output_pdt=self.tgt_padding_token,
            has_encoder=has_encoder,
            vocab_size = vocab_size,
            dynamics_embedding_size = dynamics_embedding_size,
            dynamics_dimension = dynamics_dimension
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tgt_padding_token)

    def _sample(self, batch_size=None, input_=None):
        if batch_size is None:
            batch_size = self.batch_size
            self.batch_size = batch_size
        initial_obs = self.task.reset_task(self.prior)
        initial_obs = initial_obs.expand(batch_size, initial_obs.shape[0])
        initial_obs = self.state_manager.process_state(initial_obs)

        # Get initial prior
        initial_prior = torch.from_numpy(self.prior.initial_prior()).to(DEVICE)
        initial_prior = initial_prior.expand(batch_size, self.n_choices)

        # Returns transformer emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        current_length = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)

        inputs = next_input.unsqueeze(0)
        obs_ta = []
        priors_ta = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        # Initial state
        while not all(finished):
            current_length += 1
            output = self.model(input_, inputs, tgt_actions)
            cell_output = output[-1, :, :-2]
            logits = cell_output + prior
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            tgt_actions = torch.cat((tgt_actions, action.view(1, -1)), 0)

            # Compute obs and prior
            next_obs, next_prior = GetNextObs.apply(self, tgt_actions[1:, :].permute(1, 0), obs)  # pyright: ignore
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            inputs = torch.cat((inputs, next_input.unsqueeze(0)), 0)

            obs_ta.append(obs)
            priors_ta.append(prior)

            finished = finished + (current_length >= self.max_length)
            next_lengths = torch.where(finished, lengths, (current_length + 1).expand(batch_size))  # Ever finished

            obs = next_obs
            prior = next_prior
            lengths = next_lengths

        actions = tgt_actions[1:, :].permute(1, 0)
        # (?, obs_dim, max_length)
        obs = torch.stack(obs_ta).permute(1, 2, 0)
        # (?, max_length, n_choices)
        priors = torch.stack(priors_ta, 1)
        return actions, obs, priors

    def compute_neg_log_likelihood(self, data_to_encode, true_action, B=None):
        inputs_ = None
        batch_size = 1
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int64).to(DEVICE) * self.sos_token
        actions = torch.tensor(true_action).to(DEVICE).view(1, -1)
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(data_to_encode, inputs_, tgt_actions[:-1,])
        logits = outputs[:, :, :-2]
        neg_log_likelihood = self.ce_loss(logits.permute(1, 2, 0), tgt_actions[1:, :].T)
        return neg_log_likelihood

    def train_mle_loss(self, input_, token_eqs, actions, rewards):
        batch_size = input_.shape[0]
        initial_obs = self.task.reset_task(self.prior)
        initial_obs = initial_obs.expand(batch_size, initial_obs.shape[0])
        initial_obs = self.state_manager.process_state(initial_obs)
        token_eqs = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        actions = actions.to(torch.int32).to(DEVICE)
        token_eqs = torch.cat((token_eqs, actions.permute(1,0)), 0)
        inverse_prop = (rewards).to(torch.float32).to(DEVICE) ** 2

        # Get initial prior
        initial_prior = torch.from_numpy(self.prior.initial_prior()).to(DEVICE)
        initial_prior = initial_prior.expand(batch_size, self.n_choices)

        # Returns transformer emit outputs TensorArray (i.e. logits), final cell state, and final loop state
        current_length = torch.tensor(0, dtype=torch.int32).to(DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(DEVICE)
        obs = initial_obs
        next_input = self.state_manager.get_tensor_input(obs)
        # Could add a start token for inputs - none at present
        inputs = next_input.unsqueeze(0)
        obs_ta = []
        priors_ta = []
        step_losses = []
        prior = initial_prior
        lengths = torch.ones(batch_size, dtype=torch.int32).to(DEVICE)
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        # Initial state
        while not all(finished):
            current_length += 1
            if self.randomize_ce:
                use_ground_truth = (
                    (torch.rand(batch_size) > 0.25).to(DEVICE).long()
                )  # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
                tgt_actions = use_ground_truth * token_eqs[:current_length, :] + (1 - use_ground_truth) * tgt_actions
            else:
                tgt_actions = token_eqs[:current_length, :]
            output = self.model(input_, inputs, tgt_actions)
            cell_output = output[-1, :, :-2]
            logits = cell_output + prior
            logits[logits == float("-inf")] = 0
            step_losses.append((self.ce_loss(logits, token_eqs[current_length, :].long())) * inverse_prop)
            action = torch.distributions.categorical.Categorical(logits=logits).sample()
            tgt_actions = torch.cat((tgt_actions, action.view(1, -1)), 0)

            # Compute obs and prior
            next_obs, next_prior = GetNextObs.apply(self, tgt_actions[1:, :].permute(1, 0), obs)  # pyright: ignore
            next_obs = self.state_manager.process_state(next_obs)
            next_input = self.state_manager.get_tensor_input(next_obs)
            inputs = torch.cat((inputs, next_input.unsqueeze(0)), 0)

            obs_ta.append(obs)
            priors_ta.append(prior)
        
            finished = finished + ((current_length >= self.max_length) or (current_length == (token_eqs.shape[0] - 1)))
            next_lengths = torch.where(finished, lengths, (current_length + 1).expand(batch_size))  # Ever finished
            
            obs = next_obs
            prior = next_prior
            lengths = next_lengths
        mle_loss = torch.stack(step_losses).mean()
        tgt_actions=lengths=prior=obs=finished=token_eqs=output=None
        return mle_loss

    def make_neglogp_and_entropy(self, B, test=False):
        # Generates tensor for neglogp of a given batch
        # Loop_fn is defined in the function:
        # Essentially only calculating up to the sequence_lengths given:
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/raw_rnn
        inputs = self.state_manager.get_tensor_input(B.obs)
        sequence_length = B.lengths  # pylint: disable=unused-variable  # noqa: F841
        batch_size = B.obs.shape[0]
        data_to_encode = B.data_to_encode
        tgt_actions = torch.ones((1, batch_size), dtype=torch.int32).to(DEVICE) * self.sos_token
        actions = B.actions
        tgt_actions = torch.cat((tgt_actions, actions.permute(1, 0)), 0)

        outputs = self.model(data_to_encode, inputs.permute(1, 0, 2).float(), tgt_actions[:-1,])
        logits = outputs[:, :, :-2].permute(1, 0, 2)
        logits += B.priors
        probs = torch.nn.Softmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(probs, (-1,)))):
            raise ValueError
            
        logprobs = torch.nn.LogSoftmax(dim=2)(logits)
        if any(torch.isnan(torch.reshape(logprobs, (-1,)))):
            raise ValueError

        # Generate mask from sequence lengths
        # NOTE: Using this mask for neglogp and entropy actually does NOT
        # affect training because gradients are zero outside the lengths.
        # However, the mask makes tensorflow summaries accurate.
        mask = sequence_mask(B.lengths, maxlen=self.max_length, dtype=torch.float32)

        # Negative log probabilities of sequences
        actions_one_hot = torch.nn.functional.one_hot(B.actions.to(torch.long), num_classes=self.n_choices)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, dim=2)  # Sum over action dim

        neglogp = torch.sum(neglogp_per_step * mask, dim=1)  # Sum over current_length dim

        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        entropy_gamma_decay_mask = self.entropy_gamma_decay * mask  # ->(batch_size, max_length)
        # Sum over action dim -> (batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        # Sum over current_length dim -> (batch_size, )
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        tgt_actions, outputs = None, None

        return neglogp, entropy

    def _train_loss(self, b, sampled_batch_ph, pqt_batch_ph=None, test=False):

        if self.rl_weight == 0:
            mle_loss = self.train_mle_loss(sampled_batch_ph.data_to_encode, sampled_batch_ph.tgt, sampled_batch_ph.actions, r)
            total_loss =  mle_loss
            mle_loss_out = mle_loss.item()
            return total_loss, mle_loss_out

        # Setup losses
        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch_ph, test=test)
        r = sampled_batch_ph.rewards

        # Entropy loss
        entropy_loss = -self.entropy_weight * torch.mean(entropy)
        loss = entropy_loss

        if not self.pqt or (self.pqt and self.pqt_use_pg):
            # Baseline is the worst of the current samples r
            pg_loss = torch.mean((r - b) * neglogp)
            # Loss already is set to entropy loss
            loss += pg_loss

        # Priority queue training loss
        if self.pqt:
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch_ph, test=test)
            pqt_loss = self.pqt_weight * torch.mean(pqt_neglogp)
            loss += pqt_loss

        if self.rl_weight != 1.0 and sampled_batch_ph.tgt.size != 0:
            mle_loss = self.train_mle_loss(sampled_batch_ph.data_to_encode, sampled_batch_ph.tgt, sampled_batch_ph.actions,r)
            total_loss = self.rl_weight * loss + (1 - self.rl_weight) * mle_loss
            mle_loss_out = mle_loss.item()
        else:
            total_loss = loss
            mle_loss_out = None
        return total_loss, mle_loss_out

    def _compute_probs(self, memory_batch_ph, log=False):
        # Memory batch
        memory_neglogp, _ = self.make_neglogp_and_entropy(memory_batch_ph)
        if log:
            return -memory_neglogp
        else:
            return torch.exp(-memory_neglogp)

    def sample(self, n, input_=None):
        """Sample batch of n expressions"""

        actions, obs, priors = self._sample(n, input_=input_)
        return actions.cpu().numpy(), obs.cpu().numpy(), priors.cpu().numpy()

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""
        probs = self._compute_probs(numpy_batch_to_tensor_batch(memory_batch), log=log)
        return probs.cpu().numpy()

    def train_loss(self, b, sampled_batch, pqt_batch, test=False):
        """Computes loss, trains model, and returns mle_loss if not None."""
        loss, mle_loss = self._train_loss(
            torch.tensor(b).to(DEVICE),
            numpy_batch_to_tensor_batch(sampled_batch),
            numpy_batch_to_tensor_batch(pqt_batch),
            test=test,
        )
        return loss, mle_loss


