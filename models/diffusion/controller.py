"""DiffusionController — config-selectable discrete-diffusion generator.

Drop-in alternative to `TransformerTreeEncoderController`. Honors the same controller
contract used by `utils/train.py`:
  * sample(n, input_) -> (actions, obs, priors)  numpy, shapes (n,L)/(n,4,L)/(n,L,n_choices)
  * train_loss(b, sampled_batch, pqt_batch, test) -> (loss, summaries)   differentiable
plus attributes max_length / tgt_padding_token / pqt* / rl_weight / learning_rate / task / prior.

RL objective (design §8): FKL = advantage-softmax-weighted denoising ELBO; the GP
expert-guidance update reuses the SAME train_loss with a small rl_weight, which blends
the FKL term with an unweighted MLE (= ELBO on the GP solutions).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dso.prior import LengthConstraint

from .grammar import Grammar
from .denoiser import MaskedDiffusionDenoiser
from .sampler import (diffusion_sample, diffusion_sample_random,
                      diffusion_sample_blocksemiar, diffusion_sample_opblock)


class DiffusionController(nn.Module):
    def __init__(self, prior, library, task, cfg=None, config_state_manager=None,
                 max_length=30, pqt=False, pqt_k=10, pqt_batch_size=1,
                 rl_weight=1.0, learning_rate=1e-3, entropy_weight=0.005,
                 vocab_size=None, lam=0.5, max_const=10,
                 d_model=128, nlayers=4, nhead=4, sampler_order="blocksemiar",
                 num_diffusion_steps=12, reveal_schedule="cosine", block_size=4, **kwargs):
        super().__init__()
        self.prior = prior
        self.task = task
        self.library = library
        self.n_choices = int(library.L)
        self.tgt_padding_token = self.n_choices
        self.rl_weight = rl_weight
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.pqt = pqt
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.lam = float(lam)
        self.sampler_order = str(sampler_order)          # "preorder" (semi-AR) | "random" (Phase-3 any-order) | "blocksemiar" | "opblock"
        self.num_diffusion_steps = int(num_diffusion_steps) if num_diffusion_steps else 12
        self.reveal_schedule = str(reveal_schedule)
        self.block_size = int(block_size)
        self.save_true_log_likelihood = False
        self.true_eq = []

        # max_length from the LengthConstraint prior (mirrors the transformer controller)
        self.max_length = max_length
        for p in prior.priors:
            if isinstance(p, LengthConstraint) and p.max is not None:
                self.max_length = int(p.max)
                break

        root_token = None
        for p in prior.priors:
            if p.__class__.__name__ == "RootConstraint":
                root_token = int(p.token)
        self.grammar = Grammar(library, max_const=max_const, max_len=self.max_length,
                               root_token=root_token)
        assert vocab_size is not None, "vocab_size (dynamics prefix vocab) is required"
        self.denoiser = MaskedDiffusionDenoiser(
            self.n_choices, self.max_length, dyn_vocab_size=vocab_size,
            d_model=d_model, nlayers=nlayers, nhead=nhead)

    @property
    def device(self):
        return next(self.parameters()).device

    # ---- sampling --------------------------------------------------------------
    def sample(self, n, input_=None):
        dyn = self._as_dyn(input_, n)
        if self.sampler_order == "random":
            seqs = diffusion_sample_random(self.denoiser, self.grammar, n, self.max_length,
                                           dyn, device=self.device)
        elif self.sampler_order == "blocksemiar":
            seqs = diffusion_sample_blocksemiar(
                self.denoiser, self.grammar, n, self.max_length, dyn,
                device=self.device, block_size=self.block_size,
                schedule=self.reveal_schedule)
        elif self.sampler_order == "opblock":
            seqs = diffusion_sample_opblock(
                self.denoiser, self.grammar, n, self.max_length, dyn,
                device=self.device, block_size=self.block_size,
                schedule=self.reveal_schedule)
        else:
            seqs = diffusion_sample(self.denoiser, self.grammar, n, self.max_length,
                                    dyn, device=self.device)
        term = int(self.grammar.terminal_tokens[0])
        actions = np.full((n, self.max_length), term, dtype=np.int32)
        for i, toks in enumerate(seqs):
            k = min(len(toks), self.max_length)
            actions[i, :k] = np.asarray(toks[:k], dtype=np.int32)
        obs, priors = self._obs_priors(actions)
        return actions, obs, priors

    def _as_dyn(self, input_, n):
        if input_ is None:                       # unconditional fallback (shouldn't happen in pipeline)
            return torch.zeros((n, 1), dtype=torch.long, device=self.device)
        dyn = input_ if torch.is_tensor(input_) else torch.as_tensor(input_)
        dyn = dyn.long().to(self.device)
        if dyn.dim() == 1:
            dyn = dyn.unsqueeze(0)
        if dyn.shape[0] != n:                    # tile a single dynamics row to the batch
            dyn = dyn[-1:].repeat(n, 1)
        return dyn

    def _obs_priors(self, actions):
        """Shape-compatible obs (n,4,L) and priors (n,L,n_choices). Values are unused by
        the diffusion loss; priors=0 (no masking) and obs carries (action,parent,sibling,dangling)."""
        n, L = actions.shape
        EMPTY_P = getattr(self.library, "EMPTY_PARENT", 0)
        EMPTY_S = getattr(self.library, "EMPTY_SIBLING", 0)
        parent = np.full((n, L), EMPTY_P, dtype=np.float32)
        sibling = np.full((n, L), EMPTY_S, dtype=np.float32)
        dangling = np.zeros((n, L), dtype=np.float32)
        ar = self.grammar.arities
        for i in range(n):
            slots = 1
            for t in range(L):
                dangling[i, t] = slots
                slots += int(ar[actions[i, t]]) - 1 if actions[i, t] < self.n_choices else -1
                if slots <= 0:
                    slots = 0
        obs = np.stack([actions.astype(np.float32), parent, sibling, dangling], axis=1)
        priors = np.zeros((n, L, self.n_choices), dtype=np.float32)
        return obs.astype(np.float32), priors

    # ---- training --------------------------------------------------------------
    def _denoise_ce(self, actions, lengths, dyn):
        """Per-sample masked denoising cross-entropy (surrogate -log p / ELBO term)."""
        B, L = actions.shape
        dev = self.device
        t = torch.rand(B, device=dev).clamp(0.05, 1.0)
        within = torch.arange(L, device=dev).unsqueeze(0) < lengths.clamp(min=1).unsqueeze(1)
        masked = (torch.rand(B, L, device=dev) < t.unsqueeze(1)) & within
        # guarantee >=1 masked position per sample
        none = masked.sum(1) == 0
        if none.any():
            first = within.float().argmax(1)
            masked[none, first[none]] = True
        inp = actions.clone()
        inp[masked] = self.denoiser.mask_id
        logits = self.denoiser(inp, t, dyn)                                   # (B,L,n_choices)
        ce = F.cross_entropy(logits.reshape(-1, self.n_choices),
                             actions.reshape(-1), reduction="none").reshape(B, L)
        ce = ce * masked.float()
        denom = masked.float().sum(1).clamp(min=1.0)

        # Entropy bonus (design BACKLOG "entbonus"): encourage high denoiser entropy at
        # masked positions whose true token is an operator (arity>0), the positions the
        # Flexibility-Trap mechanism collapses onto first when RL narrows the policy.
        arities = torch.as_tensor(self.grammar.arities, device=dev, dtype=torch.long)
        is_operator = arities[actions.clamp(0, self.n_choices - 1)] > 0
        op_mask = masked & is_operator
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)             # (B,L)
        op_denom = op_mask.float().sum(1).clamp(min=1.0)
        entropy_term = (ent * op_mask.float()).sum(1) / op_denom              # (B,) mean entropy at op positions

        return (ce.sum(1) / denom) / t, entropy_term                         # (B,), (B,)

    def train_loss(self, b, sampled_batch, pqt_batch=None, test=False):
        dev = self.device
        actions = torch.as_tensor(np.asarray(sampled_batch.actions), dtype=torch.long, device=dev)
        lengths = torch.as_tensor(np.asarray(sampled_batch.lengths), dtype=torch.long, device=dev)
        rewards = torch.as_tensor(np.asarray(sampled_batch.rewards), dtype=torch.float, device=dev)
        dyn = self._as_dyn(sampled_batch.data_to_encode, actions.shape[0])
        actions = actions.clamp(0, self.n_choices - 1)

        ce, entropy_term = self._denoise_ce(actions, lengths, dyn)            # (B,), (B,)
        adv = rewards - float(b)
        w = torch.softmax(adv / self.lam, dim=0).detach()                     # FKL weights
        fkl = (w * ce).sum()
        mle = ce.mean()
        entropy_bonus = entropy_term.mean()
        total = self.rl_weight * fkl + (1.0 - self.rl_weight) * mle - self.entropy_weight * entropy_bonus
        return total, {"fkl": float(fkl.item()), "mle": float(mle.item()),
                       "entropy_bonus": float(entropy_bonus.item())}

    # GP-guidance uses train_loss with small rl_weight; provide an explicit alias too.
    def train_mle_loss(self, b, sampled_batch, pqt_batch=None, test=False):
        return self.train_loss(b, sampled_batch, pqt_batch, test)

    def compute_neg_log_likelihood(self, data_to_encode, true_action, B=None):
        return torch.zeros((), device=self.device)   # unused unless save_true_log_likelihood
