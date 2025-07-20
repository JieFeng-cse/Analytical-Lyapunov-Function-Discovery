# pylint: disable=pointless-string-statement,broad-exception-caught
# pyright: reportUnboundVariable=false, reportOptionalMemberAccess=false

import itertools
import time
from copy import deepcopy
from itertools import compress

from scipy import optimize
import sympy as sym
from scipy.special import logsumexp
from scipy.optimize import direct, Bounds, shgo
import multiprocessing as mp
import contextlib
import os
import sys
import math
from torch import nn
import sympytorch
import torch.optim.lr_scheduler as lr_scheduler
import csv
import timeit
import copy

import numpy as np
import scipy
import torch
from dso.memory import Batch, make_queue
from dso.program import Program, from_tokens
from dso.task import make_task
from dso.train_stats import StatsLogger
from dso.utils import empirical_entropy, get_duration, log_and_print, weighted_quantile
from dso.variance import quantile_variance
from torch.multiprocessing import (  # pylint: disable=unused-import  # noqa: F401
    Pool,
    cpu_count,
    get_logger,
)
from torch.nn.utils.rnn import pad_sequence
from dso.task.regression.regression import make_regression_metric
from config import dynamics, get_config
import ast
import pandas as pd



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = get_logger()



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


def process_raw_batch(raw_batch, controller):
    eqs = []
    eqs_valid = []
    for eq in raw_batch[1]:
        try:
            eq = torch.Tensor(controller.task.library.actionize(eq_sympy_prefix_to_token_library(eq)))
            eqs.append(eq)
            eqs_valid.append(True)
        except Exception:  # pylint: disable=broad-exception-caught
            eqs_valid.append(False)
    if np.array(eqs_valid).sum() == 0:
        return torch.Tensor([]), torch.Tensor([])
    eqs = pad_sequence(eqs, padding_value=controller.tgt_padding_token).T
    data = raw_batch[0][eqs_valid, :, :]
    return data, eqs



def optomize_at_test(
    controller,
    pool,
    gp_controller,
    output_file,
    pre_train,
    config,
    controller_saved_path,
    n_epochs=None,
    n_samples=2000000,
    batch_size=1000,
    complexity="token",
    const_optimizer="scipy",
    const_params=None,
    alpha=0.5,
    epsilon=0.05,
    n_cores_batch=1,
    verbose=True,
    save_summary=False,
    save_all_epoch=False,
    baseline="R_e",
    b_jumpstart=False,
    early_stopping=True,
    hof=100,
    eval_all=False,
    save_pareto_front=True,
    debug=0,
    use_memory=False,
    memory_capacity=1e3,
    warm_start=None,
    memory_threshold=None,
    save_positional_entropy=False,
    save_top_samples_per_batch=0,
    save_cache=False,
    save_cache_r_min=0.9,
    save_freq=None,
    save_token_count=False,
    gradient_clip=1,
    save_true_log_likelihood=False,
    true_action=None,
):
    """
    Executes the main training loop.

    Parameters
    ----------
    controller : TransformerTreeEncoderController
        Symbolic transformer used to generate symbolic expressions.

    pool : multiprocessing.Pool or None
        Pool to parallelize reward computation. For the control task, each
        worker should have its own TensorFlow model. If None, a Pool will be
        generated if n_cores_batch > 1.

    gp_controller : dso.gp.gp_controller.GPController or None
        GP controller object used to generate Programs.

    output_file : str or None
        Path to save results each step.

    n_epochs : int or None, optional
        Number of epochs to train when n_samples is None.

    n_samples : int or None, optional
        Total number of expressions to sample when n_epochs is None. In this
        case, n_epochs = int(n_samples / batch_size).

    batch_size : int, optional
        Number of sampled expressions per epoch.

    complexity : str, optional
        Complexity function name, used computing Pareto front.

    const_optimizer : str or None, optional
        Name of constant optimizer.

    const_params : dict, optional
        Dict of constant optimizer kwargs.

    alpha : float, optional
        Coefficient of exponentially-weighted moving average of baseline.

    epsilon : float or None, optional
        Fraction of top expressions used for training. None (or
        equivalently, 1.0) turns off risk-seeking.

    n_cores_batch : int, optional
        Number of cores to spread out over the batch for constant optimization
        and evaluating reward. If -1, uses multiprocessing.cpu_count().

    verbose : bool, optional
        Whether to print progress.

    save_summary : bool, optional
        Whether to write TensorFlow summaries.

    save_all_epoch : bool, optional
        Whether to save all rewards for each iteration.

    baseline : str, optional
        Type of baseline to use: grad J = (R - b) * grad-log-prob(expression).
        Choices:
        (1) "ewma_R" : b = EWMA(<R>)
        (2) "R_e" : b = R_e
        (3) "ewma_R_e" : b = EWMA(R_e)
        (4) "combined" : b = R_e + EWMA(<R> - R_e)
        In the above, <R> is the sample average _after_ epsilon sub-sampling and
        R_e is the (1-epsilon)-quantile estimate.

    b_jumpstart : bool, optional
        Whether EWMA part of the baseline starts at the average of the first
        iteration. If False, the EWMA starts at 0.0.

    early_stopping : bool, optional
        Whether to stop early if stopping criteria is reached.

    hof : int or None, optional
        If not None, number of top Programs to evaluate after training.

    eval_all : bool, optional
        If True, evaluate all Programs. While expensive, this is useful for
        noisy data when you can't be certain of success solely based on reward.
        If False, only the top Program is evaluated each iteration.

    save_pareto_front : bool, optional
        If True, compute and save the Pareto front at the end of training.

    debug : int, optional
        Debug level, also passed to Controller. 0: No debug. 1: Print initial
        parameter means. 2: Print parameter means each step.

    use_memory : bool, optional
        If True, use memory queue for reward quantile estimation.

    memory_capacity : int
        Capacity of memory queue.

    warm_start : int or None
        Number of samples to warm start the memory queue. If None, uses
        batch_size.

    memory_threshold : float or None
        If not None, run quantile variance/bias estimate experiments after
        memory weight exceeds memory_threshold.

    save_positional_entropy : bool, optional
        Whether to save evolution of positional entropy for each iteration.

    save_top_samples_per_batch : float, optional
        Whether to store X% top-performer samples in every batch.

    save_cache : bool
        Whether to save the str, count, and r of each Program in the cache.

    save_cache_r_min : float or None
        If not None, only keep Programs with r >= r_min when saving cache.

    save_freq : int or None
        Statistics are flushed to file every save_freq epochs (default == 1). If < 0, uses save_freq = inf

    save_token_count : bool
        Whether to save token counts each batch.

    Returns
    -------
    result : dict
        A dict describing the best-fit expression (determined by reward).
    """
    controller_saved_path = controller_saved_path.replace("controller", "controller_test")
    quantile_over_fit_times_limit = float("inf")  # This could help it ?!
    quantile_over_fit_times = 0
    run_gp_meld = gp_controller is not None
    controller.rl_weight = 1.0
    eval_all = True

    total_unique_set = set()

    # Config assertions and warnings
    assert n_samples is None or n_epochs is None, "At least one of 'n_samples' or 'n_epochs' must be None."

    # Create the priority queue
    k = controller.pqt_k
    if controller.pqt and k is not None and k > 0:
        priority_queue = make_queue(priority=True, capacity=k)
    else:
        priority_queue = None

    memory_queue = None
    r_history = None


    # Main training loop
    p_final = None
    r_best = -np.inf
    prev_r_best = None
    ewma = None if b_jumpstart else 0.0  # EWMA portion of baseline
    n_epochs = n_epochs if n_epochs is not None else int(n_samples / batch_size)
    nevals = 0  # Total number of sampled expressions (from RL or GP)
    positional_entropy = np.zeros(shape=(n_epochs, controller.max_length), dtype=np.float32)

    top_samples_per_batch = list()

    logger = StatsLogger(
        output_file,
        save_summary,
        save_all_epoch,
        hof,
        save_pareto_front,
        save_positional_entropy,
        save_top_samples_per_batch,
        save_cache,
        save_cache_r_min,
        save_freq,  # pyright: ignore
        save_token_count,
    )

    start_time = time.time()
    if verbose:
        print("-- RUNNING EPOCHS START -------------")

    optimizer = torch.optim.Adam(controller.parameters(), lr=controller.learning_rate)

    ## Dynamics embedding (prepare it as input to transformer encoder)
    data_to_encode = prepare_encoder_input()[0]
    data_to_encode = data_to_encode.long().to(DEVICE)
    data_to_encode = data_to_encode.tile(batch_size, 1, 1)
    data_to_encode = data_to_encode.squeeze(dim = 1)
    print(data_to_encode.shape)
 
    gp_collection = []

    ## Set Training Dataset Parameters
    counter_example_size = Program.task.X_train.shape[0]

    conf = get_config()
    benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
    row = benchmark_df.loc[config["task"]["dataset"]]
    train_spec = ast.literal_eval(row["train_spec"])

    global low_bound
    global up_bound 

    low_bound = []
    up_bound = []

    for i in range(1, len(dynamics()[0])+1):
        input_var = "x{}".format(i)
        low, high, n = train_spec[input_var]["U"]
        low_bound.append(low)
        up_bound.append(high)


    for epoch in range(n_epochs):
        t0 = time.time()

        # Set of str representations for all Programs ever seen
        s_history = set(r_history.keys() if Program.task.stochastic else Program.cache.keys())

        # Sample batch of Programs from the Controller
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: [(batch_size, max_length)] * 3
        # Shape of priors: (batch_size, max_length, n_choices)
        actions, obs, priors = controller.sample(batch_size, data_to_encode)
        programs = [from_tokens(a) for a in actions]
        nevals += batch_size

        start = timeit.default_timer()

        ## Employ Expert Guidance or not ?
        run_gp_meld = True
    

        if run_gp_meld:
            deap_programs, deap_actions, deap_obs, deap_priors = gp_controller(actions)
            nevals += gp_controller.nevals

            # Combine transformer outputs and deap outputs - actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)
        end = timeit.default_timer()

        ## Store deap result for expert guidance 
        if run_gp_meld:
            gp_collection = deap_programs
            gp_collection_actions = deap_actions
            gp_collection_obs = deap_obs
            gp_collection_priors = deap_priors
            
        log_and_print(f"Genetic Porgramming Runtime {end - start}")
        pool = None
       
        # Compute rewards in parallel
        if pool is not None:
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)
        
        temp = Program.task.X_train
        
        ## PGD collection for each sampled candidate expression
        print("\n=========================PGD Collection Starts=========================\n")
        pgd_initial = torch.rand(Program.task.X_train.shape[0] // 100, Program.task.X_train.shape[1])
        with mp.Pool(processes= 10, initializer=init_worker) as pool:
            pgd_output = pool.starmap(pgd_check, [(i.sympy_expr[0], pgd_initial, low_bound, up_bound) for i in programs if (not i.invalid)])

        pgd_example = []
        for j in pgd_output:
            if len(j[1]) > 0:
                if len(pgd_example) > 0:
                    pgd_example = np.concatenate((pgd_example, j[1]), axis = 0)
                else:
                    pgd_example = j[1]

        ratio = 0.2
        if len(pgd_example) > Program.task.X_train.shape[0] * ratio:
            random_ind = np.random.randint(0, len(pgd_example), int(np.floor(Program.task.X_train.shape[0] * ratio)))
            print(random_ind.shape)
            pgd_example = pgd_example[random_ind]
                    
        if len(pgd_example) > 0:
            Program.task.X_train = np.vstack((Program.task.X_train, pgd_example))
            Program.task.y_train = np.zeros(Program.task.X_train.shape[0])
        print(f"\n===================Collect {len(pgd_example)} Examples===================\n")
        
        
        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.update_r() for p in programs])
        r_train = r
        
        del Program.task.X_train, Program.task.y_train
        Program.task.X_train = temp
        Program.task.y_train = np.zeros(Program.task.X_train.shape[0])
        
        # Back up programs to save them properly later
        controller_programs = programs.copy() if save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        p_train = programs

        l = np.array([len(p.traversal) for p in programs])  # noqa: E741
        s = [p.str for p in programs]  # Str representations of Programs
        on_policy = np.array([p.originally_on_policy for p in programs])
        invalid = np.array([p.invalid for p in programs], dtype=bool)
        invalid_percent = (np.sum(invalid) / invalid.shape[0]) * 100

        if save_positional_entropy:
            positional_entropy[epoch] = np.apply_along_axis(empirical_entropy, 0, actions)

        if save_top_samples_per_batch > 0:
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            one_perc = int(len(programs) * float(save_top_samples_per_batch))
            for idx in sorted_idx[:one_perc]:
                top_samples_per_batch.append([epoch, r[idx], repr(programs[idx])])
                
        # Update reward history
        if r_history is not None:
            for p in programs:
                key = p.str
                if key in r_history:
                    r_history[key].append(p.r)
                else:
                    r_history[key] = [p.r]

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)
        r_best = max(r_max, r_best)
        r_raw_sum = np.sum(r)
        r_raw_mean = r.sum() / (r != 0).sum()

        epsilon_r = 0.05

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        print("Epsilon",epsilon)

        if epsilon is not None and epsilon < 1.0:
            # Compute reward quantile estimate
            quantile = np.quantile(r, 1 - epsilon, interpolation="higher")  # pyright: ignore
            quantile_ver = np.quantile(r, 1 - epsilon_r , interpolation="higher")

            # These guys can contain the GP solutions if we run GP
            """
                Here we get the returned as well as stored programs and properties.

                If we are returning the GP programs to the controller, p and r will be exactly the same
                as p_train and r_train. Otherwise, p and r will still contain the GP programs so they
                can still fall into the hall of fame. p_train and r_train will be different and no longer
                contain the GP program items.
            """

            keep = r >= quantile
            keep_ver = r >= quantile_ver
            l = l[keep]  # noqa: E741
            s = list(compress(s, keep))  # pyright: ignore
            invalid = invalid[keep]

            # Option: don't keep the GP programs for return to controller
            # gp_controller.return_gp_obs = False
            if run_gp_meld and not gp_controller.return_gp_obs:
                """
                If we are not returning the GP components to the controller, we will remove them from
                r_train and p_train by augmenting 'keep'. We just chop off the GP elements which are indexed
                from batch_size to the end of the list.
                """
                _r = r[keep]
                _p = list(compress(programs, keep))  # pyright: ignore
                print(batch_size)
                print(keep.shape)
                print(r.shape)
                print(np.sum(keep))
                keep[batch_size:] = False
                print(np.sum(keep))
                r_train = r[keep]
                p_train = list(compress(programs, keep))  # pyright: ignore
                print(r_train.shape)

                """
                    These contain all the programs and rewards regardless of whether they are returned to the controller.
                    This way, they can still be stored in the hall of fame.
                """
                r = _r
                programs = _p
                print("Running GP")
            else:
                """
                Since we are returning the GP programs to the contorller, p and r are the same as p_train and r_train.
                """
                print(batch_size)
                print(keep.shape)
                print(r.shape)
                r_train = r = r[keep]
                p_train_ver = list(compress(programs, keep_ver))
                p_train = programs = list(compress(programs, keep))  # pyright: ignore
                print(r_train.shape)
            
            
            """
                get the action, observation, priors and on_policy status of all programs returned to the controller.
            """
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]
        else:
            keep = None
        
        ## SHGO Numerical Verification Process + Counter example feedback
        
        counter_example = np.array([])
        count = 0

        print(f"Number of Programs for Minimization: {len(p_train_ver)}")
        with mp.Pool(processes= mp.cpu_count(), initializer=init_worker) as pool:
            counter_example_splits = pool.starmap(check_options, [(i.sympy_expr[0], low_bound, up_bound) for i in p_train_ver])
            # counter_example_splits = pool.starmap(pgd_check, [(i.sympy_expr[0], Program.task.X_test) for i in p_train])

        pool = None
        print(len(counter_example_splits))
        for j in counter_example_splits:
            if len(j[1]) > 0:
                if len(counter_example) > 0:
                    count += j[1].shape[0]
                    counter_example = np.concatenate((counter_example, j[1]), axis = 0)
                else:
                    counter_example = j[1]


        if len(counter_example) > counter_example_size:
            random_ind = np.random.randint(0, len(counter_example), counter_example_size)
            counter_example = counter_example[random_ind]
        
        if len(counter_example) > 0:
            
            Program.task.X_test = np.vstack((Program.task.X_test, counter_example))

            if  len(Program.task.X_test) > counter_example_size * 2:
                random_ind = np.random.randint(0, Program.task.X_test.shape[0], counter_example_size * 2)
                Program.task.X_test = Program.task.X_test[random_ind]

        Program.task.y_test = np.zeros(Program.task.X_test.shape[0])

        if True:
            if epoch == 0:
                Program.task.X_train = Program.task.X_test
                Program.task.y_train = Program.task.y_test
            
            else:
                portion = (min((epoch - (-1)) * 0.003, 0.1))
                
                add_size = int(portion * Program.task.X_test.shape[0])
                add_part_ind = np.random.randint(0, Program.task.X_test.shape[0], add_size)
                add_part = Program.task.X_test[add_part_ind]
                
                keep_size = int((1 - portion) * Program.task.X_test.shape[0])
                keep_part_ind = np.random.randint(0, Program.task.X_train.shape[0], keep_size)
                keep_part = Program.task.X_train[keep_part_ind]
                
                Program.task.X_train = np.vstack((keep_part, add_part))
                Program.task.y_train = np.zeros(Program.task.X_train.shape[0])

        print(Program.task.X_train.shape)
        print(Program.task.y_train.shape)

        print(Program.task.X_test.shape)
        print(Program.task.y_test.shape)

        print(f"\n=======================Successfully Added {counter_example.shape} Counter Examples=======================\n")

        del counter_example

        ## Evaluate if any output expression is numerically valid
        if eval_all:
            success = [p.evaluate.get("success") for p in programs]  # pyright: ignore
            if any(success):
                minimizer_check = [check_options(programs[i].sympy_expr[0] * 1e5, low_bound, up_bound)[0] if success[i] else False for i in range(len(success))]
                overall_check = [final_check(programs[i].sympy_expr[0], low_bound, up_bound) if minimizer_check[i] else False for i in range(len(success))]
                success = overall_check
                if any(success) > 0:
                    p_final = programs[success.index(True)]
                    print(f"Number of valid equations: {np.sum(success)}. Program should stopped")
                    print(p_final.sympy_expr[0])
                    print([programs[i].sympy_expr[0] for i in range(len(success)) if success[i]])

                    log_and_print(f"Number of valid equations: {np.sum(success)}. Program should stopped")
                    log_and_print(p_final.sympy_expr[0])
                    log_and_print([programs[i].sympy_expr[0] for i in range(len(success)) if success[i]])
                else:
                    log_and_print(f"Number of potential valid equations: {np.sum(success)}")
            else:
                print(f"\n============None of the candidates fulfill the conditions. Training CONTINUES!=================\n")
                log_and_print(f"\n============None of the candidates fulfill the conditions. Training CONTINUES!=================\n")
        
        
        r_quantile_mean = np.mean(r)
        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)
        r_train = np.clip(r_train, -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if baseline == "ewma_R":
            ewma = np.mean(r_train) if ewma is None else alpha * np.mean(r_train) + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "R_e":  # Default
            ewma = -1
            b_train = quantile
        elif baseline == "ewma_R_e":
            ewma = np.min(r_train) if ewma is None else alpha * quantile + (1 - alpha) * ewma
            b_train = ewma
        elif baseline == "combined":
            ewma = (
                np.mean(r_train) - quantile
                if ewma is None
                else alpha * (np.mean(r_train) - quantile) + (1 - alpha) * ewma
            )
            b_train = quantile + ewma
        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), controller.max_length) for p in p_train], dtype=np.int32)

        if data_to_encode is not None:
            if run_gp_meld:
                data_to_encode_train = torch.cat(
                    [
                        data_to_encode.detach(),
                        data_to_encode.detach()[-1, :].tile(config["gp_meld"]["train_n"], 1),
                    ]
                )
                if keep is not None:
                    data_to_encode_train = data_to_encode_train[keep, :].cpu().numpy()
                else:
                    data_to_encode_train = data_to_encode_train[:, :].cpu().numpy()
            elif keep is not None:
                data_to_encode_train = data_to_encode[keep, :].detach().cpu().numpy()
            else:
                data_to_encode_train = data_to_encode[:, :].detach().cpu().numpy()
            print(data_to_encode_train.shape)
        else:
            data_to_encode_train = np.array([])
        tgt_train = np.array([])
        # Create the Batch
        sampled_batch = Batch(
            actions=actions,
            obs=obs,
            priors=priors,
            lengths=lengths,
            rewards=r_train,
            on_policy=on_policy,
            data_to_encode=data_to_encode_train,
            tgt=tgt_train,
        )

        # Update and sample from the priority queue
        if priority_queue is not None:
            priority_queue.push_best(sampled_batch, programs)
            pqt_batch = priority_queue.sample_batch(controller.pqt_batch_size)
        else:
            pqt_batch = None

        if save_true_log_likelihood and true_action is not None:
            if pre_train:
                nll = controller.compute_neg_log_likelihood(sampled_data.tile(1, 1, 1).detach(), true_action).item()
            else:
                nll = controller.compute_neg_log_likelihood(None, true_action, sampled_batch).item()

        # Train the controller
        controller.train()
        optimizer.zero_grad()
        loss, summaries = controller.train_loss(b_train, sampled_batch, pqt_batch, test=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            controller.parameters(), gradient_clip)
        optimizer.step()

        # wall time calculation for the epoch
        epoch_walltime = time.time() - start_time
        torch.save(controller.state_dict(), controller_saved_path)

        ## Expert Guidance from deap refined expressions 
        if any(success) == 0 and (run_gp_meld):

            print("\n==============Start Supervised Learning==============\n")

            ## Sort gp programs by reward and select the top 30% programs
            p_collection_reward = np.array([1 / (gp_p.task.evaluate(gp_p)["nmse_test"] + 1) if gp_p.task.evaluate(gp_p)["nmse_test"] != None else 0 for gp_p in gp_collection])
            gp_ratio = (config["gp_meld"]["train_n"] / len(gp_collection)) * 0.3
            gp_quantile = np.quantile(gp_collection_reward, 1 - gp_ratio, interpolation="higher")  # pyright: ignore
            gp_keep = gp_collection_reward >= gp_quantile

            gp_collection = list(compress(gp_collection, gp_keep))
            gp_mean = np.mean(gp_collection_reward[gp_collection_reward >= gp_quantile])
            log_and_print(f"GP output mean: {gp_mean}")
            gp_collection_actions = gp_collection_actions[gp_keep]
            gp_collection_obs = gp_collection_obs[gp_keep]
            gp_collection_priors = gp_collection_priors[gp_keep]
            gp_collection_reward = gp_collection_reward[gp_keep]
            gp_success = np.array([gp_p.task.evaluate(gp_p)["success"] for gp_p in gp_collection])
            log_and_print(f"GP output success rate: {np.mean(gp_success)}")

            if any(gp_success):
                log_and_print([gp_collection[i].sympy_expr[0] for i in range(len(gp_collection)) if gp_success[i]])
                minimizer_check = [check_options(gp_collection[i].sympy_expr[0] * 1e4, low_bound, up_bound)[0] if gp_success[i] else False for i in range(len(gp_success))]
                overall_check = [final_check(gp_collection[i].sympy_expr[0], low_bound, up_bound) if minimizer_check[i] else False for i in range(len(gp_success))]
                gp_success_f = overall_check
                if any(gp_success_f) > 0:
                    p_final = gp_collection[gp_success_f.index(True)]
                    log_and_print(f"Number of valid equations: {np.sum(gp_success_f)}. Program should stopped")
                    log_and_print(p_final.sympy_expr[0])
                    log_and_print([gp_collection[i].sympy_expr[0] for i in range(len(gp_success_f)) if gp_success_f[i]])
 
            deap_lengths = np.array([min(len(p.traversal), controller.max_length) for p in gp_collection], dtype=np.int32)
            deap_r = gp_collection_reward
            deap_on_policy = np.array([p.originally_on_policy for p in gp_collection])
            deap_data_to_encode_train = data_to_encode.detach()[-1, :].tile(len(gp_collection), 1).cpu().numpy()
            deap_tgt_train = np.array([])

            # Create the Batch
            sampled_batch = Batch(
                actions=gp_collection_actions,
                obs=gp_collection_obs,
                priors=gp_collection_priors,
                lengths=deap_lengths,
                rewards=deap_r,
                on_policy=deap_on_policy,
                data_to_encode=deap_data_to_encode_train,
                tgt=deap_tgt_train,
            )

            controller.rl_weight = 1e-3
            for super in range(2):

                # Train the controller
                controller.train()
                optimizer.zero_grad()
                loss_s, summaries_s = controller.train_loss(b_train, sampled_batch, pqt_batch, test=True)
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(
                    controller.parameters(), gradient_clip)
                optimizer.step()

            controller.rl_weight = 1
            print("\n==============End Supervised Learning==============\n")

        # Collect sub-batch statistics and write output
        logger.save_stats(
            r_full,
            quantile,
            l_full,
            actions_full,
            s_full,
            invalid_full,
            r,
            l,
            actions,
            s,
            invalid,
            r_best,
            r_max,
            ewma,
            summaries,
            epoch,
            s_history,
            b_train,
            epoch_walltime,
            controller_programs,
        )

        # Update the memory queue
        if memory_queue is not None:
            memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        new_r_best = False

        if prev_r_best is None or r_max > prev_r_best:
            new_r_best = True
            p_r_best = programs[np.argmax(r)]
            r2, acc_iid, acc_ood = compute_metrics(p_r_best)

        prev_r_best = r_best
        nmse_test = p_r_best.evaluate["nmse_test"]  # pyright: ignore
        if epsilon is not None:
            if save_true_log_likelihood:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| quantile={quantile:.5f} \t| r_quantile_mean={r_quantile_mean:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| nll={nll} \t| true_equation_set_count={len(total_unique_set)} \t| s/it={time.time() - t0:.5f}"
                )
            else:
                log_and_print(
                    f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| quantile={quantile:.5f} \t| r_quantile_mean={r_quantile_mean:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t| nmse_test={nmse_test} \t| true_equation_set_count={len(total_unique_set)} \t| s/it={time.time() - t0:.5f}"
                )
            if quantile > 0.9:
                quantile_over_fit_times += 1
                if quantile_over_fit_times > quantile_over_fit_times_limit:
                    log_and_print("Converged overfitting detected breaking out")
                    break
            else:
                quantile_over_fit_times = 0
        else:
            log_and_print(
                f"[Test epoch={epoch+1:04d}]  nevals={nevals} \t| train_loss={loss.item():.5f} \t| eqs_invalid %={invalid_percent:.2f} \t| r_best={r_best:.5f} \t| r_raw_sum={r_raw_sum:.5f} \t|  r_raw_mean={r_raw_mean:.5f} \t| r2={r2:.5f} \t| acc_iid={acc_iid:.5f} \t| acc_ood={acc_ood:.5f} \t|  nmse_test={nmse_test} \t| s/it={time.time() - t0:.5f}"
            )

        # Print new best expression
        if verbose and new_r_best:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )
            log_and_print("\n\t** New best")
            p_r_best.print_stats()

        
        if True:
            if eval_all and any(success):
                log_and_print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
                
                break

        if verbose and (epoch + 1) % 5 == 0:
            log_and_print(
                "[{}] Training epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if debug >= 2:
            log_and_print("\nParameter means after epoch {} of {}:".format(epoch + 1, n_epochs))
            # print_var_means()

        if verbose and (epoch + 1) == n_epochs:
            log_and_print(
                "[{}] Ending training after epoch {}/{}, current best R: {:.4f}".format(
                    get_duration(start_time), epoch + 1, n_epochs, prev_r_best
                )
            )

        if nevals > n_samples:
            break

    if verbose:
        log_and_print("-- RUNNING EPOCHS END ---------------\n")
        log_and_print("-- EVALUATION START ----------------")
        # print("\n[{}] Evaluating the hall of fame...\n".format(get_duration(start_time)))

    controller.prior.report_constraint_counts()

    # Save all results available only after all epochs are finished. Also return metrics to be added to the summary file
    results_add = logger.save_results(positional_entropy, top_samples_per_batch, r_history, pool, epoch, nevals)

    # Print the priority queue at the end of training
    if verbose and priority_queue is not None:
        for i, item in enumerate(priority_queue.iter_in_order()):
            log_and_print("\nPriority queue entry {}:".format(i))
            p = Program.cache[item[0]]
            p.print_stats()

    # Close the pool
    if pool is not None:
        pool.close()

    # Return statistics of best Program
    p = p_final if p_final is not None else p_r_best
    result = {
        "r": p.r,
    }
    result.update(p.evaluate)  # pyright: ignore
    result.update({"expression": repr(p.sympy_expr), "traversal": repr(p), "program": p})
    result.update(results_add)  # pyright: ignore

    if verbose:
        log_and_print("-- EVALUATION END ------------------")
    return result


def program_from_tokens_with_custom_data(tokens, config_task, data, on_policy=True):
    # data : (X, y)
    # X : array-like, shape = [n_samples, n_features]
    # y : shape = [n_samples,]
    config_task["dataset"] = data
    task = make_task(**config_task)
    p = Program(tokens, on_policy=on_policy, custom_task=task)
    return p


def work(p):
    """Compute reward and return it with optimized constants"""
    r = p.r  # pylint: disable=unused-variable  # noqa: F841
    return p


def compute_metrics(p):
    y_hat_test = p.execute(p.task.X_test)
    y_test = p.task.y_test
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    try:
        if y_test.size > 1:
            r2 = scipy.stats.pearsonr(np.nan_to_num(y_test), np.nan_to_num(y_hat_test))[0]
        else:
            r2 = 0
    except Exception as e:
        logger.exception("Error in pearson calculation {}".format(e))
        r2 = 0
    y_hat_test = p.execute(p.task.X_test_ood)
    y_test = p.task.y_test_ood
    acc_ood = acc_tau(y_test, y_hat_test)
    return r2, acc_iid, acc_ood


def compute_metrics_gp(ind, gp_main, dataset):
    f = gp_main.toolbox.compile(expr=ind)
    y_hat_test = f(*dataset.X_test.T)
    y_test = dataset.y_test
    acc_iid = acc_tau(y_test, y_hat_test)
    y_test, y_hat_test = remove_nans(y_test, y_hat_test)
    if y_test.size > 1:
        r2 = scipy.stats.pearsonr(y_test, y_hat_test)[0]
    else:
        r2 = 0

    y_hat_test = f(*dataset.X_test_ood.T)
    y_test = dataset.y_test_ood
    acc_ood = acc_tau(y_test, y_hat_test)
    return r2, acc_iid, acc_ood


def acc_tau(y, y_hat, tau=0.05):
    error = np.abs(((y_hat - y) / y))
    error = np.sort(error)[: -int(error.size * 0.05)]
    return (error <= tau).mean()


def make_X(spec, n_input_var, dataset_size_multiplier):
    """Creates X values based on specification"""

    features = []
    for i in range(1, n_input_var + 1):
        # Hierarchy: "all" --> "x{}".format(i)
        input_var = "x{}".format(i)
        if "all" in spec:
            input_var = "all"
        elif input_var not in spec:
            input_var = "x1"

        if "U" in spec[input_var]:
            low, high, n = spec[input_var]["U"]
            n = int(n * dataset_size_multiplier)
            feature = np.random.uniform(low=low, high=high, size=n)
        elif "E" in spec[input_var]:
            start, stop, step = spec[input_var]["E"]
            if step > stop - start:
                n = step
            else:
                n = int((stop - start) / step) + 1
            n = int(n * dataset_size_multiplier)
            feature = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        else:
            raise ValueError("Did not recognize specification for {}: {}.".format(input_var, spec[input_var]))
        features.append(feature)

    # Do multivariable combinations
    if "E" in spec[input_var] and n_input_var > 1:
        X = np.array(list(itertools.product(*features)))
    else:
        X = np.column_stack(features)

    return X


def remove_nans(y_test, y_hat_test):
    y_test_r = []
    y_hat_test_r = []
    for i in zip(y_test, y_hat_test):
        if not np.isnan(i[0]) and not np.isnan(i[1]):
            y_test_r.append(i[0])
            y_hat_test_r.append(i[1])
    y_test_r = np.array(y_test_r)
    y_hat_test_r = np.array(y_hat_test_r)
    return y_test_r, y_hat_test_r


## Helper Functions for Simplicial Homology Global Optimization (SHGO) Numerical Verification Process

def derivative_calculate(v):
    '''
    Given candidate Lyapunov function v in smypy, return the negated lie derivative in sympy. 
    '''

    state_variables, dynamics_ode = dynamics()

    lie_derivative_split = [ v.diff(state_variables[i]) * dynamics_ode[i] for i in range(len(state_variables))]

    lie_derivative = 0
    for i in range(len(lie_derivative_split)):
        lie_derivative += lie_derivative_split[i]
    lie_derivative = (-1) * lie_derivative

    return lie_derivative

def find_root(func, low_bound, up_bound):
    '''
    Perform SHGO algorithms to find global minimum on func within the feasible search space.
    '''

    ## Constraints the search space 
    bounds = Bounds(low_bound, up_bound)

    ## SHGO
    result = shgo(func, bounds, n = 2048, iters = 3, sampling_method = "simplicial")
    root = result.x

    return root

def counter_exp_finder_deri(root1, func1, root2, func2, epsilon, low_bound, up_bound, num=800):
    '''
    This function is used to perform Lyapunov condidtions checking on disrete samples for given functions

    Parameters
    ----------
    func1 : sympy expression
        Candidate Expression

    root1 : np.array
        Global minimizer of func1

    func2: sympy expression
        Negated lie derivative of func1

    root2: np.array
        Global minimizer of func2

    Returns
    ----------
    counter_example : list
        A set of data points violating Lyapunov conditions.
    '''
 
    ## discrete dataset is sampled from 2 different radius around the identified maximizer/minimizer
    counter_example = []
    pd_counter_example = []
    distance = np.random.uniform(0, 0.15, (num,len(dynamics()[0]))) # 0.3
    distance = np.concatenate((distance,np.random.randn(distance.shape[0],distance.shape[1])*0.01),axis=0)


    '''
    Evaluate Lyapunov function and its Lie derivative values on discrete sample.
    Collect any encountered counter-example for further training.
    '''
    

    for j in range(num*2):

        ## Ensure sampled dataset is in the local region
        root1_minus = root1 - distance[j]
        root1_minus = np.clip(root1_minus, low_bound, up_bound)
        root1_plus = root1 + distance[j]
        root1_plus = np.clip(root1_plus, low_bound, up_bound)
        root2_minus = root2 - distance[j]
        root2_minus = np.clip(root2_minus, low_bound, up_bound)
        root2_plus = root2 + distance[j]
        root2_plus = np.clip(root2_plus, low_bound, up_bound)

        ## Function evaluation
        value1 = func1(root1_minus)
        value2 = func1(root1_plus)
        value3 = func2(root2_minus)
        value4 = func2(root2_plus)
        
        ## Lyapunov condidtions' checking

        if value1 < 0:

            pd_counter_example.append((root1_minus).copy())

        if value2 < 0:

            pd_counter_example.append((root1_plus).copy())
        
        if value3 < -epsilon:

            counter_example.append((root2_minus).copy())
        if value4 < -epsilon:

            counter_example.append((root2_plus).copy())
    del distance

    ## Check root explicitly
    if func1(root1) < 0:
        pd_counter_example.append(root1.copy())
    if func2(root2) < -epsilon:
        counter_example.append((root2.copy()))
    return counter_example, pd_counter_example

def check_options(sympy_expr, low_bound, up_bound, epsilon=0):

    '''
    Perform numerical verification process for sympy_expr function
    '''

    ## Check if all state variables are included
    if len(list(sympy_expr.free_symbols)) < len(dynamics()[0]):
        return False, np.array([])
    
    ## Convert sympy form into numpy form. Calculate Lie derivative
    evaluated_origin = {str(i): 0 for i in dynamics()[0]}
    origin = float(sympy_expr.evalf(subs = evaluated_origin))
    sympy_expr = sympy_expr - origin
    numpy_expr = sym.lambdify(dynamics()[0], sympy_expr, "numpy")
    v_dot = derivative_calculate(sympy_expr)
    numpy_v_dot = sym.lambdify(dynamics()[0], v_dot, "numpy")
        
    function1 = lambda x: numpy_expr(*x)
    function2 = lambda x: numpy_v_dot(*x)

    ## Identify maximizer/minimizer using SHGO
    root_1 = find_root(function1, low_bound, up_bound)
    root_2 = find_root(function2, low_bound, up_bound)

    ## Counter-example identification on localized sampling
    counter_example, pd = counter_exp_finder_deri(root_1, function1, root_2, function2, epsilon, low_bound, up_bound)
    counter_exp = counter_example
    counter_exp = np.array(counter_exp)

    
    if ((len(counter_exp) + len(pd)) == 0): 
        valid = True
        return valid, counter_exp
    else:
        valid = False
        return valid, counter_exp
    

def final_check(sympy_expr, low_bound, up_bound):
    '''
    This function is used to thoroughly check the Lyapunov conditions on function sympy_expr
    by random sampling checking and pgd checking, if sympy_expr has passed the numerical verification
    process and has reward equal to 1. 
    '''

    
    evaluated_origin = {str(i): 0 for i in dynamics()[0]}
    origin = float(sympy_expr.evalf(subs = evaluated_origin))
    sympy_expr = sympy_expr - origin
    numpy_expr = sym.lambdify(dynamics()[0], sympy_expr, "numpy")
    v_dot = derivative_calculate(sympy_expr)
    numpy_v_dot = sym.lambdify(dynamics()[0], v_dot, "numpy")

    global function_check_1
    global function_check_2

    def function_check_1(x):
        return numpy_expr(*[x[:, i] for i in range(x.shape[1])])
    
    def function_check_2(x):
        return numpy_v_dot(*[x[:, i] for i in range(x.shape[1])])
    
    ## pgd check
    pgd_check_result = True
    for i in range(5):
        valid, counter_exp = pgd_check(sympy_expr, torch.ones((100000, len(dynamics()[0]))), low_bound, up_bound)
        if not valid:
            pgd_check_result = False
            break
    
    if not pgd_check_result:
        return False

    ## Random Sampling Checking
    num_points = 10 ** 7
    for i in range(10):
        
        boundary_points = []

        # Iterate over each dimension to create points near the boundaries
        for j in range(len(dynamics()[0])):
            # Points near the -1 boundary for the i-th dimension
            point_set_low = np.random.uniform(low_bound, up_bound, (num_points, len(dynamics()[0])))
            point_set_low[:, j] = low_bound[j] + np.abs(0.001 * np.random.rand(num_points))
    
            # Points near the 1 boundary for the i-th dimension
            point_set_high = np.random.uniform(low_bound, up_bound, (num_points, len(dynamics()[0])))
            point_set_high[:, j] = up_bound[j] -  np.abs(0.001 * np.random.rand(num_points))
    

            point_uniform = np.random.uniform(low_bound, up_bound, (num_points, len(dynamics()[0])))
            # Add these points to the boundary_points list
            boundary_points.append(point_set_low)
            boundary_points.append(point_set_high)
            boundary_points.append(point_uniform)

        # Combine all boundary points into one array
        boundary_points = np.vstack(boundary_points)

        check_1 = function_check_1(boundary_points)
        check_2 = function_check_2(boundary_points)

        if any(check_1 < 0) or any(check_2 < 0):
            return False
    
    return True

def pgd_attack(
    x0, f, eps, steps=10, lower_boundary=None, upper_boundary=None, direction="minimize"
):
    """
    Use adversarial attack (PGD) to find violating points.
    Args:
      x0: initialization points, in [batch, state_dim].
      f: function f(x) to find the worst case x to maximize.
      eps: perturbation added to x0.
      steps: number of pgd steps.
      lower_boundary: absolute lower bounds of x.
      upper_boundary: absolute upper bounds of x.
    """
    # Set all parameters without gradient, this can speedup things significantly
    grad_status = {}
    try:
        for p in f.parameters():
            grad_status[p] = p.requires_grad
            p.requires_grad_(False)
    except:
        pass

    step_size = eps / steps * 2
    noise = torch.randn_like(x0) * step_size
    if lower_boundary is not None:
        lower_boundary = torch.max(lower_boundary, x0 - eps)
    else:
        lower_boundary = x0 - eps
    if upper_boundary is not None:
        upper_boundary = torch.min(upper_boundary, x0 + eps)
    else:
        upper_boundary = x0 + eps
    x = x0.detach().clone().requires_grad_()
    # Save the best x and best loss.
    best_x = torch.clone(x).detach().requires_grad_(False)
    fill_value = float("-inf") if direction == "maximize" else float("inf")
    best_loss = torch.full(
        size=(x.size(0),),
        requires_grad=False,
        fill_value=fill_value,
        device=x.device,
        dtype=x.dtype,
    )
    for i in range(steps):
        num_vars = x.shape[1]
        arg_dict = {f'x{i+1}': x[:, [i]] for i in range(num_vars)}
        output = f(**arg_dict).squeeze(1).squeeze(1)
        output.mean().backward()
        if direction == "maximize":
            improved_mask = output >= best_loss
        else:
            improved_mask = output <= best_loss
        best_x[improved_mask] = x[improved_mask]
        best_loss[improved_mask] = output[improved_mask]
        noise = torch.randn_like(x0) * step_size / (i + 1)
        if direction == "maximize":
            x = (
                (
                    torch.clamp(
                        x + torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )
        else:
            x = (
                (
                    torch.clamp(
                        x - torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )

    # restore the gradient requirement for model parameters
    try:
        for p in f.parameters():
            p.requires_grad_(grad_status[p])
    except:
        pass
    return best_x

def pgd_check(sympy, X, low_bound, up_bound):

    if len(list(sympy.free_symbols)) < len(dynamics()[0]):
        return False, np.array([])

    sympy = derivative_calculate(sympy)

    if len(list(sympy.free_symbols)) < 1:
        return False, np.array([])
    
    sympy_torch = sympytorch.SymPyModule(expressions=[sympy]).to("cuda:1")
    data = torch.rand((X.shape[0], X.shape[1])).to("cuda:1")
    result = pgd_attack(data, sympy_torch, min(up_bound)*0.8, steps=100, lower_boundary=torch.tensor(low_bound).to("cuda:1"), upper_boundary=torch.tensor(up_bound).to("cuda:1"), direction="minimize")
    num_vars = result.shape[1]
    arg_dict = {f'x{i+1}': result[:, [i]] for i in range(num_vars)}
    evaluation = sympy_torch(**arg_dict).squeeze(1).squeeze(1)
    result = result[evaluation < - 1e-10]

    del sympy_torch, data, evaluation

    return (len(result) == 0), result.cpu().detach().numpy()
    
def init_worker():
    # Redirect stdout and stderr to suppress print statements in workers
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def prepare_encoder_input():

    '''
    This function convert input dynamics (in ODE form) into sequences of symbolic tokens
    and embed in the form that can be fed into transformer encoder.
    '''

    global vocab

    ## Basic vocabolary - symbolic operators + coefficients
    vocab = {'start': 0, 'add': 1, 'mul': 2, 'pow': 3, 'sin': 4, 'cos': 5, '+': 6, '-': 7, 
             '1':8 ,'2':9, '3':10, '4':11, '5':12, '6':13, '7':14, '8':15, '9': 16, '0':17, "E+1":18,
             'E+0': 19, 'E-1': 20, 'E-2':21, 'E-3':22}
    
    ## Add state varaibles into vocabolary
    for i in range(len(dynamics()[0])):
        vocab[str(dynamics()[0][i])] = len(vocab)
    vocab['end'] = len(vocab)

    ## Convert dynamics from ODE form into sequences of symbolic tokens in pre-order traversal
    f = dynamics()[1]
    polish_expr = [to_polish_with_encoding(i) for i in f]
    input_ids = [[vocab[token] for token in i] for i in polish_expr] 

    # Concant sequences together
    result = []
    for i in input_ids:
        i.insert(0, vocab['start'])
        i.append(vocab['end'])
        result.extend(i)

    return torch.tensor(result), len(vocab), len(result)


# Function to encode an integer in I10
def encode_integer_i10(number):
    sign = '+' if number >= 0 else '-'
    digits = [str(int(digit)) for digit in str(abs(number))]
    return [sign] + (digits)

# Function to encode a real number in E100
def encode_real_f10_e100(number):
    sign = '+' if number >= 0 else '-'
    number = abs(number)
    
    if number == 0:
        return [sign, 0, 'E0']
    
    exponent = math.floor(math.log10(number))
    mantissa = number / (10 ** exponent)
    
    # Round mantissa to 4 significant digits
    mantissa = round(mantissa * 1000)
    
    # Adjust exponent if mantissa rounding affects it
    if mantissa >= 10000:
        mantissa = mantissa // 10
        exponent += 1

    # Clamp the exponent to the range [-100, 100]
    exponent = max(min(exponent, 100), -100)
    
    return [sign] + [str(int(digit)) for digit in str(mantissa)] + [f'E{exponent:+d}']

# Function to convert a sympy expression to Polish notation with I10 and E100 encoding
def to_polish_with_encoding(expr):
    if isinstance(expr, sym.Add):
        return ['add'] + sum([to_polish_with_encoding(arg) for arg in expr.args], [])
    elif isinstance(expr, sym.Mul):
        return ['mul'] + sum([to_polish_with_encoding(arg) for arg in expr.args], [])
    elif isinstance(expr, sym.Pow):
        return ['pow'] + to_polish_with_encoding(expr.args[0]) + to_polish_with_encoding(expr.args[1])
    elif isinstance(expr, sym.sin):
        return ['sin'] + to_polish_with_encoding(expr.args[0])
    elif isinstance(expr, sym.cos):
        return ['cos'] + to_polish_with_encoding(expr.args[0])
    elif isinstance(expr, sym.Symbol):
        return [str(expr)]
    elif isinstance(expr, sym.Integer):
        return encode_integer_i10(int(expr))
    elif isinstance(expr, sym.Float):
        return encode_real_f10_e100(float(expr))
    elif isinstance(expr, sym.Expr) and expr.is_negative:
        return ['mul'] + encode_integer_i10(-1) + to_polish_with_encoding(-expr)
    return [str(expr)]


    