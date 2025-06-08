import logging
import os
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from dso.utils import log_and_print

from config import get_config


import os
import random
import time
from datetime import datetime
from typing import Any

import commentjson as json
import numpy as np
import torch
from dotenv import load_dotenv
from dso.config import load_config
from dso.logeval import LogEval
from dso.prior import make_prior
from dso.program import Program
from dso.task import set_task
from dso.utils import log_and_print
from omegaconf import OmegaConf
from torch.multiprocessing import get_logger
import sympy as sym
from dso.program import from_str_tokens, from_tokens
import ast



from config import (
    config_factory,
    train_config_factory,
    dynamics
)
from models.transformers2 import (
    TransformerTreeEncoderController,
)
from utils.train import (
    optomize_at_test,
    prepare_encoder_input
)

logger = get_logger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using {} device".format(DEVICE))

gradient_clip = 1.0




conf = get_config()

conf.exp.seed_runs = 1
conf.exp.n_cores_task = 1 
conf.exp.seed_start = 5
conf.exp.baselines = ["transformer"]
# User must specify the benchmark to run:
conf.exp.benchmark = "fn_d_all_o"  

Path("./logs").mkdir(parents=True, exist_ok=True)

benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
df = benchmark_df[benchmark_df.index.str.contains(conf.exp.benchmark)]
datasets = df.index.to_list()

file_name = os.path.basename(os.path.realpath(__file__)).split(".py")[0]
path_run_name = "all_{}-{}_01".format(file_name, conf.exp.benchmark)


def create_our_logger(path_run_name):
    logger = multiprocessing.get_logger()
    formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("./logs/{}_log.txt".format(path_run_name))
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info("STARTING NEW RUN ==========")
    logger.info(f"SEE LOG AT : ./logs/{path_run_name}_log.txt")
    return logger



logger = create_our_logger(path_run_name)
logger.info(f"See log at : ./logs/{path_run_name}_log.txt")
data_samples_to_use = int(float(df["train_spec"][0].split(",")[-1].split("]")[0]) * conf.exp.dataset_size_multiplier)

def perform_run(tuple_in):
    seed, dataset, baseline = tuple_in
    logger.info(
        f"[BASELINE_TESTING NOW] dataset={dataset} \t| baseline={baseline} \t| seed={seed} \t| data_samples={data_samples_to_use} \t| noise={conf.exp.noise}"
    )
    
    
    
    result = top_main(
            test_dataset=dataset,
            seed=seed,
            batch_outer_datasets=24,
            batch_inner_equations=100,
            pre_train=False,
            skip_pre_training=True,
            load_pre_trained_path="",
            priority_queue_training=conf.exp.priority_queue_training,
            gp_meld=conf.gp_meld.run_gp_meld,
            model="TransformerTreeEncoderController",
            train_path="",
            test=conf.exp.run_pool_programs_test,
            risk_seeking_pg_train=True,
            save_true_log_likelihood=conf.exp.save_true_log_likelihood,
            p_crossover=conf.gp_meld.p_crossover,
            p_mutate=conf.gp_meld.p_mutate,
            tournament_size=conf.gp_meld.tournament_size,
            generations=conf.gp_meld.generations,
            function_set=conf.exp.function_set,
            learning_rate=conf.exp.learning_rate,
            test_sample_multiplier=conf.exp.test_sample_multiplier,
            n_samples=conf.exp.n_samples,
            dataset_size_multiplier=conf.exp.dataset_size_multiplier,
            noise=conf.exp.noise,
        )
        
    result["baseline"] = baseline  # pyright: ignore
    result["run_seed"] = seed  # pyright: ignore
    result["dataset"] = dataset  # pyright: ignore
    log_and_print(f"[TEST RESULT] {result}")  # pyright: ignore
    return result  # pyright: ignore


def main(dataset, n_cores_task=conf.exp.n_cores_task):

    task_inputs = []
    for seed in range(conf.exp.seed_start, conf.exp.seed_start + conf.exp.seed_runs):
        for baseline in conf.exp.baselines:
            task_inputs.append((seed, dataset, baseline))

    if n_cores_task is None:
        n_cores_task = multiprocessing.cpu_count()
    if n_cores_task >= 2:
        pool_outer = multiprocessing.Pool(n_cores_task)
        for i, result in enumerate(pool_outer.imap(perform_run, task_inputs)):
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )
    else:
        for i, task_input in enumerate(task_inputs):
            result = perform_run(task_input)
            log_and_print(
                "INFO: Completed run {} of {} in {:.0f} s | LATEST TEST_RESULT {}".format(
                    i + 1, len(task_inputs), result["t"], result
                )
            )


            
def top_main(
    test_dataset="",
    seed=0,
    batch_outer_datasets=24,
    batch_inner_equations=100,
    pre_train=True,  # Pre-train model type
    skip_pre_training=False,
    load_pre_trained_path="",
    priority_queue_training=True,
    gp_meld=True,
    model="TransformerTreeEncoderController",
    train_path="",
    test=False,
    risk_seeking_pg_train=True,
    test_sample_multiplier=1,
    data_gen_max_len=20,  
    data_gen_max_ops=5,  
    data_gen_equal_prob_independent_vars=False,
    data_gen_remap_independent_vars_to_monotic=False,
    data_gen_force_all_independent_present=False,
    data_gen_operators=None,
    data_gen_lower_nbs_ops=3,
    data_gen_create_eqs_with_constants=False,
    use_latest_DSRNG_hyperparameters=True,
    save_true_log_likelihood=False,
    p_crossover=None,
    p_mutate=None,
    tournament_size=None,
    generations=None,
    function_set=None,
    learning_rate=None,
    rl_weight=1.0,
    epsilon=None,
    n_samples=None,
    dataset_size_multiplier=1.0,
    noise=0.0,
):  
    seed_all(seed)
    load_dotenv()
    CPU_COUNT_DIV = int(os.getenv("CPU_COUNT_DIV")) if os.getenv("CPU_COUNT_DIV") else 1  # pyright: ignore
    log_and_print(
        f"[RUN SETTINGS]: test_dataset={test_dataset} "
        f" batch_outer_datasets={batch_outer_datasets} pre_train={pre_train} "
        f"load_pre_trained_path={load_pre_trained_path} priority_queue_training={priority_queue_training} "
        f"gp_meld={gp_meld} model={model} train_path={train_path} risk_seeking_pg_train={risk_seeking_pg_train}"
    )
    
    dsoconfig = config_factory()
    nesymres_train_config: Any = train_config_factory()

    if risk_seeking_pg_train:
        batch_outer_datasets = 5
    else:
        batch_outer_datasets = os.cpu_count() // CPU_COUNT_DIV  # pyright: ignore


    # Determine library of functions, i.e. the function set name
    if function_set is not None:
        dsoconfig["task"]["function_set"] = function_set
    dsoconfig["task"]["dataset"] = test_dataset
    config = load_config(dsoconfig)
    config["controller"]["pqt"] = priority_queue_training
    config["gp_meld"]["run_gp_meld"] = gp_meld
    config["model"] = model
    log_and_print("Running model : {}".format(model))
    Program.clear_cache()
    complexity = config["training"]["complexity"]
    Program.set_complexity(complexity)
    

    # Set the constant optimizer
    const_optimizer = config["training"]["const_optimizer"]
    const_params = config["training"]["const_params"]
    const_params = const_params if const_params is not None else {}
    Program.set_const_optimizer(const_optimizer, **const_params)

    pool = None
    # Set the Task for the parent process
    set_task(config["task"])

    
    test_task: Any = Program.task
    function_names = test_task.library.names
    logger.info("Function set: {}".format(function_names))
    
    set_task(config["task"])
    

    nesymres_train_config["architecture"]["dim_input"] = 64

    
    # Save starting seed
    config["experiment"]["starting_seed"] = config["experiment"]["seed"]
    # Set timestamp once to be used by all workers
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    config["experiment"]["timestamp"] = timestamp
    # config["training"]["batch_size"] = batch_inner_equations
    if n_samples is not None:
        config["training"]["n_samples"] = n_samples


    # Save complete configuration file
    output_file = make_output_file(config, seed)
    controller_saved_path = "/".join(output_file.split("/")[:-1]) + "/" + "controller.pt"  # pyright: ignore
    config["function_names_str"] = ",".join(function_names)  # Order very important here

    # Prepare training parameters
    config["state_manager"]["embedding"] = False  
    config["state_manager"]["embedding_size"] = 32  
    config["controller"]["rl_weight"] = rl_weight  # Default 1.0

    
    prior = make_prior(test_task.library, config["prior"])
    
    if model == "TransformerTreeEncoderController":
        config["state_manager"]["embedding"] = True
        config["state_manager"]["embedding_size"] = 64  # 64 also good
        config["controller"]["num_units"] = 64
        controller = TransformerTreeEncoderController(
            prior,
            test_task.library,
            test_task,
            nesymres_train_config["architecture"],
            config_state_manager=config["state_manager"],
            encoder_input_dim= config["controller"]["num_units"],
            **config["controller"],
            vocab_size=prepare_encoder_input()[1],
            dynamics_embedding_size=prepare_encoder_input()[-1],
            dynamics_dimension=len(dynamics()[0])
        ).to(DEVICE)
    
    if model != "gp":
        log_and_print(
            f"{model} parameters: {sum(p.numel() for p in controller.parameters())} \t | "  # pyright: ignore
            f"trainable : {sum(p.numel() for p in controller.parameters() if p.requires_grad)}"  # pyright: ignore
        )
        torch.save(controller.state_dict(), controller_saved_path)  # pyright: ignore
    config["nesymres_train_config"] = OmegaConf.to_container(nesymres_train_config, resolve=True)
    save_config(output_file, config)
    if config["gp_meld"].pop("run_gp_meld", False):
        config["gp_meld"]["train_n"] = test_sample_multiplier * 50
        if p_crossover is not None:
            config["gp_meld"]["p_crossover"] = p_crossover
        if p_mutate is not None:
            config["gp_meld"]["p_mutate"] = p_mutate
        if tournament_size is not None:
            config["gp_meld"]["tournament_size"] = tournament_size
        if generations is not None:
            config["gp_meld"]["generations"] = generations
        log_and_print("GP CONFIG : {}".format(config["gp_meld"]))
        from dso.gp.gp_controller import GPController

        del config["gp_meld"]["verbose"]
        gp_controller = GPController(prior, pool, **config["gp_meld"], seed=seed)
    else:
        gp_controller = None


    logger.info("config: {}".format(config))

    Program.clear_cache()
    set_task(config["task"])

    test_start = time.time()
    result = {"seed": seed}  # Seed listed first
    controller.save_true_log_likelihood = save_true_log_likelihood  # pyright: ignore
    controller.true_eq = []  # pyright: ignore

    config["training"]["batch_size"] = config["training"]["batch_size"] * test_sample_multiplier
    config["training"]["epsilon"] = config["training"]["epsilon"]
    config["training"]["baseline"] = "ewma_R"
    
    true_a = None

    if True:
        result.update(
            optomize_at_test(
                controller,  # pyright: ignore
                pool,
                gp_controller,
                output_file,
                pre_train,
                config,
                controller_saved_path=controller_saved_path,
                **config["training"],
                save_true_log_likelihood=save_true_log_likelihood,
                true_action=true_a,
            )
        )
    result["t"] = time.time() - test_start  # pyright: ignore
    

    save_path = config["experiment"]["save_path"]
    summary_path = os.path.join(save_path, "summary.csv")

    log_and_print("== TRAINING SEED {} END ==============".format(config["experiment"]["seed"]))

    # Evaluate the log files
    log_and_print("\n== POST-PROCESS START =================")
    log = LogEval(config_path=os.path.dirname(summary_path))
    log.analyze_log(
        show_count=config["postprocess"]["show_count"],
        show_hof=config["training"]["hof"] is not None and config["training"]["hof"] > 0,
        show_pf=config["training"]["save_pareto_front"],
        save_plots=config["postprocess"]["save_plots"],
    )
    log_and_print("== POST-PROCESS END ===================")
    return result


def save_config(output_file, config):
    # Save the config file
    if output_file is not None:
        path = os.path.join(config["experiment"]["save_path"], "config.json")
        # With run.py, config.json may already exist. To avoid race
        # conditions, only record the starting seed. Use a backup seed
        # in case this worker's seed differs.
        backup_seed = config["experiment"]["seed"]
        if not os.path.exists(path):
            if "starting_seed" in config["experiment"]:
                config["experiment"]["seed"] = config["experiment"]["starting_seed"]
                del config["experiment"]["starting_seed"]
            with open(path, "w") as f:
                json.dump(config, f, indent=3)
        config["experiment"]["seed"] = backup_seed


def make_output_file(config, seed):
    """Generates an output filename"""

    # If logdir is not provided (e.g. for pytest), results are not saved
    if config["experiment"].get("logdir") is None:
        logger.info("WARNING: logdir not provided. Results will not be saved to file.")
        return None

    # When using run.py, timestamp is already generated
    timestamp = config["experiment"].get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp

    # Generate save path
    task_name = Program.task.name  # pyright: ignore
    save_path = os.path.join(config["experiment"]["logdir"], "_".join([task_name, timestamp, str(seed)]))
    config["experiment"]["task_name"] = task_name
    config["experiment"]["save_path"] = save_path
    os.makedirs(save_path, exist_ok=True)

    seed = config["experiment"]["seed"]
    output_file = os.path.join(save_path, "dso_{}_{}.csv".format(task_name, seed))

    return output_file


def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)  # pyright: ignore  # pylint: disable=not-callable

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if output_file is not None:
        path = os.path.join(config["experiment"]["save_path"], "config.json")
        # With run.py, config.json may already exist. To avoid race
        # conditions, only record the starting seed. Use a backup seed
        # in case this worker's seed differs.
        backup_seed = config["experiment"]["seed"]
        if not os.path.exists(path):
            if "starting_seed" in config["experiment"]:
                config["experiment"]["seed"] = config["experiment"]["starting_seed"]
                del config["experiment"]["starting_seed"]
            with open(path, "w") as f:
                json.dump(config, f, indent=3)
        config["experiment"]["seed"] = backup_seed


def make_output_file(config, seed):
    """Generates an output filename"""

    # If logdir is not provided (e.g. for pytest), results are not saved
    if config["experiment"].get("logdir") is None:
        logger.info("WARNING: logdir not provided. Results will not be saved to file.")
        return None

    # When using run.py, timestamp is already generated
    timestamp = config["experiment"].get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp

    # Generate save path
    task_name = Program.task.name  # pyright: ignore
    save_path = os.path.join(config["experiment"]["logdir"], "_".join([task_name, timestamp, str(seed)]))
    config["experiment"]["task_name"] = task_name
    config["experiment"]["save_path"] = save_path
    os.makedirs(save_path, exist_ok=True)

    seed = config["experiment"]["seed"]
    output_file = os.path.join(save_path, "dso_{}_{}.csv".format(task_name, seed))

    return output_file


def seed_all(seed=None):
    """
    Set the torch, numpy, and random module seeds based on the seed
    specified in config. If there is no seed or it is None, a time-based
    seed is used instead and is written to config.
    """
    # Default uses current time in milliseconds, modulo 1e9
    if seed is None:
        seed = round(time() * 1000) % int(1e9)  # pyright: ignore  # pylint: disable=not-callable

    # Set the seeds using the shifted seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    from dso.config import load_config
    from dso.task import set_task

    from config import (
        config_factory,
        train_config_factory,
    )
    
    dsoconfig = config_factory()
    log_and_print(df.to_string())
    for dataset, row in df.iterrows():
    
        covars = row["variables"]

        nesymres_train_config = train_config_factory()
        dsoconfig["task"]["dataset"] = dataset
        config = load_config(dsoconfig)
        set_task(config["task"])
        try:
            main(dataset)
        except FileNotFoundError as e:
            # pylint: disable-next=raise-missing-from
            if 'nesymres_pre_train' in str(e):
                raise FileNotFoundError(
                    f"Please download the baseline pre-trained models for NeuralSymbolicRegressionThatScales from https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales and put them into the folder `models/nesymres_pre_train`. No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
            else:                
                raise FileNotFoundError(
                    f"No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                )
    logger.info("Fin.")
    
