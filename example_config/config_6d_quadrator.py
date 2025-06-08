import collections

from omegaconf import OmegaConf
import sympy as sym


def config_factory():
    return {
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "sin", "cos", "n2"]  # Koza
        },
        "training": {
            "n_samples": 2000000,
            "batch_size": 500,
            "epsilon": 0.1,
            "n_cores_batch": 1,
        },
        "controller": {
            "learning_rate": 0.0003,
            "entropy_weight": 0.005,
            "entropy_gamma": 0.8,
        },
        "prior": {
            "length": {
                "min_": 12,
                "max_": 20,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 15, "scale": 5, "on": True},
        },
    }

def dynamics():
    x1, x2, x3, x4, x5, x6 = sym.symbols("x1, x2, x3, x4, x5, x6")


    ## 6-D Quadrator
    ## fn_d_all_z_5 is the corresponding state space and training set setting in libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv

    
    state_variables = [x1, x2, x3, x4, x5, x6]

    I_x = 2
    I_y = 2
    I_z = 5

    k1 = 5
    k2 = 20
    k3 = 4

    U2 = -I_x * x1 - k1 * x2
    U3 = -I_y * x3 - k2 * x4
    U4 = -I_z * x5 - k3 * x6


    dynamics_ode = [x2,
        float((I_y - I_z) / I_x) * (x6 * x4) + U2 * float(1/I_x) - x4 * sym.sin(x2) * sym.cos(x4) * float(1/I_x),
        x4,
        float((I_z - I_x) / I_y) * (x6 * x2) + U3 * float(1/I_y) + x2 * sym.sin(x2) * sym.cos(x4) * float(1/I_y),
        x6,
        float((I_x - I_y) / I_z) * (x4 * x2) + U4 * float(1/I_z)]
    
    return state_variables, dynamics_ode



def train_config_factory():
    return OmegaConf.create(
        {
            "architecture": {
                "sinuisodal_embeddings": False,
                "dec_pf_dim": 32,
                "dec_layers": 1,
                "dim_hidden": 32,
                "lr": 0.0001,
                "dropout": 0,
                "num_features": 2,
                "ln": True,
                "N_p": 0,
                "num_inds": 50,
                "activation": "relu",
                "bit16": True,
                "norm": True,
                "linear": False,
                "input_normalization": False,
                "src_pad_idx": 0,
                "trg_pad_idx": 0,
                "length_eq": 20,
                "n_l_enc": 5,
                "mean": 0.5,
                "std": 0.5,
                "dim_input": 4,
                "num_heads": 2,
                "output_dim": 10,
            },
        }
    )



def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_config(skip_cli=True):
    base_conf = OmegaConf.load("config.yaml")
    if skip_cli:
        return base_conf
    flat_base_conf = flatten(base_conf)
    cli_conf = OmegaConf.from_cli()
    cli_conf = OmegaConf.create({(k[2:] if k[:2] == "--" else k): v for k, v in cli_conf.items()})  # pyright: ignore
    flat_cli_conf = flatten(cli_conf)

    list_cond = [k in flat_base_conf for k in flat_cli_conf.keys()]
    contains_all_keys_bool = all(list_cond)
    assert contains_all_keys_bool, f"Input CLI keys that cannot be set {set(flat_cli_conf) - set(flat_base_conf)}"
    conf = OmegaConf.merge(base_conf, cli_conf)
    return conf


if __name__ == "__main__":
    conf = get_config()
    print("priority_queue_training: ", conf.experiment.priority_queue_training)
    print("seed_runs: ", conf.experiment.seed_runs)
    print("")
