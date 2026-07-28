import collections

from omegaconf import OmegaConf
import sympy as sym  # noqa: F401  (kept for parity with other configs)

from example_config.rotor_common import make_rotor

_N = 2
_SV, _DRIFT, _SPEC = make_rotor(_N)


def dynamics():
    return _SV, _DRIFT


def generator_spec():
    return _SPEC


def config_factory():
    return {
        "task": {
            "task_type": "regression",
            "function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "n2"],  # Koza
            "metric": "foster_lyapunov",
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
                "min_": 10,
                "max_": 35,
                "on": True,
            },
            "repeat": {"tokens": "const", "min_": None, "max_": 5, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": False},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 25, "scale": 5, "on": True},
        },
    }


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
