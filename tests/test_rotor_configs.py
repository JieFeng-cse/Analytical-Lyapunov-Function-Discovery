import importlib
import json
import pandas as pd
import pytest

ROWS = {"rotor2": ("fn_rotor_2", 3), "rotor3": ("fn_rotor_3", 5),
        "rotor4": ("fn_rotor_4", 7), "rotor5": ("fn_rotor_5", 9)}
CSV = "libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv"


@pytest.mark.parametrize("sys,expected", ROWS.items())
def test_config_module(sys, expected):
    m = importlib.import_module(f"example_config.config_{sys}")
    sv, drift = m.dynamics()
    assert len(sv) == expected[1] == len(drift)
    spec = m.generator_spec()
    assert set(spec) >= {"noise", "alpha", "c", "R", "H"}
    conf = m.config_factory()
    assert conf["task"]["metric"] == "foster_lyapunov"
    assert conf["task"]["function_set"] == ["add", "sub", "mul", "div",
                                            "sin", "cos", "exp", "n2"]
    assert conf["prior"]["length"]["min_"] == 10
    assert conf["prior"]["length"]["max_"] == 35
    assert m.train_config_factory() is not None


@pytest.mark.parametrize("sys,expected", ROWS.items())
def test_csv_row(sys, expected):
    name, dim = expected
    df = pd.read_csv(CSV, index_col=0)
    assert name in df.index
    row = df.loc[name]
    assert int(row["variables"]) == dim
    train_spec = json.loads(row["train_spec"].replace("'", '"'))
    n_r = dim - (dim + 1) // 2
    for i in range(1, dim + 1):
        lo, hi, n = train_spec[f"x{i}"]["U"]
        assert n == 15000
        if i <= n_r:
            assert (lo, hi) == (-3.14, 3.14)
        else:
            assert (lo, hi) == (-6, 6)
