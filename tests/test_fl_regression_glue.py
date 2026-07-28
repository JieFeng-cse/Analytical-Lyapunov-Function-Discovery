import numpy as np
import sympy as sym
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr, fl_penalty


def test_reward_of_reference_H_is_positive_and_below_one():
    """Sanity: V=1+H is NOT a certificate but must produce a finite (0,1] reward."""
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]
    LV = generator_expr(V, sv, drift, spec["noise"])
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3.14, 3.14, 4096),
                         rng.uniform(-6, 6, 4096), rng.uniform(-6, 6, 4096)])
    args = [X[:, i] for i in range(3)]
    v_vals = sym.lambdify(sv, V, "numpy")(*args)
    lv_vals = sym.lambdify(sv, LV, "numpy")(*args)
    h_vals = sym.lambdify(sv, spec["H"], "numpy")(*args)
    r = 1.0 / (1.0 + fl_penalty(v_vals, lv_vals, h_vals, spec))
    assert 0.0 < r <= 1.0
    # V=1+H violates the drift condition on fast-p2 states (LH=1-p1^2 can be ~1>0
    # outside K_R while phi>0), so the reward must NOT be perfect:
    assert r < 1.0
