import numpy as np
import sympy as sym
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr, phi_vals, fl_penalty, fl_violation


def test_generator_expr_matches_hand_identity():
    sv, drift, spec = make_rotor(3)
    LH = generator_expr(spec["H"], sv, drift, spec["noise"])
    p1, p3 = sv[2], sv[4]
    assert sym.simplify(LH - ((1 - p1**2) + (2 - p3**2))) == 0


def test_phi_vals_formula_and_clamp():
    v = np.array([0.5, 1.0, np.e])
    out = phi_vals(v, alpha=1, c=0.05)
    # v<1 clamps to 1: phi(1) = 0.05*1/(2+0)^1 = 0.025
    assert np.isclose(out[0], 0.025) and np.isclose(out[1], 0.025)
    assert np.isclose(out[2], 0.05 * np.e / 3.0)


def test_fl_penalty_masks_inside_KR():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([2.0, 2.0])
    lv = np.array([5.0, 5.0])          # violating drift
    h = np.array([1.0, 20.0])          # first point inside K_R -> masked out
    pen_all_in = fl_penalty(v, lv, np.array([1.0, 1.0]), spec)
    pen_half = fl_penalty(v, lv, h, spec)
    assert pen_all_in == 0.0           # only positivity term, v>=1 -> 0
    assert pen_half > 0.0


def test_fl_penalty_positivity_term():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([0.0])                # V below 1 everywhere
    lv = np.array([-100.0])
    h = np.array([0.0])
    assert np.isclose(fl_penalty(v, lv, h, spec), 2.5 * 1.0)


def test_fl_penalty_nan_guard():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    assert fl_penalty(np.array([np.nan]), np.array([0.0]),
                      np.array([20.0]), spec) >= 1e6


def test_fl_violation_unweighted():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([0.0]); lv = np.array([-100.0]); h = np.array([0.0])
    assert np.isclose(fl_violation(v, lv, h, spec), 1.0)  # no 2.5 weight


def test_fl_penalty_inf_guard():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    assert fl_penalty(np.array([2.0]), np.array([-np.inf]),
                      np.array([20.0]), spec) >= 1e6
    assert fl_penalty(np.array([np.inf]), np.array([0.0]),
                      np.array([20.0]), spec) >= 1e6


def test_fl_penalty_multielement_mean():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    # two points outside K_R: one violates drift by exactly (1+phi(1))=1.025,
    # one satisfies it; positivity satisfied everywhere (v=1 -> relu(1-v)=0).
    v = np.array([1.0, 1.0])
    lv = np.array([1.0, -100.0])
    h = np.array([20.0, 20.0])
    expected = (max(1.0 + 0.025, 0.0) + 0.0) / 2.0     # mean over BOTH points
    assert np.isclose(fl_penalty(v, lv, h, spec), expected)
