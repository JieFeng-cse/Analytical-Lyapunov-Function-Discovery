import numpy as np
import sympy as sym
from example_config.rotor_common import make_rotor
from utils.fl_blind import (Chi, CHI_MODULES, BlindJudge, reference_pair, split_pair,
                            ratio_mp_points, fl_blind_final_check, RHO_ONE)

# provisional numbers of the same shape as example_config/blind_rotor2.json
SPEC_BLIND = {"theta": 0.1, "P1max": 40.0, "P2max": 4e4, "M_max": 985.0, "R2_max": 2.5e6,
              "M_near": 64.0, "R2_near": 1.03e4, "M_0": 12.0, "R2_0": 250.0, "c_t": 5e-4,
              "A_grid": [0.1, 1, 10, 100, 1000], "grid_coarse": True}


def _spec():
    sv, drift, spec = make_rotor(2)
    spec = dict(spec)
    spec["blind"] = dict(SPEC_BLIND)
    return spec, sv, drift


def _judge():
    spec, sv, drift = _spec()
    return BlindJudge(spec, sv, drift), sv


def test_chi_derivatives_match_fd():
    z = sym.Symbol("z")
    f = sym.lambdify(z, Chi(z), modules=[CHI_MODULES, "numpy"])
    d1 = sym.lambdify(z, sym.diff(Chi(z), z), modules=[CHI_MODULES, "numpy"])
    d2 = sym.lambdify(z, sym.diff(Chi(z), z, 2), modules=[CHI_MODULES, "numpy"])
    zs = np.linspace(-1, 2, 301)
    h = 1e-5
    fd1 = (f(zs + h) - f(zs - h)) / (2 * h)
    fd2 = (f(zs + h) - 2 * f(zs) + f(zs - h)) / h**2
    assert np.allclose(d1(zs), fd1, atol=1e-5)
    assert np.allclose(d2(zs), fd2, atol=1e-3)
    assert d1(-0.5) == 0 and d1(1.5) == 0 and d2(-0.5) == 0 and f(2.0) == 1.0


def test_split_pair_and_invalid_shapes():
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair")
    step = sym.Function("step")
    assert split_pair(pair(step(x3), x3)) is not None
    assert split_pair(step(x3)) is None
    assert judge.penalty(step(x3))[2] is False                      # no pair root
    assert judge.penalty(pair(-step(x3**2 - 4), x3))[2] is False    # rho < 0
    assert judge.penalty(pair(step(x3**2 - 4), x2))[2] is False     # ptilde without p2
    pen, A, ok = judge.penalty(pair(step(x3**2 - 4), 5 * x3))       # overflow probe
    assert ok and np.isfinite(pen)
    pen0, A0, ok0 = judge.penalty(pair(sym.Integer(0) * x3 + step(x3 - x3), x3))  # rho == 0
    assert ok0 and np.isfinite(pen0) and pen0 > 0.5                 # shells violated


def test_reference_matches_mpmath():
    judge, sv = _judge()
    expr = reference_pair()
    assert judge.fields(expr) is not None
    pts = np.array([[0.3, 5.0, 10.0], [0.3, 5.0, 900.0], [0.3, 2.0, 100.0],
                    [1.0, 0.5, 40.0], [-2.0, 3.0, 300.0], [0.0, 1.0, 0.0]])
    r_num = judge.ratio_at(expr, pts, A=20.0)
    r_ref = judge.ratio_mpmath(expr, pts, A=20.0, dps=40)
    ok = np.isfinite(r_ref)
    assert ok.sum() >= 5
    assert np.allclose(r_num[ok], r_ref[ok], rtol=1e-6, atol=1e-12)
    # log-domain mpmath path agrees with the float path where float64 is trusted
    pts2 = np.array([[0.3, 5.0, 2000.0], [0.3, 5.0, 1e4], [1.0, 0.5, 40.0]])
    r_mp = ratio_mp_points(judge, expr, pts2, A=20.0, dps=30)
    assert np.allclose(judge.ratio_at(expr, pts2, A=20.0), r_mp, rtol=1e-4)


def test_reference_passes_and_truncation_fails():
    judge, sv = _judge()
    pen4, A4, ok4 = judge.penalty(reference_pair(order=4))
    pen1, A1, ok1 = judge.penalty(reference_pair(order=1))
    assert ok4 and ok1
    v_out, v_comp, v_in = judge.last_parts if A4 == judge.A_grid[-1] else judge.last_parts
    fl = judge.fields(reference_pair(order=4))
    s4 = judge.score(fl, A4)
    assert s4[1][0] == 0.0 and s4[1][1] == 0.0          # no drift violation outside K, shells ok
    assert pen4 <= 0.1                                  # only the secondary v_in term
    assert pen1 > 1.0


def test_final_check_reference():
    spec, sv, drift = _spec()
    spec["blind"].update(n_random=20000, dense=False, far_grid=(5, 4, 4), dps=30)
    ok, worst = fl_blind_final_check(reference_pair(order=4), spec, sv, drift, A=20.0)
    assert ok, worst
    ok1, _ = fl_blind_final_check(reference_pair(order=1), spec, sv, drift, A=20.0)
    assert not ok1


def test_calibration_json_loads():
    import json
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "example_config", "blind_rotor2.json")
    assert os.path.exists(p)
    d = json.load(open(p))
    for k in BlindJudge.KEYS + ("A_grid", "c_ref", "M_max_ref", "R2_near_ref", "R2_max_ref"):
        assert k in d
    assert d["theta"] == 0.1 and d["M_max"] > d["M_max_ref"] and d["c_t"] < d["c_ref"]
    assert d["R2_near"] < d["P2max"] < d["R2_max"]


def test_periodicity_backstop():
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair")
    step = sym.Function("step")
    assert judge.penalty(pair(step(x3**4), x3 + x1))[2] is False           # linear in r1
    assert judge.penalty(pair(step(x3**4 + x1), x3))[2] is False           # r1 inside the gate
    assert judge.penalty(pair(step(x3**4), x3 + sym.cos(x1)))[2] is True   # periodic


def test_oscillating_gate_is_not_exempt():
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair")
    step = sym.Function("step")
    # the exploit found by job 20502818 epoch 1: checkerboard gate, bounded ptilde
    pen, A, ok = judge.penalty(pair(step(sym.cos(2 * x3**2))**2, sym.cos(x3 + 2) + 2))
    assert ok
    fl = judge.fields(pair(step(sym.cos(2 * x3**2))**2, sym.cos(x3 + 2) + 2))
    assert fl["band"].mean() < 0.05          # essentially nothing exempted
    assert pen > 2.0                          # far below the order-3 reference (~1.3)
    # the document's gate keeps its transition band exempt
    fl_ref = judge.fields(reference_pair(order=4))
    assert fl_ref["band"].sum() > 0
    s4 = judge.score(fl_ref, 10.0)
    assert s4[1][0] == 0.0 and s4[1][1] == 0.0


def test_flat_V_is_capped():
    """V ~ huge constant (rho == 1, ptilde ~ 65537): ratio ~ 0 everywhere, a mild
    violation at every scored point. Must stay below the order-3 reference."""
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair")
    step = sym.Function("step")
    flat = pair(sym.Integer(1) + 0 * x3,
                (0.5 * step(16.0 - x2**8 / (25.0 * (0.2 * x3 - 1)**2 + 2.0)) + 65537.0)
                / (1.0 - step(step(0.45 - 27.3 * x2**2))))
    pen_flat, _, ok = judge.penalty(flat)
    pen3, _, ok3 = judge.penalty(reference_pair(order=3))
    assert ok and ok3
    assert pen_flat > 2.0 and 1.0 / (1.0 + pen_flat) < 0.34
    assert pen3 < pen_flat


def test_complexity_caps_reject_nested_monsters():
    import time
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair")
    step = sym.Function("step")
    e = x3
    for _ in range(6):
        e = step(sym.exp(e * x2) + sym.cos(x2**2) * e)
    monster = pair(step(sym.cos(x2)) * e, x3 * e)
    t0 = time.time()
    pen, A, ok = judge.penalty(monster)
    assert not ok
    assert time.time() - t0 < 10.0
    # the reference (nesting depth 1, ~150 ops) is unaffected
    assert judge.penalty(reference_pair(order=4))[2]
