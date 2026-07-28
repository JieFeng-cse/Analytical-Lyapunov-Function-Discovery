import numpy as np
from example_config.rotor_common import make_rotor
from utils.fl_reward import fl_random_falsify, fl_harvest


def test_random_falsify_rejects_bad_V():
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]          # violates drift condition outside K_R
    low, high = [-3.14, -6, -6], [3.14, 6, 6]
    ok, worst = fl_random_falsify(V, sv, drift, spec, low, high,
                                  n_rounds=2, n_pts=20000, seed=0)
    assert ok is False and worst is not None and worst.shape[-1] == 3


def test_random_falsify_accepts_trivial_negative_drift():
    """V = 2 (constant) fails the all-vars rule upstream, but as pure math:
    LV=0, phi(2)>0 -> drift violated outside K_R. Use it to assert the mask
    works: shrink the box INSIDE K_R so no test point has H>R -> ok=True."""
    sv, drift, spec = make_rotor(2)
    V = 2 + 0 * sv[0]
    low, high = [-0.1, -0.5, -0.5], [0.1, 0.5, 0.5]   # H<=~0.26 < R
    ok, worst = fl_random_falsify(V, sv, drift, spec, low, high,
                                  n_rounds=1, n_pts=1000, seed=0)
    assert ok is True and worst is None


def test_harvest_shape():
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]
    pts = fl_harvest(V, sv, drift, spec, [-3.14, -6, -6], [3.14, 6, 6],
                     n_pts=5000, k=10, seed=1)
    assert pts.shape == (10, 3)


def test_objective_fns_accepts_precomputed_lv():
    import sympy as sym
    from utils.fl_reward import fl_objective_fns, generator_expr
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]
    lv = generator_expr(V, sv, drift, spec["noise"])
    v_fn, lv_fn, h_fn = fl_objective_fns(V, sv, drift, spec, lv_expr=lv)
    x = np.array([[0.5, 1.0, 2.0]])
    args = [x[:, i] for i in range(3)]
    direct = sym.lambdify(sv, lv, "numpy")(*args)
    assert np.allclose(lv_fn(*args), direct)


def test_pointwise_violation_perpoint_nonfinite():
    from utils.fl_reward import fl_objective_fns, _pointwise_violation
    sv, drift, spec = make_rotor(2)
    V = sv[1] ** 2 + sv[2] ** 2 + 0 * sv[0] + 1   # 1+p1^2+p2^2, finite everywhere
    v_fn, lv_fn, h_fn = fl_objective_fns(V, sv, drift, spec)
    X = np.array([[0.0, 5.0, 5.0], [0.0, 5.0, 5.0]])
    base = _pointwise_violation(v_fn, lv_fn, h_fn, X, spec)
    assert np.all(np.isfinite(base))

    # inject a non-finite V at point 0 only via a wrapper fn
    def bad_v(*a):
        out = np.asarray(v_fn(*a), dtype=float).copy()
        out = np.broadcast_to(out, (X.shape[0],)).copy()
        out[0] = np.inf
        return out

    viol = _pointwise_violation(bad_v, lv_fn, h_fn, X, spec)
    assert viol[0] >= 1e6 and np.isfinite(viol[1]) and viol[1] < 1e6
