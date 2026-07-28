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
