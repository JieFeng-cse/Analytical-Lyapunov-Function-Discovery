"""Unit tests for the template evaluator core (spec 2026-10-17)."""
from fractions import Fraction

import numpy as np
import sympy as sym

from utils.fl_tmpl.asym import AsymSum, NotQuasiHomogeneous, Space
from utils.fl_tmpl.averaging import slow_momentum
from utils.fl_tmpl.fourier import Coef, CoefContext
from utils.fl_tmpl.splice import bellman_ford, solve_lp
from utils.fl_tmpl.system import RotorSystem, THETA
from utils.fl_tmpl.tropical import TropicalChecker, region_full


def test_fourier_product_is_canonical():
    x, y = sym.symbols("x y")
    ctx = CoefContext([x, y], [THETA])
    a = Coef.from_sympy(ctx, sym.sin(x) * sym.cos(x))
    b = Coef.from_sympy(ctx, sym.sin(2 * x) / 2)
    assert (a - b).is_zero()
    c = Coef.from_sympy(ctx, THETA * sym.cos(x)) * Coef.from_sympy(ctx, sym.cos(y))
    assert sym.simplify(c.to_sympy() - THETA * sym.cos(x) * sym.cos(y)) == 0
    R = np.random.default_rng(0).uniform(-3, 3, (50, 2))
    P = np.full((50, 1), 0.3)
    v = c.evaluate(R, P)
    assert np.allclose(v, 0.3 * np.cos(R[:, 0]) * np.cos(R[:, 1]))


def test_asym_power_series_and_remainder():
    S = RotorSystem(4)
    p2 = S.P(2)
    pt = S.asym(p2 - (sym.cos(S.R(1)) + sym.cos(S.R(2))) / p2)
    lead = S.space.monomial_key({p2: 1}, {p2: 1})
    half = pt.power(Fraction(1, 2), lead, nmax=2, absval=True)
    expr = half.to_sympy()
    # |p~|^(1/2) = |p|^(1/2) (1 - (cos r1 + cos r2)/(2 p^2) + ...)
    val = complex(expr.subs({p2: 100, S.R(1): 0.3, S.R(2): -1.1}))
    ref = abs(100 - (np.cos(0.3) + np.cos(-1.1)) / 100) ** 0.5
    assert abs(val.real - ref) < 1e-6
    assert half.rem                                   # truncation leaves a remainder


def test_quasi_homogeneity_rule():
    S = RotorSystem(3)
    try:
        S.asym(sym.exp(S.P(2)))
    except NotQuasiHomogeneous:
        return
    raise AssertionError("exp(p) must be rejected")


def test_slow_momentum_drifts_match_benchmark():
    # N=2: L p~2 = -1/(2 p^3) (benchmark eq. 18); N=3: -1/p^3 (eq. 23);
    # N=4: -alpha_j/p^3 with alpha_j = gamma <W^2> = 1/2 (CE Prop. 2.2)
    for N, j, want in [(2, 2, -sym.Rational(1, 2)), (3, 2, -1), (4, 2, -sym.Rational(1, 2)),
                       (4, 3, -sym.Rational(1, 2))]:
        res = slow_momentum(RotorSystem(N), j, target=3)
        assert res.mean == {-3: want}, (N, j, res.mean)


def test_tropical_toy_from_source():
    # g = p2^2 - p3^3 (source section 3 step 2): on the edge s2 = 1 the wall is s3 = 2/3
    p2, p3, q = sym.symbols("p2 p3 q")
    sp = Space([p2, p3], angles=[q])
    g = AsymSum.from_sympy(p2**2 * (-1 + sym.cos(q) / 2) - p3**2, sp)
    tc = TropicalChecker(sp, [q])
    blocks = [b for top in (0, 1) for b in tc.check(g, region_full(2, top))]
    assert blocks and all(b.status == "good" for b in blocks)
    walls = [b for b in blocks if len(b.ties) > 1]
    assert walls                                       # the wall s2 = s3 is its own block
    g2 = AsymSum.from_sympy(p2**2 - p3**3, sp)
    stat = {b.status for top in (0, 1) for b in tc.check(g2, region_full(2, top))}
    assert "positive" in stat


def test_lp_slack_and_bellman_ford():
    rows = [("need a > 0.125", {"V": 1.0}, -0.125), ("need a < 0.333", {"V": -1.0}, 0.333)]
    ok, a, sl = solve_lp(rows, ["V"])
    assert ok and 0.125 < a["V"] < 0.333
    ok, a, sl = solve_lp(rows, ["V"], fixed={"V": 0.1})
    assert not ok and abs(sl[0][1] - (0.025 + 1e-3)) < 1e-6
    ok, lam, cyc = bellman_ford(["main", "Vc"], [("main", "Vc", 2.0, "x")])
    assert ok and lam["Vc"] - lam["main"] >= 2.0
    ok, lam, cyc = bellman_ford(["main", "Vc"], [("main", "Vc", 1.0, "x"), ("Vc", "main", 0.5, "y")])
    assert not ok and cyc
