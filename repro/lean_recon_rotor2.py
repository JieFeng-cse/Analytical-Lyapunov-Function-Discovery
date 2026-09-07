"""Numerical reconnaissance for the Lean tail theorem (spec 2026-09-07, §6).

Evaluates the rotor2 certificate (C36 constants) OUTSIDE the +-15 box, region
by region, in a scaled form that cannot overflow:

    V  = exp(theta*p2^2/2) * W,        LV = exp(theta*p2^2/2) * (LW - theta*p2*sin(r1)*W)
    ratio(x) := -LV*(2+log V)/V = -(LW - theta*p2*sin(r1)*W)*(2 + theta*p2^2/2 + log W)/W

The drift condition LV + c*V/(2+log V) <= 0 holds at x iff ratio(x) >= c.
Prints, per region, min ratio and the worst point, plus the fast-region
remainder constant  sup |ptilde^4 * (L ptilde + 1/(2 ptilde^3))|.

Run from a rotor2 run dir / repo root with the alfd env:
    python repro/lean_recon_rotor2.py
Importable: build / ratio / grid / report / unscaled_check.
"""
import sys

import numpy as np
import sympy as sym

sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor  # noqa: E402
from utils.fl_reward import generator_expr  # noqa: E402
from utils.fl_scaffold import _chi  # noqa: E402

THETA, A, RRHO, M = sym.Rational(1, 50), 20, 20, 9

sv, drift, spec = make_rotor(2)
r1, p1, p2 = sv
noise = spec["noise"]
H = spec["H"]

t1 = sym.cos(r1)
t2 = p1 * sym.cos(r1)
t3 = p1**2 * sym.cos(r1) + p1 * sym.sin(r1) + sym.cos(r1)**2
t4 = (-p1**3 * sym.cos(r1) - 3 * p1**2 * sym.sin(r1) - 3 * p1 * sym.cos(r1)**2
      + p1 * sym.cos(r1) - sym.Rational(1, 2) * sym.sin(r1) * sym.cos(r1)
      + 2 * sym.sin(r1))
S = p1**4 + RRHO
rho = _chi(p2**2 / S**2)


def build(rho_expr):
    """Scaled pieces for a given gate expression (symbolic 0/1 or the real chi):
    returns numpy fns (W, LW, ptilde, L ptilde)."""
    I = lambda j: rho_expr / p2**j
    pt = p2 - t1 * I(1) - t2 * I(2) - t3 * I(3) + t4 * I(4)
    W = (sym.exp(-THETA * p2**2 / 2) + sym.exp(THETA * (H - p2**2 / 2))
         + A * rho_expr * pt**2 * sym.exp(THETA * (pt**2 - p2**2) / 2))
    LW = generator_expr(W, sv, drift, noise)
    Lpt = generator_expr(pt, sv, drift, noise)
    f = lambda e: sym.lambdify(sv, e, "numpy")
    return f(W), f(LW), f(pt), f(Lpt)


def ratio(fW, fLW, X):
    with np.errstate(all="ignore"):
        W = fW(*X.T); LW = fLW(*X.T)
    p2v, r1v = X[:, 2], X[:, 0]
    logV = 2 + float(THETA) * p2v**2 / 2 + np.log(W)
    return -(LW - float(THETA) * p2v * np.sin(r1v) * W) * logV / W


def grid(r1s, p1s, p2s):
    R, P1, P2 = np.meshgrid(r1s, p1s, p2s, indexing="ij")
    return np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)


def report(name, X, fW, fLW, extra=""):
    rr = ratio(fW, fLW, X)
    ok = np.isfinite(rr)
    i = np.argmin(np.where(ok, rr, np.inf))
    print(f"{name:34s} n={len(X):7d} min ratio={rr[i]:+.4e} at r1={X[i,0]:+.3f} "
          f"p1={X[i,1]:+.3f} p2={X[i,2]:+.4g}  nonfinite={int((~ok).sum())} {extra}",
          flush=True)
    return rr


def unscaled_check(pt_r1, pt_p1, pt_p2, dps=40):
    """Independent high-precision evaluation of the UNSCALED V, LV at one point
    (mpmath, no scaling trick) -> (V, LV, ratio). Guards against lambdify/scaling bugs."""
    import mpmath as mp
    mp.mp.dps = dps
    I = lambda j: sym.Piecewise((rho / p2**j, p2**2 > 1), (0, True))
    pt = p2 - t1 * I(1) - t2 * I(2) - t3 * I(3) + t4 * I(4)
    V = 1 + sym.exp(THETA * H) + A * rho * pt**2 * sym.exp(THETA / 2 * pt**2)
    LV = generator_expr(V, sv, drift, noise)
    subs = {r1: sym.Float(pt_r1, dps), p1: sym.Float(pt_p1, dps), p2: sym.Float(pt_p2, dps)}
    v = sym.N(V.subs(subs), dps); lv = sym.N(LV.subs(subs), dps)
    return v, lv, sym.N(-lv * (2 + sym.log(v)) / v, 15)


R1S = np.linspace(-np.pi, np.pi, 49)


def main():
    # ---- (a) slow region, rho = 0 exactly: |p2| <= S, p1^2 > M -------------
    fW0, fLW0, _, _ = build(sym.Integer(0))
    p1s = np.concatenate([np.linspace(3.0, 6, 25), np.linspace(6, 30, 25)])
    rows = []
    for p1v in p1s:
        Sv = p1v**4 + RRHO
        rows.append(grid(R1S, [p1v], np.linspace(-Sv, Sv, 41)))
    report("(a) slow rho=0, p1^2>9", np.vstack(rows), fW0, fLW0)

    # ---- (b) transition band with the real gate: S < |p2| < 2S, p1^2 > M ---
    fWc, fLWc, _, _ = build(rho)
    rows = []
    for p1v in np.concatenate([np.linspace(3.0, 4, 21), np.linspace(4, 8, 17)]):
        Sv = p1v**4 + RRHO
        p2s = np.concatenate([np.linspace(Sv, 2 * Sv, 60), -np.linspace(Sv, 2 * Sv, 60)])
        rows.append(grid(R1S, [p1v], p2s))
    Xb = np.vstack(rows)
    rb = report("(b) transition S<|p2|<2S, p1^2>9", Xb, fWc, fLWc)
    bad = Xb[rb < 0]
    if len(bad):
        q = np.abs(bad[:, 2]) / (bad[:, 1]**4 + RRHO)
        print(f"    VIOLATIONS: {len(bad)}  p1 range [{bad[:,1].min():.3f},{bad[:,1].max():.3f}] "
              f"|p2|/S range [{q.min():.3f},{q.max():.3f}]")

    # ---- (c) fast region, rho = 1 exactly: |p2| >= 2S -----------------------
    fW1, fLW1, fpt1, fLpt1 = build(sym.Integer(1))
    rows = []
    for p2v in np.concatenate([np.linspace(40, 400, 37), np.geomspace(400, 1e5, 25)]):
        p1max = (p2v / 2 - RRHO) ** 0.25
        rows.append(grid(R1S, np.linspace(-p1max, p1max, 31), [p2v, -p2v]))
    Xc = np.vstack(rows)
    rc = report("(c) fast rho=1, |p2|>=2S", Xc, fW1, fLW1)
    # asymptotic law: ratio -> theta^2/4 = 1e-4 as |p2| -> infinity.
    # NB: for |p2| >~ 1e4 the scaled evaluation loses ~1e-5 to cancellation.
    for lo, hi in [(40, 100), (100, 1e3), (1e3, 1e4), (1e4, 1e5)]:
        m = (np.abs(Xc[:, 2]) >= lo) & (np.abs(Xc[:, 2]) < hi)
        print(f"    |p2| in [{lo:g},{hi:g}): min ratio {np.nanmin(rc[m]):+.4e}   (theta^2/4 = {float(THETA)**2/4:.1e})")
    m = np.abs(Xc[:, 2]) <= 1e3          # remainder only where float64 resolves it
    with np.errstate(all="ignore"):
        pt = fpt1(*Xc[m].T); Lpt = fLpt1(*Xc[m].T)
    rem = pt**4 * (Lpt + 1 / (2 * pt**3))
    i = np.nanargmax(np.abs(rem))
    print(f"    fast remainder (|p2|<=1e3) sup |pt^4 (L pt + 1/(2 pt^3))| = {abs(rem[i]):.4e} at "
          f"r1={Xc[m][i,0]:+.3f} p1={Xc[m][i,1]:+.3f} p2={Xc[m][i,2]:+.4g}")
    w = 1 + np.abs(Xc[m][:, 1]) ** 5
    print(f"    sup |rem|/(1+|p1|^5) = {np.nanmax(np.abs(rem) / w):.4e}")


if __name__ == "__main__":
    main()
