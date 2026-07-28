"""Rotor-chain benchmark systems in bond coordinates (spec 2026-07-28).

State: x1..x_{N-1} = bond angles r_i = q_{i+1}-q_i, x_N..x_{2N-1} = momenta p_i.
Drift: dr_i = p_{i+1}-p_i;
  N=2 (one bath):  dp1 = sin r1 - p1,  dp2 = -sin r1
  N>=3 (two baths): dp1 = sin r1 - p1, dp_i = sin r_i - sin r_{i-1},
                    dpN = -sin r_{N-1} - pN
Generator second-derivative coefficients: gamma_b*T_b = 1 at p1, 2 at pN (N>=3).
"""
import sympy as sym


def make_rotor(N):
    dim = 2 * N - 1
    xs = sym.symbols(f"x1:{dim + 1}")            # x1..x_{2N-1}
    r = list(xs[: N - 1])
    p = list(xs[N - 1:])

    drift_r = [p[i + 1] - p[i] for i in range(N - 1)]
    if N == 2:
        drift_p = [sym.sin(r[0]) - p[0], -sym.sin(r[0])]
        noise = {N - 1: 1.0}                     # index of p1
    else:
        drift_p = [sym.sin(r[0]) - p[0]]
        for i in range(1, N - 1):
            drift_p.append(sym.sin(r[i]) - sym.sin(r[i - 1]))
        drift_p.append(-sym.sin(r[N - 2]) - p[N - 1])
        noise = {N - 1: 1.0, 2 * N - 2: 2.0}     # indices of p1, pN

    H = sum(pi**2 for pi in p) / 2 + sum(1 - sym.cos(ri) for ri in r)
    spec = {"noise": noise, "alpha": 3 if N == 5 else 1,
            "c": 0.05, "R": 10.0, "H": H}
    return list(xs), drift_r + drift_p, spec
