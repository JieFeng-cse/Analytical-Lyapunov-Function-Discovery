"""Full-series V_ref calibration for N=2 (CONCLUSIONS C1/C3, doc eq.13-19).

Builds the COMPLETE reference certificate -- all four eq.(17) counterterms --
and computes, on the +-15 training box and the +-30 falsifier box,

    c*(theta, A, R) = min_{x in box \\ K_R}  -LV(x)*(alpha+1+log V(x))/V(x)

(0 if LV >= 0 anywhere outside K_R). Prints the feasible frontier.

Run:  python repro/fl_calibrate_full.py   (from a rotor2 run dir/symlink env)
"""
import itertools
import sys

import numpy as np
import sympy as sym

sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor  # noqa: E402
from utils.fl_reward import generator_expr  # noqa: E402
from utils.fl_scaffold import _chi  # noqa: E402

SEED, N_UNIF, N_BAND = 0, 300_000, 200_000
alpha = 1

sv, drift, spec = make_rotor(2)
r1, p1, p2 = sv

# eq.(17) counterterms (ptilde = p2 - t1*I1 - t2*I2 - t3*I3 + t4*I4)
t1 = sym.cos(r1)
t2 = p1 * sym.cos(r1)
t3 = p1**2 * sym.cos(r1) + p1 * sym.sin(r1) + sym.cos(r1)**2
t4 = (-p1**3 * sym.cos(r1) - 3 * p1**2 * sym.sin(r1) - 3 * p1 * sym.cos(r1)**2
      + p1 * sym.cos(r1) - sym.Rational(1, 2) * sym.sin(r1) * sym.cos(r1)
      + 2 * sym.sin(r1))

S = p1**4 + 4
rho = _chi(p2**2 / S**2)
I = lambda j: sym.Piecewise((rho / p2**j, p2**2 > 1), (0, True))
pt = p2 - t1 * I(1) - t2 * I(2) - t3 * I(3) + t4 * I(4)
H = spec["H"]

def sample(pbox):
    rng = np.random.default_rng(SEED)
    low = np.array([-np.pi, -pbox, -pbox]); high = -low
    Xu = rng.uniform(low, high, size=(N_UNIF, 3))
    Xb = rng.uniform(low, high, size=(N_BAND, 3))
    Xb[:, 1] = rng.uniform(-2.5, 2.5, N_BAND)                       # p1 small
    Xb[:, 2] = rng.choice([-1, 1], N_BAND) * rng.uniform(2.5, pbox, N_BAND)
    return np.vstack([Xu, Xb])

h_fn = sym.lambdify(sv, H, "numpy")
for box_name, pbox in [("train +-15", 15.0), ("falsifier +-30", 30.0)]:
    X = sample(pbox)
    args = [X[:, i] for i in range(3)]
    h = h_fn(*args)
    print(f"\n### box {box_name}")
    print(f"{'theta':>6} {'A':>6} {'R':>5} {'c*':>12} {'#LV>=0':>7} {'maxLV/V':>10}")
    for theta, A in itertools.product([0.05, 0.1, 0.2, 0.3], [5., 20., 50., 200.]):
        V = 1 + sym.exp(theta * H) + A * rho * pt**2 * sym.exp(theta / 2 * pt**2)
        LV = generator_expr(V, sv, drift, spec["noise"])
        with np.errstate(all="ignore"):
            v = np.asarray(sym.lambdify(sv, V, "numpy")(*args), dtype=np.float64)
            lv = np.asarray(sym.lambdify(sv, LV, "numpy")(*args), dtype=np.float64)
        bad = ~(np.isfinite(v) & np.isfinite(lv))
        for R in [20., 30., 40., 60.]:
            out = (h > R) & ~bad
            if not out.any():
                continue
            lv_o, v_o = lv[out], v[out]
            n_pos = int((lv_o >= 0).sum())
            if n_pos:
                print(f"{theta:>6} {A:>6} {R:>5} {'infeasible':>12} {n_pos:>7} "
                      f"{float((lv_o / v_o).max()):>10.3g}")
            else:
                cstar = float((-lv_o * (alpha + 1 + np.log(np.maximum(v_o, 1.))) / v_o).min())
                print(f"{theta:>6} {A:>6} {R:>5} {cstar:>12.3e} {0:>7} {'':>10}")
