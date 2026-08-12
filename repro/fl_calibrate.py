"""V_ref-based calibration of scaffold/benchmark constants (CONCLUSIONS C1/C3).

For each (theta, A, R) on a small grid, plug the KNOWN stage-1 slot answer into
the scaffold and compute, over a dense sample of box\\K_R,

    c*(theta, A, R) = min_x  -LV(x) * (alpha+1+log V(x)) / V(x)     (0 if any LV>=0)

i.e. the largest drift constant the reference construction can certify on the
finite box. Report the best combos. Pure CPU, minutes.

Usage:  python repro/fl_calibrate.py <N>     (run from a rotor run dir / template env)
"""
import itertools
import sys

import numpy as np
import sympy as sym

sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor  # noqa: E402
from utils.fl_reward import generator_expr  # noqa: E402
from utils.fl_scaffold import Scaffold, _build_cfg  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SEED, N_UNIF, N_BAND = 0, 300_000, 100_000
P_BOX = 15.0

sv, drift, spec = make_rotor(N)
dim = len(sv)
n_ang = N - 1
alpha = spec["alpha"]

# stage-1 known slot answers (bond-local motif)
r = sv[:n_ang]
if N == 2:
    g_answer = sym.cos(r[0])
else:
    g_answer = sym.cos(r[0]) + sym.cos(r[1])   # N=3 (and N=4 local pattern)

rng = np.random.default_rng(SEED)
low = np.array([-np.pi] * n_ang + [-P_BOX] * (dim - n_ang))
high = -low
X_u = rng.uniform(low, high, size=(N_UNIF, dim))
# adversarial band: small endpoint momenta, fast momentum in the gate-transition zone
X_b = rng.uniform(low, high, size=(N_BAND, dim))
X_b[:, n_ang] = rng.uniform(-2, 2, N_BAND)                    # p1 small
X_b[:, n_ang + 1] = rng.choice([-1, 1], N_BAND) * rng.uniform(3, 12, N_BAND)  # p2 in band
X = np.vstack([X_u, X_b])
args = [X[:, i] for i in range(dim)]

h_fn = sym.lambdify(sv, spec["H"], "numpy")
h = h_fn(*args)

results = []
for theta, A in itertools.product([0.05, 0.1, 0.2], [5.0, 20.0, 50.0, 100.0]):
    s = dict(spec)
    s["scaffold"] = {"theta": theta, "A": A}
    scaf = Scaffold(sv, drift, spec["noise"], _build_cfg(s, sv))
    V, LV = scaf.assemble(g_answer)
    with np.errstate(all="ignore"):
        v = np.asarray(sym.lambdify(sv, V, "numpy")(*args), dtype=np.float64)
        lv = np.asarray(sym.lambdify(sv, LV, "numpy")(*args), dtype=np.float64)
    bad = ~(np.isfinite(v) & np.isfinite(lv))
    for R in [20.0, 30.0, 40.0, 50.0, 60.0]:
        out = (h > R) & ~bad
        if not out.any():
            continue
        lv_o, v_o = lv[out], v[out]
        n_pos = int((lv_o >= 0).sum())
        if n_pos:
            worst = float(lv_o.max())
            results.append((theta, A, R, 0.0, n_pos, worst))
            continue
        cstar = float((-lv_o * (alpha + 1.0 + np.log(np.maximum(v_o, 1.0))) / v_o).min())
        results.append((theta, A, R, cstar, 0, 0.0))

results.sort(key=lambda t: -t[3])
print(f"N={N}  answer g = {g_answer}")
print(f"{'theta':>6} {'A':>6} {'R':>5} {'c*':>12} {'#LV>=0':>7} {'worst LV':>10}")
for theta, A, R, cstar, n_pos, worst in results[:15]:
    print(f"{theta:>6} {A:>6} {R:>5} {cstar:>12.3e} {n_pos:>7} {worst:>10.3g}")
