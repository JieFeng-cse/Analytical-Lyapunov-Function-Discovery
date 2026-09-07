"""Calibrate the blind judge on the document's rotor2 certificate at theta=0.1
(spec 2026-09-07 section 4, K shape as in utils/fl_blind.py). Writes
example_config/blind_rotor2.json and prints a C44-style table.

Run from the repo root in the alfd env:  python repro/blind_calibrate_rotor2.py
"""
import datetime
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor          # noqa: E402
from utils.fl_blind import (BlindJudge, reference_pair, make_grid,   # noqa: E402
                            ratio_mp_points, fl_blind_final_check, RHO_ONE)

THETA, A_REF, R_RHO = 0.1, 20.0, 20
P2MAX = 4e4            # float64 trust limit for the fast-branch drift (design constant)
M_NEAR = 64.0          # near shell |p1| <= 8 (design constant)
SCALE = 1.25
OUT = "example_config/blind_rotor2.json"

sv, drift, spec = make_rotor(2)
spec = dict(spec)
expr = reference_pair(order=4, R_rho=R_RHO)
t0 = time.time()

# provisional judge just to get the machinery; numbers overwritten below
spec["blind"] = {"theta": THETA, "P1max": 60.0, "P2max": P2MAX, "M_max": 1e4, "R2_max": 1e8,
                 "M_near": M_NEAR, "R2_near": 1e4, "M_0": 1.0, "R2_0": 1.0, "c_t": 1e-4,
                 "A_grid": [A_REF]}
j = BlindJudge(spec, sv, drift)
fns = j.compile(expr)
assert fns is not None

# (a) float64 domain D0: bulk drift rate away from the core, then the core box
j.set_grid(make_grid(60.0, P2MAX, 25, 36, 60))
fl = j.fields(expr, fns=fns)
r = j.ratio(fl, A_REF)
X, rho = j.grid, fl["rho"]
p1, p2 = X[:, 1], X[:, 2]
band = (rho > 0) & (rho < RHO_ONE)
core_bad = (r < 0) & ~band
M_0_zero = float(np.max(p1[core_bad]**2)) if core_bad.any() else 0.0
R2_0_zero = float(np.max(np.abs(p2[core_bad]))) if core_bad.any() else 0.0
bulk = ~band & ((p1**2 > 4 * M_0_zero) | (np.abs(p2) > 4 * R2_0_zero))
c_bulk_D = float(np.min(r[bulk]))
print(f"(a) D0 zero-level core box: M={M_0_zero:.3f} R2={R2_0_zero:.1f}; bulk min ratio on D0 = {c_bulk_D:+.4e}"
      f"   [{time.time()-t0:.0f}s]", flush=True)

# (c) near shell: smallest |p2| beyond which rho == 1 for all |p1| <= sqrt(M_NEAR)
levels = np.geomspace(10, 1e7, 200)
def shell_level(p1max):
    Pr = np.linspace(-p1max, p1max, 41)
    R, P1, P2 = np.meshgrid([0.0], Pr, levels, indexing="ij")
    G = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
    rv = j._eval(fns, G)[0].reshape(len(Pr), len(levels))
    # for each p1 the first level from which rho == 1 for all larger levels
    first = []
    for i in range(len(Pr)):
        one = rv[i] >= RHO_ONE
        # last index where rho < 1, +1
        idx = np.where(~one)[0]
        first.append(levels[idx[-1] + 1] if len(idx) and idx[-1] + 1 < len(levels) else levels[0])
    return float(np.max(first))
R2_near_ref = shell_level(np.sqrt(M_NEAR))
print(f"(c) near shell: rho==1 for |p1|<={np.sqrt(M_NEAR):.0f}, |p2|>={R2_near_ref:.1f}", flush=True)
assert SCALE * R2_near_ref < P2MAX, "near shell must sit inside the float64 domain"

# (b) transition-band threshold in p1 (mpmath): last p1 whose band S<|p2|<2S violates
p1s = np.arange(10.0, 40.5, 1.0)
r1s = np.linspace(-np.pi, np.pi, 13)
thr = None
band_min = {}
for p1v in p1s:
    S = p1v**4 + R_RHO
    p2s = np.geomspace(S * 1.001, 2 * S * 0.999, 12)
    R, P2 = np.meshgrid(r1s, p2s, indexing="ij")
    pts = np.stack([R.ravel(), np.full(R.size, p1v), P2.ravel()], 1)
    rr = ratio_mp_points(j, expr, pts, A_REF, dps=30)
    band_min[p1v] = float(np.nanmin(rr))
    if band_min[p1v] < 0:
        thr = p1v
print("(b) transition band min ratio by p1:", {k: f"{v:+.2e}" for k, v in band_min.items()}, flush=True)
assert thr is not None
M_max_ref = float(thr**2)
print(f"(b) last violating p1 = {thr} -> M_max_ref = {M_max_ref:.0f}   [{time.time()-t0:.0f}s]", flush=True)

# far-region bulk rate (mpmath) for |p1| <= 60 outside the band exemption
M_max = (SCALE * thr)**2
R2_max_ref = shell_level(np.sqrt(M_max))
R2_max = SCALE * R2_max_ref
P1max = 1.5 * np.sqrt(M_max)
print(f"(d) far shell: rho==1 for |p1|<={np.sqrt(M_max):.1f}, |p2|>={R2_max_ref:.3e}", flush=True)
n_far = (13, 12, 12)
rr = np.linspace(-np.pi, np.pi, n_far[0])
a = np.geomspace(0.25, P1max, n_far[1]); pp1 = np.concatenate([-a[::-1], [0.0], a])
pp2 = np.geomspace(P2MAX, 4.0 * R2_max, n_far[2]); pp2 = np.concatenate([-pp2[::-1], pp2])
R, P1, P2 = np.meshgrid(rr, pp1, pp2, indexing="ij")
far = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
j.set_grid(far)
rho_far = j._eval(fns, far)[0]
wit = np.where(j.in_Mnear, j._eval(fns, j.wit_near)[0] >= RHO_ONE, j._eval(fns, j.wit_far)[0] >= RHO_ONE)
ex = (far[:, 1]**2 <= M_max) & (rho_far < RHO_ONE) & wit
rf = ratio_mp_points(j, expr, far[~ex], A_REF, dps=30)
c_bulk_far = float(np.nanmin(rf))
print(f"(e) far region outside band-K (mpmath): n={int((~ex).sum())} min ratio={c_bulk_far:+.4e}   [{time.time()-t0:.0f}s]", flush=True)
c_ref = min(c_bulk_D, c_bulk_far)
assert c_ref > 0
c_t = 0.5 * c_ref
# dense core scan at the c_t level: smallest box containing {ratio < c_t} outside the band
Pd1 = np.linspace(-2 * np.sqrt(M_0_zero) - 1, 2 * np.sqrt(M_0_zero) + 1, 161)
Pd2 = np.linspace(-2 * R2_0_zero - 50, 2 * R2_0_zero + 50, 321)
R, P1, P2 = np.meshgrid(np.linspace(-np.pi, np.pi, 49), Pd1, Pd2, indexing="ij")
dense = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
rng = np.random.default_rng(0)
dense = np.vstack([dense, rng.uniform([-np.pi, Pd1[0], Pd2[0]], [np.pi, Pd1[-1], Pd2[-1]], size=(500_000, 3))])
j.set_grid(dense)
fld = j.fields(expr, fns=fns)
rd = j.ratio(fld, A_REF)
bandd = (fld["rho"] > 0) & (fld["rho"] < RHO_ONE)
cb = (rd < c_t) & ~bandd
M_0_ref = float(np.max(dense[cb, 1]**2)) if cb.any() else 0.0
R2_0_ref = float(np.max(np.abs(dense[cb, 2]))) if cb.any() else 0.0
assert M_0_ref < 2 * M_0_zero + 1 and R2_0_ref < 2 * R2_0_zero + 40, "core box grew beyond the dense scan window"
print(f"(a') dense core box at the c_t={c_t:.3e} level: M_0_ref={M_0_ref:.3f} R2_0_ref={R2_0_ref:.1f}   [{time.time()-t0:.0f}s]", flush=True)

# final numbers
M_max_ref = float(thr**2)
M_0 = max(SCALE**2 * M_0_ref, 1.0)
R2_0 = max(SCALE * R2_0_ref, 1.0)
R2_near = SCALE * R2_near_ref

blind = {"theta": THETA, "P1max": P1max, "P2max": P2MAX, "M_max": M_max, "R2_max": R2_max,
         "M_near": M_NEAR, "R2_near": R2_near, "M_0": M_0, "R2_0": R2_0, "c_t": c_t,
         "A_grid": [0.1, 1, 10, 100, 1000]}
spec["blind"] = blind
j = BlindJudge(spec, sv, drift)
fl = j.fields(expr, fns=fns)
r = j.ratio(fl, A_REF)
out = ~j.exempt(fl)
c_D = float(np.min(r[out]))
print(f"(e) D outside K: n={int(out.sum())} min ratio={c_D:+.4e} (c_t={c_t:.3e}); near shell rho<1: "
      f"{int((fl['rho'][j.near_shell] < RHO_ONE).sum())}; witness fails: {int((~fl['witness'] & j.in_Mmax).sum())}", flush=True)
blind.update({"M_0_ref": M_0_ref, "R2_0_ref": R2_0_ref, "M_max_ref": M_max_ref, "R2_near_ref": R2_near_ref,
              "R2_max_ref": R2_max_ref, "c_ref": c_ref, "c_bulk_D": c_bulk_D, "c_bulk_far": c_bulk_far,
              "A_ref": A_REF, "scale": SCALE,
              "calibrated_on": datetime.date.today().isoformat()})
spec["blind"] = dict(blind, n_random=1_000_000)
ok, worst = fl_blind_final_check(expr, spec, sv, drift, A=A_REF, seed=0)
print(f"(f) final check on the reference with the calibrated numbers: ok={ok} worst={worst}   [{time.time()-t0:.0f}s]", flush=True)
assert ok
json.dump(blind, open(OUT, "w"), indent=2)
print("wrote", OUT)
print(json.dumps(blind, indent=2))
