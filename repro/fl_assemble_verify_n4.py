# N=4 four-piece assembled-certificate verification (CONCLUSIONS C40).
# Requires mp_htc.pkl produced by the fl_multiphase engine (see C39).
import numpy as np, sympy as sym, sys, pickle, time
sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr
from utils.fl_scaffold import _chi

t0 = time.time()
sv, drift, spec = make_rotor(4); r1, r2, r3, p1, p2, p3, p4 = sv
noise = spec["noise"]
Htc = sym.sympify(pickle.load(open("mp_htc.pkl", "rb")))
th = 0.02; g1, g4, T1, T4 = 1., 1., 1., 2.
F2 = th**2*g1*T1*(2*sym.sin(r1)-sym.sin(2*r1)/4)
F3 = th**2*g4*T4*(2*sym.sin(r3)-sym.sin(2*r3)/4)
Vc = Htc*sym.exp(th*Htc)*(1+F2/p2**3+F3/p3**3)

def chi01(z, lo, hi):
    b = lambda u: sym.Piecewise((sym.exp(-1/u), u > 0), (0, True))
    zz = (z-lo)/(hi-lo); return b(zz)/(b(zz)+b(1-zz))

psi_c = (chi01((p2**2+p3**2)/((p1**2+p4**2)**4+20.), 1., 4.)
         * chi01(p3**6/(p2**2+1), 1., 4.) * chi01(p2**6/(p3**2+1), 1., 4.))

def sf_patch(fast, S, ts):
    rho = _chi(fast**2/S**2)
    I = lambda j: sym.Piecewise((rho/fast**j, fast**2 > 1), (0, True))
    pt = fast - sum(t*I(j+1) for j, t in enumerate(ts))
    return rho*pt**2*sym.exp(th/2*pt**2)

t0s = [sym.cos(r1)+sym.cos(r2), p1*sym.cos(r1)+p3*sym.cos(r2),
       (sym.cos(r1)**2+sym.cos(r2)**2+p1**2*sym.cos(r1)+p1*sym.sin(r1)
        + p3**2*sym.cos(r2)+sym.sin(r1)*sym.sin(r2)+sym.sin(r2)*sym.sin(r3))]
t1s = [sym.cos(r2)+sym.cos(r3), p2*sym.cos(r2)+p4*sym.cos(r3),
       (p2**2*sym.cos(r2)+p4**2*sym.cos(r3)-p4*sym.sin(r3)
        + sym.cos(r2)**2+sym.cos(r3)**2+sym.cos(r2)*sym.cos(r3)+sym.sin(r1)*sym.sin(r2))]
patch2 = sf_patch(p2, (p1**2+p3**2)**2+20., t0s)
patch3 = sf_patch(p3, (p2**2+p4**2)**2+20., t1s)
H = spec["H"]; main = sym.exp(th*H)

# product rule: L(psi*Vc) = psi*LVc + Vc*Lpsi + sum_b 2 kb dpsi_b dVc_b
print("diff main/patches ...", flush=True)
Lmain = generator_expr(main, sv, drift, noise)
Lp2 = generator_expr(patch2, sv, drift, noise)
Lp3 = generator_expr(patch3, sv, drift, noise)
print("diff Vc ...", round(time.time()-t0, 1), flush=True)
LVc = generator_expr(Vc, sv, drift, noise)
Lpsi = generator_expr(psi_c, sv, drift, noise)
cross = sum(2*k*sym.diff(psi_c, sv[b])*sym.diff(Vc, sv[b]) for b, k in noise.items())
print("lambdify ...", round(time.time()-t0, 1), flush=True)
F = {k: sym.lambdify(sv, e, "numpy") for k, e in dict(
    main=main, p2f=patch2, p3f=patch3, Vc=Vc, psi=psi_c,
    Lmain=Lmain, Lp2=Lp2, Lp3=Lp3, LVc=LVc, Lpsi=Lpsi, cross=cross,
    h=H).items()}
fr2 = sym.lambdify(sv, _chi(p2**2/((p1**2+p3**2)**2+20.)**2), "numpy")
fr3 = sym.lambdify(sv, _chi(p3**2/((p2**2+p4**2)**2+20.)**2), "numpy")
print("sampling ...", round(time.time()-t0, 1), flush=True)
rng = np.random.default_rng(0); pbox = 15.
Xu = rng.uniform([-np.pi]*3+[-pbox]*4, [np.pi]*3+[pbox]*4, size=(400000, 7))
Xb = Xu.copy()[:200000]; Xb[:, 3] = rng.uniform(-2, 2, 200000); Xb[:, 6] = rng.uniform(-2, 2, 200000)
X = np.vstack([Xu, Xb]); a = [X[:, i] for i in range(7)]
Mw = 50.
with np.errstate(all="ignore"):
    vals = {k: np.asarray(f(*a), dtype=np.float64) for k, f in F.items()}
    rr2 = fr2(*a); rr3 = fr3(*a)
# guard: Vc pieces only defined where p2^2+p3^2>10 (denominators); psi_c support inside
safe = (X[:, 4]**2 + X[:, 5]**2) > 10
psi = np.where(safe, vals["psi"], 0.0)
Vcv = np.where(safe, vals["Vc"], 0.0)
LVcv = np.where(safe, vals["LVc"], 0.0)
Lpsiv = np.where(safe, vals["Lpsi"], 0.0)
crossv = np.where(safe, vals["cross"], 0.0)
v = 1 + vals["main"] + 20*vals["p2f"] + 20*vals["p3f"] + Mw*psi*Vcv
lv = vals["Lmain"] + 20*vals["Lp2"] + 20*vals["Lp3"] + Mw*(psi*LVcv + Vcv*Lpsiv + crossv)
h = vals["h"]
ok = np.isfinite(v) & np.isfinite(lv)
diag = np.abs(np.abs(X[:, 4]) - np.abs(X[:, 5]))**2 < np.abs(X[:, 4] + X[:, 5]) + 4
for M in (9., 36.):
    end = X[:, 3]**2 + X[:, 6]**2   # paper's endpoint norm p1^2+p4^2
    nogate = (rr2 < 1-1e-9) & (rr3 < 1-1e-9) & (psi < 1-1e-9)
    K = (h <= 20.) | (nogate & (end <= M)) | ((psi >= 1e-9) & (X[:, 3]**2 + X[:, 6]**2 <= M))
    for name, extra in [("base", np.zeros(len(X), bool)), ("diag-exempt", diag)]:
        out = ok & ~(K | extra)
        npos = int((lv[out] >= 0).sum())
        if npos == 0:
            c = float((-lv[out]*(2+np.log(np.maximum(v[out], 1)))/v[out]).min())
            print(f"M={M} [{name}]: FEASIBLE c*={c:.3e} (checked {int(out.sum())})", flush=True)
        else:
            P = X[out][lv[out] >= 0]
            bf = np.mean((np.abs(P[:, 4]) > 6) & (np.abs(P[:, 5]) > 6))
            dg = np.mean(np.abs(np.abs(P[:, 4]) - np.abs(P[:, 5])) < 3)
            print(f"M={M} [{name}]: pos={npos}/{int(out.sum())} both-fast={bf:.0%} near-diag={dg:.0%}", flush=True)
print("total", round(time.time()-t0, 1), "s")
