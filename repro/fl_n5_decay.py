"""Decay-curve measurement of the central-rotor dissipation law (C41 redo).
dp/dt = -kappa p^-k  =>  p(t)^{k+1} = p0^{k+1} - kappa(k+1) t.
Record ensemble-mean |p_c|(t); fit k from the curve family.
Control N=3 (proved k=3) then N=5 (conjectured k=7)."""
import numpy as np, time

def run(N, p0, T, dt=1e-3, ens=2000, rec=200, seed=0):
    rng = np.random.default_rng(seed)
    nb = N-1
    r = rng.uniform(-np.pi, np.pi, (ens, nb))
    p = np.zeros((ens, N))
    p[:, 0] = rng.normal(0, 1., ens); p[:, -1] = rng.normal(0, np.sqrt(2.), ens)
    c = N//2
    p[:, c] = p0
    s1 = np.sqrt(2*dt); sN = np.sqrt(4*dt)
    steps = int(T/dt); out = []
    trec = max(1, steps//rec)
    for i in range(steps):
        w = np.sin(r)
        dp = np.zeros_like(p)
        dp[:, 0] = (w[:, 0]-p[:, 0])*dt + s1*rng.standard_normal(ens)
        for j in range(1, N-1):
            dp[:, j] = (w[:, j]-w[:, j-1])*dt
        dp[:, -1] = (-w[:, -1]-p[:, -1])*dt + sN*rng.standard_normal(ens)
        r += (p[:, 1:]-p[:, :-1])*dt
        p += dp
        if i % trec == 0:
            m = np.mean(np.abs(p[:, c]))
            out.append((i*dt, m))
            if m < 0.35*p0:
                break
    return np.array(out)

def fit_k(curves):
    """Fit k from d(p)/dt = -kappa p^-k using log-log of -dp/dt vs p."""
    xs, ys = [], []
    for cu in curves:
        t, m = cu[:, 0], cu[:, 1]
        sl = np.gradient(m, t)
        sel = (m > 0.55*m[0]) & (sl < 0)     # early, clean part of the decay
        xs += list(np.log(m[sel])); ys += list(np.log(-sl[sel]))
    k = -np.polyfit(xs, ys, 1)[0]
    return k

for name, N, cfg in [("N=3", 3, [(3., 800.), (5., 8000.), (8., 60000.)]),
                     ("N=5", 5, [(2., 2000.), (3., 40000.), (4., 400000.)])]:
    print(f"== {name} ==", flush=True)
    curves = []
    for p0, T in cfg:
        t0 = time.time()
        cu = run(N, p0, T)
        curves.append(cu)
        half = cu[cu[:, 1] < 0.5*p0]
        th = half[0, 0] if len(half) else float("nan")
        print(f"  p0={p0}: t_half={th:.0f}  end={cu[-1,1]:.2f}@t={cu[-1,0]:.0f}  ({time.time()-t0:.0f}s)", flush=True)
        np.save(f"decay_{name.replace('=','')}_{p0}.npy", cu)
    print(f"  {name} fitted k = {fit_k(curves):.2f}  (theory/conjecture: N=3 -> 3, N=5 -> 7)", flush=True)
