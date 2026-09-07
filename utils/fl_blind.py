"""Blind skeleton search judge (spec 2026-09-07, CONCLUSIONS C45).

Candidate = pair(rho, ptilde). Assembled certificate

    V = 1 + e^{theta H} + A * rho * ptilde^2 * e^{theta ptilde^2 / 2}.

Everything is evaluated in log domain so that no exponential of a candidate
quantity is ever formed and nothing divides by rho:

    P = ptilde^2,  f = theta P / 2
    logV  = logsumexp(0, theta H, log A + log rho + log P + f)
    LV/V  = e^{theta H - logV} Q1 + A e^{f - logV} G2
    Q1 = theta LH + theta^2 sum_b k_b (d_b H)^2                     (= L e^{theta H} / e^{theta H})
    G2 = rho (LP + P (Lf + sum_b k_b (d_b f)^2) + 2 sum_b k_b d_b P d_b f)
         + P L rho + 2 sum_b k_b d_b rho (d_b P + P d_b f)         (= L(rho P e^f) / (A e^f))
    ratio = -(LV/V) (2 + logV)

The drift condition  LV <= -c V / (2 + log V)  holds at x  iff  ratio(x) >= c.
"""
import numpy as np
import sympy as sym

from utils.fl_reward import generator_expr


# ---- smooth step as a sympy Function with symbolic derivatives ------------------
# NB: class names must not collide with mpmath function names (sympy's
# Function._eval_evalf looks the lowercased name up in mpmath: "chi" is the
# hyperbolic cosine integral there). _eval_evalf is blocked explicitly too.
class SmoothStepD2(sym.Function):
    nargs = 1

    def _eval_evalf(self, prec):
        return None


class SmoothStepD1(sym.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return SmoothStepD2(self.args[0])

    def _eval_evalf(self, prec):
        return None


class SmoothStep(sym.Function):
    """step(z): 0 for z<=0, 1 for z>=1, 1/(1+exp(1/z - 1/(1-z))) inside."""
    nargs = 1

    def fdiff(self, argindex=1):
        return SmoothStepD1(self.args[0])

    def _eval_evalf(self, prec):
        return None


Chi, ChiD1, ChiD2 = SmoothStep, SmoothStepD1, SmoothStepD2


def _chi_inner_fns():
    z = sym.Symbol("z")
    inner = 1 / (1 + sym.exp(1 / z - 1 / (1 - z)))
    return tuple(sym.lambdify(z, sym.diff(inner, z, k), "numpy") for k in range(3))


_F0, _F1, _F2 = _chi_inner_fns()


def _chi_eval(z, fn, value_hi):
    z = np.asarray(z, dtype=np.float64)
    inner = (z > 0.0) & (z < 1.0)
    zi = np.where(inner, z, 0.5)
    with np.errstate(all="ignore"):
        val = np.asarray(fn(zi), dtype=np.float64)
    val = np.where(np.isfinite(val), val, 0.0)      # exp overflow deep in the tails -> 0
    return np.where(inner, val, np.where(z >= 1.0, value_hi, 0.0))


def chi_np(z):
    return _chi_eval(z, _F0, 1.0)


def chi1_np(z):
    return _chi_eval(z, _F1, 0.0)


def chi2_np(z):
    return _chi_eval(z, _F2, 0.0)


CHI_MODULES = {"SmoothStep": chi_np, "SmoothStepD1": chi1_np, "SmoothStepD2": chi2_np}
_CHI_IMPL = {Chi: chi_np, ChiD1: chi1_np, ChiD2: chi2_np}


def _to_chi(expr):
    """parse_expr yields an undefined Function('step'); swap in Chi."""
    return expr.replace(
        lambda e: isinstance(e, sym.Function) and e.func.__name__ == "step",
        lambda e: Chi(e.args[0]))


def split_pair(expr):
    """(rho, ptilde) with step -> Chi, or None if the root is not pair(.,.)."""
    if not isinstance(expr, sym.Function) or expr.func.__name__ != "pair" or len(expr.args) != 2:
        return None
    return _to_chi(expr.args[0]), _to_chi(expr.args[1])


# ---- reference certificate (document eq. 13-17) for calibration and tests -------
def reference_pair(order=4, R_rho=20):
    """pair(rho_doc, ptilde_doc) truncated after `order` counterterms, written
    with the step token (document chi ramps on z in [1,4]; step on [0,1])."""
    x1, x2, x3 = sym.symbols("x1 x2 x3")
    r1, p1, p2 = x1, x2, x3
    S = p1**4 + R_rho
    rho = sym.Function("step")((p2**2 / S**2 - 1) / 3)
    t = [sym.cos(r1),
         p1 * sym.cos(r1),
         p1**2 * sym.cos(r1) + p1 * sym.sin(r1) + sym.cos(r1)**2,
         -(-p1**3 * sym.cos(r1) - 3 * p1**2 * sym.sin(r1) - 3 * p1 * sym.cos(r1)**2
           + p1 * sym.cos(r1) - sym.Rational(1, 2) * sym.sin(r1) * sym.cos(r1)
           + 2 * sym.sin(r1))]
    pt = p2 - sum(t[j] * rho / p2**(j + 1) for j in range(order))
    return sym.Function("pair")(rho, pt)


# ---- the judge -------------------------------------------------------------------
"""Exemption set (candidate-dependent, C36 shape; calibrated numbers in spec["blind"]):

    K(rho) = {p1^2 <= M_max and rho < 1}  U  {p1^2 <= M_0 and |p2| <= R2_0}

The drift condition is scored on the float64-trusted domain
D = {|p1| <= P1max, |p2| <= P2max} outside K(rho). Compactness of K is enforced
by two shells on which rho must be identically 1:
    near shell  {p1^2 <= M_near, R2_near <= |p2| <= P2max}   (inside D: forces the
                fast branch to be scored there)
    far shell   {p1^2 <= M_max,  |p2| = R2_max}               (rho only, no drift)
Beyond P2max the drift of a good candidate is below float64 cancellation noise
(L ptilde ~ 1/(2 ptilde^3)); that region is covered by fl_blind_final_check at
high precision for candidates that already pass on D.
"""
RHO_ONE = 1.0 - 1e-9


def make_grid(P1max, P2max, n_r, n_p1, n_p2):
    """Product grid: r1 uniform, p1/p2 signed geometric plus 0."""
    r = np.linspace(-np.pi, np.pi, n_r)
    a = np.geomspace(0.25, P1max, n_p1)
    p1 = np.concatenate([-a[::-1], [0.0], a])
    b = np.geomspace(0.25, P2max, n_p2)
    p2 = np.concatenate([-b[::-1], [0.0], b])
    R, P1, P2 = np.meshgrid(r, p1, p2, indexing="ij")
    return np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)


class BlindJudge:
    KEYS = ("theta", "P1max", "P2max", "M_max", "R2_max", "M_near", "R2_near",
            "M_0", "R2_0", "c_t")

    def __init__(self, spec, state_vars, drift):
        b = spec["blind"]
        self.sv = list(state_vars)
        assert len(self.sv) == 3, "blind judge is calibrated for N=2 (dim 3)"
        self.drift = list(drift)
        self.noise = {int(k): float(v) for k, v in spec["noise"].items()}
        for k in self.KEYS:
            setattr(self, k, float(b[k]))
        self.A_grid = [float(a) for a in b.get("A_grid", [0.1, 1, 10, 100, 1000])]
        H = spec["H"]
        th = self.theta
        Q1 = th * generator_expr(H, self.sv, self.drift, self.noise) \
            + th**2 * sum(k * sym.diff(H, self.sv[i])**2 for i, k in self.noise.items())
        self._thH = self._lam(th * H)
        self._Q1 = self._lam(Q1)
        n = (13, 12, 20) if b.get("grid_coarse", False) else (25, 24, 40)
        self.set_grid(make_grid(self.P1max, self.P2max, *n))
        # far shell: rho only
        r = np.linspace(-np.pi, np.pi, n[0])
        a = np.geomspace(0.25, np.sqrt(self.M_max), n[1])
        p1 = np.concatenate([-a[::-1], [0.0], a])
        R, P1, P2 = np.meshgrid(r, p1, np.array([-self.R2_max, self.R2_max]), indexing="ij")
        self.far_shell = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)

    def set_grid(self, X):
        self.grid = np.asarray(X, dtype=np.float64)
        p1, p2 = self.grid[:, 1], self.grid[:, 2]
        self.box0 = (p1**2 <= self.M_0) & (np.abs(p2) <= self.R2_0)
        self.in_Mmax = p1**2 <= self.M_max
        self.near_shell = (p1**2 <= self.M_near) & (np.abs(p2) >= self.R2_near)

    def _lam(self, e):
        return sym.lambdify(self.sv, e, modules=[CHI_MODULES, "numpy"])

    # -- symbolic assembly (A-free) --
    def _pieces(self, rho, pt):
        sv, noise, th = self.sv, self.noise, self.theta
        d1 = [sym.diff(pt, s) for s in sv]
        r1 = [sym.diff(rho, s) for s in sv]
        Lpt = (sum(f * d for f, d in zip(self.drift, d1))
               + sum(k * sym.diff(pt, sv[b], 2) for b, k in noise.items()))
        Lrho = (sum(f * d for f, d in zip(self.drift, r1))
                + sum(k * sym.diff(rho, sv[b], 2) for b, k in noise.items()))
        P = pt**2
        f = th * P / 2
        dP = {b: 2 * pt * d1[b] for b in noise}
        df = {b: th * pt * d1[b] for b in noise}
        LP = 2 * pt * Lpt + 2 * sum(k * d1[b]**2 for b, k in noise.items())
        Lf = th * LP / 2
        sq = sum(k * df[b]**2 for b, k in noise.items())
        G2 = (rho * (LP + P * (Lf + sq) + 2 * sum(k * dP[b] * df[b] for b, k in noise.items()))
              + P * Lrho + 2 * sum(k * r1[b] * (dP[b] + P * df[b]) for b, k in noise.items()))
        return rho, P, f, G2

    def compile(self, expr):
        """Lambdified (rho, P, f, G2) or None if structurally invalid."""
        sp = split_pair(expr)
        if sp is None:
            return None
        rho, pt = sp
        if not (rho.free_symbols | pt.free_symbols) <= set(self.sv):
            return None
        if self.sv[2] not in pt.free_symbols:
            return None
        try:
            return [self._lam(e) for e in self._pieces(rho, pt)]
        except Exception:
            return None

    def _eval(self, fns, X):
        args = [X[:, i] for i in range(X.shape[1])]
        n = X.shape[0]
        with np.errstate(all="ignore"):
            out = [np.broadcast_to(np.asarray(fn(*args), dtype=np.float64), (n,)) for fn in fns]
        return out

    def fields(self, expr, fns=None):
        """A-independent numeric fields on the grid (+ rho on the far shell),
        or None if invalid: no pair root, foreign symbols, ptilde without p2,
        rho < 0 or non-finite anywhere, non-finite pieces where rho > 0."""
        fns = self.compile(expr) if fns is None else fns
        if fns is None:
            return None
        try:
            rho_v, P_v, f_v, G2_v = self._eval(fns, self.grid)
            thH, Q1 = self._eval([self._thH, self._Q1], self.grid)
            rho_far = self._eval(fns[:1], self.far_shell)[0]
        except Exception:
            return None
        if (np.any(~np.isfinite(rho_v)) or np.any(rho_v < 0)
                or np.any(~np.isfinite(rho_far)) or np.any(rho_far < 0)):
            return None
        active = rho_v > 0
        for arr in (P_v, f_v, G2_v):
            if np.any(~np.isfinite(arr[active])):
                return None
        with np.errstate(divide="ignore"):
            log_rhoP = np.where(active & (P_v > 0), np.log(rho_v) + np.log(P_v), -np.inf)
        return {"thH": thH, "Q1": Q1, "log_rhoP": log_rhoP, "rho": rho_v, "rho_far": rho_far,
                "f": np.where(active, f_v, 0.0), "G2": np.where(active, G2_v, 0.0)}

    def _weights(self, fl, A):
        """(logV, w1 = e^{thH}/V, A*w2 = A e^{f}/V), exponent differences taken
        relative to the dominant term so O(1e9) magnitudes never cancel."""
        a0 = np.zeros_like(fl["thH"])
        a1 = fl["thH"]
        la2 = np.log(A) + fl["log_rhoP"]              # -inf where rho*P == 0
        a2 = la2 + fl["f"]
        m = np.maximum(np.maximum(a0, a1), a2)
        with np.errstate(all="ignore"):
            e0, e1, e2 = np.exp(a0 - m), np.exp(a1 - m), np.exp(a2 - m)
            S = e0 + e1 + e2
            logV = m + np.log(S)
            w1 = e1 / S
            # A e^{f}/V = (e2/S) / (rho P): exactly 1/(rho P) where the third
            # term dominates, finite wherever rho P > 0.
            Aw2 = np.where(la2 > -np.inf, (e2 / S) * np.exp(-fl["log_rhoP"]), 0.0)
        return logV, w1, Aw2

    def logV(self, fl, A):
        return self._weights(fl, A)[0]

    def ratio(self, fl, A):
        lv, w1, Aw2 = self._weights(fl, A)
        with np.errstate(all="ignore"):
            LV_over_V = w1 * fl["Q1"] + Aw2 * fl["G2"]
        return -LV_over_V * (2.0 + lv)

    def exempt(self, fl):
        return (self.in_Mmax & (fl["rho"] < RHO_ONE)) | self.box0

    def score(self, fl, A):
        """(pen, parts) or None if the ratio is non-finite somewhere."""
        r = self.ratio(fl, A)
        if not np.all(np.isfinite(r)):
            return None
        ex = self.exempt(fl)
        rc = np.clip(r, -1e3, np.inf)
        v_out = float(np.mean(np.maximum(self.c_t - rc[~ex], 0.0)) / self.c_t)
        v_in = float(np.mean(r[ex] < self.c_t)) if ex.any() else 0.0
        v_comp = 0.5 * (float(np.mean(fl["rho"][self.near_shell] < RHO_ONE))
                        + float(np.mean(fl["rho_far"] < RHO_ONE)))
        return v_out + v_comp + 0.1 * v_in, (v_out, v_comp, v_in)

    def penalty(self, expr):
        """(pen, A, ok): best score over the A grid."""
        fl = self.fields(expr)
        if fl is None:
            return None, None, False
        best = None
        for A in self.A_grid:
            s = self.score(fl, A)
            if s is not None and (best is None or s[0] < best[0]):
                best = (s[0], A, s[1])
        if best is None:
            return None, None, False
        self.last_parts = best[2]
        return float(best[0]), float(best[1]), True

    # -- point-wise helpers (tests / reconnaissance) --
    def ratio_at(self, expr, pts, A):
        saved = self.grid
        try:
            self.set_grid(pts)
            fl = self.fields(expr)
            return None if fl is None else self.ratio(fl, A)
        finally:
            self.set_grid(saved)

    def ratio_mpmath(self, expr, pts, A, dps=40):
        """Independent unscaled evaluation of V, LV with sympy at `dps` digits.
        Chi and its derivatives are evaluated numerically (same float impl as
        the fast path); the check is on the log-domain assembly algebra."""
        rho, pt = split_pair(expr)
        th = sym.Rational(self.theta).limit_denominator(10**6)
        H = sum(s**2 for s in self.sv[1:]) / 2 + (1 - sym.cos(self.sv[0]))
        V = 1 + sym.exp(th * H) + A * rho * pt**2 * sym.exp(th * pt**2 / 2)
        LV = generator_expr(V, self.sv, self.drift, self.noise)

        def num(e, sub):
            e = e.xreplace(sub)
            e = e.replace(lambda q: isinstance(q, (Chi, ChiD1, ChiD2)),
                          lambda q: sym.Float(float(_CHI_IMPL[q.func](float(q.args[0]))), dps))
            return sym.N(e, dps)

        out = []
        for p in pts:
            sub = {s: sym.Float(float(v), dps) for s, v in zip(self.sv, p)}
            v, lv = num(V, sub), num(LV, sub)
            try:
                if v.is_number and v.is_real and bool(v > 0):
                    out.append(float(sym.N(-lv * (2 + sym.log(v)) / v, dps)))
                else:
                    out.append(np.nan)
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)


# ---- high-precision far-region evaluation (final check only) ---------------------
def _chi_mp_fns():
    import mpmath as mp
    z = sym.Symbol("z")
    inner = 1 / (1 + sym.exp(1 / z - 1 / (1 - z)))
    fs = [sym.lambdify(z, sym.diff(inner, z, k), "mpmath") for k in range(3)]

    def make(k):
        def fn(zz):
            zz = mp.mpf(zz)
            if zz <= 0:
                return mp.mpf(0)
            if zz >= 1:
                return mp.mpf(1 if k == 0 else 0)
            return fs[k](zz)
        return fn
    return {"SmoothStep": make(0), "SmoothStepD1": make(1), "SmoothStepD2": make(2)}


def ratio_mp_points(judge, expr, pts, A, dps=30):
    """ratio(x) at each row of pts using mpmath at `dps` digits, in the same
    log-domain assembly as the float path (so only precision differs)."""
    import mpmath as mp
    mp.mp.dps = dps
    rho, pt = split_pair(expr)
    rho_e, P_e, f_e, G2_e = judge._pieces(rho, pt)
    th = judge.theta
    H = sum(s**2 for s in judge.sv[1:]) / 2 + (1 - sym.cos(judge.sv[0]))
    Q1_e = th * generator_expr(H, judge.sv, judge.drift, judge.noise) \
        + th**2 * sum(k * sym.diff(H, judge.sv[i])**2 for i, k in judge.noise.items())
    mods = [_chi_mp_fns(), "mpmath"]
    fns = [sym.lambdify(judge.sv, e, modules=mods) for e in (rho_e, P_e, f_e, G2_e, th * H, Q1_e)]
    out = np.empty(len(pts))
    lA = mp.log(A)
    for i, p in enumerate(pts):
        args = [mp.mpf(float(v)) for v in p]
        try:
            rho_v, P_v, f_v, G2_v, thH, Q1 = [fn(*args) for fn in fns]
        except (ZeroDivisionError, ValueError):
            out[i] = np.nan
            continue
        if rho_v < 0:
            out[i] = np.nan
            continue
        if rho_v > 0 and P_v > 0:
            a2 = lA + mp.log(rho_v) + mp.log(P_v) + f_v
            V = 1 + mp.exp(thH) + mp.exp(a2)
            LV_over_V = (mp.exp(thH) * Q1 + A * mp.exp(f_v) * G2_v) / V
        else:
            V = 1 + mp.exp(thH)
            LV_over_V = mp.exp(thH) * Q1 / V
        out[i] = float(-LV_over_V * (2 + mp.log(V)))
    return out


def fl_blind_final_check(expr, spec, state_vars, drift, A, seed=0):
    """Final gate. (i) float64: 2x denser grid on D plus n_random uniform
    points in D, outside K(rho); (ii) mpmath (dps digits): a coarse far grid
    |p2| in [P2max, 4 R2_max] for |p1| <= P1max, outside K(rho). Pass iff
    ratio >= c_t at every scored point. Returns (ok, worst_point)."""
    b = dict(spec["blind"])
    j = BlindJudge(spec, state_vars, drift)
    fns = j.compile(expr)
    if fns is None:
        return False, None
    worst, worst_r = None, np.inf

    def scan(X, ratio_fn):
        nonlocal worst, worst_r
        j.set_grid(X)
        fl = j.fields(expr, fns=fns)
        if fl is None:
            return False
        r = ratio_fn(fl)
        bad = ~j.exempt(fl) & ~(r >= j.c_t)
        shell_bad = (j.near_shell & (fl["rho"] < RHO_ONE))
        bad = bad | shell_bad
        if bad.any():
            i = int(np.argmin(np.where(bad, r, np.inf)))
            if r[i] < worst_r or worst is None:
                worst_r, worst = r[i], j.grid[i].copy()
        return True

    parts = [make_grid(j.P1max, j.P2max, 49, 48, 80)] if b.get("dense", True) else [j.grid]
    n_random = int(b.get("n_random", 10_000_000))
    if n_random > 0:
        rng = np.random.default_rng(seed)
        parts.append(rng.uniform([-np.pi, -j.P1max, -j.P2max], [np.pi, j.P1max, j.P2max],
                                 size=(n_random, 3)))
    for X in parts:
        for lo in range(0, X.shape[0], 1_000_000):
            if not scan(X[lo:lo + 1_000_000], lambda fl: j.ratio(fl, A)):
                return False, None
    if np.any(j._eval(fns[:1], j.far_shell)[0] < RHO_ONE):
        return False, j.far_shell[0]
    # far region, high precision
    n_far = b.get("far_grid", (13, 12, 12))
    r = np.linspace(-np.pi, np.pi, n_far[0])
    a = np.geomspace(0.25, j.P1max, n_far[1]); p1 = np.concatenate([-a[::-1], [0.0], a])
    p2 = np.geomspace(j.P2max, 4.0 * j.R2_max, n_far[2]); p2 = np.concatenate([-p2[::-1], p2])
    R, P1, P2 = np.meshgrid(r, p1, p2, indexing="ij")
    far = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
    j.set_grid(far)
    rho_far = j._eval(fns[:1], far)[0]
    ex = (j.in_Mmax & (rho_far < RHO_ONE)) | j.box0
    pts = far[~ex]
    if len(pts):
        rf = ratio_mp_points(j, expr, pts, A, dps=int(b.get("dps", 30)))
        bad = ~(rf >= j.c_t)
        if bad.any():
            i = int(np.argmin(np.where(bad, rf, np.inf)))
            if rf[i] < worst_r or worst is None:
                worst_r, worst = rf[i], pts[i].copy()
    return worst is None, worst


_JUDGES = {}


def get_judge(spec, state_vars, drift):
    b = spec["blind"]
    key = (len(state_vars), tuple(sorted((k, str(v)) for k, v in b.items())))
    if key not in _JUDGES:
        _JUDGES[key] = BlindJudge(spec, state_vars, drift)
    return _JUDGES[key]


def fl_blind_penalty(expr, spec, state_vars, drift):
    return get_judge(spec, state_vars, drift).penalty(expr)
