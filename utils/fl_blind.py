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

    K(rho) = {p1^2 <= M_max and rho < 1 and WITNESS}  U  {p1^2 <= M_0 and |p2| <= R2_0}

WITNESS(x): on the same (r1, p1, sign p2) line the gate reaches 1 at the witness
    shell |p2| = R2_near (if p1^2 <= M_near) else |p2| = R2_max, AND x lies in the
    LAST rho<1 band below that contiguous rho == 1 tail (walking down from the
    witness: ones ... ones | band (exempt) | anything below: not exempt),
so a vanishing or infinitesimal gate exempts nothing. The drift condition is
scored on the float64-trusted domain D = {|p1| <= P1max, |p2| <= P2max}
outside K(rho). Compactness of K is enforced by requiring rho == 1 on the near
shell {p1^2 <= M_near, R2_near <= |p2| <= P2max} and on both witness shells.
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
        n = (13, 12, 20) if b.get("grid_coarse", False) else tuple(b.get("grid_n", (17, 16, 32)))
        self.set_grid(make_grid(self.P1max, self.P2max, *n), shape=(n[0], 2 * n[1] + 1, 2 * n[2] + 1))

    def set_grid(self, X, shape=None):
        """shape=(n_r, n_p1, n_p2) marks a make_grid() product grid (exact band
        scan along p2); otherwise the ladder scan is used for the exemption."""
        self.grid = np.asarray(X, dtype=np.float64)
        self.grid_shape = tuple(shape) if shape is not None else None
        p1, p2 = self.grid[:, 1], self.grid[:, 2]
        self.box0 = (p1**2 <= self.M_0) & (np.abs(p2) <= self.R2_0)
        self.in_Mmax = p1**2 <= self.M_max
        self.in_Mnear = p1**2 <= self.M_near
        self.near_shell = self.in_Mnear & (np.abs(p2) >= self.R2_near)
        # witness points: same (r1, p1), p2 moved to the witness shell (sign kept)
        sgn = np.where(p2 < 0, -1.0, 1.0)
        self.wit_near = np.column_stack([self.grid[:, 0], p1, sgn * self.R2_near])
        self.wit_far = np.column_stack([self.grid[:, 0], p1, sgn * self.R2_max])

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
        """One lambdified function returning the primitive fields
        [rho, d rho/dx_i (all i), d2 rho/dp_b^2 (noise b), pt, d pt/dx_i, d2 pt/dp_b^2]
        or None if structurally invalid. G2 is assembled numerically in
        fields(): the symbolic product form is 10-50x slower to codegen."""
        sp = split_pair(expr)
        if sp is None:
            return None
        rho, pt = sp
        if not (rho.free_symbols | pt.free_symbols) <= set(self.sv):
            return None
        if self.sv[2] not in pt.free_symbols:
            return None
        try:
            prims = []
            for e in (rho, pt):
                prims.append(e)
                prims.extend(sym.diff(e, s) for s in self.sv)
                prims.extend(sym.diff(e, self.sv[b], 2) for b in self.noise)
            return sym.lambdify(self.sv, prims, modules=[CHI_MODULES, "numpy"])
        except Exception:
            return None

    def is_periodic(self, fn, n=96, tol=1e-8):
        """rho and ptilde must be 2*pi-periodic in the bond angle (periodicity
        in q is part of the problem statement; a term linear in r1 fakes
        negative drift through dr1/dt = p2 - p1)."""
        rng = np.random.default_rng(12345)
        X = np.column_stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-8, 8, n),
                             rng.uniform(-50, 50, n)])
        Xs = X.copy()
        Xs[:, 0] += 2 * np.pi
        try:
            a = self._eval(fn, X)
            b = self._eval(fn, Xs)
        except Exception:
            return False
        k = 1 + len(self.sv) + len(self.noise)
        for i in (0, k):                      # rho, ptilde
            with np.errstate(all="ignore"):
                d = np.abs(a[i] - b[i])
                ok = np.isfinite(a[i]) & np.isfinite(b[i])
                if np.any(d[ok] > tol * (1.0 + np.abs(a[i][ok]))):
                    return False
        return True

    def _static(self, X):
        """Grid-only fields (theta H, Q1, drift_i), cached per grid object."""
        key = (id(X), X.shape)
        st = getattr(self, "_static_cache", None)
        if st is None or st[0] != key:
            args = [X[:, i] for i in range(X.shape[1])]
            n = X.shape[0]
            with np.errstate(all="ignore"):
                thH = np.broadcast_to(np.asarray(self._thH(*args), dtype=np.float64), (n,))
                Q1 = np.broadcast_to(np.asarray(self._Q1(*args), dtype=np.float64), (n,))
                dr = [np.broadcast_to(np.asarray(sym.lambdify(self.sv, d, "numpy")(*args),
                                                 dtype=np.float64), (n,)) for d in self.drift]
            self._static_cache = (key, thH, Q1, dr)
        return self._static_cache[1:]

    def _eval(self, fn, X):
        args = [X[:, i] for i in range(X.shape[1])]
        n = X.shape[0]
        with np.errstate(all="ignore"):
            out = [np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)) for v in fn(*args)]
        return out

    def _assemble(self, prim, X):
        """rho, P, f, G2 on X from the primitive fields (numpy)."""
        th, noise, dim = self.theta, self.noise, len(self.sv)
        nb = len(noise)
        bidx = list(noise)
        k = 1 + dim + nb
        rho, r1, r2 = prim[0], prim[1:1 + dim], dict(zip(bidx, prim[1 + dim:k]))
        pt, d1, d2 = prim[k], prim[k + 1:k + 1 + dim], dict(zip(bidx, prim[k + 1 + dim:2 * k]))
        _, _, dr = self._static(X)
        with np.errstate(all="ignore"):
            Lpt = sum(dr[i] * d1[i] for i in range(dim)) + sum(noise[b] * d2[b] for b in bidx)
            Lrho = sum(dr[i] * r1[i] for i in range(dim)) + sum(noise[b] * r2[b] for b in bidx)
            P = pt * pt
            f = th * P / 2
            LP = 2 * pt * Lpt + 2 * sum(noise[b] * d1[b] ** 2 for b in bidx)
            Lf = th * LP / 2
            G2 = rho * (LP + P * Lf) + P * Lrho
            for b in bidx:
                kb = noise[b]
                dP = 2 * pt * d1[b]
                df = th * pt * d1[b]
                G2 = G2 + rho * (P * kb * df * df + 2 * kb * dP * df) + 2 * kb * r1[b] * (dP + P * df)
        return rho, P, f, G2

    def fields(self, expr, fns=None):
        """A-independent numeric fields on the grid (+ rho on the far shell),
        or None if invalid: no pair root, foreign symbols, ptilde without p2,
        rho < 0 or non-finite anywhere, non-finite pieces where rho > 0."""
        fn = self.compile(expr) if fns is None else fns
        if fn is None:
            return None
        if not self.is_periodic(fn):
            return None
        try:
            prim = self._eval(fn, self.grid)
            rho_v, P_v, f_v, G2_v = self._assemble(prim, self.grid)
            thH, Q1, _ = self._static(self.grid)
            rho_wn = self._eval(fn, self.wit_near)[0]
            rho_wf = self._eval(fn, self.wit_far)[0]
        except Exception:
            return None
        for arr in (rho_v, rho_wn, rho_wf):
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                return None
        witness = np.where(self.in_Mnear, rho_wn >= RHO_ONE, rho_wf >= RHO_ONE)
        try:
            band = self._band_exempt(fn, rho_v, witness)
        except Exception:
            return None
        active = rho_v > 0
        for arr in (P_v, f_v, G2_v):
            if np.any(~np.isfinite(arr[active])):
                return None
        with np.errstate(all="ignore"):
            log_rhoP = np.where(active & (P_v > 0), np.log(rho_v) + np.log(P_v), -np.inf)
        return {"thH": thH, "Q1": Q1, "log_rhoP": log_rhoP, "rho": rho_v,
                "rho_wn": rho_wn, "rho_wf": rho_wf, "witness": witness, "band": band,
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
        return (self.in_Mmax & fl["band"]) | self.box0

    @staticmethod
    def _scan_down(ones_top_to_x, witness):
        """State machine along one line, walking DOWN from the witness through
        rungs ordered from largest |p2| to the point itself (last column).
        ones_top_to_x: (n_lines, m) bool, rho >= 1 at each rung; the point is
        the last column. Returns exempt flag of the point: it is exempt iff it
        is in the last rho<1 band under a contiguous rho==1 tail that reaches
        the witness."""
        n, m = ones_top_to_x.shape
        state = np.where(witness, 0, 2)          # 0 tail, 1 band, 2 dead
        for j in range(m - 1):
            o = ones_top_to_x[:, j]
            state = np.where((state == 0) & ~o, 1, state)
            state = np.where((state == 1) & o, 2, state)
        x_one = ones_top_to_x[:, -1]
        return (~x_one) & (state <= 1)

    def _band_exempt(self, fn, rho_v, witness):
        p1, p2 = self.grid[:, 1], self.grid[:, 2]
        ones = rho_v >= RHO_ONE
        if self.grid_shape is not None:
            n_r, n_p1, n_p2 = self.grid_shape
            mid = n_p2 // 2
            O = ones.reshape(n_r, n_p1, n_p2)
            W = witness.reshape(n_r, n_p1, n_p2)
            out = np.zeros_like(O)
            # positive side: rungs from the top (index n_p2-1) down to k
            pos = O[:, :, mid + 1:]                                   # ascending |p2|
            wpos = W[:, :, -1].reshape(-1)
            n_lines = n_r * n_p1
            state = np.where(wpos, 0, 2)
            res = np.zeros((n_lines, pos.shape[2]), dtype=bool)
            for k in range(pos.shape[2] - 1, -1, -1):
                o = pos[:, :, k].reshape(-1)
                res[:, k] = (~o) & (state <= 1)
                state = np.where((state == 0) & ~o, 1, state)
                state = np.where((state == 1) & o, 2, state)
            out[:, :, mid + 1:] = res.reshape(n_r, n_p1, -1)
            neg = O[:, :, :mid]                                       # index 0 = largest |p2|
            wneg = W[:, :, 0].reshape(-1)
            state = np.where(wneg, 0, 2)
            res = np.zeros((n_lines, neg.shape[2]), dtype=bool)
            for k in range(neg.shape[2]):
                o = neg[:, :, k].reshape(-1)
                res[:, k] = (~o) & (state <= 1)
                state = np.where((state == 0) & ~o, 1, state)
                state = np.where((state == 1) & o, 2, state)
            out[:, :, :mid] = res.reshape(n_r, n_p1, -1)
            return out.reshape(-1)
        # arbitrary points: 16-rung geometric ladder from the witness shell down to |p2|
        top = np.where(self.in_Mnear, self.R2_near, self.R2_max)
        a2 = np.maximum(np.abs(p2), 0.25)
        beyond = np.abs(p2) >= top
        m = 16
        t = np.linspace(1.0, 0.0, m + 1)[:-1]                        # 1 -> just above 0
        rungs = a2[:, None] * (top / a2)[:, None] ** t[None, :]        # descending from top to just above |p2|
        sgn = np.where(p2 < 0, -1.0, 1.0)
        R = np.column_stack([np.repeat(self.grid[:, 0], m), np.repeat(p1, m), (sgn[:, None] * rungs).reshape(-1)])
        rr = self._eval(fn, R)[0].reshape(-1, m) >= RHO_ONE
        seq = np.concatenate([rr, ones[:, None]], axis=1)
        return self._scan_down(seq, witness) & ~beyond

    def score(self, fl, A):
        """(pen, parts) or None if the ratio is non-finite somewhere.
        v_out: 2*(fraction violating outside K) + mean log1p(max(c_t - ratio, 0)/c_t)
               over the scored points: the fraction caps violate-everywhere
               candidates, the log term keeps a gradient from ratio ~ -1e6 up
               to feasibility instead of saturating at 0;
        v_comp: fraction of shell/witness points with rho < 1;
        v_in: fraction of exempt points violating (secondary: smaller K wins)."""
        r = self.ratio(fl, A)
        if not np.all(np.isfinite(r)):
            return None
        ex = self.exempt(fl)
        rc = np.clip(r, -1e6, np.inf)
        viol = np.maximum(self.c_t - rc[~ex], 0.0) / self.c_t
        # 2 x (fraction of scored points violating) + mean log-magnitude: a
        # candidate violating everywhere is capped at reward 1/3 however mild
        # the violations (job 20539551 epoch 0: V ~ huge constant makes
        # ratio ~ 0 < c_t everywhere, and the log term alone priced that at 0.57).
        v_out = 2.0 * float(np.mean(viol > 0)) + float(np.mean(np.log1p(viol)))
        v_in = float(np.mean(r[ex] < self.c_t)) if ex.any() else 0.0
        v_comp = (float(np.mean(fl["rho"][self.near_shell] < RHO_ONE))
                  + float(np.mean(fl["rho_wn"][self.in_Mnear] < RHO_ONE))
                  + float(np.mean(fl["rho_wf"][self.in_Mmax] < RHO_ONE))) / 3.0
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

    def scan(X, ratio_fn, shape=None):
        nonlocal worst, worst_r
        j.set_grid(X, shape=shape)
        fl = j.fields(expr, fns=fns)
        if fl is None:
            return False
        r = ratio_fn(fl)
        bad = ~j.exempt(fl) & ~(r >= j.c_t)
        shell_bad = ((j.near_shell & (fl["rho"] < RHO_ONE))
                     | (j.in_Mnear & (fl["rho_wn"] < RHO_ONE))
                     | (j.in_Mmax & (fl["rho_wf"] < RHO_ONE)))
        bad = bad | shell_bad
        if bad.any():
            i = int(np.argmin(np.where(bad, r, np.inf)))
            if r[i] < worst_r or worst is None:
                worst_r, worst = r[i], j.grid[i].copy()
        return True

    parts = [(make_grid(j.P1max, j.P2max, 49, 48, 80), (49, 97, 161))] if b.get("dense", True) \
        else [(j.grid, j.grid_shape)]
    n_random = int(b.get("n_random", 10_000_000))
    if n_random > 0:
        rng = np.random.default_rng(seed)
        parts.append((rng.uniform([-np.pi, -j.P1max, -j.P2max], [np.pi, j.P1max, j.P2max],
                                  size=(n_random, 3)), None))
    for X, shp in parts:
        if shp is not None:
            if not scan(X, lambda fl: j.ratio(fl, A), shp):
                return False, None
            continue
        for lo in range(0, X.shape[0], 1_000_000):
            if not scan(X[lo:lo + 1_000_000], lambda fl: j.ratio(fl, A)):
                return False, None
    # far region, high precision
    n_far = b.get("far_grid", (13, 12, 12))
    r = np.linspace(-np.pi, np.pi, n_far[0])
    a = np.geomspace(0.25, j.P1max, n_far[1]); p1 = np.concatenate([-a[::-1], [0.0], a])
    p2 = np.geomspace(j.P2max, 4.0 * j.R2_max, n_far[2]); p2 = np.concatenate([-p2[::-1], p2])
    R, P1, P2 = np.meshgrid(r, p1, p2, indexing="ij")
    far = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
    j.set_grid(far)
    rho_far = j._eval(fns, far)[0]
    wit = np.where(j.in_Mnear, j._eval(fns, j.wit_near)[0] >= RHO_ONE,
                   j._eval(fns, j.wit_far)[0] >= RHO_ONE)
    ex = (j.in_Mmax & j._band_exempt(fns, rho_far, wit)) | j.box0
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
