"""Per-part log-domain generator G_r = h^alpha (L F + Gamma(F,F)) (spec 2026-10-17 section 1).

A part is V_r = exp(F), F = sum_k phi_k(Y_k) over components:

    log      d * log Y          (prefactor P_r = Y^d)
    abs_pow  |Y|^a              (gap factor)
    energy_sq theta * Y^2 / 2   (averaged energy of a slow momentum)
    energy_lin theta * Y        (averaged energy given directly, H~_c or H)
    log1p    log(1 + Y)         (noise-averaging factor, Y small)
    free     log g(|Y|)         (free-grammar shape, param = freeform.expand_shape data)

Chain rule:  L F = sum phi'(Y) L Y + sum_b kappa_b phi''(Y) (d_b Y)^2,
             d_b F = sum phi'(Y) d_b Y,   Gamma = sum_b kappa_b (d_b F)^2.
h = 1 + H is written as the positive atom Ah = sum p_i^2 / 2 times (1 + O(1/Ah)),
so h^alpha never has to be expanded.
"""
from dataclasses import dataclass

import numpy as np
import sympy as sym

from utils.fl_tmpl.asym import AsymSum
from utils.fl_tmpl.averaging import fast_modes_cancel
from utils.fl_tmpl.macros import explicit_macro

from utils.fl_tmpl.system import THETA  # noqa: F401  (re-exported)


@dataclass
class Component:
    kind: str
    macro: object
    param: object = None

    def label(self):
        if self.kind == "free":
            return f"free({self.macro.name}, {self.param.get('g')})"
        return f"{self.kind}({self.macro.name}{'' if self.param is None else ', ' + str(self.param)})"


def phi_derivs(comp, nmax=4):
    Y = comp.macro.value
    sp = Y.space
    k = comp.kind
    if k == "log":
        d = sym.sympify(comp.param)
        Yinv = Y.power(-1, comp.macro.lead_key, nmax)
        return Yinv.scale(d), (Yinv * Yinv).scale(-d)
    if k == "abs_pow":
        a = sym.nsimplify(comp.param)
        Ya2 = Y.power(a - 2, comp.macro.lead_key, nmax, absval=True)
        return (Y * Ya2).scale(a), Ya2.scale(a * (a - 1))
    if k == "energy_sq":
        th = sym.sympify(comp.param)
        return Y.scale(th), AsymSum.const(sp, th)
    if k == "energy_lin":
        th = sym.sympify(comp.param)
        return AsymSum.const(sp, th), AsymSum(sp)
    if k == "free":
        from utils.fl_tmpl.freeform import phi_derivs_free
        return phi_derivs_free(Y, comp.macro.lead_key, comp.param, nmax)
    if k == "log1p":
        p1 = Y.series_1p([(-1) ** n for n in range(nmax + 1)], nmax)
        p2 = Y.series_1p([(-1) ** (n + 1) * (n + 1) for n in range(nmax + 1)], nmax)
        return p1, p2
    raise ValueError(f"unknown component kind {k}")


def ensure_h_atom(S):
    return S.space.add_atom("Ah", sum(pi**2 for pi in S.p) / 2)


def _delta_max(c, pruner):
    """max over the region of the order of the series variable of component c
    (-eta < 0 means the series converges asymptotically), or None if no series."""
    if c.kind in ("energy_sq", "energy_lin"):
        return None
    Y = c.macro.value
    if c.kind == "log1p":
        dl = Y
    else:
        _, dl = Y.split_lead(c.macro.lead_key)
    keys = [k[0] for k in dl.terms] + list(dl.rem)
    if not keys:
        return -np.inf
    return max(pruner.max_order(k) for k in keys)


def _top_order(A, pruner):
    keys = [k[0] for k in A.terms] + list(A.rem)
    return max((pruner.max_order(k) for k in keys), default=-np.inf)


def part_K(S, comps, nmax=4, pruner=None):
    """K = L F + Gamma(F, F) as an AsymSum.

    With a pruner (floor = -2 alpha on the part's region), components are handled in
    two passes: exact components (energies) first, then series components (gap,
    prefactor, noise factor).  A series component whose lead-level contributions --
    phi' L Y, phi'' (d_b Y)^2 and the Gamma cross term with the energy's d_b F -- all
    lie below the floor everywhere is skipped (the series terms are smaller still);
    otherwise its series order is chosen just large enough that the truncation
    remainder falls below the floor.  Every product is pruned at the floor."""
    sp = S.space
    pr = pruner if pruner is not None else (lambda A: A)
    LF = AsymSum(sp)
    dF = {b: AsymSum(sp) for b in S.baths}
    exact = [c for c in comps if c.kind in ("energy_sq", "energy_lin")]
    series = [c for c in comps if c.kind not in ("energy_sq", "energy_lin")]
    skipped = []
    for c in exact + series:
        n_c = nmax
        if pruner is not None and c in series:
            dm = _delta_max(c, pruner)
            if dm >= 0:
                raise ValueError(f"component {c.label()}: series variable is not small on the region "
                                 f"(max order {dm:.3g} >= 0)")
            q1, q2 = phi_derivs(c, 0)
            m = c.macro
            tops = [_top_order(q1 * m.L, pruner)]
            for b in S.baths:
                if m.d[b].terms or m.d[b].rem:
                    tops.append(_top_order(q2 * (m.d[b] * m.d[b]), pruner))
                    t1 = _top_order(q1 * m.d[b], pruner)
                    tops.append(2 * t1)
                    tops.append(t1 + _top_order(dF[b], pruner))
            top = max(tops)
            if top < pruner.floor - 1e-9:
                skipped.append(c.label())
                continue
            n_c = int(max(0, min(nmax, np.ceil((top - pruner.floor) / (-dm)))))
        p1, p2 = phi_derivs(c, n_c)
        m = c.macro
        LF = LF + p1.mul_pruned(m.L, pruner)
        for b in S.baths:
            kb = S.kappa[b]
            db = m.d[b]
            if db.terms or db.rem:
                LF = LF + p2.mul_pruned(db * db, pruner).scale(kb)
                dF[b] = dF[b] + p1 * db
    K = LF
    for b in S.baths:
        K = K + dF[b].mul_pruned(dF[b], pruner).scale(S.kappa[b])
    K = pr(K)
    K.skipped = skipped
    return K


def make_pruner(S, regions, alpha):
    """Floor for K: a term of K whose order is below -2*alpha at every direction of
    the region can never give G = Ah^alpha K an order >= 0 (Ah has order 2 there),
    so it cannot decide any block of G (spec 1.2)."""
    from utils.fl_tmpl.asym import Pruner
    ensure_h_atom(S)
    verts = [r.vertices() for r in regions]
    return Pruner(S.space, verts, -2.0 * alpha)


def part_G(S, comps, alpha, nmax=4, regions=None):
    """G = h^alpha K with h = Ah (1 + O(1/Ah))."""
    ensure_h_atom(S)
    pruner = make_pruner(S, regions, alpha) if regions else None
    K = part_K(S, comps, nmax, pruner)
    hk = S.space.monomial_key({"Ah": alpha})
    G = K.mul_monomial(hk)
    lower = S.space.monomial_key({"Ah": alpha - 1})
    for key in list(K.terms):
        G.rem.add(tuple(a + b for a, b in zip(key[0], lower[0])))
    for r in K.rem:
        G.rem.add(tuple(a + b for a, b in zip(r, lower[0])))
    return G, K


# ---- automatic noise averaging (spec 1.3) ------------------------------------------------
def _order_at(sp, e, s):
    nP = sp.nP
    o = float(np.dot([float(x) for x in e[:nP]], s))
    for j, atom in enumerate(sp.atoms):
        f = float(e[nP + j])
        if f:
            o += f * max(float(np.dot([float(x) for x in en], s)) for en, _ in atom.monomials)
    return o


def leading_at(K, s, tol=1e-9):
    """(tau, {key: coeff}) of the terms of K of maximal order at direction s."""
    sp = K.space
    orders = {k: _order_at(sp, k[0], s) for k in K.terms}
    if not orders:
        return None, {}
    tau = max(orders.values())
    return tau, {k: c for k, c in K.terms.items() if orders[k] >= tau - tol}


def noise_average(S, comps, fast, nmax=4, name="N", pruner=None):
    """Build the factor (1 + N) cancelling the fast-oscillating modes of K's leading
    order at the direction s* (fast momenta s = 1, others 0).  Returns
    (component or None, info dict)."""
    K = part_K(S, comps, nmax, pruner)
    s = np.zeros(S.N)
    for f in fast:
        s[f - 1] = 1.0
    tau, lead = leading_at(K, s)
    sp = S.space
    N_expr = sym.S.Zero
    resonant = []
    for key, c in lead.items():
        mono = K.monomial_sympy(key)
        num, res = fast_modes_cancel(S, c.to_sympy(), list(fast), 0)
        resonant.extend(res)
        for f, n_f in num.items():
            if n_f != 0:
                N_expr += -n_f * mono / S.P(f)
    info = {"tau": tau, "resonant": resonant, "N": N_expr}
    if N_expr == 0:
        return None, info
    N_expr = sym.expand(N_expr)
    mac = explicit_macro(S, name, N_expr, kind="explicit", fast=tuple(fast))
    return Component("log1p", mac), info


def leading_coefficient_at(S, K, fast):
    """Leading coefficient of K at s* (fast momenta s = 1, slow momenta set to 0),
    as a sympy expression in the angles, the fast momenta (units) and parameters."""
    s = np.zeros(S.N)
    for f in fast:
        s[f - 1] = 1.0
    tau, lead = leading_at(K, s)
    C = sum((c.to_sympy() * K.monomial_sympy(k) for k, c in lead.items()), sym.S.Zero)
    C = C.subs({S.P(i): 0 for i in range(1, S.N + 1) if i not in fast})
    return tau, C


def sup_over_compact(S, C, fast, subs=None, n=20000, seed=0):
    C = C.subs(subs or {})
    rng = np.random.default_rng(seed)
    ang = [a for a in S.r if a in C.free_symbols]
    fm = [S.P(f) for f in fast if S.P(f) in C.free_symbols]
    if C.free_symbols - set(ang) - set(fm):
        raise RuntimeError(f"unbound symbols {C.free_symbols - set(ang) - set(fm)}")
    if not ang and not fm:
        return float(C)
    fn = sym.lambdify(ang + fm, C, "numpy")
    cols = [rng.uniform(-np.pi, np.pi, n) for _ in ang]
    cols += [rng.choice([-1.0, 1.0], n) * 2.0 ** rng.uniform(-1, 1, n) for _ in fm]
    return float(np.max(np.broadcast_to(np.asarray(fn(*cols), dtype=float), (n,))))


def theta_window(S, comps_theta, fast, use_noise=True, nmax=4, regions=None, alpha=1,
                 checker_factory=None, hi=2.0, iters=16, ignore=None):
    """Largest theta for which every tropical block of the part (on its regions) is
    good.  comps_theta: components built with the symbol THETA (noise factor
    excluded; it is added here when use_noise).  G is built once; only the block
    check is repeated with numeric theta (bisection)."""
    from utils.fl_tmpl.tropical import CheckConfig, TropicalChecker
    pruner = make_pruner(S, regions, alpha)
    comps = list(comps_theta)
    if use_noise and fast:
        nc, _ = noise_average(S, comps, fast, nmax, pruner=pruner)
        if nc is not None:
            comps = comps + [nc]
    G, _ = part_G(S, comps, alpha, nmax, regions=regions)

    def all_good(th):
        cfg = CheckConfig(n_samples=3000, n_refine=3)
        cfg.params = {THETA: th}
        ch = checker_factory(cfg) if checker_factory else TropicalChecker(S.space, S.r, cfg)
        return all(b.status == "good" or (ignore is not None and ignore(b))
                   for r in regions for b in ch.check(G, r))

    lo_t, hi_t = 0.0, hi
    if not all_good(1e-3):
        return 0.0
    if all_good(hi_t):
        return hi_t
    for _ in range(iters):
        mid = 0.5 * (lo_t + hi_t)
        if all_good(mid):
            lo_t = mid
        else:
            hi_t = mid
    return lo_t
