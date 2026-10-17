"""Directional blow-up and leading-term block checks (spec 2026-10-17 section 3).

Direction space: p_i = u_i R^{s_i}, s in [0,1]^nP, the region's top variable has
s_top = 1.  A Region is a polytope { s : A s <= b, s_top = 1, 0 <= s <= 1 }.

check(G, region) enumerates the blocks of the AsymSum G on the region:

  * subregions: for every positive atom appearing in G, the subregion on which a
    given atom monomial attains the max (so every term has a linear order there);
  * zero sets Z: subsets of variables pinned to s_i = 0, where p_i becomes a
    compact variable on [-B, B];
  * blocks = tie-sets S of effective exponent vectors that are simultaneously
    maximal on a face with all other monomials strictly lower (LP, margin eps);
  * per block: order range [tau_min, tau_max], leading coefficient C_S over the
    compact variables (angles, units u_i, p_Z), remainder danger.

Status: good / positive / weak / degenerate / undecided / singular (see spec 3.2).
"""
import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import sympy as sym
from scipy.optimize import linprog, minimize

TOL = 1e-9


# ---- regions --------------------------------------------------------------------------
class Region:
    def __init__(self, nP, top, ineqs=(), name=""):
        """ineqs: list of (dict {i: coef}, b) meaning sum coef*s_i <= b (0-based i)."""
        self.nP = nP
        self.top = top
        self.ineqs = [(dict(a), float(b)) for a, b in ineqs]
        self.name = name

    def A_b(self):
        A, b = [], []
        for a, bb in self.ineqs:
            row = np.zeros(self.nP)
            for i, c in a.items():
                row[i] = float(c)
            A.append(row)
            b.append(bb)
        return (np.array(A).reshape(-1, self.nP), np.array(b))

    def with_extra(self, ineqs, name=None):
        return Region(self.nP, self.top, list(self.ineqs) + list(ineqs), name or self.name)

    def vertices(self, extra_eq=()):
        """Vertices of { A s <= b, 0 <= s <= 1, s_top = 1, extra equalities }.
        extra_eq: list of (dict, value) equalities."""
        nP = self.nP
        A, b = self.A_b()
        rows = [r for r in A] + [np.eye(nP)[i] for i in range(nP)] + [-np.eye(nP)[i] for i in range(nP)]
        rhs = list(b) + [1.0] * nP + [0.0] * nP
        eqA = [np.eye(nP)[self.top]]
        eqb = [1.0]
        for a, v in extra_eq:
            row = np.zeros(nP)
            for i, c in a.items():
                row[i] = c
            eqA.append(row)
            eqb.append(v)
        need = nP - len(eqA)
        verts = []
        for combo in itertools.combinations(range(len(rows)), max(need, 0)):
            M = np.vstack(eqA + [rows[i] for i in combo])
            y = np.array(eqb + [rhs[i] for i in combo])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            s = np.linalg.solve(M, y)
            if np.all(np.array([r @ s for r in rows]) <= np.array(rhs) + 1e-9) and \
                    all(abs(r @ s - v) < 1e-9 for r, v in zip(eqA, eqb)):
                if not any(np.allclose(s, w) for w in verts):
                    verts.append(s)
        return np.array(verts).reshape(-1, nP)

    def contains(self, s, tol=1e-9):
        s = np.asarray(s, float)
        if abs(s[self.top] - 1) > tol or np.any(s < -tol) or np.any(s > 1 + tol):
            return False
        A, b = self.A_b()
        return bool(np.all(A @ s <= b + tol)) if len(b) else True

    def __repr__(self):
        return f"Region({self.name}, top={self.top}, {len(self.ineqs)} ineqs)"


def region_single_fast(nP, j, k, name=None):
    """Omega_j(k): s_j = 1, s_i <= 1/k (0-based j)."""
    ineqs = [({i: 1.0}, 1.0 / k) for i in range(nP) if i != j]
    return Region(nP, j, ineqs, name or f"Omega_{j + 1}(k={k})")


def region_full(nP, top, name=None):
    return Region(nP, top, [], name or f"S(top={top + 1})")


# ---- LP helper --------------------------------------------------------------------------
def _lp(c, A_ub, b_ub, A_eq, b_eq, nvar):
    bounds = [(0.0, 1.0)] * nvar
    res = linprog(c, A_ub=A_ub if len(A_ub) else None, b_ub=b_ub if len(b_ub) else None,
                  A_eq=A_eq if len(A_eq) else None, b_eq=b_eq if len(b_eq) else None,
                  bounds=bounds, method="highs")
    return res


@dataclass
class Block:
    region: str
    sub: str
    Z: tuple
    ties: list                     # list of effective exponent vectors (tuples of float)
    tau: tuple                     # (tau_min, tau_max)
    sup: float
    inf: float
    status: str
    flags: list = field(default_factory=list)
    s_rep: tuple = ()
    detail: str = ""
    geom: object = None            # (region, Z, E_ties, E_others) for membership tests

    def contains(self, s, tol=1e-7):
        """Is direction s in the closure of this block?"""
        if self.geom is None:
            return False
        region, Z, Et, Eo = self.geom
        s = np.asarray(s, float)
        if not region.contains(s, tol):
            return False
        if any(abs(s[i]) > tol for i in Z):
            return False
        t = Et @ s
        if np.ptp(t) > tol:
            return False
        return bool(len(Eo) == 0 or np.all(Eo @ s <= t[0] + tol))

    def short(self):
        return (f"[{self.status}] {self.region}/{self.sub} Z={self.Z} s~{tuple(round(x, 3) for x in self.s_rep)} "
                f"tau=[{self.tau[0]:.3g},{self.tau[1]:.3g}] C in [{self.inf:.3g},{self.sup:.3g}] "
                f"{' '.join(self.flags)} {self.detail}").strip()


@dataclass
class CheckConfig:
    B: float = 30.0                # box for compact momenta (s_i = 0)
    n_samples: int = 6000
    n_refine: int = 6
    c_min: float = 1e-6            # G <= -c needs sup C <= -c_min on tau == 0 parts
    degenerate_tol: float = 1e-10
    seed: int = 0
    params: dict = field(default_factory=dict)       # numeric values for parameter symbols
    param_ranges: dict = field(default_factory=dict) # symbol -> (lo, hi) sampled (positive unknowns)
    refinements: list = field(default_factory=list)


# ---- the checker -----------------------------------------------------------------------
class TropicalChecker:
    def __init__(self, space, angles, cfg=None):
        self.space = space
        self.angles = list(angles)
        self.cfg = cfg or CheckConfig()
        self.u = sym.symbols(f"u1:{space.nP + 1}")

    # -- effective exponents in a subregion --
    def _atoms_used(self, G):
        used = set()
        keys = [k[0] for k in G.terms] + list(G.rem)
        for e in keys:
            for j in range(self.space.nA):
                if e[self.space.nP + j] != 0:
                    used.add(j)
        return sorted(used)

    def _subregions(self, G, region):
        """List of (label, region', choice {atom j: monomial index})."""
        used = self._atoms_used(G)
        if not used:
            return [("-", region, {})]
        out = []
        choices = [range(len(self.space.atoms[j].monomials)) for j in used]
        for combo in itertools.product(*choices):
            extra = []
            for j, mi in zip(used, combo):
                mons = self.space.atoms[j].monomials
                em = np.array([float(x) for x in mons[mi][0]])
                for nu, (en, _) in enumerate(mons):
                    if nu == mi:
                        continue
                    d = np.array([float(x) for x in en]) - em      # (e_nu - e_mu) . s <= 0
                    extra.append(({i: d[i] for i in range(self.space.nP) if d[i] != 0}, 0.0))
            r2 = region.with_extra(extra)
            if self._feasible(r2, ()):
                label = ",".join(f"{self.space.atoms[j].name}:{mi}" for j, mi in zip(used, combo))
                out.append((label, r2, dict(zip(used, combo))))
        return out

    def _eff(self, e, choice):
        nP = self.space.nP
        v = np.array([float(x) for x in e[:nP]])
        for j, mi in choice.items():
            f = float(e[nP + j])
            if f:
                v = v + f * np.array([float(x) for x in self.space.atoms[j].monomials[mi][0]])
        for j in range(self.space.nA):
            if float(e[nP + j]) and j not in choice:
                raise RuntimeError("atom without subregion choice")
        return v

    def _base_constraints(self, region, Z):
        A, b = region.A_b()
        nP = self.space.nP
        A_eq, b_eq = [], []
        row = np.zeros(nP); row[region.top] = 1.0
        A_eq.append(row); b_eq.append(1.0)
        for i in Z:
            row = np.zeros(nP); row[i] = 1.0
            A_eq.append(row); b_eq.append(0.0)
        return A, b, np.array(A_eq), np.array(b_eq)

    def _feasible(self, region, Z):
        A, b, Ae, be = self._base_constraints(region, Z)
        res = _lp(np.zeros(self.space.nP), A, b, Ae, be, self.space.nP)
        return res.status == 0

    def _zero_sets(self, region):
        free = [i for i in range(self.space.nP) if i != region.top]
        out = []
        for r in range(len(free) + 1):
            for Z in itertools.combinations(free, r):
                if self._feasible(region, Z):
                    out.append(Z)
        return out

    # -- tie-set enumeration on (region, Z) with effective exponents E (n_m x nP) --
    def _face_lp(self, region, Z, E, S, others):
        """max eps s.t. ties on S, others <= tie - eps, free s_i >= eps."""
        nP = self.space.nP
        A, b, Ae, be = self._base_constraints(region, Z)
        n = nP + 1                                  # variables s, eps
        A_ub = [np.append(row, 0.0) for row in A]
        b_ub = list(b)
        A_eq = [np.append(row, 0.0) for row in Ae]
        b_eq = list(be)
        m0 = S[0]
        for m in S[1:]:
            A_eq.append(np.append(E[m] - E[m0], 0.0)); b_eq.append(0.0)
        for k in others:
            A_ub.append(np.append(E[k] - E[m0], 1.0)); b_ub.append(0.0)
        for i in range(nP):
            if i != region.top and i not in Z:
                row = np.zeros(n); row[i] = -1.0; row[-1] = 1.0
                A_ub.append(row); b_ub.append(0.0)
        c = np.zeros(n); c[-1] = -1.0
        bounds = [(0.0, 1.0)] * nP + [(0.0, 1.0)]
        res = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
        if res.status != 0:
            return None, None
        return -res.fun, res.x[:nP]

    def _tau_range(self, region, Z, E, S, others, extra_vec=None):
        """min/max of E[S0].s (or of extra_vec.s) over the closed face."""
        nP = self.space.nP
        A, b, Ae, be = self._base_constraints(region, Z)
        A_ub, b_ub = list(A), list(b)
        A_eq, b_eq = list(Ae), list(be)
        m0 = S[0]
        for m in S[1:]:
            A_eq.append(E[m] - E[m0]); b_eq.append(0.0)
        for k in others:
            A_ub.append(E[k] - E[m0]); b_ub.append(0.0)
        vec = E[m0] if extra_vec is None else extra_vec
        out = []
        for sgn in (1.0, -1.0):
            res = _lp(sgn * vec, np.array(A_ub) if A_ub else [], np.array(b_ub) if b_ub else [],
                      np.array(A_eq), np.array(b_eq), nP)
            out.append(sgn * res.fun if res.status == 0 else np.nan)
        return out[0], out[1]

    def _relevant(self, region, Z, E):
        nP = self.space.nP
        A, b, Ae, be = self._base_constraints(region, Z)
        rel = []
        for m in range(len(E)):
            A_ub = list(A) + [E[k] - E[m] for k in range(len(E)) if k != m]
            b_ub = list(b) + [0.0] * (len(E) - 1)
            res = _lp(np.zeros(nP), np.array(A_ub) if A_ub else [], np.array(b_ub) if b_ub else [],
                      Ae, be, nP)
            if res.status == 0:
                rel.append(m)
        return rel

    # -- leading coefficient --
    def _unit_expr(self, key, Z, choice, s_rep):
        """unit of one monomial at direction s_rep: u_i (or p_i for i in Z) powers and atom units."""
        e, par = key
        nP = self.space.nP
        m = sym.S.One
        for i in range(nP):
            ex = e[i]
            if ex == 0 and par[i] == 0:
                continue
            base = self.space.momenta[i] if i in Z else self.u[i]
            if Fraction(ex).denominator == 1 and int(ex) % 2 == par[i]:
                m *= base ** int(ex)
            else:
                m *= sym.sign(base) ** par[i] * sym.Abs(base) ** sym.Rational(ex.numerator, ex.denominator)
        for j, atom in enumerate(self.space.atoms):
            f = e[nP + j]
            if f == 0:
                continue
            orders = [float(np.dot([float(x) for x in en], s_rep)) for en, _ in atom.monomials]
            top = max(orders)
            unit = sym.S.Zero
            for (en, c), o in zip(atom.monomials, orders):
                if o >= top - 1e-7:
                    t = sym.Float(c)
                    for i in range(nP):
                        if en[i]:
                            base = self.space.momenta[i] if i in Z else self.u[i]
                            t *= base ** int(en[i])
                    unit += t
            m *= unit ** sym.Rational(f.numerator, f.denominator)
        return m

    def _sample_sup(self, C_expr, Z, rng):
        cfg = self.cfg
        params = {sym.Symbol(str(k)) if isinstance(k, str) else k: v for k, v in cfg.params.items()}
        C_expr = C_expr.subs(params)
        pr = list(cfg.param_ranges.items())
        pr_syms = [sym.Symbol(str(k)) if isinstance(k, str) else k for k, _ in pr]
        free_u = [self.u[i] for i in range(self.space.nP)
                  if i not in Z and self.u[i] in C_expr.free_symbols]
        pz = [self.space.momenta[i] for i in Z if self.space.momenta[i] in C_expr.free_symbols]
        ang = [a for a in self.angles if a in C_expr.free_symbols]
        args = ang + free_u + pz + pr_syms
        extra = C_expr.free_symbols - set(args)
        if extra:
            raise RuntimeError(f"unbound symbols in leading coefficient: {extra}")
        if not args:
            v = float(C_expr)
            return v, v, False
        f = sym.lambdify(args, C_expr, "numpy")
        n = cfg.n_samples
        cols = []
        for a in ang:
            cols.append(rng.uniform(-np.pi, np.pi, n))
        for _ in free_u:
            cols.append(rng.choice([-1.0, 1.0], n) * 2.0 ** rng.uniform(-1, 1, n))
        for _ in pz:
            col = rng.uniform(-cfg.B, cfg.B, n)
            col[: n // 10] = 0.0
            cols.append(col)
        for _, (lo, hi) in pr:
            cols.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
        X = np.column_stack(cols)
        with np.errstate(all="ignore"):
            vals = np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), dtype=float)
        vals = np.broadcast_to(vals, (n,))
        ok = np.isfinite(vals)
        if not ok.all():
            return np.inf, -np.inf, True
        sup, inf = float(vals.max()), float(vals.min())
        # local refinement of the sup
        lo = [-np.pi] * len(ang) + [-2.0] * len(free_u) + [-cfg.B] * len(pz) + [r[1][0] for r in pr]
        hi = [np.pi] * len(ang) + [2.0] * len(free_u) + [cfg.B] * len(pz) + [r[1][1] for r in pr]
        for idx in np.argsort(vals)[-cfg.n_refine:]:
            x0 = X[idx].copy()
            # keep unit signs fixed (|u| in [1/2, 2] within the sign's half line)
            bl, bh = list(lo), list(hi)
            for t in range(len(free_u)):
                k = len(ang) + t
                if x0[k] > 0:
                    bl[k], bh[k] = 0.5, 2.0
                else:
                    bl[k], bh[k] = -2.0, -0.5
            try:
                r = minimize(lambda x: -float(f(*x)), x0, bounds=list(zip(bl, bh)), method="L-BFGS-B")
                if np.isfinite(r.fun):
                    sup = max(sup, -float(r.fun))
            except Exception:  # pylint: disable=broad-except
                pass
        return sup, inf, False

    def check(self, G, region, label="", rng=None):
        rng = rng or np.random.default_rng(self.cfg.seed)
        blocks = []
        for sub_label, sub, choice in self._subregions(G, region):
            for Z in self._zero_sets(sub):
                blocks.extend(self._check_face_family(G, sub, sub_label, choice, Z, label or region.name, rng))
        return blocks

    def _check_face_family(self, G, region, sub_label, choice, Z, label, rng):
        nP = self.space.nP
        keys = list(G.terms)
        effs = [self._eff(k[0], choice) for k in keys]
        # effective exponents restricted off Z (Z coordinates do not count in the order)
        groups = {}
        for k, v in zip(keys, effs):
            v2 = v.copy()
            for i in Z:
                v2[i] = 0.0
            groups.setdefault(tuple(np.round(v2, 9)), []).append(k)
        E = np.array([list(g) for g in groups]) if groups else np.zeros((0, nP))
        gkeys = list(groups.values())
        rems = [self._eff(r, choice) for r in G.rem]
        for i in Z:
            for rv in rems:
                rv[i] = 0.0
        if len(E) == 0:
            return [Block(label, sub_label, Z, [], (np.nan, np.nan), 0.0, 0.0, "degenerate",
                          detail="expression is identically zero")]
        rel = self._relevant(region, Z, E)
        out = []
        seen = set()
        frontier = [(m,) for m in rel]
        while frontier:
            S = frontier.pop()
            if S in seen:
                continue
            seen.add(S)
            if len(seen) > 20000:
                raise RuntimeError("block enumeration exploded (too many relevant monomials)")
            others = [k for k in rel if k not in S]
            eps, s_rep = self._face_lp(region, Z, E, list(S), others)
            valid = eps is not None and eps > 1e-9
            # a face's tie-set need not be reachable through valid faces one
            # element at a time (a monomial can be maximal only on a wall), so
            # small tie-sets are always extended
            if valid or len(S) <= 2:
                for k in others:
                    frontier.append(tuple(sorted(S + (k,))))
            if not valid:
                continue
            out.append(self._judge(G, region, sub_label, choice, Z, E, gkeys, rems, list(S), others,
                                   s_rep, label, rng))
        return out

    def _judge(self, G, region, sub_label, choice, Z, E, gkeys, rems, S, others, s_rep, label, rng):
        b = self._judge_inner(G, region, sub_label, choice, Z, E, gkeys, rems, S, others, s_rep, label, rng)
        b.geom = (region, tuple(Z), E[list(S)], E[list(others)] if others else np.zeros((0, E.shape[1])))
        return b

    def _judge_inner(self, G, region, sub_label, choice, Z, E, gkeys, rems, S, others, s_rep, label, rng):
        tmin, tmax = self._tau_range(region, Z, E, S, others)
        flags = []
        # remainder danger: max over face of (e_R - e_S0).s >= 0
        for rv in rems:
            _, mx = self._tau_range(region, Z, E, S, others, extra_vec=rv - E[S[0]])
            if mx >= -1e-9:
                return Block(label, sub_label, Z, [tuple(E[m]) for m in S], (tmin, tmax), np.nan, np.nan,
                             "undecided", flags, tuple(s_rep), detail=f"remainder of order {tuple(rv)} reaches leading order")
        C = sym.S.Zero
        singular = False
        for m in S:
            for key in gkeys[m]:
                if any(key[0][i] < 0 for i in Z):
                    singular = True
                C += G.terms[key].to_sympy() * self._unit_expr(key, Z, choice, s_rep)
        if singular:
            return Block(label, sub_label, Z, [tuple(E[m]) for m in S], (tmin, tmax), np.nan, np.nan,
                         "singular", flags, tuple(s_rep), detail="negative power of a bounded momentum")
        sup, inf, bad = self._sample_sup(C, Z, rng)
        if bad:
            return Block(label, sub_label, Z, [tuple(E[m]) for m in S], (tmin, tmax), sup, inf,
                         "singular", flags, tuple(s_rep), detail="non-finite leading coefficient")
        osc = (inf < -self.cfg.c_min < self.cfg.c_min < sup) and bool(C.free_symbols & set(self.angles))
        if osc:
            flags.append("oscillatory")
        if max(abs(sup), abs(inf)) < self.cfg.degenerate_tol:
            status = "degenerate"
        elif tmin < -1e-9:
            status = "weak"
        elif sup > -self.cfg.c_min:
            status = "positive"
        else:
            status = "good"
        return Block(label, sub_label, Z, [tuple(E[m]) for m in S], (tmin, tmax), sup, inf, status,
                     flags, tuple(s_rep))


def summarize(blocks):
    from collections import Counter
    return dict(Counter(b.status for b in blocks))
