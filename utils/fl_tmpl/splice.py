"""Splicing: junction ledger, LP over gap exponents, Bellman-Ford on constant
junctions (spec 2026-10-17 sections 3.5-3.6).

Vocabulary
----------
part      : one summand w_r psi_r V_r of the template (or the main part e^{theta H}).
bad set   : a polytope in direction space where part r produces an error term that
            must be absorbed: a non-good block of r inside its region, or a switch
            layer of r (the facet of its region where psi_r is cut off).
absorber  : a part s that is good at the direction and whose good term dominates
            the error term of r there.

Pointwise absorption test at a direction s (all exponents numeric):
  D = log V_r - log V_s at power level (AsymSum over the momenta, gap terms as
  extra monomials).  P = max order of terms whose coefficient is not
  negative-definite, Nn = max order of negative-definite terms.
    max(P, Nn) > 0 :  absorbed iff Nn > P            (ratio -> 0 stretched-exp.)
    else           :  l(s) = (log level of r - log level of s) + tau_r(s) - tau_s(s)
                      l < 0 absorbed, l = 0 constant junction (edge r -> s),
                      l > 0 not absorbed.

LP (diagnostics, "缺多少"): with region exponents fixed, every power-level
absorption is a set of linear inequalities in the gap exponents a at the vertices
of the piece; slack variables per inequality, minimize total slack.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sym
from scipy.optimize import linprog

TOL = 1e-7


# ---- per-part data needed by the ledger ---------------------------------------------
@dataclass
class PartLedger:
    name: str
    regions: list                 # list of tropical.Region
    blocks: list                  # tropical.Block list (with geometry)
    logV: object                  # AsymSum, power level, gap excluded
    gap: tuple = None             # (j0 0-based momentum index, a) or None
    log_level: list = field(default_factory=list)   # [(coef, ('p', i) | ('atom', j))]
    G_keys: list = field(default_factory=list)      # effective exponent data for tau
    alpha: int = 1
    is_main: bool = False
    layer_tau: float = None       # tau of the switch-derivative error (default 2*alpha)


def order_of(space, e, s):
    nP = space.nP
    o = float(np.dot([float(x) for x in e[:nP]], s))
    for j, atom in enumerate(space.atoms):
        f = float(e[nP + j])
        if f:
            o += f * max(float(np.dot([float(x) for x in en], s)) for en, _ in atom.monomials)
    return o


def tau_at(space, keys, s):
    return max((order_of(space, k, s) for k in keys), default=-np.inf)


def log_level_at(space, ll, s):
    out = 0.0
    for c, (kind, idx) in ll:
        if kind == "p":
            out += float(c) * s[idx]
        else:
            atom = space.atoms[idx]
            out += float(c) * max(float(np.dot([float(x) for x in en], s)) for en, _ in atom.monomials)
    return out


def _param_matrix(ctx, params, n, rng):
    cols = []
    for p in ctx.params:
        v = params.get(p)
        if v is None:
            v = params.get(str(p))
        if isinstance(v, tuple):              # (lo, hi) range of a positive unknown
            cols.append(np.exp(rng.uniform(np.log(v[0]), np.log(v[1]), n)))
        elif v is None:
            raise RuntimeError(f"no value for parameter {p}")
        else:
            cols.append(np.full(n, float(v)))
    return np.column_stack(cols) if cols else None


def coeff_values(c, params, n=512, seed=0):
    """Samples of a fourier.Coef over the angles (uniform) and parameters."""
    rng = np.random.default_rng(seed)
    ctx = c.ctx
    R = rng.uniform(-np.pi, np.pi, (n, ctx.nA)) if ctx.nA else np.zeros((n, 0))
    P = _param_matrix(ctx, params, n, rng)
    return c.evaluate(R, P)


def coeff_sign_class(c, params, n=512, seed=0):
    """'neg' (negative definite), 'pos' (positive definite), 'zero', 'mixed'."""
    v = coeff_values(c, params, n, seed)
    if np.all(v < -1e-12):
        return "neg"
    if np.all(v > 1e-12):
        return "pos"
    if np.all(np.abs(v) <= 1e-12):
        return "zero"
    return "mixed"


# ---- block status lookup ------------------------------------------------------------
def status_at(part, s):
    """'good' if every block of the part whose closure contains s is good,
    'bad' if one is not, None if s lies outside the part's regions."""
    hit = False
    for b in part.blocks:
        if b.contains(s):
            hit = True
            if b.status != "good":
                return "bad"
    return "good" if hit else None


def in_regions(part, s):
    return any(r.contains(s) for r in part.regions)


# ---- the ledger ---------------------------------------------------------------------
class Ledger:
    def __init__(self, space, angles, parts, params=None, n_samples=256, seed=0):
        self.space = space
        self.angles = list(angles)
        self.parts = {p.name: p for p in parts}
        self.params = params or {}
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self._sign_cache = {}
        self.edges = []          # (r, s, log c, where)
        self.failures = []       # (bad-set label, sample direction, reason)
        self.lp_rows = []        # (label, a-coefficient dict, rhs) meaning sum c_j a_j + rhs >= delta

    def _sign(self, c):
        key = tuple(sorted(c.d.items(), key=repr))
        if key not in self._sign_cache:
            self._sign_cache[key] = coeff_sign_class(c, self.params)
        return self._sign_cache[key]

    def _D_terms(self, r, s_part):
        """[(key exps, sign class)] of log V_r - log V_s, gap terms included (cached)."""
        ck = (r.name, s_part.name)
        cache = self.__dict__.setdefault("_dcache", {})
        if ck in cache:
            return cache[ck]
        res = self._D_terms_uncached(r, s_part)
        cache[ck] = res
        return res

    def _D_terms_uncached(self, r, s_part):
        D = r.logV - s_part.logV
        out = [(k[0], self._sign(c)) for k, c in D.terms.items()]
        nT = self.space.n
        for part, sgn in ((r, +1), (s_part, -1)):
            if part.gap is not None:
                j, a = part.gap
                e = [0.0] * nT
                e[j] = float(a)
                out.append((tuple(e), "pos" if sgn > 0 else "neg"))
        return out, D

    def absorbed_at(self, r, s_part, s, tau_r):
        """'yes' | 'const' | 'no' and detail."""
        terms, D = self._D_terms(r, s_part)
        P, Nn = -np.inf, -np.inf
        for e, cls in terms:
            if cls == "zero":
                continue
            o = order_of(self.space, e, s)
            if cls == "neg":
                Nn = max(Nn, o)
            else:
                P = max(P, o)
        if max(P, Nn) > TOL:
            if Nn > P + TOL:
                return "yes", f"power level: neg {Nn:.3g} > pos {P:.3g}"
            return "no", f"power level: pos {P:.3g} >= neg {Nn:.3g}"
        tau_s = tau_at(self.space, s_part.G_keys, s)
        ell = (log_level_at(self.space, r.log_level, s) - log_level_at(self.space, s_part.log_level, s)
               + tau_r - tau_s)
        if ell < -TOL:
            return "yes", f"log level {ell:.3g}"
        if ell <= TOL:
            return "const", f"log level 0 (constant ratio)"
        return "no", f"log level {ell:.3g} > 0"

    def _constant(self, r, s_part, s):
        """Rough constant c_rs for a constant junction: e^{sup of the bounded part of
        log V_r - log V_s} (bounded remainders are not quantified: factor 1)."""
        terms, D = self._D_terms(r, s_part)
        tot = None
        for k, c in D.terms.items():
            if abs(order_of(self.space, k[0], s)) <= TOL and all(
                    float(k[0][i]) == 0 or s[i] > TOL for i in range(self.space.nP)):
                tot = c if tot is None else tot + c
        if tot is None or tot.is_zero():
            return 1.0
        return float(np.exp(np.max(coeff_values(tot, self.params, 2048))))

    def sample_polytope(self, region, extra_eq=(), n=None):
        V = region.vertices(extra_eq)
        if len(V) == 0:
            return V
        n = n or self.n_samples
        W = self.rng.dirichlet(np.ones(len(V)), size=n)
        pts = np.vstack([V, W @ V, V.mean(axis=0, keepdims=True)])
        return pts

    def check_bad_set(self, r, label, pts, tau_fn):
        """Every sample direction of the bad set must be absorbed by some part."""
        unabsorbed = []
        for s in pts:
            ok = False
            reasons = []
            for name, t in self.parts.items():
                if t is r or not in_regions(t, s):
                    continue
                if status_at(t, s) != "good":
                    reasons.append(f"{name}: not good here")
                    continue
                verdict, why = self.absorbed_at(r, t, s, tau_fn(s))
                if verdict == "yes":
                    ok = True
                    break
                if verdict == "const":
                    self.edges.append((r.name, name, math.log(2 * self._constant(r, t, s)), label))
                    ok = True
                    break
                reasons.append(f"{name}: {why}")
            if not ok:
                unabsorbed.append((tuple(np.round(s, 4)), reasons or ["no other part is active here"]))
        if unabsorbed:
            self.failures.append((label, unabsorbed))
        return not unabsorbed

    # -- LP rows ---------------------------------------------------------------------
    def lp_rows_for(self, r, t, label, vertices):
        """Power-level absorption of r by t on a piece with the given vertices,
        written as linear inequalities in the gap exponents (fixed region exponents).
        Designated negative monomial: t's gap if t has one, else the strongest
        negative-definite monomial of D at the piece's center."""
        terms, D = self._D_terms(r, t)
        known = [(e, cls) for e, cls in terms if cls != "zero"]
        center = vertices.mean(axis=0)
        rows = []
        for v in vertices:
            # positive side: non-negative-definite D terms with order > 0, and r's gap
            pos_const = [order_of(self.space, e, v) for e, cls in known
                         if cls != "neg" and not (r.gap is not None and self._is_gap(e, r))
                         and order_of(self.space, e, v) > TOL]
            pos_gap = None
            if r.gap is not None and v[r.gap[0]] > TOL:
                pos_gap = (r.name, v[r.gap[0]])
            neg_const = [order_of(self.space, e, v) for e, cls in known
                         if cls == "neg" and not (t.gap is not None and self._is_gap(e, t))]
            nbest = max(neg_const, default=-np.inf)
            neg_gap = (t.name, v[t.gap[0]]) if t.gap is not None else None
            for po in pos_const + ([None] if pos_gap else []):
                # need: max(nbest, a_t v_jt) >= po (+ a_r v_jr) + delta
                coef = {}
                rhs = 0.0
                if po is None:
                    coef[pos_gap[0]] = coef.get(pos_gap[0], 0.0) - pos_gap[1]
                else:
                    rhs -= po
                if neg_gap is not None and (nbest == -np.inf or nbest < (po or 0.0)):
                    coef[neg_gap[0]] = coef.get(neg_gap[0], 0.0) + neg_gap[1]
                else:
                    rhs += nbest if nbest > -np.inf else -10.0
                rows.append((f"{label} @ s={tuple(np.round(v, 3))}", coef, rhs))
        self.lp_rows.extend(rows)
        return rows

    def _is_gap(self, e, part):
        return False


def solve_lp(rows, gap_parts, fixed=None, delta=1e-3, bounds=(0.0, 1.0)):
    """min sum sigma  s.t.  sum_j c_j a_j + rhs + sigma_i >= delta,  sigma >= 0.
    fixed: {part: a} pins gap exponents (RL's choice).  Returns
    (feasible, a dict, [(label, sigma)] with sigma > 0)."""
    names = list(gap_parts)
    nA, nR = len(names), len(rows)
    if nR == 0:
        return True, dict(fixed or {}), []
    c = np.concatenate([np.zeros(nA), np.ones(nR)])
    A_ub, b_ub = [], []
    for i, (_, coef, rhs) in enumerate(rows):
        row = np.zeros(nA + nR)
        for nm, cv in coef.items():
            row[names.index(nm)] = -cv
        row[nA + i] = -1.0
        A_ub.append(row)
        b_ub.append(rhs - delta)
    bnds = []
    for nm in names:
        if fixed and nm in fixed:
            bnds.append((fixed[nm], fixed[nm]))
        else:
            bnds.append(bounds)
    bnds += [(0.0, None)] * nR
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bnds, method="highs")
    if res.status != 0:
        return False, {}, [("LP failed", np.inf)]
    a = {nm: float(res.x[i]) for i, nm in enumerate(names)}
    slacks = [(rows[i][0], float(res.x[nA + i])) for i in range(nR) if res.x[nA + i] > 1e-9]
    return not slacks, a, slacks


def bellman_ford(nodes, edges, source="main"):
    """Difference constraints lambda_s >= lambda_r + w for edges (r, s, w).
    Feasible iff the constraint graph has no positive cycle.  Returns
    (feasible, lambdas, cycle or None)."""
    nodes = list(nodes)
    dist = {n: 0.0 for n in nodes}
    pred = {n: None for n in nodes}
    # longest path relaxation from a virtual source connected to everything
    for _ in range(len(nodes)):
        changed = False
        for r, s, w, _ in edges:
            if dist[r] + w > dist[s] + 1e-12:
                dist[s] = dist[r] + w
                pred[s] = r
                changed = True
        if not changed:
            lam = {n: dist[n] - dist.get(source, 0.0) for n in nodes}
            return True, lam, None
    # one more pass finds a node on a positive cycle
    for r, s, w, _ in edges:
        if dist[r] + w > dist[s] + 1e-12:
            cyc = [s]
            x = r
            seen = {s}
            while x is not None and x not in seen:
                cyc.append(x)
                seen.add(x)
                x = pred[x]
            return False, dist, list(reversed(cyc))
    return True, dist, None
