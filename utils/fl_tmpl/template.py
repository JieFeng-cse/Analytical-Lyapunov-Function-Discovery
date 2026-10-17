"""Template, slots and the full evaluation (spec 2026-10-17 sections 2.2-2.3, 3-4).

    V = 1 + e^{theta H} + sum_r w_r psi_r V_r,
    V_r = P_r(Y) * exp(|Y_gap|^{a_r}) * exp(theta E_r(Y)) * (1 + N_r)

evaluate(template) -> Report:
  1. per part: components, automatic noise factor, G_r = h^alpha (L F + Gamma),
     tropical blocks on the part's regions;
  2. ledger: every bad block (non-good block inside the part's region) and every
     switch layer (facet of the part's region) must be absorbed at every sampled
     direction by a part that is good there (power level / log level / constant);
  3. LP over the gap exponents at the sampled directions (per-row slack = 缺多少);
  4. Bellman-Ford on the constant junctions (weights w_r);
  5. penalty and reward.
"""
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import sympy as sym

from utils.fl_tmpl.asym import AsymSum, NotQuasiHomogeneous
from utils.fl_tmpl.logdomain import Component, make_pruner, noise_average, part_G, theta_window
from utils.fl_tmpl.splice import (Ledger, PartLedger, bellman_ford, order_of, solve_lp, status_at,
                                  in_regions, tau_at)
from utils.fl_tmpl.system import THETA
from utils.fl_tmpl.tropical import CheckConfig, Region, TropicalChecker, region_full, region_single_fast


@dataclass
class PartSpec:
    name: str
    fast: tuple                      # fast momenta (1-based); () for main
    energy: str = None               # macro name of the averaged energy (None: main)
    energy_kind: str = "sq"          # 'sq' (theta Y^2/2) or 'lin' (theta Y)
    deg: int = 0                     # prefactor degree P = Y^deg
    gap: str = None                  # macro name carrying the gap factor, or None
    a: Fraction = None               # gap exponent
    regions: list = field(default_factory=list)
    noise_avg: bool = True
    is_main: bool = False


@dataclass
class Template:
    S: object
    alpha: int
    theta: Fraction
    macros: dict
    parts: list
    slots: dict = field(default_factory=dict)
    conditions: list = field(default_factory=list)


@dataclass
class PartResult:
    name: str
    blocks: list
    theta_max: float = None
    theta_max_without_noise: float = None
    noise: object = None
    resonant: list = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0


@dataclass
class Report:
    parts: dict
    bad_blocks: int
    unabsorbed: list
    lp_feasible: bool
    lp_slacks: list
    lp_hint: dict
    bf_feasible: bool
    bf_cycle: object
    edges: list
    penalty: float
    reward: float
    conditions: list
    seconds: float

    def summary(self):
        lines = [f"penalty={self.penalty:.4g} reward={self.reward:.4g}  bad_blocks={self.bad_blocks} "
                 f"unabsorbed={len(self.unabsorbed)} lp_feasible={self.lp_feasible} bf_feasible={self.bf_feasible} "
                 f"({self.seconds:.1f}s)"]
        for n, pr in self.parts.items():
            from collections import Counter
            cnt = Counter(b.status for b in pr.blocks)
            tm = "" if pr.theta_max is None else f" theta_max={pr.theta_max:.4g} (without noise avg {pr.theta_max_without_noise:.4g})"
            lines.append(f"  part {n}: {dict(cnt)}{tm} {pr.error}")
        for lab, pts in self.unabsorbed:
            lines.append(f"  UNABSORBED {lab}: {len(pts)} directions, e.g. {pts[0][0]}: {pts[0][1][:3]}")
        for lab, sg in self.lp_slacks[:12]:
            lines.append(f"  LP slack {sg:.4g}: {lab}")
        if self.lp_hint:
            lines.append(f"  LP hint (free gap exponents): {self.lp_hint}")
        if self.bf_cycle:
            lines.append(f"  positive cycle: {self.bf_cycle}")
        if self.conditions:
            lines.append(f"  conditional on: {self.conditions}")
        return "\n".join(lines)


# ---- components and log size ---------------------------------------------------------
def part_components(T, ps, theta_sym=THETA):
    if ps.is_main:
        return [Component("energy_lin", T.macros["H"], theta_sym)]
    E = T.macros[ps.energy]
    comps = []
    if ps.energy_kind == "sq":
        comps.append(Component("energy_sq", E, theta_sym))
    else:
        comps.append(Component("energy_lin", E, theta_sym))
    if ps.deg:
        comps.append(Component("log", E, ps.deg))
    if ps.gap is not None and ps.a:
        comps.append(Component("abs_pow", T.macros[ps.gap], ps.a))
    return comps


def part_logsize(T, ps):
    """(power-level AsymSum, log-level list, gap) of log V_r with numeric theta."""
    S = T.S
    th = sym.Rational(T.theta.numerator, T.theta.denominator)
    if ps.is_main:
        return S.asym(th * S.H), [], None
    E = T.macros[ps.energy]
    if ps.energy_kind == "sq":
        logV = (E.value * E.value).scale(th / 2)
    else:
        logV = E.value.scale(th)
    logV = logV.expand_atoms()
    ll = []
    if ps.deg:
        e, par = E.lead_key
        nz = [i for i in range(S.space.n) if e[i] != 0]
        assert len(nz) == 1, "prefactor lead must be a single momentum or atom"
        i = nz[0]
        f = float(e[i])
        kind = ("p", i) if i < S.space.nP else ("atom", i - S.space.nP)
        ll.append((ps.deg * f, kind))
    gap = None
    if ps.gap is not None and ps.a:
        g = T.macros[ps.gap]
        e, _ = g.lead_key
        j = [i for i in range(S.space.nP) if e[i] != 0]
        assert len(j) == 1
        gap = (j[0], float(ps.a) * float(e[j[0]]))
    return logV, ll, gap


def block_region(b):
    """Region object for the closure of a block (for vertex sampling)."""
    region, Z, Et, Eo = b.geom
    extra = []
    for row in Eo:
        d = row - Et[0]
        extra.append(({i: d[i] for i in range(len(d)) if abs(d[i]) > 1e-12}, 0.0))
    R = region.with_extra(extra)
    eqs = [({i: 1.0}, 0.0) for i in Z]
    for row in Et[1:]:
        d = row - Et[0]
        eqs.append(({i: d[i] for i in range(len(d)) if abs(d[i]) > 1e-12}, 0.0))
    return R, eqs


# ---- evaluation ------------------------------------------------------------------------
_PART_CACHE = {}


def _part_signature(T, ps):
    regs = tuple((r.top, tuple((tuple(sorted(a.items())), b) for a, b in r.ineqs)) for r in ps.regions)
    return (T.S.N, T.alpha, str(T.theta), ps.name, ps.fast, ps.energy, ps.energy_kind, ps.deg, ps.gap,
            str(ps.a), ps.noise_avg, ps.is_main, regs, id(T.macros))


def evaluate(T, cfg=None, theta_windows=True, n_samples=64, seed=0, verbose=False):
    t0 = time.time()
    S = T.S
    cfg = cfg or CheckConfig()
    cfg.params = dict(cfg.params)
    cfg.params[THETA] = float(T.theta)
    checker = TropicalChecker(S.space, S.r, cfg)
    results = {}
    ledgers = []
    for ps in T.parts:
        t1 = time.time()
        sig = (_part_signature(T, ps), theta_windows)
        if sig in _PART_CACHE:
            pr, pl = _PART_CACHE[sig]
            results[ps.name] = pr
            if pl is not None:
                ledgers.append(pl)
            continue
        pr = PartResult(ps.name, [])
        pl = None
        try:
            comps = part_components(T, ps)
            pruner = make_pruner(S, ps.regions, T.alpha)
            if ps.noise_avg and not ps.is_main:
                nc, info = noise_average(S, comps, ps.fast, pruner=pruner, name=f"N_{ps.name}")
                pr.resonant = info["resonant"]
                pr.noise = info["N"]
                if nc is not None:
                    comps = comps + [nc]
            if theta_windows and ps.is_main:
                # e^{theta H} is bad on the cold-bath faces whatever theta is (that is
                # what the other parts absorb); its theta window is the Ito bound
                # theta < 1/T_b on every other block
                baths0 = [b - 1 for b in S.baths]
                pr.theta_max = theta_window(S, part_components(T, ps), (), use_noise=False,
                                            regions=ps.regions, alpha=T.alpha,
                                            ignore=lambda b: all(i in b.Z for i in baths0))
                pr.theta_max_without_noise = pr.theta_max
            if theta_windows and not ps.is_main:
                base = part_components(T, ps)
                pr.theta_max = theta_window(S, base, ps.fast, use_noise=ps.noise_avg,
                                            regions=ps.regions, alpha=T.alpha)
                pr.theta_max_without_noise = theta_window(S, base, ps.fast, use_noise=False,
                                                          regions=ps.regions, alpha=T.alpha)
            G, K = part_G(S, comps, T.alpha, regions=ps.regions)
            for reg in ps.regions:
                pr.blocks.extend(checker.check(G, reg, label=f"{ps.name}:{reg.name}"))
            logV, ll, gap = part_logsize(T, ps)
            pl = PartLedger(ps.name, ps.regions, pr.blocks, logV, gap, ll,
                            [k[0] for k in G.terms], T.alpha, ps.is_main)
            ledgers.append(pl)
        except (NotQuasiHomogeneous, ValueError) as exc:
            pr.error = f"invalid: {exc}"
        pr.seconds = time.time() - t1
        if verbose:
            print(f"[tmpl] part {ps.name}: {len(pr.blocks)} blocks in {pr.seconds:.1f}s {pr.error}", flush=True)
        results[ps.name] = pr
        _PART_CACHE[sig] = (pr, pl)

    if any(r.error for r in results.values()):
        pen = 1e3
        return Report(results, 0, [], False, [], {}, False, None, [], pen, 1.0 / (1.0 + pen),
                      T.conditions, time.time() - t0)

    params = {THETA: float(T.theta)}
    led = Ledger(S.space, S.r, ledgers, params=params, n_samples=n_samples, seed=seed)
    bad_blocks = 0
    rows = []
    for pl, ps in zip(ledgers, T.parts):
        tau_G = lambda s, pl=pl: tau_at(S.space, pl.G_keys, s)
        # (a) bad blocks of the part, sampled in their closure
        for b in pl.blocks:
            if b.status == "good":
                continue
            bad_blocks += 1
            R, eqs = block_region(b)
            pts = led.sample_polytope(R, eqs, n=max(8, n_samples // 4))
            if len(pts) == 0:
                pts = np.array([b.s_rep])
            lab = f"{pl.name} bad block [{b.status}] Z={b.Z} s~{tuple(np.round(b.s_rep, 3))}"
            led.check_bad_set(pl, lab, pts, tau_G)
            rows += _lp_rows(led, pl, lab, pts, ledgers)
        # (b) switch layers (facets of the part's regions)
        if not ps.is_main:
            tau_layer = lambda s, a=T.alpha: 2.0 * a
            for reg in ps.regions:
                for a_, b_ in reg.ineqs:
                    pts = led.sample_polytope(reg, [(a_, b_)], n=max(8, n_samples // 4))
                    if len(pts) == 0:
                        continue
                    lab = f"{pl.name} layer {reg.name}: {_fmt_ineq(a_, b_)}"
                    led.check_bad_set(pl, lab, pts, tau_layer)
                    rows += _lp_rows(led, pl, lab, pts, ledgers)

    gap_parts = [pl.name for pl in ledgers if pl.gap is not None]
    fixed = dict(_gaps(ledgers))
    lp_ok, _, slacks = solve_lp(rows, gap_parts, fixed=fixed)
    _, hint, hint_slacks = solve_lp(rows, gap_parts)
    bf_ok, lam, cycle = bellman_ford([pl.name for pl in ledgers], led.edges)
    # graded LP term: per bad set, its largest missing exponent
    per_set = {}
    for lab, sg in slacks:
        key = lab.split(" -> ")[0]
        per_set[key] = max(per_set.get(key, 0.0), sg if np.isfinite(sg) else 10.0)
    slack_total = sum(per_set.values())
    # bad blocks / layers that are absorbed cost nothing; unabsorbed ones count 1
    # each, LP slack adds a graded term (0.05 of missing exponent ~ one failure)
    pen = len(led.failures) + slack_total / 0.05 + (5.0 if not bf_ok else 0.0)
    led.edges = sorted(set((e[0], e[1], round(e[2], 6), e[3]) for e in led.edges))
    rep = Report(results, bad_blocks, led.failures, lp_ok, sorted(slacks, key=lambda t: -t[1]),
                 hint if gap_parts else {}, bf_ok, cycle, led.edges, pen, 1.0 / (1.0 + pen),
                 T.conditions, time.time() - t0)
    return rep


def _fmt_ineq(a, b):
    return " + ".join(f"{c:g}*s{i + 1}" for i, c in a.items()) + f" <= {b:g}"


def _pair_rows(led, r, t, label, s, gaps):
    """Rows (sum c_j g_j + rhs >= delta) for power-level absorption of r by t at s,
    with the current gap orders `gaps` deciding which negative term is designated."""
    terms, _ = led._D_terms(r, t)
    n_gap = (r.gap is not None) + (t.gap is not None)
    plain = terms[:len(terms) - n_gap] if n_gap else terms
    P = max((order_of(led.space, e, s) for e, c in plain if c not in ("neg", "zero")), default=-np.inf)
    # the bounded level (order 0) is always available as the "negative" side: a
    # positive term of order P > 0 then misses exactly P (the slack reads as 缺多少)
    Nn = max(max((order_of(led.space, e, s) for e, c in plain if c == "neg"), default=-np.inf), 0.0)
    pos_gap = (r.name, s[r.gap[0]]) if r.gap is not None and s[r.gap[0]] > 1e-9 else None
    neg_gap = (t.name, s[t.gap[0]]) if t.gap is not None and s[t.gap[0]] > 1e-9 else None
    if max(P, Nn) <= 1e-9 and pos_gap is None and neg_gap is None:
        return []
    rows = []
    pos_list = ([("c", P)] if P > 1e-9 else []) + ([("g", pos_gap)] if pos_gap else [])
    use_gap = neg_gap is not None and (Nn == -np.inf or neg_gap[1] * gaps.get(neg_gap[0], 0.0) >= Nn)
    for kind, val in pos_list:
        coef, rhs = {}, 0.0
        if kind == "c":
            rhs -= val
        else:
            coef[val[0]] = coef.get(val[0], 0.0) - val[1]
        if use_gap:
            coef[neg_gap[0]] = coef.get(neg_gap[0], 0.0) + neg_gap[1]
        elif Nn > -np.inf:
            rhs += Nn
        rows.append((f"{label} -> {t.name} @ s={tuple(np.round(s, 3))}", coef, rhs))
    return rows


def _lp_rows(led, r, label, pts, ledgers):
    """LP rows at the sample directions.  Absorber per direction: the first part that
    absorbs at the current exponents (its rows hold with zero slack); otherwise the
    candidate (good there) whose rows need the least total violation."""
    gaps = dict(_gaps(ledgers))
    rows = []
    for s in pts:
        cands = [t for t in ledgers if t is not r and in_regions(t, s) and status_at(t, s) == "good"]
        chosen = None
        best_def = np.inf
        for t in cands:
            verdict, _ = led.absorbed_at(r, t, s, 0.0 if r.is_main else 0.0)
            rr = _pair_rows(led, r, t, label, s, gaps)
            deficit = 0.0
            for _, coef, rhs in rr:
                val = rhs + sum(c * gaps.get(nm, 0.0) for nm, c in coef.items())
                deficit += max(0.0, 1e-3 - val)
            if verdict in ("yes", "const") and deficit <= 1e-12:
                chosen = rr
                break
            if deficit < best_def:
                best_def, chosen = deficit, rr
        if chosen:
            rows.extend(chosen)
    return rows


def _gaps(ledgers):
    """current gap orders g_r = a_r * (exponent of the gap variable's lead), the LP variables"""
    return [(pl.name, pl.gap[1]) for pl in ledgers if pl.gap is not None]
