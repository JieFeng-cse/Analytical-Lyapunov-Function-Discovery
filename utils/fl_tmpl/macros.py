"""Macro tokens: building blocks with known expansion data (spec 2026-10-17 section 2.1).

A Macro carries the AsymSum expansions of its value, of its generator image and of
its derivatives in every noisy momentum, plus provenance.  The evaluator never
needs more than these data (chain rule, logdomain.py), so a macro can come from
an explicit expression (computed), from a published proposition (cited), from a
verified-parts file (danus) or from a labelled hypothesis (conjecture).
"""
import json
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sym

from utils.fl_tmpl.asym import AsymSum
from utils.fl_tmpl.averaging import slow_momentum


@dataclass
class Macro:
    name: str
    kind: str                         # slow_momentum | slow_energy | explicit
    fast: tuple                       # fast momenta (1-based)
    value: AsymSum
    L: AsymSum
    d: dict                           # {bath b (1-based): AsymSum of d/dp_b}
    lead_key: tuple                   # key of the leading monomial of `value`
    provenance: str                   # computed | cited | danus | conjecture
    expr: object = None               # explicit sympy expression when available
    ref: str = ""
    conditions: list = field(default_factory=list)

    def check_bath_derivatives(self, baths):
        missing = [b for b in baths if b not in self.d]
        if missing:
            raise ValueError(f"macro {self.name}: missing d/dp_b data for baths {missing} "
                             "(design rule: every slow variable tracks its bath derivatives)")


def explicit_macro(S, name, expr, lead=None, kind="explicit", fast=()):
    """Macro from an explicit expression: exact data by symbolic L and d/dp_b."""
    value = S.asym(expr)
    Lv = S.asym(S.L(expr))
    d = {b: S.asym(S.d(expr, b)) for b in S.baths}
    lead_key = lead if lead is not None else S.space.zero_key()
    return Macro(name, kind, tuple(fast), value, Lv, d, lead_key, "computed", expr=expr)


def computed_slow_momentum(S, j, target=3):
    res = slow_momentum(S, j, target=target)
    pj = S.P(j)
    lead = S.space.monomial_key({pj: 1}, {pj: 1})
    m = explicit_macro(S, f"pt{j}", res.expr, lead=lead, kind="slow_momentum", fast=(j,))
    m.ref = f"single-fast averaging to order p^-{target}; mean {res.mean}"
    m.conditions = [f"residual {r}" for r in res.residuals]
    m.averaging = res
    return m


def _rem_poly(S, base_exps, slow, degree):
    """Remainder set O(z(slow) * base): base, base*|p_b|^degree for each slow b."""
    rems = []
    k0 = S.space.monomial_key(base_exps)[0]
    rems.append(k0)
    for b in slow:
        k = list(k0)
        k[S.space._index[S.P(b)]] += Fraction(degree)
        rems.append(tuple(k))
    return rems


def cited_Hc_CE4(S, rem_degree=2):
    """H~_c of Cuneo-Eckmann arXiv:1504.04964 Prop. 3.2 (N=4, both central rotors fast).

        H~_c       = (p2^2+p3^2)/2 + O_c(1)
        L H~_c     = -alpha_2/p2^2 - alpha_3/p3^2 + O_c(|p2|^{-5/2} + |p3|^{-5/2})
        d_p1 H~_c  = W_L/p2 + O_c(p2^{-2}),   d_p4 H~_c = W_R/p3 + O_c(p3^{-2})

    alpha_2 = gamma_1 <W_L^2>, alpha_3 = gamma_4 <W_R^2>; W = -cos (zero mean),
    W_L = W(r1), W_R = W(q3 - q4) = -cos(r3).  O_c(g) means |.| <= z(p1, p4) |g| for
    a polynomial z; the remainder sets carry z as |p_b|^rem_degree.  The paper does
    not state deg z; rem_degree is a modelling choice (spec section 8) and enters the
    "m large enough" relation: the remainder stays below the leading order on
    Omega_c(l, m) iff rem_degree/m < (1/2)/l, i.e. m > 2 l rem_degree."""
    assert S.N == 4
    p1, p2, p3, p4 = S.p
    A = S.space.add_atom("A23", p2**2 + p3**2)
    akey = S.space.monomial_key({"A23": 1})
    value = AsymSum(S.space, {akey: S.space.coef(sym.Rational(1, 2))})
    value.rem |= set(_rem_poly(S, {}, [1, 4], rem_degree))
    a2 = S.gamma[1] * sym.Rational(1, 2)
    a3 = S.gamma[4] * sym.Rational(1, 2)
    Lv = S.asym(-a2 / p2**2 - a3 / p3**2)
    Lv.rem |= set(_rem_poly(S, {p2: sym.Rational(-5, 2)}, [1, 4], rem_degree))
    Lv.rem |= set(_rem_poly(S, {p3: sym.Rational(-5, 2)}, [1, 4], rem_degree))
    d1 = S.asym(-sym.cos(S.R(1)) / p2)
    d1.rem |= set(_rem_poly(S, {p2: -2}, [1, 4], rem_degree))
    d4 = S.asym(-sym.cos(S.R(3)) / p3)
    d4.rem |= set(_rem_poly(S, {p3: -2}, [1, 4], rem_degree))
    return Macro("Hc", "slow_energy", (2, 3), value, Lv, {1: d1, 4: d4}, akey, "cited",
                 ref="Cuneo-Eckmann, CMP 345 (2016), arXiv:1504.04964, Prop. 3.2 (eqs. 3.2-3.4)")


def main_energy(S):
    """H itself (for the e^{theta H} part); exact."""
    return explicit_macro(S, "H", S.H, kind="explicit")


# ---- Danus / conjecture loaders -------------------------------------------------------
def _parse_series(S, items, locals_):
    out = AsymSum(S.space)
    for coef, exps in items:
        e = {S.P(int(k[1:])) if k.startswith("p") else k: Fraction(v) for k, v in exps.items()}
        par = {S.P(int(k[1:])): int(Fraction(v)) % 2 for k, v in exps.items()
               if k.startswith("p") and Fraction(v).denominator == 1}
        out = out + AsymSum.monomial(S.space, sym.sympify(coef, locals=locals_), e, par)
    return out


def load_macros_json(S, path):
    """Load macros from the JSON schema of spec section 2.4 (Danus verified parts)."""
    with open(path) as fh:
        spec = json.load(fh)
    locals_ = {f"r{i}": S.R(i) for i in range(1, S.N)}
    for c in spec.get("constants", {}):
        locals_[c] = sym.Symbol(c, positive=True)
    out = {}
    for m in spec["macros"]:
        value = _parse_series(S, m["value"], locals_)
        Lv = _parse_series(S, m["L"], locals_)
        d = {int(k[1:]): _parse_series(S, v, locals_) for k, v in m["d"].items()}
        for which, rems in m.get("remainders", {}).items():
            tgt = value if which == "value" else Lv if which == "L" else d[int(which[1:])]
            for r in rems:
                tgt.rem.add(S.space.monomial_key({S.P(int(k[1:])): Fraction(v) for k, v in r.items()})[0])
        lead = m.get("lead", {f"p{m['fast'][0]}": "1"})
        lead_key = S.space.monomial_key({S.P(int(k[1:])): Fraction(v) for k, v in lead.items()},
                                        {S.P(int(k[1:])): int(Fraction(v)) % 2 for k, v in lead.items()})
        mac = Macro(m["name"], m["kind"], tuple(m["fast"]), value, Lv, d, lead_key,
                    m.get("provenance", "danus"), ref=m.get("ref", ""))
        mac.check_bath_derivatives(S.baths)
        out[mac.name] = mac
    return out


# ---- N=5 (conjectural template, spec section 2.5) ---------------------------------------
C7 = sym.Symbol("c7", positive=True)       # L p~3 ~ -c7 p3^-7 (central bottleneck)
AP = sym.Symbol("ap", positive=True)       # leak rate of a fast rotor into a slow interior neighbour


def atomize(S, A, atom_name, coef):
    """Rewrite coef * (monomials of the atom) inside A as coef * atom (so the macro
    has a single lead monomial for its series)."""
    sp = S.space
    j = sp.atom_names().index(atom_name)
    out = A.copy()
    c = sp.coef(coef)
    for en, cc in sp.atoms[j].monomials:
        k = (tuple(list(en) + [Fraction(0)] * sp.nA), tuple(0 for _ in range(sp.nP)))
        cur = out.terms.get(k)
        sub = c * sp.coef(cc)
        if cur is None or not (cur - sub).is_zero():
            raise ValueError(f"atomize: {atom_name} monomial {en} not present with coefficient {coef}")
        del out.terms[k]
    akey = sp.monomial_key({atom_name: 1})
    out.terms[akey] = c
    return out, akey


def explicit_pair_energy(S, mi, mj, atom_name):
    """E = (p~_i^2 + p~_j^2)/2 from two computed slow momenta (two separated fast
    rotors, e.g. p2 and p4 in N=5): exact data by symbolic L."""
    expr = (mi.expr**2 + mj.expr**2) / 2
    m = explicit_macro(S, f"E{mi.fast[0]}{mj.fast[0]}", expr, kind="slow_energy",
                       fast=mi.fast + mj.fast)
    m.value, m.lead_key = atomize(S, m.value, atom_name, sym.Rational(1, 2))
    m.ref = f"(p~{mi.fast[0]}^2 + p~{mj.fast[0]}^2)/2, computed"
    return m


def conjecture_bottleneck(S, base, rem_degree=2, noise_order=5):
    """p~3 for N=5: value = computed first averaging layer (base), and the
    conjectured second-layer drift  L p~3 = -c7 p3^-7 + O(z p3^-8)  (benchmark eq. 29,
    CE Remark 5.3, CONCLUSIONS C41/C42); bath derivatives d_{p_b} p~3 = O(z p3^-noise_order)."""
    p3 = S.P(3)
    slow = [b for b in (1, 2, 4, 5)]
    Lv = S.asym(-C7 / p3**7)
    Lv.rem |= set(_rem_poly(S, {p3: -8}, slow, rem_degree))
    d = {}
    for b in S.baths:
        A = AsymSum(S.space)
        A.rem |= set(_rem_poly(S, {p3: -noise_order}, slow, rem_degree))
        d[b] = A
    m = Macro("pt3c", "slow_momentum", (3,), base.value, Lv, d, base.lead_key, "conjecture",
              expr=None, ref="conjecture: L p~3 = -c7 p3^-7 (second averaging layer open, C42)")
    m.conditions = ["c7 > 0 (central bottleneck rate, unproved)",
                    f"d_p_b p~3 = O(p3^-{noise_order})"]
    return m


def conjecture_pair_energy(S, fast, rem_degree=2):
    """H~ for two adjacent fast interior rotors with the third interior rotor slow
    (N=5: (2,3) or (3,4)).  Conjecture: the outer fast rotor leaks into its bath
    neighbour with the CE rate alpha = gamma <W^2> = 1/2, the inner one into its slow
    interior neighbour with an unknown rate ap > 0:
        L H~ = -alpha/p_out^2 - ap/p_in^2 + O(z (|p_out|^-5/2 + |p_in|^-5/2)).
    """
    i, j = fast
    out_, in_ = (i, j) if i == 2 else (j, i)          # rotor next to a bath first
    bath = 1 if out_ == 2 else 5
    other_bath = 5 if bath == 1 else 1
    slow = [b for b in range(1, S.N + 1) if b not in fast]
    atom = f"A{min(i, j)}{max(i, j)}"
    akey = S.space.monomial_key({atom: 1})
    value = AsymSum(S.space, {akey: S.space.coef(sym.Rational(1, 2))})
    value.rem |= set(_rem_poly(S, {}, slow, rem_degree))
    po, pi = S.P(out_), S.P(in_)
    Lv = S.asym(-sym.Rational(1, 2) / po**2 - AP / pi**2)
    Lv.rem |= set(_rem_poly(S, {po: sym.Rational(-5, 2)}, slow, rem_degree))
    Lv.rem |= set(_rem_poly(S, {pi: sym.Rational(-5, 2)}, slow, rem_degree))
    bond = S.R(1) if bath == 1 else S.R(S.N - 1)
    d_b = S.asym(-sym.cos(bond) / po)
    d_b.rem |= set(_rem_poly(S, {po: -2}, slow, rem_degree))
    d_o = AsymSum(S.space)
    d_o.rem |= set(_rem_poly(S, {pi: -2}, slow, rem_degree))
    m = Macro(f"H{min(i, j)}{max(i, j)}", "slow_energy", tuple(sorted(fast)), value, Lv,
              {bath: d_b, other_bath: d_o}, akey, "conjecture",
              ref=f"conjecture: L H~ = -1/2 p{out_}^-2 - ap p{in_}^-2")
    m.conditions = ["ap > 0 (leak into a slow interior neighbour, unproved)"]
    return m


def conjecture_receiver(S, rem_degree=2):
    """Global receiver for N=5, all three interior rotors fast, both baths cold.
    Conjecture: the total interior energy leaks only through the outer interior
    rotors, each with the CE rate:
        L H~234 = -1/2 p2^-2 - 1/2 p4^-2 + O(z (|p2|^-5/2 + |p4|^-5/2)),
        d_p1 H~234 = W(r1)/p2 + O(z p2^-2),  d_p5 H~234 = W(r4)/p4 + O(z p4^-2)."""
    slow = [1, 5]
    akey = S.space.monomial_key({"A234": 1})
    value = AsymSum(S.space, {akey: S.space.coef(sym.Rational(1, 2))})
    value.rem |= set(_rem_poly(S, {}, slow, rem_degree))
    p2, p4 = S.P(2), S.P(4)
    Lv = S.asym(-sym.Rational(1, 2) / p2**2 - sym.Rational(1, 2) / p4**2)
    Lv.rem |= set(_rem_poly(S, {p2: sym.Rational(-5, 2)}, slow, rem_degree))
    Lv.rem |= set(_rem_poly(S, {p4: sym.Rational(-5, 2)}, slow, rem_degree))
    d1 = S.asym(-sym.cos(S.R(1)) / p2)
    d1.rem |= set(_rem_poly(S, {p2: -2}, slow, rem_degree))
    d5 = S.asym(-sym.cos(S.R(4)) / p4)
    d5.rem |= set(_rem_poly(S, {p4: -2}, slow, rem_degree))
    m = Macro("H234", "slow_energy", (2, 3, 4), value, Lv, {1: d1, 5: d5}, akey, "conjecture",
              ref="conjecture: L H~234 = -1/2 p2^-2 - 1/2 p4^-2 (receiver; resonances and multi-scale averaging open, CE Remark 5.3)")
    m.conditions = ["receiver H~234 drift (unproved; resonant surfaces p2 ~ p3 ~ p4 not treated)"]
    return m
