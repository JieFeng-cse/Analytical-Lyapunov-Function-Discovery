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
