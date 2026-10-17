"""Free-grammar local shapes (spec 2026-10-17 section 10).

A part is V_r = g(|Y|) * exp(theta E_r) * (1 + N_r): the averaged quantity Y (p~_j from
the averaging macro) and the energy factor exp(theta E_r) stay fixed, the shape g of
one variable u = |Y| > 0 is an expression tree sampled by the generator over

    add sub mul div exp log sqrt n2 | u 1 2

The evaluator only needs phi = log g through phi'(u), phi''(u) as u -> oo.  They are
expanded as Puiseux series sum_q c_q u^q (t = 1/u, sympy series at t = 0+) and pushed
through Y with phi'(Y) = sign(Y) sum c_q |Y|^q, phi''(Y) = sum d_q |Y|^q (u = |Y|).

The log size (splice ledger) is phi integrated term by term: the largest positive
power c u^a becomes the part's gap (0 < a < 2, c > 0) and c_{-1} log u the log level.

Rejected (InvalidShape): g not positive/finite for large u, phi' or phi'' with log
factors (not Puiseux), phi growing like u^2 or faster (that would change theta),
a decaying leading power (c < 0), sympy timeouts.
"""
import signal
from fractions import Fraction

import numpy as np
import sympy as sym

U = sym.Symbol("u", positive=True)
_T = sym.Symbol("t", positive=True)

G_TOKENS = ["add", "sub", "mul", "div", "exp", "log", "sqrt", "n2", "u", "1", "2"]
_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "exp": 1, "log": 1, "sqrt": 1, "n2": 1}
_OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b, "exp": sym.exp, "log": sym.log, "sqrt": sym.sqrt,
        "n2": lambda a: a**2}


class InvalidShape(ValueError):
    pass


def root_name(n_free):
    return f"free{n_free}"


def _terminal(nm):
    if nm == "u":
        return U
    try:
        v = float(nm)
    except ValueError:
        return None
    return sym.nsimplify(v)


def decode_tree(names, i=0):
    """Pre-order token names from index i -> (sympy expr, next index); None if malformed."""
    if i >= len(names):
        return None, i
    nm = names[i]
    if nm in _ARITY:
        args = []
        j = i + 1
        for _ in range(_ARITY[nm]):
            a, j = decode_tree(names, j)
            if a is None:
                return None, j
            args.append(a)
        return _OPS[nm](*args), j
    t = _terminal(nm)
    return t, i + 1


def decode_free(n_free, names):
    """free<n>(g_1, ..., g_n) -> [g_1, ..., g_n] (sympy in u), or None if malformed."""
    names = list(names)
    if not names or names[0] != root_name(n_free):
        return None
    out, i = [], 1
    for _ in range(n_free):
        g, i = decode_tree(names, i)
        if g is None:
            return None
        out.append(g)
    return out if i == len(names) else None


class _Timeout:
    def __init__(self, seconds):
        self.s = int(seconds)

    def _raise(self, *_):
        raise InvalidShape("sympy timeout")

    def __enter__(self):
        if self.s > 0:
            try:
                self.old = signal.signal(signal.SIGALRM, self._raise)
                signal.alarm(self.s)
            except ValueError:          # not in the main thread
                self.s = 0
        return self

    def __exit__(self, *exc):
        if self.s > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old)
        return False


def _puiseux(expr, nterms):
    """expr(u) as u -> oo: ([(q, c)], q_rem) with expr = sum c u^q + O(u^q_rem)."""
    e = sym.simplify(expr.subs(U, 1 / _T)) if expr.has(U) else expr
    if not e.has(_T):
        c = sym.nsimplify(e)
        return ([(Fraction(0), c)] if c != 0 else []), None
    # leading exponent first, then enough terms beyond it
    lead = sym.limit(sym.log(sym.Abs(e)) / sym.log(_T), _T, 0, "+")
    if not lead.is_finite:
        raise InvalidShape(f"no power-law asymptotics: {expr}")
    n = int(np.floor(float(lead))) + nterms + 1
    ser = sym.series(e, _T, 0, n)
    if ser.has(sym.log) or ser.has(sym.exp):
        raise InvalidShape(f"non-Puiseux asymptotics: {expr}")
    O = ser.getO()
    body = sym.expand(ser.removeO())
    q_rem = None
    if O is not None:
        oe = O.args[0]
        k = sym.degree(oe, _T) if oe.is_polynomial(_T) else sym.Poly(oe, _T).degree()
        q_rem = Fraction(-int(k))
    terms = {}
    for term in sym.Add.make_args(body):
        if term == 0:
            continue
        c, rest = term.as_independent(_T, as_Add=False)
        if rest == 1:
            p = sym.S.Zero
        else:
            b, p = rest.as_base_exp()
            if b != _T or p.free_symbols:
                raise InvalidShape(f"non-Puiseux term {term} in {expr}")
        q = Fraction(-sym.Rational(p).p, sym.Rational(p).q)
        terms[q] = terms.get(q, 0) + c
    out = sorted(((q, sym.nsimplify(c)) for q, c in terms.items() if c != 0), key=lambda t: -t[0])
    return out, q_rem


MAX_OPS = 24          # count_ops(g) cap (cf. C50: nested monsters stalled the judge)
MAX_UNARY_DEPTH = 3   # nesting depth of exp/log/sqrt


def _unary_depth(e):
    own = 1 if isinstance(e, (sym.exp, sym.log)) or (isinstance(e, sym.Pow) and e.exp.is_Rational
                                                     and e.exp.q != 1) else 0
    return own + max((_unary_depth(a) for a in e.args), default=0)


_SHAPE_CACHE = {}


def expand_shape(g, nterms=4, timeout=3):
    key = (str(g), nterms)
    if key in _SHAPE_CACHE:
        hit = _SHAPE_CACHE[key]
        if isinstance(hit, Exception):
            raise hit
        return hit
    try:
        out = _expand_shape(g, nterms, timeout)
    except InvalidShape as exc:
        _SHAPE_CACHE[key] = exc
        raise
    _SHAPE_CACHE[key] = out
    return out


def _expand_shape(g, nterms, timeout):
    """Asymptotic data of phi = log g(u):
       {'phi1': [(q, c)], 'phi1_rem': q|None, 'phi2': ..., 'phi2_rem': ...,
        'gap': a|None (leading positive power of phi), 'gap_coef': c,
        'log_coef': coefficient of log u in phi}"""
    g = sym.sympify(g)
    if g.has(sym.zoo, sym.nan, sym.oo, -sym.oo, sym.I):
        raise InvalidShape(f"g is not a finite real expression: {g}")
    if sym.count_ops(g) > MAX_OPS or _unary_depth(g) > MAX_UNARY_DEPTH:
        raise InvalidShape(f"g too complex (ops {sym.count_ops(g)}, depth {_unary_depth(g)})")
    with _Timeout(timeout):
        f = sym.lambdify(U, g, "numpy")
        with np.errstate(all="ignore"):
            vals = np.array([complex(f(x)) for x in (3.0, 10.0, 30.0, 100.0)])
        if not np.all(np.isfinite(vals)) or np.any(np.abs(vals.imag) > 1e-9) or np.any(vals.real <= 0):
            raise InvalidShape(f"g not positive and finite for large u: {g}")
        if not g.has(U):
            return {"g": str(g), "phi1": [], "phi1_rem": None, "phi2": [], "phi2_rem": None,
                    "gap": None, "gap_coef": 0, "log_coef": 0}
        phi = sym.log(g)
        d1 = sym.diff(phi, U)
        d2 = sym.diff(d1, U)
        try:
            p1, r1 = _puiseux(d1, nterms)
            p2, r2 = _puiseux(d2, nterms)
        except InvalidShape:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise InvalidShape(f"asymptotic expansion failed for {g}: {type(exc).__name__}")
    # phi = sum c_q u^(q+1)/(q+1) + c_{-1} log u + const
    gap, gap_c, log_c = None, 0, 0
    for q, c in p1:
        if q == -1:
            log_c = c
        elif q > -1 and gap is None:
            gap, gap_c = q + 1, c / (q + 1)
    if gap is not None:
        if gap >= 2:
            raise InvalidShape(f"log g grows like u^{gap} (>= u^2 changes theta): {g}")
        if float(gap_c) <= 0:
            raise InvalidShape(f"decaying correction exp(-c u^{gap}): {g}")
    return {"g": str(g), "phi1": p1, "phi1_rem": r1, "phi2": p2, "phi2_rem": r2,
            "gap": gap, "gap_coef": gap_c, "log_coef": log_c}


def phi_derivs_free(Y, lead_key, data, nmax):
    """phi'(Y), phi''(Y) as AsymSums for u = |Y| (Y's lead monomial lead_key)."""
    from utils.fl_tmpl.asym import AsymSum
    sp = Y.space
    out = []
    for terms, q_rem, odd in ((data["phi1"], data["phi1_rem"], True),
                              (data["phi2"], data["phi2_rem"], False)):
        acc = AsymSum(sp)
        for q, c in terms:
            if odd:                     # sign(Y) |Y|^q = Y |Y|^(q-1)
                acc = acc + (Y * Y.power(q - 1, lead_key, nmax, absval=True)).scale(c)
            else:
                acc = acc + Y.power(q, lead_key, nmax, absval=True).scale(c)
        if q_rem is not None:
            acc.rem.add(tuple(q_rem * a for a in lead_key[0]))
        out.append(acc)
    return out[0], out[1]


def shape_str(g):
    return str(sym.simplify(g))
