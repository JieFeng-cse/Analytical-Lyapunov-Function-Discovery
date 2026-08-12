"""Multi-phase averaging engine (CONCLUSIONS C38 -> C39).

Mechanizes the counterterm construction of CEP2015/2016 for trig-polynomial
Hamiltonian chains: rewrite L(H) exactly as a finite trigonometric polynomial
in the bond angles, and cancel each oscillatory Fourier mode by dividing by its
EXACT rotation rate Omega_n = sum_i n_i * (dr_i/dt).

    mode  a(p)cos(phi) + b(p)sin(phi),   dphi/dt = Omega
    counterterm  C = (a sin(phi) - b cos(phi)) / Omega,   L(C) = mode + h.o.t.

Legality (see C38 discussion): the decomposition is an exact identity (product-
to-sum); mode-wise division is the classical solution of the cohomological
equation (the papers' Q operator is its single-phase case); small denominators
are never divided -- modes are SKIPPED when their leading rate vanishes
identically (n1=n2=...) or lies in a user-declared resonant class; the final
arbiter is always the independent pointwise verification of the assembled V.

The engine returns counterterms whose denominators are polynomials in p that
vanish only outside the intended gated region; wrap usage in the caller's gate.
"""
import sympy as sym
from sympy.simplify.fu import TR8, TR10


def _mode_table(expr, angles):
    """Exact rewrite of a trig polynomial as {n_tuple: (a, b)} with the term
    a*cos(n.r) + b*sin(n.r); the zero tuple holds the non-oscillatory part in a."""
    e = sym.expand(TR8(sym.expand_trig(sym.expand(expr))))
    table = {}
    def add(n, a=0, b=0):
        n = tuple(int(v) for v in n)
        # canonical sign: first nonzero component positive
        for v in n:
            if v != 0:
                if v < 0:
                    n = tuple(-x for x in n)
                    a, b = a, -b
                break
        A, B = table.get(n, (sym.S.Zero, sym.S.Zero))
        table[n] = (A + a, B + b)
    for term in sym.Add.make_args(e):
        trig = None
        coeff = sym.S.One
        for f in sym.Mul.make_args(term):
            if isinstance(f, (sym.sin, sym.cos)) and f.free_symbols & set(angles):
                if trig is not None:      # product survived TR8 -> reapply
                    for t2 in sym.Add.make_args(sym.expand(TR8(term))):
                        if t2 != term:
                            for n, (a, b) in _mode_table(t2, angles).items():
                                add(n, a, b)
                    trig = "handled"
                    break
                trig = f
            else:
                coeff *= f
        if trig == "handled":
            continue
        if trig is None:
            add((0,) * len(angles), a=coeff)
            continue
        arg = sym.expand(trig.args[0])
        n = [arg.coeff(t) for t in angles]
        rest = sym.simplify(arg - sum(c * t for c, t in zip(n, angles)))
        if any(not c.is_integer for c in n) or rest != 0:
            raise ValueError(f"non-integer/shifted mode: {trig}")
        if isinstance(trig, sym.cos):
            add(n, a=coeff)
        else:
            add(n, b=coeff)
    return table


def average(H0, state_vars, drift, noise, angles, rates, skip,
            n_iter=3, drop_order=None, gate=None, verbose=False):
    """Iteratively cancel oscillatory modes of L(H).

    angles: list of angle symbols; rates: dict {angle: exact d(angle)/dt expr};
    skip(n): True to leave the mode alone (slow / resonant classes);
    drop_order: optional callable(counterterm)->bool to discard negligible terms;
    Returns (H_tilde, last_mode_table).
    """
    from utils.fl_reward import generator_expr
    H = H0
    table = None
    for it in range(n_iter):
        R = generator_expr(H, state_vars, drift, noise)
        table = _mode_table(R, angles)
        added = 0
        for n, (a, b) in table.items():
            if all(v == 0 for v in n) or skip(n):
                continue
            if a == 0 and b == 0:
                continue
            Omega = sym.expand(sum(v * rates[t] for v, t in zip(n, angles)))
            phi = sum(v * t for v, t in zip(n, angles))
            C = (a * sym.sin(phi) - b * sym.cos(phi)) / Omega
            if drop_order is not None and drop_order(C):
                continue
            H = H - C
            added += 1
        if verbose:
            print(f"  iter {it}: cancelled {added} modes")
        if added == 0:
            break
    return H, table
