"""Automatic averaging steps (spec 2026-10-17 sections 1.3 and 2.1).

* slow_momentum: single-fast averaging of p_j (CE arXiv:1504.04964 section 2.1,
  CEP2015): order by order in 1/p_j, Fourier modes of L(.) that oscillate in q_j
  are cancelled by (a sin phi - b cos phi)/(omega p_j); modes that do not move
  with q_j but oscillate in slow angles are cancelled only when they are an exact
  slow L-derivative (polynomial divisibility by the slow rate -- the paper's
  L<W_L W_C> step); everything else is left and reported.
* noise_average: the factor (1 + N_r) that replaces oscillating W^2 terms of the
  part's leading order by their averages (CE Prop. 2.4 footnote 3, Prop. 3.5).
"""
import sympy as sym

from utils.fl_multiphase import _mode_table


def laurent_coeffs(expr, x, K=40):
    """{exponent: coefficient} of a Laurent polynomial in x (coefficients free of x)."""
    e = sym.expand(sym.expand(expr) * x**K)
    poly = sym.Poly(e, x)
    out = {}
    for (k,), c in poly.terms():
        out[k - K] = sym.expand(c)
    return out


class AveragingResult:
    def __init__(self, expr, mean, residuals, orders_done):
        self.expr = expr                 # the slow momentum p~_j (sympy)
        self.mean = mean                 # {exponent: mean coefficient of L p~_j}
        self.residuals = residuals       # list of (exponent, mode, reason)
        self.orders_done = orders_done

    def __repr__(self):
        return f"AveragingResult(mean={self.mean}, residuals={len(self.residuals)})"


def slow_momentum(S, j, target=3, max_iter=40, verbose=False):
    """p~_j = p_j + sum_k C_k / p_j^k with L p~_j = <mean> + O_j(p_j^{-target-1})."""
    pj = S.P(j)
    others = [S.P(i) for i in range(1, S.N + 1) if i != j]
    pt = pj
    residuals = []
    skip = set()                       # (exponent, mode) left in place
    for it in range(max_iter):
        R = S.L(pt)
        co = laurent_coeffs(R, pj)
        acted = False
        for e in sorted(co, reverse=True):
            if e < -target:
                break
            table = _mode_table(co[e], S.r)
            C = sym.S.Zero
            for n, (a, b) in table.items():
                if all(v == 0 for v in n) or (a == 0 and b == 0) or (e, n) in skip:
                    continue
                phi = sum(ni * ri for ni, ri in zip(n, S.r))
                f = S.fast_frequency(n, j)
                if f != 0:
                    C += (a * sym.sin(phi) - b * sym.cos(phi)) / f * pj**(e - 1)
                    continue
                Om = S.slow_rate(n, [j])
                if Om == 0:
                    residuals.append((e, n, "non-oscillating mode, rate identically 0"))
                    skip.add((e, n))
                    continue
                try:
                    qa, ra = sym.div(sym.expand(a), Om, *others)
                    qb, rb = sym.div(sym.expand(b), Om, *others)
                except sym.PolynomialError:
                    ra = rb = sym.S.One
                if sym.expand(ra) == 0 and sym.expand(rb) == 0:
                    C += (qa * sym.sin(phi) - qb * sym.cos(phi)) * pj**e
                else:
                    residuals.append((e, n, f"slow mode not an exact L-derivative (rate {Om})"))
                    skip.add((e, n))
            if C != 0:
                pt = sym.expand(pt - C)
                acted = True
                if verbose:
                    print(f"  iter {it}: cancelled order p^{e}")
                break
        if not acted:
            break
    co = laurent_coeffs(S.L(pt), pj)
    mean = {}
    for e in sorted(co, reverse=True):
        if e < -target:
            break
        t = _mode_table(co[e], S.r)
        m = t.get(tuple(0 for _ in S.r), (sym.S.Zero, sym.S.Zero))[0]
        if sym.simplify(m) != 0:
            mean[e] = sym.simplify(m)
    return AveragingResult(pt, mean, residuals, target)


# ---- noise averaging ----------------------------------------------------------------
def fast_modes_cancel(S, coeff, fast, order_exp):
    """Counterterm n(q, p) / p_f^... cancelling the fast-oscillating modes of the
    expression `coeff * p^order` where p is the part's scale.  For each mode that
    oscillates in exactly one fast angle q_f with frequency omega, the term
        N += (a sin phi - b cos phi) / (omega * p_f)   (times the same power)
    makes p_f d/dq_f N = mode.  Returns (N_numerator_per_fast, resonant_modes)."""
    table = _mode_table(coeff, S.r)
    out = {f: sym.S.Zero for f in fast}
    resonant = []
    for n, (a, b) in table.items():
        if all(v == 0 for v in n) or (a == 0 and b == 0):
            continue
        freqs = {f: S.fast_frequency(n, f) for f in fast}
        moving = [f for f, w in freqs.items() if w != 0]
        if len(moving) == 1:
            f = moving[0]
            phi = sum(ni * ri for ni, ri in zip(n, S.r))
            out[f] += (a * sym.sin(phi) - b * sym.cos(phi)) / freqs[f]
        elif len(moving) == 0:
            resonant.append((n, "not fast"))
        else:
            resonant.append((n, "moves with several fast angles"))
    return out, resonant
