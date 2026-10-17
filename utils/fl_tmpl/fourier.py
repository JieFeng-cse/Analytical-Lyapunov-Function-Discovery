"""Fourier coefficients for asymptotic sums (spec 2026-10-17 section 1.2).

A Coef is a finite sum  sum v * prod_k par_k^{e_k} * trig(n . r)  with trig in
{cos, sin}, n an integer vector over the bond angles in canonical sign (first
nonzero component positive), v a Fraction (exact) or float (when an irrational
number appears, e.g. (1/2)^{1/2}).  Products use product-to-sum, so the
representation is canonical and identically-zero combinations cancel exactly.

This replaces sympy coefficients inside AsymSum: sympy's TR8 on large products
was the bottleneck (minutes per part).
"""
from fractions import Fraction

import numpy as np
import sympy as sym

ZERO_TOL = 1e-13


def _canon_mode(n, trig, v):
    """(n, trig, v) -> canonical (n', trig, v') or None if the term vanishes."""
    for x in n:
        if x != 0:
            if x < 0:
                n = tuple(-y for y in n)
                if trig == 1:
                    v = -v
            break
    else:
        if trig == 1:
            return None
    return n, trig, v


def _num(x):
    if isinstance(x, (Fraction, float)):
        return x
    if isinstance(x, int):
        return Fraction(x)
    x = sym.nsimplify(x) if not isinstance(x, sym.Basic) else x
    if isinstance(x, sym.Rational):
        return Fraction(int(x.p), int(x.q))
    return float(x)


class CoefContext:
    """Angles and parameter symbols shared by all Coefs of a Space."""

    def __init__(self, angles, params):
        self.angles = tuple(angles)
        self.params = tuple(params)
        self.nA = len(self.angles)
        self.nPar = len(self.params)
        self.zero_n = tuple(0 for _ in self.angles)
        self.zero_pe = tuple(0 for _ in self.params)


class Coef:
    __slots__ = ("ctx", "d")

    def __init__(self, ctx, d=None):
        self.ctx = ctx
        self.d = d if d is not None else {}       # (n, trig, pe) -> value

    # -- construction --
    @classmethod
    def const(cls, ctx, v):
        v = _num(v)
        if v == 0:
            return cls(ctx)
        return cls(ctx, {(ctx.zero_n, 0, ctx.zero_pe): v})

    @classmethod
    def from_sympy(cls, ctx, expr):
        from utils.fl_multiphase import _mode_table
        expr = sym.expand(sym.sympify(expr))
        if expr == 0:
            return cls(ctx)
        if not (expr.free_symbols & set(ctx.angles)):
            table = {ctx.zero_n: (expr, sym.S.Zero)}
        else:
            table = _mode_table(expr, list(ctx.angles))
        out = cls(ctx)
        for n, (a, b) in table.items():
            for trig, c in ((0, a), (1, b)):
                c = sym.expand(c)
                if c == 0:
                    continue
                if ctx.params:
                    if c.free_symbols - set(ctx.params):
                        raise ValueError(f"coefficient depends on non-parameter symbols: {c}")
                    poly = sym.Poly(c, *ctx.params)
                    items = [(tuple(int(e) for e in mon), v) for mon, v in poly.terms()]
                else:
                    if c.free_symbols:
                        raise ValueError(f"coefficient has free symbols {c.free_symbols} (no params declared)")
                    items = [((), c)]
                for pe, v in items:
                    cm = _canon_mode(tuple(int(x) for x in n), trig, _num(v))
                    if cm is None:
                        continue
                    key = (cm[0], trig, pe)
                    out.d[key] = out.d.get(key, 0) + cm[2]
        return out.clean()

    # -- algebra --
    def clean(self):
        self.d = {k: v for k, v in self.d.items()
                  if not (v == 0 or (isinstance(v, float) and abs(v) < ZERO_TOL))}
        return self

    def is_zero(self):
        return not self.d

    def copy(self):
        return Coef(self.ctx, dict(self.d))

    def __add__(self, other):
        out = dict(self.d)
        for k, v in other.d.items():
            out[k] = out.get(k, 0) + v
        return Coef(self.ctx, out).clean()

    def __neg__(self):
        return Coef(self.ctx, {k: -v for k, v in self.d.items()})

    def __sub__(self, other):
        return self + (-other)

    def scale(self, c):
        c = _num(c)
        if c == 0:
            return Coef(self.ctx)
        return Coef(self.ctx, {k: v * c for k, v in self.d.items()})

    def __mul__(self, other):
        if not isinstance(other, Coef):
            return self.scale(other)
        out = {}
        zn = self.ctx.zero_n
        for (n1, t1, e1), v1 in self.d.items():
            for (n2, t2, e2), v2 in other.d.items():
                pe = tuple(a + b for a, b in zip(e1, e2))
                if n1 == zn and t1 == 0:
                    k = (n2, t2, pe)
                    out[k] = out.get(k, 0) + v1 * v2
                    continue
                if n2 == zn and t2 == 0:
                    k = (n1, t1, pe)
                    out[k] = out.get(k, 0) + v1 * v2
                    continue
                v = v1 * v2 / 2
                nm = tuple(a - b for a, b in zip(n1, n2))
                npl = tuple(a + b for a, b in zip(n1, n2))
                if t1 == 0 and t2 == 0:
                    parts = ((nm, 0, v), (npl, 0, v))
                elif t1 == 1 and t2 == 1:
                    parts = ((nm, 0, v), (npl, 0, -v))
                elif t1 == 1 and t2 == 0:
                    parts = ((npl, 1, v), (nm, 1, v))
                else:
                    parts = ((npl, 1, v), (nm, 1, -v))
                for n, t, vv in parts:
                    cm = _canon_mode(n, t, vv)
                    if cm is None:
                        continue
                    k = (cm[0], t, pe)
                    out[k] = out.get(k, 0) + cm[2]
        return Coef(self.ctx, out).clean()

    def power_number(self, c):
        """c0**c for a constant Coef c0 (used for lead coefficients)."""
        if len(self.d) != 1:
            raise ValueError("power of a non-monomial coefficient")
        (k, v), = self.d.items()
        if k[0] != self.ctx.zero_n or k[1] != 0 or any(k[2]):
            raise ValueError("power of a non-constant coefficient")
        c = sym.nsimplify(c)
        if c.is_integer:
            return Coef.const(self.ctx, Fraction(v) ** int(c) if isinstance(v, Fraction) else v ** int(c))
        val = sym.Rational(v.numerator, v.denominator) ** c if isinstance(v, Fraction) else sym.Float(v) ** c
        return Coef.const(self.ctx, _num(sym.nsimplify(val)) if sym.nsimplify(val).is_Rational else float(val))

    def abs_number(self):
        (k, v), = self.d.items()
        return Coef.const(self.ctx, abs(v))

    # -- inspection --
    def to_sympy(self):
        ctx = self.ctx
        out = sym.S.Zero
        for (n, t, pe), v in self.d.items():
            val = sym.Rational(v.numerator, v.denominator) if isinstance(v, Fraction) else sym.Float(v)
            m = val
            for p, e in zip(ctx.params, pe):
                if e:
                    m *= p ** e
            phi = sum(int(ni) * a for ni, a in zip(n, ctx.angles))
            m *= sym.cos(phi) if t == 0 else sym.sin(phi)
            out += m
        return out

    def depends_on_angles(self):
        return any(k[0] != self.ctx.zero_n for k in self.d)

    def evaluate(self, R, P=None):
        """Vectorized value: R (n x nA) angle samples, P (n x nPar) parameter samples."""
        n = R.shape[0] if R is not None and R.ndim == 2 else (P.shape[0] if P is not None else 1)
        out = np.zeros(n)
        for (nn, t, pe), v in self.d.items():
            term = np.full(n, float(v))
            if any(pe):
                for k, e in enumerate(pe):
                    if e:
                        term = term * P[:, k] ** e
            if any(nn):
                phi = R @ np.array(nn, dtype=float)
                term = term * (np.cos(phi) if t == 0 else np.sin(phi))
            elif t == 1:
                continue
            out += term
        return out

    def mean_part(self):
        """non-oscillating part (n = 0)."""
        return Coef(self.ctx, {k: v for k, v in self.d.items() if k[0] == self.ctx.zero_n})

    def __repr__(self):
        return f"Coef({self.to_sympy()})"
