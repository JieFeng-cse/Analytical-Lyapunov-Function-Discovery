"""Asymptotic sums in the momenta (spec 2026-10-17 section 1.2).

An AsymSum is a finite sum of terms

    c(q, params) * prod_i sign(p_i)^{par_i} |p_i|^{e_i} * prod_A A^{f_A}

plus remainder terms O(prod_i |p_i|^{e_i} prod_A A^{f_A}) whose coefficient is an
unknown bounded function.  p^n (integer n) is sign^n |p|^n, so integer powers keep
their sign while |p|^e does not.  Coefficients are fourier.Coef objects (exact
Fourier modes in the bond angles times polynomials in the parameters).  Positive
atoms A = sum_m c_m * (even monomial), c_m > 0, cannot cancel; their tropical order
is max_m e_m.s (see tropical.py).

Exponents are Fractions (exact), keys are (exps, parity) tuples.
"""
import itertools
from fractions import Fraction

import numpy as np
import sympy as sym

from utils.fl_tmpl.fourier import Coef, CoefContext


class NotQuasiHomogeneous(ValueError):
    """Raised when an expression has a momentum dependence that is not a product of
    momentum powers, |p|^e and registered atoms (the source's quasi-homogeneity rule)."""


def _fr(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, sym.Rational):
        return Fraction(int(x.p), int(x.q))
    if isinstance(x, (float, sym.Float)):
        return Fraction(float(x)).limit_denominator(10**6)
    raise TypeError(f"exponent {x!r} is not rational")


class Atom:
    """Positive atom: sum of monomials in the momenta with positive coefficients
    and even integer exponents, e.g. p2**2 + p3**2."""

    def __init__(self, name, expr, momenta):
        self.name = name
        self.expr = sym.expand(expr)
        self.monomials = []          # list of (exps tuple over momenta, coeff float)
        poly = sym.Poly(self.expr, *momenta)
        for mon, c in poly.terms():
            if not all(int(e) % 2 == 0 for e in mon):
                raise ValueError(f"atom {name}: odd exponent in {mon}")
            if float(c) <= 0:
                raise ValueError(f"atom {name}: non-positive coefficient {c}")
            self.monomials.append((tuple(Fraction(int(e)) for e in mon), sym.Rational(c)))
        if not self.monomials:
            raise ValueError(f"atom {name} is empty")


class Space:
    """Momenta (scaling variables) + registered positive atoms + coefficient context."""

    def __init__(self, momenta, angles=(), params=()):
        self.momenta = tuple(momenta)
        self.nP = len(self.momenta)
        self.atoms = []
        self._index = {s: i for i, s in enumerate(self.momenta)}
        self.ctx = CoefContext(angles, params)
        self._frozen = False

    @property
    def nA(self):
        return len(self.atoms)

    @property
    def n(self):
        return self.nP + self.nA

    def add_atom(self, name, expr):
        for a in self.atoms:
            if a.name == name:
                return a
        if self._frozen:
            raise RuntimeError("atoms must be registered before any AsymSum is built")
        a = Atom(name, expr, self.momenta)
        self.atoms.append(a)
        return a

    def atom_index(self, base):
        b = sym.expand(base)
        for i, a in enumerate(self.atoms):
            if sym.expand(b - a.expr) == 0:
                return i
        return None

    def atom_names(self):
        return [a.name for a in self.atoms]

    def zero_key(self):
        self._frozen = True
        return (tuple(Fraction(0) for _ in range(self.n)), tuple(0 for _ in range(self.nP)))

    def monomial_key(self, exps=None, parity=None):
        self._frozen = True
        e = [Fraction(0)] * self.n
        par = [0] * self.nP
        for s, v in (exps or {}).items():
            if isinstance(s, str):
                idx = self.nP + self.atom_names().index(s)
            else:
                idx = self._index[s]
            e[idx] += _fr(v)
        for s, v in (parity or {}).items():
            par[self._index[s]] = int(v) % 2
        return (tuple(e), tuple(par))

    def coef(self, x):
        if isinstance(x, Coef):
            return x
        return Coef.const(self.ctx, x) if not isinstance(x, sym.Basic) or not x.free_symbols \
            else Coef.from_sympy(self.ctx, x)


def _add_keys(k1, k2):
    e = tuple(a + b for a, b in zip(k1[0], k2[0]))
    p = tuple((a + b) % 2 for a, b in zip(k1[1], k2[1]))
    return (e, p)


class AsymSum:
    def __init__(self, space, terms=None, rem=None):
        self.space = space
        self.terms = dict(terms or {})    # key -> Coef
        self.rem = set(rem or ())         # set of exps tuples

    # -- construction ---------------------------------------------------------------
    @classmethod
    def const(cls, space, c):
        c = space.coef(c)
        return cls(space, {space.zero_key(): c} if not c.is_zero() else {})

    @classmethod
    def monomial(cls, space, coeff=1, exps=None, parity=None):
        return cls(space, {space.monomial_key(exps, parity): space.coef(coeff)}).clean()

    @classmethod
    def from_sympy(cls, expr, space):
        """Decompose an expression; raises NotQuasiHomogeneous on unsupported terms."""
        space._frozen = True
        mom = set(space.momenta)
        e = sym.expand(sym.sympify(expr))
        groups = {}
        for term in sym.Add.make_args(e):
            if term == 0:
                continue
            coeff = sym.S.One
            exps = [Fraction(0)] * space.n
            par = [0] * space.nP
            for f in sym.Mul.make_args(term):
                if not (f.free_symbols & mom):
                    coeff *= f
                    continue
                base, ex = (f.base, f.exp) if isinstance(f, sym.Pow) else (f, sym.S.One)
                if ex.free_symbols:
                    raise NotQuasiHomogeneous(f"symbolic exponent in {f}")
                if base in mom:
                    i = space._index[base]
                    if not ex.is_integer:
                        raise NotQuasiHomogeneous(f"non-integer power of a signed momentum: {f}")
                    exps[i] += _fr(ex)
                    par[i] = (par[i] + int(ex)) % 2
                    continue
                if isinstance(base, sym.Abs) and base.args[0] in mom:
                    exps[space._index[base.args[0]]] += _fr(ex)
                    continue
                ai = space.atom_index(base)
                if ai is not None:
                    exps[space.nP + ai] += _fr(ex)
                    continue
                raise NotQuasiHomogeneous(f"unsupported momentum dependence: {f}")
            key = (tuple(exps), tuple(par))
            groups[key] = groups.get(key, sym.S.Zero) + coeff
        out = cls(space)
        for key, c in groups.items():
            co = Coef.from_sympy(space.ctx, c)
            if not co.is_zero():
                out.terms[key] = co
        return out

    def copy(self):
        return AsymSum(self.space, dict(self.terms), set(self.rem))

    # -- algebra ----------------------------------------------------------------------
    def clean(self):
        self.terms = {k: c for k, c in self.terms.items() if not c.is_zero()}
        return self

    def __add__(self, other):
        if not isinstance(other, AsymSum):
            other = AsymSum.const(self.space, other)
        out = dict(self.terms)
        for k, c in other.terms.items():
            out[k] = out[k] + c if k in out else c
        return AsymSum(self.space, out, self.rem | other.rem).clean()

    __radd__ = __add__

    def __neg__(self):
        return AsymSum(self.space, {k: -c for k, c in self.terms.items()}, set(self.rem))

    def __sub__(self, other):
        if not isinstance(other, AsymSum):
            other = AsymSum.const(self.space, other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def scale(self, c):
        """multiply by a momentum-free coefficient (number, sympy expr in params/angles, or Coef)."""
        c = self.space.coef(c)
        if c.is_zero():
            return AsymSum(self.space, {}, set(self.rem))
        return AsymSum(self.space, {k: v * c for k, v in self.terms.items()}, set(self.rem)).clean()

    def __mul__(self, other):
        if not isinstance(other, AsymSum):
            return self.scale(other)
        out = {}
        for k1, c1 in self.terms.items():
            for k2, c2 in other.terms.items():
                k = _add_keys(k1, k2)
                p = c1 * c2
                out[k] = out[k] + p if k in out else p
        res = AsymSum(self.space, out).clean()
        allk1 = [k[0] for k in self.terms] + list(self.rem)
        allk2 = [k[0] for k in other.terms] + list(other.rem)
        for r in self.rem:
            for e2 in allk2:
                res.rem.add(tuple(a + b for a, b in zip(r, e2)))
        for r in other.rem:
            for e1 in allk1:
                res.rem.add(tuple(a + b for a, b in zip(r, e1)))
        return res

    __rmul__ = __mul__

    def mul_pruned(self, other, pruner):
        """self * other keeping only products whose order can reach pruner.floor
        somewhere on the region (pairs below the floor are never multiplied)."""
        if pruner is None:
            return self * other
        fl = pruner.floor - 1e-9
        out = {}
        for k1, c1 in self.terms.items():
            for k2, c2 in other.terms.items():
                k = _add_keys(k1, k2)
                if pruner.max_order(k[0]) < fl:
                    continue
                p = c1 * c2
                out[k] = out[k] + p if k in out else p
        res = AsymSum(self.space, out).clean()
        allk1 = [k[0] for k in self.terms] + list(self.rem)
        allk2 = [k[0] for k in other.terms] + list(other.rem)
        for r in self.rem:
            for e2 in allk2:
                e = tuple(a + b for a, b in zip(r, e2))
                if pruner.max_order(e) >= fl:
                    res.rem.add(e)
        for r in other.rem:
            for e1 in allk1:
                e = tuple(a + b for a, b in zip(r, e1))
                if pruner.max_order(e) >= fl:
                    res.rem.add(e)
        return pruner(res)

    def mul_monomial(self, key, coeff=1):
        coeff = self.space.coef(coeff)
        out = AsymSum(self.space)
        for k, c in self.terms.items():
            out.terms[_add_keys(k, key)] = coeff * c
        out.rem = {tuple(a + b for a, b in zip(r, key[0])) for r in self.rem}
        return out.clean()

    def powi(self, n):
        assert n >= 0
        out = AsymSum.const(self.space, 1)
        base = self
        while n:
            if n & 1:
                out = out * base
            n >>= 1
            if n:
                base = base * base
        return out

    def expand_atoms(self):
        """Rewrite atoms raised to non-negative integer powers as polynomials in the
        momenta (so that e.g. theta/2 (p2^2+p3^2) cancels against theta/2 A23)."""
        sp = self.space
        out = AsymSum(sp, {}, set(self.rem))
        for (e, par), c in self.terms.items():
            fs = [e[sp.nP + j] for j in range(sp.nA)]
            if all(f == 0 for f in fs) or not all(f >= 0 and f.denominator == 1 for f in fs):
                out = out + AsymSum(sp, {(e, par): c})
                continue
            base = (tuple(list(e[:sp.nP]) + [Fraction(0)] * sp.nA), par)
            acc = AsymSum(sp, {base: c})
            for j, f in enumerate(fs):
                if f:
                    poly = AsymSum(sp)
                    for en, cc in sp.atoms[j].monomials:
                        k = (tuple(list(en) + [Fraction(0)] * sp.nA), tuple(0 for _ in range(sp.nP)))
                        poly = poly + AsymSum(sp, {k: sp.coef(cc)})
                    acc = acc * poly.powi(int(f))
            out = out + acc
        return out

    # -- series ----------------------------------------------------------------------
    def split_lead(self, key):
        """(lead coefficient, delta) with self = lead_coeff * monomial(key) * (1 + delta)."""
        c0 = self.terms.get(key)
        if c0 is None or c0.is_zero():
            raise ValueError("lead monomial not present")
        inv = (tuple(-a for a in key[0]), key[1])        # sign^par / |.|^e
        rest = self.copy()
        del rest.terms[key]
        delta = rest.mul_monomial(inv, c0.power_number(-1))
        return c0, delta

    def series_1p(self, coeffs, nmax):
        """sum_{n=0}^{nmax} coeffs[n] * self^n + remainder sum_m |m|^{nmax+1}
        (self plays the role of a small delta)."""
        out = AsymSum.const(self.space, coeffs[0])
        pw = AsymSum.const(self.space, 1)
        for n in range(1, nmax + 1):
            pw = pw * self
            out = out + pw.scale(coeffs[n])
        mons = [k[0] for k in self.terms] + list(self.rem)
        for e in mons:
            out.rem.add(tuple(a * (nmax + 1) for a in e))
        return out

    def power(self, c, lead_key, nmax=4, absval=False):
        """self^c (or |self|^c with absval=True) expanded around the lead monomial."""
        c = sym.nsimplify(c)
        c0, delta = self.split_lead(lead_key)
        if not absval and not c.is_integer:
            raise ValueError("non-integer power needs absval=True")
        if absval:
            lk = (tuple(_fr(c) * a for a in lead_key[0]), tuple(0 for _ in lead_key[1]))
            lc = c0.abs_number().power_number(c)
        else:
            ci = int(c)
            lk = (tuple(ci * a for a in lead_key[0]), tuple((ci * p) % 2 for p in lead_key[1]))
            lc = c0.power_number(c)
        binom = [sym.binomial(c, n) for n in range(nmax + 1)]
        ser = delta.series_1p(binom, nmax)
        return ser.mul_monomial(lk, lc)

    def log1p(self, nmax=4):
        """log(1 + self) for small self."""
        coeffs = [0] + [sym.Rational((-1) ** (n + 1), n) for n in range(1, nmax + 1)]
        return self.series_1p(coeffs, nmax)

    # -- inspection --------------------------------------------------------------------
    def keys(self):
        return list(self.terms)

    def monomial_sympy(self, key, Z_values=None):
        sp = self.space
        e, par = key
        m = sym.S.One
        for i, s in enumerate(sp.momenta):
            if e[i] != 0 or par[i]:
                if e[i].denominator == 1 and (int(e[i]) % 2) == par[i]:
                    m *= s ** int(e[i])
                else:
                    m *= sym.sign(s) ** par[i] * sym.Abs(s) ** sym.Rational(e[i].numerator, e[i].denominator)
        for j, a in enumerate(sp.atoms):
            f = e[sp.nP + j]
            if f != 0:
                m *= a.expr ** sym.Rational(f.numerator, f.denominator)
        return m

    def to_sympy(self, with_rem=False):
        out = sym.S.Zero
        for key, c in self.terms.items():
            out += c.to_sympy() * self.monomial_sympy(key)
        if with_rem and self.rem:
            out += sym.Symbol("O_rem")
        return out

    def __repr__(self):
        return f"AsymSum({len(self.terms)} terms, {len(self.rem)} remainders)"


class Pruner:
    """Region-aware pruning (spec 1.2): a term whose order is below `floor` at every
    direction of the region(s) cannot decide any block, so it is dropped; remainders
    below the floor are dropped as well, and remainders dominated componentwise by
    another remainder are merged (atom orders are >= 0, so componentwise >= is sound).

    vertex_sets: list of vertex arrays (n_v x nP) of the region polytopes.
    """

    def __init__(self, space, vertex_sets, floor):
        self.space = space
        self.V = np.vstack(vertex_sets)
        self.floor = float(floor)
        self._cache = {}

    def max_order(self, e):
        """Upper bound of the order of monomial e over the region.  Atoms with f > 0
        contribute f*max_mu(e_mu.s) (convex: max at vertices, enumerate mu); atoms
        with f < 0 contribute f*max_mu(e_mu.s) <= f*e_mu0.s for any fixed mu0, so
        the bound is min over mu0 choices of the vertex max."""
        hit = self._cache.get(e)
        if hit is not None:
            return hit
        nP = self.space.nP
        w = np.array([float(x) for x in e[:nP]])
        pos, neg = [], []
        for j, atom in enumerate(self.space.atoms):
            f = float(e[nP + j])
            if f:
                mono = [np.array([float(x) for x in en]) * f for en, _ in atom.monomials]
                (pos if f > 0 else neg).append(mono)
        best = np.inf
        for nc in (itertools.product(*neg) if neg else [()]):
            wn = w + (sum(nc) if nc else 0.0)
            worst = -np.inf
            for pc in (itertools.product(*pos) if pos else [()]):
                wp = wn + (sum(pc) if pc else 0.0)
                worst = max(worst, float((self.V @ wp).max()))
            best = min(best, worst)
        self._cache[e] = best
        return best

    def __call__(self, A):
        keep = {k: c for k, c in A.terms.items() if self.max_order(k[0]) >= self.floor - 1e-9}
        rem = [r for r in A.rem if self.max_order(r) >= self.floor - 1e-9]
        rem = sorted(set(rem), key=lambda r: [-float(x) for x in r])
        kept = []
        for r in rem:
            if not any(all(a >= b for a, b in zip(q, r)) for q in kept):
                kept.append(r)
        return AsymSum(A.space, keep, set(kept))
