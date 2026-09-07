"""Exact expansion of L ptilde on the fully open gate (rho = 1) in u = 1/p2.

Needed to state the fast-region Lean lemmas with explicit polynomial
coefficients: L pt = sum_k c_k(r, p1) u^k, and the noise factor (d pt/d p1)^2.
Run: python repro/lean_fast_series.py
"""
import sys
import sympy as sym
sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr

sv, drift, spec = make_rotor(2)
r1, p1, p2 = sv
u = sym.symbols('u')
t1 = sym.cos(r1); t2 = p1*sym.cos(r1)
t3 = p1**2*sym.cos(r1) + p1*sym.sin(r1) + sym.cos(r1)**2
t4 = (-p1**3*sym.cos(r1) - 3*p1**2*sym.sin(r1) - 3*p1*sym.cos(r1)**2 + p1*sym.cos(r1)
      - sym.Rational(1, 2)*sym.sin(r1)*sym.cos(r1) + 2*sym.sin(r1))
pt = p2 - t1/p2 - t2/p2**2 - t3/p2**3 + t4/p2**4
Lpt = generator_expr(pt, sv, drift, spec["noise"])
Lpt_u = sym.expand(sym.simplify(Lpt.subs(p2, 1/u)))
poly = sym.Poly(Lpt_u, u)
print("L pt as polynomial in u = 1/p2 (rho = 1):")
for (k,), c in sorted(poly.terms()):
    c = sym.simplify(sym.expand(c))
    cf = sym.trigsimp(c)
    print(f"  u^{k}: {cf}")
print("\nd pt / d p1 in u:")
dpt = sym.expand(sym.diff(pt, p1).subs(p2, 1/u))
for (k,), c in sorted(sym.Poly(dpt, u).terms()):
    print(f"  u^{k}: {sym.simplify(c)}")
print("\npt in u:  pt = 1/u - (", end="")
ptu = sym.expand((pt - p2).subs(p2, 1/u))
for (k,), c in sorted(sym.Poly(ptu, u).terms()):
    print(f" [u^{k}] {sym.simplify(c)} ", end="")
print(")")
# numeric sanity: |c4|, |c5| bounds on p1 in [-4,4]
import numpy as np
c4 = sym.lambdify((r1, p1), poly.coeff_monomial(u**4), "numpy")
c5 = sym.lambdify((r1, p1), poly.coeff_monomial(u**5), "numpy")
R, P = np.meshgrid(np.linspace(-np.pi, np.pi, 361), np.linspace(-4, 4, 161))
print(f"\nsup|c4| on r x [-4,4]: {np.abs(c4(R, P)).max():.4f}   sup|c5|: {np.abs(c5(R, P)).max():.4f}")
for pm in (1, 2, 3, 4, 6, 8):
    P2 = np.linspace(-pm, pm, 161); R2, PP = np.meshgrid(np.linspace(-np.pi, np.pi, 361), P2)
    print(f"  |p1|<={pm}: sup|c4|={np.abs(c4(R2, PP)).max():.4f}  sup|c5|={np.abs(c5(R2, PP)).max():.4f}")
