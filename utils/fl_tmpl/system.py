"""Rotor chain in bond coordinates for the template evaluator (spec 2026-10-17).

Same symbols and dynamics as example_config/rotor_common.make_rotor (state
x1..x_{N-1} = bond angles r_i = q_{i+1}-q_i, x_N..x_{2N-1} = momenta p_1..p_N),
re-derived here so that the evaluator does not depend on environment flags.
"""
from fractions import Fraction

import sympy as sym

from utils.fl_reward import generator_expr
from utils.fl_tmpl.asym import AsymSum, Space

THETA = sym.Symbol("theta", positive=True)


class RotorSystem:
    def __init__(self, N, params=()):
        """params: extra parameter symbols allowed in coefficients (THETA is always one)."""
        self.N = N
        dim = 2 * N - 1
        self.xs = list(sym.symbols(f"x1:{dim + 1}"))
        self.r = self.xs[: N - 1]                 # bond angles r_1..r_{N-1}
        self.p = self.xs[N - 1:]                  # momenta p_1..p_N
        r, p = self.r, self.p
        drift_r = [p[i + 1] - p[i] for i in range(N - 1)]
        if N == 2:
            drift_p = [sym.sin(r[0]) - p[0], -sym.sin(r[0])]
            self.noise = {N - 1: sym.Integer(1)}          # state index of p1
        else:
            drift_p = [sym.sin(r[0]) - p[0]]
            for i in range(1, N - 1):
                drift_p.append(sym.sin(r[i]) - sym.sin(r[i - 1]))
            drift_p.append(-sym.sin(r[N - 2]) - p[N - 1])
            self.noise = {N - 1: sym.Integer(1), 2 * N - 2: sym.Integer(2)}
        self.drift = drift_r + drift_p
        self.H = sum(pi**2 for pi in p) / 2 + sum(1 - sym.cos(ri) for ri in r)
        self.h = 1 + self.H
        # 1-based indices of noisy momenta and their kappa = gamma_b T_b
        self.baths = [i - (N - 1) + 1 for i in self.noise]
        self.kappa = {i - (N - 1) + 1: k for i, k in self.noise.items()}
        # gamma_b (friction) is 1 for every bath; T_b = kappa_b / gamma_b
        self.gamma = {b: sym.Integer(1) for b in self.baths}
        self.T = {b: self.kappa[b] / self.gamma[b] for b in self.baths}
        self.interior = [j for j in range(1, N + 1) if j not in self.baths]
        self.params = (THETA,) + tuple(params)
        self.space = Space(self.p, angles=self.r, params=self.params)
        # fixed atom set, registered before any AsymSum exists so that all keys
        # have the same length: h's leading part and interior-group energies
        self.space.add_atom("Ah", sum(pi**2 for pi in p) / 2)
        inner = self.interior
        for i in range(len(inner)):
            for k in range(i + 1, len(inner)):
                a, b = inner[i], inner[k]
                self.space.add_atom(f"A{a}{b}", self.P(a)**2 + self.P(b)**2)
        if len(inner) >= 3:
            self.space.add_atom("A" + "".join(map(str, inner)), sum(self.P(a)**2 for a in inner))

    # -- symbols ----------------------------------------------------------------------
    def P(self, j):
        """momentum p_j (1-based)."""
        return self.p[j - 1]

    def R(self, i):
        """bond angle r_i (1-based, i = 1..N-1)."""
        return self.r[i - 1]

    def state_index(self, j):
        return self.N - 1 + (j - 1)

    # -- generator ----------------------------------------------------------------------
    def L(self, expr):
        return generator_expr(expr, self.xs, self.drift, self.noise)

    def d(self, expr, b):
        """derivative in the bath momentum p_b."""
        return sym.diff(expr, self.P(b))

    def asym(self, expr):
        return AsymSum.from_sympy(expr, self.space)

    # -- fast-angle frequencies ------------------------------------------------------------
    def fast_frequency(self, n, j):
        """q_j-frequency of the Fourier mode n (tuple over bond angles r_1..r_{N-1}):
        dr_{j-1}/dt contains +p_j and dr_j/dt contains -p_j."""
        f = 0
        if j - 2 >= 0:
            f += n[j - 2]
        if j - 1 <= self.N - 2:
            f -= n[j - 1]
        return f

    def slow_rate(self, n, fast):
        """d(n.r)/dt with the fast momenta removed (exact remaining polynomial)."""
        rates = self.drift[: self.N - 1]
        Om = sum(ni * rt for ni, rt in zip(n, rates))
        return sym.expand(Om.subs({self.P(j): 0 for j in fast}))

    def leading_rate(self, n):
        return sym.expand(sum(ni * rt for ni, rt in zip(n, self.drift[: self.N - 1])))

    def W(self, i):
        """zero-mean interaction potential W_i(r_i) = -cos r_i (paper convention <W>=0)."""
        return -sym.cos(self.R(i))


def frac(x):
    return Fraction(x)
