# Rotor2 Blind Skeleton Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third FL mode (`LYA_FL_BLIND=1`) in which the generator searches `pair(ρ, p̃)` with only the three-term skeleton of V known, scored by an overflow-free out-of-box drift judge calibrated on the reference certificate at θ=0.1; submit N=2 on both arms.

**Architecture:** A pure sympy/numpy judge (`utils/fl_blind.py`) assembles V from `pair(ρ, p̃)`, evaluates `ratio = −(LV/V)(2+log V)` in log domain on a fixed calibrated grid, and returns a penalty. Two new tokens (`step`, `pair`), one new prior (`RootConstraint`), one grammar rule (diffusion arm) plug the candidate space into the existing transformer/diffusion + GP-meld pipeline. Config/env glue in `rotor_common.py`; a calibration script writes the JSON the judge reads.

**Tech Stack:** Python 3.9 (conda env `alfd`), sympy, numpy, mpmath (tests), pytest, SLURM on Anvil.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-rotor2-blind-skeleton-search-design.md`.
- θ = 0.1 fixed. N = 2 only. Skeleton `V = 1 + e^{θH} + A·ρ·p̃²·e^{θp̃²/2}`.
- Function set (blind): `add, sub, mul, div, sin, cos, exp, n2, step, pair, "1", "2"`. No `const` token.
- `step(z)`: 0 for z ≤ 0, 1 for z ≥ 1, `1/(1+exp(1/z − 1/(1−z)))` inside.
- Judge must never evaluate an exponential of a candidate quantity (log domain throughout); no division by ρ.
- Priors: root; length min 5 max 60; soft_length loc 28 scale 8; inverse, trig, no_inputs on.
- A ∈ {0.1, 1, 10, 100, 1000}, best of five. Score `pen = v_out + 0.1·v_in`, ratio clipped at −1e3.
- C8 slot pipeline untouched; `LYA_FL_TEMPLATE` and `LYA_FL_BLIND` mutually exclusive.
- Run all tests with: `cd ~/ALFD-rotor && source ~/.conda_alfd.sh 2>/dev/null; PYTHONPATH=libs/sd3/dso python -m pytest tests/<file> -q` (activate env: `module load conda/2024.09; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate alfd`).
- Commit after each task with the session trailer.

---

### Task 1: `step` / `pair` tokens

**Files:**
- Modify: `libs/sd3/dso/dso/functions.py` (unprotected_ops list, protected_ops list)
- Test: `tests/test_blind_tokens.py`

**Interfaces:**
- Produces: tokens named `"step"` (arity 1) and `"pair"` (arity 2) in `function_map`; numpy `chi_np(z)`.
- `Program.sympy_expr` of a traversal `[pair, step, x3, x3]` parses to `pair(step(x3), x3)` with undefined sympy Functions named `pair`, `step`.

- [ ] **Step 1: failing test**

```python
# tests/test_blind_tokens.py
import numpy as np
import sympy as sym
from dso.functions import function_map, create_tokens
from dso.library import Library


def test_step_numpy_values():
    step = function_map["step"]
    z = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    out = step(z)
    assert out[0] == 0 and out[1] == 0 and out[5] == 1 and out[6] == 1
    assert abs(out[3] - 0.5) < 1e-12
    assert np.all(np.diff(out) >= 0)
    assert step.arity == 1 and function_map["pair"].arity == 2


def test_tokens_parse_to_sympy():
    from dso.program import Program, build_tree, convert_to_sympy
    from sympy.parsing.sympy_parser import parse_expr
    lib = Library(create_tokens(3, ["add", "step", "pair", "1", "2"], protected=False))
    names = [t.name for t in lib.tokens]
    trav = [lib.tokens[names.index(n)] for n in ["pair", "step", "x3", "add", "x3", "1.0"]]
    tree = convert_to_sympy(build_tree(list(trav)))
    expr = parse_expr(repr(tree))
    assert expr.func.__name__ == "pair"
    assert expr.args[0].func.__name__ == "step"
    assert expr.args[1] == sym.Symbol("x3") + 1
```

- [ ] **Step 2: run** `pytest tests/test_blind_tokens.py -q` → FAIL `KeyError: 'step'`.

- [ ] **Step 3: implement** in `functions.py` after the existing helper defs (before `unprotected_ops`):

```python
def chi_np(z):
    """Smooth step: 0 for z<=0, 1 for z>=1, C^inf bump splice inside."""
    z = np.asarray(z, dtype=np.float64)
    out = np.where(z >= 1.0, 1.0, 0.0)
    inner = (z > 0.0) & (z < 1.0)
    zi = np.clip(z, 1e-12, 1 - 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        val = 1.0 / (1.0 + np.exp(1.0 / zi - 1.0 / (1.0 - zi)))
    return np.where(inner, val, out)


def pair_np(x1, x2):
    return x1
```

and in `unprotected_ops` add
```python
    Token(chi_np, "step", arity=1, complexity=3),
    Token(pair_np, "pair", arity=2, complexity=1),
```
`protected_ops` needs nothing (both are already total functions).

- [ ] **Step 4: run** → PASS.  **Step 5: commit** `feat(blind): step/pair tokens`.

---

### Task 2: `RootConstraint` prior

**Files:**
- Modify: `libs/sd3/dso/dso/prior.py` (`make_prior` dict + new class after `NoInputsConstraint`)
- Test: `tests/test_blind_priors.py`

**Interfaces:**
- Produces: `RootConstraint(library, token="pair")`; config key `"root": {"token": "pair", "on": True}`.
- `initial_prior()` = −inf everywhere except the root token; `__call__` = −inf on the root token for every row; `is_violated(actions, parent, sibling)` True iff `actions[0] != root` or root appears later.

- [ ] **Step 1: failing test**

```python
# tests/test_blind_priors.py
import numpy as np
from dso.functions import create_tokens
from dso.library import Library
from dso.prior import make_prior


def _lib():
    return Library(create_tokens(3, ["add", "mul", "sin", "step", "pair", "1"], protected=False))


def test_root_prior_masks():
    lib = _lib()
    prior = make_prior(lib, {"root": {"token": "pair", "on": True}})
    pair = int(np.where(lib.names == "pair")[0][0]) if hasattr(lib, "names") else \
        [t.name for t in lib.tokens].index("pair")
    init = prior.initial_prior()
    assert init[pair] == 0 and np.all(np.isneginf(np.delete(init, pair)))
    actions = np.array([[pair, 0]])
    later = prior(actions, np.zeros((1,)), np.zeros((1,)), np.ones((1,)))
    assert np.isneginf(later[0, pair]) and np.isfinite(later[0]).sum() == lib.L - 1


def test_root_prior_is_violated():
    lib = _lib()
    prior = make_prior(lib, {"root": {"token": "pair", "on": True}})
    names = [t.name for t in lib.tokens]
    pair, add, x1 = names.index("pair"), names.index("add"), names.index("x1")
    assert not prior.is_violated(np.array([pair, x1, x1]), None, None)
    assert prior.is_violated(np.array([add, x1, x1]), None, None)
    assert prior.is_violated(np.array([pair, pair, x1, x1, x1]), None, None)
```

- [ ] **Step 2: run** → FAIL (custom prior import error for "root").

- [ ] **Step 3: implement**

```python
class RootConstraint(Constraint):
    """Forces the first token to be `token` and forbids it anywhere else
    (blind skeleton search: candidate = pair(rho, ptilde))."""

    def __init__(self, library, token="pair"):
        Prior.__init__(self, library)
        self.token = library.tokenize(token)[0]

    def initial_prior(self):
        prior = np.full((self.L,), -np.inf, dtype=np.float32)
        prior[self.token] = 0.0
        return prior

    def __call__(self, actions, parent, sibling, dangling):
        mask = np.ones(actions.shape[0], dtype=bool)
        return self.make_constraint(mask, [self.token])

    def is_violated(self, actions, parent, sibling):
        actions = np.asarray(actions).reshape(-1)
        return bool(actions.size == 0 or actions[0] != self.token
                    or np.any(actions[1:] == self.token))

    def describe(self):
        return "{}: first token is {}, never again.".format(
            self.__class__.__name__, self.library.names[self.token])
```
and `"root": RootConstraint,` in `prior_dict`. (Check `Library.tokenize` accepts a name string; if it needs a list use `library.tokenize([token])[0]`.)

- [ ] **Step 4: run** → PASS. **Step 5: commit** `feat(blind): RootConstraint prior`.

---

### Task 3: diffusion grammar root rule

**Files:**
- Modify: `models/diffusion/grammar.py` (`__init__` gets `root_token=None`; `valid_token_mask`)
- Modify: `models/diffusion/controller.py:60` (pass root token when the prior contains a `RootConstraint`)
- Test: `tests/test_blind_priors.py` (append)

- [ ] **Step 1: failing test**

```python
def test_diffusion_grammar_root_rule():
    from models.diffusion.grammar import Grammar
    lib = _lib()
    names = [t.name for t in lib.tokens]
    pair, add = names.index("pair"), names.index("add")
    g = Grammar(lib, root_token=pair)
    m0 = g.valid_token_mask([])
    assert m0[pair] and not m0[add]
    m1 = g.valid_token_mask([pair])
    assert m1[add] and not m1[pair]
    rep = g.repair([add, names.index("x1"), names.index("x1")])
    assert rep[0] == pair and pair not in rep[1:]
```

- [ ] **Step 2: run** → FAIL (unexpected kwarg).

- [ ] **Step 3: implement** in `Grammar.__init__`: `self.root_token = None if root_token is None else int(root_token)`. In `valid_token_mask`, after computing `const_count` and before the loop:

```python
        n_real = len(self._real(tokens))
        if self.root_token is not None:
            if n_real == 0:
                mask[self.root_token] = True
                return mask
```
and inside the loop: `if self.root_token is not None and tid == self.root_token: continue`.
`repair` must start from the root: at the top of `repair`, `if self.root_token is not None: out = [self.root_token]; toks = self._real(tokens); toks = toks[1:] if toks and toks[0] == self.root_token else toks` and iterate over `toks`. `fallback` returns `[root, t, t]` when a root is set.

In `controller.py`: 
```python
        root_token = None
        for p in prior.priors:
            if p.__class__.__name__ == "RootConstraint":
                root_token = int(p.token)
        self.grammar = Grammar(library, max_const=max_const, max_len=self.max_length, root_token=root_token)
```

- [ ] **Step 4: run** all of `tests/test_blind_priors.py` and `tests/test_diffusion_grammar.py` → PASS. **Step 5: commit** `feat(blind): diffusion grammar root rule`.

---

### Task 4: judge core — `utils/fl_blind.py`

**Files:**
- Create: `utils/fl_blind.py`
- Test: `tests/test_fl_blind.py`

**Interfaces:**
- `Chi, ChiD1, ChiD2` sympy Function classes; `CHI_MODULES` lambdify module dict.
- `reference_pair(theta_unused=None, R_rho=20)` → sympy `pair(rho_doc, ptilde_doc)` for tests/calibration (order 4). `reference_pair(order=k)` truncates.
- `BlindJudge(spec, state_vars, drift)`; `spec["blind"]` dict keys `theta, M_max, R2_max, c_t, A_grid`; `spec["noise"], spec["H"]`.
  - `judge.fields(expr) -> dict | None` (A-independent numeric arrays on the grid; None if invalid).
  - `judge.ratio(fields, A) -> np.ndarray` and `judge.logV(fields, A)`.
  - `judge.penalty(expr) -> (pen: float, A: float, ok: bool)`.
  - `judge.grid` (n,3) array; `judge.outside` boolean mask of K_max complement.
- `fl_blind_penalty(expr, spec, state_vars, drift)` module-level cached entry.
- `split_pair(expr) -> (rho, pt) | None`.

- [ ] **Step 1: failing tests**

```python
# tests/test_fl_blind.py
import numpy as np
import sympy as sym
import pytest
from example_config.rotor_common import make_rotor
from utils.fl_blind import (Chi, ChiD1, ChiD2, CHI_MODULES, BlindJudge,
                            reference_pair, split_pair)

SPEC_BLIND = {"theta": 0.1, "M_max": 900.0, "R2_max": 600.0, "c_t": 1e-3,
              "A_grid": [0.1, 1, 10, 100, 1000], "grid_coarse": True}


def _judge():
    sv, drift, spec = make_rotor(2)
    spec = dict(spec); spec["blind"] = SPEC_BLIND
    return BlindJudge(spec, sv, drift), sv


def test_chi_derivatives_match_fd():
    z = sym.Symbol("z")
    f = sym.lambdify(z, Chi(z), modules=[CHI_MODULES, "numpy"])
    d1 = sym.lambdify(z, sym.diff(Chi(z), z), modules=[CHI_MODULES, "numpy"])
    d2 = sym.lambdify(z, sym.diff(Chi(z), z, 2), modules=[CHI_MODULES, "numpy"])
    zs = np.linspace(-1, 2, 301); h = 1e-5
    fd1 = (f(zs + h) - f(zs - h)) / (2 * h)
    fd2 = (f(zs + h) - 2 * f(zs) + f(zs - h)) / h**2
    assert np.allclose(d1(zs), fd1, atol=1e-5)
    assert np.allclose(d2(zs), fd2, atol=1e-3)
    assert d1(-0.5) == 0 and d1(1.5) == 0 and d2(-0.5) == 0 and f(2.0) == 1.0


def test_split_pair_and_invalid_shapes():
    judge, sv = _judge()
    x1, x2, x3 = sv
    pair = sym.Function("pair"); step = sym.Function("step")
    assert split_pair(pair(step(x3), x3)) is not None
    assert split_pair(step(x3)) is None
    assert judge.penalty(step(x3))[2] is False                      # no pair root
    assert judge.penalty(pair(-step(x3**2 - 4), x3))[2] is False    # rho < 0
    assert judge.penalty(pair(step(x3**2 - 4), x2))[2] is False     # ptilde without p2
    pen, A, ok = judge.penalty(pair(step(x3**2 - 4), 5 * x3))       # overflow probe
    assert ok and np.isfinite(pen)


def test_reference_matches_mpmath():
    import mpmath as mp
    judge, sv = _judge()
    expr = reference_pair()
    fields = judge.fields(expr)
    assert fields is not None
    pts = np.array([[0.3, 5.0, 10.0], [0.3, 5.0, 900.0], [0.3, 2.0, 100.0],
                    [1.0, 0.5, 40.0], [-2.0, 3.0, 300.0], [0.0, 1.0, 0.0]])
    r_num = judge.ratio_at(expr, pts, A=20.0)
    r_ref = judge.ratio_mpmath(expr, pts, A=20.0, dps=40)
    ok = np.isfinite(r_ref)
    assert np.allclose(r_num[ok], r_ref[ok], rtol=1e-6, atol=1e-12)


def test_reference_passes_and_truncation_fails():
    judge, sv = _judge()
    pen4, A4, ok4 = judge.penalty(reference_pair(order=4))
    pen1, A1, ok1 = judge.penalty(reference_pair(order=1))
    assert ok4 and ok1
    assert pen4 == 0.0
    assert pen1 > pen4
```

- [ ] **Step 2: run** → FAIL (`No module named utils.fl_blind`).

- [ ] **Step 3: implement `utils/fl_blind.py`**

```python
"""Blind skeleton search judge (spec 2026-09-07, CONCLUSIONS C45).

Candidate = pair(rho, ptilde). Assembled V = 1 + e^{theta H} + A rho ptilde^2 e^{theta ptilde^2/2}.
Everything is evaluated in log domain:
    logV  = logsumexp(0, theta H, log A + log rho + log P + f),  P = ptilde^2, f = theta P/2
    LV/V  = e^{theta H - logV} Q1 + A e^{f - logV} G2
    Q1 = theta LH + theta^2 sum_b k_b (d_b H)^2
    G2 = rho (LP + P (Lf + sum k_b (d_b f)^2) + 2 sum k_b d_b P d_b f) + P L rho + 2 sum k_b d_b rho (d_b P + P d_b f)
    ratio = -(LV/V) (2 + logV);  drift condition holds at x  <=>  ratio(x) >= c.
No exponential of a candidate quantity is ever evaluated and nothing divides by rho.
"""
import numpy as np
import sympy as sym

from utils.fl_reward import generator_expr


# ---- smooth step as a sympy Function with symbolic derivatives -----------------
class ChiD2(sym.Function):
    nargs = 1


class ChiD1(sym.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return ChiD2(self.args[0])


class Chi(sym.Function):
    nargs = 1

    def fdiff(self, argindex=1):
        return ChiD1(self.args[0])


def _chi_inner_fns():
    z = sym.Symbol("z")
    inner = 1 / (1 + sym.exp(1 / z - 1 / (1 - z)))
    f0 = sym.lambdify(z, inner, "numpy")
    f1 = sym.lambdify(z, sym.diff(inner, z), "numpy")
    f2 = sym.lambdify(z, sym.diff(inner, z, 2), "numpy")
    return f0, f1, f2


_F0, _F1, _F2 = _chi_inner_fns()


def _chi_eval(z, fn, outside_hi):
    z = np.asarray(z, dtype=np.float64)
    inner = (z > 0.0) & (z < 1.0)
    zi = np.where(inner, z, 0.5)
    with np.errstate(all="ignore"):
        val = np.asarray(fn(zi), dtype=np.float64)
    return np.where(inner, val, np.where(z >= 1.0, outside_hi, 0.0))


def chi_np(z):  return _chi_eval(z, _F0, 1.0)
def chi1_np(z): return _chi_eval(z, _F1, 0.0)
def chi2_np(z): return _chi_eval(z, _F2, 0.0)


CHI_MODULES = {"Chi": chi_np, "ChiD1": chi1_np, "ChiD2": chi2_np}


def _to_chi(expr):
    """parse_expr gives an undefined Function('step'); swap in Chi."""
    return expr.replace(lambda e: isinstance(e, sym.Function) and e.func.__name__ == "step",
                        lambda e: Chi(e.args[0]))


def split_pair(expr):
    if not isinstance(expr, sym.Basic) or not isinstance(expr, sym.Function):
        return None
    if expr.func.__name__ != "pair" or len(expr.args) != 2:
        return None
    return _to_chi(expr.args[0]), _to_chi(expr.args[1])


# ---- reference certificate (document eq. 13-17), for calibration and tests ------
def reference_pair(order=4, R_rho=20):
    x1, x2, x3 = sym.symbols("x1 x2 x3")
    r1, p1, p2 = x1, x2, x3
    S = p1**4 + R_rho
    # document chi ramps on z in [1,4]; step ramps on [0,1]: z' = (z-1)/3
    rho = sym.Function("step")((p2**2 / S**2 - 1) / 3)
    t = [sym.cos(r1), p1 * sym.cos(r1),
         p1**2 * sym.cos(r1) + p1 * sym.sin(r1) + sym.cos(r1)**2,
         -(-p1**3 * sym.cos(r1) - 3 * p1**2 * sym.sin(r1) - 3 * p1 * sym.cos(r1)**2
           + p1 * sym.cos(r1) - sym.Rational(1, 2) * sym.sin(r1) * sym.cos(r1) + 2 * sym.sin(r1))]
    pt = p2 - sum(t[j] * rho / p2**(j + 1) for j in range(order))
    return sym.Function("pair")(rho, pt)


# ---- the judge ----------------------------------------------------------------
def _logsumexp3(a, b, c):
    m = np.maximum(np.maximum(a, b), c)
    with np.errstate(all="ignore"):
        return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m))


class BlindJudge:
    def __init__(self, spec, state_vars, drift):
        b = spec["blind"]
        self.sv = list(state_vars)
        self.drift = list(drift)
        self.noise = dict(spec["noise"])
        self.theta = float(b["theta"])
        self.M_max = float(b["M_max"]); self.R2_max = float(b["R2_max"])
        self.c_t = float(b["c_t"])
        self.A_grid = [float(a) for a in b.get("A_grid", [0.1, 1, 10, 100, 1000])]
        H = spec["H"]
        th = self.theta
        Q1 = th * generator_expr(H, self.sv, self.drift, self.noise) \
            + th**2 * sum(k * sym.diff(H, self.sv[i])**2 for i, k in self.noise.items())
        self._thH = self._lam(th * H); self._Q1 = self._lam(Q1)
        self.grid = self._make_grid(coarse=bool(b.get("grid_coarse", False)))
        self.outside = (self.grid[:, 1]**2 > self.M_max) | (np.abs(self.grid[:, 2]) > self.R2_max)
        self._cache = {}

    def _lam(self, e):
        return sym.lambdify(self.sv, e, modules=[CHI_MODULES, "numpy"])

    def _make_grid(self, coarse=False):
        n_r, n_p1, n_p2 = (13, 12, 20) if coarse else (25, 24, 40)
        P1max = 2.0 * np.sqrt(self.M_max); P2max = 4.0 * self.R2_max
        r = np.linspace(-np.pi, np.pi, n_r)
        a = np.geomspace(0.25, P1max, n_p1); p1 = np.concatenate([-a[::-1], [0.0], a])
        bnd = np.geomspace(0.25, P2max, n_p2); p2 = np.concatenate([-bnd[::-1], [0.0], bnd])
        R, P1, P2 = np.meshgrid(r, p1, p2, indexing="ij")
        return np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)

    # -- symbolic assembly (A-free) --
    def _pieces(self, rho, pt):
        sv, noise, th = self.sv, self.noise, self.theta
        d1 = [sym.diff(pt, s) for s in sv]
        r1 = [sym.diff(rho, s) for s in sv]
        Lpt = sum(f * d for f, d in zip(self.drift, d1)) + sum(k * sym.diff(pt, sv[b], 2) for b, k in noise.items())
        Lrho = sum(f * d for f, d in zip(self.drift, r1)) + sum(k * sym.diff(rho, sv[b], 2) for b, k in noise.items())
        P = pt**2; f = th * P / 2
        dP = {b: 2 * pt * d1[b] for b in noise}
        df = {b: th * pt * d1[b] for b in noise}
        LP = 2 * pt * Lpt + 2 * sum(k * d1[b]**2 for b, k in noise.items())
        Lf = th * LP / 2
        sq = sum(k * df[b]**2 for b, k in noise.items())
        G2 = rho * (LP + P * (Lf + sq) + 2 * sum(k * dP[b] * df[b] for b, k in noise.items())) \
            + P * Lrho + 2 * sum(k * r1[b] * (dP[b] + P * df[b]) for b, k in noise.items())
        return rho, P, f, G2

    def fields(self, expr):
        sp = split_pair(expr)
        if sp is None:
            return None
        rho, pt = sp
        if not (rho.free_symbols | pt.free_symbols) <= set(self.sv):
            return None
        if self.sv[2] not in pt.free_symbols:          # ptilde must depend on p2 (N=2)
            return None
        rho_e, P_e, f_e, G2_e = self._pieces(rho, pt)
        X = self.grid
        args = [X[:, i] for i in range(X.shape[1])]
        n = X.shape[0]
        ev = lambda e: np.broadcast_to(np.asarray(self._lam(e)(*args), dtype=np.float64), (n,))
        with np.errstate(all="ignore"):
            rho_v, P_v, f_v, G2_v = ev(rho_e), ev(P_e), ev(f_e), ev(G2_e)
            thH, Q1 = ev(self._thH), ev(self._Q1)
        if np.any(~np.isfinite(rho_v)) or np.any(rho_v < 0):
            return None
        active = rho_v > 0
        if np.any(~np.isfinite(P_v[active])) or np.any(~np.isfinite(f_v[active])) or np.any(~np.isfinite(G2_v[active])):
            return None
        with np.errstate(divide="ignore"):
            log_rhoP = np.where(active & (P_v > 0), np.log(rho_v) + np.log(P_v), -np.inf)
        return {"thH": thH, "Q1": Q1, "log_rhoP": log_rhoP, "f": f_v, "G2": np.where(active, G2_v, 0.0)}

    def logV(self, fl, A):
        return _logsumexp3(np.zeros_like(fl["thH"]), fl["thH"], np.log(A) + fl["log_rhoP"] + fl["f"])

    def ratio(self, fl, A):
        lv = self.logV(fl, A)
        with np.errstate(all="ignore"):
            LVoverV = np.exp(fl["thH"] - lv) * fl["Q1"] + A * np.exp(fl["f"] - lv) * fl["G2"]
        return -LVoverV * (2.0 + lv)

    def score(self, fl, A):
        r = self.ratio(fl, A)
        if not np.all(np.isfinite(r)):
            return None
        rc = np.clip(r, -1e3, np.inf)
        v_out = float(np.mean(np.maximum(self.c_t - rc[self.outside], 0.0)) / self.c_t)
        v_in = float(np.mean(r[~self.outside] < self.c_t))
        return v_out + 0.1 * v_in

    def penalty(self, expr):
        fl = self.fields(expr)
        if fl is None:
            return None, None, False
        best = (None, None)
        for A in self.A_grid:
            s = self.score(fl, A)
            if s is not None and (best[0] is None or s < best[0]):
                best = (s, A)
        if best[0] is None:
            return None, None, False
        return float(best[0]), float(best[1]), True

    # -- point-wise helpers (tests / recon) --
    def ratio_at(self, expr, pts, A):
        saved = self.grid; self.grid = np.asarray(pts, dtype=np.float64)
        try:
            fl = self.fields(expr)
            return self.ratio(fl, A)
        finally:
            self.grid = saved

    def ratio_mpmath(self, expr, pts, A, dps=40):
        import mpmath as mp
        mp.mp.dps = dps
        rho, pt = split_pair(expr)
        th = sym.Rational(self.theta).limit_denominator(10**6)
        H = sum(s**2 for s in self.sv[1:]) / 2 + (1 - sym.cos(self.sv[0]))
        V = 1 + sym.exp(th * H) + A * rho * pt**2 * sym.exp(th * pt**2 / 2)
        LV = generator_expr(V, self.sv, self.drift, self.noise)
        out = []
        for p in pts:
            sub = {s: sym.Float(float(v), dps) for s, v in zip(self.sv, p)}
            # replace Chi and derivatives by their closed forms for evaluation
            def num(e):
                e = e.subs(sub)
                e = e.replace(lambda q: isinstance(q, (Chi, ChiD1, ChiD2)),
                              lambda q: sym.Float(float({Chi: chi_np, ChiD1: chi1_np, ChiD2: chi2_np}[q.func](float(q.args[0]))), dps))
                return sym.N(e, dps)
            v = num(V); lv = num(LV)
            out.append(float(-lv * (2 + sym.log(v)) / v) if v > 0 else np.nan)
        return np.array(out)


_JUDGES = {}


def get_judge(spec, state_vars, drift):
    key = (len(state_vars), tuple(sorted((k, float(v)) for k, v in spec["blind"].items() if not isinstance(v, list))))
    if key not in _JUDGES:
        _JUDGES[key] = BlindJudge(spec, state_vars, drift)
    return _JUDGES[key]


def fl_blind_penalty(expr, spec, state_vars, drift):
    return get_judge(spec, state_vars, drift).penalty(expr)
```

Notes for the implementer: `ratio_mpmath` substitutes numbers first and evaluates Chi pieces with the float implementation (the mpmath check is about the *assembly/log-domain algebra*, not about chi's own precision; the chi values are identical in both paths). If `Chi(...).subs` keeps Chi unevaluated the `.replace` handles it. The N=2 rule "ptilde depends on p2" uses `self.sv[2]`; guard with `len(self.sv) == 3`.

- [ ] **Step 4: run** `pytest tests/test_fl_blind.py -q` → PASS (the mpmath test may need `rtol` loosened to 1e-5 if float64 cancellation at |p₂|=900 shows; do not loosen beyond 1e-4 — that would hide algebra bugs).

- [ ] **Step 5: commit** `feat(blind): log-domain out-of-box judge (utils/fl_blind.py)`.

---

### Task 5: calibration script + JSON

**Files:**
- Create: `repro/blind_calibrate_rotor2.py`
- Create (output): `example_config/blind_rotor2.json`
- Test: `tests/test_fl_blind.py::test_calibration_json_loads`

**Interfaces:**
- Produces JSON keys: `theta, M_max, R2_max, c_t, M_ref, R2_ref, c_ref, A_ref, A_grid, calibrated_on`.

- [ ] **Step 1: failing test**

```python
def test_calibration_json_loads():
    import json, os
    p = "example_config/blind_rotor2.json"
    assert os.path.exists(p)
    d = json.load(open(p))
    for k in ["theta", "M_max", "R2_max", "c_t", "M_ref", "R2_ref", "c_ref", "A_ref", "A_grid"]:
        assert k in d
    assert d["theta"] == 0.1 and d["M_max"] >= d["M_ref"] and d["c_t"] < d["c_ref"]
```

- [ ] **Step 2: write the script**

```python
"""Calibrate the blind judge on the document's rotor2 certificate at theta=0.1
(spec 2026-09-07 section 4). Writes example_config/blind_rotor2.json and prints
a C44-style region table. Run from the repo root in the alfd env."""
import json, sys, datetime
import numpy as np
sys.path.insert(0, ".")
from example_config.rotor_common import make_rotor
from utils.fl_blind import BlindJudge, reference_pair

THETA, A_REF = 0.1, 20.0
sv, drift, spec = make_rotor(2)
spec = dict(spec)
# wide provisional box so the grid reaches |p1|<=100, |p2|<=1e5
spec["blind"] = {"theta": THETA, "M_max": 2500.0, "R2_max": 25000.0, "c_t": 1e-3, "A_grid": [A_REF]}
judge = BlindJudge(spec, sv, drift)
expr = reference_pair(order=4)
fl = judge.fields(expr)
assert fl is not None, "reference rejected by the judge"
r = judge.ratio(fl, A_REF)
X = judge.grid
p1, p2 = X[:, 1], X[:, 2]

def smallest_box(c):
    """smallest (M, R2) with ratio>=c everywhere outside {p1^2<=M, |p2|<=R2}."""
    bad = r < c
    if not bad.any():
        return 0.0, 0.0
    # violations must lie inside the box: M >= max p1^2 over bad points with |p2|<=R2,
    # scan R2 over the p2 grid levels and take the box with the smallest area proxy.
    best = None
    for R2 in np.unique(np.abs(p2)):
        inside_p2 = np.abs(p2) <= R2
        if np.any(bad & ~inside_p2):
            # need every bad point with |p2|>R2 to be covered -> impossible, skip
            continue
        M = float(np.max(p1[bad]**2)) if bad.any() else 0.0
        cand = (M, float(R2))
        if best is None or cand[0] * cand[1] < best[0] * best[1]:
            best = cand
    return best

c_ref, box = None, None
for k in range(0, 12):
    c = (THETA**2 / 4) * 2.0**(-k)
    b = smallest_box(c)
    if b is not None:
        c_ref, box = c, b
        break
assert box is not None, "no compact exemption found for the reference"
M_ref, R2_ref = box
out = {"theta": THETA, "A_ref": A_REF, "A_grid": [0.1, 1, 10, 100, 1000],
       "M_ref": M_ref, "R2_ref": R2_ref, "c_ref": c_ref,
       "M_max": 1.25**2 * M_ref, "R2_max": 1.25 * R2_ref, "c_t": 0.5 * c_ref,
       "calibrated_on": datetime.date.today().isoformat()}
json.dump(out, open("example_config/blind_rotor2.json", "w"), indent=2)
for name, m in [("slow rho=0", fl["log_rhoP"] == -np.inf), ("gate>0", fl["log_rhoP"] > -np.inf)]:
    print(f"{name:12s} n={m.sum():7d} min ratio={np.nanmin(r[m]):+.4e}")
print("calibration:", out)
```

- [ ] **Step 3: run** `python repro/blind_calibrate_rotor2.py` (≈1–2 min). Record the printed numbers for CONCLUSIONS. Sanity: √M_ref ≈ 20–30, R2_ref ≈ 300–1000, c_ref ≈ 1e-3 (θ-scan expectations). If the reference is rejected or no box exists, stop and debug the judge before going on.

- [ ] **Step 4: run** the test → PASS. **Step 5: commit** `feat(blind): rotor2 calibration script + JSON`.

---

### Task 6: config / env glue (`rotor_common.py`) + reward branch

**Files:**
- Modify: `example_config/rotor_common.py` (FL_BLIND flag, spec["blind"], `apply_blind_overrides`, call it from `apply_slot_overrides` so configs need no change)
- Modify: `libs/sd3/dso/dso/task/regression/regression.py` (`_fl_reward`, `evaluate` blind branches)
- Test: `tests/test_rotor_configs.py` (blind case), `tests/test_fl_regression_glue.py` (append)

**Interfaces:**
- Env `LYA_FL_BLIND=1` → `generator_spec()["blind"]` = JSON dict; `config_factory()["task"]["function_set"] == ["add","sub","mul","div","sin","cos","exp","n2","step","pair","1","2"]`; priors root/length/soft_length set.
- `_fl_reward(p)` in blind mode: `pen, A, ok = fl_blind_penalty(p.sympy_expr[0], spec, sv, drift)`; reward `1/(1+pen)` or `invalid_reward`. `evaluate(p)` returns `nmse_test = pen`, `success = pen == 0.0`.

- [ ] **Step 1: failing tests**

```python
# in tests/test_rotor_configs.py
def test_blind_mode_config(monkeypatch):
    import importlib, sys
    monkeypatch.setenv("LYA_FL_BLIND", "1")
    monkeypatch.delenv("LYA_FL_TEMPLATE", raising=False)
    for m in ["example_config.rotor_common", "example_config.config_rotor2"]:
        sys.modules.pop(m, None)
    m = importlib.import_module("example_config.config_rotor2")
    conf = m.config_factory()
    assert conf["task"]["function_set"] == ["add", "sub", "mul", "div", "sin", "cos",
                                            "exp", "n2", "step", "pair", "1", "2"]
    assert conf["prior"]["root"] == {"token": "pair", "on": True}
    assert conf["prior"]["length"]["max_"] == 60 and conf["prior"]["const"]["on"] is False
    spec = m.generator_spec()
    assert spec["blind"]["theta"] == 0.1 and "scaffold" not in spec
```

```python
# in tests/test_fl_regression_glue.py (append; follow the file's existing fixture style)
def test_blind_reward_branch(monkeypatch):
    ...  # build a RegressionTask the way the file's existing FL glue test does,
         # with LYA_FL_BLIND=1, sample Program for traversal [pair, step, x3, x3],
         # assert reward is finite and evaluate()["nmse_test"] equals 1/reward - 1.
```

- [ ] **Step 2: run** → FAIL.

- [ ] **Step 3: implement** in `rotor_common.py`:

```python
FL_BLIND = os.environ.get("LYA_FL_BLIND", "") == "1"
assert not (FL_BLIND and FL_TEMPLATE), "LYA_FL_BLIND and LYA_FL_TEMPLATE are mutually exclusive"
BLIND_FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "cos", "exp", "n2", "step", "pair", "1", "2"]
BLIND_JSON = os.path.join(os.path.dirname(__file__), "blind_rotor2.json")


def apply_blind_overrides(cfg):
    if not FL_BLIND:
        return cfg
    cfg["task"]["function_set"] = list(BLIND_FUNCTION_SET)
    cfg["prior"]["root"] = {"token": "pair", "on": True}
    cfg["prior"]["length"] = {"min_": 5, "max_": 60, "on": True}
    cfg["prior"]["soft_length"] = {"loc": 28, "scale": 8, "on": True}
    cfg["prior"]["const"] = {"on": False}
    cfg["prior"].pop("repeat", None)
    return cfg
```
`apply_slot_overrides` ends with `return apply_blind_overrides(cfg)` when not template. In `make_rotor`, after the template block:
```python
    if FL_BLIND:
        import json as _json
        assert N == 2, "blind mode calibrated for N=2 only"
        spec["blind"] = _json.load(open(BLIND_JSON))
```
`regression.py::_fl_reward` — insert before `template = ...`:
```python
        if spec.get("blind") is not None:
            from utils.fl_blind import fl_blind_penalty
            try:
                pen, A, ok = fl_blind_penalty(p.sympy_expr[0], spec, state_vars, drift)
            except Exception:
                return self.invalid_reward
            if not ok:
                return self.invalid_reward
            p.blind_A = A
            return 1.0 / (1.0 + pen)
```
and in `evaluate` the mirror: `nmse_test = pen if ok else 1e6`, `success = (ok and pen == 0.0)`.

- [ ] **Step 4: run** both test files → PASS. **Step 5: commit** `feat(blind): config/env glue + reward branch`.

---

### Task 7: train loop — pristine grid, final check, logging

**Files:**
- Modify: `utils/train.py` (harvest skip, X_train pristine, final gate, top-candidate log line)
- Modify: `utils/fl_blind.py` (add `fl_blind_final_check(expr, spec, sv, drift, A, seed=0) -> (ok, worst_point)`)
- Test: `tests/test_fl_blind.py::test_final_check_reference`

- [ ] **Step 1: failing test**

```python
def test_final_check_reference():
    from utils.fl_blind import fl_blind_final_check
    sv, drift, spec = make_rotor(2)
    spec = dict(spec); spec["blind"] = dict(SPEC_BLIND, n_random=20000, dense=False)
    ok, worst = fl_blind_final_check(reference_pair(order=4), spec, sv, drift, A=20.0)
    assert ok, worst
    ok1, _ = fl_blind_final_check(reference_pair(order=1), spec, sv, drift, A=20.0)
    assert not ok1
```

- [ ] **Step 2: implement**

```python
def fl_blind_final_check(expr, spec, state_vars, drift, A, seed=0):
    """(i) 2x denser product grid, (ii) n_random uniform points in the judge box
    minus K_max. Pass iff ratio >= c_t everywhere. Returns (ok, worst_point)."""
    b = dict(spec["blind"])
    dense_spec = dict(spec); dense_spec["blind"] = dict(b, grid_coarse=False)
    j = BlindJudge(dense_spec, state_vars, drift)
    if b.get("dense", True):
        j.grid = j._make_grid(coarse=False)
        # densify: 2x points per axis
        n_r, n_p1, n_p2 = 49, 48, 80
        P1max = 2.0 * np.sqrt(j.M_max); P2max = 4.0 * j.R2_max
        r = np.linspace(-np.pi, np.pi, n_r)
        a = np.geomspace(0.25, P1max, n_p1); p1 = np.concatenate([-a[::-1], [0.0], a])
        bb = np.geomspace(0.25, P2max, n_p2); p2 = np.concatenate([-bb[::-1], [0.0], bb])
        R, P1, P2 = np.meshgrid(r, p1, p2, indexing="ij")
        j.grid = np.stack([R.ravel(), P1.ravel(), P2.ravel()], 1)
    rng = np.random.default_rng(seed)
    n_random = int(b.get("n_random", 10_000_000))
    P1max = 2.0 * np.sqrt(j.M_max); P2max = 4.0 * j.R2_max
    U = rng.uniform([-np.pi, -P1max, -P2max], [np.pi, P1max, P2max], size=(n_random, 3))
    U = U[(U[:, 1]**2 > j.M_max) | (np.abs(U[:, 2]) > j.R2_max)]
    j.grid = np.vstack([j.grid, U])
    j.outside = (j.grid[:, 1]**2 > j.M_max) | (np.abs(j.grid[:, 2]) > j.R2_max)
    fl = j.fields(expr)
    if fl is None:
        return False, None
    r = j.ratio(fl, A)
    bad = j.outside & ~(r >= j.c_t)
    if not bad.any():
        return True, None
    i = int(np.argmin(np.where(bad, r, np.inf)))
    return False, j.grid[i]
```
Evaluate random points in chunks of 1e6 inside `fields` if memory is a problem (the lambdified expressions are vectorised; 10⁷ × a few arrays of float64 ≈ 1–2 GB, acceptable on the 64 GB job).

`train.py` edits (all guarded by `FL_SPEC.get("blind")`):
- line ≈676: `if FL_SPEC.get("stage") or FL_SPEC.get("blind"): order = np.array([], dtype=int)`.
- line ≈716: same condition for the pristine `X_train`.
- final gate (≈744): `if FL_SPEC.get("blind"):` loop over successes calling `fl_blind_final_check(programs[i].sympy_expr[0], FL_SPEC, sv, dr, A=getattr(programs[i], "blind_A", 1.0), seed=i)`, log `[FL blind final] i=.. ok=.. worst=..`, `overall_check.append(ok)`. Otherwise the existing FL branch.
- after the per-epoch best-program print, add `if FL_SPEC.get("blind"): log_and_print(f"[FL blind] epoch={epoch} best_A={getattr(best_p,'blind_A',None)} expr={best_p.sympy_expr[0]}")` where `best_p` is the program the loop already identifies as the epoch's best (find the variable name at the print site).

- [ ] **Step 3: run** `tests/test_fl_blind.py` → PASS; run `tests/test_fl_reward.py tests/test_fl_falsifier.py` to confirm nothing else moved. **Step 4: commit** `feat(blind): train-loop integration + final check`.

---

### Task 8: smoke run, job scripts, submission, ledger

**Files:**
- Create: `repro/run_one_rotor_blind.sbatch`, `repro/submit_rotor_blind.sh`, `repro/setup_runs_rotor_blind.sh`
- Modify: `docs/CONCLUSIONS.md` (C46), memory file.

- [ ] **Step 1: local smoke** (login node, CPU, 2 epochs) from the repo root:

```bash
LYA_FL_BLIND=1 LYA_BENCHMARK=fn_rotor_2 LYA_NSAMPLES=1000 LYA_SEED_START=1 LYA_SEED_RUNS=1 \
LYA_GP_CORES=4 PYTHONPATH=libs/sd3/dso timeout 1500 python -u Lyapunov_test_dso.py 2>&1 | tail -40
```
Expected: `[FL blind] epoch=0 ...` lines, no traceback, rewards in (0,1]. Then the same with `LYA_CONTROLLER=DiffusionController LYA_SAMPLER_ORDER=blocksemiar`. Fix whatever breaks (typical: `Program.execute` on `pair`, `HardCodedConstant` naming `1.0` in `parse_expr`, grammar `repair` root handling).

- [ ] **Step 2: job scripts** — copy `run_one_rotor_slot.sbatch` to `run_one_rotor_blind.sbatch` replacing the template exports with `export LYA_FL_BLIND=1` (drop `LYA_FL_TEMPLATE`/`LYA_FL_STAGE`), keep `LYA_NSAMPLES=20000`, 16 h, header comment updated. `setup_runs_rotor_blind.sh` = `setup_runs_rotor.sh` restricted to `rotor2`, target `RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_blind`. `submit_rotor_blind.sh`:

```bash
#!/bin/bash
set -euo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_blind
for arm in transformer diffusion; do
  sbatch --job-name="lyablind_rotor2_${arm}" --time=16:00:00 \
         --export=ALL,SYS=rotor2,BM=fn_rotor_2,ARM="$arm" \
         --chdir="$RUNS_BASE/rotor2/$arm" \
         /home/x-cchen47/ALFD-rotor/repro/run_one_rotor_blind.sbatch
done
```

- [ ] **Step 3: submit**: `bash repro/setup_runs_rotor_blind.sh && bash repro/submit_rotor_blind.sh`; record job ids with `squeue -u $USER -n lyablind_rotor2_transformer,lyablind_rotor2_diffusion`.

- [ ] **Step 4: ledger** — append C46 to `docs/CONCLUSIONS.md`: calibration numbers (M_ref, R2_ref, c_ref, K_max, c_t), θ=0.1, encoding, vocabulary, job ids, what counts as success, expected-wall caveat. Update memory `rotor-fl-benchmark-spec.md` with the blind track status.

- [ ] **Step 5: commit** `feat(blind): rotor2 blind search jobs submitted (C46)`.
