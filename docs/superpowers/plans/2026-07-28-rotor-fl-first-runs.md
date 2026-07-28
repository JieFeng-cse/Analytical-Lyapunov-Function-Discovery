# Rotor-chain FL First Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run transformer vs best-diffusion (loop_20) on the four rotor-chain stochastic Foster–Lyapunov benchmarks (8 SLURM runs) with a reward that evaluates the full stochastic generator.

**Architecture:** Extend the existing DSO pipeline (approach A of the spec): per-system `generator_spec()` in config, new `foster_lyapunov` metric branch in `regression.py`, FL-mode verification/harvesting in `utils/train.py`, all pure math in a new `utils/fl_reward.py`. Work happens on branch `rotor-benchmarks` forked from loop/20 commit `5700a56`, in a **separate git worktree** so the main checkout (used by the 6-hourly self-improvement loop) is untouched.

**Tech Stack:** sympy (symbolic LV incl. 2nd derivatives), numpy/lambdify (vectorized eval), scipy SHGO (falsifier), SLURM on Anvil, conda env `alfd`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md` (approved 2026-07-28).
- Branch: `rotor-benchmarks` from commit `5700a56`; worktree at `/home/x-cchen47/ALFD-rotor`. NEVER switch the main checkout off `diffusion-generator-design`.
- Reward must include second-derivative noise terms: κ = {p1: 1.0} for N=2; {p1: 1.0, pN: 2.0} for N≥3. First-order-only is wrong.
- FL params: c=0.05, R=10.0; α=1 for rotor2/3/4, α=3 for rotor5.
- Boxes: r ∈ [−3.14, 3.14], p ∈ [−6, 6]; 15000 training samples per variable spec.
- State layout: x1..x_{N−1} = bond angles r, x_N..x_{2N−1} = momenta p (dims 3/5/7/9).
- function_set `["add","sub","mul","div","sin","cos","exp","n2"]`; length prior min 10 / max 35, soft_length loc 25.
- No V(0)=0 shift, no PD-at-origin logic in the FL path. Keep the all-variables-used validity rule.
- Falsifier deviation from spec (record in spec, Task 6): torch-PGD is replaced by SHGO + massive random sampling in FL mode (numerically safer for log/exp objectives); in-loop harvesting uses random-sampling top violators.
- Run tests with: `module load conda; conda activate alfd; python -m pytest <file> -v` from the worktree root.
- Commit after every task on `rotor-benchmarks`; git author must be chongwei730 <chen8596@umn.edu> (verify once with `git config user.name`).

---

### Task 1: Worktree + branch setup

**Files:** none created in repo (worktree dir `/home/x-cchen47/ALFD-rotor`).

**Interfaces:**
- Produces: worktree at `/home/x-cchen47/ALFD-rotor` on new branch `rotor-benchmarks` (fork of `5700a56`). All later tasks run inside it.

- [ ] **Step 1: Create worktree**

```bash
cd /home/x-cchen47/Analytical-Lyapunov-Function-Discovery
git worktree add /home/x-cchen47/ALFD-rotor -b rotor-benchmarks 5700a56
cd /home/x-cchen47/ALFD-rotor && git log -1 --oneline
```
Expected: `5700a56 loop loop_20_entbonus_blocksemiarb4: block_size default 2 -> 4 ...`

- [ ] **Step 2: Verify loop_20 ingredients present**

```bash
grep -n "diffusion_sample_blocksemiar" models/diffusion/sampler.py | head -2
grep -n "entropy_term" models/diffusion/controller.py | head -2
```
Expected: both found (blocksemiar sampler + entbonus wiring).

- [ ] **Step 3: Copy spec+plan into the worktree branch and commit**

```bash
cp /home/x-cchen47/Analytical-Lyapunov-Function-Discovery/docs/superpowers/specs/2026-07-27-rotor-chain-foster-lyapunov-benchmarks.md docs/superpowers/specs/
cp /home/x-cchen47/Analytical-Lyapunov-Function-Discovery/docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md docs/superpowers/specs/
cp /home/x-cchen47/Analytical-Lyapunov-Function-Discovery/docs/superpowers/plans/2026-07-28-rotor-fl-first-runs.md docs/superpowers/plans/
git add docs && git commit -m "docs: import rotor FL benchmark spec + plan onto rotor-benchmarks branch"
```

### Task 2: Rotor dynamics + generator spec (`rotor_common.py`)

**Files:**
- Create: `example_config/rotor_common.py`
- Test: `tests/test_rotor_dynamics.py`

**Interfaces:**
- Produces: `make_rotor(N) -> (state_variables: list[sym.Symbol], drift: list[sym.Expr], spec: dict)` with `spec = {"noise": {int: float}, "alpha": int, "c": float, "R": float, "H": sym.Expr}`. Noise keys are INDICES into state_variables.

- [ ] **Step 1: Write the failing test**

`tests/test_rotor_dynamics.py`:
```python
import sympy as sym
import pytest
from example_config.rotor_common import make_rotor


@pytest.mark.parametrize("N,dim", [(2, 3), (3, 5), (4, 7), (5, 9)])
def test_dimensions(N, dim):
    sv, drift, spec = make_rotor(N)
    assert len(sv) == dim and len(drift) == dim
    assert [str(s) for s in sv] == [f"x{i}" for i in range(1, dim + 1)]


@pytest.mark.parametrize("N", [3, 4, 5])
def test_generator_identity_two_bath(N):
    # L_N H_N = (1 - p1^2) + (2 - pN^2)   [benchmark eq (6)]
    sv, drift, spec = make_rotor(N)
    LH = sum(sym.diff(spec["H"], s) * f for s, f in zip(sv, drift))
    for idx, kappa in spec["noise"].items():
        LH += kappa * sym.diff(spec["H"], sv[idx], 2)
    p1, pN = sv[N - 1], sv[2 * N - 2]
    assert sym.simplify(LH - ((1 - p1**2) + (2 - pN**2))) == 0


def test_generator_identity_one_bath_N2():
    # L_2 H_2 = 1 - p1^2   [benchmark eq (9)]
    sv, drift, spec = make_rotor(2)
    LH = sum(sym.diff(spec["H"], s) * f for s, f in zip(sv, drift))
    for idx, kappa in spec["noise"].items():
        LH += kappa * sym.diff(spec["H"], sv[idx], 2)
    p1 = sv[1]
    assert sym.simplify(LH - (1 - p1**2)) == 0
    assert list(spec["noise"].keys()) == [1]  # only p1 noisy


def test_spec_params():
    for N, alpha in [(2, 1), (3, 1), (4, 1), (5, 3)]:
        _, _, spec = make_rotor(N)
        assert spec["alpha"] == alpha and spec["c"] == 0.05 and spec["R"] == 10.0


def test_H_nonnegative_and_zero_at_rest():
    sv, drift, spec = make_rotor(3)
    at_rest = {s: 0 for s in sv}
    assert spec["H"].subs(at_rest) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rotor_dynamics.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: example_config.rotor_common`

- [ ] **Step 3: Write implementation**

`example_config/rotor_common.py`:
```python
"""Rotor-chain benchmark systems in bond coordinates (spec 2026-07-28).

State: x1..x_{N-1} = bond angles r_i = q_{i+1}-q_i, x_N..x_{2N-1} = momenta p_i.
Drift: dr_i = p_{i+1}-p_i;
  N=2 (one bath):  dp1 = sin r1 - p1,  dp2 = -sin r1
  N>=3 (two baths): dp1 = sin r1 - p1, dp_i = sin r_i - sin r_{i-1},
                    dpN = -sin r_{N-1} - pN
Generator second-derivative coefficients: gamma_b*T_b = 1 at p1, 2 at pN (N>=3).
"""
import sympy as sym


def make_rotor(N):
    dim = 2 * N - 1
    xs = sym.symbols(f"x1:{dim + 1}")            # x1..x_{2N-1}
    r = list(xs[: N - 1])
    p = list(xs[N - 1:])

    drift_r = [p[i + 1] - p[i] for i in range(N - 1)]
    if N == 2:
        drift_p = [sym.sin(r[0]) - p[0], -sym.sin(r[0])]
        noise = {N - 1: 1.0}                     # index of p1
    else:
        drift_p = [sym.sin(r[0]) - p[0]]
        for i in range(1, N - 1):
            drift_p.append(sym.sin(r[i]) - sym.sin(r[i - 1]))
        drift_p.append(-sym.sin(r[N - 2]) - p[N - 1])
        noise = {N - 1: 1.0, 2 * N - 2: 2.0}     # indices of p1, pN

    H = sum(pi**2 for pi in p) / 2 + sum(1 - sym.cos(ri) for ri in r)
    spec = {"noise": noise, "alpha": 3 if N == 5 else 1,
            "c": 0.05, "R": 10.0, "H": H}
    return list(xs), drift_r + drift_p, spec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rotor_dynamics.py -v` — Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add example_config/rotor_common.py tests/test_rotor_dynamics.py
git commit -m "feat: rotor-chain dynamics + generator spec in bond coordinates"
```

### Task 3: FL reward math (`utils/fl_reward.py`)

**Files:**
- Create: `utils/fl_reward.py`
- Test: `tests/test_fl_reward.py`

**Interfaces:**
- Consumes: `make_rotor` (Task 2).
- Produces:
  - `generator_expr(V: sym.Expr, state_vars, drift, noise: dict) -> sym.Expr` — LV incl. 2nd-derivative terms.
  - `phi_vals(v: np.ndarray, alpha, c) -> np.ndarray` — c·max(v,1)/(α+1+log max(v,1))^α.
  - `fl_penalty(v_vals, lv_vals, h_vals, spec) -> float` — mean(relu(LV+φ)·1_{H>R}) + 2.5·mean(relu(1−V)); inputs clipped to ±1e30, NaN→penalty 1e6.
  - `fl_violation(v_vals, lv_vals, h_vals, spec) -> float` — max(0-margin drift violation, positivity violation) as a scalar = mean form used by evaluate(); same formula as fl_penalty but WITHOUT the 2.5 weight (plain sum of means).

- [ ] **Step 1: Write the failing test**

`tests/test_fl_reward.py`:
```python
import numpy as np
import sympy as sym
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr, phi_vals, fl_penalty, fl_violation


def test_generator_expr_matches_hand_identity():
    sv, drift, spec = make_rotor(3)
    LH = generator_expr(spec["H"], sv, drift, spec["noise"])
    p1, p3 = sv[2], sv[4]
    assert sym.simplify(LH - ((1 - p1**2) + (2 - p3**2))) == 0


def test_phi_vals_formula_and_clamp():
    v = np.array([0.5, 1.0, np.e])
    out = phi_vals(v, alpha=1, c=0.05)
    # v<1 clamps to 1: phi(1) = 0.05*1/(2+0)^1 = 0.025
    assert np.isclose(out[0], 0.025) and np.isclose(out[1], 0.025)
    assert np.isclose(out[2], 0.05 * np.e / 3.0)


def test_fl_penalty_masks_inside_KR():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([2.0, 2.0])
    lv = np.array([5.0, 5.0])          # violating drift
    h = np.array([1.0, 20.0])          # first point inside K_R -> masked out
    pen_all_in = fl_penalty(v, lv, np.array([1.0, 1.0]), spec)
    pen_half = fl_penalty(v, lv, h, spec)
    assert pen_all_in == 0.0           # only positivity term, v>=1 -> 0
    assert pen_half > 0.0


def test_fl_penalty_positivity_term():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([0.0])                # V below 1 everywhere
    lv = np.array([-100.0])
    h = np.array([0.0])
    assert np.isclose(fl_penalty(v, lv, h, spec), 2.5 * 1.0)


def test_fl_penalty_nan_guard():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    assert fl_penalty(np.array([np.nan]), np.array([0.0]),
                      np.array([20.0]), spec) >= 1e6


def test_fl_violation_unweighted():
    spec = {"alpha": 1, "c": 0.05, "R": 10.0}
    v = np.array([0.0]); lv = np.array([-100.0]); h = np.array([0.0])
    assert np.isclose(fl_violation(v, lv, h, spec), 1.0)  # no 2.5 weight
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fl_reward.py -v`
Expected: ERROR `ModuleNotFoundError: utils.fl_reward`

- [ ] **Step 3: Write implementation**

`utils/fl_reward.py`:
```python
"""Foster-Lyapunov reward math (spec 2026-07-28). Pure sympy/numpy, no framework deps."""
import numpy as np
import sympy as sym


def generator_expr(V, state_vars, drift, noise):
    """Full stochastic generator LV = grad(V).f + sum_b kappa_b d2V/dp_b^2."""
    LV = sum(sym.diff(V, s) * f for s, f in zip(state_vars, drift))
    for idx, kappa in noise.items():
        LV = LV + kappa * sym.diff(V, state_vars[idx], 2)
    return LV


def phi_vals(v, alpha, c):
    v = np.maximum(np.asarray(v, dtype=np.float64), 1.0)
    return c * v / (alpha + 1.0 + np.log(v)) ** alpha


def _terms(v_vals, lv_vals, h_vals, spec):
    v = np.clip(np.asarray(v_vals, dtype=np.float64), -1e30, 1e30)
    lv = np.clip(np.asarray(lv_vals, dtype=np.float64), -1e30, 1e30)
    h = np.asarray(h_vals, dtype=np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(lv))):
        return None
    mask = (h > spec["R"]).astype(np.float64)
    drift_viol = np.mean(np.maximum(lv + phi_vals(v, spec["alpha"], spec["c"]), 0.0) * mask)
    pos_viol = np.mean(np.maximum(1.0 - v, 0.0))
    return drift_viol, pos_viol


def fl_penalty(v_vals, lv_vals, h_vals, spec):
    t = _terms(v_vals, lv_vals, h_vals, spec)
    if t is None:
        return 1e6
    return float(t[0] + 2.5 * t[1])


def fl_violation(v_vals, lv_vals, h_vals, spec):
    t = _terms(v_vals, lv_vals, h_vals, spec)
    if t is None:
        return 1e6
    return float(t[0] + t[1])
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fl_reward.py tests/test_rotor_dynamics.py -v` — Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add utils/fl_reward.py tests/test_fl_reward.py
git commit -m "feat: FL generator/penalty math (full stochastic generator, drift condition)"
```

### Task 4: Four rotor configs + benchmark CSV + systems.tsv

**Files:**
- Create: `example_config/config_rotor2.py`, `config_rotor3.py`, `config_rotor4.py`, `config_rotor5.py`
- Modify: `libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv` (append 4 rows)
- Modify: `repro/systems.tsv` (append 4 rows)
- Test: `tests/test_rotor_configs.py`

**Interfaces:**
- Consumes: `make_rotor` (Task 2).
- Produces: importable configs each exporting `dynamics()`, `generator_spec()`, `config_factory()`, `train_config_factory()`; CSV rows named `fn_rotor_2/3/5` (unique substrings for `LYA_BENCHMARK` matching); config task dict sets `"metric": "foster_lyapunov"`.

- [ ] **Step 1: Write the failing test**

`tests/test_rotor_configs.py`:
```python
import importlib
import json
import pandas as pd
import pytest

ROWS = {"rotor2": ("fn_rotor_2", 3), "rotor3": ("fn_rotor_3", 5),
        "rotor4": ("fn_rotor_4", 7), "rotor5": ("fn_rotor_5", 9)}
CSV = "libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv"


@pytest.mark.parametrize("sys,expected", ROWS.items())
def test_config_module(sys, expected):
    m = importlib.import_module(f"example_config.config_{sys}")
    sv, drift = m.dynamics()
    assert len(sv) == expected[1] == len(drift)
    spec = m.generator_spec()
    assert set(spec) >= {"noise", "alpha", "c", "R", "H"}
    conf = m.config_factory()
    assert conf["task"]["metric"] == "foster_lyapunov"
    assert conf["task"]["function_set"] == ["add", "sub", "mul", "div",
                                            "sin", "cos", "exp", "n2"]
    assert conf["prior"]["length"]["min_"] == 10
    assert conf["prior"]["length"]["max_"] == 35
    assert m.train_config_factory() is not None


@pytest.mark.parametrize("sys,expected", ROWS.items())
def test_csv_row(sys, expected):
    name, dim = expected
    df = pd.read_csv(CSV, index_col=0)
    assert name in df.index
    row = df.loc[name]
    assert int(row["variables"]) == dim
    train_spec = json.loads(row["train_spec"].replace("'", '"'))
    n_r = dim - (dim + 1) // 2
    for i in range(1, dim + 1):
        lo, hi, n = train_spec[f"x{i}"]["U"]
        assert n == 15000
        if i <= n_r:
            assert (lo, hi) == (-3.14, 3.14)
        else:
            assert (lo, hi) == (-6, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rotor_configs.py -v` — Expected: FAIL (module + CSV rows missing)

- [ ] **Step 3: Write the four configs**

`example_config/config_rotor2.py` (rotor3/4/5 are identical except `_N = 3/4/5`):
```python
import sympy as sym  # noqa: F401  (kept for parity with other configs)
from example_config.rotor_common import make_rotor

_N = 2
_SV, _DRIFT, _SPEC = make_rotor(_N)


def dynamics():
    return _SV, _DRIFT


def generator_spec():
    return _SPEC
```
Then open `example_config/config_lossless6d.py`, copy its `config_factory()` and `train_config_factory()` **verbatim** into each rotor config, and apply exactly these overrides inside `config_factory`:
- `"function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "n2"]`
- add key `"metric": "foster_lyapunov"` to the `"task"` dict
- `"length": {"min_": 10, "max_": 35, "on": True}`
- `"soft_length": {"loc": 25, "scale": 5, "on": True}`

- [ ] **Step 4: Append CSV rows**

Append to `libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv` (no trailing blank line; r-variables first, then p-variables). Row pattern, shown in full for N=2 and N=3 — build N=4 (7 vars: x1-x3 angles, x4-x7 momenta) and N=5 (9 vars: x1-x4 angles, x5-x9 momenta) the same way with a dummy expression summing 0*xi over all vars:
```
fn_rotor_2,3,"0*x1+0*x2+0*x3","{""x1"":{""U"":[-3.14,3.14,15000]}, ""x2"":{""U"":[-6,6,15000]}, ""x3"":{""U"":[-6,6,15000]}}",None,Koza
fn_rotor_3,5,"0*x1+0*x2+0*x3+0*x4+0*x5","{""x1"":{""U"":[-3.14,3.14,15000]}, ""x2"":{""U"":[-3.14,3.14,15000]}, ""x3"":{""U"":[-6,6,15000]}, ""x4"":{""U"":[-6,6,15000]}, ""x5"":{""U"":[-6,6,15000]}}",None,Koza
```

- [ ] **Step 5: Append systems.tsv rows** (tab-separated, matching existing format in `repro/systems.tsv`):
```
rotor2	example_config/config_rotor2.py	fn_rotor_2	3	16:00:00
rotor3	example_config/config_rotor3.py	fn_rotor_3	5	16:00:00
rotor4	example_config/config_rotor4.py	fn_rotor_4	7	16:00:00
rotor5	example_config/config_rotor5.py	fn_rotor_5	9	16:00:00
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_rotor_configs.py -v` — Expected: 8 PASS

- [ ] **Step 7: Commit**

```bash
git add example_config/config_rotor*.py libs/sd3/dso/dso/task/regression/benchmarks_bkup_1.csv repro/systems.tsv tests/test_rotor_configs.py
git commit -m "feat: rotor2-5 benchmark configs, CSV rows, systems.tsv entries"
```

### Task 5: `foster_lyapunov` metric branch in regression.py + program.py

**Files:**
- Modify: `libs/sd3/dso/dso/program.py` (add cached property near `sympy_expr_lie`, ~line 586)
- Modify: `libs/sd3/dso/dso/task/regression/regression.py` (reward_function ~191-241, evaluate ~244-294, metric registry ~329-336, `__init__` metric-name capture)
- Test: `tests/test_fl_regression_glue.py`

**Interfaces:**
- Consumes: `generator_expr`, `fl_penalty`, `fl_violation` (Task 3); `generator_spec` from `config` (Task 4 configs, present only for rotor systems).
- Produces: `Program.sympy_expr_generator` (cached sympy LV); reward path returning `1/(1+fl_penalty)`; `evaluate()` returning `nmse_test = fl_violation` and the usual `success = nmse < threshold`.

- [ ] **Step 1: Write the failing test** (pure-glue test via a stub program; does not construct RegressionTask)

`tests/test_fl_regression_glue.py`:
```python
import numpy as np
import sympy as sym
from example_config.rotor_common import make_rotor
from utils.fl_reward import generator_expr, fl_penalty


def test_reward_of_reference_H_is_positive_and_below_one():
    """Sanity: V=1+H is NOT a certificate but must produce a finite (0,1] reward."""
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]
    LV = generator_expr(V, sv, drift, spec["noise"])
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-3.14, 3.14, 4096),
                         rng.uniform(-6, 6, 4096), rng.uniform(-6, 6, 4096)])
    args = [X[:, i] for i in range(3)]
    v_vals = sym.lambdify(sv, V, "numpy")(*args)
    lv_vals = sym.lambdify(sv, LV, "numpy")(*args)
    h_vals = sym.lambdify(sv, spec["H"], "numpy")(*args)
    r = 1.0 / (1.0 + fl_penalty(v_vals, lv_vals, h_vals, spec))
    assert 0.0 < r <= 1.0
    # V=1+H violates the drift condition on fast-p2 states (LH=1-p1^2 can be ~1>0
    # outside K_R while phi>0), so the reward must NOT be perfect:
    assert r < 1.0
```

- [ ] **Step 2: Run test to verify it fails** — it actually passes already (pure Task-3 math); run it to lock the sanity baseline: `python -m pytest tests/test_fl_regression_glue.py -v` — Expected: PASS. The real deliverable of this task is wiring; the smoke run (Task 8) is its integration test.

- [ ] **Step 3: program.py — add cached generator property**

Directly below the `sympy_expr_lie` cached property (~line 600) add:
```python
    @cached_property
    def sympy_expr_generator(self):
        """Full stochastic generator LV for Foster-Lyapunov tasks (spec 2026-07-28)."""
        from config import dynamics, generator_spec
        from utils.fl_reward import generator_expr
        state_variables, dynamics_ode = dynamics()
        return generator_expr(self.sympy_expr[0], state_variables,
                              dynamics_ode, generator_spec()["noise"])
```

- [ ] **Step 4: regression.py — capture metric name, register metric, branch reward_function**

(a) In `RegressionTask.__init__`, next to where the `metric` argument is passed to `make_regression_metric(...)`, add: `self.metric_name = metric`.

(b) In the `all_metrics` dict (~line 329) add a never-called placeholder entry:
```python
        "foster_lyapunov": (lambda y, y_hat: 0.0, 1),
```

(c) At the TOP of `reward_function(self, p)` (before the origin computation) insert:
```python
        if self.metric_name == "foster_lyapunov":
            return self._fl_reward(p)
```
and add the method:
```python
    def _fl_reward(self, p):
        import sympy as sym_
        from config import dynamics, generator_spec
        from utils.fl_reward import fl_penalty
        state_vars, _ = dynamics()
        spec = generator_spec()
        y_hat = p.execute(self.X_train)
        if p.invalid:
            return self.invalid_reward
        used = p.sympy_expr[0].free_symbols
        if len(used) < len(state_vars):
            return self.invalid_reward
        try:
            args = [self.X_train[:, i] for i in range(self.X_train.shape[1])]
            lv_vals = sym_.lambdify(state_vars, p.sympy_expr_generator, "numpy")(*args)
            h_vals = sym_.lambdify(state_vars, spec["H"], "numpy")(*args)
            lv_vals = np.broadcast_to(np.asarray(lv_vals, dtype=np.float64), y_hat.shape)
        except Exception:
            return self.invalid_reward
        return 1.0 / (1.0 + fl_penalty(y_hat, lv_vals, h_vals, spec))
```

(d) At the TOP of `evaluate(self, p)` insert the FL branch (mirror structure of the existing one, using the big test set):
```python
        if self.metric_name == "foster_lyapunov":
            import sympy as sym_
            from config import dynamics, generator_spec
            from utils.fl_reward import fl_violation
            state_vars, _ = dynamics()
            spec = generator_spec()
            y_hat = p.execute(self.X_test)
            if p.invalid or len(p.sympy_expr[0].free_symbols) < len(state_vars):
                nmse_test = 1e6
            else:
                try:
                    args = [self.X_test[:, i] for i in range(self.X_test.shape[1])]
                    lv = sym_.lambdify(state_vars, p.sympy_expr_generator, "numpy")(*args)
                    h = sym_.lambdify(state_vars, spec["H"], "numpy")(*args)
                    lv = np.broadcast_to(np.asarray(lv, dtype=np.float64), y_hat.shape)
                    nmse_test = fl_violation(y_hat, lv, h, spec)
                except Exception:
                    nmse_test = 1e6
            success = nmse_test < self.threshold
            return {"nmse_test": nmse_test, "nmse_test_noiseless": nmse_test,
                    "success": success}
```
Check the existing `evaluate` return dict's exact keys first and match them (add any extra keys it returns, filled with the same values).

- [ ] **Step 5: Import-sanity check** (regression.py is imported lazily with `from config import dynamics` at module level — verify non-rotor path unaffected):

```bash
python - <<'EOF'
import ast
src = open("libs/sd3/dso/dso/task/regression/regression.py").read()
ast.parse(src); print("syntax OK")
src2 = open("libs/sd3/dso/dso/program.py").read()
ast.parse(src2); print("program OK")
EOF
```
Expected: `syntax OK` / `program OK`

- [ ] **Step 6: Run all tests** — `python -m pytest tests/test_fl_regression_glue.py tests/test_fl_reward.py tests/test_rotor_dynamics.py tests/test_rotor_configs.py -v` — Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add libs/sd3/dso/dso/program.py libs/sd3/dso/dso/task/regression/regression.py tests/test_fl_regression_glue.py
git commit -m "feat: foster_lyapunov metric branch (full-generator reward + evaluate)"
```

### Task 6: FL mode in utils/train.py (harvesting + final falsifier)

**Files:**
- Modify: `utils/train.py` (module init near line 41; PGD-harvest block ~400-423; SHGO-harvest block ~555-615; final gate ~620-639)
- Modify: `utils/fl_reward.py` (add falsifier helpers)
- Modify: `docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md` (record falsifier deviation)
- Test: `tests/test_fl_falsifier.py`

**Interfaces:**
- Consumes: Task 3 helpers; `FL_SPEC` global.
- Produces in `utils/fl_reward.py`:
  - `fl_objective_fns(V, state_vars, spec) -> (v_fn, lv_fn, h_fn)` lambdified callables.
  - `fl_random_falsify(V, state_vars, drift, spec, low, high, n_rounds=10, n_pts=1_000_000, seed=0) -> (ok: bool, worst: np.ndarray|None)` — uniform box sampling, violation = relu(LV+φ(V))·1_{H>R} + relu(1−V) pointwise; ok iff max pointwise violation ≤ 1e-9.
  - `fl_shgo_falsify(V, state_vars, drift, spec, low, high) -> bool` — SHGO maximizes masked violation (minimizes its negative), penalty masking `− 1e3·relu(R−H)` keeps the search outside K_R; ok iff found max ≤ 1e-9.
  - `fl_harvest(V, state_vars, drift, spec, low, high, n_pts=100_000, k=50, seed) -> np.ndarray (k, dim)` — top-k violating points for X_train augmentation.

- [ ] **Step 1: Write the failing test**

`tests/test_fl_falsifier.py`:
```python
import numpy as np
from example_config.rotor_common import make_rotor
from utils.fl_reward import fl_random_falsify, fl_harvest


def test_random_falsify_rejects_bad_V():
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]          # violates drift condition outside K_R
    low, high = [-3.14, -6, -6], [3.14, 6, 6]
    ok, worst = fl_random_falsify(V, sv, drift, spec, low, high,
                                  n_rounds=2, n_pts=20000, seed=0)
    assert ok is False and worst is not None and worst.shape[-1] == 3


def test_random_falsify_accepts_trivial_negative_drift():
    """V = 2 (constant) fails the all-vars rule upstream, but as pure math:
    LV=0, phi(2)>0 -> drift violated outside K_R. Use it to assert the mask
    works: shrink the box INSIDE K_R so no test point has H>R -> ok=True."""
    sv, drift, spec = make_rotor(2)
    V = 2 + 0 * sv[0]
    low, high = [-0.1, -0.5, -0.5], [0.1, 0.5, 0.5]   # H<=~0.26 < R
    ok, worst = fl_random_falsify(V, sv, drift, spec, low, high,
                                  n_rounds=1, n_pts=1000, seed=0)
    assert ok is True and worst is None


def test_harvest_shape():
    sv, drift, spec = make_rotor(2)
    V = 1 + spec["H"]
    pts = fl_harvest(V, sv, drift, spec, [-3.14, -6, -6], [3.14, 6, 6],
                     n_pts=5000, k=10, seed=1)
    assert pts.shape == (10, 3)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_fl_falsifier.py -v` — Expected: ImportError on the new names.

- [ ] **Step 3: Implement helpers in `utils/fl_reward.py`** (append):

```python
def fl_objective_fns(V, state_vars, drift, spec):
    LV = generator_expr(V, state_vars, drift, spec["noise"])
    v_fn = sym.lambdify(state_vars, V, "numpy")
    lv_fn = sym.lambdify(state_vars, LV, "numpy")
    h_fn = sym.lambdify(state_vars, spec["H"], "numpy")
    return v_fn, lv_fn, h_fn


def _pointwise_violation(v_fn, lv_fn, h_fn, X, spec):
    args = [X[:, i] for i in range(X.shape[1])]
    v = np.clip(np.broadcast_to(np.asarray(v_fn(*args), dtype=np.float64),
                                (X.shape[0],)), -1e30, 1e30)
    lv = np.clip(np.broadcast_to(np.asarray(lv_fn(*args), dtype=np.float64),
                                 (X.shape[0],)), -1e30, 1e30)
    h = np.broadcast_to(np.asarray(h_fn(*args), dtype=np.float64), (X.shape[0],))
    viol = np.maximum(lv + phi_vals(v, spec["alpha"], spec["c"]), 0.0) * (h > spec["R"])
    viol = viol + np.maximum(1.0 - v, 0.0)
    viol = np.where(np.isfinite(viol), viol, 1e6)
    return viol


def fl_random_falsify(V, state_vars, drift, spec, low, high,
                      n_rounds=10, n_pts=1_000_000, seed=0):
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec)
    rng = np.random.default_rng(seed)
    worst_pt, worst_val = None, 0.0
    for _ in range(n_rounds):
        X = rng.uniform(low, high, size=(n_pts, len(low)))
        viol = _pointwise_violation(v_fn, lv_fn, h_fn, X, spec)
        i = int(np.argmax(viol))
        if viol[i] > worst_val:
            worst_val, worst_pt = float(viol[i]), X[i]
    ok = worst_val <= 1e-9
    return ok, (None if ok else worst_pt)


def fl_shgo_falsify(V, state_vars, drift, spec, low, high):
    from scipy.optimize import shgo, Bounds
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec)

    def neg_obj(x):
        X = x.reshape(1, -1)
        args = [X[:, i] for i in range(X.shape[1])]
        v = float(np.clip(v_fn(*args), -1e30, 1e30))
        lv = float(np.clip(lv_fn(*args), -1e30, 1e30))
        h = float(h_fn(*args))
        drift_v = max(lv + float(phi_vals(np.array([v]), spec["alpha"], spec["c"])[0]), 0.0)
        drift_v -= 1e3 * max(spec["R"] - h, 0.0)          # penalty masking inside K_R
        val = max(drift_v, 0.0) + max(1.0 - v, 0.0)
        return -val if np.isfinite(val) else 0.0

    res = shgo(neg_obj, Bounds(low, high), n=512, iters=2,
               sampling_method="simplicial")
    return bool(-res.fun <= 1e-9)


def fl_harvest(V, state_vars, drift, spec, low, high, n_pts=100_000, k=50, seed=0):
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec)
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n_pts, len(low)))
    viol = _pointwise_violation(v_fn, lv_fn, h_fn, X, spec)
    return X[np.argsort(viol)[-k:]]
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_fl_falsifier.py -v` — Expected: 3 PASS

- [ ] **Step 5: Wire FL mode into utils/train.py**

(a) Near line 41 (`from config import dynamics`) add:
```python
try:
    from config import generator_spec as _fl_generator_spec
    FL_SPEC = _fl_generator_spec()
except ImportError:
    FL_SPEC = None
```

(b) PGD-harvest block (~400-423): wrap the existing body in `if FL_SPEC is None:` and add the FL alternative (same X_train append mechanics as the original block — read the block first and reuse its variable names for the programs list and X_train target):
```python
        else:
            from utils.fl_reward import fl_harvest
            sv, dr = dynamics()
            for prog in <the same top-programs list the original block iterates>:
                pts = fl_harvest(prog.sympy_expr[0], sv, dr, FL_SPEC,
                                 low_bound, up_bound, n_pts=100_000, k=50,
                                 seed=epoch)
                <append pts to X_train exactly the way the original block appends
                 PGD counterexamples>
```

(c) SHGO-harvest block (~555-615): wrap in `if FL_SPEC is None:`; in FL mode skip it entirely (random harvest already covers augmentation; SHGO runs in the final gate).

(d) Final gate (~620-639): replace with FL variant when `FL_SPEC is not None`:
```python
        if FL_SPEC is not None:
            from utils.fl_reward import fl_random_falsify, fl_shgo_falsify
            sv, dr = dynamics()
            overall_check = []
            for i in range(len(programs)):
                if not success[i]:
                    overall_check.append(False); continue
                V = programs[i].sympy_expr[0]
                ok_rand, worst = fl_random_falsify(V, sv, dr, FL_SPEC,
                                                   low_bound, up_bound,
                                                   n_rounds=10, n_pts=1_000_000,
                                                   seed=i)
                ok_shgo = fl_shgo_falsify(V, sv, dr, FL_SPEC, low_bound, up_bound)
                log_and_print(f"[FL final check] i={i} random_ok={ok_rand} "
                              f"shgo_ok={ok_shgo} worst={worst}")
                overall_check.append(ok_rand and ok_shgo)
        else:
            <original minimizer_check/overall_check body unchanged>
```
Read the surrounding code first; keep `overall_check`'s downstream consumers identical.

(e) Guard the origin-based `derivative_calculate`/`check_options` per-epoch calls: any other call site that runs only makes sense for the deterministic path — wrap each with `if FL_SPEC is None:` (grep `check_options(` and `pgd_check(` call sites in the epoch loop and gate them).

- [ ] **Step 6: Record the falsifier deviation in the spec** — in `docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md`, replace the sentence "the adapted falsifier passes: PGD (sympytorch) and SHGO maximize LV + φ(V) …" with: "the adapted falsifier passes: SHGO (penalty-masked) plus 10×10⁶-point random sampling maximize LV + φ(V) outside K_R; torch-PGD is deferred (log/exp objectives are numerically unsafe under unconstrained attack); in-loop harvesting appends top-50 random-search violators per epoch."

- [ ] **Step 7: Syntax check + full test suite**

```bash
python -c "import ast; ast.parse(open('utils/train.py').read()); print('train OK')"
python -m pytest tests/ -v -x -k "rotor or fl_"
```
Expected: `train OK`, all rotor/fl tests PASS.

- [ ] **Step 8: Commit**

```bash
git add utils/train.py utils/fl_reward.py tests/test_fl_falsifier.py docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md
git commit -m "feat: FL-mode harvesting + SHGO/random final falsifier in train loop"
```

### Task 7: Encoder compatibility test for bond dynamics

**Files:**
- Test: `tests/test_rotor_encoder.py`
- Possibly modify: `utils/train.py` tokenizer (~1383-1457) IF the test fails.

**Interfaces:**
- Consumes: `prepare_encoder_input` (utils/train.py), rotor configs.

- [ ] **Step 1: Write the test**

`tests/test_rotor_encoder.py`:
```python
import pytest
from example_config.rotor_common import make_rotor


@pytest.mark.parametrize("N", [2, 3, 5])
def test_encoder_tokenizes_bond_dynamics(N):
    from utils.train import prepare_encoder_input
    sv, drift, _ = make_rotor(N)
    out = prepare_encoder_input(drift, sv)
    assert out is not None
```
NOTE: read `prepare_encoder_input`'s actual signature at utils/train.py:1383 first and adapt the call (it may take the dynamics list only, or (dynamics, state_vars)); the assertion is just "does not raise KeyError".

- [ ] **Step 2: Run it** — `python -m pytest tests/test_rotor_encoder.py -v`
If PASS: done. If `KeyError` on `sympy` node types: the bond drift `p2 − p1` is `Add(p2, Mul(-1, p1))` — extend the tokenizer's operator table (utils/train.py:1438-1457) with the missing entry following the existing pattern for `add`/`mul`, re-run until PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rotor_encoder.py utils/train.py
git commit -m "test: dynamics encoder handles bond-coordinate drift"
```

### Task 8: repro scripts + smoke run (both arms, rotor2)

**Files:**
- Create: `repro/setup_runs_rotor.sh`, `repro/run_one_rotor.sbatch`, `repro/submit_rotor.sh`, `repro/smoke_rotor.sbatch`

**Interfaces:**
- Consumes: Tasks 2-7 complete, committed on `rotor-benchmarks`.
- Produces: run dirs `/anvil/scratch/x-cchen47/LyaDis/runs_rotor/<sys>/<arm>/`; smoke jobs on gpu-debug.

- [ ] **Step 1: Read the templates** — `repro/run_one_loop.sbatch`, `repro/run_one_diffusion.sbatch`, `repro/smoke_diffusion.sbatch`, `repro/setup_runs.sh` (exact module loads, python invocation, and LYA_LAMBDA value come from these; reuse them verbatim).

- [ ] **Step 2: Write `repro/setup_runs_rotor.sh`** (modeled on confirm_5seed.sh's snapshot pattern):

```bash
#!/bin/bash
# Snapshot the rotor-benchmarks worktree into per-(system,arm) run dirs.
# Usage: setup_runs_rotor.sh   (run from the rotor worktree root)
set -euo pipefail
ROOT=/home/x-cchen47/ALFD-rotor
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor
cd "$ROOT"
for sys in rotor2 rotor3 rotor4 rotor5; do
  for arm in transformer diffusion; do
    dst="$RUNS_BASE/$sys/$arm"
    rm -rf "$dst"; mkdir -p "$dst"
    tar cf - --exclude=.git --exclude=log --exclude=logs --exclude=experiments . \
      | ( cd "$dst" && tar xf - )
    cp "$dst/example_config/config_${sys}.py" "$dst/config.py"
    mkdir -p "$dst/log" "$dst/logs"
  done
done
echo "run dirs ready under $RUNS_BASE"
```

- [ ] **Step 3: Write `repro/run_one_rotor.sbatch`** — start from `cp repro/run_one_loop.sbatch repro/run_one_rotor.sbatch`, then:
  - keep ALL module-load and `python Lyapunov_test_dso.py`-style invocation lines untouched;
  - replace the EXPID/array-seed logic with fixed `SEED=1` (`export LYA_SEED_START=1`, `LYA_SEED_RUNS=1`);
  - `LYA_BENCHMARK` comes from `--export` (`BM` variable, like `run_one.sbatch` does);
  - add the arm switch (copy the `LYA_LAMBDA` line's value verbatim from `run_one_diffusion.sbatch`):
```bash
if [[ "${ARM}" == "diffusion" ]]; then
  export LYA_CONTROLLER=DiffusionController
  export LYA_SAMPLER_ORDER=blocksemiar
  # LYA_LAMBDA line copied verbatim from run_one_diffusion.sbatch
fi
```
  (blocksemiar block_size=4 and entbonus weight come from the loop_20 commit defaults — no extra env needed.)

- [ ] **Step 4: Write `repro/submit_rotor.sh`**:

```bash
#!/bin/bash
set -euo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor
declare -A BM=( [rotor2]=fn_rotor_2 [rotor3]=fn_rotor_3 [rotor4]=fn_rotor_4 [rotor5]=fn_rotor_5 )
for sys in rotor2 rotor3 rotor4 rotor5; do
  for arm in transformer diffusion; do
    sbatch --job-name="lyarot_${sys}_${arm}" --time=16:00:00 \
           --export=ALL,SYS="$sys",BM="${BM[$sys]}",ARM="$arm" \
           --chdir="$RUNS_BASE/$sys/$arm" \
           /home/x-cchen47/ALFD-rotor/repro/run_one_rotor.sbatch
  done
done
```

- [ ] **Step 5: Write `repro/smoke_rotor.sbatch`** — copy `repro/smoke_diffusion.sbatch`, change: benchmark env to `fn_rotor_2`, run dir to the rotor2 snapshot, keep `LYA_NSAMPLES=20000` + 30-min gpu-debug partition; parameterize ARM the same way as Step 3 so one script serves both smoke jobs.

- [ ] **Step 6: Run setup + submit the two smoke jobs**

```bash
bash repro/setup_runs_rotor.sh
sbatch --export=ALL,ARM=transformer --chdir=/anvil/scratch/x-cchen47/LyaDis/runs_rotor/rotor2/transformer repro/smoke_rotor.sbatch
sbatch --export=ALL,ARM=diffusion  --chdir=/anvil/scratch/x-cchen47/LyaDis/runs_rotor/rotor2/diffusion  repro/smoke_rotor.sbatch
```
Expected: two job IDs; then poll `squeue -u x-cchen47 | grep -i rot`.

- [ ] **Step 7: Verify smoke logs** (after jobs finish, ~30 min): in each smoke `.out`, check ALL of:
  - no Python traceback;
  - `[Test epoch=0001]` line appears (training loop ran);
  - reward values strictly in (0,1) and varying across epochs (FL metric live, not invalid_reward 0 everywhere);
  - for the diffusion arm: `[diff]` sampler diagnostic line mentions the sampler and runs without repair-rate 1.0.
If any check fails: fix, re-run smoke. Do NOT proceed to Task 9 with a failing smoke.

- [ ] **Step 8: Commit**

```bash
git add repro/setup_runs_rotor.sh repro/run_one_rotor.sbatch repro/submit_rotor.sh repro/smoke_rotor.sbatch
git commit -m "feat: rotor run/smoke SLURM harness (transformer + diffusion arms)"
```

### Task 9: Submit the 8 full runs + record

**Files:**
- Modify: `docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md` (append run record section)

- [ ] **Step 1: Re-snapshot run dirs** (pick up any smoke fixes): `bash repro/setup_runs_rotor.sh`

- [ ] **Step 2: Submit**: `bash repro/submit_rotor.sh` — Expected: 8 job IDs printed.

- [ ] **Step 3: Verify queue**: `squeue -u x-cchen47 -o "%.12i %.28j %.8T" | grep lyarot` — Expected: 8 rows.

- [ ] **Step 4: Append to the spec doc** a "Run record 2026-07-XX" section listing: commit SHA of `rotor-benchmarks` HEAD, the 8 job IDs, run-dir paths. Commit:

```bash
git add docs/superpowers/specs/2026-07-28-rotor-fl-first-runs-design.md
git commit -m "docs: record rotor FL 8-run submission (job IDs + commit SHA)"
```

- [ ] **Step 5: Report to user** — table of 8 jobs (system × arm × job ID), expected completion time (≤16 h), where results will land, and the reminder that rotor5 is expected to fail by design.

---

## Self-Review (done at plan time)

- Spec coverage: branch strategy (T1), systems+CSV+tsv (T2/T4), full-generator reward (T3/T5), grammar/priors (T4), falsifier+harvesting (T6), encoder risk (T7), 8 runs + smoke (T8/T9). Out-of-scope items untouched. Falsifier deviation is recorded in the spec by T6 Step 6.
- Placeholder scan: the two "copy verbatim from <file>" steps (config_factory boilerplate, LYA_LAMBDA line) reference exact source files that exist on the branch — treated as concrete.
- Type consistency: `make_rotor` returns 3-tuple everywhere; `spec` dict keys uniform; `fl_penalty`/`fl_violation` signatures consistent across T3/T5/T6.
