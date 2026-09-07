# Rotor2 blind skeleton search — design

Date: 2026-09-07. Follows CONCLUSIONS C44 (out-of-box reconnaissance) and
C45 (gate ρ becomes a search object). Supersedes, for this track, the C8
guided-template slot search: nothing inside V is fixed except the three-term
skeleton. The C8 slot pipeline stays untouched and selectable (`LYA_FL_TEMPLATE=1`);
this track is a third mode, `LYA_FL_BLIND=1`.

## 1. Decisions taken in the 2026-09-07 session (user-decided)

| # | Question | Decision |
|---|----------|----------|
| D1 | What is "known" about V | **Option A**: only the skeleton `V = 1 + e^{θH} + A·ρ·p̃²·e^{θp̃²/2}`. ρ and p̃ are free symbolic functions of (r₁, p₁, p₂); no series structure for p̃, no χ(p₂²/S²) structure for ρ. |
| D2 | Success criterion | **Global feasibility only**: assembled V satisfies `LV ≤ −c·V/(2+log V)` outside a compact K, judged out-of-box (C44 style). No comparison to the document's answer is part of the judge. |
| D3 | Size of the exemption set K | **Calibrated on the reference** (C1 practice): the document's V at θ=0.1 is run through the judge; K_max = 1.25 × the (M, R) it needs. The search sees only the numbers. |
| D4 | Vocabulary | A smooth-step primitive `step(z)` is allowed (χ: 0 for z≤0, 1 for z≥1, C^∞ bump splice in between). It reveals "there is a smooth switch", nothing about where or on what. |
| D5 | Engine | The existing rotor pipeline: transformer / diffusion generator + GP-meld, both arms, one seed each. |
| D6 | Candidate encoding | **Single tree with a `pair` root**: candidate = `pair(ρ, p̃)`; left subtree is ρ, right subtree is p̃. Root constraint enforced by a prior (RL sampler), by the diffusion grammar mask, and by GP's `check_constraint`. |
| D7 | θ | Fixed at **0.1** for this round (θ-scan 2026-09-07: fast branch tolerates θ ≲ 0.3, slow region θ < 1; 0.1 is safe with margin and cuts the transition-band threshold from p₁≈64 to ≈24). |
| D8 | Scope | N=2 only. Budget = the August stage-1 jobs (LYA_NSAMPLES=20000, 16 h, 1 GPU + 16 CPU). |

## 2. Candidate space

State symbols `x1=r₁, x2=p₁, x3=p₂` (bond coordinates, `example_config/rotor_common.py`).

Function set (blind mode): `add, sub, mul, div, sin, cos, exp, n2, step, pair`
plus hard-coded constant terminals `1`, `2` (function_set accepts float strings;
`create_tokens` turns them into `HardCodedConstant`s). No BFGS `const` token:
each candidate's judge call is far more expensive than the stage residual, and
the inner constant optimizer would multiply it by ~50. The document's answer
needs −½, 3, 4 — all reachable from 1, 2 with div/add/mul.

New tokens (in `libs/sd3/dso/dso/functions.py`):

* `step`, arity 1: numpy `chi(z) = b(z)/(b(z)+b(1−z))`, `b(u)=exp(−1/u)·[u>0]`.
  Identically 0 for z ≤ 0, identically 1 for z ≥ 1, C^∞. Note the normalised
  argument: the document's χ has its ramp on [1,4]; here the ramp is [0,1] and
  the candidate supplies any affine/other transform of its argument.
* `pair`, arity 2: numpy implementation returns its first argument (only used by
  `Program.execute`, which blind mode never calls for scoring).

sympy side: `convert_to_sympy` leaves both names alone; `parse_expr` produces
undefined `Function('step')`, `Function('pair')`. The judge replaces `step` by a
concrete sympy class `Chi(Function)` with `fdiff` returning `ChiD1`, `ChiD2`
(first/second derivative classes) so `generator_expr` differentiates it
symbolically without Piecewise blow-up; lambdify maps the three classes to
numpy implementations.

Priors (blind mode, replacing `apply_slot_overrides`):

* `root` (new `RootConstraint`): at position 0 only `pair` is allowed; at any
  later position `pair` is forbidden. `is_violated` implements the same rule for
  GP-meld offspring.
* `length`: min 5, max 60 (the diffusion grammar's `max_len=64` is the hard cap;
  the document's answer with four counterterm orders is ≈70 tokens in this
  encoding, so the round can at best reach order 3 — recorded as a known limit).
* `soft_length`: loc 28, scale 8.
* `repeat`: not needed (no const token).
* `inverse`, `trig`, `no_inputs`: as in the rotor config.

Diffusion arm: `models/diffusion/grammar.py::valid_token_mask` gets the same
root rule (position 0 → only `pair`; elsewhere `pair` masked), and `repair`
inherits it automatically because it consults `valid_token_mask`.

## 3. The judge (`utils/fl_blind.py`)

Pure sympy/numpy, no framework imports, mirroring `utils/fl_reward.py`.

### 3.1 Assembly and log-domain generator

Given candidate sympy `pair(ρ, p̃)`, θ, H, drift, noise `{b: κ_b}`:

```
T̂₂ = p̃² · exp(θ p̃²/2)              (A factored out; ρ factored out)
log V  = logsumexp( 0,  θH,  log A + log ρ + log T̂₂ )     log ρ = −∞ where ρ = 0
LV / V = w₁·Q₁ + w₂·[ ρ·Q̂₂ + Lρ + 2 Σ_b κ_b ∂_bρ · ∂_b log T̂₂ ]
   w₁ = exp(θH − log V),   w₂ = exp(log A + log T̂₂ − log V)
   Q₁ = θ·LH + θ² Σ_b κ_b (∂_b H)²                       (= L e^{θH} / e^{θH})
   Q̂₂ = L(log T̂₂) + Σ_b κ_b (∂_b log T̂₂)²               (= L T̂₂ / T̂₂)
ratio = −(LV/V) · (2 + log V)
```

Every factor is a symbolic expression of moderate size (log T̂₂ = 2 log|p̃| +
θp̃²/2 — no exponential ever evaluated), so nothing overflows for any p̃ the
generator can write. No division by ρ anywhere; the exact-zero region of ρ
contributes w₂ = 0. This replaces C44's `e^{θp₂²/2}` scaling, which assumed
p̃ ≈ p₂.

The drift condition `LV + c·V/(2+log V) ≤ 0` holds at x iff `ratio(x) ≥ c`.

### 3.2 Evaluation grid (fixed, calibrated)

Product grid, built once from the calibration JSON:

* r₁: 25 points uniform in [−π, π].
* p₁: 0, ±(geometric from 0.25 to P₁max, 24 values) → 49 values.
* p₂: 0, ±(geometric from 0.25 to P₂max, 40 values) → 81 values.

≈ 99k points. P₁max = 2·√M_max, P₂max = 4·R₂max (see §4). The grid is the same
for every candidate and both arms (stage-mode lesson: a moving point set adds
reward noise and defeats caching); train.py's harvest/bootstrap is skipped in
blind mode exactly as in stage mode.

### 3.3 Score

With K_max = {|p₁| ≤ √M_max and |p₂| ≤ R₂max} and target rate c_t:

```
Ω_out = grid points outside K_max
v_out = mean over Ω_out of  max(c_t − clip(ratio, −1e3, ∞), 0) / c_t
v_in  = fraction of grid points inside K_max with ratio < c_t
pen   = v_out + 0.1 · v_in
reward = 1 / (1 + pen)
```

The clip keeps early candidates (ratios of −1e17) from flattening the signal;
v_in is the "smaller K is better" secondary term from D3's discussion.

A is not searched: per candidate the judge evaluates the A-independent fields
once and combines them for A ∈ {0.1, 1, 10, 100, 1000}; the reward is the best
of the five. The chosen A is stored with the candidate for reporting.

Invalid (reward = invalid_reward): root is not `pair`; ρ < 0 anywhere on the
grid; any non-finite log V or LV/V at a point with ρ > 0; expression free
symbols not ⊆ {x1, x2, x3}; p̃ has no dependence on p₂ (then T̂₂ cannot supply
fast-branch drift — a cheap early reject, kept explicit so it is auditable).

### 3.4 Final check (train.py's `eval_all` gate, blind branch)

A candidate with pen == 0 on the training grid is re-judged on (i) a 2× denser
grid and (ii) 10 × 10⁶ uniform random points in {|p₁| ≤ P₁max, |p₂| ≤ P₂max}
∖ K_max, both in the same log-domain form. Pass = every point satisfies
ratio ≥ c_t. SHGO is not used (Piecewise-free but the log-domain objective is
not smooth across ρ = 0; random search at 10⁷ points is the C44 idiom).

## 4. Calibration (`repro/blind_calibrate_rotor2.py`)

Runs once, before any job:

1. Build the document's V (eq. 13–17, `_chi(p₂²/S²)` with S = p₁⁴ + R_ρ,
   R_ρ = 20 as in C36) at θ = 0.1, A = 20.
2. Evaluate ratio on a wide grid (|p₁| ≤ 100, |p₂| ≤ 10⁵) with the §3.1 judge
   (same code path — this is also the judge's acceptance test).
3. Find the smallest box K_ref = {|p₁| ≤ √M_ref, |p₂| ≤ R₂ref} such that every
   grid point outside it has ratio ≥ c_ref, scanning c_ref over
   {θ²/4 · 2^{−k}} downward until K_ref exists (expected from the θ-scan:
   transition band clean for |p₁| ≳ 24, fast branch clean for |p₂| ≳ 500).
4. Write `example_config/blind_rotor2.json`:
   `{"theta": 0.1, "M_max": 1.25²·M_ref, "R2_max": 1.25·R₂ref, "c_t": 0.5·c_ref,
     "M_ref", "R2_ref", "c_ref", "A_ref": 20}`.
   The 0.5 on c follows C4 (c = 0.3–0.5 × c*).
5. Print the same table as C44 (region / min ratio / worst point) for the ledger.

The reference expression itself is never written into the JSON or the run dir.

## 5. Pipeline integration

| File | Change |
|------|--------|
| `libs/sd3/dso/dso/functions.py` | add `step`, `pair` tokens (unprotected and protected maps). |
| `libs/sd3/dso/dso/prior.py` | `RootConstraint(library, token)`; key `"root"` in `make_prior`. |
| `models/diffusion/grammar.py` | optional `root_token`; position-0 / elsewhere mask rule. Wired from `models/diffusion/controller.py` when the config prior has `root.on`. |
| `utils/fl_blind.py` | judge (§3), `Chi/ChiD1/ChiD2`, `fl_blind_reward(expr, spec) -> (pen, A, ok)`, `fl_blind_final_check`. |
| `libs/sd3/dso/dso/task/regression/regression.py` | `_fl_reward` / `evaluate`: `if spec.get("blind")` branch before the template branch; no `execute()`, no free-symbol-count rule. |
| `utils/train.py` | blind mode: keep X_train pristine (stage idiom); final gate calls `fl_blind_final_check`. |
| `example_config/rotor_common.py` | `FL_BLIND = env LYA_FL_BLIND == "1"`; `spec["blind"] = json.load(blind_rotor2.json)`; `apply_blind_overrides(cfg)` sets function set, priors (§2). Mutually exclusive with `FL_TEMPLATE` (assert). |
| `repro/run_one_rotor_blind.sbatch`, `repro/submit_rotor_blind.sh` | both arms, `LYA_FL_BLIND=1`, `LYA_NSAMPLES=20000`, 16 h; run dir snapshot via `setup_runs_rotor.sh` into `runs_rotor_blind/`. |
| `docs/CONCLUSIONS.md` | C46: calibration numbers + job ids at submission; later entries for readouts. |

Not changed: C8 slot path, `fl_scaffold.py`, `fl_reward.py`, existing configs' defaults.

## 6. Testing

`tests/test_fl_blind.py`:

1. **Judge vs. mpmath**: at 6 hand-picked points (slow / transition / fast, one
   with ρ = 0 exactly) the log-domain ratio of the document's V matches a
   40-digit unscaled mpmath evaluation to 1e-8 relative.
2. **Reference passes**: document's V at θ=0.1 has pen == 0 with the calibrated
   JSON (this is the calibration script's own acceptance run, re-executed on a
   coarser grid so the test runs in < 30 s).
3. **Order-1 truncation fails, order-4 passes** the same judge (monotone signal).
4. **Invalid paths**: ρ = −step(...) → invalid; p̃ without p₂ → invalid;
   `pair` missing → invalid; overflow probe p̃ = 5·p₂ at p₂ = 10⁵ → finite reward.
5. **Chi derivatives**: `ChiD1`, `ChiD2` agree with finite differences of `Chi`
   on (−1, 2) and are exactly 0 outside (0, 1).

`tests/test_blind_priors.py`: `RootConstraint` masks; `is_violated` on a GP
individual with `pair` at depth 1; diffusion `valid_token_mask` position-0
rule; a sampled batch from each controller starts with `pair` and contains no
other `pair`.

`tests/test_rotor_configs.py` gets a blind-mode case (env var set) asserting
the function set and priors.

## 7. Reporting and stop rules

* Every epoch log already prints the best reward; blind mode adds the chosen A
  and the (v_out, v_in) split for the top candidate.
* Success = a candidate passes §3.4. Then: record V, c_t, K_max, A in
  CONCLUSIONS, and only then compare with the document (attribution is
  commentary, not judgement — D2).
* Expected outcome is a wall (C23 precedent). The readout that matters even on
  failure: best (v_out, v_in) trajectory per arm, whether any candidate ever
  produced a positive ratio on the fast branch (|p₂| ≥ R₂max), and the shape
  of the best ρ. These go into the C46+ entries with the same "verdict" format
  as C42.

## 8. Out of scope

N ≥ 3; searching θ (fixed 0.1 this round — a θ-scan entry is a follow-up);
gate-only search with p̃ frozen (option B, kept as fallback if A produces no
fast-branch signal at all); Lean formalisation of any blind result.
