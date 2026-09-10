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

### 3.2 Exemption set, evaluation domain, shells (as built, 2026-09-07)

A plain box K fails for this problem: the document's certificate at θ=0.1 must
exempt the transition band for |p₁| ≤ 26, which reaches |p₂| ≈ 6.6×10⁵, while
float64 cannot resolve the fast-branch drift of a *good* candidate beyond
|p₂| ≈ 4×10⁴ (the true L p̃ ≈ 1/(2p̃³) is below cancellation noise; measured
20 % error at 2×10⁵, 10⁻⁵ at 10⁴). So:

* **Exemption set (candidate-dependent, the C36 shape)**
  `K(ρ) = {p₁² ≤ M_max and ρ < 1 and WITNESS} ∪ {p₁² ≤ M_0 and |p₂| ≤ R2_0}`.
  WITNESS(x): on the same (r₁, p₁) line, ρ = 1 at the witness shell |p₂| = R2_near
  (if p₁² ≤ M_near) else at |p₂| = R2_max. A vanishing or infinitesimal gate
  therefore exempts nothing.
* **Scored domain** D = {|p₁| ≤ P1max, |p₂| ≤ P2max = 4×10⁴} (float64-trusted),
  product grid r₁ × p₁ × p₂ = 17 × (2·16+1) × (2·32+1) ≈ 36k points, p₁/p₂ signed
  geometric from 0.25. The same grid for every candidate and both arms.
* **Compactness shells** (ρ must be 1): the near shell {p₁² ≤ M_near = 64,
  R2_near ≤ |p₂| ≤ P2max} inside D, so the fast branch is scored there for
  |p₁| ≤ 8 (order-3 counterterms plus a real gate are required at |p₂| ≈ 10⁴);
  and both witness shells.
* **Far region** |p₂| ∈ [P2max, 4·R2_max] is scored only in the final check, at 30
  digits (mpmath), on a coarse grid outside K(ρ).

### 3.3 Score

```
v_out  = 2·frac(ratio < c_t on D \ K(ρ)) + mean over D \ K(ρ) of log1p( max(c_t − ratio, 0) / c_t )   (ratio clipped at −1e6; C49)
v_comp = ( frac(ρ<1 on near shell) + frac(ρ<1 on near witnesses, p₁²≤M_near)
         + frac(ρ<1 on far witnesses, p₁²≤M_max) ) / 3
v_in   = frac(ratio < c_t on K(ρ))
pen    = v_out + v_comp + 0.1·v_in ;   reward = 1 / (1 + pen)
```

The log compression keeps a gradient from ratio ≈ −10⁶ up to feasibility
(a linear mean saturated every early reward at ≈ 10⁻⁶). Measured landscape on
the reference ladder (C49 score): trivial pair 0.17, order 1 0.16, order 2
0.19, order 3 0.42, order 4 0.98; degenerates: ρ ≡ 0 0.18, checkerboard gate 0.19,
flat V 0.27 (any violate-everywhere candidate is capped at 1/3). Exemption is the
last ρ<1 band under a contiguous ρ = 1 tail (C48).

A is not searched: the A-independent fields are evaluated once and combined
for A ∈ {0.1, 1, 10, 100, 1000}; the reward is the best of the five, the
chosen A is stored on the program (`blind_A`).

Invalid (reward = invalid_reward): root is not `pair`; ρ < 0 or non-finite
anywhere on the grid or the witness shells; non-finite P, f, G2 where ρ > 0;
free symbols not ⊆ {x1, x2, x3}; p̃ without p₂.

Implementation note: only the primitives ρ, ∂ρ, ∂²ρ, p̃, ∂p̃, ∂²p̃ are lambdified
(one call); G2 is assembled numerically. The symbolic product form was 30–100×
slower to code-generate (0.03–0.5 s per candidate now, 1.2 s for the order-4
reference).

### 3.4 Final check (train.py's `eval_all` gate, blind branch)

A candidate with `success` (pen == 0 on the training grid) is re-judged on (i) a
2× denser grid on D plus 10⁷ uniform random points in D, outside K(ρ), with
the shell/witness conditions enforced pointwise; (ii) the far region
|p₂| ∈ [P2max, 4·R2_max], |p₁| ≤ P1max, coarse grid (13 × 25 × 24) outside K(ρ),
evaluated with mpmath at 30 digits in the same log-domain assembly. Pass =
ratio ≥ c_t everywhere scored. No SHGO.

## 4. Calibration (`repro/blind_calibrate_rotor2.py`)

Runs once on the document's V (eq. 13–17, R_ρ = 20, θ = 0.1, A = 20):

1. float64 domain D0 = {|p₁| ≤ 60, |p₂| ≤ 4×10⁴}: zero-level core box and the
   bulk drift rate away from it (c_bulk_D).
2. transition-band threshold in p₁ at 30 digits: last p₁ ∈ {10, …, 40} whose
   band S < |p₂| < 2S violates → **p₁ = 26**, M_max_ref = 676.
3. near shell: ρ = 1 for |p₁| ≤ 8 from |p₂| ≥ 8407 (R2_near_ref); far shell:
   ρ = 1 for |p₁| ≤ √M_max from |p₂| ≥ 2.33×10⁶ (R2_max_ref).
4. far-region bulk rate at 30 digits (c_bulk_far); c_ref = min(c_bulk_D,
   c_bulk_far) = 1.71×10⁻³; **c_t = 0.5·c_ref = 8.56×10⁻⁴** (C4 rule).
5. dense core scan at the c_t level (161 × 321 grid + 5×10⁵ random points):
   M_0_ref = 9.24, R2_0_ref = 211.
6. scaling 1.25 (M: 1.25², lengths: 1.25): **M_max = 1056, R2_max = 2.91×10⁶,
   R2_near = 1.05×10⁴, M_0 = 14.4, R2_0 = 264, P1max = 1.5·√M_max = 48.75.**
7. the reference is run through `fl_blind_final_check` with these numbers
   (must pass) and the JSON `example_config/blind_rotor2.json` is written.

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
