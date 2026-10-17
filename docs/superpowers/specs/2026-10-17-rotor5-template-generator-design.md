# Template generator with log-domain parts, tropical tail checks and LP splicing — design

Date: 2026-10-17. Branch `rotor5-template` (forked from `rotor-benchmarks` @ ad5d8da).
Source: the user's summary "生成框架的三项改造", archived verbatim in
`2026-10-17-generator-three-changes-source.md` (the *source*). This document records
how the source is turned into code, the decisions taken where the source leaves
freedom, and the honest limits. Section numbers §1–§3 follow the source.

Status: as-built notes are added in §9 after implementation.

## 0. Summary

| Source item | Module | One-line description |
|---|---|---|
| §1 log-domain + noise in one step | `utils/fl_tmpl/logdomain.py`, `asym.py` | per part `V_r = P_r e^{F_r}`: `G_r = h^α (L F_r + Γ(F_r,F_r))` as an asymptotic sum, never forming `e^F` |
| §1 automatic noise averaging | `utils/fl_tmpl/averaging.py::noise_average` | cancels the fast-oscillating part of the leading order of `G_r` with a factor `1+N_r`, `N_r = n(q)/p_j^{k}`; reports the θ window with and without it |
| §2 macro tokens (slow variables, averaged energies) | `utils/fl_tmpl/averaging.py::slow_momentum`, `macros.py` | cohomological-equation solutions as data objects (value / `L`-image / `∂_{p_b}` expansions, remainder orders, provenance) |
| §2 template + slots | `utils/fl_tmpl/template.py` | `V = 1 + e^{θH} + Σ_r ψ_r P_r e^{|ỹ_r|^{a_r}} e^{θ Ẽ_r}(1+N_r)`; RL fills slots only |
| §3 steps 2–4 directional blow-up, leading-term blocks, walls | `utils/fl_tmpl/tropical.py` | direction space `s`, tie-cell (block) enumeration by LP, wall blocks, zero-coordinate (compact) faces, degenerate-wall refinement, oscillation flag |
| §3 step 5 splicing | `utils/fl_tmpl/splice.py` | junction inequalities → LP over gap exponents with per-junction slack; constant-ratio junctions → Bellman–Ford positive-cycle test |
| 分工与校准 | `utils/fl_tmpl/reference.py`, `tests/test_tmpl_calibration.py` | N=3/N=4 paper V as slot settings; the three calibration checks |
| RL fills slots | `functions.py` slot tokens, `prior.py::SlotPositionConstraint`, diffusion grammar slot mask, `regression.py` template branch, `example_config/rotor_common.py` (`LYA_FL_TMPL=1`), `repro/*_tmpl*` | both arms (transformer, diffusion) + GP-meld search the slot sequence; reward = 1/(1+failed blocks+LP slack+cycle) |

## 1. Log domain (source §1)

### 1.1 What is evaluated

For a part `V_r = P_r · e^{F_r}` write `F̂_r = log P_r + F_r` (so `V_r = e^{F̂_r}`):

    L e^{F̂}/e^{F̂} = L F̂ + Γ(F̂,F̂),   Γ(F,F) = Σ_b κ_b (∂_{p_b} F)²,   κ_b = γ_b T_b

and the evaluator checks `G_r = h^α (L F̂_r + Γ(F̂_r,F̂_r)) ≤ −c` on the part's region,
`h = 1 + H`. Here κ is the second-derivative coefficient of the repo's generator
(`utils/fl_reward.generator_expr`: κ_1 = 1, κ_N = 2 for N ≥ 3; N=2 κ_1 = 1).

`F̂_r` is a sum of components `φ_k(Y_k)` (prefactor `d·log Y`, gap `|Y|^a`, energy `θ E`,
noise `log(1+N)`), each a function of one building block `Y_k` whose data are known
(§2.1). By the chain rule

    L F̂ = Σ_k [ φ_k'(Y_k) L Y_k + Σ_b κ_b φ_k''(Y_k) (∂_b Y_k)² ]
    Γ   = Σ_b κ_b ( Σ_k φ_k'(Y_k) ∂_b Y_k )²

All products/powers/inverses are carried out in the asymptotic-sum algebra of §1.2.
No exponential of a candidate quantity is ever formed; every term of `G_r` is a
product of a monomial in the momenta and an O(1) trigonometric coefficient.

### 1.2 Asymptotic sums (`asym.py`)

An `AsymSum` is a finite sum of terms `c(q, params) · Π_i p_i^{n_i} |p_i|^{e_i} · Π_A A^{f_A}`
plus *remainder* terms `O(Π_i |p_i|^{e_i} Π_A A^{f_A})` with an unknown bounded coefficient.

* Scaling variables: all momenta `p_1..p_N`. Angles and parameters live in the
  coefficients.
* Monomial key = (exponent vector over momenta and atoms as `Fraction`s, parity
  vector `n_i mod 2`). `p^n` with integer `n` keeps its sign; `|p|^e` does not.
* *Positive atoms* `A = Σ_m c_m · (even monomial)` with `c_m > 0` (example
  `p_2² + p_3²`) may appear to any rational power. They cannot cancel, so their
  tropical order is `max_m (e_m·s)` and their unit on a face is the sum of the tied
  monomials' units. Linear forms that *can* vanish (`p_3 − p_2`, `2p_2 − p_3`) are not
  atoms; they are handled by refinement rules (§3.4).
* Series operations for `(1+δ)^c` and `log(1+δ)` truncate after `n_max` and add the
  remainder `Σ_m C |m|^{n_max+1}` over the monomials `m` of `δ` (valid because
  `|δ|^{n+1} ≤ n_δ^{n+1} Σ_m |m|^{n+1}`), so truncation never hides an order.
* Decomposition of a sympy expression into an `AsymSum` (`from_sympy`) accepts sums
  of products of: momentum powers, `Abs(p)^e`, registered atoms to rational powers,
  and momentum-free factors. Anything else raises `NotQuasiHomogeneous` — this is
  the source's "拟齐次约束": a slot combination that produces such a term is rejected
  before scoring.

### 1.3 Noise averaging as an automatic step (`averaging.py::noise_average`)

Input: a part's `G_r` (computed without the factor), the part's fast variables.
Algorithm (source §1 revision; paper Prop. 2.4 footnote 3, Prop. 3.5):

1. Take the terms of `L F̂ + Γ` at the part's *target order* (`p_j^{-2}` for the
   CE single-fast parts, `p_j^{-2}`/`p_3^{-2}` for the both-fast part, i.e. the order
   that becomes O(1) after the `h^α` factor).
2. Fourier-decompose their coefficients in the bond angles (`fl_multiphase._mode_table`).
   A mode is *fast in `p_f`* if its phase moves with rate `ω_f p_f + (slower)`,
   `ω_f ≠ 0`; the fast frequency is read off from the bond-angle drift
   (`dr_{f−1}/dt ∋ +p_f`, `dr_f/dt ∋ −p_f`).
3. Modes fast in exactly one `p_f` are cancelled by `N_r ∋ (a sin φ − b cos φ)/(ω_f p_f^{k+1})`
   with the sign that makes `L N_r` remove the mode at order `k` (`k` = target order).
   Modes fast in two or more momenta (resonant combinations) are left and reported.
4. Re-evaluate `G_r` with the factor `(1+N_r)` (as the component `log(1+N_r)`); the
   leading coefficient now contains the averages `⟨W²⟩` instead of `W²`.

The step also reports the θ window: the largest θ such that the leading
coefficient's supremum over the compact variables is `< 0`, computed with and
without the factor (bisection in θ; `θ_max^{with}` / `θ_max^{without}`). For the CE
parts this gives `θ < 1/T_b` vs `θ < ⟨W²⟩/(T_b sup W²)` (= 1/(2T_b) for `W = 1−cos`).

### 1.4 Design rules written as constraints

* "补偿项的指数里只放中间转子的平均能量": the energy component's building block must
  be a macro whose kind is `slow_momentum` or `slow_energy` and whose fast variables
  are interior rotors (asserted in `Part.__init__`).
* "每个慢变量都要追踪它对 p_b 的导数": every macro carries `∂_{p_b}` data for every
  noisy momentum `b` (asserted at macro construction; missing data = invalid macro).

## 2. Template and macro tokens (source §2)

### 2.1 Macros (`macros.py`)

A `Macro` is a building block with data:

| field | meaning |
|---|---|
| `name`, `kind` | e.g. `pt2` / `slow_momentum`, `Hc` / `slow_energy` |
| `fast` | fast momenta it is built for (`(2,)`, `(2,3)`) |
| `value` | `AsymSum` expansion of the macro itself |
| `L` | `AsymSum` expansion of its generator image (incl. remainder) |
| `d` | `{b: AsymSum}` expansions of `∂_{p_b}` for every noisy `b` |
| `expr` | explicit sympy expression when available (needed only by the numeric spot check) |
| `provenance` | `computed` (from `expr` by symbolic `L`), `cited` (data copied from a published proposition, with the reference), `danus` (loaded from a JSON file produced by the Danus project), `conjecture` (labelled hypothesis) |

Builders:

* `slow_momentum(system, j, order)` — single-fast averaging (paper §2.1), computed:
  starting from `p_j`, at each order the fast-oscillating modes of `L(·)` (frequency
  `ω ≠ 0` in `q_j`) are cancelled by `(a sin φ − b cos φ)/(ω p_j)`; non-`q_j`-oscillating
  modes that oscillate in slow angles are cancelled only when they are an exact slow
  `L`-derivative (polynomial divisibility by the slow rate, the paper's
  `L⟨W_L W_C⟩` step); everything else is left. Output: `p̃_j` with `L p̃_j = −α_j p_j^{-3} + O_j(p_j^{-4})`
  for N=3/N=4 (verified by the builder; the coefficient is compared with
  `α_j = γ⟨W²⟩` from (1.14)). For N=5's central rotor the builder stops where the
  first averaging layer stops (C42: residual O(p_3^{-3}), second layer open) and
  says so.
* `slow_energy_cited_CE4()` — `H̃_c` with the data of CE (arXiv:1504.04964)
  Prop. 3.2: `H̃_c = (p_2²+p_3²)/2 + O_c(1)`, `L H̃_c = −α_2 p_2^{-2} − α_3 p_3^{-2} + O_c(|p_2|^{-5/2}+|p_3|^{-5/2})`,
  `∂_{p_1} H̃_c = W_L/p_2 + O_c(p_2^{-2})`, `∂_{p_4} H̃_c = W_R/p_3 + O_c(p_3^{-2})`.
  Resonances (R12/R23, Prop. 3.12) are inside the cited statement; that is the
  meaning of "macro token" here.
* `load_danus(path)` — JSON schema for verified parts (`docs/...` §2.4); N=5 uses it
  when the file exists. **The Danus parts (incl. c7) are not available on this
  machine; no values are invented.** Without the file, rotor5 uses `conjecture`
  macros (§2.5) whose unknown constants stay symbolic with a declared positive range,
  and every result is reported as conditional.

### 2.2 Template

    V = 1 + e^{θH} + Σ_r w_r ψ_r V_r,
    V_r = P_r(Y_r^{pre}) · exp(|Y_r^{gap}|^{a_r}) · exp(θ E_r(Y_r^{en})) · (1 + N_r)

`P_r = (Y^{pre})^{d_r}` (`Y^{pre}` = the part's energy macro, `p̃_j` or `H̃_c`), `E_r = Y²/2`
for slow momenta and `E_r = Y` for slow energies, `N_r` from §1.3 (automatic), the
weights `w_r` come from Bellman–Ford (§3.6), `θ` is fixed per run (config).

Switches (source §2: "对数尺度上的区域指数"): `ψ_r = Π_c χ((ℓ_c(log|p|) − b_c)/w)` with
log-linear forms `ℓ_c`. Asymptotically only the homogeneous part matters: region
`D_r` = polytope in direction space (`ℓ_c(s) ≥ 0`), e.g. `Ω_j(k)`: `s_j = 1`,
`s_i ≤ s_j/k`; `Ω_c(ℓ,m)`: `ℓ s_3 ≥ s_2`, `ℓ s_2 ≥ s_3`, `max(s_2,s_3) ≥ m·max(s_1,s_4)`.
The finite offset `b` and width `w` enter only the numeric spot check.

### 2.3 Slots (what RL chooses)

| slot | menu | applies to |
|---|---|---|
| `gap_r` | `none` or one of the part's slow variables (`p̃_j`) | each single-fast part |
| `a_r` | `{1/8, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4}` | parts with `gap_r ≠ none` |
| `deg_r` | `{0,1,2,3,4,6}` | every non-main part |
| region exponents | `k ∈ {2,3,4,6,8}`, `ℓ ∈ {4,6,8,12,16}`, `m ∈ {8,12,16,24,32}` | per region family |
| `recv` | `none`, `E234`, `E234·p̃3^6`, ... (menu in `template.py`) | N=5 only |

N=3 has 4 slots, N=4 has 10, N=5 has 18 (§2.5). The slot sequence is the
"program": `tmpl(S_1, ..., S_n)` with one choice token per slot.

### 2.4 Danus JSON (schema)

    {"system": "rotor5", "macros": [{"name": "pt3", "kind": "slow_momentum", "fast": [3],
      "value": [[coef_str, {"p3": "1"}], ...], "L": [...], "d": {"p1": [...], "p5": [...]},
      "remainders": {"L": [{"p3": "-8"}], ...}, "provenance": "danus", "ref": "<lemma id>"}]}

Coefficients are sympy strings over the bond angles `r1..r4` and named constants
(e.g. `c7`); exponents are rational strings.

### 2.5 rotor5 template (conjectural)

Parts: main; `V_2`, `V_4` (single fast, CE-type, `L p̃_j ≈ −α_j p_j^{-3}`, computed);
`V_3` (central bottleneck, `L p̃_3 ≈ −c_7 p_3^{-7}`, Danus or conjecture);
`V_{23}`, `V_{34}` (pair-fast slow energies, conjecture: `L H̃ ≈ −α p_{outer}^{-2}` );
`V_{24}` (two separated fast rotors, product of single-fast data, computed);
receiver `V_{234}` (menu). α = 3. Unknown constants (`c_7`, pair rates) are positive
symbols sampled over a declared range; the report says "conditional on
{conjecture list}". This is the source's "find or falsify" stage: the expected
honest output is a failure map (which blocks/junctions fail), not a certificate.

## 3. Compactification (source §3)

### 3.1 Direction space

`p_i = u_i R^{s_i}`, `s ∈ [0,1]^N`, `max_i s_i = 1`, units `u_i ∈ ±[1/2, 2]`. A part's
region is `D_r ⊂` direction space with a designated top variable (`s_top = 1`).
`s_i = 0` means "bounded": on faces with `s_i ≡ 0`, `p_i` becomes a compact variable
ranging over the box `[−B, B]` (default `B = 30`), and monomials that differ only in
`p_i`'s exponent tie (they are summed into a polynomial in `p_i`).

### 3.2 Blocks (`tropical.py`)

For an `AsymSum` `G` and a region `D`:

1. Merge terms with equal keys; drop coefficients that vanish identically
   (symbolic `simplify` for small coefficients, else numeric on 64 random points).
2. For each zero-set `Z` (subset of non-top variables that can be 0 in `D`):
   restrict to `D_Z = D ∩ {s_Z = 0}`, merge monomials equal off `Z`.
3. Keep only monomials that attain the max somewhere on `D_Z` (one LP each).
4. Enumerate tie-sets `S` (blocks): `S` is a block iff `{s ∈ D_Z : e_m·s = τ ∀m∈S,
   e_k·s ≤ τ − ε ∀k∉S}` is non-empty for some `ε > 0` (LP maximizing `ε`). Singletons
   with full dimension are the source's "块"; larger `S` are "墙".
5. For each block: order range `[τ_min, τ_max]` (LP), leading coefficient
   `C_S = Σ_{m∈S} c_m(q, p_Z) · unit_m(u)` and its sup/inf over the compact variables
   (angles grid 16 per angle + random 4096 points + local refinement with L-BFGS-B
   from the 8 worst points; units sampled on `±{1/2, 1, 2}`; `p_Z` on the box grid).
6. Remainder control: a remainder term `O(e_R)` is *dangerous* on a block if
   `max_{block} (e_R·s − τ(s)) ≥ 0` (LP); then the block is `undecided`.

Block status:

| status | condition |
|---|---|
| `good` | `τ_max > 0 ⇒ sup C_S < 0`; on `τ = 0`: `sup C_S ≤ −c`; `τ_min ≥ 0`; no dangerous remainder |
| `positive` | `sup C_S > 0` somewhere where `τ ≥ 0` (flag `oscillatory` if `inf C_S < 0 < sup C_S` and `C_S` depends on angles: "an oscillating term was not averaged") |
| `weak` | `τ_min < 0` (drift decays faster than `h^{-α}`) |
| `degenerate` | `C_S ≡ 0` and no refinement rule applies (needs next order) |
| `undecided` | a remainder reaches the leading order |

Compactness argument (source §3 step 3): if every block of `D` is `good`, every
sequence `x_n → ∞` with directions in `D` eventually satisfies `G ≤ −c/2`. What it
does **not** give: `R_0`, and uniformity of the sub-leading terms (Danus).

### 3.3 Walls

Walls are exactly the non-singleton blocks of 3.2-4; their leading coefficient keeps
all tied monomials (source step 4).

### 3.4 Degenerate walls: refinement

A `Refinement(trigger, substitution, new_var, t_range)` rewrites the expression on a
degenerate wall, e.g. `p_3 = p_2 + v`, `v = R^t`, `t ∈ [0, s_2)`, and re-runs 3.2 on the
substituted expression with `t` as a new direction coordinate (source: 共振面
`p_3 − p_2 = R^t`). Refinement rules are attached to macros that contain the
corresponding linear forms. The cited `H̃_c` contains none (its resonances are inside
Prop. 3.2), so for N=4 refinement is exercised by tests only.

### 3.5 Junctions and the LP (`splice.py`)

Log size of a part: `log V_r` expanded as `AsymSum` (power level) + log level
(`d·log|Y|`, atom logs). For a *bad set* `F` of part `r` (a non-good block of `r`
inside its support, or a switch layer `B(Ω_r)`) and an absorber `s` that is good on
the piece `F ∩ D_s`:

* power level: `log V_r − log V_s` restricted to the piece; every positive-order
  monomial with a non-negative coefficient must be strictly dominated by a
  negative-definite monomial (in the LP: the absorber's gap monomial `−|p_j|^{a_s}` or
  energy monomial) at every vertex of the piece → linear inequalities in the
  unknown gap exponents `a` (region exponents are fixed numbers from the slots, so
  vertices are numeric and every inequality is linear);
* log level (when the power level is bounded): `ℓ(s) = (log-level coeff of log V_r − log V_s)(s) + τ_r(s) − τ_s(s)`;
  `< 0` → absorbed; `= 0` → *constant junction* (edge for 3.6); `> 0` → fail.

LP: maximize margin `δ` s.t. all inequalities with slack variables `σ_j ≥ 0`;
infeasible junctions are those with `σ_j > 0` in the min-total-slack solution, and
`σ_j` is "缺多少". Two modes: `fixed` (RL's `a_r` are constants — feasibility check +
slacks) and `free` (LP chooses `a_r`, reported as hint). Coverage: the union of
good absorber pieces must cover `F`; checked on vertices plus 512 random directions
of `F` (heuristic, uncovered directions are reported).

### 3.6 Constant junctions: Bellman–Ford

Edge `r → s` with constant `c_rs = sup(bad leading coeff) · sup e^{(bounded part of log V_r − log V_s)} / inf |good leading coeff|`:
constraint `λ_s ≥ λ_r + log(2 c_rs)` on `λ = log w`. `λ_main = 0`. A positive cycle
(Bellman–Ford on negated weights) = infeasible; the cycle is reported. The
"截断层放大" fallback is only noted in the report for constant junctions.

## 4. Reward and success

    pen = n_bad_blocks + n_uncovered_pieces + Σ_j σ_j / σ_scale + 5·[positive cycle]
    reward = 1/(1+pen)

`success` ⇔ `pen == 0` **and** (when every macro is explicit) the numeric spot check of
§5 passes. A success with cited/danus/conjecture macros is reported as
"asymptotically consistent, conditional on: ..." and handed to Danus.

## 5. Numeric spot check (secondary)

For explicit templates (N=3 computed `p̃_2`; N=4 single-fast parts computed), the
assembled `V` is evaluated in log domain (logsumexp of parts; the BlindJudge
assembly generalized to several parts) on random far points in each part's
region at `R ∈ {10², 10³, 10⁴}`; failures are reported, not used in the reward.

## 6. RL integration

* Tokens: `tmpl` (arity = #slots, root) and one terminal per menu entry, named
  `S<slot>_<index>` (sympy-safe identifiers). Registered dynamically by
  `create_tokens` (names starting with `S` + digit or `tmpl`) — no change to the
  existing tokens.
* Prior `slot` (`SlotPositionConstraint`): position 0 → `tmpl`; position `i` → the
  tokens of slot `i`; nothing else ever. `is_violated` implements the same rule for
  GP-meld offspring.
* Diffusion grammar: `slot_tokens` list → same positional mask; `repair` inherits.
* `regression.py`: `if spec.get("template")` branch before blind; decodes the slot
  choices from `p.traversal` names (never `execute()`), caches the report per slot
  tuple, stores `p.tmpl_report`.
* `train.py`: template branch in the final gate (success = `pen == 0` + spot check)
  and in the harvest switch (no X_train usage).
* Config: `LYA_FL_TMPL=1` (mutually exclusive with blind/template-C8); `LYA_TMPL_THETA`
  (default per N); `rotor_common.apply_tmpl_overrides`. Runs: `repro/setup_runs_rotor_tmpl.sh`,
  `run_one_rotor_tmpl.sbatch`, `smoke_rotor_tmpl.sbatch`, `submit_rotor_tmpl.sh`
  for rotor3/4/5 × {transformer, diffusion}.

## 7. Calibration (source "用 N=3、4 校准评估器")

`tests/test_tmpl_calibration.py` and `repro/tmpl_calibrate.py`:

1. N=3 and N=4 paper V as slot settings → every block good, LP feasible, no
   positive cycle.
2. Gap factor removed (`gap = none`) → the report lists the junction(s) whose
   inequality needs `a > 0` ("缺间隙"), with slack.
3. Noise averaging removed → `θ_max` shrinks (`1 → 1/2` for bath 1 in N=3/N=4's `V_2`;
   `1/2 → 1/4` for bath N in N=4's `V_3`).

## 8. Known limits (honest)

1. Leading-term checks give the sign for large `R` only; no `R_0`, no uniform
   control of sub-leading terms (source risk 2). Remainder orders are tracked, not
   their constants.
2. Coverage of bad sets by absorber pieces is checked on sample directions.
3. Compact suprema are numeric (grid + local refinement), not SMT; a dReal export of
   the per-block inequalities is a follow-up.
4. N=4 both-fast part uses cited Prop. 3.2 data (resonance counterterms R12/R23 are
   not rebuilt here); the computed engine `H̃_c` (C39/C40) has a genuine small divisor
   at `p_3 ≈ 2p_2` and is not used as the macro.
5. N=5 rests on conjectural macros until Danus parts are provided.
6. Danus forbids computation inside its project (AGENTS.md); everything here runs in
   this repo.

## 9. As built (2026-10-17)

Modules (`utils/fl_tmpl/`): `fourier.py` (exact Fourier coefficients: modes in the
bond angles x polynomials in the parameters theta/c7/ap, product-to-sum, recursive
conversion from sympy -- replaces sympy coefficients, whose TR8 canonicalisation was
the bottleneck), `asym.py` (AsymSum, positive atoms, series with remainders, region
pruner), `system.py`, `averaging.py`, `macros.py`, `logdomain.py`, `tropical.py`,
`splice.py`, `template.py`, `reference.py` (N=3/4/5 builders, paper slots),
`slots.py` (menus, token names, decode), `rl.py` (cached reward, pool prefill).

Decisions taken while building (not in the source):

* h^alpha is the positive atom Ah = sum p_i^2/2 times (1 + O(1/Ah)); the floor for
  pruning K is -2 alpha (a K term below it everywhere on the region can never give G
  an order >= 0).  Series components whose lead-level contributions (incl. the Gamma
  cross term) are below the floor are skipped -- in practice the gap factor and the
  prefactor never enter G, only the junction ledger (the source's "间隙因子对漂移几乎没有代价").
* The theta window is "largest theta with every block of the part good" (bisection
  over the full tropical check), not the coefficient at one direction: at the
  diagonal of V_c the two rotors' terms compensate and a single-direction probe gave
  0.5155 instead of 1/2.
* Switch layers are bad sets with error order 2 alpha (the log-scale switch has
  L psi = O(1) on bath momenta); main's cold-bath blocks are bad sets with the order
  of G_main.  Absorption is tested pointwise at sampled directions of each bad set;
  LP rows are generated at the same directions for the absorber that absorbs (or the
  smallest deficit), the bounded level 0 is always an admissible "negative" so a
  slack equals the missing exponent.
* Cited O_c remainders carry z(p1,p4) as |p_b|^2 (rem_degree, a modelling choice);
  the evaluator then requires m > 2 l rem_degree, the quantitative form of "m large".
* Penalty = (#unabsorbed bad sets) + sum over bad sets of the largest LP slack / 0.05
  + 5 [positive cycle].

Calibration results (tests/test_tmpl_calibration.py, all pass):

| check | N=3 | N=4 |
|---|---|---|
| paper V | pass; one constant junction main -> V2 (A large) | pass; one constant junction main -> V_c (M large); free-LP gap 0.126 in (2/l, 2/k) = (0.125, 0.333) |
| gap removed | n/a (paper uses deg 2) | V_c layer s3 = 1/l unabsorbed, slack 2/l + 0.001 |
| noise averaging removed | V2: theta_max 2/3 -> 1/3 | V2 1 -> 1/2, V3 1/2 -> 1/4, V_c 1/2 -> 1/4 |

The automatically built noise factor is N = -theta^2 sin(2 r1)/(4 p2^3), i.e.
F_2' = theta^2 gamma T (<W^2> - W^2) of CE Prop. 2.4.  Main's theta window is the
Ito bound 1/T_max = 1/2.

Single-fast averaging (computed macros) reproduces the benchmark drifts:
L p~2 = -1/(2p^3) (N=2), -1/p^3 (N=3), -1/(2p^3) for both central rotors of N=4
and for p~2, p~4 of N=5; p~3 of N=5 has zero mean through p^-3 and an oscillating
residual at p^-4 (34 modes) -- the C42 obstruction, so p~3's drift enters N=5 only
as the conjecture L p~3 = -c7 p3^-7.

RL integration: tokens `tmpl<n>` / `S<i>_<j>` registered in `create_tokens`;
`SlotConstraint` (prior key `slot`); diffusion `Grammar(slot_tokens=...)`;
`LYA_FL_TMPL=1`, `LYA_TMPL_THETA`, `LYA_TMPL_NSAMPLES`, optional `LYA_TMPL_DANUS`;
`apply_slot_overrides(cfg, N)`; jobs `repro/{setup_runs,run_one,submit}_rotor_tmpl*`.
N=2/3 slot spaces have 420 tuples and most pass (as the source predicts: "N=3
几乎没有需要搜索的部分"); N=4 has ~3e6 tuples, 4-8 s per new tuple.

Known gaps beyond section 8: R12/R23 not rebuilt (cited); N=5 parts conjectural;
`tests/test_rotor_configs.py::test_csv_row` fails on the base branch already (CSV
box changed to +-15 earlier), unrelated to this work.
