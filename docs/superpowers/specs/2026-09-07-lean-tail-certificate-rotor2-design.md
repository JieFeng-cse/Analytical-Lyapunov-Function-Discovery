# Lean formalization of the rotor2 certificate tail (design)

Date: 2026-09-07. Status: **DONE 2026-09-07** — T1 and T2 proved in
`lean/rotor-fl` (3706 lines, zero sorry, standard axioms only); see C46.
T2 is stated as `L V + φ(V) ≤ C` on K with `C = exp(10¹⁵)·10¹⁰⁰`.
Approved design (user decisions D1–D5 below) with the constants of §3.

## 1. Goal

Give the machine-built rotor2 Foster–Lyapunov certificate (CONCLUSIONS C36) a
machine-checked proof that it satisfies the drift condition **outside the
verification box**, with **explicit constants**. Inside the box (momenta in
±15) the certificate is already verified numerically (SHGO/PGD/random, C36);
the tail has so far been "human mathematics". This closes that gap for N=2.

Not in scope: N ≥ 3, the box interior (a later branch-and-bound + interval
arithmetic project), an automated LLM proving pipeline.

## 2. User decisions

| # | decision |
|---|---|
| D1 | Target: rotor stochastic FL certificates (not the deterministic main-repo systems). |
| D2 | Prove the tail (outside the box) by derivation, not the box interior by computation. |
| D3 | Explicit constants (theorem for the concrete V, concrete threshold), not the paper's existential form. |
| D4 | Proofs written interactively by Claude in Claude Code with `lake build` feedback; no API loop, no external prover model. |
| D5 | `sorry` allowed in intermediate states, each one recorded in a ledger; target is zero. |

## 3. Mathematical object (all constants fixed)

State `x = (r₁, p₁, p₂) ∈ ℝ³`, `r₁ = q₂ − q₁` (2π-periodic).

```
drift     ṙ₁ = p₂ − p₁,   ṗ₁ = sin r₁ − p₁,   ṗ₂ = −sin r₁
noise     κ₁ = 1  (second derivative in p₁ only)
LV        = ∂ᵣV·(p₂−p₁) + ∂ₚ₁V·(sin r₁−p₁) + ∂ₚ₂V·(−sin r₁) + ∂²ₚ₁V
H         = ½(p₁²+p₂²) + 1 − cos r₁
S(p₁)     = p₁⁴ + 20                       (R_ρ = 20, C28)
χ(z)      = b(z−1)/(b(z−1)+b(4−z)),  b(u)=exp(−1/u)·[u>0]
          = Real.smoothTransition((z−1)/3)   (Mathlib)
ρ         = χ(p₂²/S²)        ρ=0 for |p₂|≤S,  ρ=1 for |p₂|≥2S
I_j       = ρ/p₂ʲ  (j=1..4; 0 where ρ=0)
t₁ = cos r₁
t₂ = p₁ cos r₁
t₃ = p₁² cos r₁ + p₁ sin r₁ + cos² r₁
t₄ = −p₁³cos r₁ − 3p₁²sin r₁ − 3p₁cos²r₁ + p₁cos r₁ − ½ sin r₁ cos r₁ + 2 sin r₁
p̃₂       = p₂ − t₁I₁ − t₂I₂ − t₃I₃ + t₄I₄        (doc eq.17 sign convention)
V         = 1 + e^{θH} + A·ρ·p̃₂²·e^{θp̃₂²/2},   θ = 0.02,  A = 20
K         = {H ≤ R} ∪ {ρ < 1 ∧ p₁² ≤ M},  R = 2·10⁶,  M = 10⁴   (paper-shaped)
φ(v)      = c·v/(2 + log v),               c = 5·10⁻⁵
```

R, M, c are NOT the box-verification values (R=20, M=9, c*=0.27 of C36).
They come from the reconnaissance of §10: the C36 values are only valid on
the ±15 box, which never contains the fast branch (ρ=1 needs |p₂| ≥ 40).

## 4. Theorems to prove (Lean statements, informal)

- **T1 (negative drift outside K).** With `c = 5·10⁻⁵`, `R = 2·10⁶`,
  `M = 10⁴`:
  `∀ x, x ∉ K → LV x + φ (V x) ≤ 0`.
  No box threshold appears: K itself carves out everything that is not
  provable, so T1 is the full global drift condition and the ±15 box
  verification becomes a consistency check rather than a missing piece.
- **T2 (bounded on the exempt set).** Explicit `C`:
  `∀ x ∈ K, LV x ≤ C`. For N=2 the set K is bounded: `ρ < 1` forces
  `|p₂| < 2S(p₁) ≤ 2·(100⁴+20) ≈ 2·10⁸` when `p₁² ≤ 10⁴`, and `{H ≤ R}`
  is compact. So T2 is an explicit sup bound on a compact set; it still needs
  the gate-derivative bounds of `Gate.lean` on the part of K inside the
  transition band. The constant will be astronomically large (order
  `θ·exp(θ·H_max)` with `H_max ≈ 10¹⁵`) but explicit; only existence with an
  explicit value is required.
- **T0 (generator formula).** `LV x = ⟨explicit sympy-exported expression⟩`,
  proved by differentiation lemmas. This is the "L1" check and is used by
  T1/T2.

If the derivation only supports a larger threshold, the theorem states that
threshold and the box verification is extended to it (cheap numerically).

## 5. Proof skeleton (paper regions Ω₁/Ω₂/Ω₃ of CEP2015, constants explicit)

- **Slow region Ω₁** (`|p₂| ≤ S`, gate closed, ρ = 0): `V = 1 + e^{θH}`.
  H is conserved by the Hamiltonian part of the drift, so exactly
  `LH = 1 − p₁²` and `L e^{θH} = θ e^{θH} (1 − (1−θ) p₁²)`.
  Points of Ω₁ outside K have `p₁² > M = 10⁴`, hence
  `LV ≤ θ(1 − 10⁴(1−θ)) e^{θH} < −195·e^{θH}`, while
  `φ(V) ≤ c(1+e^{θH})/2`. Closes for every `H` with enormous margin
  (reconnaissance: ratio ≥ 0.22 already for `p₁² > 9`).
- **Fast region Ω₃** (`|p₂| ≥ 2S`, ρ = 1): core lemma
  `L p̃₂ = −1/(2p̃₂³) + R(x)`, `|R| ≤ C₃/|p̃₂|⁴` with explicit `C₃`
  (polynomial in |p₁|, controlled because `|p₂| ≥ 2(p₁⁴+20)`), then
  `L[p̃₂² e^{θp̃₂²/2}] ≤ −(θ/2)(1 − ε)·e^{θp̃₂²/2}/p̃₂² ...` and finally
  `≤ −c V/(2+log V)` using `log V ≈ θp̃₂²/2 + log A`.
- **Transition band Ω₂** (`S < |p₂| < 2S`): the part with `p₁² ≤ M` is in K
  (only T2 needed); the part with `p₁² > M = 10⁴` uses explicit bounds on
  `|χ'|, |χ''|` (own lemmas on `smoothTransition`, Mathlib has smoothness
  but no explicit derivative bounds) and the damping of `e^{θH}`. The
  dangerous term is `A p̃₂² e^{θp̃₂²/2} · Lρ` with `Lρ ≈ 8χ'(z)·z` (the gate
  opens as p₁ is damped); it is beaten only by `θ(1−θ)p₁² e^{θp₁²/2}`, which
  is why M must be ≈ 67² (reconnaissance §10) and not 9.
- **T2**: K is compact (§4): `p₁² ≤ 10⁴`, `|p₂| ≤ 2·10⁸` on the gate part
  and `H ≤ 2·10⁶` on the energy part. Bound every term of the explicit LV formula
  (T0) by its sup over that box using the gate-derivative bounds from
  `Gate.lean`; C is the sum. Numerical reconnaissance gives the actual sup
  so the crude bound can be sanity-checked (it must dominate it).

## 6. Engineering

Lean project `~/lean/rotor-fl` (elan, Lean 4 stable, Mathlib via
`lake new … math`, cache from `lake exe cache get`). Modules:

| file | content |
|---|---|
| `RotorFL/Defs.lean` | state, drift, H, S, χ, ρ, I_j, t_j, p̃₂, V, generator L, K, φ |
| `RotorFL/Gate.lean` | χ = smoothTransition composition; χ∈[0,1]; χ=0/1 regions; explicit `|χ'|`, `|χ''|` bounds |
| `RotorFL/Generator.lean` | T0: LV explicit formula (per region: ρ=0, ρ=1, general) |
| `RotorFL/Slow.lean` | Ω₁ lemma |
| `RotorFL/Fast.lean` | Ω₃ lemmas (L p̃₂ expansion, remainder bound, exponential weight) |
| `RotorFL/Transition.lean` | Ω₂ lemma |
| `RotorFL/Main.lean` | T1, T2 |

Python side (in `~/ALFD-rotor`):

- `utils/fl_lean.py`: sympy → Lean 4 term printer (Real literals as
  rationals, `Real.exp/sin/cos/log`, powers), export of V, LV, p̃₂ and the
  per-region residuals as Lean `def`s + a `#check`-able file.
- `repro/lean_recon_rotor2.py`: before each lemma is attempted, compute the
  claimed constant numerically (sup of remainder ratios over the region on a
  dense grid + SHGO), print it with a 30 % margin; the Lean lemma states the
  margined constant. Prevents attempting false bounds.
- `docs/LEAN-SORRY-LEDGER.md`: one line per remaining `sorry` (lemma, reason).

## 7. Order of work

1. Environment (elan, Mathlib cache) — in progress.
2. `Defs` + `Gate` (smoothTransition reuse; derivative bounds).
3. `Generator` (T0) with sympy export cross-check (`fl_lean.py`).
4. Slow region. 5. Fast region (risk point). 6. Transition. 7. T2. 8. Main.
9. CONCLUSIONS entry C44; ledger; report.

## 8. Acceptance

- `lake build` succeeds; T1, T2 stated exactly as in §4 with the numeric
  constants `c = 5·10⁻⁵`, `R = 2·10⁶`, `M = 10⁴`.
- Zero `sorry`, or every `sorry` listed in the ledger with the blocking reason.
- Every numeric constant in a Lean lemma has a matching reconnaissance line.
- The box verification (C36 machinery) re-run with the K of §3 as a
  consistency check: it must report zero violations outside K.

## 9. Risks

- The fast-region proof must hold for all `|p₁| ≤ (|p₂|/2 − 20)^{1/4}`, so the
  remainder constants are polynomial in `p₁`. Margins were chosen for proofs
  with crude explicit bounds: fast region ratio ≥ 8.49·10⁻⁵ vs `c = 5·10⁻⁵`
  (70 %); the triangle-inequality bound `|c₄| ≤ 13 + 22a + 19a² + 6a³ + a⁴`
  (`a = |p₁|`, generated by `utils/fl_lean.py`) overshoots the true sup by
  2.3× near `a ≈ 4`, which is why the fast-branch threshold is `|p₂| ≥ 2000`
  (`R = 2·10⁶`) rather than 500: then `Aθ·|c₄|·u ≤ 0.21 < 0.2 + damping` for
  every `a`; transition band crossover at `|p₁| ≈ 67.3` vs `M = 10⁴`
  (`|p₁| ≥ 100`, factor `e^{θ(100²−67.3²)/2} ≈ e^{55}` of slack, so
  `|χ'|, |χ''| ≤ 10⁶` would still suffice). If a bound still cannot be
  closed, raise R or M (T2's C grows, nothing else changes).
- Explicit `χ'`, `χ''` bounds on `smoothTransition` are new lemmas; if hard,
  fall back to a cruder bound obtained from monotonicity + mean value theorem.
- Mathlib differentiation of nested `exp(θ·(…)²)` compositions is verbose;
  mitigate by structuring V as named smooth pieces with `HasDerivAt` lemmas.

## 10. Reconnaissance results (2026-09-07, `repro/lean_recon_rotor2.py`)

`ratio(x) := −LV·(2+log V)/V`; the drift condition holds at x iff
`ratio(x) ≥ c`. Scaled evaluation (no overflow), spot checks with mpmath at
40 digits on the unscaled V, LV.

| region | condition | min ratio | where |
|---|---|---|---|
| Ω₁ slow (ρ=0) | p₁² > 9, |p₂| ≤ S | +0.224 | p₁ = 3, p₂ = 0 |
| Ω₂ transition | 3 < |p₁| ≤ 67, S < |p₂| < 2S | **−6·10⁶ … −5·10¹⁷** (LV > 0) | |p₂|/S ≈ 1.03–1.8, r₁ ≈ −π/2 |
| Ω₂ transition | |p₁| ≥ 67.5 | ≥ +2.5·10¹⁴ | |p₂| = S |
| Ω₃ fast (ρ=1) | 50 < |p₂| < 200, |p₁| = (|p₂|/2−20)^{1/4} | **−2.2·10⁻⁵** (LV > 0, mpmath-confirmed) | p₂ = 80, p₁ = 2.11 |
| Ω₃ fast | |p₂| ≥ 500 | ≥ +8.49·10⁻⁵ | p₁ ≈ 0, r₁ ≈ ±π |
| Ω₃ fast | |p₂| → ∞ | → θ²/4·(1−ε) ≈ 8.58·10⁻⁵ | |

Consequences:

1. The C36 statement "rotor2 certificate FEASIBLE, c* = 0.27, K with R=20,
   M=9" is a statement about the ±15 box only. Globally that K is wrong: the
   certificate violates the drift condition in the transition band for every
   3 < |p₁| ≲ 67 and in the fast region for 50 < |p₂| < 200.
2. The certificate IS valid globally with the paper's quantifier order
   (M chosen last): `M = 10⁴`, `R = 2·10⁶`, `c = 5·10⁻⁵`. These are the
   constants T1/T2 use.
3. The true drift rate is `c ≈ θ²/4`, four orders of magnitude below the
   box-measured `c*`; the box never contains the fast branch (needs
   `|p₂| ≥ 2S ≥ 40`), so its `c*` measures only the slow-region damping.
