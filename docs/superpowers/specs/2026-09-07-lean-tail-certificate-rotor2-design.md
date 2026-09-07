# Lean formalization of the rotor2 certificate tail (design)

Date: 2026-09-07. Status: approved in conversation (user decisions D1–D5 below).

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
K         = {H ≤ 20} ∪ {ρ < 1 ∧ p₁² ≤ 9}          (paper-shaped, M = 9)
φ(v)      = c·v/(2 + log v)
```

## 4. Theorems to prove (Lean statements, informal)

- **T1 (tail negative drift).** For an explicit `c > 0` (target `c ≥ 0.1`)
  and explicit threshold `P₀` (target `P₀ ≤ 15`):
  `∀ x, max |p₁| |p₂| ≥ P₀ → x ∉ K → LV x + φ (V x) ≤ 0`.
- **T2 (bounded on the exempt set).** Explicit `C`:
  `∀ x ∈ K, LV x ≤ C`. For N=2 the set K is bounded: `ρ < 1` forces
  `|p₂| < 2S(p₁) ≤ 2·(81+20) = 202` when `p₁² ≤ 9`, and `{H ≤ 20}` is
  compact. So T2 is an explicit sup bound on a compact set; it still needs
  the gate-derivative bounds of `Gate.lean` on the part of K inside the
  transition band. The constant will be crude (order `θ·exp(θ·H_max)`); that
  is acceptable, only existence with an explicit value is required.
- **T0 (generator formula).** `LV x = ⟨explicit sympy-exported expression⟩`,
  proved by differentiation lemmas. This is the "L1" check and is used by
  T1/T2.

If the derivation only supports a larger threshold, the theorem states that
threshold and the box verification is extended to it (cheap numerically).

## 5. Proof skeleton (paper regions Ω₁/Ω₂/Ω₃ of CEP2015, constants explicit)

- **Slow region Ω₁** (`|p₂| ≤ S`, gate closed, ρ = 0): `V = 1 + e^{θH}`.
  H is conserved by the Hamiltonian part of the drift, so exactly
  `LH = 1 − p₁²` and `L e^{θH} = θ e^{θH} (1 − (1−θ) p₁²)`.
  Points of Ω₁ outside K have `p₁² > 9`, hence
  `LV ≤ θ(1 − 9(1−θ)) e^{θH} = −0.1564·e^{θH}`, while
  `φ(V) ≤ c(1+e^{θH})/2`. With `c = 0.1` this closes for every `H`
  (no threshold needed in this region).
- **Fast region Ω₃** (`|p₂| ≥ 2S`, ρ = 1): core lemma
  `L p̃₂ = −1/(2p̃₂³) + R(x)`, `|R| ≤ C₃/|p̃₂|⁴` with explicit `C₃`
  (polynomial in |p₁|, controlled because `|p₂| ≥ 2(p₁⁴+20)`), then
  `L[p̃₂² e^{θp̃₂²/2}] ≤ −(θ/2)(1 − ε)·e^{θp̃₂²/2}/p̃₂² ...` and finally
  `≤ −c V/(2+log V)` using `log V ≈ θp̃₂²/2 + log A`.
- **Transition band Ω₂** (`S < |p₂| < 2S`): the part with `p₁² ≤ 9` is in K
  (only T2 needed); the part with `p₁² > 9` uses explicit bounds on
  `|χ'|, |χ''|` (own lemmas on `smoothTransition`, Mathlib has smoothness
  but no explicit derivative bounds) and the damping of `e^{θH}`.
- **T2**: K is compact (§4): `p₁² ≤ 9`, `|p₂| ≤ 202` on the gate part and
  `H ≤ 20` on the energy part. Bound every term of the explicit LV formula
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

- `lake build` succeeds; T1, T2 stated exactly as in §4 with numeric constants.
- Zero `sorry`, or every `sorry` listed in the ledger with the blocking reason.
- Every numeric constant in a Lean lemma has a matching reconnaissance line.
- Threshold `P₀` and `c` reported; if `P₀ > 15`, box verification re-run to `±P₀`.

## 9. Risks

- Fast-region remainder constants may push `P₀` above 15 → accept larger
  threshold (§4).
- Explicit `χ'`, `χ''` bounds on `smoothTransition` are new lemmas; if hard,
  fall back to a cruder bound obtained from monotonicity + mean value theorem.
- Mathlib differentiation of nested `exp(θ·(…)²)` compositions is verbose;
  mitigate by structuring V as named smooth pieces with `HasDerivAt` lemmas.
