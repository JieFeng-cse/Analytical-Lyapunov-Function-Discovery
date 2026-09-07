/-
# Rotor-chain N=2 Foster–Lyapunov certificate: definitions

Spec: ALFD-rotor/docs/superpowers/specs/2026-09-07-lean-tail-certificate-rotor2-design.md

State `(r, p₁, p₂) ∈ ℝ³`, `r = q₂ − q₁` the bond angle.
Dynamics (one bath on rotor 1, noise κ₁ = 1 in `p₁`):

    ṙ = p₂ − p₁,   ṗ₁ = sin r − p₁,   ṗ₂ = −sin r,   dp₁ += √2 dB

Generator on C² functions:

    L V = ∂ᵣV·(p₂−p₁) + ∂ₚ₁V·(sin r − p₁) + ∂ₚ₂V·(−sin r) + ∂²ₚ₁V

Certificate (machine-built, CONCLUSIONS C36; constants of C44):

    V = 1 + e^{θH} + A·ρ·p̃₂²·e^{θ p̃₂²/2}
    ρ = χ(p₂²/S²), S = p₁⁴ + 20, χ(z) = smoothTransition((z−1)/3)
    p̃₂ = p₂ − ρ·(t₁/p₂ + t₂/p₂² + t₃/p₂³ − t₄/p₂⁴)
-/
import Mathlib.Analysis.SpecialFunctions.SmoothTransition
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

namespace RotorFl

open Real

/-! ## Constants -/

/-- Exponential weight θ = 0.02. -/
noncomputable def θ : ℝ := 1 / 50
/-- Fast-branch amplitude. -/
noncomputable def A : ℝ := 20
/-- Gate offset R_ρ (C28: 20, not the draft's 4). -/
noncomputable def Rρ : ℝ := 20
/-- Drift rate c = 5·10⁻⁵ (C44: true rate is ≈ θ²/4; margin for crude bounds). -/
noncomputable def cRate : ℝ := 1 / 20000
/-- Energy exemption R = 1.25·10⁵ (|p₂| ≥ 500 outside). -/
noncomputable def Rex : ℝ := 125000
/-- Endpoint-momentum exemption M = 10⁴ (|p₁| ≥ 100 outside the gate part). -/
noncomputable def Mex : ℝ := 10000

/-! ## Dynamics and generator -/

/-- Energy `H = (p₁² + p₂²)/2 + 1 − cos r`. -/
noncomputable def H (r p₁ p₂ : ℝ) : ℝ := (p₁ ^ 2 + p₂ ^ 2) / 2 + 1 - cos r

/-- The stochastic generator, written with one-variable `deriv`s of the slices.
For a C² function this is the classical `∇V·f + κ₁ ∂²ₚ₁V` with κ₁ = 1. -/
noncomputable def L (V : ℝ → ℝ → ℝ → ℝ) (r p₁ p₂ : ℝ) : ℝ :=
    deriv (fun t => V t p₁ p₂) r * (p₂ - p₁)
  + deriv (fun t => V r t p₂) p₁ * (sin r - p₁)
  + deriv (fun t => V r p₁ t) p₂ * (-sin r)
  + deriv (deriv (fun t => V r t p₂)) p₁

/-! ## Gate -/

/-- Gate width `S = p₁⁴ + R_ρ`. -/
noncomputable def S (p₁ : ℝ) : ℝ := p₁ ^ 4 + Rρ

/-- `χ(z) = b(z−1)/(b(z−1)+b(4−z))` with `b(u) = exp(−1/u)·[u>0]`; this is
Mathlib's `smoothTransition` rescaled: `χ z = smoothTransition ((z − 1)/3)`.
`χ = 0` for `z ≤ 1`, `χ = 1` for `z ≥ 4`. -/
noncomputable def χ (z : ℝ) : ℝ := smoothTransition ((z - 1) / 3)

/-- Gate `ρ = χ(p₂²/S²)`: 0 for `|p₂| ≤ S`, 1 for `|p₂| ≥ 2S`. -/
noncomputable def ρ (p₁ p₂ : ℝ) : ℝ := χ (p₂ ^ 2 / (S p₁) ^ 2)

/-! ## Counterterms (benchmark doc eq. 17) -/

noncomputable def t₁ (r p₁ : ℝ) : ℝ := cos r
noncomputable def t₂ (r p₁ : ℝ) : ℝ := p₁ * cos r
noncomputable def t₃ (r p₁ : ℝ) : ℝ := p₁ ^ 2 * cos r + p₁ * sin r + (cos r) ^ 2
noncomputable def t₄ (r p₁ : ℝ) : ℝ :=
  -p₁ ^ 3 * cos r - 3 * p₁ ^ 2 * sin r - 3 * p₁ * (cos r) ^ 2 + p₁ * cos r
  - (1 / 2) * sin r * cos r + 2 * sin r

/-- Corrected fast momentum `p̃₂`. Division by `p₂` is harmless: `ρ = 0` on a
neighbourhood of `p₂ = 0` (indeed for `|p₂| ≤ S ≥ 20`). -/
noncomputable def pt (r p₁ p₂ : ℝ) : ℝ :=
  p₂ - ρ p₁ p₂ * (t₁ r p₁ / p₂ + t₂ r p₁ / p₂ ^ 2 + t₃ r p₁ / p₂ ^ 3 - t₄ r p₁ / p₂ ^ 4)

/-! ## Certificate, exempt set, drift function -/

/-- The certificate `V`. -/
noncomputable def V (r p₁ p₂ : ℝ) : ℝ :=
  1 + exp (θ * H r p₁ p₂) + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * exp (θ / 2 * (pt r p₁ p₂) ^ 2)

/-- Paper-shaped exempt set `K = {H ≤ R} ∪ {ρ < 1 ∧ p₁² ≤ M}` (C28, C44). -/
def K (r p₁ p₂ : ℝ) : Prop := H r p₁ p₂ ≤ Rex ∨ (ρ p₁ p₂ < 1 ∧ p₁ ^ 2 ≤ Mex)

/-- Drift function `φ(v) = c·v/(2 + log v)` (α = 1). -/
noncomputable def φ (v : ℝ) : ℝ := cRate * v / (2 + log v)

/-! ## Main statements (proved in later files) -/

/-- **T1** (negative drift outside K): `L V + φ(V) ≤ 0` on the complement of K. -/
def T1_statement : Prop :=
  ∀ r p₁ p₂ : ℝ, ¬ K r p₁ p₂ → L V r p₁ p₂ + φ (V r p₁ p₂) ≤ 0

/-- **T2** (bounded on K): `L V ≤ C` on K, for an explicit `C`. -/
def T2_statement (C : ℝ) : Prop :=
  ∀ r p₁ p₂ : ℝ, K r p₁ p₂ → L V r p₁ p₂ ≤ C

end RotorFl
