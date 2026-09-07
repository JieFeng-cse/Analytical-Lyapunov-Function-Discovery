/-
# Generator of the fast-branch certificate

    Vf = Vs + Ff(p̃f),   Ff(y) = A y² e^{θ y²/2}

    L Vf = θ e^{θH} (1 − (1−θ) p₁²) + Ff'(p̃f)·L p̃f + Ff''(p̃f)·(∂ₚ₁ p̃f)²
-/
import RotorFl.Series
import RotorFl.Chain

namespace RotorFl

open Real

/-- Fast-branch profile `Ff(y) = A y² e^{θy²/2}`. -/
noncomputable def Ff (y : ℝ) : ℝ := A * y ^ 2 * exp (θ / 2 * y ^ 2)
/-- `Ff' (y) = A (2y + θy³) e^{θy²/2}`. -/
noncomputable def Ff' (y : ℝ) : ℝ := A * (2 * y + θ * y ^ 3) * exp (θ / 2 * y ^ 2)
/-- `Ff'' (y) = A (2 + 5θy² + θ²y⁴) e^{θy²/2}`. -/
noncomputable def Ff'' (y : ℝ) : ℝ := A * (2 + 5 * θ * y ^ 2 + θ ^ 2 * y ^ 4) * exp (θ / 2 * y ^ 2)

lemma hasDerivAt_Ff (y : ℝ) : HasDerivAt Ff (Ff' y) y := by
  unfold Ff Ff'
  deriv_tac (((hasDerivAt_const y A).mul (hasDerivAt_sq y)).mul
    (((hasDerivAt_sq y).const_mul (θ / 2)).exp))

lemma hasDerivAt_Ff' (y : ℝ) : HasDerivAt Ff' (Ff'' y) y := by
  unfold Ff' Ff''
  deriv_tac (((hasDerivAt_const y A).mul
      (((hasDerivAt_id' y).const_mul 2).add (((hasDerivAt_id' y).pow 3).const_mul θ))).mul
    (((hasDerivAt_sq y).const_mul (θ / 2)).exp))

/-- `L Vf` in closed form (for `p₂ ≠ 0`). -/
theorem L_Vf (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    L Vf r p₁ p₂ = θ * exp (θ * H r p₁ p₂) * (1 - (1 - θ) * p₁ ^ 2)
      + (Ff' (ptf r p₁ p₂) * Lptf r p₁ p₂ + Ff'' (ptf r p₁ p₂) * (ptf_p₁ r p₁ p₂) ^ 2) := by
  have hV : Vf = fun a b c => Vs a b c + Ff (ptf a b c) := by
    funext a b c
    unfold Vf Vs Ff
    ring
  rw [hV, L_add (hasDerivAt_Vs_r r p₁ p₂) (fun t => hasDerivAt_Vs_p₁ r t p₂)
        (hasDerivAt_Vs_p₂ r p₁ p₂) (hasDerivAt_deriv_Vs_p₁ r p₁ p₂)
        ((hasDerivAt_Ff _).comp r (hasDerivAt_ptf_r r p₁ p₂))
        (fun t => (hasDerivAt_Ff _).comp t (hasDerivAt_ptf_p₁ r t p₂))
        ((hasDerivAt_Ff _).comp p₂ (hasDerivAt_ptf_p₂ r p₁ p₂ h₂))
        (((hasDerivAt_Ff' _).comp p₁ (hasDerivAt_ptf_p₁ r p₁ p₂)).mul (hasDerivAt_ptf_p₁p₁ r p₁ p₂))
        rfl rfl]
  unfold Lptf
  simp only [Function.comp_apply, Pi.mul_apply]
  ring

end RotorFl
