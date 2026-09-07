/-
# The averaged drift of `p̃f` as a series in `u = 1/p₂`

    L p̃f = c₃ u³ + c₄ u⁴ + c₅ u⁵,   c₃ = −1/2

This is an exact rational identity (no averaging left): the counterterms
t₁..t₄ cancel every term of order u⁰..u², and the u³ coefficient is the
non-removable mean −1/2 (CONCLUSIONS C20/C21).
-/
import RotorFl.PtDeriv

namespace RotorFl

open Real

/-- `L p̃f` written out with the explicit slice derivatives. -/
noncomputable def Lptf (r p₁ p₂ : ℝ) : ℝ :=
  ptf_r r p₁ p₂ * (p₂ - p₁) + ptf_p₁ r p₁ p₂ * (sin r - p₁)
    + ptf_p₂ r p₁ p₂ * (-sin r) + ptf_p₁p₁ r p₁ p₂

theorem Lptf_series (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    Lptf r p₁ p₂ = c3 r * (1 / p₂) ^ 3 + c4 r p₁ * (1 / p₂) ^ 4 + c5 r p₁ * (1 / p₂) ^ 5 := by
  unfold Lptf ptf_r ptf_p₁ ptf_p₂ ptf_p₁p₁ t₁ t₂ t₃ t₄ t1_r t2_r t3_r t4_r
    t1_p₁ t2_p₁ t3_p₁ t4_p₁ t1_p₁p₁ t2_p₁p₁ t3_p₁p₁ t4_p₁p₁ c3 c4 c5
  field_simp
  ring

lemma c3_eq (r : ℝ) : c3 r = -1 / 2 := by
  unfold c3
  linear_combination (-(1 : ℝ) / 2) * sin_sq_add_cos_sq r

/-- Hence `L p̃f = −u³/2 + c₄ u⁴ + c₅ u⁵`. -/
theorem Lptf_eq (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    Lptf r p₁ p₂ = -(1 / p₂) ^ 3 / 2 + c4 r p₁ * (1 / p₂) ^ 4 + c5 r p₁ * (1 / p₂) ^ 5 := by
  rw [Lptf_series r p₁ p₂ h₂, c3_eq]
  ring

end RotorFl
