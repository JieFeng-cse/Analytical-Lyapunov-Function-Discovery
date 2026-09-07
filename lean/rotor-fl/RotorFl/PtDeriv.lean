/-
# Slice derivatives of `p̃f` (gate fully open)

    p̃f = p₂ − t₁/p₂ − t₂/p₂² − t₃/p₂³ + t₄/p₂⁴

Explicit first derivatives in r, p₁, p₂, and the second derivative in p₁,
as `HasDerivAt` facts (the p₂-derivative needs `p₂ ≠ 0`).
-/
import RotorFl.FastDeriv
import RotorFl.Fast

namespace RotorFl

open Real

/-- ∂ᵣ p̃f -/
noncomputable def ptf_r (r p₁ p₂ : ℝ) : ℝ :=
  -(t1_r r p₁ / p₂) - t2_r r p₁ / p₂ ^ 2 - t3_r r p₁ / p₂ ^ 3 + t4_r r p₁ / p₂ ^ 4

/-- ∂ₚ₁ p̃f -/
noncomputable def ptf_p₁ (r p₁ p₂ : ℝ) : ℝ :=
  -(t1_p₁ r p₁ / p₂) - t2_p₁ r p₁ / p₂ ^ 2 - t3_p₁ r p₁ / p₂ ^ 3 + t4_p₁ r p₁ / p₂ ^ 4

/-- ∂²ₚ₁ p̃f -/
noncomputable def ptf_p₁p₁ (r p₁ p₂ : ℝ) : ℝ :=
  -(t1_p₁p₁ r p₁ / p₂) - t2_p₁p₁ r p₁ / p₂ ^ 2 - t3_p₁p₁ r p₁ / p₂ ^ 3 + t4_p₁p₁ r p₁ / p₂ ^ 4

/-- ∂ₚ₂ p̃f -/
noncomputable def ptf_p₂ (r p₁ p₂ : ℝ) : ℝ :=
  1 + t₁ r p₁ / p₂ ^ 2 + 2 * t₂ r p₁ / p₂ ^ 3 + 3 * t₃ r p₁ / p₂ ^ 4 - 4 * t₄ r p₁ / p₂ ^ 5

lemma hasDerivAt_ptf_r (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => ptf t p₁ p₂) (ptf_r r p₁ p₂) r := by
  unfold ptf ptf_r
  have h := ((((hasDerivAt_const r p₂).sub ((hasDerivAt_t1_r r p₁).div_const p₂)).sub
    ((hasDerivAt_t2_r r p₁).div_const (p₂ ^ 2))).sub
    ((hasDerivAt_t3_r r p₁).div_const (p₂ ^ 3))).add ((hasDerivAt_t4_r r p₁).div_const (p₂ ^ 4))
  refine h.congr_deriv ?_
  ring

lemma hasDerivAt_ptf_p₁ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => ptf r t p₂) (ptf_p₁ r p₁ p₂) p₁ := by
  unfold ptf ptf_p₁
  have h := ((((hasDerivAt_const p₁ p₂).sub ((hasDerivAt_t1_p₁ r p₁).div_const p₂)).sub
    ((hasDerivAt_t2_p₁ r p₁).div_const (p₂ ^ 2))).sub
    ((hasDerivAt_t3_p₁ r p₁).div_const (p₂ ^ 3))).add ((hasDerivAt_t4_p₁ r p₁).div_const (p₂ ^ 4))
  refine h.congr_deriv ?_
  ring

lemma hasDerivAt_ptf_p₁p₁ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => ptf_p₁ r t p₂) (ptf_p₁p₁ r p₁ p₂) p₁ := by
  unfold ptf_p₁ ptf_p₁p₁
  have h := ((((hasDerivAt_t1_p₁p₁ r p₁).div_const p₂).neg.sub
    ((hasDerivAt_t2_p₁p₁ r p₁).div_const (p₂ ^ 2))).sub
    ((hasDerivAt_t3_p₁p₁ r p₁).div_const (p₂ ^ 3))).add ((hasDerivAt_t4_p₁p₁ r p₁).div_const (p₂ ^ 4))
  refine h.congr_deriv ?_
  ring

/-- `HasDerivAt (fun t => c / t ^ (k+1)) (-((k+1) * c) / t ^ (k+2)) t` for `t ≠ 0`. -/
lemma hasDerivAt_const_div_pow (c t : ℝ) (k : ℕ) (ht : t ≠ 0) :
    HasDerivAt (fun x => c / x ^ (k + 1)) (-((k + 1) * c) / t ^ (k + 2)) t := by
  have h := ((hasDerivAt_pow (k + 1) t).inv (pow_ne_zero (k + 1) ht)).const_mul c
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun x => ?_)).congr_deriv ?_
  · simp [div_eq_mul_inv]
  · simp only [Nat.add_sub_cancel]
    field_simp
    push_cast
    ring

lemma hasDerivAt_ptf_p₂ (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    HasDerivAt (fun t => ptf r p₁ t) (ptf_p₂ r p₁ p₂) p₂ := by
  unfold ptf ptf_p₂
  have h := ((((hasDerivAt_id' p₂).sub (hasDerivAt_const_div_pow (t₁ r p₁) p₂ 0 h₂)).sub
    (hasDerivAt_const_div_pow (t₂ r p₁) p₂ 1 h₂)).sub
    (hasDerivAt_const_div_pow (t₃ r p₁) p₂ 2 h₂)).add (hasDerivAt_const_div_pow (t₄ r p₁) p₂ 3 h₂)
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun x => ?_)).congr_deriv ?_
  · simp
  · push_cast
    ring

end RotorFl
