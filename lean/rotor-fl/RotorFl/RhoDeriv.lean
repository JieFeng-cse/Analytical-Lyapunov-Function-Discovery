/-
# Slice derivatives of the gate `ρ`

`ρ p₁ p₂ = ψ(ζ)`, `ζ = (p₂²/S² − 1)/3`, `S = p₁⁴ + 20`, `ψ = smoothTransition`.

    ∂ₚ₁ζ = −8 p₁³ p₂² / (3 S³),   ∂ₚ₂ζ = 2 p₂ / (3 S²),
    ∂²ₚ₁ζ = −(8 p₂²/3) (3 p₁²/S³ − 12 p₁⁶/S⁴),
    ∂ₚ₁ρ = ψ'(ζ) ∂ₚ₁ζ,  ∂ₚ₂ρ = ψ'(ζ) ∂ₚ₂ζ,  ∂²ₚ₁ρ = ψ''(ζ) (∂ₚ₁ζ)² + ψ'(ζ) ∂²ₚ₁ζ.
-/
import RotorFl.Gate
import RotorFl.Slow

namespace RotorFl

open Real

noncomputable def ζ (p₁ p₂ : ℝ) : ℝ := (p₂ ^ 2 / (S p₁) ^ 2 - 1) / 3
noncomputable def ζ_p₁ (p₁ p₂ : ℝ) : ℝ := -(8 * p₁ ^ 3 * p₂ ^ 2) / (3 * (S p₁) ^ 3)
noncomputable def ζ_p₂ (p₁ p₂ : ℝ) : ℝ := 2 * p₂ / (3 * (S p₁) ^ 2)
noncomputable def ζ_p₁p₁ (p₁ p₂ : ℝ) : ℝ :=
  -(8 * p₂ ^ 2) / 3 * (3 * p₁ ^ 2 / (S p₁) ^ 3 - 12 * p₁ ^ 6 / (S p₁) ^ 4)

lemma ρ_eq_ψ (p₁ p₂ : ℝ) : ρ p₁ p₂ = smoothTransition (ζ p₁ p₂) := rfl

lemma hasDerivAt_S (p₁ : ℝ) : HasDerivAt S (4 * p₁ ^ 3) p₁ := by
  unfold S
  have h := (hasDerivAt_pow 4 p₁).add_const Rρ
  refine h.congr_deriv ?_
  push_cast; ring

lemma hasDerivAt_ζ_p₂ (p₁ p₂ : ℝ) : HasDerivAt (fun t => ζ p₁ t) (ζ_p₂ p₁ p₂) p₂ := by
  unfold ζ ζ_p₂
  have h := (((hasDerivAt_sq p₂).div_const ((S p₁) ^ 2)).sub_const 1).div_const 3
  refine h.congr_deriv ?_
  ring

lemma hasDerivAt_ζ_p₁ (p₁ p₂ : ℝ) : HasDerivAt (fun t => ζ t p₂) (ζ_p₁ p₁ p₂) p₁ := by
  unfold ζ ζ_p₁
  have hS := S_pos p₁
  have h1 : HasDerivAt (fun t => (S t) ^ 2) (2 * S p₁ * (4 * p₁ ^ 3)) p₁ := by
    have := (hasDerivAt_S p₁).pow 2
    refine this.congr_deriv ?_
    push_cast; ring
  have h2 := ((h1.inv (by positivity)).const_mul (p₂ ^ 2)).sub_const 1 |>.div_const 3
  refine (h2.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => ?_)).congr_deriv ?_
  · simp [div_eq_mul_inv]
  · field_simp
    ring

lemma hasDerivAt_ζ_p₁p₁ (p₁ p₂ : ℝ) :
    HasDerivAt (fun t => ζ_p₁ t p₂) (ζ_p₁p₁ p₁ p₂) p₁ := by
  unfold ζ_p₁ ζ_p₁p₁
  have hS := S_pos p₁
  have hnum : HasDerivAt (fun t : ℝ => -(8 * t ^ 3 * p₂ ^ 2)) (-(24 * p₁ ^ 2 * p₂ ^ 2)) p₁ := by
    have := (((hasDerivAt_pow 3 p₁).const_mul 8).mul_const (p₂ ^ 2)).neg
    refine this.congr_deriv ?_
    push_cast; ring
  have hden : HasDerivAt (fun t => 3 * (S t) ^ 3) (3 * (3 * (S p₁) ^ 2 * (4 * p₁ ^ 3))) p₁ := by
    have := ((hasDerivAt_S p₁).pow 3).const_mul 3
    refine this.congr_deriv ?_
    push_cast; ring
  have h := hnum.div hden (by positivity)
  refine h.congr_deriv ?_
  field_simp
  ring

/-! ## `ρ` -/

noncomputable def ρ_p₁ (p₁ p₂ : ℝ) : ℝ := ψ' (ζ p₁ p₂) * ζ_p₁ p₁ p₂
noncomputable def ρ_p₂ (p₁ p₂ : ℝ) : ℝ := ψ' (ζ p₁ p₂) * ζ_p₂ p₁ p₂
noncomputable def ρ_p₁p₁ (p₁ p₂ : ℝ) : ℝ :=
  ψ'' (ζ p₁ p₂) * (ζ_p₁ p₁ p₂) ^ 2 + ψ' (ζ p₁ p₂) * ζ_p₁p₁ p₁ p₂

lemma hasDerivAt_ρ_p₁ (p₁ p₂ : ℝ) : HasDerivAt (fun t => ρ t p₂) (ρ_p₁ p₁ p₂) p₁ :=
  (hasDerivAt_smoothTransition _).comp p₁ (hasDerivAt_ζ_p₁ p₁ p₂)

lemma hasDerivAt_ρ_p₂ (p₁ p₂ : ℝ) : HasDerivAt (fun t => ρ p₁ t) (ρ_p₂ p₁ p₂) p₂ :=
  (hasDerivAt_smoothTransition _).comp p₂ (hasDerivAt_ζ_p₂ p₁ p₂)

lemma hasDerivAt_ρ_p₁_p₁ (p₁ p₂ : ℝ) :
    HasDerivAt (fun t => ρ_p₁ t p₂) (ρ_p₁p₁ p₁ p₂) p₁ := by
  have h := ((hasDerivAt_ψ' _).comp p₁ (hasDerivAt_ζ_p₁ p₁ p₂)).mul (hasDerivAt_ζ_p₁p₁ p₁ p₂)
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => rfl)).congr_deriv ?_
  unfold ρ_p₁p₁
  simp only [Function.comp_apply]
  ring

/-! ## Vanishing of the gate derivatives where the gate is saturated -/

lemma ζ_le_zero {p₁ p₂ : ℝ} (h : |p₂| ≤ S p₁) : ζ p₁ p₂ ≤ 0 := by
  unfold ζ
  have hS := S_pos p₁
  have h2 : p₂ ^ 2 ≤ (S p₁) ^ 2 := by nlinarith [abs_nonneg p₂, sq_abs p₂, h]
  have : p₂ ^ 2 / (S p₁) ^ 2 ≤ 1 := by rw [div_le_one (by positivity)]; exact h2
  linarith

lemma one_le_ζ {p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : 1 ≤ ζ p₁ p₂ := by
  unfold ζ
  have hS := S_pos p₁
  have h2 : 4 * (S p₁) ^ 2 ≤ p₂ ^ 2 := by nlinarith [abs_nonneg p₂, sq_abs p₂, h]
  have : 4 ≤ p₂ ^ 2 / (S p₁) ^ 2 := by rw [le_div_iff₀ (by positivity)]; linarith
  linarith

lemma ρ_p₁_eq_zero_of_open {p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : ρ_p₁ p₁ p₂ = 0 := by
  unfold ρ_p₁; rw [ψ'_eq_zero_of_one_le (one_le_ζ h), zero_mul]

lemma ρ_p₂_eq_zero_of_open {p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : ρ_p₂ p₁ p₂ = 0 := by
  unfold ρ_p₂; rw [ψ'_eq_zero_of_one_le (one_le_ζ h), zero_mul]

lemma ρ_p₁p₁_eq_zero_of_open {p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : ρ_p₁p₁ p₁ p₂ = 0 := by
  unfold ρ_p₁p₁
  rw [ψ'_eq_zero_of_one_le (one_le_ζ h), ψ''_eq_zero_of_one_le (one_le_ζ h)]; ring

/-! ## Bounds in the band `|p₂| ≤ 2S` -/

section band
variable {p₁ p₂ : ℝ} (hb : |p₂| ≤ 2 * S p₁)
include hb

lemma sq_p₂_le : p₂ ^ 2 ≤ 4 * (S p₁) ^ 2 := by
  have hS := S_pos p₁
  nlinarith [abs_nonneg p₂, sq_abs p₂, hb]

lemma abs_ζ_p₁_le : |ζ_p₁ p₁ p₂| ≤ 32 * |p₁| ^ 3 / (3 * S p₁) := by
  unfold ζ_p₁
  have hS := S_pos p₁
  have h2 := sq_p₂_le hb
  rw [abs_div, abs_neg, abs_of_pos (by positivity : (0:ℝ) < 3 * (S p₁) ^ 3),
    div_le_div_iff₀ (by positivity) (by positivity)]
  rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 8), abs_pow, abs_pow, sq_abs]
  have hp := abs_nonneg p₁
  have h3 : |p₁| ^ 3 * p₂ ^ 2 ≤ |p₁| ^ 3 * (4 * (S p₁) ^ 2) :=
    mul_le_mul_of_nonneg_left h2 (pow_nonneg hp 3)
  nlinarith [h3, hS, pow_nonneg hp 3, mul_nonneg (pow_nonneg hp 3) hS.le]

lemma abs_ζ_p₂_le : |ζ_p₂ p₁ p₂| ≤ 4 / (3 * S p₁) := by
  unfold ζ_p₂
  have hS := S_pos p₁
  rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < 3 * (S p₁) ^ 2),
    div_le_div_iff₀ (by positivity) (by positivity), abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 2)]
  nlinarith [abs_nonneg p₂]

lemma abs_ζ_p₁p₁_le :
    |ζ_p₁p₁ p₁ p₂| ≤ 32 / 3 * (3 * p₁ ^ 2 / S p₁ + 12 * p₁ ^ 6 / (S p₁) ^ 2) := by
  unfold ζ_p₁p₁
  have hS := S_pos p₁
  have h2 := sq_p₂_le hb
  rw [abs_mul, abs_div, abs_neg, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 8),
    abs_of_pos (by norm_num : (0:ℝ) < 3), abs_pow, sq_abs]
  have hA : 0 ≤ 3 * p₁ ^ 2 / (S p₁) ^ 3 := by positivity
  have hB : 0 ≤ 12 * p₁ ^ 6 / (S p₁) ^ 4 := by positivity
  have hsub : |3 * p₁ ^ 2 / (S p₁) ^ 3 - 12 * p₁ ^ 6 / (S p₁) ^ 4|
      ≤ 3 * p₁ ^ 2 / (S p₁) ^ 3 + 12 * p₁ ^ 6 / (S p₁) ^ 4 := by
    rw [abs_le]; constructor <;> linarith
  have hp2 : 8 * p₂ ^ 2 / 3 ≤ 32 * (S p₁) ^ 2 / 3 := by linarith
  calc 8 * p₂ ^ 2 / 3 * |3 * p₁ ^ 2 / (S p₁) ^ 3 - 12 * p₁ ^ 6 / (S p₁) ^ 4|
      ≤ 32 * (S p₁) ^ 2 / 3 * (3 * p₁ ^ 2 / (S p₁) ^ 3 + 12 * p₁ ^ 6 / (S p₁) ^ 4) := by
        gcongr
    _ = 32 / 3 * (3 * p₁ ^ 2 / S p₁ + 12 * p₁ ^ 6 / (S p₁) ^ 2) := by
        field_simp

end band

end RotorFl
