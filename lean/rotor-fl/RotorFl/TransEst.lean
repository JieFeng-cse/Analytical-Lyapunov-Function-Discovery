/-
# Transition band Ω₂ — part 1: setup and elementary bounds

Points with `S ≤ |p₂| ≤ 2S` and `|p₁| ≥ 100` (outside K this is forced by
`p₁² > M = 10⁴`). With `u = 1/p₂`, `a = |p₁|`:

* `S ≥ a⁴ ≥ 10⁸`, `|u| ≤ 1/S ≤ 10⁻⁸`, `a|u| ≤ 1/a³ ≤ 10⁻⁶`;
* gate derivatives: `|∂ₚ₁ρ| ≤ 14`, `|∂ₚ₂ρ| ≤ 10⁻⁵`, `|∂²ₚ₁ρ| ≤ 130`, `|Lρ| ≤ 16a`;
* `|T| ≤ 1.005|u|`, `|∂ᵣp̃f| ≤ 1.01|u|`, `|∂ₚ₁p̃f| ≤ 1.01u²`, `|∂ₚ₂p̃f − 1| ≤ 1.01u²`,
  `|∂²ₚ₁p̃f| ≤ 2.03|u|³`.
-/
import RotorFl.GenForm
import RotorFl.FastEst

namespace RotorFl

open Real

structure TransPt (r p₁ p₂ : ℝ) : Prop where
  lo : S p₁ ≤ |p₂|
  hi : |p₂| ≤ 2 * S p₁
  big : 100 ≤ |p₁|

namespace TransPt

variable {r p₁ p₂ : ℝ} (h : TransPt r p₁ p₂)
include h

lemma abs_pow4 : |p₁| ^ 4 = p₁ ^ 4 := by
  rw [← abs_pow, abs_of_nonneg (by positivity)]

lemma a4_le_S : |p₁| ^ 4 ≤ S p₁ := by
  unfold S Rρ; rw [h.abs_pow4]; linarith

lemma a4_ge : (10 : ℝ) ^ 8 ≤ |p₁| ^ 4 := by
  have := pow_le_pow_left₀ (by norm_num) h.big 4
  norm_num at this ⊢; linarith

lemma S_ge : (10 : ℝ) ^ 8 ≤ S p₁ := le_trans h.a4_ge h.a4_le_S

lemma abs_p₂_ge : (10 : ℝ) ^ 8 ≤ |p₂| := le_trans h.S_ge h.lo

lemma p₂_ne : p₂ ≠ 0 := by
  intro h0; have := h.abs_p₂_ge; rw [h0, abs_zero] at this; norm_num at this

lemma abs_u_le : |1 / p₂| ≤ 1 / S p₁ := by
  rw [abs_one_div]; exact one_div_le_one_div_of_le (S_pos p₁) h.lo

lemma abs_u_le' : |1 / p₂| ≤ 1 / 10 ^ 8 := by
  rw [abs_one_div]; exact one_div_le_one_div_of_le (by norm_num) h.abs_p₂_ge

lemma a_mul_u_le : |p₁| * |1 / p₂| ≤ 1 / 10 ^ 6 := by
  have hu := h.abs_u_le
  have hS := S_pos p₁
  have hb := h.big
  have ha0 : 0 ≤ |p₁| := abs_nonneg p₁
  have h4 := h.a4_le_S
  -- a / S ≤ 10⁻⁶ since 10⁻⁶ S ≥ 10⁻⁶ a⁴ ≥ a
  have key : |p₁| ≤ 1 / 10 ^ 6 * S p₁ := by
    have : (10 : ℝ) ^ 6 * |p₁| ≤ |p₁| ^ 4 := by
      have h3 : (100 : ℝ) ^ 3 ≤ |p₁| ^ 3 := pow_le_pow_left₀ (by norm_num) hb 3
      nlinarith [h3, ha0]
    linarith
  calc |p₁| * |1 / p₂| ≤ |p₁| * (1 / S p₁) := by gcongr
    _ = |p₁| / S p₁ := by ring
    _ ≤ (1 / 10 ^ 6 * S p₁) / S p₁ := by gcongr
    _ = 1 / 10 ^ 6 := by field_simp

lemma smallU : SmallU p₁ (1 / p₂) :=
  ⟨le_trans h.abs_u_le' (by norm_num), le_trans h.a_mul_u_le (by norm_num)⟩

/-! ### Gate derivatives -/

lemma abs_ζ_p₁_le' : |ζ_p₁ p₁ p₂| ≤ 32 / 300 := by
  have h1 := abs_ζ_p₁_le h.hi
  have hS := S_pos p₁
  have h4 := h.a4_le_S
  have hb := h.big
  have ha0 : 0 ≤ |p₁| := abs_nonneg p₁
  -- 32 a³ /(3S) ≤ 32 a³/(3 a⁴) = 32/(3a) ≤ 32/300
  have : 32 * |p₁| ^ 3 / (3 * S p₁) ≤ 32 / 300 := by
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    have : |p₁| ^ 3 * 100 ≤ |p₁| ^ 4 := by nlinarith [pow_nonneg ha0 3]
    nlinarith
  linarith

lemma abs_ρ_p₁_le : |ρ_p₁ p₁ p₂| ≤ 14 := by
  unfold ρ_p₁
  rw [abs_mul]
  calc |ψ' (ζ p₁ p₂)| * |ζ_p₁ p₁ p₂| ≤ 128 * (32 / 300) := by
        gcongr
        · exact abs_ψ'_le _
        · exact h.abs_ζ_p₁_le'
    _ ≤ 14 := by norm_num

lemma abs_ρ_p₂_le : |ρ_p₂ p₁ p₂| ≤ 1 / 10 ^ 5 := by
  unfold ρ_p₂
  rw [abs_mul]
  have h1 := abs_ζ_p₂_le (p₁ := p₁) h.hi
  have hS := h.S_ge
  have : 4 / (3 * S p₁) ≤ 4 / (3 * 10 ^ 8) := by
    apply div_le_div_of_nonneg_left (by norm_num) (by norm_num); linarith
  calc |ψ' (ζ p₁ p₂)| * |ζ_p₂ p₁ p₂| ≤ 128 * (4 / (3 * 10 ^ 8)) := by
        gcongr
        · exact abs_ψ'_le _
        · linarith
    _ ≤ 1 / 10 ^ 5 := by norm_num

lemma abs_ζ_p₁p₁_le' : |ζ_p₁p₁ p₁ p₂| ≤ 160 / 10 ^ 4 := by
  have h1 := abs_ζ_p₁p₁_le h.hi
  have hS := S_pos p₁
  have h4 := h.a4_le_S
  have hb := h.big
  have ha0 : 0 ≤ |p₁| := abs_nonneg p₁
  have hp2 : p₁ ^ 2 = |p₁| ^ 2 := (sq_abs p₁).symm
  have hp6 : p₁ ^ 6 = |p₁| ^ 6 := by rw [← abs_pow, abs_of_nonneg (by positivity)]
  have hA : 3 * p₁ ^ 2 / S p₁ ≤ 3 / 10 ^ 4 := by
    rw [hp2, div_le_div_iff₀ hS (by norm_num)]
    have : |p₁| ^ 2 * 10 ^ 4 ≤ |p₁| ^ 4 := by
      have h2 : (100 : ℝ) ^ 2 ≤ |p₁| ^ 2 := pow_le_pow_left₀ (by norm_num) hb 2
      nlinarith [sq_nonneg |p₁|]
    nlinarith
  have hB : 12 * p₁ ^ 6 / (S p₁) ^ 2 ≤ 12 / 10 ^ 4 := by
    rw [hp6, div_le_div_iff₀ (by positivity) (by norm_num)]
    have h8 : |p₁| ^ 8 ≤ (S p₁) ^ 2 := by
      have := pow_le_pow_left₀ (by positivity) h4 2
      calc |p₁| ^ 8 = (|p₁| ^ 4) ^ 2 := by ring
        _ ≤ (S p₁) ^ 2 := this
    have : |p₁| ^ 6 * 10 ^ 4 ≤ |p₁| ^ 8 := by
      have h2 : (100 : ℝ) ^ 2 ≤ |p₁| ^ 2 := pow_le_pow_left₀ (by norm_num) hb 2
      nlinarith [pow_nonneg ha0 6]
    nlinarith
  calc |ζ_p₁p₁ p₁ p₂| ≤ 32 / 3 * (3 * p₁ ^ 2 / S p₁ + 12 * p₁ ^ 6 / (S p₁) ^ 2) := h1
    _ ≤ 32 / 3 * (3 / 10 ^ 4 + 12 / 10 ^ 4) := by gcongr
    _ = 160 / 10 ^ 4 := by norm_num

lemma abs_ρ_p₁p₁_le : |ρ_p₁p₁ p₁ p₂| ≤ 130 := by
  unfold ρ_p₁p₁
  have h1 := h.abs_ζ_p₁_le'
  have h2 := h.abs_ζ_p₁p₁_le'
  have hψ := abs_ψ'_le (ζ p₁ p₂)
  have hψ' := abs_ψ''_le (ζ p₁ p₂)
  calc |ψ'' (ζ p₁ p₂) * (ζ_p₁ p₁ p₂) ^ 2 + ψ' (ζ p₁ p₂) * ζ_p₁p₁ p₁ p₂|
      ≤ |ψ'' (ζ p₁ p₂) * (ζ_p₁ p₁ p₂) ^ 2| + |ψ' (ζ p₁ p₂) * ζ_p₁p₁ p₁ p₂| := abs_add_le _ _
    _ = |ψ'' (ζ p₁ p₂)| * |ζ_p₁ p₁ p₂| ^ 2 + |ψ' (ζ p₁ p₂)| * |ζ_p₁p₁ p₁ p₂| := by
        rw [abs_mul, abs_mul, abs_pow]
    _ ≤ 11000 * (32 / 300) ^ 2 + 128 * (160 / 10 ^ 4) := by gcongr
    _ ≤ 130 := by norm_num

lemma abs_Lρ_le : |Lρ r p₁ p₂| ≤ 16 * |p₁| := by
  unfold Lρ
  have h1 := h.abs_ρ_p₁_le
  have h2 := h.abs_ρ_p₂_le
  have h3 := h.abs_ρ_p₁p₁_le
  have hb := h.big
  have hs : |sin r - p₁| ≤ 1 + |p₁| := by
    calc |sin r - p₁| ≤ |sin r| + |p₁| := abs_sub _ _
      _ ≤ 1 + |p₁| := by gcongr; exact abs_sin_le_one r
  have hs' : |-sin r| ≤ 1 := by rw [abs_neg]; exact abs_sin_le_one r
  calc |ρ_p₁ p₁ p₂ * (sin r - p₁) + ρ_p₂ p₁ p₂ * (-sin r) + ρ_p₁p₁ p₁ p₂|
      ≤ |ρ_p₁ p₁ p₂ * (sin r - p₁)| + |ρ_p₂ p₁ p₂ * (-sin r)| + |ρ_p₁p₁ p₁ p₂| := abs_add_three _ _ _
    _ = |ρ_p₁ p₁ p₂| * |sin r - p₁| + |ρ_p₂ p₁ p₂| * |-sin r| + |ρ_p₁p₁ p₁ p₂| := by
        rw [abs_mul, abs_mul]
    _ ≤ 14 * (1 + |p₁|) + 1 / 10 ^ 5 * 1 + 130 := by gcongr
    _ ≤ 16 * |p₁| := by nlinarith

end TransPt

/-! ## `T` and the derivatives of `p̃f` with `SmallU` -/

lemma Tf_eq_Dc {r p₁ p₂ : ℝ} (h₂ : p₂ ≠ 0) : Tf r p₁ p₂ = Dc r p₁ (1 / p₂) := by
  unfold Tf Dc; field_simp

lemma abs_Tf_le {r p₁ p₂ : ℝ} (hU : SmallU p₁ (1 / p₂)) (h₂ : p₂ ≠ 0) :
    |Tf r p₁ p₂| ≤ (1 + 1 / 200) * |1 / p₂| := by
  rw [Tf_eq_Dc h₂]; exact abs_Dc_le hU

section smallU
variable {r p₁ p₂ : ℝ} (hU : SmallU p₁ (1 / p₂)) (h₂ : p₂ ≠ 0)
include hU h₂

omit hU in
lemma ptf_r_eq : ptf_r r p₁ p₂ = -(t1_r r p₁) * (1 / p₂) - t2_r r p₁ * (1 / p₂) ^ 2
    - t3_r r p₁ * (1 / p₂) ^ 3 + t4_r r p₁ * (1 / p₂) ^ 4 := by
  unfold ptf_r; field_simp

lemma abs_ptf_r_le : |ptf_r r p₁ p₂| ≤ (1 + 1 / 100) * |1 / p₂| := by
  have hm := hU.abs_mul
  have hv := hU.small
  have hv0 := abs_nonneg (1 / p₂)
  have ha0 := abs_nonneg p₁
  have b1 := abs_t1_r_le r p₁
  have b2 := abs_t2_r_le r p₁
  have b3 := abs_t3_r_le r p₁
  have b4 := abs_t4_r_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b1 b2 b3 b4
  rw [ptf_r_eq h₂]
  set u := 1 / p₂ with hu
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  calc |-(t1_r r p₁) * u - t2_r r p₁ * u ^ 2 - t3_r r p₁ * u ^ 3 + t4_r r p₁ * u ^ 4|
      ≤ |-(t1_r r p₁) * u - t2_r r p₁ * u ^ 2 - t3_r r p₁ * u ^ 3| + |t4_r r p₁ * u ^ 4| :=
        abs_add_le _ _
    _ ≤ |-(t1_r r p₁) * u - t2_r r p₁ * u ^ 2| + |t3_r r p₁ * u ^ 3| + |t4_r r p₁ * u ^ 4| := by
        gcongr; exact abs_sub _ _
    _ ≤ |-(t1_r r p₁) * u| + |t2_r r p₁ * u ^ 2| + |t3_r r p₁ * u ^ 3| + |t4_r r p₁ * u ^ 4| := by
        gcongr; exact abs_sub _ _
    _ = |t1_r r p₁| * v + |t2_r r p₁| * v ^ 2 + |t3_r r p₁| * v ^ 3 + |t4_r r p₁| * v ^ 4 := by
        simp only [abs_mul, abs_pow, abs_neg, hvdef]
    _ ≤ 1 * v + a * v ^ 2 + (2 + a + a ^ 2) * v ^ 3 + (3 + 7 * a + 3 * a ^ 2 + a ^ 3) * v ^ 4 := by
        gcongr <;> linarith
    _ = v * (1 + (a * v) + 2 * v ^ 2 + (a * v) * v + (a * v) ^ 2
          + 3 * v ^ 3 + 7 * (a * v) * v ^ 2 + 3 * (a * v) ^ 2 * v + (a * v) ^ 3) := by ring
    _ ≤ (1 + 1 / 100) * v := by
        have h1 : (a * v) ^ 2 ≤ (3 / 1000) ^ 2 := pow_le_pow_left₀ hm0 hm 2
        have h2 : (a * v) ^ 3 ≤ (3 / 1000) ^ 3 := pow_le_pow_left₀ hm0 hm 3
        have h3 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        have h4 : v ^ 3 ≤ (1 / 1999) ^ 3 := pow_le_pow_left₀ hv0 hv 3
        have h5 : (a * v) * v ≤ 3 / 1000 * (1 / 1999) := mul_le_mul hm hv hv0 (by norm_num)
        have h6 : (a * v) * v ^ 2 ≤ 3 / 1000 * (1 / 1999) ^ 2 :=
          mul_le_mul hm h3 (by positivity) (by norm_num)
        have h7 : (a * v) ^ 2 * v ≤ (3 / 1000) ^ 2 * (1 / 1999) := mul_le_mul h1 hv hv0 (by norm_num)
        have : 1 + (a * v) + 2 * v ^ 2 + (a * v) * v + (a * v) ^ 2
          + 3 * v ^ 3 + 7 * (a * v) * v ^ 2 + 3 * (a * v) ^ 2 * v + (a * v) ^ 3 ≤ 1 + 1 / 100 := by
          nlinarith
        exact mul_le_mul_of_nonneg_left this hv0 |>.trans_eq (by ring)

omit hU in
lemma ptf_p₂_sub_one_eq : ptf_p₂ r p₁ p₂ - 1 = t₁ r p₁ * (1 / p₂) ^ 2 + 2 * t₂ r p₁ * (1 / p₂) ^ 3
    + 3 * t₃ r p₁ * (1 / p₂) ^ 4 - 4 * t₄ r p₁ * (1 / p₂) ^ 5 := by
  unfold ptf_p₂; field_simp; ring

lemma abs_ptf_p₂_sub_one_le : |ptf_p₂ r p₁ p₂ - 1| ≤ (1 + 1 / 100) * (1 / p₂) ^ 2 := by
  have hm := hU.abs_mul
  have hv := hU.small
  have hv0 := abs_nonneg (1 / p₂)
  have ha0 := abs_nonneg p₁
  have b1 := abs_t1_le r p₁
  have b2 := abs_t2_le r p₁
  have b3 := abs_t3_le r p₁
  have b4 := abs_t4_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b1 b2 b3 b4
  rw [ptf_p₂_sub_one_eq h₂]
  set u := 1 / p₂ with hu
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  have hsq : u ^ 2 = v ^ 2 := by rw [hvdef, sq_abs]
  calc |t₁ r p₁ * u ^ 2 + 2 * t₂ r p₁ * u ^ 3 + 3 * t₃ r p₁ * u ^ 4 - 4 * t₄ r p₁ * u ^ 5|
      ≤ |t₁ r p₁ * u ^ 2 + 2 * t₂ r p₁ * u ^ 3 + 3 * t₃ r p₁ * u ^ 4| + |4 * t₄ r p₁ * u ^ 5| :=
        abs_sub _ _
    _ ≤ |t₁ r p₁ * u ^ 2 + 2 * t₂ r p₁ * u ^ 3| + |3 * t₃ r p₁ * u ^ 4| + |4 * t₄ r p₁ * u ^ 5| := by
        gcongr; exact abs_add_le _ _
    _ ≤ |t₁ r p₁ * u ^ 2| + |2 * t₂ r p₁ * u ^ 3| + |3 * t₃ r p₁ * u ^ 4| + |4 * t₄ r p₁ * u ^ 5| := by
        gcongr; exact abs_add_le _ _
    _ = |t₁ r p₁| * v ^ 2 + 2 * |t₂ r p₁| * v ^ 3 + 3 * |t₃ r p₁| * v ^ 4 + 4 * |t₄ r p₁| * v ^ 5 := by
        simp only [abs_mul, abs_pow, hvdef]
        norm_num
    _ ≤ 1 * v ^ 2 + 2 * a * v ^ 3 + 3 * (1 + a + a ^ 2) * v ^ 4
          + 4 * (5 / 2 + 4 * a + 3 * a ^ 2 + a ^ 3) * v ^ 5 := by
        gcongr <;> linarith
    _ = v ^ 2 * (1 + 2 * (a * v) + 3 * v ^ 2 + 3 * (a * v) * v + 3 * (a * v) ^ 2
          + 10 * v ^ 3 + 16 * (a * v) * v ^ 2 + 12 * (a * v) ^ 2 * v + 4 * (a * v) ^ 3) := by ring
    _ ≤ v ^ 2 * (1 + 1 / 100) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        have h1 : (a * v) ^ 2 ≤ (3 / 1000) ^ 2 := pow_le_pow_left₀ hm0 hm 2
        have h2 : (a * v) ^ 3 ≤ (3 / 1000) ^ 3 := pow_le_pow_left₀ hm0 hm 3
        have h3 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        have h4 : v ^ 3 ≤ (1 / 1999) ^ 3 := pow_le_pow_left₀ hv0 hv 3
        have h5 : (a * v) * v ≤ 3 / 1000 * (1 / 1999) := mul_le_mul hm hv hv0 (by norm_num)
        have h6 : (a * v) * v ^ 2 ≤ 3 / 1000 * (1 / 1999) ^ 2 :=
          mul_le_mul hm h3 (by positivity) (by norm_num)
        have h7 : (a * v) ^ 2 * v ≤ (3 / 1000) ^ 2 * (1 / 1999) := mul_le_mul h1 hv hv0 (by norm_num)
        nlinarith
    _ = (1 + 1 / 100) * u ^ 2 := by rw [hsq]; ring

omit hU in
lemma ptf_p₁p₁_eq : ptf_p₁p₁ r p₁ p₂ = -(t3_p₁p₁ r p₁) * (1 / p₂) ^ 3 + t4_p₁p₁ r p₁ * (1 / p₂) ^ 4 := by
  unfold ptf_p₁p₁ t1_p₁p₁ t2_p₁p₁; field_simp; ring

lemma abs_ptf_p₁p₁_le : |ptf_p₁p₁ r p₁ p₂| ≤ (2 + 3 / 100) * |1 / p₂| ^ 3 := by
  have hm := hU.abs_mul
  have hv := hU.small
  have hv0 := abs_nonneg (1 / p₂)
  have ha0 := abs_nonneg p₁
  have b3 := abs_t3_p₁p₁_le r p₁
  have b4 := abs_t4_p₁p₁_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b3 b4
  rw [ptf_p₁p₁_eq h₂]
  set u := 1 / p₂ with hu
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  calc |-(t3_p₁p₁ r p₁) * u ^ 3 + t4_p₁p₁ r p₁ * u ^ 4|
      ≤ |-(t3_p₁p₁ r p₁) * u ^ 3| + |t4_p₁p₁ r p₁ * u ^ 4| := abs_add_le _ _
    _ = |t3_p₁p₁ r p₁| * v ^ 3 + |t4_p₁p₁ r p₁| * v ^ 4 := by
        simp only [abs_mul, abs_pow, abs_neg, hvdef]
    _ ≤ 2 * v ^ 3 + (6 + 6 * a) * v ^ 4 := by gcongr <;> linarith
    _ = v ^ 3 * (2 + 6 * v + 6 * (a * v)) := by ring
    _ ≤ v ^ 3 * (2 + 3 / 100) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        linarith
    _ = (2 + 3 / 100) * v ^ 3 := by ring

end smallU

end RotorFl
