/-
# Fast region Ω₃ — part 1: setup and elementary estimates

Hypotheses of the fast region outside K: `2S ≤ |p₂|` (gate saturated, boundary included) and
`H > R`. Consequences used everywhere below, with `u = 1/p₂`, `a = |p₁|`:

* `|p₂| ≥ 1999`, `p₂ ≠ 0`, `FastU p₁ u` (so `a·|u| ≤ 3/1000`);
* `p̃f = p₂ − D`, `D = t₁u + t₂u² + t₃u³ − t₄u⁴`, `|D| ≤ 1.005 |u|`;
* `θH = θp̃f²/2 + θ(p₁²/2 + 1 + W)`, `|W| ≤ 1/250`;
* `|∂ₚ₁ p̃f| ≤ 1.01 u²`.
-/
import RotorFl.FastGen
import RotorFl.FastBounds

namespace RotorFl

open Real

/-- Fast-region point outside K. -/
structure FastPt (r p₁ p₂ : ℝ) : Prop where
  gate : 2 * S p₁ ≤ |p₂|
  energy : Rex < H r p₁ p₂

namespace FastPt

variable {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂)
include h

lemma p₁_pow_le : p₁ ^ 4 ≤ |p₂| / 2 - 20 := by
  have := h.gate
  unfold S Rρ at this
  linarith

lemma abs_p₂_ge : 1999 ≤ |p₂| := by
  by_contra hlt
  push_neg at hlt
  have h4 : p₁ ^ 4 < 1000 := by have := h.p₁_pow_le; linarith
  have h2 : p₁ ^ 2 < 32 := by
    by_contra h32
    push_neg at h32
    have : (32 : ℝ) ^ 2 ≤ (p₁ ^ 2) ^ 2 := pow_le_pow_left₀ (by norm_num) h32 2
    have e : p₁ ^ 4 = (p₁ ^ 2) ^ 2 := by ring
    linarith
  have hp2 : p₂ ^ 2 < 1999 ^ 2 := by
    have := abs_nonneg p₂
    nlinarith [sq_abs p₂]
  have hH := h.energy
  unfold Rex H at hH
  have := neg_one_le_cos r
  linarith

lemma p₂_ne : p₂ ≠ 0 := by
  intro h0
  have := h.abs_p₂_ge
  rw [h0, abs_zero] at this
  linarith

/-- `u = 1/p₂` satisfies the `FastU` bounds. -/
lemma fastU : FastU p₁ (1 / p₂) := by
  have hp := h.abs_p₂_ge
  have hpos : 0 < |p₂| := by linarith
  constructor
  · rw [abs_one_div p₂]
    exact div_le_div_of_nonneg_left (by norm_num) (by norm_num) hp
  · rw [abs_one_div p₂, div_mul_eq_mul_div, one_mul, div_le_one hpos]
    have := h.p₁_pow_le
    linarith

end FastPt

lemma abs_one_div (p₂ : ℝ) : |1 / p₂| = 1 / |p₂| := by rw [abs_div, abs_one]

/-! ## The correction `D` and the exponent shift `W` -/

/-- `D = t₁u + t₂u² + t₃u³ − t₄u⁴`; `p̃f = p₂ − D` with `u = 1/p₂`. -/
noncomputable def Dc (r p₁ u : ℝ) : ℝ :=
  t₁ r p₁ * u + t₂ r p₁ * u ^ 2 + t₃ r p₁ * u ^ 3 - t₄ r p₁ * u ^ 4

lemma ptf_eq_sub {r p₁ p₂ : ℝ} (h₂ : p₂ ≠ 0) : ptf r p₁ p₂ = p₂ - Dc r p₁ (1 / p₂) := by
  unfold ptf Dc
  field_simp
  ring

/-- `W = t₂u + t₃u² − t₄u³ − D²/2`. -/
noncomputable def Wc (r p₁ u : ℝ) : ℝ :=
  t₂ r p₁ * u + t₃ r p₁ * u ^ 2 - t₄ r p₁ * u ^ 3 - (Dc r p₁ u) ^ 2 / 2

/-- `θH = θ p̃f²/2 + θ (p₁²/2 + 1 + W)`. -/
lemma H_split {r p₁ p₂ : ℝ} (h₂ : p₂ ≠ 0) :
    θ * H r p₁ p₂ = θ / 2 * (ptf r p₁ p₂) ^ 2 + θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂)) := by
  rw [ptf_eq_sub h₂]
  unfold H Wc Dc t₁
  field_simp
  ring

/-! ## Numerical bounds in terms of `v = |u|`, `m = |p₁|·|u|` -/

section bounds
variable {r p₁ u : ℝ} (hU : FastU p₁ u)
include hU

lemma abs_Dc_le : |Dc r p₁ u| ≤ (1 + 1 / 200) * |u| := by
  have hm := hU.abs_mul_le
  have hv := hU.small
  have hv0 := abs_nonneg u
  have ha0 := abs_nonneg p₁
  have b1 := abs_t1_le r p₁
  have b2 := abs_t2_le r p₁
  have b3 := abs_t3_le r p₁
  have b4 := abs_t4_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b1 b2 b3 b4
  unfold Dc
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  calc |t₁ r p₁ * u + t₂ r p₁ * u ^ 2 + t₃ r p₁ * u ^ 3 - t₄ r p₁ * u ^ 4|
      ≤ |t₁ r p₁ * u + t₂ r p₁ * u ^ 2 + t₃ r p₁ * u ^ 3| + |t₄ r p₁ * u ^ 4| := abs_sub _ _
    _ ≤ |t₁ r p₁ * u + t₂ r p₁ * u ^ 2| + |t₃ r p₁ * u ^ 3| + |t₄ r p₁ * u ^ 4| := by
        gcongr; exact abs_add_le _ _
    _ ≤ |t₁ r p₁ * u| + |t₂ r p₁ * u ^ 2| + |t₃ r p₁ * u ^ 3| + |t₄ r p₁ * u ^ 4| := by
        gcongr; exact abs_add_le _ _
    _ = |t₁ r p₁| * v + |t₂ r p₁| * v ^ 2 + |t₃ r p₁| * v ^ 3 + |t₄ r p₁| * v ^ 4 := by
        simp only [abs_mul, abs_pow, hvdef]
    _ ≤ 1 * v + a * v ^ 2 + (1 + a + a ^ 2) * v ^ 3 + (5 / 2 + 4 * a + 3 * a ^ 2 + a ^ 3) * v ^ 4 := by
        gcongr <;> linarith
    _ = v * (1 + (a * v) + v ^ 2 + (a * v) * v + (a * v) ^ 2
          + 5 / 2 * v ^ 3 + 4 * (a * v) * v ^ 2 + 3 * (a * v) ^ 2 * v + (a * v) ^ 3) := by ring
    _ ≤ (1 + 1 / 200) * v := by
        have h1 : (a * v) ^ 2 ≤ (3 / 1000) ^ 2 := pow_le_pow_left₀ hm0 hm 2
        have h2 : (a * v) ^ 3 ≤ (3 / 1000) ^ 3 := pow_le_pow_left₀ hm0 hm 3
        have h3 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        have h4 : v ^ 3 ≤ (1 / 1999) ^ 3 := pow_le_pow_left₀ hv0 hv 3
        have h5 : (a * v) * v ≤ 3 / 1000 * (1 / 1999) := mul_le_mul hm hv hv0 (by norm_num)
        have h6 : (a * v) * v ^ 2 ≤ 3 / 1000 * (1 / 1999) ^ 2 := mul_le_mul hm h3 (by positivity) (by norm_num)
        have h7 : (a * v) ^ 2 * v ≤ (3 / 1000) ^ 2 * (1 / 1999) := mul_le_mul h1 hv hv0 (by norm_num)
        have : 1 + (a * v) + v ^ 2 + (a * v) * v + (a * v) ^ 2
          + 5 / 2 * v ^ 3 + 4 * (a * v) * v ^ 2 + 3 * (a * v) ^ 2 * v + (a * v) ^ 3 ≤ 1 + 1 / 200 := by
          nlinarith
        exact mul_le_mul_of_nonneg_left this hv0 |>.trans_eq (by ring)

lemma abs_Wc_le : |Wc r p₁ u| ≤ 1 / 250 := by
  have hm := hU.abs_mul_le
  have hv := hU.small
  have hv0 := abs_nonneg u
  have ha0 := abs_nonneg p₁
  have b2 := abs_t2_le r p₁
  have b3 := abs_t3_le r p₁
  have b4 := abs_t4_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b2 b3 b4
  have hD := abs_Dc_le (r := r) hU
  have hD2 : (Dc r p₁ u) ^ 2 ≤ ((1 + 1 / 200) * |u|) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hD 2
  unfold Wc
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  calc |t₂ r p₁ * u + t₃ r p₁ * u ^ 2 - t₄ r p₁ * u ^ 3 - (Dc r p₁ u) ^ 2 / 2|
      ≤ |t₂ r p₁ * u + t₃ r p₁ * u ^ 2 - t₄ r p₁ * u ^ 3| + |(Dc r p₁ u) ^ 2 / 2| := abs_sub _ _
    _ ≤ |t₂ r p₁ * u + t₃ r p₁ * u ^ 2| + |t₄ r p₁ * u ^ 3| + |(Dc r p₁ u) ^ 2 / 2| := by
        gcongr; exact abs_sub _ _
    _ ≤ |t₂ r p₁ * u| + |t₃ r p₁ * u ^ 2| + |t₄ r p₁ * u ^ 3| + |(Dc r p₁ u) ^ 2 / 2| := by
        gcongr; exact abs_add_le _ _
    _ = |t₂ r p₁| * v + |t₃ r p₁| * v ^ 2 + |t₄ r p₁| * v ^ 3 + (Dc r p₁ u) ^ 2 / 2 := by
        rw [abs_of_nonneg (by positivity : (0 : ℝ) ≤ (Dc r p₁ u) ^ 2 / 2)]
        simp only [abs_mul, abs_pow, hvdef]
    _ ≤ a * v + (1 + a + a ^ 2) * v ^ 2 + (5 / 2 + 4 * a + 3 * a ^ 2 + a ^ 3) * v ^ 3
          + ((1 + 1 / 200) * v) ^ 2 / 2 := by
        gcongr <;> linarith
    _ = (a * v) + v ^ 2 + (a * v) * v + (a * v) ^ 2 + 5 / 2 * v ^ 3 + 4 * (a * v) * v ^ 2
          + 3 * (a * v) ^ 2 * v + (a * v) ^ 3 + (1 + 1 / 200) ^ 2 / 2 * v ^ 2 := by ring
    _ ≤ 1 / 250 := by
        have h1 : (a * v) ^ 2 ≤ (3 / 1000) ^ 2 := pow_le_pow_left₀ hm0 hm 2
        have h2 : (a * v) ^ 3 ≤ (3 / 1000) ^ 3 := pow_le_pow_left₀ hm0 hm 3
        have h3 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        have h4 : v ^ 3 ≤ (1 / 1999) ^ 3 := pow_le_pow_left₀ hv0 hv 3
        have h5 : (a * v) * v ≤ 3 / 1000 * (1 / 1999) := mul_le_mul hm hv hv0 (by norm_num)
        have h6 : (a * v) * v ^ 2 ≤ 3 / 1000 * (1 / 1999) ^ 2 :=
          mul_le_mul hm h3 (by positivity) (by norm_num)
        have h7 : (a * v) ^ 2 * v ≤ (3 / 1000) ^ 2 * (1 / 1999) := mul_le_mul h1 hv hv0 (by norm_num)
        nlinarith

end bounds

/-! ## `∂ₚ₁ p̃f` and `p̃f·u` -/

lemma ptf_p₁_eq {r p₁ p₂ : ℝ} (h₂ : p₂ ≠ 0) :
    ptf_p₁ r p₁ p₂ = -(t2_p₁ r p₁) * (1 / p₂) ^ 2 - t3_p₁ r p₁ * (1 / p₂) ^ 3
      + t4_p₁ r p₁ * (1 / p₂) ^ 4 := by
  unfold ptf_p₁ t1_p₁
  field_simp
  ring

lemma abs_ptf_p₁_le {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    |ptf_p₁ r p₁ p₂| ≤ (1 + 1 / 100) * (1 / p₂) ^ 2 := by
  have hU := h.fastU
  have hm := hU.abs_mul_le
  have hv := hU.small
  have hv0 := abs_nonneg (1 / p₂)
  have ha0 := abs_nonneg p₁
  have b2 := abs_t2_p₁_le r p₁
  have b3 := abs_t3_p₁_le r p₁
  have b4 := abs_t4_p₁_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b2 b3 b4
  rw [ptf_p₁_eq h.p₂_ne]
  set u := 1 / p₂ with hu
  set v := |u| with hvdef
  set a := |p₁| with hadef
  have hm0 : 0 ≤ a * v := mul_nonneg ha0 hv0
  have hsq : u ^ 2 = v ^ 2 := by rw [hvdef, sq_abs]
  calc |-(t2_p₁ r p₁) * u ^ 2 - t3_p₁ r p₁ * u ^ 3 + t4_p₁ r p₁ * u ^ 4|
      ≤ |-(t2_p₁ r p₁) * u ^ 2 - t3_p₁ r p₁ * u ^ 3| + |t4_p₁ r p₁ * u ^ 4| := abs_add_le _ _
    _ ≤ |-(t2_p₁ r p₁) * u ^ 2| + |t3_p₁ r p₁ * u ^ 3| + |t4_p₁ r p₁ * u ^ 4| := by
        gcongr; exact abs_sub _ _
    _ = |t2_p₁ r p₁| * v ^ 2 + |t3_p₁ r p₁| * v ^ 3 + |t4_p₁ r p₁| * v ^ 4 := by
        simp only [abs_mul, abs_pow, abs_neg, hvdef]
    _ ≤ 1 * v ^ 2 + (1 + 2 * a) * v ^ 3 + (4 + 6 * a + 3 * a ^ 2) * v ^ 4 := by
        gcongr <;> linarith
    _ = v ^ 2 * (1 + v + 2 * (a * v) + 4 * v ^ 2 + 6 * (a * v) * v + 3 * (a * v) ^ 2) := by ring
    _ ≤ v ^ 2 * (1 + 1 / 100) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        have h1 : (a * v) ^ 2 ≤ (3 / 1000) ^ 2 := pow_le_pow_left₀ hm0 hm 2
        have h3 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        have h5 : (a * v) * v ≤ 3 / 1000 * (1 / 1999) := mul_le_mul hm hv hv0 (by norm_num)
        nlinarith
    _ = (1 + 1 / 100) * u ^ 2 := by rw [hsq]; ring

/-- `q = p̃f · u` is within `3·10⁻⁷` of 1. -/
lemma abs_q_sub_one_le {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    |ptf r p₁ p₂ * (1 / p₂) - 1| ≤ 3 / 10000000 := by
  have hU := h.fastU
  have hv := hU.small
  have hv0 := abs_nonneg (1 / p₂)
  have hD := abs_Dc_le (r := r) hU
  have h₂ := h.p₂_ne
  have e : ptf r p₁ p₂ * (1 / p₂) - 1 = -(Dc r p₁ (1 / p₂) * (1 / p₂)) := by
    rw [ptf_eq_sub h₂]; field_simp; ring
  rw [e, abs_neg, abs_mul]
  have h3 : |1 / p₂| ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  calc |Dc r p₁ (1 / p₂)| * |1 / p₂| ≤ ((1 + 1 / 200) * |1 / p₂|) * |1 / p₂| := by gcongr
    _ = (1 + 1 / 200) * |1 / p₂| ^ 2 := by ring
    _ ≤ (1 + 1 / 200) * (1 / 1999) ^ 2 := by gcongr
    _ ≤ 3 / 10000000 := by norm_num

end RotorFl
