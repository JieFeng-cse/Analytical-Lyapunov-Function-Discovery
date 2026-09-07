/-
# T2: `L V + φ(V)` is bounded on the exempt set K

`K = {H ≤ R} ∪ {ρ < 1 ∧ p₁² ≤ M}` is contained in
`{|p₂| < S} ∪ Box`, `Box = {|p₁| ≤ 2000, 20 ≤ |p₂| ≤ 2·10⁸ + 40}`.

* On `{|p₂| < S}` (gate closed) `L V = θ e^{θH}(1 − (1−θ)p₁²)`: this is `≤ 0` unless
  `p₁² ≤ 1.03`, in which case `|p₂| < S ≤ 22` and `H ≤ 250`, so `L V ≤ θ e^{5}`.
* On `Box` every factor of the general formula is bounded by a crude explicit
  constant; the total is dominated by `exp (10¹⁵)`.

The explicit constant: `C = exp (10 ^ 15)`.
-/
import RotorFl.Main

namespace RotorFl

open Real

/-! ## Small helpers -/

lemma abs_mul_le_of_le {a b A B : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ B) : |a * b| ≤ A * B := by
  rw [abs_mul]
  exact mul_le_mul ha hb (abs_nonneg _) (le_trans (abs_nonneg _) ha)

lemma abs_add_le_of_le {a b A B : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ B) : |a + b| ≤ A + B :=
  le_trans (abs_add_le _ _) (add_le_add ha hb)

lemma abs_sub_le_of_le {a b A B : ℝ} (ha : |a| ≤ A) (hb : |b| ≤ B) : |a - b| ≤ A + B :=
  le_trans (abs_sub _ _) (add_le_add ha hb)

/-! ## The box -/

structure BoxPt (p₁ p₂ : ℝ) : Prop where
  a : |p₁| ≤ 2000
  lo : 20 ≤ |p₂|
  hi : |p₂| ≤ 2 * 10 ^ 8 + 40

namespace BoxPt

variable {r p₁ p₂ : ℝ} (h : BoxPt p₁ p₂)
include h

lemma p₂_ne : p₂ ≠ 0 := by
  intro h0; have := h.lo; rw [h0, abs_zero] at this; linarith

lemma abs_u_le : |1 / p₂| ≤ 1 / 20 := by
  rw [abs_one_div]; exact one_div_le_one_div_of_le (by norm_num) h.lo

lemma S_ge : 20 ≤ S p₁ := by
  unfold S Rρ; have : 0 ≤ p₁ ^ 4 := by positivity
  linarith

lemma S_le : S p₁ ≤ 2 * 10 ^ 13 := by
  unfold S Rρ
  have h4 : |p₁| ^ 4 ≤ 2000 ^ 4 := pow_le_pow_left₀ (abs_nonneg _) h.a 4
  rw [← abs_pow, abs_of_nonneg (by positivity)] at h4
  norm_num at h4 ⊢; linarith

/-- Generic bound for `c₀ + c₁ u + c₂ u² + c₃ u³ + c₄ u⁴`-type quantities: with
`a ≤ 2000`, `v ≤ 1/20` every degree-≤ 3 polynomial in `a` times `v^k` is tiny. -/
lemma poly_bound : |p₁| ^ 2 ≤ 4 * 10 ^ 6 ∧ |p₁| ^ 3 ≤ 8 * 10 ^ 9 := by
  have ha0 := abs_nonneg p₁
  have ha := h.a
  have h2 : |p₁| ^ 2 ≤ 4 * 10 ^ 6 := by nlinarith
  have h3 : |p₁| ^ 3 ≤ 8 * 10 ^ 9 := by nlinarith
  exact ⟨h2, h3⟩

omit h in
lemma t_bounds (r : ℝ) (ha : |p₁| ≤ 2000) :
    |t₁ r p₁| ≤ 1 ∧ |t₂ r p₁| ≤ 2000 ∧ |t₃ r p₁| ≤ 5 * 10 ^ 6 ∧ |t₄ r p₁| ≤ 9 * 10 ^ 9 := by
  have b1 := abs_t1_le r p₁
  have b2 := abs_t2_le r p₁
  have b3 := abs_t3_le r p₁
  have b4 := abs_t4_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b1 b2 b3 b4
  have ha0 := abs_nonneg p₁
  have h2 : |p₁| ^ 2 ≤ 4 * 10 ^ 6 := by nlinarith
  have h3 : |p₁| ^ 3 ≤ 8 * 10 ^ 9 := by nlinarith
  refine ⟨b1, by linarith, by linarith, by linarith⟩

omit h in
lemma tr_bounds (r : ℝ) (ha : |p₁| ≤ 2000) :
    |t1_r r p₁| ≤ 1 ∧ |t2_r r p₁| ≤ 2000 ∧ |t3_r r p₁| ≤ 5 * 10 ^ 6 ∧ |t4_r r p₁| ≤ 9 * 10 ^ 9 := by
  have b1 := abs_t1_r_le r p₁
  have b2 := abs_t2_r_le r p₁
  have b3 := abs_t3_r_le r p₁
  have b4 := abs_t4_r_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b1 b2 b3 b4
  have ha0 := abs_nonneg p₁
  have h2 : |p₁| ^ 2 ≤ 4 * 10 ^ 6 := by nlinarith
  have h3 : |p₁| ^ 3 ≤ 8 * 10 ^ 9 := by nlinarith
  refine ⟨by linarith, by linarith, by linarith, by linarith⟩

omit h in
lemma tp_bounds (r : ℝ) (ha : |p₁| ≤ 2000) :
    |t2_p₁ r p₁| ≤ 1 ∧ |t3_p₁ r p₁| ≤ 5000 ∧ |t4_p₁ r p₁| ≤ 2 * 10 ^ 7
    ∧ |t3_p₁p₁ r p₁| ≤ 2 ∧ |t4_p₁p₁ r p₁| ≤ 13000 := by
  have b2 := abs_t2_p₁_le r p₁
  have b3 := abs_t3_p₁_le r p₁
  have b4 := abs_t4_p₁_le r p₁
  have c3 := abs_t3_p₁p₁_le r p₁
  have c4 := abs_t4_p₁p₁_le r p₁
  simp only [pow_zero, pow_one, one_mul] at b2 b3 b4 c3 c4
  have ha0 := abs_nonneg p₁
  have h2 : |p₁| ^ 2 ≤ 4 * 10 ^ 6 := by nlinarith
  refine ⟨by linarith, by linarith, by linarith, by linarith, by linarith⟩

lemma abs_Tf_le : |Tf r p₁ p₂| ≤ 6 * 10 ^ 4 := by
  rw [Tf_eq_Dc h.p₂_ne]
  unfold Dc
  obtain ⟨b1, b2, b3, b4⟩ := t_bounds r h.a
  have hv := h.abs_u_le
  have hv0 := abs_nonneg (1 / p₂)
  have hv2 : |1 / p₂| ^ 2 ≤ (1 / 20) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  have hv3 : |1 / p₂| ^ 3 ≤ (1 / 20) ^ 3 := pow_le_pow_left₀ hv0 hv 3
  have hv4 : |1 / p₂| ^ 4 ≤ (1 / 20) ^ 4 := pow_le_pow_left₀ hv0 hv 4
  have e1 := abs_mul_le_of_le b1 hv
  have e2 := abs_mul_le_of_le b2 (by rw [abs_pow]; exact hv2 : |(1 / p₂) ^ 2| ≤ (1 / 20) ^ 2)
  have e3 := abs_mul_le_of_le b3 (by rw [abs_pow]; exact hv3 : |(1 / p₂) ^ 3| ≤ (1 / 20) ^ 3)
  have e4 := abs_mul_le_of_le b4 (by rw [abs_pow]; exact hv4 : |(1 / p₂) ^ 4| ≤ (1 / 20) ^ 4)
  have := abs_sub_le_of_le (abs_add_le_of_le (abs_add_le_of_le e1 e2) e3) e4
  linarith [this]

lemma abs_ptf_r_le : |ptf_r r p₁ p₂| ≤ 6 * 10 ^ 4 := by
  rw [ptf_r_eq h.p₂_ne]
  obtain ⟨b1, b2, b3, b4⟩ := tr_bounds r h.a
  have hv := h.abs_u_le
  have hv0 := abs_nonneg (1 / p₂)
  have hv2 : |1 / p₂| ^ 2 ≤ (1 / 20) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  have hv3 : |1 / p₂| ^ 3 ≤ (1 / 20) ^ 3 := pow_le_pow_left₀ hv0 hv 3
  have hv4 : |1 / p₂| ^ 4 ≤ (1 / 20) ^ 4 := pow_le_pow_left₀ hv0 hv 4
  have b1' : |-(t1_r r p₁)| ≤ 1 := by rw [abs_neg]; exact b1
  have e1 := abs_mul_le_of_le b1' hv
  have e2 := abs_mul_le_of_le b2 (by rw [abs_pow]; exact hv2 : |(1 / p₂) ^ 2| ≤ (1 / 20) ^ 2)
  have e3 := abs_mul_le_of_le b3 (by rw [abs_pow]; exact hv3 : |(1 / p₂) ^ 3| ≤ (1 / 20) ^ 3)
  have e4 := abs_mul_le_of_le b4 (by rw [abs_pow]; exact hv4 : |(1 / p₂) ^ 4| ≤ (1 / 20) ^ 4)
  have := abs_add_le_of_le (abs_sub_le_of_le (abs_sub_le_of_le e1 e2) e3) e4
  linarith [this]

lemma abs_ptf_p₁_le : |ptf_p₁ r p₁ p₂| ≤ 200 := by
  rw [ptf_p₁_eq h.p₂_ne]
  obtain ⟨b2, b3, b4, -, -⟩ := tp_bounds r h.a
  have hv := h.abs_u_le
  have hv0 := abs_nonneg (1 / p₂)
  have hv2 : |1 / p₂| ^ 2 ≤ (1 / 20) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  have hv3 : |1 / p₂| ^ 3 ≤ (1 / 20) ^ 3 := pow_le_pow_left₀ hv0 hv 3
  have hv4 : |1 / p₂| ^ 4 ≤ (1 / 20) ^ 4 := pow_le_pow_left₀ hv0 hv 4
  have b2' : |-(t2_p₁ r p₁)| ≤ 1 := by rw [abs_neg]; exact b2
  have e2 := abs_mul_le_of_le b2' (by rw [abs_pow]; exact hv2 : |(1 / p₂) ^ 2| ≤ (1 / 20) ^ 2)
  have e3 := abs_mul_le_of_le b3 (by rw [abs_pow]; exact hv3 : |(1 / p₂) ^ 3| ≤ (1 / 20) ^ 3)
  have e4 := abs_mul_le_of_le b4 (by rw [abs_pow]; exact hv4 : |(1 / p₂) ^ 4| ≤ (1 / 20) ^ 4)
  have := abs_add_le_of_le (abs_sub_le_of_le e2 e3) e4
  linarith [this]

lemma abs_ptf_p₂_le : |ptf_p₂ r p₁ p₂| ≤ 12000 := by
  have e := ptf_p₂_sub_one_eq (r := r) (p₁ := p₁) h.p₂_ne
  obtain ⟨b1, b2, b3, b4⟩ := t_bounds r h.a
  have hv := h.abs_u_le
  have hv0 := abs_nonneg (1 / p₂)
  have hv2 : |1 / p₂| ^ 2 ≤ (1 / 20) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  have hv3 : |1 / p₂| ^ 3 ≤ (1 / 20) ^ 3 := pow_le_pow_left₀ hv0 hv 3
  have hv4 : |1 / p₂| ^ 4 ≤ (1 / 20) ^ 4 := pow_le_pow_left₀ hv0 hv 4
  have hv5 : |1 / p₂| ^ 5 ≤ (1 / 20) ^ 5 := pow_le_pow_left₀ hv0 hv 5
  have b2' : |2 * t₂ r p₁| ≤ 2 * 2000 := abs_mul_le_of_le (by norm_num) b2
  have b3' : |3 * t₃ r p₁| ≤ 3 * (5 * 10 ^ 6) := abs_mul_le_of_le (by norm_num) b3
  have b4' : |4 * t₄ r p₁| ≤ 4 * (9 * 10 ^ 9) := abs_mul_le_of_le (by norm_num) b4
  have e1 := abs_mul_le_of_le b1 (by rw [abs_pow]; exact hv2 : |(1 / p₂) ^ 2| ≤ (1 / 20) ^ 2)
  have e2 := abs_mul_le_of_le b2' (by rw [abs_pow]; exact hv3 : |(1 / p₂) ^ 3| ≤ (1 / 20) ^ 3)
  have e3 := abs_mul_le_of_le b3' (by rw [abs_pow]; exact hv4 : |(1 / p₂) ^ 4| ≤ (1 / 20) ^ 4)
  have e4 := abs_mul_le_of_le b4' (by rw [abs_pow]; exact hv5 : |(1 / p₂) ^ 5| ≤ (1 / 20) ^ 5)
  have hsub := abs_sub_le_of_le (abs_add_le_of_le (abs_add_le_of_le e1 e2) e3) e4
  rw [← e] at hsub
  have : |ptf_p₂ r p₁ p₂| ≤ |ptf_p₂ r p₁ p₂ - 1| + 1 := by
    have := abs_sub_abs_le_abs_sub (ptf_p₂ r p₁ p₂) 1
    rw [abs_one] at this; linarith
  linarith

lemma abs_ptf_p₁p₁_le : |ptf_p₁p₁ r p₁ p₂| ≤ 1 := by
  rw [ptf_p₁p₁_eq h.p₂_ne]
  obtain ⟨-, -, -, c3, c4⟩ := tp_bounds r h.a
  have hv := h.abs_u_le
  have hv0 := abs_nonneg (1 / p₂)
  have hv3 : |1 / p₂| ^ 3 ≤ (1 / 20) ^ 3 := pow_le_pow_left₀ hv0 hv 3
  have hv4 : |1 / p₂| ^ 4 ≤ (1 / 20) ^ 4 := pow_le_pow_left₀ hv0 hv 4
  have c3' : |-(t3_p₁p₁ r p₁)| ≤ 2 := by rw [abs_neg]; exact c3
  have e3 := abs_mul_le_of_le c3' (by rw [abs_pow]; exact hv3 : |(1 / p₂) ^ 3| ≤ (1 / 20) ^ 3)
  have e4 := abs_mul_le_of_le c4 (by rw [abs_pow]; exact hv4 : |(1 / p₂) ^ 4| ≤ (1 / 20) ^ 4)
  have := abs_add_le_of_le e3 e4
  linarith [this]

/-! ### Gate derivatives on the box -/

lemma abs_ζ_p₁_le : |ζ_p₁ p₁ p₂| ≤ 2 * 10 ^ 23 := by
  unfold ζ_p₁
  have hS := h.S_ge
  have hS0 : 0 < S p₁ := S_pos p₁
  obtain ⟨-, h3⟩ := h.poly_bound
  have hp2 : |p₂| ^ 2 ≤ (2 * 10 ^ 8 + 40) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) h.hi 2
  rw [abs_div, abs_neg, abs_mul, abs_mul, abs_pow, abs_pow, abs_of_pos (by norm_num : (0:ℝ) < 8),
    abs_of_pos (by positivity : (0:ℝ) < 3 * (S p₁) ^ 3), div_le_iff₀ (by positivity)]
  have hS3 : 20 ^ 3 ≤ (S p₁) ^ 3 := pow_le_pow_left₀ (by norm_num) hS 3
  have : 8 * |p₁| ^ 3 * |p₂| ^ 2 ≤ 8 * (8 * 10 ^ 9) * (2 * 10 ^ 8 + 40) ^ 2 := by
    gcongr
  nlinarith

lemma abs_ζ_p₂_le : |ζ_p₂ p₁ p₂| ≤ 4 * 10 ^ 5 := by
  unfold ζ_p₂
  have hS := h.S_ge
  have hS0 : 0 < S p₁ := S_pos p₁
  rw [abs_div, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 2),
    abs_of_pos (by positivity : (0:ℝ) < 3 * (S p₁) ^ 2), div_le_iff₀ (by positivity)]
  have hS2 : 20 ^ 2 ≤ (S p₁) ^ 2 := pow_le_pow_left₀ (by norm_num) hS 2
  have := h.hi
  nlinarith

lemma abs_ζ_p₁p₁_le : |ζ_p₁p₁ p₁ p₂| ≤ 10 ^ 33 := by
  unfold ζ_p₁p₁
  have hS := h.S_ge
  have hS0 : 0 < S p₁ := S_pos p₁
  obtain ⟨h2, -⟩ := h.poly_bound
  have hp2 : |p₂| ^ 2 ≤ (2 * 10 ^ 8 + 40) ^ 2 := pow_le_pow_left₀ (abs_nonneg _) h.hi 2
  have hp6 : |p₁| ^ 6 ≤ 2000 ^ 6 := pow_le_pow_left₀ (abs_nonneg _) h.a 6
  rw [← abs_pow, abs_of_nonneg (by positivity)] at hp6
  have hp2' : p₁ ^ 2 ≤ 4 * 10 ^ 6 := by rw [← sq_abs]; exact h2
  have hA : |3 * p₁ ^ 2 / (S p₁) ^ 3| ≤ 3 * (4 * 10 ^ 6) / 20 ^ 3 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < (S p₁) ^ 3), abs_mul,
      abs_of_pos (by norm_num : (0:ℝ) < 3), abs_of_nonneg (sq_nonneg p₁)]
    have hS3 : 20 ^ 3 ≤ (S p₁) ^ 3 := pow_le_pow_left₀ (by norm_num) hS 3
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  have hB : |12 * p₁ ^ 6 / (S p₁) ^ 4| ≤ 12 * 2000 ^ 6 / 20 ^ 4 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < (S p₁) ^ 4), abs_mul,
      abs_of_pos (by norm_num : (0:ℝ) < 12), abs_of_nonneg (by positivity : (0:ℝ) ≤ p₁ ^ 6)]
    have hS4 : 20 ^ 4 ≤ (S p₁) ^ 4 := pow_le_pow_left₀ (by norm_num) hS 4
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  have hfront : |-(8 * p₂ ^ 2) / 3| ≤ 8 * (2 * 10 ^ 8 + 40) ^ 2 / 3 := by
    rw [abs_div, abs_neg, abs_mul, abs_of_pos (by norm_num : (0:ℝ) < 8),
      abs_of_nonneg (sq_nonneg p₂), abs_of_pos (by norm_num : (0:ℝ) < 3)]
    rw [← sq_abs] at *
    have := pow_le_pow_left₀ (abs_nonneg p₂) h.hi 2
    gcongr
  have := abs_mul_le_of_le hfront (abs_sub_le_of_le hA hB)
  linarith [this]

lemma abs_ρ_p₁_le : |ρ_p₁ p₁ p₂| ≤ 3 * 10 ^ 25 := by
  unfold ρ_p₁
  have := abs_mul_le_of_le (abs_ψ'_le (ζ p₁ p₂)) h.abs_ζ_p₁_le
  linarith [this]

lemma abs_ρ_p₂_le : |ρ_p₂ p₁ p₂| ≤ 6 * 10 ^ 7 := by
  unfold ρ_p₂
  have := abs_mul_le_of_le (abs_ψ'_le (ζ p₁ p₂)) h.abs_ζ_p₂_le
  linarith [this]

lemma abs_ρ_p₁p₁_le : |ρ_p₁p₁ p₁ p₂| ≤ 5 * 10 ^ 50 := by
  unfold ρ_p₁p₁
  have h1 := abs_mul_le_of_le (abs_ψ''_le (ζ p₁ p₂))
    (by rw [abs_pow]; exact pow_le_pow_left₀ (abs_nonneg _) h.abs_ζ_p₁_le 2 :
      |(ζ_p₁ p₁ p₂) ^ 2| ≤ (2 * 10 ^ 23) ^ 2)
  have h2 := abs_mul_le_of_le (abs_ψ'_le (ζ p₁ p₂)) h.abs_ζ_p₁p₁_le
  have := abs_add_le_of_le h1 h2
  linarith [this]

lemma abs_Lρ_le : |Lρ r p₁ p₂| ≤ 6 * 10 ^ 50 := by
  unfold Lρ
  have hs : |sin r - p₁| ≤ 2001 := by
    have := abs_sub_le_of_le (abs_sin_le_one r) h.a; linarith
  have hs' : |-sin r| ≤ 1 := by rw [abs_neg]; exact abs_sin_le_one r
  have e1 := abs_mul_le_of_le h.abs_ρ_p₁_le hs
  have e2 := abs_mul_le_of_le h.abs_ρ_p₂_le hs'
  have := abs_add_le_of_le (abs_add_le_of_le e1 e2) h.abs_ρ_p₁p₁_le
  linarith [this]

/-! ### `p̃` and its derivatives on the box -/

lemma abs_ρ_le : |ρ p₁ p₂| ≤ 1 := by
  rw [ρ_eq_ψ, abs_of_nonneg (Real.smoothTransition.nonneg _)]; exact Real.smoothTransition.le_one _

lemma abs_pt_le : |pt r p₁ p₂| ≤ 3 * 10 ^ 8 := by
  rw [pt_eq_Tf]
  have := abs_sub_le_of_le h.hi (abs_mul_le_of_le h.abs_ρ_le (h.abs_Tf_le (r := r)))
  linarith [this]

lemma abs_pt_r_le : |pt_r r p₁ p₂| ≤ 6 * 10 ^ 4 := by
  unfold pt_r
  have := abs_mul_le_of_le h.abs_ρ_le (h.abs_ptf_r_le (r := r))
  linarith [this]

lemma abs_pt_p₁_le : |pt_p₁ r p₁ p₂| ≤ 2 * 10 ^ 30 := by
  unfold pt_p₁
  have e1 := abs_mul_le_of_le h.abs_ρ_p₁_le (h.abs_Tf_le (r := r))
  have e1' : |-(ρ_p₁ p₁ p₂ * Tf r p₁ p₂)| ≤ 3 * 10 ^ 25 * (6 * 10 ^ 4) := by rw [abs_neg]; exact e1
  have e2 := abs_mul_le_of_le h.abs_ρ_le (h.abs_ptf_p₁_le (r := r))
  have := abs_add_le_of_le e1' e2
  linarith [this]

lemma abs_pt_p₂_le : |pt_p₂ r p₁ p₂| ≤ 4 * 10 ^ 12 := by
  unfold pt_p₂
  have e1 := abs_mul_le_of_le h.abs_ρ_p₂_le (h.abs_Tf_le (r := r))
  have e2 : |1 - ptf_p₂ r p₁ p₂| ≤ 1 + 12000 := abs_sub_le_of_le (by rw [abs_one]) (h.abs_ptf_p₂_le (r := r))
  have e3 := abs_mul_le_of_le h.abs_ρ_le e2
  have := abs_sub_le_of_le (abs_sub_le_of_le (by rw [abs_one] : |(1:ℝ)| ≤ 1) e1) e3
  linarith [this]

lemma abs_pt_p₁p₁_le : |pt_p₁p₁ r p₁ p₂| ≤ 4 * 10 ^ 55 := by
  unfold pt_p₁p₁
  have e1 := abs_mul_le_of_le h.abs_ρ_p₁p₁_le (h.abs_Tf_le (r := r))
  have e1' : |-(ρ_p₁p₁ p₁ p₂ * Tf r p₁ p₂)| ≤ 5 * 10 ^ 50 * (6 * 10 ^ 4) := by rw [abs_neg]; exact e1
  have e2 := abs_mul_le_of_le (abs_mul_le_of_le (by norm_num : |(2:ℝ)| ≤ 2) h.abs_ρ_p₁_le)
    (h.abs_ptf_p₁_le (r := r))
  have e3 := abs_mul_le_of_le h.abs_ρ_le (h.abs_ptf_p₁p₁_le (r := r))
  have := abs_add_le_of_le (abs_add_le_of_le e1' e2) e3
  linarith [this]

lemma abs_Lpt_le : |Lpt r p₁ p₂| ≤ 5 * 10 ^ 55 := by
  unfold Lpt
  have hs1 : |p₂ - p₁| ≤ 2 * 10 ^ 8 + 40 + 2000 := abs_sub_le_of_le h.hi h.a
  have hs2 : |sin r - p₁| ≤ 2001 := by
    have := abs_sub_le_of_le (abs_sin_le_one r) h.a; linarith
  have hs3 : |-sin r| ≤ 1 := by rw [abs_neg]; exact abs_sin_le_one r
  have e1 := abs_mul_le_of_le (h.abs_pt_r_le (r := r)) hs1
  have e2 := abs_mul_le_of_le (h.abs_pt_p₁_le (r := r)) hs2
  have e3 := abs_mul_le_of_le (h.abs_pt_p₂_le (r := r)) hs3
  have := abs_add_le_of_le (abs_add_le_of_le (abs_add_le_of_le e1 e2) e3) (h.abs_pt_p₁p₁_le (r := r))
  linarith [this]

/-! ### The exponentials -/

/-- `E = e^{θ p̃²/2} ≤ e^{10¹⁵}` and `e^{θH} ≤ e^{10¹⁵}`. -/
lemma E_le : exp (θ / 2 * (pt r p₁ p₂) ^ 2) ≤ exp (10 ^ 15) := by
  apply exp_le_exp.mpr
  have := pow_le_pow_left₀ (abs_nonneg _) (h.abs_pt_le (r := r)) 2
  rw [sq_abs] at this
  unfold θ; nlinarith

lemma EH_le : exp (θ * H r p₁ p₂) ≤ exp (10 ^ 15) := by
  apply exp_le_exp.mpr
  unfold H θ
  have h1 := pow_le_pow_left₀ (abs_nonneg _) h.a 2
  have h2 := pow_le_pow_left₀ (abs_nonneg _) h.hi 2
  rw [sq_abs] at h1 h2
  have := neg_one_le_cos r
  nlinarith

end BoxPt

/-! ## T2 -/

/-- The explicit constant of T2. -/
noncomputable def Cbound : ℝ := exp (10 ^ 15) * 10 ^ 100

theorem T2 : T2_statement Cbound := by
  intro r p₁ p₂ hK
  unfold Cbound
  have hE15 : (1 : ℝ) ≤ exp (10 ^ 15) := one_le_exp (by norm_num)
  have hθ : θ = 1 / 50 := rfl
  -- φ(V) ≥ 0 is not needed; we bound φ(V) ≤ cRate · V / 2 later, so first the two cases
  rcases lt_or_ge |p₂| (S p₁) with h1 | h1
  · -- gate closed
    rw [L_V_slow h1, V_eq_Vs h1.le]
    unfold Vs φ
    set EH := exp (θ * H r p₁ p₂) with hEH
    have hEH0 : 0 < EH := exp_pos _
    have hlog : 0 ≤ log (1 + EH) := log_nonneg (by linarith)
    have hφ : cRate * (1 + EH) / (2 + log (1 + EH)) ≤ cRate * (1 + EH) / 2 := by
      apply div_le_div_of_nonneg_left (by unfold cRate; positivity) (by norm_num) (by linarith)
    have hc : cRate = 1 / 20000 := rfl
    rcases le_or_gt (p₁ ^ 2) (103 / 100) with hp | hp
    · -- small p₁: H ≤ 250, EH ≤ e^5
      have hS : S p₁ ≤ 22 := by
        unfold S Rρ; nlinarith [sq_nonneg (p₁ ^ 2)]
      have hp2 : p₂ ^ 2 ≤ 22 ^ 2 := by
        have : |p₂| ≤ 22 := by linarith
        have := pow_le_pow_left₀ (abs_nonneg _) this 2
        rwa [sq_abs] at this
      have hH : H r p₁ p₂ ≤ 250 := by
        unfold H; have := neg_one_le_cos r; nlinarith
      have hEH5 : EH ≤ exp 5 := by
        rw [hEH]; apply exp_le_exp.mpr; rw [hθ]; linarith
      have he5 : exp 5 ≤ 3 ^ 5 := by
        have := exp_one_lt_three
        have h5 : exp 5 = exp 1 ^ 5 := by rw [← exp_nat_mul]; try norm_num
        rw [h5]; exact pow_le_pow_left₀ (exp_pos 1).le this.le 5
      have hT : θ * EH * (1 - (1 - θ) * p₁ ^ 2) ≤ θ * EH := by
        rw [hθ]; nlinarith [sq_nonneg p₁]
      rw [hθ] at hT ⊢
      rw [hc] at hφ ⊢
      generalize hTt : 1 / 50 * EH * (1 - (1 - 1 / 50) * p₁ ^ 2) = Tt at hT ⊢
      generalize hF : 1 / 20000 * (1 + EH) / (2 + log (1 + EH)) = F at hφ ⊢
      linarith
    · -- large p₁: drift term ≤ 0, φ small relative
      have hT : θ * EH * (1 - (1 - θ) * p₁ ^ 2) ≤ 0 := by
        rw [hθ]
        have : 1 - (1 - 1 / 50) * p₁ ^ 2 ≤ 0 := by linarith
        nlinarith
      have hEH' : EH ≤ exp (10 ^ 15) := by
        -- p₁² ≤ 2S ≤ ... : on K, either H ≤ R or p₁² ≤ M; in both cases H ≤ 10¹⁷·(1/50)... we use K
        rw [hEH]; apply exp_le_exp.mpr; rw [hθ]
        unfold K at hK
        rcases hK with hR | ⟨-, hM⟩
        · unfold Rex at hR; linarith
        · unfold Mex at hM
          have hS : S p₁ ≤ 10 ^ 8 + 20 := by
            unfold S Rρ; nlinarith [sq_nonneg (p₁ ^ 2)]
          have hp2 : p₂ ^ 2 ≤ (10 ^ 8 + 20) ^ 2 := by
            have : |p₂| ≤ 10 ^ 8 + 20 := by linarith
            have := pow_le_pow_left₀ (abs_nonneg _) this 2
            rwa [sq_abs] at this
          unfold H; have := neg_one_le_cos r; nlinarith
      rw [hθ] at hT ⊢
      rw [hc] at hφ ⊢
      generalize hTt : 1 / 50 * EH * (1 - (1 - 1 / 50) * p₁ ^ 2) = Tt at hT ⊢
      generalize hF : 1 / 20000 * (1 + EH) / (2 + log (1 + EH)) = F at hφ ⊢
      linarith
  · -- gate (partially) open: the box bounds
    have hB : BoxPt p₁ p₂ := by
      unfold K at hK
      have hS20 : 20 ≤ S p₁ := by
        unfold S Rρ
        have : 0 ≤ p₁ ^ 4 := by positivity
        linarith
      rcases hK with hR | ⟨hρ, hM⟩
      · unfold Rex H at hR
        have hc := cos_le_one r
        have h1' : p₁ ^ 2 ≤ 2000 ^ 2 := by nlinarith [sq_nonneg p₂]
        have h2' : p₂ ^ 2 ≤ 2000 ^ 2 := by nlinarith [sq_nonneg p₁]
        refine ⟨?_, by linarith, ?_⟩
        · exact abs_le.mpr (abs_le_of_sq_le_sq' h1' (by norm_num))
        · have := abs_le.mpr (abs_le_of_sq_le_sq' h2' (by norm_num))
          linarith
      · unfold Mex at hM
        have hlt : |p₂| < 2 * S p₁ := by
          by_contra hge; push_neg at hge
          have := ρ_eq_one hge; linarith
        have hS : S p₁ ≤ 10 ^ 8 + 20 := by
          unfold S Rρ; nlinarith [sq_nonneg (p₁ ^ 2)]
        refine ⟨?_, by linarith, by linarith⟩
        have := abs_le.mpr (abs_le_of_sq_le_sq' (by linarith : p₁ ^ 2 ≤ 100 ^ 2) (by norm_num))
        linarith
    have h₂ := hB.p₂_ne
    rw [L_V_general r p₁ p₂ h₂]
    set E := exp (θ / 2 * (pt r p₁ p₂) ^ 2) with hE
    set EH := exp (θ * H r p₁ p₂) with hEH
    have hE0 : 0 < E := exp_pos _
    have hEH0 : 0 < EH := exp_pos _
    have hEle := hB.E_le (r := r)
    have hEHle := hB.EH_le (r := r)
    rw [← hE] at hEle; rw [← hEH] at hEHle
    -- the pieces
    have hQ : |pt r p₁ p₂| ≤ 3 * 10 ^ 8 := hB.abs_pt_le
    have hQ2 : (pt r p₁ p₂) ^ 2 ≤ (3 * 10 ^ 8) ^ 2 := by
      have := pow_le_pow_left₀ (abs_nonneg _) hQ 2; rwa [sq_abs] at this
    have hQ4 : (pt r p₁ p₂) ^ 4 ≤ (3 * 10 ^ 8) ^ 4 := by
      have := pow_le_pow_left₀ (abs_nonneg _) hQ 4
      rwa [← abs_pow, abs_of_nonneg (by positivity)] at this
    have hQ3 : |pt r p₁ p₂| ^ 3 ≤ (3 * 10 ^ 8) ^ 3 := pow_le_pow_left₀ (abs_nonneg _) hQ 3
    have hF' : |Ff' (pt r p₁ p₂)| ≤ 10 ^ 26 * E := by
      unfold Ff'
      have hb : |2 * pt r p₁ p₂ + θ * (pt r p₁ p₂) ^ 3| ≤ 2 * (3 * 10 ^ 8) + 1 / 50 * (3 * 10 ^ 8) ^ 3 := by
        have e1 := abs_mul_le_of_le (by norm_num : |(2:ℝ)| ≤ 2) hQ
        have e2 := abs_mul_le_of_le (by rw [hθ]; norm_num : |θ| ≤ 1 / 50)
          (by rw [abs_pow]; exact hQ3 : |(pt r p₁ p₂) ^ 3| ≤ (3 * 10 ^ 8) ^ 3)
        exact abs_add_le_of_le e1 e2
      rw [abs_mul, abs_mul, abs_of_pos hE0, abs_of_pos (by unfold A; norm_num : (0:ℝ) < A)]
      have hA : A = 20 := rfl
      rw [hA]
      calc 20 * |2 * pt r p₁ p₂ + θ * (pt r p₁ p₂) ^ 3| * E
          ≤ 20 * (2 * (3 * 10 ^ 8) + 1 / 50 * (3 * 10 ^ 8) ^ 3) * E := by gcongr
        _ ≤ 10 ^ 26 * E := by nlinarith
    have hF'' : Ff'' (pt r p₁ p₂) ≤ 10 ^ 32 * E ∧ 0 ≤ Ff'' (pt r p₁ p₂) := by
      unfold Ff''
      have hA : A = 20 := rfl
      rw [hA, hθ]
      constructor
      · have : 2 + 5 * (1 / 50) * (pt r p₁ p₂) ^ 2 + (1 / 50) ^ 2 * (pt r p₁ p₂) ^ 4 ≤ 10 ^ 32 / 20 := by
          nlinarith
        calc 20 * (2 + 5 * (1 / 50) * (pt r p₁ p₂) ^ 2 + (1 / 50) ^ 2 * (pt r p₁ p₂) ^ 4) * E
            ≤ 20 * (10 ^ 32 / 20) * E := by gcongr
          _ = 10 ^ 32 * E := by ring
      · positivity
    have hF : Ff (pt r p₁ p₂) ≤ 2 * 10 ^ 18 * E ∧ 0 ≤ Ff (pt r p₁ p₂) := by
      unfold Ff
      have hA : A = 20 := rfl
      rw [hA]
      exact ⟨by nlinarith, by positivity⟩
    have hL := hB.abs_Lpt_le (r := r)
    have hd := hB.abs_pt_p₁_le (r := r)
    have hd2 : (pt_p₁ r p₁ p₂) ^ 2 ≤ (2 * 10 ^ 30) ^ 2 := by
      have := pow_le_pow_left₀ (abs_nonneg _) hd 2; rwa [sq_abs] at this
    have hLρ := hB.abs_Lρ_le (r := r)
    have hρ1 := hB.abs_ρ_p₁_le
    have hρ := hB.abs_ρ_le
    -- gate terms
    have g1 : ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)
        ≤ 10 ^ 26 * E * (5 * 10 ^ 55) + 10 ^ 32 * E * (2 * 10 ^ 30) ^ 2 := by
      have e1 : Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ ≤ 10 ^ 26 * E * (5 * 10 ^ 55) := by
        have := abs_mul_le_of_le hF' hL
        linarith [le_abs_self (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂)]
      have e2 : Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2 ≤ 10 ^ 32 * E * (2 * 10 ^ 30) ^ 2 :=
        mul_le_mul hF''.1 hd2 (sq_nonneg _) (by positivity)
      have hsum : Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2
          ≤ 10 ^ 26 * E * (5 * 10 ^ 55) + 10 ^ 32 * E * (2 * 10 ^ 30) ^ 2 := add_le_add e1 e2
      have hρ0 : 0 ≤ ρ p₁ p₂ := Real.smoothTransition.nonneg _
      have hρ1' : ρ p₁ p₂ ≤ 1 := Real.smoothTransition.le_one _
      rcases le_or_gt 0 (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2) with hpos | hneg
      · calc ρ p₁ p₂ * _ ≤ 1 * (10 ^ 26 * E * (5 * 10 ^ 55) + 10 ^ 32 * E * (2 * 10 ^ 30) ^ 2) :=
              mul_le_mul hρ1' hsum hpos (by norm_num)
          _ = _ := one_mul _
      · have : ρ p₁ p₂ * _ ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hρ0 hneg.le
        have : (0:ℝ) ≤ 10 ^ 26 * E * (5 * 10 ^ 55) + 10 ^ 32 * E * (2 * 10 ^ 30) ^ 2 := by positivity
        linarith
    have g2 : Ff (pt r p₁ p₂) * Lρ r p₁ p₂ ≤ 2 * 10 ^ 18 * E * (6 * 10 ^ 50) := by
      have := abs_mul_le_of_le ((abs_of_nonneg hF.2).trans_le hF.1) hLρ
      linarith [le_abs_self (Ff (pt r p₁ p₂) * Lρ r p₁ p₂)]
    have g3 : 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)
        ≤ 2 * (3 * 10 ^ 25) * (10 ^ 26 * E * (2 * 10 ^ 30)) := by
      have := abs_mul_le_of_le (abs_mul_le_of_le (by norm_num : |(2:ℝ)| ≤ 2) hρ1)
        (abs_mul_le_of_le hF' hd)
      linarith [le_abs_self (2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂))]
    -- drift term and φ
    have hT : θ * EH * (1 - (1 - θ) * p₁ ^ 2) ≤ θ * EH := by
      rw [hθ]; nlinarith [sq_nonneg p₁]
    have hφ : φ (V r p₁ p₂) ≤ cRate * (1 + EH + A * (pt r p₁ p₂) ^ 2 * E) / 2 := by
      unfold φ V
      rw [← hE, ← hEH]
      have hρ0 : 0 ≤ ρ p₁ p₂ := Real.smoothTransition.nonneg _
      have hρ1' : ρ p₁ p₂ ≤ 1 := Real.smoothTransition.le_one _
      have hA : A = 20 := rfl
      have hVpos : 0 ≤ A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E := by rw [hA]; positivity
      have hVle : A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E ≤ A * (pt r p₁ p₂) ^ 2 * E := by
        rw [hA]
        have : ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 ≤ 1 * (pt r p₁ p₂) ^ 2 :=
          mul_le_mul_of_nonneg_right hρ1' (sq_nonneg _)
        nlinarith
      have hlog : 0 ≤ log (1 + EH + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E) := log_nonneg (by linarith)
      have hc : cRate = 1 / 20000 := rfl
      have hc0 : 0 ≤ cRate := by rw [hc]; norm_num
      calc cRate * (1 + EH + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E)
            / (2 + log (1 + EH + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E))
          ≤ cRate * (1 + EH + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E) / 2 := by
            apply div_le_div_of_nonneg_left (by rw [hc]; exact mul_nonneg (by norm_num) (by linarith)) (by norm_num) (by linarith)
        _ ≤ cRate * (1 + EH + A * (pt r p₁ p₂) ^ 2 * E) / 2 := by gcongr
    have hc : cRate = 1 / 20000 := rfl
    have hA : A = 20 := rfl
    rw [hc, hA] at hφ
    have hQ2E : 20 * (pt r p₁ p₂) ^ 2 * E ≤ 20 * (3 * 10 ^ 8) ^ 2 * E :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hQ2 (by norm_num)) hE0.le
    have hφ' : φ (V r p₁ p₂) ≤ 1 / 20000 * (1 + EH + 20 * (3 * 10 ^ 8) ^ 2 * E) / 2 := by
      linarith
    rw [hθ] at hT ⊢
    generalize hTt : 1 / 50 * EH * (1 - (1 - 1 / 50) * p₁ ^ 2) = Tt at hT ⊢
    generalize hG1 : ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2) = G1 at g1 ⊢
    generalize hG2 : Ff (pt r p₁ p₂) * Lρ r p₁ p₂ = G2 at g2 ⊢
    generalize hG3 : 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂) = G3 at g3 ⊢
    generalize hF : φ (V r p₁ p₂) = F at hφ' ⊢
    linarith

end RotorFl
