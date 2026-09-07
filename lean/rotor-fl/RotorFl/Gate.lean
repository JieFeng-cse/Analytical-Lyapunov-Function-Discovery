/-
# The gate profile: explicit derivatives and bounds

`χ(z) = smoothTransition((z−1)/3)`, `smoothTransition x = g x / (g x + g (1−x))`,
`g = expNegInvGlue` (`g x = e^{−1/x}` for `x > 0`, `0` otherwise).

This file provides `ψ' , ψ''` (explicit first and second derivatives of
`ψ = smoothTransition`) as `HasDerivAt` facts, the vanishing of `ψ', ψ''`
outside `(0,1)`, and crude explicit bounds `|ψ'| ≤ 128`, `|ψ''| ≤ 110000`.

Ingredients: `g' = x⁻² g`, `g'' = (x⁻⁴ − 2x⁻³) g`, `g ≤ 1`, `|g'| ≤ 1`,
`|g''| ≤ 48`, denominator `d = g + g(1−·) ∈ [e⁻², 2]`.
-/
import RotorFl.Defs
import Mathlib.Analysis.Complex.ExponentialBounds

namespace RotorFl

open Real Polynomial

local notation "g" => expNegInvGlue

/-! ## `expNegInvGlue` -/

/-- `g' x = x⁻² g x` (also at `x ≤ 0`, where both sides vanish). -/
noncomputable def dg (x : ℝ) : ℝ := x⁻¹ ^ 2 * g x
/-- `g'' x = (x⁻⁴ − 2 x⁻³) g x`. -/
noncomputable def ddg (x : ℝ) : ℝ := (x⁻¹ ^ 4 - 2 * x⁻¹ ^ 3) * g x

lemma hasDerivAt_g (x : ℝ) : HasDerivAt g (dg x) x := by
  have h := expNegInvGlue.hasDerivAt_polynomial_eval_inv_mul (1 : ℝ[X]) x
  simp only [eval_one, one_mul, derivative_one, sub_zero, mul_one, eval_mul, eval_pow, eval_X] at h
  unfold dg
  exact h

lemma hasDerivAt_dg (x : ℝ) : HasDerivAt dg (ddg x) x := by
  have h := expNegInvGlue.hasDerivAt_polynomial_eval_inv_mul (X ^ 2 : ℝ[X]) x
  simp only [eval_mul, eval_pow, eval_X, derivative_X_pow, eval_sub, eval_mul, eval_natCast,
    eval_pow, eval_X, Nat.cast_ofNat] at h
  unfold dg ddg
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_
  · rfl
  · simp only [eval_C, Nat.reduceSub, pow_one]
    ring

lemma g_nonneg (x : ℝ) : 0 ≤ g x := expNegInvGlue.nonneg x

lemma g_le_one (x : ℝ) : g x ≤ 1 := by
  unfold expNegInvGlue
  split_ifs with hx
  · norm_num
  · push_neg at hx
    apply exp_le_one_iff.mpr
    have : 0 < x⁻¹ := inv_pos.mpr hx
    linarith

lemma g_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : g x = 0 := expNegInvGlue.zero_of_nonpos hx

lemma dg_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : dg x = 0 := by
  unfold dg; rw [g_eq_zero_of_nonpos hx, mul_zero]

lemma ddg_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : ddg x = 0 := by
  unfold ddg; rw [g_eq_zero_of_nonpos hx, mul_zero]

/-- `y² ≤ e^y` for `y ≥ 0`. -/
lemma sq_le_exp {y : ℝ} (hy : 0 ≤ y) : y ^ 2 ≤ exp y := by
  have h := quadratic_le_exp_of_nonneg (x := y / 2) (by linarith)
  have h2 : exp y = exp (y / 2) ^ 2 := by
    rw [← exp_nat_mul]; congr 1; push_cast; ring
  rw [h2]
  have hq : y ≤ 1 + y / 2 + (y / 2) ^ 2 / 2 := by nlinarith [sq_nonneg (y - 2)]
  calc y ^ 2 ≤ (1 + y / 2 + (y / 2) ^ 2 / 2) ^ 2 := pow_le_pow_left₀ hy hq 2
    _ ≤ exp (y / 2) ^ 2 := pow_le_pow_left₀ (by positivity) h 2

/-- `y² e^{−y} ≤ 1` for `y ≥ 0`. -/
lemma sq_mul_exp_neg_le {y : ℝ} (hy : 0 ≤ y) : y ^ 2 * exp (-y) ≤ 1 := by
  have h := sq_le_exp hy
  have hpos : 0 < exp y := exp_pos y
  rw [exp_neg]
  calc y ^ 2 * (exp y)⁻¹ ≤ exp y * (exp y)⁻¹ := by gcongr
    _ = 1 := mul_inv_cancel₀ hpos.ne'

/-- `y⁴ e^{−y} ≤ 16` for `y ≥ 0`. -/
lemma pow_four_mul_exp_neg_le {y : ℝ} (hy : 0 ≤ y) : y ^ 4 * exp (-y) ≤ 16 := by
  have h := sq_mul_exp_neg_le (y := y / 2) (by linarith)
  have e : y ^ 4 * exp (-y) = (4 * ((y / 2) ^ 2 * exp (-(y / 2)))) ^ 2 := by
    have : exp (-y) = exp (-(y / 2)) ^ 2 := by
      rw [← exp_nat_mul]; congr 1; push_cast; ring
    rw [this]; ring
  rw [e]
  have h0 : 0 ≤ (y / 2) ^ 2 * exp (-(y / 2)) := by positivity
  calc (4 * ((y / 2) ^ 2 * exp (-(y / 2)))) ^ 2 ≤ (4 * 1) ^ 2 := by gcongr
    _ = 16 := by norm_num

/-- `y³ e^{−y} ≤ 16` for `y ≥ 0`. -/
lemma pow_three_mul_exp_neg_le {y : ℝ} (hy : 0 ≤ y) : y ^ 3 * exp (-y) ≤ 16 := by
  rcases le_or_gt 1 y with h1 | h1
  · have h4 := pow_four_mul_exp_neg_le hy
    have : y ^ 3 ≤ y ^ 4 := by nlinarith [pow_nonneg hy 3]
    have hE : 0 ≤ exp (-y) := (exp_pos _).le
    calc y ^ 3 * exp (-y) ≤ y ^ 4 * exp (-y) := by gcongr
      _ ≤ 16 := h4
  · have hy3 : y ^ 3 ≤ 1 := pow_le_one₀ hy h1.le
    have hE : exp (-y) ≤ 1 := exp_le_one_iff.mpr (by linarith)
    calc y ^ 3 * exp (-y) ≤ 1 * 1 := by gcongr
      _ ≤ 16 := by norm_num

lemma g_eq_exp {x : ℝ} (hx : 0 < x) : g x = exp (-x⁻¹) := by
  unfold expNegInvGlue; rw [if_neg (not_le.mpr hx)]

lemma abs_dg_le (x : ℝ) : |dg x| ≤ 1 := by
  rcases le_or_gt x 0 with hx | hx
  · rw [dg_eq_zero_of_nonpos hx, abs_zero]; norm_num
  · unfold dg
    rw [g_eq_exp hx, abs_of_nonneg (by positivity)]
    exact sq_mul_exp_neg_le (inv_pos.mpr hx).le

lemma abs_ddg_le (x : ℝ) : |ddg x| ≤ 48 := by
  rcases le_or_gt x 0 with hx | hx
  · rw [ddg_eq_zero_of_nonpos hx, abs_zero]; norm_num
  · unfold ddg
    rw [g_eq_exp hx]
    have hy : 0 ≤ x⁻¹ := (inv_pos.mpr hx).le
    have h4 := pow_four_mul_exp_neg_le hy
    have h3 := pow_three_mul_exp_neg_le hy
    have hE : 0 < exp (-x⁻¹) := exp_pos _
    have e : (x⁻¹ ^ 4 - 2 * x⁻¹ ^ 3) * exp (-x⁻¹)
        = x⁻¹ ^ 4 * exp (-x⁻¹) - 2 * (x⁻¹ ^ 3 * exp (-x⁻¹)) := by ring
    rw [e]
    have h4' : 0 ≤ x⁻¹ ^ 4 * exp (-x⁻¹) := by positivity
    have h3' : 0 ≤ x⁻¹ ^ 3 * exp (-x⁻¹) := by positivity
    rw [abs_le]; constructor <;> linarith

/-! ## `smoothTransition` -/

/-- `h x = g (1 − x)`. -/
noncomputable def gh (x : ℝ) : ℝ := g (1 - x)
/-- `d = g + h > 0`. -/
noncomputable def dd (x : ℝ) : ℝ := g x + g (1 - x)

lemma hasDerivAt_gh (x : ℝ) : HasDerivAt gh (-(dg (1 - x))) x := by
  unfold gh
  have h := (hasDerivAt_g (1 - x)).comp x ((hasDerivAt_id x).const_sub 1)
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_
  · rfl
  · simp

lemma dd_pos (x : ℝ) : 0 < dd x := Real.smoothTransition.pos_denom x

lemma dd_le_two (x : ℝ) : dd x ≤ 2 := by
  unfold dd; linarith [g_le_one x, g_le_one (1 - x)]

lemma exp_neg_two_le_dd (x : ℝ) : exp (-2) ≤ dd x := by
  unfold dd
  have hmono := expNegInvGlue.monotone
  have hhalf : g (1 / 2) = exp (-2) := by
    rw [g_eq_exp (by norm_num)]; norm_num
  rcases le_or_gt (1 / 2) x with hx | hx
  · have := hmono hx
    linarith [g_nonneg (1 - x)]
  · have : (1 / 2 : ℝ) ≤ 1 - x := by linarith
    have := hmono this
    linarith [g_nonneg x]

lemma one_div_eight_le_dd (x : ℝ) : 1 / 8 ≤ dd x := by
  have h := exp_neg_two_le_dd x
  have h2 : exp 2 ≤ 8 := by
    have h1 := exp_one_lt_d9
    have : exp 2 = exp 1 ^ 2 := by rw [← exp_nat_mul]; try norm_num
    rw [this]; nlinarith [exp_pos 1]
  have : exp (-2) = (exp 2)⁻¹ := exp_neg 2
  rw [this] at h
  have hpos : 0 < exp 2 := exp_pos 2
  have h3 : 1 / 8 ≤ 1 / exp 2 := one_div_le_one_div_of_le hpos h2
  rw [one_div (exp 2)] at h3
  linarith

lemma hasDerivAt_dd (x : ℝ) : HasDerivAt dd (dg x + -(dg (1 - x))) x :=
  (hasDerivAt_g x).add (hasDerivAt_gh x)

lemma smoothTransition_eq (x : ℝ) : smoothTransition x = g x / dd x := rfl

/-- Numerator of `ψ'`: `N = g' h + g g'(1−·)`. -/
noncomputable def ψN (x : ℝ) : ℝ := dg x * gh x + g x * dg (1 - x)
/-- `N' = g'' h − g g''(1−·)`. -/
noncomputable def ψN' (x : ℝ) : ℝ := ddg x * gh x - g x * ddg (1 - x)
/-- `ψ' = N / d²`. -/
noncomputable def ψ' (x : ℝ) : ℝ := ψN x / (dd x) ^ 2
/-- `ψ'' = N'/d² − 2 N d'/d³`. -/
noncomputable def ψ'' (x : ℝ) : ℝ :=
  ψN' x / (dd x) ^ 2 - 2 * ψN x * (dg x + -(dg (1 - x))) / (dd x) ^ 3

lemma hasDerivAt_smoothTransition (x : ℝ) : HasDerivAt smoothTransition (ψ' x) x := by
  have h := (hasDerivAt_g x).div (hasDerivAt_dd x) (dd_pos x).ne'
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => rfl)).congr_deriv ?_
  unfold ψ' ψN gh dd
  ring

lemma hasDerivAt_ψN (x : ℝ) : HasDerivAt ψN (ψN' x) x := by
  have h1 := (hasDerivAt_dg x).mul (hasDerivAt_gh x)
  have h2 := (hasDerivAt_g x).mul
    ((hasDerivAt_dg (1 - x)).comp x ((hasDerivAt_id x).const_sub 1))
  have h := h1.add h2
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => ?_)).congr_deriv ?_
  · rfl
  · unfold ψN'
    simp only [Function.comp_apply, id_eq]
    ring

lemma hasDerivAt_ψ' (x : ℝ) : HasDerivAt ψ' (ψ'' x) x := by
  have hd : HasDerivAt (fun y => (dd y) ^ 2) (2 * dd x * (dg x + -(dg (1 - x)))) x := by
    have := (hasDerivAt_dd x).pow 2
    refine this.congr_deriv ?_
    push_cast; ring
  have h := (hasDerivAt_ψN x).div hd (pow_pos (dd_pos x) 2).ne'
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => rfl)).congr_deriv ?_
  unfold ψ''
  have hne : dd x ≠ 0 := (dd_pos x).ne'
  field_simp

/-! ### Bounds -/

lemma abs_gh_le (x : ℝ) : |gh x| ≤ 1 := by
  unfold gh; rw [abs_of_nonneg (g_nonneg _)]; exact g_le_one _

lemma abs_g_le (x : ℝ) : |g x| ≤ 1 := by
  rw [abs_of_nonneg (g_nonneg _)]; exact g_le_one _

lemma abs_ψN_le (x : ℝ) : |ψN x| ≤ 2 := by
  unfold ψN
  calc |dg x * gh x + g x * dg (1 - x)| ≤ |dg x * gh x| + |g x * dg (1 - x)| := abs_add_le _ _
    _ = |dg x| * |gh x| + |g x| * |dg (1 - x)| := by rw [abs_mul, abs_mul]
    _ ≤ 1 * 1 + 1 * 1 := by
        gcongr
        · exact abs_dg_le x
        · exact abs_gh_le x
        · exact abs_g_le x
        · exact abs_dg_le _
    _ = 2 := by norm_num

lemma abs_ψN'_le (x : ℝ) : |ψN' x| ≤ 96 := by
  unfold ψN'
  calc |ddg x * gh x - g x * ddg (1 - x)| ≤ |ddg x * gh x| + |g x * ddg (1 - x)| := abs_sub _ _
    _ = |ddg x| * |gh x| + |g x| * |ddg (1 - x)| := by rw [abs_mul, abs_mul]
    _ ≤ 48 * 1 + 1 * 48 := by
        gcongr
        · exact abs_ddg_le x
        · exact abs_gh_le x
        · exact abs_g_le x
        · exact abs_ddg_le _
    _ = 96 := by norm_num

lemma abs_dd'_le (x : ℝ) : |dg x + -(dg (1 - x))| ≤ 2 := by
  calc |dg x + -(dg (1 - x))| ≤ |dg x| + |-(dg (1 - x))| := abs_add_le _ _
    _ = |dg x| + |dg (1 - x)| := by rw [abs_neg]
    _ ≤ 1 + 1 := add_le_add (abs_dg_le x) (abs_dg_le _)
    _ = 2 := by norm_num

lemma abs_ψ'_le (x : ℝ) : |ψ' x| ≤ 128 := by
  unfold ψ'
  have hd := one_div_eight_le_dd x
  have hd0 := dd_pos x
  rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < (dd x) ^ 2)]
  rw [div_le_iff₀ (by positivity)]
  have h1 : (1 / 8 : ℝ) ^ 2 ≤ (dd x) ^ 2 := pow_le_pow_left₀ (by norm_num) hd 2
  have := abs_ψN_le x
  nlinarith

lemma abs_ψ''_le (x : ℝ) : |ψ'' x| ≤ 11000 := by
  unfold ψ''
  have hd := one_div_eight_le_dd x
  have hd0 := dd_pos x
  have h2 : (1 / 8 : ℝ) ^ 2 ≤ (dd x) ^ 2 := pow_le_pow_left₀ (by norm_num) hd 2
  have h3 : (1 / 8 : ℝ) ^ 3 ≤ (dd x) ^ 3 := pow_le_pow_left₀ (by norm_num) hd 3
  have hN := abs_ψN_le x
  have hN' := abs_ψN'_le x
  have hd' := abs_dd'_le x
  have t1 : |ψN' x / (dd x) ^ 2| ≤ 96 * 64 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < (dd x) ^ 2), div_le_iff₀ (by positivity)]
    nlinarith
  have t2 : |2 * ψN x * (dg x + -(dg (1 - x))) / (dd x) ^ 3| ≤ 2 * 2 * 2 * 512 := by
    rw [abs_div, abs_of_pos (by positivity : (0:ℝ) < (dd x) ^ 3), div_le_iff₀ (by positivity)]
    rw [abs_mul, abs_mul, abs_two]
    have hN0 := abs_nonneg (ψN x)
    have hd'0 := abs_nonneg (dg x + -(dg (1 - x)))
    have : |ψN x| * |dg x + -(dg (1 - x))| ≤ 2 * 2 := mul_le_mul hN hd' hd'0 (by norm_num)
    nlinarith
  calc |ψN' x / (dd x) ^ 2 - 2 * ψN x * (dg x + -(dg (1 - x))) / (dd x) ^ 3|
      ≤ |ψN' x / (dd x) ^ 2| + |2 * ψN x * (dg x + -(dg (1 - x))) / (dd x) ^ 3| := abs_sub _ _
    _ ≤ 96 * 64 + 2 * 2 * 2 * 512 := add_le_add t1 t2
    _ ≤ 11000 := by norm_num

/-! ### Vanishing outside `(0, 1)` -/

lemma ψN_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : ψN x = 0 := by
  unfold ψN; rw [dg_eq_zero_of_nonpos hx, g_eq_zero_of_nonpos hx]; ring

lemma ψN'_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : ψN' x = 0 := by
  unfold ψN'; rw [ddg_eq_zero_of_nonpos hx, g_eq_zero_of_nonpos hx]; ring

lemma ψN_eq_zero_of_one_le {x : ℝ} (hx : 1 ≤ x) : ψN x = 0 := by
  unfold ψN gh
  rw [dg_eq_zero_of_nonpos (x := 1 - x) (by linarith), g_eq_zero_of_nonpos (x := 1 - x) (by linarith)]
  ring

lemma ψN'_eq_zero_of_one_le {x : ℝ} (hx : 1 ≤ x) : ψN' x = 0 := by
  unfold ψN' gh
  rw [ddg_eq_zero_of_nonpos (x := 1 - x) (by linarith), g_eq_zero_of_nonpos (x := 1 - x) (by linarith)]
  ring

lemma ψ'_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : ψ' x = 0 := by
  unfold ψ'; rw [ψN_eq_zero_of_nonpos hx]; simp

lemma ψ'_eq_zero_of_one_le {x : ℝ} (hx : 1 ≤ x) : ψ' x = 0 := by
  unfold ψ'; rw [ψN_eq_zero_of_one_le hx]; simp

lemma ψ''_eq_zero_of_nonpos {x : ℝ} (hx : x ≤ 0) : ψ'' x = 0 := by
  unfold ψ''; rw [ψN_eq_zero_of_nonpos hx, ψN'_eq_zero_of_nonpos hx]; simp

lemma ψ''_eq_zero_of_one_le {x : ℝ} (hx : 1 ≤ x) : ψ'' x = 0 := by
  unfold ψ''; rw [ψN_eq_zero_of_one_le hx, ψN'_eq_zero_of_one_le hx]; simp

end RotorFl
