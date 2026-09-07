/-
# Transition band Ω₂ — part 2: the drift inequality

For a `TransPt` (`S ≤ |p₂| ≤ 2S`, `|p₁| ≥ 100`), with `E = e^{θ p̃²/2}`, `P = 2S + 1`:

* `|∂ᵣp̃| ≤ 1.01|u|`, `|∂ₚ₁p̃| ≤ 15|u|`, `|∂ₚ₂p̃| ≤ 1.001`, `|∂²ₚ₁p̃| ≤ 131|u|`, `|L p̃| ≤ 2.02`;
* `|p̃| ≤ P`, gate terms `≤ 5200 S³ E`;
* `θH ≥ θp̃²/2 + θ(p₁²/2 − 1.006)`, so `E ≤ e^{θH} e^{−0.01 p₁² + 0.0202}`;
* `φ(V) ≤ 10⁻¹⁸ e^{θH} + 10⁻¹⁶ S² E`;
* the damping `θ e^{θH}(1 − 0.98 p₁²)` wins because `e^{0.01 a²} ≥ 3·10⁵ a¹⁰` for `a ≥ 100`.
-/
import RotorFl.TransEst
import Mathlib.Analysis.Complex.ExponentialBounds

namespace RotorFl

open Real

namespace TransPt

variable {r p₁ p₂ : ℝ} (h : TransPt r p₁ p₂)
include h

lemma ρ_mem : 0 ≤ ρ p₁ p₂ ∧ ρ p₁ p₂ ≤ 1 :=
  ⟨Real.smoothTransition.nonneg _, Real.smoothTransition.le_one _⟩

lemma abs_ρ_le : |ρ p₁ p₂| ≤ 1 := by
  rw [abs_of_nonneg h.ρ_mem.1]; exact h.ρ_mem.2

lemma abs_pt_r_le : |pt_r r p₁ p₂| ≤ (1 + 1 / 100) * |1 / p₂| := by
  unfold pt_r
  rw [abs_mul]
  have h1 := h.abs_ρ_le
  have h2 := abs_ptf_r_le (r := r) h.smallU h.p₂_ne
  calc |ρ p₁ p₂| * |ptf_r r p₁ p₂| ≤ 1 * ((1 + 1 / 100) * |1 / p₂|) := by gcongr
    _ = (1 + 1 / 100) * |1 / p₂| := by ring

lemma abs_pt_p₁_le : |pt_p₁ r p₁ p₂| ≤ 15 * |1 / p₂| := by
  unfold pt_p₁
  have h1 := h.abs_ρ_p₁_le
  have h2 := abs_Tf_le (r := r) h.smallU h.p₂_ne
  have h3 := h.abs_ρ_le
  have h4 := abs_ptf_p₁_le (r := r) h.smallU h.p₂_ne
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  have hsq : (1 / p₂) ^ 2 = |1 / p₂| ^ 2 := (sq_abs _).symm
  calc |-(ρ_p₁ p₁ p₂ * Tf r p₁ p₂) + ρ p₁ p₂ * ptf_p₁ r p₁ p₂|
      ≤ |-(ρ_p₁ p₁ p₂ * Tf r p₁ p₂)| + |ρ p₁ p₂ * ptf_p₁ r p₁ p₂| := abs_add_le _ _
    _ = |ρ_p₁ p₁ p₂| * |Tf r p₁ p₂| + |ρ p₁ p₂| * |ptf_p₁ r p₁ p₂| := by
        rw [abs_neg, abs_mul, abs_mul]
    _ ≤ 14 * ((1 + 1 / 200) * |1 / p₂|) + 1 * ((1 + 1 / 100) * (1 / p₂) ^ 2) := by gcongr
    _ = 14 * (1 + 1 / 200) * |1 / p₂| + (1 + 1 / 100) * |1 / p₂| ^ 2 := by rw [hsq]; ring
    _ ≤ 15 * |1 / p₂| := by nlinarith

lemma abs_pt_p₂_le : |pt_p₂ r p₁ p₂| ≤ 1 + 1 / 1000 := by
  unfold pt_p₂
  have h1 := h.abs_ρ_p₂_le
  have h2 := abs_Tf_le (r := r) h.smallU h.p₂_ne
  have h3 := h.abs_ρ_le
  have h4 := abs_ptf_p₂_sub_one_le (r := r) h.smallU h.p₂_ne
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  have hsq : (1 / p₂) ^ 2 = |1 / p₂| ^ 2 := (sq_abs _).symm
  have e : 1 - ρ_p₂ p₁ p₂ * Tf r p₁ p₂ - ρ p₁ p₂ * (1 - ptf_p₂ r p₁ p₂)
      = 1 - ρ_p₂ p₁ p₂ * Tf r p₁ p₂ + ρ p₁ p₂ * (ptf_p₂ r p₁ p₂ - 1) := by ring
  rw [e]
  calc |1 - ρ_p₂ p₁ p₂ * Tf r p₁ p₂ + ρ p₁ p₂ * (ptf_p₂ r p₁ p₂ - 1)|
      ≤ |1 - ρ_p₂ p₁ p₂ * Tf r p₁ p₂| + |ρ p₁ p₂ * (ptf_p₂ r p₁ p₂ - 1)| := abs_add_le _ _
    _ ≤ |(1 : ℝ)| + |ρ_p₂ p₁ p₂ * Tf r p₁ p₂| + |ρ p₁ p₂ * (ptf_p₂ r p₁ p₂ - 1)| := by
        gcongr; exact abs_sub _ _
    _ = 1 + |ρ_p₂ p₁ p₂| * |Tf r p₁ p₂| + |ρ p₁ p₂| * |ptf_p₂ r p₁ p₂ - 1| := by
        rw [abs_one, abs_mul, abs_mul]
    _ ≤ 1 + 1 / 10 ^ 5 * ((1 + 1 / 200) * |1 / p₂|) + 1 * ((1 + 1 / 100) * (1 / p₂) ^ 2) := by
        gcongr
    _ ≤ 1 + 1 / 1000 := by
        rw [hsq]
        have : |1 / p₂| ^ 2 ≤ (1 / 10 ^ 8) ^ 2 := pow_le_pow_left₀ hv0 hv 2
        nlinarith

lemma abs_pt_p₁p₁_le : |pt_p₁p₁ r p₁ p₂| ≤ 131 * |1 / p₂| := by
  unfold pt_p₁p₁
  have h1 := h.abs_ρ_p₁p₁_le
  have h2 := abs_Tf_le (r := r) h.smallU h.p₂_ne
  have h3 := h.abs_ρ_p₁_le
  have h4 := abs_ptf_p₁_le (r := r) h.smallU h.p₂_ne
  have h5 := h.abs_ρ_le
  have h6 := abs_ptf_p₁p₁_le (r := r) h.smallU h.p₂_ne
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  have hsq : (1 / p₂) ^ 2 = |1 / p₂| ^ 2 := (sq_abs _).symm
  calc |-(ρ_p₁p₁ p₁ p₂ * Tf r p₁ p₂) + 2 * ρ_p₁ p₁ p₂ * ptf_p₁ r p₁ p₂ + ρ p₁ p₂ * ptf_p₁p₁ r p₁ p₂|
      ≤ |-(ρ_p₁p₁ p₁ p₂ * Tf r p₁ p₂)| + |2 * ρ_p₁ p₁ p₂ * ptf_p₁ r p₁ p₂|
          + |ρ p₁ p₂ * ptf_p₁p₁ r p₁ p₂| := abs_add_three _ _ _
    _ = |ρ_p₁p₁ p₁ p₂| * |Tf r p₁ p₂| + 2 * |ρ_p₁ p₁ p₂| * |ptf_p₁ r p₁ p₂|
          + |ρ p₁ p₂| * |ptf_p₁p₁ r p₁ p₂| := by
        rw [abs_neg, abs_mul, abs_mul, abs_mul, abs_mul, abs_two]
    _ ≤ 130 * ((1 + 1 / 200) * |1 / p₂|) + 2 * 14 * ((1 + 1 / 100) * (1 / p₂) ^ 2)
          + 1 * ((2 + 3 / 100) * |1 / p₂| ^ 3) := by gcongr
    _ ≤ 131 * |1 / p₂| := by
        rw [hsq]
        have h2' : |1 / p₂| ^ 2 ≤ (1 / 10 ^ 8) * |1 / p₂| := by
          have := mul_le_mul_of_nonneg_right hv hv0; nlinarith
        have h3' : |1 / p₂| ^ 3 ≤ (1 / 10 ^ 8) * |1 / p₂| := by
          have : |1 / p₂| ^ 2 ≤ (1 / 10 ^ 8) ^ 2 := pow_le_pow_left₀ hv0 hv 2
          have := mul_le_mul_of_nonneg_right this hv0; nlinarith
        nlinarith

lemma abs_Lpt_le : |Lpt r p₁ p₂| ≤ 202 / 100 := by
  unfold Lpt
  have h1 := h.abs_pt_r_le
  have h2 := h.abs_pt_p₁_le
  have h3 := h.abs_pt_p₂_le
  have h4 := h.abs_pt_p₁p₁_le
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  have hau := h.a_mul_u_le
  have hne := h.p₂_ne
  have hup : |1 / p₂| * |p₂| = 1 := by
    rw [abs_one_div, one_div_mul_cancel (abs_pos.mpr hne).ne']
  have hs1 : |p₂ - p₁| ≤ |p₂| + |p₁| := abs_sub _ _
  have hs2 : |sin r - p₁| ≤ 1 + |p₁| := by
    calc |sin r - p₁| ≤ |sin r| + |p₁| := abs_sub _ _
      _ ≤ 1 + |p₁| := by gcongr; exact abs_sin_le_one r
  have hs3 : |-sin r| ≤ 1 := by rw [abs_neg]; exact abs_sin_le_one r
  have ha0 := abs_nonneg p₁
  calc |pt_r r p₁ p₂ * (p₂ - p₁) + pt_p₁ r p₁ p₂ * (sin r - p₁) + pt_p₂ r p₁ p₂ * (-sin r)
        + pt_p₁p₁ r p₁ p₂|
      ≤ |pt_r r p₁ p₂ * (p₂ - p₁) + pt_p₁ r p₁ p₂ * (sin r - p₁) + pt_p₂ r p₁ p₂ * (-sin r)|
          + |pt_p₁p₁ r p₁ p₂| := abs_add_le _ _
    _ ≤ |pt_r r p₁ p₂ * (p₂ - p₁)| + |pt_p₁ r p₁ p₂ * (sin r - p₁)| + |pt_p₂ r p₁ p₂ * (-sin r)|
          + |pt_p₁p₁ r p₁ p₂| := by gcongr; exact abs_add_three _ _ _
    _ = |pt_r r p₁ p₂| * |p₂ - p₁| + |pt_p₁ r p₁ p₂| * |sin r - p₁| + |pt_p₂ r p₁ p₂| * |-sin r|
          + |pt_p₁p₁ r p₁ p₂| := by rw [abs_mul, abs_mul, abs_mul]
    _ ≤ (1 + 1 / 100) * |1 / p₂| * (|p₂| + |p₁|) + 15 * |1 / p₂| * (1 + |p₁|)
          + (1 + 1 / 1000) * 1 + 131 * |1 / p₂| := by gcongr
    _ = (1 + 1 / 100) * (|1 / p₂| * |p₂|) + (1 + 1 / 100) * (|p₁| * |1 / p₂|)
          + 15 * |1 / p₂| + 15 * (|p₁| * |1 / p₂|) + (1 + 1 / 1000) + 131 * |1 / p₂| := by ring
    _ ≤ 202 / 100 := by rw [hup]; nlinarith

/-! ### Size of `p̃` and the exponent relation -/

lemma abs_pt_le : |pt r p₁ p₂| ≤ 2 * S p₁ + 1 := by
  rw [pt_eq_Tf]
  have h1 := h.abs_ρ_le
  have h2 := abs_Tf_le (r := r) h.smallU h.p₂_ne
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  calc |p₂ - ρ p₁ p₂ * Tf r p₁ p₂| ≤ |p₂| + |ρ p₁ p₂ * Tf r p₁ p₂| := abs_sub _ _
    _ = |p₂| + |ρ p₁ p₂| * |Tf r p₁ p₂| := by rw [abs_mul]
    _ ≤ 2 * S p₁ + 1 * ((1 + 1 / 200) * |1 / p₂|) := by gcongr; exact h.hi
    _ ≤ 2 * S p₁ + 1 := by nlinarith

/-- `H ≥ p̃²/2 + p₁²/2 − 1.006`. -/
lemma H_ge : (pt r p₁ p₂) ^ 2 / 2 + p₁ ^ 2 / 2 - 1006 / 1000 ≤ H r p₁ p₂ := by
  have h1 := h.ρ_mem
  have h2 := abs_le.mp (abs_Tf_le (r := r) h.smallU h.p₂_ne)
  have hv := h.abs_u_le'
  have hv0 := abs_nonneg (1 / p₂)
  have hne := h.p₂_ne
  have hup : |1 / p₂| * |p₂| = 1 := by
    rw [abs_one_div, one_div_mul_cancel (abs_pos.mpr hne).ne']
  -- p₂ T is bounded by 1.005
  have hpT : |p₂ * Tf r p₁ p₂| ≤ 1 + 1 / 200 := by
    rw [abs_mul]
    calc |p₂| * |Tf r p₁ p₂| ≤ |p₂| * ((1 + 1 / 200) * |1 / p₂|) := by
          gcongr; exact abs_le.mpr h2
      _ = (1 + 1 / 200) * (|1 / p₂| * |p₂|) := by ring
      _ = 1 + 1 / 200 := by rw [hup]; ring
  have hT2 : (Tf r p₁ p₂) ^ 2 ≤ 1 / 10 ^ 8 := by
    have : |Tf r p₁ p₂| ≤ (1 + 1 / 200) * (1 / 10 ^ 8) := by
      calc |Tf r p₁ p₂| ≤ (1 + 1 / 200) * |1 / p₂| := abs_le.mpr h2
        _ ≤ (1 + 1 / 200) * (1 / 10 ^ 8) := by gcongr
    have := pow_le_pow_left₀ (abs_nonneg _) this 2
    rw [sq_abs] at this
    nlinarith
  have e : H r p₁ p₂ = (pt r p₁ p₂) ^ 2 / 2 + p₁ ^ 2 / 2 + (1 - cos r)
      + ρ p₁ p₂ * (p₂ * Tf r p₁ p₂) - (ρ p₁ p₂) ^ 2 * (Tf r p₁ p₂) ^ 2 / 2 := by
    rw [pt_eq_Tf]; unfold H; ring
  have hc := cos_le_one r
  have hpT' := abs_le.mp hpT
  have hρ2 : (ρ p₁ p₂) ^ 2 ≤ 1 := pow_le_one₀ h1.1 h1.2
  rw [e]
  nlinarith [mul_le_mul_of_nonneg_left hpT'.1 h1.1, mul_le_mul hρ2 hT2 (by positivity) (by norm_num)]

end TransPt

end RotorFl
