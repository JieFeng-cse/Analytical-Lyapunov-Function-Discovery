/-
# Fast region Ω₃: gate fully open (|p₂| > 2S)

Part 1 (this file): locality. With `ρ = 1` on a neighbourhood, `V = Vf` where

    Vf = 1 + e^{θH} + A·p̃f²·e^{θ p̃f²/2},
    p̃f = p₂ − t₁/p₂ − t₂/p₂² − t₃/p₂³ + t₄/p₂⁴,

and `L V = L Vf` (all slice derivatives agree).
-/
import RotorFl.Slow

namespace RotorFl

open Real Filter Topology

/-- Gate fully open: `2S ≤ |p₂| ⇒ ρ = 1`. -/
lemma ρ_eq_one {p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : ρ p₁ p₂ = 1 := by
  unfold ρ χ
  apply Real.smoothTransition.one_of_one_le
  have hS := S_pos p₁
  have h2 : 4 * (S p₁) ^ 2 ≤ p₂ ^ 2 := by
    nlinarith [abs_nonneg p₂, sq_abs p₂, h]
  have : 4 ≤ p₂ ^ 2 / (S p₁) ^ 2 := by
    rw [le_div_iff₀ (by positivity)]; linarith
  linarith

/-- `p̃` on the fully open gate (no `ρ`). -/
noncomputable def ptf (r p₁ p₂ : ℝ) : ℝ :=
  p₂ - t₁ r p₁ / p₂ - t₂ r p₁ / p₂ ^ 2 - t₃ r p₁ / p₂ ^ 3 + t₄ r p₁ / p₂ ^ 4

/-- The certificate on the fully open gate. -/
noncomputable def Vf (r p₁ p₂ : ℝ) : ℝ :=
  1 + exp (θ * H r p₁ p₂) + A * (ptf r p₁ p₂) ^ 2 * exp (θ / 2 * (ptf r p₁ p₂) ^ 2)

lemma pt_eq_ptf {r p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : pt r p₁ p₂ = ptf r p₁ p₂ := by
  unfold pt ptf
  rw [ρ_eq_one h]
  ring

lemma V_eq_Vf {r p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : V r p₁ p₂ = Vf r p₁ p₂ := by
  unfold V Vf
  rw [ρ_eq_one h, pt_eq_ptf h]
  ring

/-! ## Locality -/

lemma eventually_gt_2S {p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    ∀ᶠ t in 𝓝 p₁, 2 * S t < |p₂| := by
  have hS : ContinuousAt (fun t => 2 * S t) p₁ := by unfold S; fun_prop
  exact hS.eventually_lt continuousAt_const h

lemma eventually_abs_gt {p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    ∀ᶠ t in 𝓝 p₂, 2 * S p₁ < |t| :=
  continuousAt_const.eventually_lt continuous_abs.continuousAt h

lemma deriv_V_r_fast {r p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    deriv (fun t => V t p₁ p₂) r = deriv (fun t => Vf t p₁ p₂) r := by
  have : (fun t => V t p₁ p₂) = fun t => Vf t p₁ p₂ := funext fun t => V_eq_Vf h.le
  rw [this]

lemma deriv_V_p₁_fast {r p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    deriv (fun t => V r t p₂) p₁ = deriv (fun t => Vf r t p₂) p₁ := by
  apply Filter.EventuallyEq.deriv_eq
  filter_upwards [eventually_gt_2S h] with t ht
  exact V_eq_Vf ht.le

lemma deriv_V_p₂_fast {r p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    deriv (fun t => V r p₁ t) p₂ = deriv (fun t => Vf r p₁ t) p₂ := by
  apply Filter.EventuallyEq.deriv_eq
  filter_upwards [eventually_abs_gt h] with t ht
  exact V_eq_Vf ht.le

lemma deriv2_V_p₁_fast {r p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) :
    deriv (deriv (fun t => V r t p₂)) p₁ = deriv (deriv (fun t => Vf r t p₂)) p₁ := by
  apply Filter.EventuallyEq.deriv_eq
  filter_upwards [eventually_gt_2S h] with t ht
  apply Filter.EventuallyEq.deriv_eq
  filter_upwards [eventually_gt_2S ht] with s hs
  exact V_eq_Vf hs.le

/-- On the open fast region the generator of `V` is that of `Vf`. -/
theorem L_V_eq_L_Vf {r p₁ p₂ : ℝ} (h : 2 * S p₁ < |p₂|) : L V r p₁ p₂ = L Vf r p₁ p₂ := by
  unfold L
  rw [deriv_V_r_fast h, deriv_V_p₁_fast h, deriv_V_p₂_fast h, deriv2_V_p₁_fast h]

end RotorFl
