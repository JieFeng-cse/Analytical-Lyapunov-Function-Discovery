/-
# Slow region Ω₁: gate closed (|p₂| < S), negative drift outside K

With `ρ = 0` on a neighbourhood, `V = 1 + exp(θH)` locally, `LH = 1 − p₁²`
(H is conserved by the Hamiltonian part), hence

    L V = θ e^{θH} (1 − (1−θ) p₁²).

Outside K this forces `p₁² > M = 10⁴`, giving `L V ≤ −195·e^{θH}`, while
`φ(V) ≤ c(1+e^{θH})/2`.
-/
import RotorFl.Defs

namespace RotorFl

open Real Filter Topology

/-! ## Elementary facts -/

lemma S_pos (p₁ : ℝ) : 0 < S p₁ := by
  unfold S Rρ; positivity

lemma H_nonneg (r p₁ p₂ : ℝ) : 0 ≤ H r p₁ p₂ := by
  unfold H
  have := cos_le_one r
  nlinarith [sq_nonneg p₁, sq_nonneg p₂]

/-- Gate closed: `|p₂| ≤ S ⇒ ρ = 0`. -/
lemma ρ_eq_zero {p₁ p₂ : ℝ} (h : |p₂| ≤ S p₁) : ρ p₁ p₂ = 0 := by
  unfold ρ χ
  apply Real.smoothTransition.zero_of_nonpos
  have hS := S_pos p₁
  have h2 : p₂ ^ 2 ≤ (S p₁) ^ 2 := by
    nlinarith [abs_nonneg p₂, sq_abs p₂, h]
  have : p₂ ^ 2 / (S p₁) ^ 2 ≤ 1 := by
    rw [div_le_one (by positivity)]; exact h2
  linarith

/-- The certificate with the gate closed. -/
noncomputable def Vs (r p₁ p₂ : ℝ) : ℝ := 1 + exp (θ * H r p₁ p₂)

lemma V_eq_Vs {r p₁ p₂ : ℝ} (h : |p₂| ≤ S p₁) : V r p₁ p₂ = Vs r p₁ p₂ := by
  unfold V Vs
  rw [ρ_eq_zero h]
  ring

/-! ## Derivatives of `Vs` -/

lemma hasDerivAt_H_r (r p₁ p₂ : ℝ) : HasDerivAt (fun t => H t p₁ p₂) (sin r) r := by
  unfold H
  have := (hasDerivAt_cos r).const_sub ((p₁ ^ 2 + p₂ ^ 2) / 2 + 1)
  simpa using this

lemma hasDerivAt_sq (x : ℝ) : HasDerivAt (fun t : ℝ => t ^ 2) (2 * x) x := by
  simpa using hasDerivAt_pow 2 x

lemma hasDerivAt_H_p₁ (r p₁ p₂ : ℝ) : HasDerivAt (fun t => H r t p₂) p₁ p₁ := by
  unfold H
  have h := ((((hasDerivAt_sq p₁).add_const (p₂ ^ 2)).div_const 2).add_const 1).sub_const (cos r)
  have e : 2 * p₁ / 2 = p₁ := by ring
  rw [e] at h
  exact h

lemma hasDerivAt_H_p₂ (r p₁ p₂ : ℝ) : HasDerivAt (fun t => H r p₁ t) p₂ p₂ := by
  unfold H
  have h := ((((hasDerivAt_sq p₂).const_add (p₁ ^ 2)).div_const 2).add_const 1).sub_const (cos r)
  have e : 2 * p₂ / 2 = p₂ := by ring
  rw [e] at h
  exact h

lemma hasDerivAt_Vs_r (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => Vs t p₁ p₂) (exp (θ * H r p₁ p₂) * (θ * sin r)) r := by
  unfold Vs
  exact (((hasDerivAt_H_r r p₁ p₂).const_mul θ).exp).const_add 1

lemma hasDerivAt_Vs_p₁ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => Vs r t p₂) (exp (θ * H r p₁ p₂) * (θ * p₁)) p₁ := by
  unfold Vs
  exact (((hasDerivAt_H_p₁ r p₁ p₂).const_mul θ).exp).const_add 1

lemma hasDerivAt_Vs_p₂ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => Vs r p₁ t) (exp (θ * H r p₁ p₂) * (θ * p₂)) p₂ := by
  unfold Vs
  exact (((hasDerivAt_H_p₂ r p₁ p₂).const_mul θ).exp).const_add 1

lemma deriv_Vs_p₁ (r p₂ : ℝ) :
    deriv (fun t => Vs r t p₂) = fun t => exp (θ * H r t p₂) * (θ * t) :=
  funext fun t => (hasDerivAt_Vs_p₁ r t p₂).deriv

lemma hasDerivAt_deriv_Vs_p₁ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => exp (θ * H r t p₂) * (θ * t))
      (exp (θ * H r p₁ p₂) * (θ * p₁) * (θ * p₁) + exp (θ * H r p₁ p₂) * (θ * 1)) p₁ := by
  have h1 : HasDerivAt (fun t => exp (θ * H r t p₂)) (exp (θ * H r p₁ p₂) * (θ * p₁)) p₁ :=
    ((hasDerivAt_H_p₁ r p₁ p₂).const_mul θ).exp
  have h2 : HasDerivAt (fun t : ℝ => θ * t) (θ * 1) p₁ := (hasDerivAt_id p₁).const_mul θ
  exact h1.mul h2

/-! ## Locality: `V = Vs` near a point of the open slow region -/

lemma eventually_lt_S {p₁ p₂ : ℝ} (h : |p₂| < S p₁) : ∀ᶠ t in 𝓝 p₁, |p₂| < S t := by
  have hS : ContinuousAt S p₁ := by unfold S; fun_prop
  exact continuousAt_const.eventually_lt hS h

lemma eventually_abs_lt {p₁ p₂ : ℝ} (h : |p₂| < S p₁) : ∀ᶠ t in 𝓝 p₂, |t| < S p₁ :=
  continuous_abs.continuousAt.eventually_lt continuousAt_const h

lemma deriv_V_r {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) :
    deriv (fun t => V t p₁ p₂) r = exp (θ * H r p₁ p₂) * (θ * sin r) := by
  have : (fun t => V t p₁ p₂) = fun t => Vs t p₁ p₂ := funext fun t => V_eq_Vs h.le
  rw [this]
  exact (hasDerivAt_Vs_r r p₁ p₂).deriv

lemma deriv_V_p₁ {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) :
    deriv (fun t => V r t p₂) p₁ = exp (θ * H r p₁ p₂) * (θ * p₁) := by
  have hev : (fun t => V r t p₂) =ᶠ[𝓝 p₁] (fun t => Vs r t p₂) := by
    filter_upwards [eventually_lt_S h] with t ht
    exact V_eq_Vs ht.le
  rw [hev.deriv_eq]
  exact (hasDerivAt_Vs_p₁ r p₁ p₂).deriv

lemma deriv_V_p₂ {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) :
    deriv (fun t => V r p₁ t) p₂ = exp (θ * H r p₁ p₂) * (θ * p₂) := by
  have hev : (fun t => V r p₁ t) =ᶠ[𝓝 p₂] (fun t => Vs r p₁ t) := by
    filter_upwards [eventually_abs_lt h] with t ht
    exact V_eq_Vs ht.le
  rw [hev.deriv_eq]
  exact (hasDerivAt_Vs_p₂ r p₁ p₂).deriv

lemma deriv2_V_p₁ {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) :
    deriv (deriv (fun t => V r t p₂)) p₁
      = exp (θ * H r p₁ p₂) * (θ * p₁) * (θ * p₁) + exp (θ * H r p₁ p₂) * (θ * 1) := by
  have hev : deriv (fun t => V r t p₂) =ᶠ[𝓝 p₁] deriv (fun t => Vs r t p₂) := by
    filter_upwards [eventually_lt_S h] with t ht
    have hev' : (fun s => V r s p₂) =ᶠ[𝓝 t] (fun s => Vs r s p₂) := by
      filter_upwards [eventually_lt_S ht] with s hs
      exact V_eq_Vs hs.le
    exact hev'.deriv_eq
  rw [hev.deriv_eq, deriv_Vs_p₁]
  exact (hasDerivAt_deriv_Vs_p₁ r p₁ p₂).deriv

/-! ## The generator on the slow region -/

/-- `L V = θ e^{θH} (1 − (1−θ) p₁²)` where the gate is closed. -/
theorem L_V_slow {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) :
    L V r p₁ p₂ = θ * exp (θ * H r p₁ p₂) * (1 - (1 - θ) * p₁ ^ 2) := by
  unfold L
  rw [deriv_V_r h, deriv_V_p₁ h, deriv_V_p₂ h, deriv2_V_p₁ h]
  ring

/-- **Slow region drift**: gate closed and `p₁² > M` give `L V + φ(V) ≤ 0`. -/
theorem slow_region {r p₁ p₂ : ℝ} (h : |p₂| < S p₁) (hM : Mex < p₁ ^ 2) :
    L V r p₁ p₂ + φ (V r p₁ p₂) ≤ 0 := by
  rw [L_V_slow h, V_eq_Vs h.le]
  unfold Vs φ
  have hE1 : 1 ≤ exp (θ * H r p₁ p₂) := by
    apply Real.one_le_exp
    have := H_nonneg r p₁ p₂
    unfold θ; positivity
  set E := exp (θ * H r p₁ p₂) with hE
  have hlog : 0 ≤ log (1 + E) := Real.log_nonneg (by linarith)
  have hφ : cRate * (1 + E) / (2 + log (1 + E)) ≤ cRate * (1 + E) / 2 := by
    apply div_le_div_of_nonneg_left _ (by norm_num) (by linarith)
    unfold cRate; positivity
  have hL : θ * E * (1 - (1 - θ) * p₁ ^ 2) ≤ θ * E * (1 - (1 - θ) * Mex) := by
    apply mul_le_mul_of_nonneg_left _ (by unfold θ; positivity)
    unfold θ at *; nlinarith
  unfold θ cRate Mex at *
  linarith

end RotorFl
