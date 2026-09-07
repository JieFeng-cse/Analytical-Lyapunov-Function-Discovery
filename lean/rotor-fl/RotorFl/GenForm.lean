/-
# The generator of `V` in general (gate partially open)

`V = Vs + ρ · Ff(p̃)`, `p̃ = p₂ − ρ T`, `T = t₁/p₂ + t₂/p₂² + t₃/p₂³ − t₄/p₂⁴ = p₂ − p̃f`.

    L V = θ e^{θH}(1 − (1−θ)p₁²)
          + ρ (Ff'(p̃) L p̃ + Ff''(p̃) (∂ₚ₁p̃)²) + Ff(p̃) L ρ + 2 ∂ₚ₁ρ Ff'(p̃) ∂ₚ₁p̃

with the slice derivatives of `p̃` and `ρ` written out. Where the gate is
saturated (`2S ≤ |p₂|`) all `ρ`-derivatives vanish and `L V = L Vf`.
-/
import RotorFl.RhoDeriv
import RotorFl.FastGen

namespace RotorFl

open Real

/-! ## Product rule for `L` -/

theorem L_mul {f g : ℝ → ℝ → ℝ → ℝ} {r p₁ p₂ : ℝ}
    {a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ} {e₁ e₂ : ℝ → ℝ}
    (h1r : HasDerivAt (fun t => f t p₁ p₂) a₁ r)
    (h1p₁ : ∀ t, HasDerivAt (fun s => f r s p₂) (e₁ t) t)
    (h1p₂ : HasDerivAt (fun t => f r p₁ t) c₁ p₂)
    (h1e : HasDerivAt e₁ d₁ p₁)
    (h2r : HasDerivAt (fun t => g t p₁ p₂) a₂ r)
    (h2p₁ : ∀ t, HasDerivAt (fun s => g r s p₂) (e₂ t) t)
    (h2p₂ : HasDerivAt (fun t => g r p₁ t) c₂ p₂)
    (h2e : HasDerivAt e₂ d₂ p₁)
    (hb₁ : e₁ p₁ = b₁) (hb₂ : e₂ p₁ = b₂) :
    L (fun a b c => f a b c * g a b c) r p₁ p₂
      = f r p₁ p₂ * (a₂ * (p₂ - p₁) + b₂ * (sin r - p₁) + c₂ * (-sin r) + d₂)
        + g r p₁ p₂ * (a₁ * (p₂ - p₁) + b₁ * (sin r - p₁) + c₁ * (-sin r) + d₁)
        + 2 * b₁ * b₂ := by
  unfold L
  have d1 : deriv (fun t => f t p₁ p₂ * g t p₁ p₂) r = a₁ * g r p₁ p₂ + f r p₁ p₂ * a₂ :=
    (h1r.mul h2r).deriv
  have d2 : deriv (fun t => f r t p₂ * g r t p₂) = fun t => e₁ t * g r t p₂ + f r t p₂ * e₂ t :=
    funext fun t => ((h1p₁ t).mul (h2p₁ t)).deriv
  have d3 : deriv (fun t => f r p₁ t * g r p₁ t) p₂ = c₁ * g r p₁ p₂ + f r p₁ p₂ * c₂ :=
    (h1p₂.mul h2p₂).deriv
  have d4 : deriv (fun t => e₁ t * g r t p₂ + f r t p₂ * e₂ t) p₁
      = (d₁ * g r p₁ p₂ + e₁ p₁ * e₂ p₁) + (e₁ p₁ * e₂ p₁ + f r p₁ p₂ * d₂) :=
    ((h1e.mul (h2p₁ p₁)).add ((h1p₁ p₁).mul h2e)).deriv
  simp only []
  rw [d1, d2, d3, d4]
  subst hb₁ hb₂
  ring

/-! ## `T` and the slice derivatives of `p̃` -/

/-- `T = t₁/p₂ + t₂/p₂² + t₃/p₂³ − t₄/p₂⁴`, so `p̃ = p₂ − ρ T` and `p̃f = p₂ − T`. -/
noncomputable def Tf (r p₁ p₂ : ℝ) : ℝ :=
  t₁ r p₁ / p₂ + t₂ r p₁ / p₂ ^ 2 + t₃ r p₁ / p₂ ^ 3 - t₄ r p₁ / p₂ ^ 4

lemma Tf_eq (r p₁ p₂ : ℝ) : Tf r p₁ p₂ = p₂ - ptf r p₁ p₂ := by unfold Tf ptf; ring

lemma pt_eq_Tf (r p₁ p₂ : ℝ) : pt r p₁ p₂ = p₂ - ρ p₁ p₂ * Tf r p₁ p₂ := rfl

lemma hasDerivAt_Tf_r (r p₁ p₂ : ℝ) : HasDerivAt (fun t => Tf t p₁ p₂) (-(ptf_r r p₁ p₂)) r := by
  have : (fun t => Tf t p₁ p₂) = fun t => p₂ - ptf t p₁ p₂ := funext fun t => Tf_eq t p₁ p₂
  rw [this]; exact (hasDerivAt_ptf_r r p₁ p₂).const_sub p₂

lemma hasDerivAt_Tf_p₁ (r p₁ p₂ : ℝ) : HasDerivAt (fun t => Tf r t p₂) (-(ptf_p₁ r p₁ p₂)) p₁ := by
  have : (fun t => Tf r t p₂) = fun t => p₂ - ptf r t p₂ := funext fun t => Tf_eq r t p₂
  rw [this]; exact (hasDerivAt_ptf_p₁ r p₁ p₂).const_sub p₂

lemma hasDerivAt_Tf_p₂ (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    HasDerivAt (fun t => Tf r p₁ t) (1 - ptf_p₂ r p₁ p₂) p₂ := by
  have : (fun t => Tf r p₁ t) = fun t => t - ptf r p₁ t := funext fun t => Tf_eq r p₁ t
  rw [this]; exact (hasDerivAt_id' p₂).sub (hasDerivAt_ptf_p₂ r p₁ p₂ h₂)

/-- ∂ᵣ p̃ = ρ ∂ᵣp̃f -/
noncomputable def pt_r (r p₁ p₂ : ℝ) : ℝ := ρ p₁ p₂ * ptf_r r p₁ p₂
/-- ∂ₚ₁ p̃ = −∂ₚ₁ρ · T + ρ ∂ₚ₁p̃f -/
noncomputable def pt_p₁ (r p₁ p₂ : ℝ) : ℝ := -(ρ_p₁ p₁ p₂ * Tf r p₁ p₂) + ρ p₁ p₂ * ptf_p₁ r p₁ p₂
/-- ∂ₚ₂ p̃ = 1 − ∂ₚ₂ρ · T − ρ (1 − ∂ₚ₂p̃f) -/
noncomputable def pt_p₂ (r p₁ p₂ : ℝ) : ℝ :=
  1 - ρ_p₂ p₁ p₂ * Tf r p₁ p₂ - ρ p₁ p₂ * (1 - ptf_p₂ r p₁ p₂)
/-- ∂²ₚ₁ p̃ = −∂²ₚ₁ρ · T + 2 ∂ₚ₁ρ ∂ₚ₁p̃f + ρ ∂²ₚ₁p̃f -/
noncomputable def pt_p₁p₁ (r p₁ p₂ : ℝ) : ℝ :=
  -(ρ_p₁p₁ p₁ p₂ * Tf r p₁ p₂) + 2 * ρ_p₁ p₁ p₂ * ptf_p₁ r p₁ p₂ + ρ p₁ p₂ * ptf_p₁p₁ r p₁ p₂

lemma hasDerivAt_pt_r (r p₁ p₂ : ℝ) : HasDerivAt (fun t => pt t p₁ p₂) (pt_r r p₁ p₂) r := by
  have h := ((hasDerivAt_Tf_r r p₁ p₂).const_mul (ρ p₁ p₂)).const_sub p₂
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => rfl)).congr_deriv ?_
  unfold pt_r; ring

lemma hasDerivAt_pt_p₁ (r p₁ p₂ : ℝ) : HasDerivAt (fun t => pt r t p₂) (pt_p₁ r p₁ p₂) p₁ := by
  have h := ((hasDerivAt_ρ_p₁ p₁ p₂).mul (hasDerivAt_Tf_p₁ r p₁ p₂)).const_sub p₂
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => rfl)).congr_deriv ?_
  unfold pt_p₁; ring

lemma hasDerivAt_pt_p₂ (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    HasDerivAt (fun t => pt r p₁ t) (pt_p₂ r p₁ p₂) p₂ := by
  have h := (hasDerivAt_id' p₂).sub ((hasDerivAt_ρ_p₂ p₁ p₂).mul (hasDerivAt_Tf_p₂ r p₁ p₂ h₂))
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => rfl)).congr_deriv ?_
  unfold pt_p₂; ring

lemma hasDerivAt_pt_p₁_p₁ (r p₁ p₂ : ℝ) :
    HasDerivAt (fun t => pt_p₁ r t p₂) (pt_p₁p₁ r p₁ p₂) p₁ := by
  have h := (((hasDerivAt_ρ_p₁_p₁ p₁ p₂).mul (hasDerivAt_Tf_p₁ r p₁ p₂)).neg).add
    ((hasDerivAt_ρ_p₁ p₁ p₂).mul (hasDerivAt_ptf_p₁p₁ r p₁ p₂))
  refine (h.congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => rfl)).congr_deriv ?_
  unfold pt_p₁p₁; ring

/-! ## The general formula -/

/-- `L p̃` written out. -/
noncomputable def Lpt (r p₁ p₂ : ℝ) : ℝ :=
  pt_r r p₁ p₂ * (p₂ - p₁) + pt_p₁ r p₁ p₂ * (sin r - p₁) + pt_p₂ r p₁ p₂ * (-sin r)
    + pt_p₁p₁ r p₁ p₂

/-- `L ρ` written out (`ρ` does not depend on `r`). -/
noncomputable def Lρ (r p₁ p₂ : ℝ) : ℝ :=
  ρ_p₁ p₁ p₂ * (sin r - p₁) + ρ_p₂ p₁ p₂ * (-sin r) + ρ_p₁p₁ p₁ p₂

theorem L_V_general (r p₁ p₂ : ℝ) (h₂ : p₂ ≠ 0) :
    L V r p₁ p₂ = θ * exp (θ * H r p₁ p₂) * (1 - (1 - θ) * p₁ ^ 2)
      + (ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)
        + Ff (pt r p₁ p₂) * Lρ r p₁ p₂
        + 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)) := by
  have hV : V = fun a b c => Vs a b c + ρ b c * Ff (pt a b c) := by
    funext a b c
    unfold V Vs Ff
    ring
  rw [hV, L_add (hasDerivAt_Vs_r r p₁ p₂) (fun t => hasDerivAt_Vs_p₁ r t p₂)
        (hasDerivAt_Vs_p₂ r p₁ p₂) (hasDerivAt_deriv_Vs_p₁ r p₁ p₂)
        (V₂ := fun a b c => ρ b c * Ff (pt a b c))
        (e₂ := fun t => ρ_p₁ t p₂ * Ff (pt r t p₂) + ρ t p₂ * (Ff' (pt r t p₂) * pt_p₁ r t p₂))
        ((hasDerivAt_const r (ρ p₁ p₂)).mul ((hasDerivAt_Ff _).comp r (hasDerivAt_pt_r r p₁ p₂)))
        (fun t => (hasDerivAt_ρ_p₁ t p₂).mul ((hasDerivAt_Ff _).comp t (hasDerivAt_pt_p₁ r t p₂)))
        ((hasDerivAt_ρ_p₂ p₁ p₂).mul ((hasDerivAt_Ff _).comp p₂ (hasDerivAt_pt_p₂ r p₁ p₂ h₂)))
        (((hasDerivAt_ρ_p₁_p₁ p₁ p₂).mul ((hasDerivAt_Ff _).comp p₁ (hasDerivAt_pt_p₁ r p₁ p₂))).add
          ((hasDerivAt_ρ_p₁ p₁ p₂).mul
            (((hasDerivAt_Ff' _).comp p₁ (hasDerivAt_pt_p₁ r p₁ p₂)).mul (hasDerivAt_pt_p₁_p₁ r p₁ p₂))))
        rfl rfl]
  unfold Lpt Lρ
  simp only [Function.comp_apply]
  ring

/-! ## Saturated gate: `L V = L Vf` for `2S ≤ |p₂|` (boundary included) -/

theorem L_V_eq_L_Vf_of_le {r p₁ p₂ : ℝ} (h : 2 * S p₁ ≤ |p₂|) : L V r p₁ p₂ = L Vf r p₁ p₂ := by
  have h₂ : p₂ ≠ 0 := by
    intro h0; rw [h0, abs_zero] at h; have := S_pos p₁; linarith
  rw [L_V_general r p₁ p₂ h₂, L_Vf r p₁ p₂ h₂]
  have hρ := ρ_eq_one h
  have h1 := ρ_p₁_eq_zero_of_open h
  have h2 := ρ_p₂_eq_zero_of_open h
  have h3 := ρ_p₁p₁_eq_zero_of_open h
  have hpt := pt_eq_ptf (r := r) h
  unfold Lpt Lρ pt_r pt_p₁ pt_p₂ pt_p₁p₁ Lptf
  rw [hρ, h1, h2, h3, hpt]
  ring

end RotorFl
