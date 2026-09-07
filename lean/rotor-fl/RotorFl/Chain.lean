/-
# Generator chain rule

For `F : ℝ → ℝ` twice differentiable and `g` with slice derivatives,

    L (F ∘ g) = F'(g) · L g + F''(g) · (∂ₚ₁ g)²

where `L g = ∂ᵣg·(p₂−p₁) + ∂ₚ₁g·(sin r − p₁) + ∂ₚ₂g·(−sin r) + ∂²ₚ₁g`.
Used for the fast branch `F(p̃) = p̃² e^{θ p̃²/2}`.
-/
import RotorFl.Defs

namespace RotorFl

open Real

theorem L_comp {F F' F'' : ℝ → ℝ} {g g_r g_p₁ g_p₂ g_p₁p₁ : ℝ → ℝ → ℝ → ℝ}
    {r p₁ p₂ : ℝ}
    (hF : ∀ y, HasDerivAt F (F' y) y) (hF' : ∀ y, HasDerivAt F' (F'' y) y)
    (hr : HasDerivAt (fun t => g t p₁ p₂) (g_r r p₁ p₂) r)
    (hp₁ : ∀ t, HasDerivAt (fun s => g r s p₂) (g_p₁ r t p₂) t)
    (hp₂ : HasDerivAt (fun t => g r p₁ t) (g_p₂ r p₁ p₂) p₂)
    (hp₁p₁ : HasDerivAt (fun t => g_p₁ r t p₂) (g_p₁p₁ r p₁ p₂) p₁) :
    L (fun a b c => F (g a b c)) r p₁ p₂
      = F' (g r p₁ p₂) * (g_r r p₁ p₂ * (p₂ - p₁) + g_p₁ r p₁ p₂ * (sin r - p₁)
          + g_p₂ r p₁ p₂ * (-sin r) + g_p₁p₁ r p₁ p₂)
        + F'' (g r p₁ p₂) * (g_p₁ r p₁ p₂) ^ 2 := by
  unfold L
  have d1 : deriv (fun t => F (g t p₁ p₂)) r = F' (g r p₁ p₂) * g_r r p₁ p₂ :=
    ((hF _).comp r hr).deriv
  have d2 : deriv (fun t => F (g r t p₂)) = fun t => F' (g r t p₂) * g_p₁ r t p₂ :=
    funext fun t => ((hF _).comp t (hp₁ t)).deriv
  have d3 : deriv (fun t => F (g r p₁ t)) p₂ = F' (g r p₁ p₂) * g_p₂ r p₁ p₂ :=
    ((hF _).comp p₂ hp₂).deriv
  have d4 : deriv (fun t => F' (g r t p₂) * g_p₁ r t p₂) p₁
      = F'' (g r p₁ p₂) * g_p₁ r p₁ p₂ * g_p₁ r p₁ p₂ + F' (g r p₁ p₂) * g_p₁p₁ r p₁ p₂ :=
    (((hF' _).comp p₁ (hp₁ p₁)).mul hp₁p₁).deriv
  simp only []
  rw [d1, d2, d3, d4]
  ring

/-- Linearity of `L` on a sum of two functions whose slices are differentiable
(as `HasDerivAt` data, second p₁-derivative included). -/
theorem L_add {V₁ V₂ : ℝ → ℝ → ℝ → ℝ} {r p₁ p₂ : ℝ}
    {a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ} {e₁ e₂ : ℝ → ℝ}
    (h1r : HasDerivAt (fun t => V₁ t p₁ p₂) a₁ r)
    (h1p₁ : ∀ t, HasDerivAt (fun s => V₁ r s p₂) (e₁ t) t)
    (h1p₂ : HasDerivAt (fun t => V₁ r p₁ t) c₁ p₂)
    (h1e : HasDerivAt e₁ d₁ p₁)
    (h2r : HasDerivAt (fun t => V₂ t p₁ p₂) a₂ r)
    (h2p₁ : ∀ t, HasDerivAt (fun s => V₂ r s p₂) (e₂ t) t)
    (h2p₂ : HasDerivAt (fun t => V₂ r p₁ t) c₂ p₂)
    (h2e : HasDerivAt e₂ d₂ p₁)
    (hb₁ : e₁ p₁ = b₁) (hb₂ : e₂ p₁ = b₂) :
    L (fun a b c => V₁ a b c + V₂ a b c) r p₁ p₂
      = (a₁ * (p₂ - p₁) + b₁ * (sin r - p₁) + c₁ * (-sin r) + d₁)
        + (a₂ * (p₂ - p₁) + b₂ * (sin r - p₁) + c₂ * (-sin r) + d₂) := by
  unfold L
  have d1 : deriv (fun t => V₁ t p₁ p₂ + V₂ t p₁ p₂) r = a₁ + a₂ := (h1r.add h2r).deriv
  have d2 : deriv (fun t => V₁ r t p₂ + V₂ r t p₂) = fun t => e₁ t + e₂ t :=
    funext fun t => ((h1p₁ t).add (h2p₁ t)).deriv
  have d3 : deriv (fun t => V₁ r p₁ t + V₂ r p₁ t) p₂ = c₁ + c₂ := (h1p₂.add h2p₂).deriv
  have d4 : deriv (fun t => e₁ t + e₂ t) p₁ = d₁ + d₂ := (h1e.add h2e).deriv
  simp only []
  rw [d1, d2, d3, d4]
  subst hb₁ hb₂
  ring

end RotorFl
