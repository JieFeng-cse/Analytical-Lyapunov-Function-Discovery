/-
# Fast region Ω₃ — part 3: the drift inequality

Assembling part 2: with `E = e^{θp̃²/2}`, `e^{θH} = E·X`,

    L V + φ(V) ≤ θ E [ X (1 + 1.25·10⁻⁷ − 0.98 p₁²) + 20.001 β − 4.5895 ]

and the bracket is ≤ 0 by the real-arithmetic lemma `fast_key`
(cases `p₁² ≤ 2`, `2 ≤ p₁² ≤ 30.25`, `p₁² ≥ 30.25`).
-/
import RotorFl.FastMain0
import RotorFl.GenForm

namespace RotorFl

open Real

theorem fast_region {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    L V r p₁ p₂ + φ (V r p₁ p₂) ≤ 0 := by
  have h₂ := h.p₂_ne
  have hU := h.fastU
  rw [L_V_eq_L_Vf_of_le h.gate, L_Vf r p₁ p₂ h₂]
  have hT2 := term2_le h
  have hT3 := term3_le h
  have hφ := φ_le h
  have hX := X_bounds h
  have hexp := expH_eq (r := r) (p₁ := p₁) h₂
  rw [hexp] at hφ ⊢
  set E := exp (θ / 2 * (ptf r p₁ p₂) ^ 2) with hE
  set X := exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂))) with hXdef
  set β := βc p₁ (1 / p₂) with hβ
  have hE0 : 0 < E := exp_pos _
  have hβ0 : 0 ≤ β := βc_nonneg _ _
  have hθ : θ = 1 / 50 := rfl
  have hA : A = 20 := rfl
  -- the key inequality with a = |p₁|, v = |u|
  have key : X * (1 + 1 / 8000000 - (1 - θ) * p₁ ^ 2) + (20 + 1 / 1000) * β ≤ 4589 / 1000 := by
    have ha := abs_nonneg p₁
    have hv0 := abs_nonneg (1 / p₂)
    have hsq : p₁ ^ 2 = |p₁| ^ 2 := (sq_abs p₁).symm
    have hβe : β = (13 + 22 * |p₁| + 19 * |p₁| ^ 2 + 6 * |p₁| ^ 3 + |p₁| ^ 4) * |1 / p₂|
        + (10 + 16 * |p₁| + 12 * |p₁| ^ 2 + 4 * |p₁| ^ 3) * |1 / p₂| ^ 2 := by
      rw [hβ]; unfold βc c4_bound c5_bound; ring
    rw [hβe, hsq, hθ]
    have hg := hU.gate
    have hs := hU.small
    rw [hsq] at hX
    have hX1 := hX.1
    have hX2 := hX.2
    rw [hθ] at hX1 hX2
    have h4 : |p₁| ^ 4 = p₁ ^ 4 := by rw [← abs_pow, abs_of_nonneg (by positivity)]
    exact fast_key ha hv0 hs (by rw [h4]; exact hg) hX1 hX2
  -- assemble
  have hkey := mul_le_mul_of_nonneg_left key (le_of_lt (mul_pos (by rw [hθ]; norm_num : (0:ℝ) < θ) hE0))
  rw [hθ] at hkey hT2 hT3 hφ ⊢
  rw [hA] at hT2
  nlinarith [hkey, hT2, hT3, hφ, hE0]

end RotorFl
