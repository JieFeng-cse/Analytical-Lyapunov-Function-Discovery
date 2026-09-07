/-
# Main theorem T1: negative drift outside K

    ∀ x ∉ K,  L V x + φ(V x) ≤ 0,     K = {H ≤ R} ∪ {ρ < 1 ∧ p₁² ≤ M}

by the three regional theorems: `slow_region` (|p₂| < S), `trans_region`
(S ≤ |p₂| < 2S), `fast_region` (2S ≤ |p₂|).
-/
import RotorFl.Slow
import RotorFl.TransEst3
import RotorFl.FastMain

namespace RotorFl

open Real

/-- Gate not yet saturated: `|p₂| < 2S ⇒ ρ < 1`. -/
lemma ρ_lt_one {p₁ p₂ : ℝ} (h : |p₂| < 2 * S p₁) : ρ p₁ p₂ < 1 := by
  rw [ρ_eq_ψ]
  apply Real.smoothTransition.lt_one_of_lt_one
  unfold ζ
  have hS := S_pos p₁
  have h2 : p₂ ^ 2 < 4 * (S p₁) ^ 2 := by nlinarith [abs_nonneg p₂, sq_abs p₂, h]
  have : p₂ ^ 2 / (S p₁) ^ 2 < 4 := by rw [div_lt_iff₀ (by positivity)]; linarith
  linarith

theorem T1 : T1_statement := by
  intro r p₁ p₂ hK
  unfold K at hK
  push_neg at hK
  obtain ⟨hH, hM⟩ := hK
  rcases lt_or_ge |p₂| (S p₁) with h1 | h1
  · -- slow region: ρ = 0 < 1, so p₁² > M
    have hρ : ρ p₁ p₂ < 1 := by rw [ρ_eq_zero h1.le]; norm_num
    exact slow_region h1 (hM hρ)
  · rcases lt_or_ge |p₂| (2 * S p₁) with h2 | h2
    · -- transition band: ρ < 1, so p₁² > M, i.e. |p₁| > 100
      have hρ := ρ_lt_one h2
      have hM' := hM hρ
      have hb : 100 ≤ |p₁| := by
        unfold Mex at hM'
        by_contra hlt
        push_neg at hlt
        have : |p₁| ^ 2 < 100 ^ 2 := by
          have := abs_nonneg p₁
          nlinarith
        rw [sq_abs] at this
        linarith
      exact TransPt.trans_region ⟨h1, h2.le, hb⟩
    · -- fast region (boundary included)
      exact fast_region ⟨h2, hH⟩

end RotorFl
