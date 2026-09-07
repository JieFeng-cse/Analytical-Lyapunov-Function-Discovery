/-
# Elementary bounds used by the fast-region estimates

A trig-polynomial monomial `c · p^k · sin^i · cos^j` is bounded by `|c| · |p|^k`;
sums of such bounds are assembled by `linarith` in the generated lemmas
(`FastBounds.lean`).
-/
import RotorFl.Defs

namespace RotorFl

open Real

/-- `|c · p^k · s^i · t^j| ≤ C · |p|^k` when `|c| ≤ C`, `|s| ≤ 1`, `|t| ≤ 1`. -/
lemma abs_mono_le (c C p s t : ℝ) (k i j : ℕ) (hc : |c| ≤ C) (hs : |s| ≤ 1) (ht : |t| ≤ 1) :
    |c * p ^ k * s ^ i * t ^ j| ≤ C * |p| ^ k := by
  rw [abs_mul, abs_mul, abs_mul, abs_pow, abs_pow, abs_pow]
  have hs' : |s| ^ i ≤ 1 := pow_le_one₀ (abs_nonneg s) hs
  have ht' : |t| ^ j ≤ 1 := pow_le_one₀ (abs_nonneg t) ht
  have hp : 0 ≤ |p| ^ k := pow_nonneg (abs_nonneg p) k
  have hC : 0 ≤ C := le_trans (abs_nonneg c) hc
  calc |c| * |p| ^ k * |s| ^ i * |t| ^ j ≤ C * |p| ^ k * 1 * 1 := by
        gcongr
    _ = C * |p| ^ k := by ring

/-- `|1/p₂| ≤ 1/1999` and `|1/p₂| · (2 p₁⁴ + 40) ≤ 1` is the data the fast
region provides about `u = 1/p₂`; packaged for convenience. -/
structure FastU (p₁ u : ℝ) : Prop where
  small : |u| ≤ 1 / 1999
  gate : |u| * (2 * p₁ ^ 4 + 40) ≤ 1

/-- `(|p₁| · |u|)⁴ ≤ |u|³ / 2`, hence `|p₁| · |u| ≤ 3/1000`. -/
lemma FastU.abs_mul_le {p₁ u : ℝ} (h : FastU p₁ u) : |p₁| * |u| ≤ 3 / 1000 := by
  have hu : 0 ≤ |u| := abs_nonneg u
  have hp : 0 ≤ |p₁| := abs_nonneg p₁
  have h4 : |p₁| ^ 4 * |u| ≤ 1 / 2 := by
    have : |p₁| ^ 4 = p₁ ^ 4 := by rw [← abs_pow, abs_of_nonneg (by positivity)]
    nlinarith [h.gate]
  -- (|p₁| |u|)^4 = |p₁|^4 |u| · |u|^3 ≤ (1/2) · (1/1999)^3
  have h3 : |u| ^ 3 ≤ (1 / 1999) ^ 3 := pow_le_pow_left₀ hu h.small 3
  have hprod : (|p₁| * |u|) ^ 4 ≤ (1 / 2) * (1 / 1999) ^ 3 := by
    calc (|p₁| * |u|) ^ 4 = (|p₁| ^ 4 * |u|) * |u| ^ 3 := by ring
      _ ≤ (1 / 2) * (1 / 1999) ^ 3 := by gcongr
  by_contra hcon
  push_neg at hcon
  have : (3 / 1000 : ℝ) ^ 4 < (|p₁| * |u|) ^ 4 :=
    pow_lt_pow_left₀ hcon (by norm_num) (by norm_num)
  norm_num at this hprod
  linarith

end RotorFl
