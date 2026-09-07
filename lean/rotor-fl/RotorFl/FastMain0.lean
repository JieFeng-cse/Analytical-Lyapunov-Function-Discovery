/-
# Fast region — the real-arithmetic key inequality

With `a = |p₁| ≥ 0`, `v = |1/p₂|`, `v ≤ 1/1999`, `v (2a⁴ + 40) ≤ 1`, and
`X` between `1 + θ(a²/2 + 0.996)` and `e^{θ(a²/2 + 1.004)}`:

    X (1 + 1.25·10⁻⁷ − 0.98 a²) + 20.001 (c4b(a) v + c5b(a) v²) ≤ 4.5895

Cases: `a² ≤ 2` (X ≤ 1.042, β small), `2 ≤ a², a ≤ 5.5` (damping beats the
polynomial with v ≤ 1/1999), `a ≥ 5.5` (v ≤ 1/(2a⁴+40) gives β ≤ 1.5).
-/
import RotorFl.FastEst2

namespace RotorFl

open Real

lemma fast_key {a v X : ℝ} (ha : 0 ≤ a) (hv0 : 0 ≤ v) (hv : v ≤ 1 / 1999)
    (hg : v * (2 * a ^ 4 + 40) ≤ 1)
    (hX1 : 1 + 1 / 50 * (a ^ 2 / 2 + 249 / 250) ≤ X)
    (hX2 : X ≤ exp (1 / 50 * (a ^ 2 / 2 + 251 / 250))) :
    X * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
      + (20 + 1 / 1000) * ((13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
        + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2) ≤ 4589 / 1000 := by
  have hc40 : 0 ≤ 13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4 := by positivity
  have hc50 : 0 ≤ 10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3 := by positivity
  have hv2 : v ^ 2 ≤ (1 / 1999) ^ 2 := pow_le_pow_left₀ hv0 hv 2
  rcases le_or_gt (a ^ 2) 2 with h2 | h2
  · -- a² ≤ 2
    have ha1 : a ≤ 14143 / 10000 := by nlinarith
    have hXle : X ≤ 1042 / 1000 := by
      have hx : (1 : ℝ) / 50 * (a ^ 2 / 2 + 251 / 250) ≤ 4008 / 100000 := by nlinarith
      have hexp : exp (1 / 50 * (a ^ 2 / 2 + 251 / 250)) ≤ exp (4008 / 100000) :=
        exp_le_exp.mpr hx
      have hb := (abs_le.mp (abs_exp_sub_one_sub_id_le (x := (4008 / 100000 : ℝ)) (by norm_num))).2
      norm_num at hb
      linarith
    have hβ : (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
        + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2 ≤ 6 / 100 := by
      have hc4 : 13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4 ≤ 104 := by
        nlinarith [mul_nonneg ha (sq_nonneg a)]
      have hc5 : 10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3 ≤ 68 := by
        nlinarith [mul_nonneg ha (sq_nonneg a)]
      calc (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
            + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2
          ≤ 104 * (1 / 1999) + 68 * (1 / 1999) ^ 2 := by gcongr
        _ ≤ 6 / 100 := by norm_num
    have hY : 1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2 ≤ 1 + 1 / 8000000 := by nlinarith
    have hXY : X * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2) ≤ 1043 / 1000 := by
      rcases le_or_gt 0 (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2) with hY0 | hY0
      · calc X * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
            ≤ (1042 / 1000) * (1 + 1 / 8000000) := mul_le_mul hXle hY hY0 (by norm_num)
          _ ≤ 1043 / 1000 := by norm_num
      · have hX0 : 0 < X := by nlinarith
        nlinarith
    nlinarith
  · -- a² ≥ 2
    have hY : 1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2 ≤ 0 := by nlinarith
    have hXY : X * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
        ≤ (1 + 1 / 50 * (a ^ 2 / 2 + 249 / 250)) * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2) :=
      mul_le_mul_of_nonpos_right hX1 hY
    rcases le_or_gt a (11 / 2) with h55 | h55
    · -- √2 ≤ a ≤ 5.5, use v ≤ 1/1999
      have hβ : (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
          + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2
          ≤ (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * (1 / 1999)
            + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * (1 / 1999) ^ 2 := by gcongr
      have hp1 : 0 ≤ (11 / 2 - a) * a ^ 2 := mul_nonneg (by linarith) (sq_nonneg a)
      have hp2 : 0 ≤ (11 / 2 - a) * (11 / 2 + a) * a ^ 2 :=
        mul_nonneg (mul_nonneg (by linarith) (by linarith)) (sq_nonneg a)
      have hpoly : (1 + 1 / 50 * (a ^ 2 / 2 + 249 / 250)) * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
          + (20 + 1 / 1000) * ((13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * (1 / 1999)
            + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * (1 / 1999) ^ 2) ≤ 4589 / 1000 := by
        linarith only [hp1, hp2, h55, h2, ha, sq_nonneg a]
      linarith only [hXY, hβ, hpoly]
    · -- a ≥ 5.5, use v (2a⁴ + 40) ≤ 1
      have ha3 : 0 ≤ a ^ 3 := by positivity
      have hc4 : 13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4 ≤ 3 * a ^ 4 := by
        have h1 : (11 / 2) * a ^ 3 ≤ a ^ 4 := by nlinarith
        have h2 : (11 / 2) * a ^ 2 ≤ a ^ 3 := by nlinarith
        have h3 : (11 / 2) * a ≤ a ^ 2 := by nlinarith
        nlinarith
      have hc5 : 10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3 ≤ 2 * a ^ 4 := by
        have h1 : (11 / 2) * a ^ 3 ≤ a ^ 4 := by nlinarith
        have h2 : (11 / 2) * a ^ 2 ≤ a ^ 3 := by nlinarith
        have h3 : (11 / 2) * a ≤ a ^ 2 := by nlinarith
        nlinarith
      have h4v : a ^ 4 * v ≤ 1 / 2 := by nlinarith
      have hβ : (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
          + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2 ≤ 3 / 2 + 1 / 1999 := by
        have e1 : (13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v ≤ 3 * a ^ 4 * v :=
          mul_le_mul_of_nonneg_right hc4 hv0
        have e2 : (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2 ≤ 2 * a ^ 4 * v ^ 2 :=
          mul_le_mul_of_nonneg_right hc5 (sq_nonneg v)
        have e3 : 2 * a ^ 4 * v ^ 2 = 2 * (a ^ 4 * v) * v := by ring
        have e4 : 2 * (a ^ 4 * v) * v ≤ 2 * (1 / 2) * (1 / 1999) := by
          have : 0 ≤ a ^ 4 * v := by positivity
          gcongr
        nlinarith
      have hs : (11 / 2 : ℝ) ^ 2 ≤ a ^ 2 := pow_le_pow_left₀ (by norm_num) h55.le 2
      have hs2 : 0 ≤ a ^ 2 * (a ^ 2 - (11 / 2) ^ 2) := mul_nonneg (sq_nonneg a) (by linarith)
      have hpoly2 : (1 + 1 / 50 * (a ^ 2 / 2 + 249 / 250)) * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
          ≤ -27 := by
        linarith only [hs, hs2]
      calc X * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
            + (20 + 1 / 1000) * ((13 + 22 * a + 19 * a ^ 2 + 6 * a ^ 3 + a ^ 4) * v
              + (10 + 16 * a + 12 * a ^ 2 + 4 * a ^ 3) * v ^ 2)
          ≤ (1 + 1 / 50 * (a ^ 2 / 2 + 249 / 250)) * (1 + 1 / 8000000 - (1 - 1 / 50) * a ^ 2)
            + (20 + 1 / 1000) * (3 / 2 + 1 / 1999) :=
            add_le_add hXY (mul_le_mul_of_nonneg_left hβ (by norm_num))
        _ ≤ -27 + (20 + 1 / 1000) * (3 / 2 + 1 / 1999) := by linarith only [hpoly2]
        _ ≤ 4589 / 1000 := by norm_num

end RotorFl
