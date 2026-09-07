/-
# Transition band Ω₂ — part 3: gate terms, φ(V), and the drift inequality

With `E = e^{θp̃²/2}`, `EH = e^{θH}`, `s = p₁² ≥ 10⁴`, `X = e^{s/100}`:

* gate terms `≤ 5200 S³ E`;
* `φ(V) ≤ 10⁻¹⁸ EH + 10⁻¹⁶ S² E`;
* `E · X ≤ 1.021 EH` (from `θH ≥ θp̃²/2 + θ(s/2 − 1.006)`);
* `3·10⁵ s⁵ ≤ X` (two regimes: `s ≤ 10⁶` via `e^{100}`, `s ≥ 10⁶` via `x¹²/12!`);
* hence `L V + φ(V) ≤ EH (0.02 + 10⁻¹⁸ − 0.0018 s) < 0`.
-/
import RotorFl.TransEst2

namespace RotorFl

open Real

/-- `3·10⁵ s⁵ ≤ e^{s/100}` for `s ≥ 10⁴`. -/
lemma exp_key {s : ℝ} (hs : 10 ^ 4 ≤ s) : 3 * 10 ^ 5 * s ^ 5 ≤ exp (s / 100) := by
  rcases le_or_gt s (10 ^ 6) with h6 | h6
  · -- e^{s/100} ≥ e^{100} ≥ 2.7182818283^100
    have h1 : exp 100 ≤ exp (s / 100) := exp_le_exp.mpr (by linarith)
    have h2 : exp 100 = exp 1 ^ 100 := by rw [← exp_nat_mul]; try norm_num
    have h3 : (2.7182818283 : ℝ) ^ 100 ≤ exp 1 ^ 100 :=
      pow_le_pow_left₀ (by norm_num) exp_one_gt_d9.le 100
    have h4 : (2.6 : ℝ) * 10 ^ 43 ≤ (2.7182818283 : ℝ) ^ 100 := by norm_num
    have h5 : s ^ 5 ≤ (10 ^ 6) ^ 5 := pow_le_pow_left₀ (by linarith) h6 5
    rw [h2] at h1
    nlinarith
  · -- e^{x} ≥ x¹²/12!, x = s/100 ≥ 10⁴
    have hx : (0 : ℝ) ≤ s / 100 := by linarith
    have h1 := pow_div_factorial_le_exp _ hx 12
    have hf : (Nat.factorial 12 : ℝ) = 479001600 := by norm_num [Nat.factorial]
    rw [hf] at h1
    have hs6 : (10 ^ 6 : ℝ) ≤ s := h6.le
    have h7 : (10 ^ 6 : ℝ) ^ 7 ≤ s ^ 7 := pow_le_pow_left₀ (by norm_num) hs6 7
    have hs5 : 0 ≤ s ^ 5 := by positivity
    -- s^12 / (100^12 · 12!) ≥ 3·10^5 s^5  ⇔  s^7 ≥ 3·10^5 · 10^24 · 479001600
    have : 3 * 10 ^ 5 * s ^ 5 ≤ (s / 100) ^ 12 / 479001600 := by
      rw [div_pow]
      have e : (s ^ 12 / 100 ^ 12) / 479001600 = s ^ 5 * (s ^ 7 / (100 ^ 12 * 479001600)) := by
        ring
      rw [e]
      have : (3 * 10 ^ 5 : ℝ) ≤ s ^ 7 / (100 ^ 12 * 479001600) := by
        rw [le_div_iff₀ (by norm_num)]; nlinarith
      nlinarith
    linarith

namespace TransPt

variable {r p₁ p₂ : ℝ} (h : TransPt r p₁ p₂)
include h

lemma S_ge_one : 1 ≤ S p₁ := by have := h.S_ge; linarith

lemma P_le : 2 * S p₁ + 1 ≤ 3 * S p₁ := by have := h.S_ge_one; linarith

lemma abs_Ff'_le :
    |Ff' (pt r p₁ p₂)| ≤ 41 * (2 * S p₁ + 1) ^ 3 * exp (θ / 2 * (pt r p₁ p₂) ^ 2) := by
  unfold Ff'
  have hP := h.abs_pt_le
  have hP1 : (1 : ℝ) ≤ 2 * S p₁ + 1 := by have := h.S_ge_one; linarith
  have hE : 0 < exp (θ / 2 * (pt r p₁ p₂) ^ 2) := exp_pos _
  have hθ : θ = 1 / 50 := rfl
  have hA : A = 20 := rfl
  set y := pt r p₁ p₂ with hy
  set P := 2 * S p₁ + 1 with hPdef
  rw [abs_mul, abs_mul, abs_of_pos hE, hA, hθ, abs_of_pos (by norm_num : (0:ℝ) < 20)]
  have hy3 : |y| ^ 3 ≤ P ^ 3 := pow_le_pow_left₀ (abs_nonneg _) hP 3
  have hyP : |y| ≤ P ^ 3 := by
    have : P ≤ P ^ 3 := by nlinarith [sq_nonneg P]
    linarith
  have hb : |2 * y + 1 / 50 * y ^ 3| ≤ (2 + 1 / 50) * P ^ 3 := by
    calc |2 * y + 1 / 50 * y ^ 3| ≤ |2 * y| + |1 / 50 * y ^ 3| := abs_add_le _ _
      _ = 2 * |y| + 1 / 50 * |y| ^ 3 := by
          rw [abs_mul, abs_mul, abs_pow, abs_of_pos (by norm_num : (0:ℝ) < 2),
            abs_of_pos (by norm_num : (0:ℝ) < 1 / 50)]
      _ ≤ 2 * P ^ 3 + 1 / 50 * P ^ 3 := by gcongr
      _ = (2 + 1 / 50) * P ^ 3 := by ring
  calc 20 * |2 * y + 1 / 50 * y ^ 3| * exp (1 / 50 / 2 * y ^ 2)
      ≤ 20 * ((2 + 1 / 50) * P ^ 3) * exp (1 / 50 / 2 * y ^ 2) := by gcongr
    _ ≤ 41 * P ^ 3 * exp (1 / 50 / 2 * y ^ 2) := by
        have : 0 ≤ P ^ 3 * exp (1 / 50 / 2 * y ^ 2) := by positivity
        nlinarith

lemma Ff''_nonneg : 0 ≤ Ff'' (pt r p₁ p₂) := by
  unfold Ff'' A θ; positivity

lemma Ff''_le :
    Ff'' (pt r p₁ p₂) ≤ 43 * (2 * S p₁ + 1) ^ 4 * exp (θ / 2 * (pt r p₁ p₂) ^ 2) := by
  unfold Ff''
  have hP := h.abs_pt_le
  have hP1 : (1 : ℝ) ≤ 2 * S p₁ + 1 := by have := h.S_ge_one; linarith
  have hE : 0 < exp (θ / 2 * (pt r p₁ p₂) ^ 2) := exp_pos _
  have hθ : θ = 1 / 50 := rfl
  have hA : A = 20 := rfl
  set y := pt r p₁ p₂ with hy
  set P := 2 * S p₁ + 1 with hPdef
  rw [hA, hθ]
  have hy2 : y ^ 2 ≤ P ^ 2 := by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hP 2
  have hy4 : y ^ 4 ≤ P ^ 4 := by
    have := pow_le_pow_left₀ (abs_nonneg _) hP 4
    rwa [← abs_pow, abs_of_nonneg (by positivity)] at this
  have hP2 : P ^ 2 ≤ P ^ 4 := by
    have h1 : 1 ≤ P ^ 2 := one_le_pow₀ hP1
    calc P ^ 2 = P ^ 2 * 1 := by ring
      _ ≤ P ^ 2 * P ^ 2 := by gcongr
      _ = P ^ 4 := by ring
  have hP0 : 1 ≤ P ^ 4 := one_le_pow₀ hP1
  have hb : 2 + 5 * (1 / 50) * y ^ 2 + (1 / 50) ^ 2 * y ^ 4 ≤ (43 / 20) * P ^ 4 := by nlinarith
  calc 20 * (2 + 5 * (1 / 50) * y ^ 2 + (1 / 50) ^ 2 * y ^ 4) * exp (1 / 50 / 2 * y ^ 2)
      ≤ 20 * ((43 / 20) * P ^ 4) * exp (1 / 50 / 2 * y ^ 2) := by
        gcongr
    _ = 43 * P ^ 4 * exp (1 / 50 / 2 * y ^ 2) := by ring

lemma Ff_nonneg : 0 ≤ Ff (pt r p₁ p₂) := by unfold Ff A; positivity

lemma Ff_le : Ff (pt r p₁ p₂) ≤ 20 * (2 * S p₁ + 1) ^ 2 * exp (θ / 2 * (pt r p₁ p₂) ^ 2) := by
  unfold Ff
  have hP := h.abs_pt_le
  have hA : A = 20 := rfl
  rw [hA]
  have hy2 : (pt r p₁ p₂) ^ 2 ≤ (2 * S p₁ + 1) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hP 2
  gcongr

/-- The gate terms of `L_V_general` are at most `5200 S³ E`. -/
lemma gate_sum_le :
    ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)
      + Ff (pt r p₁ p₂) * Lρ r p₁ p₂ + 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)
      ≤ 5200 * (S p₁) ^ 3 * exp (θ / 2 * (pt r p₁ p₂) ^ 2) := by
  set E := exp (θ / 2 * (pt r p₁ p₂) ^ 2) with hE
  set P := 2 * S p₁ + 1 with hPdef
  set v := |1 / p₂| with hv
  have hE0 : 0 < E := exp_pos _
  have hS := h.S_ge
  have hS0 : 0 < S p₁ := S_pos p₁
  have hP3 := h.P_le
  have hP0 : 0 ≤ P := by rw [hPdef]; positivity
  have hv0 : 0 ≤ v := abs_nonneg _
  have hvS : v * S p₁ ≤ 1 := by
    have := h.abs_u_le
    rw [hv]; calc |1 / p₂| * S p₁ ≤ 1 / S p₁ * S p₁ := by gcongr
      _ = 1 := by field_simp
  have ha : |p₁| ≤ S p₁ := by
    have h4 := h.a4_le_S
    have hb := h.big
    have ha0 := abs_nonneg p₁
    nlinarith [pow_le_pow_left₀ (by norm_num) hb 3]
  have hρ := h.abs_ρ_le
  have hF' := h.abs_Ff'_le
  have hF'' := h.Ff''_le
  have hF''0 := h.Ff''_nonneg
  have hF := h.Ff_le
  have hF0 := h.Ff_nonneg
  have hL := h.abs_Lpt_le
  have hLρ := h.abs_Lρ_le
  have hρ1 := h.abs_ρ_p₁_le
  have hd := h.abs_pt_p₁_le
  -- bound each of the three terms in absolute value
  have t1 : |ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)|
      ≤ 41 * P ^ 3 * E * (202 / 100) + 43 * P ^ 4 * E * (15 * v) ^ 2 := by
    rw [abs_mul]
    have hd2 : (pt_p₁ r p₁ p₂) ^ 2 ≤ (15 * v) ^ 2 := by
      rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hd 2
    have hinner : |Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2|
        ≤ 41 * P ^ 3 * E * (202 / 100) + 43 * P ^ 4 * E * (15 * v) ^ 2 := by
      calc |Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2|
          ≤ |Ff' (pt r p₁ p₂) * Lpt r p₁ p₂| + |Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2| :=
            abs_add_le _ _
        _ = |Ff' (pt r p₁ p₂)| * |Lpt r p₁ p₂| + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2 := by
            rw [abs_mul, abs_mul, abs_of_nonneg hF''0, abs_of_nonneg (sq_nonneg (pt_p₁ r p₁ p₂))]
        _ ≤ 41 * P ^ 3 * E * (202 / 100) + 43 * P ^ 4 * E * (15 * v) ^ 2 := by gcongr
    calc |ρ p₁ p₂| * |Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2|
        ≤ 1 * (41 * P ^ 3 * E * (202 / 100) + 43 * P ^ 4 * E * (15 * v) ^ 2) := by gcongr
      _ = _ := one_mul _
  have t2 : |Ff (pt r p₁ p₂) * Lρ r p₁ p₂| ≤ 20 * P ^ 2 * E * (16 * |p₁|) := by
    rw [abs_mul, abs_of_nonneg hF0]; gcongr
  have t3 : |2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)|
      ≤ 2 * 14 * (41 * P ^ 3 * E * (15 * v)) := by
    rw [abs_mul, abs_mul, abs_mul, abs_two]
    gcongr
  -- polynomial bookkeeping: P ≤ 3S, vS ≤ 1, |p₁| ≤ S, S ≥ 10⁸
  have hP3' : P ^ 3 ≤ 27 * (S p₁) ^ 3 := by
    have := pow_le_pow_left₀ hP0 hP3 3; linarith
  have hP2' : P ^ 2 ≤ 9 * (S p₁) ^ 2 := by
    have := pow_le_pow_left₀ hP0 hP3 2; linarith
  have hP4v : P ^ 4 * v ^ 2 ≤ 81 * (S p₁) ^ 2 := by
    have h1 : P ^ 2 * v ≤ 9 * S p₁ := by
      calc P ^ 2 * v ≤ 9 * (S p₁) ^ 2 * v := by gcongr
        _ = 9 * S p₁ * (v * S p₁) := by ring
        _ ≤ 9 * S p₁ * 1 := by gcongr
        _ = 9 * S p₁ := by ring
    calc P ^ 4 * v ^ 2 = (P ^ 2 * v) ^ 2 := by ring
      _ ≤ (9 * S p₁) ^ 2 := pow_le_pow_left₀ (by positivity) h1 2
      _ = 81 * (S p₁) ^ 2 := by ring
  have hP3v : P ^ 3 * v ≤ 27 * (S p₁) ^ 2 := by
    calc P ^ 3 * v ≤ 27 * (S p₁) ^ 3 * v := by gcongr
      _ = 27 * (S p₁) ^ 2 * (v * S p₁) := by ring
      _ ≤ 27 * (S p₁) ^ 2 * 1 := by gcongr
      _ = 27 * (S p₁) ^ 2 := by ring
  have hS2 : 1248615 * (S p₁) ^ 2 ≤ 83 * (S p₁) ^ 3 := by nlinarith
  calc ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)
        + Ff (pt r p₁ p₂) * Lρ r p₁ p₂ + 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)
      ≤ |ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂ + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)|
        + |Ff (pt r p₁ p₂) * Lρ r p₁ p₂|
        + |2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂)| := by
        have := abs_add_three (ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂
          + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2)) (Ff (pt r p₁ p₂) * Lρ r p₁ p₂)
          (2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂))
        linarith [le_abs_self (ρ p₁ p₂ * (Ff' (pt r p₁ p₂) * Lpt r p₁ p₂
          + Ff'' (pt r p₁ p₂) * (pt_p₁ r p₁ p₂) ^ 2) + Ff (pt r p₁ p₂) * Lρ r p₁ p₂
          + 2 * ρ_p₁ p₁ p₂ * (Ff' (pt r p₁ p₂) * pt_p₁ r p₁ p₂))]
    _ ≤ (41 * P ^ 3 * E * (202 / 100) + 43 * P ^ 4 * E * (15 * v) ^ 2)
        + 20 * P ^ 2 * E * (16 * |p₁|) + 2 * 14 * (41 * P ^ 3 * E * (15 * v)) :=
        add_le_add (add_le_add t1 t2) t3
    _ = E * (41 * (202 / 100) * P ^ 3 + 43 * 225 * (P ^ 4 * v ^ 2) + 320 * (P ^ 2 * |p₁|)
          + 2 * 14 * 41 * 15 * (P ^ 3 * v)) := by ring
    _ ≤ E * (41 * (202 / 100) * (27 * (S p₁) ^ 3) + 43 * 225 * (81 * (S p₁) ^ 2)
          + 320 * (9 * (S p₁) ^ 2 * S p₁) + 2 * 14 * 41 * 15 * (27 * (S p₁) ^ 2)) := by
        gcongr
    _ ≤ E * (5200 * (S p₁) ^ 3) := by
        apply mul_le_mul_of_nonneg_left _ hE0.le
        linarith [hS2]
    _ = 5200 * (S p₁) ^ 3 * E := by ring

/-- `φ(V) ≤ 10⁻¹⁸ e^{θH} + 10⁻¹⁶ S² E`. -/
lemma φ_le : φ (V r p₁ p₂) ≤ 1 / 10 ^ 18 * exp (θ * H r p₁ p₂)
    + 1 / 10 ^ 16 * (S p₁) ^ 2 * exp (θ / 2 * (pt r p₁ p₂) ^ 2) := by
  unfold φ V
  set E := exp (θ / 2 * (pt r p₁ p₂) ^ 2) with hE
  set EH := exp (θ * H r p₁ p₂) with hEH
  have hE0 : 0 < E := exp_pos _
  have hEH0 : 0 < EH := exp_pos _
  have hEH1 : 1 ≤ EH := by
    rw [hEH]; apply one_le_exp; have := H_nonneg r p₁ p₂; unfold θ; exact mul_nonneg (by norm_num) this
  have hρ := h.ρ_mem
  have hθ : θ = 1 / 50 := rfl
  have hc : cRate = 1 / 20000 := rfl
  have hA : A = 20 := rfl
  have hP := h.abs_pt_le
  have hy2 : (pt r p₁ p₂) ^ 2 ≤ (2 * S p₁ + 1) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hP 2
  have hP3 := h.P_le
  have hP2 : (2 * S p₁ + 1) ^ 2 ≤ 9 * (S p₁) ^ 2 := by
    have := pow_le_pow_left₀ (by have := S_pos p₁; linarith) hP3 2; linarith
  -- H ≥ p₂²/2 ≥ 10¹⁶/2
  have hH : 10 ^ 16 / 2 ≤ H r p₁ p₂ := by
    unfold H
    have h1 := h.abs_p₂_ge
    have h2 : (10 ^ 8 : ℝ) ^ 2 ≤ p₂ ^ 2 := by
      have := pow_le_pow_left₀ (by norm_num) h1 2; rwa [sq_abs] at this
    have := cos_le_one r
    nlinarith [sq_nonneg p₁]
  set Vv := 1 + EH + A * ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 * E with hV
  have hV0 : 0 < Vv := by
    rw [hV, hA]
    nlinarith [mul_nonneg (mul_nonneg (mul_nonneg (by norm_num : (0:ℝ) ≤ 20) hρ.1) (sq_nonneg (pt r p₁ p₂))) hE0.le]
  have hVle : Vv ≤ 2 * EH + 180 * (S p₁) ^ 2 * E := by
    rw [hV, hA]
    have : ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 ≤ 1 * (9 * (S p₁) ^ 2) := by
      calc ρ p₁ p₂ * (pt r p₁ p₂) ^ 2 ≤ 1 * (2 * S p₁ + 1) ^ 2 := by gcongr; exact hρ.2
        _ ≤ 1 * (9 * (S p₁) ^ 2) := by gcongr
    nlinarith
  have hlog : 10 ^ 14 ≤ 2 + log Vv := by
    have h1 : EH ≤ Vv := by rw [hV, hA]; nlinarith [mul_nonneg (mul_nonneg hρ.1 (sq_nonneg (pt r p₁ p₂))) hE0.le]
    have h2 : log EH ≤ log Vv := log_le_log hEH0 h1
    have h3 : log EH = θ * H r p₁ p₂ := by rw [hEH, log_exp]
    rw [hθ] at h3
    linarith
  have hden : 0 < 2 + log Vv := by linarith
  calc cRate * Vv / (2 + log Vv) ≤ cRate * Vv / 10 ^ 14 := by
        apply div_le_div_of_nonneg_left (by rw [hc]; exact mul_nonneg (by norm_num) hV0.le) (by norm_num) hlog
    _ ≤ cRate * (2 * EH + 180 * (S p₁) ^ 2 * E) / 10 ^ 14 := by
        gcongr
        rw [hc]; norm_num
    _ ≤ 1 / 10 ^ 18 * EH + 1 / 10 ^ 16 * (S p₁) ^ 2 * E := by
        rw [hc]
        have : 0 ≤ (S p₁) ^ 2 * E := by positivity
        nlinarith

/-- `E · e^{s/100} ≤ 1.021 e^{θH}` with `s = p₁²`. -/
lemma E_mul_le : exp (θ / 2 * (pt r p₁ p₂) ^ 2) * exp (p₁ ^ 2 / 100)
    ≤ (1 + 21 / 1000) * exp (θ * H r p₁ p₂) := by
  have hH := h.H_ge
  have hθ : θ = 1 / 50 := rfl
  rw [← exp_add]
  have h1 : exp (θ * H r p₁ p₂) ≥ exp (θ / 2 * (pt r p₁ p₂) ^ 2 + p₁ ^ 2 / 100 - 2012 / 100000) := by
    apply exp_le_exp.mpr; rw [hθ]; nlinarith
  have h2 : exp (θ / 2 * (pt r p₁ p₂) ^ 2 + p₁ ^ 2 / 100 - 2012 / 100000)
      = exp (θ / 2 * (pt r p₁ p₂) ^ 2 + p₁ ^ 2 / 100) * exp (-(2012 / 100000)) := by
    rw [← exp_add]; ring_nf
  have h3 : (1 : ℝ) / (1 + 21 / 1000) ≤ exp (-(2012 / 100000)) := by
    have := add_one_le_exp (-(2012 / 100000 : ℝ))
    linarith
  have hpos : 0 < exp (θ / 2 * (pt r p₁ p₂) ^ 2 + p₁ ^ 2 / 100) := exp_pos _
  rw [h2] at h1
  nlinarith

/-- **Transition-band drift**: `L V + φ(V) ≤ 0` for `S ≤ |p₂| ≤ 2S`, `|p₁| ≥ 100`. -/
theorem trans_region : L V r p₁ p₂ + φ (V r p₁ p₂) ≤ 0 := by
  rw [L_V_general r p₁ p₂ h.p₂_ne]
  have hg := h.gate_sum_le
  have hφ := h.φ_le
  have hEX := h.E_mul_le
  set E := exp (θ / 2 * (pt r p₁ p₂) ^ 2) with hE
  set EH := exp (θ * H r p₁ p₂) with hEH
  set X := exp (p₁ ^ 2 / 100) with hX
  have hE0 : 0 < E := exp_pos _
  have hEH0 : 0 < EH := exp_pos _
  have hX0 : 0 < X := exp_pos _
  have hθ : θ = 1 / 50 := rfl
  have hs : 10 ^ 4 ≤ p₁ ^ 2 := by
    have := pow_le_pow_left₀ (by norm_num) h.big 2
    rw [sq_abs] at this; norm_num at this ⊢; linarith
  have hkey := exp_key hs
  rw [← hX] at hkey
  -- S ≤ p₁⁴ + 20 ≤ (1 + 2·10⁻⁷) p₁⁴
  have hS : S p₁ ≤ (1 + 2 / 10 ^ 7) * (p₁ ^ 2) ^ 2 := by
    unfold S Rρ; nlinarith [sq_nonneg (p₁ ^ 2)]
  have hS0 : 0 < S p₁ := S_pos p₁
  -- 5200 S³ + 10⁻¹⁶ S² ≤ 0.01735 s · X   (s = p₁²)
  have hpoly : 5200 * (S p₁) ^ 3 + 1 / 10 ^ 16 * (S p₁) ^ 2 ≤ 1735 / 100000 * p₁ ^ 2 * X := by
    have h3 : (S p₁) ^ 3 ≤ ((1 + 2 / 10 ^ 7) * (p₁ ^ 2) ^ 2) ^ 3 := pow_le_pow_left₀ hS0.le hS 3
    have h2 : (S p₁) ^ 2 ≤ ((1 + 2 / 10 ^ 7) * (p₁ ^ 2) ^ 2) ^ 2 := pow_le_pow_left₀ hS0.le hS 2
    have hs5 : 0 ≤ (p₁ ^ 2) ^ 5 := by positivity
    have e1 : ((1 + 2 / 10 ^ 7) * (p₁ ^ 2) ^ 2) ^ 3 = (1 + 2 / 10 ^ 7) ^ 3 * (p₁ ^ 2) ^ 6 := by ring
    have e2 : ((1 + 2 / 10 ^ 7) * (p₁ ^ 2) ^ 2) ^ 2 = (1 + 2 / 10 ^ 7) ^ 2 * (p₁ ^ 2) ^ 4 := by ring
    rw [e1] at h3; rw [e2] at h2
    have h4 : (p₁ ^ 2) ^ 4 ≤ (p₁ ^ 2) ^ 6 := by
      have : 1 ≤ (p₁ ^ 2) ^ 2 := by nlinarith
      nlinarith [pow_nonneg (sq_nonneg p₁) 4]
    -- 5201 s⁶ ≤ 0.01735 s · 3·10⁵ s⁵ = 5205 s⁶
    nlinarith [mul_le_mul_of_nonneg_left hkey (by positivity : (0:ℝ) ≤ 1735 / 100000 * p₁ ^ 2)]
  -- gate + φ ≤ (5200 S³ + 10⁻¹⁶ S²) E + 10⁻¹⁸ EH ≤ 0.01735 s X E + 10⁻¹⁸ EH ≤ 0.0178 s EH + 10⁻¹⁸ EH
  have hgate : (5200 * (S p₁) ^ 3 + 1 / 10 ^ 16 * (S p₁) ^ 2) * E ≤ 1772 / 100000 * p₁ ^ 2 * EH := by
    calc (5200 * (S p₁) ^ 3 + 1 / 10 ^ 16 * (S p₁) ^ 2) * E
        ≤ 1735 / 100000 * p₁ ^ 2 * X * E := by gcongr
      _ = 1735 / 100000 * p₁ ^ 2 * (E * X) := by ring
      _ ≤ 1735 / 100000 * p₁ ^ 2 * ((1 + 21 / 1000) * EH) := by gcongr
      _ ≤ 1772 / 100000 * p₁ ^ 2 * EH := by nlinarith [sq_nonneg p₁]
  rw [hθ]
  nlinarith [hg, hφ, hgate, hEH0, hE0, hs, mul_le_mul_of_nonneg_left hs hEH0.le]

end TransPt

end RotorFl
