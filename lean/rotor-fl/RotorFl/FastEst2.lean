/-
# Fast region Ω₃ — part 2: the three drift terms and φ(V)

With `E = e^{θ p̃²/2}`, `X = e^{θ(p₁²/2 + 1 + W)}` (so `e^{θH} = E X`),
`u = 1/p₂`, `v = |u|`, `β = c4_bound·v + c5_bound·v²`:

* Term2 = Ff'(p̃)·L p̃ ≤ A θ E (−1/2 + β + 5·10⁻⁵ (1/2 + β))
* Term3 = Ff''(p̃)·(∂ₚ₁p̃)² ≤ 0.41 θ E
* 1 ≤ X, `1 + θ(p₁²/2 + 249/250) ≤ X ≤ e^{θ(p₁²/2 + 251/250)}`
* φ(V) ≤ 5 θ E + 1.25·10⁻⁷ θ E X
-/
import RotorFl.FastEst
import Mathlib.Analysis.Complex.ExponentialBounds

namespace RotorFl

open Real

/-- The bound `β` on `|c₄ u| + |c₅ u²|`. -/
noncomputable def βc (p₁ u : ℝ) : ℝ := c4_bound p₁ * |u| + c5_bound p₁ * |u| ^ 2

lemma βc_nonneg (p₁ u : ℝ) : 0 ≤ βc p₁ u := by
  unfold βc c4_bound c5_bound
  positivity

/-- `|c₄ u + c₅ u²| ≤ β`. -/
lemma abs_c_le (r p₁ u : ℝ) : |c4 r p₁ * u + c5 r p₁ * u ^ 2| ≤ βc p₁ u := by
  unfold βc
  calc |c4 r p₁ * u + c5 r p₁ * u ^ 2| ≤ |c4 r p₁ * u| + |c5 r p₁ * u ^ 2| := abs_add_le _ _
    _ = |c4 r p₁| * |u| + |c5 r p₁| * |u| ^ 2 := by simp only [abs_mul, abs_pow]
    _ ≤ c4_bound p₁ * |u| + c5_bound p₁ * |u| ^ 2 := by
        gcongr
        · exact abs_c4_le r p₁
        · exact abs_c5_le r p₁

/-! ## Term 2 -/

theorem term2_le {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    Ff' (ptf r p₁ p₂) * Lptf r p₁ p₂
      ≤ A * θ * exp (θ / 2 * (ptf r p₁ p₂) ^ 2)
          * (-1 / 2 + βc p₁ (1 / p₂) + 1 / 20000 * (1 / 2 + βc p₁ (1 / p₂))) := by
  have h₂ := h.p₂_ne
  have hq := abs_q_sub_one_le h
  have hU := h.fastU
  have hv := hU.small
  rw [Lptf_eq r p₁ p₂ h₂]
  unfold Ff'
  set u := 1 / p₂ with hu
  set p := ptf r p₁ p₂ with hp
  set E := exp (θ / 2 * p ^ 2) with hE
  set q := p * u with hqdef
  set β := βc p₁ u with hβ
  have hE0 : 0 < E := exp_pos _
  have hβ0 : 0 ≤ β := βc_nonneg p₁ u
  -- algebraic rewrite: (2p + θp³)(−u³/2 + c₄u⁴ + c₅u⁵) = (2qu² + θq³)(−1/2 + c₄u + c₅u²)
  have e : (2 * p + θ * p ^ 3) * (-u ^ 3 / 2 + c4 r p₁ * u ^ 4 + c5 r p₁ * u ^ 5)
      = (2 * q * u ^ 2 + θ * q ^ 3) * (-1 / 2 + (c4 r p₁ * u + c5 r p₁ * u ^ 2)) := by
    rw [hqdef]; ring
  have hM := abs_le.mp (abs_c_le r p₁ u)
  set M := c4 r p₁ * u + c5 r p₁ * u ^ 2 with hMdef
  have hq' := abs_le.mp hq
  have hu2 : u ^ 2 ≤ (1 / 1999) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg u) hv 2
  have hu0 : 0 ≤ u ^ 2 := sq_nonneg u
  have hθ : θ = 1 / 50 := rfl
  -- F = 2qu² + θq³ lies in [θ(1 − 10⁻⁶), θ(1 + 3·10⁻⁵)]
  have hq3lo : (1 - 3 / 10000000) ^ 3 ≤ q ^ 3 := pow_le_pow_left₀ (by norm_num) (by linarith) 3
  have hq3hi : q ^ 3 ≤ (1 + 3 / 10000000) ^ 3 := pow_le_pow_left₀ (by linarith) (by linarith) 3
  have hF_lo : θ * (1 - 1 / 1000000) ≤ 2 * q * u ^ 2 + θ * q ^ 3 := by
    have : 0 ≤ 2 * q * u ^ 2 := by
      have : 0 ≤ q := by linarith
      positivity
    rw [hθ] at *; nlinarith
  have hF_hi : 2 * q * u ^ 2 + θ * q ^ 3 ≤ θ * (1 + 3 / 100000) := by
    have hq1 : q ≤ 1 + 3 / 10000000 := by linarith
    have hq0 : 0 ≤ q := by linarith
    have : 2 * q * u ^ 2 ≤ 2 * (1 + 3 / 10000000) * (1 / 1999) ^ 2 := by gcongr
    rw [hθ] at *; nlinarith
  -- F·M ≤ θ(−1/2 + β) + θ·5·10⁻⁵·(1/2 + β)
  have hFM : (2 * q * u ^ 2 + θ * q ^ 3) * (-1 / 2 + M)
      ≤ θ * (-1 / 2 + β + 1 / 20000 * (1 / 2 + β)) := by
    rcases le_or_gt (-1 / 2 + M) 0 with hneg | hpos
    · -- F ≥ F_lo, factor ≤ 0
      have h1 : (2 * q * u ^ 2 + θ * q ^ 3) * (-1 / 2 + M)
          ≤ θ * (1 - 1 / 1000000) * (-1 / 2 + M) :=
        mul_le_mul_of_nonpos_right hF_lo hneg
      rw [hθ] at *
      nlinarith [hM.1, hM.2]
    · have h1 : (2 * q * u ^ 2 + θ * q ^ 3) * (-1 / 2 + M)
          ≤ θ * (1 + 3 / 100000) * (-1 / 2 + M) :=
        mul_le_mul_of_nonneg_right hF_hi hpos.le
      rw [hθ] at *
      nlinarith [hM.1, hM.2]
  calc A * (2 * p + θ * p ^ 3) * E * (-u ^ 3 / 2 + c4 r p₁ * u ^ 4 + c5 r p₁ * u ^ 5)
      = A * E * ((2 * p + θ * p ^ 3) * (-u ^ 3 / 2 + c4 r p₁ * u ^ 4 + c5 r p₁ * u ^ 5)) := by ring
    _ = A * E * ((2 * q * u ^ 2 + θ * q ^ 3) * (-1 / 2 + M)) := by rw [e]
    _ ≤ A * E * (θ * (-1 / 2 + β + 1 / 20000 * (1 / 2 + β))) := by
        apply mul_le_mul_of_nonneg_left hFM
        unfold A; positivity
    _ = A * θ * E * (-1 / 2 + β + 1 / 20000 * (1 / 2 + β)) := by ring

/-! ## Term 3 -/

theorem term3_le {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    Ff'' (ptf r p₁ p₂) * (ptf_p₁ r p₁ p₂) ^ 2 ≤ 41 / 100 * θ * exp (θ / 2 * (ptf r p₁ p₂) ^ 2) := by
  have h₂ := h.p₂_ne
  have hq := abs_le.mp (abs_q_sub_one_le h)
  have hU := h.fastU
  have hv := hU.small
  have hd := abs_ptf_p₁_le h
  unfold Ff''
  set u := 1 / p₂ with hu
  set p := ptf r p₁ p₂ with hp
  set E := exp (θ / 2 * p ^ 2) with hE
  set q := p * u with hqdef
  have hE0 : 0 < E := exp_pos _
  have hθ : θ = 1 / 50 := rfl
  have hu2 : u ^ 2 ≤ (1 / 1999) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg u) hv 2
  have hu0 : 0 ≤ u ^ 2 := sq_nonneg u
  -- (∂ₚ₁ p̃)² ≤ (1.01)² u⁴
  have hd2 : (ptf_p₁ r p₁ p₂) ^ 2 ≤ ((1 + 1 / 100) * u ^ 2) ^ 2 := by
    rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hd 2
  -- p² u⁴ = q² u², p⁴ u⁴ = q⁴
  have hq2 : q ^ 2 ≤ (1 + 3 / 10000000) ^ 2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
  have hq4 : q ^ 4 ≤ (1 + 3 / 10000000) ^ 4 := pow_le_pow_left₀ (by linarith) (by linarith) 4
  have hpos : 0 ≤ A * (2 + 5 * θ * p ^ 2 + θ ^ 2 * p ^ 4) * E := by
    unfold A; rw [hθ]; positivity
  calc A * (2 + 5 * θ * p ^ 2 + θ ^ 2 * p ^ 4) * E * (ptf_p₁ r p₁ p₂) ^ 2
      ≤ A * (2 + 5 * θ * p ^ 2 + θ ^ 2 * p ^ 4) * E * ((1 + 1 / 100) * u ^ 2) ^ 2 :=
        mul_le_mul_of_nonneg_left hd2 hpos
    _ = A * E * (1 + 1 / 100) ^ 2 * (2 * (u ^ 2) ^ 2 + 5 * θ * q ^ 2 * u ^ 2 + θ ^ 2 * q ^ 4) := by
        rw [hqdef]; ring
    _ ≤ A * E * (1 + 1 / 100) ^ 2 * (2 * ((1 / 1999) ^ 2) ^ 2
          + 5 * θ * (1 + 3 / 10000000) ^ 2 * (1 / 1999) ^ 2 + θ ^ 2 * (1 + 3 / 10000000) ^ 4) := by
        apply mul_le_mul_of_nonneg_left _ (by unfold A; positivity)
        have h1 : (u ^ 2) ^ 2 ≤ ((1 / 1999) ^ 2) ^ 2 := pow_le_pow_left₀ hu0 hu2 2
        have h2 : q ^ 2 * u ^ 2 ≤ (1 + 3 / 10000000) ^ 2 * (1 / 1999) ^ 2 :=
          mul_le_mul hq2 hu2 hu0 (by positivity)
        rw [hθ]
        nlinarith
    _ ≤ 41 / 100 * θ * E := by
        unfold A; rw [hθ]
        nlinarith

/-! ## The factor `X = e^{θ(p₁²/2 + 1 + W)}` -/

/-- `e^{θH} = E · X`. -/
lemma expH_eq {r p₁ p₂ : ℝ} (h₂ : p₂ ≠ 0) :
    exp (θ * H r p₁ p₂)
      = exp (θ / 2 * (ptf r p₁ p₂) ^ 2) * exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂))) := by
  rw [← exp_add, H_split h₂]

lemma X_bounds {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    1 + θ * (p₁ ^ 2 / 2 + 249 / 250) ≤ exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂)))
    ∧ exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂))) ≤ exp (θ * (p₁ ^ 2 / 2 + 251 / 250)) := by
  have hW := abs_le.mp (abs_Wc_le (r := r) h.fastU)
  have hθ : θ = 1 / 50 := rfl
  constructor
  · calc 1 + θ * (p₁ ^ 2 / 2 + 249 / 250) ≤ 1 + θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂)) := by
          rw [hθ]; nlinarith
      _ ≤ exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂))) := by
          have := add_one_le_exp (θ * (p₁ ^ 2 / 2 + 1 + Wc r p₁ (1 / p₂)))
          linarith
  · apply exp_le_exp.mpr
    rw [hθ]; nlinarith

/-! ## φ(V) -/

/-- `φ(V) ≤ 5θE + 1.25·10⁻⁷ θ E X`, using `2 + log V ≥ 2 + θR` on the `1 + e^{θH}`
part and `2 + log V ≥ 19 + θ p̃²/2` on the fast-branch part. -/
theorem φ_le {r p₁ p₂ : ℝ} (h : FastPt r p₁ p₂) :
    φ (V r p₁ p₂) ≤ 5 * θ * exp (θ / 2 * (ptf r p₁ p₂) ^ 2)
      + 1 / 8000000 * θ * exp (θ * H r p₁ p₂) := by
  have h₂ := h.p₂_ne
  have hq := abs_le.mp (abs_q_sub_one_le h)
  have hp := h.abs_p₂_ge
  rw [V_eq_Vf h.gate.le]
  unfold Vf φ
  set p := ptf r p₁ p₂ with hp_def
  set E := exp (θ / 2 * p ^ 2) with hE
  set G := exp (θ * H r p₁ p₂) with hG
  have hE0 : 0 < E := exp_pos _
  have hG0 : 0 < G := exp_pos _
  have hθ : θ = 1 / 50 := rfl
  have hc : cRate = 1 / 20000 := rfl
  have hA : A = 20 := rfl
  -- |p| ≥ 1998
  have hpabs : 1998 ≤ |p| := by
    have h1 : 1 - 3 / 10000000 ≤ |p * (1 / p₂)| :=
      le_trans (by linarith [hq.1]) (le_abs_self _)
    rw [abs_mul, abs_one_div] at h1
    have hpos : 0 < |p₂| := by linarith
    have h2 : (1 - 3 / 10000000) * |p₂| ≤ |p| := by
      have := mul_le_mul_of_nonneg_right h1 hpos.le
      rwa [mul_assoc, one_div_mul_cancel hpos.ne', mul_one] at this
    nlinarith
  have hpne : p ≠ 0 := by
    intro h0; rw [h0, abs_zero] at hpabs; linarith
  have hp2 : 1998 ^ 2 ≤ p ^ 2 := by
    have := pow_le_pow_left₀ (by norm_num) hpabs 2
    rwa [sq_abs] at this
  set Vv := 1 + G + A * p ^ 2 * E with hV
  have hV0 : 0 < Vv := by rw [hV, hA]; positivity
  -- log V ≥ log G = θ H ≥ θ R = 40000
  have hlogG : 40000 ≤ log Vv := by
    have h1 : G ≤ Vv := by rw [hV, hA]; nlinarith
    have h2 : log G ≤ log Vv := log_le_log hG0 h1
    have h3 : log G = θ * H r p₁ p₂ := by rw [hG, log_exp]
    have h4 := h.energy
    unfold Rex at h4
    rw [hθ] at h3
    linarith
  -- log V ≥ log (A p² E) = log 20 + log p² + θp²/2 ≥ 2 + 15 + θp²/2
  have hlogF : 17 + θ / 2 * p ^ 2 ≤ log Vv := by
    have h1 : A * p ^ 2 * E ≤ Vv := by rw [hV]; linarith
    have hAp : 0 < A * p ^ 2 * E := by rw [hA]; positivity
    have h2 : log (A * p ^ 2 * E) ≤ log Vv := log_le_log hAp h1
    have h3 : log (A * p ^ 2 * E) = log A + log (p ^ 2) + θ / 2 * p ^ 2 := by
      rw [log_mul (by rw [hA]; positivity) hE0.ne', log_mul (by rw [hA]; norm_num) (by positivity),
        hE, log_exp]
    have hlogA : 2 ≤ log A := by
      rw [hA, le_log_iff_exp_le (by norm_num)]
      have := exp_one_lt_d9
      have h2 : exp 2 = exp 1 ^ 2 := by
        rw [← exp_nat_mul]; try norm_num
      rw [h2]; nlinarith [exp_pos 1]
    have hlogp : 15 ≤ log (p ^ 2) := by
      rw [le_log_iff_exp_le (by positivity)]
      have := exp_one_lt_d9
      have h15 : exp 15 = exp 1 ^ 15 := by
        rw [← exp_nat_mul]; try norm_num
      rw [h15]
      have hpow : exp 1 ^ 15 ≤ (2.7182818286 : ℝ) ^ 15 :=
        pow_le_pow_left₀ (exp_pos 1).le this.le 15
      nlinarith
    linarith
  have hden1 : 0 < 2 + log Vv := by linarith
  -- split the numerator
  have hsplit : cRate * Vv / (2 + log Vv)
      = cRate * (1 + G) / (2 + log Vv) + cRate * (A * p ^ 2 * E) / (2 + log Vv) := by
    rw [hV]; ring
  rw [hsplit]
  have hpart1 : cRate * (1 + G) / (2 + log Vv) ≤ 1 / 8000000 * θ * G := by
    have h1 : cRate * (1 + G) / (2 + log Vv) ≤ cRate * (1 + G) / 40002 := by
      apply div_le_div_of_nonneg_left (by rw [hc]; positivity) (by norm_num) (by linarith)
    have hG1 : 1 ≤ G := by
      rw [hG]; apply one_le_exp; have := H_nonneg r p₁ p₂; rw [hθ]; positivity
    rw [hc, hθ] at *
    nlinarith
  have hpart2 : cRate * (A * p ^ 2 * E) / (2 + log Vv) ≤ 5 * θ * E := by
    have h1 : cRate * (A * p ^ 2 * E) / (2 + log Vv) ≤ cRate * (A * p ^ 2 * E) / (θ / 2 * p ^ 2) := by
      apply div_le_div_of_nonneg_left (by rw [hc, hA]; positivity) (by rw [hθ]; positivity)
        (by linarith)
    have h2 : cRate * (A * p ^ 2 * E) / (θ / 2 * p ^ 2) = 5 * θ * E := by
      rw [hc, hA, hθ]; field_simp; ring
    linarith
  linarith

end RotorFl
