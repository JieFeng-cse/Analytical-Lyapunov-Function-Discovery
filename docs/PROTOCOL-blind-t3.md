# Blind replication protocol — N=3 t₃ (C43)

Purpose: convert the guided t₃ result (C24–C26, r=0.917133) into a **blind**
score per the benchmark's evaluation protocol §5. Question being answered:
*would the pipeline have reached t₃ without a derived target?*

## Sealed information (must NOT influence any decision inside the run)

- the derived target expression and its 7-term structure;
- its reward value 0.917133;
- the term count, the identity of the cross-bond term, the coefficients (1, −1);
- any equation from the recursion (c₂, the directional-derivative equation);
- the fact that t₃ exists in closed form at all.

The sealed values are used only AFTER the run terminates, to score it.

## Admissible signals (computable without the answer)

1. **residual reward** `r = 1/(1+mean[(p^{j-1}·Lp̃)²])` — already answer-free;
2. **decay-order test**: fit `log|residual|` vs `log p₂` on a held-out ladder of
   p₂ values. Freezing is justified only if the slope improves by ≈1 (one order);
3. **multi-arm / multi-seed consensus**: a subexpression appearing verbatim in
   the top candidate of ≥2 independent runs (different seed and/or arm);
4. **snap-and-verify**: series-expand a machine composite, extract a basis, fit
   coefficients by least squares against the residual, snap to simple rationals,
   **accept only if reward does not decrease**;
5. **stagnation**: no improvement > 0.002 over 40 epochs.

## Decision rules (fixed in advance, executed by script)

- **Freeze rule**: freeze a subexpression iff (3) consensus AND (2) the frozen
  variant's decay-order does not worsen. Never freeze on "it matches the target".
- **Round advance**: after each freeze, restart search on the remainder with
  soft-length prior recentred on the observed best-length minus frozen length.
- **Snap rule**: apply (4) to any composite of the form `f(a·r + b·p)` in the
  best expression; accept per the reward-non-decrease test.
- **Stop rule**: stop when (5) stagnation AND the decay-order slope has improved
  by ≥1 relative to the stage-2 baseline, OR when the round budget (4 rounds)
  is exhausted.
- **No human/model judgement** on expression content during the run; only the
  script's rules act.

## Scoring (after termination)

| Outcome | Verdict |
|---|---|
| final expr ≡ target (mod kernel) | **blind success** |
| reward ≥ 0.917 but expr ≢ target | blind success w/ alternative representative |
| 0.899 ≤ r < 0.917 | partial — reached the deceptive basin, snap insufficient |
| r < 0.899 | blind failure |

Also record: rounds used, wall-clock, whether each freeze decision coincided
with the (sealed) truth — a false freeze is the key risk this experiment tests.

## Deliverable

`experiments/blind_t3/` with per-round logs, the decision trace, and the verdict
appended to CONCLUSIONS as C43.

## Calibration of the answer-free signals (2026-08-27, before launch)

Deep-gate decay-slope test (p₂ ladder 120–560, ρ=1 only):

| frozen extra | slope |
|---|---|
| — (stages 1–2 only) | 1.142 |
| trig triple (3 of 7 terms) | 1.167 |
| p₁ part (2 of 7) | 1.152 |
| junk `sin r₁` | 1.132 |
| **all 7 terms (sealed)** | **0.562** |

**Finding**: the decay-order test only fires on the COMPLETE set — partial
correct subsets are indistinguishable from junk (all ≈ baseline). Consequence
for the protocol: decay-order cannot select partial freezes; it works only as
(a) a veto against order-worsening additions and (b) the stop criterion.
Selection must therefore rest on consensus + reward alone. This is itself a
blind-operation finding: **there is no answer-free signal for partial progress
within an order** — an intrinsic difficulty of blind ladder search, and a
plausible explanation of why the deceptive basin is so stable.
