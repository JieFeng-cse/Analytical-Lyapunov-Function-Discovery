# N=2 / N=3: Results vs Document — Formula Tables

Source: `docs/superpowers/specs/2026-07-27-rotor-fl-benchmarks.tex`; results per
`docs/CONCLUSIONS.md` C12–C37, C41. Convention: `p̃ = p − Σ_j t_j I_j`
(uniform minus), `I_j = ρ/p^j` for `p≠0`, `0` at `p=0`.

## Table 1 — N=2 counterterms

| j | Document eq.(17) | Ours | Status |
|---|---|---|---|
| 1 | `cos r₁` | `cos r₁` | verbatim (search) |
| 2 | `p₁cos r₁` | `p₁cos r₁` | verbatim (search) |
| 3 | `p₁²cos r₁ + p₁sin r₁ + cos²r₁` | same | verbatim (search) |
| 4 | `−p₁³cos r₁ − 3p₁²sin r₁ − 3p₁cos²r₁ + p₁cos r₁ − ½sin r₁cos r₁ + 2sin r₁` (with **+**I₄) | `p₁³cos r₁ + 3p₁²sin r₁ − p₁cos r₁ + (3/2)p₁cos 2r₁ + ¼sin 2r₁ − 2sin r₁` | equivalent; difference `= (3/2)p₁ ∈ ker(∂_{r₁})` (derivation) |

## Table 2 — N=3 counterterms

| j | Document eq.(22) | Ours | Status |
|---|---|---|---|
| 1 | `cos r₁ + cos r₂` | `cos r₁ + cos r₂` | verbatim (search) |
| 2 | `p₁cos r₁ + p₃cos r₂` | `p₁cos r₁ + p₃cos r₂` | verbatim (search) |
| 3 | — (`O_f(p₂⁻³)`) | `p₁²cos r₁ + p₁sin r₁ + cos²r₁ + p₃²cos r₂ − p₃sin r₂ + cos²r₂ + cos r₁cos r₂` | **new** (pipeline + derivation) |
| 4 | — | `p₁³cos r₁ + 3p₁²sin r₁ − p₁cos r₁ + (3/2)p₁cos 2r₁ − 2sin r₁ + ¼sin 2r₁ + p₃³cos r₂ − 3p₃²sin r₂ − p₃cos r₂ + (3/2)p₃cos 2r₂ + 4sin r₂ − ¼sin 2r₂ + ¾(p₁+p₃)cos(r₁−r₂)` | **new** (derivation) |

## Table 3 — Scaffold, gates, constants

| Item | Document | Ours | Status |
|---|---|---|---|
| V | `1 + e^{θH} + Aρ p̃² e^{θp̃²/2}` (eq.13/21) | same | adopted |
| χ | `b(z−1)/[b(z−1)+b(4−z)]`, `b(u)=e^{−1/u}1_{u>0}` (eq.14) | same | adopted |
| I_j | `ρ/p₂^j`, `0` at `p₂=0` (eq.16) | same | adopted |
| S₂ (N=2) | `p₁⁴ + 4` (k=2, R_ρ=4 hardcoded) | `p₁⁴ + 20` | **corrected** (doc value infeasible) |
| ρ₃ (N=3) | not given (text only) | `χ(p₂²/S₃²)`, `S₃ = (p₁²+p₃²)² + 20` | **filled in** (per CEP2015) |
| Exempt set K | `K_R = {H ≤ R}` (energy sublevel) | `{H≤R} ∪ {ρ<1 ∧ Σp_end² ≤ M}` | **corrected** (paper-shaped) |
| θ, A | "sufficiently small θ", A unspecified | θ = 0.02 (0.05 also feasible, N=3), A = 20 | **quantified** |

## Table 4 — Drift identities

| Quantity | Document | Ours | Status |
|---|---|---|---|
| N=2 drift | `−1/(2p₂³)` (eq.18) | mechanism confirmed: `⟨c₃⟩ = −½` | confirms |
| N=3 drift | `−1/p₂³` (eq.23) | `⟨c₃⟩ = −1` | confirms (was the flagged ⚠ extraction ambiguity) |
| N=3 residual equation | — | `(∂_{r₁}−∂_{r₂}) t₃ = c₂`, `c₂ = −p₁²sin r₁ + p₁cos r₁ + p₃²sin r₂ + p₃cos r₂ − sin 2r₁ + sin 2r₂ − sin(r₁−r₂)` | **new** |
| N=3 unsolvable mode at order 4 | — | `½(p₁−p₃)sin(r₁+r₂)` (`n₁=n₂`, slow) | **new** (micro-preview of the N=5 layer-2 obstruction) |

## Table 5 — Certificate verification (not in document)

| System | Box | Violations | c\* (M=9) | c\* (M=36) |
|---|---|---|---|---|
| rotor2 | ±15 (700k pts) | 0 | 0.273 | 1.198 |
| rotor2 | ±30 falsifier | 0 | 0.281 | 1.198 |
| rotor3 | ±15 (700k pts/combo, θ×A×M grid) | 0 | 0.198 | 1.105 |
| doc instance (S₂=p₁⁴+4, K=K_R) | ±15 | **≥ 12275** ∀(θ,A,R) | infeasible | infeasible |

## Table 6 — Raw search outputs before snapping

| Target | Machine output | r | Snap |
|---|---|---|---|
| N=2 t₂ | `p₁cos r₁ − p₁ + sin p₁` | 0.9935 | drop `−p₁+sin p₁ = O(p₁³/6)` |
| N=2 t₃ | `p₁²cos r₁ + p₁sin r₁ + cos²r₁` | 0.9771 | none needed |
| N=2 t₄ | with const tokens | 0.785 | (textbook form scores 0.658) |
| N=3 t₂ | `p₁cos r₁ − p₁ + p₃cos r₂` | 0.9815 | drop `−p₁` (lowers box residual 6%) |
| N=3 t₃ r1 | `sin 2p₁ + cos²r₁ + cos r₁cos r₂ + cos²r₂ − cos(r₁+p₁) + cos(r₂+p₃)` | 0.8535 | trig triple verbatim; freeze |
| N=3 t₃ r2 | `p₁sin(r₁+p₁) + …` | 0.8992 | series: `= p₁²cos r₁ + p₁sin r₁ + O(p₁⁴)` |
| N=3 t₃ r3 | `cos(r₂+p₃) + …` | 0.8992 | basis `{p₃²cos r₂, p₃sin r₂}`, fit `(0.987, −0.953)` → `(1, −1)` |
| N=3 t₃ final | assembled | **0.917133** | == derived target |

## Table 7 — Provenance

| Item | Search | Mechanical algorithm | Derivation |
|---|---|---|---|
| N=2 t₁–t₃ | ✅ verbatim | — | — |
| N=2 t₄ | partial (const solution 0.785 > 0.658) | — | ✅ exact form |
| N=3 t₁–t₂ | ✅ verbatim | — | — |
| N=3 t₃ | trig triple verbatim + composite structures | series/coefficient snap | target + curriculum |
| N=3 t₄ | — | — | ✅ engine |
| ρ₃, S, K, θ, A, c\* | — | calibration scan | paper-shaped design |
