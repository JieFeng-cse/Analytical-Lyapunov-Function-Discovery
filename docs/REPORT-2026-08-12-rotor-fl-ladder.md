# Guided-Template Discovery of Foster–Lyapunov Counterterms for Boundary-Thermostatted Rotor Chains

Report, 2026-08-12. Covers CONCLUSIONS C1–C35 (docs/CONCLUSIONS.md holds the
dated evidence trail; this report synthesizes it). Code: branch
`rotor-benchmarks`, commit 8ff3be7.

## 1. Problem and approach

The rotor FL benchmark (spec: docs/superpowers/specs/2026-07-27-*.tex) asks a
symbolic-discovery pipeline to produce stochastic Foster–Lyapunov certificates
`L_N V ≤ C·1_K − c·V/(α+1+log V)^α` for chains of N=2..5 rotors with boundary
Langevin baths. The certificates' hard content is a series of **averaging
counterterms** g_j inside a corrected fast momentum
`p̃ = p_f − Σ_j g_j·I_j`; the reference constructions (Cuneo–Eckmann–Poquet)
publish these only partially (N=2: 4 orders; N=3: 2 orders; N=4: structure
only; N=5: conjecture).

Instead of free-form search over whole certificates (v0; defeated by
box-truncation loopholes — a trivial polynomial "solved" rotor2 in 3 epochs),
we fixed the reference **scaffold** (gates, exponential body, I_j
regularization) and made the searched object the slot expressions g_j
(CONCLUSIONS C8). Stage-j search scores the **averaging residual**
`mean_{gate fully open}[(p_f^{j-1}·L p̃)²]` — bounded numbers, independent of
the certificate-level constants θ/A/c/R (C12), evaluated via a linearized
fast path (~400×; numerically identical to the symbolic route, C17).

Searcher: the published ALFD pipeline (transformer policy + risk-seeking PG +
GP-meld) with a 6-operator slot grammar; diffusion arm (loop_20 config) used
for the A/B in §4.

## 2. Ladder results

| System | order 1 | order 2 | order 3 | order 4 |
|---|---|---|---|---|
| rotor2 (N=2) | ✅ verbatim, ep.1 | ✅ core verbatim + snap | ✅ **verbatim, zero decoration** | machine-with-constants **beats textbook** (0.785 vs 0.658); search closed, derived t₄ adopted |
| rotor3 (N=3) | ✅ verbatim, ep.1 | ✅ verbatim core (+6% box improvement) | ✅ **full reconstruction of a term absent from the literature** (r=0.917133 == derived target) | — |
| rotor4 (N=4) | ✅ tied bond-local | ✅ tied bond-local | patch-0 ✅ reconstructed, **kernel representative improved by the machine**; patch-1 derived-adopted | — |

Every counterterm published in the literature (7/7) was rediscovered exactly.
The N=3 third-order term — not written in any reference — was reconstructed
end-to-end:

```
t₃^{N=3} = p₁²cos r₁ + p₁sin r₁ + cos²r₁
         + p₃²cos r₂ − p₃sin r₂ + cos²r₂ + cos r₁·cos r₂   (cross-bond term)
```

with provenance: 3 trig terms found verbatim by direct search; 2 p₁-terms via
series-snap of a machine-invented compact composite p₁sin(r₁+p₁); 2 p₃-terms
via basis extraction + linear coefficient snap (exact integers (1,−1)). An
independent symbolic derivation (1/p₂-series + solving (∂r₁−∂r₂)t₃ = c₂)
confirms every term (C21, C26).

For rotor4, friction asymmetry and the light-cone spread of the recursion
(each order reaches one bond further) break the tied-patch design at order 3
(C22, derivation-confirmed); per-patch search then reconstructed patch-0's
term with a **better kernel representative than the derivation**: the machine
chose sin r₁sin r₂ over cos r₁cos r₂ (difference cos(r₁+r₂) ∈ ker(∂r₁−∂r₂)),
scoring 0.8832 vs 0.8801 (C34). The same substitution does NOT help patch-1
(0.845 vs 0.848) — kernel choice is patch-specific (C35).

## 3. The finite-domain micro-optimization phenomenon (7 instances)

In every completed stage, the machine's best expression scored slightly-to-
substantially better than the asymptotically-derived answer on the finite box:
decorations of order O(p³/6) in the deep-gate region (3×), a −p₁ term (+6%),
continuous coefficients (0.987, −0.953) vs integers (1, −1), free constants at
order 4 (0.785 vs 0.658 — the largest gap), and one kernel-level improvement
(C34). Asymptotic optimality and finite-domain optimality systematically
diverge; the machine reliably finds the gap.

## 4. Search-capability findings

- **Wall**: at ~25-token/7-term slots both arms plateau (transformer 0.869 @
  435 ep; diffusion 0.781 @ 215 ep) with identical failure modes — trig
  backbone found verbatim, momentum polynomials approximated by cheap trig
  composites (cos(r+p) Taylor tricks). The bottleneck is the search paradigm,
  not the generator architecture (C23).
- **Within-order curriculum** (freeze consensus terms, search the remainder)
  breaks the wall when the remainder still admits cheap approximants; when
  the remainder must be exact, it exposes rather than reduces difficulty
  (C24, C32). Single runs are coin flips (patch-0 micro-curriculum succeeded
  where its twin and patch-1's failed).
- **Snap toolbox**: series-expansion of machine composites recovers exact
  polynomial terms; linear coefficient fit (the residual is affine in the
  slot) recovers exact rationals — reliable at ≤2–3 basis elements, ill-
  conditioned at 5 (C31).
- **const tokens** cost ~20× per epoch (BFGS inner loop) and crossed three
  distinct DSO code paths (p.r / update_r / GP pool) each hiding a bug
  (C29, C30).

## 5. Benchmark-design findings (feedback to the spec)

1. The spec's finite-domain track with p∈[−6,6] boxes is trivially gameable
   (C2); ±15 training boxes + oversized falsifier boxes are the working
   minimum.
2. The spec's explicit gate instance (S=p₁⁴+4, i.e. k=2, R_ρ=4) and
   energy-sublevel exempt set K_R make the spec's own reference certificate
   infeasible pointwise (C13). The original paper uses K = {gate not fully
   open ∧ endpoint momenta ≤ M} — not an energy sublevel — and treats the
   gate transition band by pointwise exponential domination with k, R_ρ
   "large enough" (C28). With the paper-shaped K and R_ρ=20, θ=0.02, the
   full reference is numerically feasible on the ±15 box with c* ∈
   [0.27, 1.20].
3. The t₄ sign convention in spec eq.(17) (+ before I₄) and a kernel-freedom
   ambiguity (3p₁cos²r₁ vs (3/2)p₁cos2r₁ representatives) are worth
   clarifying in the next draft.

## 6. Limitations

- All stage-level acceptances are residual-order acceptances on a finite box;
  the assembled-certificate FL verification (paper-shaped K) has been
  demonstrated for the N=2 reference but not yet run for the machine-built
  certificates (next step).
- n=1 seeds per search run except where noted; search-success claims are
  existence claims, not rate claims.
- rotor4 patch-1's order-3 term and rotor2's exact t₄ are derivation-adopted,
  not search-found — recorded as such.
- The v1 scaffold omits the |p̃|^a factor of the N=4 reference and the
  resonance piece V_c entirely; N=4 assembled verification will need them or
  an honest regime restriction.

## 7. Next steps

1. **Assembled-certificate verification** (in progress): plug the ladder's
   frozen slots into the full scaffold, verify LV ≤ C·1_K − c·φ(V) with the
   paper-shaped K per system; produces the benchmark's first machine-built
   verified certificates.
2. **N=5 conjecture campaign**: derivation-first reconnaissance (low-order
   counterterms of the p̃₃⁶-weighted ansatz are mechanically derivable), then
   the ladder. All infrastructure (per-patch, curriculum, snap) is in place.
3. Report the §5 findings to the benchmark draft's author.
