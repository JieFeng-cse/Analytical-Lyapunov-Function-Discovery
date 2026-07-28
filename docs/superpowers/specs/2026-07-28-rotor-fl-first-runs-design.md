# Rotor-chain Foster–Lyapunov first runs — design

Date: 2026-07-28. Status: approved by user (this conversation).
Benchmark source: `docs/superpowers/specs/2026-07-27-rotor-chain-foster-lyapunov-benchmarks.md`.

## Goal

First A/B experiment on the four rotor-chain stochastic Foster–Lyapunov benchmarks
(N=2,3,4,5): transformer generator vs the best diffusion configuration found by the
self-improvement loop (loop_20: entbonus + blocksemiar B=4). 4 systems × 2 arms ×
1 seed = **8 runs**.

## Decisions (user-confirmed)

1. **Finite-domain track** (benchmark open question 1): reward and falsification are
   evaluated numerically on a box; no symbolic tail certification in this round. The
   reward evaluates the **full stochastic generator including second derivatives** —
   first-order-only was explicitly rejected (it defines a different problem).
2. **Bond coordinates**: state = (r_1..r_{N−1}, p_1..p_N), dimension 2N−1
   (N=2→3, 3→5, 4→7, 5→9). Dynamics and reference certificates depend on q only
   through r. Candidate expressions in raw r are not forced to be periodic — accepted
   under the finite-domain track; periodicity is checked post hoc, not in-loop.
3. **Architecture: extend the existing pipeline** (approach A). New metric in
   `regression.py` + per-system `generator_spec()` in the config. No new DSO task type.

## Branch strategy

New branch `rotor-benchmarks` forked from `loop/20-entbonus_blocksemiarb4`
(commit 5700a56) so the diffusion arm gets the blocksemiar-B4 sampler + entbonus
training term with their loop_20 defaults, and the transformer arm is selected the
usual way (`LYA_CONTROLLER` unset). Frozen-surface files (benchmarks CSV,
`regression.py`, `Lyapunov_test_dso.py`) may be edited on this branch: the frozen
guard only runs for loop-protocol submissions, and loop branches form their own
lineage — mainline A/B comparability is not affected.

## New systems

Dynamics in bond coordinates: dr_i = (p_{i+1} − p_i) dt, dp per the benchmark spec
(w(r)=sin r, endpoint friction −p_1, −p_N, noise variances 2γ₁T₁=2 and 2γ_N T_N=4;
N=2 one-bath: only p_1 damped/noisy).

| config | dim | variables | box | FL params |
|---|---|---|---|---|
| `config_rotor2.py` | 3 | r1, p1, p2 | r∈[−π,π], p∈[−6,6] | α=1, c=0.05, R=10 |
| `config_rotor3.py` | 5 | r1,r2, p1..p3 | same | α=1, c=0.05, R=10 |
| `config_rotor4.py` | 7 | r1..r3, p1..p4 | same | α=1, c=0.05, R=10 |
| `config_rotor5.py` | 9 | r1..r4, p1..p5 | same | α=3, c=0.05, R=10 |

- K_R = {H_N ≤ R=10} lies strictly inside every box (box-max of H_N ≈ 38–98).
- α is supplied to the scorer (benchmark open question 3: "given α" side).
- Each config exports the existing `dynamics()` (drift only, for the Lie part and the
  dynamics encoder) plus a new `generator_spec()` returning:
  `{"noise": {p_index: γT coefficient}, "alpha": α, "c": 0.05, "R": 10.0,
    "H": <sympy expr for H_N in bond coords>}`.
- New rows `fn_rotor_{2,3,4,5}` in `benchmarks_bkup_1.csv` with the box bounds and
  15000 training samples, dummy target expression, dims 3/5/7/9.
- New rows in `repro/systems.tsv` (rotor2..rotor5).

## Reward (new metric `foster_lyapunov` in regression.py)

For candidate V (sympy expr from tokens):

- LV = Σ_i ∂V/∂x_i · f_i + Σ_b κ_b ∂²V/∂p_b², with κ from `generator_spec().noise`
  (sympy `diff` twice + `lambdify`, cached per Program like `sympy_expr_lie`).
- φ(V) = c·V/(α+1+log V)^α, evaluated where V≥1 (clamp V below at 1 for φ's arg).
- penalty = mean(relu(LV + φ(V)) · 1_{H > R}) + 2.5 · mean(relu(1 − V))
- reward = 1/(1 + penalty); invalid_reward = 0.
- No V(0)=0 shift, no PD-at-origin logic (Foster–Lyapunov has no origin anchor).
- The "all variables used" validity rule stays.

Success criterion mirrors the existing one: violation measure < threshold on the big
test sample, then the adapted falsifier passes: SHGO (penalty-masked) plus
10×10⁶-point random sampling maximize LV + φ(V) outside K_R; torch-PGD is deferred
(log/exp objectives are numerically unsafe under unconstrained attack); in-loop
harvesting appends top-50 random-search violators per epoch. Coercivity and periodicity are
NOT checked in-loop; they are offline post-hoc checks on the winning expressions.

## Grammar and priors

- function_set: `["add","sub","mul","div","sin","cos","exp","n2"]` (`exp` required by
  the reference form 1+e^{θH}; fractional θ expressible via `div`; const optimization
  stays off, matching existing configs).
- length prior: min 10, max 35, soft_length loc 25 (reference certificates are longer
  than the ODE-benchmark solutions).
- batch_size 500, training defaults otherwise copied from `config_lossless6d.py`.

## Runs

- 8 SLURM jobs: {rotor2,rotor3,rotor4,rotor5} × {transformer, diffusion}, seed 1,
  16 h walltime, 1 GPU each.
- New `repro/setup_runs_rotor.sh` + `repro/run_one_rotor.sbatch` modeled on the loop
  runner; diffusion arm sets `LYA_CONTROLLER=DiffusionController` + the loop_20
  sampler env (blocksemiar order, B=4 default from the commit); transformer arm sets
  neither. Output under `/anvil/scratch/x-cchen47/LyaDis/runs_rotor/<sys>/<arm>/`.
- Report per run: best reward curve, best expression, violation statistics from the
  final falsifier, wall-clock; success = falsifier finds no violation.

## Known risks

1. Dynamics-encoder tokenizer supports only add/mul/pow/sin/cos — subtractions in the
   bond dynamics must be written as `+ (−1)·(...)` when building encoder input.
2. `exp` under |p|≤6 can reach e^≈100 ≈ 1e43 — float64 handles it, but lambdified
   evaluations are clipped to avoid inf propagating into the metric.
3. rotor5 (9-dim, α=3, conjectural) is expected to fail for both arms; that is itself
   a valid benchmark data point ("find or falsify").
4. Candidate expressions may exploit non-periodicity in r (finite-domain artifact);
   flagged in post-hoc checks, accepted this round.

## Out of scope this round

Global/tail certification, dReal verification of FL certificates, smooth-cutoff
grammar primitives (benchmark open question 4), parameter-sampled baths (open
question 2), multi-seed sweeps, N=5 joint search over α.
