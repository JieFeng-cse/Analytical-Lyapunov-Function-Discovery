#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=lyablind_smoke
#SBATCH --output=%x_%j.out
# Smoke test for the blind skeleton search (spec 2026-09-07): rotor2, 3 epochs,
# on gpu-debug. Submit with --export=ALL,ARM=<transformer|diffusion>; runs in the
# rotor2/<arm> snapshot built by setup_runs_rotor_blind.sh.
set -eo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_blind
RUNDIR="$RUNS_BASE/rotor2/$ARM"

module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd

cd "$RUNDIR"
export PYTHONPATH="$RUNDIR/libs/sd3/dso"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK=fn_rotor_2
export LYA_SEED_START=1
export LYA_SEED_RUNS=1
export LYA_NSAMPLES="${LYA_NSAMPLES:-1500}"
export LYA_FL_BLIND=1

if [[ "${ARM}" == "diffusion" ]]; then
  export LYA_CONTROLLER=DiffusionController
  export LYA_SAMPLER_ORDER=blocksemiar
  export LYA_LAMBDA="${LYA_LAMBDA:-0.5}"
fi

echo "[blind-smoke] arm=$ARM host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[blind-smoke] done arm=$ARM end=$(date -Is)"
