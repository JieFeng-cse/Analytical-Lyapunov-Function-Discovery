#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_%j.out
# Rotor-benchmark runner: one fixed-seed (seed=1) run per (system, arm).
# Submitted by submit_rotor.sh, which sets --job-name=lyarot_<sys>_<arm>, --time,
# and exports SYS, BM, ARM ("transformer" | "diffusion"); --chdir points at the
# pre-built run dir for (SYS, ARM) under runs_rotor/.
# NOTE: no `set -u` — conda's activate.d scripts reference unset vars.
set -eo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor
RUNDIR="$RUNS_BASE/$SYS/$ARM"

module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd

cd "$RUNDIR"
export PYTHONPATH="$RUNDIR/libs/sd3/dso"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK="$BM"
export LYA_SEED_START=1
export LYA_SEED_RUNS=1
# full reproduction: do NOT set LYA_NSAMPLES (uses paper's 2,000,000)

if [[ "${ARM}" == "diffusion" ]]; then
  export LYA_CONTROLLER=DiffusionController
  export LYA_SAMPLER_ORDER=blocksemiar
  export LYA_LAMBDA="${LYA_LAMBDA:-0.5}"
fi

echo "[rotor] sys=$SYS arm=$ARM benchmark=$BM seed=1 host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[rotor] done sys=$SYS arm=$ARM end=$(date -Is)"
