#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_%j.out
# Blind skeleton search (CONCLUSIONS C45, spec 2026-09-07): candidate =
# pair(rho, ptilde), only the three-term skeleton of V known; reward = the
# out-of-box log-domain judge calibrated in example_config/blind_rotor2.json.
# Submitted by submit_rotor_blind.sh with --export=ALL,SYS,BM,ARM and
# --chdir=<snapshot>/<arm>.
set -eo pipefail

module load conda/2024.09 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alfd

export PYTHONPATH="$PWD/libs/sd3/dso"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LYA_BENCHMARK="$BM"
export LYA_SEED_START="${LYA_SEED_START:-1}"
export LYA_SEED_RUNS=1
export LYA_NSAMPLES="${LYA_NSAMPLES:-20000}"
export LYA_FL_BLIND=1

if [[ "${ARM:-transformer}" == "diffusion" ]]; then
  export LYA_CONTROLLER=DiffusionController
  export LYA_SAMPLER_ORDER="${LYA_SAMPLER_ORDER:-blocksemiar}"
  export LYA_LAMBDA="${LYA_LAMBDA:-0.5}"
fi

echo "[rotor-blind] sys=$SYS bm=$BM arm=${ARM:-transformer} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[rotor-blind] done sys=$SYS arm=${ARM:-transformer} end=$(date -Is)"
