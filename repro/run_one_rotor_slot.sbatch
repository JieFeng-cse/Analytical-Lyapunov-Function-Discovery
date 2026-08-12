#!/bin/bash
#SBATCH --account=cis260360-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --output=%x_%j.out
# Guided-template STAGE search (CONCLUSIONS C8-C11): transformer arm, slot g_j
# inside the fixed reference scaffold; reward = gate-weighted averaging residual.
# Submitted with --export=ALL,SYS,BM,STAGE --chdir=<snapshot>/transformer.
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
export LYA_FL_TEMPLATE=1
export LYA_FL_STAGE="${STAGE:-1}"

if [[ "${ARM:-transformer}" == "diffusion" ]]; then
  export LYA_CONTROLLER=DiffusionController
  export LYA_SAMPLER_ORDER="${LYA_SAMPLER_ORDER:-blocksemiar}"
  export LYA_LAMBDA="${LYA_LAMBDA:-0.5}"
fi

echo "[rotor-slot] sys=$SYS bm=$BM stage=$LYA_FL_STAGE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
python -u Lyapunov_test_dso.py
echo "[rotor-slot] done sys=$SYS stage=$LYA_FL_STAGE end=$(date -Is)"
