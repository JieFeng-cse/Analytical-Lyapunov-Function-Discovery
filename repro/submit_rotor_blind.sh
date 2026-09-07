#!/bin/bash
set -euo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_blind
for arm in transformer diffusion; do
  sbatch --job-name="lyablind_rotor2_${arm}" --time=06:00:00 \
         --export=ALL,SYS=rotor2,BM=fn_rotor_2,ARM="$arm" \
         --chdir="$RUNS_BASE/rotor2/$arm" \
         /home/x-cchen47/ALFD-rotor/repro/run_one_rotor_blind.sbatch
done
