#!/bin/bash
set -euo pipefail
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor
declare -A BM=( [rotor2]=fn_rotor_2 [rotor3]=fn_rotor_3 [rotor4]=fn_rotor_4 [rotor5]=fn_rotor_5 )
for sys in rotor2 rotor3 rotor4 rotor5; do
  for arm in transformer diffusion; do
    sbatch --job-name="lyarot_${sys}_${arm}" --time=16:00:00 \
           --export=ALL,SYS="$sys",BM="${BM[$sys]}",ARM="$arm" \
           --chdir="$RUNS_BASE/$sys/$arm" \
           /home/x-cchen47/ALFD-rotor/repro/run_one_rotor.sbatch
  done
done
