#!/bin/bash
# Usage: submit_rotor_tmpl.sh [partition] [walltime] [systems...]
set -euo pipefail
PART="${1:-gpu}"; WALL="${2:-06:00:00}"; shift 2 || true
SYSTEMS=("${@:-rotor2 rotor3 rotor4}")
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_tmpl
for sys in ${SYSTEMS[@]}; do
  n="${sys#rotor}"
  for arm in transformer diffusion; do
    sbatch --partition="$PART" --job-name="lyatmpl_${sys}_${arm}" --time="$WALL" \
           --export=ALL,SYS="$sys",BM="fn_rotor_${n}",ARM="$arm" \
           --chdir="$RUNS_BASE/$sys/$arm" \
           /anvil/scratch/x-cchen47/ALFD-rotor5/repro/run_one_rotor_tmpl.sbatch
  done
done
