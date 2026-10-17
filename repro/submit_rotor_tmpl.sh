#!/bin/bash
# Usage: submit_rotor_tmpl.sh [partition] [walltime] [systems...]
set -euo pipefail
PART="${1:-gpu}"; WALL="${2:-06:00:00}"; shift 2 || true
SYSTEMS=("${@:-rotor2 rotor3 rotor4}")
# FREE=1: free-grammar shapes (spec section 10), run dirs under runs_rotor_free
FREE="${FREE:-0}"
if [[ "$FREE" == "1" ]]; then TAG=lyafree; DEF=runs_rotor_free; else TAG=lyatmpl; DEF=runs_rotor_tmpl; fi
RUNS_BASE="${RUNS_BASE:-/anvil/scratch/x-cchen47/LyaDis/$DEF}"
for sys in ${SYSTEMS[@]}; do
  n="${sys#rotor}"
  for arm in transformer diffusion; do
    sbatch --partition="$PART" --job-name="${TAG}_${sys}_${arm}" --time="$WALL" \
           --export=ALL,SYS="$sys",BM="fn_rotor_${n}",ARM="$arm",LYA_TMPL_FREE="$FREE" \
           --chdir="$RUNS_BASE/$sys/$arm" \
           /anvil/scratch/x-cchen47/ALFD-rotor5/repro/run_one_rotor_tmpl.sbatch
  done
done
