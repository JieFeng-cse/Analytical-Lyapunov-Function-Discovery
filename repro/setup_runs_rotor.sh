#!/bin/bash
# Snapshot the rotor-benchmarks worktree into per-(system,arm) run dirs.
# Usage: setup_runs_rotor.sh   (run from the rotor worktree root)
set -euo pipefail
ROOT=/home/x-cchen47/ALFD-rotor
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor
cd "$ROOT"
for sys in rotor2 rotor3 rotor4 rotor5; do
  for arm in transformer diffusion; do
    dst="$RUNS_BASE/$sys/$arm"
    rm -rf "$dst"; mkdir -p "$dst"
    tar cf - --exclude=.git --exclude=log --exclude=logs --exclude=experiments . \
      | ( cd "$dst" && tar xf - )
    cp "$dst/example_config/config_${sys}.py" "$dst/config.py"
    mkdir -p "$dst/log" "$dst/logs"
  done
done
echo "run dirs ready under $RUNS_BASE"
