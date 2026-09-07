#!/bin/bash
# Snapshot the rotor-benchmarks worktree into per-arm run dirs for the blind
# skeleton search (rotor2 only). Usage: setup_runs_rotor_blind.sh (from the worktree root)
set -euo pipefail
ROOT=/home/x-cchen47/ALFD-rotor
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_rotor_blind
cd "$ROOT"
for arm in transformer diffusion; do
  dst="$RUNS_BASE/rotor2/$arm"
  rm -rf "$dst"; mkdir -p "$dst"
  tar cf - --exclude=.git --exclude=log --exclude=logs --exclude=experiments --exclude=lean --exclude=repro/runs . \
    | ( cd "$dst" && tar xf - )
  cp "$dst/example_config/config_rotor2.py" "$dst/config.py"
  mkdir -p "$dst/log" "$dst/logs"
done
echo "run dirs ready under $RUNS_BASE"
