#!/bin/bash
# Snapshot the rotor5-template worktree into per-(system, arm) run dirs for the
# template slot search (spec 2026-10-17). Usage: setup_runs_rotor_tmpl.sh [systems...]
set -euo pipefail
ROOT=/anvil/scratch/x-cchen47/ALFD-rotor5
# RUNS_BASE env overrides (free grammar: runs_rotor_free)
RUNS_BASE="${RUNS_BASE:-/anvil/scratch/x-cchen47/LyaDis/runs_rotor_tmpl}"
SYSTEMS=("${@:-rotor2 rotor3 rotor4}")
cd "$ROOT"
for sys in ${SYSTEMS[@]}; do
  for arm in transformer diffusion; do
    dst="$RUNS_BASE/$sys/$arm"
    rm -rf "$dst"; mkdir -p "$dst"
    tar cf - --exclude=.git --exclude=log --exclude=logs --exclude=experiments --exclude=lean \
        --exclude=repro/runs --exclude='*.out' . | ( cd "$dst" && tar xf - )
    cp "$dst/example_config/config_${sys}.py" "$dst/config.py"
    mkdir -p "$dst/log" "$dst/logs"
  done
done
echo "run dirs ready under $RUNS_BASE"
