# rotor-fl — Lean 4 formalization of the rotor2 Foster–Lyapunov tail

Spec: `docs/superpowers/specs/2026-09-07-lean-tail-certificate-rotor2-design.md`.
Ledger of results: `docs/CONCLUSIONS.md` (C44 onwards).

The build tree lives on scratch (`/anvil/scratch/x-cchen47/lean/rotor-fl`,
symlinked from `~/lean/rotor-fl`); this directory holds the sources only.

## Build on Anvil

The Lean toolchain's bundled clang needs GLIBC ≥ 2.29 (Anvil has 2.28), so C
compilation goes through a gcc wrapper, and Mathlib's cache tool needs the
system CA bundle and the `mathlib4-master` container:

```bash
export PATH="$HOME/.elan/bin:$PATH"
module load curl/7.76.1
export LEAN_CC=~/lean/leancc.sh
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export MATHLIB_CACHE_GET_URL=https://lakecache.blob.core.windows.net/mathlib4-master
cd ~/lean/rotor-fl && lake exe cache get && lake build
```

Home quota is 25 GB; elan, the Mathlib cache and the build tree are on scratch
with symlinks (`~/.elan`, `~/.cache/mathlib`, `~/lean`).

## Modules

| file | content | status |
|---|---|---|
| `RotorFl/Defs.lean` | dynamics, generator `L`, gate `ρ`, counterterms, `V`, `K`, `φ`, statements T1/T2 | builds |
| `RotorFl/Slow.lean` | Ω₁ (gate closed): `L_V_slow`, `slow_region` | proved, no sorry |

Sync from the build tree: `rsync -a --exclude .lake ~/lean/rotor-fl/ lean/rotor-fl/`.
