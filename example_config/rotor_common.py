"""Rotor-chain benchmark systems in bond coordinates (spec 2026-07-28).

State: x1..x_{N-1} = bond angles r_i = q_{i+1}-q_i, x_N..x_{2N-1} = momenta p_i.
Drift: dr_i = p_{i+1}-p_i;
  N=2 (one bath):  dp1 = sin r1 - p1,  dp2 = -sin r1
  N>=3 (two baths): dp1 = sin r1 - p1, dp_i = sin r_i - sin r_{i-1},
                    dpN = -sin r_{N-1} - pN
Generator second-derivative coefficients: gamma_b*T_b = 1 at p1, 2 at pN (N>=3).
"""
import os

import sympy as sym
from omegaconf import OmegaConf

# Guided-template mode (CONCLUSIONS C8): LYA_FL_TEMPLATE=1 switches the run to
# slot search inside the fixed reference scaffold (utils/fl_scaffold.py).
FL_TEMPLATE = os.environ.get("LYA_FL_TEMPLATE", "") == "1"
# Blind skeleton search (CONCLUSIONS C45, spec 2026-09-07): LYA_FL_BLIND=1
# searches pair(rho, ptilde) with only the three-term skeleton of V known.
FL_BLIND = os.environ.get("LYA_FL_BLIND", "") == "1"
assert not (FL_BLIND and FL_TEMPLATE), "LYA_FL_BLIND and LYA_FL_TEMPLATE are mutually exclusive"
BLIND_FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "cos", "exp", "n2", "step", "pair", "1", "2"]
BLIND_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind_rotor2.json")

# R by the zero-search fallback rule R ~ 10+5N on the +-15 momentum box
# (CONCLUSIONS C1/C2); provisional until the V_ref calibration script lands.
_R_BY_N = {2: 20.0, 3: 25.0, 4: 30.0, 5: 35.0}

SLOT_FUNCTION_SET = ["add", "sub", "mul", "n2", "sin", "cos"]
# "const" only from stage 4 on: the stage 1-3 reference answers are const-free,
# and each const costs a BFGS inner loop per candidate. The sympy_expr cache
# staleness this used to hit is handled in regression.py's FL paths.
SLOT_FUNCTION_SET_CONST = SLOT_FUNCTION_SET + ["const"]

def _slot_prior_overrides():
    # Reference slot lengths grow with the stage (g1: 2-5 tokens, g3: ~14,
    # g4: ~40); keep the soft-length prior centered near the expected size so
    # it stops suppressing the correct answer at higher stages.
    stage = int(os.environ.get("LYA_FL_STAGE", "1"))
    loc, max_ = (6, 20) if stage <= 2 else (14, 30) if stage == 3 else (20, 45)
    # optional per-run override when the expected slot length is known
    # (e.g. rotor3 stage-3 derived target is ~25 tokens)
    loc = int(os.environ.get("LYA_FL_SOFTLOC", loc))
    max_ = int(os.environ.get("LYA_FL_LENMAX", max_))
    return {
        "length": {"min_": None, "max_": max_, "on": True},
        "soft_length": {"loc": loc, "scale": 5, "on": True},
        "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
    }


def apply_blind_overrides(cfg):
    """Mutate a config_factory() dict for blind mode (spec 2026-09-07 section 2)."""
    if not FL_BLIND:
        return cfg
    cfg["task"]["function_set"] = list(BLIND_FUNCTION_SET)
    cfg["prior"]["root"] = {"token": "pair", "on": True}
    # Periodicity in the bond angle: x1 (= r1) may only be the direct child of
    # sin/cos. Without this the sampler finds terms linear in r1, whose drift
    # dr1/dt = p2 - p1 fakes a negative drift (smoke job 20471334, epoch 0).
    cfg["prior"]["relational"] = {"targets": ["x1"],
                                  "effectors": [t for t in BLIND_FUNCTION_SET
                                                if t not in ("sin", "cos", "1", "2")],
                                  "relationship": "child", "on": True}
    # no step inside step, no exp inside exp: nested compositions blew up the
    # derivative code generation (hours per candidate, C50)
    cfg["prior"]["nonest"] = {"tokens": ["step", "exp"], "on": True}
    cfg["prior"]["length"] = {"min_": 5, "max_": 60, "on": True}
    cfg["prior"]["soft_length"] = {"loc": 28, "scale": 8, "on": True}
    cfg["prior"]["const"] = {"on": False}
    cfg["prior"].pop("repeat", None)
    return cfg


def apply_slot_overrides(cfg):
    """Mutate a config_factory() dict for template mode. No-op otherwise
    (blind mode applies its own overrides here as well)."""
    if not FL_TEMPLATE:
        return apply_blind_overrides(cfg)
    stage = int(os.environ.get("LYA_FL_STAGE", "1"))
    want_const = os.environ.get("LYA_FL_CONST", "1" if stage >= 4 else "0") == "1"
    cfg["task"]["function_set"] = list(
        SLOT_FUNCTION_SET_CONST if want_const else SLOT_FUNCTION_SET)
    for k, v in _slot_prior_overrides().items():
        cfg["prior"][k] = dict(v)
    return cfg


def get_config(skip_cli=True):
    """Mirror config.py:get_config for run dirs where a rotor config is copied to
    config.py. Lyapunov_test_dso only uses the skip_cli=True path."""
    return OmegaConf.load("config.yaml")


def make_rotor(N):
    dim = 2 * N - 1
    xs = sym.symbols(f"x1:{dim + 1}")            # x1..x_{2N-1}
    r = list(xs[: N - 1])
    p = list(xs[N - 1:])

    drift_r = [p[i + 1] - p[i] for i in range(N - 1)]
    if N == 2:
        drift_p = [sym.sin(r[0]) - p[0], -sym.sin(r[0])]
        noise = {N - 1: 1.0}                     # index of p1
    else:
        drift_p = [sym.sin(r[0]) - p[0]]
        for i in range(1, N - 1):
            drift_p.append(sym.sin(r[i]) - sym.sin(r[i - 1]))
        drift_p.append(-sym.sin(r[N - 2]) - p[N - 1])
        noise = {N - 1: 1.0, 2 * N - 2: 2.0}     # indices of p1, pN

    H = sum(pi**2 for pi in p) / 2 + sum(1 - sym.cos(ri) for ri in r)
    spec = {"noise": noise, "alpha": 3 if N == 5 else 1,
            "c": 0.05, "R": _R_BY_N[N], "H": H}
    if FL_TEMPLATE:
        # theta/A only matter in full (final-verification) mode; stage-mode
        # rewards are theta/A/c/R-free (utils/fl_scaffold.fl_stage_penalty).
        spec["scaffold"] = {"theta": 0.1, "A": 20.0}
        # Stage-j slot search (C10); frozen lower orders arrive with stage>=2
        # via LYA_FL_FROZEN, a JSON dict {order: expr} in patch-1 coordinates,
        # e.g. '{"1": "cos(x1)"}'. Parsed with sympify over the state symbols.
        spec["stage"] = int(os.environ.get("LYA_FL_STAGE", "1"))
        # N=4 stage>=3: search ONE patch at a time in actual coordinates (C22).
        if os.environ.get("LYA_FL_PATCH", "") != "":
            spec["patch_index"] = int(os.environ["LYA_FL_PATCH"])
        frozen_env = os.environ.get("LYA_FL_FROZEN", "")
        if frozen_env:
            import json as _json
            local = {str(s): s for s in xs}
            spec["frozen_slots"] = {int(k): sym.sympify(v, locals=local)
                                    for k, v in _json.loads(frozen_env).items()}
        else:
            spec["frozen_slots"] = {}
    if FL_BLIND:
        import json as _json
        assert N == 2, "blind mode is calibrated for N=2 only (example_config/blind_rotor2.json)"
        with open(BLIND_JSON) as fh:
            spec["blind"] = _json.load(fh)
    return list(xs), drift_r + drift_p, spec
