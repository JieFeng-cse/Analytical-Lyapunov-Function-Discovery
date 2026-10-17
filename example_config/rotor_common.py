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
# Template slot search (spec 2026-10-17): LYA_FL_TMPL=1 samples tmpl<n>(slot choices)
# and scores them with the log-domain / tropical / splice evaluator (utils/fl_tmpl).
FL_TMPL = os.environ.get("LYA_FL_TMPL", "") == "1"
# Free grammar (spec 2026-10-17 section 10): with LYA_FL_TMPL=1, LYA_TMPL_FREE=1 samples
# free<n>(g_1..g_n), one local shape g(u) per part (u = |p~_j| from the averaging macro)
FL_TMPL_FREE = FL_TMPL and os.environ.get("LYA_TMPL_FREE", "") == "1"
assert not (FL_TMPL and (FL_BLIND or FL_TEMPLATE)), "LYA_FL_TMPL excludes LYA_FL_BLIND / LYA_FL_TEMPLATE"
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


def apply_tmpl_overrides(cfg, N):
    """Template slot search: the library is the root tmpl<n> + the slot choice tokens;
    the only structural prior is the positional slot constraint (length is n+1)."""
    from utils.fl_tmpl.slots import SLOT_MENUS, root_name, token_names
    n = len(SLOT_MENUS[N])
    cfg["task"]["function_set"] = token_names(N)
    # DSO's load_config merges the default priors back in (config_common.json):
    # every default must be switched off explicitly -- no_inputs in particular
    # forbids every token at the last slot (no input variable in the program) and
    # turned the transformer's logits into NaN (local smoke 2026-10-17)
    off = {"on": False}
    cfg["prior"] = {"slot": {"root": root_name(N), "on": True},
                    "length": {"min_": None, "max_": n + 1, "on": True},
                    "relational": dict(off), "repeat": dict(off), "inverse": dict(off),
                    "trig": dict(off), "const": dict(off), "no_inputs": dict(off),
                    "uniform_arity": dict(off), "soft_length": dict(off)}
    return cfg


def apply_free_overrides(cfg, N):
    """Free grammar: root free<n>, shape trees over u and the constants 1, 2; the
    state variables x_i are never allowed (relational ban under every token)."""
    from utils.fl_tmpl.freeform import G_TOKENS, root_name
    from utils.fl_tmpl.reference import free_parts
    n = len(free_parts(N))
    root = root_name(n)
    fs = [root] + list(G_TOKENS)
    cfg["task"]["function_set"] = fs
    off = {"on": False}
    xs = [f"x{i}" for i in range(1, 2 * N)]
    lmax = int(os.environ.get("LYA_TMPL_LENMAX", str(1 + 14 * n)))
    cfg["prior"] = {"root": {"token": root, "on": True},
                    "relational": {"targets": xs, "effectors": fs, "relationship": "child", "on": True},
                    "nonest": {"tokens": ["exp", "log"], "on": True},
                    "inverse": {"on": True},
                    "length": {"min_": 1 + n, "max_": lmax, "on": True},
                    "soft_length": {"loc": 1 + 5 * n, "scale": 4, "on": True},
                    "repeat": dict(off), "trig": dict(off), "const": dict(off),
                    "no_inputs": dict(off), "uniform_arity": dict(off)}
    return cfg


def apply_slot_overrides(cfg, N=None):
    """Mutate a config_factory() dict for template mode. No-op otherwise
    (blind / tmpl modes apply their own overrides here as well)."""
    if FL_TMPL:
        assert N is not None, "pass N to apply_slot_overrides for LYA_FL_TMPL"
        if FL_TMPL_FREE:
            return apply_free_overrides(cfg, N)
        return apply_tmpl_overrides(cfg, N)
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
    if FL_TMPL:
        from fractions import Fraction as _Fr
        from utils.fl_tmpl.slots import THETA_DEFAULT
        spec["template"] = {"N": N, "theta": str(_Fr(os.environ.get("LYA_TMPL_THETA", str(THETA_DEFAULT[N])))),
                            "n_samples": int(os.environ.get("LYA_TMPL_NSAMPLES", "32")),
                            "free": FL_TMPL_FREE}
    if FL_BLIND:
        import json as _json
        assert N == 2, "blind mode is calibrated for N=2 only (example_config/blind_rotor2.json)"
        with open(BLIND_JSON) as fh:
            spec["blind"] = _json.load(fh)
    return list(xs), drift_r + drift_p, spec
