"""Guided-template scaffold for Foster-Lyapunov certificates (CONCLUSIONS C8-C10).

Fixed reference structure (benchmark doc eq.13/21):

    V = 1 + exp(theta*H) + sum_f A * rho_f * ptilde_f^2 * exp(theta/2 * ptilde_f^2)
    ptilde_f = p_f - sum_j t_j(neighbor vars) * I_j,     I_j = rho_f / p_f^j

Two operating modes:

* **Stage mode** (spec["stage"] = j >= 1): the searched slot G sits at series
  order j; orders < j are frozen expressions (spec["frozen_slots"]); orders > j
  are absent. The reward object is the AVERAGING RESIDUAL
      pen = sum_patches  sum(rho * (L ptilde)^2) / sum(rho)
  -- C10's acceptance criterion (each correct order knocks L ptilde down one
  power of p_f). It involves neither theta/A nor c/R, so stage jobs are
  independent of the final-verification calibration.
* **Full mode** (no "stage"): candidate slot completes the series; the assembled
  V/LV pair feeds the ordinary FL reward/falsifier (final verification only).

Gates are fixed:  b(u)=exp(-1/u) [u>0],  chi(z)=b(z-1)/(b(z-1)+b(4-z)),
S_f = sum(neighbor momenta^4)+4,  rho_f = chi(p_f^2/S_f^2).  I_j is a Piecewise
vanishing for p_f^2 <= 1 (exact: rho_f = 0 wherever |p_f| <= S_f >= 4).

Multi-patch systems (N=4) tie every patch to ONE searched bond-local slot via
index-shift substitution (cf. N=2 g1=cos r1 vs N=3 g1=cos r1+cos r2).

Heavy symbolic work is done once per process with G an undetermined Function;
assembling a candidate only substitutes G and differentiates the small slot.
"""
import sympy as sym

from utils.fl_reward import generator_expr

_BUILT = {}


def _chi(z):
    b = lambda u: sym.Piecewise((sym.exp(-1 / u), u > 0), (0, True))
    b1, b2 = b(z - 1), b(4 - z)
    return b1 / (b1 + b2)


class Scaffold:
    def __init__(self, state_vars, drift, noise, sc_cfg):
        self.state_vars = list(state_vars)
        self.allowed = set(sc_cfg["allowed"])
        self.stage = sc_cfg.get("stage")           # None => full mode
        self.drift = list(drift)
        self.noise = dict(noise)
        theta = sym.Float(sc_cfg.get("theta", 0.1))
        A = sym.Float(sc_cfg.get("A", 1.0))
        H = sc_cfg["H"]
        frozen = sc_cfg.get("frozen_slots", {})     # {order(int): sympy expr in patch-1 coords}
        # Fast stage-mode path: the searched slot enters L(ptilde) LINEARLY
        # through (G, dG/dx_i, d2G/dp_b^2), so per patch we keep the pieces
        # needed to assemble the residual numerically from precomputed fields:
        #   res = base  -  [ I_s*LG + G*L(I_s) + sum_b 2 kappa_b dG/dp_b dI_s/dp_b ]
        #   LG  = sum_i drift_i dG/dx_i + sum_b kappa_b d2G/dp_b^2
        self._fast = []                             # per patch dicts of sympy pieces

        G = sym.Function("G_slot")
        self._apps = []                             # [(applied G, subs map or None)]
        self._res_tpls = []                         # per patch: (L ptilde template, rho)
        self.rho_fns = []
        V = 1 + sym.exp(theta * H)
        for patch in sc_cfg["patches"]:
            pf = patch["fast"]
            S = sum(pn ** 4 for pn in patch["neighbors_p"]) + 4
            rho = _chi(pf ** 2 / S ** 2)
            app = G(*patch["slot_args"])
            submap = patch.get("subs")
            self._apps.append((app, submap))

            # Frozen orders stay separate from the searched slot. A frozen
            # partial sum may sit at the SAME order as the searched slot
            # (t_k = partial + G): additive within-order curriculum (C23 wall).
            orders = {int(j): (e.xreplace(submap) if submap else e)
                      for j, e in frozen.items()}
            k = self.stage if self.stage else max(orders, default=0) + 1
            I = lambda j: sym.Piecewise((rho / pf ** j, pf ** 2 > 1), (0, True))
            pt_frozen = pf - sum(tj * I(j) for j, tj in orders.items())
            pt = pt_frozen - app * I(k)
            V = V + A * rho * pt ** 2 * sym.exp(theta / 2 * pt ** 2)

            if self.stage:
                Lpt = generator_expr(pt, self.state_vars, drift, noise)
                # Stage-order weighting (C10): at stage j the residual after a
                # CORRECT slot is O(p_f^-j); scoring (p_f^{j-1} * L ptilde)^2
                # keeps the correct/wrong gap at O(1) scale at every stage
                # (unweighted, stage>=2 rewards all saturate near 1 and the
                # FKL/risk-seeking signal collapses).
                if self.stage > 1:
                    Lpt = pf ** (self.stage - 1) * Lpt
                self._res_tpls.append(Lpt)
                self.rho_fns.append(sym.lambdify(self.state_vars, rho, "numpy"))

                # fast-path pieces: base residual (frozen orders only, concrete)
                # plus the coefficient fields of the searched slot's contribution
                I_s = I(k)
                base = generator_expr(pt_frozen, self.state_vars, drift, noise)
                self._fast.append({
                    "base": base, "I_s": I_s,
                    "L_I_s": generator_expr(I_s, self.state_vars, drift, noise),
                    "dI_dpb": {b: sym.diff(I_s, self.state_vars[b])
                               for b in noise},
                    "submap": submap, "pf": pf,
                })

        self.V_tpl = V
        self.LV_tpl = None if self.stage else generator_expr(V, self.state_vars, drift, noise)

    def check_slot(self, expr):
        used = expr.free_symbols
        return bool(used) and used <= self.allowed

    def _mapping(self, expr):
        # xreplace = simultaneous symbol replacement (plain .subs applies dict
        # entries sequentially: r1->r2 then r2->r3 would corrupt shifted slots)
        return {app: (expr.xreplace(submap) if submap else expr)
                for app, submap in self._apps}

    def assemble(self, expr):
        """Full mode: (V, LV) sympy expressions."""
        m = self._mapping(expr)
        return self.V_tpl.subs(m), self.LV_tpl.subs(m).doit()

    def residual_fns(self, expr):
        """Stage mode: list of lambdified L-ptilde residuals (one per patch)."""
        m = self._mapping(expr)
        return [sym.lambdify(self.state_vars, tpl.subs(m).doit(), "numpy")
                for tpl in self._res_tpls]

    # ---- fast stage path: numeric fields + per-candidate small derivatives ----

    def _fields(self, X):
        """Per-patch numeric fields on the deep-gate subset of X (cached)."""
        import numpy as np
        key = (X.shape, X[0].tobytes(), X[-1].tobytes())
        cache = getattr(self, "_field_cache", None)
        if cache is None:
            cache = self._field_cache = {}
        if key in cache:
            return cache[key]
        out = []
        args_full = [X[:, i] for i in range(X.shape[1])]
        with np.errstate(all="ignore"):
            for fp, rho_fn in zip(self._fast, self.rho_fns):
                rho = np.broadcast_to(np.asarray(rho_fn(*args_full), dtype=np.float64),
                                      (X.shape[0],))
                deep = rho > 1.0 - 1e-9
                Xd = X[deep]
                if Xd.shape[0] == 0:
                    out.append(None)
                    continue
                a = [Xd[:, i] for i in range(Xd.shape[1])]
                lam = lambda e: np.broadcast_to(
                    np.asarray(sym.lambdify(self.state_vars, e, "numpy")(*a),
                               dtype=np.float64), (Xd.shape[0],))
                f = {
                    "a": a, "n": Xd.shape[0],
                    "base": lam(fp["base"]), "I_s": lam(fp["I_s"]),
                    "L_I_s": lam(fp["L_I_s"]),
                    "dI_dpb": {b: lam(d) for b, d in fp["dI_dpb"].items()},
                    "drift": {i: lam(self.drift[i]) for i in range(len(self.state_vars))},
                    "w": lam(fp["pf"] ** (self.stage - 1)) if self.stage > 1 else 1.0,
                    "submap": fp["submap"],
                }
                out.append(f)
        cache[key] = out
        return out

    def fast_penalty(self, expr, X):
        """Numerically equivalent to the residual_fns route, ~10-40x faster:
        only the candidate's own small derivatives are lambdified."""
        import numpy as np
        pen = 0.0
        with np.errstate(all="ignore"):
            for f in self._fields(X):
                if f is None:
                    return None, False
                g = expr.xreplace(f["submap"]) if f["submap"] else expr
                a, n = f["a"], f["n"]
                ev = lambda e: np.broadcast_to(
                    np.asarray(sym.lambdify(self.state_vars, e, "numpy")(*a),
                               dtype=np.float64), (n,))
                gv = ev(g)
                used = {i for i, s in enumerate(self.state_vars)
                        if s in g.free_symbols}
                dg = {i: ev(sym.diff(g, self.state_vars[i])) for i in used}
                LG = np.zeros(n)
                for i in used:
                    LG = LG + f["drift"][i] * dg[i]
                cross = np.zeros(n)
                for b, kappa in self.noise.items():
                    if b in used:
                        LG = LG + kappa * ev(sym.diff(g, self.state_vars[b], 2))
                        cross = cross + 2.0 * kappa * f["dI_dpb"][b] * dg[b]
                contrib = -(f["I_s"] * LG + f["L_I_s"] * gv + cross)
                res = (f["base"] + contrib) * f["w"]
                if not np.all(np.isfinite(res)):
                    return None, False
                pen += float(np.square(np.clip(res, -1e150, 1e150)).mean())
        return pen, True


def _build_cfg(spec, state_vars):
    dim = len(state_vars)
    N = (dim + 1) // 2
    r = state_vars[: N - 1]
    p = state_vars[N - 1:]
    sc = dict(spec["scaffold"])
    sc["H"] = spec["H"]
    sc["stage"] = spec.get("stage")
    sc["frozen_slots"] = spec.get("frozen_slots", {})
    if N == 2:
        patches = [{"fast": p[1], "neighbors_p": [p[0]],
                    "slot_args": [r[0], p[0]], "subs": None}]
        allowed = [r[0], p[0]]
    elif N == 3:
        patches = [{"fast": p[1], "neighbors_p": [p[0], p[2]],
                    "slot_args": [r[0], r[1], p[0], p[2]], "subs": None}]
        allowed = [r[0], r[1], p[0], p[2]]
    elif N == 4:
        pidx = spec.get("patch_index")
        if pidx is None:
            # tied mode (valid through stage 2, C22): two single-fast patches
            # sharing one bond-local slot via index shift; resonance piece Vc
            # of doc eq.(26) deliberately absent in v1.
            slot_args = [r[0], r[1], p[0], p[2]]
            shift = {r[0]: r[1], r[1]: r[2], p[0]: p[1], p[2]: p[3]}
            patches = [
                {"fast": p[1], "neighbors_p": [p[0], p[2]], "slot_args": slot_args,
                 "subs": None},
                {"fast": p[2], "neighbors_p": [p[1], p[3]],
                 "slot_args": [shift[s] for s in slot_args], "subs": shift},
            ]
            allowed = slot_args
        else:
            # per-patch mode (stage >= 3, C22): friction asymmetry + light-cone
            # spread make the patches' answers differ; search ONE patch at a
            # time in ACTUAL state coordinates (no shift map), with the allowed
            # set widened by one bond. frozen_slots are per-patch, actual coords.
            defs = [
                {"fast": p[1], "neighbors_p": [p[0], p[2]],
                 "allowed": [r[0], r[1], r[2], p[0], p[2]]},
                {"fast": p[2], "neighbors_p": [p[1], p[3]],
                 "allowed": [r[0], r[1], r[2], p[1], p[3]]},
            ]
            d = defs[int(pidx)]
            patches = [{"fast": d["fast"], "neighbors_p": d["neighbors_p"],
                        "slot_args": d["allowed"], "subs": None}]
            allowed = d["allowed"]
    else:
        raise NotImplementedError(f"scaffold not defined for N={N} (v2: regime structure search)")
    sc["patches"] = patches
    sc["allowed"] = allowed
    return sc


def get_scaffold(spec, state_vars, drift):
    key = ("scaffold", len(state_vars), spec.get("stage"), spec.get("patch_index"),
           tuple(sorted((int(j), sym.srepr(e))
                        for j, e in spec.get("frozen_slots", {}).items())))
    if key not in _BUILT:
        cfg = _build_cfg(spec, state_vars)
        _BUILT[key] = Scaffold(state_vars, drift, spec["noise"], cfg)
    return _BUILT[key]


def fl_stage_penalty(p_sympy_expr, spec, state_vars, drift, X):
    """Stage-mode penalty on sample matrix X: gate-weighted mean squared
    averaging residual, summed over patches. Returns (penalty, ok)."""
    import os

    import numpy as np
    scaf = get_scaffold(spec, state_vars, drift)
    if not scaf.check_slot(p_sympy_expr):
        return None, False
    if os.environ.get("LYA_FL_FAST", "1") != "0":
        return scaf.fast_penalty(p_sympy_expr, X)
    args = [X[:, i] for i in range(X.shape[1])]
    n = X.shape[0]
    pen = 0.0
    with np.errstate(all="ignore"):
        res_fns = scaf.residual_fns(p_sympy_expr)
        for res_fn, rho_fn in zip(res_fns, scaf.rho_fns):
            res = np.broadcast_to(np.asarray(res_fn(*args), dtype=np.float64), (n,))
            rho = np.broadcast_to(np.asarray(rho_fn(*args), dtype=np.float64), (n,))
            # Score only where the gate is FULLY open (chi==1 exactly, all gate
            # derivatives identically zero). In the transition band chi'/chi''
            # are O(1-10) and add a large candidate-independent noise floor that
            # drowns the averaging signal.
            deep = rho > 1.0 - 1e-9
            if not np.all(np.isfinite(res[deep])):
                return None, False
            w = float(deep.sum())
            if w <= 0:
                return None, False
            pen += float(np.square(np.clip(res[deep], -1e150, 1e150)).sum() / w)
    return pen, True


def fl_V_and_LV(p_sympy_expr, spec, state_vars, drift, lv_expr=None):
    """Full-mode entry: (V, LV, ok). Template runs assemble; plain mode passes
    the candidate through (LV = lv_expr or freshly differentiated)."""
    if spec.get("scaffold") is not None:
        scaf = get_scaffold(spec, state_vars, drift)
        if not scaf.check_slot(p_sympy_expr):
            return None, None, False
        V, LV = scaf.assemble(p_sympy_expr)
        return V, LV, True
    V = p_sympy_expr
    LV = lv_expr if lv_expr is not None else generator_expr(V, state_vars, drift, spec["noise"])
    return V, LV, True
