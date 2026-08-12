"""Foster-Lyapunov reward math (spec 2026-07-28). Pure sympy/numpy, no framework deps."""
import numpy as np
import sympy as sym


def generator_expr(V, state_vars, drift, noise):
    """Full stochastic generator LV = grad(V).f + sum_b kappa_b d2V/dp_b^2."""
    LV = sum(sym.diff(V, s) * f for s, f in zip(state_vars, drift))
    for idx, kappa in noise.items():
        LV = LV + kappa * sym.diff(V, state_vars[idx], 2)
    return LV


def phi_vals(v, alpha, c):
    v = np.maximum(np.asarray(v, dtype=np.float64), 1.0)
    return c * v / (alpha + 1.0 + np.log(v)) ** alpha


def _terms(v_vals, lv_vals, h_vals, spec):
    v_raw = np.asarray(v_vals, dtype=np.float64)
    lv_raw = np.asarray(lv_vals, dtype=np.float64)
    h = np.asarray(h_vals, dtype=np.float64)
    # Check finiteness on RAW inputs before clipping
    if not (np.all(np.isfinite(v_raw)) and np.all(np.isfinite(lv_raw))):
        return None
    v = np.clip(v_raw, -1e300, 1e300)
    lv = np.clip(lv_raw, -1e300, 1e300)
    mask = (h > spec["R"]).astype(np.float64)
    # Normalize over the points OUTSIDE K_R only. Dividing by the total count
    # would dilute the drift penalty as R grows (masked points contribute 0 to
    # the numerator but stay in the denominator), implicitly coupling R to the
    # effective penalty strength.
    n_out = max(float(mask.sum()), 1.0)
    drift_viol = float(np.sum(np.maximum(lv + phi_vals(v, spec["alpha"], spec["c"]), 0.0) * mask) / n_out)
    pos_viol = np.mean(np.maximum(1.0 - v, 0.0))
    return drift_viol, pos_viol


def fl_penalty(v_vals, lv_vals, h_vals, spec):
    t = _terms(v_vals, lv_vals, h_vals, spec)
    if t is None:
        return 1e6
    return float(t[0] + 2.5 * t[1])


def fl_violation(v_vals, lv_vals, h_vals, spec):
    t = _terms(v_vals, lv_vals, h_vals, spec)
    if t is None:
        return 1e6
    return float(t[0] + t[1])


def fl_objective_fns(V, state_vars, drift, spec, *, lv_expr=None):
    """Lambdify (V, LV, H). Pass ``lv_expr`` to reuse an already-computed
    generator expression (e.g. Program.sympy_expr_generator) and skip the
    symbolic differentiation."""
    LV = generator_expr(V, state_vars, drift, spec["noise"]) if lv_expr is None else lv_expr
    v_fn = sym.lambdify(state_vars, V, "numpy")
    lv_fn = sym.lambdify(state_vars, LV, "numpy")
    h_fn = sym.lambdify(state_vars, spec["H"], "numpy")
    return v_fn, lv_fn, h_fn


def _pointwise_violation(v_fn, lv_fn, h_fn, X, spec):
    args = [X[:, i] for i in range(X.shape[1])]
    # Check finiteness on RAW inputs before clipping (mirrors _terms), but
    # per-point: only the offending rows are marked maximally violating.
    v_raw = np.broadcast_to(np.asarray(v_fn(*args), dtype=np.float64), (X.shape[0],))
    lv_raw = np.broadcast_to(np.asarray(lv_fn(*args), dtype=np.float64), (X.shape[0],))
    bad = ~(np.isfinite(v_raw) & np.isfinite(lv_raw))
    v = np.clip(v_raw, -1e300, 1e300)
    lv = np.clip(lv_raw, -1e300, 1e300)
    h = np.broadcast_to(np.asarray(h_fn(*args), dtype=np.float64), (X.shape[0],))
    viol = np.maximum(lv + phi_vals(v, spec["alpha"], spec["c"]), 0.0) * (h > spec["R"])
    viol = viol + np.maximum(1.0 - v, 0.0)
    viol = np.where(np.isfinite(viol), viol, 1e6)
    viol = np.where(bad, 1e6, viol)
    return viol


def fl_random_falsify(V, state_vars, drift, spec, low, high,
                      n_rounds=10, n_pts=1_000_000, seed=0, lv_expr=None):
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec, lv_expr=lv_expr)
    rng = np.random.default_rng(seed)
    worst_pt, worst_val = None, 0.0
    for _ in range(n_rounds):
        X = rng.uniform(low, high, size=(n_pts, len(low)))
        viol = _pointwise_violation(v_fn, lv_fn, h_fn, X, spec)
        i = int(np.argmax(viol))
        if viol[i] > worst_val:
            worst_val, worst_pt = float(viol[i]), X[i]
    ok = worst_val <= 1e-9
    return ok, (None if ok else worst_pt)


def fl_shgo_falsify(V, state_vars, drift, spec, low, high, lv_expr=None):
    from scipy.optimize import shgo, Bounds
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec, lv_expr=lv_expr)

    def neg_obj(x):
        X = x.reshape(1, -1)
        args = [X[:, i] for i in range(X.shape[1])]
        v = float(np.clip(v_fn(*args), -1e300, 1e300))
        lv = float(np.clip(lv_fn(*args), -1e300, 1e300))
        h = float(h_fn(*args))
        drift_v = max(lv + float(phi_vals(np.array([v]), spec["alpha"], spec["c"])[0]), 0.0)
        drift_v -= 1e3 * max(spec["R"] - h, 0.0)          # penalty masking inside K_R
        val = max(drift_v, 0.0) + max(1.0 - v, 0.0)
        return -val if np.isfinite(val) else 0.0

    res = shgo(neg_obj, Bounds(low, high), n=512, iters=2,
               sampling_method="simplicial")
    return bool(-res.fun <= 1e-9)


def fl_harvest(V, state_vars, drift, spec, low, high, n_pts=100_000, k=50, seed=0,
               lv_expr=None):
    """Return the ``k`` worst random points for V. ``lv_expr`` reuses a cached
    generator expression instead of re-differentiating V."""
    v_fn, lv_fn, h_fn = fl_objective_fns(V, state_vars, drift, spec, lv_expr=lv_expr)
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n_pts, len(low)))
    viol = _pointwise_violation(v_fn, lv_fn, h_fn, X, spec)
    return X[np.argsort(viol)[-k:]]
