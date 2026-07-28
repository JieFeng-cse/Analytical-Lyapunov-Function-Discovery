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
    v = np.clip(np.asarray(v_vals, dtype=np.float64), -1e30, 1e30)
    lv = np.clip(np.asarray(lv_vals, dtype=np.float64), -1e30, 1e30)
    h = np.asarray(h_vals, dtype=np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(lv))):
        return None
    mask = (h > spec["R"]).astype(np.float64)
    drift_viol = np.mean(np.maximum(lv + phi_vals(v, spec["alpha"], spec["c"]), 0.0) * mask)
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
