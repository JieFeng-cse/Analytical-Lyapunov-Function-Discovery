"""Reward glue for the RL slot search (spec 2026-10-17 sections 4, 6).

tmpl_penalty(token_names, spec) -> (pen, report) with a per-process cache keyed by
the slot tuple (the slot space is small, every arm samples the same tuples many
times).  reward = 1/(1+pen); success iff pen == 0.
"""
from fractions import Fraction
import warnings

from utils.fl_tmpl.slots import THETA_DEFAULT, decode

_CACHE = {}


def tmpl_penalty(names, spec):
    N = int(spec["N"])
    slots = decode(N, list(names))
    if slots is None:
        return None, None
    theta = Fraction(spec.get("theta", THETA_DEFAULT[N]))
    key = (N, str(theta), tuple(sorted((k, str(v)) for k, v in slots.items())))
    if key in _CACHE:
        return _CACHE[key]
    from utils.fl_tmpl.reference import build
    from utils.fl_tmpl.template import evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rep = evaluate(build(N, slots, theta=theta), theta_windows=False,
                           n_samples=int(spec.get("n_samples", 32)))
            out = (float(rep.penalty), rep)
        except Exception as exc:  # pylint: disable=broad-except
            out = (None, f"error: {exc!r}")
    _CACHE[key] = out
    return out


def program_names(p):
    return [t.name for t in p.traversal]


def _light(args):
    names, spec = args
    pen, rep = tmpl_penalty(names, spec)
    summ = rep.summary() if hasattr(rep, "summary") else rep
    return tuple(names), pen, summ


def prefill(programs, spec, n_proc=16):
    """Evaluate the distinct, not-yet-cached slot tuples of a batch in a process
    pool and store (pen, summary string) in the cache, so the serial reward loop
    only hits the cache.  Returns the number of new evaluations."""
    import multiprocessing as mp
    N = int(spec["N"])
    theta = Fraction(spec.get("theta", THETA_DEFAULT[N]))
    todo, seen = [], set()
    for p in programs:
        names = tuple(program_names(p))
        slots = decode(N, list(names))
        if slots is None:
            continue
        key = (N, str(theta), tuple(sorted((k, str(v)) for k, v in slots.items())))
        if key in _CACHE or key in seen:
            continue
        seen.add(key)
        todo.append((key, names))
    if not todo:
        return 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(n_proc, len(todo))) as pool:
        res = pool.map(_light, [(list(n), spec) for _, n in todo], chunksize=1)
    for (key, _), (_, pen, summ) in zip(todo, res):
        _CACHE[key] = (pen, summ)
    return len(todo)
