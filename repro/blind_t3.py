"""Blind replication driver for N=3 t3 (protocol: docs/PROTOCOL-blind-t3.md).

Runs the search->freeze->snap loop using ONLY answer-free signals. The derived
target never enters any decision; it is loaded at the very end for scoring.

Usage:  python repro/blind_t3.py <rundir>
Each round: launch search job(s) -> parse best expr -> consensus/decay tests ->
freeze or snap -> next round. Designed to be resumable (state in state.json).
"""
import json, os, re, subprocess, sys, time

import numpy as np
import sympy as sym

RUNDIR = sys.argv[1] if len(sys.argv) > 1 else "."
STATE = os.path.join(RUNDIR, "state.json")
# decay_slope() imports the run-dir config, which only builds the template
# scaffold (spec["scaffold"]) when LYA_FL_TEMPLATE=1 -- the sbatch script sets
# it for the search jobs, the driver process must set it for itself.
os.environ.setdefault("LYA_FL_TEMPLATE", "1")
FROZEN0 = {"1": "cos(x1)+cos(x2)", "2": "x3*cos(x1)+x5*cos(x2)"}   # stages 1-2 (public)
SEEDS = [1, 2]
BUDGET = "1500000"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"round": 0, "frozen": dict(FROZEN0), "history": []}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)


def submit(frozen, seed, softloc, lenmax, jobname):
    env = os.environ.copy()
    env.update(LYA_FL_FROZEN=json.dumps(frozen), LYA_NSAMPLES=BUDGET,
               LYA_FL_SOFTLOC=str(softloc), LYA_FL_LENMAX=str(lenmax),
               LYA_SEED_START=str(seed))
    out = subprocess.run(
        f'sbatch --time=02:30:00 --job-name={jobname} '
        f'--export=ALL,SYS=rotor3,BM=fn_rotor_3,STAGE=3,ARM=transformer '
        f'--chdir={RUNDIR} /home/x-cchen47/ALFD-rotor/repro/run_one_rotor_slot.sbatch',
        shell=True, capture_output=True, text=True, env=env).stdout
    return re.search(r"(\d+)", out).group(1)


def wait(jobids, poll=120):
    while True:
        q = sh(f"squeue -h -j {','.join(jobids)} -o %i").strip()
        if not q:
            return
        time.sleep(poll)


def best_of(jobid):
    for f in os.listdir(RUNDIR):
        if jobid in f and f.endswith(".out"):
            txt = open(os.path.join(RUNDIR, f)).read()
            e = re.findall(r"'expression': '\[([^\]]*)\]'", txt)
            r = re.findall(r"current best R: ([\d.]+)", txt)
            if e:
                return e[-1], float(r[-1]) if r else float("nan")
    return None, float("nan")


def consensus_terms(exprs):
    """Additive terms appearing verbatim in >=2 of the given expressions."""
    sets = []
    for e in exprs:
        try:
            sets.append(set(sym.Add.make_args(sym.sympify(e))))
        except Exception:
            sets.append(set())
    out = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            out |= sets[i] & sets[j]
    return out


def decay_slope(frozen, extra=None):
    """Answer-free: log|residual| vs log p2 slope on a held-out ladder."""
    sys.path.insert(0, RUNDIR)
    from config import dynamics, generator_spec           # noqa: E402
    from utils.fl_scaffold import get_scaffold            # noqa: E402
    sv, dr = dynamics()
    spec = dict(generator_spec())
    fr = {int(k): sym.sympify(v) for k, v in frozen.items()}
    if extra is not None:
        fr[3] = fr.get(3, 0) + extra
    spec["frozen_slots"] = fr
    spec["stage"] = 4                                     # probe next order
    scaf = get_scaffold(spec, sv, dr)
    fn = scaf.residual_fns(sym.Integer(0))[0]
    rho_fn = scaf.rho_fns[0]
    rng = np.random.default_rng(0)
    xs, ys = [], []
    # p2 ladder must be deep inside the open gate: |p2| >= 2*S, S=(p1^2+p3^2)^2+20
    for P2 in (120., 200., 340., 560.):
        n = 4000
        X = np.zeros((n, 5))
        X[:, 0:2] = rng.uniform(-np.pi, np.pi, (n, 2))
        X[:, 2] = rng.normal(0, 1, n); X[:, 4] = rng.normal(0, 1.4, n)
        X[:, 3] = P2
        a = [X[:, i] for i in range(5)]
        with np.errstate(all="ignore"):
            v = np.asarray(fn(*a), dtype=np.float64)
            rho = np.broadcast_to(np.asarray(rho_fn(*a), dtype=np.float64), (n,))
        deep = np.isfinite(v) & (rho > 1 - 1e-9)      # gate fully open only
        if deep.sum() < 50:
            return float("nan")
        xs.append(np.log(P2)); ys.append(np.log(np.std(v[deep])))
    return float(np.polyfit(xs, ys, 1)[0])


def snap_candidates(expr):
    """Series-expand composites f(a*r+b*p) and return basis terms (answer-free)."""
    out = []
    for t in sym.Add.make_args(expr):
        for f in t.atoms(sym.sin, sym.cos):
            a = sym.expand(f.args[0])
            if len(a.free_symbols) >= 2:                   # mixed angle+momentum
                ser = sym.expand(sym.series(t, list(a.free_symbols)[0], 0, 3).removeO())
                out += [u for u in sym.Add.make_args(ser)]
    return out


def main():
    s = load_state()
    while s["round"] < 4:
        rnd = s["round"]
        frozen = s["frozen"]
        cur_len = sum(len(str(v)) for v in frozen.values())
        softloc = max(8, 22 - 4 * rnd)
        # Resume: reuse already-submitted jobs for this round (state["pending"]).
        jobs = s.get("pending") or [
            submit(frozen, sd, softloc, softloc + 10, f"blind_t3_r{rnd}s{sd}")
            for sd in SEEDS]
        s["pending"] = jobs
        save_state(s)
        print(f"[round {rnd}] jobs {jobs} frozen={frozen}", flush=True)
        wait(jobs)
        s["pending"] = None
        results = [best_of(j) for j in jobs]
        exprs = [e for e, _ in results if e]
        rs = [r for _, r in results]
        print(f"[round {rnd}] best r={rs} exprs={exprs}", flush=True)
        cons = consensus_terms(exprs)
        base_slope = decay_slope(frozen)
        acted = False
        if cons:
            cand = sym.Add(*cons)
            sl = decay_slope(frozen, extra=cand)
            print(f"[round {rnd}] consensus={cand} slope {base_slope:.2f}->{sl:.2f}", flush=True)
            if sl <= base_slope + 0.1:                     # order not worsened
                frozen["3"] = str(sym.sympify(frozen.get("3", "0")) + cand)
                acted = True
        if not acted and exprs:
            for cand in snap_candidates(sym.sympify(exprs[int(np.nanargmax(rs))])):
                sl = decay_slope(frozen, extra=cand)
                if sl < base_slope - 0.3:
                    frozen["3"] = str(sym.sympify(frozen.get("3", "0")) + cand)
                    print(f"[round {rnd}] snapped {cand} slope->{sl:.2f}", flush=True)
                    acted = True
                    break
        s["history"].append({"round": rnd, "r": rs, "exprs": exprs,
                             "frozen_after": dict(frozen), "slope": base_slope})
        s["round"] = rnd + 1
        save_state(s)
        if not acted:
            print(f"[round {rnd}] no action -> stop", flush=True)
            break
    print("FINAL frozen:", s["frozen"], flush=True)


if __name__ == "__main__":
    main()
