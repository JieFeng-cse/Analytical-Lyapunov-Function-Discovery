"""Template builders for N=3/4 and the paper's V as slot settings (spec 2026-10-17 section 7).

Slots (N=3):  k, gap2, a2, deg2
Slots (N=4):  k, gap2, a2, deg2, gap3, a3, deg3, l, m, degc

Paper references:
  N=3: benchmark eq. (21) (CEP2015): V = 1 + e^{theta H} + A rho p~_2^2 e^{theta p~_2^2/2}
       -> deg2 = 2, no gap factor.
  N=4: CE arXiv:1504.04964 sec. 4: V = 1 + e^{theta H} + psi_2 V_2 + psi_3 V_3 + M psi_c V_c,
       V_j = e^{|p~_j|^a + theta p~_j^2/2}(1 + F_j/p_j^3), V_c = H~_c e^{theta H~_c}(1 + ...),
       2/l < a < 2/k, k < l < m  -> gap_j = p~_j, deg_j = 0, degc = 1.
"""
from fractions import Fraction

import os

from utils.fl_tmpl.macros import (AP, C7, cited_Hc_CE4, computed_slow_momentum, conjecture_bottleneck,
                                  conjecture_pair_energy, conjecture_receiver, explicit_pair_energy,
                                  load_macros_json, main_energy)
from utils.fl_tmpl.system import RotorSystem
from utils.fl_tmpl.template import PartSpec, Template
from utils.fl_tmpl.tropical import Region, region_full, region_single_fast

_MACRO_CACHE = {}


def macros_for(N, danus=None):
    """Macros per system.  N=5: computed p~2, p~4, E24; conjecture p~3 (c7), pair
    energies H23/H34 (ap), receiver H234 -- replaced by Danus parts when a JSON file
    is given (argument or env LYA_TMPL_DANUS)."""
    danus = danus if danus is not None else os.environ.get("LYA_TMPL_DANUS", "")
    key = (N, danus)
    if key in _MACRO_CACHE:
        return _MACRO_CACHE[key]
    S = RotorSystem(N, params=(C7, AP) if N == 5 else ())
    M = {"H": main_energy(S)}
    if N <= 4:
        for j in S.interior:
            M[f"pt{j}"] = computed_slow_momentum(S, j, target=3)
    if N == 4:
        M["Hc"] = cited_Hc_CE4(S)
    if N == 5:
        for j in (2, 4):
            M[f"pt{j}"] = computed_slow_momentum(S, j, target=3)
        base3 = computed_slow_momentum(S, 3, target=3)
        M["pt3"] = conjecture_bottleneck(S, base3)
        M["E24"] = explicit_pair_energy(S, M["pt2"], M["pt4"], "A24")
        M["H23"] = conjecture_pair_energy(S, (2, 3))
        M["H34"] = conjecture_pair_energy(S, (3, 4))
        M["H234"] = conjecture_receiver(S)
        if danus:
            M.update(load_macros_json(S, danus))
    _MACRO_CACHE[key] = (S, M)
    return S, M


def region_group(nP, fast, l, m, name):
    """Several fast momenta (0-based indices `fast`): for each choice of top in the
    group, the other group members satisfy s >= 1/l and every other momentum s <= 1/m."""
    out = []
    for top in fast:
        ineqs = [({o: -1.0}, -1.0 / l) for o in fast if o != top]
        ineqs += [({i: 1.0}, 1.0 / m) for i in range(nP) if i not in fast]
        out.append(Region(nP, top, ineqs, f"{name}(top=p{top + 1},l={l},m={m})"))
    return out


def region_both_fast_CE4(l, m):
    """Omega_c(l, m) of CE eq. (1.10) in direction space (0-based indices 1, 2 are
    p2, p3): top p2: s3 >= s2/l, s1, s4 <= 1/m; top p3 symmetric."""
    out = []
    for top, other in ((1, 2), (2, 1)):
        ineqs = [({other: -1.0}, -1.0 / l), ({0: 1.0}, 1.0 / m), ({3: 1.0}, 1.0 / m)]
        out.append(Region(4, top, ineqs, f"Omega_c(top=p{top + 1},l={l},m={m})"))
    return out


def build(N, slots, theta=Fraction(1, 4), noise_avg=True):
    if N == 5:
        return build5(slots, theta=theta, noise_avg=noise_avg)
    S, M = macros_for(N)
    nP = N
    parts = [PartSpec("main", (), is_main=True, regions=[region_full(nP, t) for t in range(nP)])]
    k = slots["k"]
    for j in S.interior:
        gap = slots.get(f"gap{j}")
        a = slots.get(f"a{j}")
        parts.append(PartSpec(f"V{j}", (j,), energy=f"pt{j}", energy_kind="sq",
                              deg=slots.get(f"deg{j}", 0), gap=gap, a=Fraction(a) if (gap and a) else None,
                              regions=[region_single_fast(nP, j - 1, k)], noise_avg=noise_avg))
    conditions = []
    if N == 4 and slots.get("l"):
        parts.append(PartSpec("Vc", (2, 3), energy="Hc", energy_kind="lin", deg=slots.get("degc", 1),
                              regions=region_both_fast_CE4(slots["l"], slots["m"]), noise_avg=noise_avg))
        conditions.append(M["Hc"].ref)
    alpha = 3 if N == 5 else 1
    return Template(S, alpha, Fraction(theta), M, parts, dict(slots), conditions)


def build5(slots, theta=Fraction(1, 4), noise_avg=True):
    """N=5 template (spec 2.5).  Slots: k, l, m, lr; gapj/aj/degj (j = 2, 3, 4);
    deg24; degp (pair parts 23 and 34; 'none' omits them); recv ('none' | 'H234'), degr."""
    S, M = macros_for(5)
    nP = 5
    k, l, m = slots["k"], slots["l"], slots["m"]
    parts = [PartSpec("main", (), is_main=True, regions=[region_full(nP, t) for t in range(nP)])]
    for j in (2, 3, 4):
        gap = slots.get(f"gap{j}")
        a = slots.get(f"a{j}")
        parts.append(PartSpec(f"V{j}", (j,), energy=f"pt{j}", energy_kind="sq",
                              deg=slots.get(f"deg{j}", 0), gap=gap, a=Fraction(a) if (gap and a) else None,
                              regions=[region_single_fast(nP, j - 1, k)], noise_avg=noise_avg))
    if slots.get("deg24") is not None:
        parts.append(PartSpec("V24", (2, 4), energy="E24", energy_kind="lin", deg=slots["deg24"],
                              regions=region_group(nP, [1, 3], l, m, "Omega_24"), noise_avg=noise_avg))
    if slots.get("degp") is not None:
        for pair in ((2, 3), (3, 4)):
            nm = f"H{pair[0]}{pair[1]}"
            parts.append(PartSpec(f"V{pair[0]}{pair[1]}", pair, energy=nm, energy_kind="lin",
                                  deg=slots["degp"], regions=region_group(nP, [pair[0] - 1, pair[1] - 1], l, m,
                                                                          f"Omega_{pair[0]}{pair[1]}"),
                                  noise_avg=noise_avg))
    if slots.get("recv") == "H234":
        parts.append(PartSpec("Vr", (2, 3, 4), energy="H234", energy_kind="lin", deg=slots.get("degr", 0),
                              regions=region_group(nP, [1, 2, 3], slots.get("lr", l), m, "Omega_234"),
                              noise_avg=noise_avg))
    used = {ps.energy for ps in parts if ps.energy} | {ps.gap for ps in parts if ps.gap}
    conditions = sorted({c for nm in used for c in getattr(M[nm], "conditions", [])
                         if M[nm].provenance in ("conjecture", "danus")})
    T = Template(S, 3, Fraction(theta), M, parts, dict(slots), conditions)
    T.param_ranges = {C7: (1e-3, 1e2), AP: (1e-3, 1e2)}
    return T


PAPER_SLOTS = {
    3: {"k": 6, "gap2": None, "a2": None, "deg2": 2},
    4: {"k": 6, "gap2": "pt2", "a2": Fraction(1, 4), "deg2": 0,
        "gap3": "pt3", "a3": Fraction(1, 4), "deg3": 0, "l": 16, "m": 128, "degc": 1},
}

# N=5: the benchmark's heuristic starting point (eq. 29): psi_3 p~3^6 e^{theta p~3^2/2}
# plus "corrections for the other fast and resonant regimes" -- here the single-fast
# parts 2/4 with CE gaps, the separated pair 24, the adjacent pairs and the receiver.
ANSATZ5_SLOTS = {"k": 6, "l": 16, "m": 128, "lr": 16,
                 "gap2": "pt2", "a2": Fraction(1, 4), "deg2": 0,
                 "gap3": None, "a3": None, "deg3": 6,
                 "gap4": "pt4", "a4": Fraction(1, 4), "deg4": 0,
                 "deg24": 1, "degp": 1, "recv": "H234", "degr": 1}
