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

from utils.fl_tmpl.macros import cited_Hc_CE4, computed_slow_momentum, main_energy
from utils.fl_tmpl.system import RotorSystem
from utils.fl_tmpl.template import PartSpec, Template
from utils.fl_tmpl.tropical import Region, region_full, region_single_fast

_MACRO_CACHE = {}


def macros_for(N):
    if N in _MACRO_CACHE:
        return _MACRO_CACHE[N]
    S = RotorSystem(N)
    M = {"H": main_energy(S)}
    for j in S.interior:
        if N <= 4:
            M[f"pt{j}"] = computed_slow_momentum(S, j, target=3)
    if N == 4:
        M["Hc"] = cited_Hc_CE4(S)
    _MACRO_CACHE[N] = (S, M)
    return S, M


def region_both_fast_CE4(l, m):
    """Omega_c(l, m) of CE eq. (1.10) in direction space (0-based indices 1, 2 are
    p2, p3): top p2: s3 >= s2/l, s1, s4 <= 1/m; top p3 symmetric."""
    out = []
    for top, other in ((1, 2), (2, 1)):
        ineqs = [({other: -1.0}, -1.0 / l), ({0: 1.0}, 1.0 / m), ({3: 1.0}, 1.0 / m)]
        out.append(Region(4, top, ineqs, f"Omega_c(top=p{top + 1},l={l},m={m})"))
    return out


def build(N, slots, theta=Fraction(1, 4), noise_avg=True):
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


PAPER_SLOTS = {
    3: {"k": 6, "gap2": None, "a2": None, "deg2": 2},
    4: {"k": 6, "gap2": "pt2", "a2": Fraction(1, 4), "deg2": 0,
        "gap3": "pt3", "a3": Fraction(1, 4), "deg3": 0, "l": 16, "m": 128, "degc": 1},
}
