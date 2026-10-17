"""Calibration of the template evaluator on N=3/N=4 (source "用 N=3、4 校准评估器",
spec 2026-10-17 section 7):

1. the paper's V passes (all blocks good or absorbed, LP feasible, no positive cycle);
2. without the gap factor the report names the junction that misses a > 0 (缺间隙);
3. without noise averaging the theta upper bound shrinks.
"""
from fractions import Fraction
import warnings

import pytest

from utils.fl_tmpl.reference import PAPER_SLOTS, build
from utils.fl_tmpl.template import evaluate

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def n4_report():
    return evaluate(build(4, PAPER_SLOTS[4]), theta_windows=True)


def test_n3_paper_V_passes_with_constant_A():
    rep = evaluate(build(3, PAPER_SLOTS[3]), theta_windows=True)
    assert rep.penalty == 0 and rep.lp_feasible and rep.bf_feasible
    # the cold-bath ray of e^{theta H} is absorbed by V2 through a constant (A large)
    assert any(e[0] == "main" and e[1] == "V2" for e in rep.edges)
    # theta window of V2: -theta + theta^2 (k1<W_L^2> + k3<W_R^2>) < 0 -> 2/3; pointwise -> 1/3
    v2 = rep.parts["V2"]
    assert abs(v2.theta_max - 2 / 3) < 1e-3 and abs(v2.theta_max_without_noise - 1 / 3) < 1e-3
    assert abs(rep.parts["main"].theta_max - 0.5) < 1e-3          # Ito bound theta < 1/T_3


def test_n4_paper_V_passes(n4_report):
    rep = n4_report
    assert rep.penalty == 0 and rep.lp_feasible and rep.bf_feasible
    assert all(b.status == "good" for n in ("V2", "V3", "Vc") for b in rep.parts[n].blocks)
    # the only constant junction is e^{theta H} -> M V_c (CE section 4, "M large enough")
    assert {(e[0], e[1]) for e in rep.edges} == {("main", "Vc")}
    # LP hint sits inside the gap window 2/l < a < 2/k  (l = 16, k = 6)
    for v in rep.lp_hint.values():
        assert 2 / 16 < v < 2 / 6


def test_n4_theta_windows(n4_report):
    p = n4_report.parts
    assert abs(p["V2"].theta_max - 1.0) < 1e-3 and abs(p["V2"].theta_max_without_noise - 0.5) < 1e-3
    assert abs(p["V3"].theta_max - 0.5) < 1e-3 and abs(p["V3"].theta_max_without_noise - 0.25) < 1e-3
    assert abs(p["Vc"].theta_max - 0.5) < 1e-3 and abs(p["Vc"].theta_max_without_noise - 0.25) < 1e-3


def test_n4_without_gap_reports_missing_gap():
    slots = dict(PAPER_SLOTS[4], gap2=None, gap3=None)
    rep = evaluate(build(4, slots), theta_windows=False)
    assert rep.penalty > 0 and not rep.lp_feasible
    labels = [lab for lab, _ in rep.unabsorbed]
    # the V_c switch layer p3^2 ~ |p2|^{2/l}: theta p3^2/2 (order 2/l) has no absorber
    assert any("Vc layer" in lab and "-0.0625" in lab for lab in labels)
    # its LP slack is the missing exponent 2/l (+ the LP margin)
    sl = [s for lab, s in rep.lp_slacks if lab.startswith("Vc layer")]
    assert sl and abs(max(sl) - (2 / 16 + 1e-3)) < 1e-6


def test_n4_gap_outside_window_is_flagged():
    # a = 1/2 > 2/k = 1/3 violates the upper end of the window (CE (4.9))
    slots = dict(PAPER_SLOTS[4], a2=Fraction(1, 2), a3=Fraction(1, 2))
    rep = evaluate(build(4, slots), theta_windows=False)
    assert rep.penalty > 0
    assert any("V2 layer" in lab or "V3 layer" in lab for lab, _ in rep.unabsorbed)
