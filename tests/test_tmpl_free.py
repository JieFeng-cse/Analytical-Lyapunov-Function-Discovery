"""Free-grammar local shapes (spec 2026-10-17 section 10)."""
import sympy as sp

from utils.fl_tmpl.freeform import InvalidShape, U, decode_free, expand_shape
from utils.fl_tmpl.rl import free_penalty


def test_decode_prefix_tree():
    names = ["free1", "mul", "u", "exp", "sqrt", "sqrt", "u"]
    assert decode_free(1, names) == [U * sp.exp(U ** sp.Rational(1, 4))]
    assert decode_free(1, names[:-1]) is None
    assert decode_free(1, ["free1", "u", "u"]) is None
    assert decode_free(2, ["free2", "n2", "u", "1"]) == [U**2, 1]


def test_expand_shape():
    d = expand_shape(U**2)
    assert d["log_coef"] == 2 and d["gap"] is None
    d = expand_shape(sp.exp(U ** sp.Rational(1, 4)))
    assert float(d["gap"]) == 0.25 and d["log_coef"] == 0
    for bad in (sp.log(U), sp.exp(-sp.sqrt(U)), sp.exp(U**2), U - 200):
        try:
            expand_shape(bad)
        except InvalidShape:
            continue
        raise AssertionError(f"{bad} should be rejected")


def test_free_penalty_n3():
    spec = {"N": 3, "theta": "1/4", "free": True}
    assert free_penalty(["free1", "n2", "u"], spec)[0] == 0          # paper: p~^2
    assert free_penalty(["free1", "exp", "sqrt", "sqrt", "u"], spec)[0] == 0   # CE-style gap
    assert free_penalty(["free1", "u"], spec)[0] > 0                 # one power short
    assert free_penalty(["free1", "1"], spec)[0] > 0
    assert free_penalty(["free1", "log", "u"], spec)[0] >= 1e3       # invalid shape
