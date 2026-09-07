import numpy as np
import sympy as sym
from dso.functions import function_map, create_tokens
from dso.library import Library


def test_step_numpy_values():
    step = function_map["step"]
    z = np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    out = step(z)
    assert out[0] == 0 and out[1] == 0 and out[5] == 1 and out[6] == 1
    assert abs(out[3] - 0.5) < 1e-12
    assert np.all(np.diff(out) >= 0)
    assert step.arity == 1 and function_map["pair"].arity == 2


def test_tokens_parse_to_sympy():
    from dso.program import build_tree, convert_to_sympy
    from sympy.parsing.sympy_parser import parse_expr
    lib = Library(create_tokens(3, ["add", "step", "pair", "1", "2"], protected=False))
    names = [t.name for t in lib.tokens]
    trav = [lib.tokens[names.index(n)] for n in ["pair", "step", "x3", "add", "x3", "1.0"]]
    tree = convert_to_sympy(build_tree(list(trav)))
    expr = parse_expr(repr(tree))
    assert expr.func.__name__ == "pair"
    assert expr.args[0].func.__name__ == "step"
    assert sym.simplify(expr.args[1] - (sym.Symbol("x3") + 1)) == 0
