import numpy as np
from dso.functions import create_tokens
from dso.library import Library
from dso.prior import make_prior


def _lib():
    return Library(create_tokens(3, ["add", "mul", "sin", "step", "pair", "1"], protected=False))


def test_root_prior_masks():
    lib = _lib()
    prior = make_prior(lib, {"root": {"token": "pair", "on": True}})
    pair = lib.names.index("pair")
    init = prior.initial_prior()
    assert init[pair] == 0 and np.all(np.isneginf(np.delete(init, pair)))
    actions = np.array([[pair, 0]])
    later = prior(actions, np.zeros((1,)), np.zeros((1,)), np.ones((1,)))
    assert np.isneginf(later[0, pair]) and np.isfinite(later[0]).sum() == lib.L - 1


def test_root_prior_is_violated():
    lib = _lib()
    prior = make_prior(lib, {"root": {"token": "pair", "on": True}})
    pair, add, x1 = lib.names.index("pair"), lib.names.index("add"), lib.names.index("x1")
    assert not prior.is_violated(np.array([pair, x1, x1]), None, None)
    assert prior.is_violated(np.array([add, x1, x1]), None, None)
    assert prior.is_violated(np.array([pair, pair, x1, x1, x1]), None, None)


def test_diffusion_grammar_root_rule():
    from models.diffusion.grammar import Grammar
    lib = _lib()
    pair, add, x1 = lib.names.index("pair"), lib.names.index("add"), lib.names.index("x1")
    g = Grammar(lib, root_token=pair)
    m0 = g.valid_token_mask([])
    assert m0[pair] and not m0[add]
    m1 = g.valid_token_mask([pair])
    assert m1[add] and not m1[pair]
    rep = g.repair([add, x1, x1])
    assert rep[0] == pair and pair not in rep[1:] and g.is_complete(rep)
    rep2 = g.repair([pair, add, pair, x1, x1, x1])
    assert rep2[0] == pair and pair not in rep2[1:] and g.is_complete(rep2)
    assert g.fallback()[0] == pair and g.is_complete(g.fallback())


def test_angle_prior_and_grammar_rule():
    from models.diffusion.grammar import Grammar
    lib = Library(create_tokens(3, ["add", "mul", "sin", "cos", "step", "pair", "1"], protected=False))
    n = lib.names
    x1, x2, add, sin, cos, pair, step = (n.index(t) for t in ["x1", "x2", "add", "sin", "cos", "pair", "step"])
    prior = make_prior(lib, {"root": {"token": "pair", "on": True},
                             "relational": {"targets": ["x1"], "effectors": ["add", "mul", "step", "pair"],
                                            "relationship": "child", "on": True}})
    # after [pair, add] the next token's parent is add -> x1 masked; after [pair, sin] -> x1 allowed
    import numpy as np
    par_add = lib.parent_adjust[add]
    par_sin = lib.parent_adjust[sin]
    pr = prior(np.array([[pair, add]]), np.array([par_add]), np.array([lib.L]), np.ones(1))
    assert np.isneginf(pr[0, x1]) and np.isfinite(pr[0, x2])
    pr = prior(np.array([[pair, sin]]), np.array([par_sin]), np.array([lib.L]), np.ones(1))
    assert np.isfinite(pr[0, x1])
    from dso.gp.utils import jit_parents_siblings_at_once

    def aps(seq):
        actions = np.array([seq], dtype=np.int32)
        parent, sibling = jit_parents_siblings_at_once(
            actions, arities=lib.arities, parent_adjust=lib.parent_adjust)
        return actions, parent, sibling
    assert prior.is_violated(*aps([pair, add, x1, x2, x2]))
    assert not prior.is_violated(*aps([pair, sin, x1, x2]))
    g = Grammar(lib, root_token=pair, angle_tokens=[x1], angle_parents=[sin, cos])
    assert not g.valid_token_mask([pair, add])[x1] and g.valid_token_mask([pair, add])[x2]
    assert g.valid_token_mask([pair, sin])[x1]
    rep = g.repair([pair, add, x1, x1, x1])
    assert x1 not in rep and g.is_complete(rep) and rep[0] == pair
    assert x1 not in g.fallback()
