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
