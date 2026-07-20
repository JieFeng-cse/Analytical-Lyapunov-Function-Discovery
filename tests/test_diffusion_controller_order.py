"""Controller-level routing test: the `sampler_order` flag selects the Phase-3
random-order sampler vs. the pre-order baseline, and both yield valid complete
expressions end-to-end through a (random-init) denoiser (design doc §8)."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.controller import DiffusionController


class FakeLib:
    L = 8
    arities = np.array([2, 2, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    terminal_tokens = np.array([4, 5, 6, 7], dtype=np.int64)
    trig_tokens = np.array([2, 3], dtype=np.int64)
    const_token = 7
    EMPTY_PARENT = 0
    EMPTY_SIBLING = 0


class FakePrior:
    priors = []


def _make(order):
    # tiny denoiser: this test checks routing + output validity, not denoiser quality
    return DiffusionController(FakePrior(), FakeLib(), task=None, vocab_size=10,
                              max_length=8, sampler_order=order,
                              d_model=16, nlayers=1, nhead=2)


def test_default_is_preorder():
    assert _make("preorder").sampler_order == "preorder"


def test_random_order_routes_and_yields_valid():
    ctrl = _make("random")
    assert ctrl.sampler_order == "random"
    torch.manual_seed(0)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    actions, obs, priors = ctrl.sample(4, dyn)
    assert actions.shape == (4, 8)
    for row in actions:
        assert ctrl.grammar.is_complete(list(row))      # valid complete tree (short-circuits at close)


def test_blocksemiar_order_routes_and_yields_valid():
    ctrl = DiffusionController(FakePrior(), FakeLib(), task=None, vocab_size=10,
                               max_length=8, sampler_order="blocksemiar", block_size=4,
                               d_model=16, nlayers=1, nhead=2)
    assert ctrl.sampler_order == "blocksemiar"
    torch.manual_seed(0)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    actions, obs, priors = ctrl.sample(4, dyn)
    assert actions.shape == (4, 8)
    for row in actions:
        assert ctrl.grammar.is_complete(list(row))


def test_opblock_is_the_default():
    ctrl = DiffusionController(FakePrior(), FakeLib(), task=None, vocab_size=10,
                               max_length=8, d_model=16, nlayers=1, nhead=2)
    assert ctrl.sampler_order == "opblock"
    assert ctrl.block_size == 2


def test_opblock_order_routes_and_yields_valid():
    ctrl = DiffusionController(FakePrior(), FakeLib(), task=None, vocab_size=10,
                               max_length=8, sampler_order="opblock", block_size=4,
                               d_model=16, nlayers=1, nhead=2)
    assert ctrl.sampler_order == "opblock"
    torch.manual_seed(0)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    actions, obs, priors = ctrl.sample(4, dyn)
    assert actions.shape == (4, 8)
    for row in actions:
        assert ctrl.grammar.is_complete(list(row))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} controller-order tests passed.")
