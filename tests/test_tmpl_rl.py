"""RL glue for the template slot search (spec 2026-10-17 section 6)."""
import numpy as np

from dso.functions import create_tokens
from dso.library import Library
from dso.prior import SlotConstraint
from models.diffusion.grammar import Grammar
from utils.fl_tmpl.reference import PAPER_SLOTS
from utils.fl_tmpl.slots import SLOT_MENUS, decode, encode, root_name, token_names


def _library(N):
    return Library(create_tokens(2 * N - 1, token_names(N), protected=False))


def test_tokens_and_decode_roundtrip():
    lib = _library(4)
    root = lib.names.index(root_name(4))
    assert lib.arities[root] == len(SLOT_MENUS[4])
    names = encode(4, PAPER_SLOTS[4])
    assert decode(4, names) == PAPER_SLOTS[4]
    assert decode(4, names[:-1]) is None
    assert decode(4, [names[0]] + names[2:] + names[1:2]) is None


def test_slot_prior_masks_positions():
    lib = _library(3)
    pr = SlotConstraint(lib, root_name(3))
    init = pr.initial_prior()
    assert np.isfinite(init).sum() == 1 and np.isfinite(init[pr.root])
    for pos in range(1, pr.n + 1):
        row = pr(np.zeros((2, pos), dtype=np.int32), None, None, None)[0]
        allowed = set(np.where(np.isfinite(row))[0].tolist())
        assert allowed == set(pr.slot_tokens[pos - 1].tolist())
    good = [pr.root] + [int(t[0]) for t in pr.slot_tokens]
    assert not pr.is_violated(good, None, None)
    bad = [pr.root] + [int(pr.slot_tokens[1][0])] + [int(t[0]) for t in pr.slot_tokens[1:]]
    assert pr.is_violated(bad, None, None)


def test_diffusion_grammar_slot_mask_and_repair():
    lib = _library(4)
    pr = SlotConstraint(lib, root_name(4))
    g = Grammar(lib, root_token=pr.root, slot_tokens=[list(s) for s in pr.slot_tokens], max_len=32)
    seq = [pr.root]
    rng = np.random.default_rng(0)
    while not g.is_complete(seq):
        m = g.valid_token_mask(seq)
        seq.append(int(rng.choice(np.where(m[:g.L])[0])))
    names = [lib.names[t] for t in seq]
    assert decode(4, names) is not None
    rep = g.repair([pr.root] + [0] * 3)
    assert decode(4, [lib.names[t] for t in rep]) is not None
