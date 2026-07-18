"""Unit tests for the block semi-AR sampler (BACKLOG #2 blocksemiar): pre-order across
blocks of `block_size` positions, confidence-ranked commit order within each block.

CPU-only; hand-built dummy denoisers supply fixed logits so reveal-order / block
boundaries / tight-completion / repair logic is exercised deterministically.
Token map (FakeLib): 0:'+'(2) 1:'*'(2) 2:'sin'(1) 3:'cos'(1) 4:'x1' 5:'x2' 6:'1' 7:'c'
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.grammar import Grammar
from models.diffusion.sampler import _reveal_counts, _blocks, diffusion_sample_blocksemiar


class FakeLib:
    L = 8
    arities = np.array([2, 2, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    terminal_tokens = np.array([4, 5, 6, 7], dtype=np.int64)
    trig_tokens = np.array([2, 3], dtype=np.int64)
    const_token = 7


def G(max_len=16):
    return Grammar(FakeLib(), max_len=max_len)


def _is_tight_complete(grammar, seq):
    return grammar.is_complete(seq) and (len(seq) == 1 or not grammar.is_complete(seq[:-1]))


def test_blocks_partitions_cover_max_len_in_order():
    blocks = _blocks(10, 4)
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


class TreeDenoiser:
    """Prefers '+' at position 0 and 'x1' everywhere else -> builds (x1 + x1) = [0,4,4]."""
    def __init__(self, n_choices):
        self.n_choices = n_choices
        self.mask_id = n_choices

    def __call__(self, grid, t, dyn):
        B, L = grid.shape
        logits = torch.zeros(B, L, self.n_choices)
        logits[:, 0, 0] = 20.0      # '+'
        logits[:, 1:, 4] = 20.0     # 'x1'
        return logits


def test_blocksemiar_returns_valid_complete():
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((5, 1), dtype=torch.long)
    seqs = diffusion_sample_blocksemiar(den, grammar, 5, 16, dyn, device="cpu",
                                        block_size=4, rng=np.random.default_rng(0))
    assert len(seqs) == 5
    for s in seqs:
        assert _is_tight_complete(grammar, s), s


def test_blocksemiar_builds_expected_tree():
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    seqs = diffusion_sample_blocksemiar(den, grammar, 4, 16, dyn, device="cpu",
                                        block_size=4, rng=np.random.default_rng(1))
    for s in seqs:
        assert s == [0, 4, 4], s


def test_blocksemiar_reports_zero_repair_on_clean_distribution():
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((6, 1), dtype=torch.long)
    stats = {}
    diffusion_sample_blocksemiar(den, grammar, 6, 16, dyn, device="cpu", block_size=4,
                                 rng=np.random.default_rng(2), stats=stats)
    assert stats["batch"] == 6, stats
    assert stats["n_repaired"] == 0, stats
    assert stats["repair_rate"] == 0.0, stats


class BlockBoundaryDenoiser:
    """Block 0 (positions 0-3) strongly prefers 'x2'(5); block 1+ (positions >=4)
    strongly prefer 'x1'(4). If block order is respected, position 0's committed
    value must come from block 0's distribution regardless of confidence -- this
    just confirms later-block positions never leak into an earlier block's commit
    when block_size=4 on a max_len=16 root-degenerate ('x2' alone) case."""
    def __init__(self, n_choices):
        self.n_choices = n_choices
        self.mask_id = n_choices

    def __call__(self, grid, t, dyn):
        B, L = grid.shape
        logits = torch.zeros(B, L, self.n_choices)
        logits[:, :4, 5] = 20.0     # block 0: 'x2'
        logits[:, 4:, 4] = 20.0     # block 1+: 'x1'
        return logits


def test_blocksemiar_resolves_root_from_first_block_only():
    """Root is a single terminal ('x2') decided entirely within block 0; the tree
    completes before block 1 is ever touched."""
    grammar, den = G(), BlockBoundaryDenoiser(8)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    seqs = diffusion_sample_blocksemiar(den, grammar, 4, 16, dyn, device="cpu",
                                        block_size=4, rng=np.random.default_rng(3))
    for s in seqs:
        assert s == [5], s


class OpenerDenoiser:
    """Prefers '+' (binary) at every position -> would never terminate on its own."""
    def __init__(self, n_choices):
        self.n_choices = n_choices
        self.mask_id = n_choices

    def __call__(self, grid, t, dyn):
        B, L = grid.shape
        logits = torch.zeros(B, L, self.n_choices)
        logits[:, :, 0] = 20.0      # '+'
        return logits


def test_tight_completion_is_global_not_per_block():
    """block_size=2 on max_len=7 (odd, tight globally by the last block) with an
    opener-only denoiser: must still complete via the GLOBAL tight-completion guard,
    not spuriously force-close every block early."""
    grammar = Grammar(FakeLib(), max_len=7)
    den = OpenerDenoiser(8)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    stats = {}
    seqs = diffusion_sample_blocksemiar(den, grammar, 4, 7, dyn, device="cpu",
                                        block_size=2, rng=np.random.default_rng(4),
                                        stats=stats)
    for s in seqs:
        assert grammar.is_complete(s), s
    assert stats["repair_rate"] == 0.0, stats


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} blocksemiar-sampler tests passed.")
