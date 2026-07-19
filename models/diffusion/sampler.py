"""Masked-diffusion sampling loop with structural early-stop (design doc §6-7).

Phase-1 reconstruction order = pre-order, left-to-right (gap-free), so the grammar's
`dangling==0` early-stop is exact. Validity is enforced in-loop via the grammar's
valid-token mask; sequences that never complete by `max_len` are repaired.
"""
import math
import numpy as np
import torch


@torch.no_grad()
def diffusion_sample(denoiser, grammar, batch, max_len, dyn_tokens,
                     device, temperature=1.0):
    """Returns a list of `batch` complete pre-order token-id lists."""
    n_choices = grammar.L
    grid = torch.full((batch, max_len), denoiser.mask_id, dtype=torch.long, device=device)
    seqs = [[] for _ in range(batch)]
    done = [False] * batch

    for pos in range(max_len):
        if all(done):
            break
        t = torch.full((batch,), 1.0 - pos / max_len, device=device)        # noise level ↓
        logits = denoiser(grid, t, dyn_tokens)[:, pos, :]                    # (B, n_choices)
        for b in range(batch):
            if done[b]:
                continue
            mask = grammar.valid_token_mask(seqs[b])[:n_choices].copy()      # real-token mask
            # Force completion within max_len: when remaining slots are tight, terminals only.
            if (max_len - pos) <= grammar.dangling(seqs[b]):
                term_only = np.zeros(n_choices, dtype=bool)
                term_only[grammar.terminal_tokens] = True
                mask &= term_only
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
            lg = logits[b].masked_fill(~valid, float("-inf"))
            if torch.isinf(lg).all():                                       # model killed all valid -> uniform
                probs = valid.float()
            else:
                probs = torch.softmax(lg / temperature, dim=-1)
            tok = int(torch.multinomial(probs, 1))
            seqs[b].append(tok)
            grid[b, pos] = tok
            if grammar.is_complete(seqs[b]):
                done[b] = True

    return [s if grammar.is_complete(s) else grammar.repair(s) for s in seqs]


def _revealed_before(grid_row, pos, L):
    """Real tokens (< L) at positions [0, pos), in position order, skipping gaps."""
    out = []
    for j in range(pos):
        tok = int(grid_row[j])
        if tok < L:
            out.append(tok)
    return out


def _decode(grid_row, grammar):
    """Trim the gap-free prefix to its minimal complete tree; else repair."""
    slots, out = 1, []
    for tok in grammar.contiguous_prefix(grid_row):
        out.append(tok)
        slots += grammar.arity(tok) - 1
        if slots == 0:
            return out
    return grammar.repair([int(t) for t in grid_row])      # _real() drops gaps + tail


@torch.no_grad()
def diffusion_sample_random(denoiser, grammar, batch, max_len, dyn_tokens,
                            device, temperature=1.0, rng=None, order=None):
    """Phase-3 any-order reverse sampler (design doc §6-7).

    Reveals masked positions in a RANDOM order (matching the random-subset masking
    the denoiser is trained on), rather than the semi-autoregressive pre-order of
    `diffusion_sample`. Validity is the grammar's lightweight mask over the
    gap-skipping revealed context; completion uses the no-interior-MASK guard
    (`is_complete_grid`); sequences that never complete by `max_len` are repaired.

    `rng`   : np.random.Generator for the per-sample reveal permutation (testable).
    `order` : optional list of per-sample position orderings (overrides `rng`).
    """
    n_choices = grammar.L
    mask_id = denoiser.mask_id
    grid = torch.full((batch, max_len), mask_id, dtype=torch.long, device=device)
    done = [False] * batch

    if order is None:
        if rng is None:
            rng = np.random.default_rng()
        order = [list(rng.permutation(max_len)) for _ in range(batch)]
    ptr = [0] * batch

    for step in range(max_len):
        if all(done):
            break
        t = torch.full((batch,), 1.0 - step / max_len, device=device)
        logits = denoiser(grid, t, dyn_tokens)                              # (B,L,n_choices)
        for b in range(batch):
            if done[b]:
                continue
            pos = int(order[b][ptr[b]])
            ptr[b] += 1
            ctx = _revealed_before(grid[b].tolist(), pos, n_choices)
            mask = grammar.valid_token_mask(ctx)[:n_choices].copy()          # lightweight, drop PAD
            if not mask.any():                                               # ctx already complete -> PAD-only;
                mask[grammar.terminal_tokens] = True                         # denoiser can't emit PAD, trim at decode
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
            lg = logits[b, pos].masked_fill(~valid, float("-inf"))
            if torch.isinf(lg).all():                                        # model killed all valid -> uniform
                probs = valid.float()
            else:
                probs = torch.softmax(lg / temperature, dim=-1)
            grid[b, pos] = int(torch.multinomial(probs, 1))
            if grammar.is_complete_grid(grid[b].tolist()):                   # §7 guard (no interior MASK)
                done[b] = True

    return [_decode(grid[b].tolist(), grammar) for b in range(batch)]


def _reveal_counts(max_len, num_steps, schedule="cosine"):
    """Per-step count of NEW positions to reveal (MaskGIT-style schedule).

    Non-negative, sums to exactly `max_len`, length == max(1, num_steps); the
    implied masked-count sequence is non-increasing. The last step takes the
    remainder so everything is revealed by step T.
    """
    T = max(1, int(num_steps))
    if schedule == "linear":
        targets = [max_len - round(max_len * (i + 1) / T) for i in range(T)]
    else:  # cosine: masked fraction ~ cos(pi/2 * (i+1)/T)
        targets = [round(max_len * math.cos(math.pi / 2 * (i + 1) / T)) for i in range(T)]
    counts, prev = [], max_len
    for tgt in targets:
        tgt = min(prev, max(0, int(tgt)))
        counts.append(prev - tgt)
        prev = tgt
    counts[-1] += prev            # reveal any remainder on the final step
    return counts


def _committed_prefix(grid_row, committed_row, L):
    """Contiguous, committed, real-token prefix from position 0 (stops at the first
    not-yet-committed slot, PAD, or MASK). Used for the tight-completion `dangling`."""
    out = []
    for p in range(len(grid_row)):
        if not bool(committed_row[p]):
            break
        tok = int(grid_row[p])
        if tok >= L:
            break
        out.append(tok)
    return out


def _decode_flag(grid_row, grammar):
    """Trim the gap-free prefix to its minimal complete tree; else repair.
    Returns (tokens, used_repair)."""
    slots, out = 1, []
    for tok in grammar.contiguous_prefix(grid_row):
        out.append(tok)
        slots += grammar.arity(tok) - 1
        if slots == 0:
            return out, False
    return grammar.repair([int(t) for t in grid_row]), True


def _blocks(max_len, block_size):
    """Consecutive position blocks of `block_size` covering [0, max_len); last block
    may be shorter."""
    b = max(1, int(block_size))
    return [list(range(s, min(s + b, max_len))) for s in range(0, max_len, b)]


def _opfirst_rank(free, conf, is_op):
    """Rank free positions operator-candidates first, confidence as tiebreak
    within each class (inverse of pure-confidence order, which defers
    high-arity/operator commitments until context sharpens)."""
    return sorted(free, key=lambda q: (is_op[q], conf[q]), reverse=True)


@torch.no_grad()
def diffusion_sample_blocksemiar(denoiser, grammar, batch, max_len, dyn_tokens, device,
                                 *, block_size=4, steps_per_block=None, temperature=1.0,
                                 schedule="cosine", rng=None, stats=None):
    """Block semi-AR reveal order (BACKLOG #2): positions are grouped into consecutive
    blocks of `block_size` and resolved in PRE-ORDER across blocks (block 0 fully
    committed before block 1 starts, ...); WITHIN a block, resolution uses the same
    confidence-ranked multi-step commit loop as `diffusion_sample_confidence`, so the
    reveal order is a dial between pure pre-order (block_size=1) and pure confidence
    order (block_size>=max_len). Tight-completion pressure (forcing terminals) is based
    on GLOBAL remaining room across the whole row, not just the current block, so an
    early block never force-closes the tree just because ITS OWN slots are running low
    while later blocks still have room. `grammar.repair` is the last-resort decode.
    """
    n_choices = grammar.L
    mask_id = denoiser.mask_id
    grid = torch.full((batch, max_len), mask_id, dtype=torch.long, device=device)
    committed = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    done = [False] * batch
    term_mask = np.zeros(n_choices, dtype=bool)
    term_mask[grammar.terminal_tokens] = True

    for block in _blocks(max_len, block_size):
        if all(done):
            break
        counts = _reveal_counts(len(block), steps_per_block or len(block), schedule)
        for step, k in enumerate(counts):
            if all(done):
                break
            t = torch.full((batch,), 1.0 - step / len(counts), device=device)
            logits = denoiser(grid, t, dyn_tokens)                   # (B, L, n_choices)
            last = (step == len(counts) - 1)
            grid_list = grid.tolist()
            comm_list = committed.tolist()
            for b in range(batch):
                if done[b]:
                    continue
                free = [p for p in block if not comm_list[b][p]]
                if not free:
                    continue
                total_free = max_len - sum(comm_list[b])                 # GLOBAL, not block-local
                d = grammar.dangling(_committed_prefix(grid_list[b], comm_list[b], n_choices))
                tight = total_free <= d
                conf, toks = {}, {}
                for p in free:
                    ctx = _revealed_before(grid_list[b], p, n_choices)
                    mask = grammar.valid_token_mask(ctx)[:n_choices].copy()
                    if tight:                                          # force terminals to close the tree
                        m2 = mask & term_mask
                        if m2.any():
                            mask = m2
                    if not mask.any():
                        mask[int(grammar.terminal_tokens[0])] = True
                    valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    lg = logits[b, p].masked_fill(~valid, float("-inf"))
                    if torch.isinf(lg).all():
                        probs = valid.float()
                    else:
                        probs = torch.softmax(lg / temperature, dim=-1)
                    probs = probs / probs.sum()
                    tok = int(torch.multinomial(probs, 1))
                    toks[p] = tok
                    conf[p] = float(probs[tok])
                kk = len(free) if last else max(1, min(int(k), len(free)))
                reveal = sorted(free, key=lambda q: conf[q], reverse=True)[:kk]
                for p in reveal:
                    grid[b, p] = toks[p]
                    committed[b, p] = True
                if grammar.is_complete_grid(grid[b].tolist()):
                    done[b] = True

    seqs, n_repaired = [], 0
    for b in range(batch):
        seq, used_repair = _decode_flag(grid[b].tolist(), grammar)
        seqs.append(seq)
        n_repaired += int(used_repair)
    if stats is not None:
        stats["batch"] = batch
        stats["n_repaired"] = n_repaired
        stats["repair_rate"] = n_repaired / max(1, batch)
    return seqs


@torch.no_grad()
def diffusion_sample_opblock(denoiser, grammar, batch, max_len, dyn_tokens, device,
                             *, block_size=4, steps_per_block=None, temperature=1.0,
                             schedule="cosine", rng=None, stats=None):
    """Operator-first WITHIN block semi-AR reveal order (invented 2026-07-19: combines
    BACKLOG #1 opfirst and #2 blocksemiar). Identical block partition and GLOBAL
    tight-completion pressure to `diffusion_sample_blocksemiar`, but the in-block
    reveal ranking is `_opfirst_rank` (operator-candidates first, confidence as
    tiebreak) instead of confidence alone -- so within each block, any candidate with
    arity > 0 commits before every terminal candidate that step, same as
    `diffusion_sample_opfirst`, just restricted to the current block's free positions.
    """
    n_choices = grammar.L
    mask_id = denoiser.mask_id
    grid = torch.full((batch, max_len), mask_id, dtype=torch.long, device=device)
    committed = torch.zeros((batch, max_len), dtype=torch.bool, device=device)
    done = [False] * batch
    term_mask = np.zeros(n_choices, dtype=bool)
    term_mask[grammar.terminal_tokens] = True

    for block in _blocks(max_len, block_size):
        if all(done):
            break
        counts = _reveal_counts(len(block), steps_per_block or len(block), schedule)
        for step, k in enumerate(counts):
            if all(done):
                break
            t = torch.full((batch,), 1.0 - step / len(counts), device=device)
            logits = denoiser(grid, t, dyn_tokens)                   # (B, L, n_choices)
            last = (step == len(counts) - 1)
            grid_list = grid.tolist()
            comm_list = committed.tolist()
            for b in range(batch):
                if done[b]:
                    continue
                free = [p for p in block if not comm_list[b][p]]
                if not free:
                    continue
                total_free = max_len - sum(comm_list[b])                 # GLOBAL, not block-local
                d = grammar.dangling(_committed_prefix(grid_list[b], comm_list[b], n_choices))
                tight = total_free <= d
                conf, toks, is_op = {}, {}, {}
                for p in free:
                    ctx = _revealed_before(grid_list[b], p, n_choices)
                    mask = grammar.valid_token_mask(ctx)[:n_choices].copy()
                    if tight:                                          # force terminals to close the tree
                        m2 = mask & term_mask
                        if m2.any():
                            mask = m2
                    if not mask.any():
                        mask[int(grammar.terminal_tokens[0])] = True
                    valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
                    lg = logits[b, p].masked_fill(~valid, float("-inf"))
                    if torch.isinf(lg).all():
                        probs = valid.float()
                    else:
                        probs = torch.softmax(lg / temperature, dim=-1)
                    probs = probs / probs.sum()
                    tok = int(torch.multinomial(probs, 1))
                    toks[p] = tok
                    conf[p] = float(probs[tok])
                    is_op[p] = grammar.arity(tok) > 0
                kk = len(free) if last else max(1, min(int(k), len(free)))
                reveal = _opfirst_rank(free, conf, is_op)[:kk]
                for p in reveal:
                    grid[b, p] = toks[p]
                    committed[b, p] = True
                if grammar.is_complete_grid(grid[b].tolist()):
                    done[b] = True

    seqs, n_repaired = [], 0
    for b in range(batch):
        seq, used_repair = _decode_flag(grid[b].tolist(), grammar)
        seqs.append(seq)
        n_repaired += int(used_repair)
    if stats is not None:
        stats["batch"] = batch
        stats["n_repaired"] = n_repaired
        stats["repair_rate"] = n_repaired / max(1, batch)
    return seqs
