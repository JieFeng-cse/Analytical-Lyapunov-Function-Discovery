"""Pre-order expression grammar utilities for the discrete-diffusion generator.

Adapts the DDSR-style validity / early-stop / repair logic (design doc §7) to the
repo's `dso` `Library`. Operates on integer token ids:
  * real tokens   : 0 .. L-1   (arity from `library.arities`)
  * PAD (virtual) : L          (used only after an expression is complete)
  * MASK (virtual): L+1         (an unfilled slot during diffusion; never sampled)

Pure numpy — no torch — so the core is unit-testable on CPU. `library` is duck-typed:
it only needs `.L`, `.arities`, `.terminal_tokens`, optional `.trig_tokens`,
optional `.const_token`.
"""
import numpy as np


class Grammar:
    def __init__(self, library, max_const=10, ban_nested_trig=True, max_len=64,
                 root_token=None, angle_tokens=None, angle_parents=None,
                 ban_nested=None):
        self.library = library
        # blind skeleton search: token forced at position 0, forbidden elsewhere
        self.root_token = None if root_token is None else int(root_token)
        # periodicity: angle input tokens allowed only as the direct child of
        # one of angle_parents (sin/cos)
        self.angle_tokens = set(int(t) for t in (angle_tokens or []))
        self.angle_parents = set(int(t) for t in (angle_parents or []))
        self.safe_terminal = None
        # tokens that may not appear anywhere inside themselves (step, exp)
        self.ban_nested = set(int(t) for t in (ban_nested or []))
        self.L = int(library.L)
        self.arities = np.asarray(library.arities, dtype=np.int64)
        self.PAD = self.L
        self.MASK = self.L + 1
        self.terminal_tokens = np.asarray(library.terminal_tokens, dtype=np.int64)
        _trig = getattr(library, "trig_tokens", None)
        self.trig_tokens = set(int(t) for t in ([] if _trig is None else _trig))
        self.const_token = getattr(library, "const_token", None)
        self.max_const = max_const
        self.ban_nested_trig = ban_nested_trig
        self.max_len = max_len
        self.safe_terminal = int(self.terminal_tokens[0])
        for t in self.terminal_tokens:
            if int(t) not in self.angle_tokens:
                self.safe_terminal = int(t)
                break

    # --- arity-aware completion (the "open slots" / dangling counter) -----------
    def arity(self, tok):
        return 0 if tok >= self.L else int(self.arities[tok])

    def _real(self, tokens):
        return [int(t) for t in tokens if t != self.PAD and t != self.MASK]

    def dangling(self, tokens):
        """Number of still-unfilled child slots over the real-token prefix (start 1)."""
        slots = 1
        for tok in self._real(tokens):
            if slots <= 0:
                break
            slots += self.arity(tok) - 1
        return slots

    def is_complete(self, tokens):
        """True iff the real-token prefix forms exactly one complete expression tree."""
        slots = 1
        for tok in self._real(tokens):
            if slots <= 0:
                return False          # extra token after the tree already closed
            slots += self.arity(tok) - 1
            if slots == 0:
                return True
        return False

    # --- random-order (Phase-3) completion guard (design doc §7) ----------------
    def contiguous_prefix(self, grid):
        """Real tokens from position 0 up to the first non-real slot.

        Any id >= L is a boundary: PAD (=L), MASK (=L+1), and the denoiser's
        mask sentinel (=L) all stop the scan. Unlike `_real`, this does NOT skip
        over interior gaps, so it reflects the truly-revealed left prefix.
        """
        out = []
        for tok in grid:
            if int(tok) >= self.L:
                break
            out.append(int(tok))
        return out

    def is_complete_grid(self, grid):
        """§7 guard for random-order sampling: complete iff the gap-free prefix
        from position 0 is exactly one tree. Prevents false completion across
        interior MASKs (which `is_complete` would skip via `_real`)."""
        return self.is_complete(self.contiguous_prefix(grid))

    # --- open ancestors (pre-order), used for structural constraints ------------
    def open_ancestors(self, tokens):
        """Operator tokens on the path to the next slot (pre-order), innermost last."""
        stack = []  # [token, remaining_children]
        for tok in self._real(tokens):
            if stack:
                stack[-1][1] -= 1
            if self.arity(tok) > 0:
                stack.append([tok, self.arity(tok)])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return [s[0] for s in stack]

    def _would_nest_trig(self, tokens, tid):
        if not self.ban_nested_trig or tid not in self.trig_tokens:
            return False
        return any(a in self.trig_tokens for a in self.open_ancestors(tokens))

    # --- in-loop valid-token mask (design doc §7, lightweight variant) ----------
    def valid_token_mask(self, tokens):
        """Boolean mask over [0 .. L]  (real tokens + PAD). MASK is never sampled."""
        mask = np.zeros(self.L + 1, dtype=bool)
        if self.is_complete(tokens):
            mask[self.PAD] = True            # only PAD allowed once complete
            return mask
        const_count = (sum(1 for t in self._real(tokens) if t == self.const_token)
                       if self.const_token is not None else 0)
        if self.root_token is not None and len(self._real(tokens)) == 0:
            mask[self.root_token] = True     # position 0: only the root token
            return mask
        anc = self.open_ancestors(tokens) if (self.angle_tokens or self.ban_nested) else None
        angle_ok = bool(anc) and anc[-1] in self.angle_parents
        for tid in range(self.L):
            if self.const_token is not None and tid == self.const_token and const_count >= self.max_const:
                continue
            if self.root_token is not None and tid == self.root_token:
                continue
            if tid in self.angle_tokens and not angle_ok:
                continue
            if tid in self.ban_nested and anc is not None and tid in anc:
                continue
            if self._would_nest_trig(tokens, tid):
                continue
            mask[tid] = True                 # PAD stays False before completion
        if not mask.any():                    # never strand the sampler
            mask[self.safe_terminal] = True
        return mask

    # --- repair + fallback (design doc §7) --------------------------------------
    def fallback(self):
        t = self.safe_terminal
        if self.root_token is None:
            return [t]
        return [self.root_token] + [t] * self.arity(self.root_token)

    def repair(self, tokens):
        """Coerce a (possibly invalid) token list into a complete valid pre-order tree."""
        out = []
        real = self._real(tokens)
        if self.root_token is not None:
            out = [self.root_token]
            if real and real[0] == self.root_token:
                real = real[1:]
        for tok in real:
            if self.dangling(out) <= 0:
                break
            if not self.valid_token_mask(out)[tok]:   # invalid given context -> terminal
                tok = self.safe_terminal
            out.append(int(tok))
            if self.is_complete(out):
                return out
        while self.dangling(out) > 0 and len(out) < self.max_len:
            out.append(self.safe_terminal)
        return out if self.is_complete(out) else self.fallback()

    def pad_after_complete(self, tokens, length):
        """Trim to the complete prefix and right-pad with PAD to `length`."""
        out, slots = [], 1
        for tok in self._real(tokens):
            if slots <= 0:
                break
            out.append(int(tok))
            slots += self.arity(tok) - 1
            if slots == 0:
                break
        out = out[:length] + [self.PAD] * max(0, length - len(out))
        return out[:length]
