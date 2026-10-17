"""Slot menus, token names and decoding for the RL slot search (spec 2026-10-17 sections 2.3, 6).

The program sampled by the generators is  tmpl<n>(S0_j0, S1_j1, ..., S<n-1>_j<n-1>):
one choice token per slot, token S<i>_<j> = j-th entry of slot i's menu.  The prior
`slot` (dso/prior.py SlotConstraint) and the diffusion grammar enforce the order.
"""
from fractions import Fraction

A_MENU = [Fraction(1, 8), Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
          Fraction(2, 3), Fraction(3, 4)]
DEG_MENU = [0, 1, 2, 3, 4, 6]
K_MENU = [2, 3, 4, 6, 8]
L_MENU = [4, 6, 8, 12, 16, 24]
M_MENU = [16, 32, 64, 128, 256]


def _single(j):
    return [(f"gap{j}", [None, f"pt{j}"]), (f"a{j}", A_MENU), (f"deg{j}", DEG_MENU)]


SLOT_MENUS = {
    2: [("k", K_MENU)] + _single(2),
    3: [("k", K_MENU)] + _single(2),
    4: [("k", K_MENU)] + _single(2) + _single(3)
       + [("l", L_MENU), ("m", M_MENU), ("degc", [0, 1, 2])],
    5: [("k", K_MENU), ("l", L_MENU), ("m", M_MENU), ("lr", L_MENU)]
       + _single(2) + _single(3) + _single(4)
       + [("deg24", [None, 0, 1, 2, 3]), ("degp", [None, 0, 1, 2, 3]),
          ("recv", [None, "H234"]), ("degr", [0, 1, 2, 3, 6])],
}

THETA_DEFAULT = {2: Fraction(1, 4), 3: Fraction(1, 4), 4: Fraction(1, 4), 5: Fraction(1, 4)}


def root_name(N):
    return f"tmpl{len(SLOT_MENUS[N])}"


def token_names(N):
    """[root] + slot tokens in slot order."""
    out = [root_name(N)]
    for i, (_, menu) in enumerate(SLOT_MENUS[N]):
        out += [f"S{i}_{j}" for j in range(len(menu))]
    return out


def decode(N, names):
    """Traversal token names -> slots dict, or None if malformed."""
    menus = SLOT_MENUS[N]
    if len(names) != len(menus) + 1 or names[0] != root_name(N):
        return None
    slots = {}
    for i, nm in enumerate(names[1:]):
        if not nm.startswith(f"S{i}_"):
            return None
        j = int(nm.split("_")[1])
        key, menu = menus[i]
        if j >= len(menu):
            return None
        slots[key] = menu[j]
    return slots


def encode(N, slots):
    """slots dict -> token names (for tests / seeding)."""
    names = [root_name(N)]
    for i, (key, menu) in enumerate(SLOT_MENUS[N]):
        names.append(f"S{i}_{menu.index(slots.get(key))}")
    return names
