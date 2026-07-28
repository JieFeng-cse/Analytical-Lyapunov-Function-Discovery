"""Verify the dynamics-encoder tokenizer (utils.train.prepare_encoder_input)
handles the rotor (bond-coordinate) drift, which contains sympy subtraction
shapes like Add(p2, Mul(-1, p1)).

`prepare_encoder_input` takes no arguments: it reads the active system via
the module-level `dynamics` name in utils/train.py (imported from `config`
at module load time, i.e. `from config import dynamics, get_config`). To
exercise each rotor system without touching the on-disk config.py, we
monkeypatch `utils.train.dynamics` to the rotor config module's `dynamics`
function for the duration of each test.
"""
import pytest

import example_config.config_rotor2 as rotor2
import example_config.config_rotor3 as rotor3
import example_config.config_rotor4 as rotor4
import example_config.config_rotor5 as rotor5

ROTOR_MODULES = {"N2": rotor2, "N3": rotor3, "N4": rotor4, "N5": rotor5}


@pytest.mark.parametrize("name", sorted(ROTOR_MODULES))
def test_encoder_tokenizes_bond_dynamics(name, monkeypatch):
    import utils.train as train_mod

    rotor_module = ROTOR_MODULES[name]
    monkeypatch.setattr(train_mod, "dynamics", rotor_module.dynamics)

    out = train_mod.prepare_encoder_input()

    assert out is not None
    input_ids, vocab_size, seq_len = out
    assert input_ids is not None
    assert vocab_size > 0
    assert seq_len > 0
    assert len(input_ids) == seq_len
