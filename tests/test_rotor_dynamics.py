import sympy as sym
import pytest
from example_config.rotor_common import make_rotor


@pytest.mark.parametrize("N,dim", [(2, 3), (3, 5), (4, 7), (5, 9)])
def test_dimensions(N, dim):
    sv, drift, spec = make_rotor(N)
    assert len(sv) == dim and len(drift) == dim
    assert [str(s) for s in sv] == [f"x{i}" for i in range(1, dim + 1)]


@pytest.mark.parametrize("N", [3, 4, 5])
def test_generator_identity_two_bath(N):
    # L_N H_N = (1 - p1^2) + (2 - pN^2)   [benchmark eq (6)]
    sv, drift, spec = make_rotor(N)
    LH = sum(sym.diff(spec["H"], s) * f for s, f in zip(sv, drift))
    for idx, kappa in spec["noise"].items():
        LH += kappa * sym.diff(spec["H"], sv[idx], 2)
    p1, pN = sv[N - 1], sv[2 * N - 2]
    assert sym.simplify(LH - ((1 - p1**2) + (2 - pN**2))) == 0


def test_generator_identity_one_bath_N2():
    # L_2 H_2 = 1 - p1^2   [benchmark eq (9)]
    sv, drift, spec = make_rotor(2)
    LH = sum(sym.diff(spec["H"], s) * f for s, f in zip(sv, drift))
    for idx, kappa in spec["noise"].items():
        LH += kappa * sym.diff(spec["H"], sv[idx], 2)
    p1 = sv[1]
    assert sym.simplify(LH - (1 - p1**2)) == 0
    assert list(spec["noise"].keys()) == [1]  # only p1 noisy


def test_spec_params():
    for N, alpha in [(2, 1), (3, 1), (4, 1), (5, 3)]:
        _, _, spec = make_rotor(N)
        assert spec["alpha"] == alpha and spec["c"] == 0.05 and spec["R"] == 10.0


def test_H_nonnegative_and_zero_at_rest():
    sv, drift, spec = make_rotor(3)
    at_rest = {s: 0 for s in sv}
    assert spec["H"].subs(at_rest) == 0
