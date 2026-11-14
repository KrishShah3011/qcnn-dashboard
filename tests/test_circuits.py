import numpy as np
import pytest
import quimb.tensor as qtn
from qcnn.physics import ed_ground_state, dmrg_ground_state, PAULI
from qcnn.circuits import (GATES, apply_gate, sv_expectation, run_statevector, run_mps, check_depth,
                           pauli_decompose, unit_unitary, pull_back, multiscale_sop, gate_count,
                           syndrome_density)


def cluster(N):
    psi = np.ones(2 ** N, dtype=complex) / 2 ** (N / 2)
    for i in range(N - 1):
        psi = apply_gate(psi, GATES['cz'], (i, i + 1), N)
    return psi


def test_fixed_point():                       # Methods, criterion 1
    r = run_statevector(cluster(9), 9, 1)
    assert np.allclose(r['syndromes'][0], 1) and abs(r['outputs'][0] - 1) < 1e-10
    assert np.allclose(syndrome_density(r['syndromes']), [0.0], atol=1e-12)


def test_single_x_error_is_detected_and_corrected():   # Methods, criterion 2 / Fig. 3
    for e in range(1, 8):
        r = run_statevector(apply_gate(cluster(9), PAULI['X'], (e,), 9), 9, 1)
        assert min(r['syndromes'][0]) < -0.99, e
        assert abs(r['outputs'][0] - 1) < 1e-10, e


def test_mps_matches_statevector():
    for N in (9, 15):
        _, v = ed_ground_state(N, 0.5, 0.3)
        sv = run_statevector(v, N, 1)
        mp = run_mps(qtn.MatrixProductState.from_dense(v, dims=[2] * N, cutoff=0.0), N, 1, max_bond=256, cutoff=0.0)
        assert abs(sv['outputs'][0] - mp['outputs'][0]) < 1e-8
        assert np.allclose(sv['syndromes'][0], mp['syndromes'][0], atol=1e-8)


def test_depth_two_fixed_point_mps():
    _, psi = dmrg_ground_state(27, 0.0, 0.0, bond_dim=16)
    r = run_mps(psi, 27, 2, max_bond=64)
    assert all(abs(o - 1) < 1e-4 for o in r['outputs'])


def test_bad_depth():
    for N, d in [(15, 2), (10, 1), (9, 2)]:
        with pytest.raises(ValueError):
            check_depth(N, d)
    check_depth(45, 2)
    check_depth(81, 3)


def test_heisenberg_rules_match_dense():      # Eqs. (8)-(9), interior site 4 of N=9
    N = 9
    U = unit_unitary(N)
    for p in 'XYZ':
        O = U.conj().T @ np.kron(np.kron(np.eye(16), PAULI[p]), np.eye(16)) @ U
        dense = pauli_decompose(O, N)
        sym = {''.join(ops.get(i, 'I') for i in range(N)): c for c, ops in pull_back([(1.0, {1: p})])}
        assert set(dense) == set(sym), p
        assert all(abs(dense[k] - sym[k]) < 1e-9 for k in sym), p


def test_multiscale_sop_equals_circuit_output():   # Eq. (10) at d=1, random state, N=15
    N = 15
    rng = np.random.default_rng(1)
    v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    v /= np.linalg.norm(v)
    direct = run_statevector(v, N, 1)['outputs'][0]          # Z X Z on live sites (4, 7, 10)
    terms = multiscale_sop(1, centre=2)                       # level-1 sites 1,2,3 -> 4,7,10
    via = sum(c * sv_expectation(v, ops, N) for c, ops in terms)
    assert abs(direct - via) < 1e-10
    assert len(terms) == 16


def test_gate_count():
    assert gate_count(27, 1) == 61            # 26 + 8 + 9 + 16 conv/pool gates + 2 FC CZs
