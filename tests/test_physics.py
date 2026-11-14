import numpy as np
from qcnn.physics import (hamiltonian_sparse, ed_ground_state, dmrg_ground_state, sop_ops,
                          sample_complexity, mps_expectation)


def test_cluster_point_energy():
    e, _ = ed_ground_state(9, 0.0, 0.0)
    assert abs(e + 7) < 1e-8            # N-2 commuting stabilisers ZXZ


def test_hermitian():
    H = hamiltonian_sparse(7, 0.3, -0.4)
    assert abs(H - H.T).max() < 1e-12


def test_ed_reproducible():
    _, a = ed_ground_state(9, 0.2, 0.1)
    _, b = ed_ground_state(9, 0.2, 0.1)
    assert np.allclose(a, b)


def test_dmrg_matches_ed():
    for h1, h2 in [(0.5, 0.3), (1.2, -0.8)]:
        e_ed, _ = ed_ground_state(9, h1, h2)
        e_mps, _ = dmrg_ground_state(9, h1, h2, bond_dim=32)
        assert abs(e_ed - e_mps) < 1e-6


def test_sop_ops_shape():
    assert sop_ops(15, 6) == {4: 'Z', 5: 'X', 7: 'X', 9: 'X', 10: 'Z'}


def test_sop_cluster_state_is_one():
    _, psi = dmrg_ground_state(15, 0.0, 0.0, bond_dim=16)
    assert abs(mps_expectation(psi, sop_ops(15, 6)) - 1) < 1e-6


def test_sample_complexity():
    m = sample_complexity([1.0, 0.0, -0.5])
    assert abs(m[0] - 1.96 ** 2 / (np.pi / 2 - np.pi / 4) ** 2) < 1e-9
    assert np.isnan(m[1]) and np.isnan(m[2])
