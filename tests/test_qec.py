import numpy as np
import torch
from qcnn.qec import gellmann, unitary, FIXED_CODES, logical_error, bloch_map, c1_cost, train_code

P_COR = (5.8e-3 / 10, 2e-3 / 10, 2e-3 / 10)      # correlated model scaled to total 1e-3
PXX_COR = 2e-4 / 10


def test_gellmann_count_and_unitarity():
    assert gellmann(8).shape == (63, 8, 8)
    U = unitary(torch.rand(63, dtype=torch.float64) * 6.28)
    assert torch.allclose(U @ U.conj().T, torch.eye(8, dtype=U.dtype), atol=1e-10)


def test_identity_code_is_bare_qubit():
    p = (1e-3, 2e-3, 3e-3)
    L = logical_error(*FIXED_CODES['identity'], p, 0.0).item()
    assert abs(L - 6e-3) < 1e-12


def test_shor_corrects_single_errors():       # logical error O(p^2) for isotropic noise
    lo = logical_error(*FIXED_CODES['shor'], (1e-4 / 3,) * 3, 0.0).item()
    hi = logical_error(*FIXED_CODES['shor'], (1e-3 / 3,) * 3, 0.0).item()
    assert lo / hi < 0.02


def test_shor_correlated_matches_reference():   # values from the design-time numerics
    L = logical_error(*FIXED_CODES['shor'], P_COR, PXX_COR).item()
    assert abs(L - 1.296e-4) / 1.296e-4 < 0.01


def test_bloch_map_identity_channel():
    M, c = bloch_map(torch.eye(8, dtype=torch.complex128), (0.0, 0.0, 0.0), 0.0)
    assert torch.allclose(M, torch.eye(3, dtype=M.dtype), atol=1e-12) and torch.allclose(c, torch.zeros(3, dtype=c.dtype), atol=1e-12)


def test_training_beats_shor_quickly():
    p, pxx = (5.8e-3, 2e-3, 2e-3), 2e-4
    r = train_code(p, pxx, seeds=2, iters1=200, iters2=600, init_scale=1.0)
    shor = logical_error(*FIXED_CODES['shor'], p, pxx).item()
    assert r['logical'] < shor and r['history2'][-1] <= r['history2'][0]
