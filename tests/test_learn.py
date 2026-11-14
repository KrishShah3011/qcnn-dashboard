import numpy as np
import torch
from qcnn.learn import N, PARAM_LAYOUT, forward, mse, init_params, train, predict, apply
from qcnn.qec import CDT
from qcnn.physics import ed_ground_state


def test_param_count():
    assert sum(n for _, n in PARAM_LAYOUT) == 1308 and N == 15


def test_forward_is_probability_and_norm_preserving():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 2 ** N)) + 1j * rng.normal(size=(3, 2 ** N))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = forward(init_params(0), torch.tensor(v, dtype=CDT))
    assert f.shape == (3,) and torch.all((f >= -1e-12) & (f <= 1 + 1e-12))


def test_apply_matches_kron():
    U = torch.tensor(np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))[0], dtype=CDT)
    psi = torch.zeros(1, 8, dtype=CDT); psi[0, 0b011] = 1      # |011>
    out = apply(psi, U, (1, 2), 3)
    ref = torch.kron(torch.eye(2, dtype=CDT), U) @ psi[0]
    assert torch.allclose(out[0], ref)


def test_training_reduces_loss():
    h1 = np.linspace(0, 2, 6)
    S = torch.tensor(np.stack([ed_ground_state(N, h, 0.0)[1] for h in h1]), dtype=CDT)
    y = torch.tensor((h1 < 1).astype(float))
    snaps = []
    r = train(S, y, seed=0, iters=15, snapshot_iters=(0, 10), on_snapshot=lambda it, th: snaps.append(it))
    assert r['history']['mse'][-1] < r['history']['mse'][0]
    assert snaps == [0, 10, 15]
    assert predict(r['theta'], S).shape == (6,)
