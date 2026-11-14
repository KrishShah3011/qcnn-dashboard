"""Randomly initialised, trained QCNN for phase recognition (Fig. 4, Fig. 9, Eq. 1, Methods).
Statevector simulation in torch on N = 15 qubits, pooling measurements deferred."""
import numpy as np
import torch
from qcnn.qec import unitary, CDT

N = 15
PARAM_LAYOUT = [('C1', 90), ('C2', 63), ('C3', 63), ('C4', 63), ('V1', 3), ('V2', 3), ('FC', 1023)]
_I2 = torch.eye(2, dtype=CDT)
_H = torch.tensor([[1, 1], [1, -1]], dtype=CDT) / np.sqrt(2)
_X = torch.tensor([[0, 1], [1, 0]], dtype=CDT)
_P0 = torch.diag(torch.tensor([1, 0], dtype=CDT))
_P1 = torch.diag(torch.tensor([0, 1], dtype=CDT))


def apply(psi, U, qubits, n):
    """Apply a k-qubit U to a batch of n-qubit states psi (B, 2**n); qubit 0 = most significant."""
    k = len(qubits)
    t = psi.reshape([-1] + [2] * n)
    t = torch.movedim(t, [q + 1 for q in qubits], list(range(1, k + 1)))
    sh = t.shape
    t = (U @ t.reshape(sh[0], 2 ** k, -1)).reshape(sh)
    return torch.movedim(t, list(range(1, k + 1)), [q + 1 for q in qubits]).reshape(-1, 2 ** n)


def _split(theta):
    out, i = {}, 0
    for name, n in PARAM_LAYOUT:
        out[name] = theta[i:i + n]
        i += n
    return out


def _u1(c):
    """U1 = U(23) U(24) U(13) U(14) U(12) U(34) on a 4-qubit block (Methods); U(34) acts first."""
    M = torch.eye(16, dtype=CDT)
    for j, (a, b) in enumerate([(2, 3), (0, 1), (0, 3), (0, 2), (1, 3), (1, 2)]):
        M = apply(M.T, unitary(c[15 * j:15 * (j + 1)]), (a, b), 4).T
    return M


def _controlled_x_basis(V):
    """Deferred pooling: apply V to the target iff the control's X measurement gives -1."""
    HI = torch.kron(_H, _I2)
    return HI @ (torch.kron(_P0, _I2) + torch.kron(_P1, V)) @ HI


def forward(theta, states):
    """QCNN output f = (1 + <X_6>)/2 for each input state (Fig. 9 circuit, n = 3, d = 1)."""
    p = _split(theta)
    U1, U2, U3, U4 = _u1(p['C1']), unitary(p['C2']), unitary(p['C3']), unitary(p['C4'])
    V1, V2 = _controlled_x_basis(unitary(p['V1'])), _controlled_x_basis(unitary(p['V2']))
    psi = states
    for k in (0, 2, 1, 3):
        psi = apply(psi, U1, (3 * k, 3 * k + 1, 3 * k + 2, 3 * k + 3), N)
    for k in range(4):
        psi = apply(psi, U2, (3 * k + 2, 3 * k + 3, 3 * k + 4), N)
    for k in range(5):
        psi = apply(psi, U3, (3 * k, 3 * k + 1, 3 * k + 2), N)
    for k in range(4):
        psi = apply(psi, U4, (3 * k + 1, 3 * k + 2, 3 * k + 3), N)
    for r in range(0, N, 3):
        psi = apply(psi, V1, (r + 1, r), N)
        if r + 3 < N:
            psi = apply(psi, V2, (r + 2, r + 3), N)
    psi = apply(psi, unitary(p['FC']), (0, 3, 6, 9, 12), N)
    x = torch.real((psi.conj() * apply(psi, _X, (6,), N)).sum(1))
    return (1 + x) / 2


def mse(theta, states, y):
    """Eq. (1): (1/2M) sum (y - f)^2."""
    return ((y - forward(theta, states)) ** 2).sum() / (2 * len(y))


def init_params(seed):
    """All Gell-Mann coefficients uniform in [0, 2 pi) (Methods)."""
    return torch.rand(sum(n for _, n in PARAM_LAYOUT), generator=torch.Generator().manual_seed(seed),
                      dtype=torch.float64) * 2 * np.pi


def train(states, y, seed=0, iters=800, lr=0.01, snapshot_iters=(0, 10, 25, 50, 100, 200, 400),
          on_snapshot=None):
    theta = init_params(seed).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    hist = {'iter': [], 'mse': []}
    for it in range(iters + 1):
        if on_snapshot and (it in snapshot_iters or it == iters):
            on_snapshot(it, theta.detach().clone())
        opt.zero_grad()
        L = mse(theta, states, y)
        hist['iter'].append(it)
        hist['mse'].append(L.item())
        if it == iters:
            break
        L.backward()
        opt.step()
    return {'theta': theta.detach(), 'history': hist,
            'snapshot_iters': sorted(set(s for s in snapshot_iters if s < iters) | {iters})}


def predict(theta, states, chunk=64):
    with torch.no_grad():
        return torch.cat([forward(theta, states[i:i + chunk]) for i in range(0, len(states), chunk)]).numpy()
