"""QCNN-optimised quantum error correction (Fig. 5, Eqs. 5, 15-17, Methods 'Demonstration of
Learning Procedure for QEC'). Three-qubit unitaries U1 (inner, one per block) and U2 (outer);
encoding = inverse QCNN, decoding = QCNN, pooling absorbed into U1. Block-local noise makes
the 9-qubit channel a product of three identical single-qubit block channels M_U1 (Eq. 17)."""
import numpy as np
import torch

CDT = torch.complex128
torch.set_default_dtype(torch.float64)


def gellmann(d):
    """Generalised Gell-Mann basis of su(d): (d^2 - 1, d, d)."""
    G = []
    for j in range(d):
        for k in range(j + 1, d):
            S = torch.zeros(d, d, dtype=CDT); S[j, k] = S[k, j] = 1; G.append(S)
            A = torch.zeros(d, d, dtype=CDT); A[j, k] = -1j; A[k, j] = 1j; G.append(A)
    for l in range(1, d):
        D = torch.zeros(d, d, dtype=CDT)
        for j in range(l):
            D[j, j] = 1
        D[l, l] = -l
        G.append(D * np.sqrt(2 / (l * (l + 1))))
    return torch.stack(G)


_GM = {}


def unitary(c):
    """U = exp(-i sum_j c_j Lambda_j) (Methods); d from len(c) = d^2 - 1."""
    d = int(round((c.numel() + 1) ** 0.5))
    if d not in _GM:
        _GM[d] = gellmann(d)
    return torch.linalg.matrix_exp(-1j * torch.einsum('j,jab->ab', c.to(CDT), _GM[d]))


I2 = torch.eye(2, dtype=CDT)
PX = torch.tensor([[0, 1], [1, 0]], dtype=CDT)
PY = torch.tensor([[0, -1j], [1j, 0]], dtype=CDT)
PZ = torch.tensor([[1, 0], [0, -1]], dtype=CDT)
P0 = torch.tensor([[1, 0], [0, 0]], dtype=CDT)
HAD = torch.tensor([[1, 1], [1, -1]], dtype=CDT) / np.sqrt(2)


def kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = torch.kron(out, o)
    return out


def _on(q, O):
    return kron(*[O if i == q else I2 for i in range(3)])


def _perm(f):
    """8x8 permutation unitary from a map on bit tuples (qubit 0 = most significant)."""
    U = torch.zeros(8, 8, dtype=CDT)
    for s in range(8):
        b = [(s >> (2 - i)) & 1 for i in range(3)]
        t = f(b)
        U[t[0] * 4 + t[1] * 2 + t[2], s] = 1
    return U


def _cnot(c, t):
    return _perm(lambda b: [b[i] ^ (b[c] if i == t else 0) for i in range(3)])


_TOFF = _perm(lambda b: [b[0], b[1] ^ (b[0] & b[2]), b[2]])
U_BIT = _TOFF @ _cnot(1, 2) @ _cnot(1, 0)        # majority-vote decoder of |000>+|111> (data = middle)
U_PHASE = U_BIT @ kron(HAD, HAD, HAD)             # same in the X basis
I8 = torch.eye(8, dtype=CDT)
FIXED_CODES = {'identity': (I8, I8), 'shor': (U_BIT, U_PHASE), 'phase_phase': (U_PHASE, U_PHASE),
               'bit_bit': (U_BIT, U_BIT), 'phase_bit': (U_PHASE, U_BIT)}


def _noise(R, p, pxx):
    """Eq. (15) on each of the 3 block qubits, then Eq. (16) on pairs (0,1), (1,2)."""
    for q in range(3):
        out = (1 - sum(p)) * R
        for pp, S in zip(p, (PX, PY, PZ)):
            O = _on(q, S)
            out = out + pp * O @ R @ O.conj().T
        R = out
    for q in (0, 1):
        O = _on(q, PX) @ _on(q + 1, PX)
        R = (1 - pxx) * R + pxx * O @ R @ O
    return R


def _keep_middle(R):
    return torch.einsum('aibajb->ij', R.reshape(2, 2, 2, 2, 2, 2))


def _block_super(U1, p, pxx):
    """Eq. (17) as S[a,b,c,d] = M_U1(|c><d|)[a,b]."""
    outs = []
    for c in range(2):
        for d in range(2):
            E = torch.zeros(2, 2, dtype=CDT); E[c, d] = 1
            R = U1.conj().T @ kron(P0, E, P0) @ U1
            outs.append(_keep_middle(U1 @ _noise(R, p, pxx) @ U1.conj().T))
    return torch.stack(outs).reshape(2, 2, 2, 2).permute(2, 3, 0, 1)


def _apply3(S, R):
    T = R.reshape(2, 2, 2, 2, 2, 2)
    T = torch.einsum('abcd,cxydzw->axybzw', S, T)
    T = torch.einsum('abcd,xcyzdw->xayzbw', S, T)
    T = torch.einsum('abcd,xycwzd->xyawzb', S, T)
    return T.reshape(8, 8)


_STATES = [v for S in (PX, PY, PZ) for v in torch.linalg.eigh(S)[1].T]   # |+-x>, |+-y>, |+-z>


def logical_error(U1, U2, p, pxx):
    """1.5 (1 - f_q), f_q = mean over the six Pauli eigenstates of <psi| M^-1(N(M(psi))) |psi> (Eq. 5)."""
    p = tuple(float(x) for x in p)
    S = _block_super(U1, p, pxx)
    f = 0
    for s in _STATES:
        R = U2.conj().T @ kron(P0, torch.outer(s, s.conj()), P0) @ U2
        R = U2 @ _apply3(S, R) @ U2.conj().T
        f = f + torch.real(s.conj() @ _keep_middle(R) @ s)
    return 1.5 * (1 - f / 6)


def bloch_map(U1, p, pxx):
    """M_U1 as r -> M r + c on the Bloch ball (Methods)."""
    p = tuple(float(x) for x in p)
    S = _block_super(U1, p, pxx)
    sig = (PX, PY, PZ)
    out = lambda E: torch.einsum('abcd,cd->ab', S, E)
    M = torch.stack([torch.stack([0.5 * torch.real(torch.trace(sig[i] @ out(sig[j]))) for j in range(3)])
                     for i in range(3)])
    c = torch.stack([0.5 * torch.real(torch.trace(sig[i] @ out(I2))) for i in range(3)])
    return M, c


def c1_cost(U1, p, pxx):
    """C1 = p1^2 + p2 + p3 from the singular values of M (p_i = (1 + s_i - s_j - s_k)/4, sorted)."""
    s = torch.linalg.svdvals(bloch_map(U1, p, pxx)[0])
    q = torch.stack([(1 + s[i] - s[(i + 1) % 3] - s[(i + 2) % 3]) / 4 for i in range(3)])
    q = torch.sort(q, descending=True).values
    return q[0] ** 2 + q[1] + q[2]


def bold_driver(loss_fn, c0, iters, eta0=1.0):
    """Gradient descent with the bold-driver rate (+5% on success, -50% on failure), Methods."""
    c = c0.clone().requires_grad_(True)
    L = loss_fn(c)
    L.backward()
    eta, hist = eta0, [L.item()]
    for _ in range(iters):
        with torch.no_grad():
            c2 = (c - eta * c.grad).requires_grad_(True)
        L2 = loss_fn(c2)
        if L2.item() < L.item():
            L2.backward()
            c, L, eta = c2, L2, eta * 1.05
        else:
            eta *= 0.5
        hist.append(L.item())
    return c.detach(), hist


def train_code(p, pxx, seeds=4, iters1=1500, iters2=1500, init_scale=2 * np.pi):
    """Sequential optimisation (Methods): U1 minimises C1, then U2 minimises the logical error.
    Paper init is c ~ U[0, 2 pi); init_scale=1 converges faster but tends to stop at the
    phase-phase repetition code instead of beating it. Descent runs on log(cost): same minima,
    but step sizes no longer shrink with the error rate (costs are ~1e-5 at p_tot = 1e-3)."""
    best1 = None
    for s in range(seeds):
        c0 = torch.rand(63, generator=torch.Generator().manual_seed(s)) * init_scale
        c, h = bold_driver(lambda c: torch.log(c1_cost(unitary(c), p, pxx)), c0, iters1)
        if best1 is None or h[-1] < best1[1][-1]:
            best1 = (c, h)
    U1 = unitary(best1[0])
    best2 = None
    for s in range(seeds):
        c0 = torch.rand(63, generator=torch.Generator().manual_seed(100 + s)) * init_scale
        c, h = bold_driver(lambda c: torch.log(logical_error(U1, unitary(c), p, pxx)), c0, iters2)
        if best2 is None or h[-1] < best2[1][-1]:
            best2 = (c, h)
    return {'c1': best1[0], 'c2': best2[0], 'logical': float(np.exp(best2[1][-1])),
            'history1': np.exp(best1[1]).tolist(), 'history2': np.exp(best2[1]).tolist()}
