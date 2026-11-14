"""Model of Cong, Choi & Lukin (arXiv:1810.03787): Hamiltonian Eq. (2), ground states,
string order parameters Eq. (3) and sample complexity Eq. (4)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

I2 = np.eye(2)
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
PAULI = {'I': I2, 'X': X, 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': Z}


def hamiltonian_sparse(N, h1, h2, J=1.0):
    """Eq. (2) on an open chain: -J sum Z X Z - h1 sum X - h2 sum X X. Qubit 0 = most significant bit."""
    idx = np.arange(2 ** N)
    z = [1 - 2 * ((idx >> (N - 1 - i)) & 1) for i in range(N)]
    bit = lambda i: 1 << (N - 1 - i)
    rows, cols, vals = [], [], []

    def term(mask, coef):
        rows.append(idx)
        cols.append(idx ^ mask)
        vals.append(np.broadcast_to(coef, idx.shape).astype(float))

    for i in range(N - 2):
        term(bit(i + 1), -J * z[i] * z[i + 2])
    for i in range(N):
        term(bit(i), -h1)
    for i in range(N - 1):
        term(bit(i) | bit(i + 1), -h2)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(2 ** N, 2 ** N))


def ed_ground_state(N, h1, h2, J=1.0):
    """(energy, normalised ground state) by Lanczos. Fixed v0 keeps the pick inside the
    quasi-degenerate SPT manifold (open chain) reproducible."""
    v0 = np.random.default_rng(0).normal(size=2 ** N)
    e, v = sla.eigsh(hamiltonian_sparse(N, h1, h2, J), k=1, which='SA', v0=v0)
    return float(e[0]), v[:, 0] / np.linalg.norm(v[:, 0])


def hamiltonian_mpo(N, h1, h2, J=1.0):
    """Eq. (2) as a bond-dimension-5 MPO (quimb, 'lrud' ordering)."""
    import quimb.tensor as qtn
    W = np.zeros((5, 5, 2, 2))
    W[0, 0] = W[4, 4] = I2
    W[4, 2], W[2, 1], W[1, 0] = Z, X, -J * Z      # -J Z X Z
    W[4, 3], W[3, 0] = X, -h2 * X                 # -h2 X X
    W[4, 0] = -h1 * X                             # -h1 X
    return qtn.MatrixProductOperator([W[4]] + [W] * (N - 2) + [W[:, 0]], shape='lrud')


def dmrg_ground_state(N, h1, h2, J=1.0, bond_dim=64):
    """(energy, quimb MPS) by two-site DMRG."""
    import quimb.tensor as qtn
    d = qtn.DMRG2(hamiltonian_mpo(N, h1, h2, J), bond_dims=[16, 32, bond_dim], cutoffs=1e-10)
    d.solve(tol=1e-8, max_sweeps=20, verbosity=0)
    return float(np.real(d.energy)), d.state


def mps_expectation(psi, ops):
    """<psi| prod_s P_s |psi> / <psi|psi> for ops = {site: 'X'|'Y'|'Z'}."""
    phi = psi.copy()
    for s, p in ops.items():
        phi.gate_(PAULI[p], s, contract=True)
    return float(np.real(psi.H @ phi) / np.real(psi.H @ psi))


def sop_ops(N, length):
    """Centred string order parameter Eq. (3): Z_a X_{a+1} X_{a+3} ... X_{b-1} Z_b, b - a = length (even)."""
    if length % 2 or length < 2 or length >= N:
        raise ValueError(f'SOP length must be even, >= 2 and < N; got {length}')
    a = (N - 1 - length) // 2
    b = a + length
    ops = {j: 'X' for j in range(a + 1, b, 2)}
    ops[a] = ops[b] = 'Z'
    return ops


def sample_complexity(expval, p0=0.5):
    """Eq. (4), with p = (<O> + 1)/2 (the printed sqrt(arcsin p0) is a typo for arcsin sqrt(p0)).
    NaN where p <= p0, i.e. where the state is not detected as SPT."""
    p = (np.asarray(expval, dtype=float) + 1) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1.96 ** 2 / (np.arcsin(np.sqrt(np.clip(p, 0, 1))) - np.arcsin(np.sqrt(p0))) ** 2
    return np.where(p > p0, m, np.nan)
