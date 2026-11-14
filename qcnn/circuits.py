"""Exact QCNN of Fig. 2b on statevectors and MPS, its Heisenberg-picture operator (Eqs. 8-10)
and gate counting (Methods, 'Experimental Resource Analysis')."""
import numpy as np
from qcnn.physics import X, Z, PAULI, mps_expectation

H1 = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
_H2 = np.kron(H1, np.eye(2))
_H3 = np.kron(np.kron(H1, H1), H1)
_CZ = np.diag([1., 1., 1., -1.])
GATES = {
    'cz': _CZ,                                          # Eq. (12)
    'cx_z': _H2 @ _CZ @ _H2,                            # Eq. (13): Z on target iff X_control = -1
    'ccz_x': _H3 @ np.diag([1., 1, 1, 1, 1, 1, 1, -1]) @ _H3,   # Eq. (14): -1 on |---> (X-control Toffoli)
}
ZXZ = np.kron(np.kron(Z, X), Z)


def check_depth(N, depth):
    """Each level keeps one site in three and the fully connected layer needs >= 3 sites."""
    L = N
    for _ in range(depth):
        if L % 3 or L // 3 < 3:
            raise ValueError(f'N={N} cannot run depth {depth}: need N = 3^depth * m with m >= 3')
        L //= 3


def unit_layers(sites):
    """One convolution-pooling unit (Fig. 2b) on live `sites`; block centres sites[1::3] survive."""
    co = sites[1::3]
    nn = [('cz', (a, b)) for a, b in zip(sites, sites[1:])]
    cc = [('cz', (a, b)) for a, b in zip(co, co[1:])]
    tof = [('ccz_x', (sites[i - 1], sites[i], sites[i + 1])) for i in range(1, len(sites), 3)]
    pool = []
    for k in range(len(co)):
        if k > 0:
            pool.append(('cx_z', (sites[3 * k], co[k - 1])))       # left fine site -> previous centre
        if k < len(co) - 1:
            pool.append(('cx_z', (sites[3 * k + 2], co[k + 1])))   # right fine site -> next centre
    layers = [('CZ on neighbours', 'conv', nn), ('CZ on block centres', 'conv', cc),
              ('Toffoli, X-basis controls', 'conv', tof), ('X-measure, Z feed-forward', 'pool', pool)]
    return layers, co


def _levels(N, depth):
    check_depth(N, depth)
    sites = list(range(N))
    for level in range(1, depth + 1):
        layers, co = unit_layers(sites)
        yield level, layers, [s for s in sites if s not in co], co
        sites = co


def fc_sites(live):
    """Fully connected layer: CZs then X on the middle qubit == measuring Z X Z on the middle three."""
    m = len(live) // 2
    return live[m - 1], live[m], live[m + 1]


def apply_gate(psi, U, qubits, N):
    k = len(qubits)
    t = np.moveaxis(np.asarray(psi).reshape([2] * N), qubits, range(k))
    t = (U @ t.reshape(2 ** k, -1)).reshape([2] * N)
    return np.moveaxis(t, range(k), qubits).reshape(-1)


def sv_expectation(psi, ops, N):
    """<psi| prod_s P_s |psi> for ops = {site: 'X'|'Y'|'Z'}."""
    phi = psi
    for s, p in ops.items():
        phi = apply_gate(phi, PAULI[p], (s,), N)
    return float(np.real(np.vdot(psi, phi)))


def run_statevector(psi, N, depth):
    """Exact QCNN with deferred pooling measurements. Fine qubits are never touched after their
    level except as X-basis controls, so their <X> (the syndromes) are read at the end."""
    psi = np.asarray(psi, dtype=complex)
    out = {'outputs': [], 'fine': [], 'live': []}
    for _, layers, fine, co in _levels(N, depth):
        for _, _, gates in layers:
            for g, q in gates:
                psi = apply_gate(psi, GATES[g], q, N)
        a, b, c = fc_sites(co)
        out['outputs'].append(sv_expectation(psi, {a: 'Z', b: 'X', c: 'Z'}, N))
        out['fine'].append(fine)
        out['live'].append(co)
    out['syndromes'] = [[sv_expectation(psi, {f: 'X'}, N) for f in fine] for fine in out['fine']]
    return out


def run_mps(psi, N, depth, max_bond=128, cutoff=1e-10):
    """Same as run_statevector on a quimb MPS (Methods: MPS simulation with swaps)."""
    import quimb.tensor as qtn
    circ = qtn.CircuitMPS(psi0=psi.copy(), gate_opts=dict(max_bond=max_bond, cutoff=cutoff))
    out = {'outputs': [], 'fine': [], 'live': []}
    for _, layers, fine, co in _levels(N, depth):
        for _, _, gates in layers:
            for g, q in gates:
                circ.apply_gate_raw(GATES[g].astype(complex), q)
        p = circ.psi
        a, b, c = fc_sites(co)
        val = p.H @ p.gate(ZXZ.reshape([2] * 6), (a, b, c), contract=False)
        out['outputs'].append(float(np.real(val) / np.real(p.H @ p)))
        out['fine'].append(fine)
        out['live'].append(co)
    p = circ.psi
    out['syndromes'] = [[mps_expectation(p, {f: 'X'}) for f in fine] for fine in out['fine']]
    out['max_bond'] = int(p.max_bond())
    return out


def syndrome_density(syndromes):
    """Fraction of pooling measurements reading X = -1, per level (RG-flow diagnostic)."""
    return [float(np.mean((1 - np.asarray(s)) / 2)) for s in syndromes]


# ---------- Heisenberg picture: Eqs. (8)-(10) ----------

def pauli_decompose(O, N, tol=1e-9):
    """Dense 2^N x 2^N operator -> {Pauli string over 'IXYZ': coefficient}."""
    Ps = np.stack([PAULI[c] for c in 'IXYZ']).astype(complex)
    T = O.reshape([2] * (2 * N))
    for q in range(N):
        T = np.moveaxis(T, N - q, 1)
        T = np.einsum('sji,ij...->...s', Ps, T) / 2
    C = T.reshape([4] * N)
    return {''.join('IXYZ'[t] for t in idx): complex(C[idx])
            for idx in zip(*np.nonzero(np.abs(C) > tol))}


def unit_unitary(N):
    """Dense unitary of one convolution-pooling unit (pooling deferred). Small N only."""
    U = np.eye(2 ** N, dtype=complex)
    layers, _ = unit_layers(list(range(N)))
    for _, _, gates in layers:
        for g, q in gates:
            U = np.stack([apply_gate(col, GATES[g], q, N) for col in U.T], axis=1)
    return U


_PM = {('X', 'Y'): (1j, 'Z'), ('Y', 'Z'): (1j, 'X'), ('Z', 'X'): (1j, 'Y'),
       ('Y', 'X'): (-1j, 'Z'), ('Z', 'Y'): (-1j, 'X'), ('X', 'Z'): (-1j, 'Y')}


def _mul(a, b):
    phase, out = 1, dict(a)
    for s, p in b.items():
        if s not in out:
            out[s] = p
        elif out[s] == p:
            del out[s]
        else:
            f, out[s] = _PM[(out[s], p)]
            phase *= f
    return phase, out


def _mul_sums(A, B):
    acc = {}
    for ca, pa in A:
        for cb, pb in B:
            f, p = _mul(pa, pb)
            key = tuple(sorted(p.items()))
            acc[key] = acc.get(key, 0) + ca * cb * f
    return [(c, dict(k)) for k, c in acc.items() if abs(c) > 1e-12]


def _rule(p, c):
    """Image of a Pauli on block centre c (child index) through one unit. Eq. (8) and Eq. (9);
    our circuit gives + signs on the two middle terms of Eq. (9) (the paper prints -)."""
    if p == 'X':
        return [(1, {c - 2: 'X', c: 'X', c + 2: 'X'})]
    if p == 'Z':
        return [(0.5, {c: 'Z'}), (0.5, {c - 2: 'Z', c - 1: 'X'}), (0.5, {c + 1: 'X', c + 2: 'Z'}),
                (-0.5, {c - 2: 'Z', c - 1: 'X', c: 'Z', c + 1: 'X', c + 2: 'Z'})]
    return _mul_sums([(1j, {})], _mul_sums(_rule('X', c), _rule('Z', c)))     # Y = i X Z


def pull_back(terms):
    """Heisenberg image of sum_i c_i P_i through one unit; level-local index k -> child 3k+1."""
    total = {}
    for coef, ops in terms:
        acc = [(coef, {})]
        for k, p in sorted(ops.items()):
            acc = _mul_sums(acc, _rule(p, 3 * k + 1))
        for c, o in acc:
            key = tuple(sorted(o.items()))
            total[key] = total.get(key, 0) + c
    return [(c, dict(k)) for k, c in total.items() if abs(c) > 1e-12]


def multiscale_sop(depth, centre=None):
    """Eq. (10): measured Z X Z pulled back through `depth` units (bulk, boundaries ignored)."""
    m = 2 * 3 ** depth if centre is None else centre
    terms = [(1.0, {m - 1: 'Z', m: 'X', m + 1: 'Z'})]
    for _ in range(depth):
        terms = pull_back(terms)
    return terms


# ---------- resources ----------

def gate_count(N, depth):
    """Multi-qubit gates of our circuit: all conv/pool gates plus the 2 CZs of the fully connected layer."""
    return sum(len(g) for _, layers, _, _ in _levels(N, depth) for _, _, g in layers) + 2


def paper_gate_count(N, depth):
    """Methods: 7N/2 (1 - 3^(1-d)) + N 3^(1-d)."""
    return 7 * N / 2 * (1 - 3 ** (1 - depth)) + N * 3 ** (1 - depth)
