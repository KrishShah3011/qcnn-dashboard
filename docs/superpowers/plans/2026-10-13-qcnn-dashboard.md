# QCNN Reproduction + Interactive Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the computational results of Cong, Choi & Lukin, *Quantum Convolutional Neural Networks* (arXiv:1810.03787), and present them in a static, interactive, attractive dashboard.

**Architecture:** A small Python package `qcnn/` holds the science: the Hamiltonian and ground states, the exact QCNN on statevectors and on MPS, the trained QCNN, and the QEC optimisation. `build_data.py` precomputes everything into `dashboard/data/*.js`, where each file is a JS assignment so the page works from `file://`. `dashboard/` is plain HTML/CSS/JS with Plotly and KaTeX from a CDN, with one JS file per section.

**Tech Stack:** Python 3.12 in `.venv` (numpy 2.0, scipy 1.18, quimb 1.15 for DMRG/MPS circuits, torch 2.14 CPU, pytest 8). Vanilla JS, Plotly.js 2.35, KaTeX 0.16.

**Spec:** `docs/superpowers/specs/2026-10-13-qcnn-dashboard-design.md`

## Global Constraints

- **Python interpreter:** always `C:/Users/Krish/Python/QC/.venv/Scripts/python`. Never `pip install` into the global Python and never change installed package versions. Global numpy is pinned at 1.26.4 for other projects.
- **Shell:** Git Bash on Windows. Use forward slashes. Run commands from the project root `C:/Users/Krish/Python/QC`.
- **Tests:** run with `.venv/Scripts/python -m pytest tests -q`. `python -m` puts the project root on `sys.path`, so `import qcnn` works.
- **No git:** the project is not a git repository. Skip every commit step and do not run `git init`.
- **Qubit ordering:** in statevectors, qubit 0 is the most significant bit (`np.kron` order).
- **Exact-QCNN geometry:** block centres `c = 3k+1` survive each level. Valid `(N, depth)` means `N = 3^depth · m` with `m ≥ 3`. Anything else raises `ValueError`.
- **Data file format:** `dashboard/data/<key>.js` contains exactly `window.QCNN = window.QCNN || {};\nQCNN.<key> = <json>;\n`. NaN and inf become `null`.
- **Colour code** (taken from the paper's figures, use these everywhere):

  | Role | Colour |
  |---|---|
  | convolution | `#2f7fd0` |
  | pooling | `#f08a24` |
  | fully connected | `#7a3fb0` |
  | error | `#d1342c` |
  | SPT / positive | `#e9c46a` |
  | trivial | `#2a9d8f` |

- **Comments:** short. Docstrings name the paper equation or figure they implement.
- **Windows multiprocessing:** every pool lives under `if __name__ == '__main__':`. `build_data.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` **before** importing numpy.

## Review Focus

1. **Degenerate SPT ground states.** With open boundaries the SPT ground state is 4-fold quasi-degenerate, so Lanczos and DMRG can return different states on different runs. Expected: results are reproducible. `ed_ground_state` uses a fixed `v0`. Pinned by `test_ed_reproducible` in Task 1.
2. **Invalid `(N, depth)`.** Expected: a clear `ValueError`, never a silent wrong circuit. Pinned by `test_bad_depth` in Task 1.
3. **Sample complexity outside the SPT phase (p ≤ 0.5).** Eq. 4 would return a meaningless finite number there. Expected: NaN in Python, `null` in JSON, and a gap in the plot. Pinned by `test_sample_complexity` (Task 1) and `test_to_js_nan` (Task 4).
4. **Data not generated yet.** The long stages may still be running when someone opens the page. Expected: the affected section shows a styled "run `build_data.py --stage X`" notice and every other section still works. Pinned by the missing-data step in Task 5 and checked in Task 10.
5. **One section's JS throws.** Expected: that section shows an error card and the rest of the page renders. Pinned by the `QCNNUI.register` try/catch in Task 5.

---

# PHASE 1: Science core (Tasks 1 and 2 run in parallel)

### Task 1: Hamiltonian, ground states, exact QCNN (`qcnn/physics.py`, `qcnn/circuits.py`)

**Files:**
- Create: `qcnn/__init__.py` (empty)
- Create: `qcnn/physics.py`
- Create: `qcnn/circuits.py`
- Test: `tests/test_physics.py`, `tests/test_circuits.py`

**Interfaces — Produces:**
- `physics.PAULI: dict[str, ndarray]`, with keys `'I','X','Y','Z'` mapping to 2×2 arrays
- `physics.hamiltonian_sparse(N, h1, h2, J=1.0) -> scipy.sparse.csr_matrix`
- `physics.ed_ground_state(N, h1, h2, J=1.0) -> (float energy, ndarray real vector)`
- `physics.hamiltonian_mpo(N, h1, h2, J=1.0) -> quimb MatrixProductOperator`
- `physics.dmrg_ground_state(N, h1, h2, J=1.0, bond_dim=64) -> (float energy, quimb MatrixProductState)`
- `physics.mps_expectation(psi, ops: dict[int,str]) -> float`
- `physics.sop_ops(N, length) -> dict[int,str]`
- `physics.sample_complexity(expval) -> ndarray`, NaN where p ≤ 0.5
- `circuits.GATES: dict` with keys `'cz','cx_z','ccz_x'`
- `circuits.check_depth(N, depth)`
- `circuits.unit_layers(sites) -> (list[(label, kind, list[(gate, qubits)])], coarse)`
- `circuits.apply_gate(psi, U, qubits, N)`
- `circuits.sv_expectation(psi, ops, N)`
- `circuits.run_statevector(psi, N, depth) -> dict`
- `circuits.run_mps(psi_mps, N, depth, max_bond=128, cutoff=1e-10) -> dict`

  Both `run_*` functions return a dict with these keys:
  - `outputs`: list of floats, ⟨ZXZ⟩ after levels 1..d
  - `syndromes`: list of lists, ⟨X_f⟩ per level
  - `fine`: list of lists of site indices
  - `live`: list of lists of site indices
  - `max_bond`: int (MPS only)
- `circuits.syndrome_density(syndromes) -> list[float]`, mean of (1−⟨X⟩)/2 per level
- `circuits.pauli_decompose(O, N) -> dict[str, complex]`
- `circuits.unit_unitary(N) -> ndarray`
- `circuits.pull_back(terms) -> terms`
- `circuits.multiscale_sop(depth, centre=None) -> list[(complex, dict[int,str])]`
- `circuits.gate_count(N, depth) -> int`
- `circuits.paper_gate_count(N, depth) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_physics.py`:
```python
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
```

`tests/test_circuits.py`:
```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: collection errors such as `ModuleNotFoundError: No module named 'qcnn'`.

- [ ] **Step 3: Write `qcnn/physics.py`**

```python
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
```

- [ ] **Step 4: Write `qcnn/circuits.py`**

```python
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
```

Two notes on this code:
- `f, out[s] = _PM[...]` unpacks the tuple into `f` and `out[s]`. It is valid Python.
- In `sv_expectation` the coefficients from `multiscale_sop` can be complex with tiny imaginary parts. The test takes the difference against a real number, so if `abs()` of a complex value complains, wrap it: `abs(direct - np.real(via))`.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: all tests pass. `test_depth_two_fixed_point_mps` takes about 10–30 s.

If `test_mps_matches_statevector` fails because quimb's `CircuitMPS(psi0=...)` changes the norm, normalise `psi` before passing it in. Do not loosen the tolerance.

If `test_heisenberg_rules_match_dense` fails on sign alone, print `dense` next to `sym` and fix `_rule` so it matches the dense result. The dense matrix is ground truth.

- [ ] **Step 6: Commit** — skipped (no git, see Global Constraints).

---

### Task 2: QEC optimisation (`qcnn/qec.py`)

**Files:**
- Create: `qcnn/qec.py` (and an empty `qcnn/__init__.py` if Task 1 has not created it yet)
- Test: `tests/test_qec.py`

**Interfaces — Produces:**
- `CDT = torch.complex128`
- `gellmann(d) -> Tensor (d²-1, d, d)`
- `unitary(c: Tensor) -> Tensor (d, d)`, d inferred from `len(c) = d²-1`. Task 3 imports both of these.
- `FIXED_CODES: dict[str, (U1, U2)]`, with keys `identity, shor, phase_phase, bit_bit, phase_bit`. U1 is the inner decoder of each block; U2 is the outer decoder.
- `logical_error(U1, U2, p, pxx) -> Tensor scalar`. `p = (px, py, pz)` as a tuple or tensor; the result is 1.5(1 − f_q).
- `bloch_map(U1, p, pxx) -> (M: Tensor 3x3, c: Tensor 3)`
- `c1_cost(U1, p, pxx) -> Tensor`
- `bold_driver(loss_fn, c0, iters, eta0) -> (c, history list[float])`
- `train_code(p, pxx, seeds=4, iters1=1500, iters2=1500, init_scale=2*pi) -> dict` (paper init U[0,2π); the quick test passes init_scale=1.0), with keys `c1, c2` (Tensors), `logical` (float), `history1`, `history2` (lists)

- [ ] **Step 1: Write the failing test** `tests/test_qec.py`

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_qec.py -q`
Expected: `ModuleNotFoundError: No module named 'qcnn.qec'`.

- [ ] **Step 3: Write `qcnn/qec.py`**

```python
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


def train_code(p, pxx, seeds=4, iters1=600, iters2=800):
    """Sequential optimisation (Methods): U1 minimises C1, then U2 minimises the logical error."""
    best1 = None
    for s in range(seeds):
        c0 = torch.rand(63, generator=torch.Generator().manual_seed(s)) * 2 * np.pi
        c, h = bold_driver(lambda c: c1_cost(unitary(c), p, pxx), c0, iters1)
        if best1 is None or h[-1] < best1[1][-1]:
            best1 = (c, h)
    U1 = unitary(best1[0])
    best2 = None
    for s in range(seeds):
        c0 = torch.rand(63, generator=torch.Generator().manual_seed(100 + s)) * 2 * np.pi
        c, h = bold_driver(lambda c: logical_error(U1, unitary(c), p, pxx), c0, iters2)
        if best2 is None or h[-1] < best2[1][-1]:
            best2 = (c, h)
    return {'c1': best1[0], 'c2': best2[0], 'logical': best2[1][-1],
            'history1': best1[1], 'history2': best2[1]}
```

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/Scripts/python -m pytest tests/test_qec.py -q`
Expected: 6 passed, with the training test taking about 30 s.

Two fixes are allowed if a test fails:
- If `test_shor_correlated_matches_reference` fails, check the qubit and pair conventions against `_noise`. Pairs (0,1) and (1,2) are inside a block.
- If `test_training_beats_shor_quickly` fails, raise `iters2` in the test to 600.

Do not change any reference number.

- [ ] **Step 5: Commit** — skipped (no git).

---

# PHASE 2: Learning + data build (Tasks 3 and 4 run in parallel; Task 4 runs the build at the end)

### Task 3: Trained QCNN (`qcnn/learn.py`)

**Files:**
- Create: `qcnn/learn.py`
- Test: `tests/test_learn.py`

**Interfaces:**
- **Consumes:** `qcnn.qec.unitary`, `qcnn.qec.CDT`, `qcnn.physics.ed_ground_state`
- **Produces:**
  - `N = 15`
  - `PARAM_LAYOUT = [('C1', 90), ('C2', 63), ('C3', 63), ('C4', 63), ('V1', 3), ('V2', 3), ('FC', 1023)]`, total 1308
  - `forward(theta: Tensor(1308), states: Tensor(B, 2**15) complex128) -> Tensor(B)`, values in [0, 1]
  - `mse(theta, states, y) -> Tensor`, Eq. 1
  - `init_params(seed) -> Tensor`
  - `train(states, y, seed=0, iters=800, lr=0.01, snapshot_iters=(0, 10, 25, 50, 100, 200, 400), on_snapshot=None) -> dict` with keys `theta`, `history` (a dict with `iter` and `mse` lists) and `snapshot_iters`. `on_snapshot(it, theta)` is called at each snapshot iteration and once more at the final iteration.
  - `predict(theta, states, chunk=64) -> ndarray`, computed without grad and in chunks

The circuit follows Fig. 9 with n = 3, N = 15, d = 1, using these qubit indices (all must be < 15):
- **C1:** U1 on (3k, 3k+1, 3k+2, 3k+3) for k = 0, 2, then k = 1, 3, in two time slices. U1 = U(23)U(24)U(13)U(14)U(12)U(34): six two-qubit Gell-Mann unitaries with 15 parameters each. U(34) is applied first and U(23) last. Indices are 1-based inside the block.
- **C2:** U2 on (3k+2, 3k+3, 3k+4) for k = 0..3.
- **C3:** U3 on (3k, 3k+1, 3k+2) for k = 0..4.
- **C4:** U4 on (3k+1, 3k+2, 3k+3) for k = 0..3.
- **Pooling:** the survivors are r = 3k. Qubit r+1 is measured in the X basis and controls V1 on r. Qubit r+2 is measured in X and controls V2 on r+3 when r+3 < 15. Deferred-measurement gate: `(H⊗I)(|0⟩⟨0|⊗I + |1⟩⟨1|⊗V)(H⊗I)` on (control, target).
- **FC:** F, a 32×32 unitary with 1023 parameters, on (0, 3, 6, 9, 12).
- **Output:** f = (1 + ⟨X_6⟩)/2.

There are 1308 parameters. The paper quotes 1309; the difference is noted in the dashboard.

**Optimiser decision:** the paper uses finite differences with the bold driver. A design-time spike measured MSE 0.124 → 0.079 in 250 iterations with bold driver, and → 0.040 with Adam (lr = 0.01). Use autograd + Adam; the dashboard notes this deviation. Initialisation c ~ U[0, 2π] follows the paper.

- [ ] **Step 1: Write the failing test** `tests/test_learn.py`

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_learn.py -q`
Expected: `ModuleNotFoundError: No module named 'qcnn.learn'`.

- [ ] **Step 3: Write `qcnn/learn.py`**

```python
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
```

Note that `snaps == [0, 10, 15]` holds because `on_snapshot` fires at 0 and 10 (both in `snapshot_iters`) and at 15 (`it == iters`).

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.venv/Scripts/python -m pytest tests/test_learn.py -q`
Expected: 4 passed in under about 60 s.

- [ ] **Step 5: Commit** — skipped (no git).

---

### Task 4: `build_data.py`, the precompute pipeline

**Files:**
- Create: `build_data.py` (project root)
- Create: `cache/` directory, created at runtime with `os.makedirs(..., exist_ok=True)`
- Output: `dashboard/data/*.js`
- Test: `tests/test_build.py`

**Interfaces:**
- **Consumes:** everything listed above from `qcnn.physics`, `qcnn.circuits`, `qcnn.qec`, and from `qcnn.learn`: `forward`, `train`, `predict`, `PARAM_LAYOUT`, `N`.
- **Produces** these helpers plus the data contract below:
  - `to_js(key, obj, out_dir='dashboard/data')`
  - `clean(obj)`, which recursively converts numpy/torch types to Python and NaN/inf to `None`
  - `find_boundaries(h1, h2, e) -> {'upper': [[h1,h2],...], 'lower': [...]}`

**CLI:** `.venv/Scripts/python build_data.py [--stage NAME ...] [--quick] [--workers K]`.
- With no `--stage`, all stages run in this order: `circuit_demo, heisenberg, resources, qec, phase_ed, trained, phase_mps, cut`.
- `--quick` shrinks every grid and iteration count so the full build runs in under about 3 minutes. It writes `quick: true` into every data object; real runs write `quick: false`.
- Each stage prints its elapsed time. The `meta` key records the per-stage seconds and is written after every stage.

#### Data contract

The dashboard depends on these exact keys. Every 2D grid is indexed `[i_h2][i_h1]`: rows are h₂ (the y-axis) and columns are h₁ (the x-axis).

```
QCNN.meta = {generated: "<ISO time>", quick: bool, stage_seconds: {stage: float}}

QCNN.circuit_demo = {N: 9, live: [1,4,7], fine: [0,2,3,5,6,8], fc: [1,4,7], quick,
  layers: [{label, kind, gates: [{g: "cz"|"ccz_x"|"cx_z", q: [ints]}]}],          // unit_layers(range(9))
  thetas: [9 values linspace(0, pi/2)],
  runs: [{site: null|0..8, theta: [...],                                          // site null -> theta [0] only
          x: [theta][stage 0..4][qubit 0..8],                                     // <X_i> after each layer
          k: [theta][stage 0..4][i-1 for i=1..7],                                 // <Z_{i-1} X_i Z_{i+1}>
          output: [theta], syndromes: [theta][fine j]}]}
  // stage 0 = input (cluster state with exp(-i theta X_site) applied), stages 1..4 after each layer

QCNN.heisenberg = {quick,
  rules: {X: [{c: float, ops: [[P, offset]]}], Y: [...], Z: [...]},     // offsets relative to centre c
  paper: {eq8: "...", eq9: "..."},                                       // LaTeX strings, copied verbatim below
  sop_d1: [{c: float, ops: [[P, offset]]}],                              // 16 terms, offsets relative to child of the X site
  term_counts: [{depth, terms, max_weight, seconds}]}                    // depth 1, 2 (3 only if < 120 s)

QCNN.resources = {quick, rows: [{N, d, ours, paper}]}                    // N in [27, 81, 243], all valid d

QCNN.qec = {quick, codes: ["identity","shor","phase_phase","bit_bit","phase_bit","learned"],
  correlated: {train: {px: 5.8e-3, py: 2e-3, pz: 2e-3, pxx: 2e-4}, totals: [10],
               logical: {code: [10]}, history1: [...], history2: [...]},
  anisotropic: {ptot: 1e-3, ratio: [11], shor: [11], learned: [11], best_fixed: [11], best_fixed_name: [11]},
  isotropic: {ptot: 1e-3, shor: float, learned: float},
  channel: {random: {M: 3x3, c: [3], sv: [3]}, learned: {...}, shor: {...}}}

QCNN.phase_ed = {quick, N: 15, h1: [17], h2: [33], energy: [h2][h1], qcnn: {"1": [h2][h1]}}

QCNN.trained = {quick, N: 15, params: {C1: 90, C2: 63, C3: 63, C4: 63, V1: 3, V2: 3, FC: 1023, total: 1308},
  optimizer: "adam(lr=0.01), autograd", train: {h1: [40], y: [40]},
  history: {iter: [], mse: []}, h1: [17], h2: [33],
  snapshots: [{iter, grid: [h2][h1]}], final: {grid: [h2][h1], train_pred: [40]}}

QCNN.phase_mps = {quick, N: 45, dmrg_bond: 64, circuit_bond: 64, h1: [17], h2: [33],
  energy_density: [h2][h1], qcnn: {"1": [h2][h1], "2": [h2][h1]},
  syndrome_density: {"1": [h2][h1], "2": [h2][h1]}, boundary: {upper: [[h1,h2]], lower: [[h1,h2]]}}

QCNN.cut = {quick, N: 81, h1: 0.5, dmrg_bond: 100, circuit_bond: 128, h2: [sorted unique],
  energy: [...], qcnn: {"1": [...], "2": [...], "3": [...]}, syndrome_density: {"1","2","3"},
  sop: {"40": [...], "28": [...], "16": [...], "14": [...]},
  sample_complexity: {qcnn: {"1","2","3"}, sop: {"40","28","16","14"}},   // null where p <= 1/2 (Eq. 4 undefined)
  inset: {h2: float, depth: [1,2,3], ratio: [3], fit: {a, b}},          // ratio = min_SOP M / M_QCNN(d), fit a*exp(b d)
  critical_h2_paper: 0.423}
```

Grids:
- **Full runs:** h₁ = `np.round(np.arange(0, 1.6001, 0.1), 2)` (17 values) and h₂ = `np.round(np.arange(-1.6, 1.6001, 0.1), 2)` (33 values).
- **Cut:** h₂ is the union of `np.arange(-1.6, 1.6001, 0.1)` and `np.arange(0.30, 0.4601, 0.01)`, rounded to 3 decimals and deduplicated.
- **Quick runs:** h₁ = [0, 0.8, 1.6]; h₂ = [-1.6, -0.8, 0, 0.8, 1.6]; cut N = 27 and depths 1..2 (store the keys `"1"` and `"2"` only); phase_mps N = 27; trained N = 15 with 4 training points and 5 iterations; QEC with seeds = 1, iters1 = 20, iters2 = 20 and 3 anisotropy ratios.

#### Stage recipes

These are complete enough to implement directly. Use `multiprocessing.Pool(workers)` with `imap` over grid points for `phase_ed`, `phase_mps`, `cut` and the anisotropic QEC sweep. Worker functions must be module-level. The default is `workers = max(1, os.cpu_count() - 2)`.

1. **`circuit_demo`.**
   - Build the cluster state with `GATES['cz']` on all neighbouring pairs of `|+>^9`, as in `tests/test_circuits.py::cluster`.
   - For each site in `[None] + list(range(9))` and each θ (only θ = 0 when the site is `None`), apply `expm(-1j*theta*X)` on that site. In closed form this is `np.cos(theta)*I - 1j*np.sin(theta)*X`.
   - Record stage 0. Then apply `unit_layers(list(range(9)))` layer by layer, recording `x` and `k` after each layer.
   - Record `output = sv_expectation(psi, {1:'Z',4:'X',7:'Z'}, 9)` and the syndromes ⟨X_f⟩ at the end.

2. **`heisenberg`.**
   - `rules`: run `pull_back([(1.0, {1: P})])` for P in XYZ, then convert each term's sites to offsets `site - 4`.
   - `sop_d1 = multiscale_sop(1, centre=2)`, with offsets relative to site 7.
   - `term_counts`: for d = 1, 2 (and 3 if d = 2 took under 10 s), time `multiscale_sop(d)` and record `len(terms)` and `max(len(ops))`. Coefficients become floats (real part).
   - The `paper` strings must be exactly:
     - `eq8 = "U_{CP}^\\dagger X_i U_{CP} = X_{\\tilde i-2} X_{\\tilde i} X_{\\tilde i+2}"`
     - `eq9 = "U_{CP}^\\dagger Z_i U_{CP} = \\tfrac12\\left(Z_{\\tilde i} - Z_{\\tilde i-2}X_{\\tilde i-1} - X_{\\tilde i+1}Z_{\\tilde i+2} - Z_{\\tilde i-2}X_{\\tilde i-1}Z_{\\tilde i}X_{\\tilde i+1}Z_{\\tilde i+2}\\right)"`

3. **`resources`.** For N in (27, 81, 243) and every d with `check_depth` passing, record a row `{N, d, ours: gate_count(N,d), paper: paper_gate_count(N,d)}`.

4. **`qec`.**
   - **Correlated.** Train with `train_code(p=(5.8e-3,2e-3,2e-3), pxx=2e-4)`. Evaluate every code over `totals = [1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1]`, scaling p and pxx by `T / 0.01`. Learned uses `unitary(c1), unitary(c2)`.
   - **Anisotropic.** Use ptot = 1e-3, `ratio = sorted(set(np.round(np.linspace(0.05, 0.95, 10), 3)) | {round(1/3, 3)})`, and p = (r·ptot, (1−r)·ptot/2, (1−r)·ptot/2). Train one code per ratio in the pool. Record Shor, learned, and the best fixed code other than identity, with its name.
   - **Isotropic.** ptot = 1e-3, Shor vs learned (one training).
   - **Channel.** Use `bloch_map` at the correlated training point for three U1 choices: random U1 = `unitary(torch.rand(63, generator=seed 0)*2π)`, the learned U1, and `U_BIT` (Shor's inner code). Also record `sv = svdvals(M)`.

5. **`phase_ed`.**
   - In the pool, for each (h1, h2) compute `ed_ground_state(15, h1, h2)` and then `run_statevector(v, 15, 1)['outputs'][0]`.
   - Also save every ground-state vector as float32 in `cache/ed15.npz`, with keys `h1`, `h2` and `states` of shape `(33, 17, 32768)`, for the `trained` stage. Workers return their vectors; the parent writes the file.

6. **`trained`.**
   - Training set: h₁ = `np.linspace(0, 2, 40)`, h₂ = 0, ground states from `ed_ground_state(15, h1, 0.0)`, and y = 1 if h₁ < 1, else 0.
   - Call `torch.set_num_threads(os.cpu_count())` before training.
   - Run `train(..., iters=800, on_snapshot=...)`. The callback runs `predict` on all 561 grid states loaded from `cache/ed15.npz` (or recomputed if the file is missing) and appends `{iter, grid}`.
   - `final.grid` is the last snapshot.

7. **`phase_mps`.**
   - In the pool: `dmrg_ground_state(45, h1, h2, bond_dim=64)` then `run_mps(psi, 45, 2, max_bond=64)`.
   - Store `energy/45`, `outputs`, and `syndrome_density(r['syndromes'])`.
   - `boundary = find_boundaries(h1, h2, energy_density)`.

8. **`cut`.**
   - In the pool, for each h₂: `dmrg_ground_state(81, 0.5, h2, bond_dim=100)` then `run_mps(psi, 81, 3, max_bond=128)`.
   - SOP lengths come from `L(k) = 2*round(81/(2*k))` for k in (2, 3, 5, 6), which gives 40, 28, 16, 14. Use `mps_expectation(psi, sop_ops(81, L))` on the **DMRG state**, not the circuit state.
   - Sample complexity uses `sample_complexity()`.
   - Inset: choose h₂* as the grid value closest to 0.39. Compute `ratio_d = nanmin_over_SOP(M_sop(h₂*)) / M_qcnn_d(h₂*)`, then fit `log ratio = log a + b d` with `np.polyfit` on finite values.
   - Submit points to the pool sorted by `abs(h2 - 0.42)` ascending so the slowest points start first.
   - This stage takes roughly 20–40 minutes, so print progress after each completed point.

`find_boundaries(h1, h2, e)`:
- For each column j, take `d = -(e[i+1,j] - 2 e[i,j] + e[i-1,j]) / dh^2` over the interior rows.
- Find the local maxima, where `d[i] > d[i-1]` and `d[i] >= d[i+1]`, keep the two largest by value, and refine each with a parabola through the three points.
- Label the one with the larger h₂ as `upper` and the smaller as `lower`. With only one peak, label it by h₂ > −0.9 → `upper`, otherwise `lower`.

- [ ] **Step 1: Write the failing test** `tests/test_build.py`

```python
import json
import numpy as np
from build_data import to_js, clean, find_boundaries


def test_to_js_nan(tmp_path):
    to_js('demo', {'a': np.array([1.0, np.nan, np.inf]), 'b': np.float32(2.5)}, out_dir=str(tmp_path))
    txt = (tmp_path / 'demo.js').read_text(encoding='utf-8')
    assert txt.startswith('window.QCNN = window.QCNN || {};\nQCNN.demo = ')
    obj = json.loads(txt.split('QCNN.demo = ', 1)[1].rstrip().rstrip(';'))
    assert obj == {'a': [1.0, None, None], 'b': 2.5}


def test_find_boundaries_synthetic():
    h1 = np.array([0.0, 0.5])
    h2 = np.round(np.arange(-1.6, 1.6001, 0.1), 2)
    e = np.stack([-np.sqrt((h2 - 0.4) ** 2 + 0.01) - np.sqrt((h2 + 1.2) ** 2 + 0.01)] * 2, axis=1)
    b = find_boundaries(h1, h2, e)
    assert len(b['upper']) == 2 and abs(b['upper'][0][1] - 0.4) < 0.03
    assert abs(b['lower'][0][1] + 1.2) < 0.03
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_build.py -q`
Expected: `ModuleNotFoundError: No module named 'build_data'`.

- [ ] **Step 3: Write `build_data.py`**

Implement the stage recipes above. The top of the file and the helpers must be exactly:

```python
"""Precompute every figure of the dashboard: .venv/Scripts/python build_data.py [--stage S] [--quick]."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, json, math, time
from datetime import datetime
from multiprocessing import Pool
import numpy as np

OUT = 'dashboard/data'


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, 'detach'):
        o = o.detach().cpu().numpy()
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, complex):
        return clean(o.real)
    return o


def to_js(key, obj, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{key}.js'), 'w', encoding='utf-8') as f:
        f.write('window.QCNN = window.QCNN || {};\nQCNN.%s = %s;\n' % (key, json.dumps(clean(obj), separators=(',', ':'))))
```

Import `qcnn.*` modules inside stage functions or after the env setup, never before it. Then write `find_boundaries`, one function per stage named `stage_<name>(quick, workers) -> dict`, and a `main()` that parses arguments, runs the stages, writes `to_js(name, result)`, and updates and writes `meta`.

- [ ] **Step 4: Run the unit tests and a quick build**

Run: `.venv/Scripts/python -m pytest tests -q`
Expected: every test across Tasks 1–4 passes.

Run: `.venv/Scripts/python build_data.py --quick`
Expected: 9 files in `dashboard/data/` (`meta` plus the 8 stages), with no exceptions, finishing in under about 5 minutes.

Then check that every file parses:

`.venv/Scripts/python -c "import json,glob; [json.loads(open(f,encoding='utf-8').read().split(' = ',1)[1].rstrip().rstrip(';')) for f in glob.glob('dashboard/data/*.js')]; print('ok')"`

- [ ] **Step 5: Commit** — skipped (no git).

The orchestrator then launches the full build in the background. Fast stages run first, then `phase_mps`, then `cut`.

---

# PHASE 3: Dashboard (Task 5 first, then Tasks 6–9 in parallel)

## Shared design brief (all dashboard tasks)

**Audience and voice.** The page is a guided lecture by a quantum-computing professor. Write in the first-person plural ("we"). It should be rigorous and warm. Every section opens with a short paragraph of 2–4 sentences that says *why* the section matters, and puts the key equation in KaTeX.

**Aesthetic: "lab notebook at night".**
- Background `#0e1116` with a faint grid-paper texture (CSS `background-image` of two 1px `repeating-linear-gradient`s at about 4% opacity, 24px pitch).
- Ink `#e8e4da` (warm paper white), muted text `#9aa3ad`, thin rules `#2a2f37`.
- Display headings in **Fraunces** (Google Fonts, opsz 9..144, weights 400/600). Body in **Inter** 400/500. Numbers, axes and code in **JetBrains Mono**.
- Section numbers read "§3", set large in Fraunces italic and muted.
- Professor's asides are margin notes: italic Fraunces at 0.9em, with a left rule in the pooling orange. They float right on screens ≥ 1280px and sit inline below that.
- Use the colour code from Global Constraints. Interactive controls are custom-styled `<input type=range>` (thin track, round thumb in conv blue) and pill toggle buttons.
- No gradient blobs, no glassmorphism, no emoji. Plotly charts are transparent, use JetBrains Mono for tick labels and the warm-ink colours, and show no Plotly logo (`displaylogo: false`, `responsive: true`).
- Layout: a fixed left rail (220px) with the section list and a reading-progress bar, and main content max-width 1100px. Below 900px the rail becomes a top bar.

**Libraries** (CDN, `defer`):
- `https://cdn.plot.ly/plotly-2.35.2.min.js`
- `https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css`
- `https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js`
- `https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js`

**Section mount contract.** `index.html` holds one `<section id="sec-<id>" class="sec"></section>` per section, in order:
`hero, idea, physics, circuit, multiscale, phase, depth, samples, rgflow, learning, qec, hardware, verdict`.

Each section file calls `QCNNUI.register('<id>', (el, D) => { ... })`, where `el` is the section element and `D = window.QCNN`. `common.js` runs every registered renderer after `DOMContentLoaded` inside try/catch and draws an error card on failure. Renderers build all their own DOM, prose included, inside `el`.

---

### Task 5: Page shell, design system, `common.js`, intro sections

**Files:**
- Create: `dashboard/index.html`
- Create: `dashboard/styles.css`
- Create: `dashboard/js/common.js`
- Create: `dashboard/js/sec-intro.js` (sections `hero`, `idea`, `physics`)

**Interfaces — Produces** (in `common.js`, global `window.QCNNUI`):
- `register(id, fn)`
- `C`, the colour constants: `{conv, pool, fc, err, spt, triv, ink, muted, rule, bg}`
- `layout(overrides) -> Plotly layout`, with transparent background, mono ticks, ink colours, margins `{l:56, r:16, t:24, b:48}`, and a deep merge of overrides
- `plot(div, traces, layoutOverrides)`, a wrapper around `Plotly.newPlot` with the config `{displaylogo: false, responsive: true}`
- `scale`, the Plotly colorscale `[[0,'#1b3a4b'],[0.5,'#2a9d8f'],[1,'#e9c46a']]` for QCNN outputs in [0, 1]
- `needData(el, key, stage) -> bool`. If `window.QCNN?.[key]` is missing it appends a notice card ("Data for this figure has not been generated yet. Run `.venv/Scripts/python build_data.py --stage <stage>`") and returns false.
- `quickBadge(el, obj)`, which shows a small "preview data (quick build)" tag when `obj.quick`
- `head(el, num, title, kicker)`, which renders the §num heading block and returns a content container
- `note(el, html)`, a margin aside
- `slider(el, {label, min, max, step, value, fmt, onInput})`, which returns the input element and shows a live mono value
- `toggle(el, options, value, onChange)`, a pill group
- `katex(el)`, which runs auto-render with `$...$` and `$$...$$` delimiters
- `fmt(x, digits)`

`index.html` loads the files in this order: CDN libs (defer); `data/*.js` for meta, circuit_demo, heisenberg, resources, qec, phase_ed, trained, phase_mps, cut (a missing file simply fails to load, which is fine); `js/common.js`; `js/sec-intro.js`; `js/sec-circuit.js`; `js/sec-phase.js`; `js/sec-learn.js`; `js/sec-qec.js`. Plain `<script src>`, no modules, so `file://` works. Since KaTeX/Plotly are `defer`, all local scripts must also be `defer` to preserve order. Renderers run on `DOMContentLoaded`.

**Section content for `sec-intro.js`:**

- **hero**
  - Title "Quantum Convolutional Neural Networks". Subtitle "Cong · Choi · Lukin — Nature Physics 15, 1273 (2019) · arXiv:1810.03787".
  - One-paragraph thesis: a quantum circuit with the CNN's structure (translationally invariant convolutions, pooling that shrinks the system, a final fully connected layer) uses only O(log N) parameters, inherits MERA and quantum error correction, and does two things: it recognises a symmetry-protected topological phase with exponentially fewer samples than string order parameters, and it learns QEC codes tailored to the noise.
  - Four stat tiles in mono numerals: "O(log N) parameters", "1308 trained parameters (paper: 1309)", "126 QEC parameters", "d = 3 depth at N = 81 simulated".
  - A "how to read this page" line.

- **idea** (§1, "From CNN to QCNN")
  - An animated inline SVG with three stacked rows: a CNN (input grid → C → P → FC → bars); a QCNN (N qubit wires → blue convolution boxes → orange pooling meters that drop 2 of every 3 wires → purple FC → one meter); and a MERA.
  - A toggle "QCNN ⇄ MERA" animates the arrow direction and labels, flipping the row with a CSS transform and transition. The point it makes: same circuit, opposite direction. MERA builds a state from few qubits; the QCNN runs it backwards and the extra measurements become QEC syndromes.
  - Prose covers convolution = quasi-local unitary U_i, pooling = measure + controlled V_j (this is where nonlinearity comes from), FC = F, and Eq. 1 MSE in KaTeX.

- **physics** (§2, "The phase we want to recognise")
  - Eq. 2 in KaTeX, with the three terms colour-highlighted in spans.
  - An interactive chain of 15 spins drawn as circles. Hovering a site i highlights the stabiliser K_i = Z_{i-1}X_iZ_{i+1} on its three sites.
  - A button "inject X error" lets the reader click a site to flip it. The two neighbouring stabilisers turn error-red. This teaches anticommutation directly, with no data needed.
  - Text explains the Z₂×Z₂ symmetry (X_even, X_odd), the three phases (SPT, paramagnet, antiferromagnet), and that h₂ = 0 is solvable by Jordan–Wigner with the transition at h₁ = J.
  - Margin note: "Open boundaries give the SPT phase a 4-fold quasi-degenerate ground space (edge spins); our QCNN only measures bulk operators, so it does not care."

- [ ] **Step 1:** Write `styles.css` with the design brief's tokens as CSS variables, the layout grid, the rail, `.sec`, `.fig`, `.caption`, `.note`, `.stat`, `.notice`, `.errcard`, `.pill`, the range input styles, and responsive breakpoints at 1280px and 900px.
- [ ] **Step 2:** Write `index.html` (fonts preconnect, the CDN tags, rail nav links to `#sec-*`, the section containers, the script tags) and `js/common.js` implementing every function in the Produces list. Scroll-spy uses `IntersectionObserver` to highlight the current rail link; the progress bar tracks `scrollY / (scrollHeight - innerHeight)`.
- [ ] **Step 3:** Write `js/sec-intro.js` with the three renderers.
- [ ] **Step 4:** Verify in a headless browser.

  Run: `.venv/Scripts/python -c "import pathlib;print(pathlib.Path('dashboard/index.html').resolve().as_uri())"`

  Open that URL with the Playwright MCP tools (`browser_navigate`, `browser_console_messages`, `browser_take_screenshot`). Expected: no console errors from `common.js` or `sec-intro.js`. Sections without a JS file yet stay empty. Screenshots of hero, idea and physics look intentional: fonts loaded, no overlapping text.

  If Playwright is unavailable, say so in your report and check by reading the code.

- [ ] **Step 5: Commit** — skipped (no git).

---

### Task 6: Circuit + multiscale sections (`dashboard/js/sec-circuit.js`)

**Consumes:** `QCNNUI.*`, `QCNN.circuit_demo`, `QCNN.heisenberg`.

**circuit** (§3, "An exact QCNN for the cluster-state phase", Fig. 2b and Fig. 3)
- An SVG circuit for N = 9: horizontal wires labelled q0..q8 with block centres (1, 4, 7) bold. Time runs left to right through the 4 layers of `circuit_demo.layers`, then FC.
- Draw CZ as two dots joined by a line (blue). Draw `ccz_x` as three ⊕-style X boxes, with the centre box solid and the controls outlined, labelled "X". Draw `cx_z` as an orange meter "X" on the control joined to a "Z" box on the target. Draw FC as purple dots plus a meter.
- **Stage slider** (0..4) shows a vertical "now" line after that layer. Every wire gets a coloured bead whose colour encodes ⟨X_i⟩ at that stage on a diverging scale: −1 = error red, 0 = muted, +1 = conv blue.
- Under the circuit, a row of 7 stabiliser chips K_1..K_7 coloured by `k[stage]`.
- **Controls:** an error site selector (pills "none, 0 … 8") and a θ slider ("error rotation exp(−iθX)", 0..π/2).
- A readout card shows the syndromes (fine sites; any < 0 glows orange as "syndrome fired"), the output ⟨Z₁X₄Z₇⟩ in large mono, and a verdict line ("corrected ✓" when |output − 1| < 1e-6 and a syndrome fired).
- Prose:
  - The fixed-point criterion and the QEC criterion (Methods), and why the measured qubits are QEC syndromes.
  - Pooling flips the phase of a neighbouring centre when the X measurement gives −1.
  - The fully connected layer measures Z X Z.
- Margin note: "Edge sites 0 and 8 are boundary modes, not errors that the bulk code can see — try them."

**multiscale** (§4, "What the circuit really measures", Eqs. 8–10, Fig. 8)
- Show `heisenberg.paper.eq8`/`eq9` in KaTeX beside our rules. Render each term as a row of coloured Pauli tiles (X blue, Z orange, Y purple, I muted outline) at offsets −2..+2, with its coefficient in mono.
- A note that our derived signs on the two middle Z-terms are + where the paper prints −. These signs depend on convention: they flip under Z → −Z on the measured sites, X-basis outcome labelling.
- Below, the 16 terms of the d = 1 multiscale string order parameter as tile rows (offsets −8..+8). Under that, a bar chart (log y) of `term_counts`: number of terms vs depth.
- Prose: the QCNN measures not one SOP but a weighted sum of products of exponentially many SOPs, which stays sharp up to the phase boundary. This is the mechanism behind the sample-complexity advantage.

- [ ] **Step 1:** Implement both renderers using `QCNNUI.needData` first (stages `circuit_demo` and `heisenberg`).
- [ ] **Step 2:** Verify in Playwright: no console errors. Moving the slider and choosing site 3 with θ = π/2 shows an orange syndrome and output 1.000. Take screenshots.
- [ ] **Step 3: Commit** — skipped.

---

### Task 7: Phase-recognition results (`dashboard/js/sec-phase.js`)

**Consumes:** `QCNNUI.*`, `QCNN.phase_ed`, `QCNN.phase_mps`, `QCNN.cut`.

**phase** (§5, "The phase diagram, read by a circuit", Fig. 2a)
- A Plotly heatmap with x = h₁/J and y = h₂/J and the `QCNNUI.scale` colorscale, zmin 0 and zmax 1.
- Source toggle: "N=15 exact diag · d=1" (`phase_ed.qcnn["1"]`) / "N=45 MPS · d=1" / "N=45 MPS · d=2" (`phase_mps.qcnn`).
- Overlay `phase_mps.boundary` upper and lower as diamond markers, blue for upper and red for lower as in the paper, and label the regions "SPT", "Paramagnet", "Antiferromagnet".
- Hover shows `h1, h2, output`.
- Beside it, a small heatmap of the energy density.
- Prose: the circuit is fixed (no training). Deep in the SPT phase it outputs about 1, elsewhere about 0. Going from d = 1 to 2 sharpens the boundary.

**depth** (§6, "Deeper is sharper", Fig. 2c)
- Lines of `cut.qcnn[d]` vs h₂ for d = 1..3, coloured from light to dark blue as in the paper.
- A depth slider highlights one curve (thicker) and dims the others.
- A vertical dashed line at `critical_h2_paper` (0.423) labelled "paper iDMRG critical point".
- A shaded band `0.30–0.46` labelled "zoom in §7".

**samples** (§7, "How many copies do we need?", Eq. 4, Fig. 2d)
- Eq. 4 in KaTeX, with the typo note.
- Lines of `sample_complexity.qcnn[d]` (blue family) and `sample_complexity.sop[len]` (red family), restricted to h₂ ∈ [0.30, 0.46] and a y range capped at 100 (clamped, with a "↑" marker when capped). Nulls create gaps.
- Inset (a second small plot): ratio vs depth on a log scale, with markers and the fit line `a·e^{b d}`, annotated with the fitted a and b in mono. Show the paper's `1.73 e^{0.28 d}` as a dashed reference.
- Prose:
  - Measuring an SOP is a coin flip with bias p, and near criticality ⟨S⟩ → 0 so the coin becomes fair.
  - The QCNN's error correction pushes p toward 1, so fewer copies are needed, and the advantage grows exponentially with depth.

**rgflow** (§8, "Watching renormalisation happen")
- A plot of syndrome density (y) vs level 1..3 (x), one line per selected h₂ value from `cut.h2`.
- A multi-select pill row offers about 6 representative h₂ values (−1.2, −0.4, 0.2, 0.38, 0.42, 0.6), using the closest grid value for each.
- A "play" button animates a marker along each line.
- Inside the SPT phase the error density flows toward 0, which is the RG fixed point. Outside, it flows toward a nonzero value.
- Also show a heatmap of `phase_mps.syndrome_density["1"]` vs `["2"]` as a toggle.
- Prose ties this to RG flow and the QEC threshold (Fig. 10c).

Iterate over the keys of `cut.qcnn`, `cut.sop` and `cut.sample_complexity.*` rather than hardcoding depths or lengths. The quick build uses N = 27, which has different SOP lengths and only depths 1–2.

- [ ] **Step 1:** Implement the four renderers, each guarded with `needData`.
- [ ] **Step 2:** Verify in Playwright, with screenshots of all four.
- [ ] **Step 3: Commit** — skipped.

---

### Task 8: Learning section (`dashboard/js/sec-learn.js`)

**Consumes:** `QCNNUI.*`, `QCNN.trained`, `QCNN.phase_ed`.

**learning** (§9, "Learning a QCNN from 40 solvable points", Fig. 4 and Fig. 9)
- **Left:** an SVG of the Fig. 9 parameterised circuit for 15 qubits with C1 (U1 4-qubit boxes in two staggered rows), C2, C3, C4 (3-qubit boxes in the offsets given in Task 3), pooling (V1/V2 with X meters), and FC (one 5-qubit purple box). Hovering a layer shows its parameter count from `trained.params`.
- **Right:** a stacked bar of parameter counts (FC dominates, 1023/1308).
- **Below:** a training-curve plot (MSE vs iteration, log y).
- **Snapshot player:** a slider plus a play button over `trained.snapshots`. It shows the 2D heatmap of predictions with the training points overlaid as small gray dots at h₂ = 0 (h₁ ≤ 1.6 visible), next to the exact QCNN (`phase_ed.qcnn["1"]`) for comparison.
- A line plot of `final.train_pred` vs `train.h1` against the step labels.
- Prose:
  - Training data come only from the Jordan–Wigner solvable line h₂ = 0, which never crosses the lower phase boundary, yet the trained circuit reproduces the 2D diagram.
  - O(log N) parameters means little room to overfit.
  - Optimiser note: the paper uses finite differences + the bold driver; we use autograd + Adam, which gives the same gradient up to O(ε²) and is faster. The paper's outlook itself suggests backpropagation.

- [ ] **Step 1:** Implement with `needData('trained', 'trained')`.
- [ ] **Step 2:** Verify in Playwright: the snapshot player animates, with no errors. Take a screenshot.
- [ ] **Step 3: Commit** — skipped.

---

### Task 9: QEC, hardware and verdict sections (`dashboard/js/sec-qec.js`)

**Consumes:** `QCNNUI.*`, `QCNN.qec`, `QCNN.resources`, plus every key for the verdict.

**qec** (§10, "Letting the circuit design its own error-correcting code", Fig. 5 and Fig. 11, Eqs. 5, 15–17)
- An SVG of the Fig. 5a encoder–noise–decoder (U2⁻¹ → 3×U1⁻¹ → 𝒩 → 3×U1 → U2 → ρ), with U boxes in conv blue and meters in pooling orange.
- **Error-model toggle "correlated / anisotropic / isotropic":**
  - **Correlated:** a log-log plot of logical vs input error for identity (gray), Shor (blue), the best nested repetition code, and learned (orange) from `qec.correlated`. Mark the training point at total 0.01. Below it, the two training-loss curves (`history1` C₁, `history2` logical error).
  - **Anisotropic:** ratio Shor/learned vs p_x/p_tot (Fig. 11), with a horizontal line at 1.
  - **Isotropic:** two big numbers, Shor vs learned.
- **Bloch-ellipsoid view:** a Plotly 3D surface of the image of the unit sphere under r → M r + c for `qec.channel.random`, `learned` and `shor` (toggle). An exaggeration slider (×1..×200) amplifies the deviation from the identity map: r' = r + α(Mr + c − r). The paper's goal is shown visually: the learned U1 makes the effective channel "classical", squeezed along one axis only. Show the singular values in mono.
- **Deviation callout (required, honest):** "With the error model exactly as stated in Methods (p_x = 5.8·10⁻³, p_y = p_z = 2·10⁻³, p_xx = 2·10⁻⁴), our Shor code beats the bare qubit (≈0.13× at 10⁻³), whereas Fig. 5b shows Shor slightly worse than no correction. Correlated XX errors defeat Shor's inner bit-flip blocks at rate ≈ 6p_xx. The learned code still beats Shor and every nested repetition code, which is the paper's claim; the size of the gap differs."
- Prose: encoding = inverse QCNN, decoding = QCNN; there are 126 parameters; training is sequential (C₁ = p₁² + p₂ + p₃ makes the first-layer channel classical, then the fidelity sets U2); f_q can be measured on hardware without knowing 𝒩.

**hardware** (§11, "Could we run it?")
- Sliders for N (9..243 in steps of 3^k·m — offer the valid values `[27, 45, 81, 135, 243]`), d (1..valid max), Ω/2π (10..100 MHz) and τ (50..500 µs).
- Show the paper's formula 7N/2(1−3^{1−d}) + N·3^{1−d}, our exact count from `resources.rows` when (N, d) is present (otherwise "—"), the budget Ωτ/2π operations, and a feasibility bar (count/budget).
- Note: the paper's formula counts an FC layer that applies CZs on all N·3^{1−d} remaining qubits and treats d as including that layer, while ours applies d full conv–pool units plus 2 CZs. Both scale as 7N/2.
- A small table from `resources.rows`.

**verdict** (§12, "Paper vs this reproduction")
- A table with the columns claim | paper | ours | status (✓ reproduced / ≈ qualitatively / ✗ not reproduced). Fill "ours" from data at runtime.

  | Claim | Paper | Ours |
  |---|---|---|
  | Fixed point & QEC criteria | — | from circuit_demo: output at site 3, θ = π/2 |
  | Output sharpens with depth | — | `cut`: max slope of qcnn[d] vs d |
  | QCNN beats SOP sample complexity | — | `cut` inset ratios > 1 |
  | Exponential advantage | 1.73e^{0.28d} | our fit |
  | Trained QCNN recovers 2D diagram | — | trained vs exact grid mean absolute difference |
  | Parameter count | 1309 | 1308 |
  | Learned QEC beats Shor (correlated) | — | ratio at 1e-3 |
  | Shor vs no correction | worse | ratio at 1e-3 |
  | Haldane chain (Fig. 7) | — | ✗ not reproduced (spin-1 DMRG + 108-qubit circuit) |

- Closing paragraph from the professor.

- [ ] **Step 1:** Implement the three renderers with `needData`.
- [ ] **Step 2:** Verify in Playwright, with screenshots.
- [ ] **Step 3: Commit** — skipped.

---

# PHASE 4: Verification (Sonnet subagent)

### Task 10: Independent verification

1. Run `.venv/Scripts/python -m pytest tests -q`. Every test must pass. Report the exact output.
2. Confirm that every `dashboard/data/*.js` exists with `quick: false`, parses as JSON after the prefix, and has the keys listed in the Task 4 data contract. Scan for nulls where the contract does not allow them.
3. Scientific sanity checks, each with a number reported:
   - `cut.qcnn["3"]` at h₂ = 0 is > 0.95, and at h₂ = 1.0 it is < 0.1.
   - The maximum |slope| of `cut.qcnn[d]` grows with d.
   - `cut.inset.ratio` > 1 for every depth.
   - The trained `final.grid` vs `phase_ed.qcnn["1"]` mean absolute difference is < 0.25 and the training MSE is < 0.05.
   - `qec.correlated.logical.learned` < `shor` at totals ≤ 1e-2.
4. Open `dashboard/index.html` from `file://` in Playwright. Collect console messages: zero errors. Screenshot every section and use one interaction per section (slider, toggle, play). Confirm no overlapping text or empty plots.
5. Write `docs/superpowers/verification-report.md` covering pass/fail per item, the numbers, screenshots referenced by path, and defects found, with `file:line`.
