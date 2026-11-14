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


def find_boundaries(h1, h2, e):
    """Phase boundaries as the two largest peaks of -d2E/dh2^2 in each column (e is [h2][h1])."""
    e, h2 = np.asarray(e, float), np.asarray(h2, float)
    dh = h2[1] - h2[0]
    out = {'upper': [], 'lower': []}
    for j, x in enumerate(h1):
        d = -(e[2:, j] - 2 * e[1:-1, j] + e[:-2, j]) / dh ** 2      # d[k] belongs to row k + 1
        pk = [k for k in range(1, len(d) - 1) if d[k] > d[k - 1] and d[k] >= d[k + 1]]
        pts = []
        for k in sorted(pk, key=lambda k: -d[k])[:2]:
            y0, y1, y2 = d[k - 1:k + 2]
            den = y0 - 2 * y1 + y2
            pts.append(float(h2[k + 1] + (0.5 * (y0 - y2) / den if den else 0.0) * dh))
        if len(pts) == 2:
            out['lower'].append([float(x), min(pts)])
            out['upper'].append([float(x), max(pts)])
        elif pts:
            out['upper' if pts[0] > -0.9 else 'lower'].append([float(x), pts[0]])
    return out


# ---------- grids and helpers ----------

def grids(quick):
    if quick:
        return np.array([0, 0.8, 1.6]), np.array([-1.6, -0.8, 0, 0.8, 1.6])
    return np.round(np.arange(0, 1.6001, 0.1), 2), np.round(np.arange(-1.6, 1.6001, 0.1), 2)


def _pmap(fn, items, workers, unordered=False):
    if workers <= 1:
        yield from map(fn, items)
        return
    with Pool(workers) as pool:
        yield from (pool.imap_unordered if unordered else pool.imap)(fn, items)


def _grid_args(h1, h2):
    return [(float(a), float(b)) for b in h2 for a in h1]        # row-major [h2][h1]


def _reshape(vals, h1, h2):
    return np.array(vals, dtype=float).reshape(len(h2), len(h1))


def _terms(terms, centre):
    """[(c, {site: P})] -> [{c, ops: [[P, offset]]}] with offsets relative to `centre`."""
    return [{'c': float(np.real(c)), 'ops': [[p, int(s - centre)] for s, p in sorted(o.items())]} for c, o in terms]


def _cache(quick):      # separate file so a quick build never clobbers the full cache
    return 'cache/ed15_quick.npz' if quick else 'cache/ed15.npz'


# ---------- workers (module level for Windows spawn) ----------

def _ed_point(a):
    from qcnn.physics import ed_ground_state
    from qcnn.circuits import run_statevector
    e, v = ed_ground_state(15, *a)
    return e, run_statevector(v, 15, 1)['outputs'][0], v.astype(np.float32)


def _ed_train(h1):
    from qcnn.physics import ed_ground_state
    return ed_ground_state(15, h1, 0.0)[1]


def _depth_stats(d):
    from qcnn.circuits import multiscale_sop
    t0 = time.time()
    terms = multiscale_sop(d)
    return {'depth': d, 'terms': len(terms), 'max_weight': max(len(o) for _, o in terms), 'seconds': time.time() - t0}


def _qec_train(a):
    from qcnn.qec import train_code
    p, pxx, seeds, i1, i2 = a
    r = train_code(p, pxx, seeds=seeds, iters1=i1, iters2=i2)
    return {'c1': r['c1'].numpy(), 'c2': r['c2'].numpy(), 'logical': r['logical'],
            'history1': r['history1'], 'history2': r['history2']}


def _mps_point(a):
    from qcnn.physics import dmrg_ground_state
    from qcnn.circuits import run_mps, syndrome_density
    N, h1, h2 = a
    e, psi = dmrg_ground_state(N, h1, h2, bond_dim=64)
    r = run_mps(psi, N, 2, max_bond=64)
    return e / N, r['outputs'], syndrome_density(r['syndromes'])


def _cut_point(a):
    from qcnn.physics import dmrg_ground_state, mps_expectation, sop_ops
    from qcnn.circuits import run_mps, syndrome_density
    N, depth, h2, lens = a
    e, psi = dmrg_ground_state(N, 0.5, h2, bond_dim=100)
    r = run_mps(psi, N, depth, max_bond=128)
    return {'h2': h2, 'energy': e / N, 'outputs': r['outputs'], 'syn': syndrome_density(r['syndromes']),
            'sop': {L: mps_expectation(psi, sop_ops(N, L)) for L in lens}}


# ---------- stages ----------

def stage_circuit_demo(quick, workers):
    from qcnn.physics import X
    from qcnn.circuits import GATES, apply_gate, sv_expectation, unit_layers, fc_sites
    N = 9
    layers, live = unit_layers(list(range(N)))
    fine = [s for s in range(N) if s not in live]
    psi0 = np.ones(2 ** N, dtype=complex) / 2 ** (N / 2)
    for i in range(N - 1):
        psi0 = apply_gate(psi0, GATES['cz'], (i, i + 1), N)
    a, b, c = fc_sites(live)
    thetas = np.linspace(0, np.pi / 2, 9)
    runs = []
    for site in [None] + list(range(N)):
        th = [0.0] if site is None else list(thetas)
        run = {'site': site, 'theta': th, 'x': [], 'k': [], 'output': [], 'syndromes': []}
        for t in th:
            psi = psi0 if site is None else apply_gate(psi0, np.cos(t) * np.eye(2) - 1j * np.sin(t) * X, (site,), N)
            xs, ks = [], []
            for stage in range(5):
                if stage:
                    for g, q in layers[stage - 1][2]:
                        psi = apply_gate(psi, GATES[g], q, N)
                xs.append([sv_expectation(psi, {i: 'X'}, N) for i in range(N)])
                ks.append([sv_expectation(psi, {i - 1: 'Z', i: 'X', i + 1: 'Z'}, N) for i in range(1, N - 1)])
            run['x'].append(xs)
            run['k'].append(ks)
            run['output'].append(sv_expectation(psi, {a: 'Z', b: 'X', c: 'Z'}, N))
            run['syndromes'].append([sv_expectation(psi, {f: 'X'}, N) for f in fine])
        runs.append(run)
    return {'N': N, 'live': live, 'fine': fine, 'fc': list(fc_sites(live)),
            'layers': [{'label': l, 'kind': k, 'gates': [{'g': g, 'q': list(q)} for g, q in gs]} for l, k, gs in layers],
            'thetas': thetas, 'runs': runs}


def stage_heisenberg(quick, workers):
    from qcnn.circuits import pull_back, multiscale_sop
    rules = {P: _terms(pull_back([(1.0, {1: P})]), 4) for P in 'XYZ'}
    counts = [_depth_stats(d) for d in (1, 2)]
    if not quick and counts[-1]['seconds'] < 10:
        # d = 3 has millions of terms: give it 120 s in a child process, then give up
        with Pool(1) as pool:
            try:
                counts.append(pool.apply_async(_depth_stats, (3,)).get(timeout=120))
            except Exception:
                pass
    return {'rules': rules, 'sop_d1': _terms(multiscale_sop(1, centre=2), 7), 'term_counts': counts,
            'paper': {'eq8': r"U_{CP}^\dagger X_i U_{CP} = X_{\tilde i-2} X_{\tilde i} X_{\tilde i+2}",
                      'eq9': r"U_{CP}^\dagger Z_i U_{CP} = \tfrac12\left(Z_{\tilde i} - Z_{\tilde i-2}X_{\tilde i-1} - X_{\tilde i+1}Z_{\tilde i+2} - Z_{\tilde i-2}X_{\tilde i-1}Z_{\tilde i}X_{\tilde i+1}Z_{\tilde i+2}\right)"}}


def stage_resources(quick, workers):
    from qcnn.circuits import check_depth, gate_count, paper_gate_count
    rows = []
    for N in (27, 81, 243):
        for d in range(1, 8):
            try:
                check_depth(N, d)
            except ValueError:
                break
            rows.append({'N': N, 'd': d, 'ours': gate_count(N, d), 'paper': paper_gate_count(N, d)})
    return {'rows': rows}


def stage_qec(quick, workers):
    import torch
    from qcnn.qec import FIXED_CODES, unitary, logical_error, bloch_map, U_BIT
    seeds, i1, i2 = (1, 20, 20) if quick else (4, 1500, 1500)
    P_CORR, PXX, ptot = (5.8e-3, 2e-3, 2e-3), 2e-4, 1e-3
    ratios = sorted(set(np.round(np.linspace(0.05, 0.95, 3 if quick else 10), 3).tolist()) | (set() if quick else {round(1 / 3, 3)}))
    pts = [(P_CORR, PXX)] + [((r * ptot, (1 - r) * ptot / 2, (1 - r) * ptot / 2), 0.0) for r in ratios] + [((ptot / 3,) * 3, 0.0)]
    res = list(_pmap(_qec_train, [(p, x, seeds, i1, i2) for p, x in pts], workers))
    ev = lambda U, p, x: float(logical_error(*U, p, x))
    cor = res[0]
    codes = dict(FIXED_CODES, learned=(unitary(torch.tensor(cor['c1'])), unitary(torch.tensor(cor['c2']))))
    totals = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    logical = {n: [ev(U, tuple(q * T / 0.01 for q in P_CORR), PXX * T / 0.01) for T in totals] for n, U in codes.items()}

    def compare(r, p):
        fixed = {n: ev(FIXED_CODES[n], p, 0.0) for n in FIXED_CODES if n != 'identity'}
        best = min(fixed, key=fixed.get)
        return fixed['shor'], r['logical'], fixed[best], best

    rows = [compare(r, p) for r, (p, _) in zip(res[1:-1], pts[1:-1])]
    iso = compare(res[-1], pts[-1][0])

    def channel(U1):
        M, c = bloch_map(U1, P_CORR, PXX)
        return {'M': M, 'c': c, 'sv': torch.linalg.svdvals(M)}

    rnd = unitary(torch.rand(63, generator=torch.Generator().manual_seed(0)) * 2 * np.pi)
    return {'codes': list(codes),
            'correlated': {'train': {'px': P_CORR[0], 'py': P_CORR[1], 'pz': P_CORR[2], 'pxx': PXX}, 'totals': totals,
                           'logical': logical, 'history1': cor['history1'], 'history2': cor['history2']},
            'anisotropic': {'ptot': ptot, 'ratio': ratios, 'shor': [r[0] for r in rows], 'learned': [r[1] for r in rows],
                            'best_fixed': [r[2] for r in rows], 'best_fixed_name': [r[3] for r in rows]},
            'isotropic': {'ptot': ptot, 'shor': iso[0], 'learned': iso[1]},
            'channel': {'random': channel(rnd), 'learned': channel(codes['learned'][0]), 'shor': channel(U_BIT)}}


def stage_phase_ed(quick, workers):
    h1, h2 = grids(quick)
    res = list(_pmap(_ed_point, _grid_args(h1, h2), workers))
    os.makedirs('cache', exist_ok=True)
    np.savez(_cache(quick), h1=h1, h2=h2, states=np.stack([r[2] for r in res]).reshape(len(h2), len(h1), -1))
    return {'N': 15, 'h1': h1, 'h2': h2, 'energy': _reshape([r[0] for r in res], h1, h2),
            'qcnn': {'1': _reshape([r[1] for r in res], h1, h2)}}


def stage_trained(quick, workers):
    import torch
    from qcnn import learn      # lazy: the other stages must work without it
    from qcnn.qec import CDT
    h1, h2 = grids(quick)
    ntrain, iters = (4, 5) if quick else (40, 800)
    htr = np.linspace(0, 2, ntrain)
    S = torch.tensor(np.stack(list(_pmap(_ed_train, [float(h) for h in htr], workers))), dtype=CDT)
    y = torch.tensor((htr < 1).astype(float))
    try:
        z = np.load(_cache(quick))
        assert np.array_equal(z['h1'], h1) and np.array_equal(z['h2'], h2)
        grid = z['states'].reshape(len(h1) * len(h2), -1)
    except (FileNotFoundError, AssertionError):
        grid = np.stack([r[2] for r in _pmap(_ed_point, _grid_args(h1, h2), workers)])
    G = torch.tensor(grid.astype(np.float64), dtype=CDT)
    snaps = []
    torch.set_num_threads(os.cpu_count())
    r = learn.train(S, y, iters=iters, on_snapshot=lambda it, th: snaps.append(
        {'iter': it, 'grid': _reshape(learn.predict(th, G), h1, h2)}))
    return {'N': learn.N, 'params': dict(learn.PARAM_LAYOUT, total=sum(n for _, n in learn.PARAM_LAYOUT)),
            'optimizer': 'adam(lr=0.01), autograd', 'train': {'h1': htr, 'y': y.numpy()}, 'history': r['history'],
            'h1': h1, 'h2': h2, 'snapshots': snaps,
            'final': {'grid': snaps[-1]['grid'], 'train_pred': learn.predict(r['theta'], S)}}


def stage_phase_mps(quick, workers):
    N = 27 if quick else 45
    h1, h2 = grids(quick)
    res = list(_pmap(_mps_point, [(N, a, b) for a, b in _grid_args(h1, h2)], workers))
    e = _reshape([r[0] for r in res], h1, h2)
    return {'N': N, 'dmrg_bond': 64, 'circuit_bond': 64, 'h1': h1, 'h2': h2, 'energy_density': e,
            'qcnn': {str(d): _reshape([r[1][d - 1] for r in res], h1, h2) for d in (1, 2)},
            'syndrome_density': {str(d): _reshape([r[2][d - 1] for r in res], h1, h2) for d in (1, 2)},
            'boundary': find_boundaries(h1, h2, e)}


def stage_cut(quick, workers):
    from qcnn.physics import sample_complexity
    N, depths = (27, (1, 2)) if quick else (81, (1, 2, 3))
    h2 = np.array([-1.6, -0.8, 0, 0.8, 1.6]) if quick else np.unique(np.round(
        np.concatenate([np.arange(-1.6, 1.6001, 0.1), np.arange(0.30, 0.4601, 0.01)]), 3))
    lens = [2 * round(N / (2 * k)) for k in (2, 3, 5, 6)]        # 40, 28, 16, 14 for N = 81
    got = {}
    todo = sorted(h2, key=lambda h: abs(h - 0.42))               # slowest points first
    for r in _pmap(_cut_point, [(N, depths[-1], float(h), lens) for h in todo], workers, unordered=True):
        got[r['h2']] = r
        print(f'  cut: h2={r["h2"]:.3f} done ({len(got)}/{len(todo)})', flush=True)
    rows = [got[float(h)] for h in h2]
    col = lambda f: np.array([f(r) for r in rows], dtype=float)
    q = {d: col(lambda r: r['outputs'][d - 1]) for d in depths}
    sop = {L: col(lambda r: r['sop'][L]) for L in lens}
    M_q = {d: sample_complexity(v) for d, v in q.items()}
    M_s = {L: sample_complexity(v) for L, v in sop.items()}
    i = int(np.argmin(abs(h2 - 0.39)))
    s_fin = [M_s[L][i] for L in lens if np.isfinite(M_s[L][i])]
    ratio = np.array([min(s_fin) / M_q[d][i] if s_fin else np.nan for d in depths])
    ok = np.isfinite(ratio) & (ratio > 0)
    b, la = np.polyfit(np.array(depths)[ok], np.log(ratio[ok]), 1) if ok.sum() >= 2 else (None, None)
    return {'N': N, 'h1': 0.5, 'dmrg_bond': 100, 'circuit_bond': 128, 'h2': h2, 'energy': col(lambda r: r['energy']),
            'qcnn': {str(d): q[d] for d in depths},
            'syndrome_density': {str(d): col(lambda r: r['syn'][d - 1]) for d in depths},
            'sop': {str(L): sop[L] for L in lens},
            'sample_complexity': {'qcnn': {str(d): M_q[d] for d in depths}, 'sop': {str(L): M_s[L] for L in lens}},
            'inset': {'h2': h2[i], 'depth': list(depths), 'ratio': ratio,
                      'fit': {'a': None if la is None else math.exp(la), 'b': b}},
            'critical_h2_paper': 0.423}


STAGES = ['circuit_demo', 'heisenberg', 'resources', 'qec', 'phase_ed', 'trained', 'phase_mps', 'cut']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', action='append', choices=STAGES)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) - 2))
    a = ap.parse_args()
    meta = {'quick': a.quick, 'stage_seconds': {}}
    try:        # keep earlier stage timings when stages are run in separate invocations
        with open(os.path.join(OUT, 'meta.js'), encoding='utf-8') as f:
            old = json.loads(f.read().split('QCNN.meta = ', 1)[1].rstrip().rstrip(';'))
        if old['quick'] == a.quick:
            meta['stage_seconds'] = old['stage_seconds']
    except (OSError, IndexError, KeyError, ValueError):
        pass
    for name in [s for s in STAGES if not a.stage or s in a.stage]:
        t0 = time.time()
        res = globals()['stage_' + name](a.quick, a.workers)
        to_js(name, {'quick': a.quick, **res})
        meta['stage_seconds'][name] = time.time() - t0
        to_js('meta', {'generated': datetime.now().isoformat(timespec='seconds'), **meta})
        print(f'{name}: {meta["stage_seconds"][name]:.1f} s', flush=True)


if __name__ == '__main__':
    main()
