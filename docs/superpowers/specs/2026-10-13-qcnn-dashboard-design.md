# QCNN Paper Reproduction + Interactive Dashboard — Design

Paper: Cong, Choi, Lukin, *Quantum Convolutional Neural Networks*, arXiv:1810.03787v2 (Nature Physics 2019). File: `1810.03787v2.pdf`.

## Intent

**What the user asked for:** understand the paper in depth, implement it, and end with an interactive, good-looking visual dashboard that shows how the whole thing works, written from the viewpoint of a quantum-computing professor. The work should be planned in phases, with Haiku subagents implementing each phase and a Sonnet subagent verifying at the end.

**Assumptions (the goal directive said to proceed without asking):**
- "Implement the paper" means reproducing its computational results (Fig. 2a/2c/2d, 3, 4, 5b, 8/Eq. 8–9, 11, and the resource formula) with real simulations, not just illustrating them.
- The dashboard is a static page that you open in a browser. It uses data precomputed in Python and needs no server.
- Not reproduced: Fig. 7 (spin-1 Haldane chain, Methods). It needs spin-1 DMRG plus a 108-qubit mapped circuit, which is a large extra cost for a side result. The dashboard says so.

**Success criteria:**
1. Each module has an assert-based check that passes, including the fixed-point and QEC criteria, Eq. 8–9, and the known Shor-code behaviour.
2. The reproduced figures show the paper's qualitative claims: the QCNN output sharpens with depth; QCNN sample complexity is lower than SOP sample complexity near criticality; a QCNN trained only on the h₂=0 line recovers the 2D phase diagram; a learned QEC code beats Shor under correlated noise.
3. `dashboard/index.html` opens from `file://` and every panel renders with no console errors.

## Verified facts (checked numerically while designing)

**Exact QCNN unit (Fig. 2b).** Block centres are the coarse sites `c = 3k+1`. One convolution–pooling unit on L live sites, applied in this order:
1. CZ on every nearest-neighbour live pair `(i, i+1)`.
2. CZ on every neighbouring coarse pair `(c, c+3)`.
3. A "Toffoli with X-basis controls" on `(c-1, c, c+1)`: a phase of −1 on |−−−⟩, i.e. CCZ conjugated by H⊗H⊗H (Eq. 14).
4. Pooling as a deferred measurement: fine site `c-1` controls Z on `c-3`, and fine site `c+1` controls Z on `c+3`. The control fires on X = −1 (CZ conjugated by H on the control, Eq. 13). Skip any gate whose target lies outside the chain.

After the unit, the live sites are the coarse sites. The fully connected layer measures `Z_{m-1} X_m Z_{m+1}` on the middle three live sites.

Checks at N=9:
- Cluster state in: every fine ⟨X⟩ = +1 and coarse ⟨ZXZ⟩ = 1 (fixed point).
- Any single X error: at least one fine ⟨X⟩ = −1, and coarse ⟨ZXZ⟩ is still 1 (QEC criterion).

**Heisenberg picture (Eq. 8–9), checked at N=9, c=4:**
- `U†X₄U = X₂X₄X₆`
- `U†Z₄U = ½(Z₄ + Z₂X₃ + X₅Z₆ − Z₂X₃Z₄X₅Z₆)`

The paper prints minus signs on the middle two terms. Those signs depend on convention and are worth flagging in the dashboard.

**MPS (quimb 1.15, `.venv`):**
- DMRG2 on Eq. 2 at h₁=h₂=0 gives E = −(N−2). The circuit output at the fixed point is about 1 for every depth.
- At N=81 and h₂=0.42, the circuit needs max_bond ≥ 128 to converge at d=3. D=64 is enough for N=45 at d ≤ 2.
- Measured times: N=81, D=128 takes about 3–4 min per state. N=45, D=64 takes a few seconds.

**Exact diagonalization:** N=15 takes about 0.5 s per ground state with `eigsh`. E(0,0) = −13.

**QEC under the correlated model, as stated in Methods:**

| Input total | Shor logical error | Identity logical error |
|---|---|---|
| 1e-4 | 1.21e-5 | 1.02e-4 |
| 1e-2 | 2.1e-3 | 1.0e-2 |

Correlated XX errors cause a logical failure in Shor's inner bit-flip blocks at about 6·p_xx. Fig. 5b in the paper instead shows Shor slightly *worse* than identity. Our faithful parameters do not reproduce that. The dashboard reports this as a deviation and does not tune numbers to match.

## Architecture

```
QC/
  qcnn/physics.py        Eq.2: sparse H (ED), MPO, ground states (ED + DMRG), SOP Eq.3, sample complexity Eq.4
  qcnn/circuits.py       exact QCNN (statevector N≤15 + MPS any N), syndromes per level, Pauli decomposition (Eq.8-9),
                         gate counting (resource formula)
  qcnn/learn.py          Fig.9 parameterised QCNN in torch, Gell-Mann unitaries, MSE Eq.1, bold-driver GD
  qcnn/qec.py            Eq.5,15,16,17: Pauli/XX channels, block channel M_U1, Shor/repetition baselines,
                         C1 = p1²+p2+p3 then fidelity training of U2
  tests/test_*.py        one assert-based file per module (run: .venv/Scripts/python -m pytest tests -q, or python file)
  build_data.py          runs every precompute stage (multiprocessing), writes dashboard/data/*.js
  dashboard/index.html   page shell + section containers; styles.css; js/<section>.js per section; data/*.js
```

Data files have the form `window.QCNN = window.QCNN || {}; QCNN.<key> = {...};`, which loads from `file://` without hitting CORS. Plotly.js and KaTeX come from a CDN.

Runtime: `.venv` (with `--system-site-packages`) supplies numpy 2.0, scipy 1.18 and quimb, and torch comes from the system install. The global Python environment stays untouched.

## Computations (`build_data.py` stages)

| key | what | size / params |
|---|---|---|
| `circuit_demo` | N=9 exact QCNN: per-gate-layer snapshots for the cluster state and for an X error at each site: fine ⟨X⟩ syndromes, coarse ⟨ZXZ⟩ | statevector |
| `heisenberg` | Eq. 8–9 terms; term count of the pulled-back ZXZ for d=1,2 | Pauli algebra |
| `phase_ed` | N=15 ED grid: h₁∈[0,1.6] step 0.1 (17) × h₂∈[−1.6,1.6] step 0.1 (33); energy, exact QCNN d=1 | ED |
| `phase_mps` | N=45 DMRG (D=64) on the same grid; QCNN d=1,2; energy density; boundary points from peaks of −∂²e/∂h₂² | MPS |
| `cut` | N=81, h₁=0.5, h₂ ∈ [−1.6,1.6] step 0.1 plus [0.30,0.46] step 0.01; QCNN d=1..3 (D=128); SOP lengths ≈N/2,N/3,N/5,N/6 (b−a even, centred); syndrome density per level; sample complexity Eq. 4 | MPS, the longest stage, ~20–40 min with a pool |
| `trained` | Fig. 9 QCNN, N=15, d=1; 40 training points h₂=0, h₁∈linspace(0,2,40), label 1 iff h₁<1; c∼U[0,2π]; MSE; bold driver (+5% / −50%, η₀=10); gradient by autograd (Methods use finite differences; the outlook suggests backprop; the gradient is the same up to O(ε²)); loss/η history; grid predictions at ~10 snapshots | torch |
| `qec` | U1, U2 three-qubit (63 Gell-Mann params each = 126); models: isotropic, anisotropic sweep (p_tot=1e-3, p_y=p_z, p_x/p_tot ∈ 11 values), correlated (train at 5.8e-3/2e-3/2e-3/2e-4, evaluate totals 1e-4..1e-1); Shor, 4 nested repetition codes, identity, learned; channel M_U1 singular values before/after | numpy/torch |
| `resources` | gate count from our circuit vs 7N/2(1−3^{1−d}) + N·3^{1−d}; Rydberg feasibility inputs | counting |

## Dashboard

The page reads as a guided lecture that scrolls top to bottom. Each section has a short professor-voice explanation followed by an interactive figure.

**Design system.** Reuse the paper's own colour code:
- convolution = blue `#2f7fd0`
- pooling = orange `#f08a24`
- fully connected = purple `#7a3fb0`
- errors = red `#d1342c`

The ground is a deep ink background (`#0d1117`-ish) with warm paper-white text. Headings use a serif display face (e.g. Fraunces), body text a clean sans (e.g. Inter), and numbers a mono face. Charts share one Plotly template. Layout is a sticky side navigation with progress, and wide figures.

**Sections:**
1. **Hero:** paper and key numbers.
2. **CNN → QCNN → MERA:** an animated layer diagram with a direction toggle.
3. **The physics:** Eq. 2, the cluster state and stabilizers.
4. **Exact circuit:** an SVG of the N=9 circuit with a gate-step slider. The reader picks an X-error site and watches the syndrome fire and get corrected (Fig. 2b and Fig. 3).
5. **Phase diagram:** heatmaps (N=15 ED d=1 and N=45 MPS d=1,2) with a depth/source toggle, hover readouts and the boundary overlay.
6. **Depth sharpening (Fig. 2c):** curves for d=1..3 with a highlight control.
7. **Sample complexity (Fig. 2d):** QCNN against SOPs, plus a ratio-versus-depth inset with an exponential fit.
8. **RG flow:** syndrome density by level for chosen h₂, animated.
9. **Multiscale SOP:** the Eq. 8–9 terms and term-count growth.
10. **Learning:** the Fig. 9 parameter-count breakdown, loss and η curves, and a heatmap that evolves across training snapshots, set beside the exact result.
11. **QEC:** the Fig. 5a encoder/decoder diagram, the logical-versus-input log-log plot, the anisotropy ratio plot (Fig. 11), and a Bloch-ellipsoid view of M_U1.
12. **Hardware:** the Rydberg resource calculator (N, d, Ω, τ sliders).
13. **Paper vs reproduction:** a table of claims, our numbers and deviations.

## Error handling and testing

- Science modules raise errors on invalid N (for the exact QCNN, N must be divisible by 3 at each level and leave ≥3 sites).
- Each build stage checks that every value is finite before it writes data.
- Tests: `test_physics.py` (ED energy at the cluster point, MPO against ED energy at N=9), `test_circuits.py` (fixed point, QEC criterion, Eq. 8–9, MPS matching statevector at N=9 and N=15 (d=1)), `test_learn.py` (unitarity of Gell-Mann exponentials, loss falls over 5 steps), `test_qec.py` (identity logical error = p_x+p_y+p_z, Shor corrects every single-qubit error to O(p²)).
- Final check: a Sonnet subagent runs the tests, opens the dashboard with Playwright, takes screenshots and checks the console.
