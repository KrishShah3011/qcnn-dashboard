# QCNN reproduction: paper explanation, what we did, and how it compares

Paper: Cong, Choi & Lukin, *Quantum Convolutional Neural Networks*, Nature Physics 15, 1273 (2019), [arXiv:1810.03787](https://arxiv.org/abs/1810.03787). The PDF is in this folder (`1810.03787v2.pdf`).

## 1. What the paper says

**Idea.** A classical CNN alternates convolution layers (the same small filter applied everywhere) and pooling layers (throw away information, shrink the data), then ends in a fully connected layer. A QCNN does the same with qubits:

- **Convolution:** the same few-qubit unitary is applied to every neighbouring group of qubits.
- **Pooling:** some qubits are measured or traced out, and the result controls a gate on the qubits that stay. The chain gets shorter, here by a factor of 3 per layer.
- **Fully connected:** after d layers a few qubits remain, and a general unitary acts on them before one qubit is measured.

With N qubits and d layers there are only O(log N) independent parameters, because every layer reuses one filter. The circuit is a quantum cousin of the MERA tensor network and of a renormalisation-group (RG) flow: each layer zooms out by 3.

**Three applications in the paper.**

1. **Quantum phase recognition (Figs. 1-4).** The target is the symmetry-protected topological (SPT) phase of the Hamiltonian H = -J Σ Z X Z - h1 Σ X - h2 Σ X X (Eq. 2). The paper writes down an exact QCNN circuit (Fig. 2b) that maps the cluster-state fixed point to a product state and outputs ⟨Z X Z⟩ = 1. The claims:
   - The output becomes a sharper phase indicator as depth d grows (Fig. 2c).
   - The QCNN needs fewer measurement samples than the string order parameter (SOP), and the advantage grows exponentially with d (Fig. 2d inset).
   - A QCNN with random starting parameters can *learn* the phase diagram from a few labelled points on a line, with 1309 parameters (Figs. 3-4).
2. **Quantum error correction (Figs. 5, 9-11).** QCNN circuits are used as encoder/decoder pairs. Training them (two stages, bold-driver gradient descent) is claimed to find codes that beat Shor's 9-qubit code under correlated noise (Fig. 5) and under biased noise (Fig. 11).
3. **Haldane chain (Fig. 7).** A spin-1 chain mapped to qubits, with the QCNN detecting the Haldane phase.

The paper also discusses hardware cost (Methods), roughly how many gates and qubits are needed to run it.

## 2. What we built

Everything is in this repo. The science code is in `qcnn/`, the data builder is `build_data.py`, and the front end is in `dashboard/`.

| Piece | File | What it does |
|---|---|---|
| Exact QCNN circuit | `qcnn/circuits.py` | Fig. 2b circuit as an MPS (matrix-product-state) simulation. Also Heisenberg-picture operator evolution (Eqs. 8-9) and the trainable QCNN. |
| Physics | `qcnn/physics.py` | Hamiltonian of Eq. 2 (dense, and MPO for DMRG), exact diagonalisation, Eq. 4 sample-complexity estimate. |
| Learning | `qcnn/learn.py` | Trained QCNN with 1308 parameters, trained by autograd and Adam, as in Figs. 3-4. |
| Error correction | `qcnn/qec.py` | Noise model of Eqs. 15-16, two-stage code training, Shor and repetition baselines. |
| Data build | `build_data.py` | Runs all 8 stages and writes `dashboard/data/*.js`. Full build is about 1.5 h. |
| Dashboard | `dashboard/` | Offline interactive page, 12 sections plus overview. |
| Tests | `tests/` | 27 tests. All pass (re-run on 2026-11-18: `27 passed in 44.85s`). |
| Verification | `docs/superpowers/verification-report.md` | Independent check of data, physics and browser behaviour. |

## 3. Is the paper implemented correctly?

**Short answer:** the core of the paper (the exact circuit, the Hamiltonian, the phase diagram, the learned QCNN, the QEC noise model and training procedure) is implemented and checked against the paper's equations. Three claims did not reproduce. We say so openly and do not hide it.

Checks done (details in the verification report):

- **Fixed-point test.** On the cluster state the circuit outputs 1.000000. A single X error is detected: the matching syndrome qubit flips to -1. Both are unit tests, and both match the paper's stated QEC criteria.
- **Eq. 2.** Signs and open boundary match the printed Hamiltonian. The MPO uses the same signs.
- **Eqs. 8-9.** Eq. 8 matches exactly. Eq. 9 has the opposite sign on two middle terms. The two forms differ only by a sign convention for the feed-forward Z gates. A dense-matrix test confirms our signs against the explicit circuit.
- **Eq. 4.** The paper prints `arcsin√p − √(arcsin p0)`, which looks like a typo. We use `arcsin√p − arcsin√p0`, the variance-stabilising form.
- **QEC noise, Eqs. 15-16, and 126 parameters.** Match the paper (63 + 63 parameters).
- **Parameter count.** 1308 for the trained QCNN, against 1309 in the paper. The one-parameter difference is probably a different counting of the pooling parameterisation. It is unresolved.

## 4. Do the results match?

Numbers come from the full (non-quick) data build. The dashboard's section 12 ("Paper vs ours") recomputes this table from the data.

| Claim in paper | Our result | Match? |
|---|---|---|
| Fixed point gives output 1 and detects errors | 1.000000, syndrome fires | **Yes** |
| Output sharpens with depth d | max slope of output vs h2: 7.64 (d=1), 13.24 (d=2), 16.24 (d=3) | **Yes** |
| QCNN needs fewer samples than SOP | SOP copies ÷ QCNN copies at h2 = 0.39: 1.97, 2.33, 1.36 (all above 1) | **Yes** |
| Advantage grows exponentially with d (paper fit 1.73·e^(0.28d)) | Our fit 2.66·e^(-0.184d). Ratio rises from d=1 to d=2 and then falls at d=3. | **No** |
| Trained QCNN recovers the 2D phase diagram | Mean abs difference from exact QCNN output 0.129 over 561 points, training loss 0.026 (Eq. 1, includes ½ factor) | **Yes** |
| 1309 parameters | 1308 | **Yes, with a disclosed −1** |
| Learned QEC beats Shor under correlated noise | Shor ÷ learned = 6.66 at total error 1e-3, 1.50 at 1e-2. At 5e-2 and 1e-1 the learned code is worse. The best nested repetition code is 1.1× better than the learned code at 1e-3. | **Yes, in a limited range** |
| Learned QEC beats Shor under biased noise (Fig. 11) | Shor ÷ learned ranges 0.19-1.29. The learned code beats Shor at only 1 of 11 biases (it is within 5% at 3). | **No** |
| Shor is slightly worse than no correction under correlated noise (Fig. 5b) | Under the noise exactly as written in Methods, Shor is about 8× *better* than a bare qubit at total 1e-3 (ratio 0.127) | **No** |
| Haldane chain (Fig. 7) | Not attempted | **Not done** |

So the tally is **6 reproduced, 3 not reproduced, 1 not attempted** (the dashboard header counts Haldane among the 4 "not reproduced").

**Likely reasons for the three misses** (hypotheses, not proven):

- **Exponential advantage.** Finite size. At N = 81 the depth-3 circuit ends on only 3 sites, so its operator spans the whole open chain and the output dips before the transition. The paper used N = 135 and itself marks this region as finite-size limited. The proposed test is N = 243, which allows depth 4 (see `NEXT_STEPS.md`, Gap 1).
- **Biased-noise QEC.** An optimisation failure, not a model limit: the QCNN family contains Shor's code, so a perfect optimiser could not do worse. Random-start training gets stuck in local optima in stage 2.
- **Shor vs bare qubit.** The noise model as printed in Methods may differ from what produced the paper's figure (for example pair correlations across blocks, or a different definition of the x-axis). We could not tell which.

## 5. Did we improve on the paper?

Honest answer: **we did not improve the paper's method or beat its results.** Where we differ, we are mostly at or below the paper. What we added is:

1. **An interactive explanation layer.** The paper has static figures. We built a dashboard where you can inject an error on any site and step through the four circuit layers, flip between QCNN and MERA, drag the depth slider, watch the learned phase diagram form over training (iteration 0 to 800), and view learned QEC codes on the Bloch sphere.
2. **A tested, open implementation.** The paper does not ship code. This repo has 27 tests and an independent verification report.
3. **Two small corrections to the paper's text.** The Eq. 4 typo and the Eq. 9 sign convention, both checked numerically.
4. **A negative-results record.** Three claims that did not reproduce, each with a data-backed diagnosis and a concrete next step. The dashboard's section 12 reports these instead of hiding them.
5. **Optimiser comparison.** The paper trains with finite differences and a bold driver. We used autograd with Adam for the trained QCNN. In our spike runs, Adam reached lower loss faster (iteration 250: MSE 0.040 for Adam, 0.079 for bold driver). This is a small engineering finding, not a new result, and we only ran one seed per optimiser for the QCNN. For the QEC stage, the paper's bold driver was kept.
6. **Hardware resource check.** Our gate count against the paper's formula for N = 27, 81, 243: they differ for shallow circuits (for example 187 against 81 at N = 81, d = 1) and converge for deep ones (826 against 828 at N = 243, d = 4).

If your professor asks "what is new?", the accurate answer is: a faithful, tested reproduction with an explorable visual explanation, and a clear map of where it does and does not agree with the paper.

## 6. How to run it

Dashboard (no install, no server):

```
start C:\Users\Krish\Python\QC\dashboard\index.html
```

Needs internet for the Plotly, KaTeX and Google Fonts CDN scripts. All results are precomputed, so nothing runs in the background. Use Chrome or Edge.

Tests:

```
cd C:\Users\Krish\Python\QC
.venv\Scripts\python -m pytest tests -q
```

Rebuild data (optional): `.venv\Scripts\python build_data.py --quick` (about 2 min) or without `--quick` (about 1.5 h). Never `pip install` into the global Python.

## 7. Dashboard guide for the presentation

Use the left rail to jump between sections. A suggested 10-minute path:

| Section | What to say | What to click |
|---|---|---|
| Overview | The paper's claim and our headline numbers | Scroll |
| §1 From CNN to QCNN | Convolution and pooling have quantum analogues | Toggle QCNN / MERA |
| §2 The phase | The SPT phase, the Hamiltonian, what a string order parameter is | Inject an X error on a site |
| §3 An exact QCNN | The circuit of Fig. 2b, layer by layer | Pick an error site, press "step through" to see the syndrome qubit flip |
| §4 What it measures | In the Heisenberg picture the circuit measures a nonlocal operator | Hover the bars (16 terms at d=1, 5776 at d=2) |
| §5 Phase diagram | QCNN output over (h1, h2) | Toggle the source: MPS d=2 or exact diagonalisation d=1 |
| §6 Deeper is sharper | Sharper boundary with depth | Depth slider |
| §7 Sample complexity | QCNN needs fewer samples than SOP. **State here that the exponential trend did not reproduce.** | Hover points |
| §8 Renormalisation | The RG flow picture | Play, toggle levels |
| §9 Learning a QCNN | The learned phase diagram forming during training | Drag the slider or press Play |
| §10 Designing QEC codes | Learned code against Shor and repetition codes | Tabs: correlated, anisotropic, isotropic. Bloch view. |
| §11 Could we run it? | Gate counts for N = 81 and 243 | Change N and d |
| §12 Paper vs ours | The verdict table. End here. | Static |

Tips: use a window at least 1200 px wide. At 800 px the rail collapses to a top bar and everything still works. Section 12 is the one to show if time is short.

## 8. Known limitations (from the verification report)

- The Haldane chain is not done.
- All numbers are from N = 81 (phase) and N = 15 (trained QCNN), smaller than the paper's N = 135. Finite-size effects are part of why the sample-complexity trend differs.
- The QEC codes come from 4 random starts per stage, and the Adam vs bold-driver comparison for the trained QCNN was a single run each. Treat both as indicative, not statistically tested.
- The dashboard depends on CDN scripts. Offline use needs them cached.
- Last verified in a browser on 2026-10-14. On 2026-11-18 we re-ran the test suite (27 pass) but did not re-open the dashboard.

Next steps and exact commands for closing each gap are in `NEXT_STEPS.md`.
