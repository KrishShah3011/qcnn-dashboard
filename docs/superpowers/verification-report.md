# QCNN dashboard: independent verification report (Task 10)

Date: 2026-10-14. Verifier: independent Sonnet subagent. No project source files were modified.
Interpreter used: `.venv/Scripts/python`. Dashboard served at http://localhost:8790 (server killed at the end). Playwright, 1440x900, plus one 800x900 pass.

## Overall verdict: PASS with defects (0 blocking, 2 medium, 5 low)

Counts: 24 PASS, 1 borderline (MSE definition), 0 FAIL on the scientific/runtime checks.

## 1. pytest

Exact final line: `27 passed in 63.36s (0:01:03)`  -> PASS

## 2. Data files

All 9 files exist (`circuit_demo, cut, heisenberg, meta, phase_ed, phase_mps, qec, resources, trained`), each has `quick: false` (meta too, `generated 2026-10-14T14:37:34`), each parses after splitting on `QCNN.<key> = ` and stripping the trailing `;`, and all contract keys are present (checked per object; `meta` stage_seconds covers all 8 stages).  -> PASS

Nulls found (2 total):
- `circuit_demo.runs[0].site = null`: allowed by contract (no-error run). OK.
- `cut.sample_complexity.qcnn["3"][1] = null` (h2 = -1.5): not covered by the contract. It is Eq. (4) returning NaN, which `clean()` turns into null. Harmless in the UI (that point is outside the 0.30-0.46 window plotted), but undocumented. See D4.

Other contract checks: grids are 33x17 [h2][h1]; `cut.h2` has 48 values; `cut.dmrg_bond/circuit_bond = 100/128`; `phase_mps` N=45, bonds 64/64; `trained.params.total = 1308`, 8 snapshots at iter 0,10,25,50,100,200,400,800.

## 3. Scientific sanity checks

| Check | Number | Result |
|---|---|---|
| cut.qcnn["3"] at h2=0 > 0.95 | 0.99995 | PASS |
| cut.qcnn["3"] at h2=1.0 < 0.1 | 0.0408 | PASS |
| max abs slope grows with d | d1 7.64, d2 13.24, d3 16.24 (all at h2~0.405) | PASS |
| cut.inset.ratio > 1 for every depth | 1.968, 2.328, 1.361 (at h2=0.39) | PASS (but ratio falls at d=3, fit b = -0.184) |
| trained final.grid vs phase_ed.qcnn["1"] mean abs diff < 0.25 | 0.1288 (max 0.494) | PASS |
| final training MSE < 0.05 | 0.02554 per Eq. (1) (the 1/(2M) definition used in the paper and the code); plain mean square from `train_pred` vs `train.y` is 0.05108 | PASS under Eq. (1), borderline under plain MSE (see D3) |
| qec.correlated learned < shor at totals <= 1e-2 | learned/shor: 1e-4 6.7e-7/1.21e-5; 1e-3 1.95e-5/1.30e-4; 1e-2 1.43e-3/2.14e-3. Holds at every total up to 2e-2. Fails at 5e-2 and 1e-1. | PASS |

Extra data observations (consistent with the page text):
- Anisotropic sweep (ptot=1e-3): learned is worse than the best fixed code at all 11 ratios, and worse than Shor at ratios >= 0.35 (learned/shor ratio range 0.19 to 1.29). Isotropic: Shor 1.594e-5 vs learned 1.627e-5 (2% worse).
- Correlated: best nested repetition (phase_phase) beats learned at totals <= 1e-3 (1.78e-5 vs 1.95e-5 at 1e-3). The page discloses this in the orange box of §10.
- Shor/bare at 1e-3 = 1.296e-4 / 1.020e-3 = 0.127. Shor logical error at T=1e-4 is 1.21e-5, about 6*pxx (pxx = 2e-6), as expected for undetected XX pairs within a block (6 intra-block pairs).
- Resources: ours 61/78/187/246/263/565/750/809/826 vs paper formula 27/72/81/216/261/243/648/783/828. They converge for large d, as the page says.
- `heisenberg.term_counts` has d=1 (16 terms) and d=2 (5776 terms); d=3 was attempted but exceeded the 120 s cap (`build_data.py:179-185`, stage took 121.6 s), so it is absent as designed.

### Physics spot-checks against the paper

- **Fig. 2b / Methods circuit** (`qcnn/circuits.py:27-41`, gates at 10-14): PASS. Layer order is NN CZ, block-centre CZ, X-controlled Toffoli with X target, then pooling. `ccz_x` (line 13) equals a -1 phase on |---> (X-controlled X-target Toffoli). Pooling uses the crossing wiring of the figure: the left fine site of block k feeds Z onto centre k-1 and the right fine site of block k feeds centre k+1 (lines 36 and 38). The swaps in Fig. 2b mean the Z is applied to the neighbouring block's centre. Fixed-point and single-X-error tests pass (tests/test_circuits.py).
- **Eq. (2)** (`qcnn/physics.py:13-32`): PASS. H = -J sum Z_i X_{i+1} Z_{i+2} - h1 sum X_i - h2 sum X_i X_{i+1}, open chain, matches the printed form. The MPO (43-51) uses the same signs.
- **Eq. (4)** (`qcnn/physics.py:81-87`): the code uses arcsin(sqrt p) - arcsin(sqrt p0) rather than the printed "arcsin sqrt p - sqrt(arcsin p0)". This is the correct variance-stabilising form, and it is disclosed on the page and in the docstring.
- **Eqs. (8)-(9)** (`qcnn/circuits.py:169-176`, data in `heisenberg.js`): Eq. (8) matches exactly (X -> X_{c-2} X_c X_{c+2}, coefficient +1). Eq. (9): our rule has +1/2 on the two middle terms, the paper prints -1/2. The two forms differ by conjugation with Z on the fine sites (Z_{c-1} Z_{c+1}), i.e. the sign convention of the feed-forward. The dense-matrix test `test_heisenberg_rules_match_dense` validates our signs against the explicit circuit. The deviation is disclosed in the code comment (circuits.py:170) and in §4 ("Signs depend on convention", with "paper prints -1/2" badges). Not an error. `heisenberg.paper.eq8/eq9` strings match the contract verbatim.
- **QEC noise Eqs. (15)-(16)** (`qcnn/qec.py:80-91`): PASS. Eq. (15) is applied per qubit with `(1 - sum p) rho + sum p_mu s rho s`. Eq. (16) is applied on within-block pairs (0,1) and (1,2), which is the paper's i in {1,2,4,5,7,8}. The correlated sweep scales px, py, pz and pxx together by T/0.01 (the training point sums to 0.0098 + 0.0002 = 0.01). f_q is averaged over 6 states and eps_L = 1.5(1 - f_q) (qec.py:129), per Methods. M_U1 follows Eq. (17) with |0> ancillas traced. Bold-driver rule matches Methods (+5% / -50%), run on log-cost (documented at qec.py:171-175).
- Qubit counts: the QEC check uses 63+63 = 126 parameters, matching the paper's 126.

## 4. Browser run (1440x900, http://localhost:8790/index.html)

Console: 0 errors, 1 warning (Plotly's third-party Canvas2D `willReadFrequently` notice).

| Section | errcard | qbadge | plots non-empty | Screenshot | Interaction exercised | DOM updated |
|---|---|---|---|---|---|---|
| #sec-hero | 0 | 0 | SVG hero art present | `docs/superpowers/screens/verify-sec-hero.png` | none (static) | n/a |
| #sec-idea | 0 | 0 | SVG present | `verify-sec-idea.png`, `verify-sec-idea-mera.png` | QCNN/MERA toggle | yes (aria-pressed and diagram flipped to mirrored MERA) |
| #sec-physics | 0 | 0 | SVG present | `verify-sec-physics.png`, `verify-sec-physics-inject.png` | inject X error on site 5 | yes (K4 and K6 turn red, text explains) |
| #sec-circuit | 0 | 0 | SVG present | `verify-sec-circuit.png`, `verify-sec-circuit-b.png` | error site 5, "step through" to layer 4 | yes (stage slider 4/4, q6 syndrome -1.00, output 1.000) |
| #sec-multiscale | 0 | 0 | 1 plot (2 bars: 16 and 5,776) | `verify-sec-multiscale.png`, `verify-sec-multiscale-b.png` | hover/static only | n/a |
| #sec-phase | 0 | 0 | 2 plots, heatmaps 33x17 | `verify-sec-phase.png`, `verify-sec-phase-b.png` | source toggle MPS d=2 to ED d=1 | yes (caption and z-values changed; boundaries drawn) |
| #sec-depth | 0 | 0 | 1 plot, 3 traces | `verify-sec-depth.png` | depth slider 3 to 1 | yes (line widths 3.6/1.4/1.4 reassigned) |
| #sec-samples | 0 | 0 | 2 plots | `verify-sec-samples.png`, `verify-sec-samples-b.png` | hover on QCNN d=1 point | yes (tooltip "M = 17.0") |
| #sec-rgflow | 0 | 0 | 2 plots | `verify-sec-rgflow.png` (-b, -c) | pill toggle, level toggle, play | yes (traces 12 to 10, marker animates x 1.05 to 3 and button returns to "play") |
| #sec-learning | 0 | 0 | 5 plots | `verify-sec-learning.png` (-b, -c, -d) | slider to iter 800, Play | yes (snapshot index advances 0..6 in 5 s, button shows "Pause") |
| #sec-qec | 0 | 0 | 4 plots + 1 canvas | `verify-sec-qec.png` (-b, -c, -d, -iso, -bloch) | correlated/anisotropic/isotropic tabs, Bloch U1 and exaggeration slider | yes (iso shows 1.59e-5 / 1.63e-5 cards; singular values change to 0.99885/0.97619/0.97533; "x60") |
| #sec-hardware | 0 | 0 | no plots (bar and table) | `verify-sec-hardware.png`, `verify-sec-hardware-b.png` | N 81 to 243, d 3 to 4 | yes (261/263 to 828/826, highlighted table row moves) |
| #sec-verdict | 0 | 0 | table | `verify-sec-verdict.png`, `verify-sec-verdict-b.png` | static | n/a |

800 px width: no horizontal page overflow (`scrollWidth = clientWidth = 785`). The rail collapses to a top bar. Charts and tables fit. Screenshots: `verify-800-physics.png`, `verify-800-circuit.png`. (The internal MathML elements that report right > viewport are inside the KaTeX hidden-accessibility layer, not visible overflow.)

Visual review notes (screenshots looked at): no empty plots, no clipped axis labels. Defects found are listed in D1 (circuit label overlap) and D6 (heatmap gridlines).

## 5. Verdict table (§12) vs data

Page header says 6 reproduced, 0 qualitatively, 4 not reproduced, 0 pending. Each row cross-checked:

| Row | Page "ours" | Matches data? | Status consistent? |
|---|---|---|---|
| Fixed point and QEC criteria | output 1.000000, syndrome fired | yes (tests pass) | yes |
| Output sharpens with depth | 7.64 / 13.24 / 16.24 | yes | yes |
| QCNN beats SOP sample complexity | 1.97 / 2.33 / 1.36 | yes | yes (all > 1) |
| Exponential advantage | paper 1.73 e^0.28d vs ours 2.66 e^-0.184d | yes | yes, "not reproduced" (honest) |
| Trained QCNN recovers 2D diagram | 0.129 over 561 pts, MSE 0.026 | yes | yes (see D3 on MSE convention) |
| Parameter count | 1309 vs 1308 (-1) | yes | OK ("reproduced" with a disclosed -1) |
| Learned QEC beats Shor (correlated) | 6.66 at 1e-3, 1.50 at 1e-2 | yes (1.2958e-4/1.946e-5; 2.140e-3/1.4285e-3) | yes (but see D5) |
| Fig. 11 biased noise | range 0.19-1.29, "3 of 11" | range yes; the 3 uses a 0.95 tolerance (sec-qec.js:422), strictly only 1 of 11 is >= 1 | yes, "not reproduced" |
| Shor vs no correction | Shor/bare = 0.13 | yes | yes, "not reproduced" |
| Haldane chain | not attempted | n/a | yes |

Nothing is labelled reproduced that the data contradicts.

Required disclosures present on the page:
- Shor vs bare qubit: row plus orange "Where we deviate from the paper" box in §10.
- Fig. 11 partial: caption says "partly reproduced", verdict row not reproduced, explanation text.
- Exponential advantage not reproduced at N=81: verdict row plus §7 text ("our chain is too short to confirm", boundary effect at d=3).
- Haldane chain not attempted: verdict row.
- 1308 vs 1309: hero stat ("paper: 1309"), §9 Fig. 9 caption, verdict row.
- Adam vs bold driver: §9 sidebar ("Optimiser ... paper uses finite differences and bold driver. We use autograd with Adam"). The QEC stage does use the bold driver; the sidebar does not say so, but the §10 caption does.
- Eq. (9) sign convention and Eq. (4) typo: both disclosed.

## Defects

D1. `dashboard/js/sec-circuit.js` (syndrome/output labels at stage 4) — low/medium — at stage "after layer 4" the syndrome label text ("fired -1.00" on q6, "1.0" on q3) is drawn underneath the bead column, so it reads "f[bead]d -1.00" and "1[bead]" (see `docs/superpowers/screens/verify-sec-circuit-b.png` at 1440 px and `verify-800-circuit.png`). Suggested fix: move the stage-4 bead column (or the meter value labels) right by roughly 30 px, or drop the "fired" word and show only the value to the left of the meter.

D2. `dashboard/js/sec-intro.js:20` — medium — hero copy says the circuit recognises the phase "with exponentially fewer samples than string order parameters need", while §7 and the verdict row report "Exponential advantage: not reproduced" (fit b = -0.184, ratios 1.97/2.33/1.36). The hero overclaims relative to our own results. Suggested fix: rephrase to the paper's claim ("which the paper reports needs exponentially fewer samples; here we measure a 1.4 to 2.3x advantage at N=81") or "fewer samples".

D3. `qcnn/learn.py:42-44` and `dashboard/js/sec-learn.js` (MSE labels) — low — the "final training MSE 0.026" is the Eq. (1) value, which includes the 1/2 factor. The plain mean squared error is 0.0511 (> 0.05, the plan's threshold). Passing the Task-10 criterion depends on that convention, and the page labels it "MSE (Eq. 1)" only on the chart axis. Suggested fix: in the verdict row, say "final training loss (Eq. 1, with 1/2) 0.026".

D4. `build_data.py` (`clean`, stage_cut) / `dashboard/data/cut.js` — low — `cut.sample_complexity.qcnn["3"][1]` is null (NaN from Eq. 4 at h2 = -1.5), a null not allowed by the data contract. Harmless in the UI. Suggested fix: add "null where p <= p0" to the Task 4 contract, or leave as is.

D5. `dashboard/js/sec-qec.js:405-416` — low — the "Learned QEC beats Shor (correlated)" row is marked reproduced at 1e-3 (6.66x) although the best nested repetition code beats the learned code at totals <= 1e-3 (1.78e-5 vs 1.95e-5 at 1e-3). The paper's claim is "significantly better than known codes". This is disclosed in the §10 box, not in the verdict row. Suggested fix: add "(but phase/phase repetition is 1.1x better at 1e-3)" to the `ours` string.

D6. `dashboard/js/sec-learn.js` (heatmap traces; exact line not located, `verify-sec-learning-c.png`) — low (cosmetic) — the 33x17 heatmaps render with visible dark horizontal row gridlines. Suggested fix: set `zsmooth: false` with `xgap: 0, ygap: 0` (or `hoverongaps:false`, `connectgaps:true`) and check the axis line colour, or use `zsmooth: 'fast'`.

D7. `dashboard/js/sec-learn.js` (Fig. 4 line-cut legend) — low (cosmetic) — the legend row overlaps the top-left training-label crosses at y = 1 (`verify-sec-learning-d.png`). Suggested fix: raise the legend (y=1.15) or extend the y-axis range to 1.1.

## Not verified

The missing-data fallback (the errcard path from Task 5) was not re-tested here, since all 9 data files exist. No source files were changed.
