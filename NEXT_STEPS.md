# QCNN reproduction: open gaps and next steps

The project is complete and verified (`docs/superpowers/verification-report.md`). This file lists what is still open, ordered by scientific value, with the exact commands to run. The dashboard's §12 table ("Paper vs ours") updates itself from the data, so a rerun that closes a gap shows up there automatically.

## Before you start

- Run everything from `C:/Users/Krish/Python/QC` with `.venv/Scripts/python`. Never `pip install` into the global Python; numpy and scipy are pinned there for other projects.
- Tests: `.venv/Scripts/python -m pytest tests -q`. Expect 27 passed.
- Rebuild one stage: `.venv/Scripts/python build_data.py --stage <name>`. Stages are `circuit_demo, heisenberg, resources, qec, phase_ed, trained, phase_mps, cut`.
- View the result by opening `dashboard/index.html` in a browser. A hard refresh (Ctrl+F5) picks up new data.
- The machine has 16 cores and the build uses 14 workers by default (`--workers K` to change). Close heavy apps during the long runs.

---

## Gap 1: exponential sample-complexity advantage (Fig. 2d inset). Highest value.

**Status:** not reproduced.
- At h₂ = 0.39, the ratio (SOP copies ÷ QCNN copies) is 1.97, 2.33 and 1.36 for d = 1, 2, 3.
- Our fit is 2.66·e^(−0.18 d). The paper gets 1.73·e^(0.28 d).

**Why:** this looks like a finite-size effect.
- At N = 81 the depth-3 circuit ends on 3 sites, so its measured operator spans the whole open chain.
- Its output starts to drop already at h₂ ≈ 0.38, before the transition.
- The paper used N = 135 and itself shades this near-critical region as finite-size limited.

**Next step:** rerun the `cut` stage at N = 243. That allows depths 1–4 (243 → 81 → 27 → 9 → 3) and tells us whether the dip at d = 3 is finite-size.

1. In `build_data.py`, `stage_cut` (around line 288), change
   ```python
   N, depths = (27, (1, 2)) if quick else (81, (1, 2, 3))
   ```
   to
   ```python
   N, depths = (27, (1, 2)) if quick else (243, (1, 2, 3, 4))
   ```
2. To keep the run affordable, restrict the h₂ grid to the region that matters. In the line after it, replace the full-run grid with
   ```python
   np.round(np.concatenate([np.arange(-0.4, 0.2001, 0.2), np.arange(0.30, 0.4601, 0.02)]), 3)
   ```
   That gives about 13 points instead of 48.
3. Consider raising the bond dimensions in `_cut_point` (line ~133): DMRG `bond_dim=100` → 150 and circuit `max_bond=128` → 160. The paper used about 130–160. Check convergence by running one point (for example h₂ = 0.39) at two bond dimensions.
4. Run `.venv/Scripts/python build_data.py --stage cut`.
   - **Estimate:** 1–3 h, which is several minutes per point on 14 workers. This is the least certain number in this file.
   - **Memory:** each worker holds an N = 243 MPS with bond 128–160. Watch RAM in Task Manager, and lower `--workers` if it climbs past about 80%.
5. **Success check:** `cut.inset.ratio` increases with depth and `cut.inset.fit.b > 0`. The verdict row "Exponential advantage" turns ✓ or ≈.
6. The dashboard iterates over whatever depths and SOP lengths exist, so no frontend change is needed. The SOP lengths for N = 243 become 122, 82, 48, 40.

Optional: also add an h₂ selector to the §7 inset. Today it shows h₂ = 0.39 only, and the ratio-vs-depth curve is more informative across the whole window. The code is in `dashboard/js/sec-phase.js` around line 256.

---

## Gap 2: learned QEC code vs Shor under biased noise (Fig. 11)

**Status:** not reproduced.
- With ptot = 1e-3, the learned code beats Shor at 1 of 11 biases and is within 5% at 3 of them.
- For strongly X-biased noise it is up to 5× worse.
- Isotropic noise (ratio 0.98) and the correlated model (1.5–18× better than Shor) work fine.

**Why:** an optimisation failure, not a limit of the model.
- The QCNN code family contains Shor's code, so it can always do at least as well.
- Random-start training (paper init c ~ U[0, 2π), bold driver on log-cost, 4 seeds × 1500 iterations per stage) gets stuck in local optima.
- We checked the stage-1 cost directly. Shor's inner bit-flip block scores better on C₁ than the alternatives for X-biased noise. The losses come mostly from stage 2 failing to find the matching outer phase code.

**Next steps**, cheapest first. All of them live in `qcnn/qec.py` (`train_code`, about line 171) and `build_data.py` (`stage_qec`, about line 205).

1. **More random starts.** Raise `seeds=4` to 16 in `stage_qec` (line ~207, the `(4, 1500, 1500)` tuple). This costs about 4× time; the current `qec` stage takes about 10 min.
2. **Joint fine-tuning after the sequential stage.** The paper says joint training from scratch gets trapped. Starting from the sequential solution is different. After `best2`, run `bold_driver` on `torch.log(logical_error(unitary(c[:63]), unitary(c[63:]), p, pxx))` over the concatenated 126 parameters for about 1000 iterations. Keep the result only if it improves.
3. **Adam instead of the bold driver** for stage 2 (`torch.optim.Adam`, lr about 0.02). The design spike found it similar or slightly better.
4. **Disclose any warm start.** If you add a warm start (for example U2 initialised near the phase repetition decoder), say so in the dashboard. It is no longer "learned from random".

After each change:
- Run `.venv/Scripts/python -m pytest tests/test_qec.py -q`.
- Run `.venv/Scripts/python build_data.py --stage qec`.
- **Success check:** in `qec.anisotropic`, Shor ÷ learned ≥ 1 at every bias. The verdict row "Learned QEC ≥ Shor for biased noise" turns ✓.

---

## Gap 3: Shor vs no correction under correlated noise (Fig. 5b)

**Status:** this disagrees with the paper, and it is probably not a bug on our side.
- With the noise model exactly as written in Methods (px = 5.8e-3, py = pz = 2e-3, pxx = 2e-4 at total 1e-2), our Shor code is about 8× better than a bare qubit at total 1e-3.
- The paper's figure shows Shor slightly worse than no correction.

**Possible explanations to test** (run each as an extra correlated sweep, not a replacement):
1. Correlated XX errors on all neighbouring pairs, including across blocks: (3,4) and (6,7) in addition to the in-block pairs. Change the pair list in `_noise` in `qcnn/qec.py` (currently pairs (0,1), (1,2) inside a block). This needs the 9-qubit channel instead of the block-product shortcut, so it is a bigger change.
2. A larger pxx. Shor fails at a rate of about 6·pxx, so the paper's curve would need pxx about 10× larger than printed.
3. A different definition of "input error rate" on the x-axis, for example summed over all 9 qubits.

**Success check:** find which variant puts Shor above the identity line, then write it up in §10's deviation note (`dashboard/js/sec-qec.js`, the "deviation callout" block around line 192).

---

## Gap 4: Haldane chain (Fig. 7, Methods). Largest effort.

**Status:** not attempted.

**Plan:**
1. Spin-1 Hamiltonian H = J Σ Sⱼ·Sⱼ₊₁ + ω Σ (Sᶻⱼ)², N = 54 spins, open chain. The DMRG goes in `qcnn/physics.py`, using a quimb MPO with physical dimension 3.
2. Map each spin-1 site to two spin-1/2 qubits with the isometry in Methods: |x⟩ → |+−⟩, |y⟩ → −|−+⟩, |z⟩ → −i|−−⟩, plus the singlet |s⟩ → |++⟩. That gives 108 qubits.
3. Run the existing exact QCNN (`run_mps`) on the mapped state at depths 1–4. 108 = 4·27, so check which depths `check_depth` allows; this may need N = 81 spin-1 or padding.
4. Sweep ω/J across about 1.035, the critical point the paper found for N = 54.
5. Add a data stage `haldane` and a small dashboard panel. The verdict row already exists and will need its function updated.

**Estimate:** a day of work plus several hours of compute.

---

## Smaller items

- **Eq. 9 sign convention:** our derived rule has + on the two middle terms where the paper prints −. It is verified against dense matrices and disclosed in §4. A one-paragraph derivation in the dashboard would make the convention fully transparent.
- **`heisenberg.term_counts` at depth 3:** this timed out at the 120 s cap; depth 2 has 5776 terms. A sparse Pauli-algebra rewrite (bit-packed strings) would make depth 3 feasible.
- **The paper's optimiser for the trained QCNN:** we used Adam (MSE 0.124 → 0.030 in about 600 iterations). The paper's bold driver with finite differences reached only 0.079 in the same budget. To be strictly faithful, run `train()` with the bold driver for about 3000 iterations and compare the final grid.
- **1308 vs 1309 parameters:** the paper's count is one higher than ours. It is most likely a different count for the pooling V₁/V₂ parameterisation; not worth chasing unless you are curious.

## Verify after any change

1. `.venv/Scripts/python -m pytest tests -q` passes.
2. Every file in `dashboard/data/*.js` has `quick: false`.
3. Open the dashboard. There should be no red error cards, and §12 should reflect the new numbers.
