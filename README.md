# QCNN reproduction and interactive dashboard

Reproduction of Cong, Choi & Lukin, *Quantum Convolutional Neural Networks* ([arXiv:1810.03787](https://arxiv.org/abs/1810.03787)), with a static, offline dashboard that walks through the circuit, the multiscale structure, phase recognition, learning dynamics and quantum error correction.

## View the dashboard

Open `dashboard/index.html` in a browser. It works from `file://` and needs no server. All results are precomputed in `dashboard/data/*.js`.

## Reproduce

Python with an isolated virtualenv (quimb, torch, numpy, scipy, pytest):

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q          # 27 tests
.venv/Scripts/python build_data.py --quick       # ~2 min smoke build
.venv/Scripts/python build_data.py               # full build, ~1.5 h on 16 cores
```

Single stage: `build_data.py --stage <circuit_demo|heisenberg|resources|qec|phase_ed|trained|phase_mps|cut>`.

## Layout

- `qcnn/` science core (circuit, MPS simulation, physics, QEC, learning)
- `build_data.py` generates `dashboard/data/`
- `dashboard/` the front end
- `docs/superpowers/` design spec, plan and verification report

## Known deviations from the paper

Disclosed in section 12 of the dashboard and in `NEXT_STEPS.md`:

- Shor's code beats the bare qubit under the Methods noise model, unlike the paper's Fig. 5b.
- The Fig. 11 anisotropic QEC result is not reproduced; random-start training stalls in local optima.
- The exponential sample-complexity advantage is not reproduced at N = 81 (finite-size effect; N = 243 is the proposed check).
- The Haldane chain (Fig. 7) was not attempted.
