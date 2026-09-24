# RAFTS-CUSUM-UCB Final Prototype

**RAFTS — Receiver-Aware Frequency-Time Scheduler** is a prototype wideband
scan scheduler for operation without reliable prior emitter intelligence. The
final operational learner is **CUSUM-UCB**, selected by the frozen V3.1
held-out protocol.

This folder is deliberately smaller than the research package:

- one operational adaptive learner: **RAFTS-CUSUM-UCB**;
- three simple display baselines: Sequential, Random, and Coverage Sweep;
- the receiver, retune, joint-assignment, Max-Age, temporal-evidence, evaluator,
  controlled-simulator, and TSRD replay components needed by the prototype;
- all four supplied TSRD datasets;
- key frozen selection and ablation evidence.

The unused learner implementations, full bake-off runner, old result folders,
caches, and disabled transition-memory module are not included.

## Run on Windows

Open PowerShell **inside this folder**, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run_demo.py
```

The default command uses `data/tsrd/config_0.h5`, runs 300 time slots, and
creates a live dashboard, animated replay, frozen-evidence summary, CSV, and
JSON in `outputs/`.

Fast run without the GIF:

```powershell
.\.venv\Scripts\python.exe run_demo.py --slots 120 --no-gif
```

Run another supplied dataset:

```powershell
.\.venv\Scripts\python.exe run_demo.py data/tsrd/config_103.h5
```

Run the fast cross-source validation over 20 simulator/stress cases and all
four TSRD replays:

```powershell
.\.venv\Scripts\python.exe run_validation.py --output-dir validation_smoke
```

This command is deliberately labelled a smoke diagnostic. The preserved
7,680-run held-out result—not this quick run—is the learner-selection evidence.

## What the visual outputs show

`rafts_live_demo.gif` and `rafts_live_dashboard.png` show:

- hidden frequency-time activity as **evaluation context only**;
- actual receiver HIT/MISS observations;
- receiver-to-band assignments, frequency ranges, and decision reasons;
- the five oldest coverage bands and the Max-Age threshold;
- the five strongest online CUSUM-UCB beliefs;
- per-band CUSUM reset markers and the live safety/status strip.

`rafts_evidence_summary.png` separately shows the frozen held-out comparison
with three simple baselines on interception, capped P95 TTI, net TP-minus-FP
reward/receiver-time, and retune dead fraction. The interception plot includes
a 95% bootstrap confidence interval, and the P95-TTI trade-off is called out
explicitly.

The live display is a **single-replay demonstration**. The evidence summary is
loaded from the frozen 480-condition comparison in
`evidence/frozen_control_comparison.csv`; it is not recomputed from the
illustrative animation.

## Project map

| Path | Purpose |
|---|---|
| `run_demo.py` | The one command judges should run |
| `run_validation.py` | Fast simulator + four-TSRD integrity run; not selection evidence |
| `visualize.py` | Live dashboard, GIF, and frozen-evidence figure generation |
| `src/learner.py` | Selected CUSUM-UCB learner only |
| `src/engine.py` | Online scheduling loop and observation boundary |
| `src/receiver.py` | Detection, false-alarm, and retune physics |
| `src/scheduler.py` | Joint assignment and Max-Age guard |
| `src/periodic.py` | Confidence-gated timing evidence |
| `src/simulator.py` | Controlled emitter and adversarial frequency-time scenarios |
| `src/tsrd.py` | TSRD HDF5 replay adapter |
| `src/evaluator.py` | Post-run metrics; may access hidden truth |
| `data/tsrd/` | Four supplied TSRD datasets |
| `evidence/` | Frozen V3.1 result summaries and figures |
| `tests/` | Focused correctness and edge-case tests |
| `REQUIREMENTS_TRACEABILITY.md` | Frozen requirement-to-code/evidence audit |

## Frozen result—state it accurately

The predeclared selection rule recommends **RAFTS-CUSUM-UCB** as the
`ROBUSTNESS_FIRST_LEADER`. On all held-out runs it achieved 17.33% mean
interception ratio, 6.97% P10 interception ratio, 12.537-slot capped P95 TTI,
0.0906 net TP-minus-FP reward per receiver-time, 13.18% retune-dead fraction,
and zero observed coverage violations.

Do not call it universally best. Plain RAFTS-UCB was very close; Thompson
variants had higher mean interception but weaker lower-tail robustness and much
higher retune dead-time. The conclusion is conditional on the declared test
protocol, receiver model, datasets, seeds, and feasibility gates.

See [ALGORITHM.md](ALGORITHM.md), [RESULTS_AND_LIMITATIONS.md](RESULTS_AND_LIMITATIONS.md),
[REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md), and
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
