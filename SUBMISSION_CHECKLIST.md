# Prototype submission checklist

## Before recording

- [ ] Use Python 3.12 and install `requirements.txt` in this folder.
- [ ] Run `audit.py` and confirm `PASS`.
- [ ] Run `python -m pytest -q` and confirm all tests pass.
- [ ] Run `run_demo.py` and open both generated PNGs and the GIF.
- [ ] Optionally run `run_validation.py`; label its output as smoke validation,
      not learner-selection evidence.
- [ ] Keep the project name expanded as **Receiver-Aware Frequency-Time Scheduler**.
- [ ] State that the selected learner is **CUSUM-UCB**.
- [ ] Keep `REQUIREMENTS_TRACEABILITY.md` with the prototype source.

## What to show in the video

- [ ] The problem: total spectrum is wider than instantaneous receiver bandwidth.
- [ ] The input boundary: online decisions use completed HIT/MISS observations only.
- [ ] The live frequency-time replay and two receiver assignments.
- [ ] Retune state, decision reason, top coverage ages, top beliefs, and CUSUM reset markers.
- [ ] The separate frozen-evidence summary and its transparent P95-TTI trade-off.
- [ ] The frozen held-out dashboard and component-ablation figure in `evidence/`.
- [ ] One honest limitation: prototype parameters are not calibrated hardware values.

## Files to keep together

- [ ] This flat project folder or its ZIP.
- [ ] PPT/PDF submission deck.
- [ ] Short demonstration video required by the current portal.
- [ ] Any portal forms, team details, declarations, and problem-statement ID.

Portal requirements can change. Verify the current national-screening page
before final upload; do not rely only on an older checklist.
