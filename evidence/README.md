# Frozen evidence

These files are copied from the completed V3.1 held-out confirmation run. They
are read-only evidence, not inputs to the live scheduler.

- `SELECTION_REPORT.md`: protocol, scorecard, and ablation table.
- `learner_bakeoff_dashboard.png`: all learners and environments.
- `candidate_scorecard.csv`: gate and aggregate metrics.
- `component_ablation.png`: fixed-shell component evidence.
- `component_ablation_effects.csv`: paired effect estimates.
- `selection_report.json`: machine-readable decision record.
- `run_metadata.json`: seeds, run mode, and evidence status.
- `policy_summary.csv` and `environment_summary.csv`: grouped summaries.
- `frozen_control_comparison.csv`: four-policy, 480-condition values displayed
  in the evidence summary, including deterministic bootstrap confidence
  limits for mean interception ratio; derived from the frozen raw held-out
  results.

The confidence limits use 20,000 deterministic bootstrap resamples with seed
2026. Mean interception ratio is computed over defined episode-bearing runs;
undefined no-emitter IR values remain excluded, matching the frozen aggregate.

The frozen field `mean_reward_per_receiver_time` is the evaluated net
TP-minus-FP reward divided by physical receiver-time. The final runtime also
reports a separate unique-episode reward/receiver-time metric; it is an added
diagnostic and was not substituted into the frozen learner-selection table.

The full raw 7,680-run CSV is intentionally excluded from the clean demo ZIP.
It remains in the separately preserved validation bundle.
