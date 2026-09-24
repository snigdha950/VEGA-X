# Results and limitations

## Held-out V3.1 result

| Metric | RAFTS-CUSUM-UCB |
|---|---:|
| Mean interception ratio | 0.173343 |
| P10 interception ratio | 0.069714 |
| Capped P95 TTI | 12.537 slots |
| Net TP−FP reward per receiver-time | 0.090615 |
| Retune dead fraction | 0.131756 |
| Coverage violations | 0 observed |
| Mean runtime | 0.742 ms/slot |
| P95 runtime | 2.925 ms/slot |
| CUSUM change detections | 828 |

Evidence covers 7,680 policy runs: 6,400 controlled/adversarial simulator runs
and 1,280 runs derived from all four TSRD files. The component ablation contains
1,100 additional complete runs.

## What can be claimed

- CUSUM-UCB was selected by the predeclared robustness-first protocol.
- The prototype used completed receiver HIT/MISS feedback only.
- It had zero observed coverage violations under the frozen feasible protocol.
- Its shell explicitly models receiver assignment and retune dead-time.
- Temporal evidence, joint assignment, Max-Age protection, and retune awareness
  were tested by component ablation.

## What must not be claimed

- It is not proven universally optimal or “perfect.”
- Zero observed violations are not a mathematical guarantee for every hardware
  configuration or environment.
- Plain UCB was extremely close; the CUSUM-versus-UCB mean advantage is small.
- Thompson variants had higher mean interception but weaker P10 robustness and
  substantially greater retune dead-time.
- On TSRD alone, Sequential had higher mean interception ratio and lower capped
  P95 TTI, while CUSUM-UCB delivered much higher reward/receiver-time and far
  lower retune dead fraction. Therefore do not claim dominance on every metric.
- `Pd`, `Pfa`, sensitivity, slot duration, and retune delay are prototype model
  parameters, not calibrated hardware measurements.
- The executable receiver uses one discretized band and one-slot dwell. The
  sensitivity value is reported as an assumption; the binary replay does not
  perform per-cell SNR thresholding.
- TSRD pulse data are aggregated into frequency-time occupancy cells; a cell is
  not a complete operational RF intercept scenario.
- The frozen selected configuration uses deterministic exact tie resolution
  and a bounded pre-deadline age term. Randomized Coverage Sweep is a control,
  not a feature that should be claimed as active inside selected CUSUM-UCB.
- Future-hazard TTI prediction error is not claimed. The reported intercept
  time error is onset-to-first-true-detection latency.

## Why the final package has only one learner

Sixteen policies were compared during selection. Keeping them in the final
runtime would make the prototype harder to inspect and could accidentally
change the frozen decision. Their results remain in `evidence/`; only the
selected learner is executable here. Three simple baselines remain solely to
make the dashboard understandable.
