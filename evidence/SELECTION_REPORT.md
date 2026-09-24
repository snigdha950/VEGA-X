# RAFTS learner-selection report

- Evidence status: **FRESH_CONFIRMATION_HELDOUT_EVIDENCE_V3**
- Mode: **final**
- Selection result: **ROBUSTNESS_FIRST_LEADER**
- Recommended fixed-shell learner: **RAFTS-CUSUM-UCB**

This is not a claim of a universally perfect algorithm. The recommendation is conditional on the declared simulator, TSRD replay, receiver physics, seeds, and feasibility gates.

## Candidate scorecard

| Candidate | Gate | Mechanism | Mean IR | P10 IR | Capped P95 TTI | Reward/time | Dead fraction | Runtime ms/slot |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| RAFTS-SWUCB | PASS | PASS | 0.1694 | 0.0702 | 12.513 | 0.0829 | 0.1308 | 0.782 |
| RAFTS-CUSUM-UCB | PASS | PASS | 0.1733 | 0.0697 | 12.537 | 0.0906 | 0.1318 | 0.742 |
| RAFTS-UCB | PASS | PASS | 0.1721 | 0.0691 | 12.501 | 0.0905 | 0.1314 | 0.738 |
| RAFTS-DUCB | PASS | PASS | 0.1635 | 0.0656 | 12.774 | 0.0788 | 0.1297 | 1.723 |
| RAFTS-Temporal | PASS | PASS | 0.1470 | 0.0615 | 12.278 | 0.0702 | 0.1295 | 0.706 |
| RAFTS-CP-TS | PASS | PASS | 0.2058 | 0.0538 | 11.176 | 0.1451 | 0.3297 | 0.541 |
| RAFTS-TS | PASS | PASS | 0.2058 | 0.0538 | 11.164 | 0.1428 | 0.3293 | 0.510 |
| RAFTS-SWTS | PASS | PASS | 0.1805 | 0.0474 | 10.691 | 0.1393 | 0.3650 | 1.237 |
| RAFTS-DTS | PASS | PASS | 0.1777 | 0.0431 | 11.248 | 0.1387 | 0.3713 | 0.515 |
| RAFTS-EXP3.S | PASS | PASS | 0.1585 | 0.0379 | 7.914 | 0.0464 | 0.5091 | 0.387 |
| RAFTS-GLR-KL-UCB* | FAIL | PASS | 0.2052 | 0.0651 | 11.373 | 0.1277 | 0.1487 | 4.578 |
| RAFTS-KL-UCB | FAIL | PASS | 0.2062 | 0.0648 | 11.370 | 0.1275 | 0.1486 | 9.870 |
| RAFTS-LinUCB | FAIL | PASS | 0.2029 | 0.0625 | 11.713 | 0.1248 | 0.1718 | 2.052 |

## Selection protocol

1. validity, mechanism evidence, and zero feasible coverage violations.
2. paired-bootstrap 10th-percentile interception ratio.
3. paired-bootstrap mean interception ratio.
4. paired-bootstrap mean capped P95 time-to-intercept.
5. paired-bootstrap true reward per receiver-time.
6. paired-bootstrap retune dead-time fraction.
7. paired-bootstrap runtime, then declared simplicity for remaining ties.

Statistically indistinguishable leaders remain tied. A tie is resolved only by the predeclared simplicity/runtime rule; it is not described as superior performance.
Temporal and change-point candidates are ineligible unless their defining mechanism activates at least once in the evaluated runs.

## Observation boundary

Every learner receives completed receiver HIT/MISS observations only. Hidden activity and TP/FP labels are restricted to replay construction and post-run evaluation.

## Fixed-shell component ablation

`NO_MAX_AGE` removes both the EDF admission rule and its urgency bonus. All rows retain the same physical receiver model and selected learner.

| Source | Configuration | Mean IR | Capped P95 TTI | Reward/time | Dead fraction | Coverage violations/run |
|---|---|---:|---:|---:|---:|---:|
| all | FULL | 0.1764 | 11.147 | 0.0969 | 0.1141 | 0.00 |
| all | GREEDY_ASSIGNMENT | 0.1491 | 11.257 | 0.0953 | 0.1164 | 0.00 |
| all | NO_MAX_AGE | 0.2695 | 14.670 | 0.1314 | 0.0108 | 8360.75 |
| all | NO_TEMPORAL | 0.1537 | 11.382 | 0.0744 | 0.1104 | 0.00 |
| all | RETUNE_UNAWARE | 0.2479 | 8.536 | 0.1721 | 0.3197 | 0.00 |
| simulator | FULL | 0.1141 | 3.424 | 0.0131 | 0.1125 | 0.00 |
| simulator | GREEDY_ASSIGNMENT | 0.1171 | 3.436 | 0.0133 | 0.1150 | 0.00 |
| simulator | NO_MAX_AGE | 0.1675 | 3.443 | 0.0268 | 0.0121 | 9135.06 |
| simulator | NO_TEMPORAL | 0.1082 | 3.424 | 0.0117 | 0.1116 | 0.00 |
| simulator | RETUNE_UNAWARE | 0.1300 | 3.424 | 0.0155 | 0.3927 | 0.00 |
| tsrd | FULL | 0.2854 | 24.663 | 0.2437 | 0.1167 | 0.00 |
| tsrd | GREEDY_ASSIGNMENT | 0.2050 | 24.942 | 0.2388 | 0.1188 | 0.00 |
| tsrd | NO_MAX_AGE | 0.4480 | 34.318 | 0.3144 | 0.0086 | 7005.71 |
| tsrd | NO_TEMPORAL | 0.2332 | 25.309 | 0.1842 | 0.1082 | 0.00 |
| tsrd | RETUNE_UNAWARE | 0.4541 | 17.482 | 0.4462 | 0.1919 | 0.00 |

## Important implementation boundary

Transition-memory code is disabled and excluded from the frozen architecture and evidence. The GLR-KL-UCB* label denotes a bounded-history, per-band implementation inspired by GLR-klUCB; it is not a claim of theorem-equivalent reproduction.
