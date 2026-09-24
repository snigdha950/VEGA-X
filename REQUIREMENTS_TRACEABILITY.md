# Frozen-solution traceability audit

## Audit basis

This file maps the final prototype to **RAFTS — Final Frozen Solution and
Coding Specification**, frozen 23 September 2026 for SIH26055. It also records
the evidence-based learner decision made after that specification: the V3.1
held-out bake-off selected **RAFTS-CUSUM-UCB**.

Status meanings:

- **Implemented:** executable in this package.
- **Validated:** covered by tests or frozen held-out evidence.
- **Evidence only:** implementation was exercised in the preserved selection
  run but intentionally removed from the judge-facing runtime.
- **Intentional exclusion:** the frozen design said not to enable it without
  evidence.
- **Prototype limitation:** accurately disclosed and not claimed as complete.

## Problem, scope, and information boundary

| Frozen requirement | Status | Prototype location/evidence |
|---|---|---|
| Receiver-Aware Frequency-Time Scheduler identity | Implemented | `README.md`, `src/engine.py` |
| Passive scheduling only; no transmit/jam/threat classification claim | Implemented | `README.md`, `RESULTS_AND_LIMITATIONS.md` |
| Which band, when, and which receiver | Implemented | `src/engine.py`, `src/scheduler.py` |
| Strict online input is completed receiver HIT/MISS plus receiver state/history | Implemented and validated | `src/engine.py`; audit verifies one hidden-truth receiver boundary |
| Future pulses, labels, AoA, amplitude, identity, and hidden state forbidden online | Implemented and validated | `src/tsrd.py`, `src/learner.py`, tests |
| Hidden truth used only by receiver simulation/replay and evaluator | Implemented and validated | `src/receiver.py`, `src/evaluator.py`, `audit.py` |
| Unknown bands begin without threat-priority intelligence | Implemented | Symmetric CUSUM-UCB priors in `src/learner.py` |
| Optional external mission weights reported separately | Intentional exclusion | Track B is not implemented or claimed in this strict prototype |

## Receiver physics and scheduling shell

| Frozen requirement | Status | Prototype location/evidence |
|---|---|---|
| Narrow instantaneous receiver bandwidth | Implemented abstraction | One discretized band per receiver per dwell |
| Fixed/configuration-selected V1 dwell | Implemented | Fixed one-slot dwell; adaptive dwell is excluded |
| Physical retune delay and no observation during retune | Implemented and validated | `src/receiver.py`; dead-slot regression tests |
| Receiver-time includes observation plus retune-dead slots | Implemented and validated | `src/evaluator.py` |
| Configurable Pd and Pfa, including band profiles | Implemented and validated | `src/receiver.py`; edge tests and frozen stress cases |
| Sensitivity parameter | Prototype limitation | Stored and reported as a modeled assumption; binary occupancy replay has no per-cell SNR thresholding |
| Preflight Max-Age feasibility | Implemented and validated | Conservative worst-case check in `MaxAgeGuard` |
| Proactive earliest-deadline/oldest-first coverage | Implemented and validated | `src/scheduler.py`; zero feasible violations in 480 selected-policy runs |
| Joint assignment of distinct bands | Implemented and validated | Exact small assignment in `src/scheduler.py`; duplicate/in-flight tests |
| Retune-aware opportunity per service time | Implemented | Learner/temporal score divided by dwell-plus-retune service time |
| Safe fallback after invalid/numerical/over-budget decision | Implemented and validated | Deadline-safe randomized coverage, `fallback_count`, trace reason; budget enforced when configured |
| Delayed/distributed coordination stress | Prototype limitation | Centralized two-receiver coordinator is the declared prototype assumption |

## Learning and temporal behavior

| Frozen requirement | Status | Prototype location/evidence |
|---|---|---|
| Replaceable learners tested in one fixed shell | Evidence only | 13 adaptive learners plus three controls in preserved V3.1 evidence |
| Final learner chosen by evidence, not by project name | Validated | `evidence/selection_report.json` selects RAFTS-CUSUM-UCB |
| CUSUM-UCB receives binary observations only | Implemented and validated | `src/learner.py`, tests |
| Change recovery/reset diagnostic | Implemented and validated | Two-sided CUSUM and per-band reset trace |
| Confidence-gated periodic evidence | Implemented and validated | `src/periodic.py`; duplicate-hit and zero-period protections |
| Temporal evidence bounded and inactive without enough evidence | Implemented and validated | Four-hit/confidence gate and bounded bonus |
| Successful-band transition evidence not enabled without held-out support | Intentional exclusion | Transition module absent; exclusion recorded in evidence report |
| Deep RL, POMCP/MCTS, PSR, and Whittle methods screened before inclusion | Evidence only / intentional exclusion | Selection documentation records feasibility rejection; none is falsely claimed as tested |
| Oracle is evaluation-only if used | Intentional exclusion | No oracle is present in the executable prototype or selection claim |

## Data, scenarios, and fair evidence

| Frozen requirement | Status | Prototype location/evidence |
|---|---|---|
| Controlled simulator is mandatory | Implemented | `src/simulator.py`; 14 nominal/adversarial families |
| Sensor/retune stresses | Implemented and validated | Six profiles in `run_validation.py`; frozen evidence used the same six |
| Periodic, random, bursty, changing, agile, jittered, spatial visibility | Implemented and validated | `src/simulator.py`, frozen environment summary |
| No-emitter, all-active, dense, activation/deactivation, transition, sync trap | Implemented and validated | `src/simulator.py`, tests, frozen environment summary |
| All four TSRD-derived scheduling replays | Implemented and validated | `data/tsrd/`, `src/tsrd.py`, inventory tests |
| ToA to slot and carrier frequency to band | Implemented | `src/tsrd.py` |
| Unobserved future TSRD pulses remain hidden | Implemented | Replay constructs hidden truth; scheduler receives selected HIT/MISS only |
| Paired hidden worlds/noise/horizons | Validated | Frozen V3.1 run IDs, deterministic receiver noise, selection report |
| Held-out final evidence with uncertainty | Validated | 7,680 runs, 480 conditions/policy, paired selection, confidence intervals |
| Publish hard cases even when controls win | Validated | Complete environment summary retained; limitations explicitly mention TSRD cases won by Sequential |
| Quick runs not presented as selection evidence | Implemented | `run_validation.py` labels outputs `SMOKE_DIAGNOSTIC_NOT_SELECTION_EVIDENCE` |

## Reward, cost, metrics, and ablations

| Frozen requirement | Status | Prototype location/evidence |
|---|---|---|
| Online learner reward is HIT=1/MISS=0 | Implemented | `src/learner.py` |
| Observed HITs per receiver-time | Implemented | `observed_hit_rate_per_receiver_time` |
| Net TP-minus-FP evaluation reward per receiver-time | Implemented and validated | Frozen V3.1 selection metric; explicitly named in new outputs |
| Unique intercepted episodes per receiver-time | Implemented | `unique_episode_reward_per_receiver_time`; repeated HITs count once |
| Episode interception ratio and no-intercept rate | Implemented and validated | `src/evaluator.py`, frozen evidence |
| Conditional mean/median/P95 TTI | Implemented | `src/evaluator.py` |
| Non-intercept-aware capped/restricted TTI | Implemented and validated | Missed episode contributes its full eligible duration |
| Empirical Pd/Pfa and detector correctness | Implemented | `src/evaluator.py` |
| Prediction calibration and classification | Implemented | Brier, log loss, accuracy, balanced accuracy, precision, recall |
| Average intercept-time error wording | Implemented with boundary | Reported as onset-to-first-detection error; no unsupported future-hazard forecast claim |
| Coverage, age, retune, runtime, duplicate, fallback diagnostics | Implemented | CSV/JSON outputs and trace |
| Temporal/Max-Age/retune/joint-assignment ablations | Validated | `evidence/component_ablation*` |
| Deterministic-vs-randomized initial-coverage ablation | Prototype limitation | Not part of the completed V3.1 evidence and not claimed |

## Evidence-preserving implementation boundaries

The audit deliberately does **not** change the normal selected-policy decision
path or frozen numerical evidence. Two pre-coding ideals differ from the
evaluated V3.1 implementation and must be described accurately:

1. The V3.1 shell contains a bounded pre-deadline age-shaping term in addition
   to hard EDF admission. Therefore describe the implementation as “hard
   Max-Age protection plus bounded age evidence,” not as a purely unshaped
   `p_hit / cost` rule.
2. The selected V3.1 assignment resolves exact utility ties deterministically.
   Randomized permutation is implemented as the Coverage Sweep control and the
   selected learner was tested on `sync_trap`, but randomized tie-breaking is
   not active in the frozen selected configuration. Enabling it now would
   require a new full paired held-out comparison.

These are transparent evidence boundaries, not silently patched algorithm
changes. The executable corrections in this audit are non-selection changes:
simulator restoration, exceptional fallback, additional metrics, metadata,
tests, visualization labels, and documentation.

## Final claim boundary

The defensible claim is that RAFTS is a prototype receiver-aware
frequency-time scheduling framework with receiver-observable learning,
deadline-safe coverage, retune physics, coordinated receivers, controlled RF
simulation, TSRD-derived replay, and a frozen evidence-based CUSUM-UCB choice.
It is not hardware-calibrated, universally optimal, operationally deployed, or
a threat-identification system.
