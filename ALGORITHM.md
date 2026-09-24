# Selected algorithm: RAFTS-CUSUM-UCB

## Decision loop

At each time slot, RAFTS:

1. completes receiver observations that have arrived after retuning;
2. updates only from the binary receiver result: HIT or MISS;
3. updates the CUSUM-UCB state for the observed band;
4. applies confidence-gated temporal evidence when sufficient HIT history exists;
5. admits bands nearing the Max-Age limit using earliest-deadline-first logic;
6. scores receiver-band choices at their physical arrival time;
7. jointly assigns distinct bands while accounting for retune dead-time.
8. falls back to deadline-safe randomized coverage if a selected-learner
   decision is invalid, non-finite, numerically unsafe, or over a configured
   decision-time budget.

Hidden activity, future ToA/frequency, emitter labels, AoA, amplitude, and
transmitter identity never enter the online learner.

## UCB score

For band `b`, the exploitation term is its smoothed HIT estimate:

```text
mean_b = (1 + hits_b) / (2 + observations_b)
```

Its exploration bonus is:

```text
bonus_b = sqrt(2 log(total_observations + 1) / max(1, observations_b))
```

The bounded learner score is `clip(mean_b + bonus_b, 0, 1)`. Unobserved bands
receive score 1 so they cannot be permanently ignored.

## CUSUM adaptation

Each new binary observation is compared with the band’s current empirical HIT
rate. Positive and negative residual accumulators use drift `0.05`. If either
reaches threshold `4.0`, stale statistics for that band are reset to the new
observation and a change detection is recorded.

This is a practical change-reset mechanism; it is not a claim of optimal
change-point detection under every emitter process.

## Reward and cost

The online learner sees HIT/MISS, not hidden TP/FP labels. After a run, the
evaluator reports both legacy selection reward and unique-episode utility:

```text
net true reward = true intercepts - false alarms
net reward per receiver-time = net true reward / receiver-time
unique-episode reward per receiver-time = intercepted episodes / receiver-time
```

The equal `+1/-1` reward weights are transparent evaluation choices and were
used by the frozen V3.1 selection protocol. The unique-episode metric prevents
repeated HITs on one episode from producing unlimited value. Physical retune
time is included in receiver-time; neither metric is fed into the online
learner.

## Safety and feasibility shell

- **Max-Age guard:** prevents starvation and is checked against receiver count
  and retune delay before execution.
- **Joint assignment:** avoids assigning multiple free receivers to the same
  band and maximizes their combined receiver-aware utility.
- **Retune awareness:** divides opportunity value by service time when a band
  change incurs delay.
- **Temporal gate:** adds bounded evidence only after at least four observed
  HITs and adequate periodic confidence.
- **Safe fallback:** logs the failure and uses deadline-safe randomized
  coverage without exposing hidden truth.

The full component ablation is preserved in `evidence/component_ablation.png`.
Exact requirement status and evidence boundaries are recorded in
`REQUIREMENTS_TRACEABILITY.md`.
