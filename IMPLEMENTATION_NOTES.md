# Implementation / judge-defense notes

## Why Thompson Sampling?
The problem supplies sparse sequential HIT/MISS feedback and requires online
adaptation. Discounted Thompson Sampling is lightweight, uncertainty-aware and
forgets stale evidence without requiring a large labelled offline training set.

## Why Max-Age?
Pure exploitation can starve apparently unproductive bands. Max-Age is outside the learned score. A deadline-aware EDF guard reserves
receiver capacity early enough to include retune dead-time and forces revisit
before a configurable deadline.

## Why TSRD Stare?
Stare data are used as an externally generated hidden RF environment. RAFTS
then imposes its own limited-bandwidth scheduling decisions. This avoids
treating an already-scanned trace as if it were complete RF truth.

## Why not feed labels/AoA/amplitude?
The current claim is HIT/MISS-driven scheduling. Those richer fields remain
available for future observation-model extensions but are not privileged
scheduler inputs.

## Periodic handling
The temporal specialist remains OFF until minimum evidence is present and its
contribution is bounded. It is an evidence generator, not a mandatory stage.

## Transition evidence
Retained in code as an optional extension but disabled by default because
development ablation showed little incremental contribution. This is more
defensible than keeping a decorative module active.

## TTI
TTI = first successful detection slot - start slot of the corresponding
contiguous band-activity episode. Mean/P95 TTI are conditional on successful
interception and must always be shown together with Interception Ratio.

## Claims to avoid
Do not claim:
- universal guaranteed interception
- calibrated dBm sensitivity
- emitter identification/deinterleaving
- perfect frequency-hop prediction
- operational battlefield validation
