"""Common fair-comparison execution engine with physical retune dead-time.

`collect_trace=True` records judge-facing telemetry without changing scheduler
inputs or decisions. The trace is used only by visualize.py after each slot.
"""

import numpy as np
from .config import NUM_RECEIVERS, TRANSITION_ENABLED, RANDOM_SEED
from .receiver import Receiver
from .adaptive import DiscountedThompsonSampling
from .periodic import ConfidenceGatedTemporalPredictor
from .transition import DiscountedTransitionMemory
from .scheduler import MaxAgeGuard, choose_joint_assignment
from .evaluator import evaluate


def _make_receivers(seed_offset):
    return [Receiver(i, seed_offset=seed_offset) for i in range(NUM_RECEIVERS)]


def run_policy(truth, policy="rafts", seed_offset=0, collect_trace=False):
    truth = np.asarray(truth, dtype=bool)
    slots, bands = truth.shape
    receivers = _make_receivers(seed_offset)
    guard = MaxAgeGuard(bands, len(receivers))
    dts = DiscountedThompsonSampling(num_bands=bands, seed_offset=seed_offset)
    temporal = ConfidenceGatedTemporalPredictor(num_bands=bands)
    transition = DiscountedTransitionMemory(num_bands=bands)
    rng = np.random.default_rng(RANDOM_SEED + 4000 + seed_offset)

    observations = []
    coverage_violations = 0
    max_observed_age = 0
    seq_next = {r.receiver_id: r.receiver_id % bands for r in receivers}
    trace = [] if collect_trace else None

    def learn(obs):
        guard.update_observation(obs.band, obs.slot)
        if policy in ("dts", "rafts"):
            dts.update(obs.band, obs.slot, obs.detected)
        if policy == "rafts":
            temporal.update(obs.band, obs.slot, obs.detected)

    for slot in range(slots):
        slot_hit_bands = []
        slot_observations = []
        slot_assignments = []

        # 1) Receivers whose physical retune completes now MUST observe target.
        arrived_ids = set()
        for receiver in receivers:
            if receiver.arrival_ready(slot):
                obs = receiver.observe(truth[slot, receiver.current_band], slot)
                if obs is not None:
                    observations.append(obs)
                    slot_observations.append(obs)
                    learn(obs)
                    arrived_ids.add(receiver.receiver_id)
                    if obs.detected:
                        slot_hit_bands.append(obs.band)
                    if policy == "sequential":
                        seq_next[receiver.receiver_id] = (
                            seq_next[receiver.receiver_id] + len(receivers)
                        ) % bands

        # 2) Pending receivers consume this slot and cannot be reassigned.
        free = [
            r for r in receivers
            if (not r.pending) and r.receiver_id not in arrived_ids
        ]

        ages_before = guard.ages(slot)
        max_observed_age = max(max_observed_age, int(ages_before.max()))
        mandatory = guard.priority_bands(slot, len(free))
        mandatory.sort(key=lambda b: ages_before[b], reverse=True)

        if free:
            if policy == "sequential":
                assignments = [(r, seq_next[r.receiver_id]) for r in free]
            else:
                if policy == "random":
                    scores = rng.random(bands)
                elif policy == "dts":
                    scores = dts.sample_scores(slot)
                elif policy == "rafts":
                    scores = dts.sample_scores(slot)
                    for b in range(bands):
                        bonus, _ = temporal.evidence(b, slot)
                        scores[b] += bonus
                    if TRANSITION_ENABLED:
                        scores += transition.evidence_vector()
                else:
                    raise ValueError("policy must be sequential, random, dts or rafts")

                required = mandatory[:len(free)] if policy == "rafts" else []
                assignments = choose_joint_assignment(free, scores, required, slot)

            # 3) A stay observes now. A retune observes only on arrival.
            for receiver, band in assignments:
                old_band = receiver.current_band
                reason = "MAX-AGE" if policy == "rafts" and band in mandatory else (
                    "ADAPTIVE" if policy == "rafts" else policy.upper()
                )
                immediate = receiver.start_assignment(band, slot)
                slot_assignments.append({
                    "receiver": receiver.receiver_id,
                    "band": int(band),
                    "reason": reason,
                    "retune": bool(old_band is not None and old_band != band),
                    "immediate": bool(immediate),
                })
                if immediate:
                    obs = receiver.observe(truth[slot, band], slot)
                    if obs is not None:
                        observations.append(obs)
                        slot_observations.append(obs)
                        learn(obs)
                        if obs.detected:
                            slot_hit_bands.append(obs.band)
                        if policy == "sequential":
                            seq_next[receiver.receiver_id] = (
                                seq_next[receiver.receiver_id] + len(receivers)
                            ) % bands
                else:
                    receiver.dead_slots += 1

        if policy == "rafts" and TRANSITION_ENABLED:
            transition.update_hits(slot, slot_hit_bands)

        if policy == "rafts":
            coverage_violations += guard.violations(slot)

        if collect_trace:
            # Posterior mean is explanatory telemetry derived only from HIT/MISS.
            belief = dts.alpha / np.maximum(dts.alpha + dts.beta, 1e-12)
            receiver_state = [{
                "receiver": r.receiver_id,
                "band": -1 if r.current_band is None else int(r.current_band),
                "pending": bool(r.pending),
                "ready_slot": int(r.ready_slot),
            } for r in receivers]
            trace.append({
                "slot": int(slot),
                "ages": guard.ages(slot).astype(int).copy(),
                "mandatory": [int(x) for x in mandatory],
                "belief": belief.astype(float).copy(),
                "assignments": slot_assignments,
                "observations": [{
                    "receiver": int(o.receiver_id), "band": int(o.band),
                    "detected": int(o.detected), "truth_active": int(o.truth_active),
                    "false_alarm": int(o.false_alarm),
                } for o in slot_observations],
                "receivers": receiver_state,
                "retunes": int(sum(r.retune_count for r in receivers)),
                "coverage_violations": int(coverage_violations),
            })

    retunes = sum(r.retune_count for r in receivers)
    dead = sum(r.dead_slots for r in receivers)
    result = evaluate(
        truth, observations, retunes, dead,
        coverage_violations if policy == "rafts" else 0,
        max_observed_age,
    )
    if collect_trace:
        result["_trace"] = trace
    return result
