"""Execution engine for the selected RAFTS-CUSUM-UCB prototype.

Every policy receives the same hidden replay and the same deterministic sensor
noise for a given seed/receiver/band/slot. Schedulers receive completed
HIT/MISS observations only; hidden truth is accessed solely at the receiver
observation boundary and by the post-run evaluator.
"""

from time import perf_counter

import numpy as np

from .config import (
    MAX_AGE_SLOTS,
    NUM_RECEIVERS,
    PD,
    PFA,
    RANDOM_SEED,
    RETUNE_DELAY_SLOTS,
    SENSITIVITY_DB,
)
from .evaluator import evaluate
from .learner import make_selected_learner
from .periodic import ConfidenceGatedTemporalPredictor
from .receiver import Receiver
from .scheduler import MaxAgeGuard, choose_greedy_assignment, choose_joint_assignment

CONTROL_POLICIES = ("sequential", "random", "coverage_sweep")
SELECTED_POLICY = "rafts_cusum_ucb"
COMPATIBILITY_ALIASES = ("rafts", "rafts_v3")
POLICIES = (*CONTROL_POLICIES, SELECTED_POLICY, *COMPATIBILITY_ALIASES)
DEMO_POLICIES = (*CONTROL_POLICIES, SELECTED_POLICY)
RAFTS_POLICIES = {SELECTED_POLICY, *COMPATIBILITY_ALIASES}


def _validated_truth(truth):
    truth = np.asarray(truth, dtype=bool)
    if truth.ndim != 2:
        raise ValueError("truth must be a two-dimensional [slot, band] array.")
    if truth.shape[0] <= 0 or truth.shape[1] <= 0:
        raise ValueError("truth must contain at least one slot and one band.")
    return truth


def run_policy(
    truth,
    policy=SELECTED_POLICY,
    seed_offset=0,
    collect_trace=False,
    sw_window=60,
    max_age=MAX_AGE_SLOTS,
    persistence_bonus=0.0,
    persistence_horizon=4,
    urgency_bonus=0.05,
    urgency_start=0.60,
    num_receivers=NUM_RECEIVERS,
    pd=PD,
    pfa=PFA,
    sensitivity_db=SENSITIVITY_DB,
    retune_delay_slots=RETUNE_DELAY_SLOTS,
    temporal_enabled=True,
    coverage_guard_enabled=True,
    retune_aware=True,
    joint_assignment=True,
    decision_budget_seconds=None,
):
    """Run one policy under common observation and receiver physics.

    RAFTS combines CUSUM-UCB with receiver-aware assignment, a Max-Age guard,
    and bounded confidence-gated temporal and urgency evidence.
    """
    started = perf_counter()
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    truth = _validated_truth(truth)
    slots, bands = truth.shape
    if int(num_receivers) != num_receivers:
        raise ValueError("num_receivers must be an integer.")
    num_receivers = int(num_receivers)
    if num_receivers <= 0 or num_receivers > bands:
        raise ValueError("num_receivers must be between 1 and the number of bands.")
    if not 0.0 <= urgency_start < 1.0:
        raise ValueError("urgency_start must be in [0, 1).")
    if decision_budget_seconds is not None and (
        not np.isfinite(decision_budget_seconds) or decision_budget_seconds <= 0
    ):
        raise ValueError("decision_budget_seconds must be positive when provided.")
    for name, enabled in (
        ("temporal_enabled", temporal_enabled),
        ("coverage_guard_enabled", coverage_guard_enabled),
        ("retune_aware", retune_aware),
        ("joint_assignment", joint_assignment),
    ):
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
    if int(persistence_horizon) != persistence_horizon:
        raise ValueError("persistence_horizon must be an integer.")
    persistence_horizon = int(persistence_horizon)
    if urgency_bonus < 0 or persistence_bonus < 0 or persistence_horizon < 0:
        raise ValueError("Urgency and persistence parameters must be non-negative.")
    for name, profile in (("Pd", pd), ("Pfa", pfa)):
        values = np.asarray(profile)
        if values.ndim == 1 and values.size != bands:
            raise ValueError(f"{name} profile length must equal the number of bands.")

    is_rafts = policy in RAFTS_POLICIES
    receivers = [
        Receiver(
            i,
            pd=pd,
            pfa=pfa,
            sensitivity_db=sensitivity_db,
            retune_delay_slots=retune_delay_slots,
            seed_offset=seed_offset,
        )
        for i in range(num_receivers)
    ]
    guard = MaxAgeGuard(
        bands,
        len(receivers),
        max_age=max_age,
        retune_delay=retune_delay_slots,
        validate_feasibility=is_rafts and coverage_guard_enabled,
    )
    del sw_window  # Retained only for compatibility with older callers.
    learner = make_selected_learner(bands, seed_offset) if is_rafts else None
    temporal = ConfidenceGatedTemporalPredictor(num_bands=bands)
    rng = np.random.default_rng(RANDOM_SEED + 7000 + int(seed_offset))

    observations = []
    coverage_violations = 0
    max_observed_age = 0
    last_hit = np.full(bands, -10_000, dtype=int)
    seq_next = {
        receiver.receiver_id: receiver.receiver_id % bands for receiver in receivers
    }
    coverage_order = list(rng.permutation(bands))
    coverage_position = 0
    trace = [] if collect_trace else None
    temporal_evidence_activations = 0
    temporal_bonus_total = 0.0
    fallback_count = 0
    duplicate_assignments = 0

    def coverage_assignments(free_receivers, excluded_bands):
        nonlocal coverage_order, coverage_position
        chosen = []
        excluded = {int(band) for band in excluded_bands}
        for receiver in free_receivers:
            attempts = 0
            while attempts < bands * 2:
                if coverage_position >= bands:
                    coverage_order = list(rng.permutation(bands))
                    coverage_position = 0
                band = int(coverage_order[coverage_position])
                coverage_position += 1
                attempts += 1
                if band not in excluded:
                    chosen.append((receiver, band))
                    excluded.add(band)
                    break
            else:
                raise RuntimeError("Randomized coverage sweep found no free band.")
        return chosen

    def fallback_assignments(free_receivers, required_bands, excluded_bands):
        """Deadline-safe randomized coverage used after learner failure."""
        excluded = {int(band) for band in excluded_bands}
        required = [
            int(band)
            for band in dict.fromkeys(required_bands)
            if int(band) not in excluded
        ][: len(free_receivers)]
        chosen = [
            (receiver, band)
            for receiver, band in zip(free_receivers[: len(required)], required)
        ]
        excluded.update(required)
        remaining = free_receivers[len(required) :]
        chosen.extend(coverage_assignments(remaining, excluded))
        return chosen

    def posterior_probability(slot, band):
        if learner is None:
            return float("nan")
        return float(learner.posterior_mean(slot)[int(band)])

    def learn(observation):
        # Any completed observation resets coverage age, including a MISS.
        guard.update_observation(observation.band, observation.slot)
        if learner is not None:
            learner.update(observation.band, observation.slot, observation.detected)
        if is_rafts and temporal_enabled:
            temporal.update(observation.band, observation.slot, observation.detected)
        if is_rafts and observation.detected:
            last_hit[int(observation.band)] = int(observation.slot)

    def complete_observation(receiver, slot, slot_observations):
        # This is the only online execution boundary that reads hidden truth.
        band = int(receiver.current_band)
        predicted = posterior_probability(slot, band)
        observation = receiver.observe(truth[slot, band], slot)
        if observation is None:
            return
        observation.predicted_hit_probability = predicted
        observations.append(observation)
        slot_observations.append(observation)
        learn(observation)
        if policy == "sequential":
            seq_next[receiver.receiver_id] = (
                seq_next[receiver.receiver_id] + len(receivers)
            ) % bands

    for slot in range(slots):
        slot_observations = []
        slot_assignments = []
        fallback_reason = None

        # Receivers arriving after a retune must observe before any reassignment.
        arrived_ids = set()
        for receiver in receivers:
            if receiver.arrival_ready(slot):
                complete_observation(receiver, slot, slot_observations)
                arrived_ids.add(receiver.receiver_id)

        # Count every intermediate retune slot, not only the dispatch slot.
        for receiver in receivers:
            if receiver.pending and not receiver.arrival_ready(slot):
                receiver.consume_dead_slot()

        free = [
            receiver
            for receiver in receivers
            if not receiver.pending and receiver.receiver_id not in arrived_ids
        ]
        inflight_bands = {
            int(receiver.current_band) for receiver in receivers if receiver.pending
        }

        ages_before = guard.ages(slot)
        max_observed_age = max(max_observed_age, int(ages_before.max()))
        mandatory = (
            guard.priority_bands(
                slot,
                len(free),
                excluded_bands=inflight_bands,
            )
            if is_rafts and coverage_guard_enabled
            else []
        )
        mandatory.sort(key=lambda band: ages_before[band], reverse=True)

        if free:
            if policy == "sequential":
                assignments = [
                    (receiver, seq_next[receiver.receiver_id]) for receiver in free
                ]
            elif policy == "coverage_sweep":
                assignments = coverage_assignments(free, inflight_bands)
            elif policy == "random":
                score_matrix = np.repeat(
                    rng.random(bands)[np.newaxis, :], len(free), axis=0
                )
                assignments = choose_joint_assignment(
                    free,
                    score_matrix,
                    [],
                    slot,
                    excluded_bands=inflight_bands,
                    retune_aware=retune_aware,
                )
            else:
                decision_started = perf_counter()
                try:
                    with np.errstate(all="raise"):
                        base_scores = np.asarray(
                            learner.sample_scores(slot), dtype=float
                        )
                        if base_scores.shape != (bands,) or not np.all(
                            np.isfinite(base_scores)
                        ):
                            raise ValueError(
                                "learner returned invalid or non-finite band scores"
                            )
                        score_matrix = np.repeat(
                            base_scores[np.newaxis, :], len(free), axis=0
                        )

                        for receiver_index, receiver in enumerate(free):
                            for band in range(bands):
                                physical_delay = (
                                    0
                                    if receiver.current_band in (None, band)
                                    else receiver.retune_delay_slots
                                )
                                planning_delay = physical_delay if retune_aware else 0
                                arrival_slot = slot + planning_delay
                                if temporal_enabled:
                                    temporal_bonus = temporal.evidence(
                                        band, arrival_slot
                                    )[0]
                                    score_matrix[receiver_index, band] += temporal_bonus
                                    if temporal_bonus > 0:
                                        temporal_evidence_activations += 1
                                        temporal_bonus_total += float(temporal_bonus)

                                dt = arrival_slot - last_hit[band]
                                if (
                                    persistence_bonus > 0
                                    and 0 < dt <= persistence_horizon
                                ):
                                    score_matrix[receiver_index, band] += (
                                        persistence_bonus
                                        * (1.0 - dt / (persistence_horizon + 1.0))
                                    )

                                if coverage_guard_enabled:
                                    projected_age = ages_before[band] + planning_delay
                                    fraction = min(
                                        1.0,
                                        projected_age / max(1.0, float(max_age)),
                                    )
                                    if fraction > urgency_start:
                                        z = (fraction - urgency_start) / (
                                            1.0 - urgency_start
                                        )
                                        score_matrix[receiver_index, band] += (
                                            urgency_bonus * z * z
                                        )

                        required = mandatory[: len(free)]
                        assignment_function = (
                            choose_joint_assignment
                            if joint_assignment
                            else choose_greedy_assignment
                        )
                        assignments = assignment_function(
                            free,
                            score_matrix,
                            required,
                            slot,
                            excluded_bands=inflight_bands,
                            retune_aware=retune_aware,
                        )
                    elapsed = perf_counter() - decision_started
                    if (
                        decision_budget_seconds is not None
                        and elapsed > decision_budget_seconds
                    ):
                        raise TimeoutError(
                            f"decision took {elapsed:.6f}s, above "
                            f"{decision_budget_seconds:.6f}s budget"
                        )
                except (
                    ArithmeticError,
                    FloatingPointError,
                    OverflowError,
                    RuntimeError,
                    TimeoutError,
                    ValueError,
                ) as exc:
                    fallback_count += 1
                    fallback_reason = f"{type(exc).__name__}: {exc}"[:200]
                    assignments = fallback_assignments(
                        free,
                        mandatory if coverage_guard_enabled else [],
                        inflight_bands,
                    )

            assigned_bands = [int(band) for _, band in assignments]
            duplicate_assignments += len(assigned_bands) - len(set(assigned_bands))

            for receiver, band in assignments:
                old_band = receiver.current_band
                if is_rafts and band in mandatory:
                    reason = "MAX-AGE"
                elif is_rafts:
                    reason = "RAFTS"
                else:
                    reason = policy.upper()
                immediate = receiver.start_assignment(band, slot)
                slot_assignments.append(
                    {
                        "receiver": receiver.receiver_id,
                        "band": int(band),
                        "reason": reason,
                        "retune": bool(old_band is not None and old_band != band),
                        "immediate": bool(immediate),
                    }
                )
                if immediate:
                    complete_observation(receiver, slot, slot_observations)
                else:
                    receiver.consume_dead_slot()

        # All policies are measured against the same threshold. Only RAFTS
        # uses the Max-Age guard to influence its decisions.
        coverage_violations += guard.violations(slot)

        if collect_trace:
            belief = (
                learner.posterior_mean(slot).astype(float).copy()
                if learner is not None
                else np.full(bands, np.nan)
            )
            receiver_state = [
                {
                    "receiver": receiver.receiver_id,
                    "band": (
                        -1
                        if receiver.current_band is None
                        else int(receiver.current_band)
                    ),
                    "pending": bool(receiver.pending),
                    "ready_slot": int(receiver.ready_slot),
                }
                for receiver in receivers
            ]
            trace.append(
                {
                    "slot": int(slot),
                    "ages": guard.ages(slot).astype(int).copy(),
                    "mandatory": [int(x) for x in mandatory],
                    "inflight_bands": sorted(inflight_bands),
                    "belief": belief,
                    "assignments": slot_assignments,
                    "observations": [
                        {
                            "receiver": int(observation.receiver_id),
                            "band": int(observation.band),
                            "detected": int(observation.detected),
                            "truth_active": int(observation.truth_active),
                            "false_alarm": int(observation.false_alarm),
                        }
                        for observation in slot_observations
                    ],
                    "receivers": receiver_state,
                    "retunes": int(sum(r.retune_count for r in receivers)),
                    "coverage_violations": int(coverage_violations),
                    "fallback": fallback_reason is not None,
                    "fallback_reason": fallback_reason,
                    "change_detections": int(
                        np.sum(
                            getattr(
                                learner,
                                "change_count",
                                np.zeros(1, dtype=int),
                            )
                        )
                    ),
                    # Visualization-only diagnostic. This is copied after the
                    # decision and observation update; it never feeds back
                    # into the learner or scheduler.
                    "change_count_by_band": getattr(
                        learner,
                        "change_count",
                        np.zeros(bands, dtype=int),
                    ).astype(int).copy(),
                }
            )

    result = evaluate(
        truth,
        observations,
        sum(receiver.retune_count for receiver in receivers),
        sum(receiver.dead_slots for receiver in receivers),
        coverage_violations,
        max_observed_age,
    )
    result["runtime_seconds"] = perf_counter() - started
    result["temporal_evidence_activations"] = int(temporal_evidence_activations)
    result["temporal_bonus_total"] = float(temporal_bonus_total)
    result["learner_change_detections"] = int(
        np.sum(getattr(learner, "change_count", np.zeros(1, dtype=int)))
    )
    result["fallback_count"] = int(fallback_count)
    result["duplicate_assignments"] = int(duplicate_assignments)
    if collect_trace:
        result["_trace"] = trace
    return result
