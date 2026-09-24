"""Metrics aligned to SIH26055 terminology where operationally meaningful."""

import numpy as np

from .config import FALSE_ALARM_COST, TRUE_INTERCEPT_REWARD


def contiguous_episodes(truth):
    episodes = []
    slots, bands = truth.shape
    for b in range(bands):
        start = None
        for t in range(slots):
            active = bool(truth[t, b])
            if active and start is None:
                start = t
            if start is not None and (not active or t == slots - 1):
                end = t if active and t == slots - 1 else t - 1
                episodes.append((b, start, end))
                start = None
    return episodes


def evaluate(
    truth, observations, retunes, dead_slots, coverage_violations, max_observed_age
):
    obs_by_band_slot = {}
    tp = fp = tn = fn = 0
    reward = 0.0
    observed_hits = 0
    predictions = []

    for o in observations:
        key = (o.band, o.slot)
        obs_by_band_slot.setdefault(key, []).append(o)
        observed_hits += int(o.detected)
        if np.isfinite(o.predicted_hit_probability):
            predictions.append((float(o.predicted_hit_probability), int(o.detected)))
        if o.truth_active and o.detected:
            tp += 1
            reward += TRUE_INTERCEPT_REWARD
        elif o.truth_active and not o.detected:
            fn += 1
        elif not o.truth_active and o.detected:
            fp += 1
            reward -= FALSE_ALARM_COST
        else:
            tn += 1

    episodes = contiguous_episodes(truth)
    ttis = []
    capped_ttis = []
    intercepted = 0
    for band, start, end in episodes:
        first = None
        for t in range(start, end + 1):
            if any(
                o.detected and o.truth_active
                for o in obs_by_band_slot.get((band, t), [])
            ):
                first = t
                break
        if first is not None:
            intercepted += 1
            latency = first - start
            ttis.append(latency)
            capped_ttis.append(latency)
        else:
            # A miss is assigned the episode duration. This prevents conditional
            # TTI from looking artificially good when most episodes are missed.
            capped_ttis.append(end - start + 1)

    ir = intercepted / len(episodes) if episodes else float("nan")
    no_intercept_rate = 1.0 - ir if episodes else float("nan")
    pd_emp = tp / (tp + fn) if (tp + fn) else float("nan")
    pfa_emp = fp / (fp + tn) if (fp + tn) else float("nan")
    correct = (tp + tn) / max(1, tp + tn + fp + fn)

    # Intercept-time error is reported only for intercepted episodes and equals
    # onset-to-first-detection error under the frozen operational TTI definition.
    mean_tti = float(np.mean(ttis)) if ttis else float("nan")
    median_tti = float(np.median(ttis)) if ttis else float("nan")
    p95_tti = float(np.percentile(ttis, 95)) if ttis else float("nan")
    capped_mean_tti = float(np.mean(capped_ttis)) if capped_ttis else float("nan")
    capped_p95_tti = (
        float(np.percentile(capped_ttis, 95)) if capped_ttis else float("nan")
    )

    active_cells = int(np.sum(truth))
    detected_active_cells = len(
        {(o.slot, o.band) for o in observations if o.truth_active and o.detected}
    )
    avg_intercept_rate = (
        detected_active_cells / active_cells if active_cells else float("nan")
    )
    receiver_time = len(observations) + int(dead_slots)

    if predictions:
        predicted = np.asarray([p for p, _ in predictions], dtype=float)
        outcomes = np.asarray([y for _, y in predictions], dtype=float)
        predicted = np.clip(predicted, 1e-12, 1.0 - 1e-12)
        brier = float(np.mean((predicted - outcomes) ** 2))
        log_loss = float(
            -np.mean(
                outcomes * np.log(predicted) + (1 - outcomes) * np.log(1 - predicted)
            )
        )
        predicted_labels = predicted >= 0.5
        actual_labels = outcomes.astype(bool)
        prediction_tp = int(np.sum(predicted_labels & actual_labels))
        prediction_fp = int(np.sum(predicted_labels & ~actual_labels))
        prediction_tn = int(np.sum(~predicted_labels & ~actual_labels))
        prediction_fn = int(np.sum(~predicted_labels & actual_labels))
        prediction_accuracy = float(np.mean(predicted_labels == actual_labels))
        prediction_recall = (
            prediction_tp / (prediction_tp + prediction_fn)
            if prediction_tp + prediction_fn
            else float("nan")
        )
        prediction_specificity = (
            prediction_tn / (prediction_tn + prediction_fp)
            if prediction_tn + prediction_fp
            else float("nan")
        )
        prediction_balanced_accuracy = (
            0.5 * (prediction_recall + prediction_specificity)
            if np.isfinite(prediction_recall)
            and np.isfinite(prediction_specificity)
            else float("nan")
        )
        prediction_precision = (
            prediction_tp / (prediction_tp + prediction_fp)
            if prediction_tp + prediction_fp
            else float("nan")
        )
    else:
        brier = log_loss = float("nan")
        prediction_accuracy = prediction_balanced_accuracy = float("nan")
        prediction_precision = prediction_recall = float("nan")

    unique_episode_reward = float(intercepted) * TRUE_INTERCEPT_REWARD
    net_reward_per_receiver_time = reward / max(1, receiver_time)

    return {
        "episodes": len(episodes),
        "intercepted_episodes": intercepted,
        "interception_ratio": ir,
        "no_intercept_rate": no_intercept_rate,
        "avg_intercept_rate": avg_intercept_rate,
        "conditional_mean_tti": mean_tti,
        "conditional_median_tti": median_tti,
        "conditional_p95_tti": p95_tti,
        "capped_mean_tti": capped_mean_tti,
        "capped_p95_tti": capped_p95_tti,
        # SIH wording retained as a compatibility alias. This is measured
        # onset-to-first-detection latency, not a future-hazard forecast error.
        "avg_intercept_time_error": mean_tti,
        "onset_to_first_detection_error": mean_tti,
        "empirical_pd": pd_emp,
        "empirical_pfa": pfa_emp,
        "correct_observation_fraction": correct,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "observed_hits": int(observed_hits),
        "net_true_reward": float(reward),
        "unique_episode_reward": unique_episode_reward,
        "avg_reward_per_observation": reward / max(1, len(observations)),
        # Cost is physical receiver time consumed: observation slots + retune-dead slots.
        # No arbitrary weights are introduced.
        "avg_reward_per_receiver_time": net_reward_per_receiver_time,
        "net_true_reward_per_receiver_time": net_reward_per_receiver_time,
        # Legacy field retained because the frozen V3.1 evidence selected the
        # learner with this transparent TP-minus-FP evaluation reward.
        "true_reward_per_receiver_time": net_reward_per_receiver_time,
        # Frozen-solution compliance metric: unique intercepted episodes count
        # once, false alarms contribute zero, and cost is receiver-time.
        "unique_episode_reward_per_receiver_time": unique_episode_reward
        / max(1, receiver_time),
        "unique_interceptions_per_wall_slot": intercepted / max(1, truth.shape[0]),
        "unique_interceptions_per_receiver_time": intercepted
        / max(1, receiver_time),
        "observed_hit_rate_per_receiver_time": observed_hits / max(1, receiver_time),
        "receiver_time_slots": int(receiver_time),
        "useful_observation_slots": int(len(observations)),
        "learner_brier_score": brier,
        "learner_log_loss": log_loss,
        "learner_prediction_accuracy": prediction_accuracy,
        "learner_prediction_balanced_accuracy": prediction_balanced_accuracy,
        "learner_prediction_precision": prediction_precision,
        "learner_prediction_recall": prediction_recall,
        "learner_prediction_threshold": 0.5,
        "retunes": int(retunes),
        "retune_dead_slots": int(dead_slots),
        "retune_dead_fraction": int(dead_slots) / max(1, receiver_time),
        "coverage_violations": int(coverage_violations),
        "coverage_violation_band_slots": int(coverage_violations),
        "max_observed_age": int(max_observed_age),
        "observations": len(observations),
    }
