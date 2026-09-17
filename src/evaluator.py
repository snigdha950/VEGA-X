"""Metrics aligned to SIH26055 terminology where operationally meaningful."""

import numpy as np


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


def evaluate(truth, observations, retunes, dead_slots, coverage_violations, max_observed_age):
    obs_by_band_slot = {}
    tp = fp = tn = fn = 0
    reward = 0.0

    for o in observations:
        key = (o.band, o.slot)
        obs_by_band_slot.setdefault(key, []).append(o)
        if o.truth_active and o.detected:
            tp += 1
            reward += 1.0
        elif o.truth_active and not o.detected:
            fn += 1
        elif not o.truth_active and o.detected:
            fp += 1
            reward -= 1.0
        else:
            tn += 1

    episodes = contiguous_episodes(truth)
    ttis = []
    intercepted = 0
    for band, start, end in episodes:
        first = None
        for t in range(start, end + 1):
            if any(o.detected and o.truth_active for o in obs_by_band_slot.get((band, t), [])):
                first = t
                break
        if first is not None:
            intercepted += 1
            ttis.append(first - start)

    ir = intercepted / len(episodes) if episodes else float("nan")
    pd_emp = tp / (tp + fn) if (tp + fn) else float("nan")
    pfa_emp = fp / (fp + tn) if (fp + tn) else float("nan")
    correct = (tp + tn) / max(1, tp + tn + fp + fn)

    # Intercept-time error is reported only for intercepted episodes and equals
    # onset-to-first-detection error under the frozen operational TTI definition.
    mean_tti = float(np.mean(ttis)) if ttis else float("nan")
    p95_tti = float(np.percentile(ttis, 95)) if ttis else float("nan")

    active_cells = int(np.sum(truth))
    detected_active_cells = len({
        (o.slot, o.band) for o in observations if o.truth_active and o.detected
    })
    avg_intercept_rate = detected_active_cells / active_cells if active_cells else float("nan")

    return {
        "episodes": len(episodes),
        "intercepted_episodes": intercepted,
        "interception_ratio": ir,
        "avg_intercept_rate": avg_intercept_rate,
        "conditional_mean_tti": mean_tti,
        "conditional_p95_tti": p95_tti,
        "avg_intercept_time_error": mean_tti,
        "empirical_pd": pd_emp,
        "empirical_pfa": pfa_emp,
        "correct_observation_fraction": correct,
        "avg_reward_per_observation": reward / max(1, len(observations)),
        "retunes": int(retunes),
        "retune_dead_slots": int(dead_slots),
        "coverage_violations": int(coverage_violations),
        "max_observed_age": int(max_observed_age),
        "observations": len(observations),
    }
