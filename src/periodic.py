"""Confidence-gated periodic timing evidence.

This is an optional evidence generator. It does not replace the general DTS
learner and remains inactive until enough receiver-observed HIT history exists.
"""

from collections import deque

import numpy as np

from .config import (
    NUM_BANDS,
    TEMPORAL_MAX_BONUS,
    TEMPORAL_MIN_CONFIDENCE,
    TEMPORAL_MIN_HITS,
    TEMPORAL_TOLERANCE_SLOTS,
)


class ConfidenceGatedTemporalPredictor:
    def __init__(
        self,
        num_bands=NUM_BANDS,
        min_hits=TEMPORAL_MIN_HITS,
        min_confidence=TEMPORAL_MIN_CONFIDENCE,
        max_bonus=TEMPORAL_MAX_BONUS,
        tolerance=TEMPORAL_TOLERANCE_SLOTS,
        history_size=20,
    ):
        self.num_bands = int(num_bands)
        self.min_hits = int(min_hits)
        self.min_confidence = float(min_confidence)
        self.max_bonus = float(max_bonus)
        self.tolerance = int(tolerance)
        if self.num_bands <= 0 or self.min_hits < 2 or history_size < self.min_hits:
            raise ValueError("Invalid temporal-predictor dimensions or history.")
        if (
            not 0 <= self.min_confidence <= 1
            or self.max_bonus < 0
            or self.tolerance < 0
        ):
            raise ValueError("Invalid temporal-predictor thresholds.")
        self.hit_slots = [deque(maxlen=history_size) for _ in range(self.num_bands)]

    def update(self, band, slot, observation):
        if int(observation) == 1:
            history = self.hit_slots[int(band)]
            slot = int(slot)
            # Multiple receivers can report the same band in one scheduler
            # slot.  That is one temporal event, not a zero-length period.
            if not history or history[-1] != slot:
                history.append(slot)

    def evidence(self, band, arrival_slot):
        h = np.asarray(self.hit_slots[int(band)], dtype=float)
        if h.size < self.min_hits:
            return 0.0, 0.0
        intervals = np.diff(h)
        # Histories are chronological and duplicate slots are coalesced in
        # update().  Keep this check as a defensive boundary for restored or
        # externally populated state.
        intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
        if intervals.size < self.min_hits - 1:
            return 0.0, 0.0
        period = float(np.median(intervals))
        if not np.isfinite(period) or period <= 0:
            return 0.0, 0.0
        mad = float(np.median(np.abs(intervals - period)))
        confidence = float(np.clip(1.0 - mad / max(period, 1.0), 0.0, 1.0))
        if confidence < self.min_confidence:
            return 0.0, confidence
        # Project to the nearest positive future cycle. Without this step, one
        # missed predicted event permanently disables evidence until an
        # incidental new HIT arrives.
        delta = float(arrival_slot) - h[-1]
        if not np.isfinite(delta):
            return 0.0, confidence
        cycles = max(1, round(delta / period))
        predicted = h[-1] + cycles * period
        distance = abs(float(arrival_slot) - predicted)
        if distance > self.tolerance:
            return 0.0, confidence
        closeness = 1.0 - distance / max(self.tolerance + 1.0, 1.0)
        return self.max_bonus * confidence * closeness, confidence
