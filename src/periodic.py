"""Confidence-gated periodic timing evidence.

This is an optional evidence generator. It does not replace the general DTS
learner and remains inactive until enough receiver-observed HIT history exists.
"""

from collections import deque
import numpy as np
from .config import (
    NUM_BANDS, TEMPORAL_MIN_HITS, TEMPORAL_MIN_CONFIDENCE,
    TEMPORAL_MAX_BONUS, TEMPORAL_TOLERANCE_SLOTS
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
        self.hit_slots = [deque(maxlen=history_size) for _ in range(self.num_bands)]

    def update(self, band, slot, observation):
        if int(observation) == 1:
            self.hit_slots[int(band)].append(int(slot))

    def evidence(self, band, arrival_slot):
        h = np.asarray(self.hit_slots[int(band)], dtype=float)
        if h.size < self.min_hits:
            return 0.0, 0.0
        intervals = np.diff(h)
        if intervals.size < self.min_hits - 1 or np.mean(intervals) <= 0:
            return 0.0, 0.0
        period = float(np.median(intervals))
        mad = float(np.median(np.abs(intervals - period)))
        confidence = float(np.clip(1.0 - mad / max(period, 1.0), 0.0, 1.0))
        if confidence < self.min_confidence:
            return 0.0, confidence
        predicted = h[-1] + period
        distance = abs(float(arrival_slot) - predicted)
        if distance > self.tolerance:
            return 0.0, confidence
        closeness = 1.0 - distance / max(self.tolerance + 1.0, 1.0)
        return self.max_bonus * confidence * closeness, confidence
