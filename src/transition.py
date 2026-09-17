"""Optional discounted successful-band transition evidence.

It is OFF by default because development ablation showed little incremental
benefit. The implementation is retained as a documented research extension.
It never assumes emitter identity.
"""

import numpy as np
from .config import (
    NUM_BANDS, TRANSITION_DISCOUNT, TRANSITION_MIN_SUPPORT,
    TRANSITION_MIN_CONFIDENCE, TRANSITION_MAX_BONUS
)


class DiscountedTransitionMemory:
    def __init__(self, num_bands=NUM_BANDS):
        self.n = int(num_bands)
        self.counts = np.zeros((self.n, self.n), dtype=float)
        self.last_band = None
        self.last_slot = None

    def update_hits(self, slot, hit_bands):
        self.counts *= TRANSITION_DISCOUNT
        unique = sorted(set(int(x) for x in hit_bands))
        if len(unique) != 1:
            self.last_band = None
            self.last_slot = None
            return
        b = unique[0]
        if self.last_band is not None and self.last_band != b:
            self.counts[self.last_band, b] += 1.0
        self.last_band, self.last_slot = b, int(slot)

    def evidence_vector(self):
        out = np.zeros(self.n, dtype=float)
        if self.last_band is None:
            return out
        row = self.counts[self.last_band]
        total = row.sum()
        if total < TRANSITION_MIN_SUPPORT:
            return out
        probs = row / max(total, 1e-12)
        best = float(probs.max())
        if best < TRANSITION_MIN_CONFIDENCE:
            return out
        return np.minimum(probs * TRANSITION_MAX_BONUS, TRANSITION_MAX_BONUS)
