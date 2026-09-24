"""Selected online learner for the final RAFTS prototype.

Only completed receiver HIT/MISS observations enter this learner.  Hidden
emitter activity, future timestamps, labels, AoA, amplitude, and transmitter
identity are never used for online decisions.
"""

import numpy as np

from .config import CUSUM_DRIFT, CUSUM_THRESHOLD, UCB_EXPLORATION


def _binary_observation(observation):
    observation = int(observation)
    if observation not in (0, 1):
        raise ValueError("Observation must be HIT=1 or MISS=0.")
    return observation


class CUSUMUCB:
    """UCB exploration with per-band two-sided CUSUM change resets.

    UCB balances exploitation of bands with high empirical HIT probability
    against exploration of uncertain bands.  CUSUM tracks positive and
    negative residuals.  Once a residual accumulator crosses the fixed
    threshold, stale evidence for that band is reset so the scheduler can
    adapt after an operating-pattern change.
    """

    def __init__(
        self,
        num_bands,
        seed_offset=0,
        exploration=UCB_EXPLORATION,
        drift=CUSUM_DRIFT,
        threshold=CUSUM_THRESHOLD,
    ):
        del seed_offset  # Deterministic under a fixed observation stream.
        self.num_bands = int(num_bands)
        self.exploration = float(exploration)
        self.drift = float(drift)
        self.threshold = float(threshold)
        if self.num_bands <= 0 or self.exploration <= 0:
            raise ValueError("Band count and UCB exploration must be positive.")
        if self.drift < 0 or self.threshold <= 0:
            raise ValueError("CUSUM drift must be non-negative and threshold positive.")

        self.counts = np.zeros(self.num_bands, dtype=float)
        self.hits = np.zeros(self.num_bands, dtype=float)
        self.positive = np.zeros(self.num_bands, dtype=float)
        self.negative = np.zeros(self.num_bands, dtype=float)
        self.change_count = np.zeros(self.num_bands, dtype=int)

    def update(self, band, slot, observation):
        del slot
        band = int(band)
        observation = _binary_observation(observation)
        if band < 0 or band >= self.num_bands:
            raise IndexError("band is outside the configured spectrum.")

        baseline = self.hits[band] / self.counts[band] if self.counts[band] else 0.5
        residual = observation - baseline
        self.positive[band] = max(
            0.0, self.positive[band] + residual - self.drift
        )
        self.negative[band] = max(
            0.0, self.negative[band] - residual - self.drift
        )

        if max(self.positive[band], self.negative[band]) >= self.threshold:
            self.counts[band] = 1.0
            self.hits[band] = float(observation)
            self.positive[band] = 0.0
            self.negative[band] = 0.0
            self.change_count[band] += 1
        else:
            self.counts[band] += 1.0
            self.hits[band] += observation

    def posterior_mean(self, slot=None):
        del slot
        return (1.0 + self.hits) / (2.0 + self.counts)

    def sample_scores(self, slot):
        del slot
        total = max(2.0, float(self.counts.sum()) + 1.0)
        bonus = np.sqrt(
            self.exploration * np.log(total) / np.maximum(1.0, self.counts)
        )
        scores = np.clip(self.posterior_mean() + bonus, 0.0, 1.0)
        scores[self.counts == 0] = 1.0
        return scores


def make_selected_learner(num_bands, seed_offset=0):
    """Construct the frozen selected learner."""
    return CUSUMUCB(num_bands=num_bands, seed_offset=seed_offset)
