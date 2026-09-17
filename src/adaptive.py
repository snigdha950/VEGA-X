"""Discounted Thompson Sampling for non-stationary HIT/MISS learning."""

import numpy as np
from .config import (
    NUM_BANDS, DTS_DISCOUNT, DTS_PRIOR_ALPHA, DTS_PRIOR_BETA, RANDOM_SEED
)


class DiscountedThompsonSampling:
    def __init__(
        self,
        num_bands=NUM_BANDS,
        discount=DTS_DISCOUNT,
        prior_alpha=DTS_PRIOR_ALPHA,
        prior_beta=DTS_PRIOR_BETA,
        seed_offset=0,
    ):
        self.num_bands = int(num_bands)
        self.discount = float(discount)
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.alpha = np.full(self.num_bands, self.prior_alpha, dtype=float)
        self.beta = np.full(self.num_bands, self.prior_beta, dtype=float)
        self.last_update_slot = np.zeros(self.num_bands, dtype=int)
        self.rng = np.random.default_rng(RANDOM_SEED + 3000 + seed_offset)

    def _discount_band(self, band, slot):
        elapsed = max(0, int(slot) - int(self.last_update_slot[band]))
        if elapsed:
            factor = self.discount ** elapsed
            self.alpha[band] = self.prior_alpha + factor * (self.alpha[band] - self.prior_alpha)
            self.beta[band] = self.prior_beta + factor * (self.beta[band] - self.prior_beta)
            self.last_update_slot[band] = int(slot)

    def update(self, band, slot, observation):
        band, slot = int(band), int(slot)
        observation = int(observation)
        if observation not in (0, 1):
            raise ValueError("Observation must be HIT=1 or MISS=0.")
        self._discount_band(band, slot)
        if observation:
            self.alpha[band] += 1.0
        else:
            self.beta[band] += 1.0

    def sample_scores(self, slot):
        scores = np.empty(self.num_bands, dtype=float)
        for b in range(self.num_bands):
            self._discount_band(b, slot)
            scores[b] = self.rng.beta(self.alpha[b], self.beta[b])
        return scores
