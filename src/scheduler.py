"""Deadline-safe, receiver-aware RAFTS scheduler."""

import itertools
import math
import numpy as np
from .config import MAX_AGE_SLOTS, RETUNE_DELAY_SLOTS


class MaxAgeGuard:
    """Coverage guard based only on completed receiver observations.

    The guard uses an earliest-deadline-first (EDF) admission rule.  A band is
    dispatched before its literal Max-Age deadline when the remaining service
    capacity is just sufficient to clear all older bands.  This accounts for
    retune dead-time instead of waiting until a band is already at the limit.
    """

    def __init__(self, num_bands, num_receivers, max_age=MAX_AGE_SLOTS,
                 retune_delay=RETUNE_DELAY_SLOTS):
        self.num_bands = int(num_bands)
        self.num_receivers = int(num_receivers)
        self.max_age = int(max_age)
        self.retune_delay = int(retune_delay)
        if min(self.num_bands, self.num_receivers, self.max_age) <= 0:
            raise ValueError("Invalid Max-Age dimensions.")
        # Worst case: every new observation requires retuning first.
        service_period = self.retune_delay + 1
        worst_revisit = math.ceil(self.num_bands / self.num_receivers) * service_period
        if worst_revisit > self.max_age:
            raise ValueError(
                f"Max-Age={self.max_age} is infeasible: conservative worst-case "
                f"revisit is {worst_revisit} slots for {self.num_bands} bands, "
                f"{self.num_receivers} receivers and retune delay {self.retune_delay}."
            )
        self.last_observed = np.full(self.num_bands, -1, dtype=int)

    @property
    def service_period(self):
        return self.retune_delay + 1

    def update_observation(self, band, slot):
        self.last_observed[int(band)] = int(slot)

    def ages(self, slot):
        slot = int(slot)
        return np.where(self.last_observed < 0, slot + 1, slot - self.last_observed)

    def priority_bands(self, slot, free_receivers):
        """Return EDF coverage bands that must be dispatched now.

        For rank k in oldest-first order, reserve enough time to service the
        k+1 oldest bands with the available receiver fleet. This creates a
        proactive deadline boundary and prevents the burst of overdue bands
        produced by a simple age-threshold guard.
        """
        free_receivers = max(0, int(free_receivers))
        if free_receivers == 0:
            return []
        ages = self.ages(slot)
        order = list(np.argsort(-ages))
        urgent = []
        for rank, band in enumerate(order):
            batches_ahead = math.ceil((rank + 1) / self.num_receivers)
            reserve = batches_ahead * self.service_period
            latest_safe_age = self.max_age - reserve
            if int(ages[band]) >= latest_safe_age:
                urgent.append(int(band))
        return urgent[:free_receivers]

    # Backwards-compatible name used by tests/older callers.
    def mandatory_bands(self, slot, free_receivers=None):
        n = self.num_receivers if free_receivers is None else free_receivers
        return self.priority_bands(slot, n)

    def violations(self, slot):
        return int(np.sum(self.ages(slot) > self.max_age))


def choose_joint_assignment(receivers, base_scores, mandatory, slot):
    """Exact small receiver x band assignment with physical retune cost."""
    n = len(base_scores)
    r = len(receivers)
    mandatory = list(dict.fromkeys(int(x) for x in mandatory))[:r]

    top_extra = list(np.argsort(base_scores)[::-1][: max(r * 5, r)])
    candidates = list(dict.fromkeys(mandatory + [int(x) for x in top_extra]))
    if len(candidates) < r:
        candidates.extend([b for b in range(n) if b not in candidates][:r-len(candidates)])

    best, best_u = None, -np.inf
    for bands in itertools.permutations(candidates, r):
        if not set(mandatory).issubset(set(bands)):
            continue
        u = 0.0
        for receiver, band in zip(receivers, bands):
            delay = 0 if receiver.current_band in (None, band) else receiver.retune_delay_slots
            u += float(base_scores[band]) / (1.0 + delay)
            if receiver.current_band == band:
                u += 1e-6
        if u > best_u:
            best_u, best = u, bands
    if best is None:
        raise RuntimeError("No feasible receiver-band assignment.")
    return [(receiver, int(band)) for receiver, band in zip(receivers, best)]
