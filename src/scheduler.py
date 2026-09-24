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

    def __init__(
        self,
        num_bands,
        num_receivers,
        max_age=MAX_AGE_SLOTS,
        retune_delay=RETUNE_DELAY_SLOTS,
        validate_feasibility=True,
    ):
        if any(
            int(value) != value
            for value in (num_bands, num_receivers, max_age, retune_delay)
        ):
            raise ValueError("Max-Age dimensions and retune delay must be integers.")
        self.num_bands = int(num_bands)
        self.num_receivers = int(num_receivers)
        self.max_age = int(max_age)
        self.retune_delay = int(retune_delay)
        if min(self.num_bands, self.num_receivers, self.max_age) <= 0:
            raise ValueError("Invalid Max-Age dimensions.")
        if self.retune_delay < 0:
            raise ValueError("retune_delay must be non-negative.")
        # Worst case: every new observation requires retuning first.
        service_period = self.retune_delay + 1
        worst_revisit = math.ceil(self.num_bands / self.num_receivers) * service_period
        self.required_max_age = worst_revisit
        if validate_feasibility and worst_revisit > self.max_age:
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

    def priority_bands(self, slot, free_receivers, excluded_bands=()):
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
        excluded = {int(b) for b in excluded_bands}
        order = [int(b) for b in np.argsort(-ages) if int(b) not in excluded]
        critical_rank = -1
        # Use capacity available *now*, not the total installed receiver count.
        # When one receiver is still retuning, dividing the backlog by the full
        # fleet can defer an old band by one slot and violate the deadline.
        active_capacity = min(self.num_receivers, free_receivers)
        for rank, band in enumerate(order):
            batches_ahead = math.ceil((rank + 1) / active_capacity)
            reserve = batches_ahead * self.service_period
            latest_safe_age = self.max_age - reserve
            if int(ages[band]) >= latest_safe_age:
                critical_rank = rank

        if critical_rank < 0:
            return []
        # If a lower-ranked band reaches its admission boundary, every older
        # band ahead of it is part of the EDF backlog. Dispatch the oldest
        # available prefix; selecting only bands that individually crossed
        # their thresholds can skip an older band and miss its deadline.
        return order[: critical_rank + 1][:free_receivers]

    # Backwards-compatible name used by tests/older callers.
    def mandatory_bands(self, slot, free_receivers=None):
        n = self.num_receivers if free_receivers is None else free_receivers
        return self.priority_bands(slot, n)

    def violations(self, slot):
        return int(np.sum(self.ages(slot) > self.max_age))


def choose_joint_assignment(
    receivers,
    base_scores,
    mandatory,
    slot,
    excluded_bands=(),
    retune_aware=True,
):
    """Exact small receiver x band assignment with physical retune cost.

    ``base_scores`` may be one score per band or a receiver-by-band matrix.
    A matrix lets frequency-time evidence be evaluated at each receiver's
    physical arrival time. ``excluded_bands`` prevents duplicate dispatch to
    a band already targeted by a receiver in retune dead-time.
    """
    del slot  # Retained in the public signature for older callers.
    scores = np.asarray(base_scores, dtype=float)
    r = len(receivers)
    if r == 0:
        return []
    if scores.ndim == 1:
        n = scores.shape[0]
        scores = np.repeat(scores[np.newaxis, :], r, axis=0)
    elif scores.ndim == 2 and scores.shape[0] == r:
        n = scores.shape[1]
    else:
        raise ValueError("base_scores must have shape (bands,) or (receivers, bands).")
    if n < r:
        raise ValueError("At least one distinct band per free receiver is required.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Assignment scores must be finite.")

    excluded = {int(x) for x in excluded_bands}
    mandatory = list(dict.fromkeys(int(x) for x in mandatory))[:r]
    if any(b < 0 or b >= n for b in excluded | set(mandatory)):
        raise ValueError("Mandatory and excluded bands must be valid band indices.")
    if excluded.intersection(mandatory):
        raise ValueError("A mandatory band cannot also be excluded.")

    # Always include each receiver's current band. Round-1 omitted these from the
    # candidate set unless they happened to be top-ranked, which could force a
    # retune even when staying and observing immediately had better physical value.
    current = [
        int(rx.current_band)
        for rx in receivers
        if rx.current_band is not None and int(rx.current_band) not in excluded
    ]
    aggregate = scores.max(axis=0)
    top_extra = [int(x) for x in np.argsort(aggregate)[::-1] if int(x) not in excluded][
        : max(r * 5, r)
    ]
    candidates = list(dict.fromkeys(mandatory + current + top_extra))
    if len(candidates) < r:
        candidates.extend(
            [b for b in range(n) if b not in candidates and b not in excluded][
                : r - len(candidates)
            ]
        )
    if len(candidates) < r:
        raise RuntimeError("Not enough unreserved bands for the free receivers.")

    best, best_u = None, -np.inf
    for bands in itertools.permutations(candidates, r):
        if not set(mandatory).issubset(set(bands)):
            continue
        u = 0.0
        for receiver_index, (receiver, band) in enumerate(zip(receivers, bands)):
            delay = (
                0
                if receiver.current_band in (None, band)
                else receiver.retune_delay_slots
            )
            service_time = 1.0 + delay if retune_aware else 1.0
            u += float(scores[receiver_index, band]) / service_time
            if retune_aware and receiver.current_band == band:
                u += 1e-6
        if u > best_u:
            best_u, best = u, bands
    if best is None:
        raise RuntimeError("No feasible receiver-band assignment.")
    return [(receiver, int(band)) for receiver, band in zip(receivers, best)]


def choose_greedy_assignment(
    receivers,
    base_scores,
    mandatory,
    slot,
    excluded_bands=(),
    retune_aware=True,
):
    """Greedy receiver-by-receiver comparator for the joint-assignment ablation.

    It preserves distinct-band and mandatory-coverage constraints, but does not
    optimize the receiver set globally. It is not used by the full RAFTS shell.
    """
    del slot
    scores = np.asarray(base_scores, dtype=float)
    receiver_count = len(receivers)
    if receiver_count == 0:
        return []
    if scores.ndim == 1:
        band_count = scores.shape[0]
        scores = np.repeat(scores[np.newaxis, :], receiver_count, axis=0)
    elif scores.ndim == 2 and scores.shape[0] == receiver_count:
        band_count = scores.shape[1]
    else:
        raise ValueError("base_scores must have shape (bands,) or (receivers, bands).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Assignment scores must be finite.")

    excluded = {int(band) for band in excluded_bands}
    required = list(dict.fromkeys(int(band) for band in mandatory))[:receiver_count]
    if any(band < 0 or band >= band_count for band in excluded | set(required)):
        raise ValueError("Mandatory and excluded bands must be valid band indices.")
    if excluded.intersection(required):
        raise ValueError("A mandatory band cannot also be excluded.")

    chosen = []
    used = set(excluded)
    for receiver_index, receiver in enumerate(receivers):
        available_required = [band for band in required if band not in used]
        available = (
            available_required
            if available_required
            else [band for band in range(band_count) if band not in used]
        )
        if not available:
            raise RuntimeError("Not enough unreserved bands for greedy assignment.")

        def utility(band, receiver=receiver, receiver_index=receiver_index):
            delay = (
                0
                if receiver.current_band in (None, band)
                else receiver.retune_delay_slots
            )
            service_time = 1.0 + delay if retune_aware else 1.0
            stay_bonus = 1e-6 if retune_aware and receiver.current_band == band else 0.0
            return float(scores[receiver_index, band]) / service_time + stay_bonus

        band = max(available, key=utility)
        chosen.append((receiver, int(band)))
        used.add(int(band))
    return chosen
