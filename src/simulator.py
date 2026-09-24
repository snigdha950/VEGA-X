"""Controlled hidden RF scenarios for prototype validation and stress tests."""

import numpy as np

SCENARIOS = (
    "periodic",
    "random",
    "bursty",
    "changing",
    "agile",
    "jittered_periodic",
    "spatial_agile",
    "no_emitters",
    "all_active",
    "dense",
    "quiet_to_active",
    "active_to_silent",
    "periodic_to_random",
    "sync_trap",
)


def _anchor_bands(bands, count=4):
    return np.unique(np.linspace(0, bands - 1, min(count, bands), dtype=int))


def make_scenario(name, slots=500, bands=20, seed=0):
    """Return a hidden boolean ``[slot, band]`` activity grid.

    The grid is supplied to the receiver boundary. It is never exposed to the
    online learner or assignment logic.
    """
    if int(slots) != slots or int(bands) != bands or slots <= 0 or bands <= 0:
        raise ValueError("slots and bands must be positive integers.")
    if name not in SCENARIOS:
        raise ValueError(f"name must be one of {SCENARIOS}")

    slots, bands = int(slots), int(bands)
    rng = np.random.default_rng(10_000 + int(seed))
    truth = np.zeros((slots, bands), dtype=bool)
    anchors = _anchor_bands(bands)

    if name == "periodic":
        for index, band in enumerate(anchors):
            period = 18 + 4 * index
            for start in range(index, slots, period):
                truth[start : min(slots, start + 4), band] = True
    elif name == "random":
        truth = rng.random((slots, bands)) < 0.035
    elif name == "bursty":
        for _ in range(min(55, max(1, slots * bands))):
            band = int(rng.integers(bands))
            start = int(rng.integers(slots))
            width = int(rng.integers(1, min(8, slots - start) + 1))
            truth[start : start + width, band] = True
    elif name == "changing":
        midpoint = slots // 2
        early_band = round(0.2 * (bands - 1))
        late_band = round(0.8 * (bands - 1))
        for start in range(1, midpoint, 26):
            truth[start : min(midpoint, start + 4), early_band] = True
        for start in range(midpoint, slots, 17):
            truth[start : min(slots, start + 3), late_band] = True
        truth |= rng.random((slots, bands)) < 0.006
    elif name == "agile":
        band = 0
        for start in range(0, slots, 5):
            band = (band + int(rng.integers(1, min(7, bands) + 1))) % bands
            truth[start : min(slots, start + 2), band] = True
    elif name == "jittered_periodic":
        for index, band in enumerate(_anchor_bands(bands, count=3)):
            start = index
            while start < slots:
                truth[start : min(slots, start + 3), band] = True
                start += max(2, 24 + int(rng.integers(-4, 5)))
    elif name == "spatial_agile":
        band = 0
        for start in range(0, slots, 4):
            band = (band + int(rng.integers(1, min(7, bands) + 1))) % bands
            visible = ((start // 20) % 3) != 1
            if visible:
                truth[start : min(slots, start + 2), band] = True
    elif name == "no_emitters":
        pass
    elif name == "all_active":
        truth[:] = True
    elif name == "dense":
        truth = rng.random((slots, bands)) < 0.40
    elif name == "quiet_to_active":
        truth[slots // 2 :] = rng.random((slots - slots // 2, bands)) < 0.15
    elif name == "active_to_silent":
        truth[: slots // 2] = rng.random((slots // 2, bands)) < 0.15
    elif name == "periodic_to_random":
        midpoint = slots // 2
        for index, band in enumerate(anchors):
            period = 18 + 4 * index
            for start in range(index, midpoint, period):
                truth[start : min(midpoint, start + 3), band] = True
        truth[midpoint:] = rng.random((slots - midpoint, bands)) < 0.06
    elif name == "sync_trap":
        # Sparse emissions fall between a rigid sweep's visits, exposing
        # synchronization sensitivity without using responsive emitters.
        period = max(3, bands)
        target = int(anchors[-1])
        for start in range(1, slots, period):
            truth[start, target] = True

    return truth
