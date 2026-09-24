"""Physical receiver abstraction shared by every compared policy."""

import math
from dataclasses import dataclass

import numpy as np

from .config import PD, PFA, RANDOM_SEED, RETUNE_DELAY_SLOTS, SENSITIVITY_DB


def _probability_profile(value, name):
    profile = np.asarray(value, dtype=float)
    if profile.ndim == 0:
        probability = float(profile)
        if not 0 <= probability <= 1:
            raise ValueError(f"{name} must be in [0, 1].")
        return probability
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError(
            f"{name} must be a scalar or a non-empty one-dimensional profile."
        )
    if not np.all(np.isfinite(profile)) or np.any((profile < 0) | (profile > 1)):
        raise ValueError(f"Every {name} profile value must be in [0, 1].")
    return profile.copy()


@dataclass
class Observation:
    """One completed receiver observation.

    ``truth_active`` and ``false_alarm`` are evaluation-only fields. The
    scheduler is updated with ``detected`` (HIT/MISS) only.
    """

    slot: int
    receiver_id: int
    band: int
    truth_active: int
    detected: int
    false_alarm: int
    predicted_hit_probability: float = math.nan


class Receiver:
    """Single-band receiver with deterministic common-random-number noise."""

    _MASK64 = (1 << 64) - 1

    def __init__(
        self,
        receiver_id,
        pd=PD,
        pfa=PFA,
        sensitivity_db=SENSITIVITY_DB,
        retune_delay_slots=RETUNE_DELAY_SLOTS,
        seed_offset=0,
    ):
        if int(retune_delay_slots) != retune_delay_slots or retune_delay_slots < 0:
            raise ValueError("retune_delay_slots must be a non-negative integer.")
        if int(seed_offset) != seed_offset or seed_offset < 0:
            raise ValueError("seed_offset must be a non-negative integer.")

        if int(receiver_id) != receiver_id or receiver_id < 0:
            raise ValueError("receiver_id must be a non-negative integer.")
        self.receiver_id = int(receiver_id)
        self.pd = _probability_profile(pd, "Pd")
        self.pfa = _probability_profile(pfa, "Pfa")
        self.sensitivity_db = float(sensitivity_db)
        if not np.isfinite(self.sensitivity_db):
            raise ValueError("sensitivity_db must be finite.")
        self.retune_delay_slots = int(retune_delay_slots)
        self.current_band = None
        self.ready_slot = 0
        self.pending = False
        self.retune_count = 0
        self.dead_slots = 0
        self.noise_seed = int(RANDOM_SEED + 1000 + self.receiver_id + seed_offset)

    def start_assignment(self, band, slot):
        """Assign a target; return ``True`` only if it can be observed now."""
        band, slot = int(band), int(slot)
        if band < 0 or slot < 0:
            raise ValueError("band and slot must be non-negative.")
        if self.pending:
            raise RuntimeError("Cannot issue a new assignment during retune dead-time.")
        if self.current_band is None:
            self.current_band = band
            self.ready_slot = slot
            return True
        if band == self.current_band:
            self.ready_slot = slot
            return True
        self.current_band = band
        self.retune_count += 1
        self.ready_slot = slot + self.retune_delay_slots
        self.pending = self.retune_delay_slots > 0
        return not self.pending

    def arrival_ready(self, slot):
        return self.pending and int(slot) >= self.ready_slot

    def consume_dead_slot(self):
        """Account for one receiver-time slot lost to retuning."""
        if not self.pending:
            raise RuntimeError("A non-pending receiver has no retune dead slot.")
        self.dead_slots += 1

    def _uniform_draw(self, slot):
        """Return a stable draw keyed by seed, receiver, slot, and band.

        Common random numbers make the sensor outcome for a given
        receiver/band/slot identical across compared policies. This removes
        avoidable benchmark noise from policy-dependent RNG consumption.
        """
        x = (
            self.noise_seed
            + (int(slot) + 1) * 0x9E3779B97F4A7C15
            + (int(self.current_band) + 1) * 0xD1B54A32D192ED03
        ) & self._MASK64
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & self._MASK64
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & self._MASK64
        x ^= x >> 31
        return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _band_probability(self, profile):
        if np.isscalar(profile):
            return float(profile)
        band = int(self.current_band)
        if band >= len(profile):
            raise ValueError(
                "Sensor probability profile is shorter than the band count."
            )
        return float(profile[band])

    def observe(self, truth_active, slot):
        slot = int(slot)
        if self.current_band is None or slot < self.ready_slot:
            if self.pending:
                self.consume_dead_slot()
            return None
        self.pending = False
        truth_active = bool(truth_active)
        draw = self._uniform_draw(slot)
        if truth_active:
            detected = int(draw < self._band_probability(self.pd))
            false_alarm = 0
        else:
            detected = int(draw < self._band_probability(self.pfa))
            false_alarm = detected
        return Observation(
            slot,
            self.receiver_id,
            int(self.current_band),
            int(truth_active),
            detected,
            false_alarm,
        )
