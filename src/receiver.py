"""Physical receiver abstraction used by all compared policies."""

from dataclasses import dataclass
import numpy as np
from .config import PD, PFA, SENSITIVITY_DB, RETUNE_DELAY_SLOTS, RANDOM_SEED


@dataclass
class Observation:
    slot: int
    receiver_id: int
    band: int
    truth_active: int
    detected: int
    false_alarm: int


class Receiver:
    def __init__(self, receiver_id, pd=PD, pfa=PFA, sensitivity_db=SENSITIVITY_DB,
                 retune_delay_slots=RETUNE_DELAY_SLOTS, seed_offset=0):
        if not 0 <= pd <= 1 or not 0 <= pfa <= 1:
            raise ValueError("Pd and Pfa must be in [0,1].")
        self.receiver_id = int(receiver_id)
        self.pd = float(pd)
        self.pfa = float(pfa)
        self.sensitivity_db = float(sensitivity_db)
        self.retune_delay_slots = int(retune_delay_slots)
        self.current_band = None
        self.ready_slot = 0
        self.pending = False
        self.retune_count = 0
        self.dead_slots = 0
        self.rng = np.random.default_rng(RANDOM_SEED + 1000 + self.receiver_id + seed_offset)

    def start_assignment(self, band, slot):
        """Assign a target. Return True only when it can be observed immediately."""
        band, slot = int(band), int(slot)
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

    def observe(self, truth_active, slot):
        slot = int(slot)
        if self.current_band is None or slot < self.ready_slot:
            self.dead_slots += 1
            return None
        self.pending = False
        truth_active = bool(truth_active)
        if truth_active:
            detected = int(self.rng.random() < self.pd)
            false_alarm = 0
        else:
            detected = int(self.rng.random() < self.pfa)
            false_alarm = detected
        return Observation(slot, self.receiver_id, int(self.current_band),
                           int(truth_active), detected, false_alarm)
