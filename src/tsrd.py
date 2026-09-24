"""TSRD HDF5 adapter.

Important separation:
- TSRD pulse data are used to construct the hidden RF replay environment.
- RAFTS never receives future ToA/frequency, labels, or transmitter metadata.
- The scheduler receives only receiver observations (HIT/MISS).

The Alan Turing Institute TSRD stores PDWs with feature names including
ToA (microseconds), Frequency (MHz), PulseWidth, AoA and Amplitude.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .config import FREQ_MAX_MHZ, FREQ_MIN_MHZ, NUM_BANDS, NUM_SLOTS


@dataclass
class TSRDReplay:
    truth: np.ndarray
    pulse_counts: np.ndarray
    slot_edges_us: np.ndarray
    band_edges_mhz: np.ndarray
    source_file: str
    pulses_loaded: int
    pulses_in_band: int
    feature_names: list
    observed_freq_min_mhz: float
    observed_freq_max_mhz: float
    observed_toa_min_us: float
    observed_toa_max_us: float

    @property
    def num_slots(self):
        return int(self.truth.shape[0])

    @property
    def num_bands(self):
        return int(self.truth.shape[1])


def _decode_names(raw):
    return [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in raw
    ]


def _validated_layout(h5_file):
    if "data" not in h5_file or "metadata/feature_names" not in h5_file:
        raise ValueError(
            "Not a supported TSRD file: missing data or metadata/feature_names."
        )
    data = h5_file["data"]
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("TSRD data must be a non-empty two-dimensional dataset.")
    names = _decode_names(h5_file["metadata/feature_names"][:])
    if len(names) != data.shape[1]:
        raise ValueError(
            "TSRD feature_names length does not match the number of data columns."
        )
    if len(set(names)) != len(names):
        raise ValueError("TSRD feature names must be unique.")
    return data, names


def inspect_tsrd(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        data, names = _validated_layout(f)
        shape = tuple(data.shape)
        receiver_range = None
        if "metadata/receiver/freq_range_mhz" in f:
            receiver_range = (
                f["metadata/receiver/freq_range_mhz"][:].astype(float).tolist()
            )
    return {
        "file": str(path),
        "shape": shape,
        "feature_names": names,
        "receiver_freq_range_mhz": receiver_range,
    }


def load_tsrd_replay(
    path,
    num_slots=NUM_SLOTS,
    num_bands=NUM_BANDS,
    freq_min_mhz=FREQ_MIN_MHZ,
    freq_max_mhz=FREQ_MAX_MHZ,
    min_pulses_per_cell=1,
):
    """Convert pulse-level TSRD PDWs to hidden frequency x time activity.

    One RAFTS cell is active when at least `min_pulses_per_cell` TSRD pulses
    fall inside that time-slot/frequency-band cell.

    This is an explicit aggregation layer. Individual pulses are NOT treated
    as independent interception episodes.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if int(num_slots) != num_slots or int(num_bands) != num_bands:
        raise ValueError("num_slots and num_bands must be integers.")
    num_slots, num_bands = int(num_slots), int(num_bands)
    if num_slots <= 0 or num_bands <= 0:
        raise ValueError("num_slots and num_bands must be positive.")
    if not np.isfinite(freq_min_mhz) or not np.isfinite(freq_max_mhz):
        raise ValueError("Frequency limits must be finite.")
    if freq_max_mhz <= freq_min_mhz:
        raise ValueError("freq_max_mhz must be greater than freq_min_mhz.")
    if int(min_pulses_per_cell) != min_pulses_per_cell or min_pulses_per_cell <= 0:
        raise ValueError("min_pulses_per_cell must be a positive integer.")

    with h5py.File(path, "r") as f:
        data, names = _validated_layout(f)
        try:
            toa_col = names.index("ToA")
            freq_col = names.index("Frequency")
        except ValueError as exc:
            raise ValueError(
                f"TSRD file must contain ToA and Frequency. Found: {names}"
            ) from exc

        toa = np.asarray(data[:, toa_col], dtype=np.float64)
        freq = np.asarray(data[:, freq_col], dtype=np.float64)

    finite = np.isfinite(toa) & np.isfinite(freq)
    toa, freq = toa[finite], freq[finite]
    if toa.size == 0:
        raise ValueError("No finite ToA/Frequency samples found.")

    t_min, t_max = float(toa.min()), float(toa.max())
    if t_max <= t_min:
        t_max = t_min + 1.0

    slot_edges = np.linspace(t_min, t_max, num_slots + 1)
    band_edges = np.linspace(freq_min_mhz, freq_max_mhz, num_bands + 1)

    slot_idx = np.searchsorted(slot_edges, toa, side="right") - 1
    band_idx = np.searchsorted(band_edges, freq, side="right") - 1
    slot_idx = np.clip(slot_idx, 0, num_slots - 1)

    # Follow histogram conventions: the final band includes its right edge.
    at_upper_edge = (band_idx == num_bands) & (freq <= float(freq_max_mhz))
    band_idx[at_upper_edge] = num_bands - 1

    in_band = (band_idx >= 0) & (band_idx < num_bands)
    counts = np.zeros((num_slots, num_bands), dtype=np.int32)
    np.add.at(counts, (slot_idx[in_band], band_idx[in_band]), 1)
    truth = counts >= int(min_pulses_per_cell)

    return TSRDReplay(
        truth=truth,
        pulse_counts=counts,
        slot_edges_us=slot_edges,
        band_edges_mhz=band_edges,
        source_file=str(path),
        pulses_loaded=int(toa.size),
        pulses_in_band=int(np.sum(in_band)),
        feature_names=names,
        observed_freq_min_mhz=float(freq.min()),
        observed_freq_max_mhz=float(freq.max()),
        observed_toa_min_us=t_min,
        observed_toa_max_us=t_max,
    )
