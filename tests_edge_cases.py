"""Small deterministic edge-case tests."""

import tempfile
from pathlib import Path
import h5py
import numpy as np

from src.scheduler import MaxAgeGuard
from src.tsrd import load_tsrd_replay
from src.engine import run_policy


def make_h5(path):
    data = np.array([
        [0.0, 1000.0, 1.0, 0.0, -80.0],
        [10.0, 1000.0, 1.0, 0.0, -80.0],
        [20.0, 9000.0, 1.0, 0.0, -80.0],
        [30.0, 17000.0, 1.0, 0.0, -80.0],
    ], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        m = f.create_group("metadata")
        m.create_dataset("feature_names", data=np.array(
            [b"ToA", b"Frequency", b"PulseWidth", b"AoA", b"Amplitude"]
        ))


def main():
    g = MaxAgeGuard(20, 2, 40, 1)
    assert g.violations(0) == 0
    g.update_observation(0, 3)
    assert g.last_observed[0] == 3

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.h5"
        make_h5(p)
        replay = load_tsrd_replay(p, num_slots=20, num_bands=20)
        assert replay.truth.shape == (20, 20)
        assert replay.truth.any()
        result = run_policy(replay.truth, "rafts", seed_offset=0)
        assert result["coverage_violations"] >= 0
        assert 0 <= result["empirical_pfa"] <= 1 or np.isnan(result["empirical_pfa"])

    # Deadline-safety regression: even with no RF activity, feasible Max-Age
    # coverage must remain intact under physical retune dead-time.
    empty_truth = np.zeros((200, 20), dtype=bool)
    coverage_result = run_policy(empty_truth, "rafts", seed_offset=77)
    assert coverage_result["coverage_violations"] == 0, coverage_result
    assert coverage_result["max_observed_age"] <= 24, coverage_result

    print("RAFTS + TSRD edge-case tests: PASS")


if __name__ == "__main__":
    main()
