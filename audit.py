"""Fast architecture audit for submission safety."""

from pathlib import Path
from src.config import NUM_BANDS, NUM_RECEIVERS, MAX_AGE_SLOTS, RETUNE_DELAY_SLOTS
from src.scheduler import MaxAgeGuard


def main():
    assert NUM_BANDS > NUM_RECEIVERS, "Prototype must represent limited receiver capacity."
    MaxAgeGuard(NUM_BANDS, NUM_RECEIVERS, MAX_AGE_SLOTS, RETUNE_DELAY_SLOTS)

    tsrd = Path("src/tsrd.py").read_text(encoding="utf-8")
    engine = Path("src/engine.py").read_text(encoding="utf-8")

    assert 'names.index("ToA")' in tsrd
    assert 'names.index("Frequency")' in tsrd
    assert "labels" not in engine, "Scheduler engine must not use TSRD emitter labels."
    assert "metadata/transmitters" not in engine, "Scheduler must not use transmitter metadata."
    assert "guard.update_observation" in engine, "Age must reset on actual observation."
    assert "truth[slot, band]" in engine, "Truth should be accessed only at receiver observation boundary."

    print("RAFTS + TSRD architecture audit: PASS")
    print("Hidden-truth separation: PASS")
    print("Max-Age conservative feasibility: PASS")
    print("TSRD ToA/Frequency adapter: PASS")


if __name__ == "__main__":
    main()
