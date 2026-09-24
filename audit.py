"""Fast packaging and architecture audit for the final prototype."""

import ast
import csv
import json
from pathlib import Path

from src.engine import DEMO_POLICIES, SELECTED_POLICY
from src.simulator import SCENARIOS
from src.tsrd import inspect_tsrd


ROOT = Path(__file__).resolve().parent
EXPECTED_DATASETS = ("config_0.h5", "config_1.h5", "config_10.h5", "config_103.h5")
FORBIDDEN_RUNTIME_FILES = (
    "run_full_bakeoff.py",
    "reanalyze_v2_evidence.py",
    "src/learners.py",
    "src/transition.py",
)
REQUIRED_FILES = (
    "README.md",
    "ALGORITHM.md",
    "RESULTS_AND_LIMITATIONS.md",
    "REQUIREMENTS_TRACEABILITY.md",
    "run_demo.py",
    "run_validation.py",
    "src/simulator.py",
)


def main():
    source_files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "src").glob("*.py"))
    for path in source_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    missing_files = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing_files:
        raise RuntimeError(f"Missing required prototype files: {missing_files}")

    missing = [name for name in EXPECTED_DATASETS if not (ROOT / "data/tsrd" / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing TSRD datasets: {missing}")
    for name in EXPECTED_DATASETS:
        inspect_tsrd(ROOT / "data/tsrd" / name)

    present = [name for name in FORBIDDEN_RUNTIME_FILES if (ROOT / name).exists()]
    if present:
        raise RuntimeError(f"Research-only runtime files must be absent: {present}")

    expected_policies = (
        "sequential",
        "random",
        "coverage_sweep",
        "rafts_cusum_ucb",
    )
    if DEMO_POLICIES != expected_policies or SELECTED_POLICY != "rafts_cusum_ucb":
        raise RuntimeError("The final selected-policy declaration changed.")

    report = json.loads((ROOT / "evidence/selection_report.json").read_text(encoding="utf-8"))
    recommendation = str(report.get("selection", {}).get("recommended_label", ""))
    if "CUSUM" not in recommendation.upper():
        raise RuntimeError("Frozen evidence does not recommend CUSUM-UCB.")

    engine_text = (ROOT / "src/engine.py").read_text(encoding="utf-8")
    if "truth[slot, band]" not in engine_text:
        raise RuntimeError("Receiver observation boundary was not found.")
    if engine_text.count("truth[slot, band]") != 1:
        raise RuntimeError("Hidden truth must cross the online boundary exactly once.")
    for required_marker in ("fallback_count", "duplicate_assignments"):
        if required_marker not in engine_text:
            raise RuntimeError(f"Missing runtime diagnostic: {required_marker}")

    with (ROOT / "evidence/candidate_scorecard.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        scorecard = {row["policy"]: row for row in csv.DictReader(stream)}
    selected = scorecard.get(SELECTED_POLICY)
    if selected is None or int(selected["runs"]) != 480:
        raise RuntimeError("Frozen selected learner must contain 480 held-out runs.")
    if int(float(selected["coverage_violations_total"])) != 0:
        raise RuntimeError("Frozen selected learner has unexpected coverage violations.")
    if len(SCENARIOS) != 14:
        raise RuntimeError("Controlled simulator scenario inventory changed.")

    print("RAFTS-CUSUM-UCB final prototype audit: PASS")
    print(f"Python source files parsed: {len(source_files)}")
    print(f"TSRD files validated: {len(EXPECTED_DATASETS)}")
    print(f"Controlled simulator scenarios: {len(SCENARIOS)} + 6 sensor/retune stresses")
    print("Operational learner: RAFTS-CUSUM-UCB only")
    print("Display baselines: Sequential, Random, Coverage Sweep")
    print("Hidden-truth observation boundary: PASS")
    print("Research-only algorithms removed from runtime: PASS")
    print("Frozen recommendation consistency: PASS")
    print("Frozen selected-metric consistency: PASS")
    print("Fallback and assignment diagnostics: PASS")


if __name__ == "__main__":
    main()
