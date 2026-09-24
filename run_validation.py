"""Fast simulator + TSRD diagnostic for the frozen final prototype.

This command is a smoke validation, not new learner-selection evidence.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.engine import DEMO_POLICIES, run_policy
from src.simulator import SCENARIOS, make_scenario
from src.tsrd import load_tsrd_replay

DATASET_NAMES = ("config_0.h5", "config_1.h5", "config_10.h5", "config_103.h5")
STRESS_CASES = (
    ("periodic_low_pd", "periodic", {"pd": 0.45}),
    ("agile_low_pd", "agile", {"pd": 0.45}),
    ("random_high_pfa", "random", {"pfa": 0.10}),
    ("no_emitters_high_pfa", "no_emitters", {"pfa": 0.10}),
    ("changing_high_retune", "changing", {"retune_delay_slots": 3}),
    ("bursty_high_retune", "bursty", {"retune_delay_slots": 3}),
)


def _run_case(truth, policy, seed, **kwargs):
    result = run_policy(truth, policy=policy, seed_offset=seed * 100, **kwargs)
    receiver_time = max(1, result["receiver_time_slots"])
    result["retune_dead_fraction"] = result["retune_dead_slots"] / receiver_time
    result["runtime_per_slot_seconds"] = result["runtime_seconds"] / max(
        1, truth.shape[0]
    )
    return result


def run_validation(output_dir="validation_smoke", slots=120, bands=20, seed=0):
    """Run all 20 simulator cases and four TSRD replays once per policy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    simulator_cases = [(name, name, {}) for name in SCENARIOS]
    simulator_cases.extend(STRESS_CASES)
    for environment, scenario, parameters in simulator_cases:
        truth = make_scenario(scenario, slots=slots, bands=bands, seed=seed)
        for policy in DEMO_POLICIES:
            rows.append(
                {
                    "source": "simulator",
                    "environment": environment,
                    "seed": seed,
                    "policy": policy,
                    **_run_case(truth, policy, seed, **parameters),
                }
            )

    dataset_dir = Path(__file__).resolve().parent / "data/tsrd"
    for name in DATASET_NAMES:
        replay = load_tsrd_replay(
            dataset_dir / name,
            num_slots=slots,
            num_bands=bands,
        )
        for policy in DEMO_POLICIES:
            rows.append(
                {
                    "source": "tsrd",
                    "environment": Path(name).stem,
                    "seed": seed,
                    "policy": policy,
                    **_run_case(replay.truth, policy, seed),
                }
            )

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "validation_runs.csv"
    raw.to_csv(raw_path, index=False)
    metrics = [
        "interception_ratio",
        "capped_mean_tti",
        "capped_p95_tti",
        "unique_episode_reward_per_receiver_time",
        "avg_reward_per_receiver_time",
        "retune_dead_fraction",
        "coverage_violations",
        "fallback_count",
        "runtime_per_slot_seconds",
    ]
    summary = raw.groupby(["source", "policy"], as_index=False)[metrics].mean()
    summary_path = output_dir / "validation_summary.csv"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "evidence_status": "SMOKE_DIAGNOSTIC_NOT_SELECTION_EVIDENCE",
        "selected_policy": "rafts_cusum_ucb",
        "simulator_cases": len(simulator_cases),
        "tsrd_datasets": list(DATASET_NAMES),
        "policies": list(DEMO_POLICIES),
        "slots": int(slots),
        "bands": int(bands),
        "seed": int(seed),
    }
    metadata_path = output_dir / "validation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("RAFTS prototype smoke validation: COMPLETE")
    print("Status: SMOKE_DIAGNOSTIC_NOT_SELECTION_EVIDENCE")
    print(f"Runs: {len(raw)} ({len(simulator_cases)} simulator cases + 4 TSRD replays)")
    for path in (raw_path, summary_path, metadata_path):
        print(" -", path)
    return raw_path, summary_path, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="validation_smoke")
    parser.add_argument("--slots", type=int, default=120)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_validation(args.output_dir, args.slots, args.bands, args.seed)


if __name__ == "__main__":
    main()
