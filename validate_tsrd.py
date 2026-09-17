"""Multi-file TSRD validation with repeated receiver-noise seeds."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tsrd import load_tsrd_replay
from src.engine import run_policy


def ci95(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan")
    return 1.96 * float(np.std(a, ddof=1)) / np.sqrt(len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--slots", type=int, default=500)
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--min-pulses", type=int, default=1)
    args = ap.parse_args()

    records = []
    for file in args.files:
        replay = load_tsrd_replay(
            file, num_slots=args.slots, num_bands=args.bands,
            min_pulses_per_cell=args.min_pulses,
        )
        for seed in range(args.seeds):
            for policy in ("sequential", "random", "dts", "rafts"):
                r = run_policy(replay.truth, policy=policy, seed_offset=seed * 100)
                records.append({
                    "file": Path(file).name,
                    "seed": seed,
                    "policy": policy.upper(),
                    **r,
                })
        print("Completed", file)

    raw = pd.DataFrame(records)
    rows = []
    for (file, policy), g in raw.groupby(["file", "policy"], sort=False):
        rows.append({
            "file": file,
            "policy": policy,
            "IR_mean": g["interception_ratio"].mean(),
            "IR_ci95": ci95(g["interception_ratio"]),
            "MeanTTI": g["conditional_mean_tti"].mean(),
            "P95TTI": g["conditional_p95_tti"].mean(),
            "AvgInterceptRate": g["avg_intercept_rate"].mean(),
            "Pd_empirical": g["empirical_pd"].mean(),
            "Pfa_empirical": g["empirical_pfa"].mean(),
            "retunes": g["retunes"].mean(),
            "coverage_violations": g["coverage_violations"].max(),
        })
    summary = pd.DataFrame(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    raw.to_csv(out / "tsrd_validation_raw.csv", index=False)
    summary.to_csv(out / "tsrd_validation_summary.csv", index=False)

    print("\nTSRD VALIDATION SUMMARY")
    for _, x in summary.iterrows():
        print(
            f"{x['file']:16s} {x['policy']:10s} "
            f"IR={100*x['IR_mean']:.1f}% ± {100*x['IR_ci95']:.1f}%  "
            f"MeanTTI={x['MeanTTI']:.3f}  "
            f"coverage={int(x['coverage_violations'])}"
        )

    rafts = summary[summary.policy == "RAFTS"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(rafts["file"], rafts["IR_mean"] * 100, yerr=rafts["IR_ci95"] * 100, capsize=4)
    ax.set_ylabel("RAFTS Interception Ratio (%)")
    ax.set_title("TSRD Replay Validation (95% CI over receiver-noise seeds)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "tsrd_rafts_validation.png", dpi=180)
    plt.close(fig)

    print("\nSaved results/tsrd_validation_raw.csv")
    print("Saved results/tsrd_validation_summary.csv")
    print("Saved results/tsrd_rafts_validation.png")


if __name__ == "__main__":
    main()
