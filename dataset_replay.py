"""Run RAFTS and fair baselines on an Alan Turing Institute TSRD .h5 file."""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.tsrd import load_tsrd_replay, inspect_tsrd
from src.engine import run_policy


def pct(x):
    return "n/a" if pd.isna(x) else f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to TSRD .h5 file")
    ap.add_argument("--slots", type=int, default=500)
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--min-pulses", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("TSRD DATASET REPLAY")
    print("=" * 72)
    info = inspect_tsrd(args.file)
    print("Source file:", info["file"])
    print("PDW shape:", info["shape"])
    print("Features:", ", ".join(info["feature_names"]))

    replay = load_tsrd_replay(
        args.file, num_slots=args.slots, num_bands=args.bands,
        min_pulses_per_cell=args.min_pulses,
    )
    print(f"Pulses loaded: {replay.pulses_loaded:,}")
    print(f"Observed frequency: {replay.observed_freq_min_mhz:.1f}..{replay.observed_freq_max_mhz:.1f} MHz")
    print(f"Observed ToA: {replay.observed_toa_min_us:.1f}..{replay.observed_toa_max_us:.1f} us")
    print(f"Replay grid: {replay.num_slots} slots x {replay.num_bands} bands")
    print(f"Active cell fraction: {replay.truth.mean()*100:.2f}%")
    print("Scheduler input: receiver HIT/MISS only. TSRD truth remains hidden.")
    print()

    rows = []
    for policy in ("sequential", "random", "dts", "rafts"):
        r = run_policy(replay.truth, policy=policy, seed_offset=args.seed)
        rows.append({"policy": policy.upper(), **r})
        print(
            f"{policy.upper():10s} "
            f"IR={pct(r['interception_ratio']):>7s}  "
            f"AvgInterceptRate={pct(r['avg_intercept_rate']):>7s}  "
            f"CondMeanTTI={r['conditional_mean_tti']:.3f}  "
            f"P95={r['conditional_p95_tti']:.3f}  "
            f"retunes={r['retunes']}"
        )

    out = Path("results")
    out.mkdir(exist_ok=True)
    stem = Path(args.file).stem
    df = pd.DataFrame(rows)
    csv = out / f"tsrd_{stem}_comparison.csv"
    df.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["policy"], df["interception_ratio"] * 100)
    ax.set_ylabel("Interception Ratio (%)")
    ax.set_title(f"TSRD Replay: {stem}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png = out / f"tsrd_{stem}_ir.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)

    print()
    print("Saved:", csv)
    print("Saved:", png)


if __name__ == "__main__":
    main()
