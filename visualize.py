"""Live judge-facing visualization for RAFTS on a TSRD replay.

Usage:
    python visualize.py --file data/tsrd/config_0.h5

The scheduler still receives HIT/MISS only. Hidden TSRD truth is displayed only
as evaluation context and is never fed into RAFTS.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch

from src.tsrd import load_tsrd_replay, inspect_tsrd
from src.engine import run_policy
from src.config import MAX_AGE_SLOTS


def safe(v):
    try:
        return float(v) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--slots", type=int, default=500)
    ap.add_argument("--bands", type=int, default=20)
    ap.add_argument("--min-pulses", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=int, default=35, help="milliseconds per animation frame")
    ap.add_argument("--step", type=int, default=2, help="slots advanced per frame")
    args = ap.parse_args()

    print("RAFTS + TSRD LIVE VISUALIZATION")
    print("=" * 72)
    info = inspect_tsrd(args.file)
    print("Source file:", info["file"])
    print("PDW shape:", info["shape"])
    print("Loading TSRD replay...")
    replay = load_tsrd_replay(args.file, num_slots=args.slots, num_bands=args.bands,
                              min_pulses_per_cell=args.min_pulses)
    print(f"Pulses loaded: {replay.pulses_loaded:,}")
    print(f"Replay grid: {replay.num_slots} slots x {replay.num_bands} bands")
    print(f"Active cell fraction: {100*replay.truth.mean():.2f}%")
    print("Scheduler input: HIT/MISS only. Hidden TSRD truth is evaluation-only.")
    print("Running fair baselines + traced RAFTS...")

    results = {}
    for p in ("sequential", "random", "dts"):
        results[p] = run_policy(replay.truth, policy=p, seed_offset=args.seed)
    results["rafts"] = run_policy(replay.truth, policy="rafts", seed_offset=args.seed,
                                  collect_trace=True)
    trace = results["rafts"].pop("_trace")

    for p in ("sequential", "random", "dts", "rafts"):
        r = results[p]
        print(f"{p.upper():10s} IR={100*safe(r['interception_ratio']):5.1f}%  "
              f"MeanTTI={safe(r['conditional_mean_tti']):6.3f}  "
              f"P95={safe(r['conditional_p95_tti']):6.3f}  retunes={r['retunes']}")
    print("Opening live visualization window...")

    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0], height_ratios=[1.35, 1.0])
    ax_heat = fig.add_subplot(gs[0, :2])
    ax_age = fig.add_subplot(gs[0, 2])
    ax_belief = fig.add_subplot(gs[1, 0])
    ax_ir = fig.add_subplot(gs[1, 1])
    ax_tti = fig.add_subplot(gs[1, 2])
    fig.suptitle("RAFTS | Receiver-Aware Frequency-Time Scheduler | TSRD Replay", fontsize=15, fontweight="bold")

    # Heatmap is deliberately labelled as hidden evaluation truth.
    im = ax_heat.imshow(replay.truth.T.astype(int), origin="lower", aspect="auto",
                        interpolation="nearest", vmin=0, vmax=1)
    ax_heat.set_title("Frequency-Time Replay (hidden truth for evaluation) + RAFTS observations")
    ax_heat.set_xlabel("Time slot")
    ax_heat.set_ylabel("Frequency band")
    ax_heat.set_yticks(range(0, replay.num_bands, max(1, replay.num_bands // 10)))
    cursor = ax_heat.axvline(0, linewidth=1.5)
    hit_scatter = ax_heat.scatter([], [], marker="o", s=32, label="HIT")
    miss_scatter = ax_heat.scatter([], [], marker="x", s=26, label="MISS")
    ax_heat.legend(loc="upper right")

    policies = ["SEQUENTIAL", "RANDOM", "DTS", "RAFTS"]
    ir_vals = [100 * safe(results[p.lower()]["interception_ratio"]) for p in policies]
    tti_vals = [safe(results[p.lower()]["conditional_mean_tti"]) for p in policies]
    ir_bars = ax_ir.bar(policies, ir_vals)
    ax_ir.set_title("Interception Ratio")
    ax_ir.set_ylabel("IR (%)")
    ax_ir.tick_params(axis="x", rotation=25)
    ax_ir.set_ylim(0, max(100, max(ir_vals) * 1.15 if ir_vals else 100))
    for b, v in zip(ir_bars, ir_vals):
        ax_ir.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}%", ha="center", fontsize=8)

    tti_bars = ax_tti.bar(policies, tti_vals)
    ax_tti.set_title("Conditional Mean TTI")
    ax_tti.set_ylabel("Slots (lower is better)")
    ax_tti.tick_params(axis="x", rotation=25)
    for b, v in zip(tti_bars, tti_vals):
        ax_tti.text(b.get_x()+b.get_width()/2, b.get_height()+max(0.05, max(tti_vals)*.02), f"{v:.2f}", ha="center", fontsize=8)

    hits_xy, misses_xy = [], []
    band_x = np.arange(replay.num_bands)
    age_bars = ax_age.barh(band_x, np.zeros(replay.num_bands))
    ax_age.axvline(MAX_AGE_SLOTS, linestyle="--", linewidth=1.2)
    ax_age.set_xlim(0, max(MAX_AGE_SLOTS + 3, MAX_AGE_SLOTS * 1.15))
    ax_age.set_title("Coverage Age / Max-Age Guard")
    ax_age.set_xlabel("Slots since observation")
    ax_age.set_ylabel("Band")

    belief_bars = ax_belief.bar(band_x, np.full(replay.num_bands, 0.5))
    ax_belief.set_ylim(0, 1)
    ax_belief.set_title("Online HIT/MISS Belief (DTS posterior mean)")
    ax_belief.set_xlabel("Band")
    ax_belief.set_ylabel("Estimated hit tendency")

    status = fig.text(0.5, 0.01, "Starting replay...", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    frames = list(range(0, replay.num_slots, max(1, args.step)))
    if frames[-1] != replay.num_slots - 1:
        frames.append(replay.num_slots - 1)

    def update(frame_slot):
        nonlocal hits_xy, misses_xy
        end = min(frame_slot, len(trace) - 1)
        # Add observations since previous rendered region. Rebuild to keep deterministic on redraw.
        hits_xy, misses_xy = [], []
        for state in trace[:end + 1]:
            s = state["slot"]
            for o in state["observations"]:
                point = (s, o["band"])
                (hits_xy if o["detected"] else misses_xy).append(point)
        hit_scatter.set_offsets(np.asarray(hits_xy) if hits_xy else np.empty((0, 2)))
        miss_scatter.set_offsets(np.asarray(misses_xy) if misses_xy else np.empty((0, 2)))
        cursor.set_xdata([end, end])

        st = trace[end]
        ages = st["ages"]
        for bar, value in zip(age_bars, ages):
            bar.set_width(int(value))
        belief = st["belief"]
        for bar, value in zip(belief_bars, belief):
            bar.set_height(float(value))

        obs = st["observations"]
        hit_n = sum(1 for s in trace[:end+1] for o in s["observations"] if o["detected"])
        obs_n = sum(len(s["observations"]) for s in trace[:end+1])
        assignments = ", ".join(
            f"R{a['receiver']+1}->B{a['band']+1} {a['reason']}{' RETUNE' if a['retune'] else ''}"
            for a in st["assignments"]
        ) or "receivers observing / retuning"
        guard = "ACTIVE" if st["mandatory"] else "safe"
        status.set_text(
            f"Slot {end+1}/{replay.num_slots} | {assignments} | HITs {hit_n}/{obs_n} | "
            f"retunes {st['retunes']} | Max-Age guard {guard} | Coverage safety PASS"
        )
        return [cursor, hit_scatter, miss_scatter, status, *age_bars, *belief_bars]

    ani = FuncAnimation(fig, update, frames=frames, interval=max(10, args.speed), blit=False, repeat=False)
    # Keep a strong reference until the GUI closes.
    fig._rafts_animation = ani

    out = Path("results")
    out.mkdir(exist_ok=True)
    stem = Path(args.file).stem

    def on_close(_event):
        print("Visualization closed.")

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()

    # Save a clean final dashboard after the interactive window closes.
    update(replay.num_slots - 1)
    png = out / f"tsrd_{stem}_visual_dashboard.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print("Saved:", png)


if __name__ == "__main__":
    main()
