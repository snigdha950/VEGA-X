"""Judge-facing live view and frozen-evidence summary for RAFTS-CUSUM-UCB.

The live view displays hidden replay truth only as evaluation context. The
scheduler itself receives completed receiver HIT/MISS observations only.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import (
    MAX_AGE_SLOTS,
    NUM_RECEIVERS,
    PD,
    PFA,
    RETUNE_DELAY_SLOTS,
    SENSITIVITY_DB,
)
from src.engine import DEMO_POLICIES, SELECTED_POLICY, run_policy
from src.tsrd import inspect_tsrd, load_tsrd_replay

POLICIES = DEMO_POLICIES
TRACE_POLICY = SELECTED_POLICY
POLICY_LABELS = {
    "sequential": "Sequential",
    "random": "Random",
    "coverage_sweep": "Coverage Sweep",
    "rafts_cusum_ucb": "RAFTS-CUSUM-UCB",
}
POLICY_COLORS = {
    "sequential": "#94A3B8",
    "random": "#F59E0B",
    "coverage_sweep": "#64748B",
    "rafts_cusum_ucb": "#0F6CBD",
}
ROOT = Path(__file__).resolve().parent
FROZEN_COMPARISON_FILE = ROOT / "evidence/frozen_control_comparison.csv"


def safe(value):
    try:
        return float(value) if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _json_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    return value


def _finish_atomic_write(temporary, intended):
    """Publish a completed image without exposing a partially written target."""
    try:
        temporary.replace(intended)
        return intended
    except OSError:
        # Windows may lock an image that is open in a viewer. Preserve the
        # complete new output under an explicit alternate name instead of
        # truncating or destroying the previous file.
        alternate = intended.with_name(f"{intended.stem}_new{intended.suffix}")
        temporary.replace(alternate)
        return alternate


def _save_figure_atomically(figure, output_path):
    output_path = Path(output_path)
    temporary = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    figure.savefig(
        temporary,
        format=output_path.suffix.lstrip("."),
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    return _finish_atomic_write(temporary, output_path)


def _annotate_bars(axis, bars, values, formatter):
    maximum = max([safe(value) for value in values] + [0.0])
    offset = maximum * 0.025 if maximum > 0 else 0.025
    for index, (bar, value) in enumerate(zip(bars, values)):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if index == len(bars) - 1 else "normal",
            color="#0F172A",
        )


def _metric_panel(
    axis,
    labels,
    values,
    colors,
    title,
    ylabel,
    formatter,
    errors=None,
):
    bars = axis.bar(
        labels,
        values,
        color=colors,
        width=0.68,
        edgecolor=["none", "none", "none", "#083B66"],
        linewidth=[0.0, 0.0, 0.0, 2.0],
        yerr=errors,
        capsize=5 if errors is not None else 0,
        error_kw={"elinewidth": 1.3, "ecolor": "#334155"},
    )
    axis.set_title(title, fontsize=13, fontweight="bold", pad=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.tick_params(axis="x", rotation=13, labelsize=9)
    axis.tick_params(axis="y", labelsize=9)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    maximum = max([safe(value) for value in values] + [0.0])
    if errors is not None:
        maximum = max(maximum, float(np.max(np.asarray(values) + errors[1])))
    axis.set_ylim(0, maximum * 1.24 if maximum > 0 else 1.0)
    _annotate_bars(axis, bars, values, formatter)


def _band_range_label(replay, band):
    if band < 0 or band + 1 >= len(replay.band_edges_mhz):
        return "unassigned"
    low = replay.band_edges_mhz[band] / 1000.0
    high = replay.band_edges_mhz[band + 1] / 1000.0
    return f"{low:.2f}–{high:.2f} GHz"


def _recent_reason(trace, end, receiver_id):
    for state in reversed(trace[: end + 1]):
        for assignment in state["assignments"]:
            if assignment["receiver"] == receiver_id:
                return (
                    "MAX-AGE GUARD"
                    if assignment["reason"] == "MAX-AGE"
                    else "CUSUM-UCB"
                )
    return "INITIAL SEARCH"


def _change_events(trace, end):
    events = []
    previous = np.zeros(len(trace[0]["change_count_by_band"]), dtype=int)
    for state in trace[: end + 1]:
        current = np.asarray(state["change_count_by_band"], dtype=int)
        for band in np.flatnonzero(current > previous):
            events.extend(
                [(state["slot"], int(band))] * int(current[band] - previous[band])
            )
        previous = current
    return events


def _create_evidence_summary(frozen, output_path):
    display_labels = ["Sequential", "Random", "Coverage\nSweep", "RAFTS\nCUSUM-UCB"]
    colors = [POLICY_COLORS[policy] for policy in POLICIES]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor="#F8FAFC")
    fig.subplots_adjust(top=0.78, bottom=0.12, hspace=0.48, wspace=0.25)
    fig.suptitle(
        "RAFTS-CUSUM-UCB — Frozen Held-Out Evidence",
        fontsize=20,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "480 paired conditions per policy • full selection: 16 policies / 7,680 runs",
        ha="center",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.5,
        0.866,
        "Robustness-first leader  •  highest displayed mean IR and net reward  •  lowest displayed retune dead-time",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#0F6CBD",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#DBEAFE",
            "edgecolor": "#93C5FD",
        },
    )

    ir_values = 100 * frozen["mean_interception_ratio"].to_numpy(dtype=float)
    ir_low = 100 * frozen["ir_ci95_low"].to_numpy(dtype=float)
    ir_high = 100 * frozen["ir_ci95_high"].to_numpy(dtype=float)
    ir_errors = np.vstack([ir_values - ir_low, ir_high - ir_values])
    _metric_panel(
        axes[0, 0],
        display_labels,
        ir_values,
        colors,
        "Mean interception ratio ↑",
        "% of episodes",
        lambda value: f"{value:.1f}%",
        errors=ir_errors,
    )
    axes[0, 0].text(
        0.02,
        0.95,
        "95% bootstrap CI",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=9,
        color="#475569",
    )

    reward_values = frozen["mean_reward_per_receiver_time"].to_numpy(dtype=float)
    _metric_panel(
        axes[0, 1],
        display_labels,
        reward_values,
        colors,
        "Net TP−FP reward / receiver-time ↑",
        "Reward per slot",
        lambda value: f"{value:.3f}",
    )

    retune_values = 100 * frozen["mean_retune_dead_fraction"].to_numpy(dtype=float)
    _metric_panel(
        axes[1, 0],
        display_labels,
        retune_values,
        colors,
        "Retune-dead fraction ↓",
        "% receiver-time",
        lambda value: f"{value:.1f}%",
    )

    tti_values = frozen["mean_capped_p95_tti"].to_numpy(dtype=float)
    _metric_panel(
        axes[1, 1],
        display_labels,
        tti_values,
        colors,
        "Capped P95 time-to-intercept ↓",
        "Slots",
        lambda value: f"{value:.1f}",
    )
    axes[1, 1].text(
        0.5,
        0.95,
        "Transparent trade-off: selected policy is slower on this metric",
        transform=axes[1, 1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#92400E",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FEF3C7",
            "edgecolor": "#FCD34D",
        },
    )

    fig.text(
        0.5,
        0.045,
        "Frozen V3.1 held-out protocol • aggregate means are not recomputed from the illustrative live replay",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    output_path = _save_figure_atomically(fig, output_path)
    plt.close(fig)
    return output_path


def run_visualization(
    file,
    slots=300,
    bands=20,
    min_pulses=1,
    seed=0,
    speed=250,
    step=6,
    output_dir="outputs",
    save_gif=True,
    show=False,
):
    """Run policies and produce CSV, JSON, live PNG/GIF, and evidence PNG."""
    file = Path(file)
    output_dir = Path(output_dir)
    if not file.is_file():
        raise FileNotFoundError(f"TSRD dataset not found: {file}")
    if int(slots) <= 0:
        raise ValueError("slots must be greater than zero")
    if int(bands) <= 0:
        raise ValueError("bands must be greater than zero")
    if int(min_pulses) <= 0:
        raise ValueError("min_pulses must be greater than zero")
    output_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(step))

    print("RAFTS JUDGE-FACING VISUALIZATION")
    print("=" * 72)
    info = inspect_tsrd(file)
    replay = load_tsrd_replay(
        file,
        num_slots=slots,
        num_bands=bands,
        min_pulses_per_cell=min_pulses,
    )
    print("Source:", info["file"])
    print("PDW matrix:", info["shape"])
    print(f"Pulses: {replay.pulses_loaded:,} total / {replay.pulses_in_band:,} in band")
    print(f"Replay: {replay.num_slots} slots x {replay.num_bands} bands")
    print(f"Active-cell fraction: {100 * replay.truth.mean():.2f}%")
    print("Scheduler input: receiver HIT/MISS only; displayed truth is evaluation-only.")

    if not FROZEN_COMPARISON_FILE.is_file():
        raise FileNotFoundError(
            f"Frozen comparison evidence not found: {FROZEN_COMPARISON_FILE}"
        )
    frozen = pd.read_csv(FROZEN_COMPARISON_FILE).set_index("policy").loc[list(POLICIES)]
    if not np.all(frozen["runs"].to_numpy(dtype=int) == 480):
        raise ValueError("Frozen comparison must contain 480 runs for every policy.")

    results = {}
    for policy in POLICIES:
        results[policy] = run_policy(
            replay.truth,
            policy=policy,
            seed_offset=seed,
            collect_trace=policy == TRACE_POLICY,
        )
    trace = results[TRACE_POLICY].pop("_trace")

    rows = []
    for policy in POLICIES:
        result = results[policy]
        rows.append({"policy_key": policy, "policy_label": POLICY_LABELS[policy], **result})
        print(
            f"{POLICY_LABELS[policy]:14s} "
            f"IR={100 * safe(result['interception_ratio']):5.1f}%  "
            f"CappedTTI={safe(result['capped_mean_tti']):6.2f}  "
            f"Reward/time={safe(result['true_reward_per_receiver_time']):6.3f}  "
            f"RetuneDead={result['retune_dead_slots']:4d}  "
            f"Coverage={result['coverage_violations']}"
        )

    comparison = pd.DataFrame(rows)
    comparison_path = output_dir / "policy_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "dataset": {
            "file": str(file),
            "pdw_shape": list(info["shape"]),
            "feature_names": info["feature_names"],
            "pulses_loaded": replay.pulses_loaded,
            "pulses_in_band": replay.pulses_in_band,
            "slots": replay.num_slots,
            "bands": replay.num_bands,
            "active_cell_fraction": float(replay.truth.mean()),
        },
        "scheduler_boundary": "HIT/MISS only; hidden truth is evaluation-only",
        "receiver_model": {
            "receivers": NUM_RECEIVERS,
            "instantaneous_bandwidth": "one discretized frequency band per receiver",
            "dwell_slots": 1,
            "retune_delay_slots": RETUNE_DELAY_SLOTS,
            "pd": PD,
            "pfa": PFA,
            "sensitivity_db": SENSITIVITY_DB,
            "sensitivity_status": (
                "configured prototype assumption; binary replay has no per-cell SNR thresholding"
            ),
        },
        "selected_policy": "RAFTS-CUSUM-UCB",
        "evidence_status": "SINGLE-RUN LIVE DEMO; NOT LEARNER-SELECTION EVIDENCE",
        "evidence_summary": "FROZEN HELD-OUT COMPARISON; 480 PAIRED CONDITIONS PER POLICY",
        "policies": {
            policy: {key: _json_value(value) for key, value in result.items()}
            for policy, result in results.items()
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    evidence_path = _create_evidence_summary(
        frozen, output_dir / "rafts_evidence_summary.png"
    )

    # Full-HD-proportioned canvas keeps every judge-facing detail legible when
    # the dashboard is opened full screen or inserted into the presentation.
    fig = plt.figure(figsize=(19.2, 10.8), facecolor="#F8FAFC")
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.68, 1.0],
        width_ratios=[1.28, 1.28, 1.0, 1.0],
        hspace=0.38,
        wspace=0.38,
        top=0.855,
        bottom=0.125,
        left=0.055,
        right=0.985,
    )
    ax_heat = fig.add_subplot(grid[0, :])
    ax_receiver = fig.add_subplot(grid[1, :2])
    ax_age = fig.add_subplot(grid[1, 2])
    ax_belief = fig.add_subplot(grid[1, 3])

    fig.suptitle(
        "RAFTS — Live Receiver-Aware Frequency-Time Scheduling",
        fontsize=23,
        fontweight="bold",
        color="#0F172A",
        y=0.975,
    )
    fig.text(
        0.5,
        0.925,
        "Operational learner: CUSUM-UCB  •  Online input: completed receiver HIT/MISS only",
        ha="center",
        fontsize=14,
        color="#475569",
    )
    fig.text(
        0.5,
        0.895,
        "Blue activity background is hidden truth for evaluation and is never provided to the scheduler",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#1E40AF",
    )

    truth_map = ListedColormap(["#F8FAFC", "#BFDBFE"])
    ax_heat.imshow(
        replay.truth.T.astype(int),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=truth_map,
        vmin=0,
        vmax=1,
    )
    ax_heat.set_title(
        "Frequency × time replay",
        fontsize=16,
        fontweight="bold",
        loc="left",
        pad=9,
    )
    ax_heat.set_xlabel("Time slot", fontsize=12)
    ax_heat.set_ylabel("Frequency band", fontsize=12)
    ax_heat.tick_params(labelsize=10.5)
    ax_heat.set_yticks(range(0, replay.num_bands, max(1, replay.num_bands // 10)))
    cursor = ax_heat.axvline(0, color="#2563EB", linewidth=2.0)
    hit_scatter = ax_heat.scatter(
        [], [], marker="o", s=48, color="#16A34A", edgecolors="white", linewidths=0.7
    )
    miss_scatter = ax_heat.scatter([], [], marker="x", s=34, color="#64748B", linewidths=1.2)
    reset_scatter = ax_heat.scatter(
        [], [], marker="D", s=62, color="#7C3AED", edgecolors="white", linewidths=0.8
    )
    ax_heat.legend(
        handles=[
            Patch(facecolor="#BFDBFE", edgecolor="none", label="Active truth (evaluation only)"),
            Line2D([], [], marker="o", linestyle="", color="#16A34A", label="HIT"),
            Line2D([], [], marker="x", linestyle="", color="#64748B", label="MISS"),
            Line2D([], [], marker="D", linestyle="", color="#7C3AED", label="CUSUM reset"),
            Line2D([], [], color="#2563EB", linewidth=2, label="Current slot"),
        ],
        loc="upper right",
        ncol=2,
        frameon=True,
        fontsize=10.5,
    )

    receiver_count = len(trace[0]["receivers"])
    receiver_colors = ["#2563EB", "#F59E0B", "#8B5CF6", "#10B981"]
    ax_receiver.set_xlim(-0.5, replay.num_bands - 0.5)
    ax_receiver.set_ylim(-0.55, receiver_count - 0.18)
    ax_receiver.set_yticks(range(receiver_count))
    ax_receiver.set_yticklabels(
        [f"Receiver {index + 1}" for index in range(receiver_count)], fontsize=11.5
    )
    ax_receiver.set_xticks(range(replay.num_bands))
    ax_receiver.set_xticklabels([str(index + 1) for index in range(replay.num_bands)], fontsize=9.5)
    ax_receiver.set_xlabel("Assigned frequency band", fontsize=11.5)
    ax_receiver.set_title(
        "Current receiver decisions", fontsize=15, fontweight="bold", loc="left", pad=9
    )
    ax_receiver.grid(axis="x", alpha=0.15)
    receiver_markers = []
    receiver_labels = []
    for index in range(receiver_count):
        (marker,) = ax_receiver.plot(
            [], [], marker="o", markersize=16, linestyle="", color=receiver_colors[index]
        )
        label = ax_receiver.text(
            0,
            index + 0.22,
            "",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        receiver_markers.append(marker)
        receiver_labels.append(label)

    top_count = min(5, replay.num_bands)
    rank_axis = np.arange(top_count)
    age_bars = ax_age.barh(rank_axis, np.zeros(top_count), color="#60A5FA")
    age_texts = [ax_age.text(0, rank, "", va="center", fontsize=10.5) for rank in rank_axis]
    ax_age.axvline(MAX_AGE_SLOTS, color="#DC2626", linestyle="--", linewidth=1.5)
    ax_age.set_xlim(0, max(MAX_AGE_SLOTS + 8, MAX_AGE_SLOTS * 1.15))
    ax_age.set_title("Top 5 coverage ages", fontsize=15, fontweight="bold")
    ax_age.set_xlabel("Slots since observed", fontsize=11)
    ax_age.set_yticks(rank_axis, [f"B{index + 1}" for index in rank_axis])
    ax_age.invert_yaxis()
    ax_age.tick_params(labelsize=10.5)

    belief_bars = ax_belief.barh(rank_axis, np.full(top_count, 0.5), color="#94A3B8")
    belief_texts = [ax_belief.text(0, rank, "", va="center", fontsize=10.5) for rank in rank_axis]
    ax_belief.set_xlim(0, 1.08)
    ax_belief.set_title("Top 5 online beliefs", fontsize=15, fontweight="bold")
    ax_belief.set_xlabel("Estimated HIT probability", fontsize=11)
    ax_belief.set_yticks(rank_axis, [f"B{index + 1}" for index in rank_axis])
    ax_belief.invert_yaxis()
    ax_belief.tick_params(labelsize=10.5)

    status = fig.text(
        0.5,
        0.035,
        "Starting replay…",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#E2E8F0",
            "edgecolor": "#CBD5E1",
        },
    )
    frames = list(range(0, replay.num_slots, step))
    if frames[-1] != replay.num_slots - 1:
        frames.append(replay.num_slots - 1)

    def update(frame_slot):
        end = min(int(frame_slot), len(trace) - 1)
        hits = []
        misses = []
        for trace_state in trace[: end + 1]:
            for observation in trace_state["observations"]:
                point = (trace_state["slot"], observation["band"])
                (hits if observation["detected"] else misses).append(point)
        hit_scatter.set_offsets(np.asarray(hits) if hits else np.empty((0, 2)))
        miss_scatter.set_offsets(np.asarray(misses) if misses else np.empty((0, 2)))
        reset_events = _change_events(trace, end)
        reset_scatter.set_offsets(np.asarray(reset_events) if reset_events else np.empty((0, 2)))
        cursor.set_xdata([end, end])

        state = trace[end]
        selected_bands = {}
        for index, receiver in enumerate(state["receivers"]):
            band = receiver["band"]
            selected_bands[band] = index
            receiver_markers[index].set_data([band], [index])
            receiver_markers[index].set_marker("X" if receiver["pending"] else "o")
            receiver_markers[index].set_color(
                "#DC2626" if receiver["pending"] else receiver_colors[index]
            )
            reason = _recent_reason(trace, end, receiver["receiver"])
            assigned_now = any(
                assignment["receiver"] == receiver["receiver"]
                for assignment in state["assignments"]
            )
            if receiver["pending"]:
                action = "RETUNING"
            elif assigned_now:
                action = reason
            else:
                action = f"HOLD • last {reason}"
            receiver_labels[index].set_position((band, index + 0.22))
            receiver_labels[index].set_text(
                f"B{band + 1} • {_band_range_label(replay, band)}\n{action}"
            )

        ages = np.asarray(state["ages"], dtype=int)
        oldest = np.argsort(-ages, kind="stable")[:top_count]
        ax_age.set_yticklabels([f"B{band + 1}" for band in oldest])
        for rank, (bar, label, band) in enumerate(zip(age_bars, age_texts, oldest)):
            age = int(ages[band])
            bar.set_width(age)
            ratio = float(age) / max(1, MAX_AGE_SLOTS)
            bar.set_color(
                "#DC2626" if ratio >= 0.90 else "#F59E0B" if ratio >= 0.70 else "#60A5FA"
            )
            label.set_position((min(age + 1.2, ax_age.get_xlim()[1] - 5), rank))
            label.set_text(str(age))

        beliefs = np.asarray(state["belief"], dtype=float)
        strongest = np.argsort(-np.nan_to_num(beliefs, nan=-1.0), kind="stable")[:top_count]
        ax_belief.set_yticklabels([f"B{band + 1}" for band in strongest])
        for rank, (bar, label, band) in enumerate(zip(belief_bars, belief_texts, strongest)):
            belief = safe(beliefs[band])
            bar.set_width(belief)
            receiver_index = selected_bands.get(int(band))
            bar.set_color(
                receiver_colors[receiver_index] if receiver_index is not None else "#94A3B8"
            )
            label.set_position((min(belief + 0.025, 1.02), rank))
            label.set_text(f"{belief:.2f}")

        observations = sum(len(item["observations"]) for item in trace[: end + 1])
        fallbacks_seen = sum(bool(item["fallback"]) for item in trace[: end + 1])
        guard_state = "ACTIVE" if state["mandatory"] else "safe"
        status.set_text(
            f"Slot {end + 1}/{replay.num_slots}  |  HITs {len(hits)}/{observations}  |  "
            f"Retunes {state['retunes']}  |  CUSUM resets {state['change_detections']}  |  "
            f"Fallbacks {fallbacks_seen}  |  "
            f"Max-Age {guard_state}  |  Coverage violations {state['coverage_violations']}"
        )
        return [
            cursor,
            hit_scatter,
            miss_scatter,
            reset_scatter,
            status,
            *receiver_markers,
            *receiver_labels,
            *age_bars,
            *age_texts,
            *belief_bars,
            *belief_texts,
        ]

    update(replay.num_slots - 1)
    live_path = _save_figure_atomically(
        fig, output_dir / "rafts_live_dashboard.png"
    )

    gif_path = None
    animation = None
    if save_gif or show:
        animation = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=max(20, int(speed)),
            blit=False,
            repeat=True,
        )
        fig._rafts_animation = animation
    if save_gif:
        intended_gif_path = output_dir / "rafts_live_demo.gif"
        temporary_gif_path = intended_gif_path.with_name(
            f"{intended_gif_path.stem}.tmp{intended_gif_path.suffix}"
        )
        frames_per_second = max(2, min(15, round(1000 / max(20, int(speed)))))
        animation.save(
            temporary_gif_path,
            writer=PillowWriter(fps=frames_per_second),
            dpi=90,
        )
        gif_path = _finish_atomic_write(temporary_gif_path, intended_gif_path)
    if show:
        plt.show()
    plt.close(fig)

    outputs = [comparison_path, summary_path, live_path, evidence_path]
    if gif_path is not None:
        outputs.append(gif_path)
    print("\nGenerated outputs:")
    for path in outputs:
        print(" -", path)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/tsrd/config_0.h5")
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--bands", type=int, default=20)
    parser.add_argument("--min-pulses", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=int, default=250, help="Milliseconds per frame")
    parser.add_argument("--step", type=int, default=6, help="Slots advanced per frame")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    parser.add_argument("--show", action="store_true", help="Open the live window")
    args = parser.parse_args()
    run_visualization(
        file=args.file,
        slots=args.slots,
        bands=args.bands,
        min_pulses=args.min_pulses,
        seed=args.seed,
        speed=args.speed,
        step=args.step,
        output_dir=args.output_dir,
        save_gif=not args.no_gif,
        show=args.show,
    )


if __name__ == "__main__":
    main()
