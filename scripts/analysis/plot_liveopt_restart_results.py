#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "LiveOpt": "outputs/liveopt_tss_earlystop_guard_200max_10seed_20260705",
    "Fixed full": "outputs/liveopt_tss_earlystop_fixed_full_200max_10seed_20260705",
    "Fixed warm": "outputs/liveopt_tss_earlystop_fixed_warm_200max_10seed_20260705",
    "Fixed population": "outputs/liveopt_tss_earlystop_fixed_population_200max_10seed_20260705",
}

COLORS = {
    "LiveOpt": "#0B6E69",
    "Fixed full": "#333333",
    "Fixed warm": "#D9822B",
    "Fixed population": "#6C63FF",
}

LINESTYLES = {
    "LiveOpt": "-",
    "Fixed full": "--",
    "Fixed warm": "-.",
    "Fixed population": ":",
}

TRAJECTORY_LABELS = {
    "LiveOpt": "LiveOpt",
    "Fixed full": "FixedFull",
    "Fixed warm": "FixedWarm",
    "Fixed population": "FixedPopulation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LiveOpt restart replay figures.")
    parser.add_argument(
        "--trajectory-dir",
        default="outputs/liveopt_tss_earlystop_restart_compare_200max_10seed_20260705",
        help="Directory containing trajectory_summary.csv exported from all restart policies.",
    )
    parser.add_argument("--out-dir", default="outputs/liveopt_tss_earlystop_restart_compare_200max_10seed_20260705/figures")
    parser.add_argument("--episode-id", action="append", default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def stage_label(stage: int) -> str:
    return f"t{stage:02d}"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_stage_summary() -> dict[str, list[dict[str, str]]]:
    data: dict[str, list[dict[str, str]]] = {}
    for label, run_dir in RUNS.items():
        path = Path(run_dir) / "reference_metrics" / "reference_stage_summary.csv"
        data[label] = read_csv(path)
    return data


def load_stage_metrics() -> dict[str, list[dict[str, str]]]:
    data: dict[str, list[dict[str, str]]] = {}
    for label, run_dir in RUNS.items():
        path = Path(run_dir) / "reference_metrics" / "reference_stage_metrics.csv"
        data[label] = read_csv(path)
    return data


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_stage_overview(stage_summary: dict[str, list[dict[str, str]]], out_dir: Path) -> None:
    metrics = [
        ("normalized_hv_mean", "Normalized HV ↑"),
        ("igd_mean", "IGD ↓"),
        ("true_pass_ratio", "True pass ratio ↑"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.65), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label, rows in stage_summary.items():
            rows = sorted(rows, key=lambda row: int(row["stage_index"]))
            xs = [int(row["stage_index"]) for row in rows]
            ys = []
            for row in rows:
                if metric == "true_pass_ratio":
                    value = f(row.get("true_pass_ratio"))
                    if math.isnan(value):
                        value = f(row.get("feasibility_acc"))
                else:
                    value = f(row.get(metric))
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=COLORS[label],
                linestyle=LINESTYLES[label],
                label=label,
            )
        ax.axvspan(10.5, 12.5, color="#F2D5D5", alpha=0.28, lw=0)
        ax.set_xlabel("Stage")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(13))
        ax.set_xticklabels([stage_label(i) for i in range(13)], rotation=45, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=False, loc="lower left")
    savefig(fig, out_dir, "restart_stage_overview")


def plot_policy_bars(stage_summary: dict[str, list[dict[str, str]]], out_dir: Path) -> None:
    labels = list(RUNS)
    panels = [
        ("mean_hv", "Mean normalized HV ↑"),
        ("mean_igd", "Mean IGD ↓"),
        ("t12_hv", "t12 normalized HV ↑"),
        ("t12_igd", "t12 IGD ↓"),
    ]
    values: dict[str, dict[str, float]] = {label: {} for label in labels}
    for label, rows in stage_summary.items():
        rows = sorted(rows, key=lambda row: int(row["stage_index"]))
        dyn = [row for row in rows if int(row["stage_index"]) >= 1]
        t12 = next(row for row in rows if int(row["stage_index"]) == 12)
        values[label]["mean_hv"] = float(np.nanmean([f(row["normalized_hv_mean"]) for row in dyn]))
        values[label]["mean_igd"] = float(np.nanmean([f(row["igd_mean"]) for row in dyn]))
        values[label]["t12_hv"] = f(t12["normalized_hv_mean"])
        values[label]["t12_igd"] = f(t12["igd_mean"])

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.8), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), panels):
        xs = np.arange(len(labels))
        ys = [values[label][key] for label in labels]
        bars = ax.bar(xs, ys, color=[COLORS[label] for label in labels], width=0.68)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        for bar, value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    savefig(fig, out_dir, "restart_policy_bar_summary")


def plot_delta_heatmaps(stage_metrics: dict[str, list[dict[str, str]]], out_dir: Path) -> None:
    base = stage_metric_grid(stage_metrics["LiveOpt"], "normalized_total_score")
    comparisons = ["Fixed full", "Fixed warm", "Fixed population"]
    episodes = sorted(base)
    stages = list(range(13))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), constrained_layout=True)
    vmax = 0.15
    for ax, label in zip(axes, comparisons):
        other = stage_metric_grid(stage_metrics[label], "normalized_total_score")
        matrix = np.full((len(episodes), len(stages)), np.nan)
        for i, episode in enumerate(episodes):
            for j, stage in enumerate(stages):
                matrix[i, j] = base.get(episode, {}).get(stage, np.nan) - other.get(episode, {}).get(stage, np.nan)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"LiveOpt − {label}")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([stage_label(stage) for stage in stages], rotation=45, ha="right")
        ax.set_yticks(range(len(episodes)))
        ax.set_yticklabels(episodes)
        ax.set_xlabel("Stage")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Normalized total score difference")
    savefig(fig, out_dir, "restart_liveopt_delta_heatmaps")


def stage_metric_grid(rows: list[dict[str, str]], metric: str) -> dict[str, dict[int, float]]:
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        episode = row["episode_id"]
        stage = int(row["stage_index"])
        value = f(row.get(metric))
        if not math.isnan(value):
            buckets[(episode, stage)].append(value)
    grid: dict[str, dict[int, float]] = defaultdict(dict)
    for (episode, stage), values in buckets.items():
        grid[episode][stage] = float(np.nanmean(values))
    return grid


def plot_episode_trajectories(summary_rows: list[dict[str, str]], out_dir: Path, episode_id: str, metric: str) -> None:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    ylabel = "Normalized HV ↑" if metric == "normalized_hv" else "IGD ↓"
    stages = list(range(1, 13))
    fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.6), sharex=False, sharey=False, constrained_layout=True)
    for ax, stage in zip(axes.ravel(), stages):
        stage_rows = [
            row
            for row in summary_rows
            if row["episode_id"] == episode_id and int(row["stage_index"]) == stage
        ]
        for label in RUNS:
            csv_label = TRAJECTORY_LABELS[label]
            rows = sorted([row for row in stage_rows if row["label"] == csv_label], key=lambda row: int(row["generation"]))
            if not rows:
                continue
            xs = np.array([int(row["generation"]) for row in rows], dtype=float)
            ys = np.array([f(row.get(mean_key)) for row in rows], dtype=float)
            std = np.array([f(row.get(std_key), 0.0) for row in rows], dtype=float)
            ax.plot(xs, ys, color=COLORS[label], linestyle=LINESTYLES[label], linewidth=1.3, label=label)
            ax.fill_between(xs, ys - std, ys + std, color=COLORS[label], alpha=0.10, linewidth=0)
        ax.set_title(stage_label(stage))
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.5, alpha=0.75)
        x_max = max(
            [int(row["generation"]) for row in stage_rows if row.get("generation") not in {None, ""}]
            or [200]
        )
        ax.set_xlim(0, max(1, x_max))
        if stage in {1, 7}:
            ax.set_ylabel(ylabel)
        if stage >= 7:
            ax.set_xlabel("Generation")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.11))
    savefig(fig, out_dir, f"{episode_id.lower()}_{metric}_trajectory_2x6")


def trajectory_groups(summary_rows: list[dict[str, str]]) -> dict[tuple[str, int, str], list[dict[str, str]]]:
    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    reverse_labels = {csv_label: label for label, csv_label in TRAJECTORY_LABELS.items()}
    for row in summary_rows:
        label = reverse_labels.get(row["label"], row["label"])
        groups[(row["episode_id"], int(row["stage_index"]), label)].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: int(row["generation"]))
    return groups


def hv_series(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([int(row["generation"]) for row in rows], dtype=float)
    ys = np.array([f(row.get("normalized_hv_mean")) for row in rows], dtype=float)
    keep = ~np.isnan(ys)
    return xs[keep], ys[keep]


def auc_until(rows: list[dict[str, str]], end_generation: int) -> float:
    xs, ys = hv_series(rows)
    keep = xs <= end_generation
    xs = xs[keep]
    ys = ys[keep]
    if len(xs) < 2:
        return float("nan")
    return float(np.trapz(ys, xs) / max(1.0, float(end_generation)))


def time_to_threshold(rows: list[dict[str, str]], threshold: float) -> float:
    xs, ys = hv_series(rows)
    if len(xs) == 0:
        return float("nan")
    reached = xs[ys >= threshold]
    if len(reached) == 0:
        return 101.0
    return float(np.min(reached))


def plot_early_adaptation_bars(summary_rows: list[dict[str, str]], out_dir: Path) -> None:
    groups = trajectory_groups(summary_rows)
    labels = list(RUNS)
    windows = [10, 20, 50]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.9), constrained_layout=True)
    for ax, window in zip(axes, windows):
        values = []
        for label in labels:
            aucs = [
                auc_until(rows, window)
                for (episode, stage, group_label), rows in groups.items()
                if group_label == label and stage >= 1
            ]
            values.append(float(np.nanmean(aucs)))
        xs = np.arange(len(labels))
        bars = ax.bar(xs, values, color=[COLORS[label] for label in labels], width=0.68)
        ax.set_title(f"AUC-HV, gen 0-{window} ↑")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ymin = min(values) - 0.02
        ymax = max(values) + 0.02
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7.5)
    savefig(fig, out_dir, "restart_early_auc_hv_bars")


def plot_time_to_threshold(summary_rows: list[dict[str, str]], out_dir: Path) -> None:
    groups = trajectory_groups(summary_rows)
    labels = list(RUNS)
    thresholds = [0.90, 0.95, 0.98]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.9), constrained_layout=True)
    for ax, ratio in zip(axes, thresholds):
        values = []
        for label in labels:
            times = []
            for episode in sorted({key[0] for key in groups}):
                for stage in range(1, 13):
                    final_candidates = []
                    for candidate_label in labels:
                        rows = groups.get((episode, stage, candidate_label), [])
                        if rows:
                            _, ys = hv_series(rows)
                            if len(ys):
                                final_candidates.append(float(ys[-1]))
                    if not final_candidates:
                        continue
                    threshold = ratio * max(final_candidates)
                    rows = groups.get((episode, stage, label), [])
                    if rows:
                        times.append(time_to_threshold(rows, threshold))
            values.append(float(np.nanmean(times)))
        xs = np.arange(len(labels))
        bars = ax.bar(xs, values, color=[COLORS[label] for label in labels], width=0.68)
        ax.set_title(f"Gen. to {int(ratio*100)}% best HV ↓")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5)
    savefig(fig, out_dir, "restart_time_to_threshold_bars")


def plot_average_hv_gain(summary_rows: list[dict[str, str]], out_dir: Path) -> None:
    groups = trajectory_groups(summary_rows)
    generations = sorted({int(row["generation"]) for row in summary_rows})
    baselines = ["Fixed full", "Fixed warm", "Fixed population"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0), constrained_layout=True)

    for label in RUNS:
        means = []
        for gen in generations:
            vals = []
            for (episode, stage, group_label), rows in groups.items():
                if group_label != label or stage < 1:
                    continue
                row = next((item for item in rows if int(item["generation"]) == gen), None)
                if row:
                    vals.append(f(row.get("normalized_hv_mean")))
            means.append(float(np.nanmean(vals)))
        axes[0].plot(generations, means, color=COLORS[label], linestyle=LINESTYLES[label], linewidth=1.8, label=label)

    for baseline in baselines:
        gains = []
        for gen in generations:
            vals = []
            for episode in sorted({key[0] for key in groups}):
                for stage in range(1, 13):
                    live = groups.get((episode, stage, "LiveOpt"), [])
                    base = groups.get((episode, stage, baseline), [])
                    live_row = next((item for item in live if int(item["generation"]) == gen), None)
                    base_row = next((item for item in base if int(item["generation"]) == gen), None)
                    if live_row and base_row:
                        vals.append(f(live_row.get("normalized_hv_mean")) - f(base_row.get("normalized_hv_mean")))
            gains.append(float(np.nanmean(vals)))
        axes[1].plot(generations, gains, color=COLORS[baseline], linestyle=LINESTYLES[baseline], linewidth=1.8, label=f"vs {baseline}")

    axes[0].set_title("Mean normalized HV over search")
    axes[0].set_ylabel("Normalized HV ↑")
    axes[1].set_title("LiveOpt HV gain over fixed policies")
    axes[1].set_ylabel("HV difference ↑")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.set_xlim(0, max(generations) if generations else 200)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.axvspan(0, 20, color="#EAF4F4", alpha=0.65, lw=0)
        ax.legend(frameon=False)
    savefig(fig, out_dir, "restart_average_hv_gain")


def main() -> int:
    args = parse_args()
    setup_style()
    out_dir = Path(args.out_dir)
    stage_summary = load_stage_summary()
    stage_metrics = load_stage_metrics()
    trajectory_rows = read_csv(Path(args.trajectory_dir) / "trajectory_summary.csv")
    episodes = args.episode_id or sorted({row["episode_id"] for row in trajectory_rows})

    plot_stage_overview(stage_summary, out_dir)
    plot_policy_bars(stage_summary, out_dir)
    plot_delta_heatmaps(stage_metrics, out_dir)
    plot_early_adaptation_bars(trajectory_rows, out_dir)
    plot_time_to_threshold(trajectory_rows, out_dir)
    plot_average_hv_gain(trajectory_rows, out_dir)
    for episode_id in episodes:
        plot_episode_trajectories(trajectory_rows, out_dir, episode_id, "normalized_hv")
        plot_episode_trajectories(trajectory_rows, out_dir, episode_id, "igd")

    manifest = {
        "out_dir": str(out_dir),
        "stage_overview": str(out_dir / "restart_stage_overview.pdf"),
        "bar_summary": str(out_dir / "restart_policy_bar_summary.pdf"),
        "delta_heatmaps": str(out_dir / "restart_liveopt_delta_heatmaps.pdf"),
        "early_auc": str(out_dir / "restart_early_auc_hv_bars.pdf"),
        "time_to_threshold": str(out_dir / "restart_time_to_threshold_bars.pdf"),
        "average_hv_gain": str(out_dir / "restart_average_hv_gain.pdf"),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
