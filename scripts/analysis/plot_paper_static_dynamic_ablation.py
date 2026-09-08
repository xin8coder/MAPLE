#!/usr/bin/env python3
"""Build the three compact main-paper experiment figures from retained artifacts.

Static values are the exact entries exported in
``release_artifacts/paper_table_exports/static_solving_nlp4lp_rows.tex`` and the NLDO view summaries.
Dynamic trajectories are recomputed from retained evaluator rows; missing public-
baseline stages count as zero, matching the cumulative paper protocol.
"""

from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "release_artifacts" / "paper_figure_exports" / "results"
FULL = ROOT / "outputs/liveopt_tss_full_195_200init200_100dyn100_adaptive_10seed_20260705/reference_metrics_latest_20260709/reference_stage_metrics.csv"
HIDDEN = ROOT / "outputs/hidden_cold_start_200x200x10_20260708/reference_metrics_latest_20260708/reference_stage_metrics.csv"
PUBLIC_FILES = [
    ROOT / "logs/llm_tests/tss_dynamic_public_react_optimai_orllm_equal_repair_merged_20260709/dynamic_public_baseline_stage_rows.jsonl",
    ROOT / "logs/llm_tests/tss_dynamic_public_optimus_orlm_equal_repair_merged_20260709/dynamic_public_baseline_stage_rows.jsonl",
]
ABLATION = ROOT / "logs/analysis/liveopt_component_ablation_true_modules_20260709.json"
COLORS = {
    "LiveOpt": "#167C80",
    "ReAct": "#4C78A8",
    "OR-Agent": "#F28E2B",
    "OptiMUS": "#8F63B8",
    "OptimAI": "#D65F5F",
    "Hidden reference": "#6C7178",
    "w/o LSM": "#59A14F",
    "w/o TSS": "#E15759",
    "fixed warm": "#B07AA1",
}


def style_axes(ax, title: str, ylabel: str = "Score (0--100)") -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(ax, groups, series, values, width=0.16, ylim=(0, 105)) -> None:
    x = np.arange(len(groups))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for off, name in zip(offsets, series):
        vals = values[name]
        bars = ax.bar(x + off, vals, width, label=name, color=COLORS[name], edgecolor="white", linewidth=0.35)
        for bar, val in zip(bars, vals):
            if val >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.4, f"{val:.0f}", ha="center", va="bottom", fontsize=5.4)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)


def plot_static() -> None:
    methods = ["LiveOpt", "ReAct", "OR-Agent", "OptiMUS", "OptimAI"]
    acc = {
        "LiveOpt": [33.9, 80.0, 100.0],
        "ReAct": [27.1, 75.0, 88.9],
        "OR-Agent": [50.8, 65.0, 77.8],
        "OptiMUS": [69.5, 65.0, 88.9],
        "OptimAI": [47.5, 60.0, 55.6],
    }
    nlp_compile_run = {
        "LiveOpt": [93.2, 93.2], "ReAct": [91.5, 78.0], "OR-Agent": [96.6, 96.6],
        "OptiMUS": [96.6, 93.2], "OptimAI": [100.0, 98.3],
    }
    bwor_compile_run = {
        "LiveOpt": [100.0, 100.0], "ReAct": [100.0, 95.0], "OR-Agent": [95.0, 95.0],
        "OptiMUS": [100.0, 100.0], "OptimAI": [80.0, 80.0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True)
    grouped_bars(axes[0], ["NLP4LP", "BWOR20", "NLDO-SS"], methods, acc, width=0.145)
    style_axes(axes[0], "(a) Executable static accuracy")

    grouped_bars(axes[1], ["Compile", "Run"], methods, nlp_compile_run, width=0.145)
    style_axes(axes[1], "(b) NLP4LP execution funnel")

    grouped_bars(axes[2], ["Compile", "Run"], methods, bwor_compile_run, width=0.145)
    style_axes(axes[2], "(c) BWOR20 execution funnel")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=6.8)
    save(fig, "main_static_experiments")


def _profile(pid: str) -> str:
    return "scalar" if int(pid.split("P")[-1]) <= 9 else "pareto"


def _read_reference(path: Path):
    bucket = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            stage = int(row["stage_index"])
            field = "normalized_score" if _profile(row["episode_id"]) == "scalar" else "normalized_hv"
            val = float(row[field]) if row[field] else 0.0
            bucket[(_profile(row["episode_id"]), stage)].append(val)
    return {k: float(np.mean(v)) * 100 for k, v in bucket.items()}


def _read_public_scalar():
    names = {"react_tools": "ReAct", "or_llm_agent_2025": "OR-Agent", "optimus": "OptiMUS", "optimai_2025": "OptimAI"}
    found = defaultdict(dict)
    for path in PUBLIC_FILES:
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                if row["method"] not in names:
                    continue
                if _profile(row["episode_id"]) != "scalar":
                    continue
                score = float(row.get("hidden_evaluation", {}).get("normalized_score") or 0.0)
                found[(names[row["method"]], int(row["stage_index"]))][row["episode_id"]] = score
    # The cumulative protocol has nine scalar episodes; absent stages are zeros.
    return {(name, stage): sum(found[(name, stage)].values()) / 9 * 100
            for name in names.values() for stage in range(13)}


def plot_dynamic() -> None:
    live = _read_reference(FULL)
    hidden = _read_reference(HIDDEN)
    public = _read_public_scalar()
    stages = np.arange(13)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True)
    for name, lw, marker in [("LiveOpt", 2.0, "o"), ("ReAct", 1.1, None), ("OR-Agent", 1.1, None), ("OptiMUS", 1.1, None), ("OptimAI", 1.1, None)]:
        ys = [live[("scalar", s)] if name == "LiveOpt" else public[(name, s)] for s in stages]
        axes[0].plot(stages, ys, color=COLORS[name], linewidth=lw, marker=marker, markersize=2.6, label=name)
    axes[0].plot(stages, [hidden[("scalar", s)] for s in stages], color=COLORS["Hidden reference"], linestyle="--", linewidth=1.2, label="Hidden reference")
    axes[0].set_xticks([0, 3, 6, 9, 12])
    axes[0].set_ylim(-3, 105)
    axes[0].set_xlabel("Update stage", fontsize=7.5)
    style_axes(axes[0], "(a) Dynamic scalar quality")

    for name, data, ls, lw in [("LiveOpt", live, "-", 2.0), ("Hidden reference", hidden, "--", 1.35)]:
        axes[1].plot(stages, [data[("pareto", s)] for s in stages], color=COLORS[name], linestyle=ls,
                     linewidth=lw, marker="o" if name == "LiveOpt" else None, markersize=2.6, label=name)
    axes[1].set_xticks([0, 3, 6, 9, 12])
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Update stage", fontsize=7.5)
    style_axes(axes[1], "(b) Dynamic Pareto HV")

    groups = ["Scalar\nSS", "Scalar\nDS", "Pareto\nSM", "Pareto\nDM"]
    vals = {
        "LiveOpt": [100.0, 95.448, 76.676, 70.354],
        "Hidden reference": [100.0, 100.0, 96.825, 97.502],
    }
    grouped_bars(axes[2], groups, ["LiveOpt", "Hidden reference"], vals, width=0.30)
    style_axes(axes[2], "(c) Static-to-dynamic retention")

    handles, labels = axes[0].get_legend_handles_labels()
    # The Pareto panel intentionally excludes non-native archive workflows.
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, fontsize=6.5)
    save(fig, "main_dynamic_experiments")


def plot_ablation() -> None:
    data = json.loads(ABLATION.read_text())
    labels = ["LiveOpt", "w/o LSM", "w/o TSS", "fixed warm"]
    raw_names = ["LiveOpt full", "LiveOpt w/o LSM", "LiveOpt w/o TSS", "LiveOpt w/o Adaptive Restart"]
    rows = {r["label"]: [float(x) * 100 for x in r["values"]] for r in data["rows"]}
    vals = {label: rows[raw] for label, raw in zip(labels, raw_names)}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15), constrained_layout=True)
    x = np.arange(4)
    for label in labels:
        axes[0].plot(x, vals[label], marker="o", markersize=3, linewidth=1.6,
                     color=COLORS[label], label=label)
    axes[0].set_xticks(x, ["SS", "SM", "DS", "DM"])
    axes[0].set_ylim(0, 105)
    style_axes(axes[0], "(a) Subsystem profile")

    grouped_bars(axes[1], ["DS quality"], labels, {k: [v[2]] for k, v in vals.items()}, width=0.16)
    style_axes(axes[1], "(b) Dynamic scalar ablation")

    grouped_bars(axes[2], ["DM HV"], labels, {k: [v[3]] for k, v in vals.items()}, width=0.16)
    style_axes(axes[2], "(c) Dynamic Pareto ablation")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False, fontsize=6.8)
    save(fig, "main_ablation_experiments")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_static()
    plot_dynamic()
    plot_ablation()


if __name__ == "__main__":
    main()
