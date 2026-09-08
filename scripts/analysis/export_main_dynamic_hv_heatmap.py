#!/usr/bin/env python3
"""Export Table 2 as a grouped DS/DM stage heatmap.

Each method occupies one large row.  Its left block contains the nine scalar
episodes P001--P009 and its right block the six Pareto episodes P010--P015;
columns are the initial state t00 and public updates t01--t12.  Scalar cells use feasibility-gated
normalized quality and Pareto cells use feasibility-gated normalized HV.
Missing suffixes and hidden-infeasible outputs score zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DS_EPISODES = [f"NLDO-P{number:03d}" for number in range(1, 10)]
DM_EPISODES = [f"NLDO-P{number:03d}" for number in range(10, 16)]
EPISODES = DS_EPISODES + DM_EPISODES
STAGES = list(range(0, 13))
METHODS = (
    ("LiveOpt", "liveopt"),
    ("Persistent ReAct", "persistent_react"),
    ("ReAct", "react_tools"),
    ("OptiMUS", "optimus"),
    ("ORLM", "orlm"),
    ("OptimAI", "optimai_2025"),
    ("OR-LLM-Agent", "or_llm_agent_2025"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the grouped DS/DM heatmap used as Table 2.")
    parser.add_argument(
        "--liveopt-metrics",
        type=Path,
        action="append",
        default=None,
        help="LiveOpt stage-metric CSV; later files replace duplicate episode/seed/stage rows.",
    )
    parser.add_argument(
        "--public-stage-rows",
        type=Path,
        action="append",
        default=None,
        help="Merged public-baseline stage JSONL; repeat for multiple source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release_artifacts/paper_figure_exports/results/main_dynamic_hv_heatmap.pdf",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=ROOT / "release_artifacts/paper_figure_exports/results/main_dynamic_hv_heatmap.png",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "outputs/main_dynamic_hv_heatmap_20260715/summary.json",
    )
    parser.add_argument(
        "--summary-tex",
        type=Path,
        default=ROOT / "release_artifacts/paper_table_exports/main_dynamic_hv_summary_rows.tex",
    )
    parser.add_argument(
        "--omit-liveopt",
        action="store_true",
        help="Export only the five external methods (used for the Kimi baseline-only repeat).",
    )
    parser.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="Diagnostic only: score a missing row after a feasible prefix as zero instead of rejecting export.",
    )
    args = parser.parse_args()
    if args.liveopt_metrics is None:
        args.liveopt_metrics = [
            ROOT
            / "outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713/reference_metrics/reference_stage_metrics.csv",
            ROOT
            / "outputs/liveopt_ablation_llmbackbone_sequential_p007_p009_200x200x10_trace_t11pop_20260714/reference_metrics/reference_stage_metrics.csv",
            ROOT
            / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
        ]
    if args.public_stage_rows is None:
        args.public_stage_rows = [
            ROOT
            / "logs/llm_tests/table2_deepseek_continuation_20260717/merged/dynamic_public_baseline_stage_rows.rescored.jsonl",
            ROOT
            / "logs/llm_tests/tss_dynamic_public_persistent_react_final_merged_20260721/dynamic_public_baseline_stage_rows.jsonl",
        ]
    return args


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value not in {None, ""}:
            return float(value)
    return None


def metric_pass(row: dict[str, Any]) -> bool:
    true_pass = row.get("true_pass")
    if true_pass not in {None, ""}:
        return parse_bool(true_pass)
    return parse_bool(row.get("feasible"))


def stage_value(mapping: dict[str, Any], episode_id: str) -> float:
    if episode_id in DS_EPISODES:
        value = first_float(mapping, "normalized_score", "normalized_total_score")
    else:
        value = first_float(mapping, "normalized_hv", "normalized_score", "normalized_total_score")
    return 0.0 if value is None else float(value)


def load_liveopt(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    overrides = 0
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                episode_id = str(row.get("episode_id") or "")
                stage = int(float(row.get("stage_index") or 0))
                if episode_id not in EPISODES or stage not in STAGES:
                    continue
                key = (episode_id, int(float(row.get("run_seed") or 0)), stage)
                if key in merged:
                    overrides += 1
                merged[key] = row

    grouped: dict[tuple[str, int], list[tuple[float, bool]]] = {}
    for (episode_id, _, stage), row in merged.items():
        passed = metric_pass(row)
        value = stage_value(row, episode_id) if passed else 0.0
        grouped.setdefault((episode_id, stage), []).append((value, passed))

    values = np.zeros((len(EPISODES), len(STAGES)), dtype=float)
    feasible = np.zeros_like(values, dtype=bool)
    for episode_index, episode_id in enumerate(EPISODES):
        for stage_index, stage in enumerate(STAGES):
            cell = grouped.get((episode_id, stage), [])
            if len(cell) != 10:
                raise ValueError(f"{episode_id} t{stage:02d}: expected 10 LiveOpt values, got {len(cell)}")
            values[episode_index, stage_index] = fmean(value for value, _ in cell)
            feasible[episode_index, stage_index] = all(passed for _, passed in cell)
    return values, feasible, overrides


def load_public(
    paths: list[Path],
    *,
    allow_incomplete_coverage: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], list[dict[str, Any]]]:
    values = {
        key: np.zeros((len(EPISODES), len(STAGES)), dtype=float)
        for _, key in METHODS
        if key != "liveopt"
    }
    feasible = {key: np.zeros_like(matrix, dtype=bool) for key, matrix in values.items()}
    seen: dict[tuple[str, str, int], tuple[float, bool, Path]] = {}
    episode_index = {episode_id: index for index, episode_id in enumerate(EPISODES)}
    stage_index = {stage: index for index, stage in enumerate(STAGES)}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = str(row.get("method") or "")
                episode_id = str(row.get("episode_id") or "")
                stage = int(row.get("stage_index") or 0)
                if key not in values or episode_id not in episode_index or stage not in stage_index:
                    continue
                hidden = row.get("hidden_evaluation") or {}
                passed = parse_bool(hidden.get("feasible"))
                value = stage_value(hidden, episode_id) if passed else 0.0
                cell = (key, episode_id, stage)
                if cell in seen and (abs(seen[cell][0] - value) > 1e-12 or seen[cell][1] != passed):
                    raise ValueError(f"conflicting public result for {cell}: {seen[cell]} versus {(value, passed, path)}")
                seen[cell] = (value, passed, path)
    # Enforce the paper's sequential-commit contract.  A missing or hidden-
    # infeasible update terminates the episode; independently generated later
    # rows are deliberately excluded rather than reviving a failed prefix.
    coverage_gaps: list[dict[str, Any]] = []
    for key in values:
        for episode_id in EPISODES:
            active_prefix = True
            for stage in STAGES:
                record = seen.get((key, episode_id, stage))
                if not active_prefix:
                    continue
                if record is None:
                    coverage_gaps.append(
                        {"method": key, "episode_id": episode_id, "missing_stage": stage}
                    )
                    active_prefix = False
                    continue
                value, passed, _ = record
                row_index = episode_index[episode_id]
                column_index = stage_index[stage]
                values[key][row_index, column_index] = value if passed else 0.0
                feasible[key][row_index, column_index] = passed
                if not passed:
                    active_prefix = False
    if coverage_gaps and not allow_incomplete_coverage:
        preview = ", ".join(
            f"{row['method']}/{row['episode_id']}/t{row['missing_stage']:02d}"
            for row in coverage_gaps[:8]
        )
        raise ValueError(
            f"public Table 2 evidence has {len(coverage_gaps)} gaps after feasible prefixes: "
            f"{preview}. Continue those trajectories or pass --allow-incomplete-coverage for diagnostics."
        )
    counts = {key: sum(1 for method, _, _ in seen if method == key) for key in values}
    return values, feasible, counts, coverage_gaps


def block_stats(matrix: np.ndarray, feasible: np.ndarray, indices: slice) -> tuple[float, int, int]:
    block = matrix[indices]
    solved = feasible[indices]
    return float(np.mean(block)), int(np.count_nonzero(solved)), int(block.size)


def render(
    data: dict[str, np.ndarray],
    feasible: dict[str, np.ndarray],
    output: Path,
    png_output: Path,
    methods: tuple[tuple[str, str], ...] = METHODS,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    # Viridis is a conventional sequential heatmap scale: it has monotone
    # luminance and never passes through paper-white, so mid-range cells do
    # not disappear against the page background.
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        len(methods),
        1,
        figsize=(3.55, 1.48 * len(methods) / len(METHODS)),
        squeeze=False,
        gridspec_kw={
            "hspace": 0.0,
            "top": 0.91,
            "bottom": 0.21,
            "left": 0.205,
            "right": 0.992,
        },
    )
    image = None
    for method_index, (label, key) in enumerate(methods):
        ax = axes[method_index, 0]
        image = ax.imshow(data[key], vmin=0.0, vmax=1.0, cmap=cmap, interpolation="nearest", aspect="auto")
        # Every method uses the same top-to-bottom order P001--P015.  Repeating
        # 15 labels inside each very thin block makes them collide, so the
        # figure labels only the method and DS/DM split; the caption gives the
        # exact problem order.
        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(len(STAGES)))
        ax.tick_params(axis="x", length=0, pad=1.1, labelsize=4.4)
        ax.axhline(len(DS_EPISODES) - 0.5, color="#202020", linewidth=0.34, zorder=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if method_index == 0:
            ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.set_xticklabels([f"t{stage:02d}" for stage in STAGES])
        else:
            ax.tick_params(axis="x", top=False, labeltop=False, bottom=False, labelbottom=False)

        ax.text(
            -0.050,
            0.5,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.55,
            fontweight="bold",
        )
        ax.text(-0.005, 0.70, "DS", transform=ax.transAxes, ha="right", va="center", fontsize=4.0, fontweight="bold")
        ax.text(-0.005, 0.20, "DM", transform=ax.transAxes, ha="right", va="center", fontsize=4.0, fontweight="bold")

    if image is None:
        raise RuntimeError("no heatmap image was rendered")
    colorbar_axis = fig.add_axes([0.735, 0.035, 0.255, 0.050])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal", ticks=[0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=4.2, length=1.5, pad=0.7)
    colorbar.outline.set_linewidth(0.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(png_output, dpi=320, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def render_summary_tex(
    method_summary: dict[str, dict[str, Any]],
    methods: tuple[tuple[str, str], ...] = METHODS,
) -> str:
    def format_rate(value: float) -> str:
        return rf"{value:.0f}\%" if abs(value - round(value)) < 1e-9 else rf"{value:.1f}\%"

    lines = [
        "% Generated by scripts/analysis/export_main_dynamic_hv_heatmap.py.",
        r"\newcommand{\MainDynamicHVSummaryRows}{%",
    ]
    for label, _ in methods:
        row = method_summary[label]
        display = r"\method{}" if label == "LiveOpt" else label
        ds_solve_rate = 100.0 * row["ds"]["solved"] / row["ds"]["cells"]
        dm_solve_rate = 100.0 * row["dm"]["solved"] / row["dm"]["cells"]
        if label == "LiveOpt":
            lines.append(r"\rowcolor{LiveOptRowAccent}")
        elif label in {"OptiMUS", "OptimAI"}:
            lines.append(r"\rowcolor{LiveOptBaselineRowB}")
        lines.append(
            f"{display} & {format_rate(ds_solve_rate)} & "
            f"{row['ds']['mean_quality']:.3f} & {format_rate(dm_solve_rate)} & "
            f"{row['dm']['mean_hv']:.3f} " + r"\\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    methods = METHODS[1:] if args.omit_liveopt else METHODS
    liveopt_values, liveopt_feasible, liveopt_overrides = load_liveopt(list(args.liveopt_metrics))
    public_values, public_feasible, seen_counts, coverage_gaps = load_public(
        list(args.public_stage_rows),
        allow_incomplete_coverage=args.allow_incomplete_coverage,
    )
    matrices = {"liveopt": liveopt_values, **public_values}
    feasibility = {"liveopt": liveopt_feasible, **public_feasible}
    render(matrices, feasibility, args.output, args.png_output, methods)

    method_summary: dict[str, Any] = {}
    for label, key in methods:
        ds_mean, ds_solved, ds_total = block_stats(matrices[key], feasibility[key], slice(0, len(DS_EPISODES)))
        dm_mean, dm_solved, dm_total = block_stats(
            matrices[key], feasibility[key], slice(len(DS_EPISODES), len(EPISODES))
        )
        method_summary[label] = {
            "key": key,
            "ds": {"mean_quality": ds_mean, "solved": ds_solved, "cells": ds_total},
            "dm": {"mean_hv": dm_mean, "solved": dm_solved, "cells": dm_total},
            "observed_public_rows": None if key == "liveopt" else seen_counts[key],
            "matrix": matrices[key].tolist(),
        }
    summary = {
        "ds_episodes": DS_EPISODES,
        "dm_episodes": DM_EPISODES,
        "stages": STAGES,
        "missing_stage_policy": "score_zero",
        "coverage_gaps_after_feasible_prefix": coverage_gaps,
        "liveopt_source_overrides": liveopt_overrides,
        "methods": method_summary,
        "sources": {
            "liveopt_metrics": [str(path) for path in args.liveopt_metrics],
            "public_stage_rows": [str(path) for path in args.public_stage_rows],
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    args.summary_tex.parent.mkdir(parents=True, exist_ok=True)
    args.summary_tex.write_text(render_summary_tex(method_summary, methods), encoding="utf-8")
    print(
        json.dumps(
            {
                label: {
                    "DS": f"{row['ds']['solved']}/{row['ds']['cells']} @ {row['ds']['mean_quality']:.6f}",
                    "DM": f"{row['dm']['solved']}/{row['dm']['cells']} @ {row['dm']['mean_hv']:.6f}",
                }
                for label, row in method_summary.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
