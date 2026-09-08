#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt


METHODS = (
    ("LiveOpt", "liveopt", "#12335D", "-"),
    ("LiveOpt w/o TSS", "no_tss", "#9B5DE5", ":"),
    ("Fixed Warm", "fixed_warm", "#E9B44C", "--"),
    ("Fixed Full", "fixed_full", "#2A9D9F", "-."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export P010 large-shift trajectory figures from matched runs.")
    for _, key, _, _ in METHODS:
        parser.add_argument(f"--{key.replace('_', '-')}-run-dir", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Optional canonical restart-shadow source manifest. When supplied, "
            "LiveOpt/Fixed Warm/Fixed Full traces are loaded from the audited "
            "per-stage sources recorded in the manifest."
        ),
    )
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--appendix-output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-tex", type=Path, required=True)
    parser.add_argument("--episode-id", default="NLDO-P010")
    parser.add_argument("--generation-cap", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces: dict[str, dict[int, list[list[tuple[int, float]]]]] = {}
    for _, key, _, _ in METHODS:
        if args.source_manifest is not None and key in {"liveopt", "fixed_warm", "fixed_full"}:
            traces[key] = load_episode_traces_from_manifest(
                args.source_manifest,
                method_label={
                    "liveopt": "LiveOpt",
                    "fixed_warm": "Fixed Warm",
                    "fixed_full": "Fixed Full",
                }[key],
                episode_id=args.episode_id,
            )
        else:
            run_dir = getattr(args, f"{key}_run_dir")
            if run_dir is None:
                raise ValueError(f"--{key.replace('_', '-')}-run-dir is required without a manifest source")
            traces[key] = load_episode_traces(run_dir, args.episode_id)
        validate_traces(key, traces[key])
    args.main_output.parent.mkdir(parents=True, exist_ok=True)
    args.appendix_output.parent.mkdir(parents=True, exist_ok=True)
    render_panel_grid(traces, [11, 12], args.main_output, args.generation_cap, main=True)
    render_panel_grid(traces, list(range(1, 13)), args.appendix_output, args.generation_cap, main=False)
    summary = summarize(traces, args.generation_cap)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_tex.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    args.summary_tex.write_text(render_tex(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def replay_jsonl(run_dir: Path) -> Path:
    candidates = [run_dir / "NLDO" / "evo2_limit0.jsonl"]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"merged replay JSONL not found under {run_dir}")
    return path


def formal_metric_csv(run_dir: Path) -> Path:
    candidates = [
        run_dir / "reference_metrics" / "reference_stage_metrics.csv",
        run_dir / "liveopt" / "reference_metrics" / "reference_stage_metrics.csv",
    ]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"formal reference-stage metrics not found under {run_dir}")
    return path


def load_formal_stage_metrics(run_dir: Path, episode_id: str) -> dict[tuple[int, int], dict[str, str]]:
    with formal_metric_csv(run_dir).open(encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("episode_id") == episode_id
            and 1 <= int(row.get("stage_index") or 0) <= 12
        ]
    by_cell = {
        (int(row["stage_index"]), int(row["run_seed"])): row
        for row in rows
    }
    if len(rows) != 120 or len(by_cell) != 120:
        raise ValueError(
            f"{run_dir} {episode_id}: expected 120 formal stage-seed metrics, "
            f"got {len(rows)} rows/{len(by_cell)} cells"
        )
    return by_cell


def apply_formal_metric_gate(
    curve: list[tuple[int, float]],
    metric: dict[str, str],
    *,
    context: str,
) -> list[tuple[int, float]]:
    """Make plotted trajectories use the same hidden-feasibility gate as paper tables.

    The per-generation callback records a Workbench-internal HV trace.  A generic
    Workbench can consider an archive feasible even when the fixed hidden checker
    rejects its public solutions.  Such a stage scores zero in every formal table
    and must not appear as a successful trajectory in the paper figure.
    """

    true_pass = str(metric.get("true_pass") or "").strip().lower() in {"1", "true", "yes"}
    official_hv = float(metric.get("normalized_hv") or 0.0)
    if not true_pass:
        if abs(official_hv) > 1e-12:
            raise ValueError(f"{context}: hidden-infeasible formal metric has nonzero HV {official_hv}")
        return [(generation, 0.0) for generation, _ in curve]
    if not curve:
        raise ValueError(f"{context}: formally feasible stage has no trajectory")
    trace_final = float(curve[-1][1])
    if abs(trace_final - official_hv) > 1e-9:
        raise ValueError(
            f"{context}: trace/table mismatch, final trace HV={trace_final} "
            f"but formal HV={official_hv}"
        )
    return curve


def load_episode_traces(run_dir: Path, episode_id: str) -> dict[int, list[list[tuple[int, float]]]]:
    formal_metrics = load_formal_stage_metrics(run_dir, episode_id)
    by_stage_seed: dict[int, dict[int, list[tuple[int, float]]]] = {
        stage: {} for stage in range(1, 13)
    }
    with replay_jsonl(run_dir).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("episode_id")) != episode_id:
                continue
            seed = int(row.get("run_seed") or 0)
            updates = list(row.get("update_results") or [])
            for stage, update in enumerate(updates[:12], start=1):
                history = list((((update.get("solver_result") or {}).get("metadata") or {}).get("history") or []))
                curve: list[tuple[int, float]] = []
                for point in history:
                    if not isinstance(point, dict) or point.get("normalized_hv") is None:
                        continue
                    curve.append((int(point.get("generation") or 0), float(point["normalized_hv"])))
                metric = formal_metrics[(stage, seed)]
                by_stage_seed[stage][seed] = apply_formal_metric_gate(
                    curve,
                    metric,
                    context=f"{run_dir} {episode_id} seed {seed} t{stage:02d}",
                )
    return {
        stage: [seed_curves[seed] for seed in sorted(seed_curves)]
        for stage, seed_curves in by_stage_seed.items()
    }


def load_episode_traces_from_manifest(
    manifest_path: Path,
    *,
    method_label: str,
    episode_id: str,
) -> dict[int, list[list[tuple[int, float]]]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("method") == method_label and row.get("episode_id") == episode_id
        ]
    by_stage_source = {int(row["stage_index"]): Path(row["source_run_dir"]) for row in rows}
    if set(by_stage_source) != set(range(1, 13)):
        raise ValueError(
            f"{method_label} {episode_id}: canonical manifest must cover t01--t12; "
            f"got {sorted(by_stage_source)}"
        )
    cache: dict[Path, dict[int, list[list[tuple[int, float]]]]] = {}
    result: dict[int, list[list[tuple[int, float]]]] = {}
    for stage, run_dir in sorted(by_stage_source.items()):
        if run_dir not in cache:
            cache[run_dir] = load_episode_traces(run_dir, episode_id)
        result[stage] = cache[run_dir][stage]
    return result


def validate_traces(key: str, by_stage: dict[int, list[list[tuple[int, float]]]]) -> None:
    for stage in range(1, 13):
        curves = by_stage.get(stage) or []
        if len(curves) != 10:
            raise ValueError(f"{key} t{stage:02d}: expected 10 seed traces, got {len(curves)}")
        if any(not curve for curve in curves):
            raise ValueError(f"{key} t{stage:02d}: missing normalized-HV trace")


def aligned(curves: list[list[tuple[int, float]]], cap: int) -> tuple[list[int], list[float], list[float], float]:
    generations = list(range(cap + 1))
    carried: list[list[float]] = []
    stops: list[int] = []
    for curve in curves:
        mapping = {generation: value for generation, value in curve}
        last = mapping[min(mapping)]
        values: list[float] = []
        for generation in generations:
            if generation in mapping:
                last = mapping[generation]
            values.append(last)
        carried.append(values)
        stops.append(max(mapping))
    means = [fmean(seed[index] for seed in carried) for index in range(len(generations))]
    stds = [pstdev(seed[index] for seed in carried) for index in range(len(generations))]
    return generations, means, stds, fmean(stops)


def render_panel_grid(
    traces: dict[str, dict[int, list[list[tuple[int, float]]]]],
    stages: list[int],
    output: Path,
    cap: int,
    *,
    main: bool,
) -> None:
    if main:
        fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharex=True, sharey=True)
        axes_list = list(axes)
    else:
        fig, axes = plt.subplots(2, 6, figsize=(13.2, 4.45), sharex=True, sharey=True)
        axes_list = list(axes.flat)
    for ax, stage in zip(axes_list, stages):
        # Draw the Full control underneath LiveOpt.  They coincide whenever
        # the selector chooses Full, so a wider translucent line keeps that
        # equality visible instead of silently covering one curve.
        for label, key, color, linestyle in reversed(METHODS):
            x, mean, std, stop = aligned(traces[key][stage], cap)
            linewidth = (2.8 if main else 2.0) if key == "fixed_full" else (1.55 if main else 1.15)
            alpha = 0.48 if key == "fixed_full" else 1.0
            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label)
            ax.fill_between(x, [max(0.0, a - b) for a, b in zip(mean, std)], [min(1.0, a + b) for a, b in zip(mean, std)], color=color, alpha=0.10, linewidth=0)
            stop_index = min(cap, max(0, int(round(stop))))
            ax.plot([stop_index], [mean[stop_index]], marker="v", markersize=3.8, color=color, markeredgewidth=0)
        ax.set_title(f"t{stage:02d}", fontsize=9 if main else 8)
        ax.set_xlim(0, cap)
        # Leave a small unticked margin below zero so a formally failed
        # trajectory (score 0) remains visible instead of merging with the axis.
        ax.set_ylim(-0.025, 0.92)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(True, linewidth=0.35, alpha=0.28)
        ax.tick_params(labelsize=7)
    if main:
        axes_list[0].set_ylabel("Normalized HV", fontsize=8)
        for ax in axes_list:
            ax.set_xlabel("Generation", fontsize=8)
        legend_in_order(fig, axes_list[0], anchor=(0.5, 1.025), fontsize=7.0)
        fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.78, wspace=0.10)
    else:
        axes_list[0].set_ylabel("Normalized HV", fontsize=8)
        axes_list[6].set_ylabel("Normalized HV", fontsize=8)
        for ax in axes_list[6:]:
            ax.set_xlabel("Generation", fontsize=7)
        legend_in_order(fig, axes_list[0], anchor=(0.5, 1.015), fontsize=8)
        fig.subplots_adjust(left=0.05, right=0.995, bottom=0.10, top=0.88, wspace=0.10, hspace=0.24)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def legend_in_order(fig: plt.Figure, ax: plt.Axes, *, anchor: tuple[float, float], fontsize: float) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [label for label, _, _, _ in METHODS]
    fig.legend(
        [by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=4,
        frameon=False,
        fontsize=fontsize,
    )


def summarize(traces: dict[str, dict[int, list[list[tuple[int, float]]]]], cap: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stage in (11, 12):
        for label, key, _, _ in METHODS:
            _, mean, std, stop = aligned(traces[key][stage], cap)
            rows.append(
                {
                    "stage": stage,
                    "method": label,
                    "key": key,
                    "initial_hv": mean[0],
                    "final_hv": mean[-1],
                    "final_std": std[-1],
                    "mean_stop_generation": stop,
                }
            )
    return {"episode_id": "NLDO-P010", "rows": rows, "generation_cap": cap}


def render_tex(summary: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/analysis/export_liveopt_large_shift_case_study.py.",
        r"\newcommand{\LiveOptLargeShiftCaseRows}{%",
    ]
    for row in summary["rows"]:
        method = str(row["method"])
        if row["key"] == "liveopt":
            method = r"\method{}"
        lines.append(
            f"t{row['stage']:02d} & {method} & {row['initial_hv']:.3f} & "
            f"{row['final_hv']:.3f}$\\pm${row['final_std']:.3f} & "
            f"{row['mean_stop_generation']:.1f} " + r"\\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
