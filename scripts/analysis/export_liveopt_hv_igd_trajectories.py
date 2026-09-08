#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.evaluate_reference_metrics import (  # noqa: E402
    DEFAULT_NLDO_STRONG_REFERENCE,
    apply_hidden_delta,
    effective_benchmark,
    reference_step_for_stage,
    reference_trajectory_for_payload,
    score_solution,
)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(Path(args.episodes_jsonl))
    runs = parse_runs(args.run)
    unavailable_runs = parse_unavailable_runs(args.unavailable_run)
    rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for label, run_dir in runs:
        rows.extend(
            export_run_trajectories(
                label,
                run_dir,
                episodes,
                benchmark=args.benchmark,
                history_metric_source=args.history_metric_source,
                iteration_rows=iteration_rows,
            )
        )
    summary = summarize(rows)
    iteration_summary = summarize_iterations(iteration_rows)
    write_csv(out_dir / "trajectory_rows.csv", rows)
    write_csv(out_dir / "trajectory_summary.csv", summary)
    write_csv(out_dir / "iteration_rows.csv", iteration_rows)
    write_csv(out_dir / "iteration_summary.csv", iteration_summary)
    (out_dir / "trajectory_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "trajectory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "iteration_rows.json").write_text(json.dumps(iteration_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "iteration_summary.json").write_text(json.dumps(iteration_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plots = []
    for episode_id in args.episode_id:
        for metric in args.metric:
            plots.append(
                str(
                    plot_episode_2x6(
                        rows,
                        iteration_rows,
                        out_dir=out_dir,
                        episode_id=episode_id,
                        metric=metric,
                        title=args.title,
                        unavailable_runs=unavailable_runs,
                    )
                )
            )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary_rows": len(summary),
                "iteration_rows": len(iteration_rows),
                "iteration_summary_rows": len(iteration_summary),
                "metric_source_counts": dict(Counter(str(row.get("metric_source") or "") for row in rows)),
                "sparse_reference_rows": sum(
                    1
                    for row in rows
                    if row.get("reference_archive_size") not in (None, "")
                    and int(row.get("reference_archive_size") or 0) < 5
                    and row.get("normalized_hv") not in (None, "")
                ),
                "out_dir": str(out_dir),
                "plots": plots,
                "unavailable_runs": unavailable_runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and plot per-generation normalized-score/HV/IGD trajectories from traced LiveOpt replays."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run mapping label=run_dir. Repeat for ablations or main algorithms.",
    )
    parser.add_argument(
        "--unavailable-run",
        action="append",
        default=[],
        metavar="LABEL=REASON",
        help=(
            "Declare a comparison that cannot be plotted on a generation axis. "
            "The label is shown as 'not plotted' and the reason is printed below "
            "the figure; no synthetic curve is created. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--episodes-jsonl",
        default=DEFAULT_NLDO_STRONG_REFERENCE,
        help=(
            "Benchmark JSONL used only for offline trajectory scoring. Defaults "
            "to the merged 500x500x10 NLDO strong reference; replay scripts should "
            "still use the public 12-update benchmark file."
        ),
    )
    parser.add_argument("--benchmark", default="NLDO")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="Draw one 2x6 stage trajectory figure for this episode. Repeat for multiple episodes.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=["normalized_score", "normalized_hv", "igd"],
        default=[],
        help="Metric to plot. Repeat for both. Defaults to normalized_hv and igd.",
    )
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--history-metric-source",
        choices=["cached", "prefer-rescore", "rescore"],
        default="cached",
        help=(
            "How to obtain per-generation metrics. 'cached' uses metrics saved during replay. "
            "'prefer-rescore' recomputes when a per-generation archive snapshot is present. "
            "'rescore' requires per-generation archive snapshots and skips rows without them."
        ),
    )
    args = parser.parse_args()
    if not args.metric:
        args.metric = ["normalized_hv", "igd"]
    return args


def parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run must be label=run_dir, got: {item}")
        label, path = item.split("=", 1)
        runs.append((label.strip(), Path(path.strip())))
    return runs


def parse_unavailable_runs(items: list[str]) -> dict[str, str]:
    unavailable: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--unavailable-run must be LABEL=REASON, got: {item}")
        label, reason = item.split("=", 1)
        label = label.strip()
        reason = reason.strip()
        if not label or not reason:
            raise ValueError(f"--unavailable-run requires a non-empty label and reason, got: {item}")
        unavailable[label] = reason
    return unavailable


def load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"trajectory scoring benchmark not found: {path}. "
            "Use the 500x500x10 strong reference for paper HV/IGD curves."
        )
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["episode_id"])] = row
    return out


def export_run_trajectories(
    label: str,
    run_dir: Path,
    episodes: dict[str, dict[str, Any]],
    *,
    benchmark: str,
    history_metric_source: str = "cached",
    iteration_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    path = run_dir / benchmark / "evo2_limit0.jsonl"
    if not path.exists():
        path = run_dir / "NLDO" / "evo2_limit0.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"run JSONL not found for {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            run = json.loads(line)
            episode_id = str(run.get("episode_id") or run.get("base_instance_id"))
            episode = episodes.get(episode_id)
            if not episode:
                continue
            rows.extend(
                export_seed_run(
                    label,
                    str(run_dir),
                    run,
                    episode,
                    benchmark=benchmark,
                    history_metric_source=history_metric_source,
                )
            )
            if iteration_rows is not None:
                iteration_rows.extend(iteration_rows_from_run(label, run_dir, run, episode, benchmark=benchmark))
    return rows


def export_run_iterations(label: str, run_dir: Path, episodes: dict[str, dict[str, Any]], *, benchmark: str) -> list[dict[str, Any]]:
    path = run_dir / benchmark / "evo2_limit0.jsonl"
    if not path.exists():
        path = run_dir / "NLDO" / "evo2_limit0.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"run JSONL not found for {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            run = json.loads(line)
            episode_id = str(run.get("episode_id") or run.get("base_instance_id"))
            episode = episodes.get(episode_id)
            if not episode:
                continue
            rows.extend(iteration_rows_from_run(label, run_dir, run, episode, benchmark=benchmark))
    return rows


def iteration_rows_from_run(
    label: str,
    run_dir: Path,
    run: dict[str, Any],
    episode: dict[str, Any],
    *,
    benchmark: str,
) -> list[dict[str, Any]]:
    episode_id = str(run.get("episode_id") or run.get("base_instance_id"))
    rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(traced_stages(run)):
        runtime = stage.get("runtime") if isinstance(stage.get("runtime"), dict) else {}
        early_stop = runtime.get("early_stop") if isinstance(runtime.get("early_stop"), dict) else {}
        executed = runtime.get("executed_generations", runtime.get("generations"))
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "benchmark": benchmark,
                "episode_id": episode_id,
                "domain": episode.get("domain"),
                "family": episode.get("family"),
                "run_seed": run.get("run_seed"),
                "stage_index": stage_index,
                "update_id": stage.get("update_id") or ("initial" if stage_index == 0 else f"u{stage_index:03d}"),
                "restart_skill": stage.get("restart_skill") or "",
                "selection": runtime.get("selection") or "",
                "selection_mode": runtime.get("selection_mode") or "",
                "max_generations": runtime.get("generations"),
                "executed_generations": executed,
                "early_stopped": bool(runtime.get("early_stopped", early_stop.get("stopped"))),
                "early_stop_reason": early_stop.get("stop_reason") or "",
                "archive_size": len(stage.get("candidate_archive") or []),
            }
        )
    return rows


def export_seed_run(
    label: str,
    run_dir: str,
    run: dict[str, Any],
    episode: dict[str, Any],
    *,
    benchmark: str,
    history_metric_source: str = "cached",
) -> list[dict[str, Any]]:
    scorer_benchmark = effective_benchmark(benchmark, episode)
    domain = str(episode.get("domain") or "")
    family = str(episode.get("family") or "")
    state = json.loads(json.dumps(episode.get("hidden_initial_state") or {}))
    previous_solution: dict[str, Any] | None = None
    reference_steps = reference_trajectory_for_payload(scorer_benchmark, domain, episode)
    oracle = episode.get("hidden_update_oracle") or []
    out: list[dict[str, Any]] = []
    stages = traced_stages(run)
    for stage_index, stage in enumerate(stages):
        if stage_index > 0 and stage_index - 1 < len(oracle):
            state = apply_hidden_delta(scorer_benchmark, domain, state, oracle[stage_index - 1].get("hidden_delta") or {})
        update_id = str(stage.get("update_id") or ("initial" if stage_index == 0 else f"u{stage_index:03d}"))
        ref_step = reference_step_for_stage(reference_steps, update_id, stage_index)
        reference_archive_size = len(ref_step.get("pareto_archive") or []) if isinstance(ref_step, dict) else 0
        reference_update_id = ref_step.get("update_id") if isinstance(ref_step, dict) else ""
        final_metric = score_solution(
            scorer_benchmark,
            domain,
            family,
            state,
            stage.get("solution") or {},
            previous_solution,
            ref_step,
            stage.get("candidate_archive") or [],
        )
        for history in stage.get("history") or []:
            archive = history.get("candidate_archive") or []
            cached_metric = metric_from_history(history)
            metric = None
            metric_source = ""
            should_rescore = history_metric_source in {"prefer-rescore", "rescore"} and bool(archive)
            if should_rescore:
                generation_solution = first_archive_solution(archive) or stage.get("solution") or {}
                metric = score_solution(
                    scorer_benchmark,
                    domain,
                    family,
                    state,
                    generation_solution,
                    previous_solution,
                    ref_step,
                    archive,
                )
                metric_source = "rescored_history_archive"
            elif history_metric_source == "rescore":
                metric_source = "missing_history_archive"
            else:
                metric = cached_metric
                metric_source = "cached_history" if metric is not None else ""
                if metric is None and archive:
                    generation_solution = first_archive_solution(archive) or stage.get("solution") or {}
                    metric = score_solution(
                        scorer_benchmark,
                        domain,
                        family,
                        state,
                        generation_solution,
                        previous_solution,
                        ref_step,
                        archive,
                    )
                    metric_source = "rescored_history_archive"
            if not metric or (
                metric.get("normalized_score") is None
                and metric.get("normalized_hv") is None
                and metric.get("igd") is None
            ):
                continue
            out.append(
                {
                    "label": label,
                    "run_dir": run_dir,
                    "benchmark": benchmark,
                    "episode_id": episode.get("episode_id"),
                    "run_seed": run.get("run_seed"),
                    "stage_index": stage_index,
                    "update_id": update_id,
                    "reference_update_id": reference_update_id,
                    "reference_archive_size": reference_archive_size,
                    "generation": history.get("generation"),
                    "normalized_score": metric.get("normalized_score"),
                    "normalized_hv": metric.get("normalized_hv"),
                    "hv": metric.get("hv"),
                    "reference_hv": metric.get("reference_hv"),
                    "igd": metric.get("igd"),
                    "igd_score": metric.get("igd_score"),
                    "ideal_gap": metric.get("ideal_gap"),
                    "true_pass": bool(metric.get("feasible")),
                    "archive_size": history.get("archive_size") or len(archive),
                    "history_archive_size": len(archive),
                    "metric_source": metric_source,
                    "restart_skill": stage.get("restart_skill") or "",
                }
            )
        previous_solution = final_metric.get("canonical_solution") or stage.get("solution") or previous_solution
    return out


def traced_stages(run: dict[str, Any]) -> list[dict[str, Any]]:
    initial_solver = run.get("initial_solver_result") if isinstance(run.get("initial_solver_result"), dict) else {}
    initial_metadata = initial_solver.get("metadata") if isinstance(initial_solver.get("metadata"), dict) else {}
    stages = [
        {
            "update_id": "initial",
            "solution": run.get("initial_candidate") or {},
            "candidate_archive": initial_metadata.get("candidate_archive") or [],
            "history": initial_metadata.get("history") or [],
            "runtime": initial_metadata.get("runtime") if isinstance(initial_metadata.get("runtime"), dict) else {},
            "restart_skill": "",
        }
    ]
    for update in run.get("update_results", []) or []:
        solver = update.get("solver_result") if isinstance(update.get("solver_result"), dict) else {}
        metadata = solver.get("metadata") if isinstance(solver.get("metadata"), dict) else {}
        restart = update.get("restart") if isinstance(update.get("restart"), dict) else {}
        if not restart:
            restart = metadata.get("restart") if isinstance(metadata.get("restart"), dict) else {}
        stages.append(
            {
                "update_id": update.get("update_id"),
                "solution": update.get("solution") or solver.get("solution") or {},
                "candidate_archive": metadata.get("candidate_archive") or [],
                "history": metadata.get("history") or [],
                "runtime": metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {},
                "restart_skill": restart.get("restart_skill") or (update.get("impact") or {}).get("restart_skill") or "",
            }
        )
    return stages


def first_archive_solution(archive: list[dict[str, Any]]) -> dict[str, Any]:
    for item in archive:
        if isinstance(item, dict) and isinstance(item.get("solution"), dict) and item["solution"]:
            return item["solution"]
    return {}


def metric_from_history(history: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(history, dict):
        return None
    metric_keys = {"hv", "reference_hv", "normalized_hv", "igd", "igd_score", "ideal_gap", "normalized_score"}
    if not any(key in history for key in metric_keys):
        return None
    return {
        "feasible": history.get("true_pass", history.get("feasible", False)),
        "hv": history.get("hv"),
        "reference_hv": history.get("reference_hv"),
        "normalized_hv": history.get("normalized_hv"),
        "igd": history.get("igd"),
        "igd_score": history.get("igd_score"),
        "ideal_gap": history.get("ideal_gap"),
        "normalized_score": history.get("normalized_score"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["benchmark"], row["episode_id"], int(row["stage_index"]), row["update_id"], int(row["generation"]))].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (str(kv[0][2]), kv[0][3], kv[0][5], str(kv[0][0]))):
        label, benchmark, episode_id, stage_index, update_id, generation = key
        hv = finite_values(item.get("normalized_hv") for item in items)
        igd = finite_values(item.get("igd") for item in items)
        score = finite_values(item.get("normalized_score") for item in items)
        reference_sizes = finite_values(item.get("reference_archive_size") for item in items)
        history_archive_sizes = finite_values(item.get("history_archive_size") for item in items)
        metric_sources = sorted({str(item.get("metric_source") or "") for item in items if item.get("metric_source")})
        out.append(
            {
                "label": label,
                "benchmark": benchmark,
                "episode_id": episode_id,
                "stage_index": stage_index,
                "update_id": update_id,
                "generation": generation,
                "seed_count": len(items),
                "normalized_score_mean": mean(score) if score else None,
                "normalized_score_std": pstdev(score) if len(score) > 1 else 0.0 if score else None,
                "normalized_hv_mean": mean(hv) if hv else None,
                "normalized_hv_std": pstdev(hv) if len(hv) > 1 else 0.0 if hv else None,
                "igd_mean": mean(igd) if igd else None,
                "igd_std": pstdev(igd) if len(igd) > 1 else 0.0 if igd else None,
                "true_pass_ratio": sum(1 for item in items if item.get("true_pass")) / len(items) if items else None,
                "archive_size_mean": mean(finite_values(item.get("archive_size") for item in items)) if items else None,
                "history_archive_size_mean": mean(history_archive_sizes) if history_archive_sizes else None,
                "reference_archive_size_mean": mean(reference_sizes) if reference_sizes else None,
                "metric_sources": "+".join(metric_sources),
            }
        )
    return out


def summarize_iterations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["benchmark"], row["episode_id"], int(row["stage_index"]), row["update_id"])].append(row)
    out = []
    for key, items in sorted(grouped.items(), key=lambda kv: (str(kv[0][2]), kv[0][3], str(kv[0][0]))):
        label, benchmark, episode_id, stage_index, update_id = key
        generations = finite_values(item.get("executed_generations") for item in items)
        max_generations = finite_values(item.get("max_generations") for item in items)
        out.append(
            {
                "label": label,
                "benchmark": benchmark,
                "episode_id": episode_id,
                "stage_index": stage_index,
                "update_id": update_id,
                "seed_count": len(items),
                "mean_executed_generations": round(mean(generations), 6) if generations else None,
                "std_executed_generations": round(pstdev(generations), 6) if len(generations) > 1 else 0.0 if generations else None,
                "min_executed_generations": min(generations) if generations else None,
                "max_executed_generations": max(generations) if generations else None,
                "mean_max_generations": round(mean(max_generations), 6) if max_generations else None,
                "early_stop_rate": round(sum(1 for item in items if item.get("early_stopped")) / len(items), 6) if items else 0.0,
            }
        )
    return out


def finite_values(values: Any) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_episode_2x6(
    rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    episode_id: str,
    metric: str,
    title: str = "",
    unavailable_runs: dict[str, str] | None = None,
) -> Path:
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    import numpy as np

    unavailable_runs = unavailable_runs or {}
    value_key = metric
    episode_rows = [row for row in rows if str(row.get("episode_id")) == str(episode_id) and int(row.get("stage_index", -1)) >= 1]
    if not episode_rows:
        raise ValueError(f"no trajectory rows found for episode {episode_id}")
    episode_iterations = [
        row
        for row in iteration_rows
        if str(row.get("episode_id")) == str(episode_id) and int(row.get("stage_index", -1)) >= 1
    ]
    iteration_lookup = {
        (str(row.get("label")), int(row.get("stage_index", -1)), str(row.get("run_seed"))): row
        for row in episode_iterations
    }
    preferred_order = [
        "LiveOpt",
        "Fixed Full",
        "Fixed Warm",
        "Fixed warm",
        "w/o LSM",
        "w/o TSS updates",
        "Typed full regeneration",
        "Full restart",
        "Population transfer",
        "Warm restart",
    ]
    observed_labels = {str(row["label"]) for row in episode_rows}
    labels = [label for label in preferred_order if label in observed_labels]
    labels.extend(sorted(observed_labels - set(labels)))
    colors = {
        "Full restart": "#1f77b4",
        "LiveOpt": "#0B6E69",
        "Fixed Full": "#333333",
        "Fixed Warm": "#D9822B",
        "Fixed warm": "#1f77b4",
        "w/o LSM": "#2ca02c",
        "w/o TSS updates": "#9467bd",
        "Typed full regeneration": "#8c564b",
        "Population transfer": "#2ca02c",
        "Warm restart": "#d62728",
    }
    fallback_cmap = plt.get_cmap("tab10")
    for idx, label in enumerate(labels):
        colors.setdefault(label, fallback_cmap(idx % 10))
    line_widths = {
        "Full restart": 2.45,
        "LiveOpt": 2.2,
        "Fixed Full": 1.85,
        "Fixed Warm": 1.75,
        "Fixed warm": 1.65,
        "w/o LSM": 1.65,
        "w/o TSS updates": 1.65,
        "Typed full regeneration": 1.65,
        "Population transfer": 1.65,
        "Warm restart": 1.65,
    }
    line_styles = {
        "LiveOpt": "-",
        "Fixed Full": "--",
        "Fixed Warm": "-.",
    }
    generation_candidates = finite_values(row.get("generation") for row in episode_rows)
    budget_candidates = finite_values(row.get("max_generations") for row in episode_iterations)
    global_end_generation = int(max(generation_candidates + budget_candidates))
    stage_indices = sorted({int(row["stage_index"]) for row in episode_rows})
    ncols = min(6, len(stage_indices))
    nrows = int(math.ceil(len(stage_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.37 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = list(axes.ravel())
    saw_early_stop = False
    for panel_index, stage_index in enumerate(stage_indices):
        ax = axes_flat[panel_index]
        stage_rows = [row for row in episode_rows if int(row["stage_index"]) == stage_index]
        for label in labels:
            label_rows = [row for row in stage_rows if str(row["label"]) == label and row.get(value_key) not in (None, "")]
            if not label_rows:
                continue
            by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in label_rows:
                by_seed[str(row.get("run_seed"))].append(row)
            seed_series: list[np.ndarray] = []
            early_stop_generations: list[int] = []
            prepared: list[dict[int, float]] = []
            for seed, seed_rows in by_seed.items():
                values_by_generation: dict[int, float] = {}
                for item in seed_rows:
                    try:
                        gen = int(item["generation"])
                        value = float(item[value_key])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        values_by_generation[gen] = value
                if not values_by_generation:
                    continue
                runtime = iteration_lookup.get((label, stage_index, seed), {})
                if runtime.get("early_stopped"):
                    executed = finite_values([runtime.get("executed_generations")])
                    early_stop_generations.append(int(executed[0]) if executed else max(values_by_generation))
                prepared.append(values_by_generation)
            if not prepared:
                continue
            x = np.arange(global_end_generation + 1, dtype=float)
            for values_by_generation in prepared:
                first_generation = min(values_by_generation)
                last_value = float("nan")
                series = np.full(global_end_generation + 1, np.nan, dtype=float)
                for gen in range(first_generation, global_end_generation + 1):
                    if gen in values_by_generation:
                        last_value = values_by_generation[gen]
                    series[gen] = last_value
                seed_series.append(series)
            arr = np.vstack(seed_series)
            counts = np.sum(np.isfinite(arr), axis=0)
            y = np.divide(np.nansum(arr, axis=0), counts, out=np.full(arr.shape[1], np.nan), where=counts > 0)
            centered = np.where(np.isfinite(arr), arr - y, 0.0)
            variance = np.divide(
                np.sum(centered * centered, axis=0),
                counts,
                out=np.zeros(arr.shape[1], dtype=float),
                where=counts > 0,
            )
            std = np.sqrt(variance)
            valid = np.isfinite(y)
            ax.plot(
                x[valid],
                y[valid],
                linewidth=line_widths.get(label, 1.55),
                linestyle=line_styles.get(label, "-"),
                color=colors[label],
                label=label,
            )
            lower = y - std
            if metric in {"normalized_score", "normalized_hv"}:
                lower = np.maximum(0.0, lower)
            ax.fill_between(x[valid], lower[valid], (y + std)[valid], color=colors[label], alpha=0.16, linewidth=0)
            if early_stop_generations:
                saw_early_stop = True
                mean_stop = float(np.mean(early_stop_generations))
                mean_stop_index = int(round(max(0.0, min(mean_stop, float(x[-1])))))
                ax.axvline(mean_stop, color=colors[label], linestyle="--", linewidth=0.8, alpha=0.58)
                if math.isfinite(float(y[mean_stop_index])):
                    ax.scatter(
                        [mean_stop],
                        [y[mean_stop_index]],
                        marker="v",
                        s=25,
                        color=colors[label],
                        edgecolors="white",
                        linewidths=0.65,
                        zorder=6,
                    )
                ax.text(
                    mean_stop,
                    0.98,
                    rf"$\bar{{g}}$={mean_stop:.0f}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    color=colors[label],
                    fontsize=6.3,
                )
        ax.set_title(f"t{stage_index:02d}", fontsize=10, pad=3)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8, length=2)
        if panel_index % ncols == 0:
            ylabel = {
                "normalized_score": "Normalized score",
                "normalized_hv": "Normalized HV",
                "igd": "IGD",
            }[metric]
            ax.set_ylabel(ylabel, fontsize=9)
        if panel_index // ncols == nrows - 1:
            ax.set_xlabel("Generation", fontsize=9)
    for ax in axes_flat[len(stage_indices) :]:
        ax.set_visible(False)
    handles_by_label: dict[str, Any] = {}
    for ax in axes_flat:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            handles_by_label.setdefault(label, handle)
    handles = [handles_by_label[label] for label in labels if label in handles_by_label]
    labels_for_legend = [label for label in labels if label in handles_by_label]
    for label in unavailable_runs:
        handles.append(Line2D([], [], color="#777777", linestyle=":", marker="x", markersize=5))
        labels_for_legend.append(f"{label} (not plotted)")
    if saw_early_stop:
        handles.append(Line2D([], [], color="#555555", linestyle="--", marker="v", markersize=4))
        labels_for_legend.append("mean early-stop generation")
    if handles:
        fig.legend(handles, labels_for_legend, loc="upper center", ncol=min(5, len(handles)), frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, 1.02))
    metric_title = {
        "normalized_score": "normalized-score",
        "normalized_hv": "HV",
        "igd": "IGD",
    }[metric]
    fig.suptitle(title or f"{episode_id} Per-generation {metric_title} trajectories", fontsize=12, y=1.06)
    bottom = 0.0
    if unavailable_runs:
        unavailable_note = "Not plotted: " + "; ".join(f"{label}—{reason}" for label, reason in unavailable_runs.items())
        fig.text(0.5, 0.008, unavailable_note, ha="center", va="bottom", fontsize=7.6, color="#555555", wrap=True)
        bottom = 0.075
    fig.tight_layout(rect=(0, bottom, 1, 0.98), w_pad=0.5, h_pad=0.7)
    safe_metric = "hv" if metric == "normalized_hv" else metric
    layout_tag = f"{nrows}x{ncols}"
    path = out_dir / f"{episode_id.lower()}_{safe_metric}_trajectory_{layout_tag}.pdf"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
