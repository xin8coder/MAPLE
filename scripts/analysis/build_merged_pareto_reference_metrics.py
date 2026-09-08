#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.moea_metrics import nondominated
from scripts.analysis import evaluate_reference_metrics as ref


MO_BENCHMARKS = {"green_vrp_mo", "cloud_scheduling_mo"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-facing metrics with 10-seed merged Pareto fronts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default=f"NLDO={ref.DEFAULT_NLDO_STRONG_REFERENCE}")
    args = parser.parse_args()

    benchmark_name, benchmark_path = args.benchmark.split("=", 1)
    episodes = ref.load_benchmark(Path(benchmark_path))
    raw_rows = ref.evaluate_run_dir(args.run_dir, {benchmark_name: Path(benchmark_path)})
    run_rows = load_runs(args.run_dir / benchmark_name / "evo2_limit0.jsonl")

    rows: list[dict[str, Any]] = []
    for episode_id, episode in sorted(episodes.items()):
        scorer = ref.effective_benchmark(benchmark_name, episode)
        episode_raw_rows = [row for row in raw_rows if row.get("episode_id") == episode_id and row.get("mode") == "evo2"]
        if scorer in MO_BENCHMARKS:
            rows.extend(merged_mo_rows(benchmark_name, scorer, episode, run_rows.get(episode_id, []), episode_raw_rows))
        else:
            rows.extend(aggregate_non_mo_rows(episode_raw_rows))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = ref.aggregate(rows)
    totals = ref.total_scores(rows)
    for row in summary + totals:
        row["seed_run_count"] = 10
    ref.write_csv(args.out_dir / "reference_stage_metrics.csv", rows)
    ref.write_csv(args.out_dir / "reference_stage_summary.csv", summary)
    ref.write_csv(args.out_dir / "reference_total_scores.csv", totals)
    (args.out_dir / "reference_stage_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "reference_stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "reference_total_scores.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "total_rows": len(totals), "out_dir": str(args.out_dir)}, indent=2))


def load_runs(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[str(row.get("episode_id"))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item.get("run_seed") or 0))
    return grouped


def aggregate_non_mo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stage_index"])].append(row)
    out = []
    for stage_index, group in sorted(grouped.items()):
        first = group[0]
        out.append(
            {
                "benchmark": first.get("benchmark"),
                "scorer_benchmark": first.get("scorer_benchmark"),
                "mode": first.get("mode"),
                "episode_id": first.get("episode_id"),
                "stage_index": stage_index,
                "stage_type": first.get("stage_type"),
                "update_id": first.get("update_id"),
                "feasible": avg_bool(row.get("feasible") for row in group),
                "normalized_score": avg(row.get("normalized_score") for row in group),
                "normalized_total_score": avg(row.get("normalized_total_score") for row in group),
                "objective": avg(row.get("objective") for row in group),
                "reference_objective": avg(row.get("reference_objective") for row in group),
                "objective_gap": avg(row.get("objective_gap") for row in group),
                "hv": None,
                "reference_hv": None,
                "normalized_hv": None,
                "igd": None,
                "igd_score": None,
                "tokens": avg(row.get("tokens") for row in group),
                "latency_seconds": avg(row.get("latency_seconds") for row in group),
                "agent_reported_feasible": avg_bool(row.get("agent_reported_feasible") for row in group),
                "agent_reported_objective": avg(row.get("agent_reported_objective") for row in group),
                "true_pass": None if stage_index == 0 else avg_bool(row.get("true_pass") for row in group),
                "missing_stage": avg_bool(row.get("missing_stage") for row in group),
                "run_seed": "merged10",
            }
        )
    return out


def merged_mo_rows(
    benchmark_name: str,
    scorer: str,
    episode: dict[str, Any],
    runs: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not runs:
        return []
    objective_names = ref.GREEN_OBJECTIVES if scorer == "green_vrp_mo" else ref.CLOUD_OBJECTIVES
    module_name = "scripts.materialize_green_vrp_mo" if scorer == "green_vrp_mo" else "scripts.materialize_cloud_scheduling_mo"
    module = importlib.import_module(module_name)
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    reference_steps = ref.reference_trajectory_for_payload(scorer, str(episode.get("domain") or ""), episode)
    oracle = episode.get("hidden_update_oracle") or []
    stages_by_seed = [ref.extract_stages(run) for run in runs]
    by_stage_raw = group_raw_rows(raw_rows)
    out: list[dict[str, Any]] = []
    for stage_index in range(0, 1 + len(episode.get("update_stream") or [])):
        if stage_index > 0 and stage_index - 1 < len(oracle):
            state = ref.apply_hidden_delta(scorer, str(episode.get("domain") or ""), state, oracle[stage_index - 1].get("hidden_delta") or {})
        update_id = "initial" if stage_index == 0 else (episode["update_stream"][stage_index - 1].get("update_id") or f"u{stage_index:03d}")
        reference_step = ref.reference_step_for_stage(reference_steps, str(update_id), stage_index)
        has_saved_stage = any(stage_index < len(stages) for stages in stages_by_seed)
        merged_archive = hidden_merged_archive(module, scorer, state, stages_by_seed, stage_index, objective_names)
        canonical = merged_archive[0]["solution"] if merged_archive else {}
        metric = ref.mo_metric(canonical, merged_archive, objective_names, reference_step)
        # The paper-facing score is a bounded quality score. Keep the raw HV
        # ratio in normalized_hv, but avoid letting an approximate reference
        # archive make the aggregate score exceed one.
        bounded_score = min(float(metric["normalized_score"] or 0.0), 1.0)
        raw_group = by_stage_raw.get(stage_index, [])
        out.append(
            {
                "benchmark": benchmark_name,
                "scorer_benchmark": scorer,
                "mode": "evo2",
                "episode_id": episode.get("episode_id"),
                "stage_index": stage_index,
                "stage_type": "initial" if stage_index == 0 else "update",
                "update_id": update_id,
                "feasible": metric["feasible"],
                "normalized_score": round(bounded_score, 6),
                "normalized_total_score": round(bounded_score, 6),
                "objective": metric.get("objective"),
                "reference_objective": metric.get("reference_objective"),
                "objective_gap": metric.get("objective_gap"),
                "hv": metric.get("hv"),
                "reference_hv": metric.get("reference_hv"),
                "normalized_hv": metric.get("normalized_hv"),
                "igd": metric.get("igd"),
                "igd_score": metric.get("igd_score"),
                "ideal_gap": metric.get("ideal_gap"),
                "tokens": avg(row.get("tokens") for row in raw_group),
                "latency_seconds": avg(row.get("latency_seconds") for row in raw_group),
                "agent_reported_feasible": avg_bool(row.get("agent_reported_feasible") for row in raw_group),
                "agent_reported_objective": avg(row.get("agent_reported_objective") for row in raw_group),
                "true_pass": None if stage_index == 0 else bool(metric["feasible"]),
                "missing_stage": not has_saved_stage,
                "run_seed": "merged10",
                "hidden_candidate_archive_size": len(merged_archive),
                "hidden_candidate_archive": merged_archive,
            }
        )
    return out


def group_raw_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stage_index"])].append(row)
    return grouped


def hidden_merged_archive(
    module: Any,
    scorer: str,
    state: dict[str, Any],
    stages_by_seed: list[list[dict[str, Any]]],
    stage_index: int,
    objective_names: list[str],
) -> list[dict[str, Any]]:
    raw_archive: list[dict[str, Any]] = []
    final_solution: dict[str, Any] = {}
    for stages in stages_by_seed:
        if stage_index >= len(stages):
            continue
        stage = stages[stage_index]
        if isinstance(stage.get("solution"), dict) and not final_solution:
            final_solution = stage["solution"]
        raw_archive.extend(stage.get("candidate_archive") or [])
    if scorer == "green_vrp_mo":
        hidden = ref.hidden_green_archive(module, state, final_solution, None, raw_archive)
    else:
        hidden = ref.hidden_cloud_archive(module, state, final_solution, None, raw_archive)
    unique = {}
    for item in hidden:
        try:
            key = tuple(round(float(item["objectives"][name]), 9) for name in objective_names)
        except Exception:
            continue
        unique.setdefault(key, item)
    return nondominated(list(unique.values()), objective_names)


def avg(values: Any) -> float | None:
    clean = []
    for value in values:
        if value in (None, "", "None"):
            continue
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            clean.append(f)
    return round(mean(clean), 6) if clean else None


def avg_bool(values: Any) -> bool | None:
    clean = []
    for value in values:
        if value in (None, "", "None"):
            continue
        if isinstance(value, str):
            clean.append(value.lower() == "true")
        else:
            clean.append(bool(value))
    if not clean:
        return None
    return sum(1 for item in clean if item) >= len(clean) / 2


if __name__ == "__main__":
    main()
