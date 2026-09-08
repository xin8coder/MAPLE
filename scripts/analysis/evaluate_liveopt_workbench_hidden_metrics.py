#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis import evaluate_reference_metrics as ref_metrics


DEFAULT_EPISODES = ref_metrics.DEFAULT_NLDO_STRONG_REFERENCE


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(Path(args.episodes_jsonl))

    all_stage_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for summary_path in [Path(item) for item in args.scaffold_summary]:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_mode = str(summary.get("mode") or "unknown")
        for row_index, row in enumerate(summary.get("rows") or []):
            episode_id = str(row.get("episode_id") or "")
            run_id = f"{summary_path.parent.name}:{row_index}:{episode_id}"
            if episode_id not in episodes:
                failure_rows.append(
                    failure_record(summary_path, row_index, row, run_id, f"episode_not_found:{episode_id}")
                )
                continue
            if row.get("status") != "completed":
                failure_rows.append(failure_record(summary_path, row_index, row, run_id, row.get("failure_reason") or "not_completed"))
                continue
            stage = scaffold_stage(row)
            stage_rows = ref_metrics.evaluate_stage_sequence(
                "NLDO",
                f"liveopt_workbench_v1/{source_mode}",
                [stage],
                episodes[episode_id],
            )
            hidden_metric = hidden_metric_for_stage(episodes[episode_id], stage)
            for metric_row in stage_rows:
                metric_row.update(
                    {
                        "run_id": run_id,
                        "source_summary": str(summary_path),
                        "workbench_protocol": summary.get("protocol"),
                        "workbench_status": row.get("status"),
                        "scaffold_archive_count": row.get("archive_count"),
                        "scaffold_population_count": row.get("population_count"),
                    }
                )
                if not metric_row.get("missing_stage") and int(metric_row.get("stage_index") or 0) == 0:
                    metric_row.update(hidden_metric_summary(hidden_metric))
            all_stage_rows.extend(stage_rows)
            actual_rows.extend([item for item in stage_rows if not item.get("missing_stage")])
            replay_rows.append(scaffold_replay_row(row, stage, summary_path, run_id))

    summary_rows = ref_metrics.aggregate(all_stage_rows)
    total_rows = ref_metrics.total_scores(all_stage_rows)
    write_csv(out_dir / "official_stage_metrics.csv", all_stage_rows)
    write_csv(out_dir / "actual_generated_stage_metrics.csv", actual_rows)
    write_csv(out_dir / "official_stage_summary.csv", summary_rows)
    write_csv(out_dir / "official_total_scores.csv", total_rows)
    write_json(out_dir / "official_stage_metrics.json", all_stage_rows)
    write_json(out_dir / "actual_generated_stage_metrics.json", actual_rows)
    write_json(out_dir / "official_stage_summary.json", summary_rows)
    write_json(out_dir / "official_total_scores.json", total_rows)
    write_jsonl(out_dir / "scaffold_replay.jsonl", replay_rows)
    write_jsonl(out_dir / "failed_or_unscored_rows.jsonl", failure_rows)
    console = {
        "out_dir": str(out_dir),
        "scored_generated_rows": len(actual_rows),
        "official_stage_rows": len(all_stage_rows),
        "failed_or_unscored_rows": len(failure_rows),
        "actual_generated_stage_metrics": str(out_dir / "actual_generated_stage_metrics.csv"),
        "official_stage_metrics": str(out_dir / "official_stage_metrics.csv"),
        "scaffold_replay": str(out_dir / "scaffold_replay.jsonl"),
        "actual_rows": compact_console_rows(actual_rows),
    }
    print(json.dumps(console, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LiveOpt Workbench runner outputs with the hidden reference evaluator.")
    parser.add_argument("--episodes-jsonl", default=DEFAULT_EPISODES)
    parser.add_argument("--scaffold-summary", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["episode_id"])] = row
    return rows


def scaffold_stage(row: dict[str, Any]) -> dict[str, Any]:
    fitness = row.get("agent_objective_raw") if isinstance(row.get("agent_objective_raw"), dict) else {}
    solution = fitness.get("solution") if isinstance(fitness.get("solution"), dict) else best_archive_solution(row)
    return {
        "update_id": str(row.get("update_id") or "initial"),
        "solution": solution if isinstance(solution, dict) else {},
        "candidate_archive": scaffold_candidate_archive(row),
        "agent_feasible": fitness.get("feasible"),
        "agent_objective": fitness.get("scalar"),
        "stage_tokens": int((row.get("token_usage") or {}).get("total_tokens", 0) or 0),
        "latency_seconds": row.get("latency_seconds"),
    }


def best_archive_solution(row: dict[str, Any]) -> dict[str, Any]:
    for item in row.get("archive") or []:
        fitness = item.get("fitness") if isinstance(item, dict) else None
        if isinstance(fitness, dict) and isinstance(fitness.get("solution"), dict):
            return fitness["solution"]
    return {}


def scaffold_candidate_archive(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in row.get("archive") or []:
        if not isinstance(item, dict):
            continue
        fitness = item.get("fitness") if isinstance(item.get("fitness"), dict) else {}
        solution = fitness.get("solution")
        if not isinstance(solution, dict):
            continue
        out.append(
            {
                "solution": solution,
                "workbench_objectives": fitness.get("objectives"),
                "workbench_scalar": fitness.get("scalar"),
                "workbench_feasible": fitness.get("feasible"),
                "rank": item.get("rank"),
            }
        )
    return out


def hidden_metric_for_stage(episode: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    scorer_benchmark = ref_metrics.effective_benchmark("NLDO", episode)
    domain = str(episode.get("domain") or "")
    family = str(episode.get("family") or "")
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    reference_steps = ref_metrics.reference_trajectory_for_payload(scorer_benchmark, domain, episode)
    reference_step = ref_metrics.reference_step_for_stage(reference_steps, stage.get("update_id"), 0)
    return ref_metrics.score_solution(
        scorer_benchmark,
        domain,
        family,
        state,
        stage.get("solution") or {},
        None,
        reference_step,
        stage.get("candidate_archive") or [],
    )


def hidden_metric_summary(metric: dict[str, Any]) -> dict[str, Any]:
    archive = metric.get("hidden_candidate_archive") if isinstance(metric.get("hidden_candidate_archive"), list) else []
    return {
        "hidden_candidate_archive_size": metric.get("hidden_candidate_archive_size"),
        "hidden_archive_objectives": [item.get("objectives") for item in archive[:20] if isinstance(item, dict)],
    }


def scaffold_replay_row(row: dict[str, Any], stage: dict[str, Any], summary_path: Path, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "episode_id": row.get("episode_id"),
        "agent_mode": "liveopt_workbench_v1",
        "source_summary": str(summary_path),
        "initial_candidate": stage["solution"],
        "initial_feasible": stage.get("agent_feasible"),
        "initial_objective": stage.get("agent_objective"),
        "initial_solver_result": {"metadata": {"candidate_archive": stage["candidate_archive"]}},
        "update_results": [],
        "token_usage": row.get("token_usage") or {},
        "latency_seconds": row.get("latency_seconds"),
        "segments": row.get("segments") or [],
    }


def failure_record(summary_path: Path, row_index: int, row: dict[str, Any], run_id: str, reason: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source_summary": str(summary_path),
        "row_index": row_index,
        "episode_id": row.get("episode_id"),
        "status": row.get("status"),
        "failure_reason": str(reason),
    }


def compact_console_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "episode_id": row.get("episode_id"),
                "feasible": row.get("feasible"),
                "normalized_score": row.get("normalized_score"),
                "normalized_hv": row.get("normalized_hv"),
                "igd": row.get("igd"),
                "objective": row.get("objective"),
                "hidden_candidate_archive_size": (row.get("metric") or {}).get("hidden_candidate_archive_size")
                if isinstance(row.get("metric"), dict)
                else row.get("hidden_candidate_archive_size"),
                "tokens": row.get("tokens"),
                "latency_seconds": row.get("latency_seconds"),
            }
        )
    return out


def write_json(path: Path, rows: Any) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "run_id",
        "source_summary",
        "benchmark",
        "scorer_benchmark",
        "mode",
        "episode_id",
        "stage_index",
        "stage_type",
        "update_id",
        "missing_stage",
        "feasible",
        "true_pass",
        "normalized_score",
        "normalized_total_score",
        "objective",
        "reference_objective",
        "objective_gap",
        "hv",
        "reference_hv",
        "normalized_hv",
        "igd",
        "igd_score",
        "tokens",
        "latency_seconds",
        "agent_reported_feasible",
        "agent_reported_objective",
        "scaffold_archive_count",
        "scaffold_population_count",
        "hidden_candidate_archive_size",
        "hidden_archive_objectives",
    ]
    fieldnames = preferred + sorted({key for row in rows for key in row if key not in preferred and key != "metric"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


if __name__ == "__main__":
    main()
