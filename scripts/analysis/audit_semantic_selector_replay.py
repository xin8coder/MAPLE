#!/usr/bin/env python3
"""Audit a frozen semantic-selector replay against its shared prefix and Full control."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-run", type=Path, required=True)
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--full-control-run", type=Path, required=True)
    parser.add_argument("--prefix-metrics-subdir", default="reference_metrics_late_regime_final")
    parser.add_argument("--full-metrics-subdir", default="reference_metrics_late_regime_final")
    parser.add_argument("--replay-prefix-updates", type=int, default=10)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--expected-population", type=int, default=200)
    parser.add_argument("--expected-generations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selector_rows = load_rows(args.selector_run)
    prefix_rows = load_rows(args.prefix_run)
    expected_keys = set(prefix_rows)
    selector_keys = set(selector_rows)
    if selector_keys != expected_keys:
        raise AssertionError(
            f"selector/source row keys differ: selector={len(selector_keys)}, source={len(expected_keys)}"
        )

    expected_episodes = sorted({episode for episode, _ in selector_keys})
    for episode in expected_episodes:
        seeds = sorted(seed for item_episode, seed in selector_keys if item_episode == episode)
        if seeds != list(range(args.expected_seeds)):
            raise AssertionError(f"unexpected seeds for {episode}: {seeds}")

    prefix_stage_matches = 0
    late_cells = 0
    late_feasible = 0
    late_full = 0
    late_semantic_full = 0
    late_llm_only = 0
    executed_generations: list[int] = []
    trace_points: list[int] = []
    lineage_counts: dict[str, int] = {}
    exact_source_population_cells = 0
    late_restart_skills: dict[tuple[str, int, int], str] = {}
    for key, row in selector_rows.items():
        episode_id, run_seed = key
        source = prefix_rows[key]
        if canonical_solver_result(row.get("initial_solver_result")) != canonical_solver_result(
            source.get("initial_solver_result")
        ):
            raise AssertionError(f"initial artifact mismatch for {key}")
        prefix_stage_matches += 1
        selector_updates = list(row.get("update_results") or [])
        source_updates = list(source.get("update_results") or [])
        if len(selector_updates) < args.replay_prefix_updates + 2:
            raise AssertionError(f"selector row lacks late stages for {key}")
        for index in range(args.replay_prefix_updates):
            if canonical_update(selector_updates[index]) != canonical_update(source_updates[index]):
                raise AssertionError(f"rehydrated prefix mismatch for {key} stage {index + 1}")
            prefix_stage_matches += 1

        for stage in (args.replay_prefix_updates + 1, args.replay_prefix_updates + 2):
            update = selector_updates[stage - 1]
            restart = update.get("restart") or {}
            runtime = ((update.get("solver_result") or {}).get("metadata") or {}).get("runtime") or {}
            if int(runtime.get("population_size") or 0) != args.expected_population:
                raise AssertionError(f"population budget mismatch for {key} stage {stage}")
            if int(runtime.get("generations") or 0) != args.expected_generations:
                raise AssertionError(f"generation budget mismatch for {key} stage {stage}")
            executed = int(runtime.get("executed_generations") or 0)
            if not 1 <= executed <= args.expected_generations:
                raise AssertionError(f"invalid executed generations for {key} stage {stage}: {executed}")
            history = list(((update.get("solver_result") or {}).get("metadata") or {}).get("history") or [])
            if not history or int(history[-1].get("generation") or -1) != executed:
                raise AssertionError(f"metric history is incomplete for {key} stage {stage}")
            if not bool(runtime.get("record_metric_history")):
                raise AssertionError(f"metric history disabled for {key} stage {stage}")
            late_cells += 1
            late_feasible += int(bool(update.get("feasible")))
            restart_skill = str(restart.get("restart_skill") or "")
            if restart_skill not in {"full_restart_v1", "warm_restart_v1"}:
                raise AssertionError(f"unsupported LLM-only restart skill for {key} stage {stage}: {restart_skill}")
            late_restart_skills[(episode_id, stage, run_seed)] = restart_skill
            late_full += int(restart_skill == "full_restart_v1")
            late_semantic_full += int(
                bool((restart.get("semantic_restart_gate") or {}).get("verified_full_vote"))
            )
            llm_only = bool(
                restart.get("restart_selection_rule") == "verified_semantic_full_else_fixed_warm"
                and not restart.get("metric_restart_skill")
                and not restart.get("restart_fusion_rule")
                and not restart.get("objective_space_shift")
            )
            if not llm_only:
                raise AssertionError(f"non-LLM restart selector metadata for {key} stage {stage}")
            late_llm_only += 1
            executed_generations.append(executed)
            trace_points.append(len(history))
            policy = str((restart.get("population_lineage") or {}).get("previous_population_policy") or "")
            lineage_counts[policy] = lineage_counts.get(policy, 0) + 1
            source_metadata = (restart.get("population_lineage") or {}).get("source_previous_metadata") or {}
            source_population_kind = str(source_metadata.get("source_population_kind") or "")
            if policy == "source_liveopt_history":
                if source_population_kind != "final_population":
                    raise AssertionError(
                        f"matched predecessor is not the exact saved final population for {key} "
                        f"stage {stage}: {source_population_kind or 'missing'}"
                    )
                exact_source_population_cells += 1

    selector_metrics = load_metrics(args.selector_run / "reference_metrics" / "reference_stage_metrics.csv")
    prefix_metrics = load_metrics(
        args.prefix_run / args.prefix_metrics_subdir / "reference_stage_metrics.csv"
    )
    full_metrics = load_metrics(
        args.full_control_run / args.full_metrics_subdir / "reference_stage_metrics.csv"
    )
    metric_fields = ("hv", "normalized_hv", "igd", "normalized_score", "objective", "feasible")
    prefix_metric_matches = 0
    full_metric_matches = 0
    warm_metric_matches = 0
    routed_control_metric_matches = 0
    late_differences: list[float] = []
    episode_stage_rows: list[dict[str, Any]] = []
    for metric_key, selector_metric in selector_metrics.items():
        episode, stage, _ = metric_key
        if episode not in expected_episodes:
            continue
        if stage <= args.replay_prefix_updates:
            if not metrics_equal(selector_metric, prefix_metrics[metric_key], metric_fields):
                raise AssertionError(f"prefix held-out metric mismatch for {metric_key}")
            prefix_metric_matches += 1
        elif stage in (args.replay_prefix_updates + 1, args.replay_prefix_updates + 2):
            restart_skill = late_restart_skills.get(metric_key)
            if restart_skill == "full_restart_v1":
                control_metric = full_metrics[metric_key]
                if not metrics_equal(selector_metric, control_metric, metric_fields):
                    raise AssertionError(f"Full-routed replay metric mismatch for {metric_key}")
                full_metric_matches += 1
            elif restart_skill == "warm_restart_v1":
                control_metric = prefix_metrics[metric_key]
                if not metrics_equal(selector_metric, control_metric, metric_fields):
                    raise AssertionError(f"Warm-routed replay metric mismatch for {metric_key}")
                warm_metric_matches += 1
            else:
                raise AssertionError(f"missing routed restart skill for {metric_key}")
            routed_control_metric_matches += 1
            late_differences.append(
                quality_value(selector_metric) - quality_value(prefix_metrics[metric_key])
            )

    for episode in expected_episodes:
        for stage in (args.replay_prefix_updates + 1, args.replay_prefix_updates + 2):
            selector_values = [
                quality_value(selector_metrics[(episode, stage, seed)])
                for seed in range(args.expected_seeds)
            ]
            warm_values = [
                quality_value(prefix_metrics[(episode, stage, seed)])
                for seed in range(args.expected_seeds)
            ]
            full_values = [
                quality_value(full_metrics[(episode, stage, seed)])
                for seed in range(args.expected_seeds)
            ]
            episode_stage_rows.append(
                {
                    "episode_id": episode,
                    "stage": stage,
                    "selector_quality_mean": statistics.mean(selector_values),
                    "selector_quality_stdev": statistics.stdev(selector_values),
                    "fixed_warm_quality_mean": statistics.mean(warm_values),
                    "full_control_quality_mean": statistics.mean(full_values),
                    "selector_minus_warm": statistics.mean(selector_values) - statistics.mean(warm_values),
                    "selector_minus_full_control": statistics.mean(selector_values) - statistics.mean(full_values),
                }
            )

    report = {
        "status": "passed",
        "selector_run": str(args.selector_run),
        "prefix_run": str(args.prefix_run),
        "full_control_run": str(args.full_control_run),
        "episodes": expected_episodes,
        "seed_rows": len(selector_rows),
        "prefix_artifact_stage_matches": prefix_stage_matches,
        "prefix_metric_matches": prefix_metric_matches,
        "late_cells": late_cells,
        "late_feasible_cells": late_feasible,
        "late_full_cells": late_full,
        "late_semantic_full_cells": late_semantic_full,
        "late_llm_only_cells": late_llm_only,
        "late_full_metric_matches": full_metric_matches,
        "late_warm_metric_matches": warm_metric_matches,
        "late_routed_control_metric_matches": routed_control_metric_matches,
        "lineage_counts": lineage_counts,
        "exact_source_population_cells": exact_source_population_cells,
        "executed_generations": {
            "min": min(executed_generations),
            "max": max(executed_generations),
            "mean": statistics.mean(executed_generations),
        },
        "metric_trace_points": {
            "min": min(trace_points),
            "max": max(trace_points),
        },
        "selector_minus_warm_all_late_cells": statistics.mean(late_differences),
        "episode_stage_rows": episode_stage_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_rows(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = run_dir / "NLDO" / "evo2_limit0.jsonl"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("episode_id") or row.get("base_instance_id") or ""), int(row.get("run_seed") or 0))
            rows[key] = row
    return rows


def canonical_update(update: dict[str, Any]) -> str:
    payload = copy.deepcopy(update)
    solver = payload.get("solver_result") or {}
    solver["metadata"] = canonical_metadata(solver.get("metadata") or {})
    return stable_digest(payload)


def canonical_solver_result(result: dict[str, Any] | None) -> str:
    payload = copy.deepcopy(result or {})
    payload["metadata"] = canonical_metadata(payload.get("metadata") or {})
    return stable_digest(payload)


def canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(metadata)
    payload.pop("final_population", None)
    payload.pop("population", None)
    return payload


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_metrics(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["episode_id"], int(row["stage_index"]), int(row["run_seed"])): row
            for row in csv.DictReader(handle)
        }


def metrics_equal(left: dict[str, str], right: dict[str, str], fields: tuple[str, ...]) -> bool:
    return all(left.get(field) == right.get(field) for field in fields)


def quality_value(metric: dict[str, str]) -> float:
    for field in ("normalized_hv", "normalized_score"):
        value = metric.get(field)
        if value not in (None, ""):
            return float(value)
    raise AssertionError(
        f"metric row has neither normalized_hv nor normalized_score: "
        f"{metric.get('episode_id')} stage={metric.get('stage_index')} seed={metric.get('run_seed')}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
