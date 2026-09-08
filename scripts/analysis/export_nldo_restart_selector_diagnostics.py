#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ACTIONS = ("adaptive_mix_v1", "full_restart_v1", "population_transfer_v1", "warm_restart_v1")


def main() -> int:
    args = parse_args()
    episodes = {row["episode_id"]: row for row in load_jsonl(args.episodes_jsonl)}
    records = collect_records(load_jsonl(args.run_jsonl), episodes)
    rows = summarize(records)
    report = {
        "schema_version": "nldo_public_regime_warm_full_diagnostics_v3",
        "run_jsonl": str(args.run_jsonl),
        "records": len(records),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["stratum"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize public-regime Warm/Full selections by NLDO update stratum.")
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def collect_records(runs: list[dict[str, Any]], episodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in runs:
        episode_id = str(run.get("episode_id") or run.get("base_instance_id") or "")
        episode = episodes.get(episode_id) or {}
        public_updates = episode.get("update_stream") or []
        for index, update in enumerate(run.get("update_results") or []):
            impact = update.get("impact") if isinstance(update.get("impact"), dict) else {}
            raw = impact.get("raw") if isinstance(impact.get("raw"), dict) else {}
            solver = update.get("solver_result") if isinstance(update.get("solver_result"), dict) else {}
            metadata = solver.get("metadata") if isinstance(solver.get("metadata"), dict) else {}
            runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {}
            restart = runtime.get("restart") if isinstance(runtime.get("restart"), dict) else {}
            shift = raw.get("objective_space_shift") if isinstance(raw.get("objective_space_shift"), dict) else {}
            if not shift:
                shift = restart.get("objective_space_shift") if isinstance(restart.get("objective_space_shift"), dict) else {}
            if not shift:
                continue
            selected = str(impact.get("restart_skill") or restart.get("restart_skill") or "")
            public_update = public_updates[index] if index < len(public_updates) else {}
            records.append(
                {
                    "episode_id": episode_id,
                    "stage_index": index + 1,
                    "run_seed": run.get("run_seed"),
                    "stratum": str(public_update.get("difficulty") or "unlabeled"),
                    "selected": selected,
                    "objective_space_shift": as_float(shift.get("objective_space_shift")),
                    "fresh_population_ratio": as_float(
                        shift.get("fresh_population_ratio", restart.get("realized_fresh_population_ratio"))
                    ),
                    "history_population_ratio": as_float(
                        shift.get("history_population_ratio", restart.get("realized_history_population_ratio"))
                    ),
                    "direct_feasible_ratio": as_float(shift.get("direct_feasible_ratio")),
                    "public_state_churn": as_float(shift.get("public_state_churn")),
                    "public_state_changed_fact_count": as_float(shift.get("public_state_changed_fact_count")),
                    "segment_distance": as_float(shift.get("segment_distance")),
                    "objective_rank_disruption": as_float(shift.get("objective_rank_disruption")),
                    "executed_generations": runtime.get("executed_generations"),
                }
            )
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups["all"].append(record)
        groups[str(record["stratum"])].append(record)
        groups[f"episode:{record['episode_id']}"].append(record)
        groups[f"stage:{record['episode_id']}:t{int(record['stage_index']):02d}"].append(record)
    rows: list[dict[str, Any]] = []
    difficulty_groups = sorted(key for key in groups if key != "all" and not key.startswith("episode:"))
    difficulty_groups = [key for key in difficulty_groups if not key.startswith("stage:")]
    episode_groups = sorted(key for key in groups if key.startswith("episode:"))
    stage_groups = sorted(key for key in groups if key.startswith("stage:"))
    for stratum in ["all", *difficulty_groups, *episode_groups, *stage_groups]:
        items = groups[stratum]
        action_counts = Counter(str(item.get("selected") or "") for item in items)
        generations = [float(item["executed_generations"]) for item in items if item.get("executed_generations") is not None]
        shifts = [float(item["objective_space_shift"]) for item in items if item.get("objective_space_shift") is not None]
        fresh_ratios = [float(item["fresh_population_ratio"]) for item in items if item.get("fresh_population_ratio") is not None]
        history_ratios = [float(item["history_population_ratio"]) for item in items if item.get("history_population_ratio") is not None]
        feasible_ratios = [float(item["direct_feasible_ratio"]) for item in items if item.get("direct_feasible_ratio") is not None]
        state_churns = [float(item["public_state_churn"]) for item in items if item.get("public_state_churn") is not None]
        changed_facts = [float(item["public_state_changed_fact_count"]) for item in items if item.get("public_state_changed_fact_count") is not None]
        segment_distances = [float(item["segment_distance"]) for item in items if item.get("segment_distance") is not None]
        rank_disruptions = [float(item["objective_rank_disruption"]) for item in items if item.get("objective_rank_disruption") is not None]
        rows.append(
            {
                "stratum": stratum,
                "stage_seed_rows": len(items),
                "adaptive_mix_count": action_counts["adaptive_mix_v1"],
                "full_count": action_counts["full_restart_v1"],
                "transfer_count": action_counts["population_transfer_v1"],
                "warm_count": action_counts["warm_restart_v1"],
                "adaptive_mix_fraction": action_counts["adaptive_mix_v1"] / len(items) if items else None,
                "full_fraction": action_counts["full_restart_v1"] / len(items) if items else None,
                "transfer_fraction": action_counts["population_transfer_v1"] / len(items) if items else None,
                "warm_fraction": action_counts["warm_restart_v1"] / len(items) if items else None,
                "mean_objective_space_shift": mean(shifts) if shifts else None,
                "mean_fresh_population_ratio": mean(fresh_ratios) if fresh_ratios else None,
                "mean_history_population_ratio": mean(history_ratios) if history_ratios else None,
                "mean_direct_feasible_ratio": mean(feasible_ratios) if feasible_ratios else None,
                "mean_public_state_churn": mean(state_churns) if state_churns else None,
                "mean_public_state_changed_fact_count": mean(changed_facts) if changed_facts else None,
                "mean_segment_distance": mean(segment_distances) if segment_distances else None,
                "mean_objective_rank_disruption": mean(rank_disruptions) if rank_disruptions else None,
                "mean_executed_generations": mean(generations) if generations else None,
            }
        )
    return rows


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
