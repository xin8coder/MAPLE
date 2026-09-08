#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


def main() -> int:
    args = parse_args()
    by_episode: dict[str, list[dict[str, float]]] = defaultdict(list)
    for run in load_jsonl(args.run_jsonl):
        episode_id = str(run.get("episode_id") or "")
        stages = [run.get("initial_solver_result") or {}]
        stages.extend((result.get("solver_result") or {}) for result in run.get("update_results") or [])
        for stage_index, solver_result in enumerate(stages):
            metadata = solver_result.get("metadata") or {}
            archive = metadata.get("candidate_archive") or []
            runtime = metadata.get("runtime") or {}
            archive_diagnostics = metadata.get("archive_diagnostics") or runtime.get("archive_diagnostics") or {}
            objectives = [objective_tuple(item) for item in archive]
            objectives = [value for value in objectives if value]
            unique_objectives = len(set(objectives))
            unique_solutions = len({canonical_solution(item) for item in archive})
            by_episode[episode_id].append(
                {
                    "stage_index": float(stage_index),
                    "archive_size": float(len(archive)),
                    "unique_objectives": float(unique_objectives),
                    "unique_solutions": float(unique_solutions),
                    "duplicate_objective_rate": 1.0 - unique_objectives / max(1, len(objectives)),
                    "cap_active": (
                        1.0
                        if bool(archive_diagnostics.get("cap_active"))
                        else (1.0 if not archive_diagnostics and len(archive) >= args.archive_cap else 0.0)
                    ),
                }
            )
    rows = []
    for episode_id, stages in sorted(by_episode.items()):
        rows.append(
            {
                "episode_id": episode_id,
                "stage_seed_observations": len(stages),
                "mean_archive_size": mean(item["archive_size"] for item in stages),
                "mean_unique_objectives": mean(item["unique_objectives"] for item in stages),
                "mean_unique_solutions": mean(item["unique_solutions"] for item in stages),
                "mean_duplicate_objective_rate": mean(item["duplicate_objective_rate"] for item in stages),
                "cap_active_rate": mean(item["cap_active"] for item in stages),
            }
        )
    report = {
        "schema_version": "nldo_archive_diagnostics_v1",
        "run_jsonl": str(args.run_jsonl),
        "archive_cap": args.archive_cap,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["episode_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure submitted-archive saturation, objective uniqueness, and duplicates.")
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument("--archive-cap", type=int, default=200)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def objective_tuple(item: dict[str, Any]) -> tuple[float, ...]:
    raw = item.get("workbench_objectives") or (item.get("fitness") or {}).get("objectives") or []
    try:
        return tuple(round(float(value), 12) for value in raw)
    except (TypeError, ValueError):
        return ()


def canonical_solution(item: dict[str, Any]) -> str:
    solution = item.get("solution") or (item.get("fitness") or {}).get("solution") or {}
    return json.dumps(solution, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    raise SystemExit(main())
