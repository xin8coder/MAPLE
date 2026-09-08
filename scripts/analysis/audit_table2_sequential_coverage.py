#!/usr/bin/env python3
"""Audit Table 2 as 15 method trajectories over t00--t12.

The audit distinguishes a recorded first failure (a legitimate zero suffix)
from a missing row after a feasible prefix (incomplete experimental coverage).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    ROOT
    / "logs/llm_tests/table2_deepseek_continuation_20260717/merged"
    / "dynamic_public_baseline_stage_rows.rescored.jsonl",
)
DEFAULT_OUT = ROOT / "logs/analysis/table2_deepseek_sequential_coverage_20260717"
METHODS = (
    "react_tools",
    "optimus",
    "orlm",
    "optimai_2025",
    "or_llm_agent_2025",
)
EPISODES = tuple(f"NLDO-P{number:03d}" for number in range(1, 16))
STAGES = tuple(range(13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_rows(paths: list[Path]) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row.get("method") or ""),
                str(row.get("episode_id") or ""),
                int(row.get("stage_index") or 0),
            )
            rows[key] = row
    return rows


def feasible(row: dict[str, Any] | None) -> bool:
    return bool(((row or {}).get("hidden_evaluation") or {}).get("feasible"))


def audit(rows: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method in METHODS:
        for episode_id in EPISODES:
            prefix = 0
            terminal_kind = "complete"
            terminal_stage: int | None = None
            for stage in STAGES:
                row = rows.get((method, episode_id, stage))
                if row is None:
                    terminal_kind = "coverage_gap"
                    terminal_stage = stage
                    break
                if not feasible(row):
                    terminal_kind = "recorded_failure"
                    terminal_stage = stage
                    break
                prefix += 1
            raw_rows = [
                row
                for stage in STAGES
                if (row := rows.get((method, episode_id, stage))) is not None
            ]
            records.append(
                {
                    "method": method,
                    "episode_id": episode_id,
                    "view": "DS" if int(episode_id[-3:]) <= 9 else "DM",
                    "initial_recorded": rows.get((method, episode_id, 0)) is not None,
                    "initial_feasible": feasible(rows.get((method, episode_id, 0))),
                    "valid_prefix_states": prefix,
                    "terminal_kind": terminal_kind,
                    "terminal_stage": terminal_stage,
                    "raw_recorded_states": len(raw_rows),
                    "raw_feasible_states": sum(feasible(row) for row in raw_rows),
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in records if row["method"] == method]
        views = {}
        for view in ("DS", "DM"):
            block = [row for row in selected if row["view"] == view]
            denominator = len(block) * len(STAGES)
            views[view] = {
                "trajectories": len(block),
                "denominator_states": denominator,
                "valid_prefix_states": sum(int(row["valid_prefix_states"]) for row in block),
                "initial_feasible": sum(bool(row["initial_feasible"]) for row in block),
                "coverage_gaps": sum(row["terminal_kind"] == "coverage_gap" for row in block),
                "recorded_failures": sum(row["terminal_kind"] == "recorded_failure" for row in block),
                "complete_trajectories": sum(row["terminal_kind"] == "complete" for row in block),
                "raw_recorded_states": sum(int(row["raw_recorded_states"]) for row in block),
                "raw_feasible_states": sum(int(row["raw_feasible_states"]) for row in block),
            }
        methods[method] = views
    gaps = [row for row in records if row["terminal_kind"] == "coverage_gap"]
    return {
        "protocol": "table2_sequential_coverage_t00_t12_v1",
        "status": "passed" if not gaps else "incomplete",
        "states": list(STAGES),
        "missing_after_feasible_prefix": len(gaps),
        "methods": methods,
        "coverage_gap_jobs": [
            {
                "method": row["method"],
                "episode_id": row["episode_id"],
                "resume_from_stage": row["terminal_stage"],
            }
            for row in gaps
        ],
    }


def main() -> int:
    args = parse_args()
    sources = list(args.source or DEFAULT_SOURCES)
    records = audit(load_rows(sources))
    summary = summarize(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "trajectory_coverage.csv", records)
    (args.out_dir / "summary.json").write_text(
        json.dumps({**summary, "sources": [str(path) for path in sources]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if args.strict and summary["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
