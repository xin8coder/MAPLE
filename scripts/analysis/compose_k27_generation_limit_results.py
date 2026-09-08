#!/usr/bin/env python3
"""Compose the targeted Kimi generation-limit reruns into one result bundle.

The reruns are intentionally split into two provider queues.  Each queue starts
from the same complete 32k result file, so blindly overlaying one merged file on
the other would restore stale rows for the methods owned by the other queue.
This script assigns each method to exactly one shard, composes only those rows,
and keeps the bundle partial until every originally truncated trajectory has
terminal evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.rerun_k27_generation_limit_rows import (
    FAILURE_MARKER,
    STAGES_PER_EPISODE,
    atomic_json,
    atomic_jsonl,
    load_json,
    load_jsonl,
    row_key,
    summarize_rows,
)


DEFAULT_SOURCE = ROOT / (
    "logs/llm_tests/tss_full_dynamic_public_baselines_k27_complete_20260812/"
    "dynamic_public_baseline_stage_rows.jsonl"
)
DEFAULT_COST_OVERLAY = ROOT / (
    "logs/llm_tests/tss_dynamic_public_k27_98k_micro_20260812/"
    "dynamic_public_baseline_stage_rows.cost_corrected.jsonl"
)
DEFAULT_MAIN_ROOT = ROOT / "logs/llm_tests/tss_k27_generation_limit_98k_20260812"
DEFAULT_OFFICIAL_ROOT = ROOT / (
    "logs/llm_tests/tss_k27_generation_limit_98k_official_shard_20260813"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "logs/llm_tests/tss_k27_generation_limit_98k_composed_20260813"
)

MAIN_METHODS = {"optimai_2025", "persistent_react", "react_tools"}
OFFICIAL_METHODS = {"optimus", "or_llm_agent_2025"}
TERMINAL_JOB_STATUSES = {"reused", "completed", "stopped_after_failure"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cost-overlay", type=Path, default=DEFAULT_COST_OVERLAY)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def targeted_jobs(source: Path) -> set[tuple[str, str]]:
    return {
        (str(row.get("method") or ""), str(row.get("episode_id") or ""))
        for row in load_jsonl(source)
        if FAILURE_MARKER in str(row.get("failure_reason") or "")
    }


def terminal_jobs(root: Path, methods: set[str]) -> set[tuple[str, str]]:
    manifest = load_json(root / "manifest.json")
    return {
        (str(job.get("method") or ""), str(job.get("episode_id") or ""))
        for job in manifest.get("jobs") or []
        if str(job.get("method") or "") in methods
        and str(job.get("status") or "") in TERMINAL_JOB_STATUSES
    }


def apply_method_shard(
    merged: dict[tuple[str, str, int], dict[str, Any]],
    root: Path,
    methods: set[str],
) -> None:
    path = root / "merged" / "dynamic_public_baseline_stage_rows.jsonl"
    if not path.exists():
        return
    for row in load_jsonl(path):
        if str(row.get("method") or "") in methods:
            merged[row_key(row)] = row


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    merged = {row_key(row): row for row in load_jsonl(args.source)}
    if args.cost_overlay.exists():
        merged.update({row_key(row): row for row in load_jsonl(args.cost_overlay)})
    apply_method_shard(merged, args.main_root, MAIN_METHODS)
    apply_method_shard(merged, args.official_root, OFFICIAL_METHODS)

    rows_path = args.output_root / "dynamic_public_baseline_stage_rows.jsonl"
    atomic_jsonl(rows_path, sorted(merged.values(), key=row_key))
    metric_summary = summarize_rows(rows_path, args.source)

    targets = targeted_jobs(args.source)
    terminal = terminal_jobs(args.main_root, MAIN_METHODS) | terminal_jobs(
        args.official_root, OFFICIAL_METHODS
    )
    pending = sorted(targets - terminal)
    summary = {
        **metric_summary,
        "status": "completed" if not pending else "partial",
        "source": str(args.source),
        "cost_overlay": str(args.cost_overlay),
        "method_shards": {
            str(args.main_root): sorted(MAIN_METHODS),
            str(args.official_root): sorted(OFFICIAL_METHODS),
        },
        "targeted_trajectories": len(targets),
        "terminal_targeted_trajectories": len(targets & terminal),
        "pending_targeted_trajectories": [
            {"method": method, "episode_id": episode_id}
            for method, episode_id in pending
        ],
        "states_per_trajectory": STAGES_PER_EPISODE,
    }
    summary_path = args.output_root / "dynamic_public_baseline_summary.json"
    atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "targeted": summary["targeted_trajectories"],
                "terminal": summary["terminal_targeted_trajectories"],
                "pending": len(pending),
                "rows": len(merged),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
