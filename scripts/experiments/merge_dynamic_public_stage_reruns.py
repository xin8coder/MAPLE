#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    args = parse_args()
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for run_dir in args.base_run_dir:
        path = Path(run_dir) / "dynamic_public_baseline_stage_rows.jsonl"
        for row in load_jsonl(path):
            key = stage_key(row)
            if key:
                latest[key] = tag_source(row, "base", str(path))
                source_counts["base_rows"] += 1
    overlay_rows = 0
    replaced = 0
    for item in args.rerun_stage_rows:
        path = Path(item)
        if path.is_dir():
            candidates = [
                path / "dynamic_public_affected_stage_reruns.jsonl",
                path / "dynamic_public_baseline_stage_rows.jsonl",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        for row in load_jsonl(path):
            key = stage_key(row)
            if not key:
                continue
            overlay_rows += 1
            if key in latest:
                replaced += 1
            latest[key] = tag_source(row, "rerun", str(path))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "dynamic_public_baseline_stage_rows.jsonl"
    merged_rows = [latest[key] for key in sorted(latest)]
    write_jsonl(rows_path, merged_rows)
    summary = {
        "schema_version": "dynamic_public_stage_rerun_merge_summary_v1",
        "base_run_dirs": args.base_run_dir,
        "rerun_stage_rows": args.rerun_stage_rows,
        "base_rows_read": source_counts["base_rows"],
        "rerun_rows_read": overlay_rows,
        "rerun_rows_replaced_existing_keys": replaced,
        "merged_rows": len(merged_rows),
        "rows_path": str(rows_path),
        "method_counts": dict(Counter(str(row.get("method") or "") for row in merged_rows)),
        "status_counts": {
            method: dict(Counter(str(row.get("status") or "") for row in merged_rows if str(row.get("method") or "") == method))
            for method in sorted({str(row.get("method") or "") for row in merged_rows})
        },
    }
    summary_path = out_dir / "dynamic_public_baseline_merge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay stage-level public-baseline reruns onto an existing dynamic-public run directory.")
    parser.add_argument("--base-run-dir", action="append", required=True)
    parser.add_argument("--rerun-stage-rows", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    method = str(row.get("method") or "")
    episode_id = str(row.get("episode_id") or "")
    try:
        stage_index = int(row.get("stage_index"))
    except (TypeError, ValueError):
        return None
    if not method or not episode_id:
        return None
    return method, episode_id, stage_index


def tag_source(row: dict[str, Any], kind: str, path: str) -> dict[str, Any]:
    out = dict(row)
    out["merged_stage_source"] = {"kind": kind, "path": path}
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
