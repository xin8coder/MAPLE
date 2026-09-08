#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.baselines.run_static_calibration import build_messages, evaluate_official_static, evaluate_reference


def main() -> int:
    parser = argparse.ArgumentParser(description="Write explicit missing static-calibration rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--existing-glob", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="liveopt_static,optimai_2025,or_llm_agent_2025")
    parser.add_argument("--status", default="not_completed_timeout")
    parser.add_argument("--failure-reason", default="provider repair timed out before a row was written")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_file = Path(manifest["case_file"])
    if not case_file.exists():
        case_file = manifest_path.parent / case_file.name
    cases = [json.loads(line) for line in case_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    existing = load_existing(args.existing_glob)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "static_calibration_rows.jsonl"
    rows: list[dict[str, Any]] = []
    for case in cases:
        for method in methods:
            key = (str(case.get("case_id")), method)
            if key in existing:
                continue
            messages = build_messages(case, method)
            rows.append(
                {
                    "schema_version": "liveopt_static_calibration_row_v1",
                    "case_id": case.get("case_id"),
                    "source": case.get("source"),
                    "source_id": case.get("source_id"),
                    "method": method,
                    "status": args.status,
                    "problem_text": case.get("problem_text", ""),
                    "reference_available": bool(case.get("reference")),
                    "reference_keys": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
                    "messages": messages,
                    "selected_stage_prompt": messages[-1]["content"] if messages else "",
                    "raw_response": "",
                    "parsed_output": {},
                    "validation_feedback": [args.failure_reason],
                    "reference_evaluation": evaluate_reference(case, {}),
                    "official_evaluation": evaluate_official_static(case, {}),
                    "token_usage": {},
                    "latency_seconds": 0.0,
                    "failure_reason": args.failure_reason,
                    "trace_path": "",
                }
            )
    rows_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(json.dumps({"missing_rows": len(rows), "rows_path": str(rows_path)}, indent=2))
    return 0


def load_existing(patterns: list[str]) -> set[tuple[str, str]]:
    existing: set[tuple[str, str]] = set()
    for pattern in patterns:
        for path_str in glob.glob(pattern):
            path = Path(path_str)
            if path.is_dir():
                path = path / "static_calibration_rows.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                existing.add((str(row.get("case_id")), str(row.get("method"))))
    return existing


if __name__ == "__main__":
    raise SystemExit(main())
