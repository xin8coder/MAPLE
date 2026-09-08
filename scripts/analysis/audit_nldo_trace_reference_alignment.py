#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that each saved final-generation HV equals the post-hoc strong-reference stage metric."
    )
    parser.add_argument("--run-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_csv)
    compared = 0
    missing = []
    mismatches = []
    for run in load_jsonl(args.run_jsonl):
        episode_id = str(run.get("episode_id") or "")
        seed = str(run.get("run_seed"))
        solver_results = [run.get("initial_solver_result") or {}]
        solver_results.extend((update.get("solver_result") or {}) for update in run.get("update_results") or [])
        for stage_index, solver_result in enumerate(solver_results):
            key = (episode_id, seed, stage_index)
            expected = metrics.get(key)
            if expected is None:
                continue
            history = (((solver_result.get("metadata") or {}).get("history")) or [])
            traced = next(
                (
                    float(item["normalized_hv"])
                    for item in reversed(history)
                    if isinstance(item, dict) and item.get("normalized_hv") not in (None, "")
                ),
                None,
            )
            if traced is None:
                missing.append({"episode_id": episode_id, "run_seed": seed, "stage_index": stage_index})
                continue
            compared += 1
            delta = traced - expected
            if abs(delta) > args.tolerance:
                mismatches.append(
                    {
                        "episode_id": episode_id,
                        "run_seed": seed,
                        "stage_index": stage_index,
                        "traced_normalized_hv": traced,
                        "posthoc_normalized_hv": expected,
                        "delta": delta,
                    }
                )

    payload = {
        "schema_version": "nldo_trace_reference_alignment_v1",
        "run_jsonl": str(args.run_jsonl),
        "metrics_csv": str(args.metrics_csv),
        "tolerance": args.tolerance,
        "compared_stage_seed_rows": compared,
        "missing_trace_count": len(missing),
        "mismatch_count": len(mismatches),
        "passed": not missing and not mismatches,
        "missing_traces": missing,
        "mismatches": mismatches,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and (missing or mismatches) else 0


def load_metrics(path: Path) -> dict[tuple[str, str, int], float]:
    metrics: dict[tuple[str, str, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("normalized_hv")
            if value in (None, ""):
                continue
            key = (str(row.get("episode_id") or ""), str(row.get("run_seed")), int(row.get("stage_index") or 0))
            metrics[key] = float(value)
    return metrics


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    raise SystemExit(main())
