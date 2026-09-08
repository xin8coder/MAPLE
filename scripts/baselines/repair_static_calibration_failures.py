#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.agents.deepseek_client import DeepSeekDebugClient
from scripts.baselines.run_static_calibration import (
    compact_public_context,
    evaluate_official_static,
    evaluate_reference,
    extract_solver_code,
    normalize_usage,
    parsed_output_from_code_block,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one bounded public repair pass for static-calibration rows whose "
            "generated solver code failed to compile or run."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows", action="append", default=[], help="Input static_calibration_rows.jsonl files.")
    parser.add_argument("--glob", action="append", default=[], help="Glob for row files. May be repeated.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="liveopt_static,optimai_2025,or_llm_agent_2025")
    parser.add_argument("--case-ids", default="")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--cache-dir", default="outputs/deepseek_cache")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-new", type=int, default=0)
    args = parser.parse_args()

    configure_cache(args)
    manifest_path = Path(args.manifest)
    cases = load_cases(manifest_path)
    rows = latest_rows(load_rows(args.rows, args.glob))
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    case_filter = {item.strip() for item in args.case_ids.split(",") if item.strip()}
    candidates = [
        row
        for row in rows.values()
        if str(row.get("method")) in methods
        and (not case_filter or str(row.get("case_id")) in case_filter or str(row.get("source_id")) in case_filter)
        and is_code_hygiene_failure(row)
    ]

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "static_calibration_rows.jsonl"
    existing = latest_rows([json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]) if args.resume and rows_path.exists() else {}
    mode = "a" if args.resume and rows_path.exists() else "w"
    client = DeepSeekDebugClient(model=args.model)
    new_rows = 0
    skipped = 0
    with rows_path.open(mode, encoding="utf-8") as handle:
        for old in candidates:
            key = (str(old.get("case_id")), str(old.get("method")))
            if key in existing:
                skipped += 1
                continue
            case = cases.get(str(old.get("case_id")))
            if case is None:
                continue
            row = repair_one(case, old, client, args, trace_dir)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            existing[key] = row
            new_rows += 1
            if args.stop_after_new and new_rows >= args.stop_after_new:
                break

    summary = {
        "schema_version": "liveopt_static_code_repair_summary_v1",
        "input_candidate_rows": len(candidates),
        "new_rows": new_rows,
        "skipped_existing": skipped,
        "rows_path": str(rows_path),
        "methods": sorted(methods),
        "cache_only": bool(args.cache_only),
    }
    (out_dir / "static_calibration_repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def configure_cache(args: argparse.Namespace) -> None:
    os.environ["DEEPSEEK_CACHE"] = "1"
    if args.cache_dir:
        os.environ["DEEPSEEK_CACHE_DIR"] = args.cache_dir
    if args.cache_only:
        os.environ["DEEPSEEK_CACHE_ONLY"] = "1"
        os.environ.setdefault("DEEPSEEK_API_KEY", "cache-only-dummy-key")
    else:
        os.environ.pop("DEEPSEEK_CACHE_ONLY", None)


def load_cases(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_file = Path(manifest["case_file"])
    if not case_file.exists():
        candidate = manifest_path.parent / case_file.name
        if candidate.exists():
            case_file = candidate
    cases = {}
    for line in case_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases[str(row.get("case_id"))] = row
    return cases


def load_rows(paths: list[str], patterns: list[str]) -> list[dict[str, Any]]:
    row_paths = [Path(path) for path in paths]
    for pattern in patterns:
        row_paths.extend(sorted(Path().glob(pattern)))
    rows: list[dict[str, Any]] = []
    for path in row_paths:
        if path.is_dir():
            path = path / "static_calibration_rows.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def latest_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("case_id")), str(row.get("method")))] = row
    return latest


def is_code_hygiene_failure(row: dict[str, Any]) -> bool:
    eval_row = row.get("official_evaluation") if isinstance(row.get("official_evaluation"), dict) else {}
    if eval_row.get("official_accuracy") is True:
        return False
    if eval_row.get("compile_success") is False:
        return True
    return eval_row.get("compile_success") is True and eval_row.get("runtime_success") is False


def repair_one(
    case: dict[str, Any],
    old: dict[str, Any],
    client: DeepSeekDebugClient,
    args: argparse.Namespace,
    trace_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    method = str(old.get("method"))
    messages = build_repair_messages(case, old)
    raw_response = ""
    parsed: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    status = "failed"
    failure_reason = ""
    validation_feedback: list[str] = []
    try:
        response = client.chat(messages, temperature=0.0, max_tokens=5000, json_mode=False)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        raw_response = response.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(response, dict) else ""
        code = extract_python_code(raw_response)
        if not code:
            raise ValueError("repair response did not contain Python code")
        parsed = parsed_output_from_code_block(method, code)
        status = "completed"
    except Exception as exc:  # noqa: BLE001
        failure_reason = str(exc)
        status = "cache_miss_blocked" if "local response cache missed" in failure_reason else "failed"
        validation_feedback = [failure_reason]
    official = evaluate_official_static(case, parsed)
    reference = evaluate_reference(case, parsed)
    trace_path = write_trace(trace_dir, case, old, messages, raw_response, parsed, usage, status, failure_reason)
    return {
        "schema_version": "liveopt_static_calibration_row_v1",
        "prompt_version": "static_code_hygiene_repair_v1",
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "method": method,
        "model": args.model,
        "status": status,
        "problem_text": case.get("problem_text", ""),
        "reference_available": bool(case.get("reference")),
        "reference_keys": sorted((case.get("reference") or {}).keys()) if isinstance(case.get("reference"), dict) else [],
        "messages": messages,
        "selected_stage_prompt": messages[-1]["content"] if messages else "",
        "raw_response": raw_response,
        "parsed_output": parsed,
        "validation_feedback": validation_feedback,
        "reference_evaluation": reference,
        "official_evaluation": official,
        "token_usage": normalize_usage(usage),
        "latency_seconds": time.perf_counter() - started,
        "failure_reason": failure_reason,
        "trace_path": str(trace_path),
        "repair_parent_trace_path": old.get("trace_path", ""),
        "repair_parent_status": old.get("status", ""),
        "repair_parent_official_evaluation": old.get("official_evaluation", {}),
    }


def build_repair_messages(case: dict[str, Any], old: dict[str, Any]) -> list[dict[str, str]]:
    method = str(old.get("method"))
    eval_row = old.get("official_evaluation") if isinstance(old.get("official_evaluation"), dict) else {}
    public_context = compact_public_context(case.get("public_context"))
    old_code = extract_solver_code(old.get("parsed_output") if isinstance(old.get("parsed_output"), dict) else {})
    prompt = f"""Repair one public static optimization solver artifact.

Method style: {method}
Problem source: {case.get('source')}
Problem id: {case.get('source_id') or case.get('case_id')}

Natural-language optimization problem:
{case.get('problem_text', '')}

Public context JSON:
{json.dumps(public_context, ensure_ascii=False, sort_keys=True)}

Observed public failure:
- compile_success: {eval_row.get('compile_success')}
- runtime_success: {eval_row.get('runtime_success')}
- compile_error: {eval_row.get('compile_error', '')}
- runtime_error: {eval_row.get('runtime_error', '')}
- stderr_preview: {eval_row.get('stderr_preview', '')}
- stdout_preview: {eval_row.get('stdout_preview', '')}

Previous solver code:
```python
{old_code}
```

Repair rules:
- Use only the public problem text and public context above.
- Do not use hidden labels, reference objective values, private benchmark modules, gurobipy, pandas, networkx, or requests.
- The repaired code must read public parameters from `parameters.json` using the exact keys in the public context.
- Preserve the intended mathematical model; fix missing code, syntax errors, wrong parameter keys, import errors, shape mistakes, and stdout formatting.
- Print exactly one JSON object containing `objective_value` and `solution`; for infeasible/unbounded/no-optimal public cases, print `solution_status`.
- Prefer scipy.optimize.linprog for LPs and scipy.optimize.milp for integer or mixed-integer cases.
- When using scipy.optimize.milp, import and pass `Bounds(lb, ub)` rather than a list of tuple bounds; pass constraints as `LinearConstraint` objects.
- If the public prompt says to "formulate" a model but the benchmark evaluates an answer, build and solve the public model, then print the optimal objective and solution.
- Return only one complete executable Python code block.
"""
    return [
        {
            "role": "system",
            "content": "You repair public optimization solver code. Return only one complete Python code block.",
        },
        {"role": "user", "content": prompt},
    ]


def extract_python_code(text: str) -> str:
    raw = text or ""
    start = raw.find("```")
    while start >= 0:
        line_end = raw.find("\n", start + 3)
        if line_end < 0:
            break
        lang = raw[start + 3 : line_end].strip().lower()
        end = raw.find("```", line_end + 1)
        if end < 0:
            break
        body = raw[line_end + 1 : end].strip()
        if body and lang in {"", "python", "py"}:
            return body
        start = raw.find("```", end + 3)
    stripped = raw.strip()
    if stripped.startswith("import ") or stripped.startswith("from ") or "print(" in stripped:
        return stripped
    return ""


def write_trace(
    trace_dir: Path,
    case: dict[str, Any],
    old: dict[str, Any],
    messages: list[dict[str, str]],
    raw_response: str,
    parsed: dict[str, Any],
    usage: dict[str, Any],
    status: str,
    failure_reason: str,
) -> Path:
    method = str(old.get("method") or "method")
    case_id = str(case.get("case_id") or "case").replace("/", "_")
    path = trace_dir / method / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "method": method,
        "prompt_version": "static_code_hygiene_repair_v1",
        "parent_trace_path": old.get("trace_path", ""),
        "messages": messages,
        "raw_response": raw_response,
        "parsed_output": parsed,
        "usage": normalize_usage(usage),
        "status": status,
        "failure_reason": failure_reason,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
