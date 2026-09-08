#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.agents.deepseek_client import DeepSeekDebugClient, load_dotenv
from scripts.baselines.run_nldo_dynamic_public_baselines import (
    add_usage,
    configure_cache_env,
    jsonable,
    public_context_with_loaded_tables,
    run_stage_method,
    zero_usage,
)


DEFAULT_SOURCE_STAGE_ROWS = [
    "logs/llm_tests/tss_full_dynamic_public_react_optimai_orllm_provider_proxy_20260703/dynamic_public_baseline_stage_rows.jsonl",
    "logs/llm_tests/tss_full_dynamic_public_optimus_orlm_provider_proxy_20260703/dynamic_public_baseline_stage_rows.jsonl",
]


def main() -> int:
    args = parse_args()
    args.provider = getattr(args, "provider", "deepseek")  # rerun is DeepSeek-only; shared cache config expects it
    out_dir = Path(args.out_dir)
    configure_cache_env(args, out_dir)
    trace_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    episodes = {str(row.get("episode_id")): row for row in load_jsonl(Path(args.episodes_jsonl))}
    source_rows = load_source_stage_rows(args.source_stage_rows)
    targets = select_targets(source_rows, args)
    if args.limit and args.limit > 0:
        targets = targets[: args.limit]

    rows_path = out_dir / "dynamic_public_affected_stage_reruns.jsonl"
    existing_keys = existing_target_keys(rows_path) if args.resume else set()
    client = DeepSeekDebugClient(model=args.model)
    started = time.perf_counter()
    written = 0
    method_usage: dict[str, dict[str, int]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    target_rows: list[dict[str, Any]] = []
    with rows_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for target in targets:
            key = target_key(target)
            if key in existing_keys:
                continue
            episode = episodes.get(str(target.get("episode_id")))
            if not episode:
                continue
            stage_index = int(target.get("stage_index") or 0)
            stage = {
                "stage_index": stage_index,
                "update_id": str(target.get("update_id") or ("initial" if stage_index == 0 else f"t{stage_index:03d}")),
                "updates": list(episode.get("update_stream") or [])[:stage_index],
            }
            method = str(target.get("method"))
            public_context = public_context_with_loaded_tables(episode)
            previous_solution = previous_accepted_solution(source_rows, method, str(target.get("episode_id")), stage_index)
            row = run_stage_method(
                args,
                client,
                episode,
                public_context,
                method,
                stage,
                trace_dir,
                previous_solution,
                previous_scoring_solution=previous_solution,
            )
            row["rerun_source"] = {
                "source_status": target.get("status"),
                "source_failure_reason": target.get("failure_reason"),
                "source_trace_path": target.get("trace_path"),
                "target_selection_policy": target.get("target_selection_policy"),
            }
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            written += 1
            target_rows.append(row)
            method_usage[method] = add_usage(method_usage.get(method, zero_usage()), row.get("token_usage") or {})
            status_counts.setdefault(method, {})
            status = str(row.get("status") or "")
            status_counts[method][status] = status_counts[method].get(status, 0) + 1
            if args.stop_after_new and written >= args.stop_after_new:
                break

    summary = summarize(targets, target_rows, method_usage, status_counts, args, started, rows_path, trace_dir)
    summary_path = out_dir / "dynamic_public_affected_stage_rerun_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Rerun only affected NLDO dynamic public-baseline stage rows.")
    parser.add_argument("--episodes-jsonl", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")
    parser.add_argument("--source-stage-rows", action="append", default=list(DEFAULT_SOURCE_STAGE_ROWS))
    parser.add_argument("--methods", default="")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--retry-statuses", default="compile_failed,runtime_failed,schema_failed")
    parser.add_argument(
        "--include-provider-failures",
        action="store_true",
        help="Also retry provider-level failed rows such as previous request timeouts.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="logs/llm_tests/nldo_dynamic_public_affected_reruns/latest")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--cache-dir", default="outputs/deepseek_cache")
    parser.add_argument("--use-response-cache", dest="use_response_cache", action="store_true", default=True)
    parser.add_argument("--no-response-cache", dest="use_response_cache", action="store_false")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--response-protocol", choices=("json", "code_block"), default="json")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--code-repair-attempts",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CODE_REPAIR_ATTEMPTS", "3") or 3),
    )
    parser.add_argument("--total-timeout", type=float, default=900.0)
    parser.add_argument("--stop-after-new", type=int, default=0)
    parser.add_argument("--continue-after-hidden-rejection", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_source_stage_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in paths:
        if item in seen_paths:
            continue
        seen_paths.add(item)
        rows.extend(load_jsonl(Path(item)))
    return rows


def select_targets(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = {item.strip() for item in str(args.methods or "").split(",") if item.strip()}
    episodes = {str(item) for item in (args.episode_id or [])}
    retry_statuses = {item.strip() for item in str(args.retry_statuses or "").split(",") if item.strip()}
    targets: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method") or "")
        episode_id = str(row.get("episode_id") or "")
        status = str(row.get("status") or "")
        if methods and method not in methods:
            continue
        if episodes and episode_id not in episodes:
            continue
        selected = status in retry_statuses
        if args.include_provider_failures and status == "failed":
            selected = True
        if not selected:
            continue
        target = dict(row)
        target["target_selection_policy"] = "status_filtered_stage_rerun_v1"
        targets[(method, episode_id, int(row.get("stage_index") or 0))] = target
    return sorted(targets.values(), key=lambda r: (str(r.get("method")), str(r.get("episode_id")), int(r.get("stage_index") or 0)))


def target_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("method")), str(row.get("episode_id")), int(row.get("stage_index") or 0))


def existing_target_keys(path: Path) -> set[tuple[str, str, int]]:
    return {target_key(row) for row in load_jsonl(path)}


def previous_accepted_solution(rows: list[dict[str, Any]], method: str, episode_id: str, stage_index: int) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_stage = -1
    for row in rows:
        if str(row.get("method")) != method or str(row.get("episode_id")) != episode_id:
            continue
        current_stage = int(row.get("stage_index") or 0)
        if current_stage >= stage_index or current_stage <= best_stage:
            continue
        hidden = row.get("hidden_evaluation") if isinstance(row.get("hidden_evaluation"), dict) else {}
        if not hidden.get("feasible"):
            continue
        solution = hidden.get("canonical_solution") if isinstance(hidden.get("canonical_solution"), dict) else row.get("solution")
        if isinstance(solution, dict) and solution:
            best = solution
            best_stage = current_stage
    return best


def summarize(
    targets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    method_usage: dict[str, dict[str, int]],
    status_counts: dict[str, dict[str, int]],
    args: argparse.Namespace,
    started: float,
    rows_path: Path,
    trace_dir: Path,
) -> dict[str, Any]:
    method_targets: dict[str, int] = {}
    for row in targets:
        method = str(row.get("method") or "")
        method_targets[method] = method_targets.get(method, 0) + 1
    methods: dict[str, dict[str, Any]] = {}
    for method in sorted(set(method_targets) | set(status_counts)):
        method_rows = [row for row in rows if str(row.get("method")) == method]
        feasible = sum(1 for row in method_rows if (row.get("hidden_evaluation") or {}).get("feasible"))
        methods[method] = {
            "targets": method_targets.get(method, 0),
            "rerun_rows": len(method_rows),
            "status_counts": status_counts.get(method, {}),
            "hidden_feasible_rows": feasible,
            "total_tokens": int((method_usage.get(method) or {}).get("total_tokens", 0) or 0),
        }
    return {
        "schema_version": "nldo_dynamic_public_affected_stage_rerun_summary_v1",
        "mode": "cache_only" if args.cache_only else "provider_backed",
        "targets": len(targets),
        "rerun_rows": len(rows),
        "methods": methods,
        "rows_path": str(rows_path),
        "trace_dir": str(trace_dir),
        "response_protocol": args.response_protocol,
        "code_repair_attempts": int(args.code_repair_attempts or 0),
        "code_repair_policy": "equal_public_code_output_repair_v1",
        "latency_seconds": time.perf_counter() - started,
    }


def console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "targets": summary.get("targets"),
        "rerun_rows": summary.get("rerun_rows"),
        "rows_path": summary.get("rows_path"),
        "summary_json": summary.get("summary_json"),
        "methods": summary.get("methods"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
