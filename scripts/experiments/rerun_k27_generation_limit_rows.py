#!/usr/bin/env python3
"""Resume only Kimi k2.7 trajectories stopped by the former 32k cap."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/baselines/run_nldo_dynamic_public_baselines.py"
DEFAULT_SOURCE = ROOT / (
    "logs/llm_tests/tss_full_dynamic_public_baselines_k27_complete_20260812/"
    "dynamic_public_baseline_stage_rows.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "logs/llm_tests/tss_k27_generation_limit_98k_20260812"
FAILURE_MARKER = "Kimi response exhausted max_tokens before completing the visible answer"
TERMINAL_STATUSES = {"completed", "stopped_after_failure"}
STAGES_PER_EPISODE = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overlay-rows", type=Path, action="append", default=[])
    parser.add_argument("--model", default="k2.7")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--method", action="append")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true", default=True)
    parser.add_argument("--no-system-proxy", dest="use_system_proxy", action="store_false")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("method") or ""),
        str(row.get("episode_id") or ""),
        int(row.get("stage_index") or 0),
    )


def add_token_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "local_cache_hits",
        "local_cache_saved_tokens",
    }
    return {
        key: int(left.get(key, 0) or 0) + int(right.get(key, 0) or 0)
        for key in sorted(keys)
    }


def target_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = set(args.method or [])
    episodes = set(args.episode_id or [])
    targets: dict[tuple[str, str], int] = {}
    for row in load_jsonl(args.source):
        if FAILURE_MARKER not in str(row.get("failure_reason") or ""):
            continue
        method, episode_id, stage = row_key(row)
        if methods and method not in methods:
            continue
        if episodes and episode_id not in episodes:
            continue
        targets[(method, episode_id)] = stage
    jobs = [
        {
            "job_id": f"{method}__{episode_id}",
            "method": method,
            "episode_id": episode_id,
            "failed_stage": stage,
            "run_dir": str(args.output_root / "jobs" / method / episode_id),
        }
        for (method, episode_id), stage in sorted(targets.items())
    ]
    return jobs[: args.max_jobs] if args.max_jobs is not None else jobs


def run_record(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["run_dir"]) / "NLDO" / f"{job['method']}_limit0.jsonl"
    rows = load_jsonl(path)
    return rows[-1] if rows else {}


def job_is_terminal(job: dict[str, Any]) -> bool:
    return str(run_record(job).get("run_status") or "") in TERMINAL_STATUSES


def runner_command(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    command = [
        args.python,
        str(RUNNER),
        "--provider",
        "kimi",
        "--model",
        args.model,
        "--episode-id",
        str(job["episode_id"]),
        "--methods",
        str(job["method"]),
        "--out-dir",
        str(job["run_dir"]),
        "--cache-dir",
        "outputs/kimi_cache",
        "--use-response-cache",
        "--response-protocol",
        "json",
        "--max-tokens",
        "64000",
        "--max-attempts",
        "2",
        "--code-repair-attempts",
        "3",
        "--total-timeout",
        "3600",
        "--stop-after-hidden-rejection",
        "--prefix-stage-rows",
        str(args.source),
    ]
    for path in args.overlay_rows:
        command.extend(["--prefix-stage-rows", str(path)])
    if args.cache_only:
        command.append("--cache-only")
    if args.resume or (Path(job["run_dir"]) / "dynamic_public_baseline_summary.json").exists():
        command.append("--resume")
        prior_status = str(run_record(job).get("run_status") or "")
        if prior_status and prior_status not in TERMINAL_STATUSES:
            command.extend(["--retry-statuses", prior_status])
    return command


def run_job(
    args: argparse.Namespace,
    job: dict[str, Any],
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        return {**job, "status": "deferred_after_infrastructure_limit", "return_code": None}
    if job_is_terminal(job):
        return {**job, "status": "reused", "return_code": 0}
    env = dict(os.environ)
    env.update(
        {
            "LLM_PROVIDER": "kimi",
            "KIMI_MODEL": args.model,
            "KIMI_CACHE": "1",
            "KIMI_MIN_GENERATION_TOKENS": "32000",
            "KIMI_MAX_GENERATION_TOKENS": "98304",
            "KIMI_REASONING_EFFORT": "medium",
            "KIMI_STREAM": "1",
            "KIMI_STREAM_INCLUDE_USAGE": "1",
            "KIMI_USE_SYSTEM_PROXY": "1" if args.use_system_proxy else "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(runner_command(args, job), cwd=ROOT, env=env, check=False)
    record = run_record(job)
    status = str(record.get("run_status") or "failed")
    if status in {"infrastructure_limited", "quota_limited"}:
        stop_event.set()
    return {
        **job,
        "status": status,
        "return_code": int(completed.returncode),
        "reached_stages": len(record.get("stage_traces") or []),
        "next_stage_index": record.get("next_stage_index"),
    }


def merge_rows(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> Path:
    source_rows = {row_key(row): row for row in load_jsonl(args.source)}
    merged = dict(source_rows)
    for path in args.overlay_rows:
        merged.update({row_key(row): row for row in load_jsonl(path)})
    for job in jobs:
        if not job_is_terminal(job):
            continue
        path = Path(job["run_dir"]) / "dynamic_public_baseline_stage_rows.jsonl"
        for row in load_jsonl(path):
            key = row_key(row)
            source_row = source_rows.get(key)
            if (
                source_row is not None
                and FAILURE_MARKER in str(source_row.get("failure_reason") or "")
                and not bool(row.get("prior_generation_limit_usage_included"))
            ):
                prior_usage = dict(source_row.get("token_usage") or {})
                row["prior_generation_limit_usage"] = prior_usage
                row["prior_generation_limit_usage_included"] = True
                row["token_usage"] = add_token_usage(
                    prior_usage,
                    dict(row.get("token_usage") or {}),
                )
            merged[key] = row
    rows = sorted(merged.values(), key=row_key)
    output = args.output_root / "merged" / "dynamic_public_baseline_stage_rows.jsonl"
    atomic_jsonl(output, rows)
    return output


def summarize_rows(path: Path, source: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(path):
        method, episode_id, stage = row_key(row)
        grouped[(method, episode_id)][stage] = row
    methods: dict[str, Any] = {}
    for method in sorted({key[0] for key in grouped}):
        episodes = sorted(key[1] for key in grouped if key[0] == method)
        feasible_states = 0
        quality_sum = 0.0
        reached = 0
        total_tokens = 0
        complete = 0
        terminal = 0
        failures: Counter[str] = Counter()
        views: dict[str, dict[str, Any]] = {}
        for view, selected in (
            ("DS", [ep for ep in episodes if int(ep.rsplit("P", 1)[1]) <= 9]),
            ("DM", [ep for ep in episodes if int(ep.rsplit("P", 1)[1]) >= 10]),
        ):
            view_feasible = 0
            view_quality = 0.0
            for episode_id in selected:
                stages = grouped[(method, episode_id)]
                for stage in range(STAGES_PER_EPISODE):
                    row = stages.get(stage)
                    if row is None or not bool((row.get("hidden_evaluation") or {}).get("feasible")):
                        break
                    view_feasible += 1
                    metric = row.get("hidden_evaluation") or {}
                    view_quality += float(metric.get("normalized_score") or 0.0)
            denominator = len(selected) * STAGES_PER_EPISODE
            views[view] = {
                "trajectories": len(selected),
                "denominator_states": denominator,
                "feasible_states": view_feasible,
                "solve_rate": view_feasible / denominator if denominator else 0.0,
                "online_quality": view_quality / denominator if denominator else 0.0,
            }
            feasible_states += view_feasible
            quality_sum += view_quality
        for episode_id in episodes:
            stages = grouped[(method, episode_id)]
            ordered = [stages[index] for index in sorted(stages)]
            reached += len(ordered)
            total_tokens += sum(
                int((row.get("token_usage") or {}).get("total_tokens") or 0)
                for row in ordered
            )
            if len(ordered) == STAGES_PER_EPISODE and all(
                bool((row.get("hidden_evaluation") or {}).get("feasible"))
                for row in ordered
            ):
                complete += 1
                terminal += 1
            else:
                first_failure = next(
                    (
                        row
                        for row in ordered
                        if not bool((row.get("hidden_evaluation") or {}).get("feasible"))
                    ),
                    None,
                )
                if first_failure is not None:
                    terminal += 1
                    failures[str(first_failure.get("status") or "failed")] += 1
        methods[method] = {
            "episodes": len(episodes),
            "reached_stages": reached,
            "hidden_feasible_stages": feasible_states,
            "total_tokens": total_tokens,
            "mean_tokens_per_reached_stage": total_tokens / reached if reached else None,
            "complete_sequences": complete,
            "stopped_sequences": terminal - complete,
            "all_episode_terminal_evidence": terminal == len(episodes),
            "failure_status_counts": dict(sorted(failures.items())),
            "views": views,
        }
    return {
        "status": (
            "completed"
            if methods and all(item["all_episode_terminal_evidence"] for item in methods.values())
            else "partial"
        ),
        "provider": "kimi",
        "model": "k2.7",
        "protocol": "common_public_lsm_own_history_first_failure_zero_suffix_v1",
        "generation_policy": {
            "reasoning_effort": "medium",
            "known_truncation_retry_start": 64000,
            "maximum_generation_tokens": 98304,
        },
        "source": str(source),
        "merged_stage_rows": sum(len(stages) for stages in grouped.values()),
        "methods": methods,
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = target_jobs(args)
    manifest_path = args.output_root / "manifest.json"
    manifest = load_json(manifest_path) if args.resume else {}
    manifest.update(
        {
            "status": "running",
            "source": str(args.source),
            "failure_marker": FAILURE_MARKER,
            "selected_job_count": len(jobs),
            "workers": args.workers,
            "generation_policy": {
                "reasoning_effort": "medium",
                "retry_start_tokens": 64000,
                "maximum_tokens": 98304,
            },
            "started_at": manifest.get("started_at") or time.time(),
        }
    )
    atomic_json(manifest_path, manifest)
    results: list[dict[str, Any]] = []
    stop_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(run_job, args, job, stop_event): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            manifest["jobs"] = sorted(results, key=lambda item: item["job_id"])
            manifest["updated_at"] = time.time()
            atomic_json(manifest_path, manifest)
    merged = merge_rows(args, jobs)
    summary = summarize_rows(merged, args.source)
    summary_path = args.output_root / "dynamic_public_baseline_summary.json"
    atomic_json(summary_path, summary)
    manifest.update(
        {
            "status": summary["status"],
            "merged_stage_rows": str(merged),
            "summary": str(summary_path),
            "updated_at": time.time(),
        }
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
