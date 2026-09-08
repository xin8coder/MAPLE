#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.agents.deepseek_client import load_dotenv
from evo2.agents.liveopt_workbench_impl import LiveOptWorkbenchGenerator
from evo2.core.template_optimizer import EvolutionConfig


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or output_dir / "cache")
    trace_dir = Path(args.trace_dir or output_dir / "trace")
    summary_path = Path(args.summary_json or output_dir / "summary.json")
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    episodes = [_load_episode(args.episodes_jsonl, episode_id) for episode_id in args.episode_id]

    old_env = _push_env(
        {
            "DEEPSEEK_CACHE": "1",
            "DEEPSEEK_CACHE_DIR": str(cache_dir),
            "DEEPSEEK_TRACE_DIR": str(trace_dir),
            "DEEPSEEK_CACHE_ONLY": "1" if args.cache_only else None,
            "DEEPSEEK_MODEL": args.model if args.model else None,
            "DEEPSEEK_MAX_TOKENS": str(args.max_tokens) if args.max_tokens else None,
            "DEEPSEEK_MAX_ATTEMPTS": str(args.max_attempts),
            "DEEPSEEK_TOTAL_TIMEOUT": str(args.total_timeout) if args.total_timeout else None,
        }
    )
    started = time.perf_counter()
    rows = []
    for episode in episodes:
        rows.append(_run_one(args, episode))
    record = {
        "status": "completed" if all(row["status"] == "completed" for row in rows) else "partial",
        "mode": "cache_only" if args.cache_only else "provider_backed",
        "protocol": LiveOptWorkbenchGenerator.prompt_version,
        "rows": rows,
        "summary_json": str(summary_path),
        "cache_dir": str(cache_dir),
        "trace_dir": str(trace_dir),
        "latency_seconds": time.perf_counter() - started,
        "deepseek_trace": _trace_summary(trace_dir),
    }
    summary_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _pop_env(old_env)
    print(json.dumps(_console_summary(record), ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_completed and record["status"] != "completed":
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LiveOpt Workbench generation on selected NLDO benchmark episodes.")
    parser.add_argument("--episodes-jsonl", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_10updates_csv.jsonl")
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/liveopt_workbench_benchmark_micro/latest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--summary-json")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL"))
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=6500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--total-timeout", type=float, default=240.0)
    parser.add_argument("--population-size", type=int, default=60)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--archive-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def _run_one(args: argparse.Namespace, episode: dict[str, Any]) -> dict[str, Any]:
    episode_id = episode["episode_id"]
    started = time.perf_counter()
    public_context = _public_context_with_loaded_tables(episode)
    problem = episode["public_initial_problem"]
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "status": "started",
        "problem": problem,
        "public_context_summary": _public_context_summary(public_context),
        "evaluation_metrics": (episode.get("evaluation") or {}).get("metrics", []),
    }
    try:
        generator = LiveOptWorkbenchGenerator(model=args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
        project = generator.generate_project(episode_id, problem, public_context)
        result = project.run(
            EvolutionConfig(
                population_size=args.population_size,
                generations=args.generations,
                seed=args.seed,
                archive_limit=args.archive_limit,
            )
        )
        best = result.best.result
        row.update(
            {
                "status": "completed",
                "failure_reason": "",
                "setup_code": project.setup_code,
                "fitness_code": project.fitness_code,
                "segments": [_jsonable_value(segment.__dict__) for segment in project.segments],
                "data_keys": sorted(project.data),
                "problem_spec": _jsonable_problem_spec(project.problem_spec),
                "agent_verification": _jsonable_value({"feasible": bool(best.feasible), "violations": best.violations} if best else {"feasible": False}),
                "agent_objective_raw": _fitness_record(best),
                "history": _jsonable_value(result.history),
                "archive": [_candidate_record(candidate) for candidate in result.archive],
                "archive_count": len(result.archive),
                "population_count": len(result.population),
                "generation_trace": {
                    "model": project.trace.model,
                    "prompts": list(project.trace.prompts),
                    "raw_responses": list(project.trace.raw_responses),
                    "usage": list(project.trace.usage),
                    "errors": list(project.trace.errors),
                    "latency_seconds": project.trace.latency_seconds,
                },
                "token_usage": _sum_usage(project.trace.usage),
            }
        )
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        row.update(
            {
                "status": "cache_miss_blocked" if _looks_like_cache_miss(text) else "failed",
                "failure_reason": text,
                "traceback": traceback.format_exc(limit=12),
            }
        )
    row["latency_seconds"] = time.perf_counter() - started
    return row


def _load_episode(path: str, episode_id: str) -> dict[str, Any]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode.get("episode_id") == episode_id or episode.get("base_instance_id") == episode_id:
            return episode
    raise ValueError(f"episode not found: {episode_id}")


def _public_context_with_loaded_tables(episode: dict[str, Any]) -> dict[str, Any]:
    public_context = dict(episode.get("public_context") or {})
    tables = {}
    for name, meta in (public_context.get("csv_tables") or {}).items():
        path = Path(meta["path"])
        tables[name] = _read_csv_rows(path)
    public_context["tables"] = tables
    return public_context


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _coerce_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _coerce_cell(value: str) -> Any:
    text = str(value).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _public_context_summary(public_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "multi_objective": public_context.get("multi_objective"),
        "objective_sense": public_context.get("objective_sense"),
        "tables": {name: {"rows": len(rows), "columns": list(rows[0]) if rows else []} for name, rows in public_context.get("tables", {}).items()},
    }


def _jsonable_problem_spec(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    if isinstance(out.get("data"), dict):
        out["data_summary"] = {key: _summarize_value(value) for key, value in out["data"].items()}
        out.pop("data", None)
    return _jsonable_value(out)


def _summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:3]}
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "sample_keys": list(value)[:5]}
    return value


def _candidate_record(candidate) -> dict[str, Any]:
    return _jsonable_value({"genome": candidate.genome, "rank": candidate.rank, "fitness": _fitness_record(candidate.result)})


def _fitness_record(result) -> dict[str, Any]:
    if result is None:
        return {}
    return _jsonable_value({
        "scalar": result.scalar,
        "objectives": list(result.objectives),
        "feasible": result.feasible,
        "violations": result.violations,
        "base_scalar": result.base_scalar,
        "penalty": result.penalty,
        "solution": result.solution,
        "diagnostics": result.diagnostics,
    })


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sum_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value
    return total


def _trace_summary(trace_dir: Path) -> dict[str, Any]:
    candidate_paths = [trace_dir / "deepseek_calls.jsonl", trace_dir / "kimi_calls.jsonl"]
    existing_paths = [path for path in candidate_paths if path.exists()]
    rows = [
        json.loads(line)
        for path in existing_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events: dict[str, int] = {}
    token_usage: dict[str, int] = {}
    cache_paths = []
    for row in rows:
        event = str(row.get("event", "unknown"))
        events[event] = events.get(event, 0) + 1
        if row.get("cache_path"):
            cache_paths.append(row["cache_path"])
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        for key, value in usage.items():
            if isinstance(value, int):
                token_usage[key] = token_usage.get(key, 0) + value
    return {
        "trace_file": str(existing_paths[0] if existing_paths else candidate_paths[0]),
        "trace_files": [str(path) for path in existing_paths],
        "call_count": len(rows),
        "events": events,
        "cache_paths": sorted(set(cache_paths)),
        "token_usage": token_usage,
    }


def _console_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record.get("status"),
        "mode": record.get("mode"),
        "protocol": record.get("protocol"),
        "summary_json": record.get("summary_json"),
        "trace_events": (record.get("deepseek_trace") or {}).get("events", {}),
        "rows": [
            {
                "episode_id": row.get("episode_id"),
                "status": row.get("status"),
                "segments": [segment.get("kind") for segment in row.get("segments", []) if isinstance(segment, dict)],
                "feasible": (row.get("agent_verification") or {}).get("feasible"),
                "objective": (row.get("agent_objective_raw") or {}).get("scalar"),
                "objectives": (row.get("agent_objective_raw") or {}).get("objectives"),
                "archive_count": row.get("archive_count", 0),
                "token_usage": row.get("token_usage", {}),
                "failure_reason": row.get("failure_reason", ""),
            }
            for row in record.get("rows", [])
        ],
    }


def _looks_like_cache_miss(text: str) -> bool:
    lowered = text.lower()
    return "cache_only" in lowered or "local response cache missed" in lowered or "cache miss" in lowered


def _push_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return old


def _pop_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
