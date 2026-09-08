#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
from evo2.agents.llm_client import (
    ChatClient,
    create_llm_client,
    infer_llm_provider,
    is_llm_infrastructure_error,
    is_llm_quota_limit_error,
    llm_infrastructure_error_record,
    llm_quota_limit_record,
)
from scripts.analysis import evaluate_reference_metrics as reference_metrics
from scripts.baselines.run_static_calibration import (
    build_messages,
    execute_and_repair_solver_code,
    extract_solver_code,
    normalize_usage,
    parse_model_response,
    sum_usage,
    validate_parsed_output,
)


DEFAULT_METHODS = "react_tools,optimai_2025,or_llm_agent_2025"
PROMPT_VERSION = "nldo_dynamic_common_public_lsm_v6"
LSM_POPULATION_LIMIT = 200
QUOTA_EXIT_CODE = 75
INFRASTRUCTURE_EXIT_CODE = 76


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "traces"
    nldo_dir = out_dir / "NLDO"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    nldo_dir.mkdir(parents=True, exist_ok=True)
    resolve_provider_and_model(args)
    configure_cache_env(args, out_dir)

    episodes = select_episodes(load_jsonl(Path(args.episodes_jsonl)), args)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    existing = load_existing_runs(nldo_dir, methods) if args.resume else {}
    client = create_llm_client(model=args.model, provider=args.provider)
    started = time.perf_counter()
    stage_rows_path = out_dir / "dynamic_public_baseline_stage_rows.jsonl"
    if not args.resume and stage_rows_path.exists():
        stage_rows_path.unlink()
    prefix_paths = [Path(path) for path in (args.prefix_stage_rows or [])]
    if args.resume and stage_rows_path.exists():
        prefix_paths.append(stage_rows_path)
    saved_prefixes = load_feasible_stage_prefixes(prefix_paths)

    method_runs: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    stage_rows_written = 0
    quota_limit: dict[str, Any] = {}
    infrastructure_limit: dict[str, Any] = {}
    paused_job: dict[str, Any] | None = None
    for method in methods:
        replay_path = nldo_dir / f"{method}_limit0.jsonl"
        for episode in episodes:
            episode_id = str(episode.get("episode_id") or "")
            key = f"{method}::{episode_id}"
            if key in existing and not should_retry_existing(existing[key], args):
                method_runs[method].append(existing[key])
                continue
            run, stage_rows = run_episode_method(
                args,
                client,
                episode,
                method,
                trace_dir,
                prefix_rows=saved_prefixes.get((method, episode_id), []),
            )
            replace_episode_run(replay_path, run)
            replace_stage_rows(stage_rows_path, method, episode_id, stage_rows)
            method_runs[method].append(run)
            stage_rows_written += len(stage_rows)
            if run.get("run_status") == "quota_limited":
                quota_limit = dict(run.get("quota_limit") or {})
                paused_job = {
                    "method": method,
                    "episode_id": episode_id,
                    "next_stage_index": run.get("next_stage_index"),
                }
                break
            if run.get("run_status") == "infrastructure_limited":
                infrastructure_limit = dict(run.get("infrastructure_limit") or {})
                paused_job = {
                    "method": method,
                    "episode_id": episode_id,
                    "next_stage_index": run.get("next_stage_index"),
                }
                break
            if args.stop_after_new and stage_rows_written >= args.stop_after_new:
                break
        if (
            quota_limit
            or infrastructure_limit
            or (args.stop_after_new and stage_rows_written >= args.stop_after_new)
        ):
            break

    summary = summarize(method_runs, args, started, stage_rows_path, nldo_dir, trace_dir)
    if quota_limit:
        summary.update(
            {
                "status": "quota_limited",
                "quota_limit": quota_limit,
                "paused_job": paused_job,
                "resumable": True,
                "resume": "rerun the same command with --resume after the quota resets",
            }
        )
    elif infrastructure_limit:
        summary.update(
            {
                "status": "infrastructure_limited",
                "infrastructure_limit": infrastructure_limit,
                "paused_job": paused_job,
                "resumable": True,
                "resume": "rerun the same command with --resume; no zero score was recorded",
            }
        )
    summary_path = out_dir / "dynamic_public_baseline_summary.json"
    summary["summary_json"] = str(summary_path)
    atomic_write_json(summary_path, summary)
    print(json.dumps(console_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))
    if quota_limit:
        return QUOTA_EXIT_CODE
    if infrastructure_limit:
        return INFRASTRUCTURE_EXIT_CODE
    if args.require_completed and summary["status"] != "completed":
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run current NLDO public dynamic baselines without legacy IR scaffolds.")
    parser.add_argument("--episodes-jsonl", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--provider", choices=("deepseek", "kimi"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", default="logs/llm_tests/nldo_dynamic_public_baselines/latest")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--use-response-cache", dest="use_response_cache", action="store_true", default=True)
    parser.add_argument("--no-response-cache", dest="use_response_cache", action="store_false")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prefix-stage-rows",
        type=Path,
        action="append",
        help=(
            "Previously saved stage JSONL used only as a contiguous hidden-feasible prefix. "
            "The next missing stage is called with this method's own accepted output and archive."
        ),
    )
    parser.add_argument("--retry-cache-misses", action="store_true")
    parser.add_argument("--retry-statuses", default="")
    parser.add_argument("--response-protocol", choices=("json", "code_block"), default="json")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--code-repair-attempts",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CODE_REPAIR_ATTEMPTS", "3") or 3),
        help=(
            "Equal public code-output repair budget per reached stage. "
            "Repairs only compile/runtime/stdout-format failures and receives no hidden/reference feedback."
        ),
    )
    parser.add_argument("--total-timeout", type=float, default=360.0)
    parser.add_argument("--stop-after-new", type=int, default=0)
    parser.add_argument(
        "--continue-after-hidden-rejection",
        dest="continue_after_hidden_rejection",
        action="store_true",
        default=False,
        help="Diagnostic override: continue the public trajectory after a hidden scoring rejection without exposing hidden feedback.",
    )
    parser.add_argument(
        "--stop-after-hidden-rejection",
        dest="continue_after_hidden_rejection",
        action="store_false",
        help="Stop a trajectory at its first hidden-infeasible stage (default).",
    )
    parser.add_argument("--require-completed", action="store_true")
    return parser.parse_args()


def resolve_provider_and_model(args: argparse.Namespace) -> None:
    provider = infer_llm_provider(model=args.model, provider=args.provider)
    if not args.model:
        args.model = (
            os.getenv("KIMI_MODEL", "k3[1m]")
            if provider == "kimi"
            else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        )
    args.provider = provider


def configure_cache_env(args: argparse.Namespace, out_dir: Path) -> None:
    enabled = "1" if args.use_response_cache else "0"
    if args.provider == "kimi":
        os.environ["KIMI_CACHE"] = enabled
        os.environ["KIMI_CACHE_DIR"] = str(args.cache_dir or "outputs/kimi_cache")
        os.environ["KIMI_TRACE_DIR"] = str(out_dir / "provider_trace")
        os.environ["KIMI_MAX_RETRIES"] = str(args.max_attempts)
        os.environ["KIMI_TIMEOUT"] = str(max(1, int(args.total_timeout)))
        if args.cache_only:
            os.environ["KIMI_CACHE_ONLY"] = "1"
            os.environ.setdefault("KIMI_API_KEY", "cache-only-dummy-key")
        else:
            os.environ.pop("KIMI_CACHE_ONLY", None)
        return

    os.environ["DEEPSEEK_CACHE"] = enabled
    os.environ["DEEPSEEK_CACHE_DIR"] = str(args.cache_dir or "outputs/deepseek_cache")
    if args.cache_only:
        os.environ["DEEPSEEK_CACHE_ONLY"] = "1"
        os.environ.setdefault("DEEPSEEK_API_KEY", "cache-only-dummy-key")
    else:
        os.environ.pop("DEEPSEEK_CACHE_ONLY", None)
    os.environ["DEEPSEEK_MAX_ATTEMPTS"] = str(args.max_attempts)
    os.environ["DEEPSEEK_TOTAL_TIMEOUT"] = str(args.total_timeout)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_episodes(episodes: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(episodes)
    if args.episode_id:
        wanted = set(args.episode_id)
        selected = [
            episode
            for episode in selected
            if str(episode.get("episode_id")) in wanted or str(episode.get("base_instance_id")) in wanted
        ]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    return selected


def load_existing_runs(nldo_dir: Path, methods: list[str]) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for method in methods:
        path = nldo_dir / f"{method}_limit0.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            existing[f"{method}::{row.get('episode_id')}"] = row
    return existing


def load_feasible_stage_prefixes(
    paths: list[Path],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            method = str(row.get("method") or "")
            episode_id = str(row.get("episode_id") or "")
            stage = int(row.get("stage_index") or 0)
            if method and episode_id and 0 <= stage <= 12:
                merged[(method, episode_id, stage)] = row

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    run_keys = sorted({(method, episode_id) for method, episode_id, _ in merged})
    for method, episode_id in run_keys:
        prefix: list[dict[str, Any]] = []
        for stage in range(13):
            row = merged.get((method, episode_id, stage))
            if row is None or not bool((row.get("hidden_evaluation") or {}).get("feasible")):
                break
            prefix.append(row)
        if prefix:
            grouped[(method, episode_id)] = prefix
    return grouped


def should_retry_existing(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.resume and str(row.get("run_status") or row.get("status") or "") in {
        "quota_limited",
        "infrastructure_limited",
    }:
        return True
    statuses = {item.strip() for item in str(args.retry_statuses or "").split(",") if item.strip()}
    run_status = str(row.get("run_status") or row.get("status") or "")
    if statuses and run_status in statuses:
        return True
    if args.retry_cache_misses:
        return any(
            str(stage.get("status") or "") == "cache_miss_blocked"
            for stage in row.get("stage_traces", []) or []
        )
    return False


def run_episode_method(
    args: argparse.Namespace,
    client: ChatClient,
    episode: dict[str, Any],
    method: str,
    trace_dir: Path,
    prefix_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = str(episode.get("episode_id"))
    public_context = public_context_with_loaded_tables(episode)
    updates = list(episode.get("update_stream") or [])
    if args.max_updates is not None:
        updates = updates[: max(0, int(args.max_updates))]
    stage_inputs = [{"update_id": "initial", "stage_index": 0, "updates": []}]
    for index, update in enumerate(updates, start=1):
        stage_inputs.append(
            {
                "update_id": str(update.get("update_id") or f"t{index:03d}"),
                "stage_index": index,
                "updates": updates[:index],
            }
        )
    expected_stage_count = len(stage_inputs)

    run = {
        "episode_id": episode_id,
        "base_benchmark": "NLDO",
        "domain": episode.get("domain"),
        "family": episode.get("family"),
        "base_instance_id": episode.get("base_instance_id"),
        "run_seed": 0,
        "agent_mode": f"{method}_dynamic_public",
        "restart_policy": "common_public_lsm_own_history_v1",
        "state_interface": "common_public_lsm_dialogue_and_own_population_v1",
        "model": args.model,
        "budget": {
            "llm_calls": "one public code-solving request per reached stage",
            "code_repair_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
            "code_repair_policy": "equal_public_code_output_repair_v1",
            "stop_after_hidden_rejection": not args.continue_after_hidden_rejection,
        },
        "initial_candidate": {},
        "initial_feasible": False,
        "initial_objective": None,
        "initial_solver_result": {"success": False, "metadata": {"candidate_archive": []}},
        "update_results": [],
        "token_usage": zero_usage(),
        "latency_seconds": 0.0,
        "stage_traces": [],
        "run_status": "started",
    }
    stage_rows: list[dict[str, Any]] = []
    previous_public_solution: dict[str, Any] | None = None
    previous_scoring_solution: dict[str, Any] | None = None
    previous_artifact: dict[str, Any] | None = None
    previous_final_answer: dict[str, Any] | None = None
    previous_candidate_population: list[dict[str, Any]] = []
    prefix_rows = sorted(
        list(prefix_rows or []),
        key=lambda row: int(row.get("stage_index") or 0),
    )
    if prefix_rows:
        stage_rows = list(prefix_rows)
        for expected_stage, row in enumerate(prefix_rows):
            if int(row.get("stage_index") or 0) != expected_stage:
                raise ValueError(
                    f"{method} {episode_id}: non-contiguous prefix at stage {expected_stage}"
                )
            if not bool((row.get("hidden_evaluation") or {}).get("feasible")):
                raise ValueError(
                    f"{method} {episode_id}: prefix stage {expected_stage} is not hidden-feasible"
                )
        run["stage_traces"] = list(prefix_rows)
        for row in prefix_rows:
            usage = row.get("token_usage") if isinstance(row.get("token_usage"), dict) else {}
            run["token_usage"] = add_usage(run["token_usage"], usage)
            run["latency_seconds"] += float(row.get("latency_seconds") or 0.0)
            metric = row.get("hidden_evaluation") if isinstance(row.get("hidden_evaluation"), dict) else {}
            solution = row.get("solution") if isinstance(row.get("solution"), dict) else {}
            stage_index = int(row.get("stage_index") or 0)
            if stage_index == 0:
                run["initial_candidate"] = solution
                run["initial_feasible"] = True
                run["initial_objective"] = metric.get("objective")
                run["initial_solver_result"] = solver_result_from_stage(row)
            elif stage_index <= len(updates):
                run["update_results"].append(
                    update_result_from_stage(row, updates[stage_index - 1])
                )
            previous_public_solution = solution
            previous_artifact = accepted_public_artifact(row)
            previous_final_answer = accepted_public_final_answer(row)
            if method != "persistent_react":
                previous_candidate_population = list(row.get("candidate_archive") or [])[
                    :LSM_POPULATION_LIMIT
                ]
            previous_scoring_solution = (
                metric.get("canonical_solution")
                if isinstance(metric.get("canonical_solution"), dict)
                else solution
            )
        run["resumed_from_saved_prefix"] = True
        run["saved_prefix_last_stage"] = int(prefix_rows[-1].get("stage_index") or 0)
        stage_inputs = [
            stage
            for stage in stage_inputs
            if int(stage["stage_index"]) > int(prefix_rows[-1].get("stage_index") or 0)
        ]
    stopped = False
    for stage in stage_inputs:
        if stopped:
            break
        try:
            stage_row = run_stage_method(
                args,
                client,
                episode,
                public_context,
                method,
                stage,
                trace_dir,
                previous_public_solution,
                previous_artifact,
                previous_candidate_population,
                previous_scoring_solution,
                previous_final_answer=previous_final_answer,
            )
        except Exception as exc:  # noqa: BLE001
            quota_limited = is_llm_quota_limit_error(exc)
            infrastructure_limited = is_llm_infrastructure_error(exc)
            if not quota_limited and not infrastructure_limited:
                raise
            run_status = (
                "quota_limited" if quota_limited else "infrastructure_limited"
            )
            run.update(
                {
                    "run_status": run_status,
                    "resumable": True,
                    "accepted_prefix_updates": max(
                        0,
                        sum(
                            int((row.get("hidden_evaluation") or {}).get("feasible") is True)
                            for row in run["stage_traces"]
                            if int(row.get("stage_index") or 0) > 0
                        ),
                    ),
                    "last_completed_stage_index": (
                        int(run["stage_traces"][-1].get("stage_index") or 0)
                        if run["stage_traces"]
                        else None
                    ),
                    "next_stage_index": int(stage["stage_index"]),
                    "next_update_id": str(stage["update_id"]),
                }
            )
            if quota_limited:
                run["quota_limit"] = llm_quota_limit_record(exc)
            else:
                run["infrastructure_limit"] = llm_infrastructure_error_record(exc)
            return run, stage_rows
        stage_rows.append(stage_row)
        run["stage_traces"].append(stage_row)
        usage = stage_row.get("token_usage") if isinstance(stage_row.get("token_usage"), dict) else {}
        run["token_usage"] = add_usage(run["token_usage"], usage)
        run["latency_seconds"] = float(run.get("latency_seconds") or 0.0) + float(stage_row.get("latency_seconds") or 0.0)
        metric = stage_row.get("hidden_evaluation") if isinstance(stage_row.get("hidden_evaluation"), dict) else {}
        solution = stage_row.get("solution") if isinstance(stage_row.get("solution"), dict) else {}
        if int(stage["stage_index"]) == 0:
            run["initial_candidate"] = solution
            run["initial_feasible"] = bool(metric.get("feasible"))
            run["initial_objective"] = metric.get("objective")
            run["initial_solver_result"] = solver_result_from_stage(stage_row)
        else:
            run["update_results"].append(update_result_from_stage(stage_row, stage["updates"][-1]))
        if stage_row.get("public_state_available"):
            previous_public_solution = solution
            previous_artifact = accepted_public_artifact(stage_row)
            previous_final_answer = accepted_public_final_answer(stage_row)
            if method != "persistent_react":
                previous_candidate_population = list(stage_row.get("candidate_archive") or [])[:LSM_POPULATION_LIMIT]
        if metric.get("feasible"):
            previous_scoring_solution = (
                metric.get("canonical_solution")
                if isinstance(metric.get("canonical_solution"), dict)
                else solution
            )
        elif not args.continue_after_hidden_rejection:
            stopped = True
    run["run_status"] = (
        "completed"
        if len(run["stage_traces"]) == expected_stage_count
        else "stopped_after_failure"
    )
    return run, stage_rows


def run_stage_method(
    args: argparse.Namespace,
    client: ChatClient,
    episode: dict[str, Any],
    public_context: dict[str, Any],
    method: str,
    stage: dict[str, Any],
    trace_dir: Path,
    previous_solution: dict[str, Any] | None,
    previous_artifact: dict[str, Any] | None = None,
    previous_candidate_population: list[dict[str, Any]] | None = None,
    previous_scoring_solution: dict[str, Any] | None = None,
    previous_final_answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    case = dynamic_stage_case(
        episode,
        public_context,
        stage,
        method,
        previous_solution,
        previous_artifact,
        previous_candidate_population,
        previous_final_answer=previous_final_answer,
    )
    messages = build_messages(case, method, response_protocol=args.response_protocol)
    raw_response = ""
    parsed: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    validation_feedback: list[str] = []
    code_result: dict[str, Any] = {}
    repair_trace: list[dict[str, Any]] = []
    solution: dict[str, Any] = {}
    candidate_archive: list[dict[str, Any]] = []
    status = "failed"
    failure_reason = ""
    try:
        response = client.chat(messages, temperature=0.0, max_tokens=args.max_tokens, json_mode=not args.no_json_mode)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        raw_response = response.get("choices", [{}])[0].get("message", {}).get("content", "") if isinstance(response, dict) else ""
        parsed = parse_model_response(raw_response, method, protocol=args.response_protocol)
        validation_feedback = validate_parsed_output(method, parsed)
        parsed, code_result, repair_trace = execute_and_repair_solver_code(
            case=case,
            method=method,
            parsed=parsed,
            client=client,
            args=args,
            base_messages=messages,
            initial_validation_feedback=validation_feedback,
        )
        if repair_trace:
            usage = sum_usage([usage, *(attempt.get("token_usage", {}) for attempt in repair_trace)])
        validation_feedback = validate_parsed_output(method, parsed)
        output = code_result.get("output") if isinstance(code_result.get("output"), dict) else {}
        solution = extract_solution(output, parsed)
        candidate_archive = extract_candidate_archive(output, parsed, solution)
        status = "completed" if not validation_feedback else "schema_failed"
        if not code_result.get("compile_success"):
            status = "compile_failed"
            failure_reason = str(code_result.get("compile_error") or "compile_failed")
        elif not code_result.get("runtime_success"):
            status = "runtime_failed"
            failure_reason = str(code_result.get("runtime_error") or "runtime_failed")
        elif validation_feedback:
            failure_reason = "; ".join(validation_feedback)
    except Exception as exc:  # noqa: BLE001
        if is_llm_quota_limit_error(exc) or is_llm_infrastructure_error(exc):
            raise
        exception_usage = getattr(exc, "usage", None)
        if isinstance(exception_usage, dict):
            usage = exception_usage
        exception_response = getattr(exc, "response", None)
        if isinstance(exception_response, dict):
            raw_response = (
                exception_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        failure_reason = str(exc)
        status = (
            "cache_miss_blocked"
            if "CACHE_ONLY" in failure_reason or "local response cache missed" in failure_reason
            else "failed"
        )
        validation_feedback = [failure_reason]
    can_score = bool(code_result.get("compile_success")) and bool(code_result.get("runtime_success")) and bool(solution)
    hidden = (
        hidden_metric_for_stage(
            episode,
            int(stage["stage_index"]),
            str(stage["update_id"]),
            solution,
            previous_scoring_solution,
            candidate_archive,
        )
        if can_score
        else {
            "feasible": False,
            "normalized_score": 0.0,
            "not_scored_reason": status,
        }
    )
    if status == "completed" and not hidden.get("feasible"):
        status = "hidden_rejected"
        failure_reason = "hidden evaluator rejected candidate"
    latency = time.perf_counter() - started
    trace_path = write_stage_trace(
        trace_dir,
        case,
        method,
        messages,
        raw_response,
        parsed,
        usage,
        status,
        failure_reason,
        code_result,
        hidden,
        repair_trace,
        int(getattr(args, "code_repair_attempts", 0) or 0),
    )
    row = {
        "schema_version": "nldo_dynamic_public_baseline_stage_v1",
        "prompt_version": PROMPT_VERSION,
        "episode_id": episode.get("episode_id"),
        "base_instance_id": episode.get("base_instance_id"),
        "domain": episode.get("domain"),
        "family": episode.get("family"),
        "method": method,
        "model": args.model,
        "stage_index": int(stage["stage_index"]),
        "update_id": str(stage["update_id"]),
        "status": status,
        "failure_reason": failure_reason,
        "messages": messages,
        "selected_stage_prompt": messages[-1]["content"] if messages else "",
        "raw_response": raw_response,
        "parsed_output": parsed,
        "solver_code_available": bool(extract_solver_code(parsed)),
        "persistent_public_artifact_input": bool(previous_artifact),
        "public_state_available": can_score,
        "shared_lsm_input": {
            "interface": "common_public_lsm_dialogue_and_own_population_v1",
            "previous_candidate_count": len(previous_candidate_population or []),
            "population_limit": LSM_POPULATION_LIMIT,
            "state_advance_gate": "publicly_executable_output_only",
            "hidden_evaluation_controls_state": False,
            "inherits_liveopt_population": False,
        },
        "code_execution": code_result,
        "code_output_repair": {
            "policy": "equal_public_code_output_repair_v1",
            "max_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
            "attempts_used": len(repair_trace),
            "attempts": repair_trace,
            "hidden_feedback_used": False,
            "reference_feedback_used": False,
        },
        "solution": solution,
        "candidate_archive": candidate_archive,
        "hidden_evaluation": hidden,
        "validation_feedback": validation_feedback,
        "token_usage": normalize_usage(usage),
        "latency_seconds": latency,
        "trace_path": str(trace_path),
    }
    if method == "persistent_react":
        row["method_prompt_variant"] = "persistent_react_workspace_v1"
        row["carried_state"] = persistent_carried_state_flags(previous_artifact, previous_final_answer)
    return row


def dynamic_stage_case(
    episode: dict[str, Any],
    public_context: dict[str, Any],
    stage: dict[str, Any],
    method: str,
    previous_solution: dict[str, Any] | None,
    previous_artifact: dict[str, Any] | None = None,
    previous_candidate_population: list[dict[str, Any]] | None = None,
    previous_final_answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_index = int(stage["stage_index"])
    history = [
        {
            "update_id": str(update.get("update_id") or f"t{idx:03d}"),
            "time_index": int(update.get("time_index") or idx),
            "natural_language_update": str(update.get("natural_language_update") or update.get("public_update") or ""),
        }
        for idx, update in enumerate(stage.get("updates") or [], start=1)
    ]
    cumulative_dialogue = [
        {
            "role": "user",
            "turn_id": "initial",
            "content": str(episode.get("public_initial_problem") or ""),
        },
        *[
            {
                "role": "user",
                "turn_id": item["update_id"],
                "content": item["natural_language_update"],
            }
            for item in history
        ],
    ]
    state_interface_note = (
        "The shared_lsm field contains this method's own cumulative dialogue, accepted output, and previous candidate population. Reuse or repair that state when useful; never assume access to another method's trajectory."
    )
    if method == "persistent_react":
        state_interface_note = (
            "The shared_lsm field contains this method's own cumulative dialogue and accepted output; this method receives no candidate population or archive. "
            "The persistent_react_carried_state field carries this method's own previous stage solver code and previous accepted final answer; edit that code minimally for the new update and keep still-valid parts. "
            "Never assume access to another method's trajectory."
        )
    text_parts = [
        "Solve the current NLDO stage from public information only.",
        "",
        "Initial natural-language request:",
        str(episode.get("public_initial_problem") or ""),
    ]
    if history:
        text_parts.extend(["", "Apply these public natural-language updates in order:"])
        for item in history:
            text_parts.append(f"- {item['update_id']}: {item['natural_language_update']}")
    else:
        text_parts.extend(["", "There are no public updates yet; solve the initial request."])
    text_parts.extend(
        [
            "",
            state_interface_note,
            "",
            "Output a concrete solution JSON in the public solution schema. For multi-objective tasks, also output candidate_archive if your method can maintain a trade-off set.",
        ]
    )
    parameters = {
        "tables": public_context.get("tables", {}),
        "csv_schema": public_context.get("csv_schema", {}),
        "optimization_contract": public_context.get("public_optimization_contract")
        or public_context.get("optimization_contract")
        or {},
        "public_objective_spec": public_context.get("public_objective_spec") or {},
        "objective_sense": public_context.get("objective_sense"),
        "multi_objective": public_context.get("multi_objective"),
        "stage_index": stage_index,
        "public_update_history": history,
        "previous_accepted_public_output": compact_previous_solution(previous_solution),
        "shared_lsm": {
            "interface": "common_public_lsm_dialogue_and_own_population_v1",
            "cumulative_dialogue": cumulative_dialogue,
            "dialogue_turn_count": len(cumulative_dialogue),
            "previous_accepted_output": compact_previous_solution(previous_solution),
            "previous_candidate_population": compact_json(
                list(previous_candidate_population or [])[:LSM_POPULATION_LIMIT],
                max_items=80,
                max_list=LSM_POPULATION_LIMIT,
            ),
            "previous_candidate_count": min(len(previous_candidate_population or []), LSM_POPULATION_LIMIT),
            "population_limit": LSM_POPULATION_LIMIT,
            "population_origin": "this method's own most recent publicly executable stage",
            "state_advance_gate": "publicly executable output; hidden evaluation never controls LSM",
            "liveopt_population_available": False,
        },
        "public_state_summary": public_state_summary(public_context, stage_index, history, previous_solution),
    }
    if method == "persistent_react":
        carried_solver_code = str((previous_artifact or {}).get("solver_code") or "")
        carried_final_answer = previous_final_answer if isinstance(previous_final_answer, dict) else {}
        shared_lsm = parameters["shared_lsm"]
        shared_lsm["previous_candidate_population"] = []
        shared_lsm["previous_candidate_count"] = 0
        shared_lsm["population_origin"] = (
            "not provided to persistent_react; this method carries its own solver code and accepted final answer instead"
        )
        parameters["persistent_react_carried_state"] = {
            "interface": "persistent_react_own_workspace_v1",
            "carried_state_summary": (
                "Carried state from this method's own most recent publicly executable stage: the complete solver code "
                "and the accepted final_answer JSON. Apply the smallest necessary edits for the new public update, keep "
                "parts that are still valid, and return the complete updated solver code (never a diff). When an update "
                "refers to earlier decisions, rely on the carried accepted plan. At the initial stage nothing is carried."
            ),
            **persistent_carried_state_flags(previous_artifact, previous_final_answer),
            "previous_stage_solver_code": carried_solver_code,
            "previous_accepted_final_answer": compact_json(carried_final_answer, max_items=80, max_list=20),
            "forbidden": [
                "TSS workbench slots",
                "candidate populations or archives",
                "restart interfaces",
                "LiveOpt artifacts or private memory",
            ],
        }
    if method == "react_generic_workbench":
        parameters["generic_workbench_interface"] = {
            "editable_surface": "the previously accepted public solver code plus current public data",
            "available_solver_modes": ["LP/MILP with scipy.optimize", "direct Python enumeration for small public cases", "public GA-style search when coded by the agent"],
            "fixed_restart_boundary": "no adaptive LiveOpt restart; use only the common public LSM state from this method's own trajectory",
            "forbidden": ["LiveOpt-private state", "TSS patch state", "another method's population", "hidden evaluator state"],
            "edit_policy": (
                "patch the previous accepted solver artifact locally when present; emit the complete revised code; "
                "do not regenerate unchanged logic without a public reason"
            ),
        }
        parameters["previous_accepted_solver_artifact"] = compact_json(
            previous_artifact or {}, max_items=30, max_list=12
        )
    if method == "react_transcript_state":
        parameters["explicit_public_state_record"] = {
            "history_window": history[-12:],
            "previous_output_available": previous_solution is not None,
            "public_tables": sorted((public_context.get("tables") or {}).keys()),
            "contract_keys": sorted((parameters["optimization_contract"] or {}).keys()) if isinstance(parameters["optimization_contract"], dict) else [],
        }
    if method == "react_public_delta_oracle":
        parameters["structured_public_delta"] = structured_public_delta(episode, stage_index)
    return {
        "schema_version": "liveopt_static_calibration_case_v1",
        "case_id": f"{episode.get('episode_id')}_t{stage_index:02d}",
        "source": "NLDO-dynamic-public",
        "source_id": f"{episode.get('episode_id')}:t{stage_index:02d}",
        "problem_text": "\n".join(text_parts),
        "public_context": {
            "parameters": parameters,
            "metadata": {
                "dynamic_stage": True,
                "stage_index": stage_index,
                "episode_id": episode.get("episode_id"),
                "family": episode.get("family"),
                "domain": episode.get("domain"),
            },
        },
        "reference": {},
        "metadata": {"static_only": False, "dynamic_stage": True},
    }


def compact_previous_solution(previous_solution: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(previous_solution, dict) or not previous_solution:
        return {}
    return compact_json(previous_solution, max_items=80, max_list=20)


def public_state_summary(
    public_context: dict[str, Any],
    stage_index: int,
    history: list[dict[str, Any]],
    previous_solution: dict[str, Any] | None,
) -> dict[str, Any]:
    tables = public_context.get("tables") if isinstance(public_context.get("tables"), dict) else {}
    return {
        "stage_index": stage_index,
        "updates_applied": len(history),
        "previous_accepted_output_available": bool(previous_solution),
        "public_table_row_counts": {str(name): len(rows) for name, rows in tables.items() if isinstance(rows, list)},
        "multi_objective": public_context.get("multi_objective"),
        "objective_sense": public_context.get("objective_sense"),
    }


def public_delta_summary(public_context: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = history[-1] if history else {}
    text = str(latest.get("natural_language_update") or "")
    lowered = text.lower()
    affected = []
    for name in sorted((public_context.get("tables") or {}).keys()):
        if str(name).lower() in lowered:
            affected.append(str(name))
    objective_terms = []
    for term in ["cost", "latency", "deadline", "energy", "emission", "distance", "capacity", "priority", "fairness"]:
        if term in lowered:
            objective_terms.append(term)
    return {
        "source": "visible_public_update_text_only",
        "latest_update": text,
        "mentioned_public_tables": affected,
        "mentioned_objective_or_constraint_terms": objective_terms,
        "hidden_delta_available": False,
    }


def structured_public_delta(episode: dict[str, Any], stage_index: int) -> dict[str, Any]:
    """Expose the resolved public operation for the language-grounding control only.

    The benchmark-authored delta resolves entities and values stated or referenced by
    the public update.  Evaluation objects, references, difficulty labels, memory
    keys, and explanatory grounding chains are intentionally excluded.
    """
    if stage_index <= 0:
        return {
            "source": "initial_public_state",
            "operation": {"type": "initial"},
            "hidden_checker_available": False,
            "reference_available": False,
        }
    oracle = list(episode.get("hidden_update_oracle") or [])
    raw = oracle[stage_index - 1].get("hidden_delta") if stage_index <= len(oracle) else {}
    return {
        "source": "benchmark_authored_resolution_of_visible_public_update",
        "operation": sanitize_structured_public_delta(raw or {}),
        "hidden_checker_available": False,
        "reference_available": False,
    }


def sanitize_structured_public_delta(value: Any) -> Any:
    excluded = {"difficulty", "memory_key", "resolved_from_memory", "reference", "checker", "evaluator"}
    if isinstance(value, dict):
        return {
            str(key): sanitize_structured_public_delta(item)
            for key, item in value.items()
            if str(key) not in excluded
        }
    if isinstance(value, list):
        return [sanitize_structured_public_delta(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def accepted_public_artifact(stage_row: dict[str, Any]) -> dict[str, Any]:
    parsed = stage_row.get("parsed_output") if isinstance(stage_row.get("parsed_output"), dict) else {}
    artifact: dict[str, Any] = {
        "solver_code": extract_solver_code(parsed),
        "generic_workbench_plan": parsed.get("generic_workbench_plan") or {},
        "validation_plan": parsed.get("validation_plan") or [],
        "public_acceptance": "publicly_executable_submitted",
    }
    return compact_json(artifact, max_items=30, max_list=12)


def accepted_public_final_answer(stage_row: dict[str, Any]) -> dict[str, Any]:
    parsed = stage_row.get("parsed_output") if isinstance(stage_row.get("parsed_output"), dict) else {}
    final_answer = parsed.get("final_answer")
    return dict(final_answer) if isinstance(final_answer, dict) else {}


def persistent_carried_state_flags(
    previous_artifact: dict[str, Any] | None,
    previous_final_answer: dict[str, Any] | None,
) -> dict[str, bool]:
    return {
        "had_solver_code": bool(str((previous_artifact or {}).get("solver_code") or "")),
        "had_accepted_output": bool(isinstance(previous_final_answer, dict) and previous_final_answer),
    }


def compact_json(value: Any, *, max_items: int, max_list: int) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = "truncated"
                break
            out[str(key)] = compact_json(item, max_items=max_items, max_list=max_list)
        return out
    if isinstance(value, list):
        return [compact_json(item, max_items=max_items, max_list=max_list) for item in value[:max_list]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def public_context_with_loaded_tables(episode: dict[str, Any]) -> dict[str, Any]:
    context = copy.deepcopy(episode.get("public_context") if isinstance(episode.get("public_context"), dict) else {})
    tables = {}
    for name, meta in (context.get("csv_tables") or {}).items():
        path = Path(meta["path"])
        tables[str(name)] = read_csv_rows(path)
    context["tables"] = tables
    return context


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: coerce_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def coerce_cell(value: str) -> Any:
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


def extract_solution(output: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    for payload in [output, parsed.get("final_answer") if isinstance(parsed.get("final_answer"), dict) else {}, parsed]:
        if not isinstance(payload, dict):
            continue
        value = payload.get("solution") or payload.get("candidate_solution") or payload.get("final_solution")
        if isinstance(value, dict):
            nested_archive = value.get("candidate_archive") or value.get("archive") or value.get("pareto_archive")
            if isinstance(nested_archive, list):
                for item in nested_archive:
                    if not isinstance(item, dict):
                        continue
                    nested_solution = item.get("solution")
                    if isinstance(nested_solution, dict):
                        return nested_solution
                    return item
            return value
    return output if isinstance(output, dict) else {}


def extract_candidate_archive(output: dict[str, Any], parsed: dict[str, Any], solution: dict[str, Any]) -> list[dict[str, Any]]:
    raw = None
    final_answer = parsed.get("final_answer") if isinstance(parsed.get("final_answer"), dict) else {}
    payloads = [
        output,
        output.get("solution") if isinstance(output.get("solution"), dict) else {},
        final_answer,
        final_answer.get("solution") if isinstance(final_answer.get("solution"), dict) else {},
        parsed,
        parsed.get("solution") if isinstance(parsed.get("solution"), dict) else {},
    ]
    for payload in payloads:
        if isinstance(payload, dict):
            raw = payload.get("candidate_archive") or payload.get("archive") or payload.get("pareto_archive")
            if raw:
                break
    archive = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                archive.append(item if "solution" in item else {"solution": item})
    if not archive and solution:
        archive.append({"solution": solution})
    return archive


def hidden_metric_for_stage(
    episode: dict[str, Any],
    stage_index: int,
    update_id: str,
    solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    candidate_archive: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmark = reference_metrics.effective_benchmark("NLDO", episode)
    domain = str(episode.get("domain") or "")
    family = str(episode.get("family") or "")
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    oracle = episode.get("hidden_update_oracle") or []
    for offset in range(max(0, stage_index)):
        if offset >= len(oracle):
            break
        state = reference_metrics.apply_hidden_delta(benchmark, domain, state, oracle[offset].get("hidden_delta") or {})
    reference_steps = reference_metrics.reference_trajectory_for_payload(benchmark, domain, episode)
    reference_step = reference_metrics.reference_step_for_stage(reference_steps, update_id, stage_index)
    try:
        metric = reference_metrics.score_solution(
            benchmark,
            domain,
            family,
            state,
            solution or {},
            previous_solution,
            reference_step,
            candidate_archive or [],
        )
        return jsonable(metric)
    except Exception as exc:  # noqa: BLE001
        return {
            "feasible": False,
            "normalized_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }


def solver_result_from_stage(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool((stage.get("hidden_evaluation") or {}).get("feasible")),
        "solution": stage.get("solution") or {},
        "metadata": {
            "candidate_archive": stage.get("candidate_archive") or [],
            "code_execution": compact_code_execution(stage.get("code_execution") or {}),
        },
    }


def update_result_from_stage(stage: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    metric = stage.get("hidden_evaluation") if isinstance(stage.get("hidden_evaluation"), dict) else {}
    return {
        "update_id": str(update.get("update_id") or stage.get("update_id") or ""),
        "natural_language_update": update.get("natural_language_update") or update.get("public_update"),
        "solution": stage.get("solution") or {},
        "feasible": bool(metric.get("feasible")),
        "objective_value": metric.get("objective"),
        "solver_result": solver_result_from_stage(stage),
        "metrics": {"update_latency_seconds": stage.get("latency_seconds")},
        "stage_trace": {
            "outcome": {
                "stage_tokens": int((stage.get("token_usage") or {}).get("total_tokens", 0) or 0),
                "latency_seconds": stage.get("latency_seconds"),
            }
        },
        "token_usage_snapshot": stage.get("token_usage") or {},
    }


def compact_code_execution(value: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "compile_success",
        "runtime_success",
        "compile_error",
        "runtime_error",
        "stdout_preview",
        "stderr_preview",
        "output",
    ]
    return {key: value.get(key) for key in keys if key in value}


def write_stage_trace(
    trace_dir: Path,
    case: dict[str, Any],
    method: str,
    messages: list[dict[str, str]],
    raw_response: str,
    parsed: dict[str, Any],
    usage: dict[str, Any],
    status: str,
    failure_reason: str,
    code_result: dict[str, Any],
    hidden: dict[str, Any],
    repair_trace: list[dict[str, Any]] | None = None,
    repair_budget: int = 0,
) -> Path:
    path = trace_dir / method / f"{case.get('case_id')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "method": method,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "raw_response": raw_response,
        "parsed_output": parsed,
        "usage": normalize_usage(usage),
        "status": status,
        "failure_reason": failure_reason,
        "code_execution": code_result,
        "code_output_repair": {
            "policy": "equal_public_code_output_repair_v1",
            "max_attempts": int(repair_budget or 0),
            "attempts_used": len(repair_trace or []),
            "attempts": repair_trace or [],
            "hidden_feedback_used": False,
            "reference_feedback_used": False,
        },
        "hidden_evaluation": hidden,
    }
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def zero_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "local_cache_hits": 0,
        "local_cache_saved_tokens": 0,
    }


def add_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    out = zero_usage()
    for key in out:
        out[key] = int((left or {}).get(key, 0) or 0) + int((right or {}).get(key, 0) or 0)
    return out


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def replace_episode_run(path: Path, run: dict[str, Any]) -> None:
    episode_id = str(run.get("episode_id") or "")
    existing = load_jsonl(path) if path.exists() else []
    kept = [row for row in existing if str(row.get("episode_id") or "") != episode_id]
    atomic_write_jsonl(path, [*kept, run])


def replace_stage_rows(
    path: Path,
    method: str,
    episode_id: str,
    new_rows: list[dict[str, Any]],
) -> None:
    existing = load_jsonl(path) if path.exists() else []
    kept = [
        row
        for row in existing
        if not (
            str(row.get("method") or "") == method
            and str(row.get("episode_id") or "") == episode_id
        )
    ]
    atomic_write_jsonl(path, [*kept, *new_rows])


def summarize(
    method_runs: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    started: float,
    stage_rows_path: Path,
    nldo_dir: Path,
    trace_dir: Path,
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    all_completed = True
    for method, runs in sorted(method_runs.items()):
        stage_traces = [stage for run in runs for stage in run.get("stage_traces", []) or []]
        reached = len(stage_traces)
        feasible = sum(1 for stage in stage_traces if (stage.get("hidden_evaluation") or {}).get("feasible"))
        rejected = sum(1 for stage in stage_traces if stage.get("status") == "hidden_rejected")
        cache_blocked = sum(1 for stage in stage_traces if stage.get("status") == "cache_miss_blocked")
        total_tokens = sum(int((stage.get("token_usage") or {}).get("total_tokens", 0) or 0) for stage in stage_traces)
        score_values = [
            float((stage.get("hidden_evaluation") or {}).get("normalized_score"))
            for stage in stage_traces
            if (stage.get("hidden_evaluation") or {}).get("normalized_score") is not None
        ]
        methods[method] = {
            "episodes": len(runs),
            "reached_stages": reached,
            "hidden_feasible_stages": feasible,
            "hidden_rejected_stages": rejected,
            "cache_miss_blocked_stages": cache_blocked,
            "mean_normalized_score": sum(score_values) / len(score_values) if score_values else None,
            "total_tokens": total_tokens,
            "mean_tokens_per_reached_stage": total_tokens / reached if reached else None,
            "completed_sequences": sum(1 for run in runs if run.get("run_status") == "completed"),
            "stopped_sequences": sum(1 for run in runs if run.get("run_status") != "completed"),
            "replay_jsonl": str(nldo_dir / f"{method}_limit0.jsonl"),
        }
        if cache_blocked or any(run.get("run_status") != "completed" for run in runs):
            all_completed = False
    return {
        "schema_version": "nldo_dynamic_public_baseline_summary_v1",
        "status": "completed" if all_completed else "partial",
        "mode": "cache_only" if args.cache_only else "provider_backed",
        "provider": args.provider,
        "model": args.model,
        "episodes_jsonl": args.episodes_jsonl,
        "methods": methods,
        "stage_rows_path": str(stage_rows_path),
        "nldo_dir": str(nldo_dir),
        "trace_dir": str(trace_dir),
        "latency_seconds": time.perf_counter() - started,
        "response_protocol": args.response_protocol,
        "code_repair_attempts": int(getattr(args, "code_repair_attempts", 0) or 0),
        "code_repair_policy": "equal_public_code_output_repair_v1",
    }


def console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "mode": summary.get("mode"),
        "summary_json": summary.get("summary_json"),
        "stage_rows_path": summary.get("stage_rows_path"),
        "methods": summary.get("methods"),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
