#!/usr/bin/env python
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
    infer_llm_provider,
    llm_quota_limit_record,
)
from evo2.agents.liveopt_dynamic_impl import (
    RESTART_SKILLS,
    LiveOptDynamicRunner,
    accepted_result_record,
    select_restart_skill_from_semantic_evidence,
)
from evo2.agents.semantic_restart_gate import LiveOptSemanticRestartGate, SemanticRestartDecision
from evo2.agents.liveopt_workbench_impl import LiveOptWorkbenchGenerator, ScaffoldProject, ScaffoldTrace
from evo2.benchmarks.optimization_contracts import public_optimization_contract_for_episode
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult
from scripts.llm_tests.run_liveopt_workbench_benchmark_micro import (
    _candidate_record,
    _fitness_record,
    _jsonable_problem_spec,
    _jsonable_value,
    _pop_env,
    _push_env,
    _sum_usage,
    _trace_summary,
)


def main() -> int:
    args = parse_args()
    if args.disable_lsm and args.stateful_record:
        raise ValueError("--disable-lsm and --stateful-record are mutually exclusive.")
    if args.stateful_record and args.force_restart_skill:
        raise ValueError("--stateful-record fixes warm reuse by construction; do not also set --force-restart-skill.")
    if args.semantic_decisions_json and (
        args.disable_semantic_restart_gate or args.force_restart_skill or args.disable_lsm or args.stateful_record
    ):
        raise ValueError(
            "--semantic-decisions-json is a complete restart policy and cannot be combined with a disabled/forced gate."
        )
    if args.disable_tss:
        return main_without_tss(args)
    output_dir = Path(args.output_dir)
    nldo_dir = output_dir / "NLDO"
    cache_dir = Path(args.cache_dir or output_dir / "cache")
    trace_dir = Path(args.trace_dir or output_dir / "trace")
    summary_path = Path(args.summary_json or output_dir / "summary.json")
    replay_path = nldo_dir / "evo2_limit0.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    nldo_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    episodes = selected_episodes(load_episodes(Path(args.episodes_jsonl)), args.episode_id)
    seed_values = list(range(args.seed_start, args.seed_start + args.seed_count))
    if not args.resume and replay_path.exists():
        replay_path.unlink()
    if not args.resume and summary_path.exists():
        summary_path.unlink()

    old_env = _push_env(
        {
            "DEEPSEEK_CACHE": "0" if args.no_cache else "1",
            "DEEPSEEK_CACHE_DIR": str(cache_dir),
            "DEEPSEEK_TRACE_DIR": str(trace_dir),
            "DEEPSEEK_CACHE_ONLY": "1" if args.cache_only else None,
            "DEEPSEEK_MODEL": args.model if args.model else None,
            "DEEPSEEK_MAX_TOKENS": str(args.max_tokens) if args.max_tokens else None,
            "DEEPSEEK_MAX_ATTEMPTS": str(args.max_attempts),
            "DEEPSEEK_TOTAL_TIMEOUT": str(args.total_timeout) if args.total_timeout else None,
            "KIMI_CACHE": "0" if args.no_cache else "1",
            "KIMI_CACHE_DIR": str(cache_dir),
            "KIMI_TRACE_DIR": str(trace_dir),
            "KIMI_CACHE_ONLY": "1" if args.cache_only else None,
            "KIMI_MODEL": args.model if infer_llm_provider(args.model) == "kimi" else None,
            "KIMI_MAX_RETRIES": str(max(0, int(args.max_attempts) - 1)),
            "KIMI_TIMEOUT": str(max(1, int(args.total_timeout))) if args.total_timeout else None,
        }
    )
    started = time.perf_counter()
    prior_records = load_resume_episode_records(summary_path, episodes) if args.resume else {}
    episode_records: list[dict[str, Any]] = [
        prior_records[str(episode["episode_id"])]
        for episode in episodes
        if str(episode["episode_id"]) in prior_records
    ]
    pending_episodes = [
        episode
        for episode in episodes
        if prior_records.get(str(episode["episode_id"]), {}).get("status") != "completed"
    ]
    completed_rows = count_jsonl_rows(replay_path)
    status = "completed" if not pending_episodes else "running"
    write_summary(
        summary_path,
        args,
        started,
        merge_episode_records_with_queue(episodes, episode_records),
        completed_rows,
        cache_dir,
        trace_dir,
        replay_path,
    )
    try:
        for episode in pending_episodes:
            record, rows = run_episode(args, episode, seed_values)
            upsert_episode_record(episode_records, record)
            if rows:
                replace_episode_rows(replay_path, str(episode["episode_id"]), rows)
            completed_rows = count_jsonl_rows(replay_path)
            write_summary(
                summary_path,
                args,
                started,
                merge_episode_records_with_queue(episodes, episode_records),
                completed_rows,
                cache_dir,
                trace_dir,
                replay_path,
            )
            if record.get("status") == "quota_limited":
                status = "quota_limited"
                break
            if record.get("status") != "completed":
                status = "partial"
                if args.stop_on_episode_failure:
                    break
    finally:
        _pop_env(old_env)

    final_records = merge_episode_records_with_queue(episodes, episode_records)
    summary = write_summary(summary_path, args, started, final_records, completed_rows, cache_dir, trace_dir, replay_path)
    if any(item.get("status") == "quota_limited" for item in final_records):
        status = "quota_limited"
    elif any(item.get("status") != "completed" for item in final_records):
        status = "partial"
    else:
        status = "completed"
    summary["status"] = status
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))
    if status == "quota_limited":
        return 75
    if args.require_completed and status != "completed":
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run full dynamic NLDO benchmark with LiveOpt Workbench artifacts and multi-seed replay.")
    parser.add_argument("--episodes-jsonl", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--output-dir", default="outputs/liveopt_dynamic_nldo_benchmark_full/latest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--summary-json")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL"))
    parser.add_argument("--no-cache", action="store_true", help="Disable response-cache reads and writes for true provider-backed reruns.")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--stop-on-episode-failure", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=18000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--total-timeout", type=float, default=900.0)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument(
        "--initial-population-size",
        type=int,
        default=None,
        help="Population budget for the initial t00 solve. Defaults to --population-size.",
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument(
        "--initial-generations",
        type=int,
        default=None,
        help="Generation budget for the initial t00 solve. Defaults to --generations.",
    )
    parser.add_argument("--archive-limit", type=int, default=200)
    parser.add_argument("--record-metric-trace", action="store_true")
    parser.add_argument("--metric-trace-interval", type=int, default=1)
    parser.add_argument("--metric-trace-archive-limit", type=int, default=200)
    parser.add_argument("--record-trace-archives", action="store_true")
    parser.add_argument("--reference-free-early-stop", action="store_true")
    parser.add_argument("--early-stop-min-generations", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-hv-epsilon", type=float, default=5e-4)
    parser.add_argument("--early-stop-scalar-epsilon", type=float, default=5e-4)
    parser.add_argument("--early-stop-epsilon-box", type=float, default=0.01)
    parser.add_argument("--early-stop-min-archive-size", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--controller-seed",
        type=int,
        default=0,
        help="Independent controller-trajectory replicate id; does not replace numerical --seed-start.",
    )
    parser.add_argument(
        "--semantic-decisions-json",
        type=Path,
        action="append",
        help=(
            "Freeze the verified public-only Warm/Full decisions while independently rebuilding the "
            "controller Workbench. Repeat for disjoint episode sets."
        ),
    )
    parser.add_argument("--max-patch-repairs", type=int, default=3)
    parser.add_argument(
        "--code-repair-attempts",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CODE_REPAIR_ATTEMPTS", "3") or 3),
        help="Equal public code-output repair budget used by the --disable-tss generic public Workbench ablation.",
    )
    parser.add_argument(
        "--disable-lsm",
        action="store_true",
        help=(
            "Module ablation: keep the cumulative public Workbench state, but hide the previous accepted "
            "public event ledger and search result/archive from update localization, data patching, "
            "landscape probing, and restart seeding."
        ),
    )
    parser.add_argument(
        "--stateful-record",
        action="store_true",
        help=(
            "Matched LiveOpt - LSM control: retain cumulative public transcript/JSON and the previous "
            "accepted public output, but hide LSM search-population/probe state and use fixed warm seeding."
        ),
    )
    parser.add_argument(
        "--disable-tss",
        action="store_true",
        help=(
            "Module ablation: replace the TSS Workbench path with the public generic code-solver Workbench "
            "adapter, so no typed search-space scaffold, TSS patch state, or restart API is available."
        ),
    )
    parser.add_argument(
        "--regenerate-typed-workbench",
        action="store_true",
        help=(
            "Matched TSS control: regenerate both typed Workbench slots at every update while retaining "
            "the same framework-owned operators, solver budget, LSM, and restart interface."
        ),
    )
    parser.add_argument("--response-protocol", choices=("json", "code_block"), default="json")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--continue-after-hidden-rejection", action="store_true")
    parser.add_argument(
        "--force-restart-skill",
        choices=sorted(RESTART_SKILLS),
        help="Force a fixed restart skill after LLM update localization, for shared-skill restart ablations.",
    )
    parser.add_argument(
        "--disable-semantic-restart-gate",
        action="store_true",
        help="Disable the LLM selector and use Fixed Warm for ordinary LiveOpt updates.",
    )
    return parser.parse_args()


class FrozenSemanticRestartGate:
    """Provider-free semantic gate for controller-trajectory repeats."""

    def __init__(self, episode_id: str, decision_paths: list[Path]):
        self.episode_id = episode_id
        self.decisions: dict[int, SemanticRestartDecision] = {}
        for path in decision_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("rows") or []:
                if str(row.get("episode_id") or "") != episode_id:
                    continue
                stage = int(row.get("stage") or 0)
                decision_payload = dict(row.get("decision") or {})
                decision_payload["cache_hit"] = True
                self.decisions[stage] = SemanticRestartDecision.from_record(decision_payload)
        missing = [stage for stage in range(1, 13) if stage not in self.decisions]
        if missing:
            raise ValueError(f"{episode_id}: frozen semantic decisions missing stages {missing}")

    def decide(self, *, update_id: str, **_: Any) -> tuple[SemanticRestartDecision, ScaffoldTrace]:
        try:
            stage = int(str(update_id).rsplit("S", 1)[-1])
        except ValueError as exc:
            raise ValueError(f"cannot parse stage from update id {update_id!r}") from exc
        if stage not in self.decisions:
            raise ValueError(f"{self.episode_id}: no frozen semantic decision for stage {stage}")
        trace = ScaffoldTrace(model="frozen_semantic_restart_gate")
        trace.usage.append({"total_tokens": 0, "frozen_semantic_decision_hits": 1})
        return self.decisions[stage], trace


def load_episodes(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main_without_tss(args: argparse.Namespace) -> int:
    if args.force_restart_skill:
        raise ValueError("--disable-tss cannot be combined with --force-restart-skill because restart APIs are unavailable.")
    if args.seed_count != 1:
        raise ValueError("--disable-tss runs a public code-solver ablation with one LLM-generated trajectory; set --seed-count 1.")
    args.response_protocol = "code_block"
    args.no_json_mode = True
    from evo2.agents.deepseek_client import DeepSeekDebugClient
    from scripts.baselines.run_nldo_dynamic_public_baselines import (
        append_jsonl as append_public_jsonl,
        run_episode_method as run_public_episode_method,
    )

    output_dir = Path(args.output_dir)
    nldo_dir = output_dir / "NLDO"
    cache_dir = Path(args.cache_dir or output_dir / "cache")
    trace_dir = Path(args.trace_dir or output_dir / "trace")
    summary_path = Path(args.summary_json or output_dir / "summary.json")
    replay_path = nldo_dir / "evo2_limit0.jsonl"
    stage_rows_path = output_dir / "without_tss_stage_rows.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    nldo_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for path in (replay_path, stage_rows_path, summary_path):
            if path.exists():
                path.unlink()

    old_env = _push_env(
        {
            "DEEPSEEK_CACHE": "0" if args.no_cache else "1",
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
    episodes = selected_episodes(load_episodes(Path(args.episodes_jsonl)), args.episode_id)
    client = DeepSeekDebugClient(model=args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    episode_records: list[dict[str, Any]] = []
    completed_rows = 0
    try:
        for episode in episodes:
            record_started = time.perf_counter()
            try:
                run, stage_rows = run_public_episode_method(
                    args,
                    client,
                    episode,
                    "react_generic_workbench",
                    trace_dir,
                )
                run["agent_mode"] = "liveopt_workbench_v1__without_tss_generic_workbench"
                run["restart_policy"] = "none_without_tss"
                run["module_ablation"] = "without_tss"
                for stage_row in stage_rows:
                    stage_row["method"] = "liveopt_without_tss"
                    stage_row["module_ablation"] = "without_tss"
                append_jsonl(replay_path, [run])
                append_public_jsonl(stage_rows_path, stage_rows)
                completed_rows += 1
                status = "completed" if run.get("run_status") == "completed" else "partial"
                episode_records.append(
                    {
                        "episode_id": str(episode.get("episode_id")),
                        "status": status,
                        "stage_count": len(run.get("stage_traces") or []),
                        "rows": 1,
                        "failure_reason": "" if status == "completed" else str(run.get("run_status") or "stopped"),
                        "latency_seconds": time.perf_counter() - record_started,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                episode_records.append(
                    {
                        "episode_id": str(episode.get("episode_id")),
                        "status": "failed",
                        "stage_count": 0,
                        "rows": 0,
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=20),
                        "latency_seconds": time.perf_counter() - record_started,
                    }
                )
            summary = write_summary(summary_path, args, started, episode_records, completed_rows, cache_dir, trace_dir, replay_path)
            summary["protocol"] = "liveopt_without_tss_public_generic_workbench_v1"
            summary["module_ablation"] = "without_tss"
            summary["stage_rows_jsonl"] = str(stage_rows_path)
            summary["budget"]["restart_policy"] = "none_without_tss"
            summary["budget"]["code_repair_attempts"] = int(getattr(args, "code_repair_attempts", 0) or 0)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            if episode_records[-1].get("status") != "completed" and args.stop_on_episode_failure:
                break
    finally:
        _pop_env(old_env)
    summary = write_summary(summary_path, args, started, episode_records, completed_rows, cache_dir, trace_dir, replay_path)
    summary["protocol"] = "liveopt_without_tss_public_generic_workbench_v1"
    summary["module_ablation"] = "without_tss"
    summary["stage_rows_jsonl"] = str(stage_rows_path)
    summary["budget"]["restart_policy"] = "none_without_tss"
    summary["budget"]["code_repair_attempts"] = int(getattr(args, "code_repair_attempts", 0) or 0)
    summary["status"] = "completed" if episode_records and all(item.get("status") == "completed" for item in episode_records) else "partial"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not args.require_completed or summary["status"] == "completed" else 1


def selected_episodes(episodes: list[dict[str, Any]], ids: list[str] | None) -> list[dict[str, Any]]:
    if not ids:
        return episodes
    wanted = set(ids)
    found = [episode for episode in episodes if str(episode.get("episode_id")) in wanted or str(episode.get("base_instance_id")) in wanted]
    missing = wanted - {str(episode.get("episode_id")) for episode in found} - {str(episode.get("base_instance_id")) for episode in found}
    if missing:
        raise ValueError("episode not found: " + ", ".join(sorted(missing)))
    return found


def run_episode(args: argparse.Namespace, episode: dict[str, Any], seed_values: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = str(episode["episode_id"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    run_rows: dict[int, dict[str, Any]] = {}
    stage_records: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    active_update_index: int | None = None
    active_update_id = ""
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "controller_seed": int(getattr(args, "controller_seed", 0)),
        "domain": episode.get("domain"),
        "family": episode.get("family"),
        "status": "started",
        "stage_count": 0,
        "seed_count": len(seed_values),
        "failure_reason": "",
    }
    try:
        public_context = public_context_with_loaded_tables(episode)
        generator = LiveOptWorkbenchGenerator(model=args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
        project = generator.generate_project(
            episode_id,
            str(episode["public_initial_problem"]),
            public_context,
            max_repairs=args.max_patch_repairs,
        )
        cfg0 = evolution_config(args, seed_values[0], stage_index=0)
        initial_first = project.run(cfg0)
        semantic_restart_gate = None
        if args.semantic_decisions_json:
            semantic_restart_gate = FrozenSemanticRestartGate(episode_id, list(args.semantic_decisions_json))
        elif not (
            args.disable_semantic_restart_gate
            or args.force_restart_skill
            or args.disable_lsm
            or args.stateful_record
        ):
            semantic_restart_gate = LiveOptSemanticRestartGate(
                str(episode.get("public_initial_problem") or ""),
                model=args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            )
        runner = LiveOptDynamicRunner(
            project,
            public_context=public_context,
            result=initial_first,
            semantic_restart_gate=semantic_restart_gate,
        )

        per_seed_results: dict[int, EvolutionResult] = {seed_values[0]: initial_first}
        for seed in seed_values[1:]:
            per_seed_results[seed] = project.run(evolution_config(args, seed, stage_index=0))

        run_rows = {
            seed: make_seed_run_row(
                args=args,
                episode=episode,
                seed=seed,
                project=project,
                initial_result=result,
                initial_usage=_sum_usage(project.trace.usage),
                initial_latency=project.trace.latency_seconds,
            )
            for seed, result in per_seed_results.items()
        }

        updates = list(episode.get("update_stream") or [])
        if args.max_updates is not None:
            updates = updates[: max(0, args.max_updates)]

        stage_records = [initial_stage_record(project, initial_first, project.trace.latency_seconds, _sum_usage(project.trace.usage))]
        cumulative_tokens = _sum_usage(project.trace.usage)

        for update_index, update in enumerate(updates, start=1):
            update_id = str(update.get("update_id") or f"u{update_index:03d}")
            update_text = str(update.get("natural_language_update") or update.get("public_update") or "")
            active_update_index = update_index
            active_update_id = update_id
            previous_project = runner.project
            stage = runner.update(
                update_id=update_id,
                natural_language_update=update_text,
                public_data_patch=update.get("public_data_patch") or None,
                config=evolution_config(args, seed_values[0], stage_index=update_index),
                max_patch_repairs=args.max_patch_repairs,
                force_restart_skill=args.force_restart_skill,
                disable_lsm=args.disable_lsm,
                stateful_record=args.stateful_record,
                regenerate_typed_workbench=bool(getattr(args, "regenerate_typed_workbench", False)),
            )
            stage_usage = _sum_usage(stage.patch_trace.usage)
            cumulative_tokens = add_usage(cumulative_tokens, stage_usage)
            per_seed_results[seed_values[0]] = stage.result
            append_update_to_seed_row(
                run_rows[seed_values[0]],
                update,
                stage.project,
                stage.result,
                stage_usage,
                stage.latency_seconds,
                cumulative_tokens,
                impact=stage.impact.__dict__,
                restart_metadata=stage.restart_metadata,
            )

            for seed in seed_values[1:]:
                previous_seed_result = None if args.disable_lsm else per_seed_results[seed]
                if args.stateful_record:
                    objective_names = list((stage.project.problem_spec or {}).get("objective_names") or [])
                    previous_seed_result = accepted_result_record(previous_seed_result, pareto=len(objective_names) > 1)
                seed_restart_skill = stage.impact.restart_skill
                if not (args.force_restart_skill or args.stateful_record or args.disable_lsm):
                    seed_restart_skill, _ = select_restart_skill_from_semantic_evidence(
                        semantic_full_vote=bool(
                            (stage.restart_metadata.get("semantic_restart_gate") or {}).get("verified_full_vote")
                        )
                    )
                initial_genomes, restart_metadata = runner.restart_builder.build(
                    restart_skill=seed_restart_skill,
                    previous_result=previous_seed_result,
                    segments=stage.project.segments,
                    population_size=args.population_size,
                    seed=seed,
                    evaluate=stage.project.evaluate,
                    data=stage.project.data,
                    objective_shift=None,
                )
                restart_metadata = dict(restart_metadata)
                restart_metadata["data_patch"] = (stage.restart_metadata or {}).get("data_patch", {})
                restart_metadata["lsm_disabled"] = bool(args.disable_lsm)
                restart_metadata["memory_mode"] = (
                    "stateful_record" if args.stateful_record else ("none" if args.disable_lsm else "lsm")
                )
                restart_metadata["semantic_restart_gate"] = copy.deepcopy(
                    stage.restart_metadata.get("semantic_restart_gate") or {}
                )
                restart_metadata["restart_selection_rule"] = "verified_semantic_full_else_fixed_warm"
                replay_started = time.perf_counter()
                result = stage.project.run(evolution_config(args, seed, stage_index=update_index), initial_genomes=initial_genomes)
                result.metadata = dict(result.metadata)
                result.metadata["restart"] = restart_metadata
                per_seed_results[seed] = result
                append_update_to_seed_row(
                    run_rows[seed],
                    update,
                    stage.project,
                    result,
                    stage_usage,
                    time.perf_counter() - replay_started,
                    cumulative_tokens,
                    impact={**stage.impact.__dict__, "restart_skill": seed_restart_skill},
                    restart_metadata=restart_metadata,
                )
            stage_records.append(dynamic_stage_record(stage, stage_usage))

        rows = [run_rows[seed] for seed in seed_values]
        record.update(
            {
                "status": "completed",
                "stage_count": 1 + len(updates),
                "rows": len(rows),
                "latency_seconds": time.perf_counter() - started,
                "initial_segments": [segment.kind for segment in project.segments],
                "stage_records": stage_records,
            }
        )
    except Exception as exc:  # noqa: BLE001
        failure_reason = f"{type(exc).__name__}: {exc}"
        quota_limit = llm_quota_limit_record(exc)
        failure_status = "quota_limited" if quota_limit else "failed"
        rows = [run_rows[seed] for seed in seed_values if seed in run_rows]
        accepted_prefix_updates = min(
            (len(row.get("update_results") or []) for row in rows),
            default=max(0, len(stage_records) - 1),
        )
        unexecuted_update_ids = [
            str(update.get("update_id") or f"u{index:03d}")
            for index, update in enumerate(updates[accepted_prefix_updates:], start=accepted_prefix_updates + 1)
        ]
        for row in rows:
            row["run_status"] = "quota_limited" if quota_limit else "stopped_after_failure"
            row["failure_reason"] = failure_reason
            row["failure_stage_index"] = active_update_index
            row["failure_update_id"] = active_update_id
            row["accepted_prefix_updates"] = accepted_prefix_updates
            row["unexecuted_update_ids"] = unexecuted_update_ids
        record.update(
            {
                "status": failure_status,
                "failure_reason": failure_reason,
                "quota_limit": quota_limit,
                "resumable": bool(quota_limit),
                "traceback": traceback.format_exc(limit=20),
                "latency_seconds": time.perf_counter() - started,
                "stage_count": 1 + accepted_prefix_updates if rows else 0,
                "rows": len(rows),
                "accepted_prefix_updates": accepted_prefix_updates,
                "rejection_stage_index": active_update_index,
                "rejection_update_id": active_update_id,
                "unexecuted_update_ids": unexecuted_update_ids,
                "stage_records": stage_records,
            }
        )
    return record, rows


def evolution_config(args: argparse.Namespace, seed: int, *, stage_index: int = 0) -> EvolutionConfig:
    population_size = (
        int(args.initial_population_size)
        if stage_index == 0 and args.initial_population_size is not None
        else int(args.population_size)
    )
    generations = (
        int(args.initial_generations)
        if stage_index == 0 and args.initial_generations is not None
        else int(args.generations)
    )
    return EvolutionConfig(
        population_size=population_size,
        generations=generations,
        seed=seed,
        archive_limit=args.archive_limit,
        structured_initialization=True,
        record_metric_history=bool(getattr(args, "record_metric_trace", False)),
        record_archive_history=bool(args.record_trace_archives),
        history_interval=max(1, int(args.metric_trace_interval or 1)),
        history_archive_limit=max(1, int(args.metric_trace_archive_limit or args.archive_limit)),
        reference_free_early_stopping=bool(getattr(args, "reference_free_early_stop", False)),
        early_stop_min_generations=int(getattr(args, "early_stop_min_generations", 40)),
        early_stop_patience=int(getattr(args, "early_stop_patience", 20)),
        early_stop_hv_epsilon=float(getattr(args, "early_stop_hv_epsilon", 5e-4)),
        early_stop_scalar_epsilon=float(getattr(args, "early_stop_scalar_epsilon", 5e-4)),
        early_stop_epsilon_box=float(getattr(args, "early_stop_epsilon_box", 0.01)),
        early_stop_min_archive_size=int(getattr(args, "early_stop_min_archive_size", 8)),
    )


def public_context_with_loaded_tables(episode: dict[str, Any]) -> dict[str, Any]:
    public_context = dict(episode.get("public_context") or {})
    tables = {}
    for name, meta in (public_context.get("csv_tables") or {}).items():
        path = Path(meta["path"])
        tables[name] = read_csv_rows(path)
    public_context["tables"] = tables
    optimization_contract = public_optimization_contract(episode)
    if optimization_contract:
        public_context["optimization_contract"] = optimization_contract
        public_context["objective_sense"] = optimization_contract.get("objective_sense")
        public_context["multi_objective"] = optimization_contract.get("objective_mode") == "multi_objective"
    return public_context


def public_optimization_contract(episode: dict[str, Any]) -> dict[str, Any]:
    """Expose the public objective/runtime contract without leaking references.

    The benchmark payload contains hidden reference trajectories, but the solver
    choice itself is a public task requirement: exact/single-solution problems,
    scalar heuristic problems, and Pareto multi-objective problems should not be
    inferred from wording alone. This contract intentionally omits hidden deltas,
    reference archives, reference objectives, and evaluator implementations.
    """

    return public_optimization_contract_for_episode(episode)


def _public_scalar_metric_names(metrics: list[str]) -> list[str]:
    excluded = {
        "feasibility",
        "objective_gap",
        "exact_reference_objective",
        "pareto_coverage",
        "hypervolume",
        "igd",
        "token_cost",
        "latency",
        "latency_seconds",
        "disruption",
    }
    names = [item for item in metrics if item and item not in excluded]
    return names[:3]


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


def make_seed_run_row(
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    seed: int,
    project: ScaffoldProject,
    initial_result: EvolutionResult,
    initial_usage: dict[str, int],
    initial_latency: float,
) -> dict[str, Any]:
    best = initial_result.best.result
    return {
        "episode_id": episode.get("episode_id"),
        "base_benchmark": "NLDO",
        "domain": episode.get("domain"),
        "family": episode.get("family"),
        "base_instance_id": episode.get("base_instance_id"),
        "run_seed": seed,
        "controller_seed": int(getattr(args, "controller_seed", 0)),
        "agent_mode": agent_mode(args),
        "restart_policy": (
            "fixed_warm_from_accepted_public_output"
            if args.stateful_record
            else (
                "full_restart_without_lsm"
                if args.disable_lsm
                else (
                    args.force_restart_skill
                    or ("frozen_verified_semantic_full_else_fixed_warm" if args.semantic_decisions_json else "semantic_llm")
                )
            )
        ),
        "module_ablation": (
            "typed_full_regeneration"
            if bool(getattr(args, "regenerate_typed_workbench", False))
            else ("without_lsm_stateful_record" if args.stateful_record else ("without_lsm" if args.disable_lsm else "full"))
        ),
        "model": args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "budget": {
            "population_size": args.population_size,
            "generations": args.generations,
            "initial_population_size": args.initial_population_size or args.population_size,
            "initial_generations": args.initial_generations or args.generations,
            "seed_count": args.seed_count,
            "controller_seed": int(getattr(args, "controller_seed", 0)),
            "archive_limit": args.archive_limit,
        },
        "initial_candidate": best.solution if best else {},
        "initial_feasible": bool(best.feasible) if best else False,
        "initial_objective": best.scalar if best else None,
        "initial_solver_result": solver_result_record(project, initial_result),
        "initial_stage_trace": {
            "stage_tokens": int(initial_usage.get("total_tokens", 0) or 0),
            "latency_seconds": initial_latency,
        },
        "update_results": [],
        "token_usage": initial_usage,
        "latency_seconds": initial_latency,
        "setup_code": project.setup_code,
        "fitness_code": project.fitness_code,
        "segments": [_jsonable_value(segment.__dict__) for segment in project.segments],
    }


def append_update_to_seed_row(
    row: dict[str, Any],
    update: dict[str, Any],
    project: ScaffoldProject,
    result: EvolutionResult,
    stage_usage: dict[str, int],
    latency_seconds: float,
    cumulative_tokens: dict[str, int],
    *,
    impact: dict[str, Any] | None = None,
    restart_metadata: dict[str, Any] | None = None,
) -> None:
    best = result.best.result
    row["update_results"].append(
        {
            "update_id": str(update.get("update_id") or ""),
            "natural_language_update": update.get("natural_language_update") or update.get("public_update"),
            "solution": best.solution if best else {},
            "feasible": bool(best.feasible) if best else False,
            "objective_value": best.scalar if best else None,
            "solver_result": solver_result_record(project, result),
            "metrics": {"update_latency_seconds": latency_seconds},
            "impact": _jsonable_value(impact or {}),
            "restart": _jsonable_value(restart_metadata or {}),
            "stage_trace": {
                "outcome": {
                    "stage_tokens": int(stage_usage.get("total_tokens", 0) or 0),
                    "latency_seconds": latency_seconds,
                }
            },
            "token_usage_snapshot": dict(cumulative_tokens),
            "setup_code": project.setup_code,
            "fitness_code": project.fitness_code,
            "segments": [_jsonable_value(segment.__dict__) for segment in project.segments],
        }
    )
    row["token_usage"] = dict(cumulative_tokens)
    row["latency_seconds"] = float(row.get("latency_seconds") or 0.0) + latency_seconds


def agent_mode(args: argparse.Namespace) -> str:
    base = "liveopt_workbench_v1"
    if args.disable_tss:
        return f"{base}__without_tss_generic_workbench"
    if args.disable_lsm:
        return f"{base}__without_lsm"
    if args.stateful_record:
        return f"{base}__without_lsm_stateful_record"
    if bool(getattr(args, "regenerate_typed_workbench", False)):
        return f"{base}__typed_full_regeneration"
    if args.force_restart_skill:
        return f"{base}__fixed_{args.force_restart_skill}"
    return base


def solver_result_record(project: ScaffoldProject, result: EvolutionResult) -> dict[str, Any]:
    best = result.best.result
    return {
        "success": bool(best.feasible) if best else False,
        "solution": best.solution if best else {},
        "metadata": {
            "runtime": result.metadata,
            "problem_spec": _jsonable_problem_spec(project.problem_spec),
            "candidate_archive": [_archive_metric_record(candidate) for candidate in result.archive],
            "final_population": [_archive_metric_record(candidate) for candidate in result.population],
            "history": _jsonable_value(result.history),
            "agent_objective_raw": _fitness_record(best),
        },
    }


def _archive_metric_record(candidate) -> dict[str, Any]:
    fitness = _fitness_record(candidate.result)
    return _jsonable_value(
        {
            "rank": candidate.rank,
            "solution": fitness.get("solution") or {},
            "workbench_objectives": fitness.get("objectives") or [],
            "workbench_scalar": fitness.get("scalar"),
            "workbench_feasible": fitness.get("feasible"),
            "fitness": fitness,
            "genome": candidate.genome,
        }
    )


def initial_stage_record(project: ScaffoldProject, result: EvolutionResult, latency: float, usage: dict[str, int]) -> dict[str, Any]:
    best = result.best.result
    return {
        "update_id": "initial",
        "segments": [segment.kind for segment in project.segments],
        "solver_mode": (project.problem_spec or {}).get("solver_mode"),
        "feasible": bool(best.feasible) if best else False,
        "objective": best.scalar if best else None,
        "archive_count": len(result.archive),
        "population_count": len(result.population),
        "latency_seconds": latency,
        "token_usage": usage,
    }


def dynamic_stage_record(stage, usage: dict[str, int]) -> dict[str, Any]:
    best = stage.result.best.result
    return {
        "update_id": stage.update_id,
        "impact": _jsonable_value(stage.impact.__dict__),
        "restart": _jsonable_value(stage.restart_metadata),
        "segments": [segment.kind for segment in stage.project.segments],
        "solver_mode": (stage.project.problem_spec or {}).get("solver_mode"),
        "feasible": bool(best.feasible) if best else False,
        "objective": best.scalar if best else None,
        "archive_count": len(stage.result.archive),
        "population_count": len(stage.result.population),
        "latency_seconds": stage.latency_seconds,
        "token_usage": usage,
        "patch_errors": list(stage.patch_trace.errors),
    }


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, 0) + int(value)
    return out


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable_value(row), ensure_ascii=False, sort_keys=True) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def replace_episode_rows(path: Path, episode_id: str, rows: list[dict[str, Any]]) -> None:
    """Atomically replace one episode so repeated --resume runs never duplicate rows."""

    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    merged = [row for row in existing if str(row.get("episode_id") or "") != episode_id]
    merged.extend(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(_jsonable_value(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_resume_episode_records(
    summary_path: Path,
    episodes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    selected_ids = {str(episode.get("episode_id") or "") for episode in episodes}
    return {
        str(record.get("episode_id")): record
        for record in list(payload.get("episode_records") or [])
        if str(record.get("episode_id") or "") in selected_ids
    }


def upsert_episode_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    episode_id = str(record.get("episode_id") or "")
    records[:] = [item for item in records if str(item.get("episode_id") or "") != episode_id]
    records.append(record)


def merge_episode_records_with_queue(
    episodes: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(record.get("episode_id") or ""): record for record in records}
    return [
        by_id.get(
            str(episode.get("episode_id") or ""),
            {
                "episode_id": str(episode.get("episode_id") or ""),
                "status": "queued",
                "stage_count": 0,
                "rows": 0,
                "failure_reason": "",
            },
        )
        for episode in episodes
    ]


def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    started: float,
    episode_records: list[dict[str, Any]],
    completed_rows: int,
    cache_dir: Path,
    trace_dir: Path,
    replay_path: Path,
) -> dict[str, Any]:
    statuses = [str(item.get("status") or "") for item in episode_records]
    if any(status == "quota_limited" for status in statuses):
        run_status = "quota_limited"
    elif statuses and all(status == "completed" for status in statuses):
        run_status = "completed"
    elif any(status == "failed" for status in statuses):
        run_status = "partial"
    elif any(status in {"queued", "started", "running"} for status in statuses):
        run_status = "running"
    else:
        run_status = "partial"
    quota_limit = next(
        (
            dict(item.get("quota_limit") or {})
            for item in episode_records
            if item.get("status") == "quota_limited"
        ),
        {},
    )
    provider_trace = _trace_summary(trace_dir)
    summary = {
        "status": run_status,
        "protocol": "liveopt_workbench_dynamic_nldo_benchmark_full_v1",
        "mode": "cache_only" if args.cache_only else "provider_backed",
        "provider": infer_llm_provider(args.model),
        "model": args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        "quota_limit": quota_limit,
        "resumable": run_status == "quota_limited",
        "resume_hint": "rerun the same command with --resume after the provider quota resets",
        "controller_seed": int(getattr(args, "controller_seed", 0)),
        "semantic_decisions_json": [str(path) for path in (args.semantic_decisions_json or [])],
        "episodes_jsonl": args.episodes_jsonl,
        "episode_count": len(episode_records),
        "completed_episode_count": sum(1 for item in episode_records if item.get("status") == "completed"),
        "formal_replay_jsonl": str(replay_path),
        "completed_seed_rows": completed_rows,
        "budget": {
            "population_size": args.population_size,
            "generations": args.generations,
            "initial_population_size": args.initial_population_size or args.population_size,
            "initial_generations": args.initial_generations or args.generations,
            "seed_count": args.seed_count,
            "seed_start": args.seed_start,
            "controller_seed": int(getattr(args, "controller_seed", 0)),
            "archive_limit": args.archive_limit,
            "record_metric_trace": bool(args.record_metric_trace),
            "metric_trace_interval": int(args.metric_trace_interval),
            "metric_trace_archive_limit": int(args.metric_trace_archive_limit),
            "record_trace_archives": bool(args.record_trace_archives),
            "reference_free_early_stop": bool(getattr(args, "reference_free_early_stop", False)),
            "early_stop_min_generations": int(getattr(args, "early_stop_min_generations", 40)),
            "early_stop_patience": int(getattr(args, "early_stop_patience", 20)),
            "early_stop_hv_epsilon": float(getattr(args, "early_stop_hv_epsilon", 5e-4)),
            "early_stop_scalar_epsilon": float(getattr(args, "early_stop_scalar_epsilon", 5e-4)),
            "early_stop_epsilon_box": float(getattr(args, "early_stop_epsilon_box", 0.01)),
            "early_stop_min_archive_size": int(getattr(args, "early_stop_min_archive_size", 8)),
            "restart_policy": (
                "fixed_warm_from_accepted_public_output"
                if args.stateful_record
                else (
                    "full_restart_without_lsm"
                    if args.disable_lsm
                    else (
                        args.force_restart_skill
                        or ("frozen_verified_semantic_full_else_fixed_warm" if args.semantic_decisions_json else "semantic_llm")
                    )
                )
            ),
            "module_ablation": (
                "typed_full_regeneration"
                if bool(getattr(args, "regenerate_typed_workbench", False))
                else (
                    "without_lsm_stateful_record"
                    if args.stateful_record
                    else ("without_lsm" if args.disable_lsm else ("without_tss" if args.disable_tss else "full"))
                )
            ),
        },
        "episode_records": episode_records,
        "cache_dir": str(cache_dir),
        "trace_dir": str(trace_dir),
        "provider_trace": provider_trace,
        "deepseek_trace": provider_trace,
        "latency_seconds": time.perf_counter() - started,
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "summary_json": summary.get("summary_json"),
        "formal_replay_jsonl": summary.get("formal_replay_jsonl"),
        "completed_episode_count": summary.get("completed_episode_count"),
        "completed_seed_rows": summary.get("completed_seed_rows"),
        "budget": summary.get("budget"),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "quota_limit": summary.get("quota_limit") or {},
        "resumable": bool(summary.get("resumable")),
        "trace_events": (summary.get("provider_trace") or summary.get("deepseek_trace") or {}).get("events", {}),
        "episodes": [
            {
                "episode_id": item.get("episode_id"),
                "status": item.get("status"),
                "stage_count": item.get("stage_count"),
                "rows": item.get("rows"),
                "failure_reason": item.get("failure_reason"),
            }
            for item in summary.get("episode_records", [])
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
