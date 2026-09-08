#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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

from evo2.agents.liveopt_dynamic_impl import (
    RESTART_SKILLS,
    ScaffoldRestartSeedBuilder,
    accepted_result_record,
    apply_public_context_patch,
    estimate_objective_space_shift,
    normalize_public_context_tables,
    select_restart_skill_from_landscape_change,
    select_restart_skill_from_semantic_evidence,
)
from evo2.agents.generic_workbench_impl import (
    GenericWorkbenchBuilder,
    GenericWorkbenchPatcher,
    compile_generic_workbench,
    generic_artifact_from_scaffold,
    generic_restart_seeds,
)
from evo2.agents.semantic_restart_gate import SemanticRestartDecision
from evo2.agents.liveopt_workbench_impl import ScaffoldTrace, compile_scaffold_project
from evo2.core.template_optimizer import Candidate, EvolutionConfig, EvolutionResult, FitnessResult
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (
    _archive_metric_record,
    add_usage,
    append_jsonl,
    append_update_to_seed_row,
    dynamic_stage_record,
    initial_stage_record,
    load_episodes,
    make_seed_run_row,
    public_context_with_loaded_tables,
    selected_episodes,
)
from scripts.analysis.evaluate_reference_metrics import (
    DEFAULT_NLDO_STRONG_REFERENCE,
    apply_hidden_delta,
    effective_benchmark,
    reference_step_for_stage,
    reference_trajectory_for_payload,
    score_solution,
)

LEGACY_RESTART_SKILL_REMAP = {
    "diversity_restart_v1": "full_restart_v1",
    "elite_immigrant_restart_v1": "full_restart_v1",
    "feasibility_preserving_restart_v1": "full_restart_v1",
}


def main() -> int:
    args = parse_args()
    if args.stateful_record and args.force_restart_skill:
        raise ValueError("--stateful-record fixes Warm reuse; do not also pass --force-restart-skill.")
    if args.stateful_record and args.semantic_decisions_json:
        raise ValueError("--stateful-record has no semantic restart selector; do not pass frozen decisions.")
    if args.generic_workbench and args.stateful_record:
        raise ValueError("Generic Workbench and Stateful Record are separate single-component controls.")
    if args.generic_workbench and args.force_restart_skill:
        raise ValueError("Generic Workbench retains the frozen LiveOpt semantic selector.")
    if args.generic_workbench and args.rehydrate_replay_prefix:
        raise ValueError("Generic Workbench cannot rehydrate typed replay prefixes; recompute from t00.")
    if args.generic_from_typed_scaffold and not args.generic_workbench:
        raise ValueError("--generic-from-typed-scaffold requires --generic-workbench.")
    if args.generic_workbench and args.variation_mode != "typed":
        raise ValueError("--variation-mode applies to the typed Workbench, not --generic-workbench.")
    if args.semantic_decisions_json and args.force_restart_skill:
        raise ValueError("Frozen semantic decisions cannot be combined with --force-restart-skill.")
    if args.landscape_restart_selector and args.semantic_decisions_json:
        raise ValueError("The landscape comparison and LiveOpt semantic selector are separate policies.")
    if args.landscape_restart_selector and args.force_restart_skill:
        raise ValueError("The landscape comparison chooses its own Warm/Full action.")
    if args.landscape_restart_selector and (args.stateful_record or args.generic_workbench):
        raise ValueError("The landscape selector is evaluated only on the typed LSM backbone.")
    # Shared row serializers also serve live controller/ablation runners.
    # Artifact replay is the complete LSM+TSS path unless an explicit replay
    # option below says otherwise, so provide their non-ablation defaults.
    args.disable_lsm = False
    args.disable_tss = False
    args.regenerate_typed_workbench = False
    args.model = None
    output_dir = Path(args.output_dir)
    nldo_dir = output_dir / "NLDO"
    replay_path = nldo_dir / "evo2_limit0.jsonl"
    summary_path = Path(args.summary_json or output_dir / "summary.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    nldo_dir.mkdir(parents=True, exist_ok=True)
    if replay_path.exists() and not args.resume:
        replay_path.unlink()
    if summary_path.exists() and not args.resume:
        summary_path.unlink()

    episodes = selected_episodes(load_episodes(Path(args.episodes_jsonl)), args.episode_id)
    seed_values = list(range(args.seed_start, args.seed_start + args.seed_count))
    started = time.perf_counter()
    episode_records: list[dict[str, Any]] = []
    completed_rows = 0
    status = "completed"
    for episode in episodes:
        record, rows = replay_episode(args, episode, seed_values)
        episode_records.append(record)
        completed_rows += len(rows)
        append_jsonl(replay_path, rows)
        write_artifact_summary(summary_path, args, started, episode_records, completed_rows, replay_path)
        if record.get("status") != "completed":
            status = "partial"
            if args.stop_on_episode_failure:
                break
    summary = write_artifact_summary(summary_path, args, started, episode_records, completed_rows, replay_path)
    summary["status"] = status
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(console_summary(summary), ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_completed and status != "completed":
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay LiveOpt/control trajectories; from-scratch Generic mode uses public-only LLM code generation.")
    # ``make_seed_run_row`` is shared with the provider-backed runner, whose
    # namespace also contains the module-ablation switches.  Artifact replay
    # never enters either of those runner-only paths, but the shared metadata
    # helpers still inspect the attributes.  Define them explicitly here so a
    # current-code replay records the full configuration instead of depending
    # on an older argparse namespace shape.
    parser.set_defaults(disable_lsm=False, disable_tss=False)
    parser.add_argument("--episodes-jsonl", default="data/evo2_dynoptbench/public_csv/nldo_15episodes_10updates_csv.jsonl")
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--source-run-dir", required=True, help="Run directory containing replay_jobs/<episode>/summary.json and NLDO/evo2_limit0.jsonl.")
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument(
        "--reference-jsonl",
        default=str(DEFAULT_NLDO_STRONG_REFERENCE),
        help=(
            "Reference payload used only by the non-intervening per-generation metric recorder. "
            "The optimizer and reference-free stopper never read it."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--model", default="artifact_replay")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-completed", action="store_true")
    parser.add_argument("--stop-on-episode-failure", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument(
        "--replay-updates",
        type=int,
        default=0,
        help="With --rehydrate-replay-prefix, rehydrate this many saved dynamic updates before replaying the rest.",
    )
    parser.add_argument(
        "--rehydrate-replay-prefix",
        action="store_true",
        help="Rehydrate the saved replay prefix from source JSONL candidate archives instead of recomputing early stages.",
    )
    parser.add_argument(
        "--use-source-history-population",
        action="store_true",
        help=(
            "For replayed updates, build restart seeds from the source LiveOpt run's immediately previous "
            "stage population/archive instead of this replay arm's previous stage. This isolates restart "
            "counterfactuals on the same LiveOpt history."
        ),
    )
    parser.add_argument(
        "--require-source-final-population",
        action="store_true",
        help="Fail instead of falling back to a source archive when common-incoming lineage is requested.",
    )
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument(
        "--initial-population-size",
        type=int,
        default=None,
        help="Optional population budget for the initial t00 solve. Defaults to --population-size.",
    )
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument(
        "--variation-mode",
        choices=("typed", "type_agnostic_resampling"),
        default="typed",
        help=(
            "Variation operators for the typed Workbench. The type-agnostic control keeps the same "
            "SegmentSpec, repair, evaluator, population, and restart action, but uses whole-segment "
            "uniform crossover and whole-segment random resampling."
        ),
    )
    parser.add_argument(
        "--initial-generations",
        type=int,
        default=None,
        help="Optional generation budget for the initial t00 solve. Defaults to --generations.",
    )
    parser.add_argument("--archive-limit", type=int, default=200)
    parser.add_argument(
        "--reference-free-early-stop",
        action="store_true",
        help="Stop GA/NSGA-II from run-internal convergence only; never uses hidden/reference fronts.",
    )
    parser.add_argument("--early-stop-min-generations", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-hv-epsilon", type=float, default=5e-4)
    parser.add_argument("--early-stop-scalar-epsilon", type=float, default=5e-4)
    parser.add_argument("--early-stop-epsilon-box", type=float, default=0.01)
    parser.add_argument("--early-stop-min-archive-size", type=int, default=8)
    parser.add_argument(
        "--record-metric-trace",
        action="store_true",
        help="Record per-generation archive snapshots so HV/IGD trajectories can be exported later.",
    )
    parser.add_argument(
        "--metric-trace-interval",
        type=int,
        default=1,
        help="Generation interval for archive snapshots when --record-metric-trace is enabled.",
    )
    parser.add_argument(
        "--metric-trace-archive-limit",
        type=int,
        default=200,
        help="Maximum candidates used to compute traced HV/IGD per generation.",
    )
    parser.add_argument(
        "--record-trace-archives",
        action="store_true",
        help="Also store per-generation archive snapshots. Use only for selected visual case studies.",
    )
    parser.add_argument(
        "--retain-final-population",
        action="store_true",
        help=(
            "Keep each stage's serialized final population even when metric traces are thinned. "
            "Required when this run will be the exact same-seed predecessor for matched restart ablations."
        ),
    )
    parser.add_argument(
        "--retain-final-population-stage",
        action="append",
        type=int,
        default=[],
        help=(
            "Retain the serialized final population only for this stage index (t00=0). "
            "Repeat as needed. This keeps a LiveOpt backbone lightweight while preserving "
            "the exact population required by a later shadow branch."
        ),
    )
    parser.add_argument(
        "--drop-final-population",
        action="store_true",
        help=(
            "Drop serialized final populations even when per-generation metric traces are disabled. "
            "Use for terminal shadow controls that will never serve as a population source."
        ),
    )
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--controller-seed",
        type=int,
        default=0,
        help="Independent controller-trajectory replicate id; numerical seeds remain controlled separately.",
    )
    parser.add_argument(
        "--stateful-record",
        action="store_true",
        help=(
            "Matched w/o-LSM control: expose only the committed candidate/archive from the "
            "selected predecessor and force Fixed Warm; discard non-committed population and restart state."
        ),
    )
    parser.add_argument(
        "--generic-workbench",
        action="store_true",
        help=(
            "Matched w/o-TSS control: generate an untyped artifact from the public task and bare "
            "callback ABI, then patch it from public updates while retaining LSM/restart."
        ),
    )
    parser.add_argument(
        "--generic-from-typed-scaffold",
        action="store_true",
        help=(
            "Legacy diagnostic: mechanically materialize the accepted t00 TSS scaffold instead of "
            "building the Generic Workbench from scratch. Not used by the matched paper ablation."
        ),
    )
    parser.add_argument("--generic-model", default=None)
    parser.add_argument("--generic-max-patch-repairs", type=int, default=3)
    parser.add_argument(
        "--force-restart-skill",
        choices=sorted(RESTART_SKILLS),
        help="Override saved restart skills while keeping saved data/code artifacts fixed.",
    )
    parser.add_argument(
        "--freeze-workbench-after-initial",
        action="store_true",
        help=(
            "Ablation mode: keep the initial setup.py and fitness.py for every dynamic update, "
            "while still applying public data patches and restart policies from the source trace."
        ),
    )
    parser.add_argument(
        "--record-objective-shift-diagnostics",
        action="store_true",
        help=(
            "Record the public before/after objective-space diagnostics on the selected predecessor "
            "population without changing a saved or forced restart action. This is intended for "
            "same-prefix Full-vs-Warm counterfactual analysis."
        ),
    )
    parser.add_argument(
        "--landscape-restart-selector",
        action="store_true",
        help=(
            "Traditional DMOEA control: re-evaluate a deterministic 10% sensor sample under the "
            "old/new public objective functions and choose Full only for a large normalized "
            "landscape displacement. This is separate from LiveOpt's semantic selector."
        ),
    )
    parser.add_argument(
        "--semantic-decisions-json",
        type=Path,
        action="append",
        help=(
            "Frozen public-only semantic-gate audit JSON. May be repeated for disjoint episode sets. "
            "The replay performs no LLM calls: verified semantic Full selects Full, and every "
            "other semantic outcome selects Fixed Warm."
        ),
    )
    return parser.parse_args()


def replay_episode(args: argparse.Namespace, episode: dict[str, Any], seed_values: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = str(episode["episode_id"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "controller_seed": int(getattr(args, "controller_seed", 0)),
        "domain": episode.get("domain"),
        "family": episode.get("family"),
        "status": "started",
        "stage_count": 0,
        "seed_count": len(seed_values),
        "failure_reason": "",
        "source_run_dir": args.source_run_dir,
    }
    try:
        source_summary = load_source_summary(args, episode_id)
        source_rows = load_source_rows(args, episode_id)
        source_row = select_source_row(args, source_rows)
        source_rows_by_seed = {int(row.get("run_seed", -1)): row for row in source_rows}
        missing_source_seeds = [seed for seed in seed_values if seed not in source_rows_by_seed]
        if args.use_source_history_population and missing_source_seeds:
            raise ValueError(f"source rows missing seeds for source-history lineage: {missing_source_seeds}")
        updates = list(episode.get("update_stream") or [])
        if args.max_updates is not None:
            updates = updates[: max(0, args.max_updates)]
        source_stage_records = source_stage_records_for_episode(source_summary, episode_id)
        source_updates = list(source_row.get("update_results") or [])
        semantic_decisions = load_frozen_semantic_decisions(args.semantic_decisions_json or [])
        restart_builder = ScaffoldRestartSeedBuilder()
        generic_patcher = (
            GenericWorkbenchPatcher(model=args.generic_model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
            if args.generic_workbench
            else None
        )
        generic_builder = (
            GenericWorkbenchBuilder(model=args.generic_model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
            if args.generic_workbench and not args.generic_from_typed_scaffold
            else None
        )
        replay_count = min(max(0, int(args.replay_updates or 0)), len(updates), len(source_updates))
        metric_reference_episode = (
            load_reference_episode(Path(args.reference_jsonl), episode_id)
            if args.record_metric_trace
            else None
        )
        metric_context = build_metric_trace_context(
            episode,
            reference_episode=metric_reference_episode,
        )

        if args.rehydrate_replay_prefix and replay_count > 0:
            public_context, project, per_seed_results, run_rows, stage_records, cumulative_tokens = rehydrate_replay_prefix(
                args=args,
                episode=episode,
                episode_id=episode_id,
                seed_values=seed_values,
                source_rows=source_rows,
                source_stage_records=source_stage_records,
                source_updates=source_updates,
                replay_count=replay_count,
            )
        else:
            public_context = public_context_with_loaded_tables(episode)
            if args.generic_workbench:
                if args.generic_from_typed_scaffold:
                    typed_initial_project = compile_stage_project(
                        episode_id,
                        public_context,
                        source_row["setup_code"],
                        source_row["fitness_code"],
                    )
                    project = compile_generic_workbench(
                        episode_id,
                        public_context,
                        generic_artifact_from_scaffold(typed_initial_project),
                        smoke_test=True,
                    )
                else:
                    assert generic_builder is not None
                    project = generic_builder.build_project(
                        task_id=episode_id,
                        public_problem=str(episode.get("public_initial_problem") or ""),
                        public_context=public_context,
                        max_repairs=args.generic_max_patch_repairs,
                    )
            else:
                project = compile_stage_project(
                    episode_id,
                    public_context,
                    source_row["setup_code"],
                    source_row["fitness_code"],
                )
            per_seed_results: dict[int, EvolutionResult] = {}
            for seed in seed_values:
                recorder = make_metric_history_recorder(
                    args=args,
                    metric_context=metric_context,
                    stage_index=0,
                    update_id="initial",
                    previous_solution=None,
                )
                per_seed_results[seed] = project.run(replay_evolution_config(args, seed, stage_index=0, history_metric_recorder=recorder))
            initial_generic_usage = sum_trace_usage(project.trace.usage) if args.generic_workbench else {}
            zero_usage = {
                "total_tokens": int(initial_generic_usage.get("total_tokens", 0)),
                "prompt_tokens": int(initial_generic_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(initial_generic_usage.get("completion_tokens", 0)),
                "artifact_replay": 1,
                "generic_build_calls": 1 if initial_generic_usage else 0,
            }
            run_rows = {
                seed: make_seed_run_row(
                    args=args,
                    episode=episode,
                    seed=seed,
                    project=project,
                    initial_result=result,
                    initial_usage=zero_usage,
                    initial_latency=0.0,
                )
                for seed, result in per_seed_results.items()
            }
            for row in run_rows.values():
                if args.generic_workbench:
                    row["agent_mode"] = "liveopt_workbench_v1__without_tss_generic_workbench_matched"
                    row["module_ablation"] = "without_tss_generic_workbench_matched"
                    row["restart_policy"] = "frozen_verified_semantic_full_else_fixed_warm"
                    row["generic_workbench_artifact"] = project.artifact_code
                    row["generic_workbench_origin"] = (
                        "typed_scaffold_materialization_diagnostic"
                        if args.generic_from_typed_scaffold
                        else "from_scratch_public_problem_and_bare_abi"
                    )
                if args.landscape_restart_selector:
                    row["restart_policy"] = "sensor_objective_landscape_warm_full"
                row["artifact_replay_updates"] = 0
                row["rehydrated_replay_prefix"] = False
                row["freeze_workbench_after_initial"] = bool(args.freeze_workbench_after_initial)
                maybe_thin_trace_row(args, row)
            stage_records = [initial_stage_record(project, per_seed_results[seed_values[0]], 0.0, zero_usage)]
            cumulative_tokens = dict(zero_usage)

        previous_solutions_by_seed = {
            seed: canonical_previous_solution(
                metric_context=metric_context,
                stage_index=replay_count,
                update_id="initial" if replay_count == 0 else str(source_updates[replay_count - 1].get("update_id") or f"u{replay_count:03d}"),
                result=per_seed_results[seed],
                previous_solution=None,
            )
            for seed in seed_values
        }

        for update_index, update in enumerate(updates[replay_count:], start=replay_count + 1):
            if update_index > len(source_updates):
                raise ValueError(f"source row has no saved artifact for update index {update_index}")
            source_update = source_updates[update_index - 1]
            source_stage = source_stage_records[update_index] if update_index < len(source_stage_records) else {}
            previous_project = project
            data_patch = replay_data_patch(update, source_stage, source_update)
            if data_patch:
                public_context = normalize_public_context_tables(apply_public_context_patch(public_context, data_patch))
            saved_impact = source_impact(source_stage, source_update)
            generic_patch_usage: dict[str, int] = {}
            if args.generic_workbench:
                if bool(saved_impact.get("patch_setup") or saved_impact.get("patch_fitness")):
                    assert generic_patcher is not None
                    project = generic_patcher.patch_project(
                        task_id=episode_id,
                        artifact_code=previous_project.artifact_code,
                        natural_language_update=str(
                            update.get("natural_language_update")
                            or update.get("public_update")
                            or source_update.get("natural_language_update")
                            or ""
                        ),
                        public_context=public_context,
                        max_repairs=args.generic_max_patch_repairs,
                    )
                    generic_patch_usage = sum_trace_usage(project.trace.usage)
                else:
                    project = compile_generic_workbench(
                        episode_id,
                        public_context,
                        previous_project.artifact_code,
                        smoke_test=True,
                    )
            else:
                setup_code = str(source_row["setup_code"] if args.freeze_workbench_after_initial else source_update.get("setup_code") or source_row["setup_code"])
                fitness_code = str(source_row["fitness_code"] if args.freeze_workbench_after_initial else source_update.get("fitness_code") or source_row["fitness_code"])
                project = compile_stage_project(
                    episode_id,
                    public_context,
                    setup_code,
                    fitness_code,
                )
            saved_restart_skill = str(saved_impact.get("restart_skill") or "full_restart_v1")
            normalized_saved_restart_skill = normalize_replay_restart_skill(saved_restart_skill)
            restart_skill = args.force_restart_skill or normalized_saved_restart_skill
            semantic_decision = semantic_decisions.get((episode_id, update_index))
            if args.semantic_decisions_json and semantic_decision is None:
                raise ValueError(
                    f"frozen semantic decision missing for {episode_id} stage {update_index}"
                )
            stage_usage = {
                "total_tokens": int(generic_patch_usage.get("total_tokens", 0)),
                "prompt_tokens": int(generic_patch_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(generic_patch_usage.get("completion_tokens", 0)),
                "artifact_replay": 1,
                "generic_patch_calls": 1 if generic_patch_usage else 0,
            }
            cumulative_tokens = add_usage(cumulative_tokens, stage_usage)
            stage_started = time.perf_counter()
            for seed in seed_values:
                if args.use_source_history_population:
                    previous_result, previous_policy, previous_source_update_id = source_history_previous_result(
                        source_rows_by_seed[seed],
                        update_index,
                    )
                    if (
                        args.require_source_final_population
                        and previous_result.metadata.get("source_population_kind") != "final_population"
                    ):
                        raise ValueError(
                            f"source {episode_id} seed {seed} t{update_index - 1:02d} lacks final_population"
                        )
                else:
                    previous_result = per_seed_results[seed]
                    previous_policy = (
                        "rehydrated_shared_prefix"
                        if args.rehydrate_replay_prefix and update_index == replay_count + 1
                        else "initial_same_artifact"
                        if update_index == 1
                        else (
                            "frozen_semantic_selector_history"
                            if args.semantic_decisions_json
                            else (args.force_restart_skill or "saved_adaptive")
                        )
                    )
                    previous_source_update_id = None
                if args.stateful_record:
                    objective_names = list((project.problem_spec or {}).get("objective_names") or [])
                    previous_result = accepted_result_record(
                        previous_result,
                        pareto=len(objective_names) > 1,
                    )
                    previous_policy = f"{previous_policy}_accepted_only"
                seed_restart_skill = restart_skill
                restart_selection_reason = ""
                objective_shift_record = None
                objective_shift = None
                landscape_decision_record = None
                if args.record_objective_shift_diagnostics:
                    objective_shift = estimate_objective_space_shift(
                        previous_project=previous_project,
                        project=project,
                        previous_result=previous_result,
                        population_size=args.population_size,
                        seed=seed,
                    )
                    objective_shift_record = objective_shift.to_record()
                if args.landscape_restart_selector:
                    landscape_decision = select_restart_skill_from_landscape_change(
                        previous_project=previous_project,
                        project=project,
                        previous_result=previous_result,
                        seed=seed,
                        stage_index=update_index,
                    )
                    landscape_decision_record = landscape_decision.to_record()
                    seed_restart_skill = landscape_decision.selected_restart_skill
                    restart_selection_reason = landscape_decision.reason
                if semantic_decision is not None:
                    seed_restart_skill, restart_selection_reason = select_restart_skill_from_semantic_evidence(
                        semantic_decision=semantic_decision,
                    )
                if args.stateful_record:
                    seed_restart_skill = "warm_restart_v1"
                    restart_selection_reason = "stateful_record_fixed_warm"
                if args.generic_workbench:
                    initial_genomes, restart_metadata = generic_restart_seeds(
                        project=project,
                        previous_result=previous_result,
                        restart_skill=seed_restart_skill,
                        population_size=args.population_size,
                        seed=seed,
                        decoded_solution_transfer=not args.generic_from_typed_scaffold,
                    )
                    if not args.generic_from_typed_scaffold and seed_restart_skill == "warm_restart_v1":
                        restart_metadata["source_transfer_kind"] = (
                            "decoded_liveopt_solution_reencoding"
                            if args.use_source_history_population
                            else "decoded_own_solution_reencoding"
                        )
                else:
                    initial_genomes, restart_metadata = restart_builder.build(
                        restart_skill=seed_restart_skill,
                        previous_result=previous_result,
                        segments=project.segments,
                        population_size=args.population_size,
                        seed=seed,
                        evaluate=project.evaluate,
                        data=project.data,
                        objective_shift=None,
                    )
                recorder = make_metric_history_recorder(
                    args=args,
                    metric_context=metric_context,
                    stage_index=update_index,
                    update_id=str(update.get("update_id") or source_update.get("update_id") or f"u{update_index:03d}"),
                    previous_solution=previous_solutions_by_seed.get(seed),
                )
                result = project.run(
                    replay_evolution_config(args, seed, stage_index=update_index, history_metric_recorder=recorder),
                    initial_genomes=initial_genomes,
                )
                result.metadata = dict(result.metadata)
                restart_metadata = dict(restart_metadata)
                restart_metadata["artifact_replay"] = True
                restart_metadata["rehydrated_replay_prefix"] = bool(args.rehydrate_replay_prefix and replay_count > 0)
                restart_metadata["use_source_history_population"] = bool(args.use_source_history_population)
                restart_metadata["source_restart_skill"] = saved_impact.get("restart_skill")
                restart_metadata["normalized_source_restart_skill"] = normalized_saved_restart_skill
                restart_metadata["forced_restart_skill"] = args.force_restart_skill
                restart_metadata["semantic_restart_gate"] = (
                    semantic_decision.to_record() if semantic_decision is not None else {}
                )
                restart_metadata["restart_selection_rule"] = (
                    "forced_restart_skill"
                    if args.force_restart_skill
                    else "sensor_objective_landscape_warm_full"
                    if args.landscape_restart_selector
                    else "verified_semantic_full_else_fixed_warm"
                    if semantic_decision is not None
                    else "saved_artifact_policy"
                )
                restart_metadata["restart_selection_reason"] = restart_selection_reason
                if objective_shift_record is not None:
                    restart_metadata["objective_space_shift"] = objective_shift_record
                if landscape_decision_record is not None:
                    restart_metadata["landscape_restart_decision"] = landscape_decision_record
                restart_metadata["freeze_workbench_after_initial"] = bool(args.freeze_workbench_after_initial)
                restart_metadata["memory_mode"] = "stateful_record" if args.stateful_record else "lsm"
                restart_metadata["workbench_interface"] = (
                    (
                        "generic_materialized_callbacks_v1"
                        if args.generic_from_typed_scaffold
                        else "generic_blank_callbacks_v1"
                    )
                    if args.generic_workbench
                    else "typed_segment_scaffold_v1"
                )
                restart_metadata["generic_workbench_origin"] = (
                    "typed_scaffold_materialization_diagnostic"
                    if args.generic_workbench and args.generic_from_typed_scaffold
                    else "from_scratch_public_problem_and_bare_abi"
                    if args.generic_workbench
                    else ""
                )
                restart_metadata["stateful_record_fields"] = (
                    ["cumulative_public_context", "accepted_candidate_or_archive"]
                    if args.stateful_record
                    else []
                )
                restart_metadata["population_lineage"] = {
                    "previous_stage_index": update_index - 1,
                    "previous_population_policy": previous_policy,
                    "previous_source_update_id": previous_source_update_id,
                    "current_restart_policy": seed_restart_skill,
                    "uses_adaptive_population": bool(args.use_source_history_population),
                    "history_origin": (
                        "source_liveopt_same_seed" if args.use_source_history_population else "own_previous_stage_same_seed"
                    ),
                    "lineage_rule": (
                        "previous_result is rehydrated from the source LiveOpt run for the same seed, using the exact saved final_population when available"
                        if args.use_source_history_population
                        else "previous_result is the immediately preceding result produced by this replay run for the same seed"
                    ),
                    "source_previous_metadata": (
                        dict(previous_result.metadata)
                        if args.use_source_history_population and isinstance(previous_result.metadata, dict)
                        else {}
                    ),
                }
                restart_metadata["data_patch"] = data_patch
                result.metadata["restart"] = restart_metadata
                per_seed_results[seed] = result
                previous_solutions_by_seed[seed] = canonical_previous_solution(
                    metric_context=metric_context,
                    stage_index=update_index,
                    update_id=str(update.get("update_id") or source_update.get("update_id") or f"u{update_index:03d}"),
                    result=result,
                    previous_solution=previous_solutions_by_seed.get(seed),
                )
                append_update_to_seed_row(
                    run_rows[seed],
                    update,
                    project,
                    result,
                    stage_usage,
                    time.perf_counter() - stage_started,
                    cumulative_tokens,
                    impact={
                        **saved_impact,
                        "restart_skill": seed_restart_skill,
                        "restart_policy_override": bool(args.force_restart_skill),
                        "landscape_restart_selector": bool(args.landscape_restart_selector),
                        "semantic_verified_full_vote": bool(
                            semantic_decision and semantic_decision.verified_full_vote
                        ),
                        "restart_selection_reason": restart_selection_reason,
                        "freeze_workbench_after_initial": bool(args.freeze_workbench_after_initial),
                        "stateful_record": bool(args.stateful_record),
                        "generic_workbench": bool(args.generic_workbench),
                        "generic_workbench_origin": (
                            "typed_scaffold_materialization_diagnostic"
                            if args.generic_workbench and args.generic_from_typed_scaffold
                            else "from_scratch_public_problem_and_bare_abi"
                            if args.generic_workbench
                            else ""
                        ),
                    },
                    restart_metadata=restart_metadata,
                )
                maybe_thin_trace_row(args, run_rows[seed])
            best_seed = seed_values[0]
            stage_records.append(
                {
                    "update_id": str(update.get("update_id") or f"u{update_index:03d}"),
                    "impact": {
                        **saved_impact,
                        "restart_skill": per_seed_results[best_seed].metadata.get("restart", {}).get("restart_skill", restart_skill),
                        "restart_policy_override": bool(args.force_restart_skill),
                        "landscape_restart_selector": bool(args.landscape_restart_selector),
                        "semantic_verified_full_vote": bool(
                            (per_seed_results[best_seed].metadata.get("restart", {}).get("semantic_restart_gate") or {}).get("verified_full_vote")
                        ),
                        "restart_selection_reason": per_seed_results[best_seed].metadata.get("restart", {}).get("restart_selection_reason", ""),
                        "freeze_workbench_after_initial": bool(args.freeze_workbench_after_initial),
                    },
                    "restart": per_seed_results[best_seed].metadata.get("restart", {}),
                    "segments": [segment.kind for segment in project.segments],
                    "solver_mode": (project.problem_spec or {}).get("solver_mode"),
                    "feasible": bool(per_seed_results[best_seed].best.result.feasible) if per_seed_results[best_seed].best else False,
                    "objective": per_seed_results[best_seed].best.result.scalar if per_seed_results[best_seed].best else None,
                    "archive_count": len(per_seed_results[best_seed].archive),
                    "population_count": len(per_seed_results[best_seed].population),
                    "latency_seconds": time.perf_counter() - stage_started,
                    "token_usage": stage_usage,
                    "patch_errors": [],
                }
            )
        rows = [run_rows[seed] for seed in seed_values]
        record.update(
            {
                "status": "completed",
                "stage_count": 1 + len(updates),
                "rows": len(rows),
                "latency_seconds": time.perf_counter() - started,
                "initial_segments": [segment.kind for segment in project.segments],
                "artifact_replay_updates": replay_count,
                "rehydrated_replay_prefix": bool(args.rehydrate_replay_prefix and replay_count > 0),
                "use_source_history_population": bool(args.use_source_history_population),
                "freeze_workbench_after_initial": bool(args.freeze_workbench_after_initial),
                "stateful_record": bool(args.stateful_record),
                "generic_workbench": bool(args.generic_workbench),
                "landscape_restart_selector": bool(args.landscape_restart_selector),
                "generic_workbench_origin": (
                    "typed_scaffold_materialization_diagnostic"
                    if args.generic_workbench and args.generic_from_typed_scaffold
                    else "from_scratch_public_problem_and_bare_abi"
                    if args.generic_workbench
                    else ""
                ),
                "semantic_decisions_json": [str(path) for path in (args.semantic_decisions_json or [])],
                "stage_records": stage_records,
            }
        )
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
                "latency_seconds": time.perf_counter() - started,
            }
        )
    return record, rows


def load_frozen_semantic_decisions(
    paths: list[Path],
) -> dict[tuple[str, int], SemanticRestartDecision]:
    """Load immutable semantic choices and reject missing or conflicting provenance."""

    decisions: dict[tuple[str, int], SemanticRestartDecision] = {}
    source_records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"semantic decision file has no rows list: {path}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            episode_id = str(row.get("episode_id") or "")
            stage = int(row.get("stage") or 0)
            decision_record = row.get("decision") or {}
            if not episode_id or stage <= 0 or not isinstance(decision_record, dict):
                raise ValueError(f"invalid semantic decision row in {path}: {row!r}")
            decision = SemanticRestartDecision.from_record(decision_record)
            key = (episode_id, stage)
            canonical = decision.to_record()
            if key in decisions and source_records[key] != canonical:
                raise ValueError(
                    f"conflicting frozen semantic decisions for {episode_id} stage {stage}"
                )
            decisions[key] = decision
            source_records[key] = canonical
    return decisions


def compile_stage_project(episode_id: str, public_context: dict[str, Any], setup_code: str, fitness_code: str):
    return compile_scaffold_project(
        episode_id,
        public_context,
        {"setup.py": setup_code, "fitness.py": fitness_code},
        ScaffoldTrace(model="artifact_replay"),
    )


def sum_trace_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
            totals[key] = totals.get(key, 0) + int(record.get(key, 0) or 0)
    return totals


def replay_evolution_config(
    args: argparse.Namespace,
    seed: int,
    *,
    stage_index: int,
    history_metric_recorder=None,
) -> EvolutionConfig:
    population_size = args.population_size
    if stage_index == 0 and args.initial_population_size is not None:
        population_size = args.initial_population_size
    generations = args.initial_generations if stage_index == 0 and args.initial_generations is not None else args.generations
    return EvolutionConfig(
        population_size=int(population_size),
        generations=int(generations),
        seed=seed,
        archive_limit=args.archive_limit,
        variation_mode=str(args.variation_mode),
        structured_initialization=True,
        record_metric_history=bool(args.record_metric_trace),
        record_archive_history=bool(args.record_trace_archives),
        history_interval=max(1, int(args.metric_trace_interval or 1)),
        history_archive_limit=max(1, int(args.metric_trace_archive_limit or args.archive_limit)),
        history_metric_recorder=history_metric_recorder,
        reference_free_early_stopping=bool(args.reference_free_early_stop),
        early_stop_min_generations=int(args.early_stop_min_generations),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_hv_epsilon=float(args.early_stop_hv_epsilon),
        early_stop_scalar_epsilon=float(args.early_stop_scalar_epsilon),
        early_stop_epsilon_box=float(args.early_stop_epsilon_box),
        early_stop_min_archive_size=int(args.early_stop_min_archive_size),
    )


def build_metric_trace_context(
    episode: dict[str, Any],
    *,
    reference_episode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark = "NLDO"
    scorer_benchmark = effective_benchmark(benchmark, episode)
    domain = str(episode.get("domain") or "")
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    states = [copy.deepcopy(state)]
    oracle = episode.get("hidden_update_oracle") or []
    for item in oracle:
        state = apply_hidden_delta(scorer_benchmark, domain, state, item.get("hidden_delta") or {})
        states.append(copy.deepcopy(state))
    return {
        "benchmark": benchmark,
        "scorer_benchmark": scorer_benchmark,
        "domain": domain,
        "family": str(episode.get("family") or ""),
        "episode": episode,
        "states": states,
        "reference_steps": reference_trajectory_for_payload(
            scorer_benchmark,
            domain,
            reference_episode or episode,
        ),
        "metric_reference_episode_id": str((reference_episode or episode).get("episode_id") or ""),
    }


def load_reference_episode(path: Path, episode_id: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"metric-trace reference JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("episode_id") or "") == str(episode_id):
                return row
    raise ValueError(f"metric-trace reference has no episode {episode_id}: {path}")


def make_metric_history_recorder(
    *,
    args: argparse.Namespace,
    metric_context: dict[str, Any],
    stage_index: int,
    update_id: str,
    previous_solution: dict[str, Any] | None,
):
    if not args.record_metric_trace:
        return None
    state = metric_state_for_stage(metric_context, stage_index)
    reference_step = reference_step_for_stage(metric_context["reference_steps"], update_id, stage_index)
    limit = max(1, int(args.metric_trace_archive_limit or args.archive_limit))

    def recorder(generation, population, archive, multi):
        archive_records = [_archive_metric_record(candidate) for candidate in list(archive or population)[:limit]]
        solution = first_solution_from_archive(archive_records)
        metric = score_solution(
            metric_context["scorer_benchmark"],
            metric_context["domain"],
            metric_context["family"],
            state,
            solution,
            previous_solution,
            reference_step,
            archive_records,
        )
        return {
            "hv": metric.get("hv"),
            "reference_hv": metric.get("reference_hv"),
            "normalized_hv": metric.get("normalized_hv"),
            "igd": metric.get("igd"),
            "igd_score": metric.get("igd_score"),
            "ideal_gap": metric.get("ideal_gap"),
            "normalized_score": metric.get("normalized_score"),
            "true_pass": bool(metric.get("feasible")),
            "archive_size": len(archive_records),
        }

    return recorder


def canonical_previous_solution(
    *,
    metric_context: dict[str, Any],
    stage_index: int,
    update_id: str,
    result: EvolutionResult,
    previous_solution: dict[str, Any] | None,
) -> dict[str, Any] | None:
    state = metric_state_for_stage(metric_context, stage_index)
    reference_step = reference_step_for_stage(metric_context["reference_steps"], update_id, stage_index)
    archive_records = [_archive_metric_record(candidate) for candidate in result.archive]
    solution = result.best.result.solution if result.best and result.best.result else first_solution_from_archive(archive_records)
    metric = score_solution(
        metric_context["scorer_benchmark"],
        metric_context["domain"],
        metric_context["family"],
        state,
        solution or {},
        previous_solution,
        reference_step,
        archive_records,
    )
    return metric.get("canonical_solution") or solution or previous_solution


def metric_state_for_stage(metric_context: dict[str, Any], stage_index: int) -> dict[str, Any]:
    states = metric_context.get("states") or [{}]
    index = min(max(0, int(stage_index)), len(states) - 1)
    return copy.deepcopy(states[index])


def first_solution_from_archive(archive_records: list[dict[str, Any]]) -> dict[str, Any]:
    for item in archive_records:
        if isinstance(item, dict) and isinstance(item.get("solution"), dict) and item["solution"]:
            return item["solution"]
    return {}


def maybe_thin_trace_row(args: argparse.Namespace, row: dict[str, Any]) -> None:
    """Drop bulky final populations from metric-trace replay rows.

    The final candidate archive is kept because the reference evaluator uses it
    for HV/IGD. The in-memory EvolutionResult still carries the population for
    subsequent restarts; this only thins the serialized replay artifact.
    """

    should_thin = bool(getattr(args, "drop_final_population", False)) or bool(
        args.record_metric_trace and not args.record_trace_archives
    )
    if not should_thin:
        return

    if args.retain_final_population:
        return

    retained_stages = {int(stage) for stage in getattr(args, "retain_final_population_stage", [])}

    def thin_solver_result(solver_result: dict[str, Any] | None) -> None:
        if not isinstance(solver_result, dict):
            return
        metadata = solver_result.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("final_population", None)

    if 0 not in retained_stages:
        thin_solver_result(row.get("initial_solver_result"))
    for stage_index, update in enumerate(row.get("update_results") or [], start=1):
        if isinstance(update, dict):
            if stage_index not in retained_stages:
                thin_solver_result(update.get("solver_result"))


def normalize_replay_restart_skill(saved_restart_skill: str) -> str:
    if saved_restart_skill in RESTART_SKILLS:
        return saved_restart_skill
    mapped = LEGACY_RESTART_SKILL_REMAP.get(saved_restart_skill)
    if mapped and mapped in RESTART_SKILLS:
        return mapped
    raise ValueError(f"unsupported saved restart_skill under restricted policy: {saved_restart_skill}")


def rehydrate_replay_prefix(
    *,
    args: argparse.Namespace,
    episode: dict[str, Any],
    episode_id: str,
    seed_values: list[int],
    source_rows: list[dict[str, Any]],
    source_stage_records: list[dict[str, Any]],
    source_updates: list[dict[str, Any]],
    replay_count: int,
) -> tuple[dict[str, Any], Any, dict[int, EvolutionResult], dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows_by_seed = {int(row.get("run_seed", -1)): row for row in source_rows}
    missing = [seed for seed in seed_values if seed not in rows_by_seed]
    if missing:
        raise ValueError(f"source rows missing seeds for rehydration: {missing}")
    if replay_count > len(source_updates):
        raise ValueError(f"cannot rehydrate {replay_count} updates from {len(source_updates)} source updates")

    public_context = public_context_with_loaded_tables(episode)
    for update_index in range(1, replay_count + 1):
        source_update = source_updates[update_index - 1]
        source_stage = source_stage_records[update_index] if update_index < len(source_stage_records) else {}
        public_update = (episode.get("update_stream") or [])[update_index - 1]
        data_patch = replay_data_patch(public_update, source_stage, source_update)
        if data_patch:
            public_context = normalize_public_context_tables(apply_public_context_patch(public_context, data_patch))

    first_row = source_rows[0]
    stage_source = source_updates[replay_count - 1]
    project = compile_stage_project(
        episode_id,
        public_context,
        str(stage_source.get("setup_code") or first_row["setup_code"]),
        str(stage_source.get("fitness_code") or first_row["fitness_code"]),
    )
    per_seed_results = {
        seed: evolution_result_from_source_update(rows_by_seed[seed]["update_results"][replay_count - 1])
        for seed in seed_values
    }
    run_rows: dict[int, dict[str, Any]] = {}
    for seed in seed_values:
        row = json.loads(json.dumps(rows_by_seed[seed]))
        row["update_results"] = list(row.get("update_results") or [])[:replay_count]
        row["artifact_replay_updates"] = replay_count
        row["rehydrated_replay_prefix"] = True
        row["source_run_dir"] = args.source_run_dir
        row["freeze_workbench_after_initial"] = bool(args.freeze_workbench_after_initial)
        run_rows[seed] = row

    stage_records = json.loads(json.dumps(source_stage_records[: replay_count + 1]))
    for stage in stage_records:
        stage["artifact_replay"] = True
        stage["rehydrated_replay_prefix"] = True

    cumulative_tokens = dict(run_rows[seed_values[0]].get("token_usage") or {})
    if not cumulative_tokens:
        cumulative_tokens = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
    cumulative_tokens["artifact_replay"] = int(cumulative_tokens.get("artifact_replay", 0)) + replay_count
    return public_context, project, per_seed_results, run_rows, stage_records, cumulative_tokens


def load_source_summary(args: argparse.Namespace, episode_id: str) -> dict[str, Any]:
    root = Path(args.source_run_dir)
    path = root / "replay_jobs" / episode_id / "summary.json"
    if not path.exists():
        direct = root / "summary.json"
        if direct.exists():
            path = direct
    if not path.exists():
        return {
            "summary_missing": True,
            "episode_records": [
                {
                    "episode_id": episode_id,
                    "stage_records": [],
                }
            ],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_rows(args: argparse.Namespace, episode_id: str) -> list[dict[str, Any]]:
    root = Path(args.source_run_dir)
    path = root / "replay_jobs" / episode_id / "NLDO" / "evo2_limit0.jsonl"
    if not path.exists():
        direct = root / "NLDO" / "evo2_limit0.jsonl"
        if direct.exists():
            path = direct
    if not path.exists():
        raise FileNotFoundError(f"source replay jsonl not found: {path}")
    rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(fallback_rows) < 1:
                fallback_rows.append(row)
            if str(row.get("episode_id") or row.get("base_instance_id")) == str(episode_id):
                rows.append(row)
    if not rows and not fallback_rows:
        raise ValueError(f"source replay jsonl has no rows: {path}")
    return rows or fallback_rows


def select_source_row(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if int(row.get("run_seed", -1)) == args.source_seed:
            return row
    return rows[0]


def source_stage_records_for_episode(summary: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    records = summary.get("episode_records") or []
    if isinstance(records, list):
        for record in records:
            if isinstance(record, dict) and str(record.get("episode_id")) == str(episode_id):
                return list(record.get("stage_records") or [])
        if records and isinstance(records[0], dict):
            return list(records[0].get("stage_records") or [])
    return []


def source_data_patch(source_stage: dict[str, Any], source_update: dict[str, Any]) -> dict[str, Any]:
    """Return the public data patch from either summary stage records or JSONL update artifacts."""

    candidates = [
        ((source_stage.get("restart") or {}).get("data_patch") or {}),
        (source_update.get("restart") or {}).get("data_patch") or {},
        (((source_update.get("solver_result") or {}).get("metadata") or {}).get("restart") or {}).get("data_patch") or {},
    ]
    for patch in candidates:
        if isinstance(patch, dict) and patch:
            return patch
    return {}


def replay_data_patch(
    public_update: dict[str, Any],
    source_stage: dict[str, Any],
    source_update: dict[str, Any],
) -> dict[str, Any]:
    """Use a benchmark-authored public patch when the materializer provides one.

    Late benchmark-construction experiments can change public rows while reusing
    a verified table-driven Workbench and a shared pre-change population.  The
    patch is public and mirrors the natural-language update; hidden oracle data
    is never read here.  Older stages retain their saved artifact patch.
    """

    benchmark_patch = public_update.get("public_data_patch") if isinstance(public_update, dict) else None
    if isinstance(benchmark_patch, dict) and benchmark_patch:
        return copy.deepcopy(benchmark_patch)
    return source_data_patch(source_stage, source_update)


def source_impact(source_stage: dict[str, Any], source_update: dict[str, Any]) -> dict[str, Any]:
    """Return the saved public impact record from either summary or JSONL update artifacts."""

    for impact in [source_stage.get("impact"), source_update.get("impact")]:
        if isinstance(impact, dict) and impact:
            return dict(impact)
    restart = source_update.get("restart") or {}
    if isinstance(restart, dict) and restart.get("restart_skill"):
        return {"restart_skill": restart.get("restart_skill"), "artifact_replay": restart.get("artifact_replay", True)}
    return {}


def load_source_row(args: argparse.Namespace, episode_id: str) -> dict[str, Any]:
    return select_source_row(args, load_source_rows(args, episode_id))


def evolution_result_from_source_update(source_update: dict[str, Any]) -> EvolutionResult:
    solver = source_update.get("solver_result") or {}
    metadata = dict(solver.get("metadata") or {})
    archive_records = list(metadata.get("candidate_archive") or [])
    population_records = list(metadata.get("final_population") or metadata.get("population") or [])
    if not archive_records:
        solution = solver.get("solution") or source_update.get("solution") or {}
        scalar = float(source_update.get("objective_value") or 0.0)
        archive_records = [
            {
                "genome": solution,
                "fitness": {
                    "scalar": scalar,
                    "objectives": [],
                    "feasible": bool(source_update.get("feasible", True)),
                    "solution": solution,
                    "violations": {},
                },
            }
        ]
    if not population_records:
        population_records = list(archive_records)
    candidates = [candidate_from_record(record) for record in archive_records]
    population = [candidate_from_record(record) for record in population_records]
    if not candidates:
        candidates = list(population)
    best = min(candidates, key=lambda candidate: candidate.result.scalar if candidate.result else float("inf"))
    return EvolutionResult(
        best=best,
        population=list(population),
        archive=list(candidates),
        history=list(metadata.get("history") or []),
        metadata={
            "rehydrated_from_source_update": source_update.get("update_id"),
            "source_runtime": metadata.get("runtime"),
            "source_archive_count": len(archive_records),
            "source_population_count": len(population_records),
            "source_population_kind": "final_population" if metadata.get("final_population") else "archive_fallback",
        },
    )


def source_history_previous_result(source_row: dict[str, Any], update_index: int) -> tuple[EvolutionResult, str, str]:
    """Return the source LiveOpt population immediately before a replayed update."""

    if update_index <= 1:
        initial_payload = {
            "update_id": "initial",
            "solver_result": source_row.get("initial_solver_result") or {},
            "solution": source_row.get("initial_candidate") or {},
            "objective_value": source_row.get("initial_objective") or 0.0,
            "feasible": source_row.get("initial_feasible", True),
        }
        return evolution_result_from_source_update(initial_payload), "source_liveopt_initial", "initial"
    source_updates = list(source_row.get("update_results") or [])
    previous_index = update_index - 2
    if previous_index >= len(source_updates):
        raise ValueError(
            f"source row has no previous update artifact for update index {update_index}; "
            f"available updates={len(source_updates)}"
        )
    previous_update = source_updates[previous_index]
    previous_update_id = str(previous_update.get("update_id") or f"u{update_index - 1:03d}")
    return evolution_result_from_source_update(previous_update), "source_liveopt_history", previous_update_id


def candidate_from_record(record: dict[str, Any]) -> Candidate:
    fitness = dict(record.get("fitness") or {})
    result = FitnessResult(
        scalar=float(fitness.get("scalar") if fitness.get("scalar") is not None else fitness.get("base_scalar") or 0.0),
        objectives=[float(value) for value in (fitness.get("objectives") or [])],
        feasible=bool(fitness.get("feasible", True)),
        violations=dict(fitness.get("violations") or {}),
        base_scalar=float(fitness["base_scalar"]) if fitness.get("base_scalar") is not None else None,
        penalty=float(fitness.get("penalty") or 0.0),
        solution=dict(fitness.get("solution") or record.get("solution") or {}),
        diagnostics=dict(fitness.get("diagnostics") or {}),
    )
    return Candidate(
        genome=dict(record.get("genome") or {}),
        result=result,
        rank=int(record.get("rank") or 0),
        crowding_distance=float(record.get("crowding_distance") or 0.0),
    )


def write_artifact_summary(
    summary_path: Path,
    args: argparse.Namespace,
    started: float,
    episode_records: list[dict[str, Any]],
    completed_rows: int,
    replay_path: Path,
) -> dict[str, Any]:
    statuses = [str(item.get("status") or "") for item in episode_records]
    if statuses and all(status == "completed" for status in statuses):
        run_status = "completed"
    elif any(status == "failed" for status in statuses):
        run_status = "partial"
    else:
        run_status = "running"
    summary = {
        "status": run_status,
        "protocol": "liveopt_workbench_dynamic_artifact_replay_v1",
        "mode": "no_llm_artifact_replay",
        "controller_seed": int(getattr(args, "controller_seed", 0)),
        "episodes_jsonl": args.episodes_jsonl,
        "source_run_dir": args.source_run_dir,
        "source_seed": args.source_seed,
        "artifact_replay_updates": int(args.replay_updates or 0),
        "rehydrate_replay_prefix": bool(args.rehydrate_replay_prefix),
        "use_source_history_population": bool(args.use_source_history_population),
        "freeze_workbench_after_initial": bool(args.freeze_workbench_after_initial),
        "landscape_restart_selector": bool(args.landscape_restart_selector),
        "variation_mode": str(args.variation_mode),
        "semantic_decisions_json": [str(path) for path in (args.semantic_decisions_json or [])],
        "episode_count": len(episode_records),
        "completed_episode_count": sum(1 for item in episode_records if item.get("status") == "completed"),
        "formal_replay_jsonl": str(replay_path),
        "completed_seed_rows": completed_rows,
        "budget": {
            "population_size": args.population_size,
            "initial_population_size": args.initial_population_size if args.initial_population_size is not None else args.population_size,
            "initial_generations": args.initial_generations if args.initial_generations is not None else args.generations,
            "generations": args.generations,
            "seed_count": args.seed_count,
            "seed_start": args.seed_start,
            "controller_seed": int(getattr(args, "controller_seed", 0)),
            "archive_limit": args.archive_limit,
            "variation_mode": str(args.variation_mode),
            "record_metric_trace": bool(args.record_metric_trace),
            "metric_trace_interval": int(args.metric_trace_interval),
            "metric_trace_archive_limit": int(args.metric_trace_archive_limit),
            "record_trace_archives": bool(args.record_trace_archives),
            "retain_final_population": bool(args.retain_final_population),
            "retain_final_population_stages": sorted(
                {int(stage) for stage in args.retain_final_population_stage}
            ),
            "drop_final_population": bool(args.drop_final_population),
            "reference_free_early_stop": bool(args.reference_free_early_stop),
            "metric_trace_reference_jsonl": str(args.reference_jsonl),
            "early_stop_min_generations": int(args.early_stop_min_generations),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_hv_epsilon": float(args.early_stop_hv_epsilon),
            "early_stop_scalar_epsilon": float(args.early_stop_scalar_epsilon),
            "early_stop_epsilon_box": float(args.early_stop_epsilon_box),
            "early_stop_min_archive_size": int(args.early_stop_min_archive_size),
            "restart_policy": (
                "sensor_objective_landscape_warm_full"
                if args.landscape_restart_selector
                else "frozen_verified_semantic_full_else_fixed_warm"
                if args.semantic_decisions_json
                else (args.force_restart_skill or "saved_adaptive")
            ),
            "lineage": "source_liveopt_history" if args.use_source_history_population else "replay_arm_history",
            "freeze_workbench_after_initial": bool(args.freeze_workbench_after_initial),
            "stateful_record": bool(args.stateful_record),
            "generic_workbench": bool(args.generic_workbench),
            "generic_workbench_origin": (
                "typed_scaffold_materialization_diagnostic"
                if args.generic_workbench and args.generic_from_typed_scaffold
                else "from_scratch_public_problem_and_bare_abi"
                if args.generic_workbench
                else ""
            ),
            "require_source_final_population": bool(args.require_source_final_population),
            "landscape_restart_selector": bool(args.landscape_restart_selector),
        },
        "episode_records": episode_records,
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
