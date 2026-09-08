#!/usr/bin/env python3
"""Replay the frozen state-binding controller outputs at the formal 200x200 budget."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import LiveOptDynamicRunner
from evo2.agents.liveopt_state_binding import materialize_state_bindings, state_binding_mismatches
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult
from evo2.evaluation.moea_metrics import (
    approximate_hypervolume,
    ideal_anchor_point,
    reference_point,
)
from scripts.analysis.build_liveopt_state_binding_challenge import (
    BENCHMARK,
    BRANCHES,
    SOURCE_PATHS,
    context_at_stage,
    load_source_rows,
    project_at_stage,
    public_history,
    result_at_stage,
    stable_json,
    value_hash,
)
from scripts.analysis.evaluate_reference_metrics import (
    apply_hidden_delta,
    effective_benchmark,
    reference_step_for_stage,
    reference_trajectory_for_payload,
    score_solution,
)
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import load_episodes
from scripts.llm_tests.run_liveopt_state_binding_controller import FrozenPatcher


DEFAULT_CONTROLLER = (
    ROOT
    / "logs/llm_tests/liveopt_state_binding_controller_pilot_v3_20260717/controller_results.jsonl"
)
STRONG_REFERENCE = (
    ROOT
    / "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713"
    / "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
DEFAULT_OUT = ROOT / "outputs/liveopt_state_binding_formal_q0_200x200x3_20260717"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-results", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--branch-id", action="append", default=[])
    parser.add_argument("--memory-view", action="append", default=[])
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--archive-limit", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_controller_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("paraphrase_index", 0)) != 0 or int(row.get("numerical_seed", 0)) != 0:
                continue
            rows[(str(row["branch_id"]), str(row["memory_view"]))] = row
    return rows


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            (str(row["branch_id"]), str(row["memory_view"]), int(row["numerical_seed"]))
            for row in (json.loads(line) for line in handle if line.strip())
        }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_archive(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def candidates(result: EvolutionResult):
    return list(result.archive or result.population or [result.best])


def canonical_assignments(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "segment": binding.get("segment"),
                "assignments": dict(sorted((binding.get("assignments") or {}).items())),
            }
            for binding in bindings
        ],
        key=stable_json,
    )


def hidden_state_at_stage(episode: dict[str, Any], source_stage: int) -> dict[str, Any]:
    scorer = effective_benchmark("NLDO", episode)
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    oracle = list(episode.get("hidden_update_oracle") or [])
    for index in range(source_stage):
        state = apply_hidden_delta(
            scorer,
            str(episode.get("domain") or ""),
            state,
            (oracle[index].get("hidden_delta") or {}),
        )
    return state


def independent_metric(
    *,
    episode: dict[str, Any],
    source_stage: int,
    source_result: EvolutionResult,
    selected_candidates: list[Any],
) -> dict[str, Any]:
    scorer = effective_benchmark("NLDO", episode)
    domain = str(episode.get("domain") or "")
    family = str(episode.get("family") or "")
    state = hidden_state_at_stage(episode, source_stage)
    references = reference_trajectory_for_payload(scorer, domain, episode)
    reference_step = reference_step_for_stage(references, None, source_stage)
    archive = [
        {"solution": copy.deepcopy(candidate.result.solution or {})}
        for candidate in selected_candidates
        if candidate.result
    ]
    final_solution = archive[0]["solution"] if archive else {}
    return score_solution(
        scorer,
        domain,
        family,
        state,
        final_solution,
        copy.deepcopy(source_result.best.result.solution or {}),
        reference_step,
        archive,
    )


def add_oracle_relative_hv(rows_path: Path, enriched_path: Path, summary_path: Path) -> dict[str, Any]:
    with rows_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    boxes = {}
    for row in rows:
        if row.get("memory_view") != "oracle":
            continue
        items = [
            {"objectives": item}
            for item in (row.get("hidden_compliant_objectives") or [])
            if isinstance(item, dict)
        ]
        names = list(row.get("objective_names") or [])
        if not items or not names:
            continue
        boxes[(row["branch_id"], int(row["numerical_seed"]))] = {
            "objective_names": names,
            "ideal": ideal_anchor_point(items, names, margin=0.1),
            "reference": reference_point(items, names, margin=0.5),
        }
    enriched = []
    for row in rows:
        box = boxes.get((row["branch_id"], int(row["numerical_seed"])))
        items = [
            {"objectives": item}
            for item in (row.get("hidden_compliant_objectives") or [])
            if isinstance(item, dict)
        ]
        branch_hv = (
            approximate_hypervolume(
                items,
                box["objective_names"],
                ref=box["reference"],
                ideal=box["ideal"],
                samples=0,
            )
            if box and items
            else 0.0
        )
        enriched.append(
            {
                **row,
                "branch_hv": branch_hv,
                "branch_hv_box": box or {},
                "task_adjusted_branch_hv": (
                    branch_hv
                    if bool(row.get("binding_exact")) and bool(row.get("hidden_contract_pass"))
                    else 0.0
                ),
            }
        )
    oracle_hv = {
        (row["branch_id"], int(row["numerical_seed"])): float(row.get("branch_hv") or 0.0)
        for row in enriched
        if row.get("memory_view") == "oracle"
    }
    for row in enriched:
        denom = oracle_hv.get((row["branch_id"], int(row["numerical_seed"])), 0.0)
        row["oracle_relative_hv"] = (
            float(row.get("task_adjusted_branch_hv") or 0.0) / denom if denom > 1e-12 else None
        )
    enriched_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in enriched),
        encoding="utf-8",
    )
    grouped = {}
    for view in sorted({row["memory_view"] for row in enriched}):
        subset = [row for row in enriched if row["memory_view"] == view]
        ratios = [float(row["oracle_relative_hv"]) for row in subset if row.get("oracle_relative_hv") is not None]
        grouped[view] = {
            "rows": len(subset),
            "binding_exact_rate": sum(bool(row["binding_exact"]) for row in subset) / max(1, len(subset)),
            "hidden_contract_pass_rate": sum(bool(row["hidden_contract_pass"]) for row in subset)
            / max(1, len(subset)),
            "mean_archive_compliance_rate": sum(float(row["archive_compliance_rate"]) for row in subset)
            / max(1, len(subset)),
            "mean_task_adjusted_oracle_relative_hv": sum(ratios) / len(ratios) if ratios else None,
        }
    start_groups: dict[tuple[str, int], set[str]] = {}
    for row in enriched:
        start_groups.setdefault((row["branch_id"], int(row["numerical_seed"])), set()).add(
            str(row["restart_sha256"])
        )
    summary = {
        "protocol": "state_binding_formal_replay_v1",
        "row_count": len(enriched),
        "by_memory_view": grouped,
        "all_restart_populations_matched": all(len(values) == 1 for values in start_groups.values()),
        "restart_group_count": len(start_groups),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "formal_results.jsonl"
    enriched_path = args.out_dir / "formal_results_enriched.jsonl"
    archives_path = args.out_dir / "formal_archives.jsonl.gz"
    summary_path = args.out_dir / "summary.json"
    if not args.resume:
        for path in (rows_path, enriched_path, archives_path, summary_path):
            if path.exists():
                path.unlink()
    completed = load_completed(rows_path)
    controller_rows = load_controller_rows(args.controller_results)
    benchmark_episodes = {episode["episode_id"]: episode for episode in load_episodes(BENCHMARK)}
    reference_episodes = {episode["episode_id"]: episode for episode in load_episodes(STRONG_REFERENCE)}
    source_rows = {episode_id: load_source_rows(path) for episode_id, path in SOURCE_PATHS.items()}
    selected_specs = [
        spec for spec in BRANCHES if not args.branch_id or spec.branch_id in set(args.branch_id)
    ]
    selected_views = tuple(args.memory_view or ("full", "ledger_only", "accepted_only", "current_only", "oracle"))
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    failures: list[str] = []

    for spec in selected_specs:
        for seed in seeds:
            source_row = source_rows[spec.episode_id][seed]
            context = context_at_stage(benchmark_episodes[spec.episode_id], source_row, spec.source_stage)
            project = project_at_stage(spec.episode_id, source_row, context, spec.source_stage)
            source_result = result_at_stage(source_row, spec.source_stage)
            history = public_history(source_row, spec.source_stage)
            expected_bindings, expected_diagnostics = materialize_state_bindings(
                [spec.oracle_query], accepted_result=source_result, segments=project.segments
            )
            config = EvolutionConfig(
                population_size=args.population_size,
                generations=args.generations,
                seed=seed,
                archive_limit=args.archive_limit,
                structured_initialization=True,
                reference_free_early_stopping=True,
                early_stop_min_generations=40,
                early_stop_patience=25,
                early_stop_hv_epsilon=5e-4,
                early_stop_scalar_epsilon=5e-4,
                early_stop_epsilon_box=0.01,
                early_stop_min_archive_size=8,
            )
            for view in selected_views:
                key = (spec.branch_id, view, seed)
                if key in completed:
                    continue
                controller = controller_rows.get((spec.branch_id, view))
                if controller is None:
                    failures.append(f"missing controller row for {spec.branch_id}/{view}")
                    continue
                observed_queries = copy.deepcopy(controller.get("observed_queries") or [])
                runner = LiveOptDynamicRunner(
                    project,
                    public_context=context,
                    result=source_result,
                    patcher=FrozenPatcher(),
                    public_update_history=history,
                )
                started = time.perf_counter()
                try:
                    stage = runner.update(
                        update_id=f"{spec.branch_id}-formal",
                        natural_language_update=spec.paraphrases[0],
                        impact={
                            "restart_skill": "warm_restart_v1",
                            "reason": "frozen state-binding controller replay",
                            "state_binding_queries": observed_queries,
                        },
                        memory_view=view,
                        force_restart_skill="warm_restart_v1",
                        config=config,
                        smoke_test=False,
                    )
                    submitted = candidates(stage.result)
                    compliant = [
                        candidate
                        for candidate in submitted
                        if candidate.result
                        and candidate.result.feasible
                        and not state_binding_mismatches(candidate.genome, expected_bindings)
                    ]
                    observed_bindings = list(stage.restart_metadata.get("state_bindings") or [])
                    binding_exact = canonical_assignments(observed_bindings) == canonical_assignments(
                        expected_bindings
                    )
                    metric = independent_metric(
                        episode=reference_episodes[spec.episode_id],
                        source_stage=spec.source_stage,
                        source_result=source_result,
                        selected_candidates=compliant,
                    )
                    contract_pass = bool(submitted) and len(compliant) == len(submitted)
                    filtered_hv = float(metric.get("normalized_hv") or 0.0)
                    hidden_archive = list(metric.get("hidden_candidate_archive") or [])
                    row = {
                        "protocol": "state_binding_formal_replay_v1",
                        "branch_id": spec.branch_id,
                        "episode_id": spec.episode_id,
                        "source_stage": spec.source_stage,
                        "task_type": spec.task_type,
                        "memory_view": view,
                        "numerical_seed": seed,
                        "controller_paraphrase_index": 0,
                        "controller_prompt_sha256": controller.get("controller_prompt_sha256"),
                        "observed_queries": observed_queries,
                        "observed_bindings": observed_bindings,
                        "expected_bindings": expected_bindings,
                        "expected_binding_diagnostics": expected_diagnostics,
                        "binding_exact": binding_exact,
                        "hidden_contract_pass": contract_pass,
                        "submitted_candidate_count": len(submitted),
                        "compliant_candidate_count": len(compliant),
                        "archive_compliance_rate": len(compliant) / max(1, len(submitted)),
                        "filtered_normalized_hv": filtered_hv,
                        "contract_adjusted_normalized_hv": filtered_hv if contract_pass else 0.0,
                        "objective_names": list((reference_episodes[spec.episode_id].get("evaluation") or {}).get("objectives") or []),
                        "hidden_compliant_objectives": [
                            dict(item.get("objectives") or {}) for item in hidden_archive
                        ],
                        "hidden_metric": {
                            key: value
                            for key, value in metric.items()
                            if key not in {"canonical_solution", "hidden_candidate_archive"}
                        },
                        "restart_skill": stage.restart_metadata.get("restart_skill"),
                        "restart_seed_count": len(stage.initial_genomes),
                        "restart_sha256": value_hash(stage.initial_genomes),
                        "population_size": args.population_size,
                        "generation_cap": args.generations,
                        "executed_generations": len(stage.result.history),
                        "early_stopping": (stage.result.metadata or {}).get("early_stopping") or {},
                        "latency_seconds": time.perf_counter() - started,
                    }
                    append_jsonl(rows_path, row)
                    append_archive(
                        archives_path,
                        {
                            "branch_id": spec.branch_id,
                            "memory_view": view,
                            "numerical_seed": seed,
                            "submitted": [
                                {
                                    "genome": candidate.genome,
                                    "solution": candidate.result.solution if candidate.result else {},
                                    "objectives": candidate.result.objectives if candidate.result else [],
                                    "scalar": candidate.result.scalar if candidate.result else None,
                                    "feasible": candidate.result.feasible if candidate.result else False,
                                    "binding_mismatches": state_binding_mismatches(
                                        candidate.genome, expected_bindings
                                    ),
                                }
                                for candidate in submitted
                            ],
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "branch": spec.branch_id,
                                "view": view,
                                "seed": seed,
                                "binding_exact": binding_exact,
                                "contract_pass": contract_pass,
                                "hv": filtered_hv,
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failure = f"{spec.branch_id}/{view}/seed{seed}: {type(exc).__name__}: {exc}"
                    failures.append(failure)
                    print(failure, flush=True)

    summary = (
        add_oracle_relative_hv(rows_path, enriched_path, summary_path)
        if rows_path.exists()
        else {}
    )
    summary["failures"] = failures
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
