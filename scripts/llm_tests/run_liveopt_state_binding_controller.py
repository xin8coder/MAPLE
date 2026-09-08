#!/usr/bin/env python3
"""Run controller-only state-binding controls on frozen P010/P015 branches."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import (
    DynamicUpdateImpact,
    LiveOptDynamicRunner,
    LiveOptUpdateLocalizer,
)
from evo2.agents.liveopt_state_binding import (
    materialize_state_bindings,
    state_binding_mismatches,
)
from evo2.agents.liveopt_workbench_impl import ScaffoldTrace
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult
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
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import load_episodes


DEFAULT_OUT = ROOT / "logs/llm_tests/liveopt_state_binding_controller_pilot_20260717"
CONTROLLER_VIEWS = ("full", "ledger_only", "accepted_only", "current_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--branch-id", action="append", default=[])
    parser.add_argument("--memory-view", action="append", default=[])
    parser.add_argument("--paraphrase-index", type=int, default=0)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--numerical-seed", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


class FrozenPatcher:
    """Keep the formal TSS slots byte-identical in the controller experiment."""

    model = "frozen_tss_slots"

    def patch_slots(self, *, current_slots: dict[str, str], **_: Any):
        return dict(current_slots), ScaffoldTrace(model=self.model)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_completed(path: Path) -> set[tuple[str, str, int, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(
                (
                    str(row.get("branch_id")),
                    str(row.get("memory_view")),
                    int(row.get("paraphrase_index", 0)),
                    int(row.get("numerical_seed", 0)),
                )
            )
    return completed


def masked_controller_inputs(
    memory_view: str,
    result: EvolutionResult,
    history: list[dict[str, Any]],
) -> tuple[EvolutionResult | None, list[dict[str, Any]]]:
    previous_result = result if memory_view in {"full", "accepted_only"} else None
    visible_history = history if memory_view in {"full", "ledger_only"} else []
    return previous_result, visible_history


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


def compliant_candidates(result: EvolutionResult, expected_bindings: list[dict[str, Any]]):
    candidates = list(result.archive or result.population or [result.best])
    return [
        candidate
        for candidate in candidates
        if candidate.result
        and candidate.result.feasible
        and not state_binding_mismatches(candidate.genome, expected_bindings)
    ]


def submitted_candidates(result: EvolutionResult):
    return list(result.archive or result.population or [result.best])


def run_controller_arm(
    *,
    spec,
    memory_view: str,
    text: str,
    project,
    context: dict[str, Any],
    source_result: EvolutionResult,
    history: list[dict[str, Any]],
    localizer: LiveOptUpdateLocalizer,
    config: EvolutionConfig,
) -> dict[str, Any]:
    previous_result, visible_history = masked_controller_inputs(memory_view, source_result, history)
    impact, trace = localizer.classify(
        natural_language_update=text,
        project=project,
        public_context=context,
        previous_result=previous_result,
        public_update_history=visible_history,
    )
    # This experiment isolates state localization.  The formal TSS slots,
    # structured data, restart policy, and numerical budget are fixed.
    raw = dict(impact.raw)
    raw["controller_declared_data_update"] = bool(impact.data_update)
    raw["controller_declared_patch_setup"] = bool(impact.patch_setup)
    raw["controller_declared_patch_fitness"] = bool(impact.patch_fitness)
    frozen_impact = DynamicUpdateImpact(
        data_update=False,
        patch_setup=False,
        patch_fitness=False,
        restart_skill="warm_restart_v1",
        reason=impact.reason,
        raw=raw,
    )
    runner = LiveOptDynamicRunner(
        project,
        public_context=context,
        result=source_result,
        localizer=localizer,
        patcher=FrozenPatcher(),
        public_update_history=history,
    )
    stage = runner.update(
        update_id=f"{spec.branch_id}-q0",
        natural_language_update=text,
        impact=frozen_impact,
        memory_view=memory_view,
        force_restart_skill="warm_restart_v1",
        config=config,
        smoke_test=False,
    )
    return {"stage": stage, "trace": trace, "raw_impact": impact.__dict__}


def run_oracle_arm(
    *,
    spec,
    text: str,
    project,
    context: dict[str, Any],
    source_result: EvolutionResult,
    history: list[dict[str, Any]],
    config: EvolutionConfig,
):
    runner = LiveOptDynamicRunner(
        project,
        public_context=context,
        result=source_result,
        patcher=FrozenPatcher(),
        public_update_history=history,
    )
    return runner.update(
        update_id=f"{spec.branch_id}-oracle",
        natural_language_update=text,
        impact={
            "restart_skill": "warm_restart_v1",
            "reason": "hidden typed oracle",
            "state_binding_queries": [copy.deepcopy(spec.oracle_query)],
        },
        memory_view="oracle",
        force_restart_skill="warm_restart_v1",
        config=config,
        smoke_test=False,
    )


def record_for_stage(
    *,
    spec,
    memory_view: str,
    paraphrase_index: int,
    numerical_seed: int,
    text: str,
    source_result: EvolutionResult,
    project,
    stage,
    trace: ScaffoldTrace,
    raw_impact: dict[str, Any],
) -> dict[str, Any]:
    expected_bindings, expected_diagnostics = materialize_state_bindings(
        [spec.oracle_query], accepted_result=source_result, segments=project.segments
    )
    observed_bindings = list(stage.restart_metadata.get("state_bindings") or [])
    expected_assignments = canonical_assignments(expected_bindings)
    observed_assignments = canonical_assignments(observed_bindings)
    submitted = submitted_candidates(stage.result)
    compliant = compliant_candidates(stage.result, expected_bindings)
    best = min(
        compliant,
        key=lambda candidate: candidate.result.scalar if candidate.result else float("inf"),
        default=None,
    )
    usage = {}
    for item in trace.usage:
        for key, value in (item or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    return {
        "protocol": "state_binding_controller_factorial_v2",
        "branch_id": spec.branch_id,
        "episode_id": spec.episode_id,
        "source_stage": spec.source_stage,
        "task_type": spec.task_type,
        "required_channels": list(spec.required_channels),
        "memory_view": memory_view,
        "paraphrase_index": paraphrase_index,
        "numerical_seed": numerical_seed,
        "natural_language_update": text,
        "controller_input": {
            "accepted_output_visible": memory_view in {"full", "accepted_only", "oracle"},
            "event_ledger_visible": memory_view in {"full", "ledger_only"},
        },
        "controller_raw_impact": raw_impact,
        "controller_prompt_sha256": value_hash(trace.prompts),
        "controller_prompts": trace.prompts,
        "controller_raw_responses": trace.raw_responses,
        "controller_usage": usage,
        "controller_errors": trace.errors,
        "observed_queries": stage.restart_metadata.get("state_binding_queries") or [],
        "observed_bindings": observed_bindings,
        "observed_binding_diagnostics": stage.restart_metadata.get("state_binding_diagnostics") or {},
        "expected_binding_diagnostics": expected_diagnostics,
        "binding_exact": observed_assignments == expected_assignments,
        "expected_assignments": expected_assignments,
        "observed_assignments": observed_assignments,
        "hidden_any_compliant": bool(compliant),
        "hidden_contract_pass": bool(submitted) and len(compliant) == len(submitted),
        "hidden_solve": bool(submitted) and len(compliant) == len(submitted),
        "archive_compliance_rate": len(compliant) / max(1, len(submitted)),
        "hidden_submitted_candidate_count": len(submitted),
        "hidden_compliant_candidate_count": len(compliant),
        "best_compliant_scalar": best.result.scalar if best and best.result else None,
        "best_compliant_objectives": best.result.objectives if best and best.result else [],
        "restart_skill": stage.restart_metadata.get("restart_skill"),
        "restart_seed_count": len(stage.initial_genomes),
        "restart_sha256": value_hash(stage.initial_genomes),
        "population_size": len(stage.result.population),
        "archive_size": len(stage.result.archive),
        "latency_seconds": stage.latency_seconds,
    }


def write_summary(path: Path, rows_path: Path) -> dict[str, Any]:
    rows = []
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    controller_rows = [row for row in rows if row.get("memory_view") != "oracle"]
    by_view = {}
    for view in sorted({row.get("memory_view") for row in rows}):
        subset = [row for row in rows if row.get("memory_view") == view]
        by_view[view] = {
            "rows": len(subset),
            "binding_exact_rate": sum(bool(row.get("binding_exact")) for row in subset) / max(1, len(subset)),
            "hidden_solve_rate": sum(bool(row.get("hidden_solve")) for row in subset) / max(1, len(subset)),
        }
    restart_groups: dict[tuple[str, int, int], set[str]] = {}
    for row in rows:
        key = (row["branch_id"], int(row["paraphrase_index"]), int(row["numerical_seed"]))
        restart_groups.setdefault(key, set()).add(str(row["restart_sha256"]))
    summary = {
        "protocol": "state_binding_controller_factorial_v2",
        "row_count": len(rows),
        "controller_row_count": len(controller_rows),
        "by_memory_view": by_view,
        "all_restart_populations_matched": all(len(hashes) == 1 for hashes in restart_groups.values()),
        "restart_group_count": len(restart_groups),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if not 0 <= args.paraphrase_index <= 2:
        raise ValueError("--paraphrase-index must be 0, 1, or 2")
    selected_branches = [
        spec for spec in BRANCHES if not args.branch_id or spec.branch_id in set(args.branch_id)
    ]
    unknown_branches = sorted(set(args.branch_id) - {spec.branch_id for spec in BRANCHES})
    if unknown_branches:
        raise ValueError(f"unknown branch ids: {unknown_branches}")
    views = tuple(args.memory_view or CONTROLLER_VIEWS)
    unknown_views = sorted(set(views) - set(CONTROLLER_VIEWS))
    if unknown_views:
        raise ValueError(f"unsupported controller memory views: {unknown_views}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "controller_results.jsonl"
    summary_path = args.out_dir / "summary.json"
    if not args.resume:
        for path in (rows_path, summary_path):
            if path.exists():
                path.unlink()
    completed = load_completed(rows_path)
    os.environ["DEEPSEEK_TRACE_DIR"] = str(args.out_dir / "provider_trace")
    os.environ["DEEPSEEK_CACHE_ONLY"] = "1" if args.cache_only else "0"
    os.environ.setdefault("DEEPSEEK_MAX_ATTEMPTS", "3")
    os.environ.setdefault("DEEPSEEK_HARD_TIMEOUT", "360")
    os.environ.setdefault("DEEPSEEK_DEFAULT_UNLIMITED_MAX_TOKENS", "0")
    # Reasoning models may spend more than the localizer's nominal 1.2k
    # output budget before emitting the short final JSON object.
    os.environ.setdefault("DEEPSEEK_PRO_MIN_MAX_TOKENS", "8000")

    episodes = {episode["episode_id"]: episode for episode in load_episodes(BENCHMARK)}
    sources = {episode_id: load_source_rows(path) for episode_id, path in SOURCE_PATHS.items()}
    localizer = LiveOptUpdateLocalizer(model=args.model)
    failures: list[str] = []

    for spec in selected_branches:
        source_row = sources[spec.episode_id][args.numerical_seed]
        context = context_at_stage(episodes[spec.episode_id], source_row, spec.source_stage)
        project = project_at_stage(spec.episode_id, source_row, context, spec.source_stage)
        source_result = result_at_stage(source_row, spec.source_stage)
        history = public_history(source_row, spec.source_stage)
        text = spec.paraphrases[args.paraphrase_index]
        config = EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            seed=args.numerical_seed,
            archive_limit=max(100, args.population_size),
            structured_initialization=False,
        )
        for memory_view in views:
            key = (spec.branch_id, memory_view, args.paraphrase_index, args.numerical_seed)
            if key in completed:
                continue
            started = time.perf_counter()
            try:
                run = run_controller_arm(
                    spec=spec,
                    memory_view=memory_view,
                    text=text,
                    project=project,
                    context=context,
                    source_result=source_result,
                    history=history,
                    localizer=localizer,
                    config=config,
                )
                row = record_for_stage(
                    spec=spec,
                    memory_view=memory_view,
                    paraphrase_index=args.paraphrase_index,
                    numerical_seed=args.numerical_seed,
                    text=text,
                    source_result=source_result,
                    project=project,
                    stage=run["stage"],
                    trace=run["trace"],
                    raw_impact=run["raw_impact"],
                )
                row["wall_seconds"] = time.perf_counter() - started
                append_jsonl(rows_path, row)
                print(
                    json.dumps(
                        {
                            "branch": spec.branch_id,
                            "view": memory_view,
                            "binding_exact": row["binding_exact"],
                            "hidden_solve": row["hidden_solve"],
                            "tokens": row["controller_usage"].get("total_tokens", 0),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failure = f"{spec.branch_id}/{memory_view}: {type(exc).__name__}: {exc}"
                failures.append(failure)
                print(failure, flush=True)

        oracle_key = (spec.branch_id, "oracle", args.paraphrase_index, args.numerical_seed)
        if oracle_key not in completed:
            try:
                oracle_stage = run_oracle_arm(
                    spec=spec,
                    text=text,
                    project=project,
                    context=context,
                    source_result=source_result,
                    history=history,
                    config=config,
                )
                oracle_row = record_for_stage(
                    spec=spec,
                    memory_view="oracle",
                    paraphrase_index=args.paraphrase_index,
                    numerical_seed=args.numerical_seed,
                    text=text,
                    source_result=source_result,
                    project=project,
                    stage=oracle_stage,
                    trace=ScaffoldTrace(model="hidden_typed_oracle"),
                    raw_impact={"oracle": True},
                )
                append_jsonl(rows_path, oracle_row)
            except Exception as exc:  # noqa: BLE001
                failure = f"{spec.branch_id}/oracle: {type(exc).__name__}: {exc}"
                failures.append(failure)
                print(failure, flush=True)

    summary = write_summary(summary_path, rows_path)
    summary["failures"] = failures
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
