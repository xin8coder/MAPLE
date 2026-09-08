#!/usr/bin/env python3
"""Build and audit state-binding branches from frozen P010/P015 trajectories.

The public branch text is written separately from the hidden typed oracle.  No
provider is called here.  The optional pilot runs the hidden oracle with the
same Fixed-Warm restart population used by every future controller arm.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import (
    ScaffoldRestartSeedBuilder,
    apply_public_context_patch,
)
from evo2.agents.liveopt_state_binding import (
    apply_state_bindings,
    materialize_state_bindings,
    state_binding_mismatches,
)
from evo2.agents.liveopt_workbench_impl import ScaffoldProject, ScaffoldTrace, compile_scaffold_project
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult, coerce_fitness_result
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import evolution_result_from_source_update
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (
    load_episodes,
    public_context_with_loaded_tables,
)


BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
SOURCE_PATHS = {
    "NLDO-P010": ROOT
    / "outputs/liveopt_matched_full_p010_p015_200x200x10_allpop_20260714"
    / "replay_jobs/NLDO-P010/NLDO/evo2_limit0.jsonl",
    "NLDO-P015": ROOT
    / "outputs/liveopt_matched_full_retry_p013_p015_200x200x10_allpop_20260714"
    / "replay_jobs/NLDO-P015/NLDO/evo2_limit0.jsonl.gz",
}
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_state_binding_challenge_20260717"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    episode_id: str
    source_stage: int
    task_type: str
    required_channels: tuple[str, ...]
    paraphrases: tuple[str, str, str]
    oracle_query: dict[str, Any]
    indirect_ids: tuple[str, ...]


BRANCHES = (
    BranchSpec(
        branch_id="P010-event-s04",
        episode_id="NLDO-P010",
        source_stage=4,
        task_type="event_only",
        required_channels=("event_ledger",),
        paraphrases=(
            "For the next dispatch window, the customer previously marked as the congestion anchor must be served by the vehicle whose telemetry update changed its emission rate. Treat this as a hard assignment constraint; all other feasibility rules and Pareto objectives stay unchanged.",
            "In the next plan, assign the congestion-anchor customer to the vehicle identified by the earlier emission-rate telemetry event. This assignment is mandatory, while the remaining routing criteria are unchanged.",
            "Keep the original routing objectives, but impose one operational hold: the customer named by the congestion note must use the vehicle from the telemetry correction history.",
        ),
        oracle_query={
            "binding_id": "p010_event",
            "kind": "lock_assignment",
            "segment": "vehicle_assign",
            "assignments": {"O004": "V1"},
            "depends_on": ["event_ledger"],
        },
        indirect_ids=("O004", "V1"),
    ),
    BranchSpec(
        branch_id="P010-accepted-s08",
        episode_id="NLDO-P010",
        source_stage=8,
        task_type="accepted_only",
        required_channels=("accepted_output",),
        paraphrases=(
            "For the next dispatch, keep the currently accepted vehicle assignments of orders O001, O002, and O003. Other orders may move, and the original feasibility rules and Pareto objectives remain unchanged.",
            "Freeze the accepted vehicles of O001, O002, and O003 for one window. Each must stay with its present vehicle, while the remaining routes are optimized as before.",
            "Use the current accepted dispatch to hold O001, O002, and O003 on their present vehicles. No other routing rule is modified.",
        ),
        oracle_query={
            "binding_id": "p010_accepted",
            "kind": "preserve_demand_assignments",
            "segment": "vehicle_assign",
            "demands": ["O001", "O002", "O003"],
            "depends_on": ["accepted_output"],
        },
        indirect_ids=(),
    ),
    BranchSpec(
        branch_id="P010-linked-s10",
        episode_id="NLDO-P010",
        source_stage=10,
        task_type="event_and_accepted",
        required_channels=("event_ledger", "accepted_output"),
        paraphrases=(
            "Preserve the currently accepted vehicle assignment of the customer referred to as the congestion anchor. Other orders may be reassigned, and the original objectives remain unchanged.",
            "Find the customer named by the earlier congestion event, then keep that customer's vehicle exactly as in the current accepted dispatch. Reoptimize the rest of the routes normally.",
            "For this window, the congestion-anchor customer's present vehicle is fixed. Resolve the customer from the event history and the vehicle from the accepted plan; leave all other routing decisions open.",
        ),
        oracle_query={
            "binding_id": "p010_linked",
            "kind": "preserve_demand_assignments",
            "segment": "vehicle_assign",
            "demands": ["O004"],
            "depends_on": ["accepted_output", "event_ledger"],
        },
        indirect_ids=("O004",),
    ),
    BranchSpec(
        branch_id="P015-event-s04",
        episode_id="NLDO-P015",
        source_stage=4,
        task_type="event_only",
        required_channels=("event_ledger",),
        paraphrases=(
            "For the next placement window, assign the first job named when workload group burst-2 was introduced to machine M07. This is a hard assignment constraint; all other feasibility rules and Pareto objectives stay unchanged.",
            "Place the first job from the earlier burst-2 list on M07 in the next schedule. Keep the original placement constraints and optimization goals for every other decision.",
            "Add one operational hold without changing the objectives: the first workload listed under burst-2 must run on M07.",
        ),
        oracle_query={
            "binding_id": "p015_event",
            "kind": "lock_assignment",
            "segment": "job_assignment",
            "assignments": {"J003": "M07"},
            "depends_on": ["event_ledger"],
        },
        indirect_ids=("J003",),
    ),
    BranchSpec(
        branch_id="P015-accepted-s08",
        episode_id="NLDO-P015",
        source_stage=8,
        task_type="accepted_only",
        required_channels=("accepted_output",),
        paraphrases=(
            "For the next placement, keep the currently accepted machine assignments of jobs J012, J013, and J014. Other jobs may move, and the original feasibility rules and Pareto objectives remain unchanged.",
            "Freeze the accepted machines of J012, J013, and J014 for one window. Each must stay on its present machine, while the remaining placements are optimized as before.",
            "Use the accepted schedule to hold J012, J013, and J014 on their present machines. No other placement rule is modified.",
        ),
        oracle_query={
            "binding_id": "p015_accepted",
            "kind": "preserve_demand_assignments",
            "segment": "job_assignment",
            "demands": ["J012", "J013", "J014"],
            "depends_on": ["accepted_output"],
        },
        indirect_ids=(),
    ),
    BranchSpec(
        branch_id="P015-linked-s10",
        episode_id="NLDO-P015",
        source_stage=10,
        task_type="event_and_accepted",
        required_channels=("event_ledger", "accepted_output"),
        paraphrases=(
            "Keep the current accepted machine assignment of every job in the workload group called burst-2. Other jobs may move, and the original objectives remain unchanged.",
            "Resolve burst-2 from the earlier workload event, then preserve each member's machine from the accepted placement. Reoptimize all jobs outside that group normally.",
            "For this window, the present machines of the burst-2 jobs are fixed. Use the event history to identify the jobs and the accepted plan to recover their machines.",
        ),
        oracle_query={
            "binding_id": "p015_linked",
            "kind": "preserve_demand_assignments",
            "segment": "job_assignment",
            "demands": ["J003", "J004", "J005", "J006", "J007"],
            "depends_on": ["accepted_output", "event_ledger"],
        },
        indirect_ids=("J003", "J004", "J005", "J006", "J007"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pilot-population", type=int, default=40)
    parser.add_argument("--pilot-generations", type=int, default=40)
    parser.add_argument("--pilot-seeds", type=int, default=3)
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def value_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_source_rows(path: Path) -> dict[int, dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {int(row["run_seed"]): row for row in rows}


def context_at_stage(episode: dict[str, Any], source_row: dict[str, Any], stage: int) -> dict[str, Any]:
    context = public_context_with_loaded_tables(episode)
    for update in list(source_row.get("update_results") or [])[:stage]:
        context = apply_public_context_patch(context, (update.get("restart") or {}).get("data_patch") or {})
    context = copy.deepcopy(context)
    context["state_binding_challenge"] = {
        "enabled": True,
        "description": "A typed assignment hold explicitly introduced by the branch update.",
    }
    return context


def project_at_stage(
    episode_id: str,
    source_row: dict[str, Any],
    context: dict[str, Any],
    stage: int,
) -> ScaffoldProject:
    update = source_row["update_results"][stage - 1]
    return compile_scaffold_project(
        episode_id,
        context,
        {"setup.py": update["setup_code"], "fitness.py": update["fitness_code"]},
        ScaffoldTrace(model="frozen_formal_artifact"),
    )


def result_at_stage(source_row: dict[str, Any], stage: int) -> EvolutionResult:
    return evolution_result_from_source_update(source_row["update_results"][stage - 1])


def public_history(source_row: dict[str, Any], stage: int) -> list[dict[str, Any]]:
    return [
        {
            "update_id": update.get("update_id"),
            "natural_language_update": update.get("natural_language_update"),
            "public_data_patch": copy.deepcopy((update.get("restart") or {}).get("data_patch") or {}),
        }
        for update in list(source_row.get("update_results") or [])[:stage]
    ]


def candidate_pool(result: EvolutionResult):
    return list(result.archive or result.population or [result.best])


def oracle_compliant_candidates(result: EvolutionResult, bindings: list[dict[str, Any]]):
    return [
        candidate
        for candidate in candidate_pool(result)
        if candidate.result
        and candidate.result.feasible
        and not state_binding_mismatches(candidate.genome, bindings)
    ]


def modified_direct_witness(result: EvolutionResult, query: dict[str, Any]) -> dict[str, Any]:
    genome = copy.deepcopy(result.best.genome)
    assignment = genome.get(query["segment"])
    if not isinstance(assignment, dict):
        raise ValueError(f"accepted genome lacks {query['segment']}")
    assignment.update(copy.deepcopy(query["assignments"]))
    return genome


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    episodes = {episode["episode_id"]: episode for episode in load_episodes(BENCHMARK)}
    source_rows = {episode_id: load_source_rows(path) for episode_id, path in SOURCE_PATHS.items()}
    source_hashes = {episode_id: sha256_file(path) for episode_id, path in SOURCE_PATHS.items()}
    public_records: list[dict[str, Any]] = []
    hidden_records: list[dict[str, Any]] = []
    pilot_records: list[dict[str, Any]] = []
    branch_audits: list[dict[str, Any]] = []
    builder = ScaffoldRestartSeedBuilder()

    for spec in BRANCHES:
        rows = source_rows[spec.episode_id]
        if sorted(rows) != list(range(10)):
            errors.append(f"{spec.branch_id}: expected source seeds 0--9, got {sorted(rows)}")
            continue
        episode = episodes[spec.episode_id]
        seed_zero = rows[0]
        context = context_at_stage(episode, seed_zero, spec.source_stage)
        project = project_at_stage(spec.episode_id, seed_zero, context, spec.source_stage)
        history = public_history(seed_zero, spec.source_stage)
        source_update = seed_zero["update_results"][spec.source_stage - 1]
        public_record = {
            "branch_id": spec.branch_id,
            "episode_id": spec.episode_id,
            "source_stage": spec.source_stage,
            "source_update_id": source_update.get("update_id"),
            "task_type": spec.task_type,
            "required_channels": list(spec.required_channels),
            "paraphrases": [
                {"paraphrase_id": f"q{index}", "text": text}
                for index, text in enumerate(spec.paraphrases)
            ],
            "history_length": len(history),
            "history_sha256": value_hash(history),
            "segment_signature": [
                {
                    "name": segment.name,
                    "kind": segment.kind,
                    "demand_count": len(segment.demands),
                    "resource_count": len(segment.resources),
                }
                for segment in project.segments
            ],
            "source_artifact": repo_path(SOURCE_PATHS[spec.episode_id]),
            "source_artifact_sha256": source_hashes[spec.episode_id],
        }
        public_records.append(public_record)
        hidden_record = {
            "branch_id": spec.branch_id,
            "required_channels": list(spec.required_channels),
            "oracle_query": copy.deepcopy(spec.oracle_query),
            "indirect_ids": list(spec.indirect_ids),
        }
        hidden_records.append(hidden_record)

        if sorted(spec.oracle_query.get("depends_on") or []) != sorted(spec.required_channels):
            errors.append(f"{spec.branch_id}: oracle depends_on does not match required channels")
        for text in spec.paraphrases:
            leaked = [entity_id for entity_id in spec.indirect_ids if entity_id in text]
            if leaked:
                errors.append(f"{spec.branch_id}: indirect ids leaked into branch text: {leaked}")

        seed_audits = []
        for seed, row in sorted(rows.items()):
            stage_update = row["update_results"][spec.source_stage - 1]
            if stage_update.get("setup_code") != source_update.get("setup_code") or stage_update.get(
                "fitness_code"
            ) != source_update.get("fitness_code"):
                errors.append(f"{spec.branch_id} seed {seed}: Workbench slots differ from seed 0")
            result = result_at_stage(row, spec.source_stage)
            if len(result.population) != 200:
                errors.append(
                    f"{spec.branch_id} seed {seed}: expected frozen population 200, got {len(result.population)}"
                )
            accepted_input = None if spec.task_type == "event_only" else result
            bindings, diagnostics = materialize_state_bindings(
                [spec.oracle_query], accepted_result=accepted_input, segments=project.segments
            )
            if diagnostics["errors"] or len(bindings) != 1:
                errors.append(
                    f"{spec.branch_id} seed {seed}: oracle materialization failed: {diagnostics['errors']}"
                )
                continue
            if spec.task_type == "event_only":
                witness = modified_direct_witness(result, spec.oracle_query)
            else:
                witness = copy.deepcopy(result.best.genome)
            witness_fitness = coerce_fitness_result(project.evaluate(witness, project.data), witness)
            bound_project = apply_state_bindings(project, bindings)
            bound_fitness = coerce_fitness_result(bound_project.evaluate(witness, bound_project.data), witness)
            if seed == 0 and (not witness_fitness.feasible or not bound_fitness.feasible):
                errors.append(
                    f"{spec.branch_id}: seed-0 authoring witness is infeasible: "
                    f"{witness_fitness.violations} / {bound_fitness.violations}"
                )

            config = EvolutionConfig(
                population_size=args.pilot_population,
                generations=args.pilot_generations,
                seed=seed,
                archive_limit=max(100, args.pilot_population),
                structured_initialization=False,
            )
            starts_by_view = {}
            for view in ("full", "ledger_only", "accepted_only", "current_only"):
                starts, _ = builder.build(
                    restart_skill="warm_restart_v1",
                    previous_result=result,
                    segments=project.segments,
                    population_size=config.population_size,
                    seed=seed,
                    evaluate=project.evaluate,
                    data=project.data,
                )
                starts_by_view[view] = starts
            start_hashes = {view: value_hash(starts) for view, starts in starts_by_view.items()}
            if len(set(start_hashes.values())) != 1:
                errors.append(f"{spec.branch_id} seed {seed}: restart populations differ by memory view")
            seed_audits.append(
                {
                    "seed": seed,
                    "source_population_count": len(result.population),
                    "binding_count": len(bindings),
                    "locked_assignment_count": diagnostics["locked_assignment_count"],
                    "authoring_witness_feasible": witness_fitness.feasible and bound_fitness.feasible,
                    "restart_seed_count": len(starts_by_view["full"]),
                    "restart_sha256_by_view": start_hashes,
                }
            )

            if args.skip_pilot or seed >= args.pilot_seeds:
                continue
            started = time.perf_counter()
            oracle_result = bound_project.run(config, initial_genomes=starts_by_view["full"])
            compliant = oracle_compliant_candidates(oracle_result, bindings)
            elapsed = time.perf_counter() - started
            if not compliant:
                errors.append(f"{spec.branch_id} seed {seed}: oracle pilot found no compliant solution")
            pilot_records.append(
                {
                    "branch_id": spec.branch_id,
                    "episode_id": spec.episode_id,
                    "source_stage": spec.source_stage,
                    "task_type": spec.task_type,
                    "seed": seed,
                    "population_size": args.pilot_population,
                    "generations": args.pilot_generations,
                    "restart_skill": "warm_restart_v1",
                    "restart_sha256": start_hashes["full"],
                    "locked_assignment_count": diagnostics["locked_assignment_count"],
                    "compliant": bool(compliant),
                    "compliant_candidate_count": len(compliant),
                    "best_compliant_scalar": min(
                        (candidate.result.scalar for candidate in compliant if candidate.result),
                        default=None,
                    ),
                    "best_compliant_objectives": (
                        min(
                            compliant,
                            key=lambda candidate: candidate.result.scalar
                            if candidate.result
                            else float("inf"),
                        ).result.objectives
                        if compliant
                        else []
                    ),
                    "latency_seconds": elapsed,
                }
            )
        branch_audits.append(
            {
                "branch_id": spec.branch_id,
                "source_stage": spec.source_stage,
                "task_type": spec.task_type,
                "seed_audits": seed_audits,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    public_payload = {
        "protocol": "state_binding_public_branches_v1",
        "benchmark": repo_path(BENCHMARK),
        "benchmark_sha256": sha256_file(BENCHMARK),
        "branches": public_records,
    }
    hidden_payload = {
        "protocol": "state_binding_hidden_oracle_v1",
        "separation_rule": "This file is never included in a controller prompt.",
        "branches": hidden_records,
    }
    audit_payload = {
        "status": "passed" if not errors else "failed",
        "protocol": "state_binding_provider_free_preflight_v1",
        "branch_count": len(public_records),
        "task_type_counts": {
            task_type: sum(record["task_type"] == task_type for record in public_records)
            for task_type in sorted({record["task_type"] for record in public_records})
        },
        "source_seed_count": 10,
        "pilot": {
            "enabled": not args.skip_pilot,
            "population_size": args.pilot_population,
            "generations": args.pilot_generations,
            "seed_count": args.pilot_seeds,
            "run_count": len(pilot_records),
            "compliant_count": sum(bool(row["compliant"]) for row in pilot_records),
        },
        "branch_audits": branch_audits,
        "errors": errors,
    }
    (args.out_dir / "public_challenges.json").write_text(
        json.dumps(public_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "hidden_oracle.json").write_text(
        json.dumps(hidden_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out_dir / "pilot_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pilot_records),
        encoding="utf-8",
    )
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(audit_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
