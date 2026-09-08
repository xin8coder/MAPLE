#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo2.evaluation.moea_metrics import igd
from evo2.evaluation.reference_solvers.moea_reference import (
    CLOUD_OBJECTIVES,
    GREEN_OBJECTIVES,
    MOEAConfig,
    ScalarGAConfig,
    cloud_stage_reference,
    green_vrp_stage_reference,
    inrc_stage_reference,
)


DOMAIN_MODULES = {
    "green_vrp_multiobjective": "scripts.materialize_green_vrp_mo",
    "cloud_scheduling_multiobjective": "scripts.materialize_cloud_scheduling_mo",
    "inrc2": "scripts.materialize_inrc_realistic_dynamic",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def run_episode(
    ep: dict[str, Any],
    config: MOEAConfig,
    seed_offset: int,
    selected_stage_ids: set[str] | None = None,
    selected_stage_indices: set[int] | None = None,
    state_independent_objectives: bool = True,
) -> dict[str, Any]:
    domain = ep["domain"]
    module = importlib.import_module(DOMAIN_MODULES[domain])
    state = copy.deepcopy(ep["hidden_initial_state"])
    previous_solution = None
    reference_trajectory = []
    stages = [("initial", None)] + [(oracle["update_id"], oracle.get("hidden_delta", {})) for oracle in ep.get("hidden_update_oracle", [])]
    objective_names = GREEN_OBJECTIVES if domain == "green_vrp_multiobjective" else CLOUD_OBJECTIVES
    if domain == "inrc2":
        return run_scalar_episode(
            ep,
            config,
            seed_offset,
            selected_stage_ids,
            selected_stage_indices,
            state_independent_objectives=state_independent_objectives,
        )
    started = time.perf_counter()
    for stage_idx, (update_id, delta) in enumerate(stages):
        if stage_idx > 0:
            state = module._apply_delta(state, delta)
        old_step = _old_reference_step(ep, update_id)
        if not stage_selected(update_id, stage_idx, selected_stage_ids, selected_stage_indices):
            previous_solution = _reference_step_solution(old_step) or previous_solution
            continue
        scoring_previous_solution = None if state_independent_objectives else previous_solution
        if domain == "green_vrp_multiobjective":
            reference = green_vrp_stage_reference(state, scoring_previous_solution, module._score_solution, config, seed_offset + stage_idx * 10000)
        elif domain == "cloud_scheduling_multiobjective":
            reference = cloud_stage_reference(state, scoring_previous_solution, module._score_solution, config, seed_offset + stage_idx * 10000)
        else:
            raise ValueError(f"unsupported domain: {domain}")

        old_archive = old_step.get("pareto_archive", []) if old_step else []
        old_igd = igd(old_archive, reference["archive"], objective_names, reference["hv_reference_point"], reference["hv_ideal_point"]) if old_archive else None
        representative = _representative_solution(reference["archive"], objective_names)
        previous_solution = representative
        reference_trajectory.append(
            {
                "update_id": update_id,
                "reference_type": "pareto_ga_best_known",
                "objectives": objective_names,
                "pareto_archive": reference["archive"],
                "archive_size": reference["archive_size"],
                "hypervolume_reference_point": reference["hv_reference_point"],
                "hypervolume_ideal_point": reference["hv_ideal_point"],
                "reference_hv": reference["reference_hv"],
                "reference_igd_self": reference["self_igd"],
                "old_constructive_igd_to_ga": old_igd,
                "solver": "offline_multi_seed_nsga2_style_ga",
                "solver_config": config.__dict__,
                "evaluations": reference["evaluations"],
                "hidden_delta_type": (delta or {}).get("type"),
                "representative_solution": representative,
                "objective_state_scope": "current_state_only" if state_independent_objectives else "history_dependent",
                "previous_solution_role": "not_used_for_official_objectives_or_feasibility"
                if state_independent_objectives
                else "used_by_reference_scoring",
            }
        )
        print(
            json.dumps(
                {
                    "event": "stage_done",
                    "episode_id": ep["episode_id"],
                    "update_id": update_id,
                    "domain": domain,
                    "archive_size": reference["archive_size"],
                    "reference_hv": reference["reference_hv"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "episode_id": ep["episode_id"],
        "benchmark": ep["benchmark"],
        "domain": domain,
        "objective_names": objective_names,
        "reference_policy": "offline_multi_seed_ga_pareto_best_known_for_hv_igd_current_state_only"
        if state_independent_objectives
        else "offline_multi_seed_ga_pareto_best_known_for_hv_igd",
        "reference_trajectory": reference_trajectory,
        "selected_stage_ids": sorted(selected_stage_ids or []),
        "selected_stage_indices": sorted(selected_stage_indices or []),
        "partial_stage_refresh": bool(selected_stage_ids or selected_stage_indices),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run_scalar_episode(
    ep: dict[str, Any],
    moea_config: MOEAConfig,
    seed_offset: int,
    selected_stage_ids: set[str] | None = None,
    selected_stage_indices: set[int] | None = None,
    state_independent_objectives: bool = True,
) -> dict[str, Any]:
    module = importlib.import_module(DOMAIN_MODULES[ep["domain"]])
    config = ScalarGAConfig(
        population_size=moea_config.population_size,
        generations=moea_config.generations,
        seeds=moea_config.seeds,
        mutation_rate=moea_config.mutation_rate,
    )
    state = copy.deepcopy(ep["hidden_initial_state"])
    previous_solution = None
    trajectory = []
    stages = [("initial", None)] + [(oracle["update_id"], oracle.get("hidden_delta", {})) for oracle in ep.get("hidden_update_oracle", [])]
    started = time.perf_counter()
    for stage_idx, (update_id, delta) in enumerate(stages):
        if stage_idx > 0:
            state = module._apply_delta(state, delta)
        old_step = _old_reference_step(ep, update_id) or {}
        if not stage_selected(update_id, stage_idx, selected_stage_ids, selected_stage_indices):
            previous_solution = _reference_step_solution(old_step) or previous_solution
            continue
        reference = inrc_stage_reference(
            state,
            None if state_independent_objectives else previous_solution,
            module._score_roster,
            module._construct_roster,
            config,
            seed_offset + stage_idx * 10000,
        )
        previous_solution = reference["solution"]
        trajectory.append(
            {
                "update_id": update_id,
                "reference_type": "ga_best_known",
                "objective": reference["objective"],
                "previous_constructive_objective": old_step.get("objective"),
                "improvement_over_constructive": (
                    round(float(old_step["objective"]) - reference["objective"], 6)
                    if old_step.get("objective") is not None
                    else None
                ),
                "score": reference["score"],
                "lower_bound": None,
                "upper_bound": reference["objective"],
                "gap": None,
                "solver": "offline_multi_seed_scalar_ga",
                "solver_config": reference["solver_config"],
                "evaluations": reference["evaluations"],
                "hidden_delta_type": (delta or {}).get("type"),
                "solution": reference["solution"],
                "objective_state_scope": "current_state_only" if state_independent_objectives else "history_dependent",
                "previous_solution_role": "not_used_for_official_objectives_or_feasibility"
                if state_independent_objectives
                else "used_by_reference_scoring",
            }
        )
        print(
            json.dumps(
                {
                    "event": "stage_done",
                    "episode_id": ep["episode_id"],
                    "update_id": update_id,
                    "domain": ep["domain"],
                    "objective": reference["objective"],
                    "evaluations": reference["evaluations"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "episode_id": ep["episode_id"],
        "benchmark": ep["benchmark"],
        "domain": ep["domain"],
        "reference_policy": "offline_multi_seed_ga_best_known_scalar_reference_current_state_only"
        if state_independent_objectives
        else "offline_multi_seed_ga_best_known_scalar_reference",
        "reference_trajectory": trajectory,
        "selected_stage_ids": sorted(selected_stage_ids or []),
        "selected_stage_indices": sorted(selected_stage_indices or []),
        "partial_stage_refresh": bool(selected_stage_ids or selected_stage_indices),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _old_reference_step(ep: dict[str, Any], update_id: str) -> dict[str, Any] | None:
    for step in ep.get("evaluation", {}).get("reference_trajectory", []):
        if step.get("update_id") == update_id:
            return step
    return None


def _reference_step_solution(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if not step:
        return None
    return copy.deepcopy(step.get("representative_solution") or step.get("solution"))


def parse_stage_filter(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_stage_index_filter(raw: str) -> set[int]:
    indices = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        indices.add(int(item))
    return indices


def stage_selected(
    update_id: str,
    stage_index: int,
    selected_stage_ids: set[str] | None,
    selected_stage_indices: set[int] | None,
) -> bool:
    if not selected_stage_ids and not selected_stage_indices:
        return True
    aliases = {update_id, f"S{stage_index:02d}", f"s{stage_index:02d}"}
    if stage_index == 0:
        aliases.update({"initial", "S00", "s00"})
    return bool((selected_stage_ids or set()) & aliases) or stage_index in (selected_stage_indices or set())


def _representative_solution(archive: list[dict[str, Any]], objective_names: list[str]) -> dict[str, Any] | None:
    if not archive:
        return None
    return min(archive, key=lambda item: sum(float(item["objectives"][name]) for name in objective_names))["solution"]


def enhance_episode(ep: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(ep)
    out.setdefault("evaluation", {})
    original_trajectory = list(out["evaluation"].get("reference_trajectory", []))
    out["evaluation"].setdefault("reference_trajectory_constructive_backup", original_trajectory)
    if reference.get("partial_stage_refresh"):
        by_update = {str(step.get("update_id")): copy.deepcopy(step) for step in original_trajectory}
        for step in reference["reference_trajectory"]:
            by_update[str(step.get("update_id"))] = copy.deepcopy(step)
        merged = []
        seen = set()
        for step in original_trajectory:
            update_id = str(step.get("update_id"))
            merged.append(by_update[update_id])
            seen.add(update_id)
        for step in reference["reference_trajectory"]:
            update_id = str(step.get("update_id"))
            if update_id not in seen:
                merged.append(copy.deepcopy(step))
                seen.add(update_id)
        out["evaluation"]["reference_trajectory"] = merged
        out["evaluation"]["reference_refresh"] = {
            "mode": "partial_stage_refresh",
            "selected_stage_ids": reference.get("selected_stage_ids", []),
            "selected_stage_indices": reference.get("selected_stage_indices", []),
            "refreshed_update_ids": [step.get("update_id") for step in reference.get("reference_trajectory", [])],
            "preserved_stage_count": max(0, len(original_trajectory) - len(reference.get("reference_trajectory", []))),
        }
    else:
        out["evaluation"]["reference_trajectory"] = reference["reference_trajectory"]
    out["evaluation"]["reference_policy"] = reference["reference_policy"]
    if ep.get("domain") in {"green_vrp_multiobjective", "cloud_scheduling_multiobjective"}:
        out["evaluation"]["pareto_metric_protocol"] = {
            "hv": "approximate normalized hypervolume against per-stage hypervolume_reference_point",
            "igd": "normalized inverted generational distance to per-stage GA best-known archive",
            "reference_archive": "evaluation-only offline multi-seed GA; not exposed to agent",
        }
    else:
        out["evaluation"]["best_known_metric_protocol"] = {
            "objective_gap": "candidate objective minus per-stage offline GA best-known objective",
            "reference_solution": "evaluation-only offline multi-seed scalar GA; not exposed to agent",
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline GA/MOEA Pareto references for multi-objective dynamic benchmarks.")
    parser.add_argument("--inputs", nargs="+", default=[
        "data/evo2_dynoptbench/compressed_plan/green_vrp_mo_6episodes.jsonl",
        "data/evo2_dynoptbench/compressed_plan/cloud_scheduling_mo_6episodes.jsonl",
    ])
    parser.add_argument("--output-dir", default="data/evo2_dynoptbench/reference_archives/moea")
    parser.add_argument("--enhanced-output-dir", default="data/evo2_dynoptbench/compressed_plan/moea_enhanced")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--update-limit", type=int)
    parser.add_argument(
        "--stage-ids",
        default="",
        help="Comma-separated stage ids to recompute, e.g. initial,u001,u005 or S00,S01. Unselected stages keep their existing references.",
    )
    parser.add_argument(
        "--stage-indices",
        default="",
        help="Comma-separated zero-based stage indices to recompute, e.g. 0,1,5. Unselected stages keep their existing references.",
    )
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--generations", type=int, default=250)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--archive-limit", type=int, default=128)
    parser.add_argument("--hv-samples", type=int, default=0)
    parser.add_argument(
        "--history-dependent-objectives",
        action="store_true",
        help="Legacy mode: pass the previous reference solution into official scoring. The default keeps official objectives current-state-only.",
    )
    args = parser.parse_args()

    config = MOEAConfig(
        population_size=args.population,
        generations=args.generations,
        seeds=args.seeds,
        archive_limit=args.archive_limit,
        hv_samples=args.hv_samples,
    )
    out_dir = Path(args.output_dir)
    enhanced_dir = Path(args.enhanced_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    selected_stage_ids = parse_stage_filter(args.stage_ids)
    selected_stage_indices = parse_stage_index_filter(args.stage_indices)
    summary = {
        "config": config.__dict__,
        "selected_stage_ids": sorted(selected_stage_ids),
        "selected_stage_indices": sorted(selected_stage_indices),
        "partial_stage_refresh": bool(selected_stage_ids or selected_stage_indices),
        "inputs": [],
        "episodes": 0,
        "stages": 0,
        "outputs": [],
    }
    for input_idx, input_path_raw in enumerate(args.inputs):
        input_path = Path(input_path_raw)
        rows = load_jsonl(input_path)
        if args.episode_start:
            rows = rows[args.episode_start :]
        if args.episode_limit is not None:
            rows = rows[: args.episode_limit]
        enhanced_rows = []
        input_summary = {"input": str(input_path), "episodes": []}
        for ep_idx, ep in enumerate(rows):
            source_episode_index = int(args.episode_start or 0) + ep_idx
            ep_run = copy.deepcopy(ep)
            if args.update_limit is not None:
                ep_run["update_stream"] = ep_run.get("update_stream", [])[: args.update_limit]
                ep_run["hidden_update_oracle"] = ep_run.get("hidden_update_oracle", [])[: args.update_limit]
            reference = run_episode(
                ep_run,
                config,
                seed_offset=1000000 * input_idx + source_episode_index * 100000,
                selected_stage_ids=selected_stage_ids,
                selected_stage_indices=selected_stage_indices,
                state_independent_objectives=not args.history_dependent_objectives,
            )
            ref_path = out_dir / f"{reference['episode_id']}.moea_reference.json"
            ref_path.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
            enhanced_rows.append(enhance_episode(ep_run, reference))
            input_summary["episodes"].append(
                {
                    "episode_id": reference["episode_id"],
                    "stages": len(reference["reference_trajectory"]),
                    "elapsed_seconds": reference["elapsed_seconds"],
                    "reference_file": str(ref_path),
                }
            )
            summary["episodes"] += 1
            summary["stages"] += len(reference["reference_trajectory"])
            print(json.dumps(input_summary["episodes"][-1], ensure_ascii=False), flush=True)
        enhanced_path = enhanced_dir / input_path.name.replace(".jsonl", ".moea_reference.jsonl")
        write_jsonl(enhanced_path, enhanced_rows)
        input_summary["enhanced_output"] = str(enhanced_path)
        summary["inputs"].append(input_summary)
        summary["outputs"].append(str(enhanced_path))
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
