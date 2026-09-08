#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.reference_solvers.moea_reference import CLOUD_OBJECTIVES, GREEN_OBJECTIVES, evaluate_against_reference
from scripts.analysis.evaluate_green_agent_run import _canonical_solution as _canonical_green_solution
from scripts.analysis.evaluate_green_agent_run import _green_constraint_report, _safe_green_score


DEFAULT_NLDO_STRONG_REFERENCE = (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)

DEFAULT_BENCHMARKS = {
    "NLDO": DEFAULT_NLDO_STRONG_REFERENCE,
    "cobench_exact_small": "data/evo2_dynoptbench/public_csv/cobench_exact_small_6episodes_csv.jsonl",
    "fjsp_exact_small": "data/evo2_dynoptbench/public_csv/fjsp_exact_small_6episodes_csv.jsonl",
    "inrc_realistic_dynamic": "data/evo2_dynoptbench/public_csv/inrc_realistic_dynamic_6episodes_csv.jsonl",
    "green_vrp_mo": "data/evo2_dynoptbench/public_csv/green_vrp_mo_6episodes_csv.jsonl",
    "cloud_scheduling_mo": "data/evo2_dynoptbench/public_csv/cloud_scheduling_mo_6episodes_csv.jsonl",
}
FORMAL_COMPARISON_BUDGET = {
    "population_size": 200,
    "generations": 200,
    "seed_count": 10,
}

_REFERENCE_TRAJECTORY_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate benchmark runs against hidden reference trajectories.")
    parser.add_argument("--run-dir", required=True, help="Formal run directory containing one subdirectory per benchmark.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Optional benchmark mapping name=path. Defaults to the formal EMNLP five-benchmark suite.",
    )
    parser.add_argument(
        "--require-formal-budget",
        action="store_true",
        help="Reject run directories whose replay plan or replay commands are not 200x200 with 10 seeds.",
    )
    args = parser.parse_args()

    benchmark_paths = dict(DEFAULT_BENCHMARKS)
    for item in args.benchmark:
        name, path = item.split("=", 1)
        benchmark_paths[name] = path

    validate_benchmark_paths(benchmark_paths)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.require_formal_budget:
        require_formal_run_budget(Path(args.run_dir))
    rows = evaluate_run_dir(Path(args.run_dir), {name: Path(path) for name, path in benchmark_paths.items()})
    summary = aggregate(rows)
    totals = total_scores(rows)
    write_csv(out_dir / "reference_stage_metrics.csv", rows)
    write_csv(out_dir / "reference_stage_summary.csv", summary)
    write_csv(out_dir / "reference_total_scores.csv", totals)
    (out_dir / "reference_stage_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reference_stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "reference_total_scores.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    plots = plot_outputs(summary, out_dir)
    print(json.dumps({"stage_rows": len(rows), "summary_rows": len(summary), "total_rows": len(totals), "out_dir": str(out_dir), "plots": plots}, ensure_ascii=False, indent=2))


def validate_benchmark_paths(benchmark_paths: dict[str, str]) -> None:
    """Fail loudly when the default NLDO strong reference is unavailable.

    The public 12-update NLDO JSONL is the right input for replay, but paper
    metrics must be computed against the merged 500x500x10 reference front.
    Failing here prevents silent fallback to the weaker embedded reference.
    """

    nldo_path = Path(benchmark_paths.get("NLDO", ""))
    if nldo_path and not nldo_path.exists():
        raise SystemExit(
            "NLDO reference benchmark not found: "
            f"{nldo_path}. Build or restore the 500x500x10 strong reference "
            "before running paper metrics, or pass --benchmark NLDO=<path> explicitly."
        )
    if nldo_path:
        validate_nldo_mo_reference_grid(nldo_path)


def validate_nldo_mo_reference_grid(path: Path) -> None:
    """Reject embedded constructive proxies on the formal Pareto metric route.

    Constructive archives are useful while materializing a benchmark, but they
    are not a paper reference.  Formal NLDO Pareto scoring requires a complete
    independently generated 500x500x10 trajectory at every stage.  Failing
    here prevents a partially refreshed payload from silently filling the
    untouched stages with its 1--5 point deterministic bootstrap archive.
    """

    episodes = load_benchmark(path)
    errors: list[str] = []
    pareto_episodes = [
        episode
        for episode in episodes.values()
        if str(episode.get("domain") or "")
        in {"green_vrp_multiobjective", "cloud_scheduling_multiobjective"}
    ]
    for episode in pareto_episodes:
        episode_id = str(episode.get("episode_id") or "unknown")
        evaluation = episode.get("evaluation") if isinstance(episode.get("evaluation"), dict) else {}
        trajectory = list(evaluation.get("reference_trajectory") or [])
        expected_stages = 1 + len(episode.get("update_stream") or [])
        if len(trajectory) != expected_stages:
            errors.append(
                f"{episode_id}: reference trajectory has {len(trajectory)} stages, "
                f"expected {expected_stages}"
            )
            continue
        for stage_index, step in enumerate(trajectory):
            label = f"{episode_id}:t{stage_index:02d}"
            solver = str(step.get("solver") or "")
            reference_type = str(step.get("reference_type") or "")
            config = step.get("solver_config") if isinstance(step.get("solver_config"), dict) else {}
            archive = step.get("pareto_archive") if isinstance(step.get("pareto_archive"), list) else []
            reference_hv = to_float(step.get("reference_hv"))
            if solver != "offline_multi_seed_nsga2_style_ga":
                errors.append(f"{label}: solver={solver or 'missing'} is not the strong offline MOEA")
            if reference_type != "pareto_ga_best_known":
                errors.append(f"{label}: reference_type={reference_type or 'missing'} is not pareto_ga_best_known")
            for key, expected in (("population_size", 500), ("generations", 500), ("seeds", 10)):
                try:
                    actual = int(config.get(key))
                except (TypeError, ValueError):
                    actual = None
                if actual != expected:
                    errors.append(f"{label}: solver_config.{key}={actual}, expected {expected}")
            if not archive:
                errors.append(f"{label}: strong Pareto archive is empty")
            if reference_hv is None or not math.isfinite(reference_hv) or reference_hv <= 0:
                errors.append(f"{label}: reference_hv is missing or non-positive")
            if not isinstance(step.get("hypervolume_reference_point"), dict):
                errors.append(f"{label}: hypervolume_reference_point is missing")
            if not isinstance(step.get("hypervolume_ideal_point"), dict):
                errors.append(f"{label}: hypervolume_ideal_point is missing")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:24])
        suffix = f"\n- ... {len(errors) - 24} additional errors" if len(errors) > 24 else ""
        raise SystemExit(
            "NLDO Pareto reference validation failed; constructive/bootstrap "
            "references are forbidden for formal HV/IGD scoring:\n"
            f"{preview}{suffix}"
        )


def require_formal_run_budget(run_dir: Path) -> None:
    """Ensure comparison metrics are computed only from the formal GA/NSGA-II budget."""
    errors: list[str] = []
    plan_path = run_dir / "replay_plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"cannot read replay_plan.json: {exc}")
        else:
            plan_budget = plan.get("budget_config")
            if plan_budget is None:
                errors.append("replay_plan.json is missing budget_config")
            else:
                errors.extend(_budget_errors("replay_plan.json", plan_budget))
    commands_path = run_dir / "replay_commands.jsonl"
    if commands_path.exists():
        for line_no, line in enumerate(commands_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"replay_commands.jsonl:{line_no} is not valid JSON: {exc}")
                continue
            command = row.get("command")
            if isinstance(command, list):
                command_budget = _budget_from_command(command)
                errors.extend(_budget_errors(f"replay_commands.jsonl:{line_no}", command_budget))
    for status_path in sorted((run_dir / "replay_jobs").glob("*/replay_status.json")):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{status_path.relative_to(run_dir)} is not valid JSON: {exc}")
            continue
        status_budget = status.get("budget_config")
        if status_budget is not None:
            errors.extend(_budget_errors(str(status_path.relative_to(run_dir)), status_budget))
    if errors:
        raise SystemExit("formal evaluation budget check failed:\n" + "\n".join(f"- {error}" for error in errors))


def _budget_from_command(command: list[Any]) -> dict[str, int]:
    mapping = {
        "--population-size": "population_size",
        "--generations": "generations",
        "--seed-count": "seed_count",
    }
    out: dict[str, int] = {}
    for index, token in enumerate(command):
        key = mapping.get(str(token))
        if key is None:
            continue
        try:
            out[key] = int(command[index + 1])
        except Exception:
            out[key] = -1
    return out


def _budget_errors(label: str, budget: dict[str, Any]) -> list[str]:
    errors = []
    for key, expected in FORMAL_COMPARISON_BUDGET.items():
        try:
            actual = int(budget.get(key))
        except Exception:
            actual = None
        if actual != expected:
            errors.append(f"{label} has {key}={actual}, expected {expected}")
    return errors


def evaluate_run_dir(run_dir: Path, benchmark_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, path in sorted(benchmark_paths.items()):
        if not path.exists():
            continue
        episodes = load_benchmark(path)
        for jsonl in iter_result_jsonls(run_dir, benchmark):
            mode = infer_mode(jsonl)
            for run in load_jsonl(jsonl):
                episode_id = str(run.get("episode_id") or "")
                if episode_id not in episodes:
                    continue
                rows.extend(evaluate_episode_run(benchmark, mode, run, episodes[episode_id]))
    return rows


def iter_result_jsonls(run_dir: Path, benchmark: str) -> list[Path]:
    bench_dir = run_dir / benchmark
    if bench_dir.exists():
        return sorted(bench_dir.glob("*.jsonl"))
    if benchmark.upper() == "NLDO":
        combined = sorted(run_dir.glob("nldo_s*/evo2_limit0.jsonl"))
        replay_jobs = sorted(run_dir.glob("replay_jobs/*/nldo_s*/evo2_limit0.jsonl"))
        if combined:
            covered_episode_ids = set()
            for path in combined:
                covered_episode_ids.update(episode_ids_in_jsonl(path))
            uncombined_replay_jobs = []
            for path in replay_jobs:
                episode_ids = episode_ids_in_jsonl(path)
                if episode_ids and episode_ids <= covered_episode_ids:
                    continue
                uncombined_replay_jobs.append(path)
            replay_jobs = uncombined_replay_jobs
            return combined + replay_jobs
        if replay_jobs:
            return replay_jobs
        direct_replays = sorted(run_dir.glob("evo2_limit*.jsonl"))
        direct_replays.extend(sorted(run_dir.glob("*/evo2_limit*.jsonl")))
        return direct_replays
    return []


def episode_ids_in_jsonl(path: Path) -> set[str]:
    episode_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            episode_id = row.get("episode_id") if isinstance(row, dict) else None
            if episode_id:
                episode_ids.add(str(episode_id))
    return episode_ids


def load_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in load_jsonl(path)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_mode(path: Path) -> str:
    if path.parent.name.startswith("nldo_s"):
        return path.parent.name
    name = path.stem
    return name.split("_limit", 1)[0] if "_limit" in name else name


def evaluate_episode_run(benchmark: str, mode: str, run: dict[str, Any], episode: dict[str, Any]) -> list[dict[str, Any]]:
    run_seed = run.get("run_seed")
    rows = evaluate_stage_sequence(benchmark, mode, extract_stages(run), episode)
    for row in rows:
        row["run_seed"] = run_seed
    for ablation_name, stages in extract_ablation_stage_sets(run).items():
        ablation_rows = evaluate_stage_sequence(benchmark, f"{mode}__{ablation_name}", stages, episode)
        for row in ablation_rows:
            row["run_seed"] = run_seed
        rows.extend(ablation_rows)
    return rows


def evaluate_stage_sequence(benchmark: str, mode: str, stages: list[dict[str, Any]], episode: dict[str, Any]) -> list[dict[str, Any]]:
    scorer_benchmark = effective_benchmark(benchmark, episode)
    domain = str(episode.get("domain") or "")
    family = str(episode.get("family") or "")
    state = copy.deepcopy(episode.get("hidden_initial_state") or {})
    previous_solution: dict[str, Any] | None = None
    reference_steps = reference_trajectory_for_payload(scorer_benchmark, domain, episode)
    oracle = episode.get("hidden_update_oracle") or []
    rows = []
    for stage_index, stage in enumerate(stages):
        if stage_index > 0 and stage_index - 1 < len(oracle):
            state = apply_hidden_delta(scorer_benchmark, domain, state, oracle[stage_index - 1].get("hidden_delta") or {})
        update_id = stage["update_id"]
        ref_step = reference_step_for_stage(reference_steps, update_id, stage_index)
        metric = score_solution(
            scorer_benchmark,
            domain,
            family,
            state,
            stage.get("solution") or {},
            previous_solution,
            ref_step,
            stage.get("candidate_archive") or [],
        )
        row = {
            "benchmark": benchmark,
            "scorer_benchmark": scorer_benchmark,
            "mode": mode,
            "episode_id": episode.get("episode_id"),
            "stage_index": stage_index,
            "stage_type": "initial" if stage_index == 0 else "update",
            "update_id": update_id,
            "feasible": metric["feasible"],
            "normalized_score": metric["normalized_score"],
            "normalized_total_score": metric["normalized_score"],
            "objective": metric.get("objective"),
            "reference_objective": metric.get("reference_objective"),
            "objective_gap": metric.get("objective_gap"),
            "hv": metric.get("hv"),
            "reference_hv": metric.get("reference_hv"),
            "normalized_hv": metric.get("normalized_hv"),
            "igd": metric.get("igd"),
            "igd_score": metric.get("igd_score"),
            "ideal_gap": metric.get("ideal_gap"),
            "tokens": stage.get("stage_tokens"),
            "latency_seconds": stage.get("latency_seconds"),
            "agent_reported_feasible": stage.get("agent_feasible"),
            "agent_reported_objective": stage.get("agent_objective"),
            "true_pass": bool(metric["feasible"]) if stage_index > 0 else None,
            "missing_stage": False,
        }
        rows.append(row)
        previous_solution = metric.get("canonical_solution") or stage.get("solution") or previous_solution
    seen_updates = {str(row.get("update_id")) for row in rows if row.get("stage_type") == "update"}
    for expected_index, update in enumerate(episode.get("update_stream") or [], start=1):
        update_id = str(update.get("update_id") or f"u{expected_index:03d}") if isinstance(update, dict) else f"u{expected_index:03d}"
        if update_id in seen_updates:
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "scorer_benchmark": scorer_benchmark,
                "mode": mode,
                "episode_id": episode.get("episode_id"),
                "stage_index": expected_index,
                "stage_type": "update",
                "update_id": update_id,
                "feasible": False,
                "true_pass": False,
                "missing_stage": True,
                "normalized_score": 0.0,
                "normalized_total_score": 0.0,
                "objective": None,
                "reference_objective": None,
                "objective_gap": None,
                "hv": None,
                "reference_hv": None,
                "normalized_hv": 0.0,
                "igd": None,
                "igd_score": 0.0,
                "ideal_gap": None,
                "tokens": None,
                "latency_seconds": None,
                "agent_reported_feasible": None,
                "agent_reported_objective": None,
            }
        )
    return rows


def extract_stages(run: dict[str, Any]) -> list[dict[str, Any]]:
    initial_solver = run.get("initial_solver_result") if isinstance(run.get("initial_solver_result"), dict) else {}
    initial_metadata = initial_solver.get("metadata") if isinstance(initial_solver.get("metadata"), dict) else {}
    out = [
        {
            "update_id": "initial",
            "solution": run.get("initial_candidate"),
            "candidate_archive": initial_metadata.get("candidate_archive") or [],
            "agent_feasible": run.get("initial_feasible"),
            "agent_objective": run.get("initial_objective"),
            "stage_tokens": 0,
            "latency_seconds": 0.0,
        }
    ]
    previous_tokens = 0
    for update in run.get("update_results", []) or []:
        solver_result = update.get("solver_result") or (update.get("metrics") or {}).get("solver_result")
        if not isinstance(solver_result, dict):
            solver_result = {}
        solver_metadata = solver_result.get("metadata") if isinstance(solver_result.get("metadata"), dict) else {}
        snapshot = update.get("token_usage_snapshot") or {}
        total_tokens = int(snapshot.get("total_tokens", 0) or 0)
        stage_tokens = max(0, total_tokens - previous_tokens)
        previous_tokens = total_tokens
        stage_trace = update.get("stage_trace") if isinstance(update.get("stage_trace"), dict) else {}
        outcome = stage_trace.get("outcome") if isinstance(stage_trace.get("outcome"), dict) else {}
        if "stage_tokens" in outcome:
            stage_tokens = int(outcome.get("stage_tokens") or 0)
        out.append(
            {
                "update_id": update.get("update_id"),
                "solution": update.get("solution"),
                "candidate_archive": solver_metadata.get("candidate_archive") or [],
                "agent_feasible": update.get("feasible"),
                "agent_objective": update.get("objective_value"),
                "stage_tokens": stage_tokens,
                "latency_seconds": (update.get("metrics") or {}).get("update_latency_seconds") or outcome.get("latency_seconds"),
            }
        )
    return out


def extract_ablation_stage_sets(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    main_stages = extract_stages(run)
    if not main_stages:
        return {}
    initial = dict(main_stages[0])
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for main_stage, update in zip(main_stages[1:], run.get("update_results", []) or [], strict=False):
        ablations = update.get("ablation_results") if isinstance(update.get("ablation_results"), dict) else {}
        for name, payload in sorted(ablations.items()):
            if not isinstance(payload, dict) or payload.get("skipped"):
                continue
            solver_result = payload.get("solver_result") if isinstance(payload.get("solver_result"), dict) else {}
            solver_metadata = solver_result.get("metadata") if isinstance(solver_result.get("metadata"), dict) else {}
            by_name[str(name)].append(
                {
                    "update_id": update.get("update_id"),
                    "solution": payload.get("solution"),
                    "candidate_archive": solver_metadata.get("candidate_archive") or [],
                    "agent_feasible": (payload.get("verification") or {}).get("feasible") if isinstance(payload.get("verification"), dict) else None,
                    "agent_objective": payload.get("objective_value"),
                    "stage_tokens": main_stage.get("stage_tokens"),
                    "latency_seconds": payload.get("latency_seconds"),
                    "shared_llm_stage_tokens": main_stage.get("stage_tokens"),
                    "extra_llm_tokens": 0,
                }
            )
    return {name: [initial] + stages for name, stages in by_name.items()}


def effective_benchmark(benchmark: str, episode: dict[str, Any]) -> str:
    if benchmark.upper() != "NLDO":
        return benchmark
    profile = str(episode.get("hidden_evaluator_profile") or "")
    if profile:
        return profile
    domain = str(episode.get("domain") or "").lower()
    if domain == "cobench":
        return "cobench_exact_small"
    if domain == "fjsp":
        return "fjsp_exact_small"
    if domain == "inrc2":
        return "inrc_realistic_dynamic"
    if domain == "green_vrp_multiobjective":
        return "green_vrp_mo"
    if domain == "cloud_scheduling_multiobjective":
        return "cloud_scheduling_mo"
    return benchmark


def reference_trajectory_for_payload(benchmark: str, domain: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the reference trajectory embedded in the benchmark payload.

    The active scaffold/replay path does not import the retired NLDO replay
    suite to recompute exact references on demand. Reference generation is a
    separate materialization step; scoring consumes the embedded public payload.
    """

    embedded = list(((payload.get("evaluation") or {}).get("reference_trajectory") or []))
    return embedded


def reference_step_for_stage(reference_steps: list[dict[str, Any]], update_id: str | None, stage_index: int) -> dict[str, Any]:
    """Resolve a reference step by public id, then by chronological stage index.

    NLDO public stage ids are globally unique, while generated reference
    trajectories may use compact ids such as u001. The chronological index is
    the stable contract shared by both views.
    """
    if update_id:
        for step in reference_steps:
            if isinstance(step, dict) and str(step.get("update_id") or "") == str(update_id):
                return step
    try:
        index = int(stage_index)
    except Exception:
        index = -1
    if 0 <= index < len(reference_steps) and isinstance(reference_steps[index], dict):
        return reference_steps[index]
    return {}


def should_recompute_exact_reference(benchmark: str, domain: str, payload: dict[str, Any], embedded: list[dict[str, Any]]) -> bool:
    if domain not in {"cobench", "fjsp"} and benchmark not in {"cobench_exact_small", "fjsp_exact_small"}:
        return False
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
    policy = str(evaluation.get("reference_policy") or "").lower()
    if "exact" in policy:
        return True
    return any(str(step.get("reference_type") or "").lower() == "exact_optimum" for step in embedded if isinstance(step, dict))


def apply_hidden_delta(benchmark: str, domain: str, state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if benchmark == "cobench_exact_small":
        return importlib.import_module("scripts.materialize_cobench_exact_small")._apply_delta(state, delta)
    if benchmark == "fjsp_exact_small":
        return importlib.import_module("scripts.materialize_fjsp_exact_small")._apply_delta(state, delta)
    if benchmark == "inrc_realistic_dynamic":
        return importlib.import_module("scripts.materialize_inrc_realistic_dynamic")._apply_delta(state, delta)
    if benchmark == "green_vrp_mo":
        return importlib.import_module("scripts.materialize_green_vrp_mo")._apply_delta(state, delta)
    if benchmark == "cloud_scheduling_mo":
        return importlib.import_module("scripts.materialize_cloud_scheduling_mo")._apply_delta(state, delta)
    raise ValueError(f"unsupported hidden-delta benchmark={benchmark!r} domain={domain!r}")


def score_solution(
    benchmark: str,
    domain: str,
    family: str,
    state: dict[str, Any],
    solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    reference_step: dict[str, Any],
    candidate_archive: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if benchmark == "cobench_exact_small":
        return score_cobench(state, family, solution, previous_solution, reference_step)
    if benchmark == "fjsp_exact_small":
        return score_fjsp(state, solution, previous_solution, reference_step)
    if benchmark == "inrc_realistic_dynamic":
        return score_inrc(state, solution, previous_solution, reference_step)
    if benchmark == "green_vrp_mo":
        return score_green(state, solution, previous_solution, reference_step, candidate_archive or [])
    if benchmark == "cloud_scheduling_mo":
        return score_cloud(state, solution, previous_solution, reference_step, candidate_archive or [])
    return {"feasible": False, "normalized_score": 0.0}


def score_cobench(state: dict[str, Any], family: str, solution: dict[str, Any], previous_solution: dict[str, Any] | None, reference_step: dict[str, Any]) -> dict[str, Any]:
    if family == "facility_location":
        canonical = canonical_facility(solution)
        objective, feasible = facility_objective(state, canonical)
    elif family == "set_cover":
        canonical = canonical_set_cover(solution)
        objective, feasible = set_cover_objective(state, canonical)
    else:
        canonical = canonical_knapsack(solution)
        objective, feasible = knapsack_objective(state, canonical)
    return scalar_metric(objective, feasible, reference_step, canonical)


def canonical_knapsack(solution: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_solution_payload(solution)
    nested = payload.get("assignments") if isinstance(payload, dict) and isinstance(payload.get("assignments"), dict) else {}
    item_aliases = [
        "selected_items",
        "selected_item_ids",
        "selected_decisions",
        "selected_decision_ids",
        "decision_ids",
        "item_ids",
        "items",
        "selected_ids",
        "selected",
        "chosen_items",
        "chosen_item_ids",
        "chosen_decisions",
        "chosen_decision_ids",
    ]
    selected = _first_list(payload, item_aliases) or _first_list(
        nested, item_aliases
    )
    if selected is None and nested:
        selected = [key for key, value in nested.items() if _truthy_selection(value)]
    elif selected is None and isinstance(payload, dict):
        selected = [key for key, value in payload.items() if _truthy_selection(value)]
    if not selected and isinstance(payload, dict) and isinstance(payload.get("solver_values"), dict):
        selected = _selected_ids_from_solver_values(payload["solver_values"])
    return {"selected_items": _canonical_id_list(selected or [])}


def knapsack_objective(state: dict[str, Any], solution: dict[str, Any]) -> tuple[float, bool]:
    items = {item["id"]: item for item in state.get("items", [])}
    selected = [item for item in solution.get("selected_items", []) if item in items]
    ids = set(selected)
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    weight = sum(float(items[item]["weight"]) for item in selected)
    value = sum(float(items[item]["value"]) for item in selected)
    feasible = mandatory <= ids and not (ids & forbidden) and weight <= float(state.get("capacity", 0)) and len(ids) == len(selected)
    if str(state.get("objective_mode", "maximize_value")) == "minimize_weight_with_value_floor":
        feasible = feasible and value + 1e-9 >= float(state.get("value_floor", 0))
        return weight, feasible
    return -value, feasible


def canonical_facility(solution: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_solution_payload(solution)
    nested = payload.get("assignments") if isinstance(payload, dict) and isinstance(payload.get("assignments"), dict) else {}
    assignments_raw = _first_present(
        payload if isinstance(payload, dict) else {},
        ["customer_assignments", "assignments", "allocation", "allocations", "placements"],
    )
    if assignments_raw is None:
        assignments_raw = _first_present(nested, ["customer_assignments", "assignments", "allocation", "allocations", "placements"])
    assignments = _canonical_assignment_map(
        assignments_raw,
        key_aliases=["customer", "customer_id", "demand", "demand_id", "item", "item_id", "id"],
        value_aliases=["facility", "facility_id", "site", "site_id", "resource", "resource_id", "assigned_to", "value"],
    )
    if isinstance(assignments_raw, dict) and "open_facilities" in assignments_raw and isinstance(assignments_raw.get("assignments"), dict):
        assignments = _canonical_assignment_map(assignments_raw.get("assignments"), key_aliases=[], value_aliases=[])
    open_raw = _first_list(payload, ["open_facilities", "active_resources", "opened", "opened_resources", "selected_facilities", "facilities", "open_sites", "selected_resources"])
    if open_raw is None:
        open_raw = _first_list(nested, ["open_facilities", "active_resources", "opened", "opened_resources", "selected_facilities", "facilities", "open_sites", "selected_resources"])
    open_facilities = _canonical_id_list(open_raw or sorted(set(assignments.values())))
    if not open_facilities and not assignments and isinstance(payload, dict) and isinstance(payload.get("solver_values"), dict):
        open_facilities, assignments = _facility_from_solver_values(payload["solver_values"])
    return {"open_facilities": open_facilities, "assignments": assignments}


def _facility_from_solver_values(values: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    open_facilities = []
    assignments: dict[str, str] = {}
    resources = []
    for name, value in values.items():
        if not _truthy_selection(value):
            continue
        text = str(name)
        if text.startswith("open_"):
            resource = text[len("open_") :]
            open_facilities.append(resource)
            resources.append(resource)
    resources = sorted(set(resources), key=len, reverse=True)
    for name, value in values.items():
        if not _truthy_selection(value):
            continue
        text = str(name)
        if not text.startswith("x_"):
            continue
        tail = text[2:]
        customer = None
        facility = None
        for resource in resources:
            suffix = "_" + resource
            if tail.endswith(suffix):
                customer = tail[: -len(suffix)]
                facility = resource
                break
        if customer is None or facility is None:
            parts = tail.rsplit("_", 1)
            if len(parts) != 2:
                continue
            customer, facility = parts
        assignments[customer] = facility
    return _canonical_id_list(open_facilities), assignments


def _selected_ids_from_solver_values(values: dict[str, Any]) -> list[str]:
    selected = []
    for name, value in values.items():
        if not _truthy_selection(value):
            continue
        item_id = str(name)
        for prefix in ("select_", "selected_", "x_"):
            if item_id.startswith(prefix):
                item_id = item_id[len(prefix) :]
                break
        selected.append(item_id)
    return selected


def facility_objective(state: dict[str, Any], solution: dict[str, Any]) -> tuple[float, bool]:
    facilities = {item["id"]: item for item in state.get("facilities", [])}
    customers = {item["id"]: item for item in state.get("customers", [])}
    open_ids = set(solution.get("open_facilities", []))
    assignments = solution.get("assignments", {})
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    feasible = bool(open_ids) and mandatory <= open_ids and not (open_ids & forbidden)
    remaining = {fid: float(facilities.get(fid, {}).get("capacity", 0)) for fid in open_ids}
    cost = sum(float(facilities.get(fid, {}).get("open_cost", 1e6)) for fid in open_ids)
    service_time = 0.0
    for cid, customer in customers.items():
        fid = assignments.get(cid)
        if fid not in open_ids or fid not in facilities:
            feasible = False
            cost += 1e6
            service_time += 1e6
            continue
        demand = float(customer.get("demand", 0))
        remaining[fid] = remaining.get(fid, 0.0) - demand
        cost += float(customer.get("serve_cost", {}).get(fid, 1e6)) * demand
        service_time += float(customer.get("serve_time", {}).get(fid, customer.get("serve_cost", {}).get(fid, 1e6))) * demand
    if any(value < -1e-9 for value in remaining.values()):
        feasible = False
    if str(state.get("objective_mode", "minimize_cost")) == "minimize_service_time":
        return service_time, feasible
    return cost, feasible


def canonical_set_cover(solution: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_solution_payload(solution)
    nested = payload.get("assignments") if isinstance(payload, dict) and isinstance(payload.get("assignments"), dict) else {}
    set_aliases = [
        "selected_sets",
        "selected_set_ids",
        "selected_decisions",
        "selected_decision_ids",
        "decision_ids",
        "set_ids",
        "sets",
        "selected_items",
        "selected_item_ids",
        "items",
        "selected_ids",
        "selected",
        "chosen_sets",
        "chosen_set_ids",
    ]
    selected = _first_list(payload, set_aliases) or _first_list(
        nested, set_aliases
    )
    if selected is None and nested:
        selected = [key for key, value in nested.items() if _truthy_selection(value)]
    elif selected is None and isinstance(payload, dict):
        selected = [key for key, value in payload.items() if _truthy_selection(value)]
    if not selected and isinstance(payload, dict) and isinstance(payload.get("solver_values"), dict):
        selected = _selected_ids_from_solver_values(payload["solver_values"])
    return {"selected_sets": _canonical_id_list(selected or [])}


def set_cover_objective(state: dict[str, Any], solution: dict[str, Any]) -> tuple[float, bool]:
    sets = {item["id"]: item for item in state.get("sets", [])}
    selected = [item for item in solution.get("selected_sets", []) if item in sets]
    ids = set(selected)
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    active_elements = {str(item["id"]) for item in state.get("elements", []) if item.get("active", True)}
    covered = set()
    cost = 0.0
    for set_id in selected:
        row = sets[set_id]
        cost += float(row.get("cost", 0))
        covered.update(str(item) for item in row.get("covers", []))
    feasible = mandatory <= ids and not (ids & forbidden) and active_elements <= covered and len(ids) == len(selected)
    return cost, feasible


def score_fjsp(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None, reference_step: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_fjsp(solution)
    objective, feasible = fjsp_objective(state, canonical, None)
    reference_objective = to_float(reference_step.get("objective"))
    if feasible and reference_objective is not None and math.isfinite(objective) and objective < reference_objective:
        reference_step = {**reference_step, "objective": float(objective), "candidate_improved_reference": True}
    return scalar_metric(objective, feasible, reference_step, canonical)


def fjsp_reference_for_previous_solution(state: dict[str, Any], previous_solution: dict[str, Any] | None, reference_step: dict[str, Any]) -> dict[str, Any]:
    """Legacy compatibility shim.

    FJSP references are now current-state-only: disruption is an auxiliary
    diagnostic and must not condition the hidden objective or exact reference.
    """
    if not previous_solution:
        return reference_step
    return reference_step


def canonical_fjsp(solution: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_solution_payload(solution)
    if not isinstance(payload, dict):
        return {"assignments": {}}
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        assignments = payload.get("assignment")
    if isinstance(assignments, dict) and isinstance(assignments.get("assignments"), dict):
        assignments = assignments["assignments"]
    if isinstance(assignments, dict):
        return {"assignments": _canonical_fjsp_assignment_map(assignments, payload)}
    if isinstance(assignments, list):
        return {"assignments": _canonical_fjsp_schedule_list(assignments)}
    schedule = _first_list(payload, ["schedule", "operation_schedule", "operations", "jobs"])
    if isinstance(schedule, list):
        return {"assignments": _canonical_fjsp_schedule_list(schedule)}
    return {"assignments": {}}


def _canonical_fjsp_assignment_map(assignments: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    payload = payload or {}
    start_times = payload.get("start_times") if isinstance(payload.get("start_times"), dict) else {}
    end_times = None
    for key in ["end_times", "finish_times", "completion_times"]:
        if isinstance(payload.get(key), dict):
            end_times = payload[key]
            break
    end_times = end_times or {}
    for op_id, item in assignments.items():
        if isinstance(item, dict):
            output[str(op_id)] = {
                "machine": item.get("machine") or item.get("machine_id"),
                "start": item.get("start", item.get("start_time", start_times.get(op_id))),
                "end": item.get("end", item.get("end_time", end_times.get(op_id))),
            }
        elif isinstance(item, (list, tuple)) and item:
            output[str(op_id)] = {
                "machine": item[0],
                "start": item[1] if len(item) > 1 else start_times.get(op_id),
                "end": item[2] if len(item) > 2 else end_times.get(op_id),
            }
        elif item not in (None, ""):
            output[str(op_id)] = {"machine": item, "start": start_times.get(op_id), "end": end_times.get(op_id)}
    return output


def _canonical_fjsp_schedule_list(schedule: list[Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in schedule:
        if not isinstance(item, dict):
            continue
        op_id = _first_present(item, ["operation_id", "operation", "op_id", "id", "task_id", "task"])
        if op_id in (None, ""):
            continue
        output[str(op_id)] = {
            "machine": _first_present(item, ["machine", "machine_id", "resource", "resource_id", "assigned_to"]),
            "start": _first_present(item, ["start", "start_time", "begin", "begin_time"]),
            "end": _first_present(item, ["end", "end_time", "finish", "finish_time"]),
        }
    return output


def fjsp_objective(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None) -> tuple[float, bool]:
    assignments = solution.get("assignments", {})
    ops = {op["id"]: (job, op) for job in state.get("jobs", []) for op in job.get("operations", [])}
    down = set(state.get("machine_downtime", []))
    feasible = set(assignments) == set(ops)
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for op_id, item in assignments.items():
        job, op = ops.get(op_id, ({}, {}))
        machine = item.get("machine")
        start = to_float(item.get("start"))
        end = to_float(item.get("end"))
        if start is not None and end is None:
            end = start + float(op.get("processing_time", 0) or 0)
            item["end"] = end
        if start is not None:
            item["start"] = start
        if machine not in op.get("eligible_machines", []) or machine in down or start is None or end is None or end - start < float(op.get("processing_time", 0)):
            feasible = False
            continue
        intervals[str(machine)].append((start, end))
    for machine_intervals in intervals.values():
        machine_intervals.sort()
        for (_, prev_end), (start, _) in zip(machine_intervals, machine_intervals[1:]):
            if start < prev_end - 1e-9:
                feasible = False
    for job in state.get("jobs", []):
        previous_end = None
        for op in job.get("operations", []):
            item = assignments.get(op["id"], {})
            start = to_float(item.get("start"))
            end = to_float(item.get("end"))
            if start is None or end is None or (previous_end is not None and start < previous_end - 1e-9):
                feasible = False
            previous_end = end
    if set(assignments) != set(ops):
        return math.inf, False
    module = importlib.import_module("scripts.materialize_fjsp_exact_small")
    objective = module._fjsp_objective_value(state, assignments, previous_solution) if assignments else math.inf
    return float(objective), feasible


def score_inrc(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None, reference_step: dict[str, Any]) -> dict[str, Any]:
    roster = canonical_roster(solution)
    module = importlib.import_module("scripts.materialize_inrc_realistic_dynamic")
    score = module._score_roster(state, roster, None)
    feasible = score.get("coverage_shortage", 0) == 0 and score.get("absence_violations", 0) == 0
    return scalar_metric(float(score["objective"]), feasible, reference_step, roster, extra=score)


def canonical_roster(solution: dict[str, Any]) -> dict[str, str]:
    payload = _unwrap_solution_payload(solution)
    assignments = _first_dict(payload, ["assignments", "assignment", "roster", "schedule"])
    if assignments is None:
        assignment_list = _first_list(payload, ["assignments", "assignment", "roster", "schedule", "shifts"])
        if isinstance(assignment_list, list):
            return _canonical_roster_list(assignment_list)
    if assignments is None and isinstance(payload, dict):
        assignments = payload
    canonical: dict[str, str] = {}
    if not isinstance(assignments, dict):
        return canonical
    for key, value in assignments.items():
        nurse = None
        day = None
        shift = None
        if isinstance(value, list):
            slot_day, slot_shift = _parse_roster_slot_key(str(key))
            if slot_day and slot_shift:
                for nurse_id in _canonical_id_list(value):
                    canonical[f"{nurse_id}|{slot_day}"] = slot_shift
            else:
                for item in value:
                    day = None
                    shift = None
                    if isinstance(item, dict):
                        day = _first_present(item, ["day", "day_id", "date", "period"])
                        shift = _first_present(item, ["shift", "shift_id", "duty", "value"])
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        day, shift = item[0], item[1]
                    if day is None or shift in (None, "", "off", "OFF"):
                        continue
                    canonical[f"{key}|{day}"] = str(shift)
            continue
        if isinstance(value, dict):
            nurse = _first_present(value, ["nurse", "nurse_id", "staff", "staff_id", "resource", "resource_id"])
            day = _first_present(value, ["day", "day_id", "date", "period"])
            shift = _first_present(value, ["shift", "shift_id", "duty", "value"])
            if nurse is None and day is None and shift is None:
                nested_any = False
                for nested_day, nested_shift in value.items():
                    if nested_shift in (None, "", "off", "OFF"):
                        continue
                    canonical[f"{key}|{nested_day}"] = str(nested_shift)
                    nested_any = True
                if nested_any:
                    continue
            if nurse is None or day is None:
                parts = str(key).split("|")
                if len(parts) >= 2:
                    nurse = nurse if nurse is not None else parts[0]
                    day = day if day is not None else parts[1]
        else:
            slot_day, slot_shift = _parse_roster_slot_key(str(key))
            if slot_day and slot_shift and _looks_like_staff_id(value):
                canonical[f"{value}|{slot_day}"] = slot_shift
                continue
            parts = str(key).split("|")
            if len(parts) >= 2:
                nurse, day = parts[0], parts[1]
                shift = value
        if nurse is None or day is None or shift is None:
            continue
        canonical[f"{nurse}|{day}"] = str(shift)
    return canonical


def _parse_roster_slot_key(key: str) -> tuple[str | None, str | None]:
    text = str(key or "")
    underscore = text.split("_")
    if len(underscore) >= 3 and _looks_like_day_id(underscore[0]):
        return underscore[0], underscore[1]
    for separator in ["|", "_", ":", "/"]:
        if separator in text:
            left, right = text.split(separator, 1)
            if left and right:
                return left, right
    return None, None


def _looks_like_day_id(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.fullmatch(r"(d|day|date)[-_]?\d+", text))


def _looks_like_staff_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.fullmatch(r"[A-Za-z]+0*\d+", text))


def _canonical_roster_list(assignments: list[Any]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for item in assignments:
        if isinstance(item, dict):
            nurse = _first_present(item, ["nurse", "nurse_id", "staff", "staff_id", "resource", "resource_id", "employee", "employee_id"])
            day = _first_present(item, ["day", "day_id", "date", "period"])
            shift = _first_present(item, ["shift", "shift_id", "duty", "value"])
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            nurse, day, shift = item[0], item[1], item[2]
        else:
            continue
        if nurse is None or day is None or shift is None:
            continue
        canonical[f"{nurse}|{day}"] = str(shift)
    return canonical


def _unwrap_solution_payload(solution: Any) -> Any:
    payload = solution
    seen = 0
    while isinstance(payload, dict) and seen < 3:
        if isinstance(payload.get("solution"), dict):
            payload = payload["solution"]
        elif isinstance(payload.get("candidate_solution"), dict):
            payload = payload["candidate_solution"]
        elif isinstance(payload.get("final_solution"), dict):
            payload = payload["final_solution"]
        elif isinstance(payload.get("candidate"), dict):
            payload = payload["candidate"]
        else:
            break
        seen += 1
    return payload


def _first_dict(payload: Any, keys: list[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _first_list(payload: Any, keys: list[str]) -> list[Any] | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return None


def _canonical_id_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            value = _first_present(
                item,
                [
                    "id",
                    "item",
                    "item_id",
                    "set",
                    "set_id",
                    "facility",
                    "facility_id",
                    "resource",
                    "resource_id",
                    "nurse",
                    "nurse_id",
                    "staff",
                    "staff_id",
                    "value",
                ],
            )
        else:
            value = item
        if value not in (None, ""):
            out.append(str(value))
    return out


def _canonical_assignment_map(assignments: Any, *, key_aliases: list[str], value_aliases: list[str]) -> dict[str, str]:
    if isinstance(assignments, dict):
        if isinstance(assignments.get("assignments"), dict):
            assignments = assignments["assignments"]
        return {str(key): str(value) for key, value in assignments.items() if value not in (None, "")}
    if not isinstance(assignments, list):
        return {}
    out: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, dict):
            continue
        key = _first_present(item, key_aliases)
        value = _first_present(item, value_aliases)
        if key in (None, "") or value in (None, ""):
            continue
        out[str(key)] = str(value)
    return out


def _truthy_selection(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "selected", "open", "chosen", "use", "used"}
    if isinstance(value, dict):
        marker = _first_present(value, ["selected", "open", "chosen", "use", "used", "active", "value"])
        return _truthy_selection(marker)
    return False


def score_green(
    state: dict[str, Any],
    solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    reference_step: dict[str, Any],
    candidate_archive: list[dict[str, Any]],
) -> dict[str, Any]:
    module = importlib.import_module("scripts.materialize_green_vrp_mo")
    canonical = normalize_green_solution_ids(state, _canonical_green_solution(solution))
    archive = hidden_green_archive(module, state, canonical, None, candidate_archive)
    return mo_metric(canonical, archive, GREEN_OBJECTIVES, reference_step)


def score_cloud(
    state: dict[str, Any],
    solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    reference_step: dict[str, Any],
    candidate_archive: list[dict[str, Any]],
) -> dict[str, Any]:
    module = importlib.import_module("scripts.materialize_cloud_scheduling_mo")
    canonical = canonical_cloud(solution)
    archive = hidden_cloud_archive(module, state, canonical, None, candidate_archive)
    return mo_metric(canonical, archive, CLOUD_OBJECTIVES, reference_step)


def canonical_cloud(solution: dict[str, Any]) -> dict[str, Any]:
    payload = _unwrap_solution_payload(solution)
    assignments = _first_present(
        payload if isinstance(payload, dict) else {},
        ["assignments", "assignment", "placements", "allocation", "allocations", "mapping", "schedule"],
    )
    if assignments is None:
        assignments = payload
    canonical = _canonical_assignment_map(
        assignments,
        key_aliases=["workload", "workload_id", "task", "task_id", "service", "service_id", "app", "app_id", "id"],
        value_aliases=["node", "node_id", "server", "server_id", "machine", "machine_id", "host", "host_id", "resource", "resource_id", "assigned_to", "value"],
    )
    return {"assignments": canonical}


def hidden_green_archive(
    module: Any,
    state: dict[str, Any],
    final_solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    candidate_archive: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = candidate_solutions(candidate_archive, final_solution)
    out = []
    seen = set()
    for solution in candidates:
        canonical = normalize_green_solution_ids(state, _canonical_green_solution(solution))
        key = canonical_key(canonical)
        if key in seen:
            continue
        seen.add(key)
        report = _green_constraint_report(state, canonical)
        objectives = _safe_green_score(module, state, canonical, previous_solution, report)
        if report["feasible"]:
            out.append({"solution": canonical, "objectives": {name: float(objectives.get(name, 0.0)) for name in GREEN_OBJECTIVES}})
    return out


def normalize_green_solution_ids(state: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(solution, dict):
        return {}
    vehicles = {str(item.get("id")) for item in state.get("vehicles", []) if isinstance(item, dict) and item.get("id") not in (None, "")}
    orders = {str(item.get("id")) for item in state.get("orders", []) if isinstance(item, dict) and item.get("id") not in (None, "")}
    routes = []
    for route in solution.get("routes", []) or []:
        if not isinstance(route, dict):
            continue
        vehicle = normalize_public_id(str(route.get("vehicle") or route.get("vehicle_id") or ""), vehicles)
        order_ids = [normalize_public_id(str(order), orders) for order in route.get("orders", []) or [] if order not in (None, "")]
        routes.append({**route, "vehicle": vehicle, "orders": order_ids})
    return {**solution, "routes": routes}


def normalize_public_id(value: str, known_ids: set[str]) -> str:
    if value in known_ids:
        return value
    parsed = _id_prefix_number(value)
    if parsed is None:
        return value
    prefix, number = parsed
    for known in known_ids:
        if _id_prefix_number(known) == (prefix, number):
            return known
    return value


def _id_prefix_number(value: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"([A-Za-z_-]+)0*(\d+)", str(value or ""))
    if not match:
        return None
    return match.group(1), int(match.group(2))


def hidden_cloud_archive(
    module: Any,
    state: dict[str, Any],
    final_solution: dict[str, Any],
    previous_solution: dict[str, Any] | None,
    candidate_archive: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = candidate_solutions(candidate_archive, final_solution)
    out = []
    seen = set()
    for solution in candidates:
        canonical = canonical_cloud(solution)
        key = canonical_key(canonical)
        if key in seen:
            continue
        seen.add(key)
        objectives = module._score_solution(state, canonical, previous_solution)
        feasible = objectives.get("resource_violations", 0.0) == 0.0
        if feasible:
            out.append({"solution": canonical, "objectives": {name: float(objectives.get(name, 0.0)) for name in CLOUD_OBJECTIVES}})
    return out


def candidate_solutions(candidate_archive: list[dict[str, Any]], final_solution: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    if isinstance(final_solution, dict) and final_solution:
        out.append(final_solution)
    for item in candidate_archive or []:
        if isinstance(item, dict) and isinstance(item.get("solution"), dict):
            out.append(item["solution"])
    return out


def canonical_key(solution: dict[str, Any]) -> str:
    return json.dumps(solution, ensure_ascii=False, sort_keys=True)


def scalar_metric(objective: float, feasible: bool, reference_step: dict[str, Any], canonical: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    reference = to_float(reference_step.get("objective"))
    if reference is None:
        reference = to_float(reference_step.get("score", {}).get("objective") if isinstance(reference_step.get("score"), dict) else None)
    gap = objective - reference if reference is not None and math.isfinite(objective) else None
    quality = normalized_scalar_score(objective, reference, feasible)
    out = {
        "feasible": bool(feasible),
        "objective": round(float(objective), 6) if math.isfinite(objective) else None,
        "reference_objective": reference,
        "objective_gap": round(float(gap), 6) if gap is not None and math.isfinite(gap) else None,
        "normalized_score": quality,
        "canonical_solution": canonical,
    }
    if extra:
        out.update({f"score_{key}": value for key, value in extra.items() if isinstance(value, (int, float))})
    return out


def normalized_scalar_score(objective: float, reference: float | None, feasible: bool) -> float:
    if not feasible or reference is None or not math.isfinite(objective):
        return 0.0
    gap = max(0.0, objective - reference)
    if abs(reference) < 1e-12:
        return round(max(0.0, min(1.0, 1.0 / (1.0 + gap))), 6)
    denom = max(abs(objective), abs(reference), 1.0)
    return round(max(0.0, min(1.0, 1.0 - gap / denom)), 6)


def mo_metric(canonical: dict[str, Any], candidate_archive: list[dict[str, Any]], objective_names: list[str], reference_step: dict[str, Any]) -> dict[str, Any]:
    reference_archive = reference_step.get("pareto_archive") or []
    metrics = evaluate_against_reference(
        candidate_archive,
        reference_archive,
        objective_names,
        reference_step.get("hypervolume_reference_point"),
        reference_step.get("hypervolume_ideal_point"),
        hv_samples=0,
    )
    igd_value = metrics.get("igd")
    igd_score = round(1.0 / (1.0 + float(igd_value)), 6) if is_finite(igd_value) else 0.0
    hv_ratio = round(float(metrics.get("hv_ratio", 0.0)), 6)
    normalized_score = hv_ratio if float(metrics.get("reference_hv", 0.0) or 0.0) > 0 else igd_score
    scalar_objectives = [sum(float(item.get("objectives", {}).get(name, 0.0)) for name in objective_names) for item in candidate_archive]
    return {
        "feasible": bool(candidate_archive),
        "objective": round(min(scalar_objectives), 6) if scalar_objectives else None,
        "normalized_score": normalized_score,
        "hv": metrics.get("hv", 0.0),
        "reference_hv": metrics.get("reference_hv", 0.0),
        "normalized_hv": hv_ratio,
        "igd": igd_value,
        "igd_score": igd_score,
        "ideal_gap": metrics.get("ideal_gap"),
        "canonical_solution": canonical,
        "hidden_candidate_archive_size": len(candidate_archive),
        "hidden_candidate_archive": candidate_archive,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["mode"], int(row["stage_index"]))].append(row)
    out = []
    for (benchmark, mode, stage_index), items in sorted(grouped.items()):
        out.append(summary_row({"benchmark": benchmark, "mode": mode, "stage_index": stage_index}, items))
    return out


def total_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["mode"])].append(row)
    out = []
    for (benchmark, mode), items in sorted(grouped.items()):
        out.append(summary_row({"benchmark": benchmark, "mode": mode, "stage_index": "all"}, items))
    return out


def summary_row(prefix: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    dynamic = [item for item in items if int(item["stage_index"]) > 0]
    seed_values = {item.get("run_seed") for item in items if item.get("run_seed") is not None}
    return {
        **prefix,
        "episodes": len(set(item["episode_id"] for item in items)),
        "stages": len(items),
        "stage_observations": len(items),
        "unique_stage_count": len(set(int(item["stage_index"]) for item in items)),
        "seed_run_count": len(seed_values) if seed_values else None,
        "feasibility_acc": avg(1.0 if item["feasible"] else 0.0 for item in items),
        "dynamic_feasibility_acc": avg(1.0 if item["feasible"] else 0.0 for item in dynamic),
        "true_pass_ratio": avg(1.0 if item.get("true_pass") else 0.0 for item in dynamic),
        "normalized_total_score": avg(item.get("normalized_score") for item in items),
        "dynamic_normalized_score": avg(item.get("normalized_score") for item in dynamic),
        "objective_gap_mean": avg(item.get("objective_gap") for item in items),
        "normalized_hv_mean": avg(item.get("normalized_hv") for item in items),
        "igd_mean": avg(item.get("igd") for item in items),
        "ideal_gap_mean": avg(item.get("ideal_gap") for item in items),
        "token_mean": avg(item.get("tokens") for item in items),
        "latency_mean": avg(item.get("latency_seconds") for item in items),
    }


def avg(values: Any) -> float | None:
    clean = [float(value) for value in values if value is not None and is_finite(value)]
    return round(mean(clean), 6) if clean else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def plot_outputs(summary: list[dict[str, Any]], out_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]
    outputs: list[str] = []
    for metric, ylabel, filename in [
        ("normalized_total_score", "Reference-normalized score", "reference_normalized_score.png"),
        ("feasibility_acc", "Feasibility", "reference_feasibility.png"),
        ("true_pass_ratio", "True update pass ratio", "reference_true_pass_ratio.png"),
        ("normalized_hv_mean", "HV ratio", "reference_hv_ratio.png"),
        ("igd_mean", "IGD", "reference_igd.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in summary:
            if row.get(metric) is None:
                continue
            series[(row["benchmark"], row["mode"])].append(row)
        for (benchmark, mode), points in sorted(series.items()):
            points.sort(key=lambda item: int(item["stage_index"]))
            ax.plot([p["stage_index"] for p in points], [p[metric] for p in points], marker="o", linewidth=1.4, label=f"{benchmark}/{mode}")
        ax.set_xlabel("Stage index (0=initial)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if series:
            ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if is_finite(value) else None
    except (TypeError, ValueError):
        return None


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
