#!/usr/bin/env python3
"""Churn-discriminator branch suite for the LiveOpt semantic restart gate.

Each branch forks LiveOpt's committed P010/P015 trajectory at one mid-sequence
stage (default t05) and applies a programmatic public-table update whose
deterministic structural table-churn ratio is >= 0.5 (so the appendix A.5
churn rule fires Full) while population reuse remains safe.  Branch tables are
built deterministically in memory; the frozen Workbench slots from the
committed trajectory are reused unchanged, so schema and semantics are
preserved by construction.

Recorded per branch:

- the deterministic structural churn decision (per stage),
- the sensor objective-landscape decision (per numerical seed),
- the semantic restart gate decision (one real provider call per branch,
  paper prompt version liveopt_semantic_restart_gate_v2, fresh cache),
- Fixed Warm and Fixed Full matched-population arms executed from LiveOpt's
  committed, seed-matched incoming population.

Branches have no hidden reference.  Arms are scored relatively: per branch,
all arms' feasible archives are pooled, the nondominated union is the
reference, and each arm reports its normalized-HV ratio against that union.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import (  # noqa: E402
    ScaffoldRestartSeedBuilder,
    apply_public_context_patch,
    normalize_public_context_tables,
    select_restart_skill_from_landscape_change,
)
from evo2.agents.liveopt_workbench_impl import ScaffoldTrace, compile_scaffold_project  # noqa: E402
from evo2.agents.semantic_restart_gate import LiveOptSemanticRestartGate  # noqa: E402
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult  # noqa: E402
from evo2.evaluation.moea_metrics import (  # noqa: E402
    approximate_hypervolume,
    ideal_point,
    nondominated,
    reference_point,
)
from scripts.analysis.export_liveopt_structural_restart_control import (  # noqa: E402
    active_table_replacement,
)
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import (  # noqa: E402
    evolution_result_from_source_update,
)
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (  # noqa: E402
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
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_churn_discriminator_20260721"
FORK_STAGE = 5
CHURN_THRESHOLD = 0.50
BRANCH_KINDS = ("unit_scale", "relabel", "easy_inject")
EPISODES = ("NLDO-P010", "NLDO-P015")
ARMS = ("fixed_warm", "fixed_full")
ARM_RESTART_SKILLS = {"fixed_warm": "warm_restart_v1", "fixed_full": "full_restart_v1"}

UPDATE_TEXTS = {
    ("NLDO-P010", "unit_scale"): (
        "The dispatch data platform has switched to a new measurement convention: every "
        "coordinate, time value, and load quantity in the public depot, order, and vehicle "
        "tables is now recorded in tenths of the former unit, so all of those numbers are ten "
        "times larger than before. Vehicle capacities and shift limits are converted the same "
        "way. All serving rules, order priorities, and emission parameters are unchanged."
    ),
    ("NLDO-P015", "unit_scale"): (
        "The cluster telemetry pipeline now reports resource quantities in tenths of the "
        "former unit: job cpu and memory requests, machine cpu and memory capacities, and "
        "machine idle energy figures are all ten times larger than before, and job deadlines, "
        "latency sensitivities, and priorities are recorded on the same rescaled convention. "
        "Assignment rules, gpu requirements, and per-unit energy rates are unchanged."
    ),
    ("NLDO-P010", "relabel"): (
        "The customer account system was migrated this morning: every order identifier in the "
        "public order table has been re-issued under the new customer-code scheme, and the "
        "fleet registry re-registered every vehicle under new identifiers. Coordinates, time "
        "windows, demands, capacities, and all other operating parameters are exactly as "
        "before."
    ),
    ("NLDO-P015", "relabel"): (
        "The workload scheduler's job registry was migrated to a new tracking system: every "
        "job identifier in the public job table has been re-issued under new task codes, and "
        "every machine was re-registered under a new identifier. Resource requests, "
        "capacities, deadlines, and all other operating parameters are exactly as before."
    ),
    ("NLDO-P010", "easy_inject"): (
        "Twenty new same-day orders have been added to the public order table and are now "
        "active. All of them are located close to the depot with small demand, short service, "
        "and wide delivery windows. All previously active orders, all vehicles, and the "
        "carbon policy are unchanged."
    ),
    ("NLDO-P015", "easy_inject"): (
        "Twenty-six new small jobs have been added to the public job table and are now "
        "active. Each has a minimal cpu and memory request, no gpu requirement, and a loose "
        "deadline. All previously active jobs and all machines are unchanged."
    ),
}

# Numeric fields multiplied by 10 in the unit_scale branches.  Only fields
# whose scaling keeps the feasibility set identical and every objective
# uniformly x10 are touched.  For P015 the frozen Workbench uses job
# deadline/latency_sensitivity/priority only in diagnostics (not in the two
# objectives or any constraint); they are scaled anyway so that the jobs-table
# churn share of the deterministic rule reflects the unit conversion.
UNIT_SCALE_FIELDS = {
    "NLDO-P010": {
        "depot": ("x", "y"),
        "orders": ("x", "y", "demand", "ready", "due", "service"),
        "vehicles": ("capacity", "shift_end"),
        "policy": (),
    },
    "NLDO-P015": {
        "jobs": ("cpu", "mem", "deadline", "latency_sensitivity", "priority"),
        "machines": ("cpu", "mem", "energy_idle"),
        "global_params": (),
    },
}

RELABEL_PREFIX = {"O": "C", "V": "W", "J": "T", "M": "N"}
RELABEL_EXACT = {"DEPOT": "HUB"}

# easy_inject row templates (deterministic; ids are offset past the largest
# existing numeric suffix in the target table at build time).
EASY_INJECT_COUNT = {"NLDO-P010": 20, "NLDO-P015": 26}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--branch-id", action="append", default=[])
    parser.add_argument("--fork-stage", type=int, default=FORK_STAGE)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--population-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--archive-limit", type=int, default=500)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--skip-sensor", action="store_true")
    parser.add_argument("--skip-arms", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def value_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_source_rows(path: Path) -> dict[int, dict[str, Any]]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {int(row["run_seed"]): row for row in rows}


def context_at_stage(episode: dict[str, Any], source_row: dict[str, Any], stage: int) -> dict[str, Any]:
    context = public_context_with_loaded_tables(episode)
    for update in list(source_row.get("update_results") or [])[:stage]:
        context = apply_public_context_patch(
            context, (update.get("restart") or {}).get("data_patch") or {}
        )
    return normalize_public_context_tables(copy.deepcopy(context))


def stage_slots(source_row: dict[str, Any], stage: int) -> dict[str, str]:
    update = source_row["update_results"][stage - 1]
    return {"setup.py": update["setup_code"], "fitness.py": update["fitness_code"]}


def compile_stage_project(episode_id: str, context: dict[str, Any], slots: dict[str, str]):
    return compile_scaffold_project(
        episode_id,
        context,
        slots,
        ScaffoldTrace(model="frozen_committed_slots"),
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


# ---------------------------------------------------------------------------
# Branch table transforms (programmatic, deterministic; no LLM data patching)
# ---------------------------------------------------------------------------


def unit_scale_context(episode_id: str, context: dict[str, Any], factor: float = 10.0) -> dict[str, Any]:
    out = copy.deepcopy(context)
    tables = out.get("tables") or {}
    for table_name, fields in UNIT_SCALE_FIELDS[episode_id].items():
        for row in tables.get(table_name) or []:
            for field in fields:
                if field in row and isinstance(row[field], (int, float)) and not isinstance(row[field], bool):
                    value = row[field] * factor
                    row[field] = int(value) if isinstance(row[field], int) else value
    return out


def relabel_id(value: Any) -> Any:
    text = str(value)
    if text in RELABEL_EXACT:
        return RELABEL_EXACT[text]
    if text and text[0] in RELABEL_PREFIX:
        return RELABEL_PREFIX[text[0]] + text[1:]
    return text


def relabel_context(episode_id: str, context: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(context)
    tables = out.get("tables") or {}
    mapping: dict[str, str] = {}
    for rows in tables.values():
        for row in rows or []:
            if isinstance(row, dict) and row.get("id") is not None:
                old = str(row["id"])
                new = relabel_id(old)
                if new in mapping and mapping[new] != old:
                    raise ValueError(f"relabel id collision: {old} -> {new}")
                mapping[old] = new
                row["id"] = new
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("relabel mapping is not injective")
    out["relabel_id_mapping"] = mapping
    return out


def _next_ids(existing: list[str], prefix: str, count: int) -> list[str]:
    suffixes = []
    for value in existing:
        text = str(value)
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            suffixes.append(int(text[len(prefix):]))
    start = max(suffixes, default=0) + 1
    width = max(3, len(str(start + count)))
    return [f"{prefix}{index:0{width}d}" for index in range(start, start + count)]


def easy_inject_context(episode_id: str, context: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(context)
    tables = out.get("tables") or {}
    count = EASY_INJECT_COUNT[episode_id]
    if episode_id == "NLDO-P010":
        rows = tables.get("orders") or []
        ids = _next_ids([str(row.get("id")) for row in rows], "O", count)
        due = max(
            [float(row.get("shift_end") or 0) for row in tables.get("vehicles") or []] + [300.0]
        )
        for index, order_id in enumerate(ids):
            rows.append(
                {
                    "id": order_id,
                    "x": 1 + (index % 5),
                    "y": 1 + (index // 5),
                    "demand": 1,
                    "ready": 0,
                    "due": due,
                    "service": 2,
                    "priority": 1,
                    "active": True,
                }
            )
    elif episode_id == "NLDO-P015":
        rows = tables.get("jobs") or []
        ids = _next_ids([str(row.get("id")) for row in rows], "J", count)
        deadline = max([float(row.get("deadline") or 0) for row in rows] + [100.0]) * 2
        for job_id in ids:
            rows.append(
                {
                    "id": job_id,
                    "burst": "",
                    "cpu": 1,
                    "mem": 1,
                    "deadline": deadline,
                    "latency_sensitivity": 1.0,
                    "priority": 1,
                    "gpu_required": False,
                    "active": True,
                }
            )
    else:  # pragma: no cover - guarded by branch specs
        raise ValueError(f"unsupported episode for easy_inject: {episode_id}")
    return out


BRANCH_TRANSFORMS = {
    "unit_scale": unit_scale_context,
    "relabel": relabel_context,
    "easy_inject": easy_inject_context,
}


def branch_specs() -> list[dict[str, Any]]:
    specs = []
    for episode_id in EPISODES:
        for kind in BRANCH_KINDS:
            specs.append(
                {
                    "branch_id": f"{episode_id}-{kind}",
                    "episode_id": episode_id,
                    "kind": kind,
                    "update_text": UPDATE_TEXTS[(episode_id, kind)],
                }
            )
    return specs


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def churn_decision(
    previous_context: dict[str, Any],
    current_context: dict[str, Any],
    threshold: float = CHURN_THRESHOLD,
) -> dict[str, Any]:
    """Deterministic structural table-churn rule (appendix A.5).

    Full iff the largest changed non-key-cell fraction among rows active after
    the update (newly active rows count 100%) is at least the threshold.
    """

    table, ratio, changed, compared, active = active_table_replacement(
        previous_context, current_context
    )
    return {
        "policy": "structural_table_churn",
        "decision_table": table,
        "active_rows": active,
        "changed_cells": changed,
        "compared_cells": compared,
        "churn_ratio": ratio,
        "threshold": threshold,
        "selected_action": "Full" if ratio >= threshold else "Warm",
        "selected_restart_skill": "full_restart_v1" if ratio >= threshold else "warm_restart_v1",
    }


# ---------------------------------------------------------------------------
# Scoring (relative; branches have no hidden reference)
# ---------------------------------------------------------------------------


def objective_items(result: EvolutionResult, objective_names: list[str]) -> list[dict[str, Any]]:
    candidates = list(result.archive or result.population or [])
    items = []
    for candidate in candidates:
        fitness = candidate.result
        if fitness is None or not fitness.feasible:
            continue
        objectives = list(fitness.objectives or [])
        if len(objectives) != len(objective_names):
            continue
        items.append({"objectives": dict(zip(objective_names, (float(v) for v in objectives)))})
    return items


def relative_hv_ratio(
    arm_items: list[dict[str, Any]],
    union_items: list[dict[str, Any]],
    pooled_items: list[dict[str, Any]],
    objective_names: list[str],
) -> tuple[float, float, float]:
    """Return (arm_hv, union_hv, hv_ratio) on a shared normalization box."""

    if not union_items or not pooled_items:
        return 0.0, 0.0, 0.0
    ref = reference_point(union_items, objective_names)
    ideal = ideal_point(pooled_items, objective_names)
    union_hv = approximate_hypervolume(union_items, objective_names, ref=ref, ideal=ideal)
    if union_hv <= 0:
        return 0.0, union_hv, 0.0
    arm_hv = approximate_hypervolume(arm_items, objective_names, ref=ref, ideal=ideal) if arm_items else 0.0
    return arm_hv, union_hv, arm_hv / union_hv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    seeds = [int(token) for token in str(args.seeds).split(",") if token.strip()]
    specs = branch_specs()
    if args.branch_id:
        wanted = set(args.branch_id)
        unknown = sorted(wanted - {spec["branch_id"] for spec in specs})
        if unknown:
            raise ValueError(f"unknown branch ids: {unknown}")
        specs = [spec for spec in specs if spec["branch_id"] in wanted]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.out_dir / "decisions.jsonl"
    metrics_path = args.out_dir / "stage_metrics.csv"
    specs_path = args.out_dir / "branch_specs.json"
    if not args.resume:
        for path in (decisions_path, metrics_path, specs_path):
            if path.exists():
                path.unlink()
    completed_metrics: set[tuple[str, str, int]] = set()
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                completed_metrics.add((row["branch_id"], row["arm"], int(row["seed"])))
    recorded_decisions: set[str] = set()
    if decisions_path.exists():
        with decisions_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    recorded_decisions.add(
                        f"{row.get('branch_id')}|{row.get('policy')}|{row.get('seed', '')}"
                    )

    os.environ.setdefault("DEEPSEEK_MAX_ATTEMPTS", "3")
    os.environ.setdefault("DEEPSEEK_HARD_TIMEOUT", "360")
    os.environ.setdefault("DEEPSEEK_DEFAULT_UNLIMITED_MAX_TOKENS", "0")
    os.environ.setdefault("DEEPSEEK_PRO_MIN_MAX_TOKENS", "8000")
    os.environ["DEEPSEEK_TRACE_DIR"] = str(args.out_dir / "provider_trace")

    episodes = {episode["episode_id"]: episode for episode in load_episodes(BENCHMARK)}
    sources = {episode_id: load_source_rows(path) for episode_id, path in SOURCE_PATHS.items()}
    builder = ScaffoldRestartSeedBuilder()
    errors: list[str] = []
    spec_records: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))

    for spec in specs:
        branch_id = spec["branch_id"]
        episode_id = spec["episode_id"]
        kind = spec["kind"]
        transform = BRANCH_TRANSFORMS[kind]
        rows = sources[episode_id]
        missing_seeds = sorted(set(seeds) - set(rows))
        if missing_seeds:
            errors.append(f"{branch_id}: source rows lack seeds {missing_seeds}")
            continue
        seed_zero = rows[0]
        slots = stage_slots(seed_zero, args.fork_stage)
        for seed in seeds:
            if stage_slots(rows[seed], args.fork_stage) != slots:
                errors.append(f"{branch_id}: seed {seed} Workbench slots differ from seed 0")
        previous_context = context_at_stage(episodes[episode_id], seed_zero, args.fork_stage)
        branch_context = transform(episode_id, previous_context)
        previous_project = compile_stage_project(episode_id, previous_context, slots)
        branch_project = compile_stage_project(episode_id, branch_context, slots)
        history = public_history(seed_zero, args.fork_stage)

        churn = churn_decision(previous_context, branch_context)
        if churn["selected_action"] != "Full":
            errors.append(
                f"{branch_id}: churn ratio {churn['churn_ratio']:.4f} below "
                f"{churn['threshold']:.2f}; branch does not discriminate"
            )

        decision_key = f"{branch_id}|structural_table_churn|"
        if decision_key not in recorded_decisions:
            append_jsonl(
                decisions_path,
                {
                    "branch_id": branch_id,
                    "episode_id": episode_id,
                    "kind": kind,
                    "fork_stage": args.fork_stage,
                    "seed": "",
                    **churn,
                },
            )

        # Sensor objective-landscape gate (per seed; no provider calls).
        if not args.skip_sensor:
            for seed in seeds:
                decision_key = f"{branch_id}|sensor_landscape|{seed}"
                if decision_key in recorded_decisions:
                    continue
                previous_result = result_at_stage(rows[seed], args.fork_stage)
                sensor = select_restart_skill_from_landscape_change(
                    previous_project=previous_project,
                    project=branch_project,
                    previous_result=previous_result,
                    seed=seed,
                    stage_index=args.fork_stage + 1,
                )
                record = sensor.to_record()
                append_jsonl(
                    decisions_path,
                    {
                        "branch_id": branch_id,
                        "episode_id": episode_id,
                        "kind": kind,
                        "fork_stage": args.fork_stage,
                        "seed": seed,
                        "policy": "sensor_landscape",
                        "selected_action": "Full"
                        if record["selected_restart_skill"] == "full_restart_v1"
                        else "Warm",
                        **record,
                    },
                )

        # Semantic restart gate (one real provider call per branch; fresh cache).
        gate_record: dict[str, Any] = {}
        if not args.skip_gate:
            decision_key = f"{branch_id}|semantic_restart_gate|"
            if decision_key not in recorded_decisions:
                gate = LiveOptSemanticRestartGate(
                    str(episodes[episode_id].get("public_initial_problem") or ""),
                    model=args.model,
                    cache_dir=args.out_dir / "gate_cache" / branch_id,
                )
                gate.prime_public_history(history)
                started = time.perf_counter()
                decision, trace = gate.decide(
                    update_id=f"CHURN-{branch_id}",
                    natural_language_update=spec["update_text"],
                    previous_public_context=previous_context,
                    public_context=branch_context,
                    previous_project=previous_project,
                    project=branch_project,
                )
                gate_record = decision.to_record()
                gate_record.update(
                    {
                        "branch_id": branch_id,
                        "episode_id": episode_id,
                        "kind": kind,
                        "fork_stage": args.fork_stage,
                        "seed": "",
                        "policy": "semantic_restart_gate",
                        "selected_action": "Full" if decision.verified_full_vote else "Warm",
                        "selected_restart_skill": "full_restart_v1"
                        if decision.verified_full_vote
                        else "warm_restart_v1",
                        "history_length": len(history),
                        "digest_changed_field_paths": list(
                            (gate.history[-1].get("digest") or {}).get("changed_field_paths") or []
                        ),
                        "digest_table_changes": (gate.history[-1].get("digest") or {}).get(
                            "table_changes"
                        ),
                        "gate_latency_seconds": time.perf_counter() - started,
                        "gate_usage": trace.usage,
                        "gate_errors": trace.errors,
                    },
                )
                append_jsonl(decisions_path, gate_record)
                print(
                    json.dumps(
                        {
                            "branch": branch_id,
                            "gate_full_vote": decision.full_vote,
                            "verified_full_vote": decision.verified_full_vote,
                            "reuse_risk": decision.reuse_risk,
                            "reason": decision.reason[:160],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        # Matched-population arms (same incoming population per seed).
        if not args.skip_arms:
            for seed in seeds:
                previous_result = result_at_stage(rows[seed], args.fork_stage)
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
                for arm in ARMS:
                    if (branch_id, arm, seed) in completed_metrics:
                        continue
                    starts, restart_metadata = builder.build(
                        restart_skill=ARM_RESTART_SKILLS[arm],
                        previous_result=previous_result,
                        segments=branch_project.segments,
                        population_size=config.population_size,
                        seed=seed,
                        evaluate=branch_project.evaluate,
                        data=branch_project.data,
                    )
                    started = time.perf_counter()
                    result = branch_project.run(config, initial_genomes=starts)
                    elapsed = time.perf_counter() - started
                    archive = list(result.archive or result.population or [])
                    feasible = [
                        candidate
                        for candidate in archive
                        if candidate.result and candidate.result.feasible
                    ]
                    objective_names = list(
                        (branch_project.problem_spec or {}).get("objective_names") or []
                    )
                    archive_dir = args.out_dir / "archives"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    (archive_dir / f"{branch_id}__{arm}__{seed}.json").write_text(
                        json.dumps(
                            {
                                "branch_id": branch_id,
                                "arm": arm,
                                "seed": seed,
                                "objective_names": objective_names,
                                "items": objective_items(result, objective_names),
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    row = {
                        "branch_id": branch_id,
                        "episode_id": episode_id,
                        "kind": kind,
                        "fork_stage": args.fork_stage,
                        "arm": arm,
                        "seed": seed,
                        "population_size": config.population_size,
                        "generation_cap": config.generations,
                        "executed_generations": int(result.metadata.get("executed_generations") or 0),
                        "restart_skill": ARM_RESTART_SKILLS[arm],
                        "restart_seed_count": len(starts),
                        "restart_sha256": value_hash(starts),
                        "incoming_population_count": len(previous_result.population),
                        "archive_size": len(archive),
                        "feasible_archive_size": len(feasible),
                        "latency_seconds": round(elapsed, 3),
                    }
                    metric_rows.append(row)
                    print(
                        json.dumps(
                            {
                                "branch": branch_id,
                                "arm": arm,
                                "seed": seed,
                                "generations": row["executed_generations"],
                                "feasible_archive": row["feasible_archive_size"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

        spec_records.append(
            {
                "branch_id": branch_id,
                "episode_id": episode_id,
                "kind": kind,
                "fork_stage": args.fork_stage,
                "update_id": f"CHURN-{branch_id}",
                "update_text": spec["update_text"],
                "source_artifact": str(SOURCE_PATHS[episode_id].relative_to(ROOT)),
                "slots_sha256": value_hash(slots),
                "slots_identical_to_committed_stage": True,
                "history_length": len(history),
                "history_sha256": value_hash(history),
                "churn": churn,
                "unit_scale_fields": UNIT_SCALE_FIELDS.get(episode_id) if kind == "unit_scale" else None,
                "previous_segment_signature": [
                    {"name": s.name, "kind": s.kind, "demand_count": len(s.demands or []), "resource_count": len(s.resources or []), "value_count": len(s.values or [])}
                    for s in previous_project.segments
                ],
                "branch_segment_signature": [
                    {"name": s.name, "kind": s.kind, "demand_count": len(s.demands or []), "resource_count": len(s.resources or []), "value_count": len(s.values or [])}
                    for s in branch_project.segments
                ],
                "objective_names": list((branch_project.problem_spec or {}).get("objective_names") or []),
            }
        )

    # Relative scoring: per branch, pool all arms' feasible archives and use
    # the nondominated union as the reference front.  Archive objective items
    # are persisted per arm so scoring also covers rows from earlier resumes.
    if not args.skip_arms:
        archive_dir = args.out_dir / "archives"
        stored: dict[str, dict[str, Any]] = {}
        if archive_dir.exists():
            for path in sorted(archive_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                stored[f"{payload['branch_id']}|{payload['arm']}|{payload['seed']}"] = payload
        objective_names_by_branch = {
            record["branch_id"]: record["objective_names"] for record in spec_records
        }
        scored_rows = []
        by_branch = {}
        for row in metric_rows:
            by_branch.setdefault(str(row["branch_id"]), []).append(row)
        for branch_id, rows_for_branch in sorted(by_branch.items()):
            objective_names = objective_names_by_branch.get(branch_id) or []
            branch_items = {
                key: payload
                for key, payload in stored.items()
                if payload.get("branch_id") == branch_id
            }
            pooled = [
                item
                for payload in branch_items.values()
                for item in (payload.get("items") or [])
            ]
            union = nondominated(pooled, objective_names) if pooled and objective_names else []
            for row in rows_for_branch:
                payload = branch_items.get(f"{row['branch_id']}|{row['arm']}|{row['seed']}")
                items = (payload or {}).get("items")
                if items is None or not objective_names:
                    scored_rows.append({**row, "hv": "", "union_hv": "", "hv_ratio": ""})
                    continue
                arm_hv, union_hv, ratio = relative_hv_ratio(items, union, pooled, objective_names)
                scored_rows.append({**row, "hv": arm_hv, "union_hv": union_hv, "hv_ratio": ratio})
        metric_rows = scored_rows

    if metric_rows:
        fields: list[str] = []
        for row in metric_rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metric_rows)

    specs_payload = {
        "protocol": "liveopt_churn_discriminator_v1",
        "fork_stage": args.fork_stage,
        "churn_threshold": CHURN_THRESHOLD,
        "seeds": seeds,
        "population_size": args.population_size,
        "generations": args.generations,
        "archive_limit": args.archive_limit,
        "early_stopping": {
            "reference_free_early_stopping": True,
            "early_stop_min_generations": 40,
            "early_stop_patience": 25,
            "early_stop_hv_epsilon": 5e-4,
            "early_stop_scalar_epsilon": 5e-4,
            "early_stop_epsilon_box": 0.01,
            "early_stop_min_archive_size": 8,
        },
        "branches": spec_records,
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }
    specs_path.write_text(json.dumps(specs_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": specs_payload["status"], "errors": errors}, ensure_ascii=False))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
