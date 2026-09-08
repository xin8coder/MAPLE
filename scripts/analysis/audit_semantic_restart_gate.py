#!/usr/bin/env python3
"""Audit the cached, public-only semantic restart gate on NLDO late updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.agents.liveopt_dynamic_impl import apply_public_context_patch, normalize_public_context_tables
from evo2.agents.liveopt_workbench_impl import ScaffoldProject, ScaffoldTrace, compile_scaffold_project
from evo2.agents.semantic_restart_gate import LiveOptSemanticRestartGate
from evo2.benchmarks.optimization_contracts import public_optimization_contract_for_episode
from evo2.core.template_optimizer import FitnessResult, SegmentSpec
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (
    load_episodes,
    public_context_with_loaded_tables,
)


DEFAULT_EPISODE_IDS = [f"NLDO-P{index:03d}" for index in range(7, 16)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
    )
    parser.add_argument("--episode-id", action="append")
    parser.add_argument(
        "--source-run",
        type=Path,
        action="append",
        help="Saved public LiveOpt run directory used to audit every update from committed public data patches and slots.",
    )
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/semantic_restart_gate_audit/cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/semantic_restart_gate_audit/decisions.json"),
    )
    parser.add_argument("--no-verify-choice-cache", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("DEEPSEEK_CACHE", "1")
    os.environ["DEEPSEEK_CACHE_DIR"] = str(args.cache_dir / "provider")
    os.environ["LIVEOPT_SEMANTIC_RESTART_CACHE_DIR"] = str(args.cache_dir / "semantic")

    selected = set(args.episode_id or DEFAULT_EPISODE_IDS)
    episodes = [episode for episode in load_episodes(args.episodes_jsonl) if episode.get("episode_id") in selected]
    source_rows = load_source_rows(args.source_run or [])
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        source_row = source_rows.get(str(episode.get("episode_id")))
        if args.all_stages and source_row is None:
            raise ValueError(f"--all-stages requires a --source-run row for {episode['episode_id']}")
        first_pass = run_episode_gate(
            episode,
            model=args.model,
            cache_dir=args.cache_dir / "semantic",
            source_row=source_row,
            all_stages=args.all_stages,
        )
        second_pass = []
        if not args.no_verify_choice_cache:
            second_pass = run_episode_gate(
                episode,
                model=args.model,
                cache_dir=args.cache_dir / "semantic",
                source_row=source_row,
                all_stages=args.all_stages,
            )
            if not all(row["decision"]["cache_hit"] for row in second_pass):
                raise AssertionError(f"parsed-choice cache did not hit for every late update in {episode['episode_id']}")
        for index, row in enumerate(first_pass):
            if second_pass:
                row["repeat_cache_hit"] = bool(second_pass[index]["decision"]["cache_hit"])
            rows.append(row)

    output = {
        "model": args.model,
        "episodes_jsonl": str(args.episodes_jsonl),
        "episodes_jsonl_sha256": sha256_file(args.episodes_jsonl),
        "source_runs": [str(path) for path in (args.source_run or [])],
        "all_stages": bool(args.all_stages),
        "episodes": len(episodes),
        "decisions": len(rows),
        "verified_full_votes": sum(bool(row["decision"]["verified_full_vote"]) for row in rows),
        "repeat_cache_hits": sum(bool(row.get("repeat_cache_hit")) for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_episode_gate(
    episode: dict[str, Any],
    *,
    model: str,
    cache_dir: Path,
    source_row: dict[str, Any] | None = None,
    all_stages: bool = False,
) -> list[dict[str, Any]]:
    updates = list(episode.get("update_stream") or [])
    context = normalize_public_context_tables(public_context_with_loaded_tables(episode))
    gate = LiveOptSemanticRestartGate(
        str(episode.get("public_initial_problem") or ""),
        model=model,
        cache_dir=cache_dir,
    )
    if not all_stages:
        gate.prime_public_history(updates[:10])
    rows = []
    previous_project = (
        compile_scaffold_project(
            str(episode.get("episode_id")),
            context,
            {"setup.py": str(source_row["setup_code"]), "fitness.py": str(source_row["fitness_code"])},
            ScaffoldTrace(model="saved-public-artifact"),
        )
        if source_row is not None
        else probe_project(episode, context)
    )
    source_updates = list((source_row or {}).get("update_results") or [])
    stage_indices = range(1, len(updates) + 1) if all_stages else (11, 12)
    for stage_index in stage_indices:
        update = updates[stage_index - 1]
        source_update = source_updates[stage_index - 1] if stage_index <= len(source_updates) else {}
        # Match artifact replay exactly: a current benchmark-authored public
        # patch supersedes any saved patch from an older source trajectory.
        # This is essential when the benchmark is rematerialized while the
        # verified table-driven Workbench and t00--t10 prefix are reused.
        patch = (
            update.get("public_data_patch")
            or (source_update.get("restart") or {}).get("data_patch")
            or {}
        )
        if not patch.get("operations"):
            patch = {"operations": [], "reason": "no public data change"}
        new_context = apply_public_context_patch(context, patch)
        project = (
            compile_scaffold_project(
                str(episode.get("episode_id")),
                new_context,
                {
                    "setup.py": str(source_update.get("setup_code") or previous_project.setup_code),
                    "fitness.py": str(source_update.get("fitness_code") or previous_project.fitness_code),
                },
                ScaffoldTrace(model="saved-public-artifact"),
            )
            if source_row is not None
            else probe_project(episode, new_context)
        )
        decision, trace = gate.decide(
            update_id=str(update.get("update_id") or f"event-{stage_index}"),
            natural_language_update=str(update.get("natural_language_update") or update.get("public_update") or ""),
            previous_public_context=context,
            public_context=new_context,
            previous_project=previous_project,
            project=project,
        )
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "stage": stage_index,
                "decision": decision.to_record(),
                "usage": trace.usage,
                "errors": trace.errors,
            }
        )
        context = new_context
        previous_project = project
    return rows


def load_source_rows(run_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        path = run_dir / "NLDO" / "evo2_limit0.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id") or "")
            seed = int(row.get("run_seed", 0) or 0)
            if episode_id and (episode_id not in rows or seed == 0):
                rows[episode_id] = row
    return rows


def probe_project(episode: dict[str, Any], context: dict[str, Any]) -> ScaffoldProject:
    tables = context.get("tables") if isinstance(context.get("tables"), dict) else {}
    domain = str(episode.get("domain") or "")
    if domain == "green_vrp_multiobjective":
        orders = [row for row in tables.get("orders", []) if _as_bool(row.get("active", True))]
        vehicles = [row for row in tables.get("vehicles", []) if _as_bool(row.get("available", True))]
        order_ids = [str(row.get("id")) for row in orders]
        vehicle_ids = [str(row.get("id")) for row in vehicles]
        segments = [
            SegmentSpec(name="vehicle_assign", kind="assignment", demands=order_ids, resources=vehicle_ids),
            SegmentSpec(name="route_order", kind="permutation", values=order_ids),
        ]
    elif domain == "cloud_scheduling_multiobjective":
        jobs = [row for row in tables.get("jobs", []) if _as_bool(row.get("active", True))]
        machines = [row for row in tables.get("machines", []) if _as_bool(row.get("available", True))]
        segments = [
            SegmentSpec(
                name="machine_assign",
                kind="assignment",
                demands=[str(row.get("id")) for row in jobs],
                resources=[str(row.get("id")) for row in machines],
            )
        ]
    else:
        nurses = [str(row.get("id")) for row in tables.get("nurses", [])]
        days = [str(row.get("id")) for row in tables.get("days", [])]
        shifts = [str(row.get("id")) for row in tables.get("shifts", [])]
        segments = [
            SegmentSpec(
                name="roster",
                kind="choice_vector",
                length=max(1, len(nurses) * len(days)),
                options=shifts + [None],
            )
        ]
    contract = public_optimization_contract_for_episode(episode) or {}
    problem_spec = {
        "solver_mode": "moea" if len(list(contract.get("objective_names") or [])) > 1 else "scalar_ga",
        "objective_names": list(contract.get("objective_names") or []),
        "objective_senses": list(contract.get("objective_senses") or []),
    }
    return ScaffoldProject(
        task_id=str(episode.get("episode_id") or "semantic-probe"),
        data={},
        segments=segments,
        setup_code="public semantic probe; no executable setup",
        fitness_code="public semantic probe; no executable fitness",
        evaluate=lambda genome, data: FitnessResult(scalar=0.0),
        problem_spec=problem_spec,
        trace=ScaffoldTrace(model="semantic-probe"),
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
