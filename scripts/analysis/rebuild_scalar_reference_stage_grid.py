#!/usr/bin/env python3
"""Rebuild selected scalar NLDO reference stages in parallel."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.reference_solvers.moea_reference import MOEAConfig
from scripts.generate_moea_references import enhance_episode, run_episode


SCALAR_DOMAIN = "inrc2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(60, os.cpu_count() or 1)))
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--stage-index", action="append", type=int, default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    selected_episodes = set(args.episode_id or [])
    selected_stages = set(args.stage_index or [])
    scalar_rows = [
        (index, episode)
        for index, episode in enumerate(rows)
        if episode.get("domain") == SCALAR_DOMAIN
        and (not selected_episodes or str(episode.get("episode_id")) in selected_episodes)
    ]
    if not scalar_rows:
        raise ValueError("no scalar NLDO episodes matched the requested selection")

    stage_dir = args.out_dir / "stage_references"
    seed_dir = args.out_dir / "stage_seed_references"
    stage_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    config = MOEAConfig(
        population_size=args.population,
        generations=args.generations,
        seeds=args.seeds,
        archive_limit=1,
        hv_samples=0,
    )
    tasks = []
    for source_index, episode in scalar_rows:
        for stage_index in range(1 + len(episode.get("update_stream") or [])):
            if selected_stages and stage_index not in selected_stages:
                continue
            for seed_index in range(args.seeds):
                path = seed_path(seed_dir, str(episode["episode_id"]), stage_index, seed_index)
                if path.exists() and not args.force:
                    continue
                tasks.append((source_index, episode, stage_index, seed_index, path, config))

    print(
        json.dumps(
            {
                "event": "start",
                "tasks": len(tasks),
                "workers": args.workers,
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_stage_seed, *task) for task in tasks]
            for future in as_completed(futures):
                print(json.dumps({"event": "stage_seed_written", **future.result()}, ensure_ascii=False), flush=True)

    aggregate_stage_seeds(scalar_rows, selected_stages, seed_dir, stage_dir, config)

    enhanced_rows = merge_rows(rows, scalar_rows, stage_dir, config, args.out_dir, selected_stages)
    enhanced_path = args.out_dir / enhanced_filename(args.input, config)
    write_jsonl(enhanced_path, enhanced_rows)
    summary = summarize(
        enhanced_rows,
        input_path=args.input,
        enhanced_path=enhanced_path,
        config=config,
        selected_episode_ids=sorted(str(episode["episode_id"]) for _, episode in scalar_rows),
        selected_stage_indices=sorted(selected_stages),
    )
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "enhanced": str(enhanced_path), "summary": str(summary_path)}, ensure_ascii=False), flush=True)


def run_stage_seed(
    source_index: int,
    episode: dict[str, Any],
    stage_index: int,
    seed_index: int,
    path: Path,
    config: MOEAConfig,
) -> dict[str, Any]:
    single_seed_config = replace(config, seeds=1)
    reference = run_episode(
        copy.deepcopy(episode),
        single_seed_config,
        seed_offset=source_index * 100000 + seed_index * 9173,
        selected_stage_indices={stage_index},
        state_independent_objectives=True,
    )
    trajectory = list(reference.get("reference_trajectory") or [])
    if len(trajectory) != 1:
        raise RuntimeError(f"expected one scalar reference stage for {episode['episode_id']} S{stage_index:02d}")
    step = trajectory[0]
    payload = {
        "episode_id": episode["episode_id"],
        "stage_index": stage_index,
        "seed_index": seed_index,
        "update_id": step.get("update_id"),
        "reference_policy": reference.get("reference_policy"),
        "reference_step": step,
        "solver_config": single_seed_config.__dict__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "episode_id": episode["episode_id"],
        "stage_index": stage_index,
        "seed_index": seed_index,
        "update_id": step.get("update_id"),
        "objective": step.get("objective"),
        "path": str(path),
    }


def aggregate_stage_seeds(
    scalar_rows: list[tuple[int, dict[str, Any]]],
    selected_stages: set[int],
    seed_dir: Path,
    stage_dir: Path,
    config: MOEAConfig,
) -> None:
    for _, episode in scalar_rows:
        episode_id = str(episode["episode_id"])
        for stage_index in range(1 + len(episode.get("update_stream") or [])):
            if selected_stages and stage_index not in selected_stages:
                continue
            seed_payloads = []
            for seed_index in range(config.seeds):
                path = seed_path(seed_dir, episode_id, stage_index, seed_index)
                if not path.exists():
                    raise FileNotFoundError(f"missing scalar stage-seed reference {path}")
                seed_payloads.append(json.loads(path.read_text(encoding="utf-8")))
            best_payload = min(
                seed_payloads,
                key=lambda item: float((item.get("reference_step") or {}).get("objective", float("inf"))),
            )
            best_step = copy.deepcopy(best_payload["reference_step"])
            best_step["solver_config"] = config.__dict__
            best_step["evaluations"] = sum(
                int((item.get("reference_step") or {}).get("evaluations") or 0)
                for item in seed_payloads
            )
            best_step["per_seed_objectives"] = [
                {
                    "seed_index": int(item["seed_index"]),
                    "objective": float(item["reference_step"]["objective"]),
                }
                for item in sorted(seed_payloads, key=lambda item: int(item["seed_index"]))
            ]
            best_step["winning_seed_index"] = int(best_payload["seed_index"])
            payload = {
                "episode_id": episode_id,
                "stage_index": stage_index,
                "update_id": best_step.get("update_id"),
                "reference_policy": "offline_multi_seed_ga_best_known_scalar_reference_current_state_only",
                "reference_step": best_step,
                "solver_config": config.__dict__,
            }
            path = stage_path(stage_dir, episode_id, stage_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "event": "stage_aggregated",
                        "episode_id": episode_id,
                        "stage_index": stage_index,
                        "update_id": best_step.get("update_id"),
                        "objective": best_step.get("objective"),
                        "winning_seed_index": best_step.get("winning_seed_index"),
                        "path": str(path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def merge_rows(
    rows: list[dict[str, Any]],
    scalar_rows: list[tuple[int, dict[str, Any]]],
    stage_dir: Path,
    config: MOEAConfig,
    out_dir: Path,
    selected_stages: set[int],
) -> list[dict[str, Any]]:
    selected_ids = {str(episode["episode_id"]) for _, episode in scalar_rows}
    merged = []
    for episode in rows:
        if str(episode.get("episode_id")) not in selected_ids:
            merged.append(copy.deepcopy(episode))
            continue
        original = list((episode.get("evaluation") or {}).get("reference_trajectory") or [])
        trajectory = []
        refreshed = []
        for stage_index in range(1 + len(episode.get("update_stream") or [])):
            path = stage_path(stage_dir, str(episode["episode_id"]), stage_index)
            if path.exists():
                rebuilt_step = json.loads(path.read_text(encoding="utf-8"))["reference_step"]
                original_step = original[stage_index] if stage_index < len(original) else {}
                trajectory.append(best_known_scalar_step(original_step, rebuilt_step, config))
                refreshed.append(stage_index)
            elif selected_stages and stage_index < len(original):
                trajectory.append(copy.deepcopy(original[stage_index]))
            else:
                raise FileNotFoundError(f"missing scalar reference stage {path}")
        reference = {
            "episode_id": episode["episode_id"],
            "benchmark": episode.get("benchmark"),
            "domain": episode.get("domain"),
            "reference_policy": "best_known_union_constructive_and_offline_scalar_ga_current_state_only",
            "reference_trajectory": trajectory,
            "partial_stage_refresh": bool(selected_stages),
            "selected_stage_ids": [trajectory[index].get("update_id") for index in refreshed],
            "selected_stage_indices": refreshed,
            "solver_config": config.__dict__,
        }
        enhanced = enhance_episode(episode, reference)
        enhanced.setdefault("evaluation", {})["reference_rebuild"] = {
            "stage_grid_dir": str(out_dir / "stage_references"),
            "protocol": "best objective from the current constructive basin reference and an independently rebuilt current-state-only scalar GA",
            "solver_config": config.__dict__,
        }
        merged.append(enhanced)
    return merged


def best_known_scalar_step(
    constructive_step: dict[str, Any],
    ga_step: dict[str, Any],
    config: MOEAConfig,
) -> dict[str, Any]:
    constructive_objective = finite_objective(constructive_step)
    ga_objective = finite_objective(ga_step)
    if constructive_objective is None and ga_objective is None:
        raise ValueError("neither constructive nor GA scalar reference has a finite objective")
    constructive_wins = ga_objective is None or (
        constructive_objective is not None and constructive_objective <= ga_objective
    )
    selected = copy.deepcopy(constructive_step if constructive_wins else ga_step)
    selected["reference_type"] = "best_known_constructive_or_scalar_ga"
    selected["solver"] = "best_of_current_constructive_and_offline_multi_seed_scalar_ga"
    selected["reference_candidate_objectives"] = {
        "constructive": constructive_objective,
        "scalar_ga": ga_objective,
    }
    selected["reference_winner"] = "constructive" if constructive_wins else "scalar_ga"
    selected["scalar_ga_solver_config"] = config.__dict__
    return selected


def finite_objective(step: dict[str, Any]) -> float | None:
    try:
        value = float(step.get("objective"))
    except (TypeError, ValueError):
        return None
    return value if value == value and value not in {float("inf"), float("-inf")} else None


def summarize(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    enhanced_path: Path,
    config: MOEAConfig,
    selected_episode_ids: list[str],
    selected_stage_indices: list[int],
) -> dict[str, Any]:
    stages = []
    selected_ids = set(selected_episode_ids)
    for episode in rows:
        if str(episode.get("episode_id")) not in selected_ids:
            continue
        for stage_index, step in enumerate((episode.get("evaluation") or {}).get("reference_trajectory") or []):
            if selected_stage_indices and stage_index not in selected_stage_indices:
                continue
            stages.append(
                {
                    "episode_id": episode.get("episode_id"),
                    "stage_index": stage_index,
                    "update_id": step.get("update_id"),
                    "objective": step.get("objective"),
                }
            )
    return {
        "input": str(input_path),
        "enhanced_output": str(enhanced_path),
        "config": config.__dict__,
        "selected_episode_ids": selected_episode_ids,
        "selected_stage_indices": selected_stage_indices,
        "scalar_stage_count": len(stages),
        "stages": stages,
    }


def enhanced_filename(input_path: Path, config: MOEAConfig) -> str:
    return f"{input_path.stem}.strong_scalar_{config.population_size}x{config.generations}x{config.seeds}.jsonl"


def stage_path(stage_dir: Path, episode_id: str, stage_index: int) -> Path:
    return stage_dir / episode_id / f"S{stage_index:02d}.json"


def seed_path(seed_dir: Path, episode_id: str, stage_index: int, seed_index: int) -> Path:
    return seed_dir / episode_id / f"S{stage_index:02d}" / f"seed_{seed_index:02d}.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
