#!/usr/bin/env python3
"""Evaluate a deterministic public-structure restart rule without new search.

The rule measures how much of the active decision-entity table changed between
two materialized public states.  It selects Full when at least half of the
non-key fields of the currently active rows changed; otherwise it selects
Warm.  Decisions are frozen before audited Warm/Full shadow outcomes are
joined, so this script makes no provider or optimizer calls.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import (  # noqa: E402
    apply_public_context_patch,
    normalize_public_context_tables,
)
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import replay_data_patch  # noqa: E402
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (  # noqa: E402
    load_episodes,
    public_context_with_loaded_tables,
)


DEFAULT_EPISODES = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
DEFAULT_SOURCE = ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714"
DEFAULT_COMPARISON = (
    ROOT / "outputs/liveopt_ablation_semantic_shadow_comparison_p007_p015_200x200x10_v2_20260714"
)
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_structural_restart_p010_p015_20260716"
EPISODES = tuple(f"NLDO-P{index:03d}" for index in range(10, 16))
STAGES = tuple(range(1, 13))
SEEDS = tuple(range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replacement-threshold", type=float, default=0.50)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def float_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def active_table_replacement(
    previous_context: dict[str, Any],
    current_context: dict[str, Any],
) -> tuple[str, float, int, int, int]:
    """Return the largest changed-cell fraction over active entity tables."""

    candidates: list[tuple[str, float, int, int, int]] = []
    previous_tables = previous_context.get("tables") or {}
    current_tables = current_context.get("tables") or {}
    for table_name, current_rows in current_tables.items():
        if not isinstance(current_rows, list) or not any(
            isinstance(row, dict) and "active" in row for row in current_rows
        ):
            continue
        previous_rows = previous_tables.get(table_name) or []
        previous_by_id = {
            str(row["id"]): row
            for row in previous_rows
            if isinstance(row, dict) and row.get("id") is not None
        }
        active_rows = [
            row
            for row in current_rows
            if isinstance(row, dict) and row.get("active") is True
        ]
        changed_cells = 0
        compared_cells = 0
        for row in active_rows:
            fields = [field for field in row if field not in {"id", "active"}]
            compared_cells += len(fields)
            previous = previous_by_id.get(str(row.get("id")))
            if previous is None or previous.get("active") is not True:
                changed_cells += len(fields)
            else:
                changed_cells += sum(previous.get(field) != row.get(field) for field in fields)
        ratio = changed_cells / compared_cells if compared_cells else 0.0
        candidates.append((table_name, ratio, changed_cells, compared_cells, len(active_rows)))
    if not candidates:
        return "--", 0.0, 0, 0, 0
    return max(candidates, key=lambda item: item[1])


def source_rows(path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episode_id = str(row.get("episode_id") or "")
            if episode_id in EPISODES and int(row.get("run_seed", -1)) == 0:
                if episode_id in selected:
                    raise ValueError(f"duplicate seed-0 source row for {episode_id}")
                selected[episode_id] = row
    missing = sorted(set(EPISODES) - set(selected))
    if missing:
        raise ValueError(f"missing seed-0 source rows: {missing}")
    return selected


def build_decisions(
    episodes: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for episode_id in EPISODES:
        episode = episodes[episode_id]
        source = sources[episode_id]
        context = public_context_with_loaded_tables(episode)
        source_updates = list(source.get("update_results") or [])
        updates = list(episode.get("update_stream") or [])
        if len(source_updates) < len(updates):
            raise ValueError(f"{episode_id}: incomplete source update trajectory")
        for stage_index, (update, source_update) in enumerate(zip(updates, source_updates), start=1):
            previous_context = copy.deepcopy(context)
            patch = replay_data_patch(update, {}, source_update)
            if patch:
                context = normalize_public_context_tables(apply_public_context_patch(context, patch))
            table, ratio, changed, compared, active = active_table_replacement(previous_context, context)
            action = "Full" if ratio >= threshold else "Warm"
            restart = source_update.get("restart") or {}
            semantic_action = "Full" if restart.get("restart_skill") == "full_restart_v1" else "Warm"
            decisions.append(
                {
                    "episode_id": episode_id,
                    "stage_index": stage_index,
                    "update_id": str(update.get("update_id") or f"t{stage_index:02d}"),
                    "decision_table": table,
                    "active_rows": active,
                    "changed_cells": changed,
                    "compared_cells": compared,
                    "replacement_ratio": ratio,
                    "threshold": threshold,
                    "selected_action": action,
                    "semantic_action": semantic_action,
                    "agrees_with_semantic": action == semantic_action,
                }
            )
    return decisions


def load_metric(path: Path, field: str) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            if episode_id not in EPISODES:
                continue
            key = (episode_id, int(row["stage_index"]), int(row["run_seed"]))
            values[key] = float_value(row.get(field))
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    episodes = {
        str(episode["episode_id"]): episode
        for episode in load_episodes(args.episodes_jsonl)
        if str(episode.get("episode_id") or "") in EPISODES
    }
    source_path = args.source_run_dir / "NLDO/evo2_limit0.jsonl"
    decisions = build_decisions(
        episodes,
        source_rows(source_path),
        args.replacement_threshold,
    )
    action_by_stage = {
        (row["episode_id"], int(row["stage_index"])): str(row["selected_action"])
        for row in decisions
    }

    shadow: dict[str, dict[str, dict[tuple[str, int, int], float]]] = {}
    provenance: dict[str, str] = {}
    for method in ("fixed_warm", "fixed_full"):
        metric = args.comparison_dir / method / "reference_metrics/reference_stage_metrics.csv"
        runtime = args.comparison_dir / method / "reference_metrics/runtime_stage_metrics.csv"
        shadow[method] = {
            "hv": load_metric(metric, "normalized_hv"),
            "generations": load_metric(runtime, "executed_generations"),
        }
        provenance[f"{method}_metrics_sha256"] = sha256_file(metric)
        provenance[f"{method}_runtime_sha256"] = sha256_file(runtime)

    selected_hv: list[float] = []
    selected_generations: list[float] = []
    stage_regret: list[float] = []
    for episode_id in EPISODES:
        for stage_index in STAGES:
            action = action_by_stage[(episode_id, stage_index)]
            selected_method = "fixed_full" if action == "Full" else "fixed_warm"
            warm_values: list[float] = []
            full_values: list[float] = []
            for seed in SEEDS:
                key = (episode_id, stage_index, seed)
                warm_values.append(shadow["fixed_warm"]["hv"][key])
                full_values.append(shadow["fixed_full"]["hv"][key])
                selected_hv.append(shadow[selected_method]["hv"][key])
                selected_generations.append(shadow[selected_method]["generations"][key])
            selected_stage = mean(full_values) if action == "Full" else mean(warm_values)
            stage_regret.append(max(mean(warm_values), mean(full_values)) - selected_stage)

    errors: list[str] = []
    if len(decisions) != len(EPISODES) * len(STAGES):
        errors.append(f"decision coverage is {len(decisions)}, expected 72")
    if not all(bool(row["agrees_with_semantic"]) for row in decisions):
        errors.append("structural and semantic actions differ")
    full_actions = sum(row["selected_action"] == "Full" for row in decisions)
    if full_actions != 12:
        errors.append(f"Full-action count is {full_actions}, expected 12")

    ordinary = [float(row["replacement_ratio"]) for row in decisions if int(row["stage_index"]) <= 10]
    disruptive = [float(row["replacement_ratio"]) for row in decisions if int(row["stage_index"]) >= 11]
    payload = {
        "protocol": "public_active_row_replacement_restart_shadow_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "provider_or_api_calls": False,
        "optimizer_replay": False,
        "decision_frozen_before_held_out_join": True,
        "rule": (
            "Full iff the maximum changed non-key-cell fraction among currently active "
            f"decision-entity tables is at least {args.replacement_threshold:.2f}; otherwise Warm"
        ),
        "decision_units": len(decisions),
        "decision_cells": len(decisions) * len(SEEDS),
        "full_actions": full_actions,
        "full_cells": full_actions * len(SEEDS),
        "mean_hv": mean(selected_hv),
        "mean_generations": mean(selected_generations),
        "mean_stage_oracle_regret": mean(stage_regret),
        "semantic_agreement": mean([float(row["agrees_with_semantic"]) for row in decisions]),
        "ordinary_ratio_range": [min(ordinary), max(ordinary)],
        "disruptive_ratio_range": [min(disruptive), max(disruptive)],
        "sources": {
            "episodes_jsonl": str(args.episodes_jsonl),
            "episodes_sha256": sha256_file(args.episodes_jsonl),
            "source_jsonl": str(source_path),
            "source_sha256": sha256_file(source_path),
            **provenance,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "decisions.csv", decisions)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
