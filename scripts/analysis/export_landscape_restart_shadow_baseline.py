#!/usr/bin/env python3
"""Evaluate a sensor-based objective-landscape restart gate on matched shadows.

The selector sees only the incoming LiveOpt population and the public old/new
Workbench.  Its chosen Warm or Full outcome is read from the already-audited
same-population shadow controls, so this analysis makes no LLM/API calls and
does not launch an optimizer.  Held-out HV is joined only after every decision
has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2.agents.liveopt_dynamic_impl import (  # noqa: E402
    apply_public_context_patch,
    normalize_public_context_tables,
    select_restart_skill_from_landscape_change,
)
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import (  # noqa: E402
    compile_stage_project,
    replay_data_patch,
    source_history_previous_result,
)
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import (  # noqa: E402
    load_episodes,
    public_context_with_loaded_tables,
)


DEFAULT_EPISODES = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
DEFAULT_SOURCE = ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714"
DEFAULT_COMPARISON = ROOT / "outputs/liveopt_ablation_semantic_shadow_comparison_p007_p015_200x200x10_v2_20260714"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716"
DEFAULT_PAPER_TABLE = ROOT / "release_artifacts/paper_table_exports/liveopt_landscape_restart_rows.tex"
EPISODES = tuple(f"NLDO-P{index:03d}" for index in range(10, 16))
STAGES = tuple(range(1, 13))
SEEDS = tuple(range(10))
EPISODE_PATTERN = re.compile(rb'"episode_id"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-jsonl", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_PAPER_TABLE)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--sensor-fraction", type=float, default=0.10)
    parser.add_argument("--displacement-threshold", type=float, default=0.50)
    parser.add_argument("--feasibility-loss-threshold", type=float, default=0.50)
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


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def float_value(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def source_jsonl(run_dir: Path) -> Path:
    direct = run_dir / "NLDO" / "evo2_limit0.jsonl"
    if not direct.exists():
        raise FileNotFoundError(direct)
    return direct


def build_line_index(path: Path, wanted: set[str]) -> dict[str, list[tuple[int, int]]]:
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            match = EPISODE_PATTERN.search(line[:4096])
            if not match:
                raise ValueError(f"cannot identify episode at byte {offset} in {path}")
            episode_id = match.group(1).decode("utf-8")
            if episode_id in wanted:
                index[episode_id].append((offset, len(line)))
    return dict(index)


def read_source_row(handle, offset: int, length: int) -> dict[str, Any]:
    handle.seek(offset)
    payload = handle.read(length)
    return json.loads(payload)


def stage_projects(episode: dict[str, Any], source_row: dict[str, Any]) -> list[tuple[Any, Any, str]]:
    episode_id = str(episode["episode_id"])
    public_context = public_context_with_loaded_tables(episode)
    project = compile_stage_project(
        episode_id,
        public_context,
        str(source_row["setup_code"]),
        str(source_row["fitness_code"]),
    )
    pairs: list[tuple[Any, Any, str]] = []
    updates = list(episode.get("update_stream") or [])
    source_updates = list(source_row.get("update_results") or [])
    if len(source_updates) < len(updates):
        raise ValueError(f"{episode_id}: source has {len(source_updates)} updates, expected {len(updates)}")
    for stage_index, update in enumerate(updates, start=1):
        previous_project = project
        source_update = source_updates[stage_index - 1]
        data_patch = replay_data_patch(update, {}, source_update)
        if data_patch:
            public_context = normalize_public_context_tables(
                apply_public_context_patch(public_context, data_patch)
            )
        project = compile_stage_project(
            episode_id,
            public_context,
            str(source_update.get("setup_code") or source_row["setup_code"]),
            str(source_update.get("fitness_code") or source_row["fitness_code"]),
        )
        update_id = str(update.get("update_id") or source_update.get("update_id") or f"t{stage_index:02d}")
        pairs.append((previous_project, project, update_id))
    return pairs


def load_metric_csv(path: Path, value_field: str) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            if episode_id not in EPISODES:
                continue
            key = (episode_id, int(row["stage_index"]), int(row["run_seed"]))
            values[key] = float_value(row.get(value_field))
    return values


def load_runtime_csv(path: Path) -> dict[tuple[str, int, int], float]:
    return load_metric_csv(path, "executed_generations")


def load_shadow_metrics(root: Path) -> dict[str, dict[str, dict[tuple[str, int, int], float]]]:
    payload: dict[str, dict[str, dict[tuple[str, int, int], float]]] = {}
    for method in ("liveopt", "fixed_warm", "fixed_full"):
        metric_dir = root / method / "reference_metrics"
        payload[method] = {
            "hv": load_metric_csv(metric_dir / "reference_stage_metrics.csv", "normalized_hv"),
            "igd": load_metric_csv(metric_dir / "reference_stage_metrics.csv", "igd"),
            "generations": load_runtime_csv(metric_dir / "runtime_stage_metrics.csv"),
        }
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def build_decisions(
    *,
    episodes: dict[str, dict[str, Any]],
    source_path: Path,
    line_index: dict[str, list[tuple[int, int]]],
    selected_episodes: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with source_path.open("rb") as handle:
        for episode_id in selected_episodes:
            locations = line_index.get(episode_id) or []
            if len(locations) != len(SEEDS):
                raise ValueError(f"{episode_id}: found {len(locations)} source rows, expected {len(SEEDS)}")
            first = read_source_row(handle, *locations[0])
            projects = stage_projects(episodes[episode_id], first)
            seen_seeds: set[int] = set()
            for offset, length in locations:
                source_row = read_source_row(handle, offset, length)
                seed = int(source_row.get("run_seed", -1))
                seen_seeds.add(seed)
                for stage_index, (previous_project, project, update_id) in enumerate(projects, start=1):
                    previous_result, source_policy, source_update_id = source_history_previous_result(
                        source_row,
                        stage_index,
                    )
                    decision = select_restart_skill_from_landscape_change(
                        previous_project=previous_project,
                        project=project,
                        previous_result=previous_result,
                        seed=seed,
                        stage_index=stage_index,
                        sensor_fraction=args.sensor_fraction,
                        displacement_threshold=args.displacement_threshold,
                        feasibility_loss_threshold=args.feasibility_loss_threshold,
                    )
                    record = decision.to_record()
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "stage_index": stage_index,
                            "update_id": update_id,
                            "run_seed": seed,
                            "selected_action": "Full"
                            if decision.selected_restart_skill == "full_restart_v1"
                            else "Warm",
                            "selected_restart_skill": decision.selected_restart_skill,
                            "population_count": record["population_count"],
                            "sensor_count": record["sensor_count"],
                            "evaluated_sensor_count": record["evaluated_sensor_count"],
                            "objective_count": record["objective_count"],
                            "normalized_landscape_displacement": record["normalized_landscape_displacement"],
                            "mean_sensor_displacement": record["mean_sensor_displacement"],
                            "per_objective_displacement": json.dumps(record["per_objective_displacement"]),
                            "old_feasible_count": record["old_feasible_count"],
                            "new_feasible_count": record["new_feasible_count"],
                            "feasibility_flip_rate": record["feasibility_flip_rate"],
                            "feasibility_loss_rate": record["feasibility_loss_rate"],
                            "objective_dimension_changed": record["objective_dimension_changed"],
                            "source_population_kind": previous_result.metadata.get("source_population_kind"),
                            "source_population_count": previous_result.metadata.get("source_population_count"),
                            "source_policy": source_policy,
                            "source_update_id": source_update_id,
                            "policy_version": record["policy_version"],
                            "reason": record["reason"],
                        }
                    )
            if seen_seeds != set(SEEDS):
                raise ValueError(f"{episode_id}: source seeds are {sorted(seen_seeds)}, expected {list(SEEDS)}")
    return rows


def join_outcomes(
    decisions: list[dict[str, Any]],
    metrics: dict[str, dict[str, dict[tuple[str, int, int], float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        key = (decision["episode_id"], int(decision["stage_index"]), int(decision["run_seed"]))
        action_key = "fixed_full" if decision["selected_action"] == "Full" else "fixed_warm"
        warm_hv = metrics["fixed_warm"]["hv"][key]
        full_hv = metrics["fixed_full"]["hv"][key]
        selected_hv = metrics[action_key]["hv"][key]
        rows.append(
            {
                **decision,
                "warm_hv": warm_hv,
                "full_hv": full_hv,
                "liveopt_hv": metrics["liveopt"]["hv"][key],
                "selected_hv": selected_hv,
                "oracle_cell_hv": max(warm_hv, full_hv),
                "selected_cell_regret": max(warm_hv, full_hv) - selected_hv,
                "warm_igd": metrics["fixed_warm"]["igd"][key],
                "full_igd": metrics["fixed_full"]["igd"][key],
                "selected_igd": metrics[action_key]["igd"][key],
                "selected_generations": metrics[action_key]["generations"][key],
                "liveopt_generations": metrics["liveopt"]["generations"][key],
                "warm_generations": metrics["fixed_warm"]["generations"][key],
                "full_generations": metrics["fixed_full"]["generations"][key],
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["episode_id"], int(row["stage_index"]))].append(row)
    for (episode_id, stage_index), items in sorted(groups.items()):
        warm_hv = mean([float(row["warm_hv"]) for row in items])
        full_hv = mean([float(row["full_hv"]) for row in items])
        stage_rows.append(
            {
                "episode_id": episode_id,
                "stage_index": stage_index,
                "full_seed_count": sum(row["selected_action"] == "Full" for row in items),
                "mean_displacement": mean([float(row["normalized_landscape_displacement"]) for row in items]),
                "std_displacement": std([float(row["normalized_landscape_displacement"]) for row in items]),
                "mean_feasibility_loss": mean([float(row["feasibility_loss_rate"]) for row in items]),
                "landscape_hv": mean([float(row["selected_hv"]) for row in items]),
                "liveopt_hv": mean([float(row["liveopt_hv"]) for row in items]),
                "warm_hv": warm_hv,
                "full_hv": full_hv,
                "oracle_stage_action": "Full" if full_hv > warm_hv else "Warm",
                "oracle_stage_hv": max(warm_hv, full_hv),
            }
        )

    method_rows: list[dict[str, Any]] = []
    for scope, selected in (
        ("all", rows),
        ("t01-t10", [row for row in rows if int(row["stage_index"]) <= 10]),
        ("t11-t12", [row for row in rows if int(row["stage_index"]) >= 11]),
    ):
        for method, field, generation_method in (
            ("LiveOpt", "liveopt_hv", "liveopt"),
            ("Sensor landscape", "selected_hv", "selected"),
            ("Always Warm", "warm_hv", "fixed_warm"),
            ("Always Full", "full_hv", "fixed_full"),
        ):
            if generation_method == "selected":
                generation_values = [float(row["selected_generations"]) for row in selected]
            elif generation_method == "liveopt":
                generation_values = [float(row["liveopt_generations"]) for row in selected]
            elif generation_method == "fixed_warm":
                generation_values = [float(row["warm_generations"]) for row in selected]
            elif generation_method == "fixed_full":
                generation_values = [float(row["full_generations"]) for row in selected]
            else:
                generation_values = []
            method_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "cells": len(selected),
                    "full_cells": (
                        sum(row["selected_action"] == "Full" for row in selected)
                        if method == "Sensor landscape"
                        else len(selected)
                        if method == "Always Full"
                        else ""
                    ),
                    "mean_hv": mean([float(row[field]) for row in selected]),
                    "mean_igd": (
                        mean([float(row["selected_igd"]) for row in selected])
                        if method == "Sensor landscape"
                        else ""
                    ),
                    "mean_generations": mean(generation_values) if generation_values else "",
                    "mean_cell_regret": (
                        mean([float(row["selected_cell_regret"]) for row in selected])
                        if method == "Sensor landscape"
                        else ""
                    ),
                }
            )
    return stage_rows, method_rows


def threshold_sensitivity(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        chosen_hv: list[float] = []
        full_count = 0
        for row in rows:
            full = int(row["evaluated_sensor_count"]) == 0 or bool(row["objective_dimension_changed"]) or (
                float(row["normalized_landscape_displacement"]) >= threshold
            ) or (
                int(row["old_feasible_count"]) > 0
                and float(row["feasibility_loss_rate"]) >= 0.50
            )
            full_count += int(full)
            chosen_hv.append(float(row["full_hv"] if full else row["warm_hv"]))
        output.append(
            {
                "displacement_threshold": threshold,
                "full_cells": full_count,
                "cells": len(rows),
                "full_rate": full_count / max(1, len(rows)),
                "mean_hv": mean(chosen_hv),
            }
        )
    return output


def write_paper_table(path: Path, rows: list[dict[str, Any]]) -> None:
    episode_lines: list[str] = []
    for episode_id in EPISODES:
        items = [row for row in rows if row["episode_id"] == episode_id]
        early = [row for row in items if int(row["stage_index"]) <= 10]
        late = [row for row in items if int(row["stage_index"]) >= 11]
        episode_lines.append(
            f"P{int(episode_id[-3:]):03d} & "
            f"{sum(row['selected_action'] == 'Full' for row in early)}/{len(early)} & "
            f"{sum(row['selected_action'] == 'Full' for row in late)}/{len(late)} & "
            f"{mean([float(row['selected_hv']) for row in items]):.3f} & "
            f"{mean([float(row['liveopt_hv']) for row in items]):.3f} & "
            f"{mean([float(row['warm_hv']) for row in items]):.3f} \\\\"
        )
    early = [row for row in rows if int(row["stage_index"]) <= 10]
    late = [row for row in rows if int(row["stage_index"]) >= 11]
    episode_lines.append(
        "All & "
        f"{sum(row['selected_action'] == 'Full' for row in early)}/{len(early)} & "
        f"{sum(row['selected_action'] == 'Full' for row in late)}/{len(late)} & "
        f"{mean([float(row['selected_hv']) for row in rows]):.3f} & "
        f"{mean([float(row['liveopt_hv']) for row in rows]):.3f} & "
        f"{mean([float(row['warm_hv']) for row in rows]):.3f} \\\\"
    )
    text = (
        "% Generated by scripts/analysis/export_landscape_restart_shadow_baseline.py.\n"
        "\\newcommand{\\LiveOptLandscapeRestartRows}{%\n"
        + "\n".join(episode_lines)
        + "\n}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    selected_episodes = list(args.episode_id or EPISODES)
    episode_payload = {
        str(row["episode_id"]): row
        for row in load_episodes(args.episodes_jsonl)
        if str(row.get("episode_id")) in selected_episodes
    }
    if set(episode_payload) != set(selected_episodes):
        raise ValueError("requested episodes are missing from the benchmark payload")
    source_path = source_jsonl(args.source_run_dir)
    line_index = build_line_index(source_path, set(selected_episodes))
    decisions = build_decisions(
        episodes=episode_payload,
        source_path=source_path,
        line_index=line_index,
        selected_episodes=selected_episodes,
        args=args,
    )
    metrics = load_shadow_metrics(args.comparison_dir)
    outcome_rows = join_outcomes(decisions, metrics)
    stage_rows, method_rows = summarize(outcome_rows)
    sensitivity_rows = threshold_sensitivity(outcome_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "landscape_decisions.csv", outcome_rows)
    write_csv(args.out_dir / "stage_summary.csv", stage_rows)
    write_csv(args.out_dir / "method_summary.csv", method_rows)
    write_csv(args.out_dir / "threshold_sensitivity.csv", sensitivity_rows)
    write_paper_table(args.paper_table, outcome_rows)
    errors: list[str] = []
    expected = len(selected_episodes) * len(STAGES) * len(SEEDS)
    if len(outcome_rows) != expected:
        errors.append(f"decision coverage {len(outcome_rows)} != {expected}")
    if {int(row["population_count"]) for row in outcome_rows} != {200}:
        errors.append("not every decision used a 200-member incoming population")
    if {int(row["sensor_count"]) for row in outcome_rows} != {20}:
        errors.append("not every decision used 20 sensors")
    if any(str(row["source_population_kind"]) != "final_population" for row in outcome_rows):
        errors.append("at least one decision did not use the exact saved final population")
    audit = {
        "status": "passed" if not errors else "failed",
        "protocol": "sensor_objective_landscape_matched_shadow_v1",
        "episodes": selected_episodes,
        "cells": len(outcome_rows),
        "sensor_fraction": args.sensor_fraction,
        "displacement_threshold": args.displacement_threshold,
        "feasibility_loss_threshold": args.feasibility_loss_threshold,
        "selection_inputs": "old/new public Workbench plus exact incoming population only",
        "held_out_metrics_joined_after_decision": True,
        "optimizer_replay": False,
        "provider_or_api_calls": False,
        "episodes_jsonl": str(args.episodes_jsonl),
        "episodes_jsonl_sha256": sha256_file(args.episodes_jsonl),
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": sha256_file(source_path),
        "comparison_protocol_audit": str(args.comparison_dir / "protocol_audit.json"),
        "comparison_protocol_audit_sha256": sha256_file(args.comparison_dir / "protocol_audit.json"),
        "paper_table": str(args.paper_table),
        "errors": errors,
        "method_summary": method_rows,
        "threshold_sensitivity": sensitivity_rows,
    }
    (args.out_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
