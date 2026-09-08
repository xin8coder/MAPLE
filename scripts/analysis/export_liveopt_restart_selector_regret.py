#!/usr/bin/env python3
"""Export offline restart-action regret and cost diagnostics.

The three inputs are the audited same-incoming-population action shadows.  The
script treats an episode-stage as the selector's decision unit, averages the
ten numerical seeds within that unit, and computes a post-hoc Warm/Full oracle.
No selector or optimizer call is made here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = ROOT / "outputs/liveopt_ablation_semantic_shadow_comparison_p007_p015_200x200x10_v2_20260714"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_restart_selector_regret_20260715"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_restart_selector_regret_rows.tex"
DEFAULT_LANDSCAPE = ROOT / "logs/analysis/liveopt_landscape_restart_shadow_p010_p015_20260716/landscape_decisions.csv"
DEFAULT_STRUCTURAL = ROOT / "logs/analysis/liveopt_structural_restart_p010_p015_20260716/decisions.csv"
METHOD_DIRS = {"LiveOpt": "liveopt", "Always Warm": "fixed_warm", "Always Full": "fixed_full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--landscape-decisions", type=Path, default=DEFAULT_LANDSCAPE)
    parser.add_argument("--structural-decisions", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--bootstrap-repeats", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_quality(path: Path) -> dict[tuple[str, int, int], float]:
    out: dict[tuple[str, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = row["episode_id"]
            quality = row.get("normalized_hv") if episode_id >= "NLDO-P010" else row.get("normalized_score")
            out[(episode_id, int(row["stage_index"]), int(row["run_seed"]))] = float_value(quality)
    return out


def load_runtime(path: Path) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, int], str]]:
    generations: dict[tuple[str, int, int], float] = {}
    actions: dict[tuple[str, int], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = row["episode_id"]
            stage_index = int(row["stage_index"])
            seed = int(row["run_seed"])
            generations[(episode_id, stage_index, seed)] = float_value(row.get("executed_generations"))
            action = "Full" if row.get("restart_skill") == "full_restart_v1" else "Warm"
            old = actions.setdefault((episode_id, stage_index), action)
            if old != action:
                raise ValueError(f"selector action changes across seeds for {episode_id}:t{stage_index:02d}")
    return generations, actions


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def episode_bootstrap_ci(deltas: list[float], repeats: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [mean([rng.choice(deltas) for _ in deltas]) for _ in range(repeats)]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def exact_sign_pvalue(deltas: list[float]) -> float:
    nonzero = [value for value in deltas if abs(value) > 1e-12]
    n = len(nonzero)
    if not n:
        return 1.0
    positive = sum(value > 0 for value in nonzero)
    extreme = max(positive, n - positive)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def load_landscape_actions(path: Path) -> dict[tuple[str, int, int], str]:
    actions: dict[tuple[str, int, int], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            if not ("NLDO-P010" <= episode_id <= "NLDO-P015"):
                continue
            action = str(row.get("selected_action") or "")
            if action not in {"Warm", "Full"}:
                raise ValueError(f"invalid landscape action in {path}: {action!r}")
            key = (episode_id, int(row["stage_index"]), int(row["run_seed"]))
            actions[key] = action
    return actions


def load_structural_actions(path: Path) -> dict[tuple[str, int], str]:
    actions: dict[tuple[str, int], str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            if not ("NLDO-P010" <= episode_id <= "NLDO-P015"):
                continue
            action = str(row.get("selected_action") or "")
            if action not in {"Warm", "Full"}:
                raise ValueError(f"invalid structural action in {path}: {action!r}")
            key = (episode_id, int(row["stage_index"]))
            if key in actions and actions[key] != action:
                raise ValueError(f"conflicting structural action for {key}")
            actions[key] = action
    return actions


def add_structural_method(
    summary: dict[str, Any],
    stages: list[dict[str, Any]],
    quality: dict[str, dict[tuple[str, int, int], float]],
    generations: dict[str, dict[tuple[str, int, int], float]],
    actions: dict[tuple[str, int], str],
) -> None:
    selected_quality: list[float] = []
    selected_generations: list[float] = []
    stage_regret: list[float] = []
    full_actions = 0
    for stage in stages:
        episode_id = str(stage["episode_id"])
        stage_index = int(stage["stage_index"])
        action = actions[(episode_id, stage_index)]
        method = "Always Full" if action == "Full" else "Always Warm"
        keys = [(episode_id, stage_index, seed) for seed in range(10)]
        stage_quality = mean([quality[method][key] for key in keys])
        stage_generation = mean([generations[method][key] for key in keys])
        stage["structural_action"] = action
        stage["structural_quality"] = stage_quality
        stage["structural_generations"] = stage_generation
        stage["structural_regret"] = stage["oracle_quality"] - stage_quality
        selected_quality.extend(quality[method][key] for key in keys)
        selected_generations.extend(generations[method][key] for key in keys)
        stage_regret.append(stage["structural_regret"])
        full_actions += int(action == "Full")
    summary["methods"].append(
        {
            "view": summary["view"],
            "method": "Structural churn",
            "full_actions": full_actions,
            "full_cells": full_actions * 10,
            "decision_units": len(stages),
            "decision_cells": len(stages) * 10,
            "quality": mean(selected_quality),
            "mean_generations": mean(selected_generations),
            "regret": mean(stage_regret),
        }
    )


def add_landscape_method(
    summary: dict[str, Any],
    stages: list[dict[str, Any]],
    quality: dict[str, dict[tuple[str, int, int], float]],
    generations: dict[str, dict[tuple[str, int, int], float]],
    actions: dict[tuple[str, int, int], str],
) -> None:
    selected_quality: list[float] = []
    selected_generations: list[float] = []
    stage_regret: list[float] = []
    full_cells = 0
    for stage in stages:
        episode_id = str(stage["episode_id"])
        stage_index = int(stage["stage_index"])
        stage_values: list[float] = []
        stage_generations: list[float] = []
        stage_full_cells = 0
        for seed in range(10):
            key = (episode_id, stage_index, seed)
            action = actions[key]
            method = "Always Full" if action == "Full" else "Always Warm"
            stage_values.append(quality[method][key])
            stage_generations.append(generations[method][key])
            stage_full_cells += int(action == "Full")
        stage_quality = mean(stage_values)
        stage_generation = mean(stage_generations)
        stage["landscape_full_cells"] = stage_full_cells
        stage["landscape_quality"] = stage_quality
        stage["landscape_generations"] = stage_generation
        stage["landscape_regret"] = stage["oracle_quality"] - stage_quality
        selected_quality.extend(stage_values)
        selected_generations.extend(stage_generations)
        stage_regret.append(stage["landscape_regret"])
        full_cells += stage_full_cells
    summary["methods"].append(
        {
            "view": summary["view"],
            "method": "Sensor landscape",
            "full_actions": full_cells / 10.0,
            "full_cells": full_cells,
            "decision_units": len(stages),
            "decision_cells": len(stages) * 10,
            "quality": mean(selected_quality),
            "mean_generations": mean(selected_generations),
            "regret": mean(stage_regret),
        },
    )


def build_view(
    view: str,
    episodes: list[str],
    quality: dict[str, dict[tuple[str, int, int], float]],
    generations: dict[str, dict[tuple[str, int, int], float]],
    live_actions: dict[tuple[str, int], str],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    for episode_id in episodes:
        for stage_index in range(1, 13):
            keys = [(episode_id, stage_index, seed) for seed in range(10)]
            warm_quality = mean([quality["Always Warm"][key] for key in keys])
            full_quality = mean([quality["Always Full"][key] for key in keys])
            live_quality = mean([quality["LiveOpt"][key] for key in keys])
            warm_generations = mean([generations["Always Warm"][key] for key in keys])
            full_generations = mean([generations["Always Full"][key] for key in keys])
            live_generations = mean([generations["LiveOpt"][key] for key in keys])
            oracle_action = "Full" if full_quality > warm_quality else "Warm"
            oracle_quality = max(warm_quality, full_quality)
            oracle_generations = full_generations if oracle_action == "Full" else warm_generations
            stages.append(
                {
                    "view": view,
                    "episode_id": episode_id,
                    "stage_index": stage_index,
                    "liveopt_action": live_actions[(episode_id, stage_index)],
                    "oracle_action": oracle_action,
                    "liveopt_quality": live_quality,
                    "warm_quality": warm_quality,
                    "full_quality": full_quality,
                    "oracle_quality": oracle_quality,
                    "liveopt_regret": oracle_quality - live_quality,
                    "warm_regret": oracle_quality - warm_quality,
                    "full_regret": oracle_quality - full_quality,
                    "liveopt_generations": live_generations,
                    "warm_generations": warm_generations,
                    "full_generations": full_generations,
                    "oracle_generations": oracle_generations,
                }
            )

    full_rate = sum(row["liveopt_action"] == "Full" for row in stages) / len(stages)
    random_quality = mean(
        [full_rate * row["full_quality"] + (1.0 - full_rate) * row["warm_quality"] for row in stages]
    )
    random_generations = mean(
        [full_rate * row["full_generations"] + (1.0 - full_rate) * row["warm_generations"] for row in stages]
    )
    oracle_quality = mean([row["oracle_quality"] for row in stages])
    method_rows = [
        {
            "view": view,
            "method": "LiveOpt",
            "full_actions": sum(row["liveopt_action"] == "Full" for row in stages),
            "decision_units": len(stages),
            "quality": mean([row["liveopt_quality"] for row in stages]),
            "mean_generations": mean([row["liveopt_generations"] for row in stages]),
            "regret": mean([row["liveopt_regret"] for row in stages]),
        },
        {
            "view": view,
            "method": "Always Warm",
            "full_actions": 0,
            "decision_units": len(stages),
            "quality": mean([row["warm_quality"] for row in stages]),
            "mean_generations": mean([row["warm_generations"] for row in stages]),
            "regret": mean([row["warm_regret"] for row in stages]),
        },
        {
            "view": view,
            "method": "Always Full",
            "full_actions": len(stages),
            "decision_units": len(stages),
            "quality": mean([row["full_quality"] for row in stages]),
            "mean_generations": mean([row["full_generations"] for row in stages]),
            "regret": mean([row["full_regret"] for row in stages]),
        },
        {
            "view": view,
            "method": "Same-rate random expectation",
            "full_actions": full_rate * len(stages),
            "decision_units": len(stages),
            "quality": random_quality,
            "mean_generations": random_generations,
            "regret": oracle_quality - random_quality,
        },
        {
            "view": view,
            "method": "Post-hoc stage oracle",
            "full_actions": sum(row["oracle_action"] == "Full" for row in stages),
            "decision_units": len(stages),
            "quality": oracle_quality,
            "mean_generations": mean([row["oracle_generations"] for row in stages]),
            "regret": 0.0,
        },
    ]

    episode_rows: list[dict[str, Any]] = []
    for episode_id in episodes:
        items = [row for row in stages if row["episode_id"] == episode_id]
        episode_rows.append(
            {
                "view": view,
                "episode_id": episode_id,
                "liveopt_quality": mean([row["liveopt_quality"] for row in items]),
                "warm_quality": mean([row["warm_quality"] for row in items]),
                "full_quality": mean([row["full_quality"] for row in items]),
                "liveopt_minus_warm": mean([row["liveopt_quality"] - row["warm_quality"] for row in items]),
                "liveopt_minus_full": mean([row["liveopt_quality"] - row["full_quality"] for row in items]),
            }
        )
    episode_deltas = [row["liveopt_minus_warm"] for row in episode_rows]
    ci_low, ci_high = episode_bootstrap_ci(episode_deltas, bootstrap_repeats, bootstrap_seed)
    live_row = next(row for row in method_rows if row["method"] == "LiveOpt")
    warm_row = next(row for row in method_rows if row["method"] == "Always Warm")
    oracle_row = next(row for row in method_rows if row["method"] == "Post-hoc stage oracle")
    available_gain = oracle_row["quality"] - warm_row["quality"]
    recovered_gain = live_row["quality"] - warm_row["quality"]
    summary = {
        "view": view,
        "decision_units": len(stages),
        "liveopt_full_actions": live_row["full_actions"],
        "oracle_full_actions": oracle_row["full_actions"],
        "liveopt_action_accuracy_against_mean_stage_oracle": mean(
            [float(row["liveopt_action"] == row["oracle_action"]) for row in stages]
        ),
        "oracle_gain_recovered_over_always_warm": recovered_gain / available_gain if available_gain > 0 else None,
        "episode_cluster_bootstrap_liveopt_minus_warm_ci95": [ci_low, ci_high],
        "episode_exact_sign_test_liveopt_minus_warm_two_sided_p": exact_sign_pvalue(episode_deltas),
        "methods": method_rows,
    }
    return summary, stages, episode_rows


def tex_value(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_tex(path: Path, views: dict[str, dict[str, Any]]) -> None:
    def rows_for(view: str) -> str:
        methods = {row["method"]: row for row in views[view]["methods"]}
        order = ["LiveOpt", "Always Warm", "Always Full", "Post-hoc stage oracle"]
        if view == "DM" and "Structural churn" in methods:
            order.insert(1, "Structural churn")
        if view == "DM" and "Sensor landscape" in methods:
            order.insert(2, "Sensor landscape")
        labels = {
            "LiveOpt": "\\method{}",
            "Structural churn": "Table-change rule",
            "Sensor landscape": "Numerical selector",
            "Always Warm": "Always Warm",
            "Always Full": "Always Full",
            "Post-hoc stage oracle": "Best action after results",
        }
        lines = []
        for name in order:
            row = methods[name]
            prefix = "\\rowcolor{LiveOptRowAccent}\n" if name == "LiveOpt" else ""
            if view == "DM":
                numerator = row.get("full_cells", row["full_actions"] * 10)
                denominator = row.get("decision_cells", row["decision_units"] * 10)
            else:
                numerator = row["full_actions"]
                denominator = row["decision_units"]
            lines.append(
                f"{prefix}{labels[name]} & {numerator:g}/{denominator:g} & "
                f"{tex_value(row['quality'])} & {tex_value(row['mean_generations'], 1)} & {tex_value(row['regret'])} \\\\"
            )
        return "\n".join(lines)

    text = (
        "% Generated by scripts/analysis/export_liveopt_restart_selector_regret.py.\n"
        "\\newcommand{\\LiveOptRestartRegretDMRows}{%\n"
        + rows_for("DM")
        + "\n}\n"
        "\\newcommand{\\LiveOptRestartRegretDSRows}{%\n"
        + rows_for("DS")
        + "\n}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    quality: dict[str, dict[tuple[str, int, int], float]] = {}
    generations: dict[str, dict[tuple[str, int, int], float]] = {}
    actions: dict[tuple[str, int], str] = {}
    provenance: dict[str, str] = {}
    for method, directory in METHOD_DIRS.items():
        metric_path = args.comparison_dir / directory / "reference_metrics/reference_stage_metrics.csv"
        runtime_path = args.comparison_dir / directory / "reference_metrics/runtime_stage_metrics.csv"
        if not metric_path.exists() or not runtime_path.exists():
            raise FileNotFoundError(f"missing comparison rows for {method}")
        quality[method] = load_quality(metric_path)
        generations[method], method_actions = load_runtime(runtime_path)
        if method == "LiveOpt":
            actions = method_actions
        provenance[f"{directory}_metrics_sha256"] = sha256_file(metric_path)
        provenance[f"{directory}_runtime_sha256"] = sha256_file(runtime_path)

    view_specs = {
        "DS": [f"NLDO-P{index:03d}" for index in range(7, 10)],
        "DM": [f"NLDO-P{index:03d}" for index in range(10, 16)],
    }
    summaries: dict[str, dict[str, Any]] = {}
    stage_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for offset, (view, episodes) in enumerate(view_specs.items()):
        summary, stages, episode = build_view(
            view,
            episodes,
            quality,
            generations,
            actions,
            args.bootstrap_repeats,
            args.bootstrap_seed + offset,
        )
        summaries[view] = summary
        stage_rows.extend(stages)
        episode_rows.extend(episode)

    if not args.landscape_decisions.exists():
        raise FileNotFoundError(args.landscape_decisions)
    landscape_actions = load_landscape_actions(args.landscape_decisions)
    if len(landscape_actions) != 720:
        raise ValueError(f"landscape decision coverage {len(landscape_actions)} != 720")
    dm_stages = [row for row in stage_rows if row["view"] == "DM"]
    # The structural-churn row was moved out of this table: the churn rule is
    # now compared in the churn-discriminator suite instead of sharing the
    # matched-shadow comparison here.

    add_landscape_method(summaries["DM"], dm_stages, quality, generations, landscape_actions)
    provenance["landscape_decisions_sha256"] = sha256_file(args.landscape_decisions)

    errors: list[str] = []
    dm = summaries["DM"]
    if dm["decision_units"] != 72 or dm["liveopt_full_actions"] != 12:
        errors.append("DM decision denominator or Full-action count differs from 12/72")
    if abs(next(row for row in dm["methods"] if row["method"] == "LiveOpt")["quality"] - 0.7789711125) > 1e-12:
        errors.append("DM LiveOpt quality differs from the canonical shadow assembly")

    payload = {
        "protocol": "liveopt_restart_selector_regret_same_incoming_population_v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "decision_unit": "episode-stage; ten numerical seeds averaged within each unit",
        "scope_note": "The post-hoc oracle is a diagnostic, not a deployed policy.",
        "landscape_note": "The sensor gate chooses per seed from 10% old/new public objective re-evaluations; held-out quality is joined only afterward.",
        "views": summaries,
        "provenance": provenance,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        args.out_dir / "method_summary.csv",
        [row for view in summaries.values() for row in view["methods"]],
        [
            "view",
            "method",
            "full_actions",
            "full_cells",
            "decision_units",
            "decision_cells",
            "quality",
            "mean_generations",
            "regret",
        ],
    )
    write_csv(
        args.out_dir / "stage_decisions.csv",
        stage_rows,
        sorted({key for row in stage_rows for key in row}),
    )
    write_csv(
        args.out_dir / "episode_deltas.csv",
        episode_rows,
        list(episode_rows[0]),
    )
    write_tex(args.paper_table, summaries)
    print(json.dumps({"status": payload["status"], "out_dir": str(args.out_dir), "paper_table": str(args.paper_table), "errors": errors}, indent=2))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
