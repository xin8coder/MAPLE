#!/usr/bin/env python3
"""Export current NLDO results by public update type and aggregation level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    ROOT / "outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713/reference_metrics/reference_stage_metrics.csv",
    ROOT / "outputs/liveopt_ablation_llmbackbone_sequential_p007_p009_200x200x10_trace_t11pop_20260714/reference_metrics/reference_stage_metrics.csv",
    ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/reference_metrics/reference_stage_metrics.csv",
)
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_update_strata_current_20260715"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_update_strata_rows.tex"
EXPECTED_EPISODES = tuple(f"NLDO-P{number:03d}" for number in range(1, 16))
EXPECTED_STAGES = tuple(range(1, 13))
EXPECTED_SEEDS = tuple(range(10))
DISPLAY_STRATUM = {
    "maintenance": "Maintenance",
    "simple": "Local numeric",
    "memory": "Referential",
    "objective": "Objective",
    "large": "Disruptive update",
    "active": "All active",
    "all": "All updates",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        action="append",
        default=None,
        help="Formal stage-metric CSV. Later files replace duplicate cells.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.metrics_csv is None:
        args.metrics_csv = list(DEFAULT_METRICS)
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_metrics(paths: list[Path]) -> tuple[dict[tuple[str, int, int], dict[str, str]], int]:
    merged: dict[tuple[str, int, int], dict[str, str]] = {}
    overrides = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                episode = str(row.get("episode_id") or "")
                stage = int(float(row.get("stage_index") or 0))
                seed = int(float(row.get("run_seed") or 0))
                if episode not in EXPECTED_EPISODES or stage not in EXPECTED_STAGES or seed not in EXPECTED_SEEDS:
                    continue
                key = (episode, stage, seed)
                overrides += int(key in merged)
                merged[key] = row
    return merged, overrides


def metric_pass(row: dict[str, str]) -> bool:
    true_pass = row.get("true_pass")
    return truthy(true_pass) if true_pass not in (None, "") else truthy(row.get("feasible"))


def state_record(
    episode: dict[str, Any], stage: int, rows: list[dict[str, str]]
) -> dict[str, Any]:
    multi = bool((episode.get("public_context") or {}).get("multi_objective"))
    view = "DM" if multi else "DS"
    quality_field = "normalized_hv" if multi else "normalized_score"
    passes = [metric_pass(row) for row in rows]
    quality = [float(row.get(quality_field) or 0.0) if passed else 0.0 for row, passed in zip(rows, passes)]
    igd_values = [as_float(row.get("igd")) for row, passed in zip(rows, passes) if passed]
    igd_values = [value for value in igd_values if value is not None]
    update = (episode.get("update_stream") or [])[stage - 1]
    return {
        "episode_id": str(episode["episode_id"]),
        "profile": str(episode.get("domain") or episode.get("family") or "unknown"),
        "view": view,
        "stage_index": stage,
        "stratum": str(update.get("difficulty") or "unlabeled"),
        "state_solved": any(passes),
        "all_seeds_pass": all(passes),
        "seed_pass_rate": fmean(passes),
        "quality": fmean(quality),
        "igd": fmean(igd_values) if igd_values else None,
        "seed_cells": len(rows),
    }


def aggregate_states(scope: str, view: str, stratum: str, states: list[dict[str, Any]]) -> dict[str, Any]:
    igd = [float(state["igd"]) for state in states if state.get("igd") is not None]
    return {
        "scope": scope,
        "view": view,
        "stratum": stratum,
        "states": len(states),
        "seed_cells": sum(int(state["seed_cells"]) for state in states),
        "state_solve_rate": fmean(bool(state["state_solved"]) for state in states),
        "all_seed_pass_rate": fmean(bool(state["all_seeds_pass"]) for state in states),
        "seed_pass_rate": fmean(float(state["seed_pass_rate"]) for state in states),
        "quality": fmean(float(state["quality"]) for state in states),
        "igd_pass_only": fmean(igd) if igd else None,
    }


def aggregate_entity(entity_type: str, entity: str, states: list[dict[str, Any]]) -> dict[str, Any]:
    active = [state for state in states if state["stratum"] != "maintenance"]
    return {
        "entity_type": entity_type,
        "entity": entity,
        "view": states[0]["view"],
        "states": len(states),
        "active_states": len(active),
        "all_quality": fmean(float(state["quality"]) for state in states),
        "active_quality": fmean(float(state["quality"]) for state in active) if active else None,
        "state_solve_rate": fmean(bool(state["state_solved"]) for state in states),
        "seed_pass_rate": fmean(float(state["seed_pass_rate"]) for state in states),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def render_tex(rows: list[dict[str, Any]]) -> str:
    selected = []
    order = {
        "DS": ("all", "active", "maintenance", "simple", "memory", "objective", "large"),
        "DM": ("all", "memory", "objective", "large"),
    }
    lookup = {(row["scope"], row["view"], row["stratum"]): row for row in rows}
    for view in ("DS", "DM"):
        for stratum in order[view]:
            row = lookup.get(("all15", view, stratum))
            if row is not None:
                selected.append(row)
    lines = [
        "% Generated by scripts/analysis/export_nldo_update_strata.py.",
        r"\newcommand{\LiveOptUpdateStrataRows}{%",
    ]
    previous = None
    for row in selected:
        view = str(row["view"])
        view_cell = view if view != previous else ""
        previous = view
        igd = r"\textemdash{}" if row["igd_pass_only"] is None else f"{float(row['igd_pass_only']):.3f}"
        lines.append(
            f"{view_cell} & {DISPLAY_STRATUM.get(str(row['stratum']), str(row['stratum']))} & "
            f"{int(row['states'])} & {100.0 * float(row['state_solve_rate']):.1f}\\% & "
            f"{float(row['quality']):.3f} & {igd} " + r"\\"
        )
    lines.append("}")
    ds_order = ("all", "active", "maintenance", "simple", "memory", "objective", "large")
    dm_order = ("all", "memory", "objective", "large")
    lines.append(r"\newcommand{\LiveOptUpdateStrataCompactRows}{%")
    for index, ds_stratum in enumerate(ds_order):
        ds = lookup[("all15", "DS", ds_stratum)]
        if index < len(dm_order):
            dm = lookup[("all15", "DM", dm_order[index])]
            dm_cells = (
                f"{DISPLAY_STRATUM[dm_order[index]]} & {int(dm['states'])} & "
                f"{float(dm['quality']):.3f} & {float(dm['igd_pass_only']):.3f}"
            )
        else:
            dm_cells = " &  &  & "
        lines.append(
            f"{DISPLAY_STRATUM[ds_stratum]} & {int(ds['states'])} & {float(ds['quality']):.3f} & "
            + dm_cells
            + r" \\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    episodes = {str(row["episode_id"]): row for row in load_jsonl(args.episodes_jsonl)}
    metrics, overrides = load_metrics(list(args.metrics_csv))
    expected_cells = {
        (episode, stage, seed)
        for episode in EXPECTED_EPISODES
        for stage in EXPECTED_STAGES
        for seed in EXPECTED_SEEDS
    }
    errors: list[str] = []
    if set(metrics) != expected_cells:
        errors.append(
            f"metric coverage mismatch: rows={len(metrics)}, missing={len(expected_cells - set(metrics))}, "
            f"extra={len(set(metrics) - expected_cells)}"
        )

    states: list[dict[str, Any]] = []
    for episode_id in EXPECTED_EPISODES:
        episode = episodes[episode_id]
        for stage in EXPECTED_STAGES:
            rows = [metrics[(episode_id, stage, seed)] for seed in EXPECTED_SEEDS if (episode_id, stage, seed) in metrics]
            if len(rows) != 10:
                errors.append(f"{episode_id}/t{stage:02d}: expected 10 seeds, found {len(rows)}")
                continue
            states.append(state_record(episode, stage, rows))

    aggregate_rows: list[dict[str, Any]] = []
    for scope, scope_states in (
        ("all15", states),
        ("p010_p015", [state for state in states if state["episode_id"] in {f"NLDO-P{n:03d}" for n in range(10, 16)}]),
    ):
        for view in ("DS", "DM"):
            view_states = [state for state in scope_states if state["view"] == view]
            if not view_states:
                continue
            strata = sorted({str(state["stratum"]) for state in view_states})
            for stratum in strata:
                selected = [state for state in view_states if state["stratum"] == stratum]
                aggregate_rows.append(aggregate_states(scope, view, stratum, selected))
            aggregate_rows.append(aggregate_states(scope, view, "all", view_states))
            active = [state for state in view_states if state["stratum"] != "maintenance"]
            if active and len(active) != len(view_states):
                aggregate_rows.append(aggregate_states(scope, view, "active", active))

    episode_rows = [
        aggregate_entity("episode", episode_id, [state for state in states if state["episode_id"] == episode_id])
        for episode_id in EXPECTED_EPISODES
    ]
    profiles = sorted({str(state["profile"]) for state in states})
    profile_rows = [
        aggregate_entity("profile", profile, [state for state in states if state["profile"] == profile])
        for profile in profiles
    ]

    summary = {
        "schema_version": "nldo_update_strata_v2",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "benchmark": str(args.episodes_jsonl),
        "benchmark_sha256": sha256_file(args.episodes_jsonl),
        "metric_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.metrics_csv
        ],
        "metric_overrides": overrides,
        "metric_cells": len(metrics),
        "public_states": len(states),
        "aggregation_definition": {
            "state_solve": "at least one of ten numerical seeds passes held-out feasibility",
            "quality": "seed-mean feasibility-gated normalized scalar quality or HV ratio",
            "igd_pass_only": "seed-mean IGD among held-out-valid results",
            "active": "all non-maintenance public updates",
        },
        "rows": aggregate_rows,
        "episode_macro": {
            view: fmean(float(row["all_quality"]) for row in episode_rows if row["view"] == view)
            for view in ("DS", "DM")
        },
        "profile_macro": {
            view: fmean(float(row["all_quality"]) for row in profile_rows if row["view"] == view)
            for view in ("DS", "DM")
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "state_rows.csv", states)
    write_csv(args.out_dir / "strata.csv", aggregate_rows)
    write_csv(args.out_dir / "episodes.csv", episode_rows)
    write_csv(args.out_dir / "profiles.csv", profile_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.paper_table.parent.mkdir(parents=True, exist_ok=True)
    args.paper_table.write_text(render_tex(aggregate_rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
