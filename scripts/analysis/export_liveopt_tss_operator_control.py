#!/usr/bin/env python3
"""Export a matched, provider-free control for TSS's typed variation operators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / (
    "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/"
    "reference_metrics/reference_stage_metrics.csv"
)
CONTROL_ROOT = ROOT / "outputs/liveopt_tss_operator_control_p010_p015_200x200x3_20260727"
CONTROL = CONTROL_ROOT / "reference_metrics/reference_stage_metrics.csv"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_tss_operator_control_20260727"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_tss_operator_control_rows.tex"
EPISODES = ("NLDO-P010", "NLDO-P015")
SEEDS = (0, 1, 2)
STAGES = tuple(range(1, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_cells(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode_id = str(row.get("episode_id") or "")
            seed = int(row.get("run_seed") or -1)
            stage = int(row.get("stage_index") or -1)
            if episode_id not in EPISODES or seed not in SEEDS or stage not in STAGES:
                continue
            feasible = as_bool(row.get("feasible")) and not as_bool(row.get("missing_stage"))
            quality = float(row.get("normalized_hv") or 0.0) if feasible else 0.0
            cells[(episode_id, seed, stage)] = {
                "episode_id": episode_id,
                "seed": seed,
                "stage": stage,
                "feasible": feasible,
                "quality": quality,
            }
    return cells


def summarize(cells: dict[tuple[str, int, int], dict[str, Any]]) -> dict[str, Any]:
    expected = {(episode, seed, stage) for episode in EPISODES for seed in SEEDS for stage in STAGES}
    missing = sorted(expected - set(cells))
    values = [cells[key] for key in sorted(expected & set(cells))]
    by_episode = {}
    for episode in EPISODES:
        rows = [row for row in values if row["episode_id"] == episode]
        by_episode[episode] = {
            "cells": len(rows),
            "solve_rate": statistics.mean(float(row["feasible"]) for row in rows) if rows else 0.0,
            "online_quality": statistics.mean(row["quality"] for row in rows) if rows else 0.0,
        }
    return {
        "cells": len(values),
        "missing_cells": [list(key) for key in missing],
        "solve_rate": statistics.mean(float(row["feasible"]) for row in values) if values else 0.0,
        "online_quality": statistics.mean(row["quality"] for row in values) if values else 0.0,
        "by_episode": by_episode,
    }


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    for path in (args.baseline, args.control, args.control_root / "summary.json"):
        if not path.exists():
            errors.append(f"missing input: {path}")
    if errors:
        raise SystemExit("; ".join(errors))

    summary = json.loads((args.control_root / "summary.json").read_text(encoding="utf-8"))
    budget = dict(summary.get("budget") or {})
    if summary.get("status") != "completed":
        errors.append(f"control run is not complete: {summary.get('status')}")
    if budget.get("variation_mode") != "type_agnostic_resampling":
        errors.append(f"unexpected variation mode: {budget.get('variation_mode')}")
    if budget.get("lineage") != "source_liveopt_history":
        errors.append(f"unexpected lineage: {budget.get('lineage')}")
    if not budget.get("require_source_final_population"):
        errors.append("source final populations were not required")
    if int(budget.get("population_size") or 0) != 200 or int(budget.get("generations") or 0) != 200:
        errors.append("control is not a 200x200 run")
    if int(budget.get("seed_count") or 0) != 3:
        errors.append("control does not use three numerical seeds")

    baseline = summarize(load_cells(args.baseline))
    control = summarize(load_cells(args.control))
    if baseline["missing_cells"]:
        errors.append(f"baseline missing {len(baseline['missing_cells'])} cells")
    if control["missing_cells"]:
        errors.append(f"control missing {len(control['missing_cells'])} cells")

    payload = {
        "protocol": "tss_matched_type_agnostic_variation_v1",
        "design": {
            "episodes": list(EPISODES),
            "numerical_seeds": list(SEEDS),
            "stages": list(STAGES),
            "population_size": 200,
            "generation_cap": 200,
            "lineage": "same seed-matched LiveOpt population immediately before each restart",
            "kept_fixed": [
                "SegmentSpec representation",
                "Workbench setup and fitness slots",
                "typed repair and interface validation",
                "semantic restart action",
                "incoming population",
                "selection, archive, stopping, and search budget",
            ],
            "changed": (
                "type-specific crossover and mutation are replaced by whole-segment uniform "
                "crossover and whole-segment valid resampling"
            ),
        },
        "baseline": baseline,
        "type_agnostic_resampling": control,
        "delta_online_quality": control["online_quality"] - baseline["online_quality"],
        "inputs": {
            "baseline_csv": str(args.baseline.relative_to(ROOT)),
            "baseline_sha256": sha256_file(args.baseline),
            "control_csv": str(args.control.relative_to(ROOT)),
            "control_sha256": sha256_file(args.control),
            "control_summary": str((args.control_root / "summary.json").relative_to(ROOT)),
            "control_summary_sha256": sha256_file(args.control_root / "summary.json"),
        },
        "errors": errors,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = []
    for label, values in (
        ("Typed operators", baseline),
        ("Generic random sampling", control),
    ):
        lines.append(
            f"{label} & {100.0 * values['solve_rate']:.1f}\\% & "
            f"{values['by_episode']['NLDO-P010']['online_quality']:.3f} & "
            f"{values['by_episode']['NLDO-P015']['online_quality']:.3f} & "
            f"{values['online_quality']:.3f} \\\\"
        )
    args.paper_tex.parent.mkdir(parents=True, exist_ok=True)
    args.paper_tex.write_text(
        "% Generated by scripts/analysis/export_liveopt_tss_operator_control.py.\n"
        "\\newcommand{\\LiveOptTSSOperatorControlRows}{%\n"
        + "\n".join(lines)
        + "\n}\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "paper_tex": str(args.paper_tex), "errors": errors}))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
