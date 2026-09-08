#!/usr/bin/env python3
"""Aggregate the matched DeepSeek/Kimi model-family check for Q2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KIMI_ROOT = ROOT / "outputs/liveopt_model_family_kimi_k3_q2_20260717"
DEFAULT_OUT = ROOT / "logs/analysis/liveopt_model_family_q2_20260717"
DEFAULT_TEX = ROOT / "release_artifacts/paper_table_exports/liveopt_model_family_rows.tex"
BENCHMARK = ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"
REFERENCE = ROOT / (
    "outputs/reference_rebuild_mo_late_regime_final_p010_p015_500x500x10_20260713/"
    "nldo_15episodes_12updates_csv.strong_moea_500x500x10.jsonl"
)
EXPECTED_BENCHMARK_SHA256 = "15717873e60fe4a05ca8305829dbf2bbe5f3b14ca075b1c128e87a455362e303"
EXPECTED_REFERENCE_SHA256 = "50126c25eb837bdbf2dbc2e7710226472027a1a880305d331dc006438a02c5e5"
SEEDS = range(3)
STAGES = range(1, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kimi-root", type=Path, default=DEFAULT_KIMI_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def deepseek_run(episode_id: str, controller: int) -> Path:
    problem = int(episode_id.rsplit("P", 1)[-1])
    if controller == 0:
        return ROOT / "outputs/liveopt_matched_full_p010_p015_200x200x10_20260714"
    return ROOT / f"outputs/liveopt_controller_repeat_p{problem:03d}_liveopt_c{controller}_200x200x3_20260717"


def kimi_run(root: Path, episode_id: str, controller: int) -> Path:
    return root / "jobs" / episode_id / f"c{controller}"


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def load_controller(
    family: str,
    episode_id: str,
    controller: int,
    run_dir: Path,
    errors: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path = run_dir / "reference_metrics/reference_stage_metrics.csv"
    summary_path = run_dir / "summary.json"
    rows: dict[tuple[int, int], dict[str, str]] = {}
    if metrics_path.exists():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("episode_id") or "") != episode_id:
                    continue
                seed = int(row.get("run_seed") or -1)
                stage = int(row.get("stage_index") or -1)
                if seed in SEEDS and stage in STAGES:
                    rows[(seed, stage)] = row
    expected = len(SEEDS) * len(STAGES)
    if len(rows) != expected:
        errors.append(f"{family} {episode_id} c{controller}: {len(rows)}/{expected} metric cells")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    if family == "Kimi K3 (1M)":
        if summary.get("status") != "completed":
            errors.append(f"{family} {episode_id} c{controller}: summary status={summary.get('status')!r}")
        if summary.get("provider") != "kimi" or summary.get("model") != "k3[1m]":
            errors.append(
                f"{family} {episode_id} c{controller}: provider/model="
                f"{summary.get('provider')!r}/{summary.get('model')!r}"
            )
        budget = summary.get("budget") or {}
        for key, expected_value in {
            "population_size": 200,
            "initial_population_size": 200,
            "generations": 200,
            "initial_generations": 200,
            "archive_limit": 500,
            "seed_count": 3,
            "seed_start": 0,
            "controller_seed": controller,
        }.items():
            if budget.get(key) != expected_value:
                errors.append(
                    f"{family} {episode_id} c{controller}: budget.{key}={budget.get(key)!r}"
                )

    cells = []
    for seed in SEEDS:
        for stage in STAGES:
            source = rows.get((seed, stage))
            passed = bool(source and as_bool(source.get("true_pass")))
            cells.append(
                {
                    "model_family": family,
                    "episode_id": episode_id,
                    "controller_seed": controller,
                    "run_seed": seed,
                    "stage_index": stage,
                    "true_pass": passed,
                    "online_quality": as_float(source.get("normalized_hv")) if source and passed else 0.0,
                    "metric_available": source is not None,
                }
            )
    record = {
        "model_family": family,
        "episode_id": episode_id,
        "controller_seed": controller,
        "run_dir": display_path(run_dir),
        "complete": len(rows) == expected,
        "solve_rate": mean(float(row["true_pass"]) for row in cells),
        "online_quality": mean(float(row["online_quality"]) for row in cells),
    }
    return record, cells


def aggregate(controllers: list[dict[str, Any]], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    families = ["DeepSeek-V4-Pro", "Kimi K3 (1M)"]
    for family in families:
        for episode_id in ("NLDO-P010", "NLDO-P015", "ALL"):
            selected_controllers = [
                row
                for row in controllers
                if row["model_family"] == family
                and (episode_id == "ALL" or row["episode_id"] == episode_id)
            ]
            selected_cells = [
                row
                for row in cells
                if row["model_family"] == family
                and (episode_id == "ALL" or row["episode_id"] == episode_id)
            ]
            qualities = [float(row["online_quality"]) for row in selected_controllers]
            result.append(
                {
                    "model_family": family,
                    "episode_id": episode_id,
                    "controller_count": len(selected_controllers),
                    "complete_controllers": sum(bool(row["complete"]) for row in selected_controllers),
                    "cell_count": len(selected_cells),
                    "solve_rate": mean(float(row["true_pass"]) for row in selected_cells),
                    "online_quality": mean(float(row["online_quality"]) for row in selected_cells),
                    "controller_quality_sd": stdev(qualities) if len(qualities) > 1 else 0.0,
                }
            )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_tex(groups: list[dict[str, Any]]) -> str:
    lookup = {(row["model_family"], row["episode_id"]): row for row in groups}
    lines = [
        "% Generated by scripts/analysis/export_kimi_model_family_q2.py.",
        r"\newcommand{\LiveOptModelFamilyRows}{%",
    ]
    labels = {
        "DeepSeek-V4-Pro": "DeepSeek-V4-Pro (Preview)",
        "Kimi K3 (1M)": "Kimi K3 (1M)",
    }
    for family in ("DeepSeek-V4-Pro", "Kimi K3 (1M)"):
        p010 = lookup[(family, "NLDO-P010")]
        p015 = lookup[(family, "NLDO-P015")]
        overall = lookup[(family, "ALL")]
        lines.append(
            f"{labels[family]} & {overall['complete_controllers']}/6 & "
            f"{100 * p010['solve_rate']:.1f} & {p010['online_quality']:.3f} & "
            f"{100 * p015['solve_rate']:.1f} & {p015['online_quality']:.3f} & "
            f"{100 * overall['solve_rate']:.1f} & {overall['online_quality']:.3f} \\\\"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    for path, expected in (
        (BENCHMARK, EXPECTED_BENCHMARK_SHA256),
        (REFERENCE, EXPECTED_REFERENCE_SHA256),
    ):
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            errors.append(f"{path}: sha256={actual}, expected {expected}")

    controllers: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for family in ("DeepSeek-V4-Pro", "Kimi K3 (1M)"):
        for episode_id in ("NLDO-P010", "NLDO-P015"):
            for controller in range(3):
                run_dir = (
                    deepseek_run(episode_id, controller)
                    if family == "DeepSeek-V4-Pro"
                    else kimi_run(args.kimi_root, episode_id, controller)
                )
                record, controller_cells = load_controller(
                    family, episode_id, controller, run_dir, errors
                )
                controllers.append(record)
                cells.extend(controller_cells)
    groups = aggregate(controllers, cells)
    if len(controllers) != 12 or len(cells) != 432:
        errors.append(f"unexpected evidence size: {len(controllers)} controllers, {len(cells)} cells")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "controller_summary.csv", controllers)
    write_csv(args.out_dir / "stage_cells.csv", cells)
    write_csv(args.out_dir / "aggregate_summary.csv", groups)
    payload = {
        "status": "passed" if not errors else "incomplete",
        "protocol": "q2_matched_model_family_p010_p015_v1",
        "changed_factor": "controller model family only",
        "fixed": {
            "problems": ["NLDO-P010", "NLDO-P015"],
            "controller_trajectories": 3,
            "numerical_seeds": [0, 1, 2],
            "stages": list(STAGES),
            "population_size": 200,
            "generation_cap": 200,
            "archive_limit": 500,
            "restart_choices": "frozen verified public-only Warm/Full decisions",
            "held_out_reference": str(REFERENCE.relative_to(ROOT)),
        },
        "controllers": controllers,
        "aggregate": groups,
        "errors": errors,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not errors:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(render_tex(groups), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if errors and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
