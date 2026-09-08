#!/usr/bin/env python3
"""Summarize matched P007--P015 restart replays without mixing metric views.

Scalar episodes (P007--P009) use normalized_score and Pareto episodes
(P010--P015) use normalized_hv.  The two views are reported separately; an
optional overall row averages their already-normalized per-seed qualities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CellKey = tuple[str, int, int]


def main() -> int:
    args = parse_args()
    methods = parse_runs(args.run)
    stage_set = set(args.stage)
    rows_by_method: dict[str, dict[CellKey, dict[str, Any]]] = defaultdict(dict)
    sources: dict[str, list[str]] = defaultdict(list)
    for label, run_dir in methods:
        path = run_dir / "reference_metrics" / "reference_stage_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing reference metrics for {label}: {path}")
        sources[label].append(str(path))
        for row in read_csv(path):
            episode_id = str(row["episode_id"])
            stage = int(row["stage_index"])
            seed = int(row["run_seed"])
            if stage not in stage_set or not in_scope(episode_id):
                continue
            key = (episode_id, stage, seed)
            if key in rows_by_method[label]:
                raise ValueError(f"duplicate matched cell for {label}: {key}")
            metric = "normalized_score" if profile(episode_id) == "DS" else "normalized_hv"
            rows_by_method[label][key] = {
                "quality": as_float(row.get(metric), default=0.0),
                "feasible": truthy(row.get("feasible")),
                "metric": metric,
            }

    labels = list(dict.fromkeys(label for label, _ in methods))
    if not labels:
        raise ValueError("at least one --run is required")
    expected = set(rows_by_method[labels[0]])
    if not expected:
        raise ValueError("no P007--P015 cells found for the requested stages")
    coverage = {}
    for label in labels:
        observed = set(rows_by_method[label])
        coverage[label] = {
            "cells": len(observed),
            "missing_vs_first": [format_key(key) for key in sorted(expected - observed)],
            "extra_vs_first": [format_key(key) for key in sorted(observed - expected)],
        }
        if observed != expected:
            raise ValueError(f"matched-cell coverage differs for {label}: {coverage[label]}")

    episode_rows = summarize_episode_stage(labels, rows_by_method, expected)
    aggregate_label = "all" if stage_set == set(range(1, 13)) else "selected"
    group_rows = summarize_groups(labels, rows_by_method, expected, aggregate_label=aggregate_label)
    paired_rows = summarize_pairs(
        labels,
        rows_by_method,
        expected,
        reference_label=args.reference_label,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        aggregate_label=aggregate_label,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "episode_stage_summary.csv", episode_rows)
    write_csv(out_dir / "group_summary.csv", group_rows)
    write_csv(out_dir / "paired_deltas.csv", paired_rows)
    report = {
        "schema_version": "liveopt_restart_comparison_v1",
        "scope": "NLDO-P007--NLDO-P015",
        "stages": sorted(stage_set),
        "metric_contract": {
            "DS": "normalized_score (P007--P009)",
            "DM": "normalized_hv (P010--P015)",
            "All": "macro mean of the corresponding normalized per-seed quality; DS and DM remain separately reported",
        },
        "methods": labels,
        "sources": dict(sources),
        "coverage": coverage,
        "episode_stage_summary": episode_rows,
        "group_summary": group_rows,
        "paired_deltas": paired_rows,
        "bootstrap": {
            "type": "hierarchical paired bootstrap over episodes, stages within episode, and seeds within stage",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir), "coverage": coverage, "paired_deltas": paired_rows}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched LiveOpt restart controls over P007--P015.")
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR; repeat a label for split P007--P009/P010--P015 runs.")
    parser.add_argument("--stage", action="append", type=int, default=[], help="Stage to include; repeat as needed. Defaults to t11 and t12.")
    parser.add_argument("--reference-label", default="LiveOpt", help="Method on the left side of paired deltas.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.stage:
        args.stage = [11, 12]
    return args


def parse_runs(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run must be LABEL=RUN_DIR, got {item!r}")
        label, path = item.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def in_scope(episode_id: str) -> bool:
    try:
        number = int(episode_id.rsplit("P", 1)[1])
    except (IndexError, ValueError):
        return False
    return 7 <= number <= 15


def profile(episode_id: str) -> str:
    return "DS" if int(episode_id.rsplit("P", 1)[1]) <= 9 else "DM"


def summarize_episode_stage(
    labels: list[str],
    data: dict[str, dict[CellKey, dict[str, Any]]],
    keys: set[CellKey],
) -> list[dict[str, Any]]:
    out = []
    episodes_stages = sorted({(episode, stage) for episode, stage, _ in keys})
    for label in labels:
        for episode, stage in episodes_stages:
            items = [data[label][key] for key in sorted(keys) if key[0] == episode and key[1] == stage]
            values = [float(item["quality"]) for item in items]
            out.append(
                {
                    "method": label,
                    "view": profile(episode),
                    "episode_id": episode,
                    "stage_index": stage,
                    "metric": items[0]["metric"],
                    "seed_count": len(items),
                    "solve_rate": mean(float(item["feasible"]) for item in items),
                    "quality_mean": mean(values),
                    "quality_std": pstdev(values) if len(values) > 1 else 0.0,
                }
            )
    return out


def summarize_groups(
    labels: list[str],
    data: dict[str, dict[CellKey, dict[str, Any]]],
    keys: set[CellKey],
    *,
    aggregate_label: str,
) -> list[dict[str, Any]]:
    out = []
    stages: list[int | str] = sorted({stage for _, stage, _ in keys}) + [aggregate_label]
    for label in labels:
        for view in ("DS", "DM", "All"):
            for stage in stages:
                selected = [
                    key
                    for key in sorted(keys)
                    if (view == "All" or profile(key[0]) == view)
                    and (stage == aggregate_label or key[1] == stage)
                ]
                items = [data[label][key] for key in selected]
                if not items:
                    continue
                values = [float(item["quality"]) for item in items]
                out.append(
                    {
                        "method": label,
                        "view": view,
                        "stage": stage,
                        "problem_count": len({key[0] for key in selected}),
                        "cell_count": len(items),
                        "solve_rate": mean(float(item["feasible"]) for item in items),
                        "quality_mean": mean(values),
                        "quality_std": pstdev(values) if len(values) > 1 else 0.0,
                    }
                )
    return out


def summarize_pairs(
    labels: list[str],
    data: dict[str, dict[CellKey, dict[str, Any]]],
    keys: set[CellKey],
    *,
    reference_label: str,
    samples: int,
    seed: int,
    aggregate_label: str,
) -> list[dict[str, Any]]:
    if reference_label not in labels:
        raise ValueError(f"reference label {reference_label!r} is not among {labels}")
    out = []
    stages: list[int | str] = sorted({stage for _, stage, _ in keys}) + [aggregate_label]
    for comparator in labels:
        if comparator == reference_label:
            continue
        for view in ("DS", "DM", "All"):
            for stage in stages:
                selected = [
                    key
                    for key in sorted(keys)
                    if (view == "All" or profile(key[0]) == view)
                    and (stage == aggregate_label or key[1] == stage)
                ]
                if not selected:
                    continue
                diffs = {
                    key: float(data[reference_label][key]["quality"]) - float(data[comparator][key]["quality"])
                    for key in selected
                }
                point = mean(diffs.values())
                low, high = hierarchical_ci(diffs, samples=samples, seed=seed + len(out) * 7919)
                out.append(
                    {
                        "reference": reference_label,
                        "comparator": comparator,
                        "view": view,
                        "stage": stage,
                        "problem_count": len({key[0] for key in selected}),
                        "paired_cells": len(selected),
                        "quality_delta": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return out


def hierarchical_ci(diffs: dict[CellKey, float], *, samples: int, seed: int) -> tuple[float, float]:
    clusters: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (episode, stage, _), value in diffs.items():
        clusters[episode][stage].append(value)
    episodes = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        values = []
        for _ in episodes:
            episode = rng.choice(episodes)
            stages = sorted(clusters[episode])
            for _ in stages:
                stage = rng.choice(stages)
                seed_values = clusters[episode][stage]
                values.extend(rng.choice(seed_values) for _ in seed_values)
        estimates.append(mean(values))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    index = q * (len(values) - 1)
    lo = int(math.floor(index))
    hi = int(math.ceil(index))
    if lo == hi:
        return values[lo]
    weight = index - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def format_key(key: CellKey) -> str:
    return f"{key[0]}:t{key[1]:02d}:seed{key[2]}"


if __name__ == "__main__":
    raise SystemExit(main())
