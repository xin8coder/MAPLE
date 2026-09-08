#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evo2.evaluation.moea_metrics import approximate_hypervolume, expanded_reference_point


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute NLDO HV ratios under deterministic thinnings of the held-out reference archives."
    )
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Method label and reference_stage_metrics.json as LABEL=PATH.",
    )
    parser.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--stage-min", type=int, default=0)
    parser.add_argument("--stage-max", type=int)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    fractions = sorted({float(item) for item in args.fractions.split(",") if item.strip()})
    if not fractions or any(value <= 0.0 or value > 1.0 for value in fractions):
        raise ValueError("fractions must lie in (0, 1]")
    references = load_references(args.reference_jsonl)
    method_rows = {}
    for item in args.metrics:
        label, raw_path = item.split("=", 1)
        method_rows[label] = json.loads(Path(raw_path).read_text(encoding="utf-8"))

    denominator_cache: dict[tuple[str, int, float, int], float] = {}
    rows: list[dict[str, Any]] = []
    for label, metrics in sorted(method_rows.items()):
        observations = [
            row
            for row in metrics
            if row.get("hv") is not None
            and int(row.get("stage_index") or 0) >= int(args.stage_min)
            and (args.stage_max is None or int(row.get("stage_index") or 0) <= int(args.stage_max))
            and (str(row.get("episode_id") or ""), int(row.get("stage_index") or 0)) in references
        ]
        by_fraction: dict[float, list[float]] = defaultdict(list)
        for fraction in fractions:
            repeat_count = 1 if fraction == 1.0 else max(1, int(args.repeats))
            for repeat in range(repeat_count):
                ratios = []
                for metric in observations:
                    key = (str(metric["episode_id"]), int(metric["stage_index"]))
                    reference = references[key]
                    cache_key = (key[0], key[1], fraction, repeat)
                    denominator = denominator_cache.get(cache_key)
                    if denominator is None:
                        subset = thin_archive(
                            reference["pareto_archive"],
                            fraction,
                            stable_seed(args.seed, key[0], key[1], repeat),
                        )
                        fixed_reference_point = expanded_reference_point(
                            reference["pareto_archive"],
                            reference["objectives"],
                            reference.get("hypervolume_reference_point"),
                        )
                        denominator = float(
                            approximate_hypervolume(
                                subset,
                                reference["objectives"],
                                fixed_reference_point,
                                reference.get("hypervolume_ideal_point"),
                                samples=0,
                            )
                        )
                        denominator_cache[cache_key] = denominator
                    ratios.append(float(metric.get("hv") or 0.0) / denominator if denominator > 0.0 else 0.0)
                by_fraction[fraction].append(mean(ratios) if ratios else 0.0)
        full = mean(by_fraction.get(1.0) or [0.0])
        for fraction in fractions:
            values = by_fraction[fraction]
            rows.append(
                {
                    "method": label,
                    "reference_fraction": fraction,
                    "repeats": len(values),
                    "stage_seed_observations": len(observations),
                    "mean_hv_ratio": mean(values),
                    "std_hv_ratio": pstdev(values) if len(values) > 1 else 0.0,
                    "delta_from_full": mean(values) - full,
                }
            )

    payload = {
        "schema_version": "nldo_reference_archive_thinning_sensitivity_v1",
        "reference_jsonl": str(args.reference_jsonl),
        "fractions": fractions,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "stage_min": int(args.stage_min),
        "stage_max": int(args.stage_max) if args.stage_max is not None else None,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def load_references(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    references = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        episode_id = str(episode.get("episode_id") or "")
        for stage_index, step in enumerate((episode.get("evaluation") or {}).get("reference_trajectory") or []):
            if not isinstance(step, dict) or not step.get("pareto_archive"):
                continue
            references[(episode_id, stage_index)] = step
    return references


def thin_archive(archive: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    if fraction >= 1.0 or len(archive) <= 1:
        return list(archive)
    size = max(1, round(len(archive) * fraction))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(archive)), min(size, len(archive))))
    return [archive[index] for index in indices]


def stable_seed(base: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


if __name__ == "__main__":
    main()
