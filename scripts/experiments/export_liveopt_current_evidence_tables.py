#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PROFILE_BY_PID = {
    1: "Selection/allocation",
    2: "Selection/allocation",
    3: "Selection/allocation",
    4: "Precedence scheduling",
    5: "Precedence scheduling",
    6: "Precedence scheduling",
    7: "Coverage rostering",
    8: "Coverage rostering",
    9: "Coverage rostering",
    10: "NLDO-DM, ordered service",
    11: "NLDO-DM, ordered service",
    12: "NLDO-DM, ordered service",
    13: "NLDO-DM, cloud resources",
    14: "NLDO-DM, cloud resources",
    15: "NLDO-DM, cloud resources",
}

PROFILE_GROUPS = [
    ("Selection/allocation", range(1, 4)),
    ("Precedence scheduling", range(4, 7)),
    ("Coverage rostering", range(7, 10)),
    ("Ordered service routing", range(10, 13)),
    ("Cloud-resource placement", range(13, 16)),
]


def main() -> int:
    args = parse_args()
    metric_sources = [Path(path) for path in args.metrics_csv]
    replay_sources = [Path(path) for path in (args.replay_jsonl or [])]
    metric_rows, metric_override_count = merge_metric_sources(metric_sources, args.allow_source_overrides)
    archive_counts: dict[tuple[str, int, int], int] = {}
    replay_override_count = 0
    for path in replay_sources:
        incoming = archive_counts_by_seed_stage(path)
        overlap = set(archive_counts).intersection(incoming)
        if overlap and not args.allow_source_overrides:
            raise ValueError(f"duplicate replay episode/seed/stage keys across sources: {len(overlap)}")
        replay_override_count += len(overlap)
        archive_counts.update(incoming)
    episode_rows = build_episode_rows(metric_rows)
    mo_rows = build_mo_variance_rows(metric_rows, archive_counts)
    profile_rows, macros = build_profile_rows(metric_rows)
    write_episode_tex(Path(args.out_episode_tex), episode_rows, metric_sources)
    write_mo_tex(Path(args.out_mo_tex), mo_rows, metric_sources, replay_sources)
    if args.out_profile_tex:
        write_profile_tex(Path(args.out_profile_tex), profile_rows, macros, metric_sources)
    payload = {
        "metric_sources": [str(path) for path in metric_sources],
        "replay_sources": [str(path) for path in replay_sources],
        "metric_override_count": metric_override_count,
        "replay_override_count": replay_override_count,
        "episode_rows": episode_rows,
        "mo_rows": mo_rows,
        "profile_rows": profile_rows,
        "macros": macros,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current LiveOpt per-problem and MO seed-variance appendix rows.")
    parser.add_argument("--metrics-csv", action="append", required=True)
    parser.add_argument("--replay-jsonl", action="append")
    parser.add_argument(
        "--allow-source-overrides",
        action="store_true",
        help="Allow later sources to replace duplicate (episode, seed, stage) rows; counts are recorded in JSON.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-episode-tex", required=True)
    parser.add_argument("--out-mo-tex", required=True)
    parser.add_argument("--out-profile-tex")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_metric_sources(paths: list[Path], allow_overrides: bool) -> tuple[list[dict[str, str]], int]:
    merged: dict[tuple[str, int, int], dict[str, str]] = {}
    overrides = 0
    for path in paths:
        for row in read_csv(path):
            episode_id = str(row.get("episode_id") or "")
            if not episode_id:
                raise ValueError(f"metric row without episode_id in {path}")
            key = (
                episode_id,
                int(float(row.get("run_seed") or 0)),
                int(float(row.get("stage_index") or 0)),
            )
            if key in merged:
                if not allow_overrides:
                    raise ValueError(f"duplicate metric episode/seed/stage key across sources: {key}")
                overrides += 1
            merged[key] = row
    return list(merged.values()), overrides


def build_episode_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            buckets[episode_id].append(row)
    out: list[dict[str, Any]] = []
    for episode_id in sorted(buckets, key=episode_sort_key):
        pid = problem_index(episode_id)
        if pid is None:
            continue
        raw_values = [quality(row, pid) for row in buckets[episode_id]]
        if pid <= 9:
            values = [min(1.0, max(0.0, value)) for value in raw_values if value is not None and math.isfinite(value)]
        else:
            values = [value for value in raw_values if value is not None and math.isfinite(value)]
        mean_value = mean(values) if values else None
        if pid <= 9:
            gaps = [as_float(row.get("objective_gap")) for row in buckets[episode_id]]
            gaps = [value for value in gaps if value is not None and math.isfinite(value)]
            scalar_gap = mean(gaps) if gaps else None
            hv = None
            ideal_gap = None
            igd = None
            profile = PROFILE_BY_PID.get(pid, "Scalar")
        else:
            hv_values = [quality(row, pid) for row in buckets[episode_id]]
            hv_values = [value for value in hv_values if value is not None and math.isfinite(value)]
            igd_values = [as_float(row.get("igd")) for row in buckets[episode_id]]
            igd_values = [value for value in igd_values if value is not None and math.isfinite(value)]
            ideal_gap_values = [as_float(row.get("ideal_gap")) for row in buckets[episode_id]]
            ideal_gap_values = [value for value in ideal_gap_values if value is not None and math.isfinite(value)]
            scalar_gap = None
            hv = mean(hv_values) if hv_values else None
            ideal_gap = mean(ideal_gap_values) if ideal_gap_values else None
            igd = mean(igd_values) if igd_values else None
            profile = PROFILE_BY_PID.get(pid, "NLDO-DM")
        out.append(
            {
                "problem": f"P{pid:03d}",
                "profile": profile,
                "mean": mean_value,
                "scalar_gap": scalar_gap,
                "ideal_gap": ideal_gap,
                "hv": hv,
                "igd": igd,
            }
        )
    return out


def build_profile_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    profile_rows: list[dict[str, Any]] = []
    for profile, pids in PROFILE_GROUPS:
        pid_set = set(pids)
        selected = [row for row in rows if problem_index(str(row.get("episode_id") or "")) in pid_set]
        dynamic = [row for row in selected if int(float(row.get("stage_index") or 0)) > 0]
        is_multi = min(pid_set) >= 10
        overall_values = finite(quality(row, min(pid_set)) for row in selected)
        dynamic_values = finite(quality(row, min(pid_set)) for row in dynamic)
        if is_multi:
            gap_or_hv = mean(dynamic_values) if dynamic_values else None
            igd = finite(as_float(row.get("igd")) for row in dynamic)
            igd_value = mean(igd) if igd else None
        else:
            gaps = finite(as_float(row.get("objective_gap")) for row in selected)
            gap_or_hv = mean(gaps) if gaps else None
            igd_value = None
        profile_rows.append(
            {
                "profile": profile,
                "overall_mean": mean(overall_values) if overall_values else None,
                "dynamic_mean": mean(dynamic_values) if dynamic_values else None,
                "gap_or_hv": gap_or_hv,
                "igd": igd_value,
            }
        )

    all_values = finite(
        quality(row, problem_index(str(row.get("episode_id") or "")) or 0)
        for row in rows
    )
    dynamic_rows = [row for row in rows if int(float(row.get("stage_index") or 0)) > 0]
    dynamic_values = finite(
        quality(row, problem_index(str(row.get("episode_id") or "")) or 0)
        for row in dynamic_rows
    )
    dm_rows = [
        row
        for row in dynamic_rows
        if (problem_index(str(row.get("episode_id") or "")) or 0) >= 10
    ]
    dm_hv = finite(as_float(row.get("normalized_hv")) for row in dm_rows)
    dm_igd = finite(as_float(row.get("igd")) for row in dm_rows)
    token_values = finite(as_float(row.get("tokens")) for row in rows)
    macros = {
        "overall_quality": mean(all_values) if all_values else None,
        "dynamic_quality": mean(dynamic_values) if dynamic_values else None,
        "dynamic_hv": mean(dm_hv) if dm_hv else None,
        "dynamic_igd": mean(dm_igd) if dm_igd else None,
        "token_mean": mean(token_values) if token_values else None,
    }
    return profile_rows, macros


def build_mo_variance_rows(rows: list[dict[str, str]], archive_counts: dict[tuple[str, int, int], int]) -> list[dict[str, Any]]:
    by_episode_seed: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        pid = problem_index(episode_id)
        if pid is None or pid <= 9:
            continue
        seed = int(float(row.get("run_seed") or 0))
        by_episode_seed[(episode_id, seed)].append(row)
        by_episode[episode_id].append(row)
    out: list[dict[str, Any]] = []
    for episode_id in sorted(by_episode, key=episode_sort_key):
        pid = problem_index(episode_id)
        if pid is None:
            continue
        seeds = sorted(seed for ep, seed in by_episode_seed if ep == episode_id)
        seed_hv: list[float] = []
        seed_igd: list[float] = []
        seed_front: list[float] = []
        for seed in seeds:
            seed_rows = by_episode_seed[(episode_id, seed)]
            hv = [quality(row, pid) for row in seed_rows]
            hv = [value for value in hv if value is not None and math.isfinite(value)]
            igd = [as_float(row.get("igd")) for row in seed_rows]
            igd = [value for value in igd if value is not None and math.isfinite(value)]
            counts = [
                archive_counts[(episode_id, seed, int(float(row.get("stage_index") or 0)))]
                for row in seed_rows
                if (episode_id, seed, int(float(row.get("stage_index") or 0))) in archive_counts
            ]
            if hv:
                seed_hv.append(mean(hv))
            if igd:
                seed_igd.append(mean(igd))
            if counts:
                seed_front.append(mean(counts))
        merged_hv_values = [quality(row, pid) for row in by_episode[episode_id]]
        merged_hv_values = [value for value in merged_hv_values if value is not None and math.isfinite(value)]
        out.append(
            {
                "problem": f"P{pid:03d}",
                "profile": PROFILE_BY_PID.get(pid, "NLDO-DM"),
                "seeds": len(seeds),
                "steps_per_seed": max((len(by_episode_seed[(episode_id, seed)]) for seed in seeds), default=0),
                "merged_hv": mean(merged_hv_values) if merged_hv_values else None,
                "seed_hv_mean": mean(seed_hv) if seed_hv else None,
                "seed_hv_std": pstdev(seed_hv) if len(seed_hv) > 1 else 0.0 if seed_hv else None,
                "seed_hv_var": variance(seed_hv),
                "seed_igd_mean": mean(seed_igd) if seed_igd else None,
                "seed_igd_std": pstdev(seed_igd) if len(seed_igd) > 1 else 0.0 if seed_igd else None,
                "avg_front": mean(seed_front) if seed_front else None,
            }
        )
    return out


def archive_counts_by_seed_stage(path: Path) -> dict[tuple[str, int, int], int]:
    out: dict[tuple[str, int, int], int] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id") or "")
            seed = int(row.get("run_seed") or 0)
            out[(episode_id, seed, 0)] = candidate_count(row.get("initial_solver_result"))
            for idx, update in enumerate(row.get("update_results") or [], start=1):
                out[(episode_id, seed, idx)] = candidate_count(update.get("solver_result"))
    return out


def candidate_count(solver_result: Any) -> int:
    if not isinstance(solver_result, dict):
        return 0
    metadata = solver_result.get("metadata") if isinstance(solver_result.get("metadata"), dict) else {}
    archive = metadata.get("candidate_archive")
    if isinstance(archive, list):
        return len(archive)
    archive = solver_result.get("candidate_archive")
    if isinstance(archive, list):
        return len(archive)
    return 1 if solver_result.get("solution") else 0


def source_note(paths: list[Path] | Path) -> str:
    if isinstance(paths, Path):
        paths = [paths]
    return "; ".join(str(path) for path in paths) if paths else "none"


def write_episode_tex(path: Path, rows: list[dict[str, Any]], sources: list[Path]) -> None:
    lines = [
        "% Auto-generated by scripts/experiments/export_liveopt_current_evidence_tables.py.",
        f"% Sources: {source_note(sources)}",
        r"\newcommand{\LiveOptCurrentEpisodeRows}{%",
    ]
    for row in rows:
        gap = row["scalar_gap"] if row["scalar_gap"] is not None else row["ideal_gap"]
        lines.append(
            f"{row['problem']} & {escape_tex(row['profile'])} & {fmt(row['mean'])} & "
            f"{fmt(gap, digits=3)} & {fmt(row['hv'])} & {fmt(row['igd'])} \\\\"
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mo_tex(path: Path, rows: list[dict[str, Any]], metric_sources: list[Path], replay_sources: list[Path]) -> None:
    provenance = f"% Source metrics: {source_note(metric_sources)}"
    if replay_sources:
        provenance += f" ; archive counts: {source_note(replay_sources)}"
    lines = [
        "% Auto-generated by scripts/experiments/export_liveopt_current_evidence_tables.py.",
        provenance,
        r"\newcommand{\LiveOptMOHVVarianceRows}{%",
    ]
    for row in rows:
        lines.append(
            f"{row['problem']} & {escape_tex(row['profile'])} & {fmt(row['merged_hv'])} & "
            f"{fmt_pm(row['seed_hv_mean'], row['seed_hv_std'])} & "
            f"{fmt(row['seed_hv_var'], digits=5)} & {fmt_pm(row['seed_igd_mean'], row['seed_igd_std'])} \\\\"
        )
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profile_tex(path: Path, rows: list[dict[str, Any]], macros: dict[str, float | None], sources: list[Path]) -> None:
    lines = [
        "% Auto-generated by scripts/experiments/export_liveopt_current_evidence_tables.py.",
        f"% Sources: {source_note(sources)}",
        r"\newcommand{\FormalBenchmarkResultRows}{%",
    ]
    for index, row in enumerate(rows):
        lines.append(
            f"{escape_tex(row['profile'])} & \\methodours & {fmt(row['overall_mean'])} & "
            f"{fmt(row['dynamic_mean'])} & {fmt(row['gap_or_hv'])} & {fmt(row['igd'])} \\\\"
        )
        if index + 1 < len(rows):
            lines.append(r"\midrule")
    lines.extend(
        [
            "}",
            rf"\newcommand{{\LiveOptCurrentOverallQuality}}{{{fmt(macros['overall_quality'])}}}",
            rf"\newcommand{{\LiveOptCurrentDynamicQuality}}{{{fmt(macros['dynamic_quality'])}}}",
            rf"\newcommand{{\LiveOptCurrentHV}}{{{fmt(macros['dynamic_hv'])}}}",
            rf"\newcommand{{\LiveOptCurrentIGD}}{{{fmt(macros['dynamic_igd'])}}}",
            rf"\newcommand{{\LiveOptCurrentTokenMean}}{{{fmt(macros['token_mean'], digits=0)}}}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quality(row: dict[str, str], pid: int) -> float | None:
    if pid >= 10:
        return first_float(row, "normalized_hv", "normalized_score", "normalized_total_score")
    return first_float(row, "normalized_score", "normalized_total_score")


def stage_range(rows: list[dict[str, str]]) -> str:
    stages = [int(float(row.get("stage_index") or 0)) for row in rows]
    if not stages:
        return "--"
    return f"t{min(stages):02d}--t{max(stages):02d}"


def problem_index(episode_id: str) -> int | None:
    match = re.search(r"P(\d+)", episode_id)
    return int(match.group(1)) if match else None


def episode_sort_key(episode_id: str) -> tuple[int, str]:
    return (problem_index(episode_id) or 999, episode_id)


def first_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def as_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def variance(values: list[float]) -> float | None:
    if not values:
        return None
    avg = mean(values)
    return mean([(value - avg) ** 2 for value in values])


def finite(values: Any) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_pm(mean_value: Any, std_value: Any) -> str:
    if mean_value is None:
        return "--"
    if std_value is None:
        return f"{float(mean_value):.3f}"
    return f"{float(mean_value):.3f}$\\pm${float(std_value):.3f}"


def escape_tex(value: str) -> str:
    return value.replace("_", r"\_")


if __name__ == "__main__":
    raise SystemExit(main())
