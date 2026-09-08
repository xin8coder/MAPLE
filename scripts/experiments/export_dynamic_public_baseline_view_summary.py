#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


VIEW_ORDER = ["NLDO (SS)", "NLDO (SM)", "NLDO (DS)", "NLDO (DM)"]


def main() -> int:
    args = parse_args()
    episodes = select_episodes(read_jsonl(Path(args.episodes_jsonl)), args.episode_id)
    rows = load_stage_rows([Path(item) for item in args.run_dir])
    methods = requested_methods(args.methods, rows)
    summary = {
        method: summarize_method(method, episodes, rows)
        for method in methods
    }
    payload = {
        "episodes_jsonl": args.episodes_jsonl,
        "episode_ids": [str(episode.get("episode_id") or "") for episode in episodes],
        "run_dirs": args.run_dir,
        "missing_stage_policy": "score_zero",
        "views": VIEW_ORDER,
        "methods": summary,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_tex:
        write_tex(Path(args.out_tex), summary)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate dynamic public baseline stage rows into the four NLDO paper views. "
            "Unreached stages are counted as zero so stopped sequences remain in the denominator."
        )
    )
    parser.add_argument("--episodes-jsonl", required=True)
    parser.add_argument(
        "--episode-id",
        action="append",
        help="Restrict the denominator to these episode ids; repeat for multiple episodes.",
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--methods", default="", help="Optional comma-separated method ids; otherwise infer from rows.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-tex")
    return parser.parse_args()


def select_episodes(episodes: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return episodes
    wanted = {str(item) for item in requested}
    available = {str(episode.get("episode_id") or "") for episode in episodes}
    unknown = sorted(wanted - available)
    if unknown:
        raise SystemExit(f"Unknown --episode-id value(s): {', '.join(unknown)}")
    return [episode for episode in episodes if str(episode.get("episode_id") or "") in wanted]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_stage_rows(run_dirs: list[Path]) -> dict[tuple[str, str, int], dict[str, Any]]:
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run_dir in run_dirs:
        stage_path = run_dir / "dynamic_public_baseline_stage_rows.jsonl"
        if not stage_path.exists():
            continue
        for row in read_jsonl(stage_path):
            method = str(row.get("method") or "")
            episode_id = str(row.get("episode_id") or "")
            try:
                stage_index = int(row.get("stage_index"))
            except (TypeError, ValueError):
                continue
            if method and episode_id:
                latest[(method, episode_id, stage_index)] = row
    return latest


def requested_methods(methods_arg: str, rows: dict[tuple[str, str, int], dict[str, Any]]) -> list[str]:
    if methods_arg.strip():
        return [item.strip() for item in methods_arg.split(",") if item.strip()]
    return sorted({method for method, _, _ in rows})


def summarize_method(
    method: str,
    episodes: list[dict[str, Any]],
    rows: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        stage_count = 1 + len(episode.get("update_stream") or [])
        for stage_index in range(stage_count):
            view = view_for(episode_id, stage_index)
            if not view:
                continue
            row = rows.get((method, episode_id, stage_index))
            buckets[view].append(stage_record(row))
    return {
        view: summarize_bucket(buckets.get(view, []))
        for view in VIEW_ORDER
    }


def view_for(episode_id: str, stage_index: int) -> str | None:
    pid = problem_index(episode_id)
    if pid is None:
        return None
    scalar = pid <= 9
    if scalar and stage_index == 0:
        return "NLDO (SS)"
    if not scalar and stage_index == 0:
        return "NLDO (SM)"
    if scalar and stage_index > 0:
        return "NLDO (DS)"
    if not scalar and stage_index > 0:
        return "NLDO (DM)"
    return None


def problem_index(episode_id: str) -> int | None:
    match = re.search(r"P(\d+)", episode_id)
    if not match:
        return None
    return int(match.group(1))


def stage_record(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"reached": False, "feasible": False, "score": 0.0, "tokens": 0}
    hidden = row.get("hidden_evaluation") or {}
    score = as_float(hidden.get("normalized_score"))
    token_usage = row.get("token_usage") or {}
    tokens = int(token_usage.get("total_tokens") or 0)
    return {
        "reached": True,
        "feasible": bool(hidden.get("feasible")),
        "score": bounded(score),
        "raw_score": score,
        "tokens": tokens,
        "status": row.get("status"),
        "failure_reason": row.get("failure_reason"),
    }


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bounded(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def summarize_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["score"]) for item in records]
    return {
        "stages": len(records),
        "reached": sum(1 for item in records if item.get("reached")),
        "feasible": sum(1 for item in records if item.get("feasible")),
        "quality_mean": mean(scores) if scores else None,
        "quality_std": pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
        "display_0_100": 100.0 * mean(scores) if scores else None,
        "feasible_fraction": fraction(sum(1 for item in records if item.get("feasible")), len(records)),
        "reached_fraction": fraction(sum(1 for item in records if item.get("reached")), len(records)),
        "total_tokens": sum(int(item.get("tokens") or 0) for item in records),
    }


def fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "--"


def write_tex(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by scripts/experiments/export_dynamic_public_baseline_view_summary.py",
        "% Missing/unreached stages are counted as zero in the denominator.",
    ]
    for method, views in sorted(summary.items()):
        values = [format_value(views[view].get("display_0_100")) for view in VIEW_ORDER]
        reached = ", ".join(f"{view}: {views[view].get('reached_fraction')}" for view in VIEW_ORDER)
        lines.append(f"{escape_tex(method)} & {' & '.join(values)} & {reached} \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1f}"


def escape_tex(value: str) -> str:
    return value.replace("_", r"\_")


if __name__ == "__main__":
    raise SystemExit(main())
