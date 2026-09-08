#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    args = parse_args()
    base_run = Path(args.base_run_dir)
    out_run = Path(args.out_run_dir)
    out_jsonl = out_run / "NLDO" / "evo2_limit0.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    base_rows = load_rows(base_run)
    base_order = episode_order(base_rows)
    rows_by_episode = group_by_episode(base_rows, duplicate_seed_policy=args.duplicate_seed_policy)
    sources: dict[str, dict[str, Any]] = {}

    for item in args.replacement:
        run_dir, episode_ids = parse_replacement(item)
        replacement_rows = group_by_episode(
            load_rows(run_dir),
            duplicate_seed_policy=args.duplicate_seed_policy,
        )
        for episode_id in episode_ids:
            rows = replacement_rows.get(episode_id)
            if not rows:
                raise SystemExit(f"{run_dir} has no rows for {episode_id}")
            rows_by_episode[episode_id] = rows
            sources[episode_id] = {
                "source_run_dir": str(run_dir),
                "row_count": len(rows),
                "seeds": sorted({row.get("run_seed") for row in rows}),
            }
            if episode_id not in base_order:
                base_order.append(episode_id)

    with out_jsonl.open("w", encoding="utf-8") as handle:
        for episode_id in base_order:
            for row in sorted(rows_by_episode[episode_id], key=seed_key):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "base_run_dir": str(base_run),
        "out_run_dir": str(out_run),
        "output_jsonl": str(out_jsonl),
        "episode_order": base_order,
        "replacement_sources": sources,
        "duplicate_seed_policy": args.duplicate_seed_policy,
        "total_rows": sum(len(rows_by_episode[episode_id]) for episode_id in base_order),
    }
    (out_run / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge NLDO run JSONL rows by replacing complete episode row sets.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--out-run-dir", required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        required=True,
        help="Replacement in run_dir:EPISODE[,EPISODE...] form. Each episode is replaced as a complete seed row set.",
    )
    parser.add_argument(
        "--duplicate-seed-policy",
        choices=("error", "first", "last"),
        default="error",
        help=(
            "How to handle duplicate episode+seed rows inside an input run. "
            "The default rejects ambiguous provenance; use 'last' only for an audited append-only retry."
        ),
    )
    return parser.parse_args()


def parse_replacement(value: str) -> tuple[Path, list[str]]:
    if ":" not in value:
        raise SystemExit(f"replacement must be run_dir:EPISODE[,EPISODE...], got {value}")
    run_dir, episodes = value.split(":", 1)
    episode_ids = [item.strip() for item in episodes.split(",") if item.strip()]
    if not episode_ids:
        raise SystemExit(f"replacement has no episode ids: {value}")
    return Path(run_dir), episode_ids


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "NLDO" / "evo2_limit0.jsonl"
    if not path.exists():
        raise SystemExit(f"missing NLDO replay JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def episode_order(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if episode_id and episode_id not in seen:
            out.append(episode_id)
            seen.add(episode_id)
    return out


def group_by_episode(
    rows: list[dict[str, Any]],
    *,
    duplicate_seed_policy: str = "error",
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            continue
        seed = str(row.get("run_seed"))
        if seed in grouped[episode_id]:
            if duplicate_seed_policy == "error":
                raise SystemExit(
                    f"duplicate row for {episode_id} seed={seed}; "
                    "audit the retry and pass --duplicate-seed-policy first|last explicitly"
                )
            if duplicate_seed_policy == "first":
                continue
        grouped[episode_id][seed] = row
    return {
        episode_id: list(rows_by_seed.values())
        for episode_id, rows_by_seed in grouped.items()
    }


def seed_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        return (int(row.get("run_seed")), "")
    except Exception:
        return (999999, str(row.get("run_seed")))


if __name__ == "__main__":
    raise SystemExit(main())
