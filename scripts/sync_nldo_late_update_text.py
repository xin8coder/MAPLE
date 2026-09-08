#!/usr/bin/env python3
"""Sync neutral P007--P015 public update text into the merged NLDO stream.

The merged release remaps the first three episodes from each realistic source
family to NLDO-P007--P015.  Numerical states, public patches, hidden oracles,
and release identifiers remain untouched; only the user-visible business-event
prefix is synchronized from the source materializers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORING_MARKER = "\n\nScoring after this update:"
SOURCE_GROUPS = (
    (7, Path("data/evo2_dynoptbench/public_csv/inrc_realistic_dynamic_6episodes_csv.jsonl")),
    (10, Path("data/evo2_dynoptbench/public_csv/green_vrp_mo_6episodes_csv.jsonl")),
    (13, Path("data/evo2_dynoptbench/public_csv/cloud_scheduling_mo_6episodes_csv.jsonl")),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _business_text(value: Any) -> str:
    return str(value or "").split(SCORING_MARKER, 1)[0].strip()


def _replace_business_text(value: Any, business: str) -> str:
    text = str(value or "")
    if SCORING_MARKER not in text:
        return business
    return business + SCORING_MARKER + text.split(SCORING_MARKER, 1)[1]


def sync(target: Path) -> dict[str, int]:
    episodes = _read_jsonl(target)
    by_id = {str(episode.get("episode_id")): episode for episode in episodes}
    updated_episodes = 0
    updated_stages = 0
    for start_index, source_path in SOURCE_GROUPS:
        source_episodes = _read_jsonl(source_path)[:3]
        for offset, source_episode in enumerate(source_episodes):
            target_id = f"NLDO-P{start_index + offset:03d}"
            target_episode = by_id[target_id]
            source_updates = list(source_episode.get("update_stream") or [])
            target_updates = list(target_episode.get("update_stream") or [])
            if len(source_updates) != len(target_updates):
                raise ValueError(f"update-count mismatch for {target_id}")
            for source_update, target_update in zip(source_updates, target_updates):
                business = _business_text(
                    source_update.get("natural_language_update") or source_update.get("public_update")
                )
                for field in ("public_update", "natural_language_update"):
                    if field in target_update:
                        target_update[field] = _replace_business_text(target_update[field], business)
                updated_stages += 1
            updated_episodes += 1
    target.write_text(
        "\n".join(json.dumps(episode, ensure_ascii=False, separators=(",", ":")) for episode in episodes) + "\n",
        encoding="utf-8",
    )
    return {"episodes": updated_episodes, "stages": updated_stages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps({"target": str(args.target), **sync(args.target)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
