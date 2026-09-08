#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble disjoint routing/cloud shards for one matched method.")
    parser.add_argument("--source-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for run_dir in args.source_run_dir:
        path = run_dir / "NLDO" / "evo2_limit0.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        source_rows = load_jsonl(path)
        rows.extend(source_rows)
        sources.append({"run_dir": str(run_dir), "path": str(path), "sha256": sha256(path), "rows": len(source_rows)})
    keys = [(str(row.get("episode_id")), int(row.get("run_seed") or 0)) for row in rows]
    if len(keys) != 60 or len(set(keys)) != 60:
        raise ValueError(f"expected 60 unique P010--P015 seed rows, got {len(keys)} rows/{len(set(keys))} keys")
    episodes = sorted({episode for episode, _ in keys})
    expected = [f"NLDO-P{index:03d}" for index in range(10, 16)]
    if episodes != expected:
        raise ValueError(f"episode coverage mismatch: {episodes}")
    for row in rows:
        if len(row.get("update_results") or []) != 12:
            raise ValueError(f"{row.get('episode_id')} seed {row.get('run_seed')} does not contain 12 updates")
    rows.sort(key=lambda row: (str(row.get("episode_id")), int(row.get("run_seed") or 0)))
    output_path = args.output_dir / "NLDO" / "evo2_limit0.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "protocol": "disjoint_matched_routing_cloud_assembly_v1",
        "sources": sources,
        "output_jsonl": str(output_path),
        "output_sha256": sha256(output_path),
        "rows": len(rows),
        "episodes": episodes,
        "seeds": sorted({seed for _, seed in keys}),
    }
    (args.output_dir / "assembly_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
