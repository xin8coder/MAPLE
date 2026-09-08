#!/usr/bin/env python3
"""Synchronize the public preview with the current NLDO benchmark."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_EPISODE_FIELDS = (
    "agent_allowed_solvers",
    "base_benchmark",
    "base_instance_id",
    "domain",
    "episode_id",
    "family",
    "public_context",
    "public_initial_problem",
    "source_dataset",
    "source_instance_id",
)
PUBLIC_UPDATE_FIELDS = (
    "difficulty",
    "metadata",
    "public_update",
    "requires_memory",
    "time_index",
    "update_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=ROOT / "release_artifacts/public_preview",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/evo2_dynoptbench/public_csv/nldo_15episodes_12updates_csv.jsonl",
    )
    return parser.parse_args()


def public_table_path(episode_id: str, filename: str | None = None) -> str:
    root = Path("data") / "public_tables" / episode_id
    return (root / filename).as_posix() if filename else root.as_posix()


def resolve_source_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable_source_path(value: str | Path) -> str:
    path = resolve_source_path(value)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return Path(value).as_posix()


def public_episode(source: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(source["episode_id"])
    row = {field: copy.deepcopy(source.get(field)) for field in PUBLIC_EPISODE_FIELDS}
    row["schema_version"] = "liveopt_public_episode_stream_preview_v1"
    row["structured_data_path"] = public_table_path(episode_id)
    row["update_stream"] = [
        {field: copy.deepcopy(update.get(field)) for field in PUBLIC_UPDATE_FIELDS}
        for update in list(source.get("update_stream") or [])
    ]
    tables = row.get("public_context", {}).get("csv_tables", {})
    for table in tables.values():
        source_path = Path(str(table.get("path", "")))
        if not source_path.name:
            raise ValueError(f"{episode_id}: table path has no filename")
        table["allowed_root"] = public_table_path(episode_id)
        table["path"] = public_table_path(episode_id, source_path.name)
    return row


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def synchronize_tables(root: Path, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    table_root = root / "data/public_tables"
    table_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    expected_episode_dirs: set[str] = set()
    copied = 0
    for source in source_rows:
        episode_id = str(source["episode_id"])
        expected_episode_dirs.add(episode_id)
        target_dir = table_root / episode_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True)
        tables: list[dict[str, Any]] = []
        for name, descriptor in sorted(
            (source.get("public_context") or {}).get("csv_tables", {}).items()
        ):
            source_path = resolve_source_path(str(descriptor.get("path") or ""))
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)
            copied += 1
            tables.append(
                {
                    "table": str(name),
                    "file": (Path("data/public_tables") / episode_id / source_path.name).as_posix(),
                    "columns": list(descriptor.get("columns") or []),
                    "row_count": csv_row_count(source_path),
                }
            )
        index_rows.append(
            {
                "episode_id": episode_id,
                "source_path": portable_source_path(str(source.get("structured_data_path") or "")),
                "copied_dir": public_table_path(episode_id),
                "tables": tables,
            }
        )
    for child in sorted(table_root.iterdir()):
        if child.is_dir() and child.name not in expected_episode_dirs:
            shutil.rmtree(child)
    index = {
        "schema_version": "liveopt_public_table_index_v1",
        "episode_count": len(index_rows),
        "episodes": index_rows,
    }
    (table_root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"episodes": len(index_rows), "table_files": copied}


def write_stream_index(path: Path, rows: list[dict[str, Any]], benchmark: Path) -> None:
    index = {
        "schema_version": "liveopt_public_stream_index_v1",
        "source": portable_source_path(benchmark),
        "source_sha256": sha256(benchmark),
        "episode_count": len(rows),
        "episodes": [
            {
                "episode_id": row["episode_id"],
                "base_benchmark": row.get("base_benchmark"),
                "domain": row.get("domain"),
                "family": row.get("family"),
                "update_count": len(row.get("update_stream") or []),
            }
            for row in rows
        ],
    }
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_checksums(root: Path) -> None:
    manifest = root / "checksums" / "manifest.sha256"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.preview_root.resolve()
    stream_path = root / "data" / "public_streams" / "nldo_public_streams.jsonl"
    source_rows = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_rows) != 15:
        raise ValueError(f"expected 15 benchmark episodes, found {len(source_rows)}")
    rows = [public_episode(source) for source in source_rows]
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    table_summary = synchronize_tables(root, source_rows)
    for row in rows:
        for table in row.get("public_context", {}).get("csv_tables", {}).values():
            target = root / str(table["path"])
            if not target.is_file():
                raise FileNotFoundError(target)
    stream_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_stream_index(stream_path.with_name("index.json"), rows, args.benchmark)
    refresh_checksums(root)
    print(
        json.dumps(
            {
                "status": "passed",
                "benchmark_sha256": sha256(args.benchmark),
                "episodes": len(rows),
                **table_summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
