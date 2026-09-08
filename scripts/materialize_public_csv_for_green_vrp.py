#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evo2.benchmarks.optimization_contracts import (
    append_public_objective_spec_to_text,
    append_public_update_objective_notes_to_episode,
    attach_public_optimization_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize public CSV inputs for Green VRP episodes.")
    parser.add_argument("--episodes", default="data/evo2_dynoptbench/compressed_plan/green_vrp_mo_6episodes.jsonl")
    parser.add_argument("--out-root", default="data/evo2_dynoptbench/public_csv/green_vrp_mo")
    parser.add_argument("--out-episodes", default="data/evo2_dynoptbench/public_csv/green_vrp_mo_6episodes_csv.jsonl")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_episodes = Path(args.out_episodes)
    out_episodes.parent.mkdir(parents=True, exist_ok=True)

    converted = []
    for line in Path(args.episodes).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        state = episode.get("hidden_initial_state") or {}
        episode_dir = out_root / episode["episode_id"]
        episode_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(episode_dir / "depot.csv", ["id", "x", "y"], [{"id": "DEPOT", **(state.get("depot") or {"x": 0.0, "y": 0.0})}])
        _write_csv(
            episode_dir / "vehicles.csv",
            ["id", "capacity", "emission_rate", "available", "shift_end"],
            state.get("vehicles", []),
        )
        _write_csv(
            episode_dir / "orders.csv",
            ["id", "x", "y", "demand", "ready", "due", "service", "priority", "active"],
            state.get("orders", []),
        )
        _write_csv(
            episode_dir / "policy.csv",
            ["carbon_multiplier"],
            [{"carbon_multiplier": state.get("carbon_multiplier", 1.0)}],
        )
        csv_tables = {
            "depot": _table_spec(episode_dir / "depot.csv", episode_dir, ["id", "x", "y"]),
            "vehicles": _table_spec(episode_dir / "vehicles.csv", episode_dir, ["id", "capacity", "emission_rate", "available", "shift_end"]),
            "orders": _table_spec(episode_dir / "orders.csv", episode_dir, ["id", "x", "y", "demand", "ready", "due", "service", "priority", "active"]),
            "policy": _table_spec(episode_dir / "policy.csv", episode_dir, ["carbon_multiplier"]),
        }
        public_context = dict(episode.get("public_context") or {})
        public_context.update(
            {
                "input_policy": "agent sees natural-language objective plus public CSV schema/paths; hidden reference remains evaluation-only",
                "csv_tables": csv_tables,
                "csv_schema": {name: spec["columns"] for name, spec in csv_tables.items()},
            }
        )
        episode["public_context"] = public_context
        episode["structured_data_path"] = _portable_path(episode_dir)
        episode["public_initial_problem"] = (
            "Plan same-day delivery routes from a single depot for a small urban fleet. "
            "Initial depot, vehicle, and order data are provided in public CSV tables. "
            "Use the attached depot, vehicle, order, and policy tables as the source of truth. "
            "All active orders should be served once under vehicle capacity. "
            "Rows with active=false are public candidate orders that are not served until a later natural-language update explicitly activates them. "
            "Later updates may replace public order geometry, demand, time windows, priorities, vehicle emissions, depot coordinates, or carbon policy; all current rows remain public. "
            "Optimize three main criteria: travel distance, delivery lateness, and emissions using public policy parameters when present. "
            "Also report route disruption and priority-customer penalties as auxiliary diagnostics."
        )
        attach_public_optimization_contract(episode, "green_vrp_mo")
        episode["public_initial_problem"] = append_public_objective_spec_to_text(
            str(episode.get("public_initial_problem") or ""),
            episode["public_context"].get("optimization_contract"),
        )
        append_public_update_objective_notes_to_episode(episode)
        converted.append(episode)

    with out_episodes.open("w", encoding="utf-8") as handle:
        for episode in converted:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": len(converted), "out_episodes": str(out_episodes), "out_root": str(out_root)}, ensure_ascii=False, indent=2))


def _table_spec(path: Path, root: Path, columns: list[str]) -> dict[str, Any]:
    return {"path": _portable_path(path), "allowed_root": _portable_path(root), "columns": columns}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


if __name__ == "__main__":
    main()
