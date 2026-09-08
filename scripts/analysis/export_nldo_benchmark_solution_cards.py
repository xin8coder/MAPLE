#!/usr/bin/env python3
"""Render the five benchmark-card solutions from current LiveOpt artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


NAVY = "#12335D"
TEAL = "#2A9D9F"
ORANGE = "#E9B44C"
LIGHT = "#DCE7EC"
GRID = "#D7E2E8"
INK = "#24343D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("release_artifacts/paper_figure_exports/benchmark_solutions"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "p001": load_row(
            Path("outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713/NLDO/evo2_limit0.jsonl"),
            "NLDO-P001",
        ),
        "p004": load_row(
            Path("outputs/liveopt_objective_shift_p001_p009_full_200x200_10seed_20260713/NLDO/evo2_limit0.jsonl"),
            "NLDO-P004",
        ),
        "p007": load_row(
            Path("outputs/liveopt_ablation_llmbackbone_sequential_p007_p009_200x200x10_trace_t11pop_20260714/NLDO/evo2_limit0.jsonl"),
            "NLDO-P007",
        ),
        "p010": load_row(
            Path("outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/NLDO/evo2_limit0.jsonl"),
            "NLDO-P010",
        ),
        "p013": load_row(
            Path("outputs/liveopt_matched_full_p010_p015_200x200x10_20260714/NLDO/evo2_limit0.jsonl"),
            "NLDO-P013",
        ),
    }
    render_selection(rows["p001"], args.output_dir / "cobench_initial_solution.pdf")
    render_schedule(rows["p004"], args.output_dir / "fjsp_initial_solution.pdf")
    render_roster(rows["p007"], args.output_dir / "inrc_initial_solution.pdf")
    render_routes(rows["p010"], args.output_dir / "green_vrp_initial_solution.pdf")
    render_placement(rows["p013"], args.output_dir / "cloud_initial_solution.pdf")
    return 0


def load_row(path: Path, episode_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("episode_id") == episode_id and int(row.get("run_seed", -1)) == 0:
                return row
    raise ValueError(f"missing seed-0 row for {episode_id} in {path}")


def solution(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row["initial_solver_result"]["metadata"]["agent_objective_raw"]["solution"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def finish(fig: plt.Figure, ax: plt.Axes, output: Path, *, grid_axis: str | None = "y") -> None:
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#9EB2BD")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=INK, labelsize=6, width=0.6, length=2.5)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def render_selection(row: dict[str, Any], output: Path) -> None:
    records = read_csv(
        Path("data/evo2_dynoptbench/public_csv/cobench_exact_small/cobench_exact_small_knapsack_000/items.csv")
    )
    selected = set(solution(row)["selected_items"])
    labels = [record["id"] for record in records]
    values = [float(record["value"]) for record in records]
    fig, ax = new_figure()
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        color=[TEAL if label in selected else LIGHT for label in labels],
        edgecolor=[NAVY if label in selected else "#AFC0C8" for label in labels],
        linewidth=0.55,
        width=0.72,
    )
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("value", fontsize=6, color=INK, labelpad=1)
    ax.set_ylim(0, max(values) * 1.12)
    finish(fig, ax, output)


def render_schedule(row: dict[str, Any], output: Path) -> None:
    schedule = list(solution(row)["schedule"])
    machines = sorted({str(item["machine_id"]) for item in schedule})
    y = {machine: index for index, machine in enumerate(machines)}
    job_colors = {"J1": NAVY, "J2": ORANGE, "J3": TEAL}
    fig, ax = new_figure()
    for item in schedule:
        operation = str(item["operation_id"])
        start = float(item["start_time"])
        end = float(item["end_time"])
        machine = str(item["machine_id"])
        ax.barh(
            y[machine],
            end - start,
            left=start,
            height=0.58,
            color=job_colors.get(operation[:2], TEAL),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_yticks(range(len(machines)), machines)
    ax.set_xlabel("time", fontsize=6, color=INK, labelpad=1)
    ax.set_xlim(0, max(float(item["end_time"]) for item in schedule) * 1.03)
    finish(fig, ax, output, grid_axis="x")


def render_roster(row: dict[str, Any], output: Path) -> None:
    roster = dict(solution(row)["roster"])
    nurses = sorted({key.split("|")[0] for key in roster})
    days = [f"D{index}" for index in range(1, 8)]
    code = {"off": 0, "day": 1, "evening": 2, "night": 3}
    matrix = np.zeros((len(nurses), len(days)), dtype=int)
    for nurse_index, nurse in enumerate(nurses):
        for day_index, day in enumerate(days):
            matrix[nurse_index, day_index] = code.get(str(roster.get(f"{nurse}|{day}", "off")), 0)
    fig, ax = new_figure()
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap([LIGHT, NAVY, TEAL, ORANGE]), vmin=0, vmax=3)
    ax.set_xticks(range(len(days)), days)
    ax.set_yticks([0, len(nurses) - 1], [nurses[0], nurses[-1]])
    finish(fig, ax, output, grid_axis=None)


def render_routes(row: dict[str, Any], output: Path) -> None:
    order_rows = read_csv(Path("data/evo2_dynoptbench/public_csv/green_vrp_mo/green_vrp_mo_000/orders.csv"))
    coords = {record["id"]: (float(record["x"]), float(record["y"])) for record in order_rows}
    routes = list(solution(row)["routes"])
    fig, ax = new_figure()
    for route, color in zip(routes, (NAVY, ORANGE, TEAL)):
        points = [(0.0, 0.0)] + [coords[order] for order in route["orders"]] + [(0.0, 0.0)]
        ax.plot([point[0] for point in points], [point[1] for point in points], color=color, linewidth=1.35, marker="o", markersize=2.5)
    ax.scatter([0.0], [0.0], marker="s", s=18, color=INK, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.04)
    finish(fig, ax, output, grid_axis=None)


def render_placement(row: dict[str, Any], output: Path) -> None:
    assignments = dict(solution(row)["assignments"])
    jobs = {record["id"]: record for record in read_csv(Path("data/evo2_dynoptbench/public_csv/cloud_scheduling_mo/cloud_scheduling_mo_000/jobs.csv"))}
    machines = {record["id"]: record for record in read_csv(Path("data/evo2_dynoptbench/public_csv/cloud_scheduling_mo/cloud_scheduling_mo_000/machines.csv"))}
    used_cpu: defaultdict[str, float] = defaultdict(float)
    used_mem: defaultdict[str, float] = defaultdict(float)
    for job, machine in assignments.items():
        used_cpu[machine] += float(jobs[job]["cpu"])
        used_mem[machine] += float(jobs[job]["mem"])
    labels = sorted(machines)
    cpu = [used_cpu[label] / float(machines[label]["cpu"]) for label in labels]
    mem = [used_mem[label] / float(machines[label]["mem"]) for label in labels]
    fig, ax = new_figure()
    x = np.arange(len(labels))
    ax.bar(x - 0.18, cpu, width=0.36, color=NAVY, label="CPU")
    ax.bar(x + 0.18, mem, width=0.36, color=TEAL, label="MEM")
    ax.axhline(1.0, color=ORANGE, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("load", fontsize=6, color=INK, labelpad=1)
    finish(fig, ax, output)


if __name__ == "__main__":
    raise SystemExit(main())
