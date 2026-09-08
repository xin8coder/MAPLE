#!/usr/bin/env python3
"""Export one compact 3-by-12 stage atlas per NLDO episode.

Each column is one public update (t01--t12).  The three rows show the pooled
Pareto projection, the accepted seed-0 plan, and the evaluator-side
per-generation normalized-HV trace (mean +/- population standard deviation).
No problem or update text is rendered.  Optimization and hidden evaluation
remain in the replay/evaluator code paths; this script only reads their output.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.analysis.evaluate_reference_metrics import (
    DEFAULT_NLDO_STRONG_REFERENCE,
    apply_hidden_delta,
    effective_benchmark,
    reference_trajectory_for_payload,
    score_solution,
)


TEAL = "#12877d"
TEAL_LIGHT = "#5dcfc4"
INK = "#27364a"
GREY = "#2f3747"
GRID = "#d9dee8"
ORANGE = "#f5a623"
BLUE = "#3575e6"
RED = "#df3d3d"
GREEN = "#2e8b57"
PURPLE = "#7b61ff"


def open_text(path: Path) -> TextIO:
    """Open a plain or gzip-compressed JSONL file as UTF-8 text."""

    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def load_reference_payload(reference_path: Path, episode_id: str) -> dict[str, Any]:
    for row in iter_jsonl(reference_path):
        if row.get("episode_id") == episode_id:
            return row
    raise ValueError(f"episode_id={episode_id!r} not found in {reference_path}")


def load_run_rows(run_jsonl: Path, episode_id: str) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(run_jsonl) if row.get("episode_id") == episode_id]
    if not rows:
        raise ValueError(f"episode_id={episode_id!r} not found in {run_jsonl}")
    rows.sort(key=lambda row: int(row.get("run_seed") or 0))
    return rows


def stage_solver_result(row: dict[str, Any], stage_index: int) -> dict[str, Any]:
    if stage_index == 0:
        return row.get("initial_solver_result") or {}
    updates = row.get("update_results") or []
    if stage_index - 1 >= len(updates):
        return {}
    return updates[stage_index - 1].get("solver_result") or {}


def stage_solution(row: dict[str, Any], stage_index: int) -> dict[str, Any]:
    if stage_index == 0:
        return row.get("initial_candidate") or {}
    updates = row.get("update_results") or []
    if stage_index - 1 >= len(updates):
        return {}
    return updates[stage_index - 1].get("solution") or {}


def build_stage_states(payload: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = effective_benchmark("NLDO", payload)
    domain = str(payload.get("domain") or "")
    state = deepcopy(payload.get("hidden_initial_state") or {})
    states = [deepcopy(state)]
    for update in payload.get("hidden_update_oracle") or []:
        delta = update.get("hidden_delta") if isinstance(update, dict) else None
        if isinstance(delta, dict):
            state = apply_hidden_delta(benchmark, domain, state, delta)
        states.append(deepcopy(state))
    return states


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nondominated_indices(points: np.ndarray) -> list[int]:
    """Return minimization-nondominated indices across every objective."""

    keep: list[int] = []
    for idx, point in enumerate(points):
        dominated = np.any(np.all(points <= point, axis=1) & np.any(points < point, axis=1))
        if not dominated:
            keep.append(idx)
    return keep


def collect_archive_points(
    rows: list[dict[str, Any]],
    stage_index: int,
    *,
    scorer_benchmark: str,
    domain: str,
    family: str,
    state: dict[str, Any],
    reference_step: dict[str, Any],
) -> list[tuple[float, float]]:
    """Pool all seeds, filter in the full objective space, then project to 2-D."""

    objective_names = [str(name) for name in (reference_step.get("objectives") or [])]
    objective_vectors: list[tuple[float, ...]] = []
    for row in rows:
        solver_result = stage_solver_result(row, stage_index)
        archive = ((solver_result.get("metadata") or {}).get("candidate_archive") or [])
        metric = score_solution(
            scorer_benchmark,
            domain,
            family,
            state,
            stage_solution(row, stage_index),
            None,
            reference_step,
            archive,
        )
        hidden_archive = metric.get("hidden_candidate_archive") if isinstance(metric, dict) else None
        for item in hidden_archive or []:
            objectives = item.get("objectives") if isinstance(item, dict) else None
            if not isinstance(objectives, dict):
                continue
            names = objective_names or list(objectives)
            try:
                vector = tuple(float(objectives[name]) for name in names)
            except (KeyError, TypeError, ValueError):
                continue
            if len(vector) >= 2 and all(math.isfinite(value) for value in vector):
                objective_vectors.append(vector)
    if not objective_vectors:
        return []
    points = np.asarray(objective_vectors, dtype=float)
    projected = {(float(points[idx, 0]), float(points[idx, 1])) for idx in nondominated_indices(points)}
    return sorted(projected)


def collect_reference_points(
    reference_step: dict[str, Any], x_key: str, y_key: str
) -> list[tuple[float, float]]:
    points: set[tuple[float, float]] = set()
    for item in reference_step.get("pareto_archive") or []:
        objectives = item.get("objectives") if isinstance(item, dict) else None
        if not isinstance(objectives, dict):
            continue
        x = finite_float(objectives.get(x_key))
        y = finite_float(objectives.get(y_key))
        if x is not None and y is not None:
            points.add((x, y))
    return sorted(points)


def projection_bounds(
    reference_by_stage: dict[int, list[tuple[float, float]]],
    live_by_stage: dict[int, list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    """One coordinate transform shared by all 12 stages of an episode."""

    points = [point for stage_points in reference_by_stage.values() for point in stage_points]
    points.extend(point for stage_points in live_by_stage.values() for point in stage_points)
    if not points:
        return (0.0, 1.0, 0.0, 1.0)
    xs, ys = zip(*points)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    return (xmin, xmax, ymin, ymax)


def normalize_points(
    points: list[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> list[tuple[float, float]]:
    xmin, xmax, ymin, ymax = bounds
    return [((x - xmin) / (xmax - xmin), (y - ymin) / (ymax - ymin)) for x, y in points]


def draw_front(
    ax: plt.Axes,
    reference_points: list[tuple[float, float]],
    live_points: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> None:
    reference = normalize_points(reference_points, bounds)
    live = normalize_points(live_points, bounds)
    if reference:
        ax.scatter(
            [point[0] for point in reference],
            [point[1] for point in reference],
            s=5.2,
            c=GREY,
            alpha=0.50,
            linewidths=0,
        )
    if live:
        ax.scatter(
            [point[0] for point in live],
            [point[1] for point in live],
            s=5.8,
            facecolors="none",
            edgecolors=TEAL,
            linewidths=0.42,
            alpha=0.80,
        )
    ax.text(
        0.03,
        0.95,
        f"L{len(live)}/R{len(reference)}",
        transform=ax.transAxes,
        va="top",
        fontsize=5.0,
        color="#536174",
    )
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(-0.035, 1.035)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([0.0, 1.0])
    ax.grid(True, color=GRID, linewidth=0.32, alpha=0.58)


def route_coordinate_bounds(states: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for state in states:
        depot = state.get("depot") or {}
        points = [depot, *(state.get("orders") or [])]
        for point in points:
            x = finite_float(point.get("x")) if isinstance(point, dict) else None
            y = finite_float(point.get("y")) if isinstance(point, dict) else None
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return (-1.0, 1.0, -1.0, 1.0)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.04, 0.5)
    ypad = max((ymax - ymin) * 0.04, 0.5)
    return (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)


def draw_green_route(
    ax: plt.Axes,
    state: dict[str, Any],
    solution: dict[str, Any],
    coordinate_bounds: tuple[float, float, float, float],
) -> None:
    depot = state.get("depot") or {"x": 0.0, "y": 0.0}
    orders = {str(order.get("id")): order for order in state.get("orders") or []}
    ax.scatter(
        [float(order.get("x", 0.0)) for order in orders.values()],
        [float(order.get("y", 0.0)) for order in orders.values()],
        s=3.0,
        c="#c9d1dc",
        linewidths=0,
        zorder=1,
    )
    ax.scatter(
        [float(depot.get("x", 0.0))],
        [float(depot.get("y", 0.0))],
        s=13,
        c="#0f172a",
        marker="s",
        linewidths=0,
        zorder=4,
    )
    colors = [TEAL, BLUE, ORANGE, RED, PURPLE, GREEN]
    routes = list(solution.get("routes") or [])
    for idx, route in enumerate(routes):
        xs = [float(depot.get("x", 0.0))]
        ys = [float(depot.get("y", 0.0))]
        for order_id in route.get("orders") or []:
            order = orders.get(str(order_id))
            if order:
                xs.append(float(order.get("x", 0.0)))
                ys.append(float(order.get("y", 0.0)))
        xs.append(float(depot.get("x", 0.0)))
        ys.append(float(depot.get("y", 0.0)))
        ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=0.48, alpha=0.88)
    ax.text(0.03, 0.95, f"{len(routes)} routes", transform=ax.transAxes, va="top", fontsize=5.0, color="#536174")
    xmin, xmax, ymin, ymax = coordinate_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color=GRID, linewidth=0.25, alpha=0.38)


def cloud_capacity_max(states: list[dict[str, Any]]) -> float:
    capacities = [
        finite_float(machine.get("cpu"))
        for state in states
        for machine in (state.get("machines") or [])
        if isinstance(machine, dict)
    ]
    finite = [capacity for capacity in capacities if capacity is not None]
    return max(finite + [1.0]) * 1.10


def draw_cloud_assignment(
    ax: plt.Axes,
    state: dict[str, Any],
    solution: dict[str, Any],
    capacity_ymax: float,
) -> None:
    machines = list(state.get("machines") or [])
    jobs = {
        str(job.get("id")): job
        for job in state.get("jobs") or []
        if isinstance(job, dict) and job.get("active", True)
    }
    machine_ids = [str(machine.get("id")) for machine in machines]
    capacities = {str(machine.get("id")): float(machine.get("cpu") or 0.0) for machine in machines}
    used = {machine_id: 0.0 for machine_id in machine_ids}
    assignments = solution.get("assignments") if isinstance(solution.get("assignments"), dict) else {}
    for job_id, machine_id in assignments.items():
        job = jobs.get(str(job_id))
        machine_id = str(machine_id)
        if job and machine_id in used:
            used[machine_id] += float(job.get("cpu") or 0.0)
    x = np.arange(len(machine_ids))
    ax.bar(x, [used[machine_id] for machine_id in machine_ids], color=TEAL_LIGHT, width=0.74, linewidth=0)
    ax.plot(
        x,
        [capacities[machine_id] for machine_id in machine_ids],
        color=GREY,
        linewidth=0.55,
        marker="_",
        markersize=3.5,
    )
    ax.text(
        0.03,
        0.95,
        f"{len(assignments)}/{len(jobs)} jobs",
        transform=ax.transAxes,
        va="top",
        fontsize=5.0,
        color="#536174",
    )
    ax.set_xlim(-0.6, max(len(machine_ids) - 0.4, 0.6))
    ax.set_ylim(0, capacity_ymax)
    ax.set_xticks([])
    ax.grid(True, axis="y", color=GRID, linewidth=0.28, alpha=0.52)


def collect_hv_trajectory(rows: list[dict[str, Any]], stage_index: int) -> dict[str, Any]:
    """Align seed traces, carrying stopped seeds forward to the common budget."""

    prepared: list[dict[str, Any]] = []
    maximum_budget = 0
    for row in rows:
        solver_result = stage_solver_result(row, stage_index)
        metadata = solver_result.get("metadata") or {}
        history = metadata.get("history") or []
        by_generation: dict[int, float] = {}
        for item in history:
            if not isinstance(item, dict):
                continue
            value = finite_float(item.get("normalized_hv"))
            if value is None:
                continue
            try:
                generation = int(item.get("generation") or 0)
            except (TypeError, ValueError):
                continue
            by_generation[generation] = value
        if not by_generation:
            continue
        runtime = metadata.get("runtime") if isinstance(metadata.get("runtime"), dict) else {}
        early_stop = runtime.get("early_stop") if isinstance(runtime.get("early_stop"), dict) else {}
        observed_end = max(by_generation)
        try:
            budget = int(runtime.get("generations") or observed_end)
        except (TypeError, ValueError):
            budget = observed_end
        try:
            executed = int(runtime.get("executed_generations") or early_stop.get("stop_generation") or observed_end)
        except (TypeError, ValueError):
            executed = observed_end
        executed = max(observed_end, executed)
        stopped = bool(runtime.get("early_stopped") or early_stop.get("stopped"))
        prepared.append(
            {
                "values": by_generation,
                "executed": executed,
                "stopped": stopped,
            }
        )
        maximum_budget = max(maximum_budget, budget, observed_end)
    if not prepared:
        return {}

    grid = np.arange(maximum_budget + 1, dtype=float)
    seed_series: list[np.ndarray] = []
    for item in prepared:
        values = item["values"]
        first_generation = min(values)
        last_value = values[first_generation]
        series: list[float] = []
        for generation in range(maximum_budget + 1):
            if generation in values:
                last_value = values[generation]
            series.append(last_value)
        seed_series.append(np.asarray(series, dtype=float))
    values = np.vstack(seed_series)
    executed = np.asarray([item["executed"] for item in prepared], dtype=float)
    stopped = [item for item in prepared if item["stopped"]]
    return {
        "x": grid,
        "mean": np.nanmean(values, axis=0),
        "std": np.nanstd(values, axis=0),
        "mean_end": float(np.mean(executed)),
        "std_end": float(np.std(executed)),
        "stop_count": len(stopped),
        "seed_count": len(prepared),
        "budget": maximum_budget,
    }


def draw_hv(ax: plt.Axes, trajectory: dict[str, Any], y_max: float) -> None:
    if not trajectory:
        ax.text(0.5, 0.5, "no metric trace", ha="center", va="center", fontsize=5.5, color=RED)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    x = trajectory["x"]
    mean = trajectory["mean"]
    std = trajectory["std"]
    ax.plot(x, mean, color=TEAL, linewidth=0.85)
    ax.fill_between(x, np.maximum(mean - std, 0.0), mean + std, color=TEAL_LIGHT, alpha=0.27, linewidth=0)
    mean_end = trajectory["mean_end"]
    ax.axvline(mean_end, color=ORANGE, linestyle="--", linewidth=0.62, alpha=0.88)
    stop_count = trajectory["stop_count"]
    seed_count = trajectory["seed_count"]
    end_std = trajectory["std_end"]
    suffix = f"\nES {stop_count}/{seed_count}" if stop_count else ""
    ax.text(
        0.97,
        0.95,
        f"end {mean_end:.0f}$\\pm${end_std:.0f}{suffix}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.8,
        color="#536174",
        linespacing=0.90,
    )
    ax.set_xlim(0, max(float(trajectory["budget"]), 1.0))
    ax.set_ylim(0, y_max)
    ax.grid(True, color=GRID, linewidth=0.30, alpha=0.62)


def style_axis(ax: plt.Axes, *, show_x_ticks: bool, show_y_ticks: bool) -> None:
    ax.tick_params(axis="both", labelsize=4.9, length=1.5, pad=1.0)
    if not show_x_ticks:
        ax.set_xticklabels([])
    if not show_y_ticks:
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#a9b5c5")
        spine.set_linewidth(0.42)


def draw_atlas(
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    reference_steps: list[dict[str, Any]],
    out_path: Path,
    x_key: str,
    y_key: str,
    plan_kind: str,
    scorer_benchmark: str,
    domain: str,
    family: str,
) -> None:
    stage_indices = list(range(1, 13))
    if len(reference_steps) <= stage_indices[-1]:
        raise ValueError(f"expected reference stages t00--t12, found {len(reference_steps)}")
    states = build_stage_states(payload)
    if len(states) <= stage_indices[-1]:
        raise ValueError(f"expected hidden states t00--t12, found {len(states)}")

    reference_by_stage = {
        stage: collect_reference_points(reference_steps[stage], x_key, y_key) for stage in stage_indices
    }
    live_by_stage = {
        stage: collect_archive_points(
            rows,
            stage,
            scorer_benchmark=scorer_benchmark,
            domain=domain,
            family=family,
            state=states[stage],
            reference_step=reference_steps[stage],
        )
        for stage in stage_indices
    }
    hv_by_stage = {stage: collect_hv_trajectory(rows, stage) for stage in stage_indices}
    missing_hv = [stage for stage, trajectory in hv_by_stage.items() if not trajectory]
    if missing_hv:
        missing = ", ".join(f"t{stage:02d}" for stage in missing_hv)
        raise ValueError(
            f"normalized_hv metric history missing for {missing}; replay with --record-metric-trace"
        )
    hv_upper = max(
        float(np.nanmax(trajectory["mean"] + trajectory["std"]))
        for trajectory in hv_by_stage.values()
    )
    hv_ymax = max(1.02, hv_upper * 1.035)

    canonical_row = next((row for row in rows if int(row.get("run_seed") or 0) == 0), rows[0])
    route_bounds = route_coordinate_bounds(states[1:13])
    capacity_ymax = cloud_capacity_max(states[1:13])

    fig, axes = plt.subplots(3, 12, figsize=(12.4, 4.55), dpi=240, squeeze=False)
    fig.subplots_adjust(left=0.064, right=0.996, top=0.91, bottom=0.115, wspace=0.13, hspace=0.25)
    for column, stage in enumerate(stage_indices):
        front_ax = axes[0, column]
        plan_ax = axes[1, column]
        hv_ax = axes[2, column]
        front_ax.set_title(f"t{stage:02d}", fontsize=7.1, color=INK, weight="bold", pad=2.4)
        # Normalize each public update independently.  P015 has a late resource
        # regime shift whose energy scale is an order of magnitude larger; a
        # single episode-wide x-axis collapses earlier fronts into vertical
        # lines and hides the Pareto shape reviewers need to inspect.
        front_bounds = projection_bounds({stage: reference_by_stage[stage]}, {stage: live_by_stage[stage]})
        draw_front(front_ax, reference_by_stage[stage], live_by_stage[stage], front_bounds)
        if plan_kind == "cloud":
            draw_cloud_assignment(plan_ax, states[stage], stage_solution(canonical_row, stage), capacity_ymax)
        else:
            draw_green_route(plan_ax, states[stage], stage_solution(canonical_row, stage), route_bounds)
        draw_hv(hv_ax, hv_by_stage[stage], hv_ymax)
        style_axis(front_ax, show_x_ticks=column == 0, show_y_ticks=column == 0)
        style_axis(plan_ax, show_x_ticks=False, show_y_ticks=column == 0)
        style_axis(hv_ax, show_x_ticks=True, show_y_ticks=column == 0)

    fig.text(0.009, 0.765, "Pareto set\n(stage norm.)", rotation=90, va="center", ha="center", fontsize=7.0, color=INK, weight="bold")
    fig.text(0.009, 0.505, "Accepted plan", rotation=90, va="center", ha="center", fontsize=7.0, color=INK, weight="bold")
    fig.text(0.009, 0.235, "Normalized HV", rotation=90, va="center", ha="center", fontsize=7.0, color=INK, weight="bold")
    fig.text(0.53, 0.037, "Generation", va="center", ha="center", fontsize=7.0, color=INK)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.7, markerfacecolor=GREY, markeredgewidth=0, alpha=0.55, label="Reference Pareto set"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.7, markerfacecolor="none", markeredgecolor=TEAL, markeredgewidth=0.7, label="LiveOpt pooled Pareto set"),
        Line2D([], [], color=TEAL, linewidth=1.0, label="Mean normalized HV"),
        Patch(facecolor=TEAL_LIGHT, edgecolor="none", alpha=0.27, label="$\\pm$1 population SD"),
        Line2D([], [], color=ORANGE, linestyle="--", linewidth=0.75, label="Mean final generation"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.004),
        ncol=5,
        frameon=False,
        fontsize=6.2,
        handlelength=1.35,
        columnspacing=1.25,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def episode_render_config(benchmark: str, episode_id: str) -> tuple[str, str, str, str]:
    if benchmark == "cloud_scheduling_mo":
        return ("energy", "load_imbalance", "cloud", f"{episode_id.lower().replace('-', '_')}_three_row_stage_atlas")
    if benchmark == "green_vrp_mo":
        return ("distance", "lateness", "green", f"{episode_id.lower().replace('-', '_')}_three_row_stage_atlas")
    raise ValueError(f"no visual plan renderer for benchmark={benchmark!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-jsonl", type=Path, required=True, help="Current-code replay JSONL (plain or .gz).")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Deprecated compatibility option; the atlas reads normalized_hv directly from replay history.",
    )
    parser.add_argument("--reference-jsonl", type=Path, default=Path(DEFAULT_NLDO_STRONG_REFERENCE))
    parser.add_argument(
        "--episode-id",
        action="append",
        dest="episode_ids",
        help="Episode to export; repeat for multiple episodes. Defaults to NLDO-P010 and NLDO-P015.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=10,
        help="Require this many unique replay seeds per episode (default: 10; use 0 to disable).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    episode_ids = args.episode_ids or ["NLDO-P010", "NLDO-P015"]
    for episode_id in episode_ids:
        payload = load_reference_payload(args.reference_jsonl, episode_id)
        rows = load_run_rows(args.run_jsonl, episode_id)
        run_seeds = {int(row.get("run_seed") or 0) for row in rows}
        if args.expected_seeds and len(run_seeds) != args.expected_seeds:
            raise ValueError(
                f"{episode_id} has {len(run_seeds)} unique seeds {sorted(run_seeds)}; "
                f"expected {args.expected_seeds}"
            )
        incomplete = [
            int(row.get("run_seed") or 0)
            for row in rows
            if len(row.get("update_results") or []) < 12
        ]
        if incomplete:
            raise ValueError(f"{episode_id} seeds missing one or more of t01--t12: {incomplete}")
        benchmark = effective_benchmark("NLDO", payload)
        domain = str(payload.get("domain") or "")
        family = str(payload.get("family") or "")
        reference_steps = reference_trajectory_for_payload(benchmark, domain, payload)
        x_key, y_key, plan_kind, stem = episode_render_config(benchmark, episode_id)
        out_path = args.out_dir / stem
        draw_atlas(
            payload=payload,
            rows=rows,
            reference_steps=reference_steps,
            out_path=out_path,
            x_key=x_key,
            y_key=y_key,
            plan_kind=plan_kind,
            scorer_benchmark=benchmark,
            domain=domain,
            family=family,
        )
        print(out_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
