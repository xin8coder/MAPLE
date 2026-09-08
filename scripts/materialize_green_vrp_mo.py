#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


WEIGHT_PROFILES = [
    {"name": "distance_first", "distance": 1.0, "lateness": 0.5, "emission": 0.4, "priority": 0.8},
    {"name": "service_first", "distance": 0.4, "lateness": 2.0, "emission": 0.3, "priority": 1.5},
    {"name": "carbon_first", "distance": 0.5, "lateness": 0.6, "emission": 1.8, "priority": 0.8},
    {"name": "priority_first", "distance": 0.5, "lateness": 1.2, "emission": 0.4, "priority": 2.2},
    {"name": "balanced_service", "distance": 0.8, "lateness": 1.0, "emission": 0.8, "priority": 1.0},
]


def build_episodes() -> list[dict[str, Any]]:
    return [_episode(idx) for idx in range(6)]


def _episode(idx: int) -> dict[str, Any]:
    initial = _initial_state(idx)
    updates = _updates(idx)
    state = copy.deepcopy(initial)
    public_updates = []
    oracle = []
    trajectory = []
    previous_solution = None
    row, previous_solution = _reference_row("initial", state, None, previous_solution)
    trajectory.append(row)
    for uid, text, delta in updates:
        resolved = _resolve_memory_delta(state, delta)
        public_update = {
            "update_id": uid,
            "time_index": int(uid[1:]),
            "public_update": text,
            "difficulty": _difficulty(delta),
            "requires_memory": delta["type"].startswith("memory"),
        }
        public_patch = _public_data_patch(resolved)
        if public_patch:
            public_update["public_data_patch"] = public_patch
        public_updates.append(public_update)
        oracle.append(
            {
                "update_id": uid,
                "hidden_delta": resolved,
                "expected_event_types": _expected_event_types(delta, resolved),
                "difficulty": _difficulty(delta),
                "requires_memory": delta["type"].startswith("memory"),
                "memory_reference_chain": delta.get("memory_reference_chain", []),
            }
        )
        state = _apply_delta(state, resolved)
        row, previous_solution = _reference_row(uid, state, resolved, previous_solution)
        trajectory.append(row)
    return {
        "episode_id": f"green_vrp_mo_{idx:03d}",
        "layer": "realistic_dynamic",
        "benchmark": "Green-Dynamic-VRP-MO",
        "domain": "green_vrp_multiobjective",
        "source_dataset": "Olist/DPDP-inspired synthetic dynamic slice; native NL exposed to agent",
        "source_instance_id": f"green_vrp_mo_slice_{idx:03d}",
        "public_initial_problem": _public_problem(initial),
        "public_context": {
            "input_policy": "agent sees natural language and public summaries only",
            "update_count": len(updates),
            "multi_objective": True,
            "objective_sense": "minimize_all",
        },
        "update_stream": public_updates,
        "hidden_initial_state": initial,
        "hidden_update_oracle": oracle,
        "evaluation": {
            "metrics": [
                "feasibility",
                "pareto_coverage",
                "hypervolume",
                "igd",
                "distance",
                "lateness",
                "emission",
                "disruption",
                "priority_penalty",
                "token_cost",
                "latency",
            ],
            "objectives": ["distance", "lateness", "emission"],
            "auxiliary_metrics": ["disruption", "priority_penalty"],
            "reference_trajectory": trajectory,
            "reference_policy": "constructive_pareto_solver_best_known",
            "objective_definition": (
                "Maintain feasible vehicle routes while minimizing three Pareto objectives: total travel distance, "
                "delivery lateness, and estimated carbon emissions. Disruption and priority-customer penalties are "
                "reported as auxiliary stability/service diagnostics rather than Pareto dimensions."
            ),
        },
        "agent_allowed_solvers": ["ga_or_moea", "generated_repair_operator", "linear_program_solver_v1"],
    }


def _initial_state(idx: int) -> dict[str, Any]:
    vehicle_count = 4 + (idx % 2)
    vehicles = [
        {
            "id": f"V{k}",
            "capacity": 22 + 3 * ((k + idx) % 3),
            "emission_rate": round(0.7 + 0.08 * ((k + idx) % 4), 3),
            "available": True,
            "shift_end": 260 + 10 * (k % 2),
        }
        for k in range(1, vehicle_count + 1)
    ]
    orders = []
    for k in range(1, 19):
        x = 4 + ((k * 11 + idx * 7) % 48)
        y = 3 + ((k * 13 + idx * 5) % 44)
        ready = 15 + ((k * 9 + idx * 3) % 80)
        due = ready + 45 + ((k * 5 + idx) % 55)
        orders.append(
            {
                "id": f"O{k:03d}",
                "x": float(x),
                "y": float(y),
                "demand": 1 + ((k + idx) % 5),
                "ready": float(ready),
                "due": float(due),
                "service": 4.0 + (k % 4),
                "priority": 1 + ((k + idx) % 3),
                "active": True,
            }
        )
    for order in _future_emergency_orders(idx):
        inactive = dict(order)
        inactive["active"] = False
        orders.append(inactive)
    orders.extend(_late_dispatch_orders(idx))
    return {
        "depot": {"x": 0.0, "y": 0.0},
        "vehicles": vehicles,
        "orders": orders,
        "speed": 1.0,
        "carbon_multiplier": 1.0,
        "traffic_zones": [],
        "memory_refs": {},
    }


def _updates(idx: int) -> list[tuple[str, str, dict[str, Any]]]:
    urgent_id = f"O{6 + idx:03d}"
    delayed_vehicle = f"V{1 + idx % 3}"
    affected_order = f"O{4 + idx:03d}"
    pharmacy_order = f"O{10 + idx:03d}"
    express_targets = [f"O{6 + idx:03d}", f"O{9 + idx:03d}", f"O{12 + idx:03d}"]
    bottleneck_targets = [f"O{1 + idx:03d}", f"O{2 + idx:03d}", f"O{3 + idx:03d}"]
    reset_targets = [f"O{4 + idx:03d}", f"O{8 + idx:03d}", f"O{13 + idx:03d}"]
    clean_vehicle = f"V{2 + idx % 2}"
    emergency_orders = _future_emergency_orders(idx)
    updates = [
        (
            "u001",
            (
                f"Existing order {urgent_id} becomes a same-day priority delivery: increase its public priority by 2 and tighten its soft due target by 18 minutes. "
                "The order set, vehicle set, capacities, and active flags stay unchanged."
            ),
            {"type": "compound", "changes": [{"type": "priority_delta", "order": urgent_id, "delta": 2, "memory_key": "last_priority_order"}, {"type": "tighten_due", "order": urgent_id, "delta": -18}]},
        ),
        (
            "u002",
            f"Traffic around order {affected_order} slows vehicles by 25 percent; remember this customer as the congestion anchor.",
            {"type": "traffic_delay", "order": affected_order, "factor": 1.25},
        ),
        (
            "u003",
            f"Vehicle {delayed_vehicle}'s emission_rate increases by 0.18 after a telemetry update; it remains available with the same capacity.",
            {"type": "vehicle_emission_delta_for_vehicle", "vehicle": delayed_vehicle, "delta": 0.18, "memory_key": "last_emission_vehicle"},
        ),
        (
            "u004",
            "Carbon policy is tightened: set the public policy carbon_multiplier to 1.35 so emissions count 35 percent more heavily in dispatch evaluation.",
            {"type": "carbon_multiplier", "factor": 1.35},
        ),
        (
            "u005",
            "The customer mentioned in the congestion note now needs a tighter delivery window ending 20 minutes earlier; infer the order from memory.",
            {"type": "memory_tighten_order_due", "delta": -20, "memory_reference_chain": ["u002 congestion note", "congestion anchor -> affected order", "affected order -> due-time patch"]},
        ),
        (
            "u006",
            (
                "An express-service wave temporarily reverses dispatch priorities. "
                "Set policy.carbon_multiplier to 0.75 and tighten the soft due-time targets for "
                f"{express_targets[0]}, {express_targets[1]}, and {express_targets[2]} by 45, 50, and 35 minutes respectively. "
                "Increase each of those existing orders' public priority by 1. "
                "These due-time targets affect lateness and priority-service quality rather than hard feasibility, so route quality should be reconsidered instead of blindly preserving the previous carbon-oriented plan."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "carbon_multiplier", "factor": 0.75},
                    {"type": "tighten_due", "order": express_targets[0], "delta": -45},
                    {"type": "tighten_due", "order": express_targets[1], "delta": -50},
                    {"type": "tighten_due", "order": express_targets[2], "delta": -35},
                    {"type": "priority_delta", "order": express_targets[0], "delta": 1},
                    {"type": "priority_delta", "order": express_targets[1], "delta": 1},
                    {"type": "priority_delta", "order": express_targets[2], "delta": 1},
                ],
            },
        ),
        (
            "u007",
            (
                f"Existing pharmacy-like order {pharmacy_order} gets a tighter soft due target by 22 minutes and priority increases by 2. "
                "The order table remains fixed."
            ),
            {"type": "compound", "changes": [{"type": "tighten_due", "order": pharmacy_order, "delta": -22}, {"type": "priority_delta", "order": pharmacy_order, "delta": 2, "memory_key": "last_pharmacy_order"}]},
        ),
        (
            "u008",
            (
                "A carbon-rationing notice takes effect after the express wave. "
                f"Set policy.carbon_multiplier to 4.80, increase every vehicle emission_rate by 0.35, and reduce priorities for {express_targets[0]}, {express_targets[1]}, and {express_targets[2]} by 1. "
                "This is a regime change toward carbon-efficient routing over the same active orders and vehicles, so routes specialized for the previous express wave can be misleading."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "carbon_multiplier", "factor": 4.80},
                    {"type": "vehicle_emission_delta", "delta": 0.35},
                    {"type": "priority_delta", "order": express_targets[0], "delta": -1},
                    {"type": "priority_delta", "order": express_targets[1], "delta": -1},
                    {"type": "priority_delta", "order": express_targets[2], "delta": -1},
                ],
            },
        ),
        (
            "u009",
            "The vehicle from the telemetry update receives a 0.10 emission_rate correction downward; use the vehicle identity from memory.",
            {"type": "memory_vehicle_emission_delta", "delta": -0.10, "memory_reference_chain": ["u003 telemetry vehicle", "vehicle -> emission correction"]},
        ),
        (
            "u010",
            (
                f"A late emergency service wave changes priorities over the same order set. Set policy.carbon_multiplier to 0.45, subtract 0.20 from every vehicle emission_rate, "
                f"and subtract another 0.08 from cleaner vehicle {clean_vehicle}. "
                f"Tighten the soft due-time targets for {bottleneck_targets[0]}, {bottleneck_targets[1]}, and {bottleneck_targets[2]} by 40, 45, and 50 minutes, and increase their priorities by 2. "
                "These due-time targets affect lateness and priority-service quality, not hard feasibility. Use the current public tables as the source of truth."
            ),
            {
                "type": "compound",
                "changes": [
                    {"type": "carbon_multiplier", "factor": 0.45},
                    {"type": "vehicle_emission_delta", "delta": -0.20},
                    {"type": "vehicle_emission_delta_for_vehicle", "vehicle": clean_vehicle, "delta": -0.08},
                    {"type": "tighten_due", "order": bottleneck_targets[0], "delta": -40},
                    {"type": "tighten_due", "order": bottleneck_targets[1], "delta": -45},
                    {"type": "tighten_due", "order": bottleneck_targets[2], "delta": -50},
                    {"type": "priority_delta", "order": bottleneck_targets[0], "delta": 2},
                    {"type": "priority_delta", "order": bottleneck_targets[1], "delta": 2},
                    {"type": "priority_delta", "order": bottleneck_targets[2], "delta": 2},
                ],
            },
        ),
        _late_dispatch_regime_update(idx, 11),
        _late_dispatch_regime_update(idx, 12),
    ]
    return updates


def _late_dispatch_regime_update(idx: int, stage_index: int) -> tuple[str, str, dict[str, Any]]:
    regime = _dispatch_regime(idx, stage_index)
    label = "A" if stage_index == 11 else "B"
    text = (
        f"A network-wide dispatch reset activates public regime {label}. Replace the active workload by the prelisted cohort {label}, replace the depot coordinates and every "
        "active order's x, y, demand, ready, due, service, and priority fields by the supplied current rows; also "
        "replace every vehicle capacity, shift_end, availability, emission_rate, and policy.carbon_multiplier by the supplied values. Serve every "
        "public order row in the active cohort exactly once. The route encoding type and the three objective definitions remain unchanged."
    )
    return f"u{stage_index:03d}", text, {"type": "dispatch_regime_set", **regime}


def _dispatch_regime(idx: int, stage_index: int) -> dict[str, Any]:
    order_count = 30
    multiplier = 7 if stage_index == 11 else 11
    offset = (idx * 5 + (3 if stage_index == 11 else 10)) % order_count
    cohort_prefix = "A" if stage_index == 11 else "B"
    order_ids = [f"{cohort_prefix}{k:03d}" for k in range(1, order_count + 1)]
    orders: list[dict[str, Any]] = []
    for k, order_id in enumerate(order_ids, start=1):
        permuted = ((k - 1) * multiplier + offset) % order_count
        cluster = permuted % 3
        rank = permuted // 3
        if stage_index == 11:
            centers = [(-58.0, 44.0), (61.0, 49.0), (8.0, -64.0)]
        else:
            centers = [(66.0, -48.0), (-63.0, -42.0), (-4.0, 69.0)]
        center_x, center_y = centers[cluster]
        x = center_x + (rank - 2.5) * (4.0 + idx)
        y = center_y + ((rank * 7 + k * 3 + idx) % 13) - 6.0
        ready = float(8 + ((permuted * 17 + idx * 9 + stage_index) % 105))
        due = ready + float(34 + ((order_count - 1 - permuted) * 7 + k * 5 + idx) % 48)
        orders.append(
            {
                "id": order_id,
                "x": round(x, 3),
                "y": round(y, 3),
                "demand": 1 + ((permuted + idx + stage_index) % 5),
                "ready": ready,
                "due": due,
                "service": float(4 + ((permuted + k) % 4)),
                "priority": 1 + ((2 * permuted + idx + stage_index) % 5),
                "active": True,
            }
        )
    vehicle_count = 4 + (idx % 2)
    vehicle_profiles = {}
    for k in range(1, vehicle_count + 1):
        phase = (k * (3 if stage_index == 11 else 5) + idx) % vehicle_count
        vehicle_profiles[f"V{k}"] = {
            "capacity": 70 + 5 * ((phase + stage_index) % 3),
            "emission_rate": round(0.32 + 0.58 * phase, 3),
            "available": True,
            "shift_end": 1800.0,
        }
    return {
        "orders": orders,
        "active_order_ids": order_ids,
        "inactive_order_ids": [
            *[f"O{k:03d}" for k in range(1, 19)],
            *[f"{'B' if cohort_prefix == 'A' else 'A'}{k:03d}" for k in range(1, 31)],
        ],
        "depot": {
            "x": 74.0 + 3.0 * idx if stage_index == 11 else -71.0 - 2.0 * idx,
            "y": -57.0 + 2.0 * idx if stage_index == 11 else 63.0 - 3.0 * idx,
        },
        "vehicle_profiles": vehicle_profiles,
        "carbon_multiplier": 8.5 if stage_index == 11 else 5.75,
        "regime_id": f"late-{stage_index}-p{idx:02d}",
    }


def _new_order(idx: int, order_id: str, due: int, priority: int, demand: int | None = None) -> dict[str, Any]:
    return {
        "id": order_id,
        "x": float(8 + ((idx * 17 + 5) % 45)),
        "y": float(7 + ((idx * 19 + 11) % 42)),
        "demand": demand if demand is not None else 2 + (idx % 4),
        "ready": float(18 + (idx % 6) * 5),
        "due": float(due),
        "service": 5.0,
        "priority": priority,
        "active": True,
    }


def _future_emergency_orders(idx: int) -> list[dict[str, Any]]:
    return [
        _new_order(idx + 20, f"OX{idx}A", due=48 + idx * 4, priority=5, demand=3),
        _new_order(idx + 30, f"OX{idx}B", due=58 + idx * 4, priority=5, demand=4),
    ]


def _late_dispatch_orders(idx: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{cohort}{k:03d}",
            "x": float(6 + ((k * 13 + idx * 5) % 48)),
            "y": float(5 + ((k * 17 + idx * 7) % 44)),
            "demand": 1 + ((k + idx) % 5),
            "ready": float(12 + ((k * 7 + idx) % 90)),
            "due": float(80 + ((k * 11 + idx) % 110)),
            "service": float(4 + (k % 4)),
            "priority": 1 + ((k + idx) % 3),
            "active": False,
        }
        for cohort in ("A", "B")
        for k in range(1, 31)
    ]


def _format_order_row(order: dict[str, Any]) -> str:
    return (
        f"{order['id']} x {order['x']} y {order['y']} demand {order['demand']} "
        f"ready {order['ready']} due {order['due']} service {order['service']} "
        f"priority {order['priority']} active {str(order['active']).lower()}"
    )


def _public_problem(state: dict[str, Any]) -> str:
    vehicle_text = ", ".join(
        f"{v['id']} capacity {v['capacity']} emission {v['emission_rate']}" for v in state["vehicles"]
    )
    order_text = ", ".join(
        f"{o['id']} demand {o['demand']} window {int(o['ready'])}-{int(o['due'])} priority {o['priority']}"
        for o in state["orders"]
    )
    return (
        "Plan same-day delivery routes from a single depot for a small urban fleet. "
        f"Vehicles: {vehicle_text}. Orders: {order_text}. "
        "All active orders should be served once if capacity and time windows allow. "
        "Optimize three main criteria: travel distance, delivery lateness, and emissions. "
        "Also report route disruption and priority-customer penalties as auxiliary diagnostics."
    )


def _resolve_memory_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if delta["type"] == "memory_tighten_order_due":
        return {
            "type": "tighten_due",
            "order": state.get("memory_refs", {}).get("last_congestion_order"),
            "delta": delta["delta"],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_replace_vehicle":
        return {
            "type": "replace_vehicle",
            "vehicle": state.get("memory_refs", {}).get("last_unavailable_vehicle"),
            "capacity": delta["capacity"],
            "emission_rate": delta["emission_rate"],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_vehicle_emission_delta":
        return {
            "type": "vehicle_emission_delta_for_vehicle",
            "vehicle": state.get("memory_refs", {}).get("last_emission_vehicle"),
            "delta": float(delta["delta"]),
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    return delta


def _apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = copy.deepcopy(state)
    typ = delta["type"]
    if typ == "no_op":
        return state
    if typ == "dispatch_regime_set":
        replacement_by_id = {row["id"]: copy.deepcopy(row) for row in delta.get("orders", [])}
        active_order_ids = set(delta.get("active_order_ids") or replacement_by_id)
        for order in state.get("orders", []):
            replacement = replacement_by_id.get(order.get("id"))
            if replacement is not None:
                order.update(replacement)
            order["active"] = order.get("id") in active_order_ids
        state["depot"] = copy.deepcopy(delta["depot"])
        state["carbon_multiplier"] = float(delta["carbon_multiplier"])
        vehicle_profiles = delta.get("vehicle_profiles") or {}
        for vehicle in state.get("vehicles", []):
            if vehicle.get("id") in vehicle_profiles:
                vehicle.update(copy.deepcopy(vehicle_profiles[vehicle["id"]]))
        state["traffic_zones"] = []
    elif typ == "new_order":
        incoming = copy.deepcopy(delta["order"])
        incoming["active"] = True
        for order in state["orders"]:
            if order.get("id") == incoming.get("id"):
                order.update(incoming)
                break
        else:
            state["orders"].append(incoming)
    elif typ == "traffic_delay":
        state.setdefault("traffic_zones", []).append({"order": delta["order"], "factor": float(delta["factor"])})
        state.setdefault("memory_refs", {})["last_congestion_order"] = delta["order"]
    elif typ == "priority_delta":
        for order in state["orders"]:
            if order["id"] == delta.get("order"):
                order["priority"] = max(1, int(order.get("priority", 1)) + int(delta["delta"]))
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["order"]
    elif typ == "vehicle_unavailable":
        for vehicle in state["vehicles"]:
            if vehicle["id"] == delta["vehicle"]:
                vehicle["available"] = False
                state.setdefault("memory_refs", {})["last_unavailable_vehicle"] = delta["vehicle"]
    elif typ == "carbon_multiplier":
        state["carbon_multiplier"] = float(delta["factor"])
    elif typ == "depot_set":
        state.setdefault("depot", {})
        state["depot"]["x"] = float(delta["x"])
        state["depot"]["y"] = float(delta["y"])
    elif typ == "tighten_due":
        for order in state["orders"]:
            if order["id"] == delta.get("order"):
                order["due"] = max(order["ready"], order["due"] + float(delta["delta"]))
    elif typ == "cancel_order":
        for order in state["orders"]:
            if order["id"] == delta["order"]:
                order["active"] = False
    elif typ == "vehicle_emission_delta":
        for vehicle in state["vehicles"]:
            vehicle["emission_rate"] = round(vehicle["emission_rate"] + float(delta["delta"]), 3)
    elif typ == "vehicle_emission_delta_for_vehicle":
        for vehicle in state["vehicles"]:
            if vehicle["id"] == delta.get("vehicle"):
                vehicle["emission_rate"] = round(vehicle["emission_rate"] + float(delta["delta"]), 3)
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["vehicle"]
    elif typ == "vehicle_emission_set_for_vehicle":
        for vehicle in state["vehicles"]:
            if vehicle["id"] == delta.get("vehicle"):
                vehicle["emission_rate"] = round(float(delta["emission_rate"]), 3)
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["vehicle"]
    elif typ == "vehicle_capacity_set":
        for vehicle in state["vehicles"]:
            if vehicle["id"] == delta["vehicle"]:
                vehicle["capacity"] = int(delta["capacity"])
    elif typ == "replace_vehicle" and delta.get("vehicle"):
        for vehicle in state["vehicles"]:
            if vehicle["id"] == delta["vehicle"]:
                vehicle.update({"available": True, "capacity": int(delta["capacity"]), "emission_rate": float(delta["emission_rate"])})
    elif typ == "compound":
        for change in delta.get("changes", []):
            state = _apply_delta(state, change)
    return state


def _reference_row(update_id: str, state: dict[str, Any], delta: dict[str, Any] | None, previous_solution: dict[str, Any] | None):
    archive = _pareto_archive(state, previous_solution)
    representative = min(archive, key=lambda item: item["scalar_score"]) if archive else None
    next_previous = representative["solution"] if representative else previous_solution
    return {
        "update_id": update_id,
        "reference_type": "pareto_solver_best_known",
        "objectives": ["distance", "lateness", "emission"],
        "auxiliary_metrics": ["disruption", "priority_penalty"],
        "pareto_archive": archive,
        "archive_size": len(archive),
        "hypervolume_proxy": _hypervolume_proxy(archive),
        "solver": "deterministic_weighted_insertion_pareto_proxy",
        "hidden_delta_type": (delta or {}).get("type"),
        "representative_solution": next_previous,
    }, next_previous


def _pareto_archive(state: dict[str, Any], previous_solution: dict[str, Any] | None) -> list[dict[str, Any]]:
    candidates = []
    for profile in WEIGHT_PROFILES:
        solution = _construct_routes(state, previous_solution, profile)
        objectives = _score_solution(state, solution, previous_solution)
        scalar = _scalarize(objectives, profile)
        candidates.append({"profile": profile["name"], "solution": solution, "objectives": objectives, "scalar_score": scalar})
    return _nondominated(candidates)


def _construct_routes(state: dict[str, Any], previous_solution: dict[str, Any] | None, profile: dict[str, float]) -> dict[str, Any]:
    vehicles = [v for v in state["vehicles"] if v.get("available", True)]
    active_orders = [o for o in state["orders"] if o.get("active", True)]
    if profile["name"] == "service_first":
        active_orders.sort(key=lambda o: (o["due"], -o["priority"], o["ready"]))
    elif profile["name"] == "carbon_first":
        active_orders.sort(key=lambda o: (o["demand"], o["due"], -o["priority"]))
        vehicles.sort(key=lambda v: (v["emission_rate"], -v["capacity"]))
    elif profile["name"] == "priority_first":
        active_orders.sort(key=lambda o: (-o["priority"], o["due"]))
    else:
        active_orders.sort(key=lambda o: (_distance(state["depot"], o), o["due"]))
    loads = {v["id"]: 0 for v in vehicles}
    routes = {v["id"]: [] for v in vehicles}
    for order in active_orders:
        best_vehicle = None
        best_cost = float("inf")
        for vehicle in vehicles:
            if loads[vehicle["id"]] + order["demand"] > vehicle["capacity"]:
                continue
            last = _last_location(state, routes[vehicle["id"]])
            incremental = _distance(last, order) + _distance(order, state["depot"]) - _distance(last, state["depot"])
            cost = incremental * profile.get("distance", 1.0)
            cost += vehicle["emission_rate"] * state.get("carbon_multiplier", 1.0) * incremental * profile.get("emission", 0.5)
            cost += max(0.0, _arrival_time(state, routes[vehicle["id"]], order) - order["due"]) * profile.get("lateness", 1.0)
            cost -= order["priority"] * profile.get("priority", 1.0)
            if cost < best_cost:
                best_vehicle = vehicle["id"]
                best_cost = cost
        if best_vehicle is not None:
            routes[best_vehicle].append(order["id"])
            loads[best_vehicle] += order["demand"]
    return {"routes": [{"vehicle": vid, "orders": orders} for vid, orders in routes.items() if orders]}


def _score_solution(state: dict[str, Any], solution: dict[str, Any], previous_solution: dict[str, Any] | None) -> dict[str, float]:
    order_map = {o["id"]: o for o in state["orders"]}
    vehicle_map = {v["id"]: v for v in state["vehicles"]}
    active = {o["id"] for o in state["orders"] if o.get("active", True)}
    served = set()
    distance = 0.0
    lateness = 0.0
    emission = 0.0
    priority_penalty = 0.0
    for route in solution.get("routes", []):
        vehicle = vehicle_map[route["vehicle"]]
        time = 0.0
        load = 0.0
        loc = state["depot"]
        for oid in route["orders"]:
            order = order_map[oid]
            leg = _distance(loc, order) * _traffic_factor(state, oid)
            time += leg
            time = max(time, order["ready"])
            late = max(0.0, time - order["due"])
            distance += leg
            lateness += late
            priority_penalty += late * order["priority"]
            emission += leg * vehicle["emission_rate"] * state.get("carbon_multiplier", 1.0) * (1.0 + 0.02 * load)
            load += order["demand"]
            time += order["service"]
            loc = order
            served.add(oid)
        back = _distance(loc, state["depot"])
        distance += back
        emission += back * vehicle["emission_rate"] * state.get("carbon_multiplier", 1.0)
    missed = active - served
    priority_penalty += sum(order_map[oid]["priority"] * 50.0 for oid in missed)
    lateness += len(missed) * 100.0
    disruption = _disruption(solution, previous_solution)
    return {
        "distance": round(distance, 3),
        "lateness": round(lateness, 3),
        "emission": round(emission, 3),
        "disruption": round(disruption, 3),
        "priority_penalty": round(priority_penalty, 3),
    }


def _arrival_time(state: dict[str, Any], route: list[str], order: dict[str, Any]) -> float:
    order_map = {o["id"]: o for o in state["orders"]}
    time = 0.0
    loc = state["depot"]
    for oid in route:
        current = order_map[oid]
        time += _distance(loc, current) * _traffic_factor(state, oid)
        time = max(time, current["ready"]) + current["service"]
        loc = current
    return time + _distance(loc, order) * _traffic_factor(state, order["id"])


def _last_location(state: dict[str, Any], route: list[str]) -> dict[str, float]:
    if not route:
        return state["depot"]
    order_map = {o["id"]: o for o in state["orders"]}
    return order_map[route[-1]]


def _traffic_factor(state: dict[str, Any], order_id: str) -> float:
    factor = 1.0
    for zone in state.get("traffic_zones", []):
        if zone["order"] == order_id:
            factor *= float(zone["factor"])
    return factor


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _previous_vehicle_map(solution: dict[str, Any] | None) -> dict[str, str]:
    if not solution:
        return {}
    out = {}
    for route in solution.get("routes", []):
        for oid in route.get("orders", []):
            out[oid] = route["vehicle"]
    return out


def _previous_positions(solution: dict[str, Any] | None) -> dict[str, tuple[str, int]]:
    if not solution:
        return {}
    out = {}
    for route in solution.get("routes", []):
        for idx, oid in enumerate(route.get("orders", [])):
            out[oid] = (route["vehicle"], idx)
    return out


def _disruption(solution: dict[str, Any], previous_solution: dict[str, Any] | None) -> float:
    if not previous_solution:
        return 0.0
    current = _previous_vehicle_map(solution)
    previous = _previous_vehicle_map(previous_solution)
    keys = set(current) | set(previous)
    return float(sum(1 for key in keys if current.get(key) != previous.get(key)))


def _scalarize(objectives: dict[str, float], profile: dict[str, float]) -> float:
    return (
        objectives["distance"] * profile.get("distance", 1.0)
        + objectives["lateness"] * profile.get("lateness", 1.0)
        + objectives["emission"] * profile.get("emission", 1.0)
        + objectives["priority_penalty"] * profile.get("priority", 1.0)
    )


def _nondominated(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cand in candidates:
        if not any(_dominates(other["objectives"], cand["objectives"]) for other in candidates if other is not cand):
            out.append(cand)
    seen = set()
    unique = []
    for cand in sorted(out, key=lambda item: item["scalar_score"]):
        key = tuple(cand["objectives"][name] for name in ["distance", "lateness", "emission"])
        if key not in seen:
            unique.append(cand)
            seen.add(key)
    return unique


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    names = ["distance", "lateness", "emission"]
    return all(a[name] <= b[name] for name in names) and any(a[name] < b[name] for name in names)


def _hypervolume_proxy(archive: list[dict[str, Any]]) -> float:
    if not archive:
        return 0.0
    total = 0.0
    for item in archive:
        obj = item["objectives"]
        total += 1.0 / (1.0 + obj["distance"] + obj["lateness"] + obj["emission"])
    return round(total, 6)


def _event_types(delta: dict[str, Any]) -> list[str]:
    typ = delta["type"]
    if typ == "compound":
        events = []
        for change in delta.get("changes", []):
            events.extend(_event_types(change))
        return list(dict.fromkeys(events))
    if typ == "dispatch_regime_set":
        return ["geometry_change", "time_window_change", "demand_change", "objective_change"]
    if typ in {"new_order", "cancel_order"}:
        return ["legacy_demand_change"]
    if typ in {"vehicle_unavailable", "vehicle_capacity_set", "replace_vehicle"}:
        return ["legacy_resource_change"]
    if typ in {"traffic_delay", "tighten_due"}:
        return ["soft_time_cost_change", "objective_change"]
    if typ in {"carbon_multiplier", "depot_set", "vehicle_emission_delta", "vehicle_emission_delta_for_vehicle", "vehicle_emission_set_for_vehicle", "priority_delta"}:
        return ["multi_objective_weight_change", "objective_change"]
    return ["objective_change"]


def _expected_event_types(original: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    events = _event_types(resolved)
    if original["type"].startswith("memory") and "memory_reference" not in events:
        return ["memory_reference", *events]
    return events


def _difficulty(delta: dict[str, Any]) -> str:
    if delta["type"] == "dispatch_regime_set":
        return "large"
    if delta["type"].startswith("memory"):
        return "memory"
    if delta["type"] in {"vehicle_unavailable", "vehicle_capacity_set", "new_order", "replace_vehicle"}:
        return "legacy_large"
    if delta["type"] in {"compound", "carbon_multiplier", "depot_set", "vehicle_emission_delta", "vehicle_emission_delta_for_vehicle", "vehicle_emission_set_for_vehicle", "priority_delta", "traffic_delay", "tighten_due"}:
        return "objective"
    return "constraint"


def _public_data_patch(delta: dict[str, Any]) -> dict[str, Any]:
    if delta.get("type") != "dispatch_regime_set":
        return {}
    operations: list[dict[str, Any]] = [
        {
            "op": "update_row",
            "path": ["tables", "depot"],
            "match": {"id": "DEPOT"},
            "values": copy.deepcopy(delta["depot"]),
        },
        {
            "op": "set_value",
            "path": ["tables", "policy", "carbon_multiplier"],
            "value": float(delta["carbon_multiplier"]),
        },
    ]
    operations.extend(
        {
            "op": "update_row",
            "path": ["tables", "orders"],
            "match": {"id": order["id"]},
            "values": {key: value for key, value in order.items() if key != "id"},
        }
        for order in delta.get("orders", [])
    )
    operations.extend(
        {
            "op": "update_row",
            "path": ["tables", "orders"],
            "match": {"id": order_id},
            "values": {"active": False},
        }
        for order_id in delta.get("inactive_order_ids", [])
    )
    operations.extend(
        {
            "op": "update_row",
            "path": ["tables", "vehicles"],
            "match": {"id": vehicle_id},
            "values": copy.deepcopy(profile),
        }
        for vehicle_id, profile in sorted((delta.get("vehicle_profiles") or {}).items())
    )
    return {
        "operations": operations,
        "reason": f"Deterministic public table replacement for {delta.get('regime_id')}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Green Dynamic VRP multi-objective episodes.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/green_vrp_mo_6episodes.jsonl")
    args = parser.parse_args()
    episodes = build_episodes()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(episodes), "updates": sum(len(ep["update_stream"]) for ep in episodes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
