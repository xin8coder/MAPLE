#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def build_episodes() -> list[dict[str, Any]]:
    episodes = []
    for idx in range(3):
        episodes.append(_knapsack_episode(idx))
    for idx in range(3):
        episodes.append(_facility_episode(idx))
    for idx in range(3):
        episodes.append(_set_cover_episode(idx))
    return episodes


def _maintenance_updates(start: int, end: int) -> list[tuple[str, str, dict[str, Any]]]:
    texts = {
        6: "Operations asks for a checkpoint only: keep the current public objective and hard constraints, revalidate the plan, and change it only if it is no longer feasible.",
        7: "No new business event is reported for this stage. Refresh the solution against the current public tables.",
        8: "Treat this as a routine review. The model, objective terms, and public rows stay as they are; preserve feasibility under the current state.",
        9: "The planning team wants the current public contract revalidated without adding new constraints.",
    }
    return [
        (f"u{stage:03d}", texts.get(stage, "Run a maintenance update with the same public contract."), {"type": "no_op", "reason": "maintenance review"})
        for stage in range(start, end + 1)
    ]


def _knapsack_episode(idx: int) -> dict[str, Any]:
    items = [
        {"id": f"I{k:02d}", "value": 8 + ((k * 7 + idx * 3) % 19), "weight": 2 + ((k * 5 + idx) % 9)}
        for k in range(1, 13)
    ]
    state = {"family": "knapsack", "items": items, "capacity": 34 + idx * 3, "mandatory": [], "forbidden": [], "objective_mode": "maximize_value", "value_floor": 0, "memory_refs": {}}
    sponsor_item = f"I{3 + idx:02d}"
    risk_item = f"I{7 + idx:02d}"
    audit_item = f"I{10 - idx:02d}"
    updates = [
        (
            "u001",
            f"A sponsor review increases item {sponsor_item}'s public value by {6 + idx}. The capacity and item eligibility rules stay unchanged.",
            {"type": "value_delta", "item": sponsor_item, "delta": 6 + idx, "memory_key": "last_sponsor_item"},
        ),
        ("u002", f"A risk review lowers item {risk_item}'s public value by {5 + idx}; it remains selectable under the same capacity rule.", {"type": "value_delta", "item": risk_item, "delta": -(5 + idx), "memory_key": "last_risk_item"}),
        ("u003", f"Quality audit increases item {audit_item}'s public value by 4 while the feasible portfolio definition remains fixed.", {"type": "value_delta", "item": audit_item, "delta": 4, "memory_key": "last_audit_item"}),
        (
            "u004",
            "The item praised in the sponsor review receives another value increase of 6; infer which item that was from memory.",
            {"type": "memory_value_delta_for_sponsor_item", "delta": 6, "memory_reference_chain": ["u001 sponsor review", "sponsor item -> value update target"]},
        ),
        (
            "u005",
            "The item from the risk review recovers 3 value points, while the sponsor-reviewed item gains 1 more point; resolve both items from memory.",
            {"type": "memory_knapsack_compound_value_delta", "risk_delta": 3, "sponsor_delta": 1, "memory_reference_chain": ["u002 risk review item", "u001 sponsor review item"]},
        ),
    ]
    if idx == 0:
        updates.extend(_maintenance_updates(6, 9))
        updates.append(
            (
                "u010",
                (
                    "A late value-model reset changes several public item scores without changing capacity or eligibility. "
                    "Item I10 receives a value bonus of 10, item I11 receives a value penalty of 8, and item I05 receives a value bonus of 10. "
                    "Rebuild the portfolio under the same feasible subset definition rather than carrying forward the old value ranking."
                ),
                {
                    "type": "compound",
                    "changes": [
                        {"type": "value_delta", "item": "I10", "delta": 10},
                        {"type": "value_delta", "item": "I11", "delta": -8},
                        {"type": "value_delta", "item": "I05", "delta": 10},
                    ],
                },
            )
        )
    return _episode("knapsack", idx, state, updates)


def _facility_episode(idx: int) -> dict[str, Any]:
    facilities = [
        {"id": f"F{k}", "open_cost": 15 + ((k * 4 + idx) % 13), "capacity": 18 + ((k + idx) % 4) * 4}
        for k in range(1, 6)
    ]
    customers = []
    for k in range(1, 11):
        row = {
            "id": f"C{k:02d}",
            "demand": 2 + ((k + idx) % 5),
            "serve_cost": {f"F{j}": 2 + ((k * j + idx) % 11) for j in range(1, 6)},
        }
        row["serve_time"] = {facility: max(1, 16 - cost) for facility, cost in row["serve_cost"].items()}
        customers.append(row)
    state = {"family": "facility_location", "facilities": facilities, "customers": customers, "mandatory": [], "forbidden": [], "objective_mode": "minimize_cost", "memory_refs": {}}
    expensive_facility = f"F{2 + idx % 2}"
    discount_facility = f"F{5 - idx % 2}"
    changed_customer = f"C0{3 + idx}"
    preferred_facility = f"F{1 + idx % 3}"
    updates = [
        ("u001", f"Facility {expensive_facility}'s opening cost increases by {4 + idx} after an energy-price review; capacity and facility eligibility stay unchanged.", {"type": "open_cost_delta", "facility": expensive_facility, "delta": 4 + idx, "memory_key": "last_cost_facility"}),
        ("u002", f"Serving customer {changed_customer} from facility {preferred_facility} becomes 3 cost units cheaper per demand unit after a local contract update.", {"type": "serve_cost_delta", "customer": changed_customer, "facility": preferred_facility, "delta": -3, "memory_key": "last_service_pair"}),
        ("u003", f"Facility {discount_facility}'s opening cost decreases by 5 because of a regional service credit.", {"type": "open_cost_delta", "facility": discount_facility, "delta": -5, "memory_key": "last_discount_facility"}),
        (
            "u004",
            "The facility from the regional service credit receives one more opening-cost discount of 2; infer that facility from memory.",
            {"type": "memory_open_cost_delta_for_discount_facility", "delta": -2, "memory_reference_chain": ["u003 regional service credit", "discount facility -> open cost target"]},
        ),
        (
            "u005",
            "The customer-facility pair from the local contract update receives another 1 unit service-cost discount, while the facility from the energy-price review gets 1 more opening-cost penalty; resolve both references from memory.",
            {"type": "memory_facility_compound_cost_delta", "service_delta": -1, "open_delta": 1, "memory_reference_chain": ["u002 local contract pair", "u001 energy-price facility"]},
        ),
    ]
    if idx == 0:
        updates.extend(_maintenance_updates(6, 9))
        updates.append(
            (
                "u010",
                (
                    "A same-day service-level reset flips the scalar objective. Keep the same facility set, customer set, demand, and capacity limits, "
                    "but stop minimizing monetary opening plus service cost. The new objective is to minimize total service time using the public "
                    "serve_time columns, even if that requires a more expensive assignment."
                ),
                {"type": "objective_mode", "mode": "minimize_service_time"},
            )
        )
    return _episode("facility_location", idx, state, updates)


def _set_cover_episode(idx: int) -> dict[str, Any]:
    elements = [{"id": f"E{k:02d}", "active": True} for k in range(1, 11)]
    sets = []
    for sid in range(1, 9):
        covers = [
            f"E{eid:02d}"
            for eid in range(1, 11)
            if ((eid + sid + idx) % 3 == 0) or ((eid * (sid + 1) + idx) % 7 == 0)
        ]
        if not covers:
            covers = [f"E{((sid + idx) % 10) + 1:02d}"]
        sets.append({"id": f"S{sid:02d}", "cost": 4 + ((sid * 5 + idx * 2) % 13), "covers": sorted(set(covers)), "active": True})
    state = {"family": "set_cover", "sets": sets, "elements": elements, "mandatory": [], "forbidden": [], "memory_refs": {}}
    priority_set = f"S{3 + idx:02d}"
    certificate_set = f"S{6 - idx % 2:02d}"
    audit_set = f"S{2 + idx:02d}"
    updates = [
        ("u001", f"Set {priority_set} receives a public service-quality rebate of 3 cost units; all element coverage requirements stay unchanged.", {"type": "set_cost_delta", "set": priority_set, "delta": -3, "memory_key": "last_priority_set"}),
        ("u002", f"Set {certificate_set} becomes 2 cost units cheaper because it carries a compliance certificate.", {"type": "set_cost_delta", "set": certificate_set, "delta": -2, "memory_key": "last_certificate_set"}),
        ("u003", f"Provider audit increases set {audit_set}'s public cost by 4, but the set remains available and the coverage universe is unchanged.", {"type": "set_cost_delta", "set": audit_set, "delta": 4, "memory_key": "last_audit_set"}),
        (
            "u004",
            "The set carrying the compliance certificate becomes 3 cost units cheaper; infer which set that was from memory.",
            {"type": "memory_cost_delta_for_certificate_set", "delta": -3, "memory_reference_chain": ["u002 compliance certificate", "certificate set -> cost update target"]},
        ),
        (
            "u005",
            "The audited set receives a 1 cost-unit correction, and the high-priority rebate set receives another 2 unit discount; resolve both sets from memory.",
            {"type": "memory_set_compound_cost_delta", "audit_delta": -1, "priority_delta": -2, "memory_reference_chain": ["u003 audited set", "u001 priority rebate set"]},
        ),
    ]
    return _episode("set_cover", idx, state, updates)


def _episode(family: str, idx: int, initial_state: dict[str, Any], updates: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    state = json.loads(json.dumps(initial_state))
    public_updates = []
    oracle = []
    trajectory = [_reference_row("initial", state, None)]
    for uid, text, delta in updates:
        public_updates.append({"update_id": uid, "time_index": int(uid[1:]), "public_update": text, "difficulty": _difficulty(delta), "requires_memory": delta["type"].startswith("memory")})
        resolved_delta = _resolve_memory_delta(state, delta)
        oracle.append({
            "update_id": uid,
            "hidden_delta": resolved_delta,
            "expected_event_types": _expected_event_types(delta, resolved_delta),
            "difficulty": _difficulty(delta),
            "requires_memory": delta["type"].startswith("memory"),
            "memory_reference_chain": delta.get("memory_reference_chain", []),
        })
        state = _apply_delta(state, resolved_delta)
        trajectory.append(_reference_row(uid, state, resolved_delta))
    return {
        "episode_id": f"cobench_exact_small_{family}_{idx:03d}",
        "layer": "exact_small",
        "benchmark": "CO-Bench-Exact-Small",
        "domain": "cobench",
        "family": family,
        "source_dataset": "synthetic exact-small CO-Bench materialization",
        "source_instance_id": f"{family}_{idx:03d}",
        "public_initial_problem": _public_problem(initial_state),
        "public_context": {"input_policy": "agent sees natural language and public summaries only", "update_count": len(updates)},
        "update_stream": public_updates,
        "hidden_initial_state": initial_state,
        "hidden_update_oracle": oracle,
        "evaluation": {
            "metrics": ["feasibility", "objective_gap", "exact_reference_objective", "disruption", "token_cost", "latency"],
            "reference_trajectory": trajectory,
            "reference_policy": "exact_optimum_by_enumeration",
        },
        "agent_allowed_solvers": ["linear_program_solver_v1", "ga_or_moea", "generated_repair_operator"],
    }


def _public_problem(state: dict[str, Any]) -> str:
    if state["family"] == "knapsack":
        items = ", ".join(f"{i['id']} value {i['value']} weight {i['weight']}" for i in state["items"])
        return f"Select a feasible portfolio under fixed capacity {state['capacity']}. Items: {items}. Maximize value; later natural-language updates may revise public item values but do not change capacity or eligibility."
    if state["family"] == "set_cover":
        elements = ", ".join(e["id"] for e in state["elements"] if e.get("active", True))
        sets = ", ".join(f"{s['id']} cost {s['cost']} covers {'/'.join(s['covers'])}" for s in state["sets"])
        return f"Select a minimum-cost collection of public sets that covers the fixed active element universe. Elements: {elements}. Sets: {sets}. Later updates may revise public set costs but do not add/remove coverage requirements or set eligibility."
    facilities = ", ".join(f"{f['id']} open cost {f['open_cost']} capacity {f['capacity']}" for f in state["facilities"])
    customers = ", ".join(f"{c['id']} demand {c['demand']}" for c in state["customers"])
    return f"Choose facilities and assign customers to open facilities. Facilities: {facilities}. Customers: {customers}. Minimize opening plus service cost; later public updates may revise costs while capacities, customer demand, and eligibility stay fixed."


def _resolve_memory_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    if delta["type"] == "memory_forbidden_item":
        return {"type": "forbidden_item", "item": state.get("memory_refs", {}).get("last_forbidden_item")}
    if delta["type"] == "memory_forbidden_facility":
        return {"type": "forbidden_facility", "facility": state.get("memory_refs", {}).get("last_forbidden_facility")}
    if delta["type"] == "memory_forbidden_set":
        return {"type": "forbidden_set", "set": state.get("memory_refs", {}).get("last_forbidden_set")}
    if delta["type"] == "memory_value_delta_for_sponsor_item":
        return {"type": "value_delta", "item": state.get("memory_refs", {}).get("last_sponsor_item"), "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_knapsack_compound_value_delta":
        return {
            "type": "compound",
            "changes": [
                {"type": "value_delta", "item": state.get("memory_refs", {}).get("last_risk_item"), "delta": int(delta["risk_delta"])},
                {"type": "value_delta", "item": state.get("memory_refs", {}).get("last_sponsor_item"), "delta": int(delta["sponsor_delta"])},
            ],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_cost_delta_for_certificate_set":
        return {"type": "set_cost_delta", "set": state.get("memory_refs", {}).get("last_certificate_set"), "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_set_compound_cost_delta":
        return {
            "type": "compound",
            "changes": [
                {"type": "set_cost_delta", "set": state.get("memory_refs", {}).get("last_audit_set"), "delta": int(delta["audit_delta"])},
                {"type": "set_cost_delta", "set": state.get("memory_refs", {}).get("last_priority_set"), "delta": int(delta["priority_delta"])},
            ],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_forbidden_set_and_required_element":
        return {
            "type": "compound",
            "changes": [
                {"type": "forbidden_set", "set": state.get("memory_refs", {}).get("last_forbidden_set")},
                {"type": "coverage_required", "element": state.get("memory_refs", {}).get("last_required_element")},
            ],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_forbidden_item_and_capacity_replay":
        return {
            "type": "compound",
            "changes": [
                {"type": "forbidden_item", "item": state.get("memory_refs", {}).get("last_forbidden_item")},
                {"type": "capacity_set", "capacity": state.get("memory_refs", {}).get("last_budget_capacity")},
            ],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    if delta["type"] == "memory_open_cost_delta_for_discount_facility":
        return {"type": "open_cost_delta", "facility": state.get("memory_refs", {}).get("last_discount_facility"), "delta": int(delta["delta"]), "resolved_from_memory": delta.get("memory_reference_chain", [])}
    if delta["type"] == "memory_facility_compound_cost_delta":
        pair = state.get("memory_refs", {}).get("last_service_pair") or {}
        return {
            "type": "compound",
            "changes": [
                {"type": "serve_cost_delta", "customer": pair.get("customer"), "facility": pair.get("facility"), "delta": int(delta["service_delta"])},
                {"type": "open_cost_delta", "facility": state.get("memory_refs", {}).get("last_cost_facility"), "delta": int(delta["open_delta"])},
            ],
            "resolved_from_memory": delta.get("memory_reference_chain", []),
        }
    return delta


def _apply_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    state = json.loads(json.dumps(state))
    typ = delta["type"]
    if typ == "no_op":
        return state
    if typ == "capacity_delta":
        state["capacity"] += int(delta["delta"])
        state.setdefault("memory_refs", {})["last_budget_capacity"] = state["capacity"]
    elif typ == "capacity_set":
        if delta.get("capacity") is not None:
            state["capacity"] = int(delta["capacity"])
    elif typ == "mandatory_item":
        state.setdefault("mandatory", [])
        if delta["item"] not in state["mandatory"]:
            state["mandatory"].append(delta["item"])
            state.setdefault("memory_refs", {})["last_mandatory_item"] = delta["item"]
    elif typ == "forbidden_item":
        state.setdefault("forbidden", [])
        if delta.get("item") and delta["item"] not in state["forbidden"]:
            state["forbidden"].append(delta["item"])
            state.setdefault("memory_refs", {})["last_forbidden_item"] = delta["item"]
    elif typ == "value_delta":
        for item in state["items"]:
            if item["id"] == delta["item"]:
                item["value"] += int(delta["delta"])
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["item"]
    elif typ == "objective_mode":
        state["objective_mode"] = str(delta["mode"])
        if delta.get("value_floor") is not None:
            state["value_floor"] = float(delta["value_floor"])
    elif typ == "forbidden_facility":
        state.setdefault("forbidden", [])
        if delta.get("facility") and delta["facility"] not in state["forbidden"]:
            state["forbidden"].append(delta["facility"])
            state.setdefault("memory_refs", {})["last_forbidden_facility"] = delta["facility"]
    elif typ == "mandatory_facility":
        state.setdefault("mandatory", [])
        if delta["facility"] not in state["mandatory"]:
            state["mandatory"].append(delta["facility"])
            state.setdefault("memory_refs", {})["last_mandatory_facility"] = delta["facility"]
    elif typ == "demand_delta":
        for customer in state["customers"]:
            if customer["id"] == delta["customer"]:
                customer["demand"] += int(delta["delta"])
                state.setdefault("memory_refs", {})["last_demand_customer"] = delta["customer"]
    elif typ == "open_cost_delta":
        for facility in state["facilities"]:
            if facility["id"] == delta["facility"]:
                facility["open_cost"] += int(delta["delta"])
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["facility"]
    elif typ == "serve_cost_delta":
        for customer in state["customers"]:
            if customer["id"] == delta.get("customer") and delta.get("facility") in customer.get("serve_cost", {}):
                customer["serve_cost"][delta["facility"]] += int(delta["delta"])
                current_time = customer.get("serve_time", {}).get(delta["facility"], customer["serve_cost"][delta["facility"]])
                customer.setdefault("serve_time", {})[delta["facility"]] = max(1, current_time + int(delta["delta"]))
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = {"customer": delta["customer"], "facility": delta["facility"]}
    elif typ == "coverage_required":
        for element in state.get("elements", []):
            if element["id"] == delta.get("element"):
                element["active"] = True
                state.setdefault("memory_refs", {})["last_required_element"] = delta["element"]
    elif typ == "mandatory_set":
        state.setdefault("mandatory", [])
        if delta["set"] not in state["mandatory"]:
            state["mandatory"].append(delta["set"])
            state.setdefault("memory_refs", {})["last_mandatory_set"] = delta["set"]
    elif typ == "forbidden_set":
        state.setdefault("forbidden", [])
        if delta.get("set") and delta["set"] not in state["forbidden"]:
            state["forbidden"].append(delta["set"])
            state.setdefault("memory_refs", {})["last_forbidden_set"] = delta["set"]
    elif typ == "set_cost_delta":
        for set_row in state.get("sets", []):
            if set_row["id"] == delta.get("set"):
                set_row["cost"] += int(delta["delta"])
                if delta.get("memory_key"):
                    state.setdefault("memory_refs", {})[str(delta["memory_key"])] = delta["set"]
    elif typ == "compound":
        for change in delta.get("changes", []):
            state = _apply_delta(state, change)
    return state


def _reference_row(update_id: str, state: dict[str, Any], delta: dict[str, Any] | None) -> dict[str, Any]:
    if state["family"] == "knapsack":
        solution, objective = _solve_knapsack_exact(state)
    elif state["family"] == "set_cover":
        solution, objective = _solve_set_cover_exact(state)
    else:
        solution, objective = _solve_facility_exact(state)
    return {
        "update_id": update_id,
        "reference_type": "exact_optimum",
        "objective": objective,
        "lower_bound": objective,
        "upper_bound": objective,
        "gap": 0.0,
        "solver": "enumeration_exact",
        "hidden_delta_type": (delta or {}).get("type"),
        "solution": solution,
    }


def _solve_knapsack_exact(state: dict[str, Any]) -> tuple[dict[str, Any], float]:
    items = state["items"]
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    best_objective = float("inf")
    best = []
    for mask in range(1 << len(items)):
        chosen = [items[i] for i in range(len(items)) if mask & (1 << i)]
        ids = {i["id"] for i in chosen}
        if not mandatory <= ids or ids & forbidden:
            continue
        weight = sum(i["weight"] for i in chosen)
        if weight > state["capacity"]:
            continue
        value = sum(i["value"] for i in chosen)
        if str(state.get("objective_mode", "maximize_value")) == "minimize_weight_with_value_floor" and value < float(state.get("value_floor", 0)):
            continue
        objective = _knapsack_objective_value(state, value, weight)
        if objective < best_objective:
            best_objective = objective
            best = sorted(ids)
    return {"selected_items": best}, float(best_objective)


def _knapsack_objective_value(state: dict[str, Any], value: float, weight: float) -> float:
    if str(state.get("objective_mode", "maximize_value")) == "minimize_weight_with_value_floor":
        return float(weight)
    return float(-value)


def _solve_facility_exact(state: dict[str, Any]) -> tuple[dict[str, Any], float]:
    facilities = state["facilities"]
    customers = state["customers"]
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    best_cost = float("inf")
    best_solution: dict[str, Any] = {}
    facility_ids = [f["id"] for f in facilities]
    facility_map = {f["id"]: f for f in facilities}
    for mask in range(1, 1 << len(facilities)):
        open_ids = {facility_ids[i] for i in range(len(facilities)) if mask & (1 << i)}
        if not mandatory <= open_ids or open_ids & forbidden:
            continue
        remaining = {fid: facility_map[fid]["capacity"] for fid in open_ids}
        cost = sum(facility_map[fid]["open_cost"] for fid in open_ids)
        service_time = 0.0
        assignment = {}
        feasible = True
        for customer in customers:
            candidates = sorted(open_ids, key=lambda fid: _facility_assignment_metric(state, customer, fid))
            chosen = None
            for fid in candidates:
                if remaining[fid] >= customer["demand"]:
                    chosen = fid
                    break
            if chosen is None:
                feasible = False
                break
            remaining[chosen] -= customer["demand"]
            assignment[customer["id"]] = chosen
            cost += customer["serve_cost"][chosen] * customer["demand"]
            service_time += float(customer.get("serve_time", {}).get(chosen, customer["serve_cost"][chosen])) * customer["demand"]
        objective = _facility_objective_value(state, cost, service_time)
        if feasible and objective < best_cost:
            best_cost = objective
            best_solution = {"open_facilities": sorted(open_ids), "assignments": assignment}
    return best_solution, float(best_cost)


def _facility_assignment_metric(state: dict[str, Any], customer: dict[str, Any], facility_id: str) -> float:
    if str(state.get("objective_mode", "minimize_cost")) == "minimize_service_time":
        return float(customer.get("serve_time", {}).get(facility_id, customer.get("serve_cost", {}).get(facility_id, 1e6)))
    return float(customer.get("serve_cost", {}).get(facility_id, 1e6))


def _facility_objective_value(state: dict[str, Any], cost: float, service_time: float) -> float:
    if str(state.get("objective_mode", "minimize_cost")) == "minimize_service_time":
        return float(service_time)
    return float(cost)


def _solve_set_cover_exact(state: dict[str, Any]) -> tuple[dict[str, Any], float]:
    sets = state.get("sets", [])
    active_elements = {str(item["id"]) for item in state.get("elements", []) if item.get("active", True)}
    mandatory = set(state.get("mandatory", []))
    forbidden = set(state.get("forbidden", []))
    best_cost = float("inf")
    best: list[str] = []
    for mask in range(1 << len(sets)):
        chosen = [sets[i] for i in range(len(sets)) if mask & (1 << i)]
        ids = {str(item["id"]) for item in chosen}
        if not mandatory <= ids or ids & forbidden:
            continue
        covered = set()
        for item in chosen:
            covered.update(str(x) for x in item.get("covers", []))
        if not active_elements <= covered:
            continue
        cost = sum(float(item.get("cost", 0)) for item in chosen)
        if cost < best_cost:
            best_cost = cost
            best = sorted(ids)
    return {"selected_sets": best}, float(best_cost)


def _event_types(delta: dict[str, Any]) -> list[str]:
    typ = delta["type"]
    if typ == "compound":
        events = []
        for change in delta.get("changes", []):
            events.extend(_event_types(change))
        return list(dict.fromkeys(events))
    if typ.startswith("memory"):
        return ["memory_reference", "objective_change"]
    if typ in {"mandatory_item", "forbidden_item", "forbidden_facility", "mandatory_facility", "mandatory_set", "forbidden_set", "coverage_required"}:
        return ["legacy_hard_constraint_change"]
    if typ == "no_op":
        return ["re_evaluate_current_state"]
    if typ in {"capacity_delta", "capacity_set", "demand_delta"}:
        return ["legacy_feasible_domain_change"]
    return ["objective_change"]


def _expected_event_types(original: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    events = _event_types(resolved)
    if original["type"].startswith("memory") and "memory_reference" not in events:
        return ["memory_reference", *events]
    return events


def _difficulty(delta: dict[str, Any]) -> str:
    if delta["type"] == "no_op":
        return "maintenance"
    if delta["type"] == "compound":
        return "objective"
    if delta["type"].startswith("memory"):
        return "memory"
    if delta["type"] in {"mandatory_item", "forbidden_item", "forbidden_facility", "mandatory_facility", "mandatory_set", "forbidden_set", "coverage_required"}:
        return "legacy_constraint"
    return "simple"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize CO-Bench Exact-Small episodes with exact reference trajectories.")
    parser.add_argument("--output", default="data/evo2_dynoptbench/compressed_plan/cobench_exact_small_6episodes.jsonl")
    args = parser.parse_args()
    episodes = build_episodes()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(ep, ensure_ascii=False) for ep in episodes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "episodes": len(episodes), "updates": sum(len(ep["update_stream"]) for ep in episodes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
