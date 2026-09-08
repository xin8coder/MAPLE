from __future__ import annotations

from scripts.analysis.evaluate_green_agent_run import _canonical_solution as canonical_green_solution
from scripts.analysis.evaluate_green_agent_run import _green_constraint_report
from scripts.analysis.evaluate_reference_metrics import canonical_facility, canonical_knapsack, canonical_roster, canonical_set_cover, normalized_scalar_score, score_green


def test_canonical_facility_recovers_prefix_solver_values() -> None:
    solution = {
        "solver_values": {
            "open_F1": 1.0,
            "open_F2": 0.0,
            "x_C01_F1": 1.0,
            "x_C01_F2": 0.0,
            "x_C_02_F1": 1.0,
            "x_C_02_F2": 0.0,
        }
    }

    assert canonical_facility(solution) == {
        "open_facilities": ["F1"],
        "assignments": {"C01": "F1", "C_02": "F1"},
    }


def test_canonical_knapsack_recovers_binary_solver_values() -> None:
    solution = {"solver_values": {"x_I01": 1.0, "x_I02": 0.0, "x_I03": 1.0}}

    assert canonical_knapsack(solution) == {"selected_items": ["I01", "I03"]}


def test_canonical_set_cover_recovers_solver_values() -> None:
    solution = {"solver_values": {"S01": 1.0, "S02": 0.0, "S03": 1.0}}

    assert canonical_set_cover(solution) == {"selected_sets": ["S01", "S03"]}


def test_canonical_roster_accepts_staff_to_day_shift_pairs() -> None:
    solution = {
        "roster": {
            "N001": [["D1", "day"], ["D2", "off"], ["D3", "night"]],
            "N002": [{"day": "D1", "shift": "evening"}],
        }
    }

    assert canonical_roster(solution) == {
        "N001|D1": "day",
        "N001|D3": "night",
        "N002|D1": "evening",
    }


def test_zero_reference_scalar_score_does_not_collapse_feasible_gap_to_zero() -> None:
    assert normalized_scalar_score(0.0, 0.0, True) == 1.0
    assert normalized_scalar_score(3.0, 0.0, True) == 0.25
    assert normalized_scalar_score(3.0, 0.0, False) == 0.0


def test_green_route_dict_accepts_route_key_alias() -> None:
    solution = {
        "routes": {
            "V1": {"route": ["O1", "O2"], "distance": 12.0},
            "V2": {"orders": ["O3"]},
        }
    }

    assert canonical_green_solution(solution)["routes"] == [
        {"vehicle": "V1", "orders": ["O1", "O2"]},
        {"vehicle": "V2", "orders": ["O3"]},
    ]


def test_hidden_green_checker_rejects_repeated_vehicle_and_zeros_hv() -> None:
    state = {
        "depot": {"x": 0.0, "y": 0.0},
        "orders": [
            {"id": "O1", "x": 1.0, "y": 0.0, "demand": 1.0, "ready": 0.0, "due": 10.0, "service": 0.0, "priority": 1.0, "active": True},
            {"id": "O2", "x": 0.0, "y": 1.0, "demand": 1.0, "ready": 0.0, "due": 10.0, "service": 0.0, "priority": 1.0, "active": True},
        ],
        "vehicles": [
            {"id": "V1", "capacity": 2.0, "available": True, "emission_rate": 1.0, "shift_end": 100.0},
        ],
        "carbon_multiplier": 1.0,
    }
    solution = {
        "routes": [
            {"vehicle": "V1", "orders": ["O1"]},
            {"vehicle": "V1", "orders": ["O2"]},
        ]
    }
    reference_step = {
        "pareto_archive": [
            {"solution": {"routes": [{"vehicle": "V1", "orders": ["O1", "O2"]}]}, "objectives": {"distance": 3.414, "lateness": 0.0, "emission": 3.414}},
        ],
        "hypervolume_reference_point": {"distance": 10.0, "lateness": 10.0, "emission": 10.0},
        "hypervolume_ideal_point": {"distance": 0.0, "lateness": 0.0, "emission": 0.0},
    }

    report = _green_constraint_report(state, solution)
    metric = score_green(state, solution, None, reference_step, [{"solution": solution}])

    assert report == {"feasible": False, "violations": {"duplicate_vehicles": ["V1"]}}
    assert metric["feasible"] is False
    assert metric["hidden_candidate_archive_size"] == 0
    assert metric["normalized_hv"] == 0.0
