from __future__ import annotations

import random

import pytest

from evo2.agents.liveopt_workbench_impl import LiveOptWorkbenchGenerator, compile_scaffold_project
from evo2.agents.liveopt_workbench_protocol import build_scaffold_prompt, runnable_mvp_project_code
from evo2.core.template_optimizer import EvolutionConfig, SegmentSpec, crossover_genome, mutate_genome, random_genome
from evo2.skills.solver.linear_programming_solver import solution_from_linear_program_spec


class FakeScaffoldClient:
    def __init__(self, response: str):
        self.response = response
        self.payloads = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        return {"choices": [{"message": {"content": self.response}}], "usage": {"total_tokens": len(self.response.split())}}


class FakeSequenceScaffoldClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        response = self.responses.pop(0)
        return {"choices": [{"message": {"content": response}}], "usage": {"total_tokens": len(response.split())}}


SCAFFOLD_RESPONSE = """
### setup.py
```python
def build_problem(public_context):
    raw = public_context["public_data"]
    tasks = [{"id": row["job_code"], "load": int(row["hours"])} for row in raw["jobs"]]
    workers = [
        {"id": row["worker_code"], "capacity": int(row["max_hours"]), "cost": float(row["rate"])}
        for row in raw["staff"]
    ]
    data = {
        "tasks": tasks,
        "workers": workers,
        "worker_by_id": {row["id"]: row for row in workers},
    }
    return {
        "data": data,
        "segments": [
            {
                "name": "assign",
                "kind": "assignment",
                "demands": [row["id"] for row in tasks],
                "resources": [row["id"] for row in workers],
            }
        ],
        "sense": "min",
        "objective_names": ["cost"],
        "constraint_names": ["capacity"],
    }
```
### fitness.py
```python
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    workers = data["worker_by_id"]
    used = {worker_id: 0 for worker_id in workers}
    cost = 0.0
    missing = 0
    for task in data["tasks"]:
        worker_id = assignments.get(task["id"])
        if worker_id not in workers:
            missing += 1
            continue
        used[worker_id] += task["load"]
        cost += task["load"] * workers[worker_id]["cost"]
    excess = sum(max(0, used[worker_id] - workers[worker_id]["capacity"]) for worker_id in workers)
    violations = {}
    if missing:
        violations["missing_tasks"] = missing
    if excess:
        violations["capacity_excess"] = excess
    solution = {"assignments": assignments, "used_capacity": used}
    return penalty_result(cost, violations, objectives=[cost], solution=solution)
```
"""


def test_scaffold_prompt_uses_liveopt_workbench_without_ir() -> None:
    prompt = build_scaffold_prompt("assign jobs", {"public_data": {"jobs": [{"job_code": "A"}]}})

    assert "fixed LiveOpt Workbench" in prompt
    assert "Do not generate a separate intermediate model" in prompt
    assert "Runnable MVP editing base" in prompt
    assert "### setup.py" in prompt
    assert "### fitness.py" in prompt
    assert '"kind": "assignment"' in prompt
    assert '"kind": "permutation"' in prompt
    assert '"kind": "binary_vector"' in prompt
    assert '"kind": "real_vector"' in prompt
    assert '"solver_mode": "linear_mip" | "scalar_ga" | "moea"' in prompt
    assert 'solver_mode="linear_mip"' in prompt
    assert "Do not use enumeration" in prompt
    assert "several soft terms" in prompt
    assert "def build_problem(public_context):" in runnable_mvp_project_code()
    assert "def evaluate(genome, data):" in runnable_mvp_project_code()


def test_scaffold_prompt_exposes_public_multiobjective_contract() -> None:
    prompt = build_scaffold_prompt(
        "optimize distance, lateness, and emission",
        {
            "optimization_contract": {
                "objective_mode": "multi_objective",
                "solution_cardinality": "pareto_archive",
                "required_solver_mode": "moea",
                "objective_sense": "minimize_all",
                "objective_names": ["distance", "lateness", "emission"],
                "objective_terms": [
                    {"name": "distance", "kind": "pareto_objective", "formula": "sum public route distances"},
                    {"name": "lateness", "kind": "pareto_objective", "formula": "sum public lateness"},
                    {"name": "emission", "kind": "pareto_objective", "formula": "sum public emissions"},
                ],
                "required_diagnostics": ["distance", "lateness", "emission"],
                "instruction": "Return a Pareto archive.",
            },
            "tables": {"orders": [{"id": "O1"}]},
        },
    )

    assert "Public objective/scoring specification" in prompt
    assert "objective_mode: multi_objective" in prompt
    assert "required_solver_mode: moea" not in prompt
    assert "objective_terms" in prompt
    assert "required_diagnostics" in prompt
    assert "solver_mode=\"moea\"" in prompt
    assert "fitness.py must return objectives=[...] in" in prompt
    assert "Do not collapse these objectives into one scalar" in prompt
    assert "compute each" in prompt


def test_scalar_public_contract_requires_component_diagnostics() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "single_objective",
            "required_solver_mode": "scalar_ga",
            "objective_names": ["public_roster_penalty"],
            "objective_terms": [
                {"name": "coverage_shortage", "required_diagnostic": True},
                {"name": "preference_penalty", "required_diagnostic": True},
            ],
            "required_diagnostics": ["coverage_shortage", "preference_penalty"],
            "scalar_formula": "1000*coverage_shortage + preference_weight*preference_penalty",
        }
    }
    setup = """
def build_problem(public_context):
    return {
        "data": {"nurses": ["N1"], "slots": ["D1|E"]},
        "segments": [{"name": "assign", "kind": "assignment", "demands": ["D1|E"], "resources": ["N1"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["public_roster_penalty"],
    }
"""
    missing_diag_fitness = """
def evaluate(genome, data):
    coverage_shortage = 0.0
    preference_penalty = 0.0
    base = 1000.0 * coverage_shortage + preference_penalty
    return penalty_result(base, {}, objectives=[base], solution={"assignments": dict(genome["assign"])}, diagnostics={"coverage_shortage": coverage_shortage})
"""

    with pytest.raises(ValueError, match="requires fitness.py diagnostics"):
        compile_scaffold_project("bad_scalar_contract", context, {"setup.py": setup, "fitness.py": missing_diag_fitness})

    good_fitness = """
def evaluate(genome, data):
    coverage_shortage = 0.0
    preference_penalty = 0.0
    base = 1000.0 * coverage_shortage + preference_penalty
    return penalty_result(
        base,
        {},
        objectives=[base],
        solution={"assignments": dict(genome["assign"])},
        diagnostics={"coverage_shortage": coverage_shortage, "preference_penalty": preference_penalty},
    )
"""
    project = compile_scaffold_project("good_scalar_contract", context, {"setup.py": setup, "fitness.py": good_fitness})

    assert project.run(EvolutionConfig(population_size=4, generations=1, seed=0)).best.result.feasible


def test_exact_linear_solver_contract_rejects_scalar_ga_fallback() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "exact_single_objective",
            "required_solver_mode": "linear_mip_when_representable",
            "objective_names": ["public_cost"],
        }
    }
    setup = """
def build_problem(public_context):
    return {
        "data": {"items": ["A"]},
        "segments": [{"name": "x", "kind": "binary_vector", "length": 1}],
        "solver_mode": "scalar_ga",
        "objective_names": ["public_cost"],
    }
"""
    fitness = """
def evaluate(genome, data):
    value = float(genome["x"][0])
    return penalty_result(value, {}, objectives=[value], solution={"genome": list(genome["x"])})
"""

    with pytest.raises(ValueError, match="requires solver_mode='linear_mip'"):
        compile_scaffold_project("bad_exact_solver_mode", context, {"setup.py": setup, "fitness.py": fitness})


def test_exact_linear_solver_contract_accepts_complete_mip_spec() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "exact_single_objective",
            "required_solver_mode": "linear_mip_when_representable",
            "objective_names": ["public_cost"],
        }
    }
    project = compile_scaffold_project(
        "good_exact_solver_mode",
        context,
        {
            "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [],
        "solver_mode": "linear_mip",
        "linear_program_spec": {
            "sense": "minimize",
            "variables": [{"name": "x_A", "type": "binary", "lb": 0, "ub": 1}],
            "objective": {"coefficients": {"x_A": 1.0}},
            "constraints": [{"name": "choose", "coefficients": {"x_A": 1.0}, "sense": "=", "rhs": 1.0}],
            "solution_extraction": {
                "type": "binary_selection",
                "variables": [{"var": "x_A", "id": "A"}],
                "selected_field": "selected_items",
            },
        },
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    return penalty_result(0, {}, objectives=[0], solution={})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=4, generations=1, seed=0))

    assert result.metadata["selection"] == "linear_mip"
    assert result.best.result.solution["selected_items"] == ["A"]


def test_public_contract_rejects_ignored_dynamic_objective_table() -> None:
    context = {
        "tables": {"preferences": [], "policy": [{"preference_weight": 35}]},
        "optimization_contract": {
            "objective_mode": "single_objective",
            "required_solver_mode": "scalar_ga",
            "objective_names": ["public_score"],
            "objective_terms": [
                {
                    "name": "preference_penalty",
                    "required_diagnostic": True,
                    "required_public_tables": ["preferences", "policy"],
                }
            ],
            "required_diagnostics": ["preference_penalty"],
            "scalar_formula": "policy.preference_weight*preference_penalty",
        },
    }
    setup = """
def build_problem(public_context):
    return {
        "data": {"tasks": ["T1"], "workers": ["W1"], "preferences": public_context["tables"].get("preferences", []), "policy": public_context["tables"].get("policy", [{}])[0]},
        "segments": [{"name": "assign", "kind": "assignment", "demands": ["T1"], "resources": ["W1"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["public_score"],
    }
"""
    bad_fitness = """
def evaluate(genome, data):
    preference_penalty = 0.0
    return penalty_result(0.0, {}, objectives=[0.0], solution={"assignments": dict(genome["assign"])}, diagnostics={"preference_penalty": preference_penalty})
"""

    with pytest.raises(ValueError, match="assigned only literal constants"):
        compile_scaffold_project("bad_ignored_dynamic_term", context, {"setup.py": setup, "fitness.py": bad_fitness})

    good_fitness = """
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    preference_penalty = 0.0
    for row in data["preferences"]:
        if assignments.get(row.get("task")) == row.get("avoid_worker"):
            preference_penalty += 1.0
    score = float(data["policy"].get("preference_weight", 1.0)) * preference_penalty
    return penalty_result(score, {}, objectives=[score], solution={"assignments": assignments}, diagnostics={"preference_penalty": preference_penalty})
"""
    project = compile_scaffold_project("good_dynamic_term", context, {"setup.py": setup, "fitness.py": good_fitness})

    assert project.run(EvolutionConfig(population_size=4, generations=1, seed=0)).best.result.feasible


def test_public_contract_rejects_missing_optional_field_gate() -> None:
    context = {
        "tables": {
            "preferences": [{"nurse": "N1", "day": "D1", "preference": "off"}],
            "policy": [{"preference_weight": 35}],
        },
        "optimization_contract": {
            "objective_mode": "single_objective",
            "required_solver_mode": "scalar_ga",
            "objective_names": ["public_score"],
            "objective_terms": [
                {
                    "name": "preference_penalty",
                    "required_diagnostic": True,
                    "required_public_tables": ["preferences", "policy"],
                    "optional_field_semantics": [
                        {
                            "table": "preferences",
                            "field": "shift",
                            "when_missing": "the preference applies to the whole public day for that nurse",
                            "must_not_gate_term": True,
                        }
                    ],
                }
            ],
            "required_diagnostics": ["preference_penalty"],
            "scalar_formula": "policy.preference_weight*preference_penalty",
        },
    }
    setup = """
def build_problem(public_context):
    data = {
        "preferences": public_context["tables"]["preferences"],
        "policy": public_context["tables"]["policy"][0],
        "slot_info": {"D1|day_0": ("D1", "day")},
    }
    return {
        "data": data,
        "segments": [{"name": "assign", "kind": "assignment", "demands": ["D1|day_0"], "resources": ["N1"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["public_score"],
    }
"""
    bad_fitness = """
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    by_nurse_day = {"N1": {"D1": ["day"]}}
    preference_penalty = 0.0
    for pref_row in data["preferences"]:
        nurse = pref_row.get("nurse")
        day = pref_row.get("day")
        shift = pref_row.get("shift")
        if nurse and day and shift:
            if shift in by_nurse_day.get(nurse, {}).get(day, []):
                preference_penalty += 1.0
    score = float(data["policy"].get("preference_weight", 1.0)) * preference_penalty
    return penalty_result(score, {}, objectives=[score], solution={"roster": {"N1|D1": "day"}}, diagnostics={"preference_penalty": preference_penalty})
"""
    with pytest.raises(ValueError, match="missing optional field"):
        compile_scaffold_project("bad_missing_optional_field_gate", context, {"setup.py": setup, "fitness.py": bad_fitness})

    good_fitness = """
def evaluate(genome, data):
    assignments = dict(genome["assign"])
    by_nurse_day = {"N1": {"D1": ["day"]}}
    preference_penalty = 0.0
    for pref_row in data["preferences"]:
        nurse = pref_row.get("nurse")
        day = pref_row.get("day")
        preference = pref_row.get("preference")
        if nurse and day and preference in ("off", "avoid"):
            if by_nurse_day.get(nurse, {}).get(day, []):
                preference_penalty += 1.0
    score = float(data["policy"].get("preference_weight", 1.0)) * preference_penalty
    return penalty_result(score, {}, objectives=[score], solution={"roster": {"N1|D1": "day"}}, diagnostics={"preference_penalty": preference_penalty})
"""
    project = compile_scaffold_project("good_missing_optional_field_semantics", context, {"setup.py": setup, "fitness.py": good_fitness})

    assert project.run(EvolutionConfig(population_size=4, generations=1, seed=0)).best.result.feasible


def test_public_contract_rejects_underexpressive_search_space() -> None:
    context = {
        "tables": {"coverage": [{"day": "D1", "shift": "day", "required": 2}]},
        "optimization_contract": {
            "objective_mode": "single_objective",
            "required_solver_mode": "scalar_ga",
            "objective_names": ["roster_score"],
            "objective_terms": [{"name": "coverage_shortage", "required_diagnostic": True, "required_public_tables": ["coverage"]}],
            "required_diagnostics": ["coverage_shortage"],
            "scalar_formula": "1000*coverage_shortage",
            "search_space_requirements": [
                {
                    "name": "per_coverage_slot_staff_assignment",
                    "allowed_segment_kinds": ["assignment", "choice_vector"],
                    "min_demands_from_table": {"table": "coverage", "column": "required"},
                }
            ],
        },
    }
    bad_setup = """
def build_problem(public_context):
    return {
        "data": {"coverage": public_context["tables"]["coverage"]},
        "segments": [{"name": "priority", "kind": "permutation", "values": ["N1", "N2"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["roster_score"],
    }
"""
    fitness = """
def evaluate(genome, data):
    coverage_shortage = 0.0
    return penalty_result(coverage_shortage, {}, objectives=[coverage_shortage], solution={"roster": {}}, diagnostics={"coverage_shortage": coverage_shortage})
"""
    with pytest.raises(ValueError, match="requires one of segment kinds"):
        compile_scaffold_project("bad_underexpressive", context, {"setup.py": bad_setup, "fitness.py": fitness})

    good_setup = """
def build_problem(public_context):
    demands = ["D1|day|0", "D1|day|1"]
    return {
        "data": {"coverage": public_context["tables"]["coverage"], "demands": demands},
        "segments": [{"name": "assign", "kind": "assignment", "demands": demands, "resources": ["N1", "N2"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["roster_score"],
    }
"""
    good_fitness = """
def evaluate(genome, data):
    coverage_shortage = sum(max(0, int(row.get("required", 0)) - int(row.get("required", 0))) for row in data["coverage"])
    return penalty_result(coverage_shortage, {}, objectives=[coverage_shortage], solution={"roster": {}}, diagnostics={"coverage_shortage": coverage_shortage})
"""
    project = compile_scaffold_project("good_expressive", context, {"setup.py": good_setup, "fitness.py": good_fitness})

    assert project.segments[0].kind == "assignment"


def test_public_contract_requires_declared_group_capacity_metadata() -> None:
    context = {
        "tables": {"coverage": [{"day": "D1", "shift": "day", "required": 2}]},
        "optimization_contract": {
            "objective_mode": "single_objective",
            "required_solver_mode": "scalar_ga",
            "objective_names": ["roster_score"],
            "objective_terms": [{"name": "coverage_shortage", "required_diagnostic": True, "required_public_tables": ["coverage"]}],
            "required_diagnostics": ["coverage_shortage"],
            "scalar_formula": "1000*coverage_shortage",
            "search_space_requirements": [
                {
                    "name": "per_group_assignment",
                    "allowed_segment_kinds": ["assignment"],
                    "min_demands_from_table": {"table": "coverage", "column": "required"},
                    "required_segment_metadata": {"exclusive_resource_per_group": "one resource per demand group"},
                }
            ],
        },
    }
    setup_without_metadata = """
def build_problem(public_context):
    demands = ["D1|day|0", "D1|day|1"]
    return {
        "data": {"coverage": public_context["tables"]["coverage"]},
        "segments": [{"name": "assign", "kind": "assignment", "demands": demands, "resources": ["N1", "N2"]}],
        "solver_mode": "scalar_ga",
        "objective_names": ["roster_score"],
    }
"""
    setup_with_metadata = """
def build_problem(public_context):
    demands = ["D1|day|0", "D1|day|1"]
    return {
        "data": {"coverage": public_context["tables"]["coverage"]},
        "segments": [{
            "name": "assign",
            "kind": "assignment",
            "demands": demands,
            "resources": ["N1", "N2"],
            "exclusive_resource_per_group": {
                "demand_to_group": {"D1|day|0": "D1", "D1|day|1": "D1"},
                "capacity": 1,
            },
        }],
        "solver_mode": "scalar_ga",
        "objective_names": ["roster_score"],
    }
"""
    fitness = """
def evaluate(genome, data):
    coverage_shortage = sum(max(0, int(row.get("required", 0)) - int(row.get("required", 0))) for row in data["coverage"])
    return penalty_result(coverage_shortage, {}, objectives=[coverage_shortage], solution={"roster": {}}, diagnostics={"coverage_shortage": coverage_shortage})
"""

    with pytest.raises(ValueError, match="requires assignment segment metadata"):
        compile_scaffold_project("missing_group_capacity_metadata", context, {"setup.py": setup_without_metadata, "fitness.py": fitness})

    project = compile_scaffold_project("with_group_capacity_metadata", context, {"setup.py": setup_with_metadata, "fitness.py": fitness})

    assert project.segments[0].metadata["exclusive_resource_per_group"]["capacity"] == 1


def test_assignment_group_capacity_metadata_repairs_generic_genomes() -> None:
    demands = ["D1-day-0", "D1-night-0", "D2-day-0", "D2-night-0"]
    demand_to_group = {"D1-day-0": "D1", "D1-night-0": "D1", "D2-day-0": "D2", "D2-night-0": "D2"}
    segment = SegmentSpec(
        name="assign",
        kind="assignment",
        demands=demands,
        resources=["R1", "R2"],
        metadata={"exclusive_resource_per_group": {"demand_to_group": demand_to_group, "capacity": 1}},
    )
    rng = random.Random(7)

    for _ in range(20):
        genome = random_genome([segment], rng)
        _assert_group_capacity(genome["assign"], demand_to_group, capacity=1)

    parent_a = {"assign": {demand: "R1" for demand in demands}}
    parent_b = {"assign": {demand: "R2" for demand in demands}}
    child = crossover_genome(parent_a, parent_b, [segment], rng)
    mutated = mutate_genome(child, [segment], rng, mutation_rate=1.0)

    _assert_group_capacity(child["assign"], demand_to_group, capacity=1)
    _assert_group_capacity(mutated["assign"], demand_to_group, capacity=1)


def test_assignment_resource_total_capacity_metadata_repairs_generic_genomes() -> None:
    demands = [f"D{i}" for i in range(6)]
    segment = SegmentSpec(
        name="assign",
        kind="assignment",
        demands=demands,
        resources=["R1", "R2", "R3"],
        metadata={"resource_total_capacity": {"resource_capacity": {"R1": 2, "R2": 2, "R3": 2}}},
    )
    rng = random.Random(11)

    for _ in range(20):
        genome = random_genome([segment], rng)
        _assert_resource_capacity(genome["assign"], {"R1": 2, "R2": 2, "R3": 2})

    parent_a = {"assign": {demand: "R1" for demand in demands}}
    parent_b = {"assign": {demand: "R2" for demand in demands}}
    child = crossover_genome(parent_a, parent_b, [segment], rng)
    mutated = mutate_genome(child, [segment], rng, mutation_rate=1.0)

    _assert_resource_capacity(child["assign"], {"R1": 2, "R2": 2, "R3": 2})
    _assert_resource_capacity(mutated["assign"], {"R1": 2, "R2": 2, "R3": 2})


def test_assignment_resource_open_gate_metadata_repairs_generic_genomes() -> None:
    demands = [f"D{i}" for i in range(8)]
    resources = ["R1", "R2", "R3"]
    assignment = SegmentSpec(
        name="assign",
        kind="assignment",
        demands=demands,
        resources=resources,
        metadata={"resource_open_gate": {"segment": "open_resource", "resources": resources, "open_value": 1}},
    )
    open_gate = SegmentSpec(name="open_resource", kind="binary_vector", length=len(resources))
    segments = [assignment, open_gate]
    rng = random.Random(13)

    for _ in range(20):
        genome = random_genome(segments, rng)
        _assert_assigned_resources_are_open(genome, "assign", "open_resource", resources)

    parent_a = {"assign": {demand: "R1" for demand in demands}, "open_resource": [0, 0, 0]}
    parent_b = {"assign": {demand: "R2" for demand in demands}, "open_resource": [0, 0, 0]}
    child = crossover_genome(parent_a, parent_b, segments, rng)
    mutated = mutate_genome(child, segments, rng, mutation_rate=1.0)

    _assert_assigned_resources_are_open(child, "assign", "open_resource", resources)
    _assert_assigned_resources_are_open(mutated, "assign", "open_resource", resources)


def test_segment_metadata_field_is_flattened_for_resource_gate() -> None:
    context = {"public_data": {}}
    setup = """
def build_problem(public_context):
    resources = ["R1", "R2"]
    return {
        "data": {},
        "segments": [
            {
                "name": "assign",
                "kind": "assignment",
                "demands": ["D1", "D2"],
                "resources": resources,
                "metadata": {"resource_open_gate": {"segment": "open_resource", "resources": resources, "open_value": 1}},
            },
            {"name": "open_resource", "kind": "binary_vector", "length": len(resources)},
        ],
        "solver_mode": "scalar_ga",
        "objective_names": ["score"],
    }
"""
    fitness = """
def evaluate(genome, data):
    return penalty_result(0.0, {}, objectives=[0.0], solution=genome)
"""

    project = compile_scaffold_project("metadata_flattening", context, {"setup.py": setup, "fitness.py": fitness})

    assert "resource_open_gate" in project.segments[0].metadata
    assert "metadata" not in project.segments[0].metadata


def _assert_group_capacity(assignments: dict[str, str], demand_to_group: dict[str, str], capacity: int) -> None:
    counts = {}
    for demand, resource in assignments.items():
        group = demand_to_group[demand]
        counts[(group, resource)] = counts.get((group, resource), 0) + 1
    assert all(count <= capacity for count in counts.values())


def _assert_resource_capacity(assignments: dict[str, str], capacities: dict[str, int]) -> None:
    counts = {}
    for resource in assignments.values():
        counts[resource] = counts.get(resource, 0) + 1
    for resource, count in counts.items():
        assert count <= capacities[resource]


def _assert_assigned_resources_are_open(genome: dict, assignment_name: str, gate_name: str, resources: list[str]) -> None:
    open_bits = genome[gate_name]
    resource_index = {resource: idx for idx, resource in enumerate(resources)}
    for resource in genome[assignment_name].values():
        assert open_bits[resource_index[resource]] == 1


def test_multiobjective_contract_rejects_scalarized_scaffold() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "multi_objective",
            "required_solver_mode": "moea",
            "objective_names": ["distance", "lateness", "emission"],
        }
    }

    try:
        compile_scaffold_project(
            "bad_scalarized_mo",
            context,
            {
                "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 3}],
        "solver_mode": "scalar_ga",
        "objective_names": ["total_cost"],
    }
""",
                "fitness.py": """
def evaluate(genome, data):
    total_cost = int(genome["x"][0])
    return penalty_result(total_cost, {}, objectives=[total_cost], solution={"x": total_cost})
""",
            },
        )
    except ValueError as exc:
        assert "requires solver_mode='moea'" in str(exc)
    else:
        raise AssertionError("multi-objective contract should reject scalar_ga")


def test_multiobjective_contract_rejects_objective_name_drift() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "multi_objective",
            "required_solver_mode": "moea",
            "objective_names": ["distance", "lateness", "emission"],
        }
    }

    try:
        compile_scaffold_project(
            "bad_objective_names",
            context,
            {
                "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 3}],
        "solver_mode": "moea",
        "objective_names": ["distance", "lateness", "emissions"],
    }
""",
                "fitness.py": """
def evaluate(genome, data):
    value = int(genome["x"][0])
    return penalty_result(value, {}, objectives=[value, value + 1, value + 2], solution={"x": value})
""",
            },
        )
    except ValueError as exc:
        assert "requires objective_names exactly" in str(exc)
    else:
        raise AssertionError("multi-objective contract should reject objective name drift")


def test_multiobjective_contract_rejects_literal_constant_dimension() -> None:
    context = {
        "optimization_contract": {
            "objective_mode": "multi_objective",
            "required_solver_mode": "moea",
            "objective_names": ["energy", "load_imbalance"],
        }
    }

    try:
        compile_scaffold_project(
            "bad_constant_mo_dim",
            context,
            {
                "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 3}],
        "solver_mode": "moea",
        "objective_names": ["energy", "load_imbalance"],
    }
""",
                "fitness.py": """
def evaluate(genome, data):
    imbalance = int(genome["x"][0])
    return penalty_result(imbalance, {}, objectives=[0.0, imbalance], solution={"x": imbalance})
""",
            },
        )
    except ValueError as exc:
        assert "literal constant objective dimensions" in str(exc)
    else:
        raise AssertionError("multi-objective contract should reject constant objective dimensions")


def test_multiobjective_contract_accepts_matching_moea_scaffold() -> None:
    project = compile_scaffold_project(
        "valid_mo",
        {
            "optimization_contract": {
                "objective_mode": "multi_objective",
                "required_solver_mode": "moea",
                "objective_names": ["cost", "capacity_balance"],
            }
        },
        {
            "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 3}],
        "solver_mode": "moea",
        "objective_names": ["cost", "capacity_balance"],
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    value = int(genome["x"][0])
    cost = value
    balance = abs(2 - value)
    return penalty_result(cost + balance, {}, objectives=[cost, balance], solution={"x": value})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=8, generations=3, seed=2))

    assert project.problem_spec["solver_mode"] == "moea"
    assert result.metadata["selection"] == "nsga2"


def test_one_call_scaffold_generator_builds_and_runs_project() -> None:
    client = FakeScaffoldClient(SCAFFOLD_RESPONSE)
    generator = LiveOptWorkbenchGenerator(model="fake", client=client)
    public_context = {
        "public_data": {
            "jobs": [{"job_code": "A", "hours": "2"}, {"job_code": "B", "hours": "1"}],
            "staff": [{"worker_code": "U1", "max_hours": "2", "rate": "1.0"}, {"worker_code": "U2", "max_hours": "1", "rate": "2.0"}],
        }
    }

    project = generator.generate_project("scaffold_fake", "assign jobs to staff", public_context)
    result = project.run(EvolutionConfig(population_size=12, generations=4, seed=3))

    assert len(client.payloads) == 1
    assert project.segments[0].kind == "assignment"
    assert project.segments[0].demands == ["A", "B"]
    assert project.segments[0].resources == ["U1", "U2"]
    assert result.best.result.feasible
    assert set(result.best.result.solution["assignments"]) == {"A", "B"}


def test_scaffold_generator_repairs_compile_error_with_feedback() -> None:
    bad_response = """
### setup.py
```python
def build_problem(public_context):
    raise KeyError(0)
```
### fitness.py
```python
def evaluate(genome, data):
    return penalty_result(0, {}, objectives=[0], solution={})
```
"""
    client = FakeSequenceScaffoldClient([bad_response, SCAFFOLD_RESPONSE])
    generator = LiveOptWorkbenchGenerator(model="fake", client=client)
    public_context = {
        "public_data": {
            "jobs": [{"job_code": "A", "hours": "2"}, {"job_code": "B", "hours": "1"}],
            "staff": [{"worker_code": "U1", "max_hours": "2", "rate": "1.0"}, {"worker_code": "U2", "max_hours": "1", "rate": "2.0"}],
        }
    }

    project = generator.generate_project("repair_fake", "assign jobs to staff", public_context, max_repairs=3)

    assert project.segments[0].kind == "assignment"
    assert len(client.payloads) == 2
    assert "KeyError" in client.payloads[1]["messages"][1]["content"]
    assert generator.last_trace.errors


def test_compile_scaffold_project_requires_only_setup_and_fitness() -> None:
    project = compile_scaffold_project(
        "compiled",
        {
            "public_data": {
                "jobs": [{"job_code": "A", "hours": "1"}],
                "staff": [{"worker_code": "U1", "max_hours": "1", "rate": "1"}],
            }
        },
        {
            "setup.py": SCAFFOLD_RESPONSE.split("### setup.py", 1)[1].split("### fitness.py", 1)[0].split("```python", 1)[1].rsplit("```", 1)[0].strip(),
            "fitness.py": SCAFFOLD_RESPONSE.split("### fitness.py", 1)[1].split("```python", 1)[1].rsplit("```", 1)[0].strip(),
        },
    )

    assert sorted(project.problem_spec) == ["constraint_names", "data", "objective_names", "segments", "sense"]


def test_exact_enumeration_solver_mode_finds_small_subset_optimum() -> None:
    project = compile_scaffold_project(
        "exact_subset",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    data = {
        "items": [
            {"id": "A", "weight": 4, "value": 7},
            {"id": "B", "weight": 3, "value": 6},
            {"id": "C", "weight": 5, "value": 3},
        ],
        "capacity": 7,
    }
    return {
        "data": data,
        "segments": [{"name": "select", "kind": "binary_vector", "length": 3}],
        "solver_mode": "exact_enumeration",
        "max_exact_candidates": 16,
        "sense": "min",
        "objective_names": ["negative_value"],
        "constraint_names": ["capacity"],
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    flags = [int(x) for x in genome["select"]]
    weight = 0
    value = 0
    selected = []
    for flag, item in zip(flags, data["items"]):
        if flag:
            weight += item["weight"]
            value += item["value"]
            selected.append(item["id"])
    violations = {}
    if weight > data["capacity"]:
        violations["capacity_excess"] = weight - data["capacity"]
    return penalty_result(-value, violations, objectives=[-value], solution={"selected": selected, "weight": weight})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=8, generations=50, seed=4))

    assert result.metadata["selection"] == "exact_enumeration"
    assert result.metadata["enumerated_candidates"] == 8
    assert result.best.result.feasible
    assert result.best.result.solution["selected"] == ["A", "B"]
    assert result.best.result.base_scalar == -13


def test_linear_mip_solver_mode_solves_small_subset_without_segments() -> None:
    project = compile_scaffold_project(
        "linear_mip_subset",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    items = [
        {"id": "A", "weight": 4, "value": 7},
        {"id": "B", "weight": 3, "value": 6},
        {"id": "C", "weight": 5, "value": 3},
    ]
    variables = [{"name": "x_" + item["id"], "type": "binary", "lb": 0, "ub": 1} for item in items]
    objective = {"coefficients": {"x_" + item["id"]: item["value"] for item in items}}
    capacity = {"name": "capacity", "coefficients": {"x_" + item["id"]: item["weight"] for item in items}, "sense": "<=", "rhs": 7}
    extraction = {
        "type": "binary_selection",
        "variables": [{"var": "x_" + item["id"], "id": item["id"]} for item in items],
        "selected_field": "selected_items",
    }
    return {
        "data": {"items": items},
        "segments": [],
        "solver_mode": "linear_mip",
        "linear_program_spec": {
            "sense": "maximize",
            "variables": variables,
            "objective": objective,
            "constraints": [capacity],
            "solution_extraction": extraction,
        },
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    return penalty_result(0, {}, objectives=[0], solution={})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=4, generations=10, seed=9))

    assert project.segments == []
    assert result.metadata["selection"] == "linear_mip"
    assert result.metadata["solver"]["solver"] == "scipy.milp"
    assert result.best.result.feasible
    assert result.best.result.solution["selected_items"] == ["A", "B"]
    assert result.best.result.diagnostics["solver_objective_value"] == 13
    assert result.best.result.scalar == -13


def test_linear_mip_generic_assignment_extracts_variable_map() -> None:
    project = compile_scaffold_project(
        "linear_mip_assignment",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    meetings = ["M-A", "M-B"]
    resources = ["R1|T1", "R1|T2"]
    variables = []
    objective = {}
    constraints = []
    mapping = {}
    for meeting in meetings:
        coeffs = {}
        for resource in resources:
            room, slot = resource.split("|")
            var = "x_" + meeting + "_" + room + "_" + slot
            variables.append({"name": var, "type": "binary", "lb": 0, "ub": 1})
            objective[var] = 0 if meeting.endswith("A") and slot == "T1" else 1
            coeffs[var] = 1
            mapping[var] = [meeting, resource]
        constraints.append({"name": "assign_" + meeting, "coefficients": coeffs, "sense": "=", "rhs": 1})
    for resource in resources:
        coeffs = {}
        room, slot = resource.split("|")
        for meeting in meetings:
            coeffs["x_" + meeting + "_" + room + "_" + slot] = 1
        constraints.append({"name": "capacity_" + resource, "coefficients": coeffs, "sense": "<=", "rhs": 1})
    return {
        "data": {},
        "segments": [{"name": "unused", "kind": "binary_vector", "length": 0}],
        "solver_mode": "linear_mip",
        "linear_program_spec": {
            "sense": "minimize",
            "variables": variables,
            "objective": {"coefficients": objective},
            "constraints": constraints,
            "solution_extraction": {"type": "generic_assignment", "assignment_from_variables": mapping},
        },
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    return penalty_result(0, {}, objectives=[0], solution={})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=4, generations=2, seed=0))

    assert project.segments == []
    assert result.best.result.solution["assignments"] == {"M-A": "R1|T1", "M-B": "R1|T2"}


def test_linear_mip_precedence_schedule_rejects_machine_overlap() -> None:
    project = compile_scaffold_project(
        "bad_overlap_schedule",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    return {
        "data": {"job_ops": {"J1": ["O1"], "J2": ["O2"]}},
        "segments": [],
        "solver_mode": "linear_mip",
        "linear_program_spec": {
            "sense": "minimize",
            "variables": [
                {"name": "start_O1", "type": "continuous", "lb": 0, "ub": 0},
                {"name": "start_O2", "type": "continuous", "lb": 0, "ub": 0},
                {"name": "assign_O1_M1", "type": "binary", "lb": 1, "ub": 1},
                {"name": "assign_O2_M1", "type": "binary", "lb": 1, "ub": 1},
            ],
            "objective": {"coefficients": {"start_O1": 0, "start_O2": 0}},
            "constraints": [],
            "solution_extraction": {
                "type": "precedence_schedule",
                "operations": [
                    {"id": "O1", "start_var": "start_O1", "processing_time": 5, "machine_vars": [{"machine": "M1", "var": "assign_O1_M1"}]},
                    {"id": "O2", "start_var": "start_O2", "processing_time": 5, "machine_vars": [{"machine": "M1", "var": "assign_O2_M1"}]},
                ],
            },
        },
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    return penalty_result(0, {}, objectives=[0], solution={})
""",
        },
    )

    with pytest.raises(ValueError, match="overlap"):
        project.run(EvolutionConfig(population_size=4, generations=1, seed=0))


def test_workbench_rejects_row_dict_as_solution_identifier() -> None:
    setup = """
def build_problem(public_context):
    vehicles = [{"id": "V1", "capacity": 10}]
    return {
        "data": {"vehicles": vehicles},
        "segments": [{"name": "x", "kind": "binary_vector", "length": 1}],
        "solver_mode": "scalar_ga",
        "sense": "min",
        "objective_names": ["cost"],
        "constraint_names": [],
    }
"""
    fitness = """
def evaluate(genome, data):
    return penalty_result(
        0.0,
        {},
        objectives=[0.0],
        solution={"routes": [{"vehicle": data["vehicles"][0], "orders": ["O1"]}]},
    )
"""
    project = compile_scaffold_project("bad_public_id", {}, {"setup.py": setup, "fitness.py": fitness})

    with pytest.raises(ValueError, match="scalar public identifier"):
        project.run(EvolutionConfig(population_size=4, generations=1, seed=0))


def test_open_resource_assignment_extracts_prefix_only_variables() -> None:
    spec = {
        "variables": [
            {"name": "open_F1", "type": "binary"},
            {"name": "open_F2", "type": "binary"},
            {"name": "x_C01_F1", "type": "binary"},
            {"name": "x_C01_F2", "type": "binary"},
            {"name": "x_C_02_F1", "type": "binary"},
            {"name": "x_C_02_F2", "type": "binary"},
        ],
        "solution_extraction": {
            "type": "open_resource_assignment",
            "open_var_prefix": "open_",
            "assign_var_prefix": "x_",
            "resources": ["F1", "F2"],
        },
    }
    solution = solution_from_linear_program_spec(
        spec,
        {"open_F1": 1.0, "open_F2": 0.0, "x_C01_F1": 1.0, "x_C01_F2": 0.0, "x_C_02_F1": 1.0, "x_C_02_F2": 0.0},
    )

    assert solution["open_facilities"] == ["F1"]
    assert solution["assignments"] == {"C01": "F1", "C_02": "F1"}


def test_open_resource_assignment_infers_assign_prefix_variables() -> None:
    spec = {
        "variables": [
            {"name": "open_F1", "type": "binary"},
            {"name": "open_F2", "type": "binary"},
            {"name": "assign_C01_F1", "type": "binary"},
            {"name": "assign_C01_F2", "type": "binary"},
            {"name": "assign_C_02_F1", "type": "binary"},
            {"name": "assign_C_02_F2", "type": "binary"},
        ],
        "solution_extraction": {"type": "open_resource_assignment"},
    }
    solution = solution_from_linear_program_spec(
        spec,
        {
            "open_F1": 1.0,
            "open_F2": 0.0,
            "assign_C01_F1": 1.0,
            "assign_C01_F2": 0.0,
            "assign_C_02_F1": 1.0,
            "assign_C_02_F2": 0.0,
        },
    )

    assert solution["open_facilities"] == ["F1"]
    assert solution["assignments"] == {"C01": "F1", "C_02": "F1"}


def test_binary_selection_accepts_string_variable_list() -> None:
    solution = solution_from_linear_program_spec(
        {
            "variables": [
                {"name": "x_I01", "type": "binary"},
                {"name": "x_I02", "type": "binary"},
            ],
            "solution_extraction": {"type": "binary_selection", "variables": ["x_I01", "x_I02"]},
        },
        {"x_I01": 1.0, "x_I02": 0.0},
    )

    assert solution["selected_items"] == ["I01"]
    assert solution["genome"] == [1, 0]


def test_exact_enumeration_rejects_continuous_segments_explicitly() -> None:
    project = compile_scaffold_project(
        "bad_exact",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    return {
        "data": {},
        "segments": [{"name": "x", "kind": "real_vector", "length": 2, "lower": 0, "upper": 1}],
        "solver_mode": "exact_enumeration",
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    return penalty_result(sum(genome["x"]), {}, objectives=[sum(genome["x"])], solution={})
""",
        },
    )

    try:
        project.run(EvolutionConfig(population_size=4, generations=1))
    except ValueError as exc:
        assert "cannot be exactly enumerated" in str(exc)
    else:
        raise AssertionError("continuous exact enumeration should fail")


def test_scalar_ga_solver_mode_keeps_single_objective_selection() -> None:
    project = compile_scaffold_project(
        "scalar_ga",
        {},
        {
            "setup.py": """
def build_problem(public_context):
    return {
        "data": {"target": 3},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 8}],
        "solver_mode": "scalar_ga",
        "sense": "min",
        "objective_names": ["distance"],
    }
""",
            "fitness.py": """
def evaluate(genome, data):
    value = int(genome["x"][0])
    distance = abs(value - data["target"])
    return penalty_result(distance, {}, objectives=[distance], solution={"x": value})
""",
        },
    )

    result = project.run(EvolutionConfig(population_size=12, generations=12, seed=1))

    assert result.metadata["selection"] == "scalar_ga"
    assert result.best.result.solution["x"] == 3
    assert result.best.result.scalar == 0
