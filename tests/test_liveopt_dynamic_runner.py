from __future__ import annotations

from evo2.agents.liveopt_dynamic_impl import (
    DynamicUpdateImpact,
    LiveOptDataPatcher,
    LiveOptDynamicRunner,
    LiveOptWorkbenchPatcher,
    ScaffoldRestartSeedBuilder,
    apply_public_context_patch,
    build_data_patch_prompt,
    estimate_objective_space_shift,
    select_restart_skill_from_landscape_change,
    dynamic_patch_contract_errors,
)
from evo2.agents.liveopt_workbench_impl import compile_scaffold_project
from evo2.core.template_optimizer import EvolutionConfig, EvolutionResult, coerce_fitness_result
from evo2.core.template_optimizer import Candidate, FitnessResult, _archive


class FakePatchClient:
    def __init__(self, response: str):
        self.response = response
        self.payloads = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        return {"choices": [{"message": {"content": self.response}}], "usage": {"total_tokens": len(self.response.split())}}


class FakeSequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        response = self.responses.pop(0)
        return {"choices": [{"message": {"content": response}}], "usage": {"total_tokens": len(response.split())}}


def test_pareto_archive_is_nondominated_and_objective_unique() -> None:
    candidates = [
        Candidate({"id": "a"}, FitnessResult(scalar=2.0, objectives=[1.0, 1.0], feasible=True)),
        Candidate({"id": "a-duplicate"}, FitnessResult(scalar=2.0, objectives=[1.0, 1.0], feasible=True)),
        Candidate({"id": "dominated"}, FitnessResult(scalar=4.0, objectives=[2.0, 2.0], feasible=True)),
        Candidate({"id": "tradeoff"}, FitnessResult(scalar=2.5, objectives=[0.5, 2.0], feasible=True)),
    ]

    archive = _archive(candidates, [], 200)

    assert [candidate.result.objectives for candidate in archive] == [[1.0, 1.0], [0.5, 2.0]]


SETUP_CODE = """
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
"""


FITNESS_CODE = """
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
"""


PATCHED_FITNESS_RESPONSE = """
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
        cost += 2.0 * task["load"] * workers[worker_id]["cost"]
    excess = sum(max(0, used[worker_id] - workers[worker_id]["capacity"]) for worker_id in workers)
    violations = {}
    if missing:
        violations["missing_tasks"] = missing
    if excess:
        violations["capacity_excess"] = excess
    solution = {"assignments": assignments, "used_capacity": used, "objective_scale": 2.0}
    return penalty_result(cost, violations, objectives=[cost], solution=solution)
```
"""


def public_context(hours_a: str = "2", hours_b: str = "1"):
    return {
        "public_data": {
            "jobs": [{"job_code": "A", "hours": hours_a}, {"job_code": "B", "hours": hours_b}],
            "staff": [{"worker_code": "U1", "max_hours": "4", "rate": "1.0"}, {"worker_code": "U2", "max_hours": "4", "rate": "2.0"}],
        }
    }


def make_project(context=None):
    return compile_scaffold_project("dynamic_scaffold_test", context or public_context(), {"setup.py": SETUP_CODE, "fitness.py": FITNESS_CODE})


def test_dynamic_contract_rejects_hardcoded_public_entity_id() -> None:
    bad_fitness = FITNESS_CODE.replace(
        'assignments = dict(genome["assign"])',
        'assignments = dict(genome["assign"])\n    preferred_worker = "U1"',
    )
    project = compile_scaffold_project(
        "dynamic_scaffold_test",
        public_context(),
        {"setup.py": SETUP_CODE, "fitness.py": bad_fitness},
    )

    errors = dynamic_patch_contract_errors(
        natural_language_update="Revalidate the current public task.",
        impact=DynamicUpdateImpact(),
        project=project,
        data_patch={},
        public_context=public_context(),
    )

    assert any("fitness.py: U1" in error for error in errors)


def test_dynamic_contract_allows_stable_semantic_category_literal() -> None:
    context = public_context()
    context["public_data"]["shifts"] = [
        {"shift_id": "day", "capacity": 2},
        {"shift_id": "night", "capacity": 2},
    ]
    category_fitness = FITNESS_CODE.replace(
        'assignments = dict(genome["assign"])',
        'assignments = dict(genome["assign"])\n    active_shift = "day"',
    )
    project = compile_scaffold_project(
        "dynamic_scaffold_test",
        context,
        {"setup.py": SETUP_CODE, "fitness.py": category_fitness},
    )

    errors = dynamic_patch_contract_errors(
        natural_language_update="Revalidate the public day shift.",
        impact=DynamicUpdateImpact(),
        project=project,
        data_patch={},
        public_context=context,
    )

    assert not any("hard-code public entity identifiers" in error for error in errors)


def test_optional_repair_word_in_classifier_reason_does_not_change_assignment_contract() -> None:
    project = make_project()
    errors = dynamic_patch_contract_errors(
        natural_language_update="Revalidate the current feasible plan without changing the public task.",
        impact=DynamicUpdateImpact(reason="warm restart permits optional repair"),
        project=project,
        data_patch={},
        public_context=public_context(),
    )

    assert not any("unassigned/skipped" in error for error in errors)


def test_runner_forces_cleanup_of_persisted_entity_literal() -> None:
    bad_fitness = FITNESS_CODE.replace(
        'assignments = dict(genome["assign"])',
        'assignments = dict(genome["assign"])\n    preferred_worker = "U1"',
    )
    project = compile_scaffold_project(
        "dynamic_scaffold_test",
        public_context(),
        {"setup.py": SETUP_CODE, "fitness.py": bad_fitness},
    )
    initial = project.run(EvolutionConfig(population_size=8, generations=1, seed=1, archive_limit=8))
    patch_client = FakePatchClient(
        "### fitness.py\n```python\n" + FITNESS_CODE.strip() + "\n```"
    )
    runner = LiveOptDynamicRunner(
        project,
        public_context=public_context(),
        result=initial,
        patcher=LiveOptWorkbenchPatcher(model="fake", client=patch_client),
    )

    stage = runner.update(
        update_id="u-contract-cleanup",
        natural_language_update="Revalidate the current public task.",
        impact=DynamicUpdateImpact(restart_skill="warm_restart_v1", reason="maintenance"),
        config=EvolutionConfig(population_size=8, generations=1, seed=2, archive_limit=8),
    )

    assert len(patch_client.payloads) == 1
    assert "Requested slots: fitness.py" in patch_client.payloads[0]["messages"][1]["content"]
    assert stage.impact.raw["contract_repair_slots"] == {"fitness.py": ["U1"]}
    assert 'preferred_worker = "U1"' not in stage.project.fitness_code


def public_context_with_capacity(u1_capacity: str, u2_capacity: str = "4"):
    return {
        "public_data": {
            "jobs": [{"job_code": "A", "hours": "2"}, {"job_code": "B", "hours": "2"}],
            "staff": [
                {"worker_code": "U1", "max_hours": u1_capacity, "rate": "1.0"},
                {"worker_code": "U2", "max_hours": u2_capacity, "rate": "2.0"},
            ],
        }
    }


def test_scaffold_project_run_accepts_restart_initial_genomes() -> None:
    project = make_project()
    seed = {"assign": {"A": "U2", "B": "U2"}}

    result = project.run(
        EvolutionConfig(population_size=1, generations=0, seed=0, structured_initialization=False),
        initial_genomes=[seed],
    )

    assert result.best.genome == seed
    assert result.best.result.solution["assignments"] == seed["assign"]


def test_dynamic_data_only_update_keeps_fixed_warm_for_local_change() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=10, generations=2, seed=1, archive_limit=10))
    runner = LiveOptDynamicRunner(project, public_context=public_context(), result=initial)

    stage = runner.update(
        update_id="u001",
        natural_language_update="Job A now takes 3 hours. Existing objective and encoding remain unchanged.",
        public_context=public_context(hours_a="3"),
        impact={
            "data_update": True,
            "patch_setup": False,
            "patch_fitness": False,
            "restart_skill": "warm_restart_v1",
            "reason": "only public table values changed",
        },
        config=EvolutionConfig(population_size=8, generations=1, seed=2, archive_limit=8, structured_initialization=False),
    )

    assert stage.project.setup_code == SETUP_CODE
    assert stage.project.fitness_code == FITNESS_CODE
    assert stage.project.data["tasks"][0]["load"] == 3
    assert stage.restart_metadata["restart_skill"] == "warm_restart_v1"
    assert stage.restart_metadata["source"] == "greedy_repair_from_previous_population"
    assert "objective_space_shift" not in stage.restart_metadata
    assert stage.restart_metadata["restart_selection_rule"] == "verified_semantic_full_else_fixed_warm"
    assert stage.initial_genomes
    assert stage.result.best.result.feasible


def test_local_capacity_change_does_not_force_full_from_infeasibility_alone() -> None:
    old_project = make_project(public_context_with_capacity("4", "4"))
    initial = old_project.run(EvolutionConfig(population_size=12, generations=1, seed=7, archive_limit=12))
    new_project = make_project(public_context_with_capacity("2", "4"))

    shift = estimate_objective_space_shift(
        previous_project=old_project,
        project=new_project,
        previous_result=initial,
        population_size=8,
        seed=7,
    )

    assert shift.direct_feasible_ratio < 1.0
    assert shift.selected_restart_skill == "warm_restart_v1"
    assert shift.selected_response_function == "public_regime_warm_full_selector"
    assert shift.fresh_population_ratio == 0.5
    assert abs(shift.history_population_ratio + shift.fresh_population_ratio - 1.0) < 1e-12


def test_public_state_replacement_selects_full_restart() -> None:
    old_context = {
        "public_data": {
            "jobs": [{"job_code": f"J{index:02d}", "hours": "1"} for index in range(12)],
            "staff": [
                {"worker_code": "U1", "max_hours": "100", "rate": "1.0"},
                {"worker_code": "U2", "max_hours": "100", "rate": "2.0"},
            ],
        }
    }
    new_context = {
        "public_data": {
            "jobs": [{"job_code": f"J{index:02d}", "hours": "2"} for index in range(12)],
            "staff": list(old_context["public_data"]["staff"]),
        }
    }
    old_project = make_project(old_context)
    initial = old_project.run(EvolutionConfig(population_size=12, generations=1, seed=7, archive_limit=12))
    new_project = make_project(new_context)

    shift = estimate_objective_space_shift(
        previous_project=old_project,
        project=new_project,
        previous_result=initial,
        population_size=8,
        seed=7,
    )

    assert shift.public_state_churn >= 0.15
    assert shift.public_state_changed_fact_count >= 8
    assert shift.selected_restart_skill == "full_restart_v1"
    assert shift.fresh_population_ratio == 1.0
    assert "public_state_replacement" in shift.large_change_signals


def test_objective_space_shift_never_launches_trial_optimization() -> None:
    old_project = make_project(public_context_with_capacity("4", "4"))
    initial = old_project.run(EvolutionConfig(population_size=12, generations=1, seed=7, archive_limit=12))
    new_project = make_project(public_context_with_capacity("2", "4"))

    def forbidden_trial(*args, **kwargs):
        raise AssertionError("objective-space shift estimation must not call project.run")

    new_project.run = forbidden_trial
    shift = estimate_objective_space_shift(
        previous_project=old_project,
        project=new_project,
        previous_result=initial,
        population_size=8,
        seed=7,
    )

    record = shift.to_record()
    assert record["policy_version"] == "public_regime_warm_full_selector_v2"
    assert not any("rollout" in key or "pilot" in key for key in record)


LANDSCAPE_SETUP_CODE = """
def build_problem(public_context):
    raw = public_context["public_data"]
    return {
        "data": {"factor": float(raw["factor"]), "limit": int(raw["limit"])},
        "segments": [{"name": "x", "kind": "int_vector", "length": 1, "lower": 0, "upper": 9}],
        "sense": "min",
        "solver_mode": "moea",
        "objective_names": ["left", "right"],
        "constraint_names": ["limit"],
    }
"""


LANDSCAPE_FITNESS_CODE = """
def evaluate(genome, data):
    x = int(genome["x"][0])
    factor = float(data["factor"])
    violations = {"limit": x - data["limit"]} if x > data["limit"] else {}
    objectives = [factor * x, factor * (9 - x)]
    return penalty_result(sum(objectives), violations, objectives=objectives, solution={"x": x})
"""


def make_landscape_project(*, factor: float, limit: int = 9):
    return compile_scaffold_project(
        "landscape-test",
        {"public_data": {"factor": factor, "limit": limit}},
        {"setup.py": LANDSCAPE_SETUP_CODE, "fitness.py": LANDSCAPE_FITNESS_CODE},
    )


def make_landscape_result(project) -> EvolutionResult:
    population = []
    for value in range(10):
        genome = {"x": [value]}
        result = coerce_fitness_result(project.evaluate(genome, project.data), genome)
        population.append(Candidate(genome, result))
    return EvolutionResult(
        best=population[0],
        population=population,
        archive=list(population),
        history=[],
        metadata={"selection": "nsga2"},
    )


def test_sensor_landscape_selector_uses_objective_displacement() -> None:
    old_project = make_landscape_project(factor=1.0)
    previous_result = make_landscape_result(old_project)

    mild = select_restart_skill_from_landscape_change(
        previous_project=old_project,
        project=make_landscape_project(factor=1.1),
        previous_result=previous_result,
        sensor_fraction=1.0,
    )
    severe = select_restart_skill_from_landscape_change(
        previous_project=old_project,
        project=make_landscape_project(factor=4.0),
        previous_result=previous_result,
        sensor_fraction=1.0,
    )

    assert mild.selected_restart_skill == "warm_restart_v1"
    assert mild.normalized_landscape_displacement < 1.0
    assert severe.selected_restart_skill == "full_restart_v1"
    assert severe.normalized_landscape_displacement >= severe.displacement_threshold
    assert severe.to_record()["policy_version"] == "sensor_objective_landscape_warm_full_v1"


def test_sensor_landscape_selector_uses_constraint_landscape_loss() -> None:
    old_project = make_landscape_project(factor=1.0, limit=9)
    previous_result = make_landscape_result(old_project)
    decision = select_restart_skill_from_landscape_change(
        previous_project=old_project,
        project=make_landscape_project(factor=1.0, limit=3),
        previous_result=previous_result,
        sensor_fraction=1.0,
    )

    assert decision.feasibility_loss_rate == 0.6
    assert decision.selected_restart_skill == "full_restart_v1"


def test_public_data_patch_paths_alias_loaded_tables_root() -> None:
    context = {"tables": {"jobs": [{"job_code": "A", "hours": "2"}], "policy": {"budget": "10", "enabled": "true"}}}

    patched = apply_public_context_patch(
        context,
        {
            "operations": [
                {"op": "update_row", "path": ["public_data", "jobs"], "match": {"job_code": "A"}, "values": {"hours": "3"}},
                {"op": "append_row", "path": ["public_data", "tables", "jobs"], "row": {"job_code": "B", "hours": "1"}},
                {"op": "set_value", "path": ["public_data", "policy", "budget"], "value": "12.5"},
            ]
        },
    )

    assert patched["tables"]["jobs"] == [{"job_code": "A", "hours": 3}, {"job_code": "B", "hours": 1}]
    assert patched["tables"]["policy"] == [{"budget": 12.5, "enabled": True}]


def test_data_patch_repair_loop_receives_error_feedback() -> None:
    project = make_project()
    client = FakeSequenceClient(
        [
            '{"operations": [{"op": "update_row", "path": ["tables", "jobs"], "match": {"job_code": "Z"}, "values": {"hours": "3"}}]}',
            '{"operations": [{"op": "update_row", "path": ["tables", "jobs"], "match": {"job_code": "A"}, "values": {"hours": "3"}}]}',
        ]
    )
    patcher = LiveOptDataPatcher(model="fake", client=client)

    patched, trace, payload = patcher.patch_context(
        public_context={"tables": {"jobs": [{"job_code": "A", "hours": 2}], "staff": [{"worker_code": "U1", "max_hours": 4, "rate": 1.0}]}},
        natural_language_update="Job A now takes 3 hours.",
        project=project,
        max_patch_repairs=3,
    )

    assert patched["tables"]["jobs"][0]["hours"] == 3
    assert payload["operations"][0]["match"] == {"job_code": "A"}
    assert len(client.payloads) == 2
    assert "matched no rows" in client.payloads[1]["messages"][1]["content"]
    assert trace.errors


def test_data_patch_prompt_receives_public_event_reference_ledger() -> None:
    prompt = build_data_patch_prompt(
        public_context=public_context(),
        natural_language_update="The item mentioned in the sponsor note changes again.",
        project=make_project(),
        public_update_history=[
            {
                "update_id": "t01",
                "natural_language_update": "Sponsor note names item I03.",
                "public_data_patch": {"operations": [{"match": {"id": "I03"}}]},
            }
        ],
    )

    assert "Prior public event/reference ledger" in prompt
    assert "Sponsor note names item I03" in prompt
    assert '"id": "I03"' in prompt


def test_dynamic_fitness_patch_updates_only_requested_slot() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=10, generations=2, seed=1, archive_limit=10))
    patch_client = FakePatchClient(PATCHED_FITNESS_RESPONSE)
    runner = LiveOptDynamicRunner(
        project,
        public_context=public_context(),
        result=initial,
        patcher=LiveOptWorkbenchPatcher(model="fake", client=patch_client),
    )

    stage = runner.update(
        update_id="u002",
        natural_language_update="Double the operating cost objective; data schema and constraints are unchanged.",
        impact=DynamicUpdateImpact(patch_fitness=True, restart_skill="population_transfer_v1", reason="objective changed"),
        config=EvolutionConfig(population_size=8, generations=1, seed=3, archive_limit=8, structured_initialization=False),
    )

    assert len(patch_client.payloads) == 1
    assert stage.project.setup_code == SETUP_CODE
    assert "objective_scale" in stage.project.fitness_code
    assert stage.restart_metadata["restart_skill"] == "warm_restart_v1"
    assert stage.restart_metadata["source"] == "greedy_repair_from_previous_population"
    assert "objective_space_shift" not in stage.restart_metadata
    assert stage.restart_metadata["restart_selection_rule"] == "verified_semantic_full_else_fixed_warm"
    assert stage.result.best.result.solution["objective_scale"] == 2.0


def test_typed_full_regeneration_requests_complete_slots_with_interface_reference() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=8, generations=1, seed=1, archive_limit=8))
    response = "\n".join(
        [
            "### setup.py",
            "```python",
            SETUP_CODE.strip(),
            "```",
            "### fitness.py",
            "```python",
            FITNESS_CODE.strip(),
            "```",
        ]
    )
    patch_client = FakePatchClient(response)
    runner = LiveOptDynamicRunner(
        project,
        public_context=public_context(),
        result=initial,
        patcher=LiveOptWorkbenchPatcher(model="fake", client=patch_client),
    )

    stage = runner.update(
        update_id="u002",
        natural_language_update="Revalidate the unchanged public task.",
        impact=DynamicUpdateImpact(restart_skill="warm_restart_v1", reason="maintenance"),
        regenerate_typed_workbench=True,
        config=EvolutionConfig(population_size=8, generations=1, seed=3, archive_limit=8),
    )

    prompt = patch_client.payloads[0]["messages"][1]["content"]
    assert "Requested slots: setup.py, fitness.py" in prompt
    assert "typed_full_regeneration_control" in prompt
    assert "complete standalone replacements" in prompt
    assert "setup.py must define build_problem(public_context)" in prompt
    assert "fitness.py must define evaluate(genome, data)" in prompt
    assert SETUP_CODE.strip() in prompt
    assert stage.project.setup_code.strip() == SETUP_CODE.strip()
    assert stage.project.fitness_code.strip() == FITNESS_CODE.strip()


def test_dynamic_update_can_force_restart_skill_for_ablation() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=10, generations=2, seed=1, archive_limit=10))
    runner = LiveOptDynamicRunner(project, public_context=public_context(), result=initial)

    stage = runner.update(
        update_id="u003",
        natural_language_update="Capacity changed locally; use the same scaffold.",
        public_context=public_context(),
        impact={
            "data_update": False,
            "patch_setup": False,
            "patch_fitness": False,
            "restart_skill": "population_transfer_v1",
            "reason": "localizer selected transfer",
        },
        force_restart_skill="warm_restart_v1",
        config=EvolutionConfig(population_size=8, generations=1, seed=5, archive_limit=8, structured_initialization=False),
    )

    assert stage.impact.restart_skill == "warm_restart_v1"
    assert stage.impact.raw["original_restart_skill"] == "population_transfer_v1"
    assert stage.impact.raw["restart_policy_override"] is True
    assert stage.restart_metadata["restart_skill"] == "warm_restart_v1"


def test_stateful_record_replaces_lsm_with_fixed_public_output_reuse() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=10, generations=2, seed=1, archive_limit=10))
    runner = LiveOptDynamicRunner(
        project,
        public_context=public_context(),
        result=initial,
        public_update_history=[
            {
                "update_id": "u001",
                "natural_language_update": "Keep the public assignment task unchanged.",
                "public_data_patch": {},
            }
        ],
    )

    stage = runner.update(
        update_id="u002",
        natural_language_update="Revalidate the same public task.",
        impact=DynamicUpdateImpact(restart_skill="population_transfer_v1", reason="provided control impact"),
        stateful_record=True,
        config=EvolutionConfig(population_size=8, generations=1, seed=5, archive_limit=8, structured_initialization=False),
    )

    assert stage.impact.restart_skill == "warm_restart_v1"
    assert stage.restart_metadata["memory_mode"] == "stateful_record"
    assert "objective_space_shift" not in stage.restart_metadata
    assert stage.restart_metadata["restart_selection_rule"] == "stateful_record_fixed_warm"
    assert stage.restart_metadata["source"] == "greedy_repair_from_previous_population"
    assert stage.restart_metadata["previous_candidate_count"] == 1
    assert stage.restart_metadata["stateful_record_fields"] == [
        "public_update_history",
        "committed_public_data_patches",
        "previous_accepted_public_output",
    ]
    assert stage.initial_genomes


def test_stateful_record_and_legacy_no_lsm_are_mutually_exclusive() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=8, generations=1, seed=1, archive_limit=8))
    runner = LiveOptDynamicRunner(project, public_context=public_context(), result=initial)

    try:
        runner.update(
            update_id="u-conflict",
            natural_language_update="Revalidate the same public task.",
            impact=DynamicUpdateImpact(),
            disable_lsm=True,
            stateful_record=True,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutually exclusive memory controls to fail")


def test_restart_seed_builder_coerces_population_to_new_segments() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=10, generations=1, seed=4, archive_limit=10))
    expanded = make_project(
        {
            "public_data": {
                "jobs": [
                    {"job_code": "A", "hours": "2"},
                    {"job_code": "B", "hours": "1"},
                    {"job_code": "C", "hours": "1"},
                ],
                "staff": [{"worker_code": "U1", "max_hours": "4", "rate": "1.0"}, {"worker_code": "U2", "max_hours": "4", "rate": "2.0"}],
            }
        }
    )

    seeds, metadata = ScaffoldRestartSeedBuilder().build(
        restart_skill="population_transfer_v1",
        previous_result=initial,
        segments=expanded.segments,
        population_size=6,
        seed=5,
        evaluate=expanded.evaluate,
        data=expanded.data,
    )

    assert metadata["restart_skill"] == "population_transfer_v1"
    assert metadata["source"] == "direct_population_migration"
    assert metadata["migration_operator"] == "direct_population_migration"
    assert seeds
    assert all(set(seed["assign"]) == {"A", "B", "C"} for seed in seeds)
    assert all(seed["assign"]["C"] in {"U1", "U2"} for seed in seeds)


def test_adaptive_mix_requires_measured_objective_shift() -> None:
    project = make_project()
    initial = project.run(EvolutionConfig(population_size=8, generations=1, seed=4, archive_limit=8))

    try:
        ScaffoldRestartSeedBuilder().build(
            restart_skill="adaptive_mix_v1",
            previous_result=initial,
            segments=project.segments,
            population_size=8,
            seed=5,
            evaluate=project.evaluate,
            data=project.data,
        )
    except ValueError as exc:
        assert "objective_space_shift" in str(exc)
    else:
        raise AssertionError("adaptive mix must not silently fall back to a fixed ratio")


def test_population_transfer_is_direct_migration_without_repair() -> None:
    old_project = make_project(public_context_with_capacity("4", "4"))
    initial = old_project.run(EvolutionConfig(population_size=18, generations=2, seed=11, archive_limit=18))
    new_project = make_project(public_context_with_capacity("3", "4"))

    seeds, metadata = ScaffoldRestartSeedBuilder().build(
        restart_skill="population_transfer_v1",
        previous_result=initial,
        segments=new_project.segments,
        population_size=12,
        seed=12,
        evaluate=new_project.evaluate,
        data=new_project.data,
    )

    assert metadata["source"] == "direct_population_migration"
    assert metadata["migration_operator"] == "direct_population_migration"
    assert metadata["direct_evaluated_count"] > 0
    assert metadata["repaired_feasible_count"] == 0
    assert metadata["local_variant_count"] == 0
    assert 0 < len(seeds) <= 12
    assert metadata["feasible_seed_count"] == len(filter_feasible_for_test(seeds, new_project))


def filter_feasible_for_test(seeds, project):
    feasible = []
    for seed in seeds:
        try:
            result = project.evaluate(seed, project.data)
        except Exception:
            continue
        is_feasible = result.feasible if hasattr(result, "feasible") else bool(result.get("feasible", True))
        if is_feasible:
            feasible.append(seed)
    return feasible


def test_supplied_public_patch_reaches_localizer_workbench_gate_and_ledger(monkeypatch):
    import copy
    from evo2.agents.liveopt_workbench_impl import ScaffoldTrace

    original = public_context()
    project = make_project(original)
    config = EvolutionConfig(population_size=8, generations=1, seed=2, archive_limit=8)
    initial = project.run(config)
    supplied = {"operations": [{"op": "update_row", "path": ["public_data", "jobs"], "match": {"job_code": "A"}, "values": {"hours": 3}}]}
    saved = copy.deepcopy(supplied)
    seen = {}

    class Localizer:
        def classify(self, **kwargs):
            seen["localizer"] = copy.deepcopy(kwargs["public_context"])
            # Explicit public input must apply even if the model misses this flag.
            return DynamicUpdateImpact.from_mapping({
                "data_update": False, "patch_setup": False, "patch_fitness": False,
                "restart_skill": "warm_restart_v1", "reason": "same model",
            }), ScaffoldTrace(model="fake")

    class DataPatcher:
        def patch_context(self, **kwargs):
            raise AssertionError("supplied public rows must not be regenerated by a model")

    class Gate:
        def decide(self, **kwargs):
            seen["old"] = copy.deepcopy(kwargs["previous_public_context"])
            seen["new"] = copy.deepcopy(kwargs["public_context"])
            return None, ScaffoldTrace(model="fake")

    runner = LiveOptDynamicRunner(project, public_context=original, result=initial,
                                 localizer=Localizer(), data_patcher=DataPatcher(),
                                 semantic_restart_gate=Gate())
    stage = runner.update(update_id="supplied-rows", natural_language_update="Use the supplied current rows.",
                          public_data_patch=supplied, config=config)
    assert int(seen["localizer"]["public_data"]["jobs"][0]["hours"]) == 3
    assert int(seen["old"]["public_data"]["jobs"][0]["hours"]) == 2
    assert int(seen["new"]["public_data"]["jobs"][0]["hours"]) == 3
    assert stage.project.data["tasks"][0]["load"] == 3
    assert stage.restart_metadata["data_patch"] == saved
    assert stage.restart_metadata["data_patch_source"] == "supplied_public_update"
    assert runner.public_update_history[-1]["public_data_patch"] == saved
    assert supplied == saved and original == public_context()
    supplied["operations"][0]["values"]["hours"] = 99
    assert runner.public_update_history[-1]["public_data_patch"] == saved


def test_supplied_patch_rejects_ambiguous_input_without_committing():
    import pytest
    runner = LiveOptDynamicRunner(make_project(), public_context=public_context())
    with pytest.raises(ValueError, match="either public_context or public_data_patch"):
        runner.update(update_id="bad", natural_language_update="update",
                      public_context=public_context(), public_data_patch={})
    assert runner.public_context == public_context()
    assert runner.public_update_history == []


def test_full_benchmark_forwards_supplied_patch_to_all_seed_records(monkeypatch):
    from types import SimpleNamespace
    from evo2.agents.liveopt_workbench_impl import ScaffoldTrace
    from scripts.llm_tests import run_liveopt_dynamic_nldo_benchmark_full as benchmark

    patch = {"operations": [{"op": "update_row", "path": ["public_data", "jobs"],
                             "match": {"job_code": "A"}, "values": {"hours": 3}}]}
    episode = {"episode_id": "supplied-public-patch-test", "public_initial_problem": "Assign jobs.",
               "update_stream": [{"update_id": "t01", "natural_language_update": "Use supplied rows.",
                                  "public_data_patch": patch}]}
    args = SimpleNamespace(model="fake", max_patch_repairs=0, semantic_decisions_json=None,
                           disable_semantic_restart_gate=True, force_restart_skill=None,
                           disable_lsm=False, stateful_record=False, disable_tss=False,
                           max_updates=None, population_size=8, generations=1, archive_limit=8,
                           initial_population_size=None, initial_generations=None, seed_count=2)
    project = make_project()
    monkeypatch.setattr(benchmark, "public_context_with_loaded_tables", lambda e: public_context())
    monkeypatch.setattr(benchmark.LiveOptWorkbenchGenerator, "generate_project", lambda *a, **k: project)
    monkeypatch.setattr(benchmark, "evolution_config", lambda a, seed, **k:
                        EvolutionConfig(population_size=8, generations=1, seed=seed, archive_limit=8))

    class Localizer:
        def classify(self, **kwargs):
            assert int(kwargs["public_context"]["public_data"]["jobs"][0]["hours"]) == 3
            return DynamicUpdateImpact.from_mapping({"data_update": True, "patch_setup": False,
                    "patch_fitness": False, "restart_skill": "warm_restart_v1"}), ScaffoldTrace(model="fake")

    monkeypatch.setattr(benchmark, "LiveOptDynamicRunner", lambda *a, **k:
                        LiveOptDynamicRunner(*a, **k, localizer=Localizer()))
    record, rows = benchmark.run_episode(args, episode, [0, 1])
    assert record["status"] == "completed", record.get("failure_reason")
    assert len(rows) == 2
    for row in rows:
        stage = row["update_results"][0]
        assert stage["restart"]["data_patch"] == patch
        assert stage["objective_value"] >= 4  # Three hours for A and one for B.
