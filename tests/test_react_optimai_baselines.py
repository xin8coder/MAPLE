from __future__ import annotations

import json


class FakeJsonClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def chat(self, messages, temperature=0.0, json_mode=False, max_tokens=None):
        self.payloads.append({"messages": messages, "temperature": temperature, "json_mode": json_mode, "max_tokens": max_tokens})
        content = json.dumps(self.responses.pop(0))
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "choices": [{"message": {"content": content}}]}


def test_react_direct_baseline_stays_direct_and_memory_free() -> None:
    from evo2.agents.react_optimai_planner import ReActPlanner

    planner = ReActPlanner(model="fake")
    planner.client = FakeJsonClient(
        [
            {
                "reasoning_summary": "direct assignment",
                "actions": [{"type": "replan", "description": "assign jobs"}],
                "solution_plan": {"representation": "direct_solution_json", "content": {"assignments": {"J1": "M1"}}},
                "solver_choice": "direct_llm",
                "confidence": "medium",
            }
        ]
    )

    result = planner.plan(
        {"task": "assign jobs"},
        memory={
            "public_update_history": ["initial request", "new job arrives"],
            "population_summary": {"archive_size": 25},
            "strategy_memory": [{"restart": "warm_restart"}],
            "retrieved_memory": [{"private": "liveopt"}],
        },
        update_text="new job arrives",
    )

    assert result["solver_choice"] == "direct_llm"
    assert result["solution_plan"]["content"]["assignments"] == {"J1": "M1"}
    prompt = planner.client.payloads[0]["messages"][1]["content"]
    assert "Do not output code" in prompt
    assert "Do not infer from hidden benchmark" in prompt
    assert "public_update_history" in prompt
    assert "population_summary" not in prompt
    assert "strategy_memory" not in prompt
    assert "retrieved_memory" not in prompt
    assert "nldo_ir" not in prompt


def test_react_tools_baseline_exposes_public_tools_without_liveopt_restart() -> None:
    from evo2.agents.react_optimai_planner import ReActToolsPlanner

    planner = ReActToolsPlanner(model="fake")
    planner.client = FakeJsonClient(
        [
            {
                "thought": "use optimizer",
                "actions": [{"tool": "ga_full_restart", "arguments": {}, "purpose": "solve"}],
                "final_solution": {"assignments": {"J1": "M1"}},
                "confidence": "medium",
            }
        ]
    )

    result = planner.plan(
        {"task": "assign jobs"},
        memory={
            "accepted_state_summary": {"stage": "t01"},
            "population_summary": {"archive_size": 10},
            "solution_record": {"solution": {"secret": "not for public baseline"}},
        },
        update_text="",
        observations=[],
    )

    assert result["actions"][0]["tool"] == "ga_full_restart"
    prompt = planner.client.payloads[0]["messages"][1]["content"]
    assert "Do not use EVO2 memory, restart policies, artifact patches, or population transfer" in prompt
    assert "available_tools" in prompt
    assert "accepted_state_summary" in prompt
    assert "population_summary" not in prompt
    assert "solution_record" not in prompt
    assert "nldo_ir" not in prompt


def test_optimai_faithful_pipeline_debugs_generated_solver() -> None:
    from evo2.agents.react_optimai_planner import OptimAIPlanner

    planner = OptimAIPlanner(model="fake")
    responses = [
        {
            "problem_type": "combinatorial",
            "decision_variables": ["assignments"],
            "objectives": ["minimize"],
            "hard_constraints": [],
            "public_data_dependencies": [],
            "dynamic_context": "",
            "solver_requirements": [],
        },
        {
            "plans": [
                {"plan_id": "bad", "solver_choice": "ga_full_restart", "algorithm": "bad code first", "score": 9},
                {"plan_id": "good", "solver_choice": "ga_full_restart", "algorithm": "direct ga", "score": 8},
            ]
        },
        {"code_lines": ["def solver(problem, memory, tools):", "    raise ValueError('boom')"]},
        {
            "code_lines": [
                "def solver(problem, memory, tools):",
                "    result = tools['ga_full_restart']({})",
                "    return {'solver_choice': 'ga_full_restart', 'solution': result['solution']}",
            ]
        },
        {
            "diagnosis": "fix failing branch",
            "code_lines": [
                "def solver(problem, memory, tools):",
                "    result = tools['ga_full_restart']({})",
                "    return {'solver_choice': 'ga_full_restart', 'solution': result['solution']}",
            ],
        },
    ]
    planner.client = FakeJsonClient(responses)
    tools = {
        "ga_full_restart": lambda options=None: {"success": True, "solution": {"assignments": {"J1": "M1"}, "genome": [0]}, "metadata": {}},
        "verifier": lambda solution: {"feasible": bool(solution.get("assignments")), "violations": {}},
        "objective": lambda solution: 0.0,
    }

    result = planner.run({"task_id": "p", "domain": "toy"}, memory={}, update_text="", tools=tools, budget={"optimai_num_plans": 2, "optimai_debug_rounds": 2})

    assert result["verification"]["feasible"]
    assert result["solver_decision"]["skill_id"] == "ga_full_restart"
    assert planner.last_trace.plan["status"] == "success"
    assert planner.last_trace.usage["total_tokens"] >= 8
