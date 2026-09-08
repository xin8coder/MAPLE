from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from scripts.baselines.run_static_calibration import (
    build_messages,
    evaluate_official_static,
    parse_model_response,
    evaluate_reference,
    parse_json_object,
    run_case_method,
    select_cases,
    validate_parsed_output,
)


class FakeContentClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.payloads = []

    def chat(self, messages, temperature=0.0, json_mode=False, max_tokens=None):
        self.payloads.append({"messages": messages, "temperature": temperature, "json_mode": json_mode, "max_tokens": max_tokens})
        content = self.contents.pop(0)
        if isinstance(content, dict):
            content = json.dumps(content)
        return {
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "choices": [{"message": {"content": content}}],
        }
from scripts.baselines.summarize_static_calibration_runs import render_tex, summarize


def test_static_calibration_prompts_are_method_specific() -> None:
    case = {
        "case_id": "toy",
        "source": "unit",
        "problem_text": "Choose production quantities to maximize profit under capacity.",
        "reference": {"answer": "x=1"},
    }

    liveopt = build_messages(case, "liveopt_static")
    optimai = build_messages(case, "optimai_2025")
    or_agent = build_messages(case, "or_llm_agent_2025")

    assert "LiveOpt in static mode" in liveopt[0]["content"]
    assert "OptimAI-style" in optimai[0]["content"]
    assert "OR-LLM-Agent-style" in or_agent[0]["content"]
    assert "encoding_plan" in liveopt[1]["content"]
    assert "candidate_solver_plans" in optimai[1]["content"]
    assert "self_repair" in or_agent[1]["content"]
    assert "final_answer" in liveopt[1]["content"]
    assert "final_answer" in optimai[1]["content"]
    assert "final_answer" in or_agent[1]["content"]


def test_strong_react_controls_have_distinct_public_schemas() -> None:
    case = {
        "case_id": "toy",
        "source": "NLDO-dynamic-public",
        "problem_text": "Minimize cost after a public update.",
        "public_context": {
            "parameters": {
                "stage_index": 1,
                "public_update_history": [{"natural_language_update": "tighten the deadline"}],
                "previous_accepted_public_output": {"assignment": {"J1": "M1"}},
            }
        },
        "reference": {},
    }
    schemas = {}
    for method in ["react_transcript_state", "react_generic_workbench", "react_public_delta_oracle"]:
        messages = build_messages(case, method)
        payload = json.loads(messages[1]["content"])
        schemas[method] = payload["required_output_schema"]
        assert "dynamic optimization stage" in payload["task"]
        assert "previous_accepted_public_output" in payload["public_context"]["parameters"]
    assert "public_state_reconstruction" in schemas["react_transcript_state"]
    assert "generic_workbench_plan" in schemas["react_generic_workbench"]
    assert "public_delta_use" in schemas["react_public_delta_oracle"]


def test_dynamic_public_case_includes_previous_public_output_and_delta() -> None:
    from scripts.baselines.run_nldo_dynamic_public_baselines import dynamic_stage_case

    episode = {
        "episode_id": "NLDO-P001",
        "public_initial_problem": "Assign tasks.",
        "update_stream": [{"update_id": "t001", "time_index": 1, "natural_language_update": "tighten deadline in jobs table"}],
        "hidden_update_oracle": [
            {
                "hidden_delta": {
                    "type": "deadline_delta",
                    "job": "J1",
                    "delta": -1,
                    "difficulty": "memory",
                    "resolved_from_memory": ["public note"],
                }
            }
        ],
    }
    public_context = {
        "tables": {"jobs": [{"job_id": "J1", "deadline": 5}]},
        "csv_schema": {"jobs": ["job_id", "deadline"]},
        "multi_objective": False,
        "objective_sense": "min",
    }
    case = dynamic_stage_case(
        episode,
        public_context,
        {"stage_index": 1, "update_id": "t001", "updates": episode["update_stream"]},
        "react_public_delta_oracle",
        {"assignment": {"J1": "M1"}},
        None,
        [{"solution": {"assignment": {"J1": "M1"}}, "objectives": [1.0]}],
    )
    params = case["public_context"]["parameters"]
    assert params["previous_accepted_public_output"]["assignment"]["J1"] == "M1"
    assert params["shared_lsm"]["interface"] == "common_public_lsm_dialogue_and_own_population_v1"
    assert params["shared_lsm"]["dialogue_turn_count"] == 2
    assert params["shared_lsm"]["cumulative_dialogue"][0]["turn_id"] == "initial"
    assert params["shared_lsm"]["cumulative_dialogue"][1]["turn_id"] == "t001"
    assert len(params["shared_lsm"]["previous_candidate_population"]) == 1
    assert params["shared_lsm"]["population_limit"] == 200
    assert "hidden evaluation never controls LSM" in params["shared_lsm"]["state_advance_gate"]
    assert params["shared_lsm"]["liveopt_population_available"] is False
    assert params["public_state_summary"]["previous_accepted_output_available"] is True
    assert params["structured_public_delta"]["hidden_checker_available"] is False
    assert params["structured_public_delta"]["reference_available"] is False
    assert params["structured_public_delta"]["operation"] == {
        "type": "deadline_delta",
        "job": "J1",
        "delta": -1,
    }


def test_generic_workbench_receives_previous_accepted_code_artifact() -> None:
    from scripts.baselines.run_nldo_dynamic_public_baselines import dynamic_stage_case

    episode = {
        "episode_id": "NLDO-P001",
        "public_initial_problem": "Assign tasks.",
        "update_stream": [{"update_id": "t001", "natural_language_update": "tighten deadline"}],
    }
    case = dynamic_stage_case(
        episode,
        {"tables": {}, "multi_objective": False},
        {"stage_index": 1, "update_id": "t001", "updates": episode["update_stream"]},
        "react_generic_workbench",
        {"assignment": {"J1": "M1"}},
        {
            "solver_code": "print('previous accepted solver')",
            "generic_workbench_plan": {"solver_or_search": "MILP"},
        },
    )
    params = case["public_context"]["parameters"]
    assert "patch the previous accepted solver artifact locally" in params["generic_workbench_interface"]["edit_policy"]
    assert params["previous_accepted_solver_artifact"]["solver_code"] == "print('previous accepted solver')"


def test_dynamic_public_stage_uses_equal_code_output_repair(tmp_path: Path) -> None:
    from scripts.baselines.run_nldo_dynamic_public_baselines import run_stage_method

    bad = {
        "thought": "solve with public code",
        "generic_workbench_plan": {"solver_or_search": "direct_python"},
        "react_trace": [{"thought": "try code", "action": "run", "observation": "public"}],
        "tool_plan": [{"tool": "direct_python", "purpose": "toy"}],
        "validation_plan": ["run"],
        "solver_code": {"language": "python", "code": "raise RuntimeError('boom')"},
        "final_answer": {"objective_value": 1, "solution": {"x": 1}},
    }
    repaired = "```python\nimport json\nprint(json.dumps({'objective_value': 1, 'solution': {'x': 1}}))\n```"
    client = FakeContentClient([bad, repaired])
    args = SimpleNamespace(
        response_protocol="json",
        no_json_mode=False,
        max_tokens=1000,
        model="fake",
        code_repair_attempts=2,
    )
    episode = {
        "episode_id": "NLDO-TOY",
        "public_initial_problem": "Choose integer x <= 1 to maximize x.",
        "domain": "toy",
        "family": "toy",
        "hidden_initial_state": {},
        "hidden_update_oracle": [],
    }

    row = run_stage_method(
        args=args,
        client=client,
        episode=episode,
        public_context={"tables": {}, "multi_objective": False, "objective_sense": "max"},
        method="react_generic_workbench",
        stage={"stage_index": 0, "update_id": "initial", "updates": []},
        trace_dir=tmp_path,
        previous_solution=None,
    )

    assert row["code_execution"]["runtime_success"] is True
    assert row["code_output_repair"]["max_attempts"] == 2
    assert row["code_output_repair"]["attempts_used"] == 1
    assert row["token_usage"]["total_tokens"] == 10
    assert row["public_state_available"] is True
    assert row["shared_lsm_input"]["hidden_evaluation_controls_state"] is False
    trace = json.loads(Path(row["trace_path"]).read_text())
    assert trace["code_output_repair"]["attempts_used"] == 1


def test_dynamic_public_lsm_passes_bounded_own_population_without_liveopt_state() -> None:
    from scripts.baselines.run_nldo_dynamic_public_baselines import dynamic_stage_case

    population = [{"solution": {"x": index}} for index in range(205)]
    case = dynamic_stage_case(
        {
            "episode_id": "NLDO-TOY",
            "public_initial_problem": "Choose x.",
            "update_stream": [{"update_id": "t001", "natural_language_update": "change x"}],
        },
        {"tables": {}, "multi_objective": False},
        {
            "stage_index": 1,
            "update_id": "t001",
            "updates": [{"update_id": "t001", "natural_language_update": "change x"}],
        },
        "react_tools",
        {"x": 204},
        None,
        population,
    )
    lsm = case["public_context"]["parameters"]["shared_lsm"]
    assert lsm["previous_candidate_count"] == 200
    assert len(lsm["previous_candidate_population"]) == 200
    assert lsm["previous_candidate_population"][0]["solution"]["x"] == 0
    assert lsm["previous_candidate_population"][-1]["solution"]["x"] == 199
    assert lsm["liveopt_population_available"] is False


def test_dynamic_public_lsm_advances_on_public_output_not_hidden_score(monkeypatch, tmp_path: Path) -> None:
    from scripts.baselines import run_nldo_dynamic_public_baselines as dynamic

    received: list[dict[str, object]] = []

    def fake_run_stage(*call_args, **call_kwargs):
        stage = call_args[5]
        previous_public = call_args[7]
        previous_population = call_args[9]
        previous_scoring = call_args[10]
        received.append(
            {
                "stage": stage["stage_index"],
                "previous_public": previous_public,
                "previous_population": previous_population,
                "previous_scoring": previous_scoring,
            }
        )
        index = int(stage["stage_index"])
        solution = {"x": index}
        return {
            "stage_index": index,
            "update_id": stage["update_id"],
            "status": "hidden_rejected",
            "hidden_evaluation": {"feasible": False, "normalized_score": 0.0},
            "solution": solution,
            "candidate_archive": [{"solution": solution}],
            "public_state_available": True,
            "parsed_output": {"solver_code": "print('{}')"},
            "code_execution": {"compile_success": True, "runtime_success": True},
            "token_usage": dynamic.zero_usage(),
            "latency_seconds": 0.0,
        }

    monkeypatch.setattr(dynamic, "run_stage_method", fake_run_stage)
    args = SimpleNamespace(
        max_updates=1,
        model="fake",
        code_repair_attempts=0,
        continue_after_hidden_rejection=True,
    )
    episode = {
        "episode_id": "NLDO-TOY",
        "domain": "toy",
        "family": "toy",
        "public_initial_problem": "Choose x.",
        "public_context": {"csv_tables": {}},
        "update_stream": [{"update_id": "t001", "natural_language_update": "change x"}],
    }
    run, _ = dynamic.run_episode_method(args, None, episode, "react_tools", tmp_path)

    assert run["run_status"] == "completed"
    assert received[0]["previous_public"] is None
    assert received[1]["previous_public"] == {"x": 0}
    assert received[1]["previous_population"] == [{"solution": {"x": 0}}]
    assert received[1]["previous_scoring"] is None


def test_dynamic_public_stop_first_policy_ends_after_hidden_failure(monkeypatch, tmp_path: Path) -> None:
    from scripts.baselines import run_nldo_dynamic_public_baselines as dynamic

    reached: list[int] = []

    def fake_run_stage(*call_args, **call_kwargs):
        stage = call_args[5]
        reached.append(int(stage["stage_index"]))
        return {
            "stage_index": int(stage["stage_index"]),
            "update_id": stage["update_id"],
            "status": "hidden_rejected",
            "hidden_evaluation": {"feasible": False, "normalized_score": 0.0},
            "solution": {"x": 0},
            "candidate_archive": [{"solution": {"x": 0}}],
            "public_state_available": True,
            "parsed_output": {"solver_code": "print('{}')"},
            "code_execution": {"compile_success": True, "runtime_success": True},
            "token_usage": dynamic.zero_usage(),
            "latency_seconds": 0.0,
        }

    monkeypatch.setattr(dynamic, "run_stage_method", fake_run_stage)
    args = SimpleNamespace(
        max_updates=2,
        model="fake",
        code_repair_attempts=3,
        continue_after_hidden_rejection=False,
    )
    episode = {
        "episode_id": "NLDO-TOY",
        "domain": "toy",
        "family": "toy",
        "public_initial_problem": "Choose x.",
        "public_context": {"csv_tables": {}},
        "update_stream": [
            {"update_id": "t001", "natural_language_update": "change x"},
            {"update_id": "t002", "natural_language_update": "change x again"},
        ],
    }

    run, rows = dynamic.run_episode_method(args, None, episode, "react_tools", tmp_path)

    assert reached == [0]
    assert len(rows) == 1
    assert run["run_status"] == "stopped_after_failure"


def test_dynamic_public_extracts_archive_nested_under_solution() -> None:
    from scripts.baselines import run_nldo_dynamic_public_baselines as dynamic

    output = {
        "objective_value": None,
        "solution": {
            "candidate_archive": [
                {"routes": [{"vehicle": "V1", "orders": ["O1"]}], "objectives": {"distance": 1.0}},
                {
                    "solution": {"routes": [{"vehicle": "V2", "orders": ["O2"]}]},
                    "objectives": {"distance": 2.0},
                },
            ]
        },
    }

    solution = dynamic.extract_solution(output, {})
    archive = dynamic.extract_candidate_archive(output, {}, solution)

    assert solution == {"routes": [{"vehicle": "V1", "orders": ["O1"]}], "objectives": {"distance": 1.0}}
    assert archive == [
        {"solution": {"routes": [{"vehicle": "V1", "orders": ["O1"]}], "objectives": {"distance": 1.0}}},
        {
            "solution": {"routes": [{"vehicle": "V2", "orders": ["O2"]}]},
            "objectives": {"distance": 2.0},
        },
    ]


def test_static_calibration_bwor_prompts_use_compact_code_first_schema() -> None:
    case = {
        "case_id": "bwor_toy",
        "source": "BWOR",
        "problem_text": "Minimize feed cost with nutrition requirements.",
        "reference": {"answer": 1.0, "solution_status": "optimal"},
    }

    messages = build_messages(case, "or_llm_agent_2025")
    prompt = messages[1]["content"]

    assert "Keep all reasoning fields short" in prompt
    assert "scipy.optimize" in prompt
    assert "Do not import gurobipy" in prompt
    assert "short modeling summary" in prompt
    assert "parameters.json" not in prompt


def test_static_calibration_schema_validation_requires_core_fields() -> None:
    assert not validate_parsed_output(
        "or_llm_agent_2025",
        {
            "reasoning_modeling_trace": {},
            "formulation": {},
            "code_generation_plan": {},
            "self_verification": {},
            "self_repair": {},
            "solver_code": {"language": "python", "code": "print({})"},
            "final_answer": {"objective_value": 19, "solution": {}},
        },
    )
    feedback = validate_parsed_output("optimai_2025", {"formulation": {}, "candidate_solver_plans": {}})
    assert "missing required field: selected_plan" in feedback
    assert "candidate_solver_plans must be a list" in feedback


def test_static_calibration_parser_extracts_echoed_required_schema() -> None:
    parsed = parse_json_object(
        json.dumps(
            {
                "case_id": "toy",
                "required_output_schema": {
                    "formulation": {},
                    "candidate_solver_plans": [],
                    "selected_plan": {},
                    "debug_validation_plan": [],
                    "solver_code": {"language": "python", "code": "print({})"},
                    "final_answer": {"objective_value": 1},
                },
            }
        )
    )

    assert set(parsed) == {"formulation", "candidate_solver_plans", "selected_plan", "debug_validation_plan", "solver_code", "final_answer"}


def test_static_calibration_parser_extracts_json_from_normal_text() -> None:
    parsed = parse_json_object(
        "Here is the result:\n"
        '{"formulation": {}, "candidate_solver_plans": [], "selected_plan": {}, '
        '"debug_validation_plan": [], "solver_code": {"code": "print({})"}, '
        '"final_answer": {"objective_value": 1}}'
    )

    assert parsed["final_answer"]["objective_value"] == 1


def test_static_calibration_code_block_protocol_wraps_solver_code() -> None:
    parsed = parse_model_response(
        "```python\nimport json\nprint(json.dumps({'objective_value': 1, 'solution': {}}))\n```",
        "or_llm_agent_2025",
        protocol="code_block",
    )

    assert parsed["solver_code"]["language"] == "python"
    assert "objective_value" in parsed["solver_code"]["code"]
    assert "self_verification" in parsed


def test_static_calibration_equal_code_output_repair_budget(tmp_path: Path) -> None:
    bad = {
        "problem_understanding": "toy",
        "data_assumptions": [],
        "decision_variables": [{"name": "x", "type": "integer", "meaning": "toy"}],
        "objectives": [{"sense": "max", "expression": "x"}],
        "constraints": [{"expression": "x <= 1", "type": "hard"}],
        "encoding_plan": [{"variable": "x", "encoding_skill": "integer_vector"}],
        "solver_plan": {"solver": "direct", "reason": "toy"},
        "verification_plan": ["run"],
        "output_solution_format": {"objective_value": "number", "solution": "object"},
        "solver_code": {"language": "python", "code": "raise RuntimeError('boom')"},
        "final_answer": {"objective_value": 1, "solution": {"x": 1}},
    }
    repaired = "```python\nimport json\nprint(json.dumps({'objective_value': 1, 'solution': {'x': 1}}))\n```"
    client = FakeContentClient([bad, repaired])
    args = SimpleNamespace(
        response_protocol="json",
        no_json_mode=False,
        max_tokens=1000,
        model="fake",
        code_repair_attempts=2,
    )
    case = {
        "case_id": "repair_toy",
        "source": "unit",
        "problem_text": "Choose integer x <= 1 to maximize x.",
        "public_context": {"parameters": {}},
        "reference": {"solution": json.dumps({"variables": {"x": 1}, "objective": 1})},
    }

    row = run_case_method(case, "liveopt_static", client, args, tmp_path)

    assert row["status"] == "completed"
    assert row["official_evaluation"]["official_accuracy"] is True
    assert row["code_output_repair"]["max_attempts"] == 2
    assert row["code_output_repair"]["attempts_used"] == 1
    assert row["token_usage"]["total_tokens"] == 10
    assert row["code_output_repair"]["attempts"][0]["hidden_feedback_used"] is False


def test_static_calibration_repairs_missing_solver_code(tmp_path: Path) -> None:
    missing_code = {
        "problem_understanding": "toy",
        "data_assumptions": [],
        "decision_variables": [{"name": "x", "type": "integer", "meaning": "toy"}],
        "objectives": [{"sense": "max", "expression": "x"}],
        "constraints": [{"expression": "x <= 1", "type": "hard"}],
        "encoding_plan": [{"variable": "x", "encoding_skill": "integer_vector"}],
        "solver_plan": {"solver": "direct", "reason": "toy"},
        "verification_plan": ["run"],
        "output_solution_format": {"objective_value": "number", "solution": "object"},
        "final_answer": {"objective_value": 1, "solution": {"x": 1}},
    }
    repaired = "```python\nimport json\nprint(json.dumps({'objective_value': 1, 'solution': {'x': 1}}))\n```"
    client = FakeContentClient([missing_code, repaired])
    args = SimpleNamespace(
        response_protocol="json",
        no_json_mode=False,
        max_tokens=1000,
        model="fake",
        code_repair_attempts=2,
    )
    case = {
        "case_id": "missing_code_toy",
        "source": "unit",
        "problem_text": "Choose integer x <= 1 to maximize x.",
        "public_context": {"parameters": {}},
        "reference": {"solution": json.dumps({"variables": {"x": 1}, "objective": 1})},
    }

    row = run_case_method(case, "liveopt_static", client, args, tmp_path)

    assert row["status"] == "completed"
    assert row["official_evaluation"]["official_accuracy"] is True
    assert row["code_output_repair"]["attempts_used"] == 1
    assert row["code_output_repair"]["attempts"][0]["reference_feedback_used"] is False


def test_code_block_prompt_exposes_solution_schema_without_values() -> None:
    case = {
        "case_id": "toy",
        "source": "NLP4LP-Hard",
        "problem_text": "Choose investments.",
        "reference": {"solution": json.dumps({"variables": {"InvestClothing": 2500.0, "InvestTech": 500.0}, "objective": 225.0})},
    }

    prompt = build_messages(case, "liveopt_static", response_protocol="code_block")[1]["content"]

    assert "Expected public solution schema hint" in prompt
    assert "InvestClothing" in prompt
    assert "InvestTech" in prompt
    assert "number" in prompt
    assert "2500" not in prompt
    assert "500.0" not in prompt
    assert "225" not in prompt


def test_static_calibration_numeric_reference_evaluation() -> None:
    case = {"reference": {"en_answer": "19.0"}}
    parsed = {"final_answer": {"objective_value": 19, "solution": {"x": 0, "y": 19}}}

    evaluation = evaluate_reference(case, parsed)

    assert evaluation["status"] == "numeric_compared"
    assert evaluation["numeric_match"] is True


def test_static_calibration_reads_nlp4lp_solution_objective() -> None:
    case = {"reference": {"solution": json.dumps({"variables": {"x": 1}, "objective": 42.0})}}
    parsed = {"final_answer": {"objective_value": 42}}

    evaluation = evaluate_reference(case, parsed)

    assert evaluation["reference_value"] == 42.0
    assert evaluation["numeric_match"] is True


def test_static_calibration_official_metric_runs_code_and_checks_solution() -> None:
    case = {
        "public_context": {"parameters": {"profit": 42}},
        "reference": {"solution": json.dumps({"variables": {"x": 2.0}, "objective": 42.0})},
    }
    parsed = {
        "solver_code": {
            "language": "python",
            "code": (
                "import json\n"
                "with open('parameters.json', 'r', encoding='utf-8') as f:\n"
                "    data = json.load(f)\n"
                "print(json.dumps({'objective_value': data['profit'], 'solution': {'x': 2.0}}))\n"
            ),
        },
        "final_answer": {"objective_value": 42, "solution": {"x": 2.0}},
    }

    evaluation = evaluate_official_static(case, parsed)

    assert evaluation["compile_success"] is True
    assert evaluation["runtime_success"] is True
    assert evaluation["objective_exact"] is True
    assert evaluation["solution_exact"] is True
    assert evaluation["official_accuracy"] is True


def test_static_calibration_bwor_metric_uses_objective_only_with_public_tolerance() -> None:
    case = {
        "source": "BWOR",
        "metadata": {
            "official_tolerance_abs": 0.1,
            "official_tolerance_rel": 0.0,
            "reference_solution_required": False,
        },
        "reference": {"answer": 32.43, "solution_status": "optimal"},
    }
    parsed = {
        "solver_code": {
            "language": "python",
            "code": "import json\nprint(json.dumps({'objective_value': 32.49, 'solution': {'feed4': 116.7}}))\n",
        },
        "final_answer": {"objective_value": 32.49, "solution": {"feed4": 116.7}},
    }

    evaluation = evaluate_official_static(case, parsed)

    assert evaluation["compile_success"] is True
    assert evaluation["runtime_success"] is True
    assert evaluation["objective_exact"] is True
    assert evaluation["objective_tolerance"] == 0.1
    assert evaluation["reference_solution_available"] is False
    assert evaluation["solution_exact"] is None
    assert evaluation["official_accuracy"] is True


def test_static_calibration_bwor_metric_handles_no_optimal_status() -> None:
    case = {
        "source": "BWOR",
        "metadata": {
            "official_tolerance_abs": 0.1,
            "reference_solution_required": False,
        },
        "reference": {"solution_status": "no_optimal", "answer": None},
    }
    parsed = {
        "solver_code": {
            "language": "python",
            "code": "import json\nprint(json.dumps({'solution_status': 'infeasible'}))\n",
        },
        "final_answer": {"solution_status": "infeasible"},
    }

    evaluation = evaluate_official_static(case, parsed)

    assert evaluation["status_exact"] is True
    assert evaluation["objective_exact"] is None
    assert evaluation["official_accuracy"] is True


def test_static_calibration_solution_match_allows_public_aliases() -> None:
    case = {
        "reference": {
            "solution": json.dumps(
                {"variables": {"InvestClothing": 2500.0, "InvestTech": 500.0}, "objective": 225.0}
            )
        },
    }
    parsed = {
        "solver_code": {
            "language": "python",
            "code": "import json\nprint(json.dumps({'objective_value': 225.0, 'solution': {'clothing': 2500.0, 'tech': 500.0}}))\n",
        },
        "final_answer": {"objective_value": 225.0, "solution": {"clothing": 2500.0, "tech": 500.0}},
    }

    evaluation = evaluate_official_static(case, parsed)

    assert evaluation["solution_exact"] is True
    assert evaluation["official_accuracy"] is True


def test_static_calibration_solution_match_allows_unnamed_numeric_vector_when_unique() -> None:
    case = {
        "reference": {
            "solution": json.dumps(
                {"variables": {"InvestClothing": 2500.0, "InvestTech": 500.0}, "objective": 225.0}
            )
        },
    }
    parsed = {
        "solver_code": {
            "language": "python",
            "code": "import json\nprint(json.dumps({'objective_value': 225.0, 'solution': {'x': 2500.0, 'y': 500.0}}))\n",
        },
        "final_answer": {"objective_value": 225.0, "solution": {"x": 2500.0, "y": 500.0}},
    }

    evaluation = evaluate_official_static(case, parsed)

    assert evaluation["solution_exact"] is True
    assert evaluation["official_accuracy"] is True


def test_static_calibration_case_sharding() -> None:
    class Args:
        case_ids = ""
        case_offset = 1
        case_stride = 3

    cases = [{"case_id": f"c{i}", "source_id": str(i)} for i in range(8)]

    assert [case["case_id"] for case in select_cases(cases, Args)] == ["c1", "c4", "c7"]


def test_static_calibration_summary_exports_adjusted_match_rate() -> None:
    rows = [
        _row("c1", "liveopt_static", 1, 1),
        _row("c1", "optimai_2025", 1, 1),
        _row("c2", "liveopt_static", 2, 3),
        _row("c2", "optimai_2025", 2, 4),
    ]

    result = summarize(rows)

    assert result["methods"]["liveopt_static"]["numeric_matches"] == 1
    assert result["methods"]["liveopt_static"]["ambiguity_adjusted_total"] == 1
    assert result["common_nonmatch_ambiguities"][0]["case_id"] == "c2"


def test_static_calibration_summary_counts_timeout_in_e2e_rate() -> None:
    rows = [
        _row("c1", "liveopt_static", 1, 1),
        {
            "case_id": "c2",
            "method": "liveopt_static",
            "status": "not_completed_timeout",
            "source_id": "c2",
            "problem_text": "toy",
            "token_usage": {"total_tokens": 0},
            "latency_seconds": 0.0,
            "reference_evaluation": {"status": "missing_numeric_prediction"},
            "official_evaluation": {
                "official_accuracy": False,
                "compile_success": False,
                "runtime_success": False,
                "objective_exact": False,
                "solution_exact": False,
            },
        },
    ]

    result = summarize(rows)
    metrics = result["methods"]["liveopt_static"]

    assert metrics["end_to_end_numeric_success_rate"] == 0.5
    assert metrics["official_accuracy_rate"] == 0.5
    assert metrics["numeric_match_rate"] == 1.0
    assert "1/2 (50.0\\%)" in render_tex(result)
    assert "\\StaticSolvingNLPFourLPRows" in render_tex(result)


def test_static_calibration_render_tex_can_omit_solution_column() -> None:
    rows = [_row("c1", "or_llm_agent_2025", 1, 1)]
    result = summarize(rows)

    tex = render_tex(result, macro_name="StaticSolvingBWORRows", include_solution_column=False)

    assert "\\StaticSolvingBWORRows" in tex
    assert "OR-LLM-Agent" in tex
    assert tex.count("&") == 6


def _row(case_id: str, method: str, reference: float, predicted: float) -> dict:
    return {
        "case_id": case_id,
        "method": method,
        "status": "completed",
        "source_id": case_id,
        "problem_text": "toy",
        "token_usage": {"total_tokens": 10},
        "latency_seconds": 1.0,
        "reference_evaluation": {
            "status": "numeric_compared",
            "reference_value": reference,
            "predicted_value": predicted,
            "numeric_match": reference == predicted,
        },
        "official_evaluation": {
            "official_accuracy": reference == predicted,
            "compile_success": True,
            "runtime_success": True,
            "objective_exact": reference == predicted,
            "solution_exact": reference == predicted,
        },
    }


def test_static_calibration_cache_only_writes_complete_miss_rows(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": "liveopt_static_calibration_case_v1",
                "case_id": "toy_static",
                "source": "unit",
                "source_id": "0",
                "problem_text": "Minimize cost while meeting two demands.",
                "reference": {"answer": "toy"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "liveopt_static_calibration_manifest_v1",
                "case_file": str(cases_path),
                "methods": ["liveopt_static", "optimai_2025"],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/baselines/run_static_calibration.py",
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
        "--methods",
        "liveopt_static,optimai_2025",
        "--model",
        "fake",
        "--cache-only",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)

    summary = json.loads(result.stdout)
    assert summary["status_counts"] == {"cache_miss_blocked": 2}
    rows = [json.loads(line) for line in (out_dir / "static_calibration_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["selected_stage_prompt"]
        assert row["messages"]
        assert row["trace_path"]
        assert row["failure_reason"]
        assert row["validation_feedback"]
