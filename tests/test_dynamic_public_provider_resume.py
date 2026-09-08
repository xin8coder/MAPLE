from __future__ import annotations

import json
from types import SimpleNamespace

from evo2.agents.kimi_client import KimiQuotaLimitError, KimiTransportError
from scripts.baselines import run_nldo_dynamic_public_baselines as dynamic


def _row(stage: int, *, feasible: bool = True) -> dict:
    solution = {"x": stage}
    return {
        "method": "react_tools",
        "episode_id": "NLDO-TOY",
        "stage_index": stage,
        "update_id": "initial" if stage == 0 else f"t{stage:03d}",
        "status": "completed" if feasible else "hidden_rejected",
        "hidden_evaluation": {
            "feasible": feasible,
            "normalized_score": 1.0 if feasible else 0.0,
            "canonical_solution": solution if feasible else {},
        },
        "solution": solution,
        "candidate_archive": [{"solution": solution}],
        "parsed_output": {"solver_code": {"code": "print('{}')"}},
        "code_execution": {"compile_success": True, "runtime_success": True},
        "token_usage": dynamic.zero_usage(),
        "latency_seconds": 0.0,
    }


def _episode() -> dict:
    return {
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


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        max_updates=2,
        model="k3[1m]",
        code_repair_attempts=3,
        continue_after_hidden_rejection=False,
    )


def _persistent_row(stage: int, *, feasible: bool = True, public_ok: bool = True) -> dict:
    row = _row(stage, feasible=feasible)
    row["method"] = "persistent_react"
    row["public_state_available"] = public_ok
    row["parsed_output"] = {
        "solver_code": {"language": "python", "code": f"print({stage})"},
        "final_answer": {"objective_value": float(stage), "solution": {"x": stage}},
    }
    return row


def test_saved_prefix_starts_at_first_missing_state(monkeypatch, tmp_path) -> None:
    called: list[int] = []

    # Keep positional-call compatibility explicit because the production runner
    # passes the previous public state and archive as positional arguments.
    def fake_stage(*call_args, **call_kwargs):
        del call_kwargs
        stage = call_args[5]
        called.append(int(stage["stage_index"]))
        return _row(int(stage["stage_index"]))

    monkeypatch.setattr(dynamic, "run_stage_method", fake_stage)
    run, rows = dynamic.run_episode_method(
        _args(),
        None,
        _episode(),
        "react_tools",
        tmp_path,
        prefix_rows=[_row(0), _row(1)],
    )
    assert called == [2]
    assert [row["stage_index"] for row in rows] == [0, 1, 2]
    assert run["run_status"] == "completed"
    assert run["saved_prefix_last_stage"] == 1


def test_quota_keeps_feasible_prefix_and_next_state(monkeypatch, tmp_path) -> None:
    def limited_stage(*call_args, **call_kwargs):
        del call_args, call_kwargs
        raise KimiQuotaLimitError(
            "weekly quota exhausted",
            status=403,
            detail="usage limit for this billing cycle",
            limit_scope="weekly",
        )

    monkeypatch.setattr(dynamic, "run_stage_method", limited_stage)
    run, rows = dynamic.run_episode_method(
        _args(),
        None,
        _episode(),
        "react_tools",
        tmp_path,
        prefix_rows=[_row(0)],
    )
    assert [row["stage_index"] for row in rows] == [0]
    assert run["run_status"] == "quota_limited"
    assert run["next_stage_index"] == 1
    assert run["quota_limit"]["limit_scope"] == "weekly"


def test_transport_failure_keeps_prefix_without_recording_algorithm_failure(
    monkeypatch, tmp_path
) -> None:
    def interrupted_stage(*call_args, **call_kwargs):
        del call_args, call_kwargs
        raise KimiTransportError(
            "stream interrupted",
            detail="RemoteDisconnected",
            stream_stats={"data_event_count": 17, "done_seen": False},
        )

    monkeypatch.setattr(dynamic, "run_stage_method", interrupted_stage)
    run, rows = dynamic.run_episode_method(
        _args(),
        None,
        _episode(),
        "react_tools",
        tmp_path,
        prefix_rows=[_row(0)],
    )
    assert [row["stage_index"] for row in rows] == [0]
    assert run["run_status"] == "infrastructure_limited"
    assert run["next_stage_index"] == 1
    assert run["resumable"] is True
    assert run["infrastructure_limit"]["stream"]["data_event_count"] == 17


def test_prefix_loader_stops_at_gap_or_recorded_failure(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        _row(0),
        _row(1),
        _row(3),
        {**_row(0), "episode_id": "NLDO-FAIL"},
        {**_row(1, feasible=False), "episode_id": "NLDO-FAIL"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prefixes = dynamic.load_feasible_stage_prefixes([path])
    assert [row["stage_index"] for row in prefixes[("react_tools", "NLDO-TOY")]] == [0, 1]
    assert [row["stage_index"] for row in prefixes[("react_tools", "NLDO-FAIL")]] == [0]


def test_persistent_react_carries_code_and_accepted_answer(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_stage(*call_args, **call_kwargs):
        stage = call_args[5]
        captured.append(
            {
                "stage_index": int(stage["stage_index"]),
                "previous_artifact": call_args[8],
                "previous_population": call_args[9],
                "previous_final_answer": call_kwargs.get("previous_final_answer"),
            }
        )
        return _persistent_row(int(stage["stage_index"]))

    monkeypatch.setattr(dynamic, "run_stage_method", fake_stage)
    run, rows = dynamic.run_episode_method(_args(), None, _episode(), "persistent_react", tmp_path)

    assert run["run_status"] == "completed"
    assert [row["stage_index"] for row in rows] == [0, 1, 2]
    assert captured[0]["previous_final_answer"] is None
    assert captured[0]["previous_artifact"] is None
    # Stage 1 carries stage 0's full solver code and accepted final_answer JSON.
    assert captured[1]["previous_artifact"]["solver_code"] == "print(0)"
    assert captured[1]["previous_final_answer"] == {"objective_value": 0.0, "solution": {"x": 0}}
    assert captured[2]["previous_artifact"]["solver_code"] == "print(1)"
    assert captured[2]["previous_final_answer"] == {"objective_value": 1.0, "solution": {"x": 1}}
    # persistent_react never receives a candidate population.
    assert all(item["previous_population"] == [] for item in captured)


def test_persistent_react_carried_state_requires_public_success(monkeypatch, tmp_path) -> None:
    captured: list[dict] = []

    def fake_stage(*call_args, **call_kwargs):
        stage = call_args[5]
        captured.append(call_kwargs.get("previous_final_answer"))
        index = int(stage["stage_index"])
        return _persistent_row(index, public_ok=index != 1)

    monkeypatch.setattr(dynamic, "run_stage_method", fake_stage)
    args = _args()
    args.continue_after_hidden_rejection = True
    run, rows = dynamic.run_episode_method(args, None, _episode(), "persistent_react", tmp_path)

    assert len(rows) == 3
    # Stage 1 produced no publicly executable output, so stage 2 still carries stage 0's accepted answer.
    assert captured[2] == {"objective_value": 0.0, "solution": {"x": 0}}


def test_persistent_react_case_embeds_carried_state() -> None:
    episode = _episode()
    stage = {"update_id": "t001", "stage_index": 1, "updates": episode["update_stream"][:1]}
    case = dynamic.dynamic_stage_case(
        episode,
        {"tables": {}},
        stage,
        "persistent_react",
        {"x": 0},
        {"solver_code": "print(0)"},
        [],
        previous_final_answer={"objective_value": 0.0, "solution": {"x": 0}},
    )
    parameters = case["public_context"]["parameters"]
    carried = parameters["persistent_react_carried_state"]
    assert carried["had_solver_code"] is True
    assert carried["had_accepted_output"] is True
    assert carried["previous_stage_solver_code"] == "print(0)"
    assert carried["previous_accepted_final_answer"]["objective_value"] == 0.0
    assert parameters["shared_lsm"]["previous_candidate_count"] == 0
    assert parameters["shared_lsm"]["previous_candidate_population"] == []
    assert "persistent_react_carried_state" in case["problem_text"]

    initial = dynamic.dynamic_stage_case(
        episode,
        {"tables": {}},
        {"update_id": "initial", "stage_index": 0, "updates": []},
        "persistent_react",
        None,
    )
    initial_carried = initial["public_context"]["parameters"]["persistent_react_carried_state"]
    assert initial_carried["had_solver_code"] is False
    assert initial_carried["had_accepted_output"] is False
    assert initial_carried["previous_stage_solver_code"] == ""


def test_persistent_react_schema_matches_react_tools_fields() -> None:
    from scripts.baselines.run_static_calibration import (
        METHOD_REQUIRED_FIELDS,
        SYSTEM_PROMPTS,
        build_messages,
        validate_parsed_output,
    )

    assert METHOD_REQUIRED_FIELDS["persistent_react"] == METHOD_REQUIRED_FIELDS["react_tools"]
    assert "persistent" in SYSTEM_PROMPTS["persistent_react"].lower()
    parsed = {
        "thought": "reuse carried code",
        "react_trace": [{"thought": "t", "action": "a", "observation": "o"}],
        "tool_plan": [{"tool": "scipy.optimize", "purpose": "solve"}],
        "validation_plan": ["compile", "run"],
        "solver_code": {"language": "python", "code": "print('{}')"},
        "final_answer": {"objective_value": 0.0, "solution": {}},
    }
    assert validate_parsed_output("persistent_react", parsed) == []
    assert validate_parsed_output("persistent_react", {})  # missing required fields are flagged
    messages = build_messages(
        {"case_id": "toy", "source": "NLDO-dynamic-public", "problem_text": "Choose x.", "reference": {}},
        "persistent_react",
    )
    assert "Persistent ReAct" in messages[0]["content"]
    assert "never a diff" in messages[1]["content"]


def test_table2_grid_includes_initial_state() -> None:
    from scripts.analysis import export_main_dynamic_hv_heatmap as table2

    assert table2.STAGES == list(range(13))
    assert len(table2.DS_EPISODES) * len(table2.STAGES) == 117
    assert len(table2.DM_EPISODES) * len(table2.STAGES) == 78


def test_table2_requeues_legacy_transport_failure() -> None:
    from scripts.experiments import run_table2_model_family_q2 as table2_run

    row = {
        "run_status": "stopped_after_failure",
        "stage_traces": [
            {"failure_reason": "Remote end closed connection without response"}
        ],
    }
    assert table2_run.is_legacy_infrastructure_failure(row) is True
    row["stage_traces"][0]["failure_reason"] = "hidden evaluator rejected candidate"
    assert table2_run.is_legacy_infrastructure_failure(row) is False
