from types import SimpleNamespace

from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import (
    build_metric_trace_context,
    maybe_thin_trace_row,
)


def test_metric_trace_context_uses_explicit_strong_reference_payload() -> None:
    public_episode = {
        "episode_id": "NLDO-PTEST",
        "domain": "unknown",
        "family": "test",
        "hidden_initial_state": {},
        "hidden_update_oracle": [],
        "evaluation": {"reference_trajectory": [{"update_id": "old"}]},
    }
    strong_reference_episode = {
        "episode_id": "NLDO-PTEST",
        "evaluation": {"reference_trajectory": [{"update_id": "strong"}]},
    }

    context = build_metric_trace_context(
        public_episode,
        reference_episode=strong_reference_episode,
    )

    assert context["reference_steps"] == [{"update_id": "strong"}]
    assert context["metric_reference_episode_id"] == "NLDO-PTEST"


def test_metric_trace_thinning_can_retain_final_population_for_matched_replay() -> None:
    row = {
        "initial_solver_result": {"metadata": {"final_population": [{"genome": {"x": 1}}]}},
        "update_results": [
            {"solver_result": {"metadata": {"final_population": [{"genome": {"x": 2}}]}}}
        ],
    }
    args = SimpleNamespace(
        record_metric_trace=True,
        record_trace_archives=False,
        retain_final_population=True,
        retain_final_population_stage=[],
        drop_final_population=False,
    )

    maybe_thin_trace_row(args, row)

    assert row["initial_solver_result"]["metadata"]["final_population"]
    assert row["update_results"][0]["solver_result"]["metadata"]["final_population"]


def test_metric_trace_thinning_drops_final_population_by_default() -> None:
    row = {
        "initial_solver_result": {"metadata": {"final_population": [{"genome": {"x": 1}}]}},
        "update_results": [
            {"solver_result": {"metadata": {"final_population": [{"genome": {"x": 2}}]}}}
        ],
    }
    args = SimpleNamespace(
        record_metric_trace=True,
        record_trace_archives=False,
        retain_final_population=False,
        retain_final_population_stage=[],
        drop_final_population=False,
    )

    maybe_thin_trace_row(args, row)

    assert "final_population" not in row["initial_solver_result"]["metadata"]
    assert "final_population" not in row["update_results"][0]["solver_result"]["metadata"]


def test_metric_trace_thinning_retains_only_requested_stage_population() -> None:
    row = {
        "initial_solver_result": {"metadata": {"final_population": [{"genome": {"x": 0}}]}},
        "update_results": [
            {"solver_result": {"metadata": {"final_population": [{"genome": {"x": 1}}]}}},
            {"solver_result": {"metadata": {"final_population": [{"genome": {"x": 2}}]}}},
        ],
    }
    args = SimpleNamespace(
        record_metric_trace=True,
        record_trace_archives=False,
        retain_final_population=False,
        retain_final_population_stage=[1],
        drop_final_population=False,
    )

    maybe_thin_trace_row(args, row)

    assert "final_population" not in row["initial_solver_result"]["metadata"]
    assert row["update_results"][0]["solver_result"]["metadata"]["final_population"]
    assert "final_population" not in row["update_results"][1]["solver_result"]["metadata"]


def test_terminal_shadow_can_drop_population_without_metric_trace() -> None:
    row = {
        "initial_solver_result": {"metadata": {"final_population": [{"genome": {"x": 0}}]}},
        "update_results": [
            {"solver_result": {"metadata": {"final_population": [{"genome": {"x": 1}}]}}}
        ],
    }
    args = SimpleNamespace(
        record_metric_trace=False,
        record_trace_archives=False,
        retain_final_population=False,
        retain_final_population_stage=[],
        drop_final_population=True,
    )

    maybe_thin_trace_row(args, row)

    assert "final_population" not in row["initial_solver_result"]["metadata"]
    assert "final_population" not in row["update_results"][0]["solver_result"]["metadata"]
