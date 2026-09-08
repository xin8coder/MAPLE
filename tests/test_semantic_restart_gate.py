from __future__ import annotations

import json

from evo2.agents.liveopt_workbench_impl import ScaffoldProject, ScaffoldTrace
from evo2.agents.liveopt_dynamic_impl import select_restart_skill_from_semantic_evidence
from evo2.agents.semantic_restart_gate import (
    LiveOptSemanticRestartGate,
    SemanticRestartDecision,
    build_public_change_digest,
)
from evo2.core.template_optimizer import FitnessResult, SegmentSpec
from scripts.llm_tests.replay_liveopt_dynamic_nldo_artifacts import load_frozen_semantic_decisions
from scripts.llm_tests.run_liveopt_dynamic_nldo_benchmark_full import FrozenSemanticRestartGate


class FakeGateClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def chat(self, messages, temperature=0.0, max_tokens=0, **kwargs):
        self.payloads.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, **kwargs})
        if not self.responses:
            raise AssertionError("semantic decision should have been served from the parsed-choice cache")
        content = self.responses.pop(0)
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }


def make_project(task_ids, *, objective_names=None):
    resources = ["V1", "V2"]
    segment = SegmentSpec(name="assign", kind="assignment", demands=list(task_ids), resources=resources)
    return ScaffoldProject(
        task_id="test",
        data={},
        segments=[segment],
        setup_code="def build_problem(public_context):\n    return {}\n",
        fitness_code="def evaluate(genome, data):\n    return None\n",
        evaluate=lambda genome, data: FitnessResult(scalar=0.0),
        problem_spec={"solver_mode": "moea", "objective_names": objective_names or ["distance", "emission"]},
        trace=ScaffoldTrace(model="fake"),
    )


def contexts():
    previous = {
        "tables": {
            "orders": [
                {"id": "A", "active": True, "x": 0, "priority": 1},
                {"id": "B", "active": True, "x": 1, "priority": 1},
                {"id": "C", "active": False, "x": 20, "priority": 4},
                {"id": "D", "active": False, "x": 21, "priority": 4},
            ]
        }
    }
    current = {
        "tables": {
            "orders": [
                {"id": "A", "active": False, "x": -20, "priority": 4},
                {"id": "B", "active": False, "x": -21, "priority": 4},
                {"id": "C", "active": True, "x": 40, "priority": 1},
                {"id": "D", "active": True, "x": 41, "priority": 1},
            ]
        }
    }
    return previous, current


def high_risk_response():
    return json.dumps(
        {
            "reuse_risk": "high",
            "change_mechanisms": ["decision_support_replacement"],
            "full_vote": True,
            "evidence_paths": ["tables/orders/active", "tables/orders/x"],
            "reason": "The active decision support is disjoint and its geometry is replaced.",
        }
    )


def test_public_change_digest_records_active_support_and_field_changes():
    previous, current = contexts()
    digest = build_public_change_digest(
        previous_public_context=previous,
        public_context=current,
        previous_project=make_project(["A", "B"]),
        project=make_project(["C", "D"]),
    )

    assert "tables/orders/active" in digest["changed_field_paths"]
    assert "tables/orders/x" in digest["changed_field_paths"]
    assert "workbench/segments" in digest["changed_field_paths"]
    assert digest["table_changes"]["orders"]["active_identity_overlap"] == 0.0


def test_restart_selection_uses_only_verified_semantic_vote():
    semantic_full = SemanticRestartDecision(
        reuse_risk="high",
        full_vote=True,
        verified_full_vote=True,
        reason="verified support replacement",
    )
    semantic_warm = SemanticRestartDecision(reuse_risk="low", reason="local change")

    selected, _ = select_restart_skill_from_semantic_evidence(semantic_decision=semantic_full)
    assert selected == "full_restart_v1"
    selected, _ = select_restart_skill_from_semantic_evidence(semantic_decision=semantic_warm)
    assert selected == "warm_restart_v1"
    selected, _ = select_restart_skill_from_semantic_evidence(semantic_decision=None)
    assert selected == "warm_restart_v1"


def test_semantic_vote_requires_evidence_from_verified_public_delta(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "1")
    previous, current = contexts()
    client = FakeGateClient([high_risk_response()])
    gate = LiveOptSemanticRestartGate(
        "Route all active public orders.",
        model="fake",
        client=client,
        cache_dir=tmp_path,
    )

    decision, trace = gate.decide(
        update_id="opaque-update",
        natural_language_update="Replace the active order rows with the supplied current rows.",
        previous_public_context=previous,
        public_context=current,
        previous_project=make_project(["A", "B"]),
        project=make_project(["C", "D"]),
    )

    assert decision.full_vote
    assert decision.verified_full_vote
    assert decision.verified_evidence_paths == ("tables/orders/active", "tables/orders/x")
    assert trace.usage[0]["total_tokens"] == 30


def test_semantic_vote_rejects_a_mechanism_not_supported_by_the_public_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    previous, _ = contexts()
    local = {
        "tables": {
            "orders": [
                {"id": "A", "active": True, "x": 0, "priority": 2},
                {"id": "B", "active": True, "x": 1, "priority": 1},
                {"id": "C", "active": False, "x": 20, "priority": 4},
                {"id": "D", "active": False, "x": 21, "priority": 4},
            ]
        }
    }
    unsupported = json.dumps(
        {
            "reuse_risk": "high",
            "change_mechanisms": ["decision_support_replacement"],
            "full_vote": True,
            "evidence_paths": ["tables/orders/priority"],
            "reason": "Claims replacement despite a local priority edit.",
        }
    )
    gate = LiveOptSemanticRestartGate(
        "Route all active public orders.",
        model="fake",
        client=FakeGateClient([unsupported]),
        cache_dir=tmp_path,
    )

    decision, _ = gate.decide(
        update_id="local-update",
        natural_language_update="Increase one priority.",
        previous_public_context=previous,
        public_context=local,
        previous_project=make_project(["A", "B"]),
        project=make_project(["A", "B"]),
    )

    assert decision.full_vote
    assert not decision.verified_mechanisms
    assert not decision.verified_full_vote


def test_semantic_choice_cache_skips_repeated_model_call(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "1")
    previous, current = contexts()
    first_client = FakeGateClient([high_risk_response()])
    first_gate = LiveOptSemanticRestartGate(
        "Route all active public orders.", model="fake", client=first_client, cache_dir=tmp_path
    )
    kwargs = {
        "update_id": "same-update",
        "natural_language_update": "Replace active orders.",
        "previous_public_context": previous,
        "public_context": current,
        "previous_project": make_project(["A", "B"]),
        "project": make_project(["C", "D"]),
    }
    first, _ = first_gate.decide(**kwargs)

    cached_client = FakeGateClient([])
    cached_gate = LiveOptSemanticRestartGate(
        "Route all active public orders.", model="fake", client=cached_client, cache_dir=tmp_path
    )
    second, trace = cached_gate.decide(**kwargs)

    assert first.verified_full_vote and second.verified_full_vote
    assert second.cache_hit
    assert not cached_client.payloads
    assert trace.usage[0]["semantic_choice_cache_hits"] == 1


def test_successive_calls_keep_a_provider_cacheable_message_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_CACHE", "0")
    low = json.dumps(
        {
            "reuse_risk": "low",
            "change_mechanisms": [],
            "full_vote": False,
            "evidence_paths": ["tables/orders/priority"],
            "reason": "Only local priorities changed.",
        }
    )
    client = FakeGateClient([low, high_risk_response()])
    gate = LiveOptSemanticRestartGate("Route all active public orders.", model="fake", client=client, cache_dir=tmp_path)
    previous, current = contexts()
    local = {
        "tables": {
            "orders": [
                {"id": "A", "active": True, "x": 0, "priority": 2},
                {"id": "B", "active": True, "x": 1, "priority": 1},
                {"id": "C", "active": False, "x": 20, "priority": 4},
                {"id": "D", "active": False, "x": 21, "priority": 4},
            ]
        }
    }
    gate.decide(
        update_id="first",
        natural_language_update="Increase one order priority.",
        previous_public_context=previous,
        public_context=local,
        previous_project=make_project(["A", "B"]),
        project=make_project(["A", "B"]),
    )
    gate.decide(
        update_id="second",
        natural_language_update="Replace active orders.",
        previous_public_context=local,
        public_context=current,
        previous_project=make_project(["A", "B"]),
        project=make_project(["C", "D"]),
    )

    first_messages = client.payloads[0]["messages"]
    second_messages = client.payloads[1]["messages"]
    assert second_messages[: len(first_messages)] == first_messages
    assert second_messages[len(first_messages)]["role"] == "assistant"


def test_artifact_replay_loads_a_frozen_verified_choice(tmp_path):
    decision = SemanticRestartDecision(
        reuse_risk="high",
        mechanisms=("decision_support_replacement",),
        verified_mechanisms=("decision_support_replacement",),
        full_vote=True,
        verified_full_vote=True,
        evidence_paths=("tables/orders/active",),
        verified_evidence_paths=("tables/orders/active",),
        reason="The active support is replaced.",
        model="fake",
        cache_key="frozen-key",
    )
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "episode_id": "NLDO-P010",
                        "stage": 11,
                        "decision": decision.to_record(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_frozen_semantic_decisions([path])

    assert loaded[("NLDO-P010", 11)] == decision


def test_provider_backed_runner_reuses_a_complete_frozen_choice_trajectory(tmp_path):
    rows = []
    for stage in range(1, 13):
        decision = SemanticRestartDecision(
            reuse_risk="high" if stage == 11 else "low",
            full_vote=stage == 11,
            verified_full_vote=stage == 11,
            reason="frozen controller-repeat decision",
            model="fake",
        )
        rows.append(
            {
                "episode_id": "NLDO-P008",
                "stage": stage,
                "decision": decision.to_record(),
            }
        )
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    gate = FrozenSemanticRestartGate("NLDO-P008", [path])
    warm, warm_trace = gate.decide(update_id="NLDO-P008-S01")
    full, full_trace = gate.decide(update_id="NLDO-P008-S11")

    assert not warm.verified_full_vote
    assert full.verified_full_vote
    assert warm.cache_hit and full.cache_hit
    assert warm_trace.usage == [{"total_tokens": 0, "frozen_semantic_decision_hits": 1}]
    assert full_trace.usage == [{"total_tokens": 0, "frozen_semantic_decision_hits": 1}]
