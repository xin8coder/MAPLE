from evo2.evaluation import moea_metrics
from evo2.evaluation.reference_solvers.moea_reference import evaluate_against_reference


def test_hypervolume_uses_exact_2d_when_pymoo_unavailable(monkeypatch):
    monkeypatch.setattr(moea_metrics, "HV", None)
    monkeypatch.setattr(moea_metrics, "np", None)

    value = moea_metrics.approximate_hypervolume(
        [
            {"objectives": {"a": 0.2, "b": 0.8}},
            {"objectives": {"a": 0.8, "b": 0.2}},
        ],
        ["a", "b"],
        ref={"a": 1.0, "b": 1.0},
        ideal={"a": 0.0, "b": 0.0},
        samples=0,
    )

    assert value == 0.28


def test_hypervolume_uses_exact_3d_when_pymoo_unavailable(monkeypatch):
    monkeypatch.setattr(moea_metrics, "HV", None)
    monkeypatch.setattr(moea_metrics, "np", None)

    value = moea_metrics.approximate_hypervolume(
        [{"objectives": {"a": 0.2, "b": 0.3, "c": 0.4}}],
        ["a", "b", "c"],
        ref={"a": 1.0, "b": 1.0, "c": 1.0},
        ideal={"a": 0.0, "b": 0.0, "c": 0.0},
        samples=0,
    )

    assert value == 0.336


def test_reference_hv_uses_reference_archive_only():
    reference_archive = [
        {"objectives": {"a": 1.0, "b": 4.0}},
        {"objectives": {"a": 2.0, "b": 2.0}},
        {"objectives": {"a": 4.0, "b": 1.0}},
    ]
    near_candidate = [{"objectives": {"a": 1.5, "b": 3.0}}]
    far_candidate = [{"objectives": {"a": 100.0, "b": 100.0}}]

    near_metrics = evaluate_against_reference(near_candidate, reference_archive, ["a", "b"])
    far_metrics = evaluate_against_reference(far_candidate, reference_archive, ["a", "b"])

    assert near_metrics["reference_hv"] == far_metrics["reference_hv"]


def test_reference_hv_point_has_enough_margin_to_rank_weaker_feasible_archives():
    reference_archive = [
        {"objectives": {"a": 1.0, "b": 4.0}},
        {"objectives": {"a": 2.0, "b": 2.0}},
        {"objectives": {"a": 4.0, "b": 1.0}},
    ]
    weaker_candidate = [{"objectives": {"a": 5.0, "b": 2.0}}]

    metrics = evaluate_against_reference(weaker_candidate, reference_archive, ["a", "b"])

    assert metrics["reference_hv"] > 0.0
    assert metrics["hv"] > 0.0
    assert 0.0 < metrics["hv_ratio"] < 1.0
