from scripts.analysis.audit_nldo_public_state_alignment import equivalent


def test_pipe_serialized_public_collection_matches_hidden_list() -> None:
    assert equivalent("E02|E05|E07|E08", ["E02", "E05", "E07", "E08"])
    assert equivalent(["E02", "E05"], "E02|E05")


def test_pipe_serialized_public_collection_preserves_order_and_membership() -> None:
    assert not equivalent("E02|E05", ["E05", "E02"])
    assert not equivalent("E02|E05", ["E02", "E07"])
