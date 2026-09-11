import pytest

from agent_lvl.context import (
    DEFAULT_SUMMARY_BATCH_MESSAGES,
    summary_batch_messages,
)


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, DEFAULT_SUMMARY_BATCH_MESSAGES),
        ({"SUMMARY_BATCH_MESSAGES": "2"}, 2),
        ({"SUMMARY_BATCH_MESSAGES": "20"}, 20),
    ],
)
def test_summary_batch_messages_reads_environment(
    environment: dict[str, str], expected: int
) -> None:
    assert summary_batch_messages(environment) == expected


@pytest.mark.parametrize("value", ["invalid", "0", "-2", "3"])
def test_summary_batch_messages_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="SUMMARY_BATCH_MESSAGES"):
        summary_batch_messages({"SUMMARY_BATCH_MESSAGES": value})
