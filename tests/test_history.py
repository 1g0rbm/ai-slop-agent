from datetime import UTC, datetime

import pytest

from agent_lvl.history import (
    DEFAULT_HISTORY_LIMIT,
    Exchange,
    SQLiteHistory,
    history_limit,
)


def make_exchange(number: int) -> Exchange:
    timestamp = datetime(2026, 1, number, 12, tzinfo=UTC)
    return Exchange(
        user_content=f"Вопрос {number}",
        assistant_content=f"Ответ {number}",
        user_created_at=timestamp,
        assistant_created_at=timestamp,
        model=f"model-{number}",
    )


def test_history_stores_and_loads_exchanges(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    exchange = make_exchange(1)

    history.add(exchange)

    assert history.all() == [exchange]
    assert history.count() == 1


def test_history_returns_recent_exchanges_in_chronological_order(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 4):
        history.add(make_exchange(number))

    assert history.recent(0) == []
    assert history.recent(2) == [make_exchange(2), make_exchange(3)]
    assert history.recent(-1) == [
        make_exchange(1),
        make_exchange(2),
        make_exchange(3),
    ]


def test_history_clear_removes_all_exchanges(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange(1))

    history.clear()

    assert history.all() == []
    assert history.count() == 0


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, DEFAULT_HISTORY_LIMIT),
        ({"HISTORY_LIMIT": "0"}, 0),
        ({"HISTORY_LIMIT": "-1"}, -1),
    ],
)
def test_history_limit_reads_environment(
    environment: dict[str, str], expected: int
) -> None:
    assert history_limit(environment) == expected


@pytest.mark.parametrize("value", ["invalid", "-2"])
def test_history_limit_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="HISTORY_LIMIT"):
        history_limit({"HISTORY_LIMIT": value})
