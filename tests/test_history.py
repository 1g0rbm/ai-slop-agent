import sqlite3
from datetime import UTC, datetime

import pytest

from agent_lvl.history import (
    DEFAULT_HISTORY_LIMIT,
    ConversationSummary,
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


def test_history_sums_tokens_and_reports_missing_usage(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    history.add(
        Exchange(
            user_content="Учтённый вопрос",
            assistant_content="Учтённый ответ",
            user_created_at=timestamp,
            assistant_created_at=timestamp,
            model="model",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        )
    )
    history.add(make_exchange(2))

    totals = history.token_totals()

    assert totals.input_tokens == 100
    assert totals.output_tokens == 20
    assert totals.total_tokens == 120
    assert totals.untracked_exchanges == 1
    assert not totals.complete


def test_history_migrates_existing_database(tmp_path) -> None:
    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                user_created_at TEXT NOT NULL,
                assistant_created_at TEXT NOT NULL,
                model TEXT NOT NULL
            )
            """
        )
        exchange = make_exchange(1)
        connection.execute(
            """
            INSERT INTO exchanges (
                user_content, assistant_content, user_created_at,
                assistant_created_at, model
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                exchange.user_content,
                exchange.assistant_content,
                exchange.user_created_at.isoformat(),
                exchange.assistant_created_at.isoformat(),
                exchange.model,
            ),
        )

    history = SQLiteHistory(path)

    assert history.all() == [make_exchange(1)]
    assert history.token_totals().untracked_exchanges == 1


def test_history_selects_summary_batch_before_recent_exchanges(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 6):
        history.add(make_exchange(number))

    batch = history.summary_batch(recent_limit=2, batch_exchanges=2)

    assert [item.exchange for item in batch] == [make_exchange(1), make_exchange(2)]


def test_history_stores_summary_and_continues_after_checkpoint(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 5):
        history.add(make_exchange(number))
    summary = ConversationSummary(
        content="Сводка первых двух пар",
        through_exchange_id=2,
        summarized_messages=4,
        model="summary-model",
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
    )

    history.save_summary(summary)

    assert history.summary() == summary
    assert [
        item.exchange for item in history.recent_after(summary.through_exchange_id, -1)
    ] == [make_exchange(3), make_exchange(4)]


def test_history_clear_removes_exchanges_and_summary(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange(1))
    history.save_summary(
        ConversationSummary(
            content="Сводка",
            through_exchange_id=1,
            summarized_messages=2,
            model="model",
            updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
    )

    history.clear()

    assert history.all() == []
    assert history.count() == 0
    assert history.summary() is None


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
