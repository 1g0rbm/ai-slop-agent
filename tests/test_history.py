import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from agent_lvl.history import (
    DEFAULT_HISTORY_LIMIT,
    ConversationSummary,
    Exchange,
    FactOperation,
    MaintenanceUsage,
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


def test_history_creates_default_conversation_and_main_branch(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")

    conversation = history.conversation()
    branch = history.active_branch()

    assert conversation.id == 1
    assert conversation.title == "Основной диалог"
    assert conversation.strategy == "summary"
    assert conversation.active_branch_id == branch.id
    assert conversation.created_at.tzinfo is not None
    assert conversation.updated_at.tzinfo is not None
    assert branch.name == "main"
    assert branch.parent_branch_id is None
    assert branch.checkpoint_exchange_id is None
    assert branch.head_exchange_id is None
    assert branch.created_at.tzinfo is not None
    assert history.list_branches() == [branch]


def test_history_persists_strategy_and_rejects_invalid_value(tmp_path) -> None:
    path = tmp_path / "history.db"
    history = SQLiteHistory(path)

    history.set_strategy("branching")

    assert SQLiteHistory(path).strategy() == "branching"
    with pytest.raises(ValueError, match="стратег"):
        history.set_strategy("unknown")
    assert history.strategy() == "branching"


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
    legacy_summary = ConversationSummary(
        content="Старая сводка",
        through_exchange_id=1,
        summarized_messages=2,
        model="legacy-summary-model",
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
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
        connection.execute(
            """
            CREATE TABLE conversation_summary (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                content TEXT NOT NULL,
                through_exchange_id INTEGER NOT NULL,
                summarized_messages INTEGER NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        for number in (1, 2):
            exchange = make_exchange(number)
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
        connection.execute(
            """
            INSERT INTO conversation_summary (
                id, content, through_exchange_id, summarized_messages,
                model, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?)
            """,
            (
                legacy_summary.content,
                legacy_summary.through_exchange_id,
                legacy_summary.summarized_messages,
                legacy_summary.model,
                legacy_summary.updated_at.isoformat(),
            ),
        )

    history = SQLiteHistory(path)

    assert history.all() == [make_exchange(1), make_exchange(2)]
    assert history.token_totals().untracked_exchanges == 2
    assert history.summary() == legacy_summary
    main = history.active_branch()
    assert main.name == "main"
    assert main.parent_branch_id is None
    assert main.head_exchange_id == 2
    with sqlite3.connect(path) as connection:
        migrated = connection.execute(
            """
            SELECT id, conversation_id, branch_id, parent_exchange_id
            FROM exchanges
            ORDER BY id
            """
        ).fetchall()
        old_summary_count = connection.execute(
            "SELECT COUNT(*) FROM conversation_summary"
        ).fetchone()[0]
    assert migrated == [(1, 1, main.id, None), (2, 1, main.id, 1)]
    assert old_summary_count == 0


def test_history_keeps_independent_paths_for_branches_from_checkpoint(
    tmp_path,
) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange(1))
    checkpoint = history.create_checkpoint("Общая точка")

    left = history.create_branch("Левая", checkpoint.name)
    right = history.create_branch("Правая", checkpoint.name)

    assert checkpoint.exchange_id == 1
    assert left.checkpoint_exchange_id == checkpoint.exchange_id
    assert right.checkpoint_exchange_id == checkpoint.exchange_id
    assert left.parent_branch_id == checkpoint.branch_id
    assert right.parent_branch_id == checkpoint.branch_id

    history.switch_branch(left.name)
    history.add(make_exchange(2))
    assert history.all() == [make_exchange(1), make_exchange(2)]

    history.switch_branch(right.name)
    history.add(make_exchange(3))
    assert history.all() == [make_exchange(1), make_exchange(3)]

    history.switch_branch(left.name)
    assert history.all() == [make_exchange(1), make_exchange(2)]


def test_history_creates_branch_from_current_head(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange(1))
    history.add(make_exchange(2))

    branch = history.create_branch("От текущей вершины")

    assert branch.checkpoint_exchange_id == 2
    assert branch.head_exchange_id == 2
    assert branch.parent_branch_id == history.active_branch().id
    history.switch_branch(branch.name)
    history.add(make_exchange(3))
    assert history.all() == [
        make_exchange(1),
        make_exchange(2),
        make_exchange(3),
    ]


def test_history_selects_summary_batch_before_recent_exchanges(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 6):
        history.add(make_exchange(number))

    batch = history.summary_batch(recent_limit=2, batch_exchanges=2)

    assert [item.exchange for item in batch] == [make_exchange(1), make_exchange(2)]


def test_new_branch_summarizes_shared_path_from_root(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    for number in range(1, 6):
        history.add(make_exchange(number))
    history.create_branch("child")
    history.switch_branch("child")

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


def test_history_keeps_summary_isolated_per_branch(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange(1))
    history.create_checkpoint("Основа")
    history.create_branch("Первая", "Основа")
    history.create_branch("Вторая", "Основа")

    history.switch_branch("Первая")
    history.add(make_exchange(2))
    first_summary = ConversationSummary(
        content="Сводка первой ветки",
        through_exchange_id=2,
        summarized_messages=2,
        model="summary-model",
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    history.save_summary(first_summary)

    history.switch_branch("Вторая")
    assert history.summary() is None
    history.add(make_exchange(3))
    second_summary = ConversationSummary(
        content="Сводка второй ветки",
        through_exchange_id=3,
        summarized_messages=2,
        model="summary-model",
        updated_at=datetime(2026, 2, 2, tzinfo=UTC),
    )
    history.save_summary(second_summary)

    history.switch_branch("Первая")
    assert history.summary() == first_summary
    history.switch_branch("Вторая")
    assert history.summary() == second_summary


def test_history_applies_fact_operations_atomically_with_exchange(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(
        make_exchange(1),
        fact_operations=[FactOperation("set", "goal", "project", "Первое значение")],
    )
    fact = history.facts()[0]
    assert (fact.category, fact.key, fact.value) == (
        "goal",
        "project",
        "Первое значение",
    )
    assert fact.source_exchange_id == 1
    assert fact.created_at.tzinfo is not None
    assert fact.updated_at.tzinfo is not None

    history.add(
        make_exchange(2),
        fact_operations=[
            FactOperation("set", "goal", "project", "Новое значение"),
            FactOperation("set", "preference", "language", "Русский"),
        ],
    )
    assert [
        (fact.category, fact.key, fact.value, fact.source_exchange_id)
        for fact in history.facts()
    ] == [
        ("goal", "project", "Новое значение", 2),
        ("preference", "language", "Русский", 2),
    ]

    with pytest.raises(ValueError, match="категория"):
        history.add(
            make_exchange(3),
            fact_operations=[FactOperation("set", "unknown", "key", "value")],
        )
    assert history.count() == 2
    assert history.facts()[0].value == "Новое значение"

    history.add(
        make_exchange(3),
        fact_operations=[
            FactOperation("delete", "goal", "project"),
            FactOperation("delete", "preference", "language"),
        ],
    )
    assert history.facts() == []
    assert history.count() == 3


def test_history_groups_maintenance_and_reports_overall_tokens(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(
        replace(
            make_exchange(1),
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
    )
    history.record_maintenance(
        "summary",
        "summary",
        "summary-model",
        usage=MaintenanceUsage(2, 3, 5),
        request_id="summary-1",
    )
    history.record_maintenance("summary", "summary", "summary-model")
    history.record_maintenance(
        "facts",
        "sticky",
        "facts-model",
        input_tokens=4,
        output_tokens=6,
        total_tokens=10,
        request_id="facts-1",
    )

    totals = history.maintenance_totals()

    assert set(totals) == {"summary", "facts"}
    assert totals["summary"].operations == 2
    assert totals["summary"].total_tokens == 5
    assert totals["summary"].untracked_operations == 1
    assert not totals["summary"].complete
    assert totals["facts"].operations == 1
    assert totals["facts"].input_tokens == 4
    assert totals["facts"].output_tokens == 6
    assert totals["facts"].total_tokens == 10
    assert totals["facts"].complete

    token_totals = history.token_totals()
    assert token_totals.input_tokens == 10
    assert token_totals.output_tokens == 5
    assert token_totals.total_tokens == 15
    assert token_totals.maintenance_input_tokens == 6
    assert token_totals.maintenance_output_tokens == 9
    assert token_totals.maintenance_total_tokens == 15
    assert token_totals.untracked_maintenance == 1
    assert token_totals.overall_input_tokens == 16
    assert token_totals.overall_output_tokens == 14
    assert token_totals.overall_total_tokens == 30


def test_token_totals_include_all_independent_branches(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(replace(make_exchange(1), total_tokens=10))
    history.create_checkpoint("fork")
    history.create_branch("left", "fork")
    history.create_branch("right", "fork")
    history.switch_branch("left")
    history.add(replace(make_exchange(2), total_tokens=20))
    history.switch_branch("right")
    history.add(replace(make_exchange(3), total_tokens=30))

    assert history.token_totals().total_tokens == 60


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


def test_history_clear_resets_all_extended_state(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.set_strategy("branching")
    history.add(
        make_exchange(1),
        fact_operations=[FactOperation("set", "decision", "mode", "strict")],
    )
    history.create_checkpoint("Точка")
    history.create_branch("Эксперимент", "Точка")
    history.switch_branch("Эксперимент")
    history.add(make_exchange(2))
    history.save_summary(
        ConversationSummary(
            content="Сводка ветки",
            through_exchange_id=2,
            summarized_messages=2,
            model="summary-model",
            updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        )
    )
    history.record_maintenance("summary", "branching", "summary-model", 1, 2, 3)

    history.clear()

    conversation = history.conversation()
    branches = history.list_branches()
    assert conversation.strategy == "summary"
    assert len(branches) == 1
    assert branches[0].name == "main"
    assert conversation.active_branch_id == branches[0].id
    assert branches[0].head_exchange_id is None
    assert branches[0].checkpoint_exchange_id is None
    assert history.all() == []
    assert history.summary() is None
    assert history.facts() == []
    assert history.maintenance_totals() == {}
    assert history.token_totals().overall_total_tokens == 0
    assert history.create_checkpoint("Точка").exchange_id is None


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
