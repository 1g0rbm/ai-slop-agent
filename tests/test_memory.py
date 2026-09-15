import sqlite3
from datetime import UTC, datetime

import pytest

from agent_lvl.history import Exchange, SQLiteHistory
from agent_lvl.memory import DEFAULT_OWNER_ID, MemoryType, TaskStatus


def make_exchange() -> Exchange:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Exchange("Вопрос", "Ответ", timestamp, timestamp, "model")


def test_memory_levels_have_separate_storage_and_stable_ids(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange())
    task = history.create_task("Реализовать память")

    working = history.set_memory(MemoryType.WORKING, "context", "language", "Python")
    updated = history.set_memory(
        MemoryType.WORKING, "context", "language", "Python 3.12"
    )
    long_term = history.set_memory(
        MemoryType.LONG_TERM, "profile", "language", "Русский"
    )

    assert working.id == updated.id
    assert updated.task_id == task.id
    assert updated.value == "Python 3.12"
    assert long_term.owner_id == DEFAULT_OWNER_ID
    assert updated.created_at.tzinfo is not None
    assert long_term.updated_at.tzinfo is not None
    assert history.list_memory(MemoryType.SHORT_TERM)[0].id == 1
    assert history.list_memory(MemoryType.WORKING) == [updated]
    assert history.list_memory(MemoryType.LONG_TERM) == [long_term]


def test_long_term_memory_uses_conversation_owner(tmp_path) -> None:
    path = tmp_path / "history.db"
    history = SQLiteHistory(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE conversations SET owner_id = 42 WHERE id = 1")

    entry = history.set_memory(
        MemoryType.LONG_TERM, "profile", "timezone", "Europe/Moscow"
    )

    assert entry.owner_id == 42
    assert history.list_memory(MemoryType.LONG_TERM) == [entry]
    with sqlite3.connect(path) as connection:
        stored_owner = connection.execute(
            "SELECT owner_id FROM long_term_memory WHERE id = ?", (entry.id,)
        ).fetchone()
    assert stored_owner == (42,)


def test_memory_validates_scopes_categories_and_limits(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")

    with pytest.raises(ValueError, match="активная задача"):
        history.list_memory(MemoryType.WORKING)
    with pytest.raises(ValueError, match="только для чтения"):
        history.set_memory(MemoryType.SHORT_TERM, "exchange", "1", "value")

    history.create_task("Задача")
    with pytest.raises(ValueError, match="категория"):
        history.set_memory(MemoryType.WORKING, "profile", "key", "value")
    with pytest.raises(ValueError, match="100"):
        history.set_memory(MemoryType.WORKING, "task", "k" * 101, "value")
    with pytest.raises(ValueError, match="4000"):
        history.set_memory(MemoryType.LONG_TERM, "knowledge", "key", "v" * 4001)
    with pytest.raises(ValueError, match="200"):
        history.create_task("t" * 201)


def test_task_lifecycle_and_single_active_task(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    first = history.create_task("Первая")

    assert first.status is TaskStatus.ACTIVE
    assert history.active_task() == first
    with pytest.raises(ValueError, match="уже есть"):
        history.create_task("Вторая")

    completed = history.complete_task()
    assert completed.status is TaskStatus.COMPLETED
    assert completed.completed_at is not None
    second = history.create_task("Вторая")
    cancelled = history.cancel_task(second.id)

    assert cancelled.status is TaskStatus.CANCELLED
    assert history.active_task() is None
    assert history.list_tasks() == [cancelled, completed]


def test_promote_is_atomic_and_leaves_working_source(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.create_task("Задача")
    source = history.set_memory(MemoryType.WORKING, "artifact", "report", "Черновик")

    promoted = history.promote_memory("artifact", "report", "knowledge", "final-report")

    assert promoted.value == source.value
    assert promoted.key == "final-report"
    assert history.list_memory(MemoryType.WORKING) == [source]
    with pytest.raises(ValueError, match="не найдена"):
        history.promote_memory("artifact", "missing", "knowledge")
    assert history.list_memory(MemoryType.LONG_TERM) == [promoted]


def test_clear_cancels_task_clears_working_and_preserves_long_term(tmp_path) -> None:
    history = SQLiteHistory(tmp_path / "history.db")
    history.add(make_exchange())
    task = history.create_task("Задача")
    history.set_memory(MemoryType.WORKING, "progress", "step", "Первый")
    long_term = history.set_memory(
        MemoryType.LONG_TERM, "decision", "database", "SQLite"
    )

    history.clear()

    assert history.task(task.id).status is TaskStatus.CANCELLED
    assert history.list_memory(MemoryType.WORKING, task_id=task.id) == []
    assert history.list_memory(MemoryType.LONG_TERM) == [long_term]
    assert history.list_memory(MemoryType.SHORT_TERM) == []


def test_migrates_owner_id_before_querying_legacy_conversations(tmp_path) -> None:
    path = tmp_path / "history.db"
    timestamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                strategy TEXT NOT NULL,
                active_branch_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO conversations VALUES (1, 'Старый', 'summary', NULL, ?, ?)
            """,
            (timestamp, timestamp),
        )

    history = SQLiteHistory(path)

    assert history.conversation().owner_id == DEFAULT_OWNER_ID
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversations)")
        }
    assert "owner_id" in columns
