"""Постоянное SQLite-хранилище истории диалога."""

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HISTORY_DB_NAME = "history.db"
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_CONVERSATION_ID = 1
DEFAULT_CONVERSATION_TITLE = "Основной диалог"
DEFAULT_BRANCH_NAME = "main"
VALID_STRATEGIES = frozenset({"sliding", "sticky", "branching", "summary"})
VALID_FACT_CATEGORIES = frozenset(
    {"goal", "constraint", "preference", "decision", "agreement"}
)
VALID_FACT_OPERATIONS = frozenset({"set", "delete"})
VALID_MAINTENANCE_OPERATIONS = frozenset({"summary", "facts"})


@dataclass(frozen=True)
class Exchange:
    """Одна завершённая пара запрос-ответ."""

    user_content: str
    assistant_content: str
    user_created_at: datetime
    assistant_created_at: datetime
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class StoredExchange:
    """Сохранённая пара вместе с идентификатором записи."""

    id: int
    exchange: Exchange


@dataclass(frozen=True)
class ConversationSummary:
    """Накопительная сводка старой части ветки диалога."""

    content: str
    through_exchange_id: int
    summarized_messages: int
    model: str
    updated_at: datetime


@dataclass(frozen=True)
class Conversation:
    """Диалог и его текущая стратегия контекста."""

    id: int
    title: str
    strategy: str
    active_branch_id: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Branch:
    """Ветка диалога с точкой ответвления и текущей вершиной."""

    id: int
    conversation_id: int
    name: str
    parent_branch_id: int | None
    checkpoint_exchange_id: int | None
    head_exchange_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class Checkpoint:
    """Именованная точка на пути ветки."""

    id: int
    conversation_id: int
    branch_id: int
    name: str
    exchange_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class Fact:
    """Актуальный структурированный факт диалога."""

    conversation_id: int
    category: str
    key: str
    value: str
    source_exchange_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class FactOperation:
    """Отложенная операция над фактом, сохраняемая вместе с парой."""

    operation: str
    category: str
    key: str
    value: str | None = None


@dataclass(frozen=True)
class MaintenanceUsage:
    """Необязательная статистика токенов служебной операции."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class OperationTotals:
    """Суммарные токены одного вида служебных операций."""

    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    untracked_operations: int
    operations: int

    @property
    def complete(self) -> bool:
        """Вернуть True, если статистика есть для каждой операции."""
        return self.untracked_operations == 0


@dataclass(frozen=True)
class TokenTotals:
    """Накопленная статистика токенов диалога и его обслуживания."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    untracked_exchanges: int
    maintenance_input_tokens: int = 0
    maintenance_output_tokens: int = 0
    maintenance_total_tokens: int = 0
    untracked_maintenance: int = 0

    @property
    def complete(self) -> bool:
        """Вернуть True, если статистика есть для каждой пары."""
        return self.untracked_exchanges == 0

    @property
    def overall_input_tokens(self) -> int:
        """Вернуть входные токены диалога вместе с обслуживанием."""
        return self.input_tokens + self.maintenance_input_tokens

    @property
    def overall_output_tokens(self) -> int:
        """Вернуть выходные токены диалога вместе с обслуживанием."""
        return self.output_tokens + self.maintenance_output_tokens

    @property
    def overall_total_tokens(self) -> int:
        """Вернуть все токены диалога вместе с обслуживанием."""
        return self.total_tokens + self.maintenance_total_tokens


class SQLiteHistory:
    """SQLite-хранилище диалога с ветками и стратегиями контекста."""

    def __init__(self, path: str | Path = HISTORY_DB_NAME) -> None:
        self._path = Path(path)
        self._initialize()

    def conversation(self) -> Conversation:
        """Вернуть диалог по умолчанию."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, strategy, active_branch_id,
                       created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (DEFAULT_CONVERSATION_ID,),
            ).fetchone()
        if row is None:
            raise RuntimeError("диалог по умолчанию не найден")
        return self._conversation_from_row(row)

    def strategy(self) -> str:
        """Вернуть текущую стратегию диалога."""
        return self.conversation().strategy

    def set_strategy(self, name: str) -> None:
        """Установить одну из поддерживаемых стратегий диалога."""
        self._validate_strategy(name)
        now = self._now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET strategy = ?, updated_at = ?
                WHERE id = ?
                """,
                (name, now, DEFAULT_CONVERSATION_ID),
            )

    def active_branch(self) -> Branch:
        """Вернуть активную ветку диалога."""
        with self._connect() as connection:
            row = self._active_branch_row(connection)
        return self._branch_from_row(row)

    def list_branches(self) -> list[Branch]:
        """Вернуть ветки диалога в порядке создания."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, name, parent_branch_id,
                       checkpoint_exchange_id, head_exchange_id, created_at
                FROM branches
                WHERE conversation_id = ?
                ORDER BY id
                """,
                (DEFAULT_CONVERSATION_ID,),
            ).fetchall()
        return [self._branch_from_row(row) for row in rows]

    def create_checkpoint(self, name: str) -> Checkpoint:
        """Создать именованную точку на вершине активной ветки."""
        self._validate_name(name, "имя checkpoint")
        now = self._now_iso()
        try:
            with self._connect() as connection:
                branch = self._active_branch_row(connection)
                cursor = connection.execute(
                    """
                    INSERT INTO checkpoints (
                        conversation_id, branch_id, name, exchange_id, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        DEFAULT_CONVERSATION_ID,
                        int(branch[0]),
                        name,
                        branch[5],
                        now,
                    ),
                )
                checkpoint_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"checkpoint с именем {name!r} уже существует") from error
        return Checkpoint(
            id=checkpoint_id,
            conversation_id=DEFAULT_CONVERSATION_ID,
            branch_id=int(branch[0]),
            name=name,
            exchange_id=int(branch[5]) if branch[5] is not None else None,
            created_at=datetime.fromisoformat(now),
        )

    def create_branch(self, name: str, checkpoint_name: str | None = None) -> Branch:
        """Создать неактивную ветку от checkpoint или текущей вершины."""
        self._validate_name(name, "имя ветки")
        now = self._now_iso()
        try:
            with self._connect() as connection:
                if checkpoint_name is None:
                    parent = self._active_branch_row(connection)
                    parent_branch_id = int(parent[0])
                    checkpoint_exchange_id = (
                        int(parent[5]) if parent[5] is not None else None
                    )
                else:
                    checkpoint = connection.execute(
                        """
                        SELECT branch_id, exchange_id
                        FROM checkpoints
                        WHERE conversation_id = ? AND name = ?
                        """,
                        (DEFAULT_CONVERSATION_ID, checkpoint_name),
                    ).fetchone()
                    if checkpoint is None:
                        raise ValueError(
                            f"checkpoint с именем {checkpoint_name!r} не найден"
                        )
                    parent_branch_id = int(checkpoint[0])
                    checkpoint_exchange_id = (
                        int(checkpoint[1]) if checkpoint[1] is not None else None
                    )
                cursor = connection.execute(
                    """
                    INSERT INTO branches (
                        conversation_id, name, parent_branch_id,
                        checkpoint_exchange_id, head_exchange_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        DEFAULT_CONVERSATION_ID,
                        name,
                        parent_branch_id,
                        checkpoint_exchange_id,
                        checkpoint_exchange_id,
                        now,
                    ),
                )
                branch_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"ветка с именем {name!r} уже существует") from error
        return Branch(
            id=branch_id,
            conversation_id=DEFAULT_CONVERSATION_ID,
            name=name,
            parent_branch_id=parent_branch_id,
            checkpoint_exchange_id=checkpoint_exchange_id,
            head_exchange_id=checkpoint_exchange_id,
            created_at=datetime.fromisoformat(now),
        )

    def switch_branch(self, name: str) -> Branch:
        """Сделать указанную ветку активной и вернуть её."""
        self._validate_name(name, "имя ветки")
        now = self._now_iso()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, conversation_id, name, parent_branch_id,
                       checkpoint_exchange_id, head_exchange_id, created_at
                FROM branches
                WHERE conversation_id = ? AND name = ?
                """,
                (DEFAULT_CONVERSATION_ID, name),
            ).fetchone()
            if row is None:
                raise ValueError(f"ветка с именем {name!r} не найдена")
            connection.execute(
                """
                UPDATE conversations
                SET active_branch_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (int(row[0]), now, DEFAULT_CONVERSATION_ID),
            )
        return self._branch_from_row(row)

    def all(self) -> list[Exchange]:
        """Вернуть путь активной ветки в хронологическом порядке."""
        return [item.exchange for item in self._active_path()]

    def recent(self, limit: int) -> list[Exchange]:
        """Вернуть последние пары активного пути в хронологическом порядке."""
        return [item.exchange for item in self.recent_after(0, limit)]

    def recent_after(self, exchange_id: int, limit: int) -> list[StoredExchange]:
        """Вернуть последние пары активного пути после указанной пары."""
        if exchange_id < 0:
            raise ValueError("идентификатор checkpoint должен быть неотрицательным")
        if limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if limit == 0:
            return []
        result = [item for item in self._active_path() if item.id > exchange_id]
        if limit != -1:
            result = result[-limit:]
        return result

    def summary_batch(
        self, recent_limit: int, batch_exchanges: int
    ) -> list[StoredExchange]:
        """Вернуть старый блок активной ветки после её checkpoint."""
        if recent_limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if batch_exchanges <= 0:
            raise ValueError("размер блока должен быть больше нуля")
        if recent_limit == -1:
            return []

        summary = self.summary()
        checkpoint = summary.through_exchange_id if summary is not None else 0
        pending = self.recent_after(checkpoint, -1)
        eligible_count = max(0, len(pending) - recent_limit)
        if eligible_count < batch_exchanges:
            return []
        return pending[:batch_exchanges]

    def add(
        self,
        exchange: Exchange,
        request_id: str | None = None,
        fact_operations: Sequence[FactOperation] = (),
    ) -> None:
        """Атомарно сохранить полную пару и отложенные операции фактов."""
        self._validate_exchange(exchange)
        operations = tuple(fact_operations)
        for operation in operations:
            self._validate_fact_operation(operation)
        now = self._now_iso()
        with self._connect() as connection:
            branch = self._active_branch_row(connection)
            branch_id = int(branch[0])
            cursor = connection.execute(
                """
                INSERT INTO exchanges (
                    conversation_id, branch_id, parent_exchange_id, request_id,
                    user_content, assistant_content,
                    user_created_at, assistant_created_at, model,
                    input_tokens, output_tokens, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_CONVERSATION_ID,
                    branch_id,
                    branch[5],
                    request_id,
                    exchange.user_content,
                    exchange.assistant_content,
                    self._datetime_iso(exchange.user_created_at, "время запроса"),
                    self._datetime_iso(exchange.assistant_created_at, "время ответа"),
                    exchange.model,
                    exchange.input_tokens,
                    exchange.output_tokens,
                    exchange.total_tokens,
                ),
            )
            exchange_id = int(cursor.lastrowid)
            for operation in operations:
                self._apply_fact_operation(connection, operation, exchange_id, now)
            connection.execute(
                "UPDATE branches SET head_exchange_id = ? WHERE id = ?",
                (exchange_id, branch_id),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, DEFAULT_CONVERSATION_ID),
            )

    def facts(self) -> list[Fact]:
        """Вернуть актуальные факты диалога в устойчивом порядке."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, category, key, value,
                       source_exchange_id, created_at, updated_at
                FROM facts
                WHERE conversation_id = ?
                ORDER BY category, key
                """,
                (DEFAULT_CONVERSATION_ID,),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def count(self) -> int:
        """Вернуть число пар на пути активной ветки."""
        return len(self._active_path())

    def clear(self) -> None:
        """Полностью сбросить диалог к основной ветке и стратегии summary."""
        now = self._now_iso()
        with self._connect() as connection:
            main_row = connection.execute(
                """
                SELECT id FROM branches
                WHERE conversation_id = ? AND name = ?
                """,
                (DEFAULT_CONVERSATION_ID, DEFAULT_BRANCH_NAME),
            ).fetchone()
            if main_row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO branches (
                        conversation_id, name, parent_branch_id,
                        checkpoint_exchange_id, head_exchange_id, created_at
                    ) VALUES (?, ?, NULL, NULL, NULL, ?)
                    """,
                    (DEFAULT_CONVERSATION_ID, DEFAULT_BRANCH_NAME, now),
                )
                main_id = int(cursor.lastrowid)
            else:
                main_id = int(main_row[0])
            connection.execute("DELETE FROM conversation_summaries")
            connection.execute("DELETE FROM facts")
            connection.execute("DELETE FROM checkpoints")
            connection.execute("DELETE FROM maintenance_usage")
            connection.execute("DELETE FROM exchanges")
            connection.execute(
                "DELETE FROM branches WHERE conversation_id = ? AND id != ?",
                (DEFAULT_CONVERSATION_ID, main_id),
            )
            connection.execute(
                """
                UPDATE branches
                SET parent_branch_id = NULL, checkpoint_exchange_id = NULL,
                    head_exchange_id = NULL
                WHERE id = ?
                """,
                (main_id,),
            )
            connection.execute(
                """
                UPDATE conversations
                SET strategy = 'summary', active_branch_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (main_id, now, DEFAULT_CONVERSATION_ID),
            )
            if self._table_exists(connection, "conversation_summary"):
                connection.execute("DELETE FROM conversation_summary")

    def summary(self) -> ConversationSummary | None:
        """Вернуть сводку активной ветки."""
        with self._connect() as connection:
            branch = self._active_branch_row(connection)
            row = connection.execute(
                """
                SELECT content, through_exchange_id, summarized_messages,
                       model, updated_at
                FROM conversation_summaries
                WHERE branch_id = ?
                """,
                (int(branch[0]),),
            ).fetchone()
        if row is None:
            return None
        return self._summary_from_row(row)

    def save_summary(self, summary: ConversationSummary) -> None:
        """Атомарно сохранить сводку активной ветки и её позицию."""
        if summary.through_exchange_id < 0:
            raise ValueError("идентификатор сводки должен быть неотрицательным")
        updated_at = self._datetime_iso(summary.updated_at, "время сводки")
        with self._connect() as connection:
            branch = self._active_branch_row(connection)
            path_ids = {item.id for item in self._active_path(connection)}
            if summary.through_exchange_id not in path_ids:
                raise ValueError("позиция сводки не принадлежит активной ветке")
            connection.execute(
                """
                INSERT INTO conversation_summaries (
                    conversation_id, branch_id, content, through_exchange_id,
                    summarized_messages, model, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(branch_id) DO UPDATE SET
                    content = excluded.content,
                    through_exchange_id = excluded.through_exchange_id,
                    summarized_messages = excluded.summarized_messages,
                    model = excluded.model,
                    updated_at = excluded.updated_at
                """,
                (
                    DEFAULT_CONVERSATION_ID,
                    int(branch[0]),
                    summary.content,
                    summary.through_exchange_id,
                    summary.summarized_messages,
                    summary.model,
                    updated_at,
                ),
            )

    def record_maintenance(
        self,
        operation: str,
        strategy: str,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        request_id: str | None = None,
        usage: MaintenanceUsage | None = None,
    ) -> None:
        """Записать статистику суммаризации или извлечения фактов."""
        if operation not in VALID_MAINTENANCE_OPERATIONS:
            raise ValueError("служебная операция должна быть summary или facts")
        self._validate_strategy(strategy)
        if not model.strip():
            raise ValueError("модель служебной операции не должна быть пустой")
        if usage is not None:
            explicit_usage = (input_tokens, output_tokens, total_tokens)
            if any(value is not None for value in explicit_usage):
                raise ValueError("статистика токенов передана двумя способами")
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = usage.total_tokens
        self._validate_token_values(input_tokens, output_tokens, total_tokens)
        with self._connect() as connection:
            branch = self._active_branch_row(connection)
            connection.execute(
                """
                INSERT INTO maintenance_usage (
                    conversation_id, branch_id, operation, strategy, model,
                    input_tokens, output_tokens, total_tokens,
                    request_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_CONVERSATION_ID,
                    int(branch[0]),
                    operation,
                    strategy,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    request_id,
                    self._now_iso(),
                ),
            )

    def maintenance_totals(self) -> dict[str, OperationTotals]:
        """Вернуть статистику обслуживания, сгруппированную по операции."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT operation,
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0),
                       SUM(CASE WHEN total_tokens IS NULL THEN 1 ELSE 0 END),
                       COUNT(*)
                FROM maintenance_usage
                WHERE conversation_id = ?
                GROUP BY operation
                ORDER BY operation
                """,
                (DEFAULT_CONVERSATION_ID,),
            ).fetchall()
        return {
            str(row[0]): OperationTotals(
                operation=str(row[0]),
                input_tokens=int(row[1]),
                output_tokens=int(row[2]),
                total_tokens=int(row[3]),
                untracked_operations=int(row[4]),
                operations=int(row[5]),
            )
            for row in rows
        }

    def token_totals(self) -> TokenTotals:
        """Вернуть токены всех веток и обслуживания всего диалога."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(total_tokens), 0),
                       COALESCE(SUM(
                           CASE WHEN total_tokens IS NULL THEN 1 ELSE 0 END
                       ), 0)
                FROM exchanges
                WHERE conversation_id = ?
                """,
                (DEFAULT_CONVERSATION_ID,),
            ).fetchone()
        maintenance = self.maintenance_totals().values()
        return TokenTotals(
            input_tokens=int(row[0]),
            output_tokens=int(row[1]),
            total_tokens=int(row[2]),
            untracked_exchanges=int(row[3]),
            maintenance_input_tokens=sum(item.input_tokens for item in maintenance),
            maintenance_output_tokens=sum(item.output_tokens for item in maintenance),
            maintenance_total_tokens=sum(item.total_tokens for item in maintenance),
            untracked_maintenance=sum(
                item.untracked_operations for item in maintenance
            ),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            self._create_schema(connection)
            self._migrate_exchange_columns(connection)
            self._migrate_fact_columns(connection)
            self._create_indexes(connection)
            now = self._now_iso()
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations (
                    id, title, strategy, active_branch_id, created_at, updated_at
                ) VALUES (?, ?, 'summary', NULL, ?, ?)
                """,
                (
                    DEFAULT_CONVERSATION_ID,
                    DEFAULT_CONVERSATION_TITLE,
                    now,
                    now,
                ),
            )
            self._validate_stored_strategy(connection)
            main_id = self._ensure_main_branch(connection, now)
            self._migrate_exchanges(connection, main_id)
            self._migrate_summary(connection, main_id)
            self._normalize_timestamps(connection)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
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
            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                parent_branch_id INTEGER,
                checkpoint_exchange_id INTEGER,
                head_exchange_id INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE (conversation_id, name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                branch_id INTEGER NOT NULL,
                parent_exchange_id INTEGER,
                request_id TEXT,
                user_content TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                user_created_at TEXT NOT NULL,
                assistant_created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                branch_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                exchange_id INTEGER,
                created_at TEXT NOT NULL,
                UNIQUE (conversation_id, name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                conversation_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_exchange_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (conversation_id, category, key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                branch_id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                through_exchange_id INTEGER NOT NULL,
                summarized_messages INTEGER NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS maintenance_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                branch_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                strategy TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                request_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _create_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS exchanges_parent_idx
            ON exchanges(parent_exchange_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS maintenance_conversation_idx
            ON maintenance_usage(conversation_id, operation)
            """
        )

    @staticmethod
    def _migrate_exchange_columns(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(exchanges)").fetchall()
        }
        definitions = {
            "conversation_id": "INTEGER",
            "branch_id": "INTEGER",
            "parent_exchange_id": "INTEGER",
            "request_id": "TEXT",
            "input_tokens": "INTEGER",
            "output_tokens": "INTEGER",
            "total_tokens": "INTEGER",
        }
        for column, definition in definitions.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE exchanges ADD COLUMN {column} {definition}"
                )

    @staticmethod
    def _migrate_fact_columns(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(facts)").fetchall()
        }
        if "source_exchange_id" not in columns:
            connection.execute(
                "ALTER TABLE facts ADD COLUMN source_exchange_id INTEGER"
            )

    @staticmethod
    def _ensure_main_branch(connection: sqlite3.Connection, now: str) -> int:
        row = connection.execute(
            """
            SELECT id FROM branches
            WHERE conversation_id = ? AND name = ?
            """,
            (DEFAULT_CONVERSATION_ID, DEFAULT_BRANCH_NAME),
        ).fetchone()
        if row is None:
            cursor = connection.execute(
                """
                INSERT INTO branches (
                    conversation_id, name, parent_branch_id,
                    checkpoint_exchange_id, head_exchange_id, created_at
                ) VALUES (?, ?, NULL, NULL, NULL, ?)
                """,
                (DEFAULT_CONVERSATION_ID, DEFAULT_BRANCH_NAME, now),
            )
            main_id = int(cursor.lastrowid)
        else:
            main_id = int(row[0])
        connection.execute(
            """
            UPDATE conversations
            SET active_branch_id = COALESCE(active_branch_id, ?)
            WHERE id = ?
            """,
            (main_id, DEFAULT_CONVERSATION_ID),
        )
        return main_id

    @staticmethod
    def _migrate_exchanges(connection: sqlite3.Connection, main_id: int) -> None:
        rows = connection.execute(
            """
            SELECT id FROM exchanges
            WHERE conversation_id IS NULL OR branch_id IS NULL
            ORDER BY id
            """
        ).fetchall()
        if not rows:
            return
        head_row = connection.execute(
            "SELECT head_exchange_id FROM branches WHERE id = ?", (main_id,)
        ).fetchone()
        parent_id = int(head_row[0]) if head_row[0] is not None else None
        for row in rows:
            exchange_id = int(row[0])
            connection.execute(
                """
                UPDATE exchanges
                SET conversation_id = ?, branch_id = ?, parent_exchange_id = ?
                WHERE id = ?
                """,
                (DEFAULT_CONVERSATION_ID, main_id, parent_id, exchange_id),
            )
            parent_id = exchange_id
        connection.execute(
            "UPDATE branches SET head_exchange_id = ? WHERE id = ?",
            (parent_id, main_id),
        )

    @classmethod
    def _migrate_summary(cls, connection: sqlite3.Connection, main_id: int) -> None:
        if not cls._table_exists(connection, "conversation_summary"):
            return
        row = connection.execute(
            """
            SELECT content, through_exchange_id, summarized_messages,
                   model, updated_at
            FROM conversation_summary
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return
        updated_at = cls._normalized_datetime_text(str(row[4]))
        connection.execute(
            """
            INSERT OR IGNORE INTO conversation_summaries (
                branch_id, conversation_id, content, through_exchange_id,
                summarized_messages, model, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                main_id,
                DEFAULT_CONVERSATION_ID,
                row[0],
                row[1],
                row[2],
                row[3],
                updated_at,
            ),
        )
        connection.execute("DELETE FROM conversation_summary WHERE id = 1")

    @classmethod
    def _normalize_timestamps(cls, connection: sqlite3.Connection) -> None:
        targets = (
            ("conversations", "id", ("created_at", "updated_at")),
            ("branches", "id", ("created_at",)),
            ("exchanges", "id", ("user_created_at", "assistant_created_at")),
            ("checkpoints", "id", ("created_at",)),
            ("facts", "rowid", ("created_at", "updated_at")),
            ("conversation_summaries", "branch_id", ("updated_at",)),
            ("maintenance_usage", "id", ("created_at",)),
        )
        for table, identity, columns in targets:
            selected = ", ".join((identity, *columns))
            rows = connection.execute(f"SELECT {selected} FROM {table}").fetchall()
            for row in rows:
                values = [
                    cls._normalized_datetime_text(str(value)) for value in row[1:]
                ]
                assignments = ", ".join(f"{column} = ?" for column in columns)
                connection.execute(
                    f"UPDATE {table} SET {assignments} WHERE {identity} = ?",
                    (*values, row[0]),
                )

    @staticmethod
    def _validate_stored_strategy(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            "SELECT strategy FROM conversations WHERE id = ?",
            (DEFAULT_CONVERSATION_ID,),
        ).fetchone()
        if row is None or str(row[0]) not in VALID_STRATEGIES:
            raise ValueError("в базе данных сохранена неизвестная стратегия")

    def _active_path(
        self, connection: sqlite3.Connection | None = None
    ) -> list[StoredExchange]:
        if connection is None:
            with self._connect() as opened_connection:
                return self._active_path(opened_connection)
        branch = self._active_branch_row(connection)
        if branch[5] is None:
            return []
        rows = connection.execute(
            """
            WITH RECURSIVE path AS (
                SELECT id, parent_exchange_id, user_content, assistant_content,
                       user_created_at, assistant_created_at, model,
                       input_tokens, output_tokens, total_tokens
                FROM exchanges
                WHERE id = ? AND conversation_id = ?
                UNION ALL
                SELECT e.id, e.parent_exchange_id,
                       e.user_content, e.assistant_content,
                       e.user_created_at, e.assistant_created_at, e.model,
                       e.input_tokens, e.output_tokens, e.total_tokens
                FROM exchanges AS e
                JOIN path AS child ON e.id = child.parent_exchange_id
                WHERE e.conversation_id = ?
            )
            SELECT id, user_content, assistant_content,
                   user_created_at, assistant_created_at, model,
                   input_tokens, output_tokens, total_tokens
            FROM path
            ORDER BY id
            """,
            (branch[5], DEFAULT_CONVERSATION_ID, DEFAULT_CONVERSATION_ID),
        ).fetchall()
        return [
            StoredExchange(id=int(row[0]), exchange=self._exchange_from_row(row[1:]))
            for row in rows
        ]

    @staticmethod
    def _active_branch_row(connection: sqlite3.Connection) -> tuple[object, ...]:
        row = connection.execute(
            """
            SELECT b.id, b.conversation_id, b.name, b.parent_branch_id,
                   b.checkpoint_exchange_id, b.head_exchange_id, b.created_at
            FROM branches AS b
            JOIN conversations AS c ON c.active_branch_id = b.id
            WHERE c.id = ?
            """,
            (DEFAULT_CONVERSATION_ID,),
        ).fetchone()
        if row is None:
            raise RuntimeError("активная ветка диалога не найдена")
        return row

    @staticmethod
    def _apply_fact_operation(
        connection: sqlite3.Connection,
        operation: FactOperation,
        source_exchange_id: int,
        now: str,
    ) -> None:
        if operation.operation == "delete":
            connection.execute(
                """
                DELETE FROM facts
                WHERE conversation_id = ? AND category = ? AND key = ?
                """,
                (DEFAULT_CONVERSATION_ID, operation.category, operation.key),
            )
            return
        connection.execute(
            """
            INSERT INTO facts (
                conversation_id, category, key, value, source_exchange_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, category, key) DO UPDATE SET
                value = excluded.value,
                source_exchange_id = excluded.source_exchange_id,
                updated_at = excluded.updated_at
            """,
            (
                DEFAULT_CONVERSATION_ID,
                operation.category,
                operation.key,
                operation.value,
                source_exchange_id,
                now,
                now,
            ),
        )

    @staticmethod
    def _validate_fact_operation(operation: FactOperation) -> None:
        if not isinstance(operation, FactOperation):
            raise ValueError("операция факта должна иметь тип FactOperation")
        if operation.operation not in VALID_FACT_OPERATIONS:
            raise ValueError("операция факта должна быть set или delete")
        if operation.category not in VALID_FACT_CATEGORIES:
            raise ValueError("неизвестная категория факта")
        if not operation.key.strip():
            raise ValueError("ключ факта не должен быть пустым")
        if operation.operation == "set" and (
            operation.value is None or not operation.value.strip()
        ):
            raise ValueError("значение факта для set не должно быть пустым")

    @classmethod
    def _validate_exchange(cls, exchange: Exchange) -> None:
        if not exchange.user_content.strip() or not exchange.assistant_content.strip():
            raise ValueError("можно сохранить только полную непустую пару")
        if not exchange.model.strip():
            raise ValueError("модель пары не должна быть пустой")
        cls._datetime_iso(exchange.user_created_at, "время запроса")
        cls._datetime_iso(exchange.assistant_created_at, "время ответа")
        cls._validate_token_values(
            exchange.input_tokens, exchange.output_tokens, exchange.total_tokens
        )

    @staticmethod
    def _validate_token_values(*values: int | None) -> None:
        if any(value is not None and value < 0 for value in values):
            raise ValueError("число токенов не должно быть отрицательным")

    @staticmethod
    def _validate_strategy(name: str) -> None:
        if name not in VALID_STRATEGIES:
            allowed = ", ".join(sorted(VALID_STRATEGIES))
            raise ValueError(f"неизвестная стратегия; допустимы: {allowed}")

    @staticmethod
    def _validate_name(name: str, label: str) -> None:
        if not name.strip():
            raise ValueError(f"{label} не должно быть пустым")

    @staticmethod
    def _conversation_from_row(row: tuple[object, ...]) -> Conversation:
        return Conversation(
            id=int(row[0]),
            title=str(row[1]),
            strategy=str(row[2]),
            active_branch_id=int(row[3]),
            created_at=SQLiteHistory._datetime_from_db(row[4]),
            updated_at=SQLiteHistory._datetime_from_db(row[5]),
        )

    @staticmethod
    def _branch_from_row(row: tuple[object, ...]) -> Branch:
        return Branch(
            id=int(row[0]),
            conversation_id=int(row[1]),
            name=str(row[2]),
            parent_branch_id=int(row[3]) if row[3] is not None else None,
            checkpoint_exchange_id=int(row[4]) if row[4] is not None else None,
            head_exchange_id=int(row[5]) if row[5] is not None else None,
            created_at=SQLiteHistory._datetime_from_db(row[6]),
        )

    @staticmethod
    def _fact_from_row(row: tuple[object, ...]) -> Fact:
        return Fact(
            conversation_id=int(row[0]),
            category=str(row[1]),
            key=str(row[2]),
            value=str(row[3]),
            source_exchange_id=int(row[4]) if row[4] is not None else None,
            created_at=SQLiteHistory._datetime_from_db(row[5]),
            updated_at=SQLiteHistory._datetime_from_db(row[6]),
        )

    @staticmethod
    def _summary_from_row(row: tuple[object, ...]) -> ConversationSummary:
        return ConversationSummary(
            content=str(row[0]),
            through_exchange_id=int(row[1]),
            summarized_messages=int(row[2]),
            model=str(row[3]),
            updated_at=SQLiteHistory._datetime_from_db(row[4]),
        )

    @staticmethod
    def _exchange_from_row(row: tuple[object, ...]) -> Exchange:
        return Exchange(
            user_content=str(row[0]),
            assistant_content=str(row[1]),
            user_created_at=SQLiteHistory._datetime_from_db(row[2]),
            assistant_created_at=SQLiteHistory._datetime_from_db(row[3]),
            model=str(row[4]),
            input_tokens=row[5] if isinstance(row[5], int) else None,
            output_tokens=row[6] if isinstance(row[6], int) else None,
            total_tokens=row[7] if isinstance(row[7], int) else None,
        )

    @staticmethod
    def _datetime_from_db(value: object) -> datetime:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=UTC)
        return parsed

    @staticmethod
    def _datetime_iso(value: datetime, label: str) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{label} должно содержать часовой пояс")
        return value.isoformat()

    @staticmethod
    def _normalized_datetime_text(value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.isoformat()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
        row = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (name,),
        ).fetchone()
        return row is not None

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def history_limit(environ: Mapping[str, str]) -> int:
    """Получить число предыдущих пар, передаваемых модели."""
    value = environ.get("HISTORY_LIMIT")
    if value is None:
        return DEFAULT_HISTORY_LIMIT
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError("переменная HISTORY_LIMIT должна быть целым числом") from error
    if result < -1:
        raise ValueError("переменная HISTORY_LIMIT должна быть не меньше -1")
    return result
