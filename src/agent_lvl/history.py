"""Постоянное SQLite-хранилище истории диалога."""

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HISTORY_DB_NAME = "history.db"
DEFAULT_HISTORY_LIMIT = 20


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
    """Накопительная сводка старой части диалога."""

    content: str
    through_exchange_id: int
    summarized_messages: int
    model: str
    updated_at: datetime


@dataclass(frozen=True)
class TokenTotals:
    """Накопленная статистика токенов сохранённого диалога."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    untracked_exchanges: int

    @property
    def complete(self) -> bool:
        """Вернуть True, если статистика есть для каждой пары."""
        return self.untracked_exchanges == 0


class SQLiteHistory:
    """Хранилище единственного диалога в SQLite."""

    def __init__(self, path: str | Path = HISTORY_DB_NAME) -> None:
        self._path = Path(path)
        self._initialize()

    def all(self) -> list[Exchange]:
        """Вернуть все пары в хронологическом порядке."""
        return self._select()

    def recent(self, limit: int) -> list[Exchange]:
        """Вернуть последние пары в хронологическом порядке."""
        return [item.exchange for item in self.recent_after(0, limit)]

    def recent_after(self, exchange_id: int, limit: int) -> list[StoredExchange]:
        """Вернуть последние пары после checkpoint в хронологическом порядке."""
        if exchange_id < 0:
            raise ValueError("идентификатор checkpoint должен быть неотрицательным")
        if limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if limit == 0:
            return []
        if limit == -1:
            query = """
                SELECT id, user_content, assistant_content,
                       user_created_at, assistant_created_at, model,
                       input_tokens, output_tokens, total_tokens
                FROM exchanges
                WHERE id > ?
                ORDER BY id
            """
            return self._select_stored(query, (exchange_id,))

        query = """
            SELECT id, user_content, assistant_content,
                   user_created_at, assistant_created_at, model,
                   input_tokens, output_tokens, total_tokens
            FROM (
                SELECT id, user_content, assistant_content,
                       user_created_at, assistant_created_at, model,
                       input_tokens, output_tokens, total_tokens
                FROM exchanges
                WHERE id > ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id
        """
        return self._select_stored(query, (exchange_id, limit))

    def summary_batch(
        self, recent_limit: int, batch_exchanges: int
    ) -> list[StoredExchange]:
        """Вернуть очередной полный блок пар, вышедших из свежего контекста."""
        if recent_limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if batch_exchanges <= 0:
            raise ValueError("размер блока должен быть больше нуля")
        if recent_limit == -1:
            return []

        summary = self.summary()
        checkpoint = summary.through_exchange_id if summary else 0
        pending = self.recent_after(checkpoint, -1)
        eligible_count = max(0, len(pending) - recent_limit)
        if eligible_count < batch_exchanges:
            return []
        return pending[:batch_exchanges]

    def add(self, exchange: Exchange) -> None:
        """Атомарно сохранить завершённую пару."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO exchanges (
                    user_content, assistant_content,
                    user_created_at, assistant_created_at, model,
                    input_tokens, output_tokens, total_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange.user_content,
                    exchange.assistant_content,
                    exchange.user_created_at.isoformat(),
                    exchange.assistant_created_at.isoformat(),
                    exchange.model,
                    exchange.input_tokens,
                    exchange.output_tokens,
                    exchange.total_tokens,
                ),
            )

    def count(self) -> int:
        """Вернуть число сохранённых пар."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM exchanges").fetchone()
        return int(row[0])

    def clear(self) -> None:
        """Удалить всю историю диалога и её сводку."""
        with self._connect() as connection:
            connection.execute("DELETE FROM exchanges")
            connection.execute("DELETE FROM conversation_summary")

    def summary(self) -> ConversationSummary | None:
        """Вернуть накопительную сводку старой части диалога."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content, through_exchange_id, summarized_messages,
                       model, updated_at
                FROM conversation_summary
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return None
        return ConversationSummary(
            content=row[0],
            through_exchange_id=int(row[1]),
            summarized_messages=int(row[2]),
            model=row[3],
            updated_at=datetime.fromisoformat(row[4]),
        )

    def save_summary(self, summary: ConversationSummary) -> None:
        """Атомарно сохранить сводку и её checkpoint."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_summary (
                    id, content, through_exchange_id, summarized_messages,
                    model, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    through_exchange_id = excluded.through_exchange_id,
                    summarized_messages = excluded.summarized_messages,
                    model = excluded.model,
                    updated_at = excluded.updated_at
                """,
                (
                    summary.content,
                    summary.through_exchange_id,
                    summary.summarized_messages,
                    summary.model,
                    summary.updated_at.isoformat(),
                ),
            )

    def token_totals(self) -> TokenTotals:
        """Вернуть сумму известных токенов всех завершённых запросов."""
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
                """
            ).fetchone()
        return TokenTotals(
            input_tokens=int(row[0]),
            output_tokens=int(row[1]),
            total_tokens=int(row[2]),
            untracked_exchanges=int(row[3]),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                CREATE TABLE IF NOT EXISTS conversation_summary (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    content TEXT NOT NULL,
                    through_exchange_id INTEGER NOT NULL,
                    summarized_messages INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(exchanges)").fetchall()
            }
            for column in ("input_tokens", "output_tokens", "total_tokens"):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE exchanges ADD COLUMN {column} INTEGER"
                    )

    def _select(
        self,
        query: str = """
            SELECT user_content, assistant_content,
                   user_created_at, assistant_created_at, model,
                   input_tokens, output_tokens, total_tokens
            FROM exchanges
            ORDER BY id
        """,
        parameters: tuple[object, ...] = (),
    ) -> list[Exchange]:
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._exchange_from_row(row) for row in rows]

    def _select_stored(
        self, query: str, parameters: tuple[object, ...]
    ) -> list[StoredExchange]:
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            StoredExchange(id=int(row[0]), exchange=self._exchange_from_row(row[1:]))
            for row in rows
        ]

    @staticmethod
    def _exchange_from_row(row: tuple[object, ...]) -> Exchange:
        return Exchange(
            user_content=str(row[0]),
            assistant_content=str(row[1]),
            user_created_at=datetime.fromisoformat(str(row[2])),
            assistant_created_at=datetime.fromisoformat(str(row[3])),
            model=str(row[4]),
            input_tokens=row[5] if isinstance(row[5], int) else None,
            output_tokens=row[6] if isinstance(row[6], int) else None,
            total_tokens=row[7] if isinstance(row[7], int) else None,
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
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
