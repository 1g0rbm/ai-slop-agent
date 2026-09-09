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
        if limit < -1:
            raise ValueError("лимит истории должен быть не меньше -1")
        if limit == -1:
            return self.all()
        if limit == 0:
            return []

        query = """
            SELECT user_content, assistant_content,
                   user_created_at, assistant_created_at, model,
                   input_tokens, output_tokens, total_tokens
            FROM (
                SELECT id, user_content, assistant_content,
                       user_created_at, assistant_created_at, model,
                       input_tokens, output_tokens, total_tokens
                FROM exchanges
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id
        """
        return self._select(query, (limit,))

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
        """Удалить всю историю диалога."""
        with self._connect() as connection:
            connection.execute("DELETE FROM exchanges")

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
        return [
            Exchange(
                user_content=row[0],
                assistant_content=row[1],
                user_created_at=datetime.fromisoformat(row[2]),
                assistant_created_at=datetime.fromisoformat(row[3]),
                model=row[4],
                input_tokens=row[5],
                output_tokens=row[6],
                total_tokens=row[7],
            )
            for row in rows
        ]

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
