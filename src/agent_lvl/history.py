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
                   user_created_at, assistant_created_at, model
            FROM (
                SELECT id, user_content, assistant_content,
                       user_created_at, assistant_created_at, model
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
                    user_created_at, assistant_created_at, model
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    exchange.user_content,
                    exchange.assistant_content,
                    exchange.user_created_at.isoformat(),
                    exchange.assistant_created_at.isoformat(),
                    exchange.model,
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
                    model TEXT NOT NULL
                )
                """
            )

    def _select(
        self,
        query: str = """
            SELECT user_content, assistant_content,
                   user_created_at, assistant_created_at, model
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
