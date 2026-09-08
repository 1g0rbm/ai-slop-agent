"""Интерактивный интерфейс командной строки для агента."""

import os

import typer
from dotenv import load_dotenv

from .agent import Agent
from .history import HISTORY_DB_NAME, SQLiteHistory, history_limit
from .providers import create_provider

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def chat() -> None:
    """Запустить интерактивный сеанс чата."""
    load_dotenv()
    provider_name = os.environ.get("AI_PROVIDER", "deepseek").lower()
    try:
        history = SQLiteHistory(HISTORY_DB_NAME)
        agent = Agent(
            create_provider(),
            history=history,
            history_limit=history_limit(os.environ),
        )
    except ValueError as error:
        typer.echo(f"Ошибка конфигурации: {error}", err=True)
        raise typer.Exit(1) from error

    exchanges = history.all()
    if exchanges:
        typer.echo(f"Сохранённая история: {len(exchanges)} пар.")
        for exchange in exchanges:
            typer.echo(f"Вы: {exchange.user_content}")
            typer.echo(f"Агент: {exchange.assistant_content}")
    else:
        typer.echo("Сохранённой истории нет.")

    typer.echo(
        f"Чат запущен. Провайдер: {provider_name}. "
        "Введите '/clear' для очистки истории, 'exit' или 'quit' для завершения."
    )
    while True:
        try:
            message = typer.prompt("Вы")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        command = message.strip().lower()
        if command in {"exit", "quit"}:
            break
        if command == "/clear":
            if typer.confirm("Удалить всю историю?", default=False):
                history.clear()
                typer.echo("История удалена.")
            else:
                typer.echo("Очистка отменена.")
            continue
        if not message.strip():
            continue

        try:
            typer.echo(f"Агент: {agent.respond(message)}")
        except Exception as error:
            typer.echo(f"Ошибка запроса ({provider_name}): {error}", err=True)


def main() -> None:
    """Запустить приложение Typer."""
    app()
