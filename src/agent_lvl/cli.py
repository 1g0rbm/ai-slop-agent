"""Интерактивный интерфейс командной строки для агента."""

import os

import typer
from dotenv import load_dotenv

from .agent import Agent, AgentResponse
from .context import summary_batch_messages
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
            summary_batch_messages=summary_batch_messages(os.environ),
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
            result = agent.respond_with_usage(message)
            typer.echo(f"Агент: {result.content}")
            _display_token_usage(result)
        except Exception as error:
            typer.echo(f"Ошибка запроса ({provider_name}): {error}", err=True)


def _display_token_usage(result: AgentResponse) -> None:
    if result.usage is None:
        typer.echo("Токены текущего запроса и ответа: недоступны.")
    else:
        typer.echo(
            "Токены: "
            f"текущий запрос — {_format_tokens(result.usage.input_tokens)}, "
            f"ответ модели — {_format_tokens(result.usage.output_tokens)}."
        )

    totals = result.conversation_totals
    suffix = ""
    if not totals.complete:
        suffix = f" (без данных для {totals.untracked_exchanges} пар)"
    typer.echo(f"Токены всего диалога — {_format_tokens(totals.total_tokens)}{suffix}.")


def _format_tokens(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def main() -> None:
    """Запустить приложение Typer."""
    app()
