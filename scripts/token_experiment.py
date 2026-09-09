"""Сравнить расход токенов в локальной модели на диалогах разной длины."""

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from agent_lvl.agent import Agent
from agent_lvl.history import SQLiteHistory
from agent_lvl.providers import ChatProvider, create_provider

SECRET_CODE = "SAPPHIRE-7319"


@dataclass(frozen=True)
class Measurement:
    """Результат одного успешного вызова модели."""

    turn: int
    input_tokens: int
    output_tokens: int
    call_tokens: int
    conversation_tokens: int
    answer: str


def local_provider(context_window: int | None = None) -> ChatProvider:
    """Создать локального провайдера с короткими ответами для эксперимента."""
    environment = dict(os.environ)
    environment.update(
        {
            "AI_PROVIDER": "local",
            "LOCAL_MAX_TOKENS": "64",
            "LOCAL_NUM_PREDICT": "12",
            "LOCAL_THINK": "false",
        }
    )
    if context_window is not None:
        environment["LOCAL_NUM_CTX"] = str(context_window)
    else:
        environment.pop("LOCAL_NUM_CTX", None)
    return create_provider(environment)


def measure(
    prompts: list[str], context_window: int | None = None
) -> tuple[list[Measurement], str | None]:
    """Выполнить независимый диалог, не изменяя рабочую базу истории."""
    with tempfile.TemporaryDirectory(prefix="agent-lvl-token-") as directory:
        history = SQLiteHistory(Path(directory) / "history.db")
        agent = Agent(
            local_provider(context_window),
            history=history,
            history_limit=-1,
        )
        measurements: list[Measurement] = []
        for turn, prompt in enumerate(prompts, start=1):
            try:
                result = agent.respond_with_usage(prompt)
            except Exception as error:
                return measurements, f"{type(error).__name__}: {error}"

            if result.usage is None:
                return measurements, "локальный API не вернул статистику токенов"
            measurements.append(
                Measurement(
                    turn=turn,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    call_tokens=result.usage.total_tokens,
                    conversation_tokens=result.conversation_totals.total_tokens,
                    answer=result.content.strip().replace("\n", " "),
                )
            )
        return measurements, None


def print_measurements(name: str, measurements: list[Measurement]) -> None:
    """Вывести измерения сценария как Markdown-таблицу."""
    print(f"\n## {name}\n")
    print("| Ход | Вход | Ответ | Вызов | Накоплено |")
    print("|---:|---:|---:|---:|---:|")
    for item in measurements:
        print(
            f"| {item.turn} | {item.input_tokens} | {item.output_tokens} "
            f"| {item.call_tokens} | {item.conversation_tokens} |"
        )


def short_prompts() -> list[str]:
    return [
        "Ответь одним словом: столица Франции?",
        "Ответь одним словом: а столица Германии?",
    ]


def long_prompts(turns: int) -> list[str]:
    filler = " ".join(f"элемент-{number}" for number in range(40))
    return [
        (
            f"Запомни факт {turn}: код CODE-{turn:03}. {filler}. "
            "Ответь только словом «принято»."
        )
        for turn in range(1, turns + 1)
    ]


def secret_prompt() -> str:
    return (
        f"Секретный код: {SECRET_CODE}. Какой секретный код указан? "
        "Ответь только кодом."
    )


def overflow_prompt(filler_words: int) -> str:
    filler = " ".join("нейтральный-текст" for _ in range(filler_words))
    return (
        f"Секретный код в начале сообщения: {SECRET_CODE}. "
        f"{filler}. Какой секретный код был в начале сообщения? "
        "Ответь только кодом."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-turns", type=int, default=12)
    parser.add_argument("--overflow-context", type=int, default=512)
    parser.add_argument("--overflow-words", type=int, default=1500)
    arguments = parser.parse_args()

    load_dotenv()
    probe = local_provider()
    print(f"# Эксперимент с токенами: {probe.model}")
    print("\nДенежная стоимость API локальной модели: 0.")
    print("Стоимость оборудования и электричества в расчёт не входит.")

    short, short_error = measure(short_prompts())
    print_measurements("Короткий диалог", short)
    if short_error:
        print(f"\nОшибка: `{short_error}`")

    long, long_error = measure(long_prompts(arguments.long_turns))
    print_measurements("Длинный диалог", long)
    if long_error:
        print(f"\nОшибка: `{long_error}`")

    control, control_error = measure(
        [secret_prompt()], context_window=arguments.overflow_context
    )
    print_measurements("Контрольная проверка секрета", control)
    if control_error:
        print(f"\nОшибка: `{control_error}`")
    elif control:
        print(f"\nОтвет модели: `{control[-1].answer}`")

    overflow, overflow_error = measure(
        [overflow_prompt(arguments.overflow_words)],
        context_window=arguments.overflow_context,
    )
    print_measurements(
        f"Переполнение запрошенного окна {arguments.overflow_context} токенов",
        overflow,
    )
    if overflow_error:
        print(f"\nСервер отклонил запрос: `{overflow_error}`")
    elif overflow:
        result = overflow[-1]
        print(f"\nОтвет модели: `{result.answer}`")
        print(
            f"Запрошенное окно: {arguments.overflow_context}; "
            f"фактически учтённый вход: {result.input_tokens}."
        )
        if SECRET_CODE.casefold() in result.answer.casefold():
            print("Результат: секрет из начала запроса сохранился в контексте.")
        else:
            print(
                "Результат: модель потеряла секрет из начала запроса. "
                "Сервер молча обрезал контекст либо модель не смогла его извлечь."
            )
        if result.input_tokens != arguments.overflow_context:
            print(
                "Сервер использует собственный эффективный размер окна, "
                "отличающийся от запрошенного num_ctx."
            )


if __name__ == "__main__":
    main()
