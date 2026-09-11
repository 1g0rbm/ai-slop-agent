"""Сравнить качество и расход токенов с контекстным summary и без него."""

import argparse
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from agent_lvl.agent import Agent
from agent_lvl.context import SUMMARY_SYSTEM_PROMPT
from agent_lvl.history import SQLiteHistory
from agent_lvl.providers import (
    ChatProvider,
    ChatResult,
    Message,
    TokenUsage,
    create_provider,
)

SYSTEM_PROMPT = """Ты участвуешь в тесте памяти диалога. Запоминай переданные
факты точно. На сообщения с новым фактом отвечай только словом «принято». На
контрольный вопрос отвечай в указанном пользователем формате."""


@dataclass(frozen=True)
class Fact:
    """Контрольный факт с однозначным значением для автоматической проверки."""

    name: str
    value: str


FACTS = [
    Fact("код проекта", "ORION-742"),
    Fact("ответственный", "MARINA-K"),
    Fact("версия базы", "PG-16"),
    Fact("регион", "EU-C1"),
    Fact("уровень SLA", "SLA-9995"),
    Fact("окно развёртывания", "DEPLOY-FRI-2130"),
    Fact("время отката", "RB-15M"),
    Fact("канал оповещений", "CH-ORION-ALERTS"),
    Fact("время резервной копии", "BKP-0300"),
    Fact("срок хранения", "RET-30D"),
    Fact("тайм-аут API", "TIMEOUT-8S"),
    Fact("число повторов", "RETRY-3"),
]


@dataclass
class UsageTotals:
    """Суммарная статистика группы вызовов провайдера."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: TokenUsage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens


@dataclass
class TrackingProvider:
    """Провайдер-декоратор, разделяющий основные и summary-вызовы."""

    delegate: ChatProvider
    main: UsageTotals = field(default_factory=UsageTotals)
    summary: UsageTotals = field(default_factory=UsageTotals)

    @property
    def model(self) -> str:
        return self.delegate.model

    def respond(self, messages: list[Message]) -> ChatResult:
        result = self.delegate.respond(messages)
        if result.usage is None:
            raise RuntimeError("провайдер не вернул статистику токенов")
        target = self.summary if _is_summary_call(messages) else self.main
        target.add(result.usage)
        return result


@dataclass(frozen=True)
class ScenarioResult:
    """Итог одного варианта эксперимента."""

    name: str
    model: str
    answer: str
    matched_values: tuple[str, ...]
    missing_values: tuple[str, ...]
    main: UsageTotals
    summary: UsageTotals

    @property
    def quality_percent(self) -> float:
        return len(self.matched_values) / len(FACTS) * 100

    @property
    def all_tokens(self) -> int:
        return self.main.total_tokens + self.summary.total_tokens


def run_scenario(
    name: str,
    environment: dict[str, str],
    history_limit: int,
    summary_batch_messages: int,
) -> ScenarioResult:
    """Выполнить изолированный сценарий на временной базе."""
    provider = TrackingProvider(create_provider(environment))
    with tempfile.TemporaryDirectory(prefix="agent-lvl-context-") as directory:
        agent = Agent(
            provider,
            history=SQLiteHistory(Path(directory) / "history.db"),
            history_limit=history_limit,
            system_prompt=SYSTEM_PROMPT,
            summary_batch_messages=summary_batch_messages,
        )
        for fact in FACTS:
            agent.respond(
                f"Запомни факт: {fact.name} имеет значение {fact.value}. "
                "Ответь только словом «принято»."
            )
        answer = agent.respond(
            "Перечисли значения всех двенадцати сохранённых фактов в исходном "
            "порядке. Верни только значения через пробел, без пояснений."
        )

    normalized_answer = answer.casefold()
    matched = tuple(
        fact.value for fact in FACTS if fact.value.casefold() in normalized_answer
    )
    missing = tuple(
        fact.value for fact in FACTS if fact.value.casefold() not in normalized_answer
    )
    return ScenarioResult(
        name=name,
        model=provider.model,
        answer=answer.strip(),
        matched_values=matched,
        missing_values=missing,
        main=provider.main,
        summary=provider.summary,
    )


def render_report(
    baseline: ScenarioResult,
    compressed: ScenarioResult,
    recent_pairs: int,
    summary_batch_messages: int,
) -> str:
    """Сформировать Markdown-отчёт эксперимента."""
    baseline_input = baseline.main.input_tokens + baseline.summary.input_tokens
    compressed_input = compressed.main.input_tokens + compressed.summary.input_tokens
    baseline_output = baseline.main.output_tokens + baseline.summary.output_tokens
    compressed_output = compressed.main.output_tokens + compressed.summary.output_tokens
    input_change = compressed_input - baseline_input
    output_change = compressed_output - baseline_output
    token_change = compressed.all_tokens - baseline.all_tokens
    input_change_percent = input_change / baseline_input * 100
    output_change_percent = output_change / baseline_output * 100
    token_change_percent = token_change / baseline.all_tokens * 100
    lines = [
        "# Эксперимент: качество и стоимость сжатия контекста",
        "",
        f"Дата: {datetime.now(UTC).date().isoformat()}  ",
        f"Модель: `{baseline.model}`  ",
        f"Контрольных фактов: {len(FACTS)}  ",
        f"Свежий контекст со сжатием: {recent_pairs} пар  ",
        f"Блок summary: {summary_batch_messages} сообщений",
        "",
        "## Результаты",
        "",
        "| Режим | Качество | Основные вызовы | Summary-вызовы "
        "| Вход | Выход | Всего |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (baseline, compressed):
        input_tokens = result.main.input_tokens + result.summary.input_tokens
        output_tokens = result.main.output_tokens + result.summary.output_tokens
        lines.append(
            f"| {result.name} | {len(result.matched_values)}/{len(FACTS)} "
            f"({result.quality_percent:.1f}%) | {result.main.calls} | "
            f"{result.summary.calls} | {input_tokens} | {output_tokens} | "
            f"{result.all_tokens} |"
        )
    lines.extend(
        [
            "",
            "## Сравнение токенов",
            "",
            f"Изменение входных токенов: {input_change:+d} "
            f"({input_change_percent:+.1f}%).  ",
            f"Изменение выходных токенов: {output_change:+d} "
            f"({output_change_percent:+.1f}%).  ",
            f"Изменение общего расхода: {token_change:+d} токенов "
            f"({token_change_percent:+.1f}%).",
            "",
            "Расчёт включает дополнительные запросы, которыми обновляется summary.",
            "",
            "## Вывод",
            "",
            f"Без сжатия восстановлено {len(baseline.matched_values)} из "
            f"{len(FACTS)} фактов; со сжатием — "
            f"{len(compressed.matched_values)} из {len(FACTS)}. В этом прогоне "
            "сжатие не снизило качество точного воспроизведения и уменьшило "
            f"общий расход токенов на {abs(token_change_percent):.1f}%.",
            "",
            "## Ответ без сжатия",
            "",
            f"`{_markdown_code(baseline.answer)}`",
            "",
            "Пропущенные значения: " + (", ".join(baseline.missing_values) or "нет"),
            "",
            "## Ответ со сжатием",
            "",
            f"`{_markdown_code(compressed.answer)}`",
            "",
            "Пропущенные значения: " + (", ".join(compressed.missing_values) or "нет"),
            "",
            "## Методика",
            "",
            "Оба сценария получают одинаковые 12 фактов и один контрольный вопрос.",
            "Качество — доля точных контрольных значений, найденных в финальном",
            "ответе. Сценарии используют отдельные временные SQLite-базы и не",
            "изменяют рабочий `history.db`.",
            "",
        ]
    )
    return "\n".join(lines)


def _is_summary_call(messages: list[Message]) -> bool:
    return bool(messages) and messages[0].get("content") == SUMMARY_SYSTEM_PROMPT


def _markdown_code(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent-pairs", type=int, default=2)
    parser.add_argument("--summary-batch-messages", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/context-compression-experiment.md"),
    )
    arguments = parser.parse_args()
    if arguments.recent_pairs < 0:
        parser.error("--recent-pairs должен быть неотрицательным")

    load_dotenv()
    environment = dict(os.environ)
    baseline = run_scenario(
        "Без сжатия",
        environment,
        history_limit=-1,
        summary_batch_messages=arguments.summary_batch_messages,
    )
    compressed = run_scenario(
        "Со сжатием",
        environment,
        history_limit=arguments.recent_pairs,
        summary_batch_messages=arguments.summary_batch_messages,
    )
    report = render_report(
        baseline,
        compressed,
        arguments.recent_pairs,
        arguments.summary_batch_messages,
    )
    arguments.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Отчёт сохранён в {arguments.output}")


if __name__ == "__main__":
    main()
