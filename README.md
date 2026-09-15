# agent-lvl

[![CI](https://github.com/1g0rbm/ai-slop-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/1g0rbm/ai-slop-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/badge/Ruff-checked-blue.svg)](https://docs.astral.sh/ruff/)

Минимальный интерактивный AI-агент на Python с поддержкой DeepSeek, Qwen Cloud и локального API в формате Ollama.

## Требования

Сначала установите [mise](https://mise.jdx.dev/). Конфигурация проекта автоматически установит Python и uv.

## Настройка

```sh
mise install
mise run install
export DEEPSEEK_API_KEY="..."
```

При необходимости выберите модель. По умолчанию используется `deepseek-chat`:

```sh
export DEEPSEEK_MODEL="deepseek-chat"
```

История основного разговора и структурированная память хранятся в `history.db`
в корне проекта. Агент поддерживает четыре стратегии управления контекстом:

- `sliding` — системный prompt, последние N завершённых пар и текущий запрос;
- `sticky` — обновляемые key-value facts, последние N пар и текущий запрос;
- `branching` — последние N пар только по пути активной ветки;
- `summary` — накопительное краткое содержание старой части и последние N пар.

По умолчанию используется `summary`. Стратегия хранится в SQLite и
переключается во время диалога командой `/strategy`, а не переменной окружения:

```text
/strategy
/strategy sliding
/strategy sticky
/strategy branching
/strategy summary
```

Размер свежего окна измеряется полными парами `user`/`assistant`. По умолчанию
используются последние 20 пар; лимит можно изменить переменной:

```sh
export HISTORY_LIMIT=20
```

`HISTORY_LIMIT=0` исключает предыдущие пары из prompt, а `-1` передаёт весь путь
активной ветки. Системный prompt и текущий запрос передаются всегда. Полная
история остаётся в SQLite независимо от выбранной стратегии.

В режиме `summary` старые пары заменяются накопительной сводкой. Она хранится
отдельно для каждой ветки вместе с checkpoint последней обработанной пары.
Summary обновляется отдельным вызовом выбранной модели после того, как за
пределами свежего окна накопятся 10 новых сообщений, то есть 5 полных пар.
Размер блока можно изменить:

```sh
export SUMMARY_BATCH_MESSAGES=10
```

`SUMMARY_BATCH_MESSAGES` должен быть положительным чётным числом. Новая сводка
строится из предыдущей сводки и очередного блока, поэтому вся старая история
повторно не отправляется. При переключении на другую стратегию summary
сохраняется, но не обновляется и не передаётся модели.

В режиме `sticky` перед основным ответом выполняется отдельный запрос к модели.
Он обновляет внутренние facts категорий `goal`, `constraint`, `preference`,
`decision` и `agreement`. Изменения facts сохраняются атомарно с успешным
ответом. При ошибке основного запроса история и facts не изменяются. В других
стратегиях facts остаются в SQLite, но не обновляются и не передаются модели.
Эти facts являются отдельным внутренним состоянием стратегии и не заменяют
структурированную память пользователя.

Служебные токены обновления summary и извлечения facts учитываются отдельно от
основных запросов и показываются после ответа.

## Модель памяти и задачи

Память разделена по сроку жизни и scope:

- `short-term` — завершённые пары активного пути разговора; только чтение;
- `working` — рабочие данные активной задачи: `task`, `context`, `artifact`,
  `progress`;
- `long-term` — устойчивые данные владельца: `profile`, `decision`, `knowledge`.

Для будущего web-интерфейса схема явно связывает long-term с `owner_id`,
разговоры и задачи — с `conversation_id`, а working — с `task_id`. Сейчас CLI
работает только с владельцем и разговором по умолчанию (`id=1`). В одном
разговоре может быть только одна активная задача. Завершённые и отменённые
задачи сохраняются вместе со своей working memory до явной очистки разговора.

Непустые long-term и working блоки добавляются во все основные prompt сразу
после базового system prompt, до summary/facts и короткой истории. Они явно
помечаются как данные, а не инструкции. Ключ ограничен 100, значение — 4000,
название задачи — 200 символами. Время хранится как timezone-aware UTC.

Управление задачами:

```text
/task
/task new <название...>
/task list
/task show <id>
/task complete
/task cancel
```

Управление памятью:

```text
/memory
/memory list short-term|working|long-term
/memory set working|long-term <категория> <ключ> <значение...>
/memory delete working|long-term <категория> <ключ>
/memory clear working|long-term
/memory promote <working-категория> <ключ> <long-категория> [новый-ключ]
```

`promote` атомарно копирует запись в long-term и оставляет working-источник.
Для working-команд нужна активная задача. Очистка long-term требует отдельного
подтверждения.

## Провайдеры

Выбор провайдера выполняется переменной `AI_PROVIDER`. По умолчанию используется `deepseek`.
CLI автоматически загружает файл `.env` из корня проекта. Переменные, заданные в shell, имеют приоритет над значениями из этого файла.

### DeepSeek

```sh
export AI_PROVIDER=deepseek
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL=deepseek-chat
```

### Qwen Cloud

Укажите URL DashScope для региона своего аккаунта.

```sh
export AI_PROVIDER=qwen
export DASHSCOPE_API_KEY="..."
export QWEN_BASE_URL="https://.../compatible-mode/v1"
export QWEN_MODEL=qwen-plus
```

### Локальный API

Профиль `local` не требует ключа. Все параметры можно изменить без правок кода:

```sh
export AI_PROVIDER=local
export LOCAL_API_URL="http://10.36.201.230:11435/api/chat"
export LOCAL_MODEL=gemma3:27b
export LOCAL_TEMPERATURE=0.0
export LOCAL_MAX_TOKENS=2048
export LOCAL_NUM_PREDICT=300
export LOCAL_NUM_CTX=8192
export LOCAL_THINK=false
export LOCAL_TIMEOUT_SECONDS=120
```

`LOCAL_NUM_CTX` необязателен и задаёт размер контекстного окна локальной модели.
Если переменная не указана, используется настройка сервера. При необходимости
добавьте `LOCAL_API_KEY`: он будет отправлен в заголовке `Authorization`.

Для локального API в `.env` должно быть указано как минимум:

```sh
AI_PROVIDER=local
```

## Запуск

```sh
mise run run
```

При запуске CLI выводит путь активной ветки и продолжает диалог. Введите
сообщение в приглашении `Вы:`. После каждого ответа отображаются токены
основного запроса, ответа, служебных операций и накопленный итог разговора. Для
старых записей без usage CLI сообщает число неучтённых операций.

В режиме `branching` доступны команды:

```text
/branch
/branch list
/branch checkpoint <имя>
/branch create <имя> [checkpoint]
/branch switch <имя>
```

Ветку можно создать от текущей вершины или от именованного checkpoint. Каждая
ветка продолжает общий путь независимо, а в prompt попадает только путь
активной ветки. Выбранная ветка и стратегия сохраняются между запусками.

Команда `/clear` после подтверждения удаляет текущий диалог, legacy-состояние
стратегий и working memory, отменяет активную задачу, оставляет пустую ветку
`main` и возвращает стратегию `summary`. Long-term memory владельца сохраняется.
Для завершения сеанса используйте `exit`, `quit`, Ctrl-D или Ctrl-C.

## Эксперимент с токенами

Для воспроизводимого сравнения короткого, длинного и переполненного диалогов на
настроенной локальной модели выполните:

```sh
uv run python scripts/token_experiment.py
```

Скрипт использует отдельные временные базы и не изменяет рабочий `history.db`.
Последние результаты и выводы сохранены в
[`docs/local-token-experiment.md`](docs/local-token-experiment.md).

Для сравнения качества ответов и полного расхода токенов с контекстным summary
и без него выполните:

```sh
uv run python scripts/context_compression_experiment.py
```

Скрипт проверяет воспроизведение 12 контрольных фактов в двух отдельных
диалогах и учитывает токены дополнительных summary-вызовов. Фактические
результаты для локальной модели сохранены в
[`docs/context-compression-experiment.md`](docs/context-compression-experiment.md).

## Разработка

```sh
mise run lint
mise run test
```
