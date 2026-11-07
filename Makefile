SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv из PATH, иначе стандартное место установки официального скрипта
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
COMPOSE ?= docker compose

# Путь внутри бакета для make s3-ls (например: make s3-ls S3_PATH=raw/)
S3_PATH ?=

# Параметры конвейера данных (переопределяются так: make data VERSION=v2 SEED=7)
VERSION ?= v1
VAL_SIZE ?= 5000
SEED ?= 42
PY = $(UV) run --frozen python -m

.PHONY: help setup check lint format-check typecheck pre-commit format clean \
	s3-up s3-down s3-purge s3-ls ingest data \
	data-check-version data-download data-convert data-split data-stats data-publish

help: ## Показать список команд
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-20s %s\n", $$1, $$2}'

setup: ## Поднять окружение с нуля: uv, Python, зависимости, git-хуки
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --frozen
	$(UV) run --frozen pre-commit install --install-hooks

check: ## Все проверки (ничего не исправляет, при ошибках возвращает ненулевой код)
	@status=0; \
	echo ">>> ruff check";          $(UV) run --frozen ruff check . || status=1; \
	echo ">>> ruff format --check"; $(UV) run --frozen ruff format --check --diff . || status=1; \
	echo ">>> mypy";                $(UV) run --frozen mypy || status=1; \
	if [ $$status -ne 0 ]; then echo "Checks FAILED."; else echo "All checks passed."; fi; \
	exit $$status

lint: ## Линтер Ruff
	$(UV) run --frozen ruff check .

format-check: ## Проверка форматирования Ruff (без изменений файлов)
	$(UV) run --frozen ruff format --check --diff .

typecheck: ## Проверка типов mypy (strict)
	$(UV) run --frozen mypy

pre-commit: ## Прогнать pre-commit хуки по всем файлам
	$(UV) run --frozen pre-commit run --all-files --show-diff-on-failure

format: ## Автоисправление стиля и форматирование (меняет файлы!)
	$(UV) run --frozen ruff check --fix .
	$(UV) run --frozen ruff format .

clean: ## Удалить окружение и кеши
	rm -rf .venv .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------- S3 (RustFS в docker compose) ----------

.env:
	cp .env.example .env
	@echo "Created .env from .env.example"

s3-up: .env ## Поднять S3 (RustFS) и создать бакет
	$(COMPOSE) up -d --wait s3
	$(COMPOSE) run --rm s3-init
	@echo "S3 is up. Web console: http://localhost:9001/rustfs/console/ (port S3_CONSOLE_PORT from .env)"

s3-down: ## Остановить S3 (данные в volume сохраняются)
	$(COMPOSE) down

s3-purge: ## Остановить S3 и удалить volume с данными
	$(COMPOSE) down -v

s3-ls: .env ## Показать содержимое бакета (путь задаётся через S3_PATH=...)
	$(COMPOSE) run --rm --no-deps --entrypoint sh s3-init \
		-c 'aws --endpoint-url http://s3:9000 s3 ls "s3://$$S3_BUCKET/$(S3_PATH)"'

# ---------- Данные ----------

ingest: .env ## Скачать CIFAR-10 с официального сайта и положить в S3 (raw/)
	$(PY) pad_hw.data.ingest

data: .env ## Весь конвейер: S3 -> PNG + JSON -> split -> stats -> S3 (VERSION, VAL_SIZE, SEED)
	$(MAKE) data-check-version data-download data-convert data-split data-stats data-publish

data-check-version: ## Шаг 0: убедиться, что версии VERSION ещё нет в S3
	$(PY) pad_hw.data.publish --version $(VERSION) --check-only

data-download: ## Шаг 1: скачать сырой архив из S3 в data/raw/
	$(PY) pad_hw.data.download

data-convert: ## Шаг 2: pickle -> PNG-картинки и JSON-разметка
	$(PY) pad_hw.data.convert

data-split: ## Шаг 3: стратифицированное разбиение train/val (VAL_SIZE, SEED)
	$(PY) pad_hw.data.split --val-size $(VAL_SIZE) --seed $(SEED)

data-stats: ## Шаг 4: среднее и std по каналам (только train)
	$(PY) pad_hw.data.stats

data-publish: ## Шаг 5: выгрузить датасет в S3 как processed/cifar10/VERSION/
	$(PY) pad_hw.data.publish --version $(VERSION)
