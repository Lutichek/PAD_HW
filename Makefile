SHELL := /bin/bash
.DEFAULT_GOAL := help

# uv из PATH, иначе стандартное место установки официального скрипта
UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: help setup check lint format-check typecheck format clean

help: ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-14s %s\n", $$1, $$2}'

setup: ## Поднять окружение с нуля: uv, Python, зависимости, git-хуки
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --frozen
	$(UV) run --frozen pre-commit install --install-hooks

check: lint format-check typecheck ## Все проверки (ничего не исправляет)
	@echo "All checks passed."

lint: ## Линтер Ruff
	$(UV) run --frozen ruff check .

format-check: ## Проверка форматирования Ruff (без изменений файлов)
	$(UV) run --frozen ruff format --check --diff .

typecheck: ## Проверка типов mypy (strict)
	$(UV) run --frozen mypy

format: ## Автоисправление стиля и форматирование (меняет файлы!)
	$(UV) run --frozen ruff check --fix .
	$(UV) run --frozen ruff format .

clean: ## Удалить окружение и кеши
	rm -rf .venv .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
