PYTHON ?= python

.PHONY: install lint format test run clean

install:
	poetry install --with dev

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run ruff check --fix src tests

test:
	poetry run pytest

run:
	poetry run python -m traffic_risk

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build
