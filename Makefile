PY ?= .venv/bin/python
UV ?= uv

.PHONY: install test lint format cov check bump clean

install:
	$(UV) pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=mafialab --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

check: test lint

part ?= patch
bump:
	$(UV) version --bump $(part) --frozen
	$(MAKE) install

clean:
	rm -rf .pytest_cache .ruff_cache build dist
