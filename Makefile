.PHONY: install test lint format demo learn ui pages clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check . --fix

demo:
	python examples/01_quickstart.py

learn:
	python examples/02_learning_across_runs.py

ui:
	python -m cogniweave ui

pages:
	python -m cogniweave ui --export docs/demo

clean:
	rm -rf .cogniweave .pytest_cache .ruff_cache build dist *.egg-info
