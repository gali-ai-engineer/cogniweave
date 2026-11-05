.PHONY: install test lint format demo learn clean

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

clean:
	rm -rf .cogniweave .pytest_cache .ruff_cache build dist *.egg-info
