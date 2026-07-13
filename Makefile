.PHONY: install test lint corpus compose-config

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

corpus:
	.venv/bin/gmoney-corpus build --config corpus/source-registry.json --output data/catalog.json

compose-config:
	docker compose config --quiet

