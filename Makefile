.PHONY: install test lint corpus release-corpus compose-config

install:
	python3 -m venv .venv
	.venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

corpus:
	.venv/bin/gmoney-corpus build --config corpus/source-registry.json --output data/catalog.json

release-corpus:
	@test -n "$(RELEASE_CORPUS_MANIFEST)" || (echo "RELEASE_CORPUS_MANIFEST is required" >&2; exit 2)
	@test -n "$(RELEASE_CORPUS_DATA_ROOT)" || (echo "RELEASE_CORPUS_DATA_ROOT is required" >&2; exit 2)
	@test -n "$(RELEASE_CORPUS_PRODUCTION14)" || (echo "RELEASE_CORPUS_PRODUCTION14 is required" >&2; exit 2)
	@test -n "$(RELEASE_CORPUS_PASSING36)" || (echo "RELEASE_CORPUS_PASSING36 is required" >&2; exit 2)
	@test -n "$(RELEASE_CORPUS_STAGING159)" || (echo "RELEASE_CORPUS_STAGING159 is required" >&2; exit 2)
	@test -n "$(RELEASE_CORPUS_REPORT)" || (echo "RELEASE_CORPUS_REPORT is required" >&2; exit 2)
	@test -n "$(RELEASE_REVISION)" || (echo "RELEASE_REVISION is required" >&2; exit 2)
	@test -n "$(RELEASE_API_IMAGE_DIGEST)" || (echo "RELEASE_API_IMAGE_DIGEST is required" >&2; exit 2)
	@test -n "$(RELEASE_FRONTEND_IMAGE_DIGEST)" || (echo "RELEASE_FRONTEND_IMAGE_DIGEST is required" >&2; exit 2)
	@test -n "$(RELEASE_WORKER_IMAGE_DIGEST)" || (echo "RELEASE_WORKER_IMAGE_DIGEST is required" >&2; exit 2)
	.venv/bin/gmoney-release-corpus evaluate \
		--manifest "$(RELEASE_CORPUS_MANIFEST)" \
		--source-root "production14=$(RELEASE_CORPUS_PRODUCTION14)" \
		--source-root "passing36=$(RELEASE_CORPUS_PASSING36)" \
		--source-root "staging159=$(RELEASE_CORPUS_STAGING159)" \
		--data-root "$(RELEASE_CORPUS_DATA_ROOT)" \
		--release-revision "$(RELEASE_REVISION)" \
		--api-image-digest "$(RELEASE_API_IMAGE_DIGEST)" \
		--frontend-image-digest "$(RELEASE_FRONTEND_IMAGE_DIGEST)" \
		--worker-image-digest "$(RELEASE_WORKER_IMAGE_DIGEST)" \
		--output "$(RELEASE_CORPUS_REPORT)"

compose-config:
	docker compose config --quiet
	docker compose -f compose.demo.yaml config --quiet
	docker compose -f compose.demo.yaml -f compose.gpu.yaml config --quiet
	docker compose -f compose.demo.yaml --profile admin config --quiet
	docker compose -f compose.demo.yaml -f compose.gpu.yaml --profile admin config --quiet
