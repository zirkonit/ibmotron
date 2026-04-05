PYTHON_BOOTSTRAP ?= $(shell command -v python3.13 || command -v python3)
PYTHON ?= .venv/bin/python

.PHONY: fetch-sources build-simh smoke test dev-venv

.venv/bin/python:
	$(PYTHON_BOOTSTRAP) -m venv .venv
	.venv/bin/pip install pytest

fetch-sources:
	$(PYTHON_BOOTSTRAP) scripts/fetch_sources.py

build-simh:
	./scripts/build_simh.sh

smoke:
	./scripts/smoke_examples.sh

dev-venv: .venv/bin/python

test: .venv/bin/python
	$(PYTHON) -m pytest
