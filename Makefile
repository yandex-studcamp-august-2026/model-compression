PYTHON ?= .venv/bin/python
UV ?= uv
EXPERIMENT ?=
WEIGHTS ?=
DATASET ?=
BUNDLE ?=
RESULTS ?= results

.PHONY: help sync install install-dev format lint test check discover validate export benchmark-cpu benchmark-gpu clean

help:
	@echo "make install                         Reproduce the locked runtime"
	@echo "make install-dev                     Install development tools"
	@echo "make sync                            Reproduce the locked environment"
	@echo "make check                           Run formatting, lint, and tests"
	@echo "make discover                        List benchmark candidates"
	@echo "make validate EXPERIMENT=...         Validate one experiment contract"
	@echo "make export EXPERIMENT=... DATASET=... [WEIGHTS=...]"
	@echo "make benchmark-cpu BUNDLE=..."
	@echo "make benchmark-gpu BUNDLE=... DATASET=..."

sync:
	$(UV) sync --frozen --extra dev --extra storage --extra export --extra cpu

install:
	$(UV) sync --frozen --extra storage --extra export --extra cpu

install-dev:
	$(UV) sync --frozen --extra dev --extra storage --extra export --extra cpu

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

lint:
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m ruff check .

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check: lint test

discover:
	PYTHONPATH=src $(PYTHON) -m model_bench discover

validate:
	@test -n "$(EXPERIMENT)" || { echo "EXPERIMENT is required" >&2; exit 2; }
	PYTHONPATH=src $(PYTHON) -m model_bench validate "$(EXPERIMENT)"

export:
	@test -n "$(EXPERIMENT)" || { echo "EXPERIMENT is required" >&2; exit 2; }
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	PYTHONPATH=src $(PYTHON) -m model_bench export "$(EXPERIMENT)" \
		--dataset "$(DATASET)" --output bundles \
		$(if $(WEIGHTS),--weights "$(WEIGHTS)",)

benchmark-cpu:
	@test -n "$(BUNDLE)" || { echo "BUNDLE is required" >&2; exit 2; }
	PYTHONPATH=src $(PYTHON) -m model_bench benchmark-cpu \
		--bundle "$(BUNDLE)" --results "$(RESULTS)"

benchmark-gpu:
	@test -n "$(BUNDLE)" || { echo "BUNDLE is required" >&2; exit 2; }
	@test -n "$(DATASET)" || { echo "DATASET is required" >&2; exit 2; }
	PYTHONPATH=src $(PYTHON) -m model_bench benchmark-gpu \
		--bundle "$(BUNDLE)" --dataset "$(DATASET)" --results "$(RESULTS)"

clean:
	rm -rf bundles results
