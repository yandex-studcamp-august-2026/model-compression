PYTHON ?= .venv/bin/python
UV ?= uv
EXPERIMENT ?=
WEIGHTS ?=
DATASET ?=
BUNDLE ?=
RESULTS ?= results

.PHONY: help sync install install-dev format lint test check discover validate export benchmark clean

help:
	@echo "make install                         Reproduce the locked runtime"
	@echo "make install-dev                     Install development tools"
	@echo "make sync                            Reproduce the locked environment"
	@echo "make check                           Run formatting, lint, and tests"
	@echo "make discover                        List benchmark candidates"
	@echo "make validate EXPERIMENT=...         Validate one experiment contract"
	@echo "make export EXPERIMENT=... DATASET=... [WEIGHTS=...]"
	@echo "make benchmark EXPERIMENT=... BUNDLE=..."

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

benchmark:
	@test -n "$(EXPERIMENT)" || { echo "EXPERIMENT is required" >&2; exit 2; }
	@test -n "$(BUNDLE)" || { echo "BUNDLE is required" >&2; exit 2; }
	@if test -f "$(EXPERIMENT)/CPU"; then \
		PYTHONPATH=src $(PYTHON) -m model_bench benchmark-cpu \
			--bundle "$(BUNDLE)" --results "$(RESULTS)"; \
	elif test -f "$(EXPERIMENT)/GPU"; then \
		PYTHONPATH=src $(PYTHON) -m model_bench benchmark-gpu \
			--bundle "$(BUNDLE)" --results "$(RESULTS)"; \
	else \
		echo "$(EXPERIMENT) must contain CPU or GPU" >&2; exit 2; \
	fi

clean:
	rm -rf bundles results
