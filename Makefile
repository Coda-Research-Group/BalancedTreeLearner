.PHONY: test lint smoke fmt check install typecheck

# If already in the batl conda env, don't wrap commands in conda run to avoid the EnvironmentLocationNotFound bug.
CONDA_CMD = $(shell if [ "$$CONDA_DEFAULT_ENV" = "batl" ]; then echo ""; else echo "conda run -n batl "; fi)

PYTHON = $(CONDA_CMD)python
PYTEST = $(CONDA_CMD)pytest
RUFF   = $(CONDA_CMD)ruff
PIP    = $(CONDA_CMD)pip
PYRIGHT = $(CONDA_CMD)pyright

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST)

lint:
	$(RUFF) check $(PATHS)

fmt:
	$(RUFF) format $(PATHS)

typecheck:
	$(PYRIGHT) batl/

# Run lint then tests — useful before committing
check: lint typecheck test

smoke:
	$(PYTEST) tests/experiments/scripts/test_build_search_smoke.py
