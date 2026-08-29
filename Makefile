.PHONY: all dgp test diagnostics clean

PYTHON ?= python

all: dgp

## The reviewed first deliverable: DGP unit tests plus the six diagnostic reports.
dgp: test diagnostics

test:
	$(PYTHON) -m pytest

diagnostics:
	$(PYTHON) src/data_analysis/01_dgp_diagnostics.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
