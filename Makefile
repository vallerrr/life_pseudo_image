.PHONY: all dgp test diagnostics cnn-smoke cnn-probe clean

PYTHON ?= python

all: dgp

## The reviewed first deliverable: DGP unit tests plus the six diagnostic reports.
dgp: test diagnostics

test:
	$(PYTHON) -m pytest

diagnostics:
	$(PYTHON) src/data_analysis/01_dgp_diagnostics.py

## Analysis 02. The smoke profile proves the plumbing; the study profile is not run
## by make -- size it with cnn-probe and run it through the .sh driver.
cnn-smoke:
	PROFILE=smoke PYTHON=$(PYTHON) src/utils/data_analysis_02_pseudo_image_cnn_job.sh local

cnn-probe:
	PROFILE=study PYTHON=$(PYTHON) src/utils/data_analysis_02_pseudo_image_cnn_job.sh probe

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
