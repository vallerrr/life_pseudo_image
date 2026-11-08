.PHONY: all dgp test diagnostics cnn-smoke cnn-probe cnn-ablation cnn-architecture oracle hrs-fixture hrs-schema hrs-counts hrs-run hrs-images hrs-kernels hrs-stability hrs-learning-curve hrs-scaling hrs-subgroups hrs-pathways hrs-pathways-no-bmi hrs-outcomes hrs-outcomes-stability hrs-outcomes-learning-curve hrs-outcomes-scaling hrs-outcomes-subgroups hrs-outcomes-pathways hrs-b hrs-comparisons hrs-ages hrs-bmi-trajectory hrs-ses-order hrs-bmi-changers clean

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

## Development comparison: the CNN with and without the midstream channels.
cnn-ablation:
	PROFILE=midstream_ablation PYTHON=$(PYTHON) src/utils/data_analysis_02_pseudo_image_cnn_job.sh local

## Development comparison: which CNN architecture the study uses (D13). Fold 0 only.
cnn-architecture:
	for i in 0 1 2 3 20 21 22 23; do $(PYTHON) src/utils/data_analysis_02_pseudo_image_cnn_job.py local --profile architecture_dev --shard $$i; done
	$(PYTHON) src/utils/data_analysis_02_pseudo_image_cnn_job.py merge --profile architecture_dev

## Analysis 03: the bounds a correctly specified model sets (prediction and attribution).
oracle:
	$(PYTHON) src/data_analysis/03_oracle_checks.py

## HRS (D07: blind). The fixture is synthetic, written from the codebook; the real
## ingest (shared_hrs_ingest.ingest_rand) is run by a human and has no target.
hrs-fixture:
	$(PYTHON) tests/fixtures/rand_hrs_fixture.py

## HRS is blind (D07, D22): these are for the user to run. Each writes one aggregate
## report under output/reports/blind/; send that file, nothing else.
hrs-schema:
	$(PYTHON) src/data_creation/01_hrs_schema_report.py

hrs-counts:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py counts

hrs-run:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run

## D30: the kernel-size experiment on diabetes (15 CNN arms, ~5 h on the Apple GPU).
hrs-kernels:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --profile kernels

## D31: map stability and faithfulness on diabetes (4 networks + LightGBM SHAP, ~3 h).
hrs-stability:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py stability --profile stability

## D32: learning curve, diabetes and 10-y death (CNN, RNN, 4 benchmarks at 10-100% of training data, ~5 h).
hrs-learning-curve:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py learning-curve --profile learning_curve

## D33: per-age vs per-variable image scaling, CNN and RNN, diabetes and death.
hrs-scaling:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --profile scaling

## D34: who is predictable, and subgroups where the CNN/RNN differ from LightGBM; diabetes (~1 h).
hrs-subgroups:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py subgroups --profile subgroups

## D35: do cases' individual EG maps fall into pathways that recur across seeds and folds? diabetes (~1.5-2 h).
hrs-pathways:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py pathways --profile pathways

## D36: D35 with the BMI row left out, so the smaller rows can separate; diabetes (~40 min).
hrs-pathways-no-bmi:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py pathways --profile pathways_no_bmi

hrs-images:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py images

## D31-D35 extended from diabetes (and death) to the other three outcomes; same profiles, folds
## and seeds, outcomes overridden on the command line. `make -k hrs-outcomes` runs all five and
## keeps going past a failure; each writes its own report. Times scaled from the diabetes runs.
## The stability profile's networks include mix_stacked and k1x5, so it also re-checks D30 per outcome.
hrs-outcomes: hrs-outcomes-stability hrs-outcomes-learning-curve hrs-outcomes-scaling hrs-outcomes-subgroups hrs-outcomes-pathways

hrs-outcomes-stability:  # ~8 h
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py stability --profile stability --conditions hibpe hearte death

hrs-outcomes-learning-curve:  # ~5 h
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py learning-curve --profile learning_curve --conditions hibpe hearte

hrs-outcomes-scaling:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --profile scaling --conditions hibpe hearte

hrs-outcomes-subgroups:  # ~3 h
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py subgroups --profile subgroups --conditions hibpe hearte death

hrs-outcomes-pathways:  # ~5 h
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py pathways --profile pathways --conditions hibpe hearte death

## Step B (STATE question 19). `make -k hrs-b` runs all three; each command writes its own report(s).
hrs-b: hrs-comparisons hrs-ages hrs-bmi-trajectory

## D38-D40: time-order shuffles, image only vs statics only, and stacking, against the original
## fits on the run's folds; all four outcomes (~3 h each, ~12 h in all).
hrs-comparisons:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py comparisons

## D37: the main run (every benchmark, CNN, RNN, maps) at three more ages of the status
## interview: bin 9 is 66-68, 11 is 70-72, 15 is 78-80; one report per age (~3-4 h each).
hrs-ages:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --status-bin 9
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --status-bin 11
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py run --status-bin 15

## D41: the BMI row removed, kept at one wave, flattened to its level, or shuffled; networks and
## LightGBM refitted against the original fits; diabetes and hypertension (~2.5 h each, ~5 h in all).
hrs-bmi-trajectory:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile bmi_trajectory

## D42-D43: socioeconomic rows removed / at the person mean / shuffled together, fitted on people with a
## socioeconomic shock only (wealth or income fall, Medicaid, poverty, home loss); four outcomes (under ~9 h).
hrs-ses-order:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile ses_order

## D44: the D41 BMI arms fitted on people whose BMI rose or fell by 3+ kg/m2 only; diabetes and
## hypertension (under ~5 h).
hrs-bmi-changers:
	$(PYTHON) src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile bmi_changers

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
