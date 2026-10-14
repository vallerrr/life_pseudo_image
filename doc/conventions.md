# Project conventions — deltas

*Last updated: 2026-10-14 (rev. 3).*

This project follows [`doc/00_project_conventions.md`](00_project_conventions.md), the
universal standard. **This file records only where this project differs from it, and the
constants the standard leaves to the project.** Read the standard for the rule; read this
for what this project chose. Reasons live in [`decisions_shared.md`](decisions_shared.md);
the resulting rules live here, in the imperative.

---

## 1. Constants

| Constant | Value | Where |
|---|---|---|
| environment name | `pesudo_image` (misspelling retained from the original project request) | [`environment.yml`](../environment.yml) |
| master seed | `20260724` | `MASTER_SEED` in [`src/utils/shared_dgp.py`](../src/utils/shared_dgp.py) |
| diagnostic sample size | `10_000`, replicate `0` | `DIAGNOSTIC_N` in [`src/data_analysis/01_dgp_diagnostics.py`](../src/data_analysis/01_dgp_diagnostics.py) |
| palette | `#255F85` `#2A9D8F` `#E9C46A` `#E76F51` | `style.base.colors` in [`universal_settings.yml`](../universal_settings.yml) |
| the analyses so far | `00_data_creation`, `01_dgp_diagnostics`, `02_pseudo_image_cnn` | see §3 |
| the life image's clock | ages 50-90 as 20 two-year bins; birth year read as `1940 +/- 5` | `AGE_AT_FIRST_BIN`, `BIRTH_YEAR_MEAN`, `BIRTH_YEAR_SD` in [`src/utils/data_creation_00_data_creation_life_image.py`](../src/utils/data_creation_00_data_creation_life_image.py) |

## 2. Deltas from the standard

**Scripts under `data_analysis/`, a notebook under `data_creation/`.** The
standard's §3 contract is written for `.ipynb`. The `data_analysis` entry points are
`.py` files run directly (`python src/data_analysis/01_dgp_diagnostics.py`) because
they are batch jobs that write reports. `00_data_creation` is a genuine notebook: its
output *is* the rendered figure, and a reader wants it beside the prose. Both obey the
rest of §3 unchanged — the first cell is the bootstrap header plus
`load_style("<stem>")`, the entry point orchestrates and narrates, the heavy lifting
lives in `src/utils/`, and it runs top to bottom in a fresh kernel. Because the stem
starts with a digit neither file is importable as a module; that is deliberate, and it
is why the bootstrap header (not a package import) is the entry mechanism.

**`src/data_creation/` holds presentation of the simulated data, not acquisition.**
Nothing in this project acquires data yet — the DGP *is* the data, and it lives in
`src/utils/shared_dgp.py` as shared domain logic. The directory currently holds one
notebook, which renders what the DGP emits. Real ingestion joins it when the HRS work
starts.

**The life image's clock is presentational and lives outside the DGP.** The simulation
has no calendar time: its birth-cohort covariate is a standardised normal with no year
attached. The age and calendar-year axes therefore come from constants declared in
`data_creation_00_data_creation_life_image.py`. In the figures they only label an axis.
In the per-person PNG export they also set the layout: columns are calendar years on
one span shared by every person (`calendar_span`, the data's min to max), so the
declared clock decides which column a cell lands in. No table or manifest may read a
year from them, and a column-position effect learned from the export is a property of
the clock, not of the DGP.

**No `config/` directory.** The project queries no credentialed service, so there is
nothing to gitignore and no `.ini.example` to commit. Add both the moment a service
appears; the standard's §7 rules apply unchanged when it does.

**Simulation parameters are not in `universal_settings.yml`.** That file owns plotting
style only. The DGP's structural parameters live in
[`src/utils/shared_dgp_config.yaml`](../src/utils/shared_dgp_config.yaml), beside the
module that reads them, and reach code only through `P.DGP_CONFIG`. They are a property
of the simulation, not of the project's presentation, and they are recorded verbatim in
every run manifest.

**`src/legacy/` is retained and is not part of any pipeline.** It holds the
pre-reorganisation HRS converter prototype. It has no place in the standard's tree; it is
kept because it encodes the HRS variable selection and recode maps, which are expensive
to reconstruct. It is not imported by any current module and is not covered by tests.
Delete it once the real ingestion code supersedes it (D01, D06).

**Tests are a top-level `tests/` directory.** The standard does not name one.
[`tests/conftest.py`](../tests/conftest.py) runs the same bootstrap the entry points do,
so a test imports `utils.shared_dgp` by exactly the path a notebook would.

## 3. The five-place slug

Notebook `00_data_creation` occupies, identically:

| | path |
|---|---|
| entry point | [`src/data_creation/00_data_creation.ipynb`](../src/data_creation/00_data_creation.ipynb) |
| its module | [`src/utils/data_creation_00_data_creation_life_image.py`](../src/utils/data_creation_00_data_creation_life_image.py) |
| its data directory | `data/analysis/data_creation/` — the model-input `.npz` files (§5) |
| its figure directory | `output/figures/data_creation/00_data_creation/` |
| its style section | `style.notebooks.00_data_creation` in `universal_settings.yml` |

Analysis `01_dgp_diagnostics` occupies, identically:

| | path |
|---|---|
| entry point | [`src/data_analysis/01_dgp_diagnostics.py`](../src/data_analysis/01_dgp_diagnostics.py) |
| its module | [`src/utils/data_analysis_01_dgp_diagnostics_report.py`](../src/utils/data_analysis_01_dgp_diagnostics_report.py) |
| its data directory | `data/analysis/dgp_diagnostics/` (registered, not yet written to) |
| its figure directory | `output/figures/data_analysis/01_dgp_diagnostics/` |
| its style section | `style.notebooks.01_dgp_diagnostics` in `universal_settings.yml` |

Analysis `02_pseudo_image_cnn` occupies, identically:

| | path |
|---|---|
| entry point | [`src/data_analysis/02_pseudo_image_cnn.ipynb`](../src/data_analysis/02_pseudo_image_cnn.ipynb) — reads results, trains nothing |
| its modules | `src/utils/data_analysis_02_pseudo_image_cnn_{model,job,figures}.py`, the job's `.sh` driver, and its `_config.yaml` |
| its data directory | `data/analysis/pseudo_image_cnn/<profile>/` — shards, then `results_<profile>.parquet` |
| its figure directory | `output/figures/data_analysis/02_pseudo_image_cnn/` |
| its style section | `style.notebooks.02_pseudo_image_cnn` in `universal_settings.yml` |

Model-agnostic code that every model shares lives in two `shared_` modules:
[`shared_model_input.py`](../src/utils/shared_model_input.py) (the input file, the observed
outcome, per-fold standardisation, folds) and
[`shared_evaluation.py`](../src/utils/shared_evaluation.py) (recovery and prediction metrics).
A new model reuses both rather than re-implementing either.

## 4. Storage

`data/` is a plain directory inside the repo, not a symlink into shared storage.
`data/hrs/` is a 4.8 GB local pull and is gitignored wholesale; see
[`doc/data/hrs.md`](data/hrs.md) for its provenance and its sensitivity tier.

## 5. Artifact naming

Reports, manifests and figures carry the standard's ISO date (§4):
`dgp_<scenario>_YYYYMMDD.pdf`, `attribution_<profile>_<scenario>_YYYYMMDD.pdf`.

Pipeline artifacts under `data/analysis/` take stable names and a registry constant:

| artifact | name | constant |
|---|---|---|
| model input, one per dataset | `model_input_<scenario>_n<N>_r<replicate>.npz` | `P.MODEL_INPUT_DIR` |
| CNN shard | `<scenario>_n<N>_r<replicate>_f<fold>.{parquet,npz,json}` | `P.PSEUDO_IMAGE_CNN_DATA_DIR` |
| CNN results | `results_<profile>.parquet` | `P.PSEUDO_IMAGE_CNN_DATA_DIR` |

The model-input file is a cache of a deterministic generation, so it is safe to delete:
the next reader rewrites it, identically, from the seed tuple. Keys prefixed `truth_` are
evaluation-only and are never passed to a model.

**Run profiles.** A model job runs under a named profile (`smoke`, `study`) defined in its
config, and every result row carries the profile. Only `study` rows are results; `smoke`
exists to prove the plumbing and its numbers are never quoted.
