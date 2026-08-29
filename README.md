# Life-course pseudo-image simulation

Tests whether hierarchical, mask-aware temporal models can recover known life-course
social-determinant exposure structures — critical periods, chains of risk, mobility
effects — that cumulative-exposure regression cannot represent.

The first deliverable is a vectorized data-generating process and six scenario-level
diagnostic reports. **Model fitting is deliberately deferred until the DGP is reviewed.**

## Setup

Miniforge environment, named `pesudo_image` (misspelling retained from the original
project request):

```bash
mamba env create -f environment.yml
mamba activate pesudo_image
```

## Run

```bash
make dgp     # unit tests, then the six diagnostic reports
make test    # python -m pytest
make all     # currently aliases `dgp`; expands as later stages are approved
```

The diagnostics entry point can also be run directly:

```bash
python src/data_analysis/01_dgp_diagnostics.py
```

It writes six one-page PDFs to `output/figures/data_analysis/01_dgp_diagnostics/` and a
run manifest — git hash, master seed, full config, realized calibration — to
`output/tables/`.

## Where to start reading

New session, or returning after a break: read these three, in this order, and nothing
else up front.

1. [`doc/STATE.md`](doc/STATE.md) — the current position: active decisions, the ledger of
   established figures, open questions. One page, always current.
2. [`doc/conventions.md`](doc/conventions.md) — this project's deltas from the standard.
3. [`doc/01_dgp_diagnostics_report.md`](doc/01_dgp_diagnostics_report.md) — what the DGP
   currently produces and what it does not establish.

Do **not** open the decision log to orient yourself — it is long and append-only, so it
gets skimmed rather than read. `STATE.md` exists precisely so that is unnecessary.

## Documents

| | |
|---|---|
| [`doc/00_project_conventions.md`](doc/00_project_conventions.md) | the universal standard — same for every project, never edited here |
| [`doc/conventions.md`](doc/conventions.md) | this project's deltas from it |
| [`doc/STATE.md`](doc/STATE.md) | current position, ledger, open questions |
| [`doc/simulation_study_specification.md`](doc/simulation_study_specification.md) | the full study specification, all stages |
| [`doc/01_dgp_diagnostics_methodology.md`](doc/01_dgp_diagnostics_methodology.md) | how the DGP runs today |
| [`doc/01_dgp_diagnostics_report.md`](doc/01_dgp_diagnostics_report.md) | what it found |
| [`doc/decisions_shared.md`](doc/decisions_shared.md) | the append-only decision log |
| [`doc/data/simulation.md`](doc/data/simulation.md) | what the DGP emits: field dictionary, shapes, traps — tier `open` |
| [`doc/data/hrs.md`](doc/data/hrs.md) | HRS provenance, dictionary, and sensitivity tier — `blind` |

## Layout

```
universal_settings.yml            plotting style for the whole project
src/
  data_analysis/
    01_dgp_diagnostics.py         analysis entry point (run as a script)
  utils/
    shared_paths.py               the path registry — bootstrap(), ensure_dirs()
    shared_style.py               the style loader — load_style(<stem>)
    shared_dgp.py                 the simulation: scenarios, hazards, mask, ground truth
    shared_dgp_config.yaml        its structural parameters
    shared_reproducibility.py     stable seeds and git provenance
    data_analysis_01_dgp_diagnostics_report.py    the diagnostic page
  legacy/                         retained pre-reorganisation prototype; not in any pipeline
data/
  hrs/                            source pull — immutable, gitignored, tier `blind` (see doc/data/hrs.md)
  analysis/                       derived tables, one directory per analysis
output/figures/ output/tables/    generated artifacts, ISO-dated
logs/                             machine-generated run logs
tests/                            DGP unit tests
```

Paths are never hardcoded — every location comes from `src/utils/shared_paths.py`, and
styling comes from `universal_settings.yml`. See
[`doc/conventions.md`](doc/conventions.md) §3 for the analysis slug convention that keeps
those five locations in step.

## Data sensitivity

`data/hrs/` is declared **`blind`** — provisionally, pending confirmation of the
governing data use agreement clause. No row from it should be read into an assistant's
context. Nothing in the current pipeline reads it. See
[`doc/data/hrs.md`](doc/data/hrs.md) and open question 1 in
[`doc/STATE.md`](doc/STATE.md).
