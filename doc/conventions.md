# Project conventions — deltas

*Last updated: 2026-08-29 (rev. 1).*

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
| the one analysis so far | `01_dgp_diagnostics` | see §3 |

## 2. Deltas from the standard

**Scripts, not notebooks.** The standard's §3 notebook contract is written for
`.ipynb`. This project's analysis entry points are `.py` files under
`src/data_analysis/` run directly (`python src/data_analysis/01_dgp_diagnostics.py`).
Everything else in §3 applies unchanged: the first block is the bootstrap header plus
`load_style("<stem>")`, the entry point orchestrates and narrates, and the heavy lifting
lives in `src/utils/`. Because the stem starts with a digit the file is not importable
as a module — that is deliberate, and it is why the bootstrap header (not a package
import) is the entry mechanism.

**Only one pipeline exists.** The standard's `src/data_creation/` is absent: nothing in
this project acquires data yet — the DGP *is* the data, and it lives in
`src/utils/shared_dgp.py` as shared domain logic. Create `src/data_creation/` when the
HRS ingestion work actually starts, not before.

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

Analysis `01_dgp_diagnostics` occupies, identically:

| | path |
|---|---|
| entry point | [`src/data_analysis/01_dgp_diagnostics.py`](../src/data_analysis/01_dgp_diagnostics.py) |
| its module | [`src/utils/data_analysis_01_dgp_diagnostics_report.py`](../src/utils/data_analysis_01_dgp_diagnostics_report.py) |
| its data directory | `data/analysis/dgp_diagnostics/` (registered, not yet written to) |
| its figure directory | `output/figures/data_analysis/01_dgp_diagnostics/` |
| its style section | `style.notebooks.01_dgp_diagnostics` in `universal_settings.yml` |

## 4. Storage

`data/` is a plain directory inside the repo, not a symlink into shared storage.
`data/hrs/` is a 4.8 GB local pull and is gitignored wholesale; see
[`doc/data/hrs.md`](data/hrs.md) for its provenance and its sensitivity tier.

## 5. Artifact naming

The project currently produces no stable-named pipeline artifacts under `data/` — every
generated file is a dated report or manifest, so the standard's ISO-date rule (§4)
applies to all of them: `dgp_<scenario>_YYYYMMDD.pdf`, `dgp_manifest_YYYYMMDD.json`. When
the first cross-analysis table appears under `data/analysis/`, it takes a stable name and
a registry constant instead.
