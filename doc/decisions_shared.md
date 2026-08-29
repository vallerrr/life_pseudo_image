# Decision log — shared

Append-only project-wide records. Current position lives in [`STATE.md`](STATE.md), never
here. Ids are permanent and never renumbered; a superseded entry's body is never edited,
only its status line. Kind is `data`, `method`, or `bug`.

Entries D01–D05 were written on 2026-07-24 in an earlier, looser format and were
restated in the standard entry shape on 2026-08-29 (see D06). Their dates, content, and
conclusions are unchanged; the *check* and *alternative rejected* fields were filled in
from the session record, and where the record does not support a field it says so.

---

## D01 — The repository root, not `image_converter/`, is the canonical root
**Date:** 2026-07-24 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** The project root, rather than the `image_converter/` subdirectory the
first commit created, is where `data/`, `doc/`, and `src/` belong.

**Check.** Root-level `data/` and `doc/` already existed and held content; three
prototype scripts existed under `image_converter/src/` (`converter.py`,
`hrs_preprocess.py`, `params.py`) and nothing else did. The 4.8 GB HRS pull was already
at `data/hrs/`, so moving the root would have meant moving it.

**Decision.** Adopt the root as canonical.

**Effect.** The three prototype scripts were preserved under `src/legacy/`; no HRS file
was moved or modified. `image_converter/` was deleted.

**Alternative rejected.** Keeping `image_converter/` as the project root — rejected
because it would have required relocating the data pull and because the directory name
described a prototype, not the study.

---

## D02 — Comparable scenario signal strength
**Date:** 2026-07-24 · **Kind:** `method` · **Status:** `active` · **Scope:** `01`

**Assumption.** A single hazard coefficient shared across the six scenarios is
interpretable only if each scenario's history score is first put on a common scale.

**Check.** The raw accumulation (a running mean), mobility (a regression slope), and
chain-of-risk (a product of two latent means) scores have different units and different
variances by construction, so an identical `beta` would encode a different effect size in
each.

**Decision.** Standardise each non-constant history score cross-sectionally within each
bin, then multiply by `log(1.5)`, giving a fixed effect of 1.5× hazard per SD.

**Effect.** Recovery is compared under a common per-SD effect. `null` is exempt: its
score is identically zero and standardising it would divide by zero, so
`_scenario_score` skips any bin whose SD is below `1e-8`.

**Alternative rejected.** Hand-tuning `beta` per scenario to equalise realized incidence
— rejected because it makes the effect size a fitted quantity rather than a
pre-specified one.

**Caveat carried forward.** Per-bin standardisation is a simulation design device, not a
proposed epidemiological estimand.

---

## D03 — Joint competing-risk calibration of the 15% / 20% targets
**Date:** 2026-07-24 · **Kind:** `method` · **Status:** `active` · **Scope:** `01`

**Assumption.** The specification's 15% disease and 20% mortality targets refer to
cumulative incidences in the **joint** competing-risk process, not to two independent
marginal processes.

**Check.** The two cause-specific intercepts are not separable: each enters the shared
event-free survival term, so raising the death intercept lowers the disease CIF and vice
versa. Solving them one at a time does not converge on either target.

**Decision.** Alternate bisection on both intercepts (up to 8 outer rounds, 28 bisection
steps each) until both joint expected CIFs hit target, and report expected and realized
cumulative incidences separately.

**Effect.** Expected CIFs land on 0.150 and 0.200 to within `1e-8` in all six scenarios.
Realized incidences at `N = 10,000` span 0.1427–0.1573 (disease) and 0.1964–0.2082
(death); the gap between expected and realized is Monte Carlo error, not miscalibration,
and reporting both is what makes that visible. Attrition is calibrated separately to 30%
and lands at 0.2969–0.3030.

**Alternative rejected.** Calibrating each cause marginally and accepting the drift —
rejected because the drift is scenario-dependent, so it would confound scenario with
baseline risk.

---

## D04 — Ground truth is population-level structural support
**Date:** 2026-07-24 · **Kind:** `method` · **Status:** `active` · **Scope:** `01`

**Assumption.** Evaluating attribution recovery needs one fixed `C × T` target per
scenario that a model's attribution map can be scored against.

**Check.** Two obstacles rule out a per-individual gradient as the target. The
chain-of-risk hazard is multiplicative (`early_U × late_D`), so its derivative is
individual-specific and can differ in sign between people. And the observed channels are
noisy indicators of three latent streams, so no stream-level truth maps onto a single
channel.

**Decision.** Define `w_true` as the normalised mean absolute analytical gradient at the
stream level, then divide each stream's support across its four channels in proportion to
their sampled loadings.

**Effect.** Every scenario gets one `12 × 20` non-negative map summing to 1, except
`null`, which is exactly zero. `critical_period` support is confined to bins 5–8;
`chain_of_risk` support splits between upstream bins 1–6 and downstream bins 10–14. All
four properties are asserted in
[`tests/test_shared_dgp.py`](../tests/test_shared_dgp.py).

**Alternative rejected.** Individual perturbation-based chain detection — not rejected
outright, but deferred: it is retained as a later endpoint rather than as the primary
target, because it cannot produce the fixed map the primary comparison needs.

---

## D05 — Study module boundaries live beneath `src/`
**Date:** 2026-07-24 · **Kind:** `method` · **Status:** `partly superseded (layout → D06)`
· **Scope:** project

**Assumption.** The study specification's requested `dgp`, `models`, `eval`,
`experiments`, and `figures` modules can sit beneath a canonical `src/` without changing
their responsibilities.

**Check.** After moving `dgp/` and `experiments/` beneath `src/`, the DGP tests returned
identical deterministic outputs and all five continued to pass.

**Decision.** Use the conventions document as the authoritative layout while keeping the
study's module boundaries.

**Effect.** DGP configuration was colocated at `src/dgp/config.yaml`, runtime settings and
style were centralised in `src/params.py`, and dated reports were written to
`outputs/reports/`.

**Superseding note.** The *principle* — conventions are authoritative, module boundaries
are preserved — still stands and is why D06 was possible. The *paths* in this entry are
all obsolete: see D06 for the current ones. Empty `models/`, `eval/`, and `figures/`
placeholders created under this decision were removed by D06.

---

## D06 — Adopt `doc/00_project_conventions.md` as the authoritative layout
**Date:** 2026-08-29 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** The repository can be moved from the older root `CONVENTIONS.md` to the
newer universal standard in `doc/00_project_conventions.md` without changing a single
number the project has already reported — the reorganisation is structural, so any
numeric change would be a bug introduced by the move.

**Check.** A baseline was captured before the move: for all six scenarios at
`N = 800, replicate = 1`, a SHA-256 digest over all thirteen output arrays
(`X, M, Delta, S, event_time, event_type, w_true, latent, random_intercepts, loadings,
dropout_time, disease_probability, death_probability`) plus the full metadata dict. After
the move, **all six digests matched byte for byte and every pre-existing metadata field
was unchanged**; the only difference was one added key, `master_seed`. The 5 unit tests
passed before and after. Re-running the full diagnostics reproduced the 2026-07-24
manifest exactly: identical seeds, identical realized incidence, mortality, attrition,
missing fraction, and all three calibrated intercepts for all six scenarios.

**Decision.** Adopt the universal standard. Record deltas — and only deltas — in
[`doc/conventions.md`](conventions.md).

**Effect.** The shared code layer replaced the flat one:

| before | after |
|---|---|
| `src/params.py` | `src/utils/shared_paths.py` (paths) + `universal_settings.yml` (style) + module constants |
| `src/utils.py` | `src/utils/shared_reproducibility.py` |
| `src/dgp/generator.py` | `src/utils/shared_dgp.py` |
| `src/dgp/config.yaml` | `src/utils/shared_dgp_config.yaml` |
| `src/experiments/generate_diagnostics.py` | `src/data_analysis/01_dgp_diagnostics.py` + `src/utils/data_analysis_01_dgp_diagnostics_report.py` |
| `outputs/reports/*.pdf` | `output/figures/data_analysis/01_dgp_diagnostics/*.pdf` |
| `outputs/reports/*manifest*.json` | `output/tables/*manifest*.json` |
| `doc/methodology.md` | `doc/01_dgp_diagnostics_methodology.md` |
| `doc/decisions.md` | `doc/decisions_shared.md` (this file) |
| `data/README.md` | `doc/data/simulation.md` (per the standard, per-folder READMEs are replaced by `doc/data/<source>.md`) |
| — | `doc/STATE.md`, `doc/conventions.md`, `doc/data/hrs.md`, `doc/01_dgp_diagnostics_report.md` |

Four smaller changes rode along, each fixing something the old layout permitted:

- The `STYLE` dict left the code for `universal_settings.yml`, so a palette change is now
  one edit and cannot drift between modules.
- `MASTER_SEED` became a **named default argument** of `generate_dataset` rather than a
  field read off the config object, per the standard's §12 — a seed reachable only
  through `cfg.master_seed` is one no caller can override without rebuilding the config.
- Plot target lines (`0.15`, `0.20`, `0.30`) and the bin-count `20` were literals in the
  report code; they now come from the loaded config, so a dotted "target" on a figure
  cannot disagree with the target the calibrator solved for.
- `master_seed` was added to run metadata, so a manifest identifies its own seed lineage.

The empty `src/models/`, `src/eval/`, `src/figures/`, `data/raw/`, `data/processed/`, and
`doc/decisions/` placeholders were deleted per the standard's "do not build the hierarchy
on day one"; the root `CONVENTIONS.md` was deleted as superseded rather than left to
disagree with the new standard.

**Alternative rejected.** Keeping the old layout and writing only the new documents —
rejected because two conventions files describing two different trees is precisely the
failure the standard's *supersede explicitly* rule exists to prevent, and because the
divergence would grow with every analysis added.

---

## D07 — HRS is provisionally `blind`
**Date:** 2026-08-29 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** Until the governing clause is read, the HRS pull must be treated at the
strictest tier rather than at a convenient one.

**Check.** No data use agreement, licence file, or acquisition record is present anywhere
in the repository — `data/hrs/` contains only `.dta` files and codebook PDFs, and the
download date is unrecoverable from the filesystem (directory mtimes read 2026-07-24,
the reorganisation date, not the acquisition date). The standard's default therefore
applies rather than a judgement.

**Decision.** Declare `blind` in [`doc/data/hrs.md`](data/hrs.md), and flag it as
provisional rather than presenting a default as a finding.

**Effect.** No HRS row is read into an assistant's context. Note that the tier is
currently **unenforced in one respect**: no synthetic fixture matching the RAND schema
exists, so any code written against this source would be code that has never executed.
Building that fixture is the outstanding cost of the tier, and it is recorded as open
question 1 in [`STATE.md`](STATE.md). Nothing is blocked today because nothing in the
current pipeline reads HRS at all.

**Alternative rejected.** Declaring `schema-only` on the reasoning that HRS public files
are widely used — rejected because "widely used" is not a clause, and because contract
governs before policy does.
