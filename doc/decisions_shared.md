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

---

## D08 — Model input is persisted, raw, and censored at attrition
**Date:** 2026-10-14 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** Every model must see an identical input and an identical outcome, or the
scenario × model table compares definitions rather than models.

**Check.** The DGP samples dropout only strictly before the event and blanks every cell
from `dropout_time` on, but it still records the event the person would have had. Of
10,000 `critical_period` people, 2,969 drop out; their later events are invisible to any
real panel. Peer-standardising `X` over all 10,000 people (as the life images do) would
let a test fold's mean and SD into training.

**Decision.** One `.npz` per `(scenario, N, replicate)` in `data/analysis/data_creation/`,
written by `shared_model_input`: raw `X` with NaN, `M`, `Delta`, `S`; the outcome as
observed (`obs_time`, `obs_type`, censored at `dropout_time − 1`); and evaluation-only
arrays under a `truth_` prefix. Each model standardises on its own training fold.

**Effect.** Settles STATE §3 open question 4: `data/analysis/` is written to. The file is
a cache of a deterministic generation, written on demand by whichever reader needs it.

**Alternative rejected.** Regenerating in memory per model run — rejected because two
models would then each re-derive the observed outcome, which is where they would drift.

---

## D09 — The 2D pseudo-image CNN is causal in time and scored on observed cells
**Date:** 2026-10-14 · **Kind:** `method` · **Status:** `active` · **Scope:** `02`

**Assumption.** The spec's comparator (§2: a 2D CNN over `C x T`, 4×5 kernels) leaves
open how it produces a discrete-time hazard, and how its attribution map is pooled.

**Check.** (i) Post-event cells stay observed in the DGP and the mask stops recording
attrition at the event, so a non-causal CNN can read the outcome from the image; the
test `test_non_causal_cnn_does_see_the_future` demonstrates the path exists. (ii) On one
fold of `critical_period`, N = 10,000, averaging |EG| over *all* explained people gave a
weights-re-drawn similarity of 0.96 even for an untrained model: unobserved cells get no
attribution, so every map carried the missingness pattern. (iii) With the per-bin baseline
initialised at zero (hazard ≈ 1/3), the decile calibration error was 0.090 and early
stopping fired at epoch 4; initialised at the pooled log-odds it was 0.017 at epoch 31.
(iv) Re-drawing that baseline in the parameter-randomisation control put 76% of the
control map in bin 1, an artefact of the hazard scale.

**Decision.** Causal (left-padded) time convolutions, symmetric channel padding; a per-bin
competing-risks head with the spec's multinomial NLL; static covariates `cohort, sex`
only (frailty is unmeasurable in a real cohort); the population map is each cell's mean
|EG| over the people who observed it; the baseline starts at the pooled log-odds; the
Adebayo weight control re-draws every input weight and keeps the baseline.

**Effect.** The CNN and the proposed TCN differ in their encoders only. On the
development fold the CNN reached AUC 0.606 at bin 20 against 0.599 for cohort + sex
alone, and its map resembled both controls (0.90, 0.81) — recorded in the notebook as a
development observation, not a result.

**Alternative rejected.** A whole-history image classifier predicting the outcome by `T`
— rejected because it cannot be causal and is not a hazard model.

---

## D10 — The per-person PNG is laid out on a shared calendar-year grid
**Date:** 2026-10-14 · **Kind:** `method` · **Status:** `active` · **Scope:** `00`

**Assumption.** A life image read by people should put the same year in the same column
for everyone.

**Check.** Birth years read from the cohort covariate span 1922–1958 in the N = 10,000
`critical_period` panel, so ages 50–90 cover calendar years 1972–2047: 76 one-year columns.

**Decision.** The export is pixels only: one column per calendar year over the panel's
min-to-max span, rows = 12 channels, age, then the three static covariates, all on one
SD colour scale, gray where there is no value. Age uses one clock-only mapping so a given
age is the same colour in every file.

**Effect.** The fabricated 1940 ± 5 clock now sets which column a cell lands in. The
model pipeline (D08, D09) does not read these PNGs; it reads the `C x T` array.

**Alternative rejected.** Training on the PNGs — rejected because colour mapping loses
precision, a missing cell has no honest pixel value, and the grid would misalign `w_true`.

---

## D11 — Pre-registered kill criteria
**Date:** 2026-10-14 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** The spec's §8 criteria and its §6 prediction are stated in words
("fails to beat", "structured maps", "requires N > 20000", "turns out to be stable") that
cannot be checked without a number. Fixed after the fact, any of them could be bent to fit.

**Check.** Written before any `study`-profile fit exists: `data/analysis/pseudo_image_cnn/`
held only `smoke/` when this entry was written, and no proposed-model code exists. The
one development fold reported in D09 used a hand-set grid point outside the study design.

**Decision.** The paper reports that the idea does not work if **any** criterion below
fires. Shared rules first:

- *Unit.* Per `(scenario, N, model)`, one value per replicate: the mean over its 5 outer
  folds. Summaries are the median and IQR across replicates (spec §5).
- *Paired comparison.* Model A **beats** model B when the median of the per-replicate
  differences (A − B) is > 0 **and** its 95% percentile bootstrap interval (10,000
  resamples of replicates) excludes 0. Replicates are paired because both models are fit
  to the same `(scenario, N, replicate)` dataset and the same outer folds.
- *Default N.* Where a criterion names no N, it is evaluated at N = 10,000.
- *Fold-0 metrics.* The label-shuffled control is refit on outer fold 0 only (it costs a
  full refit), so for `sanity_label_random_*` the per-replicate value is fold 0's.

**K1 — Flexible temporal modelling is not needed where it should be.** Fires if, in
`chain_of_risk` **or** `mobility`, the proposed model fails to beat **either** XGBoost
**or** penalised Cox on `auc_disease_t20` — or beats them only while being worse
calibrated, i.e. its median `ici_decile` exceeds the comparator's by more than 0.01.
(Four comparisons; any failure fires.) Recovery is not used here because the tree and
Cox models' attributions are not computed by the same method.

**K2 — The attributions fail the sanity checks.** Fires if, for the proposed model, in any
non-null scenario, the median `sanity_param_random_similarity` **or** the median
`sanity_label_random_similarity` is ≥ 0.50. Any *other* model that fails the same test
is disqualified from interpretive claims (spec §3) but does not fire K2.

**K3 — The null scenario produces structured maps.** Under `null`, a map that a model
fitted to signal-free data draws must be no more structured than the map of the same
model trained on shuffled labels. Fires if, for the proposed model at any N, the median
`flatness_cv` of its null map exceeds 1.10 × the median `flatness_cv` of its
label-shuffled control map on the same null datasets.

**K4 — Recovery needs more data than any cohort has.** *Recovery* at a given N means
median `spearman_cells` ≥ 0.50 **and** median `window_auprc` ≥ 2 × the fraction of
`w_true` cells that are non-zero. Fires if, for the proposed model, there is a scenario
among `critical_period`, `sensitive_period`, `chain_of_risk`, `mobility` where recovery is
not reached at any N ≤ 20,000. (`accumulation` is excluded: its truth covers every
upstream cell, so window AUPRC has almost no negatives to rank against.)

**K5 — The 2D CNN is permutation-stable (spec §6).** Over the K = 20 within-stream
permutations of scenarios 2 and 4 at N = 10,000, the 2D CNN counts as stable if **both**
(a) its SD of test `auc_disease_t20` is ≤ 1.5 × the proposed model's, **and** (b) its mean
pairwise Spearman between permutation-aligned maps is ≥ the proposed model's minus 0.05.
Fires if the 2D CNN is stable in **either** scenario. The spec's consequence applies: the
architectural argument is retracted, not reframed.

**Effect.** STATE §3 open question 3 is closed. The thresholds (0.50, 1.10, 2×, 1.5×,
0.05, 0.01) were set by the assistant at the user's request, transcribing the spec, before
any study fit. They may be amended **only before the first `study` result is read**, by a
superseding entry that says what changed and why; after that they are fixed.

**Alternative rejected.** Leaving the criteria in the spec's prose — rejected because
"fails to beat" with no interval and "structured" with no reference map can each be met
or missed by choosing the test afterwards.

---

## D12 — The per-person PNG drops the fixed traits and draws absent years black
**Date:** 2026-10-16 · **Kind:** `method` · **Status:** `active` · **Scope:** `00` · **Amends:** D10

**Assumption.** D10 drew birth cohort, sex and frailty as three constant rows under age.
The user asked for them out of the generated images, and for the years a person is
not in the panel to be told apart from cells that are genuinely missing.

**Check.** They are constant across a life, so they add no pattern to read along the
year axis, and frailty is unmeasurable in a real cohort (D09). No model reads the PNG
(D10), so nothing downstream depends on these rows.

**Decision.** The export's rows are the 12 channels, then age. A year outside the
person's ages 50-90 is black (`absent_color`); gray now means only a cell with no value
inside those ages: not asked, item nonresponse, or after leaving the panel. Every other
part of D10 stands: pixels only, shared calendar-year grid, one SD colour scale.

**Effect.** Each PNG is 13 rows instead of 16, and its black margins show where in
calendar time the person's 40 years sit. `S` is unchanged in the DGP and in the
model-input file, and the CNN still sees `cohort, sex` (D09).

**Alternative rejected.** Keeping the rows but leaving out frailty only — rejected
because the request was to remove the fixed traits, not to filter them.

---

## D13 — The 2D CNN must see the whole history: 4x5 kernels, time dilations 1-2-4
**Date:** 2026-10-25 · **Kind:** `method` · **Status:** `active` · **Scope:** `02` · **Amends:** D09

**Assumption.** A comparator that cannot see the cells a scenario's hazard depends on
is a straw man, whatever it scores: D09's two undilated 5-wide layers read 9 bins back,
so the `critical_period` window (bins 5-8) left the view from bin 17 on, and the
`chain_of_risk` pair (bins 1-6 with 10-14) never fit one view after bin 14.

**Check.** `architecture_dev` profile, fold 0, N = 10,000, replicate 0, with the
sanity controls (dev only: no `study` shard read). AUC at bin 20 / `spearman_cells`:

| arm | receptive field | `critical_period` | `chain_of_risk` | shard time |
|---|---|---|---|---|
| D09, 4x5 x 2 | 9 | 0.619 / 0.351 | 0.610 / 0.409 | 3-4 min |
| 4x5, dilations 1-2-4 | 29 | 0.615 / 0.285 | 0.608 / 0.281 | 6 min |
| time widths 3, 5, 9 in parallel, dilations 1-2 | 25 | 0.619 / 0.343 | 0.619 / 0.335 | 16-21 min |
| channel heights 1, 4, 12 in parallel, dilations 1-2-4 | 29 | 0.616 / 0.332 | 0.606 / 0.323 | 42-47 min |

Every arm predicts at the oracle's bound (D14: 0.621, 0.625 on this fold), and none
recovers better than D09's on this fold: the full view did not help, because the
observed data carries too little signal to locate a window (D14). One fold only.

**Decision.** Chosen on principle, not on the recovery figures, which would tune the
comparator to the paper's endpoint: the network must see the whole panel, and
`build_model` refuses one that cannot (`require_full_history`). Of the two
full-history candidates the user allowed, the plain 4x5 with dilations 1-2-4 is used:
it is the spec's kernel and 3-4x cheaper than the multi-width one, which did no better.
The parallel-kernel code stays for ablations; the D09 network is the `d09_short` arm.

**Effect.** The CNN study cost grows ~1.5x over the D09 probe (ledger). The CNN job's
outer folds are still seeded by model name (`_seed(shard, "outer")`), so a second
model would get different folds and D11's paired comparison would not be paired; the
oracle borrows the CNN's seed for that reason. Making folds model-independent is open.

**Alternative rejected.** Multi-width time kernels — no better on fold 0 at 3-4x the
cost. Mixed channel heights — slowest, and the worst sanity-control similarity.

---

## D14 — Oracle checks: the kill criteria's yardsticks fail a correct model
**Date:** 2026-10-25 · **Kind:** `method` · **Status:** `active (finding; amendments not adopted)` · **Scope:** project

**Assumption.** A recovery or sanity metric is only a test of a model if a model known
to be right passes it.

**Check.** `03_oracle_checks`: the true hazard's form applied to the observed cells,
coefficients fitted, scored by the CNN's own code. Six scenarios x N 2k-20k x 5 folds,
replicate 0. At N = 10,000 (mean of 5 folds): oracle AUC at bin 20 is 0.618-0.649
against 0.591-0.598 for cohort + sex, so observed histories add 0.02-0.05 at most; the
CNN (D09) reached 0.633 and 0.609 on the same folds in `critical_period` and
`chain_of_risk`. The oracle's label-shuffled similarity is 0.96-1.00 in every scenario,
though the shuffled fit holds ~11% of its attribution mass. Its `critical_period`
`spearman_cells` is 0.488 at every N (0.999 with the death term removed): explaining
the disease CIF credits the downstream cells that drive the competing death.

**Decision.** Recorded as a finding. As written, K2 (similarity >= 0.50) fires on a
correct model in every scenario, and K4 (`spearman_cells` >= 0.50) fires on it in
`critical_period`; K1 compares models inside a 0.03-AUC band. The proposed amendments --
attribute the cumulative disease hazard, not the CIF; base K2's label control on the
attribution-mass ratio; keep parameter randomisation diagnostic -- are **not adopted**:
the user moved the work to HRS (2026-10-25), and D11 stays amendable while no `study`
result has been read.

**Effect.** Any attribution claim, simulation or HRS, needs these yardsticks fixed
first. Figures: `data/analysis/oracle_checks/results_03_oracle_checks.parquet`.

**Alternative rejected.** Reading the CNN's sanity failures (D09) as model failures --
the oracle shows the same failures with a model that is right by construction.

---

<!-- D15–D21 were written under `blind` (D07): their checks are against the codebook
and the synthetic fixture, and each names the ingest-report figure to fill in after the
first real run. -->

---

## D15 — HRS channels: twelve RAND variables, Leave-Behind as the midstream
**Date:** 2026-10-23 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** The spec's streams (structural/material, psychosocial/relational,
behavioural/biological; §1.1) can each be filled with four RAND variables that exist in
enough waves, and the midstream should carry the Leave-Behind's half-sample rotation,
since the spec built that rotation into the DGP to mirror HRS (§1.4).

**Check.** Codebook wave listings: upstream `HwITOT`, `HwATOTB`, `HwINPOV`, `RwWORK`
exist in all 16 waves; midstream `RwLBLONELY3`, `RwLBSATWLF` (8–16), `RwLBONCHRSTR`
(8, 10–16), `RwLBNEGAFFECT` (9–16); downstream `RwBMI`, `RwSMOKEN` (1–16), `RwDRINKD`
(3–16), `RwVIGACT`/`RwVGACTX` (1–6 / 7–16). Each household is Leave-Behind-eligible
every other wave from 2006, the other half coded `.A` [2068].

**Decision.** The twelve channels in `doc/data/hrs.md`, in that order.

**Effect.** No midstream data before 2006: people aged 50–89 only before then have an
empty midstream. Fill in from the report: observed fraction per channel.

**Alternative rejected.** CES-D (`RwCESD`, waves 2–16) in the midstream — more data, but
it would not reproduce the rotation the DGP was built to mirror, so simulation and HRS
results would not be comparable on the midstream.

---

## D16 — HRS outcome: diabetes first reported after entry; death only when nothing was missed
**Date:** 2026-10-23 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** The disease is detected at interviews, not timed; death is timed but
says nothing about disease since the last interview.

**Check.** `RwDIABE` is 0/1 for every respondent in all 16 waves, with no item codes
[427]. Disputes rewrite earlier waves and don't-know is coded 0 [438]. `RwDIABE2`
needs a later confirming report [464]. `RwIWSTAT = 5` marks a death since the last
wave [75].

**Decision.** Disease at the bin of the first interview after entry with `RwDIABE = 1`.
Death at the bin of `RADAGE_M` only when the last interview was disease-free and the
next wave records the death, or no next wave exists. Otherwise censored at the last
disease-free interview, capped at bin 20; events at 90+ are censored.

**Effect.** Fill in from the report: outcome counts and `deaths_not_counted`.

**Alternative rejected.** Counting every recorded death — it would turn a person whose
diabetes began in an unobserved gap into a death, biasing the competing risk toward
death exactly where attrition is informative.

---

## D17 — HRS time axis: age at interview end, later value wins within a bin
**Date:** 2026-10-23 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** The life image's bins are ages (spec §1.1), so a wave belongs to the bin
of the respondent's age at that interview, not to a calendar slot.

**Check.** RAND recommends interview-end age (`RwAGEM_E`) [140]. Waves are about two
years apart, so a bin can hold two interviews or none.

**Decision.** Bin `⌊(RwAGEM_E − 600)/24⌋ + 1`. When two waves share a bin the later
non-missing value wins. Nothing after `obs_time` is written into `X`.

**Effect.** Fill in from the report: cells overwritten (not yet counted — add if wanted).

**Alternative rejected.** Averaging two waves in a bin — it would blend values measured
up to two years apart and hide the later one from a causal model.

---

## D18 — HRS left truncation: entry at the first interview aged 50–89
**Date:** 2026-10-23 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** A person first interviewed at 75 was never at risk in this study before
75; treating bins 1–12 as event-free would add immortal time.

**Check.** Entry cohorts join in 1992–2022 [13]; on the fixture `entry_time` ranges over
bins 1–20 and more than 20% enter after bin 1 (`tests/test_hrs_ingest.py`).

**Decision.** `entry_time` = bin of the first interview at ages 50–89. Bins before it
are unobserved (`M = 0`). People with diabetes at or before entry, including interviews
before 50, are excluded.

**Effect.** `ModelInput` gained `entry_time` (all 1 for the simulation, so no
simulation figure moves) and `person_id`; `shared_evaluation` and the CNN and oracle
losses count a person at risk only in bins `entry_time..obs_time`, and horizon metrics
treat someone entering after the horizon as unknown at it. On real data: fill in
`entry_bin_counts` and `excluded` from the report after the first run.

**Alternative rejected.** Restricting to people who enter by bin 1 — it would discard
every cohort sampled after age 52 and most of the old-age bins.

---

## D19 — HRS money: CPI-U to 2022 dollars, asinh of thousands
**Date:** 2026-10-23 · **Kind:** `data` · **Status:** `active (CPI table verified against the BLS API 2026-10-25: 1991–2023 exact, 2024 added)` · **Scope:** project

**Assumption.** Nominal dollars at the same age differ by cohort through inflation alone,
which a model would read as a cohort effect.

**Check.** Income and wealth are nominal; income is last calendar year's [1241, 1353].
Wealth can be negative (minimum −720,000 in wave 1) [1240]. RAND's CPI is a SAS format only [50].

**Decision.** Deflate `HwITOT` by CPI-U of `RwIWENDY − 1`, `HwATOTB` by CPI-U of
`RwIWENDY`, to 2022 dollars; take asinh of thousands. The CPI table in
`shared_hrs_ingest.CPI_U` is hand-transcribed and must be checked against BLS
(CUUR0000SA0) before the first real run.

**Effect.** Fill in from the report: income/wealth cells lost to a missing interview year.

**Alternative rejected.** Within-wave percentile ranks — no CPI needed, but ranks are
computed over every person, so the test fold would shape the training values (D08).

---

## D20 — HRS harmonisation seams
**Date:** 2026-10-23 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** A change of instrument should not appear in `X` as a change in people.

**Check.** `R8LBSATWLF` is 1–6, later 1–7 [2112]; `R8LBNEGAFFECT6` is a different
6-item scale [2097]; `RwVIGACT` (3+/week, yes/no, waves 1–6) became `RwVGACTX` (five
frequencies, waves 7–16) [596–598]; `RwDRINKD` starts in wave 3 [599].

**Decision.** Map wave-8 life satisfaction linearly onto 1–7; leave wave-8 negative
affect missing; vigorous activity = 1 if `RwVIGACT = 1` or `RwVGACTX ≤ 2` ("more than
once a week", the nearest cut to "3+"); no drinking channel before wave 3.

**Effect.** The activity seam remains a known level shift at wave 7. Fill in from the
report: observed fraction of `vigorous_activity` either side of 2004.

**Alternative rejected.** Dropping vigorous activity before wave 7 — loses a channel for
every cohort aged 50–60 before 2004.

---

## D21 — HRS static covariates: birth year and sex; frailty unmeasured
**Date:** 2026-10-23 · **Kind:** `method` · **Status:** `active` · **Scope:** project

**Assumption.** `S` keeps the DGP's three named columns so `ModelInput.static()` works
unchanged.

**Check.** No RAND variable measures the DGP's frailty; D09 already keeps it out of model
inputs.

**Decision.** `cohort` = z-scored `RABYEAR`; `sex` = z-scored indicator of
`RAGENDER = 2`; `frailty` = NaN.

**Effect.** A run asking for `frailty` on HRS produces NaN inputs and fails visibly.

**Alternative rejected.** Entry cohort (`HACOHORT`) as `cohort` — it records when a
household was sampled, not when its members were born [82].

---

## D22 — Blind sources come back as aggregate reports, run by the user
**Date:** 2026-10-25 · **Kind:** `data` · **Status:** `active` · **Scope:** project

**Assumption.** HRS stays `blind` (D07): the user will not give an assistant access to
its rows. The user will run code and send aggregate results back.

**Check.** The user's instruction, 2026-10-25. The standard (§7) leaves one point open:
how a `blind` run's result comes back, and when an aggregate may relax the tier. It
asks for that to be a logged decision. Testing the new report against the fixture
caught one leak before any real run: percentiles of the person id `HHIDPN` are
person ids.

**Decision.** Run and report, as written in [`conventions.md`](conventions.md) ("Blind
sources: run and report"). Every run on a blind source is one entry-point script. It
writes one JSON report through `shared_secure_report`: aggregates only, counts 1–10
written as `"<11"`, cells over fewer than 11 people null, no min or max, identifiers
counted but never summarised, and errors recorded by type and line only. That report is
the only thing the assistant reads. When the source's structure is needed, a report
script answers it (`01_hrs_schema_report`).

**Effect.** These aggregates are below the tier and may be read. Nothing per-person is
written to disk by analysis 04. `output/reports/` is gitignored. Suppression is primary
only, which is enough for an assistant to read but not for publication.

**Alternative rejected.** Asking the user to paste values or terminal output: the
terminal carries tracebacks, and pasting is where a row slips through.

---

## D23 — HRS binary task: chronic condition by age A, from the life image at 50 to A
**Date:** 2026-10-25 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user asked for a binary classification of a chronic disease (any
for now) and an activation-map check of where in the image the disease is read. The
user does not want reverse causation handled, so values recorded after a diagnosis may
stay in the image.

**Check.** The codebook's eight "ever had" conditions, `RwHIBPE` `RwDIABE` `RwCANCRE`
`RwLUNGE` `RwHEARTE` `RwSTROKE` `RwPSYCHE` `RwARTHRE`, are 0/1 in waves 1–16 with no
item missing codes [425–436]. Labelling "ever reported during follow-up" was rejected:
in HRS, entry age and follow-up length vary by cohort, so the people seen longest have
the most interviews in which to report. The classifier, and its map, would learn the
observation window. The same concern drove the sanity controls in D14.

**Decision.** Everyone kept has a responding interview in one status bin, the two-year
bin holding age A (default 13, ages 74–76; chosen from the `counts` report). The label
is the condition flag at that interview. The image is bins 1 to A−1, so every image has
the same width, and the status interview's own answers are excluded. Prevalent cases
stay in (the user's call). One CNN per condition: the analysis 02 encoder (4×5, time
dilations 1-2-4, D13), uncausal, because every cell precedes the status interview.
Three baselines: cohort + sex; logistic on per-channel summaries; gradient boosting on
every cell. Folds are grouped by household, and seeds depend on the dataset, not the
model. Maps: Grad-CAM on the last conv layer and Expected Gradients on P(case), each
averaged per cell over the test people who observed it, for cases and controls. Controls:
re-drawn weights on every fold and shuffled labels on fold 0, each scored by Spearman
similarity and attribution-mass ratio (D14).

**Effect.** Survivorship is built in: the dead never reach age A. Raw Grad-CAM also
tracks feature magnitude, so the case-minus-control panel is the one to read. Fixture
smoke run only; the real numbers come from the user's `run` report.

**Alternative rejected.** A landmark design (history to A, onset after A): cleaner
causally, but the user chose prevalence, and it would cut the image off at A.

---

## D24 — An RNN joins the HRS model family, on the same image, folds and maps
**Date:** 2026-10-25 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user asked for an RNN. Comparing it with the CNN means
something only if both read the same input, on the same folds, and are explained by the
same method.

**Check.** The spec's GRU-D baseline (§2) differs from a plain GRU only in a learnt
decay toward the mean over gaps. Mask and Delta already enter as inputs here, so the
plain GRU sees the same missingness information.

**Decision.** A GRU (`RecurrentClassifier`) over bins. Each step's input is every
channel's standardised value, mask and Delta (3 × 12 = 36 numbers). It is bidirectional,
because every bin precedes the status interview, and it uses the CNN's logistic head and
prevalence start. It has its own inner grid (hidden 32/64, lr 3e-4/1e-3) and shares the
CNN's folds, explained people, Expected Gradients baselines and shuffled labels. Maps:
Expected Gradients only, since Grad-CAM needs convolution layers. Sanity controls as for
the CNN.

**Effect.** Every `run` report carries `cnn` and `rnn` metrics and maps side by side.
The figure's bottom row compares their Expected Gradients maps. The RNN treats the 12
channels as an unordered vector at each step, while the CNN's 4×5 kernel treats channel
order as spatial, which is the property the spec's permutation test (§6) targets.

**Alternative rejected.** Full GRU-D with learnt decay: a further module to verify, for a
difference the mask and Delta inputs already cover.

---

## D25 — Death within a horizon after the age-A interview as a binary outcome
**Date:** 2026-10-25 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user proposed death age as a prediction task. Age at death itself is
right-censored for most of the sample, and the early deaths that are observed come
mostly from the older cohorts. A regression on observed death ages would learn birth
cohort.

**Check.** RAND gives `RADAGE_M` (age at death, months; `.X` when no death date) [140]
and `RwIWSTAT` (4 alive but not interviewed, 5/6 died, 7 dropped, 9 vital status unknown)
[75]. A non-respondent wave has no interview age.

**Decision.** Same status interview and image as D23. The label is 1 if `RADAGE_M` falls
within `death_horizon_years` (default 10) of the status interview's age. The label is 0
only if the person is *known* alive at the horizon: a death date after it, a later
interview at or past it, or a later `RwIWSTAT = 4` wave at least a year past it by the
wave's nominal year. Everyone else is left out as `vital_status_unknown_at_horizon`.

**Effect.** It is a cleaner label than the self-reported conditions. Its exclusions
depend on the end of follow-up: a status interview after about 2012 cannot reach a
10-year horizon by 2022, so later cohorts drop out of it. The `counts` report shows this
per age. A time-to-death model would use the hazard pipeline (analysis 02), which already
has death as a competing risk.

**Alternative rejected.** Treating everyone without a death date as alive at the
horizon: attrition would turn unknown deaths into controls.

---

## D26 — Extended HRS dictionary: every Yan (2026) risk factor RAND carries
**Date:** 2026-10-25 · **Kind:** `data` · **Status:** `active (rows amended by D27)` · **Scope:** `04`

**Assumption.** The user wants the image to carry the risk factors of Yan (2026, AJE
195(3), Table S2): 61 HRS factors in seven domains, from Puterman et al.'s
one-row-per-person file. The user asked to use RAND only and to keep every factor
RAND carries.

**Check.** Each factor was looked up in the RAND 2022 v1 codebook. About 11 are fixed for
life. About 18 have a repeated RAND measure, some only through a proxy: per-wave
Medicaid for "history of Medicaid", not owning a home for "history of renting", drinking
frequency for the CAGE abuse screen, chronic stress for adult adversity. The rest
(purpose, mastery, optimism, pessimism, hopelessness, constraints, anxiety, anger,
hostility, religiosity, social support and strain, discrimination, neighbourhood, food
insecurity and stamps, four childhood items) are not in RAND. RAND's Leave-Behind
section holds only the Big Five, loneliness, chronic stress, affect and life
satisfaction [2063–2114]. Codings came from the codebook; the fixture reproduces them.

**Decision.** `shared_hrs_variables` holds a recode table with the user's SHARE/ELSA
record fields, plus `paper_factor` and whether the paper had that factor in SHARE and
ELSA. The `extended` channel set has 29 rows in five domains (socioeconomic 8, adversity
1, connections 2, psychological 10, behaviours 8). The 12 core channels are among them,
and CES-D is added as the only every-wave psychological measure. The `extended` static
set has 11 covariates. Recodes that change meaning:
- unemployment `.X`/`.A` (not in the labour force) count as 0;
- sleep problems = mean of the 4 sleep items scored 0–1, at least 2 answered;
- moderate activity = more than once a week, as vigorous activity;
- separated/divorced from `RwMSTAT` 4–6;
- never married read at the first image interview, with `.N` counted as 1;
- lower occupation = 1980 Census longest job 5–16 versus 1–4; later cohorts' other
  coding (`.B`) is missing.

A missing static covariate gets the training-fold mean plus a missing indicator. The
core set and the hazard pipeline are unchanged.

**Effect.** Analysis 04 defaults to `extended`. The Leave-Behind rows (13 of 29) exist
only from 2006, for each person every other wave, so for older cohorts they fill only the
right edge of the image. The static logistic baseline now includes education, race,
nativity and parents' education, so "the image helps" is measured against more.

**Alternative rejected.** Scoring the missing scales from the raw core and Leave-Behind
files: the user asked for RAND only (2026-10-25).

---

## D27 — A married-or-partnered row joins the extended set; financial strain skips wave 9
**Date:** 2026-10-25 · **Kind:** `data` · **Status:** `active` · **Scope:** `04` · **Amends:** D26

**Assumption.** D26's marital rows record only marriages ending (separated/divorced,
number of divorces); the user asked for current marriage or partnership as a row too.

**Check.** `RwMSTAT` [177] is in all 16 waves of the real file (schema report
2026-10-25 22:26). The first extended run failed on `r9lbfinprb`: that report showed
`RwLBFINPRB` is in waves 8 and 10–16, not 9, as with chronic stress. Every other
extended column's waves were then checked against the same report: none missing.

**Decision.** `married_partnered` = `RwMSTAT` 1–3, connections domain, before
`separated_divorced`. Financial strain is read from waves 8 and 10–16. The fixture
carries both, and the test now requires the code's waves to equal the fixture's.

**Effect.** 30 image rows (connections 3). Widowhood remains only as a 0 on both marital
rows.

**Alternative rejected.** A single categorical marital row: a 1–8 code in one cell would
be read as a scale by the convolution.

---

## D28 — The average life image: peer-standardised group means, thin cells null
**Date:** 2026-10-26 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user wants a sample life image from HRS for the paper. One
respondent's image is row-level data: it cannot come back under D22 and cannot be
published under the HRS conditions of use. An average over many people can.

**Check.** A cell mean over at least 11 observers is an aggregate of the kind D22
already allows for the attribution maps (`SecureReport.mean_map`). On the fixture, the
mean over everyone of a standardised cell is 0 to 1e-9, and every cell with fewer than
11 observers in a group is null (`tests/test_hrs_chronic.py`).

**Decision.** `04_hrs_chronic_cnn.py images` builds each outcome's dataset as `run`
does (D23, D25, extended set). Each cell is standardised against everyone kept who
observed it, at that channel and age bin, which is the simulation images' "SD from
same-age peers". The report holds, per outcome: the cases' and controls' mean of that
value, their difference, the share of people observing each cell (overall and per
group), and the raw per-cell mean and SD. Any cell with fewer than 11 observers is
null. `figures` draws four panels: share observed, cases, controls, cases minus
controls. No model is fitted.

**Effect.** The figure for the paper is aggregate only. Cases and controls share one
colour scale. The share panel shows the Leave-Behind rotation and late entry that make
about 64% of the image empty. A mean is taken over the people who observed a cell, and
those people differ by cell, so a column is not a cohort followed through time.

**Alternative rejected.** A real respondent's image, drawn locally and kept off the
report: it could not be put in the paper, which was the point of the request.

---

## D29 — Analysis 04's benchmarks: tuned LightGBM, age-window, elastic-net and snapshot models
**Date:** 2026-10-26 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The networks' 0.01–0.02 AUC lead over the summary logistic (ledger,
2026-10-26) is evidence only if the strongest reasonable non-network models were in the
comparison. The boosting model was untuned, and the spec's "manual life-course" model
(§2) was missing.

**Check.** The user asked for benchmarks 1–4 from the proposed list. Yan (2026) found
LightGBM the best of 12 classifiers on HRS mortality. Timed on a synthetic training fold
the size of the real one (12,000 people, 30 × 12 cells): all eight benchmarks fit in
about 90 s per outer fold.

**Decision.** `data_analysis_04_hrs_chronic_cnn_benchmarks` fits eight models per outer
fold:
- **logistic models:** static only; ridge on per-row summaries; ridge on per-row means
  over ages 50–59, 60–69 and 70+; elastic net on every cell plus missing indicators
  (`saga`, l1 ratio 0.2/0.5/0.8);
- **sklearn boosting,** untuned, on every cell, kept for continuity with earlier runs;
- **LightGBM,** tuned, on three inputs: the last observed value per row (Yan's
  one-time-point design), the per-row summaries, and every cell.

Everything that is tuned is tuned by log loss on household-grouped inner folds. That now
includes the ridge models, whose inner folds were ungrouped before. LightGBM searches a
2 × 2 × 2 grid (leaves, learning rate, minimum child size) with early stopping, then
refits on the whole training fold with the mean best number of trees. Cells observed by
fewer than 20 training people are dropped from the cell-level models.

**Effect.** Every `run` report carries ten models per outcome, plus each benchmark's
tuning record. `logistic_summary` moves slightly from earlier runs, because its inner
folds are now grouped. `lightgbm` is added to `environment.yml`.

**Alternative rejected.** XGBoost: no gain over LightGBM for this, and it was not in
the environment either.

---

## D30 — Kernel-size experiment on diabetes: 15 CNN arms on the same folds
**Date:** 2026-10-26 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user asked whether kernel size matters: small, medium (4 rows) and
large kernels, varying in time as well as in rows, plus a whole-trajectory kernel
(1 × 12), a whole-cross-section kernel (30 × 1), and two ways of mixing sizes (parallel
and stacked). The user approved the plan before any code was written.

**Check.** Every arm builds, runs on the Apple GPU (MPS), and gives a 30 × 12 Grad-CAM
map. The whole `kernels` profile ran end to end on the fixture (test
`test_kernel_experiment_runs_every_arm_on_the_fixture`). Parameter counts run from 5.7k
(30 × 1) to 226k (10 × 9) at width 32. Size differs by design and is reported with the
results.

**Decision.** The `kernels` profile: diabetes, ages 74–76, extended set, the same 5
outer and 3 inner household-grouped folds as the 2026-10-26 run, with the CNN's tuning
grid, head and training. No benchmarks or RNN: their diabetes results share these folds
and serve as reference. `KernelNet` takes any sequence of stages; each stage is a set of
parallel kernels with "same" or "valid" padding, and "all" means the whole axis. Width is
split across a stage's branches, so the mixes match the single-kernel networks in size.

| arms | kernels (rows × bins) | layers |
|---|---|---|
| grid (9) | {1, 4, 10} × {3, 5, 9}, time dilations 1-2-4, same padding | 3 |
| trajectory | 1 × 12, valid: one variable's whole life course | 1 |
| cross_section | 30 × 1, valid: every variable at one age; it does not look across ages (`require_full_history: false`) | 1 |
| mix_parallel | 1×3 ∥ 4×5 ∥ 10×9 in every layer | 3 |
| mix_stacked | 1×3 → 4×5 → 10×9 | 3 |
| mix_separable | 30×1 → 1×12 | 2 |
| mix_two_view | 1×12 ∥ 30×1, each view collapsed separately, joined at the head | 1 |

The time-reach check is relaxed for these uncausal, bin-pooled classifiers: the reach
must span the image (time 3 reaches 15 bins), not every bin's full history. A whole-axis
kernel's Grad-CAM map is spread along the axis it collapsed: rows for `trajectory`,
columns for `cross_section`. `mix_separable` ends in one value per filter, so its
Grad-CAM map is flat and only its Expected Gradients map is informative.

**Effect.** One `run --profile kernels` report (`make hrs-kernels`), about 5 hours. The
`figures` command draws an AUC-per-arm plot (with `--reference rnn 0.750` lines) and
cases-minus-controls map grids for Grad-CAM and |Expected Gradients|.

**Alternative rejected.** Varying rows only, with time fixed at 5: the user asked for time
widths to vary too.

---

## D31 — Map stability and faithfulness: seeds, folds, row order, other models, deletion
**Date:** 2026-10-27 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** HRS has no true map, so a map can be judged only by whether it holds
still when nothing that matters changes, and by whether it points at what the model
uses. The kernel experiment (D30) showed equal accuracy alongside different maps.

**Check.** The user asked for the maps to be tested. The whole `stability` command ran
end to end on the fixture (`test_stability_runs_on_the_fixture`). The row-permutation
helper is checked to keep rows within their domain and to round-trip.

**Decision.** Four networks on diabetes, ages 74–76, on the same outer folds: the 4×5 CNN,
`mix_stacked`, `k1x5` (blind to row order) and the RNN. Per fold, each is tuned once,
then refitted with 3 seeds and on 3 within-domain row shuffles, using seed 0's
initialisation. LightGBM on the cells is tuned, and its TreeSHAP gives a map from another
model family. Every map is a population |attribution| map, with cells under 11 observers
set to null. Maps are compared by Spearman over cells, Jaccard overlap of the top 10% of
cells, and Spearman of the row and column profiles. The summaries:
- seed (the floor);
- fold;
- permutation (original against un-shuffled);
- cross-model;
- faithfulness: AUC and mean |ΔP| after setting each person's top-k cells by
  |Expected Gradients| (k = 1, 3, 5, 10, 20) to the training mean, against k random
  observed cells.

**Effect.** One `stability` report (`make hrs-stability`, about 3 hours), with fold-mean
maps and per-fold seed-0 maps. `figures` draws similarity bars, faithfulness curves, the
cross-model matrix and the maps. The RNN gets every test except Grad-CAM, which needs
convolution layers.

**Alternative rejected.** Comparing maps against the simulation's truth instead: that
answers recovery under the DGP (D14), not whether these HRS maps are stable.

---

## D32 — Learning curve: every model refitted on nested shares of each training fold
**Date:** 2026-10-27 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user asked whether the method would pull ahead with a larger sample.
More data favours the networks only if their AUC rises faster per doubling of the training
sample than the best benchmark's; a level comparison at one N cannot show that.

**Check.** The whole `learning-curve` command ran end to end on the fixture
(`test_learning_curve_runs_on_the_fixture`). The subsample helper is checked to keep whole
households and to nest each share inside the next (`test_household_subsample_is_nested_and_whole_households`).

**Decision.** Diabetes and 10-year death, ages 74–76, extended set, the same 5 outer folds
as every earlier run. In each fold the training part is cut to 10%, 25%, 50% and 100% of
its households, nested, with 2 household draws per share below 100%. At each share the CNN
(4×5), the RNN and four benchmarks (`logistic_static`, `logistic_summary`,
`lightgbm_summary`, `lightgbm_cells`) are tuned and fitted on that share alone, with the
same inner folds and grids as a `run`, and scored on the whole outer test fold. The
100% point repeats the `run` seeds. Reported per share: each model's mean and SD of test
AUC; each network's fold-paired gap to each benchmark and to the best benchmark at that
share; each model's AUC gain per doubling of N, and over the last doubling alone (50% to
100%); and the gap's slope per doubling. The samples are nested, so the slope's standard
error is too small and is reported as a guide only.

**Effect.** One `learning-curve` report (`make hrs-learning-curve`, about 5 hours) and one
figure per outcome: AUC against training N on a log scale, and each network minus the best
benchmark with 95% paired bars. `fit_benchmarks` gained `only`, to fit a subset of the
benchmarks; nothing else changes. Reading: a gap that widens per doubling supports the
claim that a larger cohort would favour the networks; parallel or flattened curves do not.

**Alternative rejected.** Extrapolating a curve past 100%: a fitted log-N slope beyond the
data is a guess. The report gives slopes, not projections.

---

## D33 — Image scaling per variable, pooled over ages and people, as an alternative to per age
**Date:** 2026-10-27 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The user pointed out that per-(channel, bin) standardisation (`cell_z`,
every run so far) removes each variable's age profile: a BMI of 30 reads differently at
55 and at 75, and the population's rise with age is gone. A convolution kernel shares its
weights over ages, so it cannot add a per-age offset back. Scaling each variable once over
every age and person keeps the trajectory. A 0-1 scale also puts the values on the range
of the mask (0/1) and Delta/K (0-1) planes.

**Check.** On synthetic data, a channel that rises with age still rises under
`channel_z` and `channel_minmax` and does not under `cell_z`. Min-max values stay in
[0, 1] with a 1e6 outlier in the data, and a binary channel with 0.2% ones stays 0/1
(`tests/test_model_pipeline.py`). The `scaling` profile runs end to end on the fixture
(`test_scaling_profile_runs_every_arm_on_the_fixture`).

**Decision.** `fit_standardiser(X, method, quantiles)` takes `cell_z` (the default, so
analysis 02, the oracle and every earlier run are unchanged), `channel_z` (one mean and
SD per channel, all bins and people of the training fold) or `channel_minmax` (per
channel, 0-1 between the training 1st and 99th percentiles, clipped, falling back to the
min and max when both percentiles land on one value). Everything is fitted on the training
fold only. Each scaler keeps mean, SD, min, max, the 0-1 bounds and the clipped mean. An
unobserved cell gets the fill value: the training mean on the scaled axis (for min-max,
the clipped mean). The deletion test (D31) now sets cells to that fill value, not to 0.
Analysis 04 reads `image_scaling` from the config, and a network's own `image_scaling`
overrides it. The `scaling` profile (`make hrs-scaling`) runs the CNN and the RNN under
each of the three scalings, on diabetes and 10-year death, with the usual folds and grids
and without benchmarks. The report gives the per-channel mean, SD, bounds and clipped mean
for each fold. Min and max are left out because each is one person's value (D22).

**Effect.** Six networks per outcome, read against `cnn` / `rnn` (`cell_z`) on paired
folds. Maps change meaning: under per-channel scaling an attribution is relative to the
variable's overall mean, not to same-age peers. The average life image (D28) keeps
peer standardisation.

**Alternative rejected.** Min-max on the raw extremes: one outlier (BMI, wealth) squashes
the rest of the channel into a narrow band. A rank (quantile) transform: also 0-1, but it
throws away distances between values; left for later if min-max loses.

---

## D34 — Who is predictable, and subgroups where the networks differ from LightGBM
**Date:** 2026-10-28 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The networks tie with tuned benchmarks on average (D29) and at every
training size (D32). An average can hide a group the networks predict better. The user
asked who is more predictable within cases and within controls, and whether any subgroup
favours the CNN or RNN over LightGBM. Diabetes is the most predictable outcome
(AUC 0.75), so it goes first.

**Check.** `test_concordance_averages_to_the_auc`: per-person concordance, averaged over
the cases or over the controls, equals the AUC, including with tied scores.
`test_subgroups_runs_on_the_fixture` runs the whole `subgroups` command and draws its figures, and
checks the report holds no per-person arrays.

**Decision.** The same outer folds and seeds as a `run`. The CNN, the RNN and four
benchmarks (`logistic_static`, `logistic_summary`, `lightgbm_summary`, `lightgbm_cells`)
give every test person an out-of-fold probability. Each network gives every test person an
Expected Gradients map. These stay local; the report holds group means over at least 11 people.
- *Predictability.* A person's concordance under a model is the share of the other class
  in the same test fold that the model ranks them correctly against, ties counted half.
  Its mean over the cases is the AUC. Cases and controls are each cut into quintiles of
  the mean concordance over LightGBM-summary, the CNN and the RNN. For each quintile the
  report gives: each image row's mean cohort z; the statics; how much of the image was
  observed; the mean life image; and the mean signed and absolute EG maps with each row's
  share of |EG|.
- *Subgroups.* These are fixed in advance from the covariates (sex, race/ethnicity,
  birthplace, education, birth cohort, occupation, marriage, missing parental education,
  observed bins, entry age, leave-behind response, last BMI, last wealth) and from
  quintiles of the static logistic's risk. They are never defined by a compared model's
  errors. For each subgroup the report gives each model's AUC and each network's
  fold-paired gap to `lightgbm_summary` and to `lightgbm_cells`, with a t interval over
  folds and a Holm-adjusted p.
- *Disagreement.* In each fold, take the 10% of people whose rank under a network and under
  `lightgbm_summary` differs most. Split them by class and by which model ranked them
  better, then profile each group as for the quintiles.

**Effect.** One `subgroups` report (`make hrs-subgroups`, about 1 h). There are three
figures: a predictability heat map, a forest plot of subgroup gaps, and a disagreement
heat map. The EG maps of each group are in the report, to be drawn once the results are in.

**Alternative rejected.** (1) Ordering the learning-curve samples from least to most
predictable: that ties sample size to sample composition, and "predictable" depends on
the model that defines it. (2) Subgroups defined by LightGBM's own errors: by regression to
the mean, any other model would look better there. (3) Clustering cases' individual EG
maps into pathways: deferred, see STATE open question 16.


---

## D35 — Pathways: do cases' individual EG maps fall into clusters that recur?
**Date:** 2026-10-28 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The networks tie with LightGBM overall (D29), at every training size (D32)
and in every fixed subgroup (D34). Any contribution must come from the maps. An average map
can hide several routes to the same outcome, for example cases flagged by weight gain and
cases flagged by economic decline or stress. The user asked to cluster cases' individual
EG maps into such pathways, on diabetes first.

**Check.** `test_pathway_stability_finds_planted_pathways_and_not_noise`: two planted
pathways come back stable under seed, bootstrap and fold replicates at k = 2; k = 4 on
pure noise does not. `test_pathways_runs_on_the_fixture` runs the whole `pathways` command
with its figures and checks the report holds no per-person arrays.

**Decision.** The same outer folds, seeds and fits as a `run` and D34's `subgroups`. Every
test case gets an out-of-fold EG map (sigmoid output) from the CNN and the RNN.
- *Pathway vector.* Per case, the signed EG summed over ages within each image row, divided
  by the case's total |EG| over observed cells. The vector says which rows pushed the case's
  risk up or down and by what share, whatever the case's overall risk. It uses the image
  only: the statics' attribution is not part of a pathway. `level: domain` sums by the five
  domains instead.
- *Clustering.* k-means (20 starts) for k = 2–6, per network, on all cases with any attribution.
- *Recurrence.* (1) Seeds: each fold's networks are refitted with two other seeds (same tuned
  hyperparameters and baselines), and the same cases' new vectors are clustered again.
  (2) Bootstrap: 20 k-means refits on resampled cases; every case is assigned to the nearest
  new centroid. (3) Folds: each test fold's cases are clustered on their own, and their
  centroids are matched to the pooled ones (Hungarian, cosine). A cluster is *stable* if
  its clusterwise Jaccard (Hennig 2007) is ≥ 0.75 over seeds and over bootstraps, and its
  matched centroid cosine averages ≥ 0.8 over folds. The chosen k is the largest at which
  every cluster is stable; if none passes, there is no stable pathway, and that is the result.
- *Profiles.* Each cluster with at least 11 cases is profiled as D34's quintiles are: row
  means, statics, observed share, concordance under each model, mean life image, mean signed
  and absolute EG maps. The report also gives the CNN–RNN agreement on cluster membership
  (ARI, clusterwise Jaccard) at each k.

**Effect.** One `pathways` report (`make hrs-pathways`, about 1.5–2 h: D34's fits plus 20
refits). Four figures: recurrence by k, cluster centroids, cluster profiles, and one mean
EG map per cluster for each network. Per-person maps, vectors and labels stay local (D22).

**Alternative rejected.** (1) Clustering the raw 30 × 12 maps: dominated by how many ages a
person was observed and by the overall risk level, not by which rows matter. (2) Choosing k
by silhouette alone: it rewards separation that need not recur under a refit. Silhouette
is reported but does not decide. (3) Clustering only cases the models rank well: that
conditions on the models' success, and D34 already profiles the hard cases.

---

## D36 — Pathways without BMI: can the smaller rows separate?
**Date:** 2026-10-29 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** D35 found one stable split of diabetes cases, and it is the sign of BMI's
attribution. BMI takes 26–37% of a case's |EG|, so it sets most of the distance between two
pathway vectors and can hide structure in the other 29 rows. At k = 3 an economic group (low
wealth, Medicaid, renting) passed the bootstrap and fold checks but not the seed check. The
user asked to re-cluster with the BMI row left out.

**Check.** `test_dropping_a_dominant_row_lets_smaller_pathways_separate`: a dominant row
whose sign is unrelated to two planted pathways splits the cases on that sign; with the row
dropped, the planted pathways come back (> 95% agreement) and are stable.
`test_pathways_runs_on_the_fixture[pathways_no_bmi]` runs the profile end to end and checks
that BMI is absent from the vectors.

**Decision.** D35 unchanged except `pathways.drop_rows: [bmi]` (profile `pathways_no_bmi`).
The BMI row leaves both the vector and its total: each case's vector is the signed EG per
remaining row as a share of the case's |EG| over the remaining rows. The networks, folds,
seeds, refits, maps, k grid and stability thresholds are D35's, so the two reports are
comparable case by case. The profiles still report BMI (row means, EG maps, row shares), so
it can be seen how each new cluster stands on BMI.

**Effect.** One `pathways` report with `dropped_rows: ["bmi"]` (`make hrs-pathways-no-bmi`,
about 40 min). Figures are named `pathways_no_bmi_*` and sit beside D35's.

**Alternative rejected.** (1) Keeping BMI in the total and only dropping it from the vector:
the remaining shares would then still scale with how much BMI took, which brings BMI back as
the vector's length. (2) Residualising each row's attribution on BMI: it assumes a linear
relation between BMI's attribution and the others, and is harder to read. (3) Standardising
each row before clustering: it lets near-zero rows count as much as wealth or activity.

---

## D37 — The main run at other ages of the status interview
**Date:** 2026-10-31 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** Every result so far is at ages 74–76 (bin 13, a 12-bin image). The tie with
the benchmarks could depend on how long a history the image holds: at younger ages it is
shorter, so there is less trajectory for a network to use; at older ages it is longer, but
fewer people are left. The heart-disease learning curve (D32) is the one result where the
networks were still gaining on the best benchmark at full size. The user chose this as B1.

**Check.** The smoke profile runs at bins 9, 11 and 15 on the fixture for all four outcomes.
The 4x5 CNN with dilations 1-2-4 reaches 14 bins either side, so bin 15 (14 bins) is the
widest image it sees whole (D13); a wider one would fail the reach check.

**Decision.** `run` unchanged (profile `full`, every benchmark, both networks, maps) at
`--status-bin` 9 (ages 66–68), 11 (70–72) and 15 (78–80), one report per age. Folds and
seeds follow from the outcome and bin (`seed_for`), as at bin 13.

**Effect.** `make hrs-ages`: three `run` reports, about 3–4 h each.

**Alternative rejected.** (1) A single model across ages with age as an input: it answers a
different question (one model for everyone), and mixes people seen at several ages.
(2) Bins 7 and 17: too short an image (6 bins) and too few people.

---

## D38 — Time order: do the networks use the order of the life course?
**Date:** 2026-10-31 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The networks tie with LightGBM on cells and summaries, which ignore the order
of the bins. If the networks gained nothing from order, the tie would be expected. D31
shuffled image *rows*; this is the age-axis counterpart. The user chose this as B2.

**Check.** `test_time_shuffles_keep_each_persons_values`: both shuffles keep each row's
observations and values; `fixed` moves every person's bins the same way; `person` leaves
whole-bin gaps (before entry, after the last interview) in place; `Delta` is recomputed.

**Decision.** Two arms, each retuned and refitted with the run's seeds on the same folds.
`fixed`: one random permutation of the bins for everyone (per fold). All information is
kept, only adjacency changes, so a loss means the architecture uses order. `person`: each
person's observed bins are shuffled among themselves. The values and the entry pattern stay,
but timing is lost, so a loss means timing carries information. The `person` shuffle is also
given to LightGBM summary and cells. A fixed order changes nothing for a tree on cells.
Reported as fold-paired AUC (and log-loss, Brier) gaps to the original fits.

**Effect.** Part `time_order` of `comparisons`.

**Alternative rejected.** (1) Shuffling only at test time: it measures sensitivity to
inputs the model never saw, not whether order can be learnt from. (2) Reversing time: one
fixed order that keeps adjacency, so it tests direction only.

---

## D39 — Inputs: image only, statics only, both
**Date:** 2026-10-31 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The static covariates alone reach AUC 0.60–0.65. Every image model also sees
them, so its gain over them is the image's whole contribution. Question 11 found that the
extended set adds mostly through the statics. The user chose this as B3.

**Check.** `test_image_only_drops_the_statics_from_every_input`: no static column reaches the
networks' heads, the summary features or the cell features.

**Decision.** Each network (retuned), the summary logistic, LightGBM summary and LightGBM cells
are refitted with the static covariates removed. LightGBM is tuned on the statics alone,
beside the static logistic. Gaps: image only minus both, and both minus statics-only LightGBM.

**Effect.** Part `inputs` of `comparisons`.

**Alternative rejected.** A network with its image zeroed as "statics only": it is a logistic
regression with extra steps. LightGBM on the statics is the stronger statics-only reference.

---

## D40 — Stacking the networks with the benchmarks
**Date:** 2026-10-31 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The networks and LightGBM tie on AUC, and D34 found their per-person rankings
agree at Spearman 0.79–0.94, not 1. Whatever is left could be combined. The question is whether
adding the networks to a stack of benchmarks helps. The user chose this as B4.

**Check.** `test_stack_weighs_the_informative_model_and_gaps_are_paired`: the stacker puts its
weight on an informative model over noise, from inner predictions alone.

**Decision.** In each outer fold, every stacked model is refitted on each inner fit part (the
run's inner folds) at the outer fold's hyperparameters, predicting its inner validation part.
The summary logistic re-chooses its ridge strength inside. A logistic regression (C = 1) on
those logits is applied to the outer models' test logits, so the outer test fold never
informs the weights. Three stacks are fitted: all five models; the three benchmarks (summary
logistic, LightGBM summary, LightGBM cells); and CNN plus RNN. A plain average of the five
probabilities is also scored. The deciding gap is the full stack minus the benchmark stack.

**Effect.** Part `ensemble` of `comparisons`; mean stack weights and inner AUCs are reported.

**Alternative rejected.** (1) Weights fitted on the other outer folds' test predictions: those
models were trained on the fold being scored, so its labels leak into the weights. (2) Using
the predictions from tuning: LightGBM's early stopping used the same validation part, so they
are optimistic and would over-weight it.

---

## D41 — BMI: trajectory or level?
**Date:** 2026-10-31 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** The EG maps show BMI as the dominant row all along the life course
(19–30% of |EG|, D31, D35). The user asked whether its trajectory matters or one wave would
do. D38 tests order over every row at once. This asks the question of one row, keeping the
other 29 as they are. The LightGBM snapshot benchmark (D29) mixes every row (summary minus
snapshot +0.018 for diabetes), so it cannot answer for BMI alone.

**Check.** `test_row_variants_rewrite_only_the_bmi_row`: every other row, its mask and
`Delta` are unchanged. `none` leaves no BMI cell. `single` keeps exactly the last observed
cell, at its value. `level` and `shuffled` keep the BMI mask; `level` puts the person's mean
in every observed bin; `shuffled` keeps each person's values and moves them.
`test_bmi_trajectory_profile_runs_on_the_fixture`: the profile runs only this part.

**Decision.** A fourth `comparisons` part, `bmi_trajectory`. On the run's folds and seeds,
the CNN, the RNN, LightGBM summary and LightGBM cells are retuned and refitted on four
rewrites of the BMI row: `none` (removed), `single` (latest observed value only, the wave
nearest the status interview), `level` (the person's mean in each observed bin: level and
observation pattern kept, shape lost) and `shuffled` (observed values permuted among the
person's observed bins: spread kept, order lost). Each is reported as a fold-paired gap to
the original fit and to `none`. A gap of the original over `level` and `single` that is
clear of zero means the trajectory adds something beyond the level. Run through the
`bmi_trajectory` profile, for diabetes and hypertension, where BMI dominates the maps (D35).
`single` also drops observation cells, so it changes the mask a model can read, as a
single-wave survey would.

**Effect.** Part `bmi_trajectory` of `comparisons`; `make hrs-bmi-trajectory`, in `hrs-b`.
Not in the default `parts`, so `make hrs-comparisons` (D38–D40) is unchanged.

**Caveat.** Cases include prevalent diabetes and values recorded after diagnosis stay in
the image (D23). A trajectory that helps may partly be weight change *after* diagnosis, the
disease shaping BMI. Restricting to incident cases, or cutting the image at the first
report, would separate the two. Not built.

**Alternative rejected.** (1) BMI as a static covariate at one wave: the networks would
read it through a different path from the other rows, so the arms would differ in more
than the trajectory. (2) The level broadcast over all twelve bins: that also changes the
mask, mixing "no shape" with "observed everywhere".

## D42 — Socioeconomic order, and for whom it matters
**Date:** 2026-11-02 · **Kind:** `method` · **Status:** `active` · **Scope:** `04`

**Assumption.** D41 found BMI's order adds nothing on average. The user asked whether the
same holds for people whose socioeconomic position changed. An average gap can hide a loss
among a minority, and a shuffle cannot touch people whose rows never move, so it is diluted
towards zero by construction. The question needs gaps within groups, and the groups need
per-person data, so they are formed and scored inside the run (D22).

**Check.** `test_rows_variant_shuffles_the_socioeconomic_rows_together`: other rows, their
masks and `Delta` are unchanged. `none` leaves no cell of the group. `shuffled` moves whole
columns of the group, so values seen together stay together, and bins where none of the
group was seen stay empty. `test_strata_split_people_by_change_and_events`: people with under
two waves stand apart, thirds of change partition the rest with flat people in the lowest,
and an event marks a later fall or rise of at least its size. `test_ses_order_profile_runs_on_the_fixture`:
the profile runs, and suppressed strata report no gaps.

**Decision.** A fifth `comparisons` part, `ses_order`, over the seven socioeconomic rows with
more than two waves (income, wealth, poverty, working, unemployed, Medicaid, not a homeowner;
financial strain has two waves and is left as it is). Arms: `none` (removed), `level` (each
row at the person's mean) and `shuffled` (the group shuffled in time with one permutation per
person over the bins where any of it was seen). The networks, LightGBM summary and LightGBM
cells are retuned and refitted, gaps paired over folds as in D41.
Strata, set from the original image and applied to the test fold of every arm in
`strata.parts` (`time_order`, `ses_order`): under two waves of these rows; thirds of a change
score (each row z-scaled over everyone, its within-person SD, averaged over rows; ties at a
cut go below); and six events with their complements among people with two or more
observations of the row: wealth or income falling by at least 1 on the asinh scale (about
63%), leaving work, entering Medicaid, falling below the poverty line, ceasing to own a home.
Within each stratum: AUC and log loss of each arm minus the original, paired over folds, and
the difference of that gap between top and bottom thirds and between each event and its
complement. Log loss is per person and so less noisy than a small group's AUC. A stratum with
under 11 cases or controls in any test fold reports a suppressed count only.

**Effect.** Part `ses_order` of `comparisons`; profile `ses_order`, all four outcomes;
`make hrs-ses-order` (~9 h). Strata are in the default `comparisons` config, so `make
hrs-comparisons` (D38) now also reports the time-order gaps by stratum, at negligible cost.

**Caveat.** The groups describe the observed image, which already contains the change, so
they are descriptive, not exogenous: people who lost wealth differ in other ways too. A
larger gap among changers says the networks use order there; it does not say the order
causes the outcome. Leaving work is near universal from 50 to 76 (retirement), so that event
mostly separates people who were working at 50 from those who were not. The shuffle keeps
the group aligned with itself but not with the other 23 rows, so a loss may come from broken
timing between, say, wealth and depressive symptoms, not within wealth.

**Alternative rejected.** (1) Shuffling only the changers: retuned on a mixture, a model
could learn which people were shuffled. (2) Groups from the models' own errors, as D34's
concordance quintiles: they select on the outcome. These groups use the image alone.

## D43 — Socioeconomic order tested on changers only
**Date:** 2026-11-02 · **Kind:** `method` · **Status:** `active` · **Scope:** `04` · **Amends:** D42

**Assumption.** The user wants the socioeconomic-order test (D42) made only on people who
went through significant socioeconomic change. Under D42 the models are fitted on everyone
and changers are only a stratum of the test fold, so a model fitted mostly to people whose
rows barely move sets the baseline.

**Check.** `test_population_keeps_only_the_named_strata`: the kept sample is exactly the
union of the named strata, every per-person array is cut with it, `null` keeps everyone, and
an unknown stratum is an error. `test_ses_order_profile_runs_on_the_fixture`: the report
carries the population's size, below the full sample, and no under-two-wave person is left.

**Decision.** A `comparisons.population` option naming D42 strata. When set, the dataset is
cut to the people in any of them before the outer folds are drawn, so every arm, the
original included, is tuned, fitted and scored on those people alone. The `ses_order`
profile sets it to anyone with a socioeconomic shock in the image: wealth or income falling
by at least 1 on the asinh scale (about 63%), entering Medicaid, falling below the poverty
line, or ceasing to own a home. Leaving work is not a shock here (mostly retirement). The
report's `population` section gives the kept n, cases and controls (suppressed below 11).
D42's strata still apply inside this sample (thirds of change, each event against the rest
of the changers).

**Effect.** `make hrs-ses-order` now runs on changers only; the default `comparisons`
(D38–D40) keeps everyone. Smaller samples make it faster than D42's ~9 h estimate.

**Caveat.** A smaller sample widens every interval, and the cut changes who the models are,
so AUCs here are not comparable to the main run's; only gaps between arms on the same people
are. Changers differ from the rest in ways the image shows (they are poorer and less often
working), so the result is about order among these people, not order in general.

**Alternative rejected.** Shuffling only the changers within the full sample (D42's first
rejected alternative): fitted on a mixture, a model could learn who was shuffled. Cutting
the sample keeps every arm on the same people with the same treatment.

## D44 — BMI trajectory tested on BMI changers only
**Date:** 2026-11-02 · **Kind:** `method` · **Status:** `active` · **Scope:** `04` · **Builds on:** D41, D43

**Assumption.** D41 found BMI's order and shape add nothing on average, but most people's BMI
moves little from 50 to 76, so their shuffled and mean rows are near copies of the original.
The user asked whether the result holds for people whose BMI did change, as D43 asks for
socioeconomic order.

**Check.** `test_bmi_changers_profile_runs_on_the_fixture`: the profile keeps the D41 part,
replaces D42's strata with BMI ones while the rest of `comparisons` survives the merge, cuts
the sample below the full one, and reports gaps for every arm-to-original pair by stratum.

**Decision.** A `bmi_changers` profile: D41's four arms (`none`, `single`, `level`,
`shuffled`) with `comparisons.population` set to people with a BMI gain or loss of at least
3 kg/m² between any earlier and later wave (about 8–9 kg at 1.70 m; `_swing`, D42). Strata
inside the sample: thirds of the within-person SD of BMI, and gain and loss each against the
other changers. Diabetes and hypertension, as D41. The threshold and groups are config, so
other cuts (a gain only, a loss only, the top third) need no code.

**Effect.** `make hrs-bmi-changers` (under ~5 h). D41's `bmi_trajectory` profile, on
everyone, is unchanged.

**Caveat.** Cases include prevalent diabetes (D23). Weight loss after diagnosis is common,
so among losers a trajectory that predicts may be the disease's effect on BMI, not a risk
factor's. A gain-versus-loss contrast is read with that in mind; incident cases (future
direction 5) would settle it. Self-reported BMI is noisy, and one bad wave can make a 3-unit
swing, so a few changers are artefacts; the thirds of SD are less sensitive to one wave.
