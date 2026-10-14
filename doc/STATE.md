# STATE

*Overwritten in place — this file holds only what is true now. History lives in
[`decisions_shared.md`](decisions_shared.md).*

**As of:** 2026-10-14 · **Stage:** DGP built and self-checking; the pseudo-image CNN
pipeline (analysis 02) built and smoke-tested. No study model run has been made.

---

## 1. Active decisions

| id | one line | kind |
|---|---|---|
| D01 | The repository root, not `image_converter/`, is the canonical root | `data` |
| D02 | Scenario history scores are standardised per bin, effect fixed at `log(1.5)` per SD | `method` |
| D03 | The 15% / 20% targets are joint competing-risk cumulative incidences, solved by alternating bisection | `method` |
| D04 | Ground truth is a population-level `C × T` structural support map, not an individual gradient | `method` |
| D05 | The study's `dgp` / `models` / `eval` module boundaries live beneath `src/` | `method` |
| D06 | `doc/00_project_conventions.md` is the authoritative layout; deltas in `doc/conventions.md` | `method` |
| D07 | HRS is provisionally `blind` until its data use agreement clause is confirmed | `data` |
| D08 | Model input is one persisted `.npz` per dataset: raw `X`, outcome censored at attrition, `truth_*` apart | `data` |
| D09 | The 2D pseudo-image CNN is causal in time, sees `cohort, sex`, and is scored on observed cells | `method` |
| D10 | The per-person PNG is pixels only, on a shared calendar-year grid; models never read it | `method` |
| D11 | Kill criteria K1–K5 pre-registered with numeric thresholds, before any `study` fit | `method` |

## 2. Ledger

Every figure the project currently treats as established. Rows are replaced, never
appended to. All figures below are `N = 10,000`, `replicate = 0`, `master_seed = 20260724`.

| quantity | value | set by | as of |
|---|---|---|---|
| expected disease CIF at T (all six scenarios) | 0.150 (to 1e-8) | D03 | 2026-08-29 |
| expected mortality CIF at T (all six scenarios) | 0.200 (to 1e-8) | D03 | 2026-08-29 |
| realized disease incidence, range over scenarios | 0.1427–0.1573 (target 0.15) | D03 | 2026-08-29 |
| realized mortality, range over scenarios | 0.1964–0.2082 (target 0.20) | D03 | 2026-08-29 |
| realized attrition, range over scenarios | 0.2969–0.3030 (target 0.30) | D03 | 2026-08-29 |
| missing cell fraction, range over scenarios | 0.4549–0.4565 | D02 | 2026-08-29 |
| generation time per scenario at N = 10,000 | 1.25–1.51 s | — | 2026-08-29 |
| unit tests passing (DGP 5, model pipeline 7) | 12 / 12 | D08, D09 | 2026-10-14 |
| true-hazard ceiling, `critical_period`, AUC at bin 20 / C-index / interval NLL | 0.710 / 0.659 / 0.110 | D08 | 2026-10-14 |
| cohort + sex alone (true coefficients), `critical_period`, AUC at bin 20 | 0.599 | D08 | 2026-10-14 |
| CNN `study` shard time on Apple GPU, N = 2k / 5k / 10k / 20k (fold 0, upper bound) | 49 / 124 / 211 / 498 s | D09 | 2026-10-14 |
| CNN `study` profile, full design, run serially | ~147 h for 2,400 shards | D09 | 2026-10-14 |
| observed outcomes, `critical_period` | 7,339 event-free or censored · 1,085 disease · 1,576 death | D08 | 2026-10-14 |
| scenarios implemented | 6 / 6 | — | 2026-08-29 |
| `null` ground-truth support | exactly zero, `12 × 20` | D04 | 2026-08-29 |
| HRS local pull size | 4.8 GB, 18 wave directories + RAND longitudinal file | D07 | 2026-08-29 |
| declared sources | 2 — `simulation` (`open`), `hrs` (`blind`, provisional) | D07 | 2026-08-29 |

Source: [`output/tables/dgp_manifest_20260829.json`](../output/tables/dgp_manifest_20260829.json).
A result that contradicts a row here is reporting a bug until proven otherwise.

## 3. Open questions

1. **What sensitivity tier does HRS actually carry?** D07 sets `blind` provisionally.
   Settled by locating the clause of the HRS/RAND data use agreement that governs
   transfer to a third party, and recording it in [`doc/data/hrs.md`](data/hrs.md). Until
   then no HRS row is read into an assistant's context, and a synthetic fixture matching
   the RAND schema does not yet exist — which the standard (§7) says the tier requires
   before any code can be written against it.
2. **Is the DGP approved?** The first deliverable is deliberately DGP-only; model
   implementation is blocked on a human review of the six diagnostic reports in
   `output/figures/data_analysis/01_dgp_diagnostics/`.
3. **Are the D11 thresholds the ones to keep?** The kill criteria are written down (D11),
   with thresholds the assistant set. They can be amended only before the first `study`
   result is read. Settled by a human read of D11.
4. **How is the study profile run?** Probed at ~147 h serially on the Apple GPU (ledger).
   Four probe shards (accumulation, replicate 0, fold 0, each N) are written and will be
   reused; their metrics have not been read. Settled by choosing between running as is, the
   spec's minimum scenario set, fewer replicates, tuning once per cell, or a cluster.
5. **Does the CNN learn anything beyond the static covariates?** On one development fold
   it did not (D09). Nested tuning over the widened grid is the test.
