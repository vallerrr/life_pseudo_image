# STATE

*Overwritten in place — this file holds only what is true now. History lives in
[`decisions_shared.md`](decisions_shared.md).*

**As of:** 2026-08-29 · **Stage:** first deliverable (DGP) built and self-checking; model
fitting not started.

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
| DGP unit tests passing | 5 / 5 | — | 2026-08-29 |
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
3. **Which kill criteria are pre-registered?** [Section 8 of the study
   specification](simulation_study_specification.md) requires them fixed before any model
   runs. They are not yet transcribed into a decision record.
4. **Does `data/analysis/` ever get written to?** The diagnostics currently regenerate
   datasets in memory each run. If replicate sweeps arrive, decide whether to persist
   them or keep re-deriving from the seed tuple.
