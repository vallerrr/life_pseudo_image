# 01 — DGP diagnostics: report

**Purpose.** What the six diagnostic runs found: whether the DGP hits its pre-specified
targets, and what a reviewer should check before model work is unblocked.

*Last updated: 2026-08-29 (rev. 1).*

| | |
|---|---|
| **Run** | 2026-08-29, `N = 10,000`, `replicate = 0`, `master_seed = 20260724`, git `ee4ba09` |
| **Produced by** | [`src/data_analysis/01_dgp_diagnostics.py`](../src/data_analysis/01_dgp_diagnostics.py) |
| **How it works** | [`01_dgp_diagnostics_methodology.md`](01_dgp_diagnostics_methodology.md) |
| **Figures** | `output/figures/data_analysis/01_dgp_diagnostics/dgp_<scenario>_20260829.pdf` (6 pages) |
| **Source table** | [`output/tables/dgp_manifest_20260829.json`](../output/tables/dgp_manifest_20260829.json) |

---

## §0 At a glance

| scenario | seed | disease CIF | mortality CIF | attrition | missing cells | gen. time |
|---|---|---|---|---|---|---|
| `accumulation` | 4073343629 | 0.1573 | 0.1964 | 0.3012 | 0.4565 | 1.51 s |
| `critical_period` | 1544072896 | 0.1493 | 0.2040 | 0.2969 | 0.4554 | 1.38 s |
| `sensitive_period` | 1228263828 | 0.1536 | 0.1988 | 0.2996 | 0.4558 | 1.36 s |
| `chain_of_risk` | 3499874961 | 0.1482 | 0.2082 | 0.2984 | 0.4549 | 1.30 s |
| `mobility` | 3299260623 | 0.1427 | 0.1991 | 0.3030 | 0.4553 | 1.27 s |
| `null` | 3876413547 | 0.1445 | 0.2020 | 0.2970 | 0.4556 | 1.25 s |
| **target** | — | **0.150** | **0.200** | **0.300** | — | — |

Expected (closed-form) CIFs are 0.150 and 0.200 to within 1e-8 in **all six** scenarios.

**The one fact that must not be missed.** *The realized column above is not the
calibration check — the expected column is.* Calibration is solved in closed form and is
exact to 1e-8; the realized figures deviate by up to 0.0073 purely as Monte Carlo error
at `N = 10,000`. Reading `mobility`'s 0.1427 as "under target" and re-tuning would be
fitting noise, and would break the equal-effect-size property (D02) that makes the six
scenarios comparable at all.

## §1 Calibration hits every target

- **Disease.** Expected 0.150 everywhere; realized 0.1427–0.1573, a spread of 0.0146
  across six independent draws.
- **Mortality.** Expected 0.200 everywhere; realized 0.1964–0.2082.
- **Attrition.** Realized 0.2969–0.3030 against a 0.300 target — the tightest of the
  three, because dropout has 20 bins of opportunity rather than competing against a
  second cause.
- The three calibrated intercepts vary slightly by scenario (disease −4.84 to −4.78),
  which is expected: each scenario's history score has a different variance contribution
  to the linear predictor, so a different intercept is needed to land on the same CIF.

**Verdict:** the calibrator does what D03 claims. No target is missed.

## §2 Ground truth is where the scenarios say it is

Checked by assertion in [`tests/test_shared_dgp.py`](../tests/test_shared_dgp.py), not by
eye:

- `critical_period` support is non-zero **only** in bins 5–8 and zero in all other 16.
- `chain_of_risk` support is confined to upstream channels 0–3 in bins 1–6 and downstream
  channels 8–11 in bins 10–14; midstream channels 4–7 carry exactly zero.
- `null` support is exactly the `12 × 20` zero matrix — not approximately zero. This is
  the falsification anchor: any model reporting structure here is reporting an artifact.

## §3 The mask behaves as designed

Observation rate is ~54.5% of cells in every scenario (0.4549–0.4565 missing), and the
per-channel/per-bin panel on each report page shows the intended three-band structure:
upstream and downstream at ~0.90 (item non-response only), midstream at 0 on odd bins and
~0.45 on even bins (rotating half-sample × item non-response), and a uniform decay across
all channels as dropout accumulates.

## §4 Performance

Each scenario generates in 1.25–1.51 s at `N = 10,000`; the six-scenario suite plus PDF
rendering completes in well under a minute on one core. Nothing here needs a cluster job,
so the standard's §8 machinery is deliberately unused. Re-check this before the first
replicate sweep — 500 replicates × 6 scenarios is ~70 minutes serial, which is the point
at which sizing it properly starts to matter.

## §5 Tricky points

**Tricky point — realized vs. expected.** See §0. Both are in the manifest; only one is
the calibration check.

**Tricky point — outcome follow-up survives dropout.** Attrition masks *covariates* from
the dropout bin onward, not outcomes. Event times and types remain observed for everyone.
A model that treats dropout as censoring of the outcome is answering a different
question than this DGP poses.

**Tricky point — the midstream half-sample is shared.** The rotating selection is drawn
once per individual per even bin and applied to all four midstream channels together, not
independently per channel. Anything that assumes independent channel-level missingness
within midstream will mis-estimate the mask's dependence structure.

**Tricky point — `null` is not "no data".** It is a full dataset with a genuine 15%/20%
event process driven entirely by static covariates and the frailty/downstream mortality
path. Only the *history* effect is zero.

## §6 What this does not establish

- **No model has been fitted.** Nothing here is evidence for or against the paper's
  recovery claim; the DGP exists to make that claim testable, not to test it.
- **No kill criteria are pre-registered yet.** [Section 8 of the
  specification](simulation_study_specification.md) requires them fixed *before* any model
  runs. They are not yet transcribed into a decision record — open question 3 in
  [`STATE.md`](STATE.md).
- **Six draws is not a sampling distribution.** The spread in §0 is one realisation per
  scenario. It is consistent with Monte Carlo error but does not measure it; a replicate
  sweep would.
- **No real data has been touched.** Targets are plausible, not estimated from HRS.

## §7 Next

Model implementation is blocked pending human review of the six report pages. On approval:
pre-register the kill criteria (open question 3), then build the model stage under
`src/data_analysis/02_*`.
