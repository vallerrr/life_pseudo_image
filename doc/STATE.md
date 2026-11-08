# STATE

*Overwritten in place — this file holds only what is true now. History lives in
[`decisions_shared.md`](decisions_shared.md).*

**As of:** 2026-11-03 · **Stage:** DGP built and self-checking; the pseudo-image CNN
pipeline (analysis 02) built and smoke-tested. No study model run has been made. Work has
moved to HRS (2026-10-25): the HRS outcome models (CNN and RNN, chronic conditions and death) with activation maps (analysis 04) and
the run-and-report protocol (D22) are built and tested on the fixture. Real runs through D34 are in
(ledger). The networks tie with LightGBM everywhere tested; D35 found no pathway in the EG maps
beyond the sign of BMI, and D36 (the same without BMI) only a socioeconomic gradient
(ledger). D37 repeats the main run at ages 66–68, 70–72 and 78–80: the tie holds at every age, longer images do not favour the networks, and the networks fall behind only on death, the CNN most (ledger). D41: BMI's contribution is its level averaged over waves; its order in time adds nothing for any model, and D44 finds the same among people whose BMI changed by 3+ kg/m². D38–D40: timing is worth at most ~0.01 AUC (death only), and the networks add nothing to a stack of the benchmarks (ledger). D42–D43: among people with a socioeconomic shock, the socioeconomic rows add nothing beyond the statics and other rows, so their order cannot matter (ledger). Individual maps say how much of the risk pattern a case carries, not which route. Open: what the
paper claims (question 12).

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
| D10 | The per-person PNG is pixels only, on a shared calendar-year grid; models never read it (rows amended by D12) | `method` |
| D11 | Kill criteria K1–K5 pre-registered with numeric thresholds, before any `study` fit | `method` |
| D12 | The per-person PNG drops the fixed traits and draws years outside ages 50–90 black | `method` |
| D13 | The 2D CNN must see the whole history: 4x5 kernels, time dilations 1-2-4 (amends D09) | `method` |
| D14 | Oracle checks: K2 and K4 as written fail a correct model; amendments not adopted | `method` |
| D15 | HRS channels: twelve RAND variables, four per stream; the midstream is the Leave-Behind | `data` |
| D16 | HRS outcome: diabetes first reported after entry; death counts only straight after the last interview | `method` |
| D17 | HRS time axis is age at interview end; the later value wins within a bin; nothing after `obs_time` | `method` |
| D18 | Left truncation: `entry_time` in `ModelInput`; risk sets, losses and metrics start there | `method` |
| D19 | HRS money deflated by CPI-U to 2022 dollars, asinh of thousands | `data` |
| D20 | HRS seams: wave-8 life satisfaction rescaled, vigorous activity binarised at wave 7 | `data` |
| D21 | HRS `S`: birth year, sex; frailty unmeasured (NaN) | `method` |
| D22 | Blind sources return as aggregate JSON reports run by the user; cells < 11 suppressed | `data` |
| D23 | HRS binary task: chronic condition at an interview at age A, image ages 50 to A; Grad-CAM + EG maps | `method` |
| D24 | An RNN (bidirectional GRU over bins) joins the CNN on the same image, folds and EG maps | `method` |
| D25 | Death within 10 years of the age-A interview as a binary outcome; unknown vital status excluded | `method` |
| D26 | Extended HRS dictionary: the Yan (2026) factors RAND carries — 29 image rows in 5 domains, 11 static covariates | `data` |
| D27 | Married/partnered row added (30 rows); financial strain read from waves 8, 10–16 | `data` |
| D28 | The average life image: per-cell peer-standardised means for cases and controls, cells < 11 null | `method` |
| D29 | Analysis 04 benchmarks: tuned LightGBM (snapshot, summary, cells), age-window ridge, elastic net on cells | `method` |
| D30 | Kernel-size experiment on diabetes: 15 CNN arms (row × time grid, 1×12, 30×1, four mixes), same folds | `method` |
| D31 | Map stability and faithfulness on diabetes: seeds, folds, within-domain row shuffles, cross-model (incl. LightGBM SHAP), deletion | `method` |
| D32 | Learning curve: CNN, RNN and four benchmarks refitted on 10–100% of each training fold's households, diabetes and death | `method` |
| D33 | Image scaling option: per variable over all ages and people (z-score or 0–1 min-max) vs per age; `scaling` profile compares | `method` |
| D34 | Who is predictable (concordance quintiles within cases and controls) and fixed subgroups where the CNN/RNN may beat LightGBM; diabetes, `subgroups` command | `method` |
| D35 | Pathways: cases' EG maps reduced to signed row shares, k-means k = 2–6, kept only if stable over seeds, bootstraps and folds; diabetes, `pathways` command | `method` |
| D36 | D35 re-run with the BMI row left out of the pathway vectors and their totals, so the smaller rows can separate; `pathways_no_bmi` profile | `method` |
| D37 | The main run at three more status ages: bins 9, 11, 15 (66–68, 70–72, 78–80), one report each | `method` |
| D38 | Time order: networks retuned on a fixed bin permutation (adjacency) and a per-person shuffle (timing); LightGBM too under the latter | `method` |
| D39 | Inputs: every network and benchmark refitted on the image only; LightGBM on the statics only | `method` |
| D40 | Stacking: logistic on logits, weights from inner out-of-fold refits; all five vs the three benchmarks vs CNN + RNN | `method` |
| D41 | BMI trajectory or level: the BMI row removed / latest wave only / person mean / shuffled; networks and LightGBM refitted; diabetes, hypertension; `bmi_trajectory` profile | `method` |
| D42 | Socioeconomic order: the seven socioeconomic rows removed / at the person mean / shuffled together; gaps also within thirds of socioeconomic change and six events (wealth or income fall, left work, Medicaid, poverty, home loss); `ses_order` profile, all outcomes; strata also on D38 | `method` |
| D43 | D42 run on changers only: the sample cut to people with a socioeconomic shock (wealth or income fall, Medicaid, poverty, home loss) before the folds; every arm fitted on them; `comparisons.population` | `method` |
| D44 | D41's BMI arms on BMI changers only (a gain or loss of 3+ kg/m² between waves); strata by BMI change; diabetes, hypertension; `bmi_changers` profile | `method` |

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
| unit tests passing (all suites) | 84 / 84 | D08, D09, D18, D22–D44 | 2026-11-02 |
| true-hazard ceiling, `critical_period`, AUC at bin 20 / C-index / interval NLL | 0.710 / 0.659 / 0.110 | D08 | 2026-10-14 |
| cohort + sex alone (true coefficients), `critical_period`, AUC at bin 20 | 0.599 | D08 | 2026-10-14 |
| CNN `study` shard time on Apple GPU, N = 2k / 5k / 10k / 20k (fold 0, upper bound) | 49 / 124 / 211 / 498 s | D09 | 2026-10-14 |
| CNN `study` profile, full design, run serially | ~147 h for 2,400 shards | D09 | 2026-10-14 |
| observed outcomes, `critical_period` | 7,339 event-free or censored · 1,085 disease · 1,576 death | D08 | 2026-10-14 |
| scenarios implemented | 6 / 6 | — | 2026-08-29 |
| `null` ground-truth support | exactly zero, `12 × 20` | D04 | 2026-08-29 |
| HRS local pull size | 4.8 GB, 18 wave directories + RAND longitudinal file | D07 | 2026-08-29 |
| declared sources | 2 — `simulation` (`open`), `hrs` (`blind`, provisional) | D07 | 2026-08-29 |
| HRS dictionary | 12 / 12 channels mapped; real schema matches it (341 / 341 columns) | D15 | 2026-10-25 |
| HRS RAND file | 45,234 people, 19,880 columns; interview end years up to 2023 | D22 | 2026-10-25 |
| HRS analysis 04 **extended** (30 rows, 11 statics), ages 74–76, AUC: static / summary logistic / boosting / CNN / RNN | diabetes 0.646 / 0.738 / 0.739 / 0.748 / 0.750 · hypertension 0.634 / 0.668 / 0.679 / 0.686 / 0.691 · heart 0.596 / 0.644 / 0.620 / 0.643 / 0.646 · 10-y death 0.637 / 0.724 / 0.726 / 0.723 / 0.733 | D26, D27 | 2026-10-26 |
| HRS analysis 04 with benchmarks (D29), extended set, ages 74–76, AUC — best benchmark vs CNN / RNN | diabetes LightGBM-summary 0.748 vs 0.748 / 0.750 · hypertension elastic net 0.688 vs 0.686 / 0.691 · heart summary ridge 0.648 vs 0.643 / 0.646 · 10-y death LightGBM-cells 0.736 vs 0.723 / 0.733; LightGBM summary minus snapshot +0.018 / +0.004 / +0.001 / +0.005 | D29 | 2026-10-26 |
| HRS kernel experiment (D30), diabetes 74–76, 15 CNN arms, AUC | all arms 0.745–0.749 (best mix_stacked 0.749, k4x5 0.748); fold-to-fold SD 0.006–0.010; no arm beats the RNN (0.750) on average; label-shuffle Grad-CAM mass high for mix_parallel 0.53, mix_two_view 0.36 | D30 | 2026-10-27 |
| HRS map stability (D31), diabetes 74–76, cell Spearman seed / fold / row-shuffle | EG: cnn 0.75 / 0.66 / 0.67 · mix_stacked 0.77 / 0.69 / 0.73 · k1x5 0.79 / 0.69 / 0.93 · rnn 0.66 / 0.57 / 0.82; Grad-CAM fold 0.25–0.35 and ≈0 vs every EG map and LightGBM SHAP; EG vs LightGBM SHAP 0.38–0.46; EG top-20 cell deletion AUC −0.085 to −0.090 vs random −0.009 to −0.012; BMI 19–22 % of EG mass (SHAP 36 %) | D31 | 2026-10-27 |
| HRS map stability (D31), other outcomes, ages 74–76, EG cell Spearman seed / fold / row-shuffle | hypertension: cnn 0.70 / 0.67 / 0.70 · mix_stacked 0.74 / 0.71 / 0.74 · k1x5 0.78 / 0.69 / 0.93 · rnn 0.68 / 0.60 / 0.80 · heart: cnn 0.79 / 0.66 / 0.75 · mix_stacked 0.83 / 0.81 / 0.77 · k1x5 0.86 / 0.81 / 0.97 · rnn 0.76 / 0.67 / 0.86 · death: cnn 0.83 / 0.77 / 0.79 · mix_stacked 0.88 / 0.81 / 0.84 · k1x5 0.87 / 0.83 / 0.96 · rnn 0.81 / 0.77 / 0.90; Grad-CAM fold 0.13–0.79; EG vs LightGBM SHAP 0.41–0.45 / 0.32–0.38 / 0.53–0.58; top-20 deletion AUC −0.049 to −0.051 / −0.032 to −0.039 / −0.050 to −0.067 vs random −0.007 to −0.024; kernel arms within 0.006 AUC of the 4x5 CNN on every outcome | D31 | 2026-10-31 |
| HRS image scaling (D33), ages 74–76, AUC per-age / per-variable z / per-variable 0–1, paired gap 95% | diabetes cnn 0.748 / 0.747 / 0.745 (−0.001 ±0.005, −0.003 ±0.011) · rnn 0.748 / 0.749 / 0.746 (+0.001 ±0.006, −0.002 ±0.006) · death cnn 0.723 / 0.723 / 0.724 (+0.000 ±0.008, +0.001 ±0.004) · rnn 0.732 / 0.731 / 0.729 (−0.002 ±0.005, −0.003 ±0.010); EG cell Spearman to per-age map: diabetes 0.82–0.85 (z), 0.58–0.66 (0–1) · death 0.93–0.96 (z), 0.88–0.89 (0–1) | D33 | 2026-10-28 |
| HRS image scaling (D33), hypertension and heart, AUC per-age / per-variable z / 0–1, paired gap 95% | hypertension cnn 0.686 / 0.688 / 0.685 (+0.002 ±0.005, −0.000 ±0.007) · rnn 0.690 / 0.690 / 0.687 (+0.000 ±0.004, −0.003 ±0.004) · heart cnn 0.643 / 0.642 / 0.639 (−0.001 ±0.004, −0.004 ±0.004) · rnn 0.648 / 0.644 / 0.645 (−0.004 ±0.006, −0.003 ±0.009); \|EG\| map Spearman to per-age: 0.87–0.97 (z), 0.62–0.82 (0–1) | D33 | 2026-10-31 |
| HRS learning curve (D32), ages 74–76, test AUC at 10 / 25 / 50 / 100% of training households (≈960→9,600 diabetes, ≈670→6,700 death) | diabetes: cnn 0.699 / 0.737 / 0.744 / 0.748 · rnn 0.716 / 0.736 / 0.745 / 0.750 · lightgbm_summary 0.725 / 0.736 / 0.741 / 0.748 · lightgbm_cells 0.716 / 0.733 / 0.740 / 0.747 · death: cnn 0.651 / 0.696 / 0.716 / 0.723 · rnn 0.695 / 0.714 / 0.723 / 0.733 · lightgbm_summary 0.695 / 0.716 / 0.728 / 0.735 · lightgbm_cells 0.687 / 0.711 / 0.724 / 0.736; gap to best benchmark per doubling, all shares / 25–100%: diabetes cnn +0.009 / −0.000 · rnn +0.004 / +0.001 · death cnn +0.011 / +0.004 · rnn −0.001 / −0.000; last-doubling gain 0.004–0.011 for every non-static model | D32 | 2026-10-28 |
| HRS learning curve (D32), hypertension and heart, ages 74–76, test AUC at 10 / 25 / 50 / 100% | hypertension: cnn 0.646 / 0.671 / 0.681 / 0.686 · rnn 0.661 / 0.678 / 0.685 / 0.691 · best benchmark (LightGBM summary) 0.663 / 0.674 / 0.682 / 0.687; network gap flat from 25% (cnn −0.004→−0.001, rnn +0.002→+0.004) · heart: cnn 0.592 / 0.622 / 0.635 / 0.643 · rnn 0.600 / 0.623 / 0.638 / 0.646 · best benchmark (summary logistic) 0.617 / 0.636 / 0.645 / 0.648; the gap narrows with every doubling (cnn −0.026→−0.004, rnn −0.017→−0.002) and the last doubling gains 0.008–0.012 for the networks and LightGBM vs 0.003 for summary logistic | D32 | 2026-10-31 |
| HRS predictability and subgroups (D34), diabetes 74–76 | AUC lightgbm_summary 0.748 · cnn 0.748 · rnn 0.750; Q1 vs Q5 cases: BMI row z −0.50 vs +1.46, wealth +0.21 vs −0.64, observed bins 5.2 vs 9.1; Q1 vs Q5 controls: BMI +0.76 vs −1.07, wealth −0.40 vs +0.47; concordance Spearman between lightgbm, cnn, rnn 0.90–0.94; subgroup gaps to lightgbm_summary within ±0.016 (largest CIs from small groups), 0 of 156 Holm < 0.05, best raw static-risk Q2 cnn +0.011 [+0.002, +0.020]; top-10% disagreement: cnn better for 52% of cases / 48% of controls, rnn 50% / 51%; BMI share of \|EG\| Q1→Q5: cases 0.33→0.37 (cnn), controls 0.29→0.45 | D34 | 2026-10-28 |
| HRS subgroups (D34), other outcomes, ages 74–76 | of 156 network-minus-LightGBM gaps: hypertension 7 raw p < 0.05, 1 Holm (cnn − lightgbm_cells, static-risk Q2, −0.014) · heart 5 raw, 0 Holm · death 32 raw, every one negative (network worse; 27 CNN), 1 Holm (cnn − lightgbm_summary, occupation other, −0.022); concordance agreement lightgbm/cnn/rnn 0.90–0.94 / 0.82–0.87 / 0.79–0.89; top-10% disagreement network better 41–58% | D34 | 2026-10-31 |
| HRS pathways in cases' EG maps (D35), diabetes 74–76, 2,964 cases | stable only at k = 2 for both networks (seed Jaccard 0.85–0.91, bootstrap 0.99, fold cosine ≥ 0.98): CNN 1,847 cases where BMI pushes risk up (BMI row z +0.95, concordance 0.86) vs 1,117 where it pushes down (z −0.44, concordance 0.56; born earlier, 6.3 vs 7.5 ages observed); RNN 1,775 / 1,189, CNN–RNN membership ARI 0.76; k = 3 adds a low-BMI-weight, low-wealth group (CNN 954, RNN 1,028; Medicaid, renting, unemployed; concordance 0.78) that fails seed recurrence (Jaccard 0.63 CNN, 0.71 RNN); k ≥ 4 unstable; silhouette 0.43 / 0.37 at k = 2 | D35 | 2026-10-29 |
| HRS pathways without BMI (D36), diabetes 74–76, 2,964 cases | RNN stable only at k = 2 (seed Jaccard 0.80–0.82, bootstrap 0.97, fold cosine 0.88–0.90): 1,407 cases where wealth, depressive symptoms, drinking and activity push risk up (wealth row z −0.92, depressive +0.56; concordance 0.83) vs 1,557 where they push down (wealth +0.38; concordance 0.67); CNN finds the same split but it fails (seed 0.71, fold cosine 0.73); silhouette 0.13–0.14, variance explained 0.13–0.14; CNN–RNN membership ARI 0.37; no k ≥ 3 stable for either; BMI still 23–30% of every cluster's \|EG\| | D36 | 2026-10-29 |
| HRS pathways in cases' EG maps (D35), other outcomes, ages 74–76 | every outcome, both networks stable only at k = 2 · hypertension (7,536 cases): the sign of BMI, as diabetes — CNN 3,941 BMI up (z +0.77, concordance 0.83) vs 3,595 down (z −0.63, 0.52), RNN 3,802 / 3,734, CNN–RNN ARI 0.74, silhouette 0.41 / 0.36, BMI 25–29% of \|EG\| · heart (3,717): depressive symptoms push risk up for 1,582 (CNN; CES-D z +0.91, wealth −0.51, concordance 0.75) vs down for 2,135 (z −0.42, 0.57), RNN 1,354 / 2,363, ARI 0.57, silhouette 0.17–0.19, no row over 16% of \|EG\|; RNN k = 4 has 3 of 4 clusters stable (ever smoked, never smoked, depressed with poor sleep) · 10-y death (4,090): a current-smoker group, CNN 729 / RNN 759 (smoking row z +2.2, smoking 30% of their \|EG\|, concordance 0.89 / 0.93, LightGBM 0.92 / 0.94) vs the rest, ARI 0.76; the same group recurs at k = 3 and 4 in both networks (fold cosine 1.00), silhouette 0.25 | D35 | 2026-10-31 |
| HRS pathways without BMI (D36), hypertension, ages 74–76, 7,536 cases | as diabetes: RNN stable only at k = 2 and only on the mean fold cosine (seed Jaccard 0.81–0.84, bootstrap 0.96–0.97, fold cosine mean 0.80 / 0.84, weakest 0.66 / 0.68): 3,442 cases where wealth and depressive symptoms push risk up (wealth row z −0.72, CES-D +0.56, Medicaid +0.37; concordance 0.80) vs 4,094 where they push down (wealth +0.44; 0.60); CNN fails at every k (seed Jaccard 0.61–0.68, fold cosine ≤ 0.12); silhouette 0.15–0.16, variance explained 0.15–0.16, CNN–RNN ARI 0.24; BMI still 23–30% of every cluster's \|EG\| (`04_hrs_chronic_cnn_pathways_20261031_213526`) | D36 | 2026-10-31 |
| HRS status ages (D37), AUC best benchmark vs CNN / RNN, and network minus best benchmark (paired 95%), at ages 66–68 (8 bins) / 70–72 (10) / 78–80 (14) | diabetes LightGBM-summary 0.762 vs 0.757 / 0.761 · 0.756 vs 0.754 / 0.760 · 0.732 vs 0.729 / 0.729 · hypertension 0.703 vs 0.700 / 0.704 · 0.695 vs 0.693 / 0.695 · 0.682 (elastic net) vs 0.678 / 0.681 · heart 0.667 (window ridge) vs 0.665 / 0.668 · 0.652 vs 0.647 / 0.652 · 0.638 (summary logistic) vs 0.623 / 0.636 · 10-y death 0.759 vs 0.752 / 0.750 · 0.744 (LightGBM-cells) vs 0.735 / 0.740 · 0.732 vs 0.715 / 0.727; of 24 gaps, 19 within ±0.006, every interval excluding zero is negative and on death (CNN −0.007 ±0.004, −0.009 ±0.007, −0.017 ±0.004; RNN −0.009 ±0.005, −0.005 ±0.002); no gap grows with image length; kept / cases: 14,424 / 3,466 diabetes at 66–68 → 10,364 / 2,532 at 78–80, death 9,190 → 7,189; LightGBM summary minus snapshot, diabetes +0.019 to +0.021 at every age (`04_hrs_chronic_cnn_run_20261101_091739`, `_112252`, `_134238`) | D37 | 2026-11-01 |
| HRS status ages (D37), top rows of cases' \|EG\| (CNN / RNN), ages 66–68 / 70–72 / 78–80 | diabetes and hypertension: BMI first at every age, 0.17–0.21 of \|EG\|, then vigorous activity (0.05–0.11) · heart: depressive symptoms first at every age (0.09–0.12) · 10-y death: smoking and working at 66–68, vigorous activity first from 70–72 (0.09–0.12), with smoking or ever smoked next | D37 | 2026-11-01 |
| HRS BMI trajectory vs level (D41), ages 74–76, AUC full row / removed / latest wave only / person mean / shuffled in time | diabetes: cnn 0.748 / 0.698 / 0.733 / 0.746 / 0.748 · rnn 0.750 / 0.700 / 0.729 / 0.748 / 0.747 · lightgbm_summary 0.748 / 0.701 / 0.734 / 0.747 / 0.748 · lightgbm_cells 0.747 / 0.697 / 0.733 / 0.746 / 0.744 · hypertension: cnn 0.686 / 0.657 / 0.681 / 0.686 / 0.686 · rnn 0.691 / 0.663 / 0.684 / 0.691 / 0.690 · lightgbm_summary 0.687 / 0.662 / 0.686 / 0.687 / 0.686 · lightgbm_cells 0.687 / 0.659 / 0.683 / 0.685 / 0.686; paired gap to the full row: removed −0.047 to −0.050 (diabetes), −0.025 to −0.029 (hypertension), every p ≤ 0.002; latest wave −0.014 to −0.020 (diabetes, every p < 0.03), −0.001 to −0.007 (hypertension); person mean and shuffled within ±0.003 AUC for every model and outcome, every p ≥ 0.05 (two at 0.050–0.051, both ≤ 0.003); the latest wave recovers 59–71% of BMI's value for diabetes, 76–95% for hypertension, the mean all of it (`04_hrs_chronic_cnn_comparisons_20261101_231106`) | D41 | 2026-11-02 |
| HRS time order (D38), ages 74–76, AUC gap to the original fit, fixed bin order / per-person shuffle (paired 95%) | CNN: diabetes −0.001 / −0.001 · hypertension +0.000 / −0.002 · heart +0.000 / −0.004 · death −0.002 / −0.007, none clear of zero · RNN: −0.002 / −0.005 · −0.004 / −0.000 · −0.006 / −0.005 · −0.007 ±0.004 / −0.007 ±0.007, clear only for death · per-person shuffle, LightGBM summary / cells: −0.002 / −0.006 ±0.006 · +0.001 / −0.002 · −0.002 / −0.005 · −0.004 ±0.003 / −0.012 ±0.008; timing is worth at most ~0.01 AUC and only for death, and LightGBM on cells loses as much as the networks; the CNN does not use bin adjacency (`04_hrs_chronic_cnn_comparisons_20261102_214320`, run before D42, so no strata) | D38 | 2026-11-03 |
| HRS inputs (D39), ages 74–76, AUC image only minus both / both minus statics-only LightGBM | diabetes −0.006 to −0.008 / +0.098 to +0.104 · hypertension −0.005 to −0.008 / +0.052 to +0.056 · heart −0.041 to −0.048 / +0.043 to +0.050 · death −0.009 to −0.018 / +0.057 to +0.070 (ranges over CNN, RNN, summary logistic, LightGBM summary and cells; every image-only loss clear of zero except the RNN on hypertension); the networks lose the statics no less than the benchmarks; statics-only LightGBM = static logistic except death (+0.029) | D39 | 2026-11-03 |
| HRS stacking (D40), ages 74–76, AUC gap: all five stacked minus LightGBM summary / minus the three benchmarks stacked; networks stacked minus benchmarks stacked | diabetes +0.006 ±0.003 / +0.002 ±0.002; −0.000 · hypertension +0.005 ±0.003 / +0.001 ±0.003; +0.000 · heart +0.008 ±0.006 / +0.002 ±0.003; −0.000 · death +0.006 ±0.003 / +0.001 ±0.001; −0.004; stacking gains ~0.005–0.008 over the best single model, nearly all from stacking the benchmarks; adding the networks adds +0.001–0.002 (none clear of zero; diabetes p = 0.054); networks carry 0.24–0.42 of the five-model stack's total weight (least for death) | D40 | 2026-11-03 |
| HRS BMI trajectory among BMI changers (D44), ages 74–76, 5,499 of 12,011 with a 3+ kg/m² gain or loss (diabetes 1,819 cases, hypertension 4,007), AUC full row / removed / latest wave / person mean / shuffled | diabetes: cnn 0.714 / 0.660 / 0.667 / 0.715 / 0.715 · rnn 0.724 / 0.668 / 0.691 / 0.721 / 0.721 · lightgbm_summary 0.722 / 0.662 / 0.695 / 0.720 / 0.721 · lightgbm_cells 0.714 / 0.659 / 0.685 / 0.711 / 0.711 · hypertension: cnn 0.645 / 0.609 / 0.624 / 0.644 / 0.639 · rnn 0.654 / 0.616 / 0.639 / 0.650 / 0.650 · lightgbm_summary 0.652 / 0.611 / 0.640 / 0.649 / 0.648 · lightgbm_cells 0.650 / 0.617 / 0.643 / 0.652 / 0.650; the latest wave loses more than in the full sample (diabetes −0.027 to −0.047, recovering 13–56% of BMI's value vs 59–71%); person mean and shuffled within −0.006 to +0.002, none clear of zero; within changers, no gap differs between top and bottom thirds of BMI SD or between gainers, losers and the rest (5 of 144 contrasts p < 0.05, about 7 expected); the most variable third is the least predictable for diabetes (RNN 0.735 → 0.689) (`04_hrs_chronic_cnn_comparisons_20261102_232523`) | D44 | 2026-11-03 |
| HRS socioeconomic order among people with a socioeconomic shock (D42–D43), ages 74–76, 7,132 of 12,011 (death 4,043 of 8,359), AUC gap to the original: seven socioeconomic rows removed / at the person mean / shuffled together | every gap within −0.004 to +0.003 for CNN, RNN, LightGBM summary and cells on all four outcomes; 47 of 48 intervals include zero (the exception: LightGBM cells on hypertension, shuffled +0.001); removing the rows costs nothing measurable (−0.004 to +0.003), so their information is redundant with the statics (education, parents' education, occupation) and the other 23 rows, and the order test has nothing to act on; within the shock group, gaps do not differ between top and bottom thirds of change or between each event and the rest: 19 of 448 contrasts p < 0.05 (about 22 expected), no consistent sign; shock groups: wealth fall 4,289, income fall 5,433, Medicaid entry 1,202, poverty entry 1,842, home loss 1,342 (diabetes sample) (`04_hrs_chronic_cnn_comparisons_20261103_155826`) | D42, D43 | 2026-11-03 |
| HRS analysis 04 sample, ages 74–76 | 12,011 kept per condition (diabetes 2,964, hypertension 7,536, heart 3,717 cases); death 8,359 kept, 4,090 died | D23, D25 | 2026-10-25 |
| HRS analysis 04, ages 74–76, AUC (5-fold mean): static / summary logistic / boosting / CNN / RNN | diabetes 0.604 / 0.732 / 0.738 / 0.747 / 0.746 · hypertension 0.609 / 0.661 / 0.668 / 0.683 / 0.683 · heart 0.573 / 0.612 / 0.591 / 0.611 / 0.613 · 10-y death 0.609 / 0.711 / 0.715 / 0.719 / 0.724 | D23–D25 | 2026-10-25 |
| HRS status bin 13 (ages 74–76), cases / kept | hypertension 7,536 / 12,011 · heart 3,717 · diabetes 2,964 · 10-y death 4,090 / 8,359 | D23, D25 | 2026-10-25 |
| oracle AUC at bin 20 vs cohort + sex, range over non-null scenarios (5 folds) | 0.618–0.649 vs 0.591–0.598 | D14 | 2026-10-25 |
| oracle label-shuffled similarity / attribution-mass ratio, all scenarios | 0.96–1.00 / 0.04–0.12 | D14 | 2026-10-25 |
| oracle `spearman_cells`, `critical_period`, every N (death term removed) | 0.488 (0.999) | D14 | 2026-10-25 |

Source: [`output/tables/dgp_manifest_20260829.json`](../output/tables/dgp_manifest_20260829.json).
A result that contradicts a row here is reporting a bug until proven otherwise.

## 3. Open questions

1. **What sensitivity tier does HRS actually carry?** D07 sets `blind` provisionally.
   Settled by locating the clause of the HRS/RAND data use agreement that governs
   transfer to a third party, and recording it in [`doc/data/hrs.md`](data/hrs.md). Until
   then no HRS row is read into an assistant's context. The codebook names where the
   clause is: the HRS "Conditions of Use" agreed to at registration. The synthetic
   fixture the tier requires now exists (`tests/fixtures/rand_hrs_fixture.py`).
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
5. **Does the CNN learn anything beyond the static covariates?** Little can be learnt:
   the oracle bounds the gain at 0.02–0.05 AUC, and the CNN reaches it (D14). The open
   question is now whether recovery is testable at this signal strength at all.
6. *(closed 2026-10-25)* The HRS CPI-U table matches the BLS API for 1991–2023; 2024 added (D19).
7. **Are K2 and K4 fixed before any attribution claim?** D14 shows both fire on a correct
   model. Settled by adopting or rejecting D14's amendments; needed before attribution is
   read on HRS as much as on the simulation.
8. **Are outer folds model-independent?** No: the CNN job seeds them by model name (D13).
   Settled before a second model is fitted to the same data.
9. **Which condition and age for the HRS task?** Settled by the user's `counts` report
   (`04_hrs_chronic_cnn.py counts`): pick status bins with enough cases and controls.
   The config defaults to diabetes, hypertension, heart disease and 10-year death at ages 74–76.
10. *(closed 2026-10-25)* The schema report (`01_hrs_schema_report_20261025_163735`)
   matches the dictionary: all 341 columns present, lowercase; wave-8 life satisfaction
   tops out at 6 (D20); chronic stress runs 0–8. The real file carries item codes the
   fixture lacks (`.m` `.p` `.d` `.r` `.j`); ingestion treats every code as missing.
11. *(closed 2026-10-26)* The extended set adds mostly through the static covariates (+0.02–0.04 AUC for the
   static model); the image's added rows add ≤ 0.01 except for heart (+0.03, with CES-D the top row). Ledger.
12. **Do the networks beat tuned benchmarks?** No, as of the 2026-10-26 run: every network-minus-best-benchmark gap is
   within ±0.005 AUC except the CNN on death (−0.013). Ledger. Open: what the paper claims instead.
13. *(closed 2026-10-27)* Kernel size does not matter for diabetes: the 15 arms span 0.004 AUC, less than one arm's
   fold-to-fold SD. Ledger.
14. *(closed 2026-10-28)* A larger sample does not favour the networks. From 25% to 100% of the training data the
   network-minus-best-benchmark gap is flat (slope −0.001 to +0.004 per doubling); the positive whole-range slopes
   come only from the CNN's unstable 10% point. Every model still gains 0.004–0.011 AUC over the last doubling
   (`04_hrs_chronic_cnn_learning_curve_20261028_144701`). Ledger.
15. *(closed 2026-10-28)* Per-variable scaling does not change AUC: every gap to per-age scaling is within
   ±0.004, inside the paired 95% interval (`04_hrs_chronic_cnn_run_20261027_210545`). Ledger.
16. *(closed 2026-10-28)* Diabetes: the hardest people are those who look like the other class. Hardest cases are
   lean, wealthier and active; hardest controls are heavy, poorer and active less; every model ranks the same
   people well or badly (concordance Spearman 0.86–0.94). No subgroup favours the CNN or RNN over LightGBM:
   4 of 156 gaps have raw p < 0.05 (about 8 expected by chance), none survives Holm. Where they disagree most,
   each network ranks better for 48–52% (`04_hrs_chronic_cnn_subgroups_20261028_175408`). Ledger.
17. *(closed 2026-10-29)* Diabetes cases' EG maps hold one stable split, and it is the sign of BMI: for 62% of
   cases BMI pushes risk up (well predicted), for 38% it pushes down (lean cases, mostly missed). This restates D34's
   predictability result. A third, economic group (low wealth, Medicaid, renting) appears at k = 3 but does not survive
   seed refits (`04_hrs_chronic_cnn_pathways_20261028_234108`). Ledger.
18. *(closed 2026-10-29)* Without BMI, the maps' only stable split (RNN only, just over the thresholds) is a
   socioeconomic gradient: for about half the cases low wealth, depressive symptoms and inactivity push risk up, for
   the rest they push down. Silhouette 0.13: a continuum cut in two, not separate groups. Distinct small groups (heavy
   drinkers, depressed early leavers) appear at k = 5 but fail the seed check
   (`04_hrs_chronic_cnn_pathways_20261029_011044`). Ledger.
19. **Do the diabetes follow-ups hold for hypertension, heart disease and 10-year death?** *A done 2026-10-31; B1–B4 (D37–D40) and D41–D44 read 2026-11-01 to 11-03 (ledger). B5, B6 not built.*
   **A (built 2026-10-30, for the user to run: `make -k hrs-outcomes`; all five reports and the hypertension no-BMI pathways read 2026-10-31, ledger):** the D30–D35 diabetes results hold on every outcome: networks tie with the best benchmark at full size (on heart they close a gap to summary logistic as data grows), scaling and kernel size do not matter, maps are stable and faithful, no subgroup favours a network. Only death adds a distinct pathway (current smokers). stability and faithfulness (D31, which also
   re-checks D30's mix_stacked and k1x5 arms), subgroups (D34) and pathways (D35) for hypertension, heart and death;
   learning curve (D32) and scaling (D33) for hypertension and heart. ~25 h in all, scaled from the diabetes runs.
   Tested on the fixture for every outcome. A D36-style re-run, leaving out each outcome's dominant row, is chosen
   once its D35 report is read: run for hypertension only (heart has no dominant row; dropping smoking would remove death's one group).
   **B1–B4 (built 2026-10-31, `make -k hrs-b`, ~29 h; tested on the fixture):** B1 is D37 (`make hrs-ages`; *read 2026-11-01, ledger: the tie holds at ages 66–68, 70–72 and 78–80 and does not move with image length; the networks fall short only on death*), B2–B4 are
   D38–D40 (*read 2026-11-03, ledger: timing worth ≤ 0.01 AUC and only for death; the networks need the statics as much as the benchmarks; stacking gains ~0.006 over the best model, nearly all from the benchmarks*), one `comparisons` report that fits the originals once per fold (`make hrs-comparisons`). Added the same day:
   D41, BMI trajectory vs level (`make hrs-bmi-trajectory`, ~5 h, also in `hrs-b`). *D41 read 2026-11-02, ledger: BMI's value is its level over the waves, not its order or shape; one wave loses a third of it for diabetes, the person mean none, and shuffling the waves in time costs nothing, for networks and LightGBM alike.* Added 2026-11-02: D42–D43, whether socioeconomic order matters for people who went through a socioeconomic shock, fitted on them alone (`make hrs-ses-order`, under ~9 h); and D44, D41's BMI arms on people whose BMI rose or fell by 3+ kg/m² (`make hrs-bmi-changers`, under ~5 h); its strata also apply to D38's time shuffles in `make hrs-comparisons`. Settled by those
   reports, read into the ledger. B5 and B6 are not built.
   **B (as listed):** (B1) the main run at other status bins, e.g. ages 66–68, 70–72, 78–80 (config only);
   (B2) a time-order shuffle within each person, the column counterpart of D31's row shuffle; (B3) networks fed the
   image only, the statics only, and both; (B4) a stacked CNN + RNN + LightGBM ensemble from out-of-fold predictions;
   (B5) EG row shares compared across outcomes; (B6) calibration and decision curves. B2–B6 need new code.

## 4. Possible extensions

*Not planned work: directions for the manuscript's discussion, recorded 2026-10-31. Each
is an argument for the life-course grid as a representation, not for convolution: a GRU or
a masked transformer over the same grid would serve as well.*

1. **More people, longer lives: other cohorts.** HRS is sparse two ways: in N (about 12,000
   per task, and every model still gains 0.004–0.011 AUC over the last doubling, D32) and in
   span (only from age 50, 12 two-year bins). Candidates: the 1970 British Cohort Study
   (BCS70; from birth, sweeps at irregular, widening intervals), NCDS, and the harmonised
   ELSA / SHARE family (Gateway to Global Aging). A pooled grid would test whether the
   networks pull ahead at a larger N, which the HRS learning curve cannot answer (question
   14). A long, irregular span is exactly what the mask and Δ planes are built for.
2. **A foundation model for missing cells.** Pretrain a model on the grid by hiding observed
   cells and predicting them (masked reconstruction), then use it to fill in missing cells or
   generate trajectories, and fine-tune it per outcome. What it would need to show: (a)
   held-out cell reconstruction better than per-person mean, last value carried forward and
   MICE; (b) a downstream gain, a high bar, since the networks already read mask and Δ and
   LightGBM handles NaN itself (D29). Limits: cells outside a person's observed window, or
   before age 50, are extrapolated, not recovered. Synthetic records generated from HRS may
   count as a transfer under the data use agreement (question 1, D22).
3. **A portable pipeline.** The grid builder, models, EG maps and run-and-report protocol
   do not depend on the dataset, only on a variable dictionary and an age clock. To claim
   portability, show it: the same pipeline on a second cohort, and, in its strongest form,
   pretraining on one cohort and fine-tuning on another. Harmonising variables, not the
   models, is the cost.
4. **Learning curve, sharpened.** Fit a power law per model to the D32 curves and project
   the network-minus-benchmark gap with an interval. Repeat the curve within strata of
   observed bins (sparse vs dense people): if the networks gain faster among the sparse,
   that supports the missingness argument directly.
5. **Incident cases.** D23 keeps prevalent cases and values recorded after diagnosis.
   Restricting to incident cases, or cutting the image at the first report, would separate
   risk factors from the disease reshaping them. That matters for any trajectory reading
   (D41's caveat).
