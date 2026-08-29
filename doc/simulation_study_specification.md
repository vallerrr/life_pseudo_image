# Simulation Study Specification — Recovery Properties of Hierarchical Temporal Models for Life-Course SDOH

> **How to use this file.** Paste the whole thing as the opening prompt to a coding agent with a Python
> environment. It is written as a specification, not a conversation. Push back on anything that is
> underspecified rather than guessing.

---

## 0. Role and objective

You are a research engineer building a simulation study for a methods paper in life-course
epidemiology. The paper's claim is:

> A hierarchical, mask-aware temporal convolutional network over social-determinant-of-health (SDOH)
> channels recovers life-course exposure structures — critical periods, chains of risk, and mobility
> effects — that cumulative-exposure regression cannot represent, and does so with attribution maps
> that are stable to arbitrary within-stream variable ordering.

The simulation exists to establish **ground truth for the interpretability claim**, which observational
data can never provide. Predictive performance is secondary; *recovery* is the primary endpoint.

**This study must be capable of falsifying the claim.** Pre-specify kill criteria (Section 8) before
running anything, and report them honestly whichever way they come out.

---

## 1. Data-generating process

### 1.1 Structure

- `N` individuals, `T = 20` discrete age-bins (interpret as ages 50–90 in 2-year bins).
- `C = 12` observed channels in three streams of 4:
  - **Upstream** (structural/material): channels 0–3
  - **Midstream** (psychosocial/relational): channels 4–7
  - **Downstream** (behavioural/biological): channels 8–11
- `P = 3` static covariates: birth cohort, sex, a baseline frailty term.

### 1.2 Latent trajectories

Generate one latent process per stream, then load observed channels onto it. **This is deliberate:
within-stream channel order must be exchangeable by construction, so the permutation test in Section 6
is a valid test of the architecture rather than of the DGP.**

```
U_t = rho_u * U_{t-1} + xi_i^u + eps_t          # upstream, AR(1) + individual intercept
M_t = a * U_{t-1} + rho_m * M_{t-1} + xi_i^m + eps_t
D_t = b * M_{t-1} + c * U_{t-1} + rho_d * D_{t-1} + xi_i^d + eps_t
```

- `rho_u = 0.9` (SES is persistent), `rho_m = 0.7`, `rho_d = 0.8`
- `a = 0.4`, `b = 0.4`, `c = 0.2` (partial mediation, with a direct upstream path)
- `xi_i ~ N(0, 0.5)` individual random intercepts; `eps_t ~ N(0, 1)`
- Observed: `X[c,t] = lambda_c * L_stream(c)[t] + nu_{c,t}`, loadings `lambda_c ~ U(0.6, 1.0)`,
  measurement noise `nu ~ N(0, 0.5)`

Make `a`, `b`, `c`, and all `rho` configurable — a sensitivity sweep over mediation strength is a
secondary analysis.

### 1.3 Hazard models (the scenarios)

Discrete-time hazard `logit h_t = beta_0 + f(history up to t) + gamma' S`. Calibrate `beta_0` per
scenario by bisection to a target cumulative incidence of **15% by T** — do not hand-tune it, and
verify the achieved incidence in the output.

Implement all six. Scenarios 1, 2, 4, 6 are the minimum viable set if compute is constrained.

| # | Scenario | `f(history)` | What it tests |
|---|---|---|---|
| 1 | **Accumulation** | `beta * mean(U_s : s <= t)` | Baseline case; regression should win or tie |
| 2 | **Critical period** | `beta * mean(U_s : s in [5,8])`, zero elsewhere | Sharp window localization |
| 3 | **Sensitive period** | `beta * sum_s w(s) U_s`, `w` Gaussian kernel peaked at `s*=6`, sd 2 | Temporal resolution; can the model tell 3 from 2? |
| 4 | **Chain of risk** | `beta * mean(U_{s<=6}) * mean(D_{s in [10,14]})` (multiplicative, no main effects) | Sequence dependence; additive models are blind by construction |
| 5 | **Mobility** | `beta * slope(U_{1..T})` with mean(U) orthogonalized out | Trajectory shape at matched level |
| 6 | **Null** | `gamma' S` only | False-positive control — attribution maps must be flat |

Add a **competing risk of death** in every scenario: `logit h_t^death = alpha_0 + alpha_1 * frailty +
alpha_2 * mean(D_s : s<=t)`, calibrated to ~20% cumulative mortality. Death must be
outcome-correlated so that naive complete-case analysis is visibly biased.

### 1.4 Missingness — mirror HRS, do not simplify

Three mechanisms, applied in this order:

1. **Item nonresponse**: MCAR, 10% per cell.
2. **Rotating half-sample**: midstream channels (4–7) observed only in even-numbered bins, and only
   for a random 50% of individuals per bin — reproducing the HRS Leave-Behind design.
3. **Informative attrition**: per-bin dropout hazard increasing in the disease hazard, tuned to ~30%
   attrition by `T`. Once dropped, all channels missing thereafter.

Emit `X`, `M` (mask), `Delta` (bins since last observation), the static matrix `S`, event times,
event type, **and the ground-truth weight profile `w_true[c,t]`** used in the hazard.

### 1.5 Design grid

- `N in {2000, 5000, 10000, 20000}`
- 6 scenarios
- **50 replicates** per cell (drop to 20 if compute-bound; state which you used)
- Fixed, logged seeds; every dataset regenerable from `(scenario, N, replicate)`

---

## 2. Models to fit

**Proposed model** — hierarchical mask-aware TCN:
- Depthwise conv (`groups=C`) over `[X; M; Delta]`
- 4 gated residual TCN blocks, `groups=3` (stream-wise), kernel 3, dilations 1/2/4/8, **causal
  (left-padded) convolutions only**
- Per-stream attention pooling over time
- Mediation-structured fusion head (upstream predicts mid/down; residuals enter the head)
- Discrete-time competing-risks head; per-interval multinomial NLL

**Ablations** (each isolates one design decision):
- (a) `groups=1` — removes the hierarchy
- (b) mean pooling instead of attention — removes timing
- (c) concat head instead of mediation fusion
- (d) no mask/Delta channels, median imputation instead
- (e) non-causal convolutions — quantifies the leakage you are avoiding

**Baselines:**
- Penalized Cox on cumulative/mean exposures
- Cox on three pre-specified age-window means (the "manual life-course" model)
- XGBoost on flattened `X` plus engineered summaries (mean, slope, min, max, last per channel) —
  **tune this properly; it is the honest competitor and a weak version invalidates the paper**
- GRU-D
- Transformer with sinusoidal time encodings
- **2D pseudo-image CNN** on the `C x T` array with 4x5 kernels — the original proposal, as the
  headline comparator

Nested CV: 5 outer folds for performance, 3 inner for hyperparameters. Never report inner-tuned
performance on inner folds.

---

## 3. Primary endpoints — recovery

For each fitted model, compute attribution `A[c,t]` via **Expected Gradients** (Integrated Gradients
averaged over ~50 training-set baselines; do *not* use a zero baseline). Compare to `w_true[c,t]`:

1. **Profile correlation** — Spearman between `|A|` and `|w_true|`, over all `(c,t)`, and over `t`
   after summing within stream.
2. **Peak-time error** — `|argmax_t A_stream - argmax_t w_true_stream|` in bins.
3. **Window detection AUPRC** — treat `w_true != 0` as the positive class, rank by `|A|`.
4. **Stream attribution share** — estimated vs. true share of total weight per stream; report as
   ternary plot.
5. **Null-scenario flatness** — coefficient of variation of `A` over `(c,t)` under scenario 6. Any
   model producing structured maps on null data is disqualified from interpretive claims.
6. **Chain detection** (scenario 4 only) — fit a probe comparing model predictions under factorized
   vs. joint perturbation of early-upstream and late-downstream; report whether the interaction is
   detected at the individual level.

Run the **Adebayo sanity checks** (model-parameter randomization, label randomization) on every
attribution method. Report them. A method that fails these cannot support any substantive claim in the
paper.

---

## 4. Secondary endpoints — prediction

Time-dependent AUC, IPCW Brier score, Antolini C-index for competing risks, and **calibration**
(decile plots plus integrated calibration index). Report calibration with equal prominence to
discrimination — a model that discriminates well and calibrates badly is not a win.

---

## 5. Reporting

Per cell, report **median and IQR across replicates**, never a single run. Present scenario x model as
the primary table. Do not average across scenarios — the entire point is that performance is
scenario-dependent, and the headline finding is expected to be that no model dominates everywhere.

---

## 6. Permutation-stability experiment (adjudicates the architecture choice)

For scenarios 2 and 4, at `N = 10000`:

- Draw `K = 20` random within-stream channel permutations.
- Refit **the proposed model** and **the 2D pseudo-image CNN** under each.
- Report, for each: (i) SD of test AUC across permutations, (ii) mean pairwise correlation of
  attribution maps across permutations (after inverting the permutation to a common ordering).

**Pre-registered prediction:** the grouped model is invariant to within-stream permutation up to
numerical/optimization noise; the 2D CNN is not, with materially higher AUC variance and attribution
maps that fail to align. If the 2D CNN turns out to be stable, the central architectural argument of
the paper is wrong and must be retracted rather than reframed.

---

## 7. Engineering requirements

- Python, PyTorch. Structure: `dgp/`, `models/`, `eval/`, `experiments/`, `figures/`.
- DGP fully vectorized in NumPy; a 20k-person dataset should generate in under 5 seconds.
- Config-driven (Hydra or plain YAML); every run writes a config + git hash + seed manifest.
- Results to Parquet, one row per `(scenario, N, replicate, model, fold, metric)`.
- Unit tests on the DGP: verify achieved incidence matches target, verify scenario 6 attributions of
  the true hazard function are exactly flat, verify the mediation coefficients are recoverable by OLS
  on the *latent* processes.
- `make all` reproduces every figure from scratch.
- Log wall-clock per fit; report total compute in the paper.

---

## 8. Kill criteria — state these before running

The idea does **not** work, and the paper should say so, if:

- The proposed model fails to beat XGBoost and penalized Cox on **scenarios 4 and 5** (chain and
  mobility). These are the scenarios where flexible temporal modelling is supposed to be necessary; if
  it isn't needed there, it isn't needed.
- Attribution maps fail the Adebayo sanity checks.
- Scenario 6 produces structured attribution maps (interpretability is spurious).
- Recovery requires `N > 20000`, which no realistic cohort will supply.

---

## 9. First deliverable

Before writing any model code, deliver:

1. The DGP module with unit tests.
2. A one-page diagnostic report per scenario: incidence curves, attrition curves, missingness
   patterns, latent trajectory examples, and the ground-truth weight profile visualized as the
   `C x T` heatmap the paper is ultimately trying to recover.

Then stop and wait for review. If the DGP is wrong, everything downstream is wasted.