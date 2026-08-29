# Source: simulated DGP output

*Last updated: 2026-08-29 (rev. 1).*

**Sensitivity:** `open` — synthetic throughout. No real person contributes to any value
here, so there is no licence, no data use agreement, and no restriction on reading rows
into context. This is the only source in the project that carries that tier; contrast
[`hrs.md`](hrs.md), which is `blind`.

*Supersedes `data/README.md`, which was removed in the 2026-08-29 reorganisation (D06);
the dictionary below is that file's, with paths corrected.*

---

## Provenance

Generated in memory by
[`utils.shared_dgp.generate_dataset`](../../src/utils/shared_dgp.py) from
[`shared_dgp_config.yaml`](../../src/utils/shared_dgp_config.yaml). **No person-level
dataset is committed or written to disk**, because each is fully reproducible from
`(master_seed, scenario, N, replicate)` — see
[`01_dgp_diagnostics_methodology.md` §7](../01_dgp_diagnostics_methodology.md).
`data/analysis/dgp_diagnostics/` is registered as `P.DGP_DIAGNOSTICS_DATA_DIR` but is
currently unwritten; re-deriving from the seed is cheap (~1.4 s at `N = 10,000`).

What *is* persisted is the run manifest, `output/tables/dgp_manifest_YYYYMMDD.json`,
carrying the git hash, the master seed, the full config, and per-scenario realized
calibration figures.

## Dictionary

One `DGPResult` per `(scenario, N, replicate)`. `N` individuals, `C = 12` channels,
`T = 20` bins.

### Observable — what a model is allowed to see

| field | shape | meaning |
|---|---|---|
| `X` | `N × 12 × 20` | observed channel values; masked cells are `NaN`, never a sentinel |
| `M` | `N × 12 × 20` | boolean observation mask; `X[~M]` is `NaN` by assertion |
| `Delta` | `N × 12 × 20` | bins since that channel was last observed, per cell |
| `S` | `N × 3` | standardised static covariates: birth cohort, sex, baseline frailty |
| `event_time` | `N` | bin of first disease or death, **or 21 if event-free through `T`** |
| `event_type` | `N` | `0` event-free, `1` disease, `2` death |

### Latent truth — for evaluation only, never a model input

| field | shape | meaning |
|---|---|---|
| `w_true` | `12 × 20` | normalised structural attribution target; sums to 1, or exactly 0 for `null` |
| `latent` | `N × 3 × 20` | the three latent streams U, M, D before loadings and noise |
| `random_intercepts` | `N × 3` | per-individual stream intercepts |
| `loadings` | `12` | `lambda_c ~ U(0.6, 1.0)`, how each channel loads on its stream |
| `dropout_time` | `N` | bin from which covariates are masked, or 21 if never |
| `disease_probability` | `N × 20` | calibrated per-bin cause-specific probability |
| `death_probability` | `N × 20` | calibrated per-bin cause-specific probability |
| `metadata` | dict | scenario, N, replicate, master seed, derived seed, three calibrated intercepts, expected and realized CIFs, missing fraction |

## Traps

- **`event_time = 21` is not bin 21.** It is the event-free sentinel, `T + 1`. Anything
  that treats it as a real time will invent a 21st bin.
- **`dropout_time` and `event_time` are different clocks.** Dropout masks *covariates*
  from that bin onward; the outcome remains observed. An individual can drop out at bin 8
  and still have a recorded disease at bin 15.
- **Channel index carries stream membership.** 0–3 upstream, 4–7 midstream, 8–11
  downstream. Within a stream the order is arbitrary *by construction* — that
  exchangeability is what makes the specification's permutation test valid, so never
  reorder channels across stream boundaries.
- **`latent` and `w_true` are ground truth.** Passing either into a model silently
  invalidates every recovery result the project exists to produce.
- **`w_true` is population-level.** It is not a per-individual gradient; under
  chain of risk individual gradients differ in sign (see D04).
