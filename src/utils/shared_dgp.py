"""The data-generating process: known life-course truth for six scenarios.

Owns everything about the simulation itself -- the latent stream dynamics, the
competing-risk calibration, the informative attrition, and the ground-truth
attribution map -- so that the analysis entry points only orchestrate and plot.
Structural parameters live beside this module in ``shared_dgp_config.yaml`` and are
never inlined here, because a constant edited in code and a constant recorded in a
manifest are how a report ends up describing a run that never happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils import shared_paths as P
from utils.shared_reproducibility import stable_seed

# The one seed the whole project derives from (conventions §12).
MASTER_SEED = 20260724

SCENARIOS = (
    "accumulation",
    "critical_period",
    "sensitive_period",
    "chain_of_risk",
    "mobility",
    "null",
)

SCENARIO_DISPLAY_NAMES = {
    "accumulation": "Accumulation",
    "critical_period": "Critical period",
    "sensitive_period": "Sensitive period",
    "chain_of_risk": "Chain of risk",
    "mobility": "Mobility",
    "null": "Null",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Resolved scalar DGP configuration."""

    T: int
    C: int
    P: int
    channels_per_stream: int
    rho_u: float
    rho_m: float
    rho_d: float
    a: float
    b: float
    c: float
    intercept_sd: float
    innovation_sd: float
    loading_low: float
    loading_high: float
    measurement_sd: float
    target_disease: float
    target_death: float
    beta: float
    gamma: tuple[float, float, float]
    alpha_frailty: float
    alpha_downstream: float
    item_nonresponse: float
    rotating_fraction: float
    attrition_target: float
    attrition_log_odds_ratio: float


@dataclass
class DGPResult:
    """All observable outputs plus latent truth needed for DGP validation."""

    X: np.ndarray
    M: np.ndarray
    Delta: np.ndarray
    S: np.ndarray
    event_time: np.ndarray
    event_type: np.ndarray
    w_true: np.ndarray
    latent: np.ndarray
    random_intercepts: np.ndarray
    loadings: np.ndarray
    dropout_time: np.ndarray
    disease_probability: np.ndarray
    death_probability: np.ndarray
    metadata: dict[str, Any]


def load_config(path: str | Path | None = None) -> SimulationConfig:
    """Load the plain-YAML configuration into a typed flat dataclass."""
    config_path = Path(path) if path else P.DGP_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return SimulationConfig(
        **raw["structure"],
        **raw["latent"],
        **raw["observation"],
        gamma=tuple(raw["hazard"]["gamma"]),
        **{k: v for k, v in raw["hazard"].items() if k != "gamma"},
        **raw["missingness"],
    )


def _zscore(values: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    sd = values.std(axis=axis, keepdims=True)
    return (values - mean) / np.maximum(sd, 1e-8)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    out = np.empty_like(values, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _generate_latent(
    rng: np.random.Generator, n: int, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray]:
    intercepts = rng.normal(0.0, cfg.intercept_sd, size=(n, 3))
    innovations = rng.normal(0.0, cfg.innovation_sd, size=(n, 3, cfg.T))
    latent = np.empty((n, 3, cfg.T), dtype=np.float64)
    latent[:, :, 0] = intercepts + innovations[:, :, 0]
    for t in range(1, cfg.T):
        latent[:, 0, t] = (
            cfg.rho_u * latent[:, 0, t - 1]
            + intercepts[:, 0]
            + innovations[:, 0, t]
        )
        latent[:, 1, t] = (
            cfg.a * latent[:, 0, t - 1]
            + cfg.rho_m * latent[:, 1, t - 1]
            + intercepts[:, 1]
            + innovations[:, 1, t]
        )
        latent[:, 2, t] = (
            cfg.b * latent[:, 1, t - 1]
            + cfg.c * latent[:, 0, t - 1]
            + cfg.rho_d * latent[:, 2, t - 1]
            + intercepts[:, 2]
            + innovations[:, 2, t]
        )
    return latent, intercepts


def _cumulative_mean(values: np.ndarray) -> np.ndarray:
    denom = np.arange(1, values.shape[1] + 1, dtype=np.float64)
    return np.cumsum(values, axis=1) / denom


def _scenario_score(
    scenario: str, latent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the causal history score and its final-time temporal support."""
    n, _, T = latent.shape
    upstream = latent[:, 0, :]
    downstream = latent[:, 2, :]
    support = np.zeros((3, T), dtype=np.float64)

    if scenario == "accumulation":
        raw = _cumulative_mean(upstream)
        support[0, :] = 1.0 / T
    elif scenario == "critical_period":
        raw = np.zeros((n, T), dtype=np.float64)
        start, stop = 4, 8  # human-readable bins 5-8, inclusive
        for t in range(start, T):
            available_stop = min(t + 1, stop)
            raw[:, t] = upstream[:, start:available_stop].mean(axis=1)
        support[0, start:stop] = 1.0 / (stop - start)
    elif scenario == "sensitive_period":
        times = np.arange(1, T + 1)
        weights = np.exp(-0.5 * ((times - 6.0) / 2.0) ** 2)
        weighted_sum = np.cumsum(upstream * weights[None, :], axis=1)
        raw = weighted_sum / np.cumsum(weights)[None, :]
        support[0, :] = weights / weights.sum()
    elif scenario == "chain_of_risk":
        early_u = upstream[:, :6].mean(axis=1)
        raw = np.zeros((n, T), dtype=np.float64)
        for t in range(9, T):
            available_stop = min(t + 1, 14)
            late_d = downstream[:, 9:available_stop].mean(axis=1)
            raw[:, t] = early_u * late_d
        mean_abs_d = np.mean(np.abs(downstream[:, 9:14].mean(axis=1)))
        mean_abs_u = np.mean(np.abs(early_u))
        support[0, :6] = mean_abs_d / 6.0
        support[2, 9:14] = mean_abs_u / 5.0
    elif scenario == "mobility":
        raw = np.zeros((n, T), dtype=np.float64)
        for t in range(1, T):
            time = np.arange(1, t + 2, dtype=np.float64)
            centered = time - time.mean()
            slope = upstream[:, : t + 1] @ centered / np.sum(centered**2)
            level = upstream[:, : t + 1].mean(axis=1)
            coefficient = np.cov(slope, level, ddof=0)[0, 1] / max(
                np.var(level), 1e-8
            )
            raw[:, t] = slope - coefficient * level
        time = np.arange(1, T + 1, dtype=np.float64)
        centered = time - time.mean()
        slope_weights = centered / np.sum(centered**2)
        final_slope = upstream @ slope_weights
        final_level = upstream.mean(axis=1)
        coefficient = np.cov(final_slope, final_level, ddof=0)[0, 1] / max(
            np.var(final_level), 1e-8
        )
        support[0, :] = np.abs(slope_weights - coefficient / T)
    elif scenario == "null":
        raw = np.zeros((n, T), dtype=np.float64)
    else:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}")

    standardized = np.zeros_like(raw)
    for t in range(T):
        if np.std(raw[:, t]) > 1e-8:
            standardized[:, t] = _zscore(raw[:, t])
    if support.sum() > 0:
        support /= support.sum()
    return standardized, support


def _channel_weight_map(
    stream_support: np.ndarray, loadings: np.ndarray, cfg: SimulationConfig
) -> np.ndarray:
    weights = np.zeros((cfg.C, cfg.T), dtype=np.float64)
    for stream in range(3):
        start = stream * cfg.channels_per_stream
        stop = start + cfg.channels_per_stream
        channel_shares = loadings[start:stop] / loadings[start:stop].sum()
        weights[start:stop, :] = channel_shares[:, None] * stream_support[stream]
    if weights.sum() > 0:
        weights /= weights.sum()
    return weights


def _competing_probabilities(
    disease_lp: np.ndarray,
    death_lp: np.ndarray,
    disease_intercept: float,
    death_intercept: float,
) -> tuple[np.ndarray, np.ndarray]:
    disease_exp = np.exp(np.clip(disease_lp + disease_intercept, -40, 40))
    death_exp = np.exp(np.clip(death_lp + death_intercept, -40, 40))
    denom = 1.0 + disease_exp + death_exp
    return disease_exp / denom, death_exp / denom


def _expected_cif(
    disease_probability: np.ndarray, death_probability: np.ndarray
) -> tuple[float, float]:
    survival = np.ones(disease_probability.shape[0], dtype=np.float64)
    disease_cif = np.zeros_like(survival)
    death_cif = np.zeros_like(survival)
    for t in range(disease_probability.shape[1]):
        disease_cif += survival * disease_probability[:, t]
        death_cif += survival * death_probability[:, t]
        survival *= 1.0 - disease_probability[:, t] - death_probability[:, t]
    return float(disease_cif.mean()), float(death_cif.mean())


def _calibrate_intercepts(
    disease_lp: np.ndarray,
    death_lp: np.ndarray,
    target_disease: float,
    target_death: float,
) -> tuple[float, float, np.ndarray, np.ndarray, tuple[float, float]]:
    disease_intercept, death_intercept = -5.0, -5.0
    for _ in range(8):
        low, high = -15.0, 2.0
        for _ in range(28):
            middle = (low + high) / 2.0
            probabilities = _competing_probabilities(
                disease_lp, death_lp, middle, death_intercept
            )
            incidence, _ = _expected_cif(*probabilities)
            if incidence < target_disease:
                low = middle
            else:
                high = middle
        disease_intercept = (low + high) / 2.0

        low, high = -15.0, 2.0
        for _ in range(28):
            middle = (low + high) / 2.0
            probabilities = _competing_probabilities(
                disease_lp, death_lp, disease_intercept, middle
            )
            _, mortality = _expected_cif(*probabilities)
            if mortality < target_death:
                low = middle
            else:
                high = middle
        death_intercept = (low + high) / 2.0
        current = _competing_probabilities(
            disease_lp, death_lp, disease_intercept, death_intercept
        )
        incidence, mortality = _expected_cif(*current)
        if (
            abs(incidence - target_disease) < 1e-8
            and abs(mortality - target_death) < 1e-8
        ):
            break

    disease_probability, death_probability = _competing_probabilities(
        disease_lp, death_lp, disease_intercept, death_intercept
    )
    expected = _expected_cif(disease_probability, death_probability)
    return (
        disease_intercept,
        death_intercept,
        disease_probability,
        death_probability,
        expected,
    )


def _sample_events(
    rng: np.random.Generator,
    disease_probability: np.ndarray,
    death_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, T = disease_probability.shape
    event_time = np.full(n, T + 1, dtype=np.int16)
    event_type = np.zeros(n, dtype=np.int8)
    at_risk = np.ones(n, dtype=bool)
    uniforms = rng.random((n, T))
    for t in range(T):
        disease = at_risk & (uniforms[:, t] < disease_probability[:, t])
        death = (
            at_risk
            & ~disease
            & (
                uniforms[:, t]
                < disease_probability[:, t] + death_probability[:, t]
            )
        )
        event_time[disease | death] = t + 1
        event_type[disease] = 1
        event_type[death] = 2
        at_risk[disease | death] = False
    return event_time, event_type


def _calibrate_dropout(
    risk_score: np.ndarray,
    event_time: np.ndarray,
    target: float,
    log_odds_ratio: float,
) -> tuple[float, np.ndarray]:
    n, T = risk_score.shape
    eligible = np.arange(1, T + 1)[None, :] < event_time[:, None]

    def expected_dropout(intercept: float) -> float:
        hazard = _sigmoid(intercept + log_odds_ratio * risk_score) * eligible
        never = np.prod(1.0 - hazard, axis=1)
        return float(np.mean(1.0 - never))

    low, high = -15.0, 2.0
    for _ in range(50):
        middle = (low + high) / 2.0
        if expected_dropout(middle) < target:
            low = middle
        else:
            high = middle
    intercept = (low + high) / 2.0
    probability = _sigmoid(intercept + log_odds_ratio * risk_score) * eligible
    return intercept, probability


def _sample_dropout(
    rng: np.random.Generator, probability: np.ndarray
) -> np.ndarray:
    n, T = probability.shape
    dropout_time = np.full(n, T + 1, dtype=np.int16)
    active = np.ones(n, dtype=bool)
    uniforms = rng.random((n, T))
    for t in range(T):
        dropped = active & (uniforms[:, t] < probability[:, t])
        dropout_time[dropped] = t + 1
        active[dropped] = False
    return dropout_time


def _apply_missingness(
    rng: np.random.Generator,
    X_complete: np.ndarray,
    dropout_time: np.ndarray,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, C, T = X_complete.shape
    mask = rng.random((n, C, T)) >= cfg.item_nonresponse

    for t in range(T):
        human_bin = t + 1
        if human_bin % 2 == 1:
            mask[:, 4:8, t] = False
        else:
            selected = rng.random(n) < cfg.rotating_fraction
            mask[:, 4:8, t] &= selected[:, None]

    after_dropout = (
        np.arange(1, T + 1)[None, :] >= dropout_time[:, None]
    )
    mask &= ~after_dropout[:, None, :]
    X = np.where(mask, X_complete, np.nan)

    delta = np.zeros((n, C, T), dtype=np.int16)
    elapsed = np.zeros((n, C), dtype=np.int16)
    for t in range(T):
        observed = mask[:, :, t]
        elapsed = np.where(observed, 0, elapsed + 1)
        delta[:, :, t] = elapsed
    return X, mask, delta


def generate_dataset(
    scenario: str,
    n: int,
    replicate: int = 0,
    config: SimulationConfig | None = None,
    master_seed: int = MASTER_SEED,
) -> DGPResult:
    """Generate one fully reproducible dataset from its identifying tuple.

    ``(master_seed, scenario, n, replicate)`` fully determines the output; nothing
    else in this function reads global state.
    """
    cfg = config or load_config()
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    if n < 2:
        raise ValueError("n must be at least 2")
    seed = stable_seed(master_seed, scenario, n, replicate)
    rng = np.random.default_rng(seed)

    latent, random_intercepts = _generate_latent(rng, n, cfg)
    loadings = rng.uniform(cfg.loading_low, cfg.loading_high, size=cfg.C)
    stream_index = np.repeat(np.arange(3), cfg.channels_per_stream)
    X_complete = (
        latent[:, stream_index, :]
        * loadings[None, :, None]
        + rng.normal(0.0, cfg.measurement_sd, size=(n, cfg.C, cfg.T))
    )

    cohort = rng.normal(size=n)
    sex = rng.binomial(1, 0.5, size=n)
    frailty = rng.normal(size=n)
    S = _zscore(np.column_stack((cohort, sex, frailty)))

    scenario_score, stream_support = _scenario_score(scenario, latent)
    disease_lp = cfg.beta * scenario_score + S @ np.asarray(cfg.gamma)[:, None]
    cumulative_downstream = _zscore(_cumulative_mean(latent[:, 2, :]))
    death_lp = (
        cfg.alpha_frailty * S[:, 2, None]
        + cfg.alpha_downstream * cumulative_downstream
    )
    (
        disease_intercept,
        death_intercept,
        disease_probability,
        death_probability,
        expected_incidence,
    ) = _calibrate_intercepts(
        disease_lp,
        death_lp,
        cfg.target_disease,
        cfg.target_death,
    )
    event_time, event_type = _sample_events(
        rng, disease_probability, death_probability
    )

    attrition_risk = _zscore(disease_lp)
    dropout_intercept, dropout_probability = _calibrate_dropout(
        attrition_risk,
        event_time,
        cfg.attrition_target,
        cfg.attrition_log_odds_ratio,
    )
    dropout_time = _sample_dropout(rng, dropout_probability)
    X, M, Delta = _apply_missingness(rng, X_complete, dropout_time, cfg)
    w_true = _channel_weight_map(stream_support, loadings, cfg)

    realized_incidence = float(np.mean(event_type == 1))
    realized_mortality = float(np.mean(event_type == 2))
    realized_attrition = float(np.mean(dropout_time <= cfg.T))
    metadata: dict[str, Any] = {
        "scenario": scenario,
        "N": n,
        "replicate": replicate,
        "master_seed": master_seed,
        "seed": seed,
        "disease_intercept": disease_intercept,
        "death_intercept": death_intercept,
        "dropout_intercept": dropout_intercept,
        "expected_disease_incidence": expected_incidence[0],
        "expected_mortality": expected_incidence[1],
        "realized_disease_incidence": realized_incidence,
        "realized_mortality": realized_mortality,
        "realized_attrition": realized_attrition,
        "missing_fraction": float(1.0 - M.mean()),
    }
    return DGPResult(
        X=X,
        M=M,
        Delta=Delta,
        S=S,
        event_time=event_time,
        event_type=event_type,
        w_true=w_true,
        latent=latent,
        random_intercepts=random_intercepts,
        loadings=loadings,
        dropout_time=dropout_time,
        disease_probability=disease_probability,
        death_probability=death_probability,
        metadata=metadata,
    )
