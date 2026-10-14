"""Scoring: recovery against the ground truth, and prediction on held-out people.

Every model in the study is scored by these functions and nothing else. Each one
takes plain arrays -- an attribution map, or per-bin hazards -- so a regression, a
boosted tree and a network are compared by identical code, and a metric cannot drift
between two models because each implemented it its own way.

Two conventions hold throughout:

* **Time is discrete.** A person is at risk in bins ``1..obs_time``; ``obs_type``
  says how bin ``obs_time`` ended (``shared_model_input``). Hazards are ``(n, T)``
  arrays, bin ``t`` in column ``t - 1``.
* **Censoring is attrition.** Inverse-probability-of-censoring weights come from a
  Kaplan-Meier estimate of *still being observed*, so metrics that need a known
  status at a horizon are not computed on the people who stayed.

What is *not* here yet: the scenario-4 chain-detection probe (spec §3.6), which needs
a model to perturb rather than a map to score, and therefore lives with the model.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from utils.shared_model_input import CENSORED, DEATH, DISEASE

STREAM_NAMES = ("upstream", "midstream", "downstream")


# --- recovery (spec §3) -----------------------------------------------------
def stream_profile(weights: np.ndarray, n_streams: int = 3) -> np.ndarray:
    """``(C, T)`` -> ``(streams, T)``, summing the channels within each stream."""
    C, T = weights.shape
    return weights.reshape(n_streams, C // n_streams, T).sum(axis=1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")  # undefined against a constant map, e.g. the null truth
    return float(spearmanr(a, b).statistic)


def average_precision(positive: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve, ties broken pessimistically."""
    positive, score = np.ravel(positive).astype(bool), np.ravel(score)
    if positive.all() or not positive.any():
        return float("nan")  # no negative (or no positive) class to rank against
    order = np.lexsort((positive, -score))  # within a tie, negatives rank first
    hits = positive[order]
    precision = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float(precision[hits].mean())


def recovery_metrics(attribution: np.ndarray, w_true: np.ndarray) -> dict[str, float]:
    """Compare a population attribution map ``A[c, t]`` with ``w_true[c, t]``.

    ``attribution`` is the mean absolute per-person attribution, so it is already
    non-negative; ``|w_true|`` is compared as the spec asks. Metrics that the truth
    leaves undefined (any correlation against the flat null map, peak time for a
    stream the truth never uses) come back NaN rather than as a misleading number.
    """
    A, w = np.abs(attribution), np.abs(w_true)
    A_stream, w_stream = stream_profile(A), stream_profile(w)
    metrics: dict[str, float] = {
        "spearman_cells": _spearman(A, w),
        "spearman_stream_time": _spearman(A_stream, w_stream),
        "window_auprc": average_precision(w > 0, A),
        # Spec §3.5: dispersion of the map. Only *read* under the null scenario.
        "flatness_cv": float(A.std() / A.mean()) if A.mean() > 0 else float("nan"),
    }

    total_A, total_w = A.sum(), w.sum()
    share_error = 0.0
    for s, name in enumerate(STREAM_NAMES):
        share = A_stream[s].sum() / total_A if total_A > 0 else float("nan")
        true_share = w_stream[s].sum() / total_w if total_w > 0 else float("nan")
        metrics[f"stream_share_{name}"] = float(share)
        metrics[f"true_stream_share_{name}"] = float(true_share)
        share_error += abs(share - true_share)
        metrics[f"peak_time_error_{name}"] = (
            float(abs(int(A_stream[s].argmax()) - int(w_stream[s].argmax())))
            if w_stream[s].any()
            else float("nan")
        )
    # Total-variation distance between the two stream shares: 0 = identical, 1 = disjoint.
    metrics["stream_share_tv"] = float(share_error / 2.0) if total_w > 0 else float("nan")
    return metrics


def population_map(per_person: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, C, T)`` per-person attributions -> mean |A| and mean A per cell.

    Each cell is averaged over the people who *observed* it, not over everyone. An
    unobserved cell carries no attribution by construction, so a plain mean would
    stamp the missingness pattern (midstream is unobserved in ~75% of person-bins)
    onto every map, trained or not -- and the Adebayo checks would then compare two
    copies of the missingness pattern and call them similar.
    """
    count = observed.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs = np.where(count > 0, np.abs(per_person).sum(axis=0) / count, 0.0)
        mean_signed = np.where(count > 0, per_person.sum(axis=0) / count, 0.0)
    return mean_abs, mean_signed


def map_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman between two attribution maps: the Adebayo checks' similarity."""
    return _spearman(np.abs(a), np.abs(b))


# --- discrete-time competing risks ------------------------------------------
def cumulative_incidence(
    disease_hazard: np.ndarray, death_hazard: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-person CIF of disease and of death at the end of each bin, ``(n, T)``."""
    survival = np.cumprod(1.0 - disease_hazard - death_hazard, axis=1)
    before = np.concatenate([np.ones_like(survival[:, :1]), survival[:, :-1]], axis=1)
    return np.cumsum(before * disease_hazard, axis=1), np.cumsum(before * death_hazard, axis=1)


def still_observed(obs_time: np.ndarray, obs_type: np.ndarray, n_bins: int) -> np.ndarray:
    """Kaplan-Meier ``K[t]`` = P(observation reaches bin t), for ``t = 0..T``.

    A censored person with ``obs_time = c < T`` was seen through bin ``c`` and lost
    before bin ``c + 1``; someone censored at ``T`` simply reached the end. Events
    are not censoring. ``K[0] = 1``.
    """
    K = np.ones(n_bins + 1)
    for t in range(1, n_bins + 1):
        at_risk = obs_time >= t - 1  # entered bin t still event-free and observed...
        at_risk &= ~((obs_time == t - 1) & (obs_type != CENSORED))  # ...and no event in t-1
        lost = at_risk & (obs_time == t - 1) & (obs_type == CENSORED)
        n_at_risk = at_risk.sum()
        K[t] = K[t - 1] * (1.0 - lost.sum() / n_at_risk if n_at_risk else 1.0)
    return K


def _horizon_status(
    obs_time: np.ndarray, obs_type: np.ndarray, K: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disease by ``horizon``? Known at all? And the IPC weight of each person."""
    event_by = (obs_type != CENSORED) & (obs_time <= horizon)
    known_free = obs_time >= horizon  # observed event-free through the horizon
    known_free &= ~event_by
    known = event_by | known_free
    weight = np.zeros(obs_time.shape, dtype=float)
    weight[event_by] = 1.0 / K[obs_time[event_by]]
    weight[known_free] = 1.0 / K[horizon]
    case = event_by & (obs_type == DISEASE)
    return case, known, weight


def disease_auc(
    risk: np.ndarray, obs_time: np.ndarray, obs_type: np.ndarray, K: np.ndarray, horizon: int
) -> float:
    """IPCW cumulative/dynamic AUC for disease by ``horizon``.

    Cases: disease by the horizon. Controls: everyone else whose status at the
    horizon is known -- event-free through it, *or* dead first (Blanche et al.'s
    second definition, under which a competing death is a control).
    """
    case, known, weight = _horizon_status(obs_time, obs_type, K, horizon)
    control = known & ~case
    if not case.any() or not control.any():
        return float("nan")
    # Weighted Mann-Whitney: for each case, weight of controls ranked below it.
    order = np.argsort(risk[control], kind="stable")
    control_risk, control_weight = risk[control][order], weight[control][order]
    cumulative = np.concatenate([[0.0], np.cumsum(control_weight)])
    below = cumulative[np.searchsorted(control_risk, risk[case], side="left")]
    tied = cumulative[np.searchsorted(control_risk, risk[case], side="right")] - below
    concordant = np.sum(weight[case] * (below + 0.5 * tied))
    return float(concordant / (weight[case].sum() * control_weight.sum()))


def disease_brier(
    cif: np.ndarray, obs_time: np.ndarray, obs_type: np.ndarray, K: np.ndarray, horizon: int
) -> float:
    """IPCW Brier score of the predicted disease CIF at ``horizon``."""
    case, _, weight = _horizon_status(obs_time, obs_type, K, horizon)
    return float(np.mean(weight * (case.astype(float) - cif) ** 2))


def disease_concordance(risk: np.ndarray, obs_time: np.ndarray, obs_type: np.ndarray) -> float:
    """Wolbers' competing-risks C-index for disease, without censoring weights.

    A pair (i, j) is comparable when i had disease in bin e and j was still
    disease-free at e: observed past e, censored at e, or already dead (a competing
    death can never become disease). Harrell-type: this is the simple version, not
    the IPCW-adjusted Antolini/Wolbers estimator, and it is reported as such.
    """
    cases = np.flatnonzero(obs_type == DISEASE)
    if cases.size == 0:
        return float("nan")
    concordant = comparable = 0.0
    for i in cases:
        e = obs_time[i]
        j = (obs_time > e) | ((obs_time == e) & (obs_type == CENSORED)) | (obs_type == DEATH)
        comparable += j.sum()
        concordant += np.sum(risk[i] > risk[j]) + 0.5 * np.sum(risk[i] == risk[j])
    return float(concordant / comparable) if comparable else float("nan")


def aalen_johansen_disease(obs_time: np.ndarray, obs_type: np.ndarray, horizon: int) -> float:
    """Nonparametric disease CIF at ``horizon`` in discrete time."""
    survival, cif = 1.0, 0.0
    for t in range(1, horizon + 1):
        at_risk = obs_time >= t
        n = at_risk.sum()
        if n == 0:
            break
        disease = np.sum(at_risk & (obs_time == t) & (obs_type == DISEASE))
        death = np.sum(at_risk & (obs_time == t) & (obs_type == DEATH))
        cif += survival * disease / n
        survival *= 1.0 - (disease + death) / n
    return float(cif)


def calibration_table(
    cif: np.ndarray, obs_time: np.ndarray, obs_type: np.ndarray, horizon: int, groups: int = 10
) -> np.ndarray:
    """``(groups, 3)``: mean predicted CIF, observed (Aalen-Johansen) CIF, group size."""
    edges = np.quantile(cif, np.linspace(0, 1, groups + 1))
    group = np.clip(np.searchsorted(edges, cif, side="right") - 1, 0, groups - 1)
    rows = []
    for g in range(groups):
        members = group == g
        if members.any():
            rows.append(
                (
                    cif[members].mean(),
                    aalen_johansen_disease(obs_time[members], obs_type[members], horizon),
                    members.sum(),
                )
            )
    return np.asarray(rows, dtype=float)


def interval_nll(
    disease_hazard: np.ndarray,
    death_hazard: np.ndarray,
    obs_time: np.ndarray,
    obs_type: np.ndarray,
) -> float:
    """Mean per-interval multinomial NLL over the bins each person was at risk."""
    n, T = disease_hazard.shape
    bins = np.arange(1, T + 1)[None, :]
    at_risk = bins <= obs_time[:, None]
    last = bins == obs_time[:, None]
    probability = np.where(
        last & (obs_type[:, None] == DISEASE),
        disease_hazard,
        np.where(
            last & (obs_type[:, None] == DEATH),
            death_hazard,
            1.0 - disease_hazard - death_hazard,
        ),
    )
    log_p = np.log(np.clip(probability, 1e-12, 1.0))
    return float(-(log_p * at_risk).sum() / max(at_risk.sum(), 1))


def prediction_metrics(
    disease_hazard: np.ndarray,
    death_hazard: np.ndarray,
    obs_time: np.ndarray,
    obs_type: np.ndarray,
    horizons: tuple[int, ...],
) -> tuple[dict[str, float], np.ndarray]:
    """Spec §4 for one test fold; returns the metrics and the calibration table at T.

    ``ici_decile`` is the size-weighted mean |predicted - observed| over risk deciles:
    a binned stand-in for the integrated calibration index, which proper ICI (a
    smoothed calibration curve) would replace.
    """
    T = disease_hazard.shape[1]
    disease_cif, _ = cumulative_incidence(disease_hazard, death_hazard)
    K = still_observed(obs_time, obs_type, T)
    metrics = {
        "interval_nll": interval_nll(disease_hazard, death_hazard, obs_time, obs_type),
        "c_index_disease": disease_concordance(disease_cif[:, -1], obs_time, obs_type),
    }
    for horizon in horizons:
        risk = disease_cif[:, horizon - 1]
        metrics[f"auc_disease_t{horizon}"] = disease_auc(risk, obs_time, obs_type, K, horizon)
        metrics[f"brier_disease_t{horizon}"] = disease_brier(risk, obs_time, obs_type, K, horizon)
    table = calibration_table(disease_cif[:, -1], obs_time, obs_type, T)
    metrics["ici_decile"] = float(np.average(np.abs(table[:, 0] - table[:, 1]), weights=table[:, 2]))
    return metrics, table
