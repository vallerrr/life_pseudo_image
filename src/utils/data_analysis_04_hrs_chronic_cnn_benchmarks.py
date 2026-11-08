"""The non-network benchmarks of analysis 04: what the CNN and RNN have to beat (D29).

Owned by src/data_analysis/04_hrs_chronic_cnn.py, run inside each outer fold by
``run_fold``. Every model sees the same training rows and is scored on the same test
rows as the networks. Every model that is tuned is tuned on inner folds grouped by
household -- never on the test fold -- by log loss.

| model | input | what it stands for |
|---|---|---|
| ``logistic_static`` | static covariates | no life history at all |
| ``logistic_summary`` | per row: mean, last, share observed; statics | a summarised history, ridge |
| ``logistic_windows`` | per row: mean over ages 50-59, 60-69, 70+; statics | the "manual life-course" model (spec §2) |
| ``elastic_net_cells`` | every cell + its missing indicator; statics | timing, but linear |
| ``boosting_cells`` | every cell; statics (sklearn defaults) | an untuned flexible model, kept for continuity |
| ``lightgbm_snapshot`` | per row: last observed value; statics | the one-time-point design of Yan (2026) |
| ``lightgbm_summary`` | the summary features | a tuned flexible model on summaries |
| ``lightgbm_cells`` | every cell; statics | the honest competitor (spec §2), tuned |

LightGBM is tuned over a small grid with early stopping on each inner fold; the
final model is refitted on the whole training fold with the inner folds' mean best
number of trees.
"""

from __future__ import annotations

import itertools
import time
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.shared_hrs_chronic import ChronicDataset

BENCHMARKS = (
    "logistic_static",
    "logistic_summary",
    "logistic_windows",
    "elastic_net_cells",
    "boosting_cells",
    "lightgbm_snapshot",
    "lightgbm_summary",
    "lightgbm_cells",
)
# A cell observed by fewer training people than this is dropped from the cell-level
# models: boosting cannot bin it, and nothing can be learnt from it.
MIN_CELL_OBSERVERS = 20


# --- features -------------------------------------------------------------------------------
def static_features(data: ChronicDataset, rows: np.ndarray) -> np.ndarray:
    return data.static[rows].astype(np.float64)


def _row_means(X: np.ndarray, M: np.ndarray, bins: slice) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row never observed in the window: NaN
        return np.nanmean(np.where(M[:, :, bins], X[:, :, bins], np.nan), axis=2)


def _last_values(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    K = X.shape[2]
    last_index = np.where(M.any(axis=2), K - 1 - M[:, :, ::-1].argmax(axis=2), 0)
    return np.where(M.any(axis=2), np.take_along_axis(X, last_index[:, :, None], axis=2)[:, :, 0], np.nan)


def summary_features(data: ChronicDataset, rows: np.ndarray) -> np.ndarray:
    """Per row: mean of observed bins, last observed value, share of bins observed."""
    X, M = data.X[rows].astype(np.float64), data.M[rows]
    return np.column_stack([_row_means(X, M, slice(None)), _last_values(X, M), M.mean(axis=2),
                            static_features(data, rows)])


def snapshot_features(data: ChronicDataset, rows: np.ndarray) -> np.ndarray:
    """Per row: the last observed value only -- one time point, as in Yan (2026)."""
    X, M = data.X[rows].astype(np.float64), data.M[rows]
    return np.column_stack([_last_values(X, M), static_features(data, rows)])


def window_features(data: ChronicDataset, rows: np.ndarray, windows: list[list[int]]) -> np.ndarray:
    """Per row: the mean over each age window ``[lo, hi)`` that the image reaches."""
    X, M = data.X[rows].astype(np.float64), data.M[rows]
    K = X.shape[2]
    starts = 50 + 2 * np.arange(K)  # age at the start of each bin (life image clock)
    blocks = []
    for lo, hi in windows:
        inside = np.flatnonzero((starts >= lo) & (starts < hi))
        if inside.size:
            blocks.append(_row_means(X, M, slice(inside[0], inside[-1] + 1)))
    return np.column_stack(blocks + [static_features(data, rows)])


def cell_features(data: ChronicDataset, rows: np.ndarray) -> np.ndarray:
    """Every cell, NaN where unobserved, then the statics."""
    return np.column_stack([data.X[rows].reshape(len(rows), -1).astype(np.float64), static_features(data, rows)])


def _seen_columns(train: np.ndarray) -> np.ndarray:
    return (~np.isnan(train)).sum(axis=0) >= MIN_CELL_OBSERVERS


# --- the models ---------------------------------------------------------------------------------
def _linear(model) -> Any:
    """Mean-impute with a missing indicator per column, standardise, then ``model``."""
    return make_pipeline(SimpleImputer(strategy="mean", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), model)


def _ridge_cv(config: dict[str, Any], inner: list) -> LogisticRegressionCV:
    return LogisticRegressionCV(Cs=config["benchmarks"]["logistic_Cs"], cv=inner, max_iter=3000,
                                scoring="neg_log_loss", l1_ratios=(0.0,), use_legacy_attributes=False)


def _elastic_net_cv(config: dict[str, Any], inner: list) -> LogisticRegressionCV:
    spec = config["benchmarks"]["elastic_net"]
    # saga is single-threaded and, on the real cells, mostly runs to max_iter: ~8-10 min per
    # outer fold on one core (2026-10-26 run). The inner folds and l1 ratios are independent
    # fits, so n_jobs spreads them over the cores without changing any result.
    return LogisticRegressionCV(Cs=spec["Cs"], cv=inner, solver="saga", max_iter=spec["max_iter"], tol=1e-3,
                                scoring="neg_log_loss", l1_ratios=tuple(spec["l1_ratios"]), use_legacy_attributes=False,
                                n_jobs=spec.get("n_jobs", -1))


def tune_lightgbm(X: np.ndarray, y: np.ndarray, inner: list, config: dict[str, Any], seed: int):
    """Grid search by inner-fold log loss with early stopping; refit on all of ``X``."""
    import lightgbm

    spec = config["benchmarks"]["lightgbm"]
    grid = [dict(zip(spec["grid"], v)) for v in itertools.product(*spec["grid"].values())]
    base = {**spec["fixed"], "n_estimators": spec["max_estimators"], "random_state": seed % 2**31, "verbose": -1,
            "n_jobs": spec.get("n_jobs", -1)}
    scores = []
    for params in grid:
        losses, rounds = [], []
        for fit_idx, val_idx in inner:
            model = lightgbm.LGBMClassifier(**base, **params)
            model.fit(X[fit_idx], y[fit_idx], eval_X=(X[val_idx],), eval_y=(y[val_idx],), eval_metric="binary_logloss",
                      callbacks=[lightgbm.early_stopping(spec["early_stopping_rounds"], verbose=False)])
            losses.append(log_loss(y[val_idx], model.predict_proba(X[val_idx])[:, 1], labels=[0, 1]))
            rounds.append(model.best_iteration_ or spec["max_estimators"])
        scores.append({**params, "inner_log_loss": float(np.mean(losses)), "n_estimators": int(np.mean(rounds))})
    best = min(scores, key=lambda s: s["inner_log_loss"])
    final = lightgbm.LGBMClassifier(**{**base, "n_estimators": max(best["n_estimators"], 1)},
                                    **{k: best[k] for k in spec["grid"]})
    final.fit(X, y)
    return final, best, scores


def fit_benchmarks(data: ChronicDataset, train: np.ndarray, test: np.ndarray, config: dict[str, Any], seed: int,
                   inner: list, only: tuple[str, ...] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, float]]:
    """Fit every benchmark (or those in ``only``) on ``train``; return test-fold P(case), tuning records, seconds.

    ``inner`` holds (fit, validation) index pairs into ``train``, grouped by household.
    """
    unknown = set(only or ()) - set(BENCHMARKS)
    if unknown:
        raise ValueError(f"unknown benchmark(s) {sorted(unknown)}; choose from {BENCHMARKS}")
    y = data.y[train]
    predictions: dict[str, np.ndarray] = {}
    tuning: dict[str, Any] = {}
    seconds: dict[str, float] = {}

    def timed(name: str, fit_predict) -> None:
        if only is not None and name not in only:
            return
        start = time.perf_counter()
        predictions[name] = fit_predict()
        seconds[name] = time.perf_counter() - start

    def linear(features, model_factory):
        def fit_predict() -> np.ndarray:
            tr, te = features(data, train), features(data, test)
            model = _linear(model_factory())
            model.fit(tr, y)
            return model.predict_proba(te)[:, 1]
        return fit_predict

    timed("logistic_static", linear(static_features, lambda: LogisticRegression(max_iter=2000)))
    timed("logistic_summary", linear(summary_features, lambda: _ridge_cv(config, inner)))
    windows = config["benchmarks"]["windows"]
    timed("logistic_windows", linear(lambda d, r: window_features(d, r, windows), lambda: _ridge_cv(config, inner)))

    def elastic_net() -> np.ndarray:
        tr, te = cell_features(data, train), cell_features(data, test)
        seen = _seen_columns(tr)
        model = _linear(_elastic_net_cv(config, inner))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # saga's ConvergenceWarning at the loosest penalties
            model.fit(tr[:, seen], y)
        fitted = model[-1]
        tuning["elastic_net_cells"] = {"C": float(np.ravel(fitted.C_)[0]), "l1_ratio": float(np.ravel(fitted.l1_ratio_)[0]),
                                       "nonzero": int((fitted.coef_ != 0).sum())}
        return model.predict_proba(te[:, seen])[:, 1]

    timed("elastic_net_cells", elastic_net)

    def boosting() -> np.ndarray:
        tr, te = cell_features(data, train), cell_features(data, test)
        seen = _seen_columns(tr)
        model = HistGradientBoostingClassifier(early_stopping=True, random_state=seed % 2**31)
        model.fit(tr[:, seen], y)
        return model.predict_proba(te[:, seen])[:, 1]

    timed("boosting_cells", boosting)

    def lightgbm_on(name: str, features):
        def fit_predict() -> np.ndarray:
            tr, te = features(data, train), features(data, test)
            seen = _seen_columns(tr)
            model, best, scores = tune_lightgbm(tr[:, seen], y, inner, config, seed)
            tuning[name] = {"best": best, "grid": scores}
            return model.predict_proba(te[:, seen])[:, 1]
        return fit_predict

    timed("lightgbm_snapshot", lightgbm_on("lightgbm_snapshot", snapshot_features))
    timed("lightgbm_summary", lightgbm_on("lightgbm_summary", summary_features))
    timed("lightgbm_cells", lightgbm_on("lightgbm_cells", cell_features))
    return predictions, tuning, seconds
