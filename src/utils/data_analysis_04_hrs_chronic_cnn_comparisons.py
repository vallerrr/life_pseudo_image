"""What the networks' tie with LightGBM is made of: time order, inputs, stacking, rows' trajectories (D38-D42).

Owned by ``src/data_analysis/04_hrs_chronic_cnn.py comparisons``. In each outer fold the
original models are fitted once, with the run's folds, seeds and grids (``run_fold``), and
three experiments are set against them on the same test fold:

* **Time order (D38).** The networks are retuned and refitted on images whose age axis is
  shuffled. ``fixed``: one permutation of the bins for everyone. It keeps every value and
  its age, but bins that were next to each other no longer are, so it tests whether the
  convolutions and the GRU use the order. ``person``: each person's observed bins are
  shuffled among themselves, so the values and the entry pattern stay but which value came
  at which age is lost. It tests whether timing carries anything at all. The ``person``
  shuffle is also given to the benchmarks named in ``time_order.benchmarks``, so a loss
  there can be read against theirs. ``Delta`` is recomputed from the shuffled mask.
* **Inputs (D39).** Each network and benchmark is refitted with the static covariates
  removed (image only), and LightGBM is tuned on the static covariates alone (beside the
  static logistic). Together with the original fits, that gives image only / statics only /
  both.
* **Stacking (D40).** A logistic regression on the logits of chosen models' P(case). Its
  weights are fitted on inner out-of-fold predictions from the training fold: each model is
  refitted on each inner fit part at the outer fold's hyperparameters, so the outer test fold
  never informs the weights. Plain averages of the probabilities need no weights.
* **BMI trajectory (D41).** One row (``bmi_trajectory.row``) is rewritten, every other row
  left as it was, and the networks and the named benchmarks are retuned and refitted.
  ``none``: the row removed. ``single``: only the person's latest observed value kept.
  ``level``: the person's mean put in every bin they were observed, so the row keeps its
  level and its observation pattern but loses its shape. ``shuffled``: the person's observed
  values permuted among their observed bins, so the spread stays but the order goes. If
  the original beats ``level`` and ``single``, the trajectory carries something the level
  does not.
* **Socioeconomic order (D42).** The same for a group of rows (``ses_order.rows``) at once.
  ``none`` and ``level`` act on each row as in D41; ``shuffled`` moves the group's values and
  masks with one permutation per person, over the bins where any of them was observed, so the
  rows stay aligned with each other but not with age.
* **Strata (D42).** Each arm of the parts in ``strata.parts`` is also scored within groups of
  the test fold, set from the original image: people with under two waves of the
  ``strata.rows`` (whom a shuffle cannot touch), thirds of a change score (each row's
  within-person SD on a z scale, averaged over rows), and named events (a later value at least
  ``fall`` below, or ``rise`` above, an earlier one) with their complements. A gap that is
  larger among people whose rows changed says order matters for them.
* **Population (D43).** ``comparisons.population`` names strata; when set, the dataset is cut
  to the people in any of them before the folds are drawn, so every arm, the original
  included, is tuned, fitted and scored on those people alone. The ``ses_order`` profile
  uses it to test socioeconomic order only on people who went through a socioeconomic shock.

``summarise`` reports each arm's mean metrics and fold-paired gaps (t interval over folds);
strata with fewer than ``MIN_CELL`` cases or controls in a test fold report a suppressed count
only. Per-person predictions stay on the machine (D22).
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from utils.data_analysis_04_hrs_chronic_cnn import (
    _fit_network,
    classification_metrics,
    household_folds,
    predict,
    seed_for,
    tensors,
    tune,
)
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import (
    _linear,
    _ridge_cv,
    _seen_columns,
    cell_features,
    fit_benchmarks,
    static_features,
    summary_features,
    tune_lightgbm,
)
from utils.shared_hrs_chronic import ChronicDataset
from utils.shared_hrs_ingest import _delta
from utils.shared_secure_report import MIN_CELL

ORIGINAL_BENCHMARKS = ("logistic_static", "logistic_summary", "lightgbm_summary", "lightgbm_cells")
STACKABLE_BENCHMARKS = ("logistic_summary", "lightgbm_summary", "lightgbm_cells")


# --- transformed datasets ---------------------------------------------------------------------------
def time_shuffled(data: ChronicDataset, how: str, rng: np.random.Generator) -> ChronicDataset:
    """``data`` with its age axis shuffled: ``fixed`` (one order for all) or ``person`` (observed bins, per person)."""
    n, _, K = data.X.shape
    if how == "fixed":
        order = np.broadcast_to(rng.permutation(K), (n, K))
    elif how == "person":
        seen = data.M.any(axis=1)  # (n, K): bins with any observation
        order = np.tile(np.arange(K), (n, 1))
        for i in range(n):
            bins = np.flatnonzero(seen[i])
            order[i, bins] = rng.permutation(bins)
    else:
        raise ValueError(f"time shuffle must be 'fixed' or 'person', got {how!r}")
    take = lambda a: np.take_along_axis(a, order[:, None, :], axis=2)  # noqa: E731
    M = take(data.M)
    return dataclasses.replace(data, X=take(data.X), M=M, Delta=_delta(M).astype(data.Delta.dtype))


ROW_ARMS = ("none", "single", "level", "shuffled")


def row_variant(data: ChronicDataset, row: str, how: str, rng: np.random.Generator) -> ChronicDataset:
    """``data`` with one row rewritten (``ROW_ARMS``); people with no observation of it are unchanged."""
    if how not in ROW_ARMS:
        raise ValueError(f"row arm must be one of {ROW_ARMS}, got {how!r}")
    if row not in data.channel_names:
        raise ValueError(f"row {row!r} is not in the image: {data.channel_names}")
    c = data.channel_names.index(row)
    X, M = data.X.copy(), data.M.copy()
    x, m = X[:, c, :], M[:, c, :]  # views
    K = x.shape[1]
    seen = m.any(axis=1)
    if how == "none":
        m[:] = False
    elif how == "single":
        last = K - 1 - np.argmax(m[:, ::-1], axis=1)
        keep = np.zeros_like(m)
        keep[np.flatnonzero(seen), last[seen]] = True
        m &= keep
    elif how == "level":
        total = np.where(m, x, 0).sum(axis=1)
        mean = np.divide(total, m.sum(axis=1), out=np.zeros_like(total), where=seen)
        x[:] = np.where(m, mean[:, None], x)
    else:  # shuffled
        for i in np.flatnonzero(m.sum(axis=1) > 1):
            bins = np.flatnonzero(m[i])
            x[i, bins] = x[i, rng.permutation(bins)]
    x[~m] = np.nan
    return dataclasses.replace(data, X=X, M=M, Delta=_delta(M).astype(data.Delta.dtype))


GROUP_ARMS = ("none", "level", "shuffled")


def rows_variant(data: ChronicDataset, rows: list[str], how: str, rng: np.random.Generator) -> ChronicDataset:
    """``data`` with a group of rows rewritten together (``GROUP_ARMS``); the shuffle keeps the rows aligned."""
    if how not in GROUP_ARMS:
        raise ValueError(f"group arm must be one of {GROUP_ARMS}, got {how!r}")
    unknown = [r for r in rows if r not in data.channel_names]
    if unknown:
        raise ValueError(f"rows {unknown} are not in the image: {data.channel_names}")
    if how != "shuffled":
        for row in rows:
            data = row_variant(data, row, how, rng)
        return data
    c = [data.channel_names.index(r) for r in rows]
    X, M = data.X.copy(), data.M.copy()
    seen = data.M[:, c, :].any(axis=1)
    for i in np.flatnonzero(seen.sum(axis=1) > 1):
        bins = np.flatnonzero(seen[i])
        order = rng.permutation(bins)
        X[i][np.ix_(c, bins)] = data.X[i][np.ix_(c, order)]
        M[i][np.ix_(c, bins)] = data.M[i][np.ix_(c, order)]
    return dataclasses.replace(data, X=X, M=M, Delta=_delta(M).astype(data.Delta.dtype))


def _swing(x: np.ndarray, m: np.ndarray, direction: str, size: float) -> np.ndarray:
    """Whether a later observed value is at least ``size`` below (``fall``) or above (``rise``) an earlier one."""
    v = np.where(m, x if direction == "fall" else -x, -np.inf)
    with np.errstate(invalid="ignore"):
        return (m & (np.maximum.accumulate(v, axis=1) - v >= size)).any(axis=1)


def strata(data: ChronicDataset, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Groups of people by how much ``spec['rows']`` changed in the original image, and by named events (D42)."""
    c = [data.channel_names.index(r) for r in spec["rows"]]
    X, M = data.X[:, c, :], data.M[:, c, :]
    waves = M.any(axis=1).sum(axis=1)
    groups = {"under_2_waves": waves < 2}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN people and rows
        z = (X - np.nanmean(X, axis=(0, 2), keepdims=True)) / np.nanstd(X, axis=(0, 2), keepdims=True)
        sd = np.where(M.sum(axis=2) >= 2, np.nanstd(z, axis=2), np.nan)
        score = np.round(np.nanmean(sd, axis=1), 10)  # no change is 0, not a rounding error above it
    many = (waves >= 2) & np.isfinite(score)
    q = spec["quantiles"]
    edges = np.quantile(score[many], np.linspace(0, 1, q + 1)[1:-1]) if many.any() else np.zeros(q - 1)
    level = np.searchsorted(edges, np.nan_to_num(score), side="left")  # ties at a cut (e.g. no change) go below
    for k in range(q):
        groups[f"change_q{k + 1}"] = many & (level == k)
    for name, event in spec["events"].items():
        k = data.channel_names.index(event["row"])
        direction = "fall" if "fall" in event else "rise"
        x, m = data.X[:, k, :], data.M[:, k, :]
        eligible = m.sum(axis=1) >= 2
        groups[name] = eligible & _swing(x, m, direction, event[direction])
        groups[f"no_{name}"] = eligible & ~groups[name]
    return groups


def restrict(data: ChronicDataset, keep: np.ndarray) -> ChronicDataset:
    """``data`` cut to the people in ``keep``; every per-person array follows."""
    return dataclasses.replace(data, X=data.X[keep], M=data.M[keep], Delta=data.Delta[keep], static=data.static[keep],
                               y=data.y[keep], household=data.household[keep])


def population(data: ChronicDataset, spec: dict[str, Any]) -> np.ndarray:
    """Who is in the analysis: everyone, or the union of the strata named in ``spec['population']`` (D43)."""
    names = spec.get("population")
    if not names:
        return np.ones(data.n, dtype=bool)
    groups = strata(data, spec["strata"])
    unknown = [name for name in names if name not in groups]
    if unknown:
        raise ValueError(f"population strata {unknown} are not defined: {list(groups)}")
    return np.logical_or.reduce([groups[name] for name in names])


def image_only(data: ChronicDataset) -> ChronicDataset:
    """``data`` with no static covariates: the networks' heads and the benchmarks' features lose them."""
    return dataclasses.replace(data, static=data.static[:, :0], static_names=())


# --- fitting --------------------------------------------------------------------------------------
def _networks(data: ChronicDataset, train: np.ndarray, test: np.ndarray, config: dict[str, Any], seed: int,
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Each network tuned and fitted on ``train`` with ``run_fold``'s seeds; test-fold P(case) and hyperparameters."""
    p, hyper = {}, {}
    for n, network in enumerate(config["networks"]):
        net_seed = seed + 1000 * n
        hyper[network], _ = tune(data, train, network, config, net_seed)
        model, scaler, _ = _fit_network(data, train, network, hyper[network], config, net_seed + 1)
        p[network] = predict(model, *tensors(data, test, scaler)[:2])
    return p, hyper


def _lightgbm_refit(X: np.ndarray, y: np.ndarray, best: dict[str, Any], config: dict[str, Any], seed: int):
    """LightGBM at hyperparameters already chosen (``tune_lightgbm``'s ``best``), with no search."""
    import lightgbm

    spec = config["benchmarks"]["lightgbm"]
    model = lightgbm.LGBMClassifier(**spec["fixed"], n_estimators=max(best["n_estimators"], 1),
                                    random_state=seed % 2**31, verbose=-1, n_jobs=spec.get("n_jobs", -1),
                                    **{k: best[k] for k in spec["grid"]})
    return model.fit(X, y)


def inner_predictions(data: ChronicDataset, train: np.ndarray, inner: list, models: list[str], hyper: dict[str, Any],
                      tuning: dict[str, Any], config: dict[str, Any], seed: int) -> dict[str, np.ndarray]:
    """Out-of-fold P(case) over ``train`` for each of ``models``, from refits on each inner fit part.

    Networks and LightGBM keep the outer fold's hyperparameters; the summary logistic re-chooses
    its ridge strength on folds of the inner fit part, as it does on the outer one.
    """
    y = data.y
    p = {m: np.full(len(train), np.nan) for m in models}
    for k, (fit_idx, val_idx) in enumerate(inner):
        fit, val = train[fit_idx], train[val_idx]
        for n, network in enumerate(config["networks"]):
            if network in models:
                model, scaler, _ = _fit_network(data, fit, network, hyper[network], config, seed + 1000 * n + 17 + k)
                p[network][val_idx] = predict(model, *tensors(data, val, scaler)[:2])
        for name, features in (("lightgbm_summary", summary_features), ("lightgbm_cells", cell_features)):
            if name in models:
                X = features(data, fit)
                seen = _seen_columns(X)
                model = _lightgbm_refit(X[:, seen], y[fit], tuning[name]["best"], config, seed)
                p[name][val_idx] = model.predict_proba(features(data, val)[:, seen])[:, 1]
        if "logistic_summary" in models:
            folds = household_folds(y[fit], data.household[fit], config["inner_folds"], seed + 23 + k)
            model = _linear(_ridge_cv(config, folds))
            model.fit(summary_features(data, fit), y[fit])
            p["logistic_summary"][val_idx] = model.predict_proba(summary_features(data, val))[:, 1]
    return p


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack(p_inner: dict[str, np.ndarray], y_inner: np.ndarray, p_test: dict[str, np.ndarray], models: list[str],
          C: float) -> tuple[np.ndarray, dict[str, float]]:
    """Logistic regression on the models' logits, fitted on the inner predictions; test P(case) and its weights."""
    Z = np.column_stack([_logit(p_inner[m]) for m in models])
    fit = LogisticRegression(C=C, max_iter=1000).fit(Z, y_inner)
    weights = {m: float(w) for m, w in zip(models, fit.coef_[0])}
    return fit.predict_proba(np.column_stack([_logit(p_test[m]) for m in models]))[:, 1], weights


# --- one outer fold ---------------------------------------------------------------------------------
def comparisons_fold(data: ChronicDataset, fold: int, train: np.ndarray, test: np.ndarray,
                     config: dict[str, Any]) -> dict[str, Any]:
    """Every arm's test metrics on this fold, keyed ``<arm>/<model>``."""
    spec = config["comparisons"]
    parts = spec["parts"]
    seed = seed_for(data.condition, data.status_bin, f"fold{fold}")  # as run_fold
    y_train, y_test = data.y[train], data.y[test]
    inner = household_folds(y_train, data.household[train], config["inner_folds"], seed + 7)
    out: dict[str, Any] = {"metrics": {}, "hyperparameters": {}, "seconds": {}}
    strata_spec = spec.get("strata")
    stratify = bool(strata_spec) and any(part in parts for part in strata_spec["parts"])
    if stratify:
        groups = {name: g[test] for name, g in strata(data, strata_spec).items()}
        out["strata"] = {name: {} for name in groups}
        out["strata_counts"] = {name: {"n": int(g.sum()), "cases": int(y_test[g].sum())} for name, g in groups.items()}

    def score(arm: str, predictions: dict[str, np.ndarray]) -> None:
        for model, p in predictions.items():
            out["metrics"][f"{arm}/{model}"] = classification_metrics(y_test, p)
            if stratify:
                for name, g in groups.items():
                    if 0 < y_test[g].sum() < g.sum():
                        q = np.clip(p[g], 1e-6, 1 - 1e-6)
                        out["strata"][name][f"{arm}/{model}"] = {"auc": float(roc_auc_score(y_test[g], q)),
                                                                 "log_loss": float(log_loss(y_test[g], q, labels=[0, 1]))}

    start = time.perf_counter()
    p_bench, tuning, _ = fit_benchmarks(data, train, test, config, seed, inner, only=ORIGINAL_BENCHMARKS)
    p_net, hyper = _networks(data, train, test, config, seed)
    original = {**p_bench, **p_net}
    score("original", original)
    out["hyperparameters"]["original"] = hyper
    out["seconds"]["original"] = time.perf_counter() - start

    if "time_order" in parts:
        for k, how in enumerate(spec["time_order"]["arms"]):
            start = time.perf_counter()
            shuffled = time_shuffled(data, how, np.random.default_rng(seed + 41 + k))
            p, out["hyperparameters"][f"time_{how}"] = _networks(shuffled, train, test, config, seed)
            if how == "person" and spec["time_order"]["benchmarks"]:
                p_b, _, _ = fit_benchmarks(shuffled, train, test, config, seed, inner,
                                           only=tuple(spec["time_order"]["benchmarks"]))
                p.update(p_b)
            score(f"time_{how}", p)
            out["seconds"][f"time_{how}"] = time.perf_counter() - start

    if "bmi_trajectory" in parts:
        row_spec = spec["bmi_trajectory"]
        for k, how in enumerate(row_spec["arms"]):
            start = time.perf_counter()
            arm = f"{row_spec['row']}_{how}"
            varied = row_variant(data, row_spec["row"], how, np.random.default_rng(seed + 61 + k))
            p, out["hyperparameters"][arm] = _networks(varied, train, test, config, seed)
            if row_spec["benchmarks"]:
                p_b, _, _ = fit_benchmarks(varied, train, test, config, seed, inner, only=tuple(row_spec["benchmarks"]))
                p.update(p_b)
            score(arm, p)
            out["seconds"][arm] = time.perf_counter() - start

    if "ses_order" in parts:
        group_spec = spec["ses_order"]
        for k, how in enumerate(group_spec["arms"]):
            start = time.perf_counter()
            varied = rows_variant(data, group_spec["rows"], how, np.random.default_rng(seed + 71 + k))
            p, out["hyperparameters"][f"ses_{how}"] = _networks(varied, train, test, config, seed)
            if group_spec["benchmarks"]:
                p_b, _, _ = fit_benchmarks(varied, train, test, config, seed, inner, only=tuple(group_spec["benchmarks"]))
                p.update(p_b)
            score(f"ses_{how}", p)
            out["seconds"][f"ses_{how}"] = time.perf_counter() - start

    if "inputs" in parts:
        start = time.perf_counter()
        bare = image_only(data)
        p, out["hyperparameters"]["image_only"] = _networks(bare, train, test, config, seed)
        p_b, _, _ = fit_benchmarks(bare, train, test, config, seed, inner, only=tuple(spec["inputs"]["benchmarks"]))
        score("image_only", {**p_b, **p})
        model, best, _ = tune_lightgbm(static_features(data, train), y_train, inner, config, seed)
        score("statics_only", {"lightgbm_static": model.predict_proba(static_features(data, test))[:, 1]})
        out["hyperparameters"]["statics_only"] = {"lightgbm_static": best}
        out["seconds"]["inputs"] = time.perf_counter() - start

    if "ensemble" in parts:
        start = time.perf_counter()
        ens = spec["ensemble"]
        members = sorted({m for group in [*ens["stacks"].values(), *ens["averages"].values()] for m in group})
        unknown = [m for m in members if m not in STACKABLE_BENCHMARKS and m not in config["networks"]]
        if unknown:
            raise ValueError(f"cannot stack {unknown}: choose from {[*STACKABLE_BENCHMARKS, *config['networks']]}")
        stacked = sorted({m for group in ens["stacks"].values() for m in group})
        p_inner = inner_predictions(data, train, inner, stacked, hyper, tuning, config, seed)
        found, out["stack_weights"] = {}, {}
        for name, models in ens["stacks"].items():
            found[name], out["stack_weights"][name] = stack(p_inner, y_train, original, models, ens["C"])
        score("stack", found)
        score("average", {name: np.mean([original[m] for m in models], axis=0) for name, models in ens["averages"].items()})
        out["inner_metrics"] = {m: classification_metrics(y_train, p_inner[m]) for m in stacked}
        out["seconds"]["ensemble"] = time.perf_counter() - start
    return out


# --- summary --------------------------------------------------------------------------------------
def paired_gap(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean of ``a - b`` over folds, its SE, 95% t interval and two-sided p."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    se = float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
    half = float(stats.t.ppf(0.975, k - 1) * se) if k > 1 else float("nan")
    p = float(2 * stats.t.sf(abs(d.mean()) / se, k - 1)) if k > 1 and se > 0 else float("nan")
    return {"mean": float(d.mean()), "se": se, "lo": float(d.mean() - half), "hi": float(d.mean() + half), "p": p}


def gap_pairs(keys: set[str], config: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """The (arm, reference) pairs each part compares, kept where both were fitted."""
    networks = list(config["networks"])
    spec = config["comparisons"]
    pairs: dict[str, list[tuple[str, str]]] = {"time_order": [], "bmi_trajectory": [], "ses_order": [], "inputs": [],
                                               "ensemble": []}
    for how in spec["time_order"]["arms"]:
        for m in [*networks, *spec["time_order"]["benchmarks"]]:
            pairs["time_order"].append((f"time_{how}/{m}", f"original/{m}"))
    row_spec = spec["bmi_trajectory"]
    row_arms = row_spec["arms"]
    for m in [*networks, *row_spec["benchmarks"]]:
        for how in row_arms:  # each arm against the full row, then against the row removed
            pairs["bmi_trajectory"].append((f"{row_spec['row']}_{how}/{m}", f"original/{m}"))
        for how in row_arms:
            if how != "none" and "none" in row_arms:
                pairs["bmi_trajectory"].append((f"{row_spec['row']}_{how}/{m}", f"{row_spec['row']}_none/{m}"))
    if "ses_order" in spec:
        group_arms = spec["ses_order"]["arms"]
        for m in [*networks, *spec["ses_order"]["benchmarks"]]:
            for how in group_arms:
                pairs["ses_order"].append((f"ses_{how}/{m}", f"original/{m}"))
            for how in group_arms:
                if how != "none" and "none" in group_arms:
                    pairs["ses_order"].append((f"ses_{how}/{m}", f"ses_none/{m}"))
    for m in [*networks, *spec["inputs"]["benchmarks"]]:
        pairs["inputs"].append((f"image_only/{m}", f"original/{m}"))
        pairs["inputs"].append((f"original/{m}", "statics_only/lightgbm_static"))
    pairs["inputs"].append(("statics_only/lightgbm_static", "original/logistic_static"))
    for ref in spec["ensemble"]["references"]:
        for name in spec["ensemble"]["stacks"]:
            if f"stack/{name}" != ref:
                pairs["ensemble"].append((f"stack/{name}", ref))
        for name in spec["ensemble"]["averages"]:
            pairs["ensemble"].append((f"average/{name}", ref))
    return {part: [(a, b) for a, b in found if a in keys and b in keys] for part, found in pairs.items()}


def summarise(per_fold: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    keys = set.intersection(*(set(f["metrics"]) for f in per_fold))
    metrics = sorted(next(iter(per_fold[0]["metrics"].values())))
    mean = {k: {m: float(np.mean([f["metrics"][k][m] for f in per_fold])) for m in metrics} for k in sorted(keys)}
    sd = {k: float(np.std([f["metrics"][k]["auc"] for f in per_fold], ddof=1)) for k in sorted(keys)}
    gaps = {}
    for part, pairs in gap_pairs(keys, config).items():
        gaps[part] = {f"{a} - {b}": {
            metric: paired_gap([f["metrics"][a][metric] for f in per_fold], [f["metrics"][b][metric] for f in per_fold])
            for metric in ("auc", "log_loss", "brier")} for a, b in pairs}
    out = {"models": sorted(keys), "metrics_mean": mean, "auc_sd": sd, "gaps": gaps}
    if "strata" in per_fold[0]:
        out["strata"] = summarise_strata(per_fold, config, keys)
    if "stack_weights" in per_fold[0]:
        out["stack_weights"] = {name: {m: float(np.mean([f["stack_weights"][name][m] for f in per_fold])) for m in w}
                                for name, w in per_fold[0]["stack_weights"].items()}
        out["inner_auc"] = {m: float(np.mean([f["inner_metrics"][m]["auc"] for f in per_fold]))
                            for m in per_fold[0]["inner_metrics"]}
    return out


def summarise_strata(per_fold: list[dict[str, Any]], config: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """Arm-minus-original gaps within each stratum, and differences of those gaps between strata (D42).

    A stratum is reported only if every test fold holds ``MIN_CELL`` cases and controls in it.
    """
    spec = config["comparisons"]["strata"]
    names = list(per_fold[0]["strata_counts"])
    shown, counts = [], {}
    for name in names:
        n = sum(f["strata_counts"][name]["n"] for f in per_fold)
        cases = sum(f["strata_counts"][name]["cases"] for f in per_fold)
        thin = any(min(f["strata_counts"][name]["cases"], f["strata_counts"][name]["n"] - f["strata_counts"][name]["cases"])
                   < MIN_CELL for f in per_fold)
        counts[name] = {"n": n if n >= MIN_CELL or n == 0 else f"<{MIN_CELL}",
                        "cases": cases if cases >= MIN_CELL or cases == 0 else f"<{MIN_CELL}", "suppressed": thin}
        if not thin:
            shown.append(name)
    found = gap_pairs(keys, config)
    pairs = [(a, b) for part in spec["parts"] for a, b in found.get(part, []) if b.startswith("original/")]

    def per_fold_gap(a: str, b: str, name: str, metric: str) -> np.ndarray:
        return np.array([f["strata"][name][a][metric] - f["strata"][name][b][metric] for f in per_fold])

    usable = lambda a, b, name: all(a in f["strata"][name] and b in f["strata"][name] for f in per_fold)  # noqa: E731
    contrasts = [(f"change_q{spec['quantiles']}", "change_q1"), *[(e, f"no_{e}") for e in spec["events"]]]
    gaps, differences, original = {}, {}, {}
    for a, b in pairs:
        gaps[f"{a} - {b}"] = {name: {metric: paired_gap(per_fold_gap(a, b, name, metric), np.zeros(len(per_fold)))
                                     for metric in ("auc", "log_loss")}
                              for name in shown if usable(a, b, name)}
        differences[f"{a} - {b}"] = {f"{g1} - {g2}": {metric: paired_gap(per_fold_gap(a, b, g1, metric), per_fold_gap(a, b, g2, metric))
                                                      for metric in ("auc", "log_loss")}
                                     for g1, g2 in contrasts if g1 in shown and g2 in shown and usable(a, b, g1) and usable(a, b, g2)}
    for model in sorted({b for _, b in pairs}):
        original[model] = {name: float(np.mean([f["strata"][name][model]["auc"] for f in per_fold]))
                           for name in shown if all(model in f["strata"][name] for f in per_fold)}
    return {"counts": counts, "original_auc": original, "gaps": gaps, "differences": differences}


# --- figure --------------------------------------------------------------------------------------
PART_TITLES = {"time_order": "Time order shuffled minus original", "bmi_trajectory": "One row rewritten",
               "ses_order": "Socioeconomic rows rewritten",
               "inputs": "Inputs",
               "ensemble": "Stacking and averages"}


def comparisons_figure(section: dict[str, Any], style: dict[str, Any]):
    """One panel per part: each fold-paired AUC gap with its 95% t interval, against zero."""
    from matplotlib.figure import Figure

    gaps = {part: found for part, found in section["summary"]["gaps"].items() if found}
    heights = [max(2, len(found)) for found in gaps.values()]
    fig = Figure(figsize=(style["figsize"][0], 0.32 * sum(heights) + 1.6), dpi=style["dpi"])
    axes = fig.subplots(len(gaps), 1, gridspec_kw={"height_ratios": heights}, squeeze=False)[:, 0]
    colors = style["colors"]
    for ax, (part, found) in zip(axes, gaps.items()):
        labels = list(found)
        g = [found[k]["auc"] for k in labels]
        ys = np.arange(len(labels))[::-1]
        mean = np.array([x["mean"] for x in g])
        err = np.array([[x["mean"] - x["lo"] for x in g], [x["hi"] - x["mean"] for x in g]])
        ax.errorbar(mean, ys, xerr=err, fmt="o", color=colors[0], capsize=3, markersize=4)
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set_yticks(ys, [k.replace(" - ", "  minus  ") for k in labels], fontsize=style["legend_fs"])
        ax.set_title(PART_TITLES.get(part, part), fontsize=style["label_fs"], loc="left")
        ax.set_xlabel("test AUC difference (fold-paired, 95% t interval)", fontsize=style["label_fs"])
    title = section["dataset"].get("title", section["dataset"]["condition"])
    ages = section["dataset"]["status_ages"]
    fig.suptitle(f"Comparisons: {title} {ages[0]}-{ages[1]}", fontsize=style["title_fs"])
    fig.tight_layout()
    return fig
