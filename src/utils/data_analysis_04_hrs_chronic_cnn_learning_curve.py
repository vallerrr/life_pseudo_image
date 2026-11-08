"""Does the networks' standing improve with more data? A learning curve on HRS (D32).

Owned by ``src/data_analysis/04_hrs_chronic_cnn.py learning-curve``. In each outer fold
the training part is cut to a share of its households -- nested, so the 25% sample holds
the 10% one -- and every model is tuned and fitted on that share alone, exactly as on the
whole fold (inner household-grouped folds, the same grids). Every share is scored on the
whole outer test fold, so a curve's points differ only in how much the models were given.

The question is the slope, not the level: if the networks gain more per doubling of the
training sample than the best benchmark does, more data would favour them; if the curves
run parallel, or have flattened before 100%, it would not. ``summarise`` reports, per
model and share, the mean test AUC; per network, its fold-paired gap to each benchmark
and to the best benchmark at that share; and the gap's slope per doubling of N, fitted
over every (fold, repeat, share) by least squares. Units at different shares share
people (the samples are nested), so that slope's standard error is too small and is
reported as a rough guide only.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from utils.data_analysis_04_hrs_chronic_cnn import (
    _fit_network,
    classification_metrics,
    household_folds,
    predict,
    tensors,
    tune,
)
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import fit_benchmarks
from utils.shared_hrs_chronic import ChronicDataset


def household_subsample(rows: np.ndarray, household: np.ndarray, share: float, rng: np.random.Generator,
                        order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The rows of the first ``share`` of ``rows``' households in a random order; and that order.

    Pass the returned order back in to draw a larger share that contains this one.
    """
    houses = np.unique(household[rows])
    if order is None:
        order = rng.permutation(houses)
    keep = order[: max(1, math.ceil(share * len(houses)))]
    return rows[np.isin(household[rows], keep)], order


def learning_curve_fold(data: ChronicDataset, fold: int, train: np.ndarray, test: np.ndarray,
                        config: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """One outer fold: every share x repeat, every model; one record per (share, repeat)."""
    spec = config["learning_curve"]
    shares = sorted(spec["shares"])
    y_test = data.y[test]
    records = []
    for repeat in range(spec["repeats"]):
        rng = np.random.default_rng(seed + 31 * (repeat + 1))
        order = None
        for share in shares:
            if share >= 1 and repeat > 0:
                continue  # the whole fold is the same sample in every repeat
            start = time.perf_counter()
            rows, order = household_subsample(train, data.household, share, rng, order)
            rows = np.sort(rows)
            record: dict[str, Any] = {"fold": fold, "repeat": repeat, "share": share, "n_train": int(len(rows)),
                                      "train_cases": int(data.y[rows].sum()), "metrics": {}, "hyperparameters": {}}
            if spec["benchmarks"]:
                inner = household_folds(data.y[rows], data.household[rows], config["inner_folds"], seed + 7)
                predictions, _, _ = fit_benchmarks(data, rows, test, config, seed, inner, only=tuple(spec["benchmarks"]))
                for name, p in predictions.items():
                    record["metrics"][name] = classification_metrics(y_test, p)
            for n, network in enumerate(config["networks"]):
                net_seed = seed + 1000 * n  # as run_fold, so the 100% point repeats the run's seeds
                hyper, _ = tune(data, rows, network, config, net_seed)
                model, scaler, _ = _fit_network(data, rows, network, hyper, config, net_seed + 1)
                image, static, _ = tensors(data, test, scaler)
                record["metrics"][network] = classification_metrics(y_test, predict(model, image, static))
                record["hyperparameters"][network] = hyper
            record["seconds"] = time.perf_counter() - start
            records.append(record)
    return records


def _ols_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Slope of ``y`` on ``x`` and its naive standard error."""
    if len(x) < 3 or np.ptp(x) == 0:
        return {"slope": float("nan"), "se": float("nan")}
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / (len(x) - 2)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return {"slope": float(beta[1]), "se": float(np.sqrt(cov[1, 1]))}


def summarise(records: list[dict[str, Any]], config: dict[str, Any], metric: str = "auc") -> dict[str, Any]:
    """Per share: mean and SD of each model's test metric; each network's paired gaps and their slope."""
    networks = list(config["networks"])
    benchmarks = list(config["learning_curve"]["benchmarks"])
    shares = sorted({r["share"] for r in records})
    by_share = {s: [r for r in records if r["share"] == s] for s in shares}

    curve: dict[str, Any] = {}
    for model in benchmarks + networks:
        curve[model] = {
            "share": shares,
            "n_train": [float(np.mean([r["n_train"] for r in by_share[s]])) for s in shares],
            "mean": [float(np.mean([r["metrics"][model][metric] for r in by_share[s]])) for s in shares],
            "sd": [float(np.std([r["metrics"][model][metric] for r in by_share[s]], ddof=1))
                   if len(by_share[s]) > 1 else float("nan") for s in shares],
            "units": [len(by_share[s]) for s in shares],
        }

    gaps: dict[str, Any] = {}
    for net in networks:
        gaps[net] = {}
        # "best": at each share, the benchmark with the highest mean there.
        best_at = {s: max(benchmarks, key=lambda b: curve[b]["mean"][shares.index(s)]) for s in shares} if benchmarks else {}
        for ref in [*benchmarks, *(["best"] if benchmarks else [])]:
            diffs = {s: np.array([r["metrics"][net][metric] - r["metrics"][best_at[s] if ref == "best" else ref][metric]
                                  for r in by_share[s]]) for s in shares}
            x = np.concatenate([np.full(len(diffs[s]), math.log2(np.mean([r["n_train"] for r in by_share[s]])))
                                for s in shares])
            gaps[net][ref] = {
                "mean": [float(diffs[s].mean()) for s in shares],
                "se": [float(diffs[s].std(ddof=1) / math.sqrt(len(diffs[s]))) if len(diffs[s]) > 1 else float("nan")
                       for s in shares],
                "per_doubling": _ols_slope(x, np.concatenate([diffs[s] for s in shares])),
                **({"which": [best_at[s] for s in shares]} if ref == "best" else {}),
            }

    per_doubling = {}
    for model in benchmarks + networks:
        x = np.array([math.log2(r["n_train"]) for r in records])
        per_doubling[model] = _ols_slope(x, np.array([r["metrics"][model][metric] for r in records]))
    # The last doubling alone (50% -> 100%), where a curve that has flattened shows it.
    last = {}
    if len(shares) >= 2:
        a, b = shares[-2], shares[-1]
        for model in benchmarks + networks:
            ma, mb = curve[model]["mean"][-2], curve[model]["mean"][-1]
            doublings = math.log2(curve[model]["n_train"][-1] / curve[model]["n_train"][-2])
            last[model] = {"from_share": a, "to_share": b, "gain": mb - ma, "per_doubling": (mb - ma) / doublings}
    return {"metric": metric, "curve": curve, "gaps": gaps, "per_doubling": per_doubling, "last_doubling": last}


# --- figure --------------------------------------------------------------------------------------
def learning_curve_figure(section: dict[str, Any], style: dict[str, Any]):
    """Left: test AUC against training N (log scale), SD bands on the networks. Right: each network minus the best benchmark."""
    from matplotlib.figure import Figure

    summary = section["summary"]
    curve, gaps = summary["curve"], summary["gaps"]
    colors = style["colors"]
    fig = Figure(figsize=(style["figsize"][0], style["figsize"][1] * 0.55), dpi=style["dpi"])
    axes = fig.subplots(1, 2)
    networks = list(gaps)
    benchmarks = [m for m in curve if m not in networks]
    # Networks take the palette; benchmarks are greys told apart by marker.
    color_of = {net: colors[i % len(colors)] for i, net in enumerate(networks)}
    greys = ["#4d4d4d", "#7a7a7a", "#a3a3a3", "#c4c4c4"]
    markers = ["s", "^", "D", "v"]
    for i, model in enumerate(benchmarks):
        c = curve[model]
        axes[0].plot(c["n_train"], c["mean"], marker=markers[i % len(markers)], color=greys[i % len(greys)],
                     linewidth=1.2, linestyle="--", label=model)
    for net in networks:
        c = curve[net]
        n, mean, sd = np.array(c["n_train"]), np.array(c["mean"]), np.array(c["sd"])
        axes[0].plot(n, mean, marker="o", color=color_of[net], linewidth=2.2, label=net)
        axes[0].fill_between(n, mean - sd, mean + sd, color=color_of[net], alpha=0.12)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("training people (log scale)", fontsize=style["label_fs"])
    axes[0].set_ylabel(f"test {summary['metric'].upper()}", fontsize=style["label_fs"])
    axes[0].legend(fontsize=style["legend_fs"])
    for i, net in enumerate(networks):
        if "best" not in gaps[net]:
            continue
        g = gaps[net]["best"]
        n = np.array(curve[net]["n_train"])
        mean, se = np.array(g["mean"]), np.array(g["se"])
        color = color_of[net]
        slope = g["per_doubling"]
        axes[1].errorbar(n, mean, yerr=1.96 * se, marker="o", color=color, capsize=3,
                         label=f"{net} ({slope['slope']:+.4f} per doubling)")
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("training people (log scale)", fontsize=style["label_fs"])
    axes[1].set_ylabel("network minus best benchmark (paired)", fontsize=style["label_fs"])
    axes[1].legend(fontsize=style["legend_fs"])
    title = section["dataset"].get("title", section["dataset"]["condition"])
    ages = section["dataset"]["status_ages"]
    fig.suptitle(f"Learning curve: {title} {ages[0]}-{ages[1]} (mean over folds and repeats; bands: network SD; bars: 95% paired)",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
