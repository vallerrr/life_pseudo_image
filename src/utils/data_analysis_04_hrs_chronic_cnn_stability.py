"""Are the attribution maps of analysis 04 stable, and do they point at what the model uses? (D31)

Owned by ``src/data_analysis/04_hrs_chronic_cnn.py stability``. HRS has no true map, so
each map is tested against itself and against other models, never against a truth:

* **Seeds.** The same network refitted with ``n_seeds`` initialisations on the same fold.
  The floor: a map that moves this much between seeds can move this much for no reason.
* **Folds.** The seed-0 maps of different outer folds: other training people, other
  explained people. Compared with the seed floor.
* **Row order** (spec §6). The rows are permuted within each domain, the network refitted,
  and its map put back in the original order. A model that reads the image as an
  unordered set of rows (the RNN, a 1-row kernel) should move no more than between seeds;
  a taller kernel may move more, and that excess is the cost of treating row order as
  spatial.
* **Across models.** Every network's map against the others' and against LightGBM's
  TreeSHAP on the cells -- an attribution from a different model family and method.
* **Faithfulness.** Each explained person's top-k cells by |Expected Gradients| are set
  to the training mean (0 after standardisation, the mask kept) and the test AUC and the
  mean |change in P(case)| recorded, against the same number of random observed cells.
  A map that points at what the model uses hurts it more than chance does.

A map here is the population map of |attribution|: per cell, the mean over the explained
people who observed it, null under ``min_cell`` observers -- what the run reports already.
Maps are compared by Spearman over the cells both have, by the Jaccard overlap of their
top ``top_share`` cells, and by Spearman of their row (variable) and column (age) profiles.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from utils.data_analysis_02_pseudo_image_cnn_model import expected_gradients
from utils.data_analysis_04_hrs_chronic_cnn import (
    _fit_network,
    grad_cam,
    household_folds,
    predict,
    seed_for,
    tensors,
    tune,
)
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import _seen_columns, cell_features, tune_lightgbm
from utils.shared_hrs_chronic import ChronicDataset


# --- maps and their comparison -----------------------------------------------------------------
def population_abs_map(per_person: np.ndarray, observed: np.ndarray, min_cell: int) -> np.ndarray:
    """Per-cell mean |attribution| over the people who observed it; NaN under ``min_cell``."""
    count = observed.sum(axis=0)
    total = (np.abs(per_person) * observed).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_cell, total / np.maximum(count, 1), np.nan)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 3 or np.ptp(a[both]) == 0 or np.ptp(b[both]) == 0:
        return float("nan")
    return float(spearmanr(a[both], b[both]).statistic)


def compare_maps(a: np.ndarray, b: np.ndarray, top_share: float) -> dict[str, float]:
    """Cell Spearman, top-cell Jaccard, row-profile and column-profile Spearman."""
    both = np.isfinite(a) & np.isfinite(b)
    k = max(1, int(round(top_share * both.sum())))
    top_a = set(np.argsort(-np.where(both, a, -np.inf), axis=None)[:k])
    top_b = set(np.argsort(-np.where(both, b, -np.inf), axis=None)[:k])
    masked_a, masked_b = np.where(both, a, np.nan), np.where(both, b, np.nan)
    with np.errstate(invalid="ignore"):
        rows = (np.nanmean(masked_a, axis=1), np.nanmean(masked_b, axis=1))
        cols = (np.nanmean(masked_a, axis=0), np.nanmean(masked_b, axis=0))
    return {
        "cells": _spearman(a, b),
        "top_jaccard": len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else float("nan"),
        "rows": _spearman(*rows),
        "columns": _spearman(*cols),
    }


def mean_pairwise(maps: list[np.ndarray], top_share: float) -> dict[str, float]:
    pairs = [compare_maps(a, b, top_share) for a, b in itertools.combinations(maps, 2)]
    return {k: float(np.nanmean([p[k] for p in pairs])) for k in pairs[0]} if pairs else {}


# --- row order --------------------------------------------------------------------------------
def domain_permutation(domains: tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    """A row order that shuffles rows within each domain and keeps the domains in place."""
    order = np.arange(len(domains))
    for domain in dict.fromkeys(domains):
        rows = np.flatnonzero(np.array(domains) == domain)
        order[rows] = rng.permutation(rows)
    return order


def permuted(data: ChronicDataset, order: np.ndarray) -> ChronicDataset:
    """``data`` with its image rows in ``order``: row r of the result is row ``order[r]``."""
    return dataclasses.replace(
        data, X=data.X[:, order], M=data.M[:, order], Delta=data.Delta[:, order],
        channel_names=tuple(np.array(data.channel_names)[order]),
        channel_domains=tuple(np.array(data.channel_domains)[order]),
    )


def unpermute(map_: np.ndarray, order: np.ndarray) -> np.ndarray:
    back = np.empty_like(map_)
    back[order] = map_
    return back


# --- faithfulness -------------------------------------------------------------------------------
def deletion(model, image: torch.Tensor, static: torch.Tensor, y: np.ndarray, attribution: np.ndarray,
             observed: np.ndarray, ks: list[int], rng: np.random.Generator, draws: int,
             fill: np.ndarray | None = None) -> dict[str, Any]:
    """Test AUC and mean |change in P(case)| after setting k cells per person to the mean.

    ``fill`` is the training mean on the scaled axis, ``(C, T)`` or ``(C, 1)`` (the
    scaler's ``fill``); 0, the default, is that mean under a z-score.
    """
    n = len(y)
    base = predict(model, image, static)
    flat_obs = observed.reshape(n, -1)
    n_obs = flat_obs.sum(axis=1)

    def ablate(scores: np.ndarray, k: int) -> np.ndarray:
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
        chosen = (ranks < np.minimum(k, n_obs)[:, None]) & flat_obs
        ablated = image.clone()
        values = ablated[:, 0].reshape(n, -1)
        mean = np.broadcast_to(np.float32(0.0) if fill is None else fill, image.shape[2:]).reshape(1, -1)
        mask = torch.from_numpy(chosen)
        values[mask] = torch.from_numpy(np.ascontiguousarray(np.broadcast_to(mean, chosen.shape)))[mask].to(values.dtype)
        ablated[:, 0] = values.reshape(ablated[:, 0].shape)
        return predict(model, ablated, static)

    top_scores = np.where(flat_obs, np.abs(attribution).reshape(n, -1), -np.inf)
    out: dict[str, Any] = {"auc_full": float(roc_auc_score(y, base)), "k": list(ks), "top": [], "random": []}
    for k in ks:
        p_top = ablate(top_scores, k)
        p_random = [ablate(np.where(flat_obs, rng.random(flat_obs.shape), -np.inf), k) for _ in range(draws)]
        out["top"].append({"auc": float(roc_auc_score(y, p_top)), "mean_abs_change": float(np.abs(p_top - base).mean())})
        out["random"].append({
            "auc": float(np.mean([roc_auc_score(y, p) for p in p_random])),
            "mean_abs_change": float(np.mean([np.abs(p - base).mean() for p in p_random])),
        })
    return out


# --- one fold -------------------------------------------------------------------------------------
def stability_fold(data: ChronicDataset, fold: int, train: np.ndarray, test: np.ndarray, config: dict[str, Any],
                   min_cell: int) -> dict[str, Any]:
    """Every map this fold contributes, and the fold's faithfulness curves."""
    spec = config["stability"]
    seed = seed_for(data.condition, data.status_bin, f"stability{fold}")
    att = config["attribution"]
    rng = np.random.default_rng(seed)
    explain = np.sort(rng.choice(len(test), size=min(att["n_explain"], len(test)), replace=False))
    baseline_rows = rng.choice(len(train), size=min(att["n_baselines"], len(train)), replace=False)
    observed = data.M[test][explain]
    y_explain = data.y[test][explain]
    orders = [domain_permutation(data.channel_domains, np.random.default_rng(seed + 100 + p))
              for p in range(spec["permutations"])]

    out: dict[str, Any] = {"maps": {}, "faithfulness": {}, "hyperparameters": {}, "seconds": {}}
    for n, network in enumerate(spec["networks"]):
        start = time.perf_counter()
        net_seed = seed + 1000 * (n + 1)
        hyper, _ = tune(data, train, network, config, net_seed)
        out["hyperparameters"][network] = hyper

        def maps_of(model, scaler, source: ChronicDataset) -> dict[str, np.ndarray]:
            image, static, _ = tensors(source, test, scaler)
            baselines = tensors(source, train, scaler)[0][baseline_rows]
            eg = expected_gradients(model, image[explain], static[explain], baselines, att["n_samples"], net_seed + 3,
                                    target=torch.sigmoid)
            found = {"eg": eg}
            if hasattr(model, "features"):
                found["gradcam"] = grad_cam(model, image[explain], static[explain])
            return found

        for r in range(spec["seeds"]):
            model, scaler, _ = _fit_network(data, train, network, hyper, config, net_seed + 10 * (r + 1))
            found = maps_of(model, scaler, data)
            for kind, per_person in found.items():
                out["maps"][f"{network}/{kind}/seed{r}"] = population_abs_map(per_person, observed, min_cell)
            if r == 0:
                image, static, _ = tensors(data, test, scaler)
                out["faithfulness"][network] = deletion(
                    model, image[explain], static[explain], y_explain, found["eg"], observed,
                    spec["deletion_k"], np.random.default_rng(net_seed + 5), spec["random_draws"], scaler.image.fill,
                )
        for p, order in enumerate(orders):
            shuffled = permuted(data, order)
            model, scaler, _ = _fit_network(shuffled, train, network, hyper, config, net_seed + 10)  # seed 0's init
            observed_p = shuffled.M[test][explain]
            for kind, per_person in maps_of(model, scaler, shuffled).items():
                out["maps"][f"{network}/{kind}/perm{p}"] = unpermute(
                    population_abs_map(per_person, observed_p, min_cell), order)
        out["seconds"][network] = time.perf_counter() - start

    # LightGBM's own attribution: TreeSHAP on the cells, the same explained people.
    start = time.perf_counter()
    inner = household_folds(data.y[train], data.household[train], config["inner_folds"], seed + 7)
    cells_train = cell_features(data, train)
    seen = _seen_columns(cells_train)
    model, _, _ = tune_lightgbm(cells_train[:, seen], data.y[train], inner, config, seed)
    contributions = model.predict(cell_features(data, test)[explain][:, seen], pred_contrib=True)  # log-odds SHAP
    full = np.zeros((len(explain), seen.size))
    full[:, seen] = contributions[:, :-1]  # the last column is the expected value
    C, K = data.X.shape[1:]
    shap = full[:, : C * K].reshape(len(explain), C, K)
    out["maps"]["lightgbm_cells/shap/seed0"] = population_abs_map(shap, observed, min_cell)
    out["seconds"]["lightgbm_cells"] = time.perf_counter() - start
    return out


def summarise(folds: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Seed, fold, permutation and cross-model similarities, averaged over folds."""
    spec = config["stability"]
    top = spec["top_share"]
    networks = list(spec["networks"])
    kinds = {net: sorted({k.split("/")[1] for f in folds for k in f["maps"] if k.startswith(f"{net}/")}) for net in networks}
    result: dict[str, Any] = {"seed": {}, "fold": {}, "permutation": {}, "cross_model": {}, "faithfulness": {}}
    for net in networks:
        for kind in kinds[net]:
            key = f"{net}/{kind}"
            seeds = [mean_pairwise([f["maps"][f"{key}/seed{r}"] for r in range(spec["seeds"])], top) for f in folds]
            result["seed"][key] = {m: float(np.nanmean([s[m] for s in seeds])) for m in seeds[0]}
            result["fold"][key] = mean_pairwise([f["maps"][f"{key}/seed0"] for f in folds], top)
            perms = [compare_maps(f["maps"][f"{key}/seed0"], f["maps"][f"{key}/perm{p}"], top)
                     for f in folds for p in range(spec["permutations"])]
            result["permutation"][key] = {m: float(np.nanmean([s[m] for s in perms])) for m in perms[0]}
    # Across models: Expected Gradients of every network, Grad-CAM of the CNNs, LightGBM's SHAP.
    references = [f"{net}/eg" for net in networks] + [f"{net}/gradcam" for net in networks if "gradcam" in kinds[net]]
    references.append("lightgbm_cells/shap")
    for a, b in itertools.combinations(references, 2):
        pairs = [compare_maps(f["maps"][f"{a}/seed0"], f["maps"][f"{b}/seed0"], top) for f in folds]
        result["cross_model"][f"{a} ~ {b}"] = {m: float(np.nanmean([p[m] for p in pairs])) for m in pairs[0]}
    for net in networks:
        curves = [f["faithfulness"][net] for f in folds]
        result["faithfulness"][net] = {
            "k": curves[0]["k"],
            "auc_full": float(np.mean([c["auc_full"] for c in curves])),
            **{f"{which}_{metric}": [float(np.mean([c[which][i][metric] for c in curves])) for i in range(len(curves[0]["k"]))]
               for which in ("top", "random") for metric in ("auc", "mean_abs_change")},
        }
    return result


# --- figures --------------------------------------------------------------------------------------
def stability_figures(section: dict[str, Any], meta: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
    """Similarity bars, faithfulness curves, the cross-model matrix and the fold-mean maps."""
    from matplotlib.figure import Figure

    summary = section["summary"]
    figs: dict[str, Any] = {}
    colors = style["colors"]

    keys = list(summary["seed"])
    fig = Figure(figsize=(style["figsize"][0], style["figsize"][1] * 0.55), dpi=style["dpi"])
    axes = fig.subplots(1, 2)
    for ax, metric, label in ((axes[0], "cells", "Spearman over cells"), (axes[1], "top_jaccard", "top-10% cell overlap (Jaccard)")):
        width = 0.27
        for j, (test, color) in enumerate((("seed", colors[0]), ("fold", colors[1]), ("permutation", colors[3]))):
            values = [summary[test][k][metric] for k in keys]
            ax.bar(np.arange(len(keys)) + (j - 1) * width, values, width, color=color, label=test)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=40, ha="right", fontsize=style["tick_fs"])
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=style["label_fs"], loc="left")
    axes[0].legend(fontsize=style["legend_fs"])
    fig.suptitle("How much a map moves: between seeds (floor), between folds, under within-domain row shuffles",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs["similarity"] = fig

    faith = summary["faithfulness"]
    fig = Figure(figsize=(style["figsize"][0], style["figsize"][1] * 0.55), dpi=style["dpi"])
    axes = fig.subplots(1, 2)
    for i, (net, curve) in enumerate(faith.items()):
        color = colors[i % len(colors)]
        for ax, metric in ((axes[0], "auc"), (axes[1], "mean_abs_change")):
            ax.plot(curve["k"], curve[f"top_{metric}"], marker="o", color=color, label=f"{net}: top cells")
            ax.plot(curve["k"], curve[f"random_{metric}"], marker="o", color=color, linestyle="--", alpha=0.6,
                    label=f"{net}: random cells")
    axes[0].set_ylabel("test AUC after ablation", fontsize=style["label_fs"])
    axes[1].set_ylabel("mean |change in P(case)|", fontsize=style["label_fs"])
    for ax in axes:
        ax.set_xlabel("cells per person set to the mean", fontsize=style["label_fs"])
    axes[1].legend(fontsize=style["legend_fs"] - 1, ncol=2)
    fig.suptitle("Faithfulness: ablating each person's top-attributed cells vs as many random observed cells",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs["faithfulness"] = fig

    cross = summary["cross_model"]
    names = list(dict.fromkeys([n for pair in cross for n in pair.split(" ~ ")]))
    matrix = np.eye(len(names))
    for pair, value in cross.items():
        a, b = pair.split(" ~ ")
        matrix[names.index(a), names.index(b)] = matrix[names.index(b), names.index(a)] = value["cells"]
    fig = Figure(figsize=(7.5, 6.5), dpi=style["dpi"])
    ax = fig.subplots()
    mesh = ax.imshow(matrix, cmap=style["diverging_cmap"], vmin=-1, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=style["tick_fs"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=style["tick_fs"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=style["tick_fs"])
    fig.colorbar(mesh, ax=ax, fraction=0.04)
    ax.set_title("Cross-model agreement: Spearman over cells (seed-0 maps, mean over folds)",
                 fontsize=style["label_fs"], loc="left")
    fig.tight_layout()
    figs["cross_model"] = fig

    maps = section["maps"]
    names_rows, domains = meta["channel_names"], meta["channel_domains"]
    boundaries = [i - 0.5 for i in range(1, len(domains)) if domains[i] != domains[i - 1]]
    ages = [50 + 2 * b for b in range(section["dataset"]["image_bins"])]
    keys = list(maps)
    cols = 4
    rows = -(-len(keys) // cols)
    fig = Figure(figsize=(style["figsize"][0] * 1.1, 4.4 * rows), dpi=style["dpi"])
    axes = np.atleast_2d(fig.subplots(rows, cols))
    for ax in axes.ravel()[len(keys):]:
        ax.set_axis_off()
    for i, (ax, key) in enumerate(zip(axes.ravel(), keys)):
        values = np.array([[np.nan if v is None else v for v in r] for r in maps[key]], dtype=float)
        ax.imshow(values, aspect="auto", interpolation="nearest", cmap=style["map_cmap"], vmin=0)
        ax.set_title(key, fontsize=style["label_fs"], loc="left")
        ax.set_xticks(range(0, len(ages), 3))
        ax.set_xticklabels([str(a) for a in ages[::3]], fontsize=style["tick_fs"])
        ax.set_yticks(range(len(names_rows)) if i % cols == 0 else [])
        if i % cols == 0:
            ax.set_yticklabels(names_rows, fontsize=style["tick_fs"] - 2)
        for boundary in boundaries:
            ax.axhline(boundary, color=style["separator_color"], linewidth=0.8)
    fig.suptitle("Fold-mean |attribution| maps (seed 0), each on its own scale", fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figs["maps"] = fig
    return figs
