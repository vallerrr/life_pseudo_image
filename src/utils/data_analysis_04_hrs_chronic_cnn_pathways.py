"""Do cases' individual EG maps fall into stable pathways? (D35)

Owned by ``src/data_analysis/04_hrs_chronic_cnn.py pathways``. The same outer folds,
seeds and fits as a ``run`` (and as D34's ``subgroups``) give every test case an
out-of-fold Expected Gradients map from each network. Each case's map is reduced to a
*pathway vector*: the signed attribution summed over ages within each image row (or
domain), divided by the case's total absolute attribution, so it says which rows pushed
that case's risk up or down and by what share, whatever the case's overall risk. ``drop_rows`` leaves rows out of the vector and its
total (D36: BMI, so the smaller rows can separate once it no longer dominates).

The vectors are clustered with k-means for each ``k`` in the grid. A cluster is a
pathway only if it recurs:

* **seeds** - each fold's networks are refitted with other seeds (same tuned
  hyperparameters, same baselines); the same cases' new maps are clustered again;
* **bootstrap** - k-means refitted on resampled cases, every case assigned to the
  nearest new centroid;
* **folds** - each test fold's cases (one fold's model) are clustered on their own and
  their centroids matched to the pooled ones.

Seed and bootstrap recurrence is each cluster's clusterwise Jaccard (Hennig 2007): the
best overlap with any replicate cluster, averaged over replicates. Fold recurrence is
the cosine of the matched centroid, averaged over folds. The chosen ``k`` is the largest
at which every cluster passes all three; none passing means no stable pathway. Each
cluster is profiled as D34's quintiles are (row means, statics, observed share, mean
life image, mean EG maps). Per-person maps, vectors and labels stay on the machine.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from utils.data_analysis_02_pseudo_image_cnn_model import expected_gradients
from utils.data_analysis_04_hrs_chronic_cnn import _fit_network, tensors
from utils.data_analysis_04_hrs_chronic_cnn_subgroups import (
    _feature_matrix,
    _label,
    _title,
    cohort_z,
    concordance,
    fold_predictions,
    person_features,
    profile,
)
from utils.shared_hrs_chronic import ChronicDataset, bin_ages

SEED_STRIDE = 10_000  # replicate seeds sit far from the run's own (net_seed + 1, + 3)


# --- per fold: maps from the run's fits and from refits with other seeds (stay local) ----------------
def fold_maps(data: ChronicDataset, train: np.ndarray, test: np.ndarray, config: dict[str, Any],
              seed: int) -> dict[str, Any]:
    """The run's predictions and EG maps on ``test``, plus each network's cases' maps under other seeds."""
    spec = config["pathways"]
    found = fold_predictions(data, train, test, config, seed, benchmarks=spec["benchmarks"])
    att = config["attribution"]
    cases = test[data.y[test] == 1]
    found["eg_seeds"] = {}
    for n, network in enumerate(config["networks"]):
        net_seed = seed + 1000 * n
        found["eg_seeds"][network] = []
        for s in range(1, spec["seeds"] + 1):
            model, scaler, _ = _fit_network(data, train, network, found["hyperparameters"][network], config,
                                            net_seed + 1 + SEED_STRIDE * s)
            image, static_t, _ = tensors(data, cases, scaler)
            baselines = tensors(data, train, scaler)[0][found["baseline_rows"]]
            found["eg_seeds"][network].append(expected_gradients(
                model, image, static_t, baselines, att["n_samples"], net_seed + 3 + SEED_STRIDE * s,
                target=torch.sigmoid))
    return found


# --- pathway vectors and clustering ------------------------------------------------------------------
def pathway_vectors(eg: np.ndarray, M: np.ndarray, groups: list[str] | None = None,
                    rows_kept: list[int] | None = None) -> np.ndarray:
    """Per person, signed EG per row (or per group of rows) as a share of the person's total |EG|.

    ``rows_kept`` restricts both the vector and the total to those rows, so a dropped row
    (e.g. BMI) neither appears nor dilutes the shares of the rest; ``groups`` then labels
    the kept rows only.
    """
    v = eg * M
    if rows_kept is not None:
        v = v[:, rows_kept]
    rows = v.sum(axis=2)
    if groups is not None:
        names = list(dict.fromkeys(groups))
        rows = np.stack([rows[:, [i for i, g in enumerate(groups) if g == name]].sum(axis=1) for name in names], axis=1)
    total = np.abs(v).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total[:, None] > 0, rows / total[:, None], np.nan)


def kmeans(V: np.ndarray, k: int, seed: int, n_init: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels and centroids, clusters numbered from largest to smallest."""
    fit = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(V)
    order = np.argsort(-np.bincount(fit.labels_, minlength=k), kind="stable")
    relabel = np.empty(k, int)
    relabel[order] = np.arange(k)
    return relabel[fit.labels_], fit.cluster_centers_[order]


def nearest(V: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return ((V[:, None, :] - centroids[None]) ** 2).sum(axis=2).argmin(axis=1)


def clusterwise_jaccard(reference: np.ndarray, replicate: np.ndarray, k: int) -> np.ndarray:
    """For each reference cluster, its best Jaccard overlap with any replicate cluster."""
    out = np.zeros(k)
    for a in range(k):
        in_a = reference == a
        if not in_a.any():
            continue
        out[a] = max((in_a & (replicate == b)).sum() / (in_a | (replicate == b)).sum()
                     for b in np.unique(replicate))
    return out


def matched_cosine(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Cosine of each reference centroid with its Hungarian-matched centroid in ``other``."""
    a = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    b = other / np.linalg.norm(other, axis=1, keepdims=True)
    cos = a @ b.T
    rows, cols = linear_sum_assignment(-cos)
    out = np.full(len(reference), np.nan)
    out[rows] = cos[rows, cols]
    return out


def stability(V: np.ndarray, labels: np.ndarray, centroids: np.ndarray, V_seeds: list[np.ndarray],
              fold: np.ndarray, k: int, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """Seed, bootstrap and fold recurrence of each cluster in one solution."""
    rng = np.random.default_rng(seed)
    seeds = [clusterwise_jaccard(labels, kmeans(W, k, seed, spec["n_init"])[0], k) for W in V_seeds]
    boot = []
    for _ in range(spec["bootstrap"]):
        draw = rng.choice(len(V), size=len(V), replace=True)
        boot.append(clusterwise_jaccard(labels, nearest(V, kmeans(V[draw], k, seed, spec["n_init"])[1]), k))
    folds, fold_ari = [], []
    for f in np.unique(fold):
        m = fold == f
        own_labels, own = kmeans(V[m], k, seed, spec["n_init"])
        folds.append(matched_cosine(centroids, own))
        fold_ari.append(adjusted_rand_score(labels[m], own_labels))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no seed replicates
        out = {
            "seed_jaccard": np.mean(seeds, axis=0) if seeds else np.full(k, np.nan),
            "seed_ari": float(np.mean([adjusted_rand_score(labels, kmeans(W, k, seed, spec["n_init"])[0])
                                       for W in V_seeds])) if V_seeds else float("nan"),
            "bootstrap_jaccard": np.mean(boot, axis=0),
            "fold_cosine": np.nanmean(folds, axis=0),
            "fold_cosine_min": np.nanmin(folds, axis=0),
            "fold_ari": float(np.mean(fold_ari)),
        }
    thresholds = spec["stable"]
    out["stable"] = [bool((s >= thresholds["jaccard"] if np.isfinite(s) else True)
                          and b >= thresholds["jaccard"] and c >= thresholds["cosine"])
                     for s, b, c in zip(out["seed_jaccard"], out["bootstrap_jaccard"], out["fold_cosine"])]
    return out


def _cluster_name(centroid: np.ndarray, names: list[str], top: int = 2) -> str:
    """The rows with the largest |share|, signed: e.g. ``bmi+ wealth-``."""
    order = np.argsort(-np.abs(centroid))[:top]
    return " ".join(f"{names[i]}{'+' if centroid[i] > 0 else '-'}" for i in order)


# --- the whole condition --------------------------------------------------------------------------
def summarise(data: ChronicDataset, folds: list[tuple[np.ndarray, np.ndarray]], found: list[dict[str, Any]],
              config: dict[str, Any], report, domains: list[str]) -> dict[str, Any]:
    """Cluster cases' pathway vectors for every k and network; report recurrence and group means only."""
    spec = config["pathways"]
    networks = list(config["networks"])
    models = [*spec["benchmarks"], *networks]
    y, n = data.y.astype(int), data.n
    case = y == 1
    fold = np.full(n, -1)
    p = {m: np.empty(n) for m in models}
    conc = {m: np.empty(n) for m in models}
    eg = {net: np.zeros(data.M.shape) for net in networks}
    eg_seeds = {net: [np.zeros(data.M.shape) for _ in range(spec["seeds"])] for net in networks}
    for f, ((_, test), result) in enumerate(zip(folds, found)):
        fold[test] = f
        cases = test[case[test]]
        for m in models:
            p[m][test] = result["p"][m]
            conc[m][test] = concordance(y[test], result["p"][m])
        for net in networks:
            eg[net][test] = result["eg"][net]
            for s in range(spec["seeds"]):
                eg_seeds[net][s][cases] = result["eg_seeds"][net][s]

    dropped = [c for c in spec.get("drop_rows", []) if c in data.channel_names]
    rows_kept = [i for i, c in enumerate(data.channel_names) if c not in dropped]
    groups = [domains[i] for i in rows_kept] if spec["level"] == "domain" else None
    names = list(dict.fromkeys(groups)) if groups else [data.channel_names[i] for i in rows_kept]
    Z = cohort_z(data)
    features = person_features(data, Z, config["subgroups"])
    seed = int(spec["seed"])

    out: dict[str, Any] = {
        "models": models, "networks": networks, "level": spec["level"], "vector_names": names,
        "dropped_rows": dropped,
        "features": {"names": list(features),
                     "mean": [float(np.nanmean(v[case])) if np.isfinite(v[case]).sum() >= report.min_cell
                              else float("nan") for v in features.values()],
                     "sd": [float(np.nanstd(v[case])) if np.isfinite(v[case]).sum() >= report.min_cell
                            else float("nan") for v in features.values()]},
        "stable": spec["stable"], "k": list(spec["k"]),
        "cases": profile(case, features, Z, data.M, eg, conc, report),
        "by_k": {}, "chosen_k": {}, "cross_model": {},
    }
    labels_at: dict[tuple[str, int], np.ndarray] = {}
    for net in networks:
        V_all = pathway_vectors(eg[net], data.M, groups, rows_kept)
        keep = case & np.isfinite(V_all).all(axis=1)
        V = V_all[keep]
        V_seeds = [pathway_vectors(e, data.M, groups, rows_kept)[keep] for e in eg_seeds[net]]
        V_seeds = [np.nan_to_num(W) for W in V_seeds]
        out["by_k"][net] = {"n_cases": report.count(int(keep.sum())),
                            "mean_vector": V.mean(axis=0) if keep.sum() >= report.min_cell else None}
        chosen = None
        for k in spec["k"]:
            labels, centroids = kmeans(V, k, seed, spec["n_init"])
            full = np.full(n, -1)
            full[keep] = labels
            labels_at[(net, k)] = full
            total_ss = ((V - V.mean(axis=0)) ** 2).sum()
            within = sum(((V[labels == c] - centroids[c]) ** 2).sum() for c in range(k))
            sample = np.random.default_rng(seed).choice(len(V), size=min(len(V), spec["silhouette_n"]), replace=False)
            st = stability(V, labels, centroids, V_seeds, fold[keep], k, spec, seed)
            clusters = []
            for c in range(k):
                size = int((labels == c).sum())
                entry = {"n": report.count(size), "stable": st["stable"][c],
                         "seed_jaccard": st["seed_jaccard"][c], "bootstrap_jaccard": st["bootstrap_jaccard"][c],
                         "fold_cosine": st["fold_cosine"][c], "fold_cosine_min": st["fold_cosine_min"][c]}
                if size >= report.min_cell:
                    entry["centroid"] = centroids[c]
                    entry["name"] = _cluster_name(centroids[c], names)
                    entry["profile"] = profile(full == c, features, Z, data.M, eg, conc, report)
                clusters.append(entry)
            out["by_k"][net][str(k)] = {
                "explained": float(1 - within / total_ss) if total_ss > 0 else float("nan"),
                "silhouette": float(silhouette_score(V[sample], labels[sample])) if len(set(labels[sample])) > 1
                else float("nan"),
                "seed_ari": st["seed_ari"], "fold_ari": st["fold_ari"],
                "all_stable": all(st["stable"]) and all(e.get("centroid") is not None for e in clusters),
                "clusters": clusters,
            }
            if out["by_k"][net][str(k)]["all_stable"]:
                chosen = k
        out["chosen_k"][net] = chosen

    # does the CNN find the same groups of cases as the RNN?
    if len(networks) >= 2:
        a, b = networks[:2]
        for k in spec["k"]:
            la, lb = labels_at[(a, k)], labels_at[(b, k)]
            both = (la >= 0) & (lb >= 0)
            out["cross_model"][str(k)] = {"pair": [a, b], "ari": float(adjusted_rand_score(la[both], lb[both])),
                                          "jaccard": clusterwise_jaccard(la[both], lb[both], k)}
    return out


# --- figures ----------------------------------------------------------------------------------------
def _dropped(s: dict[str, Any]) -> str:
    return f", clustered without {', '.join(s['dropped_rows'])}" if s.get("dropped_rows") else ""


def _shown_k(s: dict[str, Any], net: str) -> int:
    return s["chosen_k"].get(net) or min(s["k"])


def stability_figure(section: dict[str, Any], style: dict[str, Any]):
    """Per network and k: the weakest cluster's seed, bootstrap and fold recurrence, and the silhouette."""
    from matplotlib.figure import Figure

    s = section["summary"]
    fig = Figure(figsize=(style["figsize"][0] * 0.9, style["figsize"][1] * 0.6), dpi=style["dpi"])
    axes = fig.subplots(1, len(s["networks"]), sharey=True, squeeze=False)[0]
    colors = style["colors"]
    for ax, net in zip(axes, s["networks"]):
        ks = s["k"]
        rows = [s["by_k"][net][str(k)] for k in ks]

        def weakest(key: str) -> list[float]:
            return [min((c[key] for c in r["clusters"] if c[key] is not None), default=np.nan) for r in rows]

        for i, (key, label) in enumerate((("seed_jaccard", "seed Jaccard (weakest cluster)"),
                                          ("bootstrap_jaccard", "bootstrap Jaccard (weakest)"),
                                          ("fold_cosine", "fold centroid cosine (weakest)"))):
            ax.plot(ks, weakest(key), "o-", color=colors[i % len(colors)], label=label)
        ax.plot(ks, [r["silhouette"] for r in rows], "s--", color="#888888", label="silhouette")
        ax.axhline(s["stable"]["jaccard"], color=colors[0], linewidth=0.6, linestyle=":")
        ax.axhline(s["stable"]["cosine"], color=colors[2 % len(colors)], linewidth=0.6, linestyle=":")
        ax.set_xticks(ks)
        ax.set_xlabel("k", fontsize=style["label_fs"])
        chosen = s["chosen_k"].get(net)
        ax.set_title(f"{net}: chosen k = {chosen if chosen else 'none stable'}", fontsize=style["label_fs"])
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=style["legend_fs"])
    fig.suptitle(f"Pathway recurrence: {_title(section)} cases{_dropped(s)} (dotted: thresholds)",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def centroid_figure(section: dict[str, Any], style: dict[str, Any]):
    """Each cluster's mean pathway vector (signed share of |EG| per row) at the chosen k."""
    from matplotlib.figure import Figure

    s = section["summary"]
    names = s["vector_names"]
    fig = Figure(figsize=(style["figsize"][0] * 0.9, max(4, 0.2 * len(names) + 1.5)), dpi=style["dpi"])
    axes = fig.subplots(1, len(s["networks"]), sharey=True, squeeze=False)[0]
    for ax, net in zip(axes, s["networks"]):
        k = _shown_k(s, net)
        clusters = s["by_k"][net][str(k)]["clusters"]
        values = np.array([c["centroid"] if c.get("centroid") is not None else [np.nan] * len(names)
                           for c in clusters], float).T
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        mesh = ax.imshow(values, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(k), [f"{c + 1}{'' if cl['stable'] else '*'}\n(n {cl['n']})" for c, cl in enumerate(clusters)],
                      fontsize=style["legend_fs"])
        ax.set_title(f"{net}, k = {k}", fontsize=style["label_fs"])
        fig.colorbar(mesh, ax=ax, fraction=0.04, label="signed share of the case's |EG|")
    axes[0].set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=style["legend_fs"])
    fig.suptitle(f"Pathways: {_title(section)} cases{_dropped(s)}, mean pathway vector per cluster (* not stable)",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def profile_figure(section: dict[str, Any], style: dict[str, Any]):
    """How each cluster differs from all cases on every descriptor, at the chosen k."""
    from matplotlib.figure import Figure

    s = section["summary"]
    names, sd = s["features"]["names"], np.array(s["features"]["sd"], float)
    fig = Figure(figsize=(style["figsize"][0] * 0.9, style["figsize"][1] * 1.1), dpi=style["dpi"])
    axes = fig.subplots(1, len(s["networks"]), sharey=True, squeeze=False)[0]
    for ax, net in zip(axes, s["networks"]):
        k = _shown_k(s, net)
        clusters = s["by_k"][net][str(k)]["clusters"]
        shown = {c: cl["profile"] for c, cl in enumerate(clusters) if cl.get("profile", {}).get("features")}
        values = _feature_matrix(shown, names, s["cases"], sd)
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        mesh = ax.imshow(values, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(len(shown)), [f"{c + 1}: {clusters[c]['name']}" for c in shown],
                      fontsize=style["legend_fs"] * 0.85, rotation=30, ha="right")
        ax.set_title(f"{net}, k = {k}", fontsize=style["label_fs"])
        fig.colorbar(mesh, ax=ax, fraction=0.04, label="minus all cases (cohort SD)")
    axes[0].set_yticks(range(len(names)), [_label(n) for n in names], fontsize=style["legend_fs"])
    fig.suptitle(f"Who is on each pathway: {_title(section)} cases{_dropped(s)}", fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def map_figure(section: dict[str, Any], channel_names: list[str], style: dict[str, Any], net: str):
    """One network's mean signed EG map per cluster at the chosen k: rows x ages."""
    from matplotlib.figure import Figure

    s = section["summary"]
    k = _shown_k(s, net)
    clusters = s["by_k"][net][str(k)]["clusters"]
    shown = {c: np.array(cl["profile"]["eg"][net]["signed"], float) for c, cl in enumerate(clusters)
             if cl.get("profile", {}).get("eg")}
    finite = [m for m in shown.values() if np.isfinite(m).any()]
    lim = max(np.nanmax(np.abs(m)) for m in finite) if finite else 1
    ages = [bin_ages(b + 1)[0] for b in range(section["dataset"]["image_bins"])]
    fig = Figure(figsize=(max(7, style["figsize"][0] * 0.3 * len(shown) + 2.5), max(4, 0.2 * len(channel_names) + 1.5)),
                 dpi=style["dpi"], layout="constrained")
    axes = fig.subplots(1, max(1, len(shown)), sharey=True, squeeze=False)[0]
    for ax, (c, m) in zip(axes, shown.items()):
        mesh = ax.imshow(m, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(len(ages)), [str(a) for a in ages], fontsize=style["legend_fs"] * 0.8, rotation=90)
        ax.set_title(f"{c + 1}: {clusters[c]['name']}\n(n {clusters[c]['n']}{'' if clusters[c]['stable'] else ', not stable'})",
                     fontsize=style["legend_fs"])
    axes[0].set_yticks(range(len(channel_names)), [n.replace("_", " ") for n in channel_names],
                       fontsize=style["legend_fs"])
    if shown:
        fig.colorbar(mesh, ax=list(axes), fraction=0.02, label="mean signed EG on P(case), observed cells")
    axes[0].set_xlabel("age at bin start", fontsize=style["legend_fs"])
    fig.suptitle(f"{net} mean EG per pathway: {_title(section)}{_dropped(s)}, k = {k}", fontsize=style["title_fs"])
    return fig
