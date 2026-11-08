"""Who is predictable, and is there a subgroup the networks predict better than LightGBM? (D34)

Owned by ``src/data_analysis/04_hrs_chronic_cnn.py subgroups``. One pass over the same
outer folds and seeds as a ``run``: the CNN, the RNN and the comparison benchmarks are
fitted on each training fold, and every test person gets an out-of-fold probability
from each model and an Expected Gradients map from each network. Those per-person
arrays stay on the machine; the report holds only group means over at least
``min_cell`` people (D22).

* **Predictability.** A person's *concordance* under a model is the share of the other
  class, in the same test fold, the model ranks them correctly against (ties count
  half); its mean over cases is the AUC. Cases and controls are each cut into
  quintiles of the mean concordance over ``consensus`` models (Q1 hardest, Q5 easiest),
  and each quintile is profiled: its row means, statics, how much of the image it
  observed, its mean life image and its mean EG maps.
* **Subgroups.** Fixed in advance from covariates and from the static logistic's risk,
  never from a compared model's errors (selecting on LightGBM's misses would favour any
  other model by regression to the mean). Per subgroup and fold, each model's AUC; per
  network, the fold-paired gap to each comparator with a t interval over folds and a
  Holm-adjusted p over the subgroups.
* **Disagreement.** Per fold, the ``top_share`` of people on whom a network and the
  reference benchmark disagree most in rank; split by class and by which model ranked
  them better, and profiled like the quintiles. Selecting on disagreement is symmetric
  in the two models, so it does not favour either.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from utils.data_analysis_02_pseudo_image_cnn_model import expected_gradients
from utils.data_analysis_04_hrs_chronic_cnn import _fit_network, household_folds, predict, tensors, tune
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import _last_values, fit_benchmarks
from utils.shared_hrs_chronic import ChronicDataset


# --- per fold: out-of-fold predictions and maps (stay local) ------------------------------------
def fold_predictions(data: ChronicDataset, train: np.ndarray, test: np.ndarray, config: dict[str, Any],
                     seed: int, benchmarks: list[str] | None = None) -> dict[str, Any]:
    """Every model's P(case) on ``test`` and each network's signed EG map of every test person."""
    benchmarks = config["subgroups"]["benchmarks"] if benchmarks is None else benchmarks
    inner = household_folds(data.y[train], data.household[train], config["inner_folds"], seed + 7)
    p, _, _ = fit_benchmarks(data, train, test, config, seed, inner, only=tuple(benchmarks))
    att = config["attribution"]
    rng = np.random.default_rng(seed + 2)
    rng.choice(len(test), size=min(att["n_explain"], len(test)), replace=False)  # as run_fold, so the baselines match
    baseline_rows = rng.choice(len(train), size=min(att["n_baselines"], len(train)), replace=False)
    eg, hyper = {}, {}
    for n, network in enumerate(config["networks"]):
        net_seed = seed + 1000 * n  # as run_fold: the same fits as the run
        hyper[network], _ = tune(data, train, network, config, net_seed)
        model, scaler, _ = _fit_network(data, train, network, hyper[network], config, net_seed + 1)
        image, static_t, _ = tensors(data, test, scaler)
        p[network] = predict(model, image, static_t)
        baselines = tensors(data, train, scaler)[0][baseline_rows]
        eg[network] = expected_gradients(model, image, static_t, baselines, att["n_samples"], net_seed + 3,
                                         target=torch.sigmoid)
    return {"p": p, "eg": eg, "hyperparameters": hyper, "baseline_rows": baseline_rows}


def concordance(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per person, the share of the other class ranked correctly against them, ties half."""
    cases, controls = np.sort(p[y == 1]), np.sort(p[y == 0])
    out = np.empty(len(y))
    pc, pn = p[y == 1], p[y == 0]
    out[y == 1] = (np.searchsorted(controls, pc, "left") + np.searchsorted(controls, pc, "right")) / 2 / len(controls)
    out[y == 0] = 1 - (np.searchsorted(cases, pn, "left") + np.searchsorted(cases, pn, "right")) / 2 / len(cases)
    return out


# --- per-person descriptors (stay local) -----------------------------------------------------------
def cohort_z(data: ChronicDataset) -> np.ndarray:
    """Each observed cell as SD from the cohort mean of that cell; NaN where unobserved."""
    X = np.where(data.M, data.X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a cell nobody observed
        mean, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        return (X - mean) / np.where(sd > 0, sd, 1.0)


def person_features(data: ChronicDataset, Z: np.ndarray, spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """One value per person per descriptor: row means of cohort z, raw statics, how much was observed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a row never observed
        row_means = np.nanmean(Z, axis=2)
    seen_bins = data.M.any(axis=1)
    leave_behind = [data.channel_names.index(c) for c in spec["leave_behind_rows"] if c in data.channel_names]
    features = {f"row:{c}": row_means[:, i] for i, c in enumerate(data.channel_names)}
    features |= {f"static:{s}": data.static[:, i].astype(float) for i, s in enumerate(data.static_names)}
    features |= {
        "observed:bins": seen_bins.sum(axis=1).astype(float),
        "observed:first_bin": np.where(seen_bins.any(axis=1), seen_bins.argmax(axis=1) + 1, np.nan).astype(float),
        "observed:leave_behind": data.M[:, leave_behind].any(axis=(1, 2)).astype(float) if leave_behind
        else np.full(data.n, np.nan),
    }
    return features


def profile(mask: np.ndarray, features: dict[str, np.ndarray], Z: np.ndarray, M: np.ndarray,
            eg: dict[str, np.ndarray], conc: dict[str, np.ndarray], report) -> dict[str, Any]:
    """Means over the people in ``mask``; any mean over fewer than ``min_cell`` is null."""
    k = report.min_cell

    def mean_of(v: np.ndarray) -> float:
        v = v[mask]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size >= k else float("nan")

    out: dict[str, Any] = {"n": report.count(int(mask.sum()))}
    if mask.sum() < k:
        return out
    observed = M[mask]
    out["features"] = {name: mean_of(v) for name, v in features.items()}
    out["concordance"] = {model: mean_of(c) for model, c in conc.items()}
    out["image_z"] = report.mean_map(np.nansum(Z[mask], axis=0), observed.sum(axis=0))
    out["eg"] = {}
    for net, values in eg.items():
        v = values[mask] * observed
        row_totals = np.abs(v).sum(axis=(0, 2))
        out["eg"][net] = {
            "signed": report.mean_map(v.sum(axis=0), observed.sum(axis=0)),
            "abs": report.mean_map(np.abs(v).sum(axis=0), observed.sum(axis=0)),
            "row_share": row_totals / row_totals.sum() if row_totals.sum() > 0 else np.full(len(row_totals), np.nan),
        }
    return out


# --- subgroups --------------------------------------------------------------------------------------
def _thirds(v: np.ndarray, prefix: str, labels=("low", "mid", "high")) -> dict[str, np.ndarray]:
    finite = np.isfinite(v)
    if finite.sum() < 3:
        return {}
    lo, hi = np.quantile(v[finite], [1 / 3, 2 / 3])
    return {f"{prefix}_{labels[0]}": finite & (v <= lo), f"{prefix}_{labels[1]}": finite & (v > lo) & (v <= hi),
            f"{prefix}_{labels[2]}": finite & (v > hi)}


def subgroup_masks(data: ChronicDataset, features: dict[str, np.ndarray], p_static: np.ndarray) -> dict[str, np.ndarray]:
    """The fixed subgroups, as (family/name) -> mask. Missing values fall in no group of a family."""
    s = {name: features[f"static:{name}"] for name in data.static_names}
    groups: dict[str, np.ndarray] = {}

    def add(family: str, masks: dict[str, np.ndarray]) -> None:
        groups.update({f"{family}/{name}": mask for name, mask in masks.items()})

    if "female" in s:
        add("sex", {"female": s["female"] == 1, "male": s["female"] == 0})
    if {"black", "other_race", "hispanic"} <= set(s):
        hisp = s["hispanic"] == 1
        add("race_ethnicity", {"white": (s["black"] == 0) & (s["other_race"] == 0) & (s["hispanic"] == 0),
                               "black": (s["black"] == 1) & ~hisp, "hispanic": hisp,
                               "other": (s["other_race"] == 1) & ~hisp})
    if "foreign_born" in s:
        add("birthplace", {"us_born": s["foreign_born"] == 0, "foreign_born": s["foreign_born"] == 1})
    if "education_years" in s:
        e = s["education_years"]
        add("education", {"under_12": e < 12, "12": e == 12, "13_15": (e > 12) & (e < 16), "16_plus": e >= 16})
    if "birth_year" in s:
        add("birth_cohort", _thirds(s["birth_year"], "born", ("earliest", "middle", "latest")))
    if "lower_occupation" in s:
        add("occupation", {"manual_service": s["lower_occupation"] == 1, "other": s["lower_occupation"] == 0})
    if "never_married" in s:
        add("marriage", {"never_married": s["never_married"] == 1, "ever_married": s["never_married"] == 0})
    if {"father_education", "mother_education"} <= set(s):
        missing = np.isnan(s["father_education"]) | np.isnan(s["mother_education"])
        add("parent_education", {"missing": missing, "known": ~missing})
    add("observed_bins", _thirds(features["observed:bins"], "bins"))
    first = features["observed:first_bin"]
    add("entry", {"by_56": first <= 3, "after_56": first > 3})
    lb = features["observed:leave_behind"]
    add("leave_behind", {"any": lb == 1, "none": lb == 0})
    channels = list(data.channel_names)
    last = _last_values(data.X, data.M)
    if "bmi" in channels:
        b = last[:, channels.index("bmi")]
        add("last_bmi", {"under_25": b < 25, "25_30": (b >= 25) & (b < 30), "30_plus": b >= 30, "missing": np.isnan(b)})
    if "wealth" in channels:
        add("last_wealth", _thirds(last[:, channels.index("wealth")], "wealth"))
    edges = np.quantile(p_static, np.linspace(0, 1, 6)[1:-1])
    quintile = np.searchsorted(edges, p_static, side="right")
    add("static_risk", {f"q{q + 1}": quintile == q for q in range(5)})
    return groups


def compare_subgroup(mask: np.ndarray, y: np.ndarray, p: dict[str, np.ndarray], fold: np.ndarray,
                     networks: list[str], comparators: list[str], min_cell: int) -> dict[str, Any]:
    """Each model's AUC in the subgroup (pooled and per fold) and each network's fold-paired gaps."""
    per_fold = []
    for f in np.unique(fold):
        m = mask & (fold == f)
        if y[m].sum() >= min_cell and (1 - y[m]).sum() >= min_cell:
            per_fold.append({model: roc_auc_score(y[m], p[model][m]) for model in p})
    out: dict[str, Any] = {"folds": len(per_fold), "auc": {}, "gaps": {}}
    if y[mask].sum() >= min_cell and (1 - y[mask]).sum() >= min_cell:
        out["auc"] = {model: roc_auc_score(y[mask], p[model][mask]) for model in p}
    for net in networks:
        out["gaps"][net] = {}
        for ref in comparators:
            d = np.array([f[net] - f[ref] for f in per_fold])
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = stats.t.ppf(0.975, len(d) - 1)
            p_value = 2 * stats.t.sf(abs(d.mean() / se), len(d) - 1) if se > 0 else float("nan")
            out["gaps"][net][ref] = {"mean": d.mean(), "se": se, "lo": d.mean() - t * se, "hi": d.mean() + t * se,
                                     "p": p_value}
    return out


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p values over the finite ones."""
    finite = sorted((v, k) for k, v in p_values.items() if np.isfinite(v))
    adjusted, running = {}, 0.0
    for i, (v, k) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - i) * v))
        adjusted[k] = running
    return adjusted


# --- disagreement -------------------------------------------------------------------------------
def disagreement(y: np.ndarray, p_net: np.ndarray, p_ref: np.ndarray, fold: np.ndarray,
                 top_share: float) -> dict[str, np.ndarray]:
    """Per fold, the ``top_share`` with the largest rank difference, by class and by which model ranked better."""
    top, net_better = np.zeros(len(y), bool), np.zeros(len(y), bool)
    for f in np.unique(fold):
        m = fold == f
        r_net = stats.rankdata(p_net[m]) / m.sum()
        r_ref = stats.rankdata(p_ref[m]) / m.sum()
        d = r_net - r_ref
        top[m] = np.abs(d) >= np.quantile(np.abs(d), 1 - top_share)
        net_better[m] = np.where(y[m] == 1, d > 0, d < 0)
    case = y == 1
    return {"cases_network_better": top & case & net_better, "cases_reference_better": top & case & ~net_better,
            "controls_network_better": top & ~case & net_better, "controls_reference_better": top & ~case & ~net_better}


# --- the whole condition --------------------------------------------------------------------------
def summarise(data: ChronicDataset, folds: list[tuple[np.ndarray, np.ndarray]], found: list[dict[str, Any]],
              config: dict[str, Any], report) -> dict[str, Any]:
    """Assemble the out-of-fold arrays and reduce them to group means (the only thing reported)."""
    spec = config["subgroups"]
    networks = list(config["networks"])
    models = [*spec["benchmarks"], *networks]
    y, n = data.y.astype(int), data.n
    fold = np.empty(n, int)
    p = {m: np.empty(n) for m in models}
    eg = {net: np.empty(data.M.shape) for net in networks}
    conc = {m: np.empty(n) for m in models}
    for f, ((_, test), result) in enumerate(zip(folds, found)):
        fold[test] = f
        for m in models:
            p[m][test] = result["p"][m]
            conc[m][test] = concordance(y[test], result["p"][m])
        for net in networks:
            eg[net][test] = result["eg"][net]

    Z = cohort_z(data)
    features = person_features(data, Z, spec)
    out: dict[str, Any] = {
        "models": models,
        "features": {"names": list(features),
                     "mean": [float(np.nanmean(v)) if np.isfinite(v).sum() >= report.min_cell else float("nan")
                              for v in features.values()],
                     "sd": [float(np.nanstd(v)) if np.isfinite(v).sum() >= report.min_cell else float("nan")
                            for v in features.values()]},
        "auc": {m: {"pooled": roc_auc_score(y, p[m]),
                    "fold_mean": float(np.mean([roc_auc_score(y[fold == f], p[m][fold == f]) for f in range(len(folds))]))}
                for m in models},
        "hyperparameters": [r["hyperparameters"] for r in found],
    }

    # 1. who is predictable, within each class
    consensus = np.mean([conc[m] for m in spec["consensus"]], axis=0)
    case = y == 1
    groups, agreement = {}, {}
    for label, cls in (("cases", case), ("controls", ~case)):
        cut = np.quantile(consensus[cls], np.linspace(0, 1, 6)[1:-1])
        q = np.searchsorted(cut, consensus, side="right")
        for k in range(5):
            groups[f"{label}_q{k + 1}"] = profile(cls & (q == k), features, Z, data.M, eg, conc, report)
        groups[label] = profile(cls, features, Z, data.M, eg, conc, report)
        agreement[label] = {a: {b: float(stats.spearmanr(conc[a][cls], conc[b][cls])[0]) for b in models} for a in models}
    out["predictability"] = {"consensus": spec["consensus"], "groups": groups, "agreement": agreement}

    # 2. subgroups where the networks may differ from the benchmarks
    masks = subgroup_masks(data, features, p["logistic_static"] if "logistic_static" in p else consensus)
    compared = {}
    for name, mask in masks.items():
        compared[name] = {"n": report.count(int(mask.sum())), "cases": report.count(int((mask & case).sum())),
                          **compare_subgroup(mask, y, p, fold, networks, spec["comparators"], report.min_cell)}
    for net in networks:
        for ref in spec["comparators"]:
            adjusted = holm({name: c["gaps"][net][ref]["p"] for name, c in compared.items() if ref in c["gaps"].get(net, {})})
            for name, value in adjusted.items():
                compared[name]["gaps"][net][ref]["p_holm"] = value
    out["subgroups"] = compared

    # 3. where a network and the reference disagree most
    ref = spec["reference"]
    out["disagreement"] = {"reference": ref, "top_share": spec["top_share"]}
    for net in networks:
        split = disagreement(y, p[net], p[ref], fold, spec["top_share"])
        out["disagreement"][net] = {
            "network_better_share": {
                label: float(split[f"{label}_network_better"].sum()
                             / max(1, split[f"{label}_network_better"].sum() + split[f"{label}_reference_better"].sum()))
                for label in ("cases", "controls")},
            "groups": {name: profile(mask, features, Z, data.M, eg, conc, report) for name, mask in split.items()},
        }
    return out


# --- figures ----------------------------------------------------------------------------------------
def _feature_matrix(groups: dict[str, Any], names: list[str], baseline: dict[str, Any], sd: np.ndarray) -> np.ndarray:
    """Each group's feature means minus the baseline group's, in cohort SD units."""
    base = np.array([baseline["features"][f] if baseline["features"][f] is not None else np.nan for f in names], float)
    cols = []
    for g in groups.values():
        vals = np.array([g.get("features", {}).get(f) if g.get("features", {}).get(f) is not None else np.nan
                         for f in names], float)
        cols.append((vals - base) / np.where(sd > 0, sd, np.nan))
    return np.array(cols).T


def _title(section: dict[str, Any]) -> str:
    d = section["dataset"]
    return f"{d.get('title', d['condition'])} {d['status_ages'][0]}-{d['status_ages'][1]}"


def _label(name: str) -> str:
    return name.split(":", 1)[1].replace("_", " ") + ("" if name.startswith("row:") else f" ({name.split(':')[0]})")


def predictability_figure(section: dict[str, Any], style: dict[str, Any]):
    """Per class: how each quintile of predictability differs from the class on every descriptor."""
    from matplotlib.figure import Figure

    s = section["summary"]
    names, sd = s["features"]["names"], np.array(s["features"]["sd"], float)
    groups = s["predictability"]["groups"]
    fig = Figure(figsize=(style["figsize"][0] * 0.9, style["figsize"][1] * 1.1), dpi=style["dpi"])
    axes = fig.subplots(1, 2, sharey=True)
    for ax, label in zip(axes, ("cases", "controls")):
        quint = {k: groups[f"{label}_q{k}"] for k in range(1, 6)}
        values = _feature_matrix(quint, names, groups[label], sd)
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        mesh = ax.imshow(values, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(5), [f"Q{k}" for k in range(1, 6)])
        conc = [quint[k].get("concordance", {}) for k in range(1, 6)]
        ax.set_title(f"{label}: Q1 hardest to Q5 easiest", fontsize=style["label_fs"])
        ax.set_xlabel("consensus concordance " + ", ".join(
            f"{np.mean(list(c.values())):.2f}" if c else "-" for c in conc), fontsize=style["legend_fs"])
        fig.colorbar(mesh, ax=ax, fraction=0.04, label="minus class mean (cohort SD)")
    axes[0].set_yticks(range(len(names)), [_label(n) for n in names], fontsize=style["legend_fs"])
    fig.suptitle(f"Who is predictable: {_title(section)} (row values: mean cohort z over observed ages)",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def subgroup_figure(section: dict[str, Any], style: dict[str, Any]):
    """Forest plot: each network minus the reference benchmark in every subgroup, 95% fold-paired intervals."""
    from matplotlib.figure import Figure

    s = section["summary"]
    ref = s["disagreement"]["reference"]
    names = list(s["subgroups"])
    networks = [m for m in s["disagreement"] if m not in ("reference", "top_share")]
    fig = Figure(figsize=(style["figsize"][0] * 0.7, max(4, 0.22 * len(names) + 1.5)), dpi=style["dpi"])
    ax = fig.subplots()
    colors = style["colors"]
    for i, net in enumerate(networks):
        offset = (i - (len(networks) - 1) / 2) * 0.25
        for j, name in enumerate(names):
            g = s["subgroups"][name]["gaps"].get(net, {}).get(ref)
            if not g:
                continue
            strong = g.get("p_holm") is not None and g["p_holm"] < 0.05
            ax.errorbar(g["mean"], j + offset, xerr=[[g["mean"] - g["lo"]], [g["hi"] - g["mean"]]], fmt="o",
                        color=colors[i % len(colors)], markersize=4 if not strong else 7, capsize=2,
                        label=net if j == 0 else None)
    ax.axvline(0, color="#888888", linewidth=0.8)
    overall = [s["auc"][net]["fold_mean"] - s["auc"][ref]["fold_mean"] for net in networks]
    for i, v in enumerate(overall):
        ax.axvline(v, color=colors[i % len(colors)], linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(names)), [f"{n}  (n {s['subgroups'][n]['n']})" for n in names], fontsize=style["legend_fs"])
    ax.invert_yaxis()
    ax.set_xlabel(f"AUC gap to {ref} within the subgroup (fold-paired 95% CI; dashed: whole sample)",
                  fontsize=style["label_fs"])
    ax.legend(fontsize=style["legend_fs"])
    fig.suptitle(f"Subgroups: {_title(section)} (large marker: Holm p < 0.05)", fontsize=style["title_fs"])
    fig.tight_layout()
    return fig


def disagreement_figure(section: dict[str, Any], style: dict[str, Any]):
    """Where a network and the reference disagree most: who each one ranks better, against the class."""
    from matplotlib.figure import Figure

    s = section["summary"]
    names, sd = s["features"]["names"], np.array(s["features"]["sd"], float)
    groups = s["predictability"]["groups"]
    d = s["disagreement"]
    networks = [m for m in d if m not in ("reference", "top_share")]
    fig = Figure(figsize=(style["figsize"][0] * 0.9, style["figsize"][1] * 1.1), dpi=style["dpi"])
    axes = fig.subplots(1, len(networks), sharey=True, squeeze=False)[0]
    for ax, net in zip(axes, networks):
        cols, labels = [], []
        for label in ("cases", "controls"):
            for who in ("network", "reference"):
                g = d[net]["groups"][f"{label}_{who}_better"]
                cols.append(_feature_matrix({0: g}, names, groups[label], sd)[:, 0])
                labels.append(f"{label}\n{net if who == 'network' else d['reference']} better\n(n {g['n']})")
        values = np.array(cols).T
        lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1
        mesh = ax.imshow(values, aspect="auto", cmap=style["diverging_cmap"], vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(len(labels)), labels, fontsize=style["legend_fs"] * 0.85)
        share = d[net]["network_better_share"]
        ax.set_title(f"{net} vs {d['reference']}: {net} ranks better for {share['cases']:.0%} of cases, "
                     f"{share['controls']:.0%} of controls", fontsize=style["legend_fs"])
        fig.colorbar(mesh, ax=ax, fraction=0.04, label="minus class mean (cohort SD)")
    axes[0].set_yticks(range(len(names)), [_label(n) for n in names], fontsize=style["legend_fs"])
    fig.suptitle(f"Largest disagreements (top {d['top_share']:.0%} per fold): {_title(section)}",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
