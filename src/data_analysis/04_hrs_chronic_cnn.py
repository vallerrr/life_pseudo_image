"""Analysis 04 - HRS outcomes at age A: the pseudo-image CNN, an RNN, and their maps.

``counts``, ``run`` and ``images`` read HRS, so the user runs them (D07, D22); each writes one
aggregate JSON report under ``output/reports/blind/``, and that file is what comes
back. ``figures`` reads only such a report, so anyone may run it.

Run:
    python src/data_analysis/04_hrs_chronic_cnn.py counts            # cases/controls per outcome x age
    python src/data_analysis/04_hrs_chronic_cnn.py run               # models, metrics, maps (profile full)
    python src/data_analysis/04_hrs_chronic_cnn.py run --conditions diabe --status-bin 11
    python src/data_analysis/04_hrs_chronic_cnn.py stability --profile stability   # map stability (D31)
    python src/data_analysis/04_hrs_chronic_cnn.py learning-curve --profile learning_curve   # AUC by training N (D32)
    python src/data_analysis/04_hrs_chronic_cnn.py run --profile scaling   # per-age vs per-variable image scaling (D33)
    python src/data_analysis/04_hrs_chronic_cnn.py subgroups --profile subgroups   # who is predictable; subgroups (D34)
    python src/data_analysis/04_hrs_chronic_cnn.py pathways --profile pathways     # clusters of cases' EG maps (D35)
    python src/data_analysis/04_hrs_chronic_cnn.py pathways --profile pathways_no_bmi   # the same, BMI row left out (D36)
    python src/data_analysis/04_hrs_chronic_cnn.py run --status-bin 9     # the run at other ages: 9 is 66-68 (D37)
    python src/data_analysis/04_hrs_chronic_cnn.py comparisons      # time order, inputs, stacking (D38-D40)
    python src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile bmi_trajectory   # BMI trajectory vs level (D41)
    python src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile ses_order        # socioeconomic order, changers only (D42-D43)
    python src/data_analysis/04_hrs_chronic_cnn.py comparisons --profile bmi_changers     # BMI trajectory, BMI changers only (D44)
    python src/data_analysis/04_hrs_chronic_cnn.py images            # average life image per outcome (D28)
    python src/data_analysis/04_hrs_chronic_cnn.py figures --report output/reports/blind/04_hrs_chronic_cnn_run_<stamp>.json
Add ``--rand-file tests/fixtures/randhrs_fixture.dta --profile smoke`` to try it on the fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap (conventions §2) -------------------------------------------
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P  # noqa: E402

P.bootstrap()

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from utils.data_analysis_04_hrs_chronic_cnn import (  # noqa: E402
    MapTotals,
    average_image,
    average_image_figure,
    household_folds,
    kernel_auc_figure,
    kernel_map_figure,
    map_figure,
    observed_share,
    run_fold,
    seed_for,
)
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import BENCHMARKS  # noqa: E402
from utils.shared_hrs_chronic import OUTCOMES, bin_ages, build_chronic_dataset, chronic_columns  # noqa: E402
from utils.shared_hrs_ingest import read_rand  # noqa: E402
from utils.shared_hrs_variables import CHANNEL_SETS, STATIC_SETS, channel_domains  # noqa: E402
from utils.shared_secure_report import SecureReport, guarded  # noqa: E402
from utils.shared_style import load_style, savedir  # noqa: E402

NOTEBOOK = "04_hrs_chronic_cnn"
REPLACED_BY_PROFILE = ("networks", "benchmarks")


def load_config(profile: str) -> dict[str, Any]:
    raw = yaml.safe_load(P.HRS_CHRONIC_CNN_CONFIG.read_text())
    profiles = raw.pop("profiles")
    if profile not in profiles:
        raise ValueError(f"profile must be one of {sorted(profiles)}, got {profile!r}")
    config = copy.deepcopy(raw)
    for key, value in (profiles[profile] or {}).items():
        # A profile's `networks` and `benchmarks` replace the defaults: an arm list that
        # merged would silently keep networks the profile did not ask for.
        merge = isinstance(value, dict) and key not in REPLACED_BY_PROFILE and isinstance(config.get(key), dict)
        config[key] = {**config[key], **value} if merge else value
    config["profile"] = profile
    return config


def dataset_section(report: SecureReport, raw: dict[str, Any]) -> dict[str, Any]:
    """The builder's raw counts, every head count through suppression."""
    return {
        **{k: raw[k] for k in ("condition", "label", "title", "death_horizon_years", "status_bin", "status_ages", "image_bins")},
        "n_file": report.count(raw["n_file"]),
        "n_kept": report.count(raw["n_kept"]),
        "excluded": report.counts(raw["excluded"]),
        "cases": report.count(raw["cases"]),
        "controls": report.count(raw["controls"]),
        "first_observed_bin": [report.count(v) for v in raw["first_observed_bin"]],
        "channel_set": raw["channel_set"],
        "static_set": raw["static_set"],
        "observed_fraction": raw["observed_fraction"],
        "static_missing_fraction": raw["static_missing_fraction"],
        "missing_codes": {name: report.counts(codes) for name, codes in raw["missing_codes"].items()},
    }


def counts(args, config: dict[str, Any]) -> None:
    report = SecureReport(f"{NOTEBOOK}_counts", source="hrs", out_dir=args.report_dir)
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(channel_set=config["channel_set"]))
        lo, hi = config["counts_bins"]
        table = {}
        for condition in OUTCOMES:
            for status_bin in range(lo, hi + 1):
                data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                             config["channel_set"], config["static_set"])
                ages = bin_ages(status_bin)
                table.setdefault(condition, {})[f"bin{status_bin}_ages{ages[0]}-{ages[1]}"] = {
                    "kept": report.count(data.n),
                    "cases": report.count(int(data.y.sum())),
                    "controls": report.count(int((data.y == 0).sum())),
                }
        report.add("counts", table)
        report.add("note", "kept = interviewed in the status bin, with an interview before it (D23); "
                           f"for death, also with vital status known {config['death_horizon_years']} years on (D25)")


def run(args, config: dict[str, Any]) -> None:
    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    report = SecureReport(f"{NOTEBOOK}_run", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    channels = CHANNEL_SETS[config["channel_set"]]
    report.add("meta/channel_names", [c.name for c in channels])
    report.add("meta/channel_domains", channel_domains(channels))
    report.add("meta/static_names", list(STATIC_SETS[config["static_set"]]))
    models = [*(BENCHMARKS if config.get("run_benchmarks", True) else ()), *config["networks"]]
    report.add("meta/models", models)
    report.add("meta/torch", torch.__version__)
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            maps = MapTotals.empty()
            per_fold = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                result = run_fold(data, fold, train, test, config, maps)
                result["n_test"] = report.count(len(test))
                result["test_cases"] = report.count(int(data.y[test].sum()))
                per_fold.append(result)
                aucs = ", ".join(f"{net} auc {result['metrics'][net]['auc']:.3f}" for net in config["networks"])
                print(f"{condition} fold {fold}: {aucs}, {time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/folds", per_fold)
            report.add(f"{section}/metrics_mean", {
                model: {m: float(np.mean([f["metrics"][model][m] for f in per_fold])) for m in per_fold[0]["metrics"][model]}
                for model in models
            })
            means = {name: report.mean_map(maps.sums[name], maps.counts[name]) for name in maps.sums}
            nested: dict[str, Any] = {}
            for name, value in means.items():
                network, kind, group = name.split("/")
                nested.setdefault(network, {}).setdefault(kind, {})[group] = value
            for kinds in nested.values():
                for groups in kinds.values():  # NaN (suppressed) stays null
                    groups["difference"] = np.array(groups["cases"], dtype=float) - np.array(groups["controls"], dtype=float)
            report.add(f"{section}/maps", nested)
            report.add(f"{section}/maps_observers", {name: [[report.count(v) for v in row] for row in maps.counts[name]]
                                                     for name in maps.counts if name.endswith("/all")})
            report.add(f"{section}/seconds", time.perf_counter() - start)


def stability(args, config: dict[str, Any]) -> None:
    """Map stability and faithfulness (D31): seeds, folds, row order, other models."""
    from utils.data_analysis_04_hrs_chronic_cnn_stability import stability_fold, summarise

    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    channels = CHANNEL_SETS[config["channel_set"]]
    report = SecureReport(f"{NOTEBOOK}_stability", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    report.add("meta/channel_names", [c.name for c in channels])
    report.add("meta/channel_domains", channel_domains(channels))
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            per_fold = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                per_fold.append(stability_fold(data, fold, train, test, config, report.min_cell))
                print(f"{condition} stability fold {fold}: {time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/summary", summarise(per_fold, config))
            keys = sorted({k.rsplit("/", 1)[0] for k in per_fold[0]["maps"]})
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a cell suppressed in every fold
                report.add(f"{section}/maps", {k: np.nanmean([f["maps"][f"{k}/seed0"] for f in per_fold], axis=0)
                                               for k in keys})
            report.add(f"{section}/fold_maps", {k: [f["maps"][f"{k}/seed0"] for f in per_fold] for k in keys})
            report.add(f"{section}/faithfulness_by_fold", [f["faithfulness"] for f in per_fold])
            report.add(f"{section}/hyperparameters", [f["hyperparameters"] for f in per_fold])
            report.add(f"{section}/seconds", time.perf_counter() - start)


def learning_curve(args, config: dict[str, Any]) -> None:
    """Every model refitted on nested household shares of each training fold (D32)."""
    from utils.data_analysis_04_hrs_chronic_cnn_learning_curve import learning_curve_fold, summarise

    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    report = SecureReport(f"{NOTEBOOK}_learning_curve", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    report.add("meta/models", [*config["learning_curve"]["benchmarks"], *config["networks"]])
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            records = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                found = learning_curve_fold(data, fold, train, test, config, seed_for(condition, status_bin, f"fold{fold}"))
                records.extend(found)
                full = next(r for r in found if r["share"] == max(x["share"] for x in found))
                aucs = ", ".join(f"{m} {v['auc']:.3f}" for m, v in full["metrics"].items())
                print(f"{condition} learning-curve fold {fold}: at {full['share']:.0%} {aucs}; "
                      f"{time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/summary", summarise(records, config))
            report.add(f"{section}/units", [
                {**r, "n_train": report.count(r["n_train"]), "train_cases": report.count(r["train_cases"])}
                for r in records
            ])
            report.add(f"{section}/seconds", time.perf_counter() - start)


def subgroups(args, config: dict[str, Any]) -> None:
    """Who is predictable, and subgroups where the networks differ from LightGBM (D34)."""
    from utils.data_analysis_04_hrs_chronic_cnn_subgroups import fold_predictions, summarise

    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    channels = CHANNEL_SETS[config["channel_set"]]
    report = SecureReport(f"{NOTEBOOK}_subgroups", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    report.add("meta/channel_names", [c.name for c in channels])
    report.add("meta/channel_domains", channel_domains(channels))
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            found = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                found.append(fold_predictions(data, train, test, config, seed_for(condition, status_bin, f"fold{fold}")))
                print(f"{condition} subgroups fold {fold}: {time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/summary", summarise(data, folds, found, config, report))
            report.add(f"{section}/seconds", time.perf_counter() - start)


def pathways(args, config: dict[str, Any]) -> None:
    """Do cases' individual EG maps fall into pathways that recur across seeds and folds? (D35)"""
    from utils.data_analysis_04_hrs_chronic_cnn_pathways import fold_maps, summarise

    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    channels = CHANNEL_SETS[config["channel_set"]]
    report = SecureReport(f"{NOTEBOOK}_pathways", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    report.add("meta/channel_names", [c.name for c in channels])
    report.add("meta/channel_domains", channel_domains(channels))
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            found = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                found.append(fold_maps(data, train, test, config, seed_for(condition, status_bin, f"fold{fold}")))
                print(f"{condition} pathways fold {fold}: {time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/summary", summarise(data, folds, found, config, report, channel_domains(channels)))
            report.add(f"{section}/seconds", time.perf_counter() - start)


def comparisons(args, config: dict[str, Any]) -> None:
    """Time order, inputs, stacking and rows' trajectories against the original fits, on the run's folds (D38-D42)."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import comparisons_fold, population, restrict, summarise

    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    report = SecureReport(f"{NOTEBOOK}_comparisons", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            start = time.perf_counter()
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            if config["comparisons"].get("population"):  # D43: only the people in the named strata
                keep = population(data, config["comparisons"])
                report.add(f"{section}/population", {"strata": config["comparisons"]["population"],
                                                     "n": report.count(keep.sum()), "cases": report.count(data.y[keep].sum()),
                                                     "controls": report.count((data.y[keep] == 0).sum())})
                data = restrict(data, keep)
            folds = household_folds(data.y, data.household, config["outer_folds"], seed_for(condition, status_bin, "outer"))
            per_fold = []
            for fold, (train, test) in enumerate(folds):
                fold_start = time.perf_counter()
                found = comparisons_fold(data, fold, train, test, config)
                per_fold.append(found)
                aucs = ", ".join(f"{k} {v['auc']:.3f}" for k, v in found["metrics"].items() if k.split("/")[1] in ("cnn", "rnn", "all"))
                print(f"{condition} comparisons fold {fold}: {aucs}; {time.perf_counter() - fold_start:.0f} s", flush=True)
            report.add(f"{section}/summary", summarise(per_fold, config))
            report.add(f"{section}/folds", [{k: f[k] for k in ("metrics", "hyperparameters", "seconds", "stack_weights")
                                             if k in f} for f in per_fold])
            report.add(f"{section}/seconds", time.perf_counter() - start)


def images(args, config: dict[str, Any]) -> None:
    """The average life image of cases and controls per outcome: means only, thin cells null (D28)."""
    conditions = args.conditions or config["conditions"]
    status_bin = args.status_bin or config["status_bin"]
    channels = CHANNEL_SETS[config["channel_set"]]
    report = SecureReport(f"{NOTEBOOK}_images", source="hrs", out_dir=args.report_dir)
    report.add("config", {**config, "conditions": conditions, "status_bin": status_bin})
    report.add("meta/channel_names", [c.name for c in channels])
    report.add("meta/channel_domains", channel_domains(channels))
    with guarded(report, debug=args.debug):
        frame = read_rand(args.rand_file, columns=chronic_columns(conditions, config["channel_set"]))
        for condition in conditions:
            data = build_chronic_dataset(frame, condition, status_bin, config["death_horizon_years"],
                                         config["channel_set"], config["static_set"])
            section = f"conditions/{condition}"
            report.add(f"{section}/dataset", dataset_section(report, data.report))
            totals = average_image(data, report.min_cell)
            means = {group: np.array(report.mean_map(totals[f"{group}/total"], totals[f"{group}/count"]), dtype=float)
                     for group in ("cases", "controls", "all")}
            report.add(f"{section}/images", {
                "min_cell": report.min_cell,
                "cases": means["cases"],
                "controls": means["controls"],
                "difference": means["cases"] - means["controls"],  # NaN (suppressed) stays null
                "observed_share": observed_share(totals["all/count"], int(totals["all/n"]), report.min_cell),
                "cases_observed_share": observed_share(totals["cases/count"], int(totals["cases/n"]), report.min_cell),
                "controls_observed_share": observed_share(totals["controls/count"], int(totals["controls/n"]),
                                                          report.min_cell),
                "peer_mean": report.mean_map(totals["peer_mean/total"], totals["peer_mean/count"]),
                "peer_sd": totals["peer_sd"],
            })


def figures(args) -> None:
    body = json.loads(Path(args.report).read_text())
    style = load_style(NOTEBOOK)
    out = savedir(style)
    out.mkdir(parents=True, exist_ok=True)
    stamp = Path(args.report).stem.rsplit("_", 2)[-2]
    average = body["report"].endswith("_images")
    if body["report"].endswith("_stability"):  # D31
        from utils.data_analysis_04_hrs_chronic_cnn_stability import stability_figures

        for condition, section in body["conditions"].items():
            for name, fig in stability_figures(section, body["meta"], style).items():
                for fmt in style["formats"]:
                    path = out / f"stability_{condition}_bin{section['dataset']['status_bin']}_{stamp}_{name}.{fmt}"
                    fig.savefig(path)
                    print(f"wrote {P.relative(path)}")
        return
    if body["report"].endswith("_subgroups"):  # D34
        from utils.data_analysis_04_hrs_chronic_cnn_subgroups import (
            disagreement_figure,
            predictability_figure,
            subgroup_figure,
        )

        for condition, section in body["conditions"].items():
            figs = {"predictability": predictability_figure(section, style), "forest": subgroup_figure(section, style),
                    "disagreement": disagreement_figure(section, style)}
            for name, fig in figs.items():
                for fmt in style["formats"]:
                    path = out / f"subgroups_{condition}_bin{section['dataset']['status_bin']}_{stamp}_{name}.{fmt}"
                    fig.savefig(path)
                    print(f"wrote {P.relative(path)}")
        return
    if body["report"].endswith("_comparisons"):  # D38-D40
        from utils.data_analysis_04_hrs_chronic_cnn_comparisons import comparisons_figure

        for condition, section in body["conditions"].items():
            fig = comparisons_figure(section, style)
            for fmt in style["formats"]:
                path = out / f"comparisons_{condition}_bin{section['dataset']['status_bin']}_{stamp}.{fmt}"
                fig.savefig(path)
                print(f"wrote {P.relative(path)}")
        return
    if body["report"].endswith("_pathways"):  # D35
        from utils.data_analysis_04_hrs_chronic_cnn_pathways import (
            centroid_figure,
            profile_figure,
            stability_figure,
        )
        from utils.data_analysis_04_hrs_chronic_cnn_pathways import map_figure as pathway_map_figure

        for condition, section in body["conditions"].items():
            dropped = section["summary"].get("dropped_rows")
            tag = "_no_" + "_".join(dropped) if dropped else ""  # D36 figures sit beside D35's
            figs = {"stability": stability_figure(section, style), "centroids": centroid_figure(section, style),
                    "profiles": profile_figure(section, style)}
            for net in section["summary"]["networks"]:
                figs[f"maps_{net}"] = pathway_map_figure(section, body["meta"]["channel_names"], style, net)
            for name, fig in figs.items():
                for fmt in style["formats"]:
                    path = out / f"pathways{tag}_{condition}_bin{section['dataset']['status_bin']}_{stamp}_{name}.{fmt}"
                    fig.savefig(path)
                    print(f"wrote {P.relative(path)}")
        return
    if body["report"].endswith("_learning_curve"):  # D32
        from utils.data_analysis_04_hrs_chronic_cnn_learning_curve import learning_curve_figure

        for condition, section in body["conditions"].items():
            fig = learning_curve_figure(section, style)
            for fmt in style["formats"]:
                path = out / f"learning_curve_{condition}_bin{section['dataset']['status_bin']}_{stamp}.{fmt}"
                fig.savefig(path)
                print(f"wrote {P.relative(path)}")
        return
    profile = body.get("config", {}).get("profile")
    if profile in ("kernels", "scaling"):  # D30 kernel arms, D33 scaling arms: one bar and one map per arm
        meta = body["meta"]
        kinds = ("gradcam", "eg_abs") if profile == "kernels" else ("eg_abs",)  # the RNN has no Grad-CAM
        for condition, section in body["conditions"].items():
            stem = f"{profile}_{condition}_bin{section['dataset']['status_bin']}_{stamp}"
            references = {name: float(value) for name, value in (args.reference or [])}
            figs = {"auc": kernel_auc_figure(section, style, references, title=f"{profile} arms")}
            for kind in kinds:
                figs[kind] = kernel_map_figure(section, meta["channel_names"], style, meta["channel_domains"], kind)
            for name, fig in figs.items():
                for fmt in style["formats"]:
                    path = out / f"{stem}_{name}.{fmt}"
                    fig.savefig(path)
                    print(f"wrote {P.relative(path)}")
        return
    draw, prefix = (average_image_figure, "average_image") if average else (map_figure, "maps")
    for condition, section in body["conditions"].items():
        fig = draw(section, body["meta"]["channel_names"], style, body["meta"].get("channel_domains"))
        for fmt in style["formats"]:
            path = out / f"{prefix}_{condition}_bin{section['dataset']['status_bin']}_{stamp}.{fmt}"
            fig.savefig(path)
            print(f"wrote {P.relative(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("counts", "run", "images", "stability", "learning-curve", "subgroups", "pathways",
                                            "comparisons", "figures"))
    parser.add_argument("--profile", default="full")
    parser.add_argument("--rand-file", type=Path, default=P.RAND_HRS_FILE)
    parser.add_argument("--conditions", nargs="*", choices=list(OUTCOMES))
    parser.add_argument("--status-bin", type=int)
    parser.add_argument("--report", help="figures: the run or images report to draw")
    parser.add_argument("--reference", nargs=2, action="append", metavar=("NAME", "AUC"),
                        type=lambda v: v, help="figures, kernels or scaling profile: a dashed reference line, e.g. rnn 0.750")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="print the traceback on failure (do not paste it)")
    args = parser.parse_args()
    if args.command == "figures":
        figures(args)
        return
    config = load_config(args.profile)
    commands = {"counts": counts, "run": run, "images": images, "stability": stability, "learning-curve": learning_curve,
                "subgroups": subgroups, "pathways": pathways, "comparisons": comparisons}
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
