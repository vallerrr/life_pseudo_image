"""Analysis 03 - oracle checks: the bounds a correctly specified model sets.

For every scenario and N, on replicate 0 and the CNN's own five outer folds, fits

* ``static``  -- cohort + sex and a per-bin baseline: the floor;
* ``oracle``  -- the scenario's true form on the observed cells (see the module);
* ``oracle_no_death_history`` -- the same, with the death hazard's downstream term
  dropped, to show how much attribution reaches the map through the competing risk;

and scores each with the CNN's prediction, attribution and Adebayo code. Nothing
here is a study result, and no ``study`` shard is read: it calibrates the
yardsticks before the study is run.

Run:  python src/data_analysis/03_oracle_checks.py   (or `make oracle`)
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap (conventions §2) -------------------------------------------
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P  # noqa: E402

P.bootstrap()

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from utils.data_analysis_02_pseudo_image_cnn_model import (  # noqa: E402
    expected_gradients,
    image_tensor,
    interval_targets,
    predict_hazards,
)
from utils.data_analysis_03_oracle_checks import fit_oracle, randomised_oracle  # noqa: E402
from utils.shared_dgp import MASTER_SEED, SCENARIOS  # noqa: E402
from utils.shared_evaluation import (  # noqa: E402
    map_similarity,
    population_map,
    prediction_metrics,
    recovery_metrics,
)
from utils.shared_model_input import (  # noqa: E402
    DEFAULT_STATIC,
    fit_standardiser,
    load_model_input,
    stratified_folds,
)
from utils.shared_reproducibility import git_hash, stable_seed  # noqa: E402

NOTEBOOK = "03_oracle_checks"
SIZES = (2000, 5000, 10000, 20000)
REPLICATE = 0
OUTER_FOLDS = 5
HORIZONS = (10, 20)
# arm -> (form, death_history); a form of None means "the scenario's own".
ARMS = {
    "static": ("static", False),
    "oracle": (None, True),
    "oracle_no_death_history": (None, False),
}


def _seed(scenario: str, n: int, tag: str) -> int:
    return stable_seed(MASTER_SEED, f"{scenario}|oracle_checks|{tag}", n, REPLICATE)


def _cnn_outer_seed(scenario: str, n: int) -> int:
    """The CNN job's outer-fold seed, so oracle and CNN are scored on the same people."""
    return stable_seed(MASTER_SEED, f"{scenario}|pseudo_image_cnn|outer", n, REPLICATE)


def run_fold(scenario: str, n: int, fold: int, attribution: dict) -> tuple[list[dict], dict]:
    data = load_model_input(scenario, n, REPLICATE)
    train_idx, test_idx = stratified_folds(data.obs_type, OUTER_FOLDS, _cnn_outer_seed(scenario, n))[fold]
    train, test = data.subset(train_idx), data.subset(test_idx)
    T = data.X.shape[2]
    standardiser = fit_standardiser(train.X)

    def tensors(part, obs_time=None, obs_type=None, entry_time=None):
        target, at_risk = interval_targets(
            part.obs_time if obs_time is None else obs_time,
            part.obs_type if obs_type is None else obs_type,
            T,
            part.entry_time if entry_time is None else entry_time,
        )
        static = torch.from_numpy(part.static(DEFAULT_STATIC).astype(np.float32))
        return image_tensor(part, standardiser), static, target, at_risk

    train_t, test_t = tensors(train), tensors(test)
    rng = np.random.default_rng(_seed(scenario, n, f"explain{fold}"))
    explain = rng.choice(test.n, size=min(attribution["n_explain"], test.n), replace=False)
    baselines = train_t[0][rng.choice(train.n, size=min(attribution["n_baselines"], train.n), replace=False)]
    observed = test.M[explain]
    w_true = data.truth["truth_w"]

    def attribution_map(model) -> np.ndarray:
        per_person = expected_gradients(
            model, test_t[0][explain], test_t[1][explain], baselines,
            attribution["n_samples"], _seed(scenario, n, f"eg{fold}"),
        )
        return population_map(per_person, observed)[0]

    rows, maps = [], {}
    for arm, (form, death_history) in ARMS.items():
        start = time.perf_counter()
        model = fit_oracle(form or scenario, *train_t, death_history=death_history)
        metrics, _ = prediction_metrics(*predict_hazards(model, *test_t[:2]), test.obs_time, test.obs_type, HORIZONS, test.entry_time)
        metrics["fitted_beta"] = model.beta.item()
        metrics["fitted_alpha"] = model.alpha.item()
        if arm != "static":
            A = attribution_map(model)
            metrics.update(recovery_metrics(A, w_true))
            A_param = attribution_map(randomised_oracle(model, _seed(scenario, n, f"rand{fold}")))
            metrics["sanity_param_random_similarity"] = map_similarity(A, A_param)
            permutation = np.random.default_rng(_seed(scenario, n, f"labels{fold}")).permutation(train.n)
            shuffled = fit_oracle(
                form or scenario,
                *tensors(train, train.obs_time[permutation], train.obs_type[permutation], train.entry_time[permutation]),
                death_history=death_history,
            )
            A_label = attribution_map(shuffled)
            metrics["sanity_label_random_similarity"] = map_similarity(A, A_label)
            metrics["sanity_label_random_flatness_cv"] = recovery_metrics(A_label, w_true)["flatness_cv"]
            # Spearman is blind to scale: a shuffled-label fit with a coefficient near
            # zero ranks cells the same way and scores as "similar". The mass ratio says
            # how much attribution the control actually has, relative to the model.
            metrics["sanity_label_random_mass_ratio"] = float(A_label.sum() / A.sum()) if A.sum() > 0 else float("nan")
            maps.update({f"{arm}_f{fold}_A": A, f"{arm}_f{fold}_A_param": A_param, f"{arm}_f{fold}_A_label": A_label})
        metrics["seconds"] = time.perf_counter() - start
        rows += [
            {"scenario": scenario, "n": n, "replicate": REPLICATE, "model": "oracle_checks",
             "arm": arm, "fold": fold, "metric": k, "value": float(v)}
            for k, v in metrics.items()
        ]
    maps["w_true"] = w_true
    return rows, maps


def summarise(results: pd.DataFrame) -> None:
    """Mean over folds; the study unit (D11), here for one replicate."""
    mean = results.groupby(["scenario", "n", "arm", "metric"])["value"].mean().unstack("metric")
    pd.set_option("display.width", 200)
    prediction = ["auc_disease_t20", "auc_disease_t10", "c_index_disease", "interval_nll", "ici_decile", "fitted_beta"]
    print("\n== prediction, N = 10,000 (mean of 5 folds) ==")
    print(mean.xs(10000, level="n")[prediction].round(3).to_string())
    attribution = [
        "spearman_cells", "window_auprc", "stream_share_upstream", "stream_share_downstream",
        "sanity_param_random_similarity", "sanity_label_random_similarity", "sanity_label_random_mass_ratio",
        "flatness_cv", "sanity_label_random_flatness_cv",
    ]
    print("\n== attribution, N = 10,000 (mean of 5 folds) ==")
    oracle = mean.xs(10000, level="n").drop(index="static", level="arm")
    print(oracle[attribution].round(3).to_string())
    print("\n== recovery by N, oracle arm (K4: spearman_cells >= 0.50 and window_auprc >= 2x non-zero share) ==")
    recovery = mean.xs("oracle", level="arm")[["spearman_cells", "window_auprc", "auc_disease_t20"]]
    print(recovery.unstack("n").round(3).to_string())


def main() -> None:
    attribution = yaml.safe_load(P.PSEUDO_IMAGE_CNN_CONFIG.read_text())["attribution"]
    out = P.ORACLE_CHECKS_DATA_DIR
    units = out / "units"
    units.mkdir(exist_ok=True)
    tables = []
    for scenario in SCENARIOS:
        for n in SIZES:
            # One finished (scenario, N) is kept, so a run that dies resumes there.
            table = units / f"{scenario}_n{n}_r{REPLICATE}.parquet"
            if not table.exists():
                start = time.perf_counter()
                rows, maps = [], {}
                for fold in range(OUTER_FOLDS):
                    fold_rows, fold_maps = run_fold(scenario, n, fold, attribution)
                    rows += fold_rows
                    maps.update(fold_maps)
                np.savez_compressed(out / f"maps_{scenario}_n{n}_r{REPLICATE}.npz", **maps)
                pd.DataFrame(rows).to_parquet(table, index=False)  # last: marks the unit done
                print(f"{scenario:<17} N={n:>6,}: {time.perf_counter() - start:6.1f} s", flush=True)
            tables.append(table)
    results = pd.concat([pd.read_parquet(t) for t in tables], ignore_index=True)
    destination = out / f"results_{NOTEBOOK}.parquet"
    results.to_parquet(destination, index=False)
    manifest = {
        "git_hash": git_hash(), "master_seed": MASTER_SEED, "sizes": SIZES, "replicate": REPLICATE,
        "outer_folds": OUTER_FOLDS, "arms": {k: list(v) for k, v in ARMS.items()},
        "attribution": attribution, "static": list(DEFAULT_STATIC), "torch": torch.__version__,
    }
    (out / f"manifest_{NOTEBOOK}.json").write_text(json.dumps(manifest, indent=2))
    print(f"results -> {P.relative(destination)}")
    summarise(results)


if __name__ == "__main__":
    main()
