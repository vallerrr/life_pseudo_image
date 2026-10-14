"""The pseudo-image CNN job: one shard = one outer fold of one dataset.

Owned by src/data_analysis/02_pseudo_image_cnn.ipynb, which reads what this writes
and never trains anything itself (conventions §8). A shard is the unit of work,
of failure and of resumption: ``(profile, scenario, N, replicate, outer fold)``.
It tunes on inner folds, refits on the outer training set, and writes

* ``shards/<id>.parquet`` -- one row per metric (spec §7's long format),
* ``shards/<id>.npz``     -- the attribution maps and the calibration table,
* ``shards/<id>.json``    -- config, seeds, git hash, chosen hyperparameters, timings.

A finished shard is skipped on re-run, so a job that dies half way resumes. ``merge``
concatenates the shards into ``results_<profile>.parquet``.

Run (or use the .sh beside this file):
    python src/utils/data_analysis_02_pseudo_image_cnn_job.py shards --profile smoke
    python src/utils/data_analysis_02_pseudo_image_cnn_job.py probe  --profile study
    python src/utils/data_analysis_02_pseudo_image_cnn_job.py local  --profile smoke [--shard i]
    python src/utils/data_analysis_02_pseudo_image_cnn_job.py merge  --profile smoke
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
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from utils.data_analysis_02_pseudo_image_cnn_model import (  # noqa: E402
    expected_gradients,
    fit,
    image_tensor,
    interval_targets,
    predict_hazards,
    randomised_copy,
    resolve_device,
)
from utils.shared_dgp import MASTER_SEED  # noqa: E402
from utils.shared_evaluation import (  # noqa: E402
    interval_nll,
    map_similarity,
    population_map,
    prediction_metrics,
    recovery_metrics,
)
from utils.shared_model_input import (  # noqa: E402
    ModelInput,
    fit_standardiser,
    load_model_input,
    stratified_folds,
)
from utils.shared_reproducibility import git_hash, stable_seed  # noqa: E402

MODEL = "pseudo_image_cnn"


# --- configuration ----------------------------------------------------------
def load_profile(profile: str) -> dict[str, Any]:
    """The config with the profile's overrides applied, and the profile inlined."""
    with P.PSEUDO_IMAGE_CNN_CONFIG.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if profile not in raw["profiles"]:
        raise ValueError(f"profile must be one of {sorted(raw['profiles'])}, got {profile!r}")
    run = copy.deepcopy(raw["profiles"][profile])
    config = {key: copy.deepcopy(value) for key, value in raw.items() if key != "profiles"}
    for section, values in run.pop("overrides", {}).items():
        config[section].update(values)
    config["run"] = run
    config["profile"] = profile
    return config


@dataclass(frozen=True)
class Shard:
    profile: str
    scenario: str
    n: int
    replicate: int
    fold: int

    @property
    def id(self) -> str:
        return f"{self.scenario}_n{self.n}_r{self.replicate}_f{self.fold}"


def shard_dir(profile: str) -> Path:
    return P.PSEUDO_IMAGE_CNN_DATA_DIR / profile / "shards"


def results_path(profile: str) -> Path:
    return P.PSEUDO_IMAGE_CNN_DATA_DIR / profile / f"results_{profile}.parquet"


def enumerate_shards(config: dict[str, Any]) -> list[Shard]:
    run = config["run"]
    return [
        Shard(config["profile"], scenario, n, replicate, fold)
        for scenario, n, replicate, fold in itertools.product(
            run["scenarios"], run["n"], range(run["replicates"]), range(run["outer_folds"])
        )
    ]


def _seed(shard: Shard, tag: str) -> int:
    return stable_seed(MASTER_SEED, f"{shard.scenario}|{MODEL}|{tag}", shard.n, shard.replicate)


# --- one fit, from arrays to a trained model --------------------------------
def _tensors(data: ModelInput, standardiser, static_names) -> tuple[torch.Tensor, ...]:
    target, at_risk = interval_targets(data.obs_time, data.obs_type, data.X.shape[2])
    static = torch.from_numpy(data.static(tuple(static_names)).astype(np.float32))
    return image_tensor(data, standardiser), static, target, at_risk


def _fit_on(data: ModelInput, hyper: dict[str, Any], config: dict[str, Any], seed: int):
    """Standardise on ``data``, carve an early-stopping holdout from it, train."""
    standardiser = fit_standardiser(data.X)
    keep, stop = train_test_split(
        np.arange(data.n),
        test_size=config["training"]["early_stop_fraction"],
        stratify=data.obs_type,
        random_state=seed % 2**31,
    )
    static = config["static"]
    result = fit(
        _tensors(data.subset(keep), standardiser, static),
        _tensors(data.subset(stop), standardiser, static),
        config["architecture"],
        hyper,
        config["training"],
        seed,
        resolve_device(config["device"]),
    )
    return result, standardiser


def _grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(config["grid"])
    return [dict(zip(keys, values)) for values in itertools.product(*config["grid"].values())]


def tune(train: ModelInput, config: dict[str, Any], shard: Shard) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick hyperparameters by mean inner-fold NLL. Inner scores are never reported."""
    grid = _grid(config)
    if len(grid) == 1:
        return grid[0], []
    inner = stratified_folds(train.obs_type, config["run"]["inner_folds"], _seed(shard, f"inner{shard.fold}"))
    scores = []
    for g, hyper in enumerate(grid):
        nll = []
        for k, (fit_idx, val_idx) in enumerate(inner):
            result, standardiser = _fit_on(train.subset(fit_idx), hyper, config, _seed(shard, f"f{shard.fold}g{g}k{k}"))
            val = train.subset(val_idx)
            image, static, _, _ = _tensors(val, standardiser, config["static"])
            nll.append(interval_nll(*predict_hazards(result.model, image, static), val.obs_time, val.obs_type))
        scores.append({**hyper, "inner_nll": float(np.mean(nll))})
    best = min(range(len(grid)), key=lambda g: scores[g]["inner_nll"])
    return grid[best], scores


# --- one shard ----------------------------------------------------------------
def run_shard(shard: Shard, config: dict[str, Any], force: bool = False) -> Path:
    out = shard_dir(shard.profile)
    out.mkdir(parents=True, exist_ok=True)
    table = out / f"{shard.id}.parquet"
    if table.exists() and not force:
        return table

    start = time.perf_counter()
    data = load_model_input(shard.scenario, shard.n, shard.replicate)
    outer = stratified_folds(data.obs_type, config["run"]["outer_folds"], _seed(shard, "outer"))
    train_idx, test_idx = outer[shard.fold]
    train, test = data.subset(train_idx), data.subset(test_idx)
    T = data.X.shape[2]

    hyper, inner_scores = tune(train, config, shard)
    final, standardiser = _fit_on(train, hyper, config, _seed(shard, f"final{shard.fold}"))
    model = final.model

    # Prediction on the outer test fold (spec §4).
    image, static, _, _ = _tensors(test, standardiser, config["static"])
    disease_hazard, death_hazard = predict_hazards(model, image, static)
    metrics, calibration = prediction_metrics(
        disease_hazard, death_hazard, test.obs_time, test.obs_type, tuple(config["run"]["horizons"])
    )

    # Recovery (spec §3): Expected Gradients on a test subsample, training baselines.
    att = config["attribution"]
    rng = np.random.default_rng(_seed(shard, f"explain{shard.fold}"))
    explain = rng.choice(test.n, size=min(att["n_explain"], test.n), replace=False)
    train_image, _, _, _ = _tensors(train, standardiser, config["static"])
    baselines = train_image[rng.choice(train.n, size=min(att["n_baselines"], train.n), replace=False)]

    observed = test.M[explain]

    def attribution_map(m) -> tuple[np.ndarray, np.ndarray]:
        per_person = expected_gradients(
            m, image[explain], static[explain], baselines, att["n_samples"], _seed(shard, f"eg{shard.fold}")
        )
        return population_map(per_person, observed)

    A_abs, A_signed = attribution_map(model)
    w_true = data.truth["truth_w"]
    metrics.update(recovery_metrics(A_abs, w_true))

    maps = {"A_abs": A_abs, "A_signed": A_signed, "w_true": w_true, "calibration": calibration}
    if config["run"]["sanity_checks"]:
        # Adebayo et al.: a map that survives re-drawing the weights, or training on
        # shuffled labels, is explaining the input, not the model.
        A_param, _ = attribution_map(randomised_copy(model, _seed(shard, f"rand{shard.fold}")))
        metrics["sanity_param_random_similarity"] = map_similarity(A_abs, A_param)
        maps["A_param_random"] = A_param
        if shard.fold == 0:
            permutation = np.random.default_rng(_seed(shard, "labels")).permutation(train.n)
            shuffled = ModelInput(
                X=train.X, M=train.M, Delta=train.Delta, S=train.S,
                obs_time=train.obs_time[permutation], obs_type=train.obs_type[permutation],
                truth={}, metadata=train.metadata,
            )
            shuffled_fit, _ = _fit_on(shuffled, hyper, config, _seed(shard, "labels_fit"))
            A_label, _ = attribution_map(shuffled_fit.model)
            metrics["sanity_label_random_similarity"] = map_similarity(A_abs, A_label)
            # The reference K3 (D11) compares a null map's flatness against.
            metrics["sanity_label_random_flatness_cv"] = recovery_metrics(A_label, w_true)["flatness_cv"]
            maps["A_label_random"] = A_label

    metrics["best_epoch"] = float(final.epochs)
    metrics["fit_seconds"] = final.seconds
    seconds = time.perf_counter() - start
    metrics["shard_seconds"] = seconds

    rows = pd.DataFrame(
        {
            "profile": shard.profile,
            "scenario": shard.scenario,
            "n": shard.n,
            "replicate": shard.replicate,
            "model": MODEL,
            "fold": shard.fold,
            "metric": list(metrics),
            "value": [float(v) for v in metrics.values()],
        }
    )
    np.savez_compressed(out / f"{shard.id}.npz", **maps)
    manifest = {
        "shard": shard.__dict__,
        "git_hash": git_hash(),
        "device": str(resolve_device(config["device"])),
        "torch": torch.__version__,
        "master_seed": MASTER_SEED,
        "dataset_seed": data.metadata.get("seed"),
        "config": config,
        "hyperparameters": hyper,
        "final_val_history": [round(v, 6) for v in final.history],
        "inner_scores": inner_scores,
        "n_train": int(train.n),
        "n_test": int(test.n),
        "test_outcomes": {int(k): int(v) for k, v in zip(*np.unique(test.obs_type, return_counts=True))},
        "shard_seconds": seconds,
    }
    (out / f"{shard.id}.json").write_text(json.dumps(manifest, indent=2, default=str))
    rows.to_parquet(table, index=False)  # written last: its existence marks the shard done
    return table


def merge(profile: str) -> Path:
    tables = sorted(shard_dir(profile).glob("*.parquet"))
    if not tables:
        raise FileNotFoundError(f"no finished shards under {P.relative(shard_dir(profile))}")
    destination = results_path(profile)
    pd.concat([pd.read_parquet(t) for t in tables], ignore_index=True).to_parquet(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("shards", "probe", "local", "merge"))
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--shard", type=int, help="run only this shard index (an array task)")
    parser.add_argument("--force", action="store_true", help="re-run finished shards")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    config = load_profile(args.profile)
    shards = enumerate_shards(config)

    if args.command == "shards":
        for i, shard in enumerate(shards):
            print(i, shard.id)
        print(f"{len(shards)} shards in profile {args.profile!r}")
    elif args.command == "probe":
        # Shard cost grows with N, so time the first shard at every N and scale each
        # by how many shards share it. Fold 0 carries the label-shuffled refit, so it
        # is the most expensive fold and the estimate errs high.
        total = 0.0
        for n in config["run"]["n"]:
            shard = next(s for s in shards if s.n == n)
            start = time.perf_counter()
            run_shard(shard, config, force=True)
            seconds = time.perf_counter() - start
            count = sum(s.n == n for s in shards)
            total += seconds * count
            print(f"N={n:>6,}: {shard.id} {seconds:6.1f} s  x {count} shards", flush=True)
        print(f"estimate: ~{total / 3600:.1f} hours for {len(shards)} shards "
              f"on {resolve_device(config['device'])}, run one at a time")
    elif args.command == "local":
        chosen = [shards[args.shard]] if args.shard is not None else shards
        for shard in chosen:
            start = time.perf_counter()
            run_shard(shard, config, force=args.force)
            print(f"{shard.id}: {time.perf_counter() - start:.1f} s", flush=True)
        if args.shard is None:
            print(f"merged -> {P.relative(merge(args.profile))}")
    else:
        print(f"merged -> {P.relative(merge(args.profile))}")


if __name__ == "__main__":
    main()
