"""Model input: what every fitted model is handed, and what it is scored against.

One ``.npz`` per dataset ``(scenario, N, replicate)`` under ``P.MODEL_INPUT_DIR``.
Every model reads the same file, so no two models can disagree about what the
outcome was, which cells were observed, or which static covariates exist.

The file holds three kinds of array, kept apart by name:

* **inputs** -- ``X`` (raw channel values, NaN where unobserved), ``M`` (the
  observation mask), ``Delta`` (bins since last observation) and ``S`` (static
  covariates). This is everything a model may see.
* **observed outcome** -- ``obs_time`` and ``obs_type``: the event as a study would
  record it, i.e. censored at attrition. This is what a model is trained on.
* **truth** -- every key prefixed ``truth_``: the DGP's own event times, the true
  hazards and the ground-truth weight map ``w_true``. Evaluation reads these; a model
  never does. The prefix exists so that a leak is visible in one grep.

``X`` is stored raw, not standardised. Standardising on the whole panel would let the
test fold's mean and SD into training; each model standardises on its own training
fold instead (:func:`fit_standardiser`).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils import shared_paths as P
from utils.shared_dgp import MASTER_SEED, DGPResult, SimulationConfig, generate_dataset, load_config
from utils.shared_reproducibility import git_hash

# Columns of S, in order (spec §1.1).
STATIC_NAMES = ("cohort", "sex", "frailty")
# Frailty is unobserved heterogeneity: no survey measures it, and it drives the death
# hazard directly. A model sees it only when a run asks for it by name.
DEFAULT_STATIC = ("cohort", "sex")

# Observed outcome codes, shared by every model's competing-risks head.
CENSORED, DISEASE, DEATH = 0, 1, 2
OUTCOME_NAMES = ("event-free / censored", "disease", "death")


@dataclass(frozen=True)
class ModelInput:
    """One dataset as models see it, plus the truth evaluation scores against."""

    X: np.ndarray  # (n, C, T) raw values, NaN where unobserved
    M: np.ndarray  # (n, C, T) bool, True where observed
    Delta: np.ndarray  # (n, C, T) bins since last observation
    S: np.ndarray  # (n, P) standardised static covariates, columns STATIC_NAMES
    obs_time: np.ndarray  # (n,) last bin the outcome is known for, 1-based; 0 = never observed
    obs_type: np.ndarray  # (n,) CENSORED / DISEASE / DEATH at obs_time
    truth: dict[str, np.ndarray]  # truth_* arrays, never a model input
    metadata: dict[str, Any]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def static(self, names: tuple[str, ...] = DEFAULT_STATIC) -> np.ndarray:
        """The static covariates a run asked for, by name."""
        return self.S[:, [STATIC_NAMES.index(name) for name in names]]

    def subset(self, index: np.ndarray) -> ModelInput:
        """Rows ``index`` of every per-person array; the weight map is not per person."""
        return ModelInput(
            X=self.X[index],
            M=self.M[index],
            Delta=self.Delta[index],
            S=self.S[index],
            obs_time=self.obs_time[index],
            obs_type=self.obs_type[index],
            truth={
                key: value if key == "truth_w" else value[index]
                for key, value in self.truth.items()
            },
            metadata=self.metadata,
        )


# --- the observed outcome ---------------------------------------------------
def observed_outcome(result: DGPResult, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """The event as the panel records it: censored at attrition.

    Dropout is only ever sampled strictly before the event (``shared_dgp``), and a
    dropped person is unobserved from ``dropout_time`` on. So a dropout means the
    person is known event-free through ``dropout_time - 1`` and nothing after; an
    event that the DGP places later was never seen. Without dropout, the event is
    seen in its own bin, and an event-free life is known through bin ``T``.
    """
    dropped = result.dropout_time <= n_bins
    event_seen = ~dropped & (result.event_type > 0)
    obs_time = np.where(
        dropped,
        result.dropout_time - 1,
        np.where(event_seen, result.event_time, n_bins),
    ).astype(np.int16)
    obs_type = np.where(event_seen, result.event_type, CENSORED).astype(np.int8)
    return obs_time, obs_type


# --- files ------------------------------------------------------------------
def model_input_path(scenario: str, n: int, replicate: int) -> Path:
    return P.MODEL_INPUT_DIR / f"model_input_{scenario}_n{n}_r{replicate}.npz"


def build_model_input(result: DGPResult, cfg: SimulationConfig) -> ModelInput:
    obs_time, obs_type = observed_outcome(result, cfg.T)
    return ModelInput(
        X=result.X.astype(np.float32),
        M=result.M.astype(bool),
        Delta=result.Delta.astype(np.int16),
        S=result.S.astype(np.float32),
        obs_time=obs_time,
        obs_type=obs_type,
        truth={
            "truth_w": result.w_true,
            "truth_event_time": result.event_time,
            "truth_event_type": result.event_type,
            "truth_dropout_time": result.dropout_time,
            "truth_disease_probability": result.disease_probability.astype(np.float32),
            "truth_death_probability": result.death_probability.astype(np.float32),
        },
        metadata={
            **{k: v for k, v in result.metadata.items() if _jsonable(v)},
            "static_names": list(STATIC_NAMES),
            "git_hash": git_hash(),
        },
    )


def write_model_input(data: ModelInput, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        X=data.X,
        M=data.M,
        Delta=data.Delta,
        S=data.S,
        obs_time=data.obs_time,
        obs_type=data.obs_type,
        metadata=np.array(json.dumps(data.metadata)),
        **data.truth,
    )
    return destination


def read_model_input(source: Path) -> ModelInput:
    with np.load(source) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return ModelInput(
        X=arrays.pop("X"),
        M=arrays.pop("M"),
        Delta=arrays.pop("Delta"),
        S=arrays.pop("S"),
        obs_time=arrays.pop("obs_time"),
        obs_type=arrays.pop("obs_type"),
        metadata=json.loads(str(arrays.pop("metadata"))),
        truth=arrays,
    )


def load_model_input(
    scenario: str,
    n: int,
    replicate: int = 0,
    config: SimulationConfig | None = None,
    master_seed: int = MASTER_SEED,
) -> ModelInput:
    """Read the dataset's file, writing it from the seed tuple first if absent.

    The file is a cache of a deterministic generation, so writing it on demand cannot
    produce a dataset that differs from one written by ``00_data_creation``.
    """
    path = model_input_path(scenario, n, replicate)
    if not path.exists():
        cfg = config or load_config()
        result = generate_dataset(scenario, n=n, replicate=replicate, config=cfg, master_seed=master_seed)
        write_model_input(build_model_input(result, cfg), path)
    return read_model_input(path)


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


# --- per-fold standardisation -----------------------------------------------
@dataclass(frozen=True)
class Standardiser:
    """Per-(channel, bin) mean and SD from a training fold only."""

    mean: np.ndarray  # (C, T)
    sd: np.ndarray  # (C, T)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise; an unobserved cell becomes 0 (the training mean)."""
        z = (X - self.mean) / self.sd
        return np.where(np.isnan(z), 0.0, z).astype(np.float32)


def fit_standardiser(X: np.ndarray) -> Standardiser:
    """A cell no training row observed gets mean 0, SD 1, so it passes through as 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing cells
        mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    sd = np.where(np.isnan(sd) | (sd < 1e-8), 1.0, sd)
    return Standardiser(mean=mean.astype(np.float32), sd=sd.astype(np.float32))


# --- folds ------------------------------------------------------------------
def stratified_folds(
    obs_type: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``k`` (train, test) index pairs, stratified on the observed outcome."""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros_like(obs_type), obs_type))
