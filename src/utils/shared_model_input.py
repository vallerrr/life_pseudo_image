"""Model input: what every fitted model is handed, and what it is scored against.

One ``.npz`` per dataset ``(scenario, N, replicate)`` under ``P.MODEL_INPUT_DIR``.
Every model reads the same file, so no two models can disagree about what the
outcome was, which cells were observed, or which static covariates exist.

The file holds three kinds of array, kept apart by name:

* **inputs** -- ``X`` (raw channel values, NaN where unobserved), ``M`` (the
  observation mask), ``Delta`` (bins since last observation) and ``S`` (static
  covariates). This is everything a model may see.
* **observed outcome** -- ``obs_time`` and ``obs_type``: the event as a study would
  record it, i.e. censored at attrition; and ``entry_time``, the first bin a person was
  at risk. A person is at risk in bins ``entry_time..obs_time``. In the simulation
  everyone enters at bin 1; in a real panel people join at different ages (left
  truncation), and a model that counted the bins before entry would add immortal time.
  This is what a model is trained on.
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
from dataclasses import dataclass, field
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

# The streams, top to bottom of the C axis, each `channels_per_stream` channels tall.
STREAM_NAMES = ("upstream", "midstream", "downstream")

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
    entry_time: np.ndarray = field(default=None)  # (n,) first bin at risk, 1-based; None -> all 1
    person_id: np.ndarray | None = None  # (n,) source row id for real data; None for simulation

    def __post_init__(self) -> None:
        if self.entry_time is None:
            object.__setattr__(self, "entry_time", np.ones(self.X.shape[0], dtype=np.int16))

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
            entry_time=self.entry_time[index],
            person_id=None if self.person_id is None else self.person_id[index],
        )


def mask_streams(data: ModelInput, streams: tuple[str, ...] | list[str]) -> ModelInput:
    """``data`` with every channel of ``streams`` made never-observed, for an ablation.

    The channels are blanked rather than removed, so the image keeps its ``C`` rows
    and an ablated map lines up cell for cell with a full one. A blanked cell is what
    the standardiser makes of any unobserved cell (value 0, mask 0), and its ``Delta``
    is the bin index for everyone, so the rows carry no information about the person.
    ``truth_w`` is untouched: the truth does not change because a model was blinded.
    """
    if not streams:
        return data
    unknown = set(streams) - set(STREAM_NAMES)
    if unknown:
        raise ValueError(f"streams must be among {STREAM_NAMES}, got {sorted(unknown)}")
    C, T = data.X.shape[1:]
    per_stream = C // len(STREAM_NAMES)
    channels = np.concatenate(
        [np.arange(per_stream) + STREAM_NAMES.index(s) * per_stream for s in streams]
    )
    X, M, Delta = data.X.copy(), data.M.copy(), data.Delta.copy()
    X[:, channels] = np.nan
    M[:, channels] = False
    Delta[:, channels] = np.arange(1, T + 1, dtype=Delta.dtype)  # as the DGP counts it
    return ModelInput(
        X=X, M=M, Delta=Delta, S=data.S, obs_time=data.obs_time, obs_type=data.obs_type,
        truth=data.truth, metadata={**data.metadata, "masked_streams": list(streams)},
        entry_time=data.entry_time, person_id=data.person_id,
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
        entry_time=data.entry_time,
        metadata=np.array(json.dumps(data.metadata)),
        **({} if data.person_id is None else {"person_id": data.person_id}),
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
        entry_time=arrays.pop("entry_time", None),  # absent in files written before it existed
        person_id=arrays.pop("person_id", None),
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
SCALING_METHODS = ("cell_z", "channel_z", "channel_minmax")


@dataclass(frozen=True)
class Standardiser:
    """Image scaling fitted on a training fold only: ``(X - loc) / scale``.

    * ``cell_z`` -- per (channel, bin) mean and SD: each cell reads as SD from same-age
      peers, and a variable's age profile is removed.
    * ``channel_z`` -- one mean and SD per channel, pooled over every bin and person: the
      age profile is kept and a value means the same thing at every age.
    * ``channel_minmax`` -- per channel, pooled the same way, mapped to 0-1 between the
      ``quantiles`` of the training values and clipped there, so one outlier does not
      squash the rest. 0 and 1 are the bounds, not the extremes; both are stored.

    An unobserved cell becomes ``fill``, the training mean on the scaled axis (0 under a
    z-score; under min-max the mean after clipping, so an outlier cannot drag it out of
    0-1); the mask plane tells the model it was unobserved.
    """

    mean: np.ndarray  # (C, T), or (C, 1) per channel
    sd: np.ndarray
    method: str = "cell_z"
    minimum: np.ndarray | None = None  # per-channel extremes and 0-1 bounds (channel_minmax)
    maximum: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    clipped_mean: np.ndarray | None = None  # the mean after clipping to [lower, upper]: the fill

    @property
    def loc(self) -> np.ndarray:
        return self.lower if self.method == "channel_minmax" else self.mean

    @property
    def scale(self) -> np.ndarray:
        return self.upper - self.lower if self.method == "channel_minmax" else self.sd

    @property
    def fill(self) -> np.ndarray:
        """The value an unobserved cell gets, broadcastable to ``(C, T)``."""
        centre = self.mean if self.clipped_mean is None else self.clipped_mean
        return ((centre - self.loc) / self.scale).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.loc) / self.scale
        if self.method == "channel_minmax":
            z = np.clip(z, 0.0, 1.0)
        return np.where(np.isnan(z), np.broadcast_to(self.fill, z.shape[-2:]), z).astype(np.float32)

    def stats(self) -> dict[str, list]:
        """Per-channel parameters, for a report (``cell_z`` gives ``(C, T)`` lists)."""
        out = {"mean": self.mean, "sd": self.sd, "lower": self.lower, "upper": self.upper,
               "clipped_mean": self.clipped_mean, "minimum": self.minimum, "maximum": self.maximum}
        return {k: np.asarray(v).squeeze(-1).tolist() if v.shape[-1] == 1 else v.tolist()
                for k, v in out.items() if v is not None}


def fit_standardiser(X: np.ndarray, method: str = "cell_z", quantiles: tuple[float, float] = (0.01, 0.99)) -> Standardiser:
    """A cell (or channel) no training row observed gets mean 0, SD 1, so it passes through as 0."""
    if method not in SCALING_METHODS:
        raise ValueError(f"scaling method must be one of {SCALING_METHODS}, got {method!r}")
    pooled = method != "cell_z"
    if pooled:  # (n, C, T) -> (C, n * T): every person and every bin of a channel together
        X = np.moveaxis(X, 1, 0).reshape(X.shape[1], -1).T[:, :, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing cells
        mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean).astype(np.float32)
    sd = np.where(np.isnan(sd) | (sd < 1e-8), 1.0, sd).astype(np.float32)
    if method != "channel_minmax":
        return Standardiser(mean=mean, sd=sd, method=method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        minimum, maximum = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        lower, upper = (np.nanquantile(X, q, axis=0) for q in quantiles)
    if quantiles != (0.0, 1.0):  # a binary or lumpy channel can have both quantiles on one value
        thin = ~(upper - lower > 1e-8)
        lower, upper = np.where(thin, minimum, lower), np.where(thin, maximum, upper)
    empty = ~(upper - lower > 1e-8)  # unobserved or constant: 0 passes through as 0, the fill
    lower = np.where(empty, mean, lower)
    upper = np.where(empty, mean + 1.0, upper)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clipped = np.nanmean(np.clip(X, lower, upper), axis=0)
    clipped = np.where(empty | np.isnan(clipped), mean, clipped)
    as32 = lambda a: np.nan_to_num(a).astype(np.float32)  # noqa: E731
    return Standardiser(mean=mean, sd=sd, method=method, minimum=as32(minimum), maximum=as32(maximum),
                        lower=as32(lower), upper=as32(upper), clipped_mean=as32(clipped))


# --- folds ------------------------------------------------------------------
def stratified_folds(
    obs_type: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``k`` (train, test) index pairs, stratified on the observed outcome."""
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros_like(obs_type), obs_type))
