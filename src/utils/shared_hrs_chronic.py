"""HRS binary outcomes at a given age: one label and one life image per person (D23, D25).

The task: does a person have chronic condition ``c`` by age ``A``, read from their life
image at ages 50 to ``A``? It is built so that every person is looked at the same way:

* **Status.** Everyone kept has a responding interview in the *status bin*, the two-year
  bin holding age ``A``; the label is RAND's "ever had" flag (``RwHIBPE`` ...) at that
  interview, the later one if two fall in the bin. Cases include people who already had
  the condition when they entered the panel: the task is prevalence by ``A``, and what
  happened after a diagnosis stays in the image (the user's choice, 2026-10-25).
* **Image.** The 12 channels of ``shared_hrs_ingest`` over bins ``1 .. status_bin - 1``
  only, so the image is the same width for everyone and the status interview's own
  answers are not in it.

Why not "ever diagnosed during follow-up"? Then people seen to older ages have more
interviews in which to report a condition, and a classifier -- and its activation map --
learns *how long someone was observed*. Requiring an interview at one age for everyone
removes that difference. What remains is survivorship: people who died before ``A`` are
not in the data, and the report counts nobody for them.

**Death** (D25) uses the same status interview and image; the label is death within
``death_horizon_years`` of that interview (``RADAGE_M``). A control must be *known* alive
at the horizon: a later death date beyond it, a later interview at or past it, or a later
wave recorded "alive, not interviewed" (``RwIWSTAT = 4``) at least a year past it by the
wave's nominal year (a non-respondent has no interview date). Anyone else's vital status
at the horizon is unknown and they are left out, counted in the report.

The channel transforms, missing codes and age clock are ``shared_hrs_ingest``'s, so the
images here and the hazard pipeline's agree cell for cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.shared_hrs_ingest import (
    ALL_WAVES,
    AGE_AT_FIRST_BIN,
    N_BINS,
    RESPONDED,
    YEARS_PER_BIN,
    HRSIngestError,
    RandFrame,
    _channel_array,
    _delta,
    _first_true,
    _last_true,
    _pick,
    _wave_matrix,
    age_bin,
    required_columns,
)
from utils.shared_hrs_variables import CHANNEL_SETS, STATIC_SETS, channel_domains, static_columns, static_matrix

# RAND "R ever had ..." [425-436]: 0.No / 1.Yes in waves 1-16, no item missing codes.
CONDITIONS = {
    "hibpe": "high blood pressure",
    "diabe": "diabetes",
    "cancre": "cancer",
    "lunge": "lung disease",
    "hearte": "heart problems",
    "stroke": "stroke",
    "psyche": "psychiatric problems",
    "arthre": "arthritis",
}
DEATH = "death"
# Everything the builder can label: the conditions, and death within a horizon.
OUTCOMES = (*CONDITIONS, DEATH)
ALIVE_NOT_INTERVIEWED = 4  # RwIWSTAT [75]


def condition_stem(condition: str) -> str:
    return f"r{{w}}{condition}"


def chronic_columns(conditions: tuple[str, ...] | list[str] = OUTCOMES, channel_set: str = "extended") -> list[str]:
    """Every column the datasets read: the channels' and ingestion's (with ``RADAGE_M``), the static
    covariates', and the conditions'. The extended set includes the core's, so it reads everything."""
    columns = list(required_columns(CHANNEL_SETS[channel_set]))
    columns += [c for c in static_columns() if c not in columns]
    for condition in [c for c in conditions if c != DEATH]:
        columns += [c for c in (condition_stem(condition).format(w=w) for w in ALL_WAVES) if c not in columns]
    return columns


def nominal_wave_year(wave: int) -> int:
    """The calendar year a wave is named for (1992, 1994, ...); AHEAD's 2-3 ran a year earlier."""
    return 1992 + 2 * (wave - 1)


def bin_ages(status_bin: int) -> tuple[int, int]:
    """The ages ``[lo, hi)`` a bin covers."""
    lo = AGE_AT_FIRST_BIN + YEARS_PER_BIN * (status_bin - 1)
    return lo, lo + YEARS_PER_BIN


@dataclass(frozen=True)
class ChronicDataset:
    """One condition at one status bin. Per-person arrays never leave the machine."""

    X: np.ndarray  # (n, C, K) raw channel values, NaN unobserved; K = status_bin - 1
    M: np.ndarray  # (n, C, K) observed
    Delta: np.ndarray  # (n, C, K) bins since last observation
    static: np.ndarray  # (n, P) raw static covariates, NaN where unknown
    static_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    channel_domains: tuple[str, ...]  # stream (core) or domain (extended) of each row
    y: np.ndarray  # (n,) 0/1: condition by the status interview
    household: np.ndarray  # (n,) HHID, for grouping folds
    condition: str  # a condition code or "death"
    status_bin: int
    report: dict[str, Any]  # raw counts: suppress through SecureReport before writing

    @property
    def n(self) -> int:
        return len(self.y)


def build_chronic_dataset(
    frame: RandFrame, condition: str, status_bin: int, death_horizon_years: int = 10,
    channel_set: str = "core", static_set: str = "core",
) -> ChronicDataset:
    if condition not in OUTCOMES:
        raise HRSIngestError(f"outcome must be one of {list(OUTCOMES)}, got {condition!r}")
    if not 2 <= status_bin <= N_BINS:
        raise HRSIngestError(f"status_bin must be in 2..{N_BINS}, got {status_bin}")
    channels = CHANNEL_SETS[channel_set]
    channel_names = tuple(c.name for c in channels)
    needed = chronic_columns([condition], channel_set)
    if static_set == "core":  # the core statics need none of the extended columns
        needed = [c for c in needed if c not in static_columns() or c in ("rabyear", "ragender")]
    absent = [c for c in needed if c not in frame.values.columns]
    if absent:
        raise HRSIngestError(f"{len(absent)} required column(s) absent: {', '.join(absent[:10])}")

    n_file = frame.n
    iwstat = _wave_matrix(frame, "r{w}iwstat")
    age_m = _wave_matrix(frame, "r{w}agem_e")
    usable = (iwstat == RESPONDED) & ~np.isnan(age_m)
    bins = age_bin(age_m)
    birth_year = frame.values["rabyear"].to_numpy()
    sex = frame.values["ragender"].to_numpy()
    history = usable & (bins >= 1) & (bins < status_bin)

    if condition == DEATH:
        status_wave = _last_true(usable & (bins == status_bin))
        label, unknown = _death_label(frame, iwstat, age_m, birth_year, status_wave, death_horizon_years)
        title = f"death within {death_horizon_years} years of an interview at"
    else:
        status = _wave_matrix(frame, condition_stem(condition))
        status_wave = _last_true(usable & (bins == status_bin) & ~np.isnan(status))
        label, unknown = _pick(status, status_wave), np.zeros(n_file, dtype=bool)
        title = f"{CONDITIONS[condition]} by"

    reasons = {
        "no_interview_in_status_bin": status_wave < 0,
        "birth_year_or_sex_missing": np.isnan(birth_year) | np.isnan(sex),
        "no_interview_before_status_bin": ~history.any(axis=1),
        "vital_status_unknown_at_horizon": unknown,
    }
    excluded = np.zeros(n_file, dtype=bool)
    exclusions = {}
    for reason, hit in reasons.items():  # first reason wins, so the counts add up
        exclusions[reason] = int((hit & ~excluded).sum())
        excluded |= hit
    keep = ~excluded

    K = status_bin - 1
    X, code_counts, _ = _channel_array(
        frame, bins, history & keep[:, None], np.full(n_file, K, dtype=np.int16), channels
    )
    X = X[keep][:, :, :K].astype(np.float32)
    M = ~np.isnan(X)
    y = label[keep].astype(np.int8)
    first_bin = np.where(history[keep], bins[keep], N_BINS + 1).min(axis=1)

    static = static_matrix(frame, STATIC_SETS[static_set], _first_true(history), status_wave)[keep]
    report = {
        "condition": condition,
        "label": CONDITIONS.get(condition, f"death within {death_horizon_years} years"),
        "title": title,
        "death_horizon_years": death_horizon_years if condition == DEATH else None,
        "status_bin": status_bin,
        "status_ages": list(bin_ages(status_bin)),
        "image_bins": K,
        "n_file": n_file,
        "n_kept": int(keep.sum()),
        "excluded": exclusions,
        "cases": int(y.sum()),
        "controls": int((y == 0).sum()),
        "first_observed_bin": np.bincount(first_bin, minlength=K + 1)[1 : K + 1].tolist(),
        "channel_set": channel_set,
        "static_set": static_set,
        "observed_fraction": {name: float(M[:, c].mean()) if M.size else 0.0 for c, name in enumerate(channel_names)},
        "missing_codes": {name: code_counts[c] for c, name in enumerate(channel_names)},
        "static_missing_fraction": {},  # filled below, once the static matrix exists
    }
    report["static_missing_fraction"] = {
        name: float(np.isnan(static[:, j]).mean()) if len(static) else 0.0 for j, name in enumerate(STATIC_SETS[static_set])
    }
    return ChronicDataset(
        X=X,
        M=M,
        Delta=_delta(M),
        static=static,
        static_names=STATIC_SETS[static_set],
        channel_names=channel_names,
        channel_domains=tuple(channel_domains(channels)),
        y=y,
        household=(frame.values["hhidpn"].to_numpy()[keep] // 1000).astype(np.int64),
        condition=condition,
        status_bin=status_bin,
        report=report,
    )


def _death_label(
    frame: RandFrame, iwstat: np.ndarray, age_m: np.ndarray, birth_year: np.ndarray,
    status_wave: np.ndarray, horizon_years: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(1 = died within the horizon / 0 = known alive at it, NaN otherwise; unknown mask)."""
    death_m = frame.values["radage_m"].to_numpy()
    horizon_m = _pick(age_m, status_wave) + 12 * horizon_years
    died = np.isfinite(death_m) & (death_m <= horizon_m)
    later = np.arange(iwstat.shape[1])[None, :] > status_wave[:, None]
    interviewed_past = (iwstat == RESPONDED) & (age_m >= horizon_m[:, None])
    nominal_age_m = (np.array([nominal_wave_year(w) for w in ALL_WAVES])[None, :] - birth_year[:, None]) * 12
    seen_alive_past = (iwstat == ALIVE_NOT_INTERVIEWED) & (nominal_age_m >= horizon_m[:, None] + 12)
    alive = ~died & (
        (np.isfinite(death_m) & (death_m > horizon_m)) | (later & (interviewed_past | seen_alive_past)).any(axis=1)
    )
    label = np.where(died, 1.0, np.where(alive, 0.0, np.nan))
    return label, (status_wave >= 0) & np.isnan(label)
