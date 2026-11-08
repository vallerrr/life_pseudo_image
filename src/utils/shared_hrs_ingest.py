"""HRS ingestion: the RAND HRS Longitudinal File, wide, into model-input arrays.

The RAND file has one row per person and one column per (concept, wave): ``r8bmi`` is
BMI at wave 8. This module turns the columns named in ``doc/data/hrs.md`` into the
arrays every model reads (``shared_model_input``): ``X``, ``M``, ``Delta``, ``S``,
``obs_time`` and ``obs_type``, on the life image's clock of 20 two-year bins at ages
50-90.

HRS is ``blind`` (D07). This code is written against the codebook alone and tested on
the synthetic fixture in ``tests/fixtures/rand_hrs_fixture.py``; the real pass is run
by a human. Error messages name columns and counts, never values, so an exception
raised on the real file can be reported without leaking a row.

Three things differ from the simulation, and each is handled here rather than left to
the models:

* **Left truncation.** People enter the panel at different ages: an AHEAD respondent
  first interviewed at 75 starts in bin 13. ``ModelInput.entry_time`` records the first
  bin at risk, and evaluation and the models' losses start each risk set there. Bins before it are unobserved by design, not by nonresponse, and a person
  with the disease already at entry is not at risk and is excluded.
* **Time is age, not wave.** A wave lands in the bin holding the respondent's age at
  that interview. Two waves can share a bin and a bin can hold none; when two share,
  the later non-missing value wins.
* **The outcome is detected, not timed.** Diabetes is seen at the first interview
  reporting it; death counts only when it follows the last interview with no
  interview missed in between. Otherwise the person is censored at their last
  disease-free interview, as ``observed_outcome`` censors the simulation at attrition.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

from utils import shared_paths as P
from utils.shared_model_input import (
    CENSORED,
    DEATH,
    DISEASE,
    STATIC_NAMES,
    STREAM_NAMES,
    ModelInput,
)
from utils.shared_reproducibility import git_hash

N_WAVES = 16
# The life image's clock. Duplicated from data_creation_00_data_creation_life_image,
# which imports matplotlib; tests/test_hrs_ingest.py fails if the two drift apart.
AGE_AT_FIRST_BIN = 50
YEARS_PER_BIN = 2
N_BINS = 20
FIRST_BIN_MONTHS = AGE_AT_FIRST_BIN * 12
BIN_MONTHS = YEARS_PER_BIN * 12

RESPONDED, DIED_THIS_WAVE = 1, 5  # RwIWSTAT codes
FEMALE = 2  # RAGENDER

# CPI-U, U.S. city average, all items, annual average (1982-84 = 100), BLS series
# CUUR0000SA0. RAND ships its own copy only as a SAS format, which the Stata release
# lacks. Transcribed by hand; checked against the BLS API (annual average, period M13)
# on 2026-10-25: all 33 years 1991-2023 matched exactly, and 2024 was added (D19).
CPI_U = {
    1991: 136.2, 1992: 140.3, 1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9,
    1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2, 2001: 177.1, 2002: 179.9,
    2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.342, 2008: 215.303,
    2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957,
    2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107,
    2019: 255.657, 2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702,
    2024: 313.689,
}
CPI_BASE_YEAR = 2022


class HRSIngestError(ValueError):
    """An ingestion failure whose message carries names and counts, never values."""


# --- the dictionary, as code -------------------------------------------------
def _identity(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    return values


def _real_thousands(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Nominal dollars of calendar ``years`` to asinh of thousands of base-year dollars.

    asinh is log-like for large values and defined at zero and below, which total
    wealth reaches. A year outside the CPI table yields NaN rather than a guess.
    """
    index = np.array([CPI_U.get(int(y), np.nan) if np.isfinite(y) else np.nan for y in years])
    return np.arcsinh(values * (CPI_U[CPI_BASE_YEAR] / index) / 1000.0)


def _income(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # HwITOT is last calendar year's income, so it is deflated by the year before.
    return _real_thousands(values, frame.values[f"r{wave}iwendy"].to_numpy() - 1)


def _wealth(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    return _real_thousands(values, frame.values[f"r{wave}iwendy"].to_numpy())


def _life_satisfaction(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # Wave 8 items were scored 1-6; from wave 9 they are 1-7. Map 1-6 onto 1-7.
    return 1.0 + (values - 1.0) * 6.0 / 5.0 if wave == 8 else values


def _vigorous_weekly(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # RwVGACTX: 1 every day, 2 more than once a week, 3 once a week, 4 1-3 a month,
    # 5 never. "More than once a week" is the nearest cut to RwVIGACT's "3+ a week".
    return np.where(np.isnan(values), np.nan, (values <= 2).astype(float))


@dataclass(frozen=True)
class Source:
    """One RAND variable family feeding a channel over some waves."""

    stem: str  # lowercase RAND name with the wave as "{w}", e.g. "h{w}itot"
    waves: tuple[int, ...]
    transform: Callable[[np.ndarray, int, RandFrame], np.ndarray] = _identity
    # More stems read on the same waves, for a transform that combines items
    # (``shared_hrs_variables``' sleep score); the transform reads them from the frame.
    extra: tuple[str, ...] = ()

    def column(self, wave: int) -> str:
        return self.stem.format(w=wave)

    def columns(self, wave: int) -> list[str]:
        return [self.column(wave)] + [e.format(w=wave) for e in self.extra]


@dataclass(frozen=True)
class Channel:
    name: str
    stream: str
    sources: tuple[Source, ...]


ALL_WAVES = tuple(range(1, N_WAVES + 1))
LB_WAVES = tuple(range(8, N_WAVES + 1))

# Row order is channel order: four per stream, upstream first (doc/data/hrs.md).
CHANNELS: tuple[Channel, ...] = (
    Channel("income", "upstream", (Source("h{w}itot", ALL_WAVES, _income),)),
    Channel("wealth", "upstream", (Source("h{w}atotb", ALL_WAVES, _wealth),)),
    Channel("poverty", "upstream", (Source("h{w}inpov", ALL_WAVES),)),
    Channel("working", "upstream", (Source("r{w}work", ALL_WAVES),)),
    Channel("loneliness", "midstream", (Source("r{w}lblonely3", LB_WAVES),)),
    Channel("life_satisfaction", "midstream", (Source("r{w}lbsatwlf", LB_WAVES, _life_satisfaction),)),
    Channel("chronic_stress", "midstream", (Source("r{w}lbonchrstr", (8,) + tuple(range(10, N_WAVES + 1))),)),
    Channel("negative_affect", "midstream", (Source("r{w}lbnegaffect", tuple(range(9, N_WAVES + 1))),)),
    Channel("bmi", "downstream", (Source("r{w}bmi", ALL_WAVES),)),
    Channel("smoking", "downstream", (Source("r{w}smoken", ALL_WAVES),)),
    Channel("drinking_days", "downstream", (Source("r{w}drinkd", tuple(range(3, N_WAVES + 1))),)),
    Channel(
        "vigorous_activity",
        "downstream",
        (Source("r{w}vigact", tuple(range(1, 7))), Source("r{w}vgactx", tuple(range(7, N_WAVES + 1)), _vigorous_weekly)),
    ),
)
CHANNEL_NAMES = tuple(c.name for c in CHANNELS)

STATIC_COLUMNS = ("hhidpn", "hacohort", "rabyear", "ragender", "radage_m")
WAVE_STEMS = ("r{w}iwstat", "r{w}agem_e", "r{w}iwendy", "r{w}diabe")


def required_columns(channels: tuple[Channel, ...] | None = None) -> list[str]:
    """Every column ingestion reads, lowercase, in a stable order."""
    columns = list(STATIC_COLUMNS)
    columns += [stem.format(w=w) for stem in WAVE_STEMS for w in ALL_WAVES]
    for column in (c for ch in (channels or CHANNELS) for s in ch.sources for w in s.waves for c in s.columns(w)):
        if column not in columns:
            columns.append(column)
    return columns


# --- reading -------------------------------------------------------------------
@dataclass(frozen=True)
class RandFrame:
    """Selected RAND columns split into values and Stata missing codes.

    ``values`` is float64 with NaN wherever a value is missing; ``codes`` holds the
    Stata missing code there (``"."``, ``".d"``, ``".a"``, ...) and ``""`` elsewhere.
    Column names are lowercase whatever the file used.
    """

    values: pd.DataFrame
    codes: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.values)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], columns: list[str] | None = None) -> RandFrame:
        """Build a frame by hand: a string starting with ``"."`` is a missing code.

        A column a record leaves out is plain missing (``"."``), as RAND codes a wave
        the person did not respond to.
        """
        columns = columns or required_columns()
        values = np.full((len(records), len(columns)), np.nan)
        codes = np.full((len(records), len(columns)), "", dtype=object)
        for i, record in enumerate(records):
            for j, column in enumerate(columns):
                cell = record.get(column, ".")
                if isinstance(cell, str):
                    codes[i, j] = cell.lower()
                else:
                    values[i, j] = cell
        return cls(pd.DataFrame(values, columns=columns), pd.DataFrame(codes, columns=columns))


def _split_missing(column: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """A column read with ``convert_missing=True`` into (values, codes)."""
    if column.dtype != object:
        values = column.to_numpy(dtype=float)
        codes = np.where(np.isnan(values), ".", "").astype(object)
        return values, codes
    raw = column.to_numpy()
    is_code = np.fromiter((isinstance(v, StataMissingValue) for v in raw), dtype=bool, count=len(raw))
    values = np.full(len(raw), np.nan)
    values[~is_code] = raw[~is_code].astype(float)
    codes = np.full(len(raw), "", dtype=object)
    codes[is_code] = [v.string for v in raw[is_code]]
    return values, codes


def read_rand(path: Path = P.RAND_HRS_FILE, columns: list[str] | None = None, chunksize: int = 2000) -> RandFrame:
    """Read ``columns`` of a RAND Stata file, matching names case-insensitively.

    Read in chunks: pandas builds every column of a chunk before selecting, and the
    RAND file is ~1.7 GB wide. A requested column the file lacks is an error, because
    the dictionary says it exists.
    """
    columns = columns or required_columns()
    with pd.read_stata(path, iterator=True) as reader:
        actual = {name.lower(): name for name in reader.variable_labels()}
    absent = [c for c in columns if c not in actual]
    if absent:
        raise HRSIngestError(f"{len(absent)} dictionary column(s) not in the file: {', '.join(absent)}")
    names = [actual[c] for c in columns]
    chunks = []
    with pd.read_stata(
        path,
        columns=names,
        convert_categoricals=False,
        convert_dates=False,
        convert_missing=True,
        preserve_dtypes=False,
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)
    values, codes = {}, {}
    for name, column in zip(columns, names):
        values[name], codes[name] = _split_missing(data[column])
    return RandFrame(pd.DataFrame(values), pd.DataFrame(codes))


# --- building the model input ---------------------------------------------------
def age_bin(months: np.ndarray) -> np.ndarray:
    """1-based bin of an age in months; 0 below age 50, ``N_BINS + 1`` from age 90."""
    bins = np.floor((months - FIRST_BIN_MONTHS) / BIN_MONTHS) + 1
    bins = np.where(np.isnan(months), 0, np.clip(bins, 0, N_BINS + 1))
    return bins.astype(np.int16)


def _wave_matrix(frame: RandFrame, stem: str) -> np.ndarray:
    return np.column_stack([frame.values[stem.format(w=w)].to_numpy() for w in ALL_WAVES])


def _zscore(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / (sd if sd > 0 else 1.0)


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Column of the first True in each row, -1 where none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


def _last_true(mask: np.ndarray) -> np.ndarray:
    last = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    return np.where(mask.any(axis=1), last, -1)


def _pick(matrix: np.ndarray, column: np.ndarray, fill: float = np.nan) -> np.ndarray:
    rows = np.arange(matrix.shape[0])
    return np.where(column >= 0, matrix[rows, np.maximum(column, 0)], fill)


def build_hrs_model_input(frame: RandFrame) -> ModelInput:
    """The RAND columns of ``frame`` as model input, with an ingestion report."""
    missing = [c for c in required_columns() if c not in frame.values.columns]
    if missing:
        raise HRSIngestError(f"{len(missing)} required column(s) absent: {', '.join(missing)}")

    n_file = frame.n
    iwstat = _wave_matrix(frame, "r{w}iwstat")
    age_m = _wave_matrix(frame, "r{w}agem_e")
    diabe = _wave_matrix(frame, "r{w}diabe")
    responded = iwstat == RESPONDED
    usable = responded & ~np.isnan(age_m)
    bins = age_bin(age_m)
    in_window = usable & (bins >= 1) & (bins <= N_BINS)

    # Entry: the first interview at ages 50-89. Waves are chronological, so it is
    # also the youngest; any earlier interview was below 50.
    entry_wave = _first_true(in_window)
    entry_time = _pick(bins, entry_wave, 0).astype(np.int16)
    status_at_entry = _pick(diabe, entry_wave)
    upto_entry = np.arange(N_WAVES)[None, :] <= entry_wave[:, None]
    prevalent = (responded & upto_entry & (diabe == 1)).any(axis=1)

    # Outcome. Disease is detected at the first later interview reporting it.
    after_entry = np.arange(N_WAVES)[None, :] > entry_wave[:, None]
    onset_wave = _first_true(responded & after_entry & (diabe == 1))
    onset_bin = _pick(bins, onset_wave, N_BINS + 1)
    before_onset = (onset_wave[:, None] < 0) | (np.arange(N_WAVES)[None, :] < onset_wave[:, None])
    free_wave = _last_true(usable & before_onset & (diabe == 0))
    free_bin = np.minimum(_pick(bins, free_wave, 0), N_BINS)

    # Death counts only when nothing could have been missed between the last
    # interview and death: that interview was disease-free and the next wave
    # records the death (or there is no next wave yet).
    death_m = frame.values["radage_m"].to_numpy()
    last_wave = _last_true(responded)
    next_status = _pick(np.column_stack([iwstat, np.full(len(iwstat), np.nan)]), last_wave + 1)
    death_follows = (last_wave == N_WAVES - 1) | (next_status == DIED_THIS_WAVE)
    death_after_last = death_m >= _pick(age_m, last_wave)
    death_seen = (
        ~np.isnan(death_m) & death_follows & death_after_last & (free_wave == last_wave) & (onset_wave < 0)
    )
    death_bin = age_bin(death_m)

    disease_seen = (onset_wave >= 0) & (onset_bin <= N_BINS)
    death_seen &= death_bin <= N_BINS
    obs_type = np.where(disease_seen, DISEASE, np.where(death_seen, DEATH, CENSORED)).astype(np.int8)
    obs_time = np.where(disease_seen, onset_bin, np.where(death_seen, death_bin, free_bin)).astype(np.int16)

    birth_year = frame.values["rabyear"].to_numpy()
    sex = frame.values["ragender"].to_numpy()
    reasons = {
        "no_interview_aged_50_89": entry_wave < 0,
        "birth_year_or_sex_missing": np.isnan(birth_year) | np.isnan(sex),
        "disease_status_missing_at_entry": np.isnan(status_at_entry),
        "disease_prevalent_at_entry": prevalent,
    }
    excluded = np.zeros(n_file, dtype=bool)
    exclusions = {}
    for reason, hit in reasons.items():  # first reason wins, so the counts add up
        exclusions[reason] = int((hit & ~excluded).sum())
        excluded |= hit
    keep = ~excluded
    inconsistent = keep & (obs_time < entry_time)
    if inconsistent.any():
        raise HRSIngestError(f"{int(inconsistent.sum())} person(s) with outcome before entry")

    X, code_counts, stray = _channel_array(frame, bins, in_window & keep[:, None], obs_time)
    X = X[keep]
    M = ~np.isnan(X)
    S = np.column_stack(
        (_zscore(birth_year[keep]), _zscore((sex[keep] == FEMALE).astype(float)), np.full(keep.sum(), np.nan))
    ).astype(np.float32)

    report = {
        "n_file": n_file,
        "n_kept": int(keep.sum()),
        "excluded": exclusions,
        "outcomes": {name: int((obs_type[keep] == code).sum()) for code, name in ((CENSORED, "censored"), (DISEASE, "disease"), (DEATH, "death"))},
        # Recorded deaths censored instead: an interview was missed first, or age 90+.
        "deaths_not_counted": int((keep & ~np.isnan(death_m) & ~death_seen & ~disease_seen).sum()),
        "entry_bin_counts": np.bincount(entry_time[keep], minlength=N_BINS + 1)[1:].tolist(),
        "observed_fraction": {name: float(M[:, c].mean()) for c, name in enumerate(CHANNEL_NAMES)},
        "missing_codes": {name: code_counts[c] for c, name in enumerate(CHANNEL_NAMES)},
        "values_without_response": stray,
    }
    return ModelInput(
        X=X.astype(np.float32),
        M=M,
        Delta=_delta(M),
        S=S,
        obs_time=obs_time[keep],
        obs_type=obs_type[keep],
        truth={},
        metadata={
            "source": "hrs",
            "rand_file": P.RAND_HRS_FILE.name,
            "static_names": list(STATIC_NAMES),
            "channel_names": list(CHANNEL_NAMES),
            "stream_names": list(STREAM_NAMES),
            "disease": "r{w}diabe",
            "cpi_base_year": CPI_BASE_YEAR,
            "ingest_report": report,
            "git_hash": git_hash(),
        },
        entry_time=entry_time[keep],
        person_id=frame.values["hhidpn"].to_numpy()[keep].astype(np.int64),  # HHIDPN
    )


def _channel_array(
    frame: RandFrame, bins: np.ndarray, in_window: np.ndarray, obs_time: np.ndarray,
    channels: tuple[Channel, ...] | None = None,
) -> tuple[np.ndarray, list[dict[str, int]], int]:
    """(n, C, T) values on the age grid, missing-code tallies, and stray values.

    A wave writes into its age bin up to and including ``obs_time``; nothing after
    the outcome reaches a model. Waves are visited in order, so a later non-missing
    value in the same bin overwrites an earlier one.
    """
    channels = channels or CHANNELS
    n = frame.n
    X = np.full((n, len(channels), N_BINS), np.nan)
    code_counts: list[dict[str, int]] = []
    stray = 0
    rows = np.arange(n)
    for c, channel in enumerate(channels):
        counts: dict[str, int] = {}
        for source in channel.sources:
            for w in source.waves:
                column = source.column(w)
                raw = frame.values[column].to_numpy()
                eligible = in_window[:, w - 1]
                stray += int((~np.isnan(raw) & (frame.values[f"r{w}iwstat"].to_numpy() != RESPONDED)).sum())
                for code, k in pd.Series(frame.codes[column].to_numpy()[eligible]).value_counts().items():
                    if code:
                        counts[code] = counts.get(code, 0) + int(k)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # asinh/deflator of NaN
                    value = source.transform(raw, w, frame)
                b = bins[:, w - 1]
                write = eligible & (b <= obs_time) & ~np.isnan(value)
                X[rows[write], c, b[write] - 1] = value[write]
        code_counts.append(dict(sorted(counts.items())))
    return X, code_counts, stray


def _delta(M: np.ndarray) -> np.ndarray:
    """Bins since last observation, counted as the DGP counts it (``shared_dgp``)."""
    n, C, T = M.shape
    delta = np.zeros((n, C, T), dtype=np.int16)
    elapsed = np.zeros((n, C), dtype=np.int16)
    for t in range(T):
        elapsed = np.where(M[:, :, t], 0, elapsed + 1).astype(np.int16)
        delta[:, :, t] = elapsed
    return delta


def ingest_rand(path: Path = P.RAND_HRS_FILE, chunksize: int = 2000) -> ModelInput:
    """Read the RAND file and build its model input. Run by a human: HRS is blind.

    Write it with ``write_model_input(data, hrs_model_input_path())``.
    """
    return build_hrs_model_input(read_rand(path, chunksize=chunksize))


def hrs_model_input_path(rand_file: Path = P.RAND_HRS_FILE) -> Path:
    """``model_input_hrs_<release>.npz``; the release is the RAND file's stem."""
    return P.HRS_MODEL_INPUT_DIR / f"model_input_hrs_{rand_file.stem}.npz"


def stem_pattern(stem: str) -> re.Pattern[str]:
    """A regex matching every wave of ``stem``: ``r{w}bmi`` -> ``r(\\d+)bmi``."""
    head, tail = stem.split("{w}")
    return re.compile(rf"^{re.escape(head)}(\d+){re.escape(tail)}$")
