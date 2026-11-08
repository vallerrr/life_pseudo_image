"""HRS ingestion on the synthetic RAND fixture: schema, outcome rules, left truncation.

HRS is blind (D07), so nothing here touches ``data/hrs/``. The fixture is written from
the codebook (``tests/fixtures/rand_hrs_fixture.py``); single-person frames built by
hand pin down each outcome rule exactly.
"""

import re

import numpy as np
import pandas as pd
import pytest
from pandas.io.stata import StataMissingValue

from fixtures import rand_hrs_fixture as fx
from utils import shared_paths as P
from utils.shared_hrs_ingest import (
    CHANNEL_NAMES,
    CHANNELS,
    CPI_BASE_YEAR,
    CPI_U,
    HRSIngestError,
    N_BINS,
    STATIC_COLUMNS,
    WAVE_STEMS,
    RandFrame,
    build_hrs_model_input,
    hrs_model_input_path,
    ingest_rand,
    required_columns,
)
from utils.shared_model_input import CENSORED, DEATH, DISEASE, STREAM_NAMES, read_model_input, write_model_input

C = {name: c for c, name in enumerate(CHANNEL_NAMES)}


@pytest.fixture(scope="module")
def fixture():
    return fx.build_fixture()


@pytest.fixture(scope="module")
def fixture_path(fixture, tmp_path_factory):
    return fx.write_fixture(tmp_path_factory.mktemp("rand") / "randhrs_fixture.dta", fixture)


@pytest.fixture(scope="module")
def ingested(fixture_path):
    return ingest_rand(fixture_path, chunksize=97)  # a chunk size that does not divide n


# --- hand-built people -----------------------------------------------------------
def person(waves: dict[int, dict], **static) -> dict:
    """One RAND row. ``waves[w]`` holds ``age`` (years), optional ``status`` (default
    1), ``diabe`` (default 0), ``year`` and any ``"r{w}..."`` column stem -> value."""
    record = {"hhidpn": 1010, "hacohort": 3, "rabyear": 1940, "ragender": 2, "radage_m": ".x", **static}
    for w in range(1, 17):
        spec = dict(waves.get(w, {"status": 0}))
        record[f"r{w}iwstat"] = spec.pop("status", 1)
        if "age" in spec:
            record[f"r{w}agem_e"] = round(spec.pop("age") * 12)
            record[f"r{w}iwendy"] = spec.pop("year", 2000)
            record[f"r{w}diabe"] = spec.pop("diabe", 0)
        for stem, value in spec.items():
            record[stem.format(w=w)] = value
    return record


def ingest_people(*records: dict):
    return build_hrs_model_input(RandFrame.from_records(list(records)))


def test_left_truncation_sets_entry_and_leaves_earlier_bins_empty() -> None:
    result = ingest_people(person({1: {"age": 66.5, "r{w}bmi": 27.0}, 2: {"age": 68.5, "r{w}bmi": 28.0}}))
    assert result.entry_time.tolist() == [9]
    X, M = result.X[0], result.M[0]
    assert not M[:, :8].any()
    assert X[C["bmi"], 8] == 27.0 and X[C["bmi"], 9] == 28.0
    assert result.obs_time.tolist() == [10] and result.obs_type.tolist() == [CENSORED]
    # Never-observed bins count from bin 1, as the DGP counts them.
    assert result.Delta[0, C["bmi"], :9].tolist() == list(range(1, 9)) + [0]


def test_interviews_before_50_are_not_entry_but_count_for_prevalence() -> None:
    young = person({1: {"age": 47}, 2: {"age": 49}, 3: {"age": 51, "r{w}bmi": 30.0}, 4: {"age": 53}})
    prevalent = person({1: {"age": 47}, 2: {"age": 49, "diabe": 1}, 3: {"age": 51, "diabe": 1}}, hhidpn=2010)
    result = ingest_people(young, prevalent)
    assert result.person_id.tolist() == [1010]
    assert result.entry_time.tolist() == [1]
    assert result.metadata["ingest_report"]["excluded"]["disease_prevalent_at_entry"] == 1


def test_disease_is_dated_at_the_interview_that_detects_it() -> None:
    record = person({5: {"age": 60}, 6: {"age": 62, "status": 4}, 7: {"age": 64, "diabe": 1}})
    record["r6agem_e"] = "."  # a non-respondent has no interview age
    result = ingest_people(record)
    assert result.obs_type.tolist() == [DISEASE]
    assert result.obs_time.tolist() == [8]


def test_death_right_after_the_last_interview_is_an_event() -> None:
    record = person({5: {"age": 70}, 6: {"age": 72}, 7: {"status": 5}, 8: {"status": 6}}, radage_m=74.5 * 12)
    result = ingest_people(record)
    assert result.obs_type.tolist() == [DEATH]
    assert result.obs_time.tolist() == [13]


def test_death_after_a_missed_interview_is_censored_at_the_last_one() -> None:
    record = person({5: {"age": 70}, 6: {"age": 72}, 7: {"status": 4}, 8: {"status": 5}}, radage_m=75 * 12)
    result = ingest_people(record)
    assert result.obs_type.tolist() == [CENSORED]
    assert result.obs_time.tolist() == [12]
    assert result.metadata["ingest_report"]["deaths_not_counted"] == 1


def test_two_waves_in_one_bin_keep_the_later_non_missing_value() -> None:
    record = person(
        {
            5: {"age": 60.0, "r{w}bmi": 25.0, "r{w}smoken": 1},
            6: {"age": 61.9, "r{w}bmi": 30.0, "r{w}smoken": ".d"},
        }
    )
    X = ingest_people(record).X[0]
    assert X[C["bmi"], 5] == 30.0  # both interviews fall in bin 6
    assert X[C["smoking"], 5] == 1.0  # the later one is missing, so the earlier stands


def test_nothing_after_the_outcome_reaches_the_model() -> None:
    record = person({5: {"age": 60, "r{w}bmi": 25.0}, 6: {"age": 64, "diabe": 1}, 7: {"age": 66, "diabe": 1, "r{w}bmi": 31.0}})
    result = ingest_people(record)
    assert result.obs_time.tolist() == [8]
    assert not result.M[0, :, 8:].any()


def test_life_past_90_is_censored_at_the_last_bin() -> None:
    record = person({14: {"age": 88, "r{w}bmi": 24.0}, 15: {"age": 90.5, "r{w}bmi": 23.0}, 16: {"age": 92.5, "diabe": 1}})
    result = ingest_people(record)
    assert result.obs_type.tolist() == [CENSORED]
    assert result.obs_time.tolist() == [N_BINS]
    assert result.X[0, C["bmi"], N_BINS - 1] == 24.0


def test_channel_transforms() -> None:
    record = person(
        {
            6: {"age": 60, "r{w}vigact": 1},
            7: {"age": 62, "r{w}vgactx": 2},
            8: {"age": 64, "r{w}lbsatwlf": 6.0, "r{w}vgactx": 3, "year": 2001, "h{w}itot": 50_000.0},
            9: {"age": 66, "r{w}lbsatwlf": 7.0, "r{w}lbonchrstr": 2.0, "r{w}lbnegaffect": ".a"},
        }
    )
    X = ingest_people(record).X[0]
    assert X[C["vigorous_activity"], [5, 6, 7]].tolist() == [1.0, 1.0, 0.0]
    assert X[C["life_satisfaction"], [7, 8]].tolist() == [7.0, 7.0]  # wave 8's 1-6 mapped onto 1-7
    real = 50_000.0 * CPI_U[CPI_BASE_YEAR] / CPI_U[2000] / 1000  # last year's income, deflated
    assert X[C["income"], 7] == pytest.approx(np.arcsinh(real), rel=1e-6)
    # r9lbonchrstr does not exist (not asked in 2008), so wave 9's value is never read.
    assert np.isnan(X[C["chronic_stress"], 8]) and np.isnan(X[C["negative_affect"], 8])


def test_missing_required_column_names_it() -> None:
    frame = RandFrame.from_records([person({1: {"age": 60}})])
    frame = RandFrame(frame.values.drop(columns="r9lbsatwlf"), frame.codes.drop(columns="r9lbsatwlf"))
    with pytest.raises(HRSIngestError, match="r9lbsatwlf"):
        build_hrs_model_input(frame)


# --- the fixture itself ------------------------------------------------------------
def test_fixture_round_trips_codes_and_labels(fixture, fixture_path) -> None:
    data = pd.read_stata(fixture_path, convert_missing=True, convert_categoricals=False)
    assert list(data.columns) == list(fixture.columns)
    for name, (values, codes) in fixture.columns.items():
        read = data[name].to_numpy()
        got = np.array([v.string if isinstance(v, StataMissingValue) else "" for v in read], dtype=object)
        assert (got == codes).all(), name
        valid = codes == ""
        assert np.allclose(read[valid].astype(float), values[valid], equal_nan=False), name
    with pd.read_stata(fixture_path, iterator=True) as reader:
        labels = reader.value_labels()
        variable_labels = reader.variable_labels()
    assert labels["r3work"][1] == "1.Working for pay"
    assert variable_labels["r8lbsatwlf"].startswith("R8LBSATWLF:W8 ")
    codes_used = {c for _, codes in fixture.columns.values() for c in codes}
    assert {".", ".a", ".d", ".i", ".j", ".m", ".n", ".p", ".q", ".r", ".x"} <= codes_used


def test_fixture_leave_behind_rotates_by_household_half(fixture) -> None:
    for stem in ("r{w}lblonely3", "r{w}lbsatwlf"):
        seen = np.column_stack([fixture.columns[stem.format(w=w)][1] == "" for w in range(8, 17)])
        # Never in two consecutive waves, except the 2022 entrants who were never assigned a half.
        consecutive = (seen[:, 1:] & seen[:, :-1]).any(axis=1)
        assert not consecutive[fixture.people.hacohort.to_numpy() != 8].any()


def test_ingest_reads_exactly_the_waves_the_codebook_lists() -> None:
    wanted: dict[str, set[int]] = {}
    for channel in CHANNELS:
        for source in channel.sources:
            wanted.setdefault(source.stem, set()).update(source.waves)
    for stem in WAVE_STEMS:
        wanted[stem] = set(range(1, 17))
    for stem, waves in wanted.items():
        assert waves == set(fx.COLUMN_WAVES[stem]), stem
    assert set(STATIC_COLUMNS) <= set(fx.STATIC)
    assert "r8lbnegaffect6" not in required_columns()  # a different instrument


# --- ingesting the fixture --------------------------------------------------------
def test_arrays_have_the_model_input_shape(ingested) -> None:
    data = ingested
    n = data.n
    assert data.X.shape == data.M.shape == data.Delta.shape == (n, 12, N_BINS)
    assert (data.X.dtype, data.M.dtype, data.Delta.dtype, data.S.dtype) == (np.float32, bool, np.int16, np.float32)
    assert data.S.shape == (n, 3) and np.isnan(data.S[:, 2]).all()  # frailty is unmeasured
    assert np.array_equal(data.M, ~np.isnan(data.X))
    assert len(CHANNEL_NAMES) == 4 * len(STREAM_NAMES)
    assert [c.stream for c in CHANNELS] == [s for s in STREAM_NAMES for _ in range(4)]


def test_entry_outcome_and_mask_agree(ingested) -> None:
    data, entry = ingested, ingested.entry_time
    bins = np.arange(1, N_BINS + 1)
    assert ((entry >= 1) & (entry <= data.obs_time) & (data.obs_time <= N_BINS)).all()
    assert not (data.M & (bins < entry[:, None])[:, None, :]).any()  # nothing before entry
    assert not (data.M & (bins > data.obs_time[:, None])[:, None, :]).any()  # nor after the outcome
    assert set(np.unique(data.obs_type)) == {CENSORED, DISEASE, DEATH}
    assert (entry > 1).mean() > 0.2  # left truncation is actually exercised
    expected = np.zeros_like(data.Delta)
    elapsed = np.zeros(data.Delta.shape[:2], dtype=np.int16)
    for t in range(N_BINS):
        elapsed = np.where(data.M[:, :, t], 0, elapsed + 1)
        expected[:, :, t] = elapsed
    assert np.array_equal(data.Delta, expected)


def test_report_accounts_for_every_row(ingested, fixture) -> None:
    report = ingested.metadata["ingest_report"]
    assert report["n_file"] == fixture.n
    assert report["n_kept"] + sum(report["excluded"].values()) == fixture.n
    assert sum(report["outcomes"].values()) == report["n_kept"]
    assert report["values_without_response"] == 0
    assert report["missing_codes"]["loneliness"][".a"] > 0  # the half not asked
    assert report["missing_codes"]["vigorous_activity"][".q"] > 0  # AHEAD wave 2A


def test_midstream_is_empty_before_the_leave_behind_existed(ingested, fixture) -> None:
    # A person whose last interview at ages 50-89 predates 2006 (wave 8) has no midstream.
    ages = np.column_stack([fixture.columns[f"r{w}agem_e"][0] for w in range(1, 17)])
    in_window = (ages >= 600) & (ages < 1080)
    late = in_window[:, 7:].any(axis=1)
    early_only = dict(zip(fixture.columns["hhidpn"][0].astype(int), ~late))
    rows = np.array([early_only[h] for h in ingested.person_id])
    midstream = [C[n] for n in ("loneliness", "life_satisfaction", "chronic_stress", "negative_affect")]
    assert rows.any() and not ingested.M[rows][:, midstream].any()


def test_upper_case_names_read_the_same(fixture, ingested, tmp_path) -> None:
    upper = ingest_rand(fx.write_fixture(tmp_path / "upper.dta", fixture, uppercase=True))
    assert np.array_equal(upper.M, ingested.M)
    assert np.allclose(upper.X, ingested.X, equal_nan=True)
    assert np.array_equal(upper.obs_time, ingested.obs_time)


def test_written_file_reads_back(ingested, tmp_path) -> None:
    back = read_model_input(write_model_input(ingested, tmp_path / "hrs.npz"))
    assert np.array_equal(back.entry_time, ingested.entry_time)
    assert np.array_equal(back.person_id, ingested.person_id)
    assert np.array_equal(back.M, ingested.M)
    assert back.truth == {}  # HRS has no truth; entry_time and the ids are not filed under it
    assert hrs_model_input_path().parent == P.HRS_MODEL_INPUT_DIR


# --- drift guards ----------------------------------------------------------------
def test_clock_matches_the_life_image() -> None:
    from utils import data_creation_00_data_creation_life_image as life
    from utils import shared_hrs_ingest as hrs

    assert (hrs.AGE_AT_FIRST_BIN, hrs.YEARS_PER_BIN) == (life.AGE_AT_FIRST_BIN, life.YEARS_PER_BIN)


def test_dictionary_documents_every_column_read() -> None:
    text = (P.PROJECT_ROOT / "doc" / "data" / "hrs.md").read_text()
    stems = {s.stem for c in CHANNELS for s in c.sources} | set(WAVE_STEMS)
    for stem in stems:
        head, tail = stem.split("{w}")
        assert f"{head.upper()}w{tail.upper()}" in text, stem
    for name in STATIC_COLUMNS:
        assert re.search(rf"\b{name.upper()}\b", text), name
