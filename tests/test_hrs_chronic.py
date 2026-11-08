"""The blind-data protocol and the chronic-disease task, on hand-built rows and the fixture.

Nothing here reads ``data/hrs/``: HRS is blind (D07, D22).
"""

import importlib.util
import json

import numpy as np
import pytest
import torch

from fixtures import rand_hrs_fixture as fx
from utils import shared_paths as P
from utils.data_analysis_04_hrs_chronic_cnn import (
    PseudoImageClassifier,
    RecurrentClassifier,
    average_image,
    average_image_figure,
    build_classifier,
    grad_cam,
    household_folds,
)
from utils.data_creation_01_hrs_schema_report import schema_report, stem_of
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import BENCHMARKS, window_features
from utils.shared_hrs_chronic import CONDITIONS, build_chronic_dataset, chronic_columns
from utils.shared_hrs_ingest import CHANNELS
from utils.shared_hrs_variables import DOMAINS, EXTENDED, EXTENDED_VARIABLES, STATICS
from utils.shared_hrs_ingest import RandFrame, read_rand
from utils.shared_secure_report import SecureReport, guarded


@pytest.fixture(scope="module")
def fixture():
    return fx.build_fixture(1200)


@pytest.fixture(scope="module")
def fixture_path(fixture, tmp_path_factory):
    return fx.write_fixture(tmp_path_factory.mktemp("rand") / "randhrs_fixture.dta", fixture)


def person(waves: dict[int, dict], hhidpn: int = 1010, **static) -> dict:
    """One RAND row; ``waves[w]`` holds ``age`` in years and any ``"r{w}..."`` stem -> value."""
    record = {"hhidpn": hhidpn, "hacohort": 3, "rabyear": 1930, "ragender": 1, "radage_m": ".x", **static}
    for w in range(1, 17):
        spec = dict(waves.get(w, {"status": 0}))
        record[f"r{w}iwstat"] = spec.pop("status", 1)
        if "age" in spec:
            record[f"r{w}agem_e"] = round(spec.pop("age") * 12)
            record[f"r{w}iwendy"] = 2000
            for condition in CONDITIONS:
                record[f"r{w}{condition}"] = spec.pop(condition, 0)
        for stem, value in spec.items():
            record[stem.format(w=w)] = value
    return record


def chronic(*records: dict, condition: str = "diabe", status_bin: int = 13):
    frame = RandFrame.from_records(list(records), columns=chronic_columns())
    return build_chronic_dataset(frame, condition, status_bin)


# --- the report --------------------------------------------------------------------
def test_small_counts_and_thin_cells_are_suppressed(tmp_path) -> None:
    report = SecureReport("t", "hrs", out_dir=tmp_path)
    assert report.count(0) == 0 and report.count(10) == "<11" and report.count(11) == 11
    assert report.summary(np.arange(5.0)) == {"n": "<11"}
    summary = report.summary(np.arange(100.0))
    assert summary["n"] == 100 and "min" not in summary and "max" not in summary
    mean = report.mean_map(np.array([[20.0, 5.0]]), np.array([[20, 5]]))
    assert mean == [[1.0, None]]


def test_a_failure_is_reported_by_place_never_by_message(tmp_path) -> None:
    report = SecureReport("t", "hrs", out_dir=tmp_path)
    with pytest.raises(SystemExit):
        with guarded(report):
            raise ValueError("secret value 123456")
    written = next(tmp_path.glob("t_*.json")).read_text()
    assert "123456" not in written and "secret" not in written
    error = json.loads(written)["error"]
    assert error["type"] == "ValueError" and error["file"] == "tests/test_hrs_chronic.py"


# --- the task ------------------------------------------------------------------------
def test_label_is_the_status_interview_and_the_image_stops_before_it() -> None:
    case = person({5: {"age": 62, "r{w}bmi": 30.0}, 6: {"age": 64}, 11: {"age": 75, "diabe": 1}}, hhidpn=1010)
    control = person({8: {"age": 70}, 11: {"age": 74.5}}, hhidpn=2010)
    data = chronic(case, control, status_bin=13)  # ages 74-76
    assert data.n == 2 and data.y.tolist() == [1, 0]
    assert data.X.shape[2] == 12  # bins 1-12, ages 50-74
    assert data.X[0, 8, 6] == pytest.approx(30.0)  # BMI at 62 sits in bin 7
    assert data.household.tolist() == [1, 2]


def test_status_answers_are_not_in_the_image() -> None:
    record = person({8: {"age": 70}, 11: {"age": 75, "diabe": 1, "r{w}bmi": 40.0}})
    data = chronic(record, status_bin=13)
    assert not np.isclose(data.X, 40.0).any()


def test_without_an_interview_at_the_age_a_person_is_left_out() -> None:
    data = chronic(person({8: {"age": 70}, 11: {"age": 77}}), status_bin=13)
    assert data.n == 0 and data.report["excluded"]["no_interview_in_status_bin"] == 1


def test_prevalent_cases_stay_in() -> None:
    # Diabetic from the first interview on: a case, as the task is prevalence by A (D23).
    data = chronic(person({6: {"age": 64, "diabe": 1}, 11: {"age": 75, "diabe": 1}}), status_bin=13)
    assert data.y.tolist() == [1]


# death within the horizon (D25): status interview at 75, horizon 10 years -> age 85
def test_death_inside_the_horizon_is_a_case() -> None:
    data = chronic(person({8: {"age": 70}, 11: {"age": 75}}, radage_m=80 * 12), condition="death", status_bin=13)
    assert data.y.tolist() == [1]


def test_known_alive_past_the_horizon_is_a_control() -> None:
    by_death = person({8: {"age": 70}, 11: {"age": 75}}, hhidpn=1010, radage_m=90 * 12)
    by_interview = person({8: {"age": 70}, 11: {"age": 75}, 16: {"age": 86}}, hhidpn=2010)
    # Alive but not interviewed in wave 16 (2022), nominal age 2022 - 1930 = 92.
    by_status = person({8: {"age": 70}, 11: {"age": 75}, 16: {"status": 4}}, hhidpn=3010)
    data = chronic(by_death, by_interview, by_status, condition="death", status_bin=13)
    assert data.y.tolist() == [0, 0, 0]


def test_vital_status_unknown_at_the_horizon_is_left_out() -> None:
    lost = person({8: {"age": 70}, 11: {"age": 75}, 13: {"status": 7}})  # dropped, no death date
    data = chronic(lost, condition="death", status_bin=13)
    assert data.n == 0 and data.report["excluded"]["vital_status_unknown_at_horizon"] == 1


def test_window_means_follow_the_age_clock() -> None:
    # BMI 30 at 52 and 34 at 56 (window 50-59), 40 at 62 (60-69), nothing at 70+.
    record = person({2: {"age": 52, "r{w}bmi": 30.0}, 4: {"age": 56, "r{w}bmi": 34.0},
                     7: {"age": 62, "r{w}bmi": 40.0}, 11: {"age": 75}})
    data = chronic(record, status_bin=13)
    features = window_features(data, np.arange(data.n), [[50, 60], [60, 70], [70, 200]])
    C = data.X.shape[1]
    bmi = list(data.channel_names).index("bmi")
    assert features[0, bmi] == pytest.approx(32.0)
    assert features[0, C + bmi] == pytest.approx(40.0)
    assert np.isnan(features[0, 2 * C + bmi])


def test_households_never_straddle_folds() -> None:
    rng = np.random.default_rng(0)
    household = np.repeat(np.arange(200), 2)
    y = rng.integers(0, 2, size=400)
    for train, test in household_folds(y, household, 5, seed=1):
        assert not set(household[train]) & set(household[test])


# --- the extended dictionary (D26) --------------------------------------------------
def extended(*records: dict, status_bin: int = 13):
    frame = RandFrame.from_records(list(records), columns=chronic_columns())
    return build_chronic_dataset(frame, "diabe", status_bin, channel_set="extended", static_set="extended")


ROW = {c.name: i for i, c in enumerate(EXTENDED)}
STATIC = {s.varname: j for j, s in enumerate(STATICS)}


def test_extended_set_contains_the_core_and_is_grouped_by_domain() -> None:
    assert {c.name for c in CHANNELS} <= set(ROW)
    domains = [c.stream for c in EXTENDED]
    assert domains == sorted(domains, key=DOMAINS.index)  # rows contiguous by domain, in DOMAINS order


def test_extended_waves_match_the_fixture() -> None:
    for variable in EXTENDED_VARIABLES:
        for source in variable.sources:
            for stem in (source.stem, *source.extra):
                assert set(source.waves) == set(fx.COLUMN_WAVES[stem]), stem  # equal, so a missing wave fails here


def test_extended_dictionary_is_documented() -> None:
    text = (P.PROJECT_ROOT / "doc" / "data" / "hrs.md").read_text()
    for variable in EXTENDED_VARIABLES:
        assert f"`{variable.varname}`" in text, variable.varname
    for static in STATICS:
        assert f"`{static.varname}`" in text, static.varname


def test_extended_recodes() -> None:
    record = person({
        5: {"age": 62, "r{w}unemp": ".x", "h{w}ahous": 0.0, "r{w}mstat": 5, "r{w}mnev": ".n", "r{w}jlocc": 12},
        6: {"age": 64, "r{w}unemp": 1, "h{w}ahous": 250000.0, "r{w}mstat": 1,
            "r{w}sleepfal": 1, "r{w}sleepwkn": 1, "r{w}sleepwke": 3, "r{w}sleeprt": ".d"},  # sleep from wave 6
        7: {"age": 66, "r{w}sleepfal": 2, "r{w}sleeprt": ".d"},
        11: {"age": 75},
    }, rabplace=13, raracem=2, rahispan=0, raedyrs=12, rafeduc=".d", rameduc=8.5)
    data = extended(record)
    X = data.X[0]
    assert X[ROW["unemployed"], 6] == 0.0 and X[ROW["unemployed"], 7] == 1.0  # .x is not unemployed
    assert X[ROW["not_homeowner"], 6] == 1.0 and X[ROW["not_homeowner"], 7] == 0.0
    assert X[ROW["separated_divorced"], 6] == 1.0 and X[ROW["separated_divorced"], 7] == 0.0
    assert X[ROW["married_partnered"], 6] == 0.0 and X[ROW["married_partnered"], 7] == 1.0
    assert np.isnan(X[ROW["sleep_problems"], 6])  # wave 5 predates the sleep items
    assert X[ROW["sleep_problems"], 7] == pytest.approx((1 + 1 + 0) / 3)  # three items answered
    assert np.isnan(X[ROW["sleep_problems"], 8])  # one item only
    S = data.static[0]
    assert S[STATIC["foreign_born"]] == 1 and S[STATIC["black"]] == 1 and S[STATIC["hispanic"]] == 0
    assert S[STATIC["never_married"]] == 1  # .N counts as never married
    assert S[STATIC["lower_occupation"]] == 1  # 12 construction: manual
    assert np.isnan(S[STATIC["father_education"]]) and S[STATIC["mother_education"]] == 8.5


# --- the network --------------------------------------------------------------------------
def test_grad_cam_is_a_non_negative_map_on_the_input_grid() -> None:
    torch.manual_seed(0)
    model = PseudoImageClassifier(12, 2, 8, ((4, 5),), (1, 2, 4), 8, 0.0).eval()
    cam = grad_cam(model, torch.randn(5, 3, 12, 12), torch.randn(5, 2))
    assert cam.shape == (5, 12, 12) and (cam >= 0).all()


def test_rnn_reads_the_first_and_the_last_bin() -> None:
    torch.manual_seed(0)
    model = RecurrentClassifier(12, 2, 8, 1, 0.0).eval()
    image, static = torch.randn(3, 3, 12, 12), torch.randn(3, 2)
    for t in (0, 11):  # first and last bin both reach the logit (bidirectional)
        changed = image.clone()
        changed[:, :, :, t] += 1.0
        with torch.no_grad():
            assert not torch.allclose(model(image, static), model(changed, static))


def test_classifier_must_see_the_whole_image() -> None:
    architecture = {"kernels": [[4, 5]], "dilations": [1], "hidden": 8, "causal": False}
    with pytest.raises(ValueError, match="receptive field"):
        build_classifier(architecture, 12, 12, {"width": 8, "dropout": 0.0})


# --- end to end on the fixture ----------------------------------------------------------
def test_schema_report_on_the_fixture_has_no_rows(fixture, fixture_path, tmp_path) -> None:
    report = SecureReport("schema", "hrs", out_dir=tmp_path)
    schema_report(fixture_path, chronic_columns() + ["notacolumn"], report)
    body = json.loads(report.write().read_text())
    assert body["requested"]["absent"] == ["notacolumn"]
    assert body["file"]["stems"]["r{w}diabe"]["waves"] == list(range(1, 17))
    assert "levels" in body["requested"]["columns"]["r3diabe"]
    assert set(body["requested"]["columns"]["rabyear"]["summary"]) >= {"n", "p50"}
    ids = {str(int(v)) for v in fixture.columns["hhidpn"][0]}
    assert not any(i in json.dumps(body) for i in ids if len(i) > 6)  # no person id anywhere
    assert stem_of("r12bmi") == ("r{w}bmi", 12) and stem_of("rabyear") == ("rabyear", None)


def test_the_analysis_runs_end_to_end_on_the_fixture(fixture_path, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("smoke")
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.run(args, config)
    body = json.loads(next(tmp_path.glob("04_hrs_chronic_cnn_run_*.json")).read_text())
    assert set(body["conditions"]) == {"diabe", "death"}
    section = body["conditions"]["diabe"]
    assert set(section["metrics_mean"]) == {*BENCHMARKS, "cnn", "rnn"}
    tuning = section["folds"][0]["benchmarks"]
    assert {"lightgbm_snapshot", "lightgbm_summary", "lightgbm_cells", "elastic_net_cells"} <= set(tuning)
    rows = len(EXTENDED)
    assert np.array(section["maps"]["cnn"]["gradcam"]["difference"], dtype=float).shape == (rows, 12)
    assert np.array(section["maps"]["rnn"]["eg_signed"]["all"], dtype=float).shape == (rows, 12)
    assert len(body["meta"]["channel_domains"]) == rows and len(body["meta"]["static_names"]) == len(STATICS)
    assert set(section["dataset"]["static_missing_fraction"]) == set(body["meta"]["static_names"])
    assert "gradcam" not in section["maps"]["rnn"]  # Grad-CAM needs conv layers
    sanity = section["folds"][0]["sanity"]
    assert {"cnn_param_random_eg_mass_ratio", "rnn_label_random_eg_similarity"} <= set(sanity)


def test_average_image_is_peer_standardised_and_suppressed(fixture_path, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("smoke")
    frame = read_rand(fixture_path, columns=chronic_columns(["diabe"], config["channel_set"]))
    data = build_chronic_dataset(frame, "diabe", 13, 10, config["channel_set"], config["static_set"])
    totals = average_image(data, 11)
    thick = totals["all/count"] >= 11
    mean_all = totals["all/total"][thick] / totals["all/count"][thick]
    assert np.allclose(mean_all, 0.0, atol=1e-9)  # everyone together sits at their peers' mean

    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": ["diabe"], "status_bin": 13})()
    module.images(args, config)
    path = next(tmp_path.glob("04_hrs_chronic_cnn_images_*.json"))
    images = json.loads(path.read_text())["conditions"]["diabe"]["images"]
    cases = np.array(images["cases"], dtype=float)
    assert cases.shape == (len(EXTENDED), 12)
    thin = (totals["cases/count"] < 11) | ~np.isfinite(totals["peer_sd"])
    assert np.isnan(cases[thin]).all()
    difference = np.array(images["difference"], dtype=float)
    both = np.isfinite(cases) & np.isfinite(np.array(images["controls"], dtype=float))
    assert np.allclose(difference[both], (cases - np.array(images["controls"], dtype=float))[both])
    share = np.array(images["observed_share"], dtype=float)
    assert np.nanmax(share) <= 1.0 and np.isnan(share[totals["all/count"] < 11]).all()

    out = tmp_path / "figures"
    style = module.load_style(module.NOTEBOOK)
    fig = average_image_figure(json.loads(path.read_text())["conditions"]["diabe"],
                               json.loads(path.read_text())["meta"]["channel_names"], style,
                               json.loads(path.read_text())["meta"]["channel_domains"])
    out.mkdir()
    fig.savefig(out / "average.png")
    assert (out / "average.png").stat().st_size > 0


def test_kernel_experiment_runs_every_arm_on_the_fixture(fixture_path, tmp_path) -> None:
    """D30: the `kernels` profile, shrunk to seconds, end to end with its figures."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("kernels")
    smoke = module.load_config("smoke")
    assert config["run_benchmarks"] is False and len(config["networks"]) == 15 and config["conditions"] == ["diabe"]
    for arm in config["networks"].values():
        arm["grid"] = {"width": [6], "lr": [0.003], "weight_decay": [0.0001], "dropout": [0.1]}
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "attribution", "device")})
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.run(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_run_*.json"))
    body = json.loads(report.read_text())
    section = body["conditions"]["diabe"]
    assert set(section["metrics_mean"]) == set(config["networks"])  # no benchmarks, no RNN
    for arm in config["networks"]:
        assert np.array(section["maps"][arm]["gradcam"]["difference"], dtype=float).shape == (len(EXTENDED), 12)
    figures = type("Args", (), {"report": str(report), "reference": [("rnn", "0.750")]})()
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir  # keep the test's figures out of output/
    module.figures(figures)
    assert len(list(style_dir.glob("kernels_diabe_*"))) == 3 * len(module.load_style(module.NOTEBOOK)["formats"])


def test_row_permutation_round_trips() -> None:
    from utils.data_analysis_04_hrs_chronic_cnn_stability import domain_permutation, unpermute

    domains = tuple(c.stream for c in EXTENDED)
    order = domain_permutation(domains, np.random.default_rng(0))
    assert [domains[i] for i in order] == list(domains)  # rows stay inside their domain
    original = np.arange(len(domains) * 3, dtype=float).reshape(len(domains), 3)
    assert np.array_equal(unpermute(original[order], order), original)


def test_stability_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D31: the `stability` command, shrunk to seconds, end to end with its figures."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("stability")
    smoke = module.load_config("smoke")
    for name, net in config["networks"].items():
        net["grid"] = smoke["networks"]["rnn" if name == "rnn" else "cnn"]["grid"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "attribution", "device", "benchmarks")})
    config["stability"].update({"seeds": 2, "permutations": 1, "deletion_k": [1, 3], "random_draws": 1})
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.stability(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_stability_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    assert {"cnn/eg", "cnn/gradcam", "rnn/eg", "k1x5/gradcam"} <= set(summary["seed"])
    assert "rnn/eg ~ lightgbm_cells/shap" in summary["cross_model"]
    assert len(summary["faithfulness"]["cnn"]["top_auc"]) == 2
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("stability_diabe_*"))) == 4 * len(module.load_style(module.NOTEBOOK)["formats"])


def test_household_subsample_is_nested_and_whole_households() -> None:
    from utils.data_analysis_04_hrs_chronic_cnn_learning_curve import household_subsample

    household = np.repeat(np.arange(50), 2)  # two people per household
    rows = np.arange(100)
    rng = np.random.default_rng(0)
    small, order = household_subsample(rows, household, 0.2, rng)
    large, _ = household_subsample(rows, household, 0.6, rng, order)
    assert set(small) < set(large)  # the larger share holds the smaller one
    assert len(small) == 20 and len(large) == 60
    assert all((household[small] == h).sum() == 2 for h in np.unique(household[small]))  # never half a household


def test_learning_curve_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D32: the `learning-curve` command, shrunk to seconds, end to end with its figure."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("learning_curve")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks")})
    config["conditions"] = ["diabe"]
    config["learning_curve"].update({"shares": [0.5, 1.0], "repeats": 2,
                                     "benchmarks": ["logistic_static", "lightgbm_summary"]})
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.learning_curve(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_learning_curve_*.json"))
    body = json.loads(report.read_text())
    section = body["conditions"]["diabe"]
    summary = section["summary"]
    assert set(summary["curve"]) == {"logistic_static", "lightgbm_summary", "cnn", "rnn"}
    assert summary["curve"]["cnn"]["share"] == [0.5, 1.0]
    # 2 folds x (2 repeats at 50% + 1 at 100%)
    assert len(section["units"]) == 6 and summary["curve"]["rnn"]["units"] == [4, 2]
    assert summary["curve"]["cnn"]["n_train"][0] < summary["curve"]["cnn"]["n_train"][1]
    assert {"logistic_static", "lightgbm_summary", "best"} == set(summary["gaps"]["rnn"])
    assert "per_doubling" in summary["gaps"]["cnn"]["best"]
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("learning_curve_diabe_*"))) == len(module.load_style(module.NOTEBOOK)["formats"])


def test_concordance_averages_to_the_auc() -> None:
    from sklearn.metrics import roc_auc_score
    from utils.data_analysis_04_hrs_chronic_cnn_subgroups import concordance

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200) + 0.3 * y, 1)  # rounded, so there are ties
    c = concordance(y, p)
    assert np.isclose(c[y == 1].mean(), roc_auc_score(y, p)) and np.isclose(c[y == 0].mean(), roc_auc_score(y, p))
    assert ((c >= 0) & (c <= 1)).all()


def test_subgroups_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D34: the `subgroups` command, shrunk to seconds, end to end with its figures."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("subgroups")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks", "attribution")})
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.subgroups(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_subgroups_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    groups = summary["predictability"]["groups"]
    assert {f"cases_q{k}" for k in range(1, 6)} | {"cases", "controls"} <= set(groups)
    assert "sex/female" in summary["subgroups"] and "static_risk/q5" in summary["subgroups"]
    assert set(summary["disagreement"]["cnn"]["groups"]) == {
        "cases_network_better", "cases_reference_better", "controls_network_better", "controls_reference_better"}
    text = report.read_text()
    assert "hhidpn" not in text and '"p": {' not in text  # no per-person arrays
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("subgroups_diabe_*"))) == 3 * len(module.load_style(module.NOTEBOOK)["formats"])


def test_pathway_stability_finds_planted_pathways_and_not_noise() -> None:
    """D35: two planted pathways recur under seeds, bootstrap and folds at k = 2; pure noise does not."""
    from utils.data_analysis_04_hrs_chronic_cnn_pathways import kmeans, pathway_vectors, stability

    rng = np.random.default_rng(0)
    n, C, T = 600, 6, 4
    M = np.ones((n, C, T))
    path = rng.integers(0, 2, n)
    eg = rng.normal(0, 0.05, (n, C, T))
    eg[path == 0, 0] += 1.0   # pathway A: row 0 pushes up
    eg[path == 1, 3] += 1.0   # pathway B: row 3 pushes up
    V = pathway_vectors(eg, M)
    assert np.allclose(np.abs(V).sum(axis=1) <= 1 + 1e-9, True)
    replicate = pathway_vectors(eg + rng.normal(0, 0.05, eg.shape), M)
    fold = np.arange(n) % 5
    spec = {"n_init": 5, "bootstrap": 5, "stable": {"jaccard": 0.75, "cosine": 0.8}}
    labels, centroids = kmeans(V, 2, 0, 5)
    st = stability(V, labels, centroids, [replicate], fold, 2, spec, 0)
    assert all(st["stable"]) and st["seed_ari"] > 0.95
    noise = rng.normal(0, 1, (n, C))
    labels, centroids = kmeans(noise, 4, 0, 5)
    st = stability(noise, labels, centroids, [rng.normal(0, 1, (n, C))], fold, 4, spec, 0)
    assert not all(st["stable"])


def test_dropping_a_dominant_row_lets_smaller_pathways_separate() -> None:
    """D36: a dominant row splits cases on its sign; left out, two planted pathways in smaller rows come back."""
    from utils.data_analysis_04_hrs_chronic_cnn_pathways import kmeans, pathway_vectors, stability

    rng = np.random.default_rng(1)
    n, C, T = 600, 6, 4
    M = np.ones((n, C, T))
    path = rng.integers(0, 2, n)
    eg = rng.normal(0, 0.02, (n, C, T))
    eg[:, 0] += rng.choice([-2.0, 2.0], n)[:, None]   # dominant row, sign unrelated to the pathway
    eg[path == 0, 2] += 0.3
    eg[path == 1, 4] += 0.3
    kept = [1, 2, 3, 4, 5]
    V = pathway_vectors(eg, M, rows_kept=kept)
    assert V.shape == (n, 5) and np.all(np.abs(V).sum(axis=1) <= 1 + 1e-9)
    with_row = kmeans(pathway_vectors(eg, M), 2, 0, 5)[0]
    labels, centroids = kmeans(V, 2, 0, 5)
    agree = lambda a: max((a == path).mean(), (a != path).mean())
    assert agree(with_row) < 0.7 and agree(labels) > 0.95
    replicate = pathway_vectors(eg + rng.normal(0, 0.02, eg.shape), M, rows_kept=kept)
    spec = {"n_init": 5, "bootstrap": 5, "stable": {"jaccard": 0.75, "cosine": 0.8}}
    assert all(stability(V, labels, centroids, [replicate], np.arange(n) % 5, 2, spec, 0)["stable"])


@pytest.mark.parametrize("profile", ["pathways", "pathways_no_bmi"])
def test_pathways_runs_on_the_fixture(fixture_path, tmp_path, profile) -> None:
    """D35, D36: the `pathways` command, shrunk to seconds, end to end with its figures."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config(profile)
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks", "attribution")})
    config["pathways"] = {**config["pathways"], "k": [2, 3], "seeds": 1, "bootstrap": 2, "n_init": 2}
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.pathways(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_pathways_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    assert set(summary["by_k"]) == {"cnn", "rnn"} and set(summary["cross_model"]) == {"2", "3"}
    dropped = ["bmi"] if profile == "pathways_no_bmi" else []
    assert summary["dropped_rows"] == dropped and ("bmi" in summary["vector_names"]) == (not dropped)
    assert len(summary["vector_names"]) == len(body["meta"]["channel_names"]) - len(dropped)
    for net in ("cnn", "rnn"):
        assert len(summary["by_k"][net]["3"]["clusters"]) == 3
        assert summary["chosen_k"][net] in (None, 2, 3)
    text = report.read_text()
    assert "hhidpn" not in text and '"labels"' not in text and '"p": {' not in text  # no per-person arrays
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    stem = "pathways_no_bmi_diabe_*" if dropped else "pathways_diabe_*"
    assert len(list(style_dir.glob(stem))) == 5 * len(module.load_style(module.NOTEBOOK)["formats"])


def test_scaling_profile_runs_every_arm_on_the_fixture(fixture_path, tmp_path) -> None:
    """D33: the `scaling` profile, shrunk to seconds; per-channel arms report their parameters."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("scaling")
    smoke = module.load_config("smoke")
    assert config["run_benchmarks"] is False and len(config["networks"]) == 6
    for name, arm in config["networks"].items():
        kind = "rnn" if name.startswith("rnn") else "cnn"
        arm["architecture"] = {**smoke["networks"][kind]["architecture"], "type": kind}
        arm["grid"] = smoke["networks"][kind]["grid"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "attribution", "device")})
    config["conditions"] = ["diabe"]
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.run(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_run_*.json"))
    section = json.loads(report.read_text())["conditions"]["diabe"]
    assert set(section["metrics_mean"]) == set(config["networks"])
    fold = section["folds"][0]
    assert "image_scaling" not in fold["cnn"]
    scaling = fold["cnn_minmax"]["image_scaling"]
    assert scaling["method"] == "channel_minmax" and len(scaling["lower"]) == len(EXTENDED)
    assert "minimum" not in scaling and "maximum" not in scaling  # one person's values stay local (D22)
    figures = type("Args", (), {"report": str(report), "reference": None})()
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(figures)
    assert len(list(style_dir.glob("scaling_diabe_*"))) == 2 * len(module.load_style(module.NOTEBOOK)["formats"])


# --- D38-D40: time order, inputs, stacking ------------------------------------------------------
@pytest.fixture(scope="module")
def extended_diabetes(fixture_path):
    frame = read_rand(fixture_path, columns=chronic_columns(["diabe"], "extended"))
    return build_chronic_dataset(frame, "diabe", 13, 10, "extended", "extended")


def test_time_shuffles_keep_each_persons_values(extended_diabetes) -> None:
    """D38: ``fixed`` moves every person's bins the same way; ``person`` shuffles only observed bins."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import time_shuffled
    from utils.shared_hrs_ingest import _delta

    data = extended_diabetes
    seen = data.M.any(axis=1)
    fixed = time_shuffled(data, "fixed", np.random.default_rng(0))
    order = [next(j for j in range(data.X.shape[2]) if np.array_equal(fixed.M[:, :, k], data.M[:, :, j])
                  and np.allclose(fixed.X[:, :, k], data.X[:, :, j], equal_nan=True)) for k in range(data.X.shape[2])]
    assert sorted(order) == list(range(data.X.shape[2])) and order != sorted(order)
    person = time_shuffled(data, "person", np.random.default_rng(0))
    assert np.array_equal(person.M.any(axis=1), seen)  # whole-bin gaps (before entry, after exit) stay put
    for shuffled in (fixed, person):
        assert np.array_equal(shuffled.Delta, _delta(shuffled.M))
        assert np.array_equal(shuffled.M.sum(axis=2), data.M.sum(axis=2))  # each row keeps its observations
        assert np.allclose(np.nansum(shuffled.X, axis=2), np.nansum(data.X, axis=2))
    moved = [(person.M[i] != data.M[i]).any() for i in range(data.n) if seen[i].sum() > 2]
    assert np.mean(moved) > 0.5
    with pytest.raises(ValueError):
        time_shuffled(data, "reverse", np.random.default_rng(0))


def test_row_variants_rewrite_only_the_bmi_row(extended_diabetes) -> None:
    """D41: each arm touches the BMI row alone, as defined; people never observed are unchanged."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import row_variant
    from utils.shared_hrs_ingest import _delta

    data = extended_diabetes
    c = data.channel_names.index("bmi")
    others = np.arange(len(data.channel_names)) != c
    x, m = data.X[:, c, :], data.M[:, c, :]
    seen = m.any(axis=1)
    assert seen.mean() > 0.5 and (m.sum(axis=1) > 1).mean() > 0.3
    arms = {how: row_variant(data, "bmi", how, np.random.default_rng(0)) for how in ("none", "single", "level", "shuffled")}
    for how, varied in arms.items():
        assert np.array_equal(varied.M[:, others], data.M[:, others]), how
        assert np.allclose(varied.X[:, others], data.X[:, others], equal_nan=True), how
        assert np.array_equal(varied.Delta, _delta(varied.M)), how
        assert np.isnan(varied.X[:, c][~varied.M[:, c]]).all(), how
    assert not arms["none"].M[:, c].any()
    single = arms["single"]
    assert np.array_equal(single.M[:, c].sum(axis=1), seen.astype(int))
    last = data.X.shape[2] - 1 - np.argmax(m[:, ::-1], axis=1)
    rows = np.flatnonzero(seen)
    assert np.allclose(single.X[rows, c, last[rows]], x[rows, last[rows]])
    for how in ("level", "shuffled"):
        assert np.array_equal(arms[how].M[:, c], m), how  # observation pattern kept
    level = arms["level"].X[:, c]
    for i in rows[:200]:
        assert np.allclose(level[i, m[i]], x[i, m[i]].mean())
        assert np.allclose(np.sort(arms["shuffled"].X[i, c, m[i]]), np.sort(x[i, m[i]]))
    moved = [not np.allclose(arms["shuffled"].X[i, c, m[i]], x[i, m[i]]) for i in rows if np.unique(x[i, m[i]]).size > 2]
    assert np.mean(moved) > 0.5
    with pytest.raises(ValueError):
        row_variant(data, "bmi", "reverse", np.random.default_rng(0))
    with pytest.raises(ValueError):
        row_variant(data, "not_a_row", "none", np.random.default_rng(0))


def test_image_only_drops_the_statics_from_every_input(extended_diabetes) -> None:
    """D39: no static column reaches the networks' heads or the benchmarks' features."""
    from utils.data_analysis_04_hrs_chronic_cnn import build_network, fit_scaler, tensors
    from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import cell_features, summary_features
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import image_only

    data, bare = extended_diabetes, image_only(extended_diabetes)
    rows = np.arange(50)
    P_ = data.static.shape[1]
    assert bare.static.shape == (data.n, 0) and bare.static_names == ()
    assert summary_features(bare, rows).shape[1] == summary_features(data, rows).shape[1] - P_
    assert cell_features(bare, rows).shape[1] == cell_features(data, rows).shape[1] - P_
    image, static, _ = tensors(bare, rows, fit_scaler(bare, rows))
    assert static.shape == (50, 0)
    for network, arch, hyper in (("cnn", {"kernels": [[4, 5]], "dilations": [1, 2, 4], "hidden": 8, "causal": False},
                                  {"width": 4, "dropout": 0.0}),
                                 ("rnn", {"layers": 1, "bidirectional": True}, {"hidden": 4, "dropout": 0.0})):
        model = build_network(network, arch, image.shape[2], image.shape[3], hyper, n_static=0)
        assert model(image, static).shape == (50,)


def test_stack_weighs_the_informative_model_and_gaps_are_paired() -> None:
    """D40: inner predictions alone set the weights; a gap's interval is over folds."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import paired_gap, stack

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    signal = lambda: 1 / (1 + np.exp(-(2 * y - 1) * 1.5 + rng.normal(0, 1, len(y))))  # noqa: E731
    p_inner = {"good": signal(), "noise": rng.uniform(0.05, 0.95, len(y))}
    p_test = {"good": signal(), "noise": rng.uniform(0.05, 0.95, len(y))}
    p, weights = stack(p_inner, y, p_test, ["good", "noise"], C=1.0)
    assert weights["good"] > 0.5 and abs(weights["noise"]) < 0.2 and p.shape == y.shape
    gap = paired_gap([0.71, 0.72, 0.70, 0.73, 0.72], [0.70, 0.71, 0.70, 0.71, 0.71])
    assert gap["mean"] == pytest.approx(0.01) and gap["lo"] < gap["mean"] < gap["hi"] and 0 < gap["p"] < 1


def test_comparisons_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D38-D40: the `comparisons` command, shrunk to seconds, end to end with its figure."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("full")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks")})
    config["conditions"] = ["diabe"]
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.comparisons(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_comparisons_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    for key in ("original/cnn", "original/logistic_static", "time_fixed/rnn", "time_person/lightgbm_cells",
                "image_only/cnn", "image_only/lightgbm_summary", "statics_only/lightgbm_static",
                "stack/all", "stack/benchmarks", "stack/networks", "average/all"):
        assert key in summary["models"], key
    assert "time_fixed/lightgbm_cells" not in summary["models"]  # a fixed order changes nothing for a tree on cells
    assert "stack/all - stack/benchmarks" in summary["gaps"]["ensemble"]
    assert "image_only/rnn - original/rnn" in summary["gaps"]["inputs"]
    assert set(summary["stack_weights"]["all"]) == {"logistic_summary", "lightgbm_summary", "lightgbm_cells", "cnn", "rnn"}
    assert len(body["conditions"]["diabe"]["folds"]) == config["outer_folds"]
    text = report.read_text()
    assert "hhidpn" not in text and '"p": [' not in text  # no per-person arrays
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("comparisons_diabe_*"))) == len(module.load_style(module.NOTEBOOK)["formats"])


def test_bmi_trajectory_profile_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D41: the `bmi_trajectory` profile, shrunk to seconds: only its part, gaps to the original and to no BMI."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("bmi_trajectory")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks")})
    config["conditions"] = ["diabe"]
    assert config["comparisons"]["parts"] == ["bmi_trajectory"] and "time_order" in config["comparisons"]
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.comparisons(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_comparisons_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    for how in ("none", "single", "level", "shuffled"):
        for m in ("cnn", "rnn", "lightgbm_summary", "lightgbm_cells"):
            assert f"bmi_{how}/{m}" in summary["models"], (how, m)
    assert not any(k.startswith(("time_", "image_only", "stack/")) for k in summary["models"])
    gaps = summary["gaps"]["bmi_trajectory"]
    assert "bmi_level/rnn - original/rnn" in gaps and "bmi_single/lightgbm_cells - bmi_none/lightgbm_cells" in gaps
    assert "bmi_none/cnn - bmi_none/cnn" not in gaps
    assert not summary["gaps"]["time_order"] and not summary["gaps"]["ensemble"]
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("comparisons_diabe_*"))) == len(module.load_style(module.NOTEBOOK)["formats"])


def test_rows_variant_shuffles_the_socioeconomic_rows_together(extended_diabetes) -> None:
    """D42: the group moves with one permutation per person; other rows and the group's values are kept."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import rows_variant
    from utils.shared_hrs_ingest import _delta

    data = extended_diabetes
    rows = ["income", "wealth", "working", "medicaid"]
    c = [data.channel_names.index(r) for r in rows]
    others = np.setdiff1d(np.arange(len(data.channel_names)), c)
    arms = {how: rows_variant(data, rows, how, np.random.default_rng(0)) for how in ("none", "level", "shuffled")}
    for how, varied in arms.items():
        assert np.array_equal(varied.M[:, others], data.M[:, others]), how
        assert np.allclose(varied.X[:, others], data.X[:, others], equal_nan=True), how
        assert np.array_equal(varied.Delta, _delta(varied.M)), how
    assert not arms["none"].M[:, c].any()
    shuffled = arms["shuffled"]
    seen = data.M[:, c, :].any(axis=1)
    many = np.flatnonzero(seen.sum(axis=1) > 2)
    assert many.size > 20
    moved = 0
    for i in many[:200]:
        bins = np.flatnonzero(seen[i])
        assert np.array_equal(shuffled.M[i][:, ~seen[i]][c], data.M[i][:, ~seen[i]][c])  # unseen bins stay empty
        before = {tuple(np.nan_to_num(data.X[i, c, b], nan=-9)) for b in bins}
        after = {tuple(np.nan_to_num(shuffled.X[i, c, b], nan=-9)) for b in bins}
        assert before == after  # whole columns of the group move together
        moved += not np.allclose(shuffled.X[i][np.ix_(c, bins)], data.X[i][np.ix_(c, bins)], equal_nan=True)
    assert moved > 0.5 * min(len(many), 200)
    with pytest.raises(ValueError):
        rows_variant(data, rows, "single", np.random.default_rng(0))
    with pytest.raises(ValueError):
        rows_variant(data, ["income", "not_a_row"], "none", np.random.default_rng(0))


def test_strata_split_people_by_change_and_events() -> None:
    """D42: under-two-wave people apart, thirds of change among the rest, events from a later fall or rise."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import _swing, strata

    x = np.array([[5.0, 4.5, 3.9, np.nan], [1.0, np.nan, 2.0, 2.5], [3.0, 3.0, np.nan, np.nan], [2.0, np.nan, np.nan, np.nan]])
    m = ~np.isnan(x)
    assert _swing(x, m, "fall", 1.0).tolist() == [True, False, False, False]
    assert _swing(x, m, "rise", 1.0).tolist() == [False, True, False, False]
    assert _swing(x, m, "fall", 1.2).tolist() == [False, False, False, False]

    rng = np.random.default_rng(0)
    n, K = 300, 6
    X = rng.normal(size=(n, 2, K))
    X[:100] = X[:100, :, :1]  # flat
    X[200:] *= 4  # volatile
    M = np.ones_like(X, dtype=bool)
    M[-10:, :, 1:] = False  # one wave only
    X[~M] = np.nan
    data = type("D", (), {"X": X, "M": M, "channel_names": ("wealth", "working")})()
    spec = {"rows": ["wealth", "working"], "quantiles": 3, "events": {"wealth_fall": {"row": "wealth", "fall": 1.0}}}
    groups = strata(data, spec)
    assert groups["under_2_waves"].sum() == 10
    thirds = [groups[f"change_q{k}"] for k in (1, 2, 3)]
    assert sum(g.sum() for g in thirds) == n - 10 and not (thirds[0] & thirds[2]).any()
    assert thirds[0][:100].all() and thirds[2][200:-10].mean() > 0.9
    assert not groups["wealth_fall"][:100].any()
    assert np.array_equal(groups["wealth_fall"] | groups["no_wealth_fall"], ~groups["under_2_waves"])


def test_ses_order_profile_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D42: the `ses_order` profile, shrunk to seconds: its arms, gaps, and strata with suppression."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("ses_order")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks")})
    config["conditions"] = ["diabe"]
    assert config["comparisons"]["parts"] == ["ses_order"] and "strata" in config["comparisons"]
    assert "left_work" not in config["comparisons"]["population"]
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.comparisons(args, config)
    report = next(tmp_path.glob("04_hrs_chronic_cnn_comparisons_*.json"))
    body = json.loads(report.read_text())
    summary = body["conditions"]["diabe"]["summary"]
    kept = body["conditions"]["diabe"]["population"]
    assert kept["strata"] == config["comparisons"]["population"]
    assert 0 < kept["n"] < body["conditions"]["diabe"]["dataset"]["n_kept"]  # D43: changers only
    assert summary["strata"]["counts"]["under_2_waves"]["n"] == 0
    for how in ("none", "level", "shuffled"):
        for m in ("cnn", "rnn", "lightgbm_summary", "lightgbm_cells"):
            assert f"ses_{how}/{m}" in summary["models"], (how, m)
    assert "ses_shuffled/rnn - ses_none/rnn" in summary["gaps"]["ses_order"]
    found = summary["strata"]
    assert {"under_2_waves", "change_q1", "change_q3", "left_work", "no_left_work"} <= set(found["counts"])
    for name, count in found["counts"].items():
        assert isinstance(count["n"], int) or count["n"] == "<11", name
        if count["suppressed"]:
            assert all(name not in by for by in found["gaps"].values()), name
    assert set(found["gaps"]) == {k for k in summary["gaps"]["ses_order"] if " - original/" in k}
    for by in found["gaps"].values():
        for gap in by.values():
            assert set(gap) == {"auc", "log_loss"}
    text = report.read_text()
    assert "hhidpn" not in text and '"p": [' not in text
    style_dir = tmp_path / "figs"
    module.savedir = lambda style: style_dir
    module.figures(type("Args", (), {"report": str(report), "reference": None})())
    assert len(list(style_dir.glob("comparisons_diabe_*"))) == len(module.load_style(module.NOTEBOOK)["formats"])


def test_population_keeps_only_the_named_strata(extended_diabetes) -> None:
    """D43: the sample is cut to the union of the named strata, every per-person array with it."""
    from utils.data_analysis_04_hrs_chronic_cnn_comparisons import population, restrict, strata

    data = extended_diabetes
    spec = {"population": ["wealth_fall", "poverty_entry"], "strata": {
        "rows": ["income", "wealth", "poverty"], "quantiles": 3,
        "events": {"wealth_fall": {"row": "wealth", "fall": 1.0}, "poverty_entry": {"row": "poverty", "rise": 1}}}}
    keep = population(data, spec)
    groups = strata(data, spec["strata"])
    assert np.array_equal(keep, groups["wealth_fall"] | groups["poverty_entry"]) and 0 < keep.sum() < data.n
    cut = restrict(data, keep)
    assert cut.n == keep.sum() and cut.X.shape[0] == cut.static.shape[0] == len(cut.household) == keep.sum()
    assert np.array_equal(cut.y, data.y[keep])
    assert population(data, {**spec, "population": None}).all()
    with pytest.raises(ValueError):
        population(data, {**spec, "population": ["wealth_rise"]})


def test_bmi_changers_profile_runs_on_the_fixture(fixture_path, tmp_path) -> None:
    """D44: the D41 arms on BMI changers only, with strata by BMI change."""
    spec = importlib.util.spec_from_file_location("a04", P.SRC_DIR / "data_analysis" / "04_hrs_chronic_cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config("bmi_changers")
    smoke = module.load_config("smoke")
    config["networks"] = smoke["networks"]
    config.update({k: smoke[k] for k in ("outer_folds", "inner_folds", "training", "device", "benchmarks")})
    config["conditions"] = ["diabe"]
    comparisons = config["comparisons"]
    assert comparisons["parts"] == ["bmi_trajectory"] and comparisons["strata"]["rows"] == ["bmi"]
    assert "ses_order" in comparisons and "inputs" in comparisons  # the rest of the section survives the merge
    args = type("Args", (), {"rand_file": fixture_path, "report_dir": tmp_path, "debug": True,
                             "conditions": None, "status_bin": None})()
    module.comparisons(args, config)
    body = json.loads(next(tmp_path.glob("04_hrs_chronic_cnn_comparisons_*.json")).read_text())
    section = body["conditions"]["diabe"]
    assert section["population"]["strata"] == ["bmi_gain", "bmi_loss"]
    assert 0 < section["population"]["n"] < section["dataset"]["n_kept"]
    assert {"bmi_gain", "no_bmi_gain", "bmi_loss", "change_q1"} <= set(section["summary"]["strata"]["counts"])
    assert set(section["summary"]["strata"]["gaps"]) == {k for k in section["summary"]["gaps"]["bmi_trajectory"]
                                                        if " - original/" in k}
