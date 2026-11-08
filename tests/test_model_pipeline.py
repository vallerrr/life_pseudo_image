"""The model pipeline's invariants: outcome encoding, causality, scoring, attribution."""

import numpy as np
import pytest
import torch

from utils.shared_dgp import generate_dataset, load_config
from utils.shared_evaluation import (
    aalen_johansen_disease,
    disease_auc,
    population_map,
    still_observed,
)
from utils.shared_model_input import (
    CENSORED,
    DEATH,
    DISEASE,
    build_model_input,
    fit_standardiser,
    mask_streams,
    observed_outcome,
)
from utils.data_analysis_02_pseudo_image_cnn_model import (
    PseudoImageCNN,
    build_model,
    expected_gradients,
    image_tensor,
)


@pytest.fixture(scope="module")
def dataset():
    cfg = load_config()
    return generate_dataset("critical_period", n=2000, replicate=0, config=cfg), cfg


def test_observed_outcome_censors_at_attrition(dataset) -> None:
    result, cfg = dataset
    obs_time, obs_type = observed_outcome(result, cfg.T)
    dropped = result.dropout_time <= cfg.T
    # A dropout is known event-free through the bin before it, and nothing after.
    assert np.all(obs_time[dropped] == result.dropout_time[dropped] - 1)
    assert np.all(obs_type[dropped] == CENSORED)
    # Without dropout the DGP's own event is seen in its own bin.
    stayed = ~dropped & (result.event_type > 0)
    assert np.all(obs_time[stayed] == result.event_time[stayed])
    assert np.all(obs_type[stayed] == result.event_type[stayed])
    assert np.all(obs_time[~dropped & (result.event_type == 0)] == cfg.T)
    # Nothing is recorded where the panel no longer observed the person.
    assert not np.any(result.M[dropped, :, :][:, :, -1])


def test_cnn_is_causal_in_time() -> None:
    torch.manual_seed(0)
    net = PseudoImageCNN(12, 20, 2, width=8, kernels=((4, 3), (4, 5)), dilations=(1, 2, 4), hidden=8, dropout=0.0).eval()
    image, static = torch.randn(4, 3, 12, 20), torch.randn(4, 2)
    changed = image.clone()
    changed[:, :, :, 9:] = torch.randn_like(changed[:, :, :, 9:])
    with torch.no_grad():
        before, after = net(image, static), net(changed, static)
    # Bins 1..9 (columns 0..8) cannot see anything from bin 10 on.
    assert torch.allclose(before[:, :, :9], after[:, :, :9])
    assert not torch.allclose(before[:, :, 9:], after[:, :, 9:])


def test_configured_cnn_reads_the_whole_panel() -> None:
    import yaml

    from utils import shared_paths as P

    architecture = yaml.safe_load(P.PSEUDO_IMAGE_CNN_CONFIG.read_text())["architecture"]
    torch.manual_seed(0)
    net = build_model(architecture, 12, 20, 2, {"width": 8, "dropout": 0.0}).eval()
    assert net.receptive_field >= 20
    image, static = torch.randn(4, 3, 12, 20), torch.randn(4, 2)
    changed = image.clone()
    changed[:, :, :, 0] += 1.0  # bin 1 only
    with torch.no_grad():
        assert not torch.allclose(net(image, static)[:, :, -1], net(changed, static)[:, :, -1])
    # The D09 network, two undilated 5-wide layers, could not: build_model refuses it.
    with pytest.raises(ValueError, match="receptive field 9"):
        build_model({**architecture, "kernels": [[4, 5]], "dilations": [1, 1]}, 12, 20, 2,
                    {"width": 8, "dropout": 0.0})


def test_non_causal_cnn_does_see_the_future() -> None:
    torch.manual_seed(0)
    net = PseudoImageCNN(12, 20, 2, 8, ((4, 5),), (1, 1), 8, 0.0, causal=False).eval()
    image, static = torch.randn(2, 3, 12, 20), torch.randn(2, 2)
    changed = image.clone()
    changed[:, :, :, 9:] += 1.0
    with torch.no_grad():
        assert not torch.allclose(net(image, static)[:, :, :9], net(changed, static)[:, :, :9])


def test_unobserved_cells_get_no_attribution() -> None:
    torch.manual_seed(0)
    net = PseudoImageCNN(12, 20, 2, 8, ((4, 5),), (1, 2), 8, 0.0).eval()
    image = torch.randn(3, 3, 12, 20)
    image[:, 1] = (torch.rand(3, 12, 20) > 0.4).float()  # mask plane
    image[:, 0] *= image[:, 1]  # unobserved values sit at 0, as the standardiser leaves them
    attribution = expected_gradients(net, image, torch.randn(3, 2), torch.randn(10, 3, 12, 20), 16, seed=0)
    assert np.all(attribution[image[:, 1].numpy() == 0] == 0)
    assert np.any(attribution != 0)


def test_scoring_against_known_answers() -> None:
    T = 5
    obs_time = np.array([2, 3, 5, 5, 4, 1])
    obs_type = np.array([DISEASE, DISEASE, CENSORED, CENSORED, DEATH, CENSORED])
    K = still_observed(obs_time, obs_type, T)
    assert K[0] == 1.0 and K[1] == 1.0 and K[2] == pytest.approx(5 / 6)
    # Risk that orders every case above every control scores exactly 1.
    risk = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.5])
    assert disease_auc(risk, obs_time, obs_type, K, T) == pytest.approx(1.0)
    assert disease_auc(-risk, obs_time, obs_type, K, T) == pytest.approx(0.0)
    # With no censoring, Aalen-Johansen is the plain proportion.
    no_censoring = np.array([DISEASE, DEATH, DISEASE, CENSORED])
    assert aalen_johansen_disease(np.array([1, 2, 3, 5]), no_censoring, 5) == pytest.approx(0.5)


def test_population_map_ignores_who_was_not_asked() -> None:
    per_person = np.zeros((4, 1, 2))
    per_person[:2, 0, 0] = 2.0  # only two people observed cell (0, 0)
    observed = np.zeros((4, 1, 2), dtype=bool)
    observed[:2, 0, 0] = True
    observed[:, 0, 1] = True
    mean_abs, _ = population_map(per_person, observed)
    assert mean_abs[0, 0] == pytest.approx(2.0)  # not 1.0, which a plain mean would give
    assert mean_abs[0, 1] == 0.0


def test_model_input_keeps_truth_apart(dataset) -> None:
    result, cfg = dataset
    data = build_model_input(result, cfg)
    assert all(key.startswith("truth_") for key in data.truth)
    assert data.X.shape == (2000, cfg.C, cfg.T) and data.S.shape == (2000, 3)
    assert np.array_equal(np.isnan(data.X), ~data.M)


def test_masked_stream_carries_nothing_about_the_person(dataset) -> None:
    result, cfg = dataset
    data = build_model_input(result, cfg)
    masked = mask_streams(data, ["midstream"])
    mid = slice(cfg.channels_per_stream, 2 * cfg.channels_per_stream)
    image = image_tensor(masked, fit_standardiser(masked.X)).numpy()
    # Every person's midstream rows are identical in all three planes.
    assert np.all(image[:, :, mid] == image[:1, :, mid])
    # The other streams and the truth are untouched.
    keep = np.ones(cfg.C, dtype=bool)
    keep[mid] = False
    assert np.array_equal(masked.M[:, keep], data.M[:, keep])
    assert masked.truth["truth_w"] is data.truth["truth_w"]
    assert mask_streams(data, []) is data


def test_oracle_reads_only_the_true_cells(dataset) -> None:
    from utils.data_analysis_03_oracle_checks import OracleHazard

    result, cfg = dataset
    data = build_model_input(result, cfg)
    image = image_tensor(data, fit_standardiser(data.X))
    static = torch.from_numpy(data.static().astype(np.float32))
    oracle = OracleHazard("critical_period", cfg.T, 2, death_history=False)
    oracle.calibrate(image)
    with torch.no_grad():
        oracle.beta.fill_(0.4)
    attribution = expected_gradients(oracle, image[:200], static[:200], image[200:250], 16, seed=0)
    support = np.zeros((cfg.C, cfg.T), dtype=bool)
    support[:4, 4:8] = True  # upstream channels, bins 5-8
    assert np.all(attribution[:, ~support] == 0)
    assert np.any(attribution[:, support] != 0)
    # Causal: the hazard at bin 6 does not move when bin 7 on changes.
    changed = image.clone()
    changed[:, 0, :, 6:] += 1.0
    with torch.no_grad():
        assert torch.allclose(oracle(image, static)[:, :, :6], oracle(changed, static)[:, :, :6])


def test_late_entry_shrinks_risk_sets_and_simulation_is_unchanged() -> None:
    from utils.shared_evaluation import aalen_johansen_disease, disease_concordance, interval_nll
    from utils.data_analysis_02_pseudo_image_cnn_model import interval_targets

    obs_time = np.array([3, 5, 5, 6])
    obs_type = np.array([DISEASE, CENSORED, DEATH, CENSORED])
    ones = np.ones(4, dtype=np.int16)
    # Default entry is bin 1: identical to passing ones, so simulation scores cannot move.
    assert np.array_equal(still_observed(obs_time, obs_type, 6), still_observed(obs_time, obs_type, 6, ones))
    entry = np.array([1, 4, 1, 2])
    _, at_risk = interval_targets(obs_time, obs_type, 6, entry)
    assert at_risk[1].tolist() == [False, False, False, True, True, False]
    # Person 1 had not entered by bin 3, so is not a comparison for person 0's disease.
    risk = np.array([0.5, 0.9, 0.1, 0.2])
    assert disease_concordance(risk, obs_time, obs_type) == pytest.approx(2 / 3)
    assert disease_concordance(risk, obs_time, obs_type, entry) == pytest.approx(1.0)
    # The risk set at bin 3 is three people, not four.
    assert aalen_johansen_disease(obs_time, obs_type, 3, entry) == pytest.approx(1 / 3)
    hazard = np.full((4, 6), 0.1)
    assert interval_nll(hazard, hazard, obs_time, obs_type, entry) != interval_nll(hazard, hazard, obs_time, obs_type)


def test_model_input_carries_entry_and_ids(dataset, tmp_path) -> None:
    from utils.shared_model_input import read_model_input, write_model_input

    result, cfg = dataset
    data = build_model_input(result, cfg)
    assert (data.entry_time == 1).all() and data.person_id is None
    back = read_model_input(write_model_input(data, tmp_path / "m.npz"))
    assert np.array_equal(back.entry_time, data.entry_time) and back.person_id is None
    assert "entry_time" not in back.truth
    part = data.subset(np.arange(10))
    assert part.entry_time.shape == (10,)


def test_per_channel_scaling_keeps_the_age_profile_and_lands_in_0_1() -> None:
    """D33: pooled over bins and people, a rising channel still rises; min-max stays in [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2, 6)) + np.arange(6)[None, None, :]  # every channel rises one unit per bin
    X[rng.random(X.shape) < 0.3] = np.nan
    X[0, 1, 0] = 1e6  # one outlier, clipped at the upper quantile
    cell = fit_standardiser(X).transform(X)
    assert abs(np.nanmean(np.where(np.isnan(X), np.nan, cell)[:, 0, -1])) < 0.1  # per-cell: the profile is gone
    pooled = fit_standardiser(X, "channel_z")
    assert pooled.mean.shape == (2, 1) and pooled.transform(X)[:, 0, -1].mean() > pooled.transform(X)[:, 0, 0].mean()
    minmax = fit_standardiser(X, "channel_minmax")
    scaled = minmax.transform(X)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0 and minmax.maximum[1, 0] == pytest.approx(1e6)
    fill = np.broadcast_to(minmax.fill, X.shape)
    assert np.allclose(scaled[np.isnan(X)], fill[np.isnan(X)])  # unobserved -> the training mean
    assert 0.0 < minmax.fill[0, 0] < 1.0
    stats = minmax.stats()
    assert set(stats) == {"mean", "sd", "lower", "upper", "clipped_mean", "minimum", "maximum"} and len(stats["mean"]) == 2


def test_minmax_keeps_a_rare_binary_channel_binary() -> None:
    X = np.zeros((1000, 1, 3))
    X[:5, 0, 1] = 1.0  # 0.17% ones: both quantiles sit at 0
    scaled = fit_standardiser(X, "channel_minmax").transform(X)
    assert set(np.unique(scaled)) == {0.0, 1.0}
