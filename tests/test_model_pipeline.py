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
from utils.shared_model_input import CENSORED, DEATH, DISEASE, build_model_input, observed_outcome
from utils.data_analysis_02_pseudo_image_cnn_model import PseudoImageCNN, expected_gradients


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
    net = PseudoImageCNN(12, 20, 2, width=8, depth=3, kernel=(4, 5), hidden=8, dropout=0.0).eval()
    image, static = torch.randn(4, 3, 12, 20), torch.randn(4, 2)
    changed = image.clone()
    changed[:, :, :, 9:] = torch.randn_like(changed[:, :, :, 9:])
    with torch.no_grad():
        before, after = net(image, static), net(changed, static)
    # Bins 1..9 (columns 0..8) cannot see anything from bin 10 on.
    assert torch.allclose(before[:, :, :9], after[:, :, :9])
    assert not torch.allclose(before[:, :, 9:], after[:, :, 9:])


def test_non_causal_cnn_does_see_the_future() -> None:
    torch.manual_seed(0)
    net = PseudoImageCNN(12, 20, 2, 8, 2, (4, 5), 8, 0.0, causal=False).eval()
    image, static = torch.randn(2, 3, 12, 20), torch.randn(2, 2)
    changed = image.clone()
    changed[:, :, :, 9:] += 1.0
    with torch.no_grad():
        assert not torch.allclose(net(image, static)[:, :, :9], net(changed, static)[:, :, :9])


def test_unobserved_cells_get_no_attribution() -> None:
    torch.manual_seed(0)
    net = PseudoImageCNN(12, 20, 2, 8, 2, (4, 5), 8, 0.0).eval()
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
