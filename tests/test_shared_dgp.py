"""Unit tests for utils.shared_dgp, the first-deliverable DGP."""

from dataclasses import replace

import numpy as np

from utils.shared_dgp import generate_dataset, load_config


def test_shapes_reproducibility_and_missingness_order() -> None:
    first = generate_dataset("critical_period", n=600, replicate=3)
    second = generate_dataset("critical_period", n=600, replicate=3)

    assert first.X.shape == (600, 12, 20)
    assert first.M.shape == first.X.shape
    assert first.Delta.shape == first.X.shape
    assert first.S.shape == (600, 3)
    assert first.w_true.shape == (12, 20)
    np.testing.assert_array_equal(first.M, second.M)
    np.testing.assert_allclose(first.X, second.X, equal_nan=True)
    np.testing.assert_array_equal(first.event_type, second.event_type)

    assert np.isnan(first.X[~first.M]).all()
    assert np.isfinite(first.X[first.M]).all()
    assert not first.M[:, 4:8, 0::2].any()  # human bins 1, 3, ..., 19
    # Ten-percent item nonresponse plus calibrated monotone attrition.
    assert 0.65 < first.M[:, :4, :].mean() < 0.82


def test_joint_incidence_and_mortality_hit_targets() -> None:
    result = generate_dataset("chain_of_risk", n=20_000, replicate=0)
    assert abs(result.metadata["expected_disease_incidence"] - 0.15) < 1e-6
    assert abs(result.metadata["expected_mortality"] - 0.20) < 1e-6
    assert abs(result.metadata["realized_disease_incidence"] - 0.15) < 0.015
    assert abs(result.metadata["realized_mortality"] - 0.20) < 0.015
    assert abs(result.metadata["realized_attrition"] - 0.30) < 0.02


def test_null_true_hazard_attribution_is_exactly_flat() -> None:
    result = generate_dataset("null", n=500, replicate=0)
    np.testing.assert_array_equal(result.w_true, np.zeros((12, 20)))


def test_latent_mediation_coefficients_recoverable_by_ols() -> None:
    cfg = replace(load_config(), target_disease=0.10, target_death=0.10)
    result = generate_dataset("accumulation", n=12_000, replicate=5, config=cfg)
    latent = result.latent
    intercepts = result.random_intercepts

    predictors_m = np.column_stack(
        (
            latent[:, 0, :-1].ravel(),
            latent[:, 1, :-1].ravel(),
            np.repeat(intercepts[:, 1], cfg.T - 1),
            np.ones(result.latent.shape[0] * (cfg.T - 1)),
        )
    )
    coefficient_m = np.linalg.lstsq(
        predictors_m, latent[:, 1, 1:].ravel(), rcond=None
    )[0]
    np.testing.assert_allclose(coefficient_m[:3], [cfg.a, cfg.rho_m, 1.0], atol=0.02)

    predictors_d = np.column_stack(
        (
            latent[:, 1, :-1].ravel(),
            latent[:, 0, :-1].ravel(),
            latent[:, 2, :-1].ravel(),
            np.repeat(intercepts[:, 2], cfg.T - 1),
            np.ones(result.latent.shape[0] * (cfg.T - 1)),
        )
    )
    coefficient_d = np.linalg.lstsq(
        predictors_d, latent[:, 2, 1:].ravel(), rcond=None
    )[0]
    np.testing.assert_allclose(
        coefficient_d[:4], [cfg.b, cfg.c, cfg.rho_d, 1.0], atol=0.02
    )


def test_true_support_matches_scenario_windows() -> None:
    critical = generate_dataset("critical_period", n=300, replicate=0)
    nonzero = np.flatnonzero(critical.w_true.sum(axis=0))
    np.testing.assert_array_equal(nonzero, np.arange(4, 8))

    chain = generate_dataset("chain_of_risk", n=300, replicate=0)
    assert np.all(chain.w_true[4:8] == 0)
    assert np.all(chain.w_true[:4, 6:] == 0)
    assert np.all(chain.w_true[8:, :9] == 0)
    assert np.all(chain.w_true[8:, 14:] == 0)
