"""Oracle checks: what a correctly specified model achieves on the observed panel.

Owned by src/data_analysis/03_oracle_checks.py. The ledger's true-hazard ceiling
(AUC 0.710 in ``critical_period``) reads the *latent* streams, which no model sees,
so it cannot say whether a model that falls short is failing to learn or has nothing
to learn from. The oracle here is the bound that answers that. It is allowed to know
the scenario's functional form -- which stream, which bins, how they combine -- and
applies it to what a model actually gets: per-fold standardised channel values with
cells missing. Only the coefficients and the per-bin baselines are fitted.

* **Prediction.** If the oracle beats the CNN by a wide margin, the signal is in the
  observed data and the CNN is not finding it. If it does not, no architecture will.
* **Attribution.** The oracle puts weight on the true cells by construction. Run
  through the same Expected Gradients, population map, recovery metrics and Adebayo
  controls as the CNN, it shows what those metrics give a model known to be right.
  A metric the oracle cannot pass is a metric, not a model, problem.

``OracleHazard`` takes the CNN's ``(image, static)`` and returns the CNN's logits,
so the CNN module's prediction and attribution functions score it unchanged.

The stream estimate in a bin is the mean of the person's observed standardised
channels in that stream; a bin with none observed reads 0, the training mean -- the
same value the standardiser gives an unobserved cell.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from utils.data_analysis_02_pseudo_image_cnn_model import _nll

# The scenario's history feature; "static" has none, and is the cohort + sex model.
FORMS = ("static", "accumulation", "critical_period", "sensitive_period", "chain_of_risk", "mobility", "null")


def stream_estimates(image: torch.Tensor, n_streams: int = 3) -> torch.Tensor:
    """``(B, 3, C, T)`` -> ``(B, streams, T)``: mean of each stream's observed channels."""
    values, mask = image[:, 0], image[:, 1]
    B, C, T = values.shape
    total = (values * mask).reshape(B, n_streams, C // n_streams, T).sum(dim=2)
    count = mask.reshape(B, n_streams, C // n_streams, T).sum(dim=2)
    return total / count.clamp(min=1.0)


def _mobility_parts(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per bin, the slope and level of the upstream path so far (as ``shared_dgp``)."""
    B, T = U.shape
    slope, level = torch.zeros_like(U), torch.zeros_like(U)
    for t in range(1, T):
        time = torch.arange(1, t + 2, dtype=U.dtype, device=U.device)
        centred = time - time.mean()
        slope[:, t] = U[:, : t + 1] @ centred / (centred**2).sum()
        level[:, t] = U[:, : t + 1].mean(dim=1)
    return slope, level


def history_feature(form: str, U: torch.Tensor, D: torch.Tensor, mobility_coef: torch.Tensor) -> torch.Tensor:
    """The scenario's ``f(history)`` before standardisation, ``(B, T)``, causal in time.

    Mirrors ``shared_dgp._scenario_score`` bin for bin, on stream estimates instead of
    the latent streams. Kept separate rather than shared because the DGP works in
    numpy on latents and this must be differentiable in the observed values.
    """
    B, T = U.shape
    zero = torch.zeros(B, dtype=U.dtype, device=U.device)
    if form in ("static", "null"):
        return torch.zeros_like(U)
    if form == "accumulation":
        return U.cumsum(dim=1) / torch.arange(1, T + 1, dtype=U.dtype, device=U.device)
    if form == "critical_period":
        start, stop = 4, 8  # bins 5-8
        return torch.stack(
            [zero if t < start else U[:, start : min(t + 1, stop)].mean(dim=1) for t in range(T)], dim=1
        )
    if form == "sensitive_period":
        times = torch.arange(1, T + 1, dtype=U.dtype, device=U.device)
        weights = torch.exp(-0.5 * ((times - 6.0) / 2.0) ** 2)
        return (U * weights).cumsum(dim=1) / weights.cumsum(dim=0)
    if form == "chain_of_risk":
        early = U[:, :6].mean(dim=1)
        return torch.stack(
            [zero if t < 9 else early * D[:, 9 : min(t + 1, 14)].mean(dim=1) for t in range(T)], dim=1
        )
    if form == "mobility":
        slope, level = _mobility_parts(U)
        return slope - mobility_coef * level
    raise ValueError(f"form must be one of {FORMS}, got {form!r}")


class OracleHazard(nn.Module):
    """The true hazard's form on observed cells; the CNN's interface and head.

    ``logit h_disease[t] = b[t] + beta * z_t + gamma' S`` with ``z_t`` the scenario
    feature standardised per bin (as the DGP does, D02), and
    ``logit h_death[t] = b'[t] + alpha * cummean(D)_t + gamma_d' S`` when
    ``death_history`` -- the DGP's death hazard -- or without the downstream term when
    not, which isolates how much attribution reaches through the competing risk.
    """

    def __init__(self, form: str, n_bins: int, n_static: int, death_history: bool = True) -> None:
        super().__init__()
        if form not in FORMS:
            raise ValueError(f"form must be one of {FORMS}, got {form!r}")
        self.form, self.death_history = form, death_history
        self.beta = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(n_static))
        self.gamma_death = nn.Parameter(torch.zeros(n_static))
        self.baseline = nn.Parameter(torch.zeros(2, n_bins))
        # Per-bin standardisation of the two features, from the training fold.
        self.register_buffer("feature_mean", torch.zeros(2, n_bins))
        self.register_buffer("feature_sd", torch.ones(2, n_bins))
        self.register_buffer("mobility_coef", torch.zeros(n_bins))

    def _raw(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        streams = stream_estimates(image)
        U, D = streams[:, 0], streams[:, 2]
        disease = history_feature(self.form, U, D, self.mobility_coef)
        death = D.cumsum(dim=1) / torch.arange(1, D.shape[1] + 1, dtype=D.dtype, device=D.device)
        return disease, death

    @torch.no_grad()
    def calibrate(self, image: torch.Tensor) -> None:
        """Set the feature standardisation (and mobility's level coefficient) on ``image``."""
        if self.form == "mobility":
            slope, level = _mobility_parts(stream_estimates(image)[:, 0])
            centred_s, centred_l = slope - slope.mean(0), level - level.mean(0)
            var = (centred_l**2).mean(0)
            self.mobility_coef.copy_(torch.where(var > 1e-8, (centred_s * centred_l).mean(0) / var, 0.0))
        for row, raw in enumerate(self._raw(image)):
            self.feature_mean[row] = raw.mean(dim=0)
            self.feature_sd[row] = raw.std(dim=0, unbiased=False)

    def _standardised(self, raw: torch.Tensor, row: int) -> torch.Tensor:
        sd = self.feature_sd[row]
        live = sd > 1e-8  # a bin where the feature is constant (e.g. before the window) is 0
        return torch.where(live, (raw - self.feature_mean[row]) / torch.where(live, sd, 1.0), 0.0)

    def forward(self, image: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """``(B, 3, C, T)``, ``(B, P)`` -> logits ``(B, 3, T)``, class 0 fixed at 0."""
        disease_raw, death_raw = self._raw(image)
        disease = self.baseline[0] + self.beta * self._standardised(disease_raw, 0) + (static @ self.gamma)[:, None]
        death = self.baseline[1] + (static @ self.gamma_death)[:, None]
        if self.death_history:
            death = death + self.alpha * self._standardised(death_raw, 1)
        events = torch.stack([disease, death], dim=1)
        return torch.cat([torch.zeros_like(events[:, :1]), events], dim=1)


def fit_oracle(
    form: str,
    image: torch.Tensor,
    static: torch.Tensor,
    target: torch.Tensor,
    at_risk: torch.Tensor,
    death_history: bool = True,
) -> OracleHazard:
    """Maximum likelihood, full batch: a handful of parameters, so L-BFGS converges.

    The per-interval multinomial NLL is the CNN's own, so the two are fitted to the
    same objective on the same rows.
    """
    model = OracleHazard(form, image.shape[3], static.shape[1], death_history)
    model.calibrate(image)
    with torch.no_grad():  # the pooled log-odds start, as the CNN's (D09)
        counts = torch.bincount(target[at_risk], minlength=3).double().clamp(min=1)
        model.baseline.copy_(torch.log(counts[1:] / counts[0]).float()[:, None].expand_as(model.baseline))
    optimiser = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = _nll(model(image, static), target, at_risk)
        loss.backward()
        return loss

    optimiser.step(closure)
    model.eval()
    return model


def randomised_oracle(model: OracleHazard, seed: int) -> OracleHazard:
    """Adebayo parameter randomisation: every coefficient on an input re-drawn N(0, 1).

    The baseline is kept, as the CNN's control keeps it. The *form* cannot be
    randomised -- it is the oracle's architecture -- which is the point: what survives
    here is what the architecture alone puts in the map.
    """
    generator = torch.Generator().manual_seed(seed)
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in (clone.beta, clone.alpha, clone.gamma, clone.gamma_death):
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return clone.eval()
