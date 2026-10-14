"""The 2D pseudo-image CNN: network, training, hazards, and Expected Gradients.

Owned by src/data_analysis/02_pseudo_image_cnn.ipynb and run by the job beside it.
This is the spec's headline comparator (§2): a 2D CNN over the ``C x T`` array with
4x5 kernels, which treats channel order as spatial -- the property the permutation
experiment (§6) exists to test.

Two choices the spec leaves open are fixed here, both for the same reason: a
discrete-time hazard for bin ``t`` may only depend on what was known by bin ``t``.

* **Convolutions are causal in time.** Each layer pads only on the left of the time
  axis, so the output at bin ``t`` sees bins ``<= t``. The DGP keeps every channel
  observed after an event, and the mask is informative (attrition stops at the
  event), so a non-causal CNN would read the outcome off the future. Causal is the
  default; ``causal: false`` exists to measure that leak, as ablation (e) does for
  the proposed model. The channel axis is padded symmetrically: there, a 4-tall
  kernel straddling two streams is the architecture, not a leak.
* **The head is per bin.** After the convolutions the channel axis is flattened
  (position-specific, as an image model is), static covariates are appended, and a
  1x1 head gives two logits per bin -- disease and death against "neither" -- plus a
  learnt per-bin baseline. That is the same competing-risks head and per-interval
  multinomial NLL the spec prescribes for the proposed model, so the two differ in
  their encoders only.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.shared_model_input import DEATH, DISEASE, ModelInput, Standardiser

N_PLANES = 3  # standardised values, observation mask, bins since last observation


def resolve_device(name: str) -> torch.device:
    """``auto`` picks Apple's GPU when present: ~8x faster than CPU on these convs.

    MPS kernels are not bit-reproducible run to run, so a shard records the device it
    used; ``cpu`` is the choice when a rerun must match to the last digit.
    """
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


# --- inputs -----------------------------------------------------------------
def image_tensor(data: ModelInput, standardiser: Standardiser) -> torch.Tensor:
    """``(n, 3, C, T)``: values, mask, and Delta scaled to [0, 1] by the panel length."""
    T = data.X.shape[2]
    planes = np.stack(
        [
            standardiser.transform(data.X),
            data.M.astype(np.float32),
            data.Delta.astype(np.float32) / T,
        ],
        axis=1,
    )
    return torch.from_numpy(planes)


def interval_targets(obs_time: np.ndarray, obs_type: np.ndarray, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(person, bin) class 0/1/2 and an at-risk mask, for the NLL."""
    bins = np.arange(1, n_bins + 1)[None, :]
    at_risk = bins <= obs_time[:, None]
    target = np.where(bins == obs_time[:, None], obs_type[:, None], 0)
    return torch.from_numpy(target.astype(np.int64)), torch.from_numpy(at_risk)


# --- network ----------------------------------------------------------------
class PseudoImageCNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_bins: int,
        n_static: int,
        width: int,
        depth: int,
        kernel: tuple[int, int],
        hidden: int,
        dropout: float,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.kernel = tuple(kernel)
        self.causal = causal
        self.convs = nn.ModuleList(
            nn.Conv2d(N_PLANES if i == 0 else width, width, self.kernel) for i in range(depth)
        )
        self.dropout = nn.Dropout(dropout)
        self.collapse = nn.Conv1d(width * n_channels, hidden, 1)
        self.head = nn.Sequential(
            nn.Conv1d(hidden + n_static, hidden, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 2, 1),
        )
        self.baseline = nn.Parameter(torch.zeros(2, n_bins))

    def _pad(self, h: torch.Tensor) -> torch.Tensor:
        kc, kt = self.kernel
        top = (kc - 1) // 2
        left = kt - 1 if self.causal else (kt - 1) // 2
        return F.pad(h, (left, kt - 1 - left, top, kc - 1 - top))

    def forward(self, image: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """``(B, 3, C, T)``, ``(B, P)`` -> logits ``(B, 3, T)``, class 0 fixed at 0."""
        h = image
        for conv in self.convs:
            h = self.dropout(F.relu(conv(self._pad(h))))
        B, width, C, T = h.shape
        h = F.relu(self.collapse(h.reshape(B, width * C, T)))
        h = torch.cat([h, static[:, :, None].expand(-1, -1, T)], dim=1)
        events = self.head(h) + self.baseline
        return torch.cat([torch.zeros_like(events[:, :1]), events], dim=1)


def disease_cif_at_end(logits: torch.Tensor) -> torch.Tensor:
    """Predicted disease CIF at bin T, per person: the scalar attribution explains."""
    p = torch.softmax(logits, dim=1)
    survival = torch.cumprod(p[:, 0], dim=1)
    before = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    return (before * p[:, DISEASE]).sum(dim=1)


# --- training ---------------------------------------------------------------
@dataclass(frozen=True)
class FitResult:
    model: PseudoImageCNN
    epochs: int
    best_val_nll: float
    seconds: float
    history: tuple[float, ...]  # validation NLL after each epoch


def _nll(logits: torch.Tensor, target: torch.Tensor, at_risk: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits, dim=1)
    picked = log_p.gather(1, target[:, None, :]).squeeze(1)
    return -(picked * at_risk).sum() / at_risk.sum().clamp(min=1)


def fit(
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    architecture: dict[str, Any],
    hyper: dict[str, Any],
    training: dict[str, Any],
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> FitResult:
    """Adam with early stopping on ``val``; returns the best-on-``val`` weights.

    Each of ``train`` / ``val`` is ``(image, static, target, at_risk)``.
    """
    start = time.perf_counter()
    torch.manual_seed(seed)
    train = tuple(t.to(device) for t in train)
    val = tuple(t.to(device) for t in val)
    image, static, target, at_risk = train
    model = PseudoImageCNN(
        n_channels=image.shape[2],
        n_bins=image.shape[3],
        n_static=static.shape[1],
        width=hyper["width"],
        depth=architecture["depth"],
        kernel=tuple(architecture["kernel"]),
        hidden=architecture["hidden"],
        dropout=hyper["dropout"],
        causal=architecture["causal"],
    ).to(device)
    with torch.no_grad():
        # Start the per-bin baseline at the pooled log-odds of each event against
        # "neither" (~1% a bin). From zero -- a one-in-three hazard -- Adam spends the
        # first epochs walking the baseline down, early stopping fires before any
        # history is learnt, and the CIF comes out several-fold too high.
        counts = torch.bincount(target[at_risk].cpu(), minlength=3).double().clamp(min=1)
        log_odds = torch.log(counts[1:] / counts[0]).float().to(device)
        model.baseline.copy_(log_odds[:, None].expand_as(model.baseline))
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    generator = torch.Generator().manual_seed(seed)

    best_nll, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[float] = []
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        order = torch.randperm(image.shape[0], generator=generator).to(device)
        for batch in order.split(training["batch_size"]):
            optimiser.zero_grad()
            loss = _nll(model(image[batch], static[batch]), target[batch], at_risk[batch])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_nll = float(_nll(model(val[0], val[1]), val[2], val[3]))
        history.append(val_nll)
        if val_nll < best_nll - 1e-6:
            best_nll, best_state, best_epoch, stale = val_nll, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= training["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, best_epoch, best_nll, time.perf_counter() - start, tuple(history))


@torch.no_grad()
def predict_hazards(model: PseudoImageCNN, image: torch.Tensor, static: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin disease and death hazards, each ``(n, T)``."""
    model.eval()
    device = next(model.parameters()).device
    p = torch.softmax(model(image.to(device), static.to(device)), dim=1).cpu().numpy().astype(np.float64)
    return p[:, DISEASE], p[:, DEATH]


def randomised_copy(model: PseudoImageCNN, seed: int) -> PseudoImageCNN:
    """Adebayo model-parameter randomisation: every weight that reads the input re-drawn.

    The per-bin baseline is kept. It is an intercept, not a weight on any cell, and
    re-drawing it at random puts the hazard near one-in-three a bin -- everyone then
    has an event within a few bins, the CIF depends on bin 1 alone, and the control
    map is an artefact of the hazard scale rather than of the weights.
    """
    torch.manual_seed(seed)
    clone = copy.deepcopy(model)
    for module in clone.modules():
        if hasattr(module, "reset_parameters") and module is not clone:
            module.reset_parameters()
    clone.eval()
    return clone


# --- attribution ------------------------------------------------------------
def expected_gradients(
    model: PseudoImageCNN,
    image: torch.Tensor,
    static: torch.Tensor,
    baselines: torch.Tensor,
    n_samples: int,
    seed: int,
    chunk: int = 64,
) -> np.ndarray:
    """Per-person Expected Gradients on the value plane, ``(n, C, T)``.

    Integrated Gradients averaged over baselines drawn from the training set, one
    random path position per draw (Erion et al.), explaining the predicted disease
    CIF at bin T. Only the value plane moves along the path; the mask and Delta stay
    the explicand's own. A cell the explicand never observed is held at its own
    value too, so (x - x') is zero there and it receives no attribution: a model
    cannot be credited for reading a value nobody measured.
    """
    model.eval()
    device = next(model.parameters()).device
    image, static, baselines = image.to(device), static.to(device), baselines.to(device)
    rng = np.random.default_rng(seed)
    out = np.empty((image.shape[0],) + tuple(image.shape[2:]), dtype=np.float32)
    for start in range(0, image.shape[0], chunk):
        x = image[start : start + chunk]
        b = x.shape[0]
        pick = torch.from_numpy(rng.integers(baselines.shape[0], size=b * n_samples)).to(device)
        alpha = torch.from_numpy(rng.random(b * n_samples).astype(np.float32))[:, None, None].to(device)

        x_rep = x.repeat_interleave(n_samples, dim=0)
        values = x_rep[:, 0]
        reference = torch.where(x_rep[:, 1] > 0.5, baselines[pick, 0], values)
        point = (reference + alpha * (values - reference)).requires_grad_(True)
        path = torch.cat([point[:, None], x_rep[:, 1:]], dim=1)

        cif = disease_cif_at_end(model(path, static[start : start + b].repeat_interleave(n_samples, dim=0)))
        (grad,) = torch.autograd.grad(cif.sum(), point)
        attribution = ((values - reference) * grad).reshape(b, n_samples, *values.shape[1:])
        out[start : start + b] = attribution.mean(dim=1).detach().cpu().numpy()
    return out
