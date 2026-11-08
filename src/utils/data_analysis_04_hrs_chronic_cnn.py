"""Analysis 04: which cells of the HRS life image a CNN and an RNN use to classify an outcome.

Owned by src/data_analysis/04_hrs_chronic_cnn.py. HRS is blind (D07), so everything
here runs on the user's machine and returns only what ``shared_secure_report`` lets
through: metrics and per-cell means over at least ``min_cell`` people (D22).

* **Task** (D23, D25, ``shared_hrs_chronic``): a chronic condition at the status
  interview at age A, or death within a horizon after it, from the life image at ages
  50 to A.
* **Models.** Two networks on the same image tensor: the pseudo-image CNN -- the
  analysis 02 encoder, uncausal because every cell precedes the status interview, then
  a per-bin collapse averaged over time and a logistic head -- and a GRU reading the
  image bin by bin (D24). The benchmarks they must beat -- static, summary, age-window
  and elastic-net logistic regressions, and tuned LightGBM on a snapshot, summaries
  and every cell -- live in ``data_analysis_04_hrs_chronic_cnn_benchmarks`` (D29).
* **Activation maps.** Grad-CAM on the encoder's last layer: the gradient of the logit,
  averaged over each feature map, weights the maps, and the positive part of the sum
  is the cell's activation (Selvaraju et al.). Padding keeps the last layer on the
  input grid, so no upsampling is needed; it needs conv layers, so the RNN has none.
  Expected Gradients (analysis 02's), for both networks, explains
  the predicted probability on the input values themselves, signed. Both are averaged
  per cell over the test people who *observed* that cell (as ``population_map``),
  separately for cases and controls.
* **Controls.** Every fold: the same network with re-drawn weights. Chosen folds: the
  network refitted on shuffled labels. Each reported as Spearman similarity to the
  model's map *and* the ratio of attribution mass -- similarity alone is blind to scale
  and fails a correct model (D14).
"""

from __future__ import annotations

import copy
import itertools
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from utils.data_analysis_02_pseudo_image_cnn_model import (
    N_PLANES,
    ConvEncoder,
    expected_gradients,
    randomised_copy,
    resolve_device,
)
from utils.data_analysis_04_hrs_chronic_cnn_benchmarks import BENCHMARKS, fit_benchmarks
from utils.shared_dgp import MASTER_SEED
from utils.shared_evaluation import map_similarity, population_map
from utils.shared_hrs_chronic import ChronicDataset
from utils.shared_model_input import fit_standardiser
from utils.shared_reproducibility import stable_seed

MODELS = (*BENCHMARKS, "cnn", "rnn")


def seed_for(condition: str, status_bin: int, tag: str) -> int:
    """Seeds depend on the dataset and a tag, never on the model, so folds are shared."""
    return stable_seed(MASTER_SEED, f"hrs_chronic|{condition}|{tag}", status_bin, 0)


# --- folds ----------------------------------------------------------------------
def household_folds(y: np.ndarray, household: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``k`` (train, test) pairs stratified on ``y``, never splitting a household.

    Spouses share every household channel (income, wealth, poverty) exactly; split
    apart, one would be scored on the other's numbers.
    """
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed % 2**31)
    return list(splitter.split(np.zeros_like(y), y, household))


# --- inputs -----------------------------------------------------------------------
@dataclass(frozen=True)
class Scaler:
    """Per-fold standardisation of the image values and the static covariates.

    A static covariate missing in the training fold (parents' education often is) is
    filled with the training mean and gets a 0/1 missing indicator after the values,
    so the networks see the same inputs a logistic regression with indicators would.
    """

    image: Any
    static_mean: np.ndarray
    static_sd: np.ndarray
    flagged: np.ndarray  # columns that get a missing indicator

    def static(self, raw: np.ndarray) -> np.ndarray:
        missing = np.isnan(raw)
        filled = np.where(missing, self.static_mean, raw)
        scaled = (filled - self.static_mean) / self.static_sd
        return np.column_stack([scaled, missing[:, self.flagged].astype(float)]).astype(np.float32)


def image_scaling(config: dict[str, Any], network: str) -> dict[str, Any]:
    """A network's image scaling: its own ``image_scaling`` if it has one, else the config's (D33)."""
    return {"method": "cell_z", **config.get("image_scaling", {}), **config["networks"][network].get("image_scaling", {})}


def fit_scaler(data: ChronicDataset, rows: np.ndarray, scaling: dict[str, Any] | None = None) -> Scaler:
    """``scaling`` is ``{method, quantiles}`` for ``fit_standardiser``; the static covariates are always z-scored."""
    scaling = scaling or {"method": "cell_z"}
    static = _static_raw(data)[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a covariate nobody in the fold has
        mean = np.nan_to_num(np.nanmean(static, axis=0))
        sd = np.nan_to_num(np.nanstd(static, axis=0))
    image = fit_standardiser(data.X[rows], scaling["method"], tuple(scaling.get("quantiles", (0.01, 0.99))))
    return Scaler(image, mean, np.where(sd > 0, sd, 1.0), np.isnan(static).any(axis=0))


def _static_raw(data: ChronicDataset) -> np.ndarray:
    return data.static.astype(np.float64)


def tensors(data: ChronicDataset, rows: np.ndarray, scaler: Scaler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(image (n, 3, C, K), static (n, P'), y (n,))`` for ``rows``; P' adds missing indicators."""
    K = data.X.shape[2]
    image = np.stack(
        [scaler.image.transform(data.X[rows]), data.M[rows].astype(np.float32), data.Delta[rows].astype(np.float32) / K],
        axis=1,
    )
    static = scaler.static(_static_raw(data)[rows])
    return torch.from_numpy(image), torch.from_numpy(static), torch.from_numpy(data.y[rows].astype(np.float32))


# --- the networks --------------------------------------------------------------------
class PseudoImageClassifier(nn.Module):
    """:class:`ConvEncoder`, a per-bin collapse averaged over bins, then a logistic head."""

    def __init__(
        self, n_channels: int, n_static: int, width: int, kernels, dilations, hidden: int, dropout: float,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder(width, kernels, dilations, dropout, causal)
        self.collapse = nn.Conv1d(self.encoder.out_planes * n_channels, hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        """The last conv layer's feature maps, as a one-element list (see :class:`KernelNet`)."""
        return [self.encoder(image)]

    def head_from(self, features: torch.Tensor | list[torch.Tensor], static: torch.Tensor) -> torch.Tensor:
        """Last-layer feature maps ``(B, F, C, K)`` -> logits ``(B,)``."""
        if isinstance(features, list):
            (features,) = features
        B, planes, C, K = features.shape
        pooled = F.relu(self.collapse(features.reshape(B, planes * C, K))).mean(dim=2)
        return self.head(torch.cat([pooled, static], dim=1)).squeeze(1)

    def forward(self, image: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head_from(self.encoder(image), static)


class RecurrentClassifier(nn.Module):
    """A GRU over the bins, then the same logistic head as the CNN (D24).

    Each bin's input is every channel's standardised value, mask and Delta (``3C``
    numbers): the image read as a sequence, so the channels have no spatial order and
    no kernel can straddle two streams. Missingness enters as inputs, as in GRU-D's
    simplest form, without its learnt decay. Bidirectional by default: every bin
    precedes the status interview, as for the uncausal CNN.
    """

    def __init__(
        self, n_channels: int, n_static: int, hidden: int, layers: int, dropout: float, bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            3 * n_channels, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional,
        )
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out + n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, image: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        B, planes, C, K = image.shape
        _, last = self.gru(image.permute(0, 3, 1, 2).reshape(B, K, planes * C))
        summary = torch.cat([last[-2], last[-1]], dim=1) if self.bidirectional else last[-1]
        return self.head(torch.cat([summary, static], dim=1)).squeeze(1)


class KernelNet(nn.Module):
    """Convolution stages with any kernels, then the CNN's collapse and head (D30).

    ``stages`` is a list of stages; a stage is a list of branches, each a dict with
    ``kernel: [rows, bins]`` (either may be ``"all"``, the whole axis), ``dilation`` along
    time (default 1) and ``padding`` ``"same"`` (default; zeros, the output keeps the input's
    rows x bins) or ``"valid"`` (none; a whole-axis kernel then collapses that axis to 1).
    Branches of one stage read the same input in parallel, each with its own ReLU; outputs
    of equal shape are concatenated along features and feed the next stage. A last stage
    whose branches end in different shapes (the two-view mix) keeps them apart: each gets
    its own per-bin collapse and bin average, and the vectors meet at the head.

    With ``split_width`` a stage's ``width`` filters are shared out among its branches, so a
    mix has about the parameters of a single-kernel network rather than a multiple.
    """

    def __init__(self, n_channels: int, n_bins: int, n_static: int, width: int, stages: list, hidden: int,
                 dropout: float, split_width: bool = True) -> None:
        super().__init__()
        self.spec: list[list[tuple[tuple[int, int], int, str]]] = []
        self.stages = nn.ModuleList()
        planes, shape = N_PLANES, (n_channels, n_bins)
        out_shapes: list[tuple[int, int, int]] = []
        for s, stage in enumerate(stages):
            branch_width = max(1, round(width / len(stage))) if split_width else width
            convs, specs, out_shapes = nn.ModuleList(), [], []
            for branch in stage:
                rows, bins = (axis if k == "all" else int(k) for k, axis in zip(branch["kernel"], shape))
                dilation, padding = int(branch.get("dilation", 1)), branch.get("padding", "same")
                convs.append(nn.Conv2d(planes, branch_width, (rows, bins), dilation=(1, dilation)))
                specs.append(((rows, bins), dilation, padding))
                if padding == "same":
                    out = shape
                else:
                    out = (shape[0] - rows + 1, shape[1] - dilation * (bins - 1))
                if min(out) < 1:
                    raise ValueError(f"stage {s}: kernel {rows}x{bins} does not fit a {shape[0]}x{shape[1]} input")
                out_shapes.append((branch_width, *out))
            self.stages.append(convs)
            self.spec.append(specs)
            if s < len(stages) - 1:
                if len({o[1:] for o in out_shapes}) != 1:
                    raise ValueError(f"stage {s}: branches end in different shapes, so only a last stage may differ")
                planes, shape = sum(o[0] for o in out_shapes), out_shapes[0][1:]
        # The last stage's outputs, grouped: one group if every branch keeps the same shape.
        same = len({o[1:] for o in out_shapes}) == 1
        self.groups = [list(range(len(out_shapes)))] if same else [[i] for i in range(len(out_shapes))]
        self.collapses = nn.ModuleList(
            nn.Conv1d(sum(out_shapes[i][0] for i in g) * out_shapes[g[0]][1], hidden, 1) for g in self.groups
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * len(self.groups) + n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.time_reach = 1 + sum(max(d * (b - 1) for (_, b), d, _ in specs) for specs in self.spec)

    @staticmethod
    def _pad(h: torch.Tensor, kernel: tuple[int, int], dilation: int, padding: str) -> torch.Tensor:
        if padding == "valid":
            return h
        rows, bins = kernel
        span = dilation * (bins - 1)
        top = (rows - 1) // 2
        return F.pad(h, (span // 2, span - span // 2, top, rows - 1 - top))

    def features(self, image: torch.Tensor) -> list[torch.Tensor]:
        """The last stage's feature maps, one tensor per group (see the class docstring)."""
        h = image
        for convs, specs in zip(self.stages, self.spec):
            outs = [self.dropout(F.relu(conv(self._pad(h, *spec)))) for conv, spec in zip(convs, specs)]
            h = torch.cat(outs, dim=1) if len({o.shape[2:] for o in outs}) == 1 else outs
        if isinstance(h, torch.Tensor):
            return [h]
        return [torch.cat([h[i] for i in g], dim=1) for g in self.groups]

    def head_from(self, features: list[torch.Tensor], static: torch.Tensor) -> torch.Tensor:
        pooled = []
        for collapse, f in zip(self.collapses, features):
            B, planes, rows, bins = f.shape
            pooled.append(F.relu(collapse(f.reshape(B, planes * rows, bins))).mean(dim=2))
        return self.head(torch.cat([*pooled, static], dim=1)).squeeze(1)

    def forward(self, image: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.head_from(self.features(image), static)


NETWORKS = ("cnn", "rnn", "kernel_cnn")


def build_network(
    network: str, architecture: dict[str, Any], n_channels: int, n_bins: int, hyper: dict[str, Any], n_static: int = 2
) -> nn.Module:
    kind = architecture.get("type", network)  # a profile may name arms freely and give their type
    if kind == "rnn":
        return RecurrentClassifier(
            n_channels, n_static, hyper["hidden"], architecture["layers"], hyper["dropout"], architecture["bidirectional"]
        )
    if kind == "kernel_cnn":
        model = KernelNet(n_channels, n_bins, n_static, hyper["width"], architecture["stages"], architecture["hidden"],
                          hyper["dropout"], architecture.get("split_width", True))
        # Uncausal and pooled over bins, so the whole image is in view if the reach spans it.
        if architecture.get("require_full_history", True) and model.time_reach < n_bins:
            raise ValueError(f"time reach {model.time_reach} < {n_bins} bins (D13); set require_full_history: false "
                             "only for an arm that is meant not to look across ages")
        return model
    if kind != "cnn":
        raise ValueError(f"network must be one of {NETWORKS}, got {network!r}")
    model = PseudoImageClassifier(
        n_channels, n_static, hyper["width"], tuple(tuple(k) for k in architecture["kernels"]),
        tuple(architecture["dilations"]), architecture["hidden"], hyper["dropout"], architecture["causal"],
    )
    reach = model.encoder.receptive_field if architecture["causal"] else (model.encoder.receptive_field + 1) // 2
    if reach < n_bins:
        raise ValueError(f"receptive field reaches {reach} bins, the image has {n_bins} (D13)")
    return model


def build_classifier(architecture: dict[str, Any], n_channels: int, n_bins: int, hyper: dict[str, Any]) -> PseudoImageClassifier:
    return build_network("cnn", architecture, n_channels, n_bins, hyper)


def fit_classifier(
    train: tuple[torch.Tensor, ...], val: tuple[torch.Tensor, ...], network: str, architecture: dict[str, Any],
    hyper: dict[str, Any], training: dict[str, Any], seed: int, device: torch.device,
) -> tuple[nn.Module, int, float]:
    """Adam on binary cross-entropy, early stopping on ``val``; best-on-``val`` weights."""
    torch.manual_seed(seed)
    train = tuple(t.to(device) for t in train)
    val = tuple(t.to(device) for t in val)
    image, static, y = train
    model = build_network(network, architecture, image.shape[2], image.shape[3], hyper, static.shape[1]).to(device)
    with torch.no_grad():  # start at the prevalence, as the hazard CNN starts at the pooled log-odds (D09)
        prevalence = y.mean().clamp(1e-4, 1 - 1e-4)
        model.head[-1].bias.fill_(float(torch.log(prevalence / (1 - prevalence))))
    optimiser = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, training["max_epochs"] + 1):
        model.train()
        for batch in torch.randperm(image.shape[0], generator=generator).to(device).split(training["batch_size"]):
            optimiser.zero_grad()
            F.binary_cross_entropy_with_logits(model(image[batch], static[batch]), y[batch]).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            loss = float(F.binary_cross_entropy_with_logits(model(val[0], val[1]), val[2]))
        if loss < best - 1e-6:
            best, best_state, best_epoch, stale = loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= training["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval(), best_epoch, best


@torch.no_grad()
def predict(model: nn.Module, image: torch.Tensor, static: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    return torch.sigmoid(model(image.to(device), static.to(device))).cpu().numpy().astype(np.float64)


def grad_cam(model: nn.Module, image: torch.Tensor, static: torch.Tensor, chunk: int = 256) -> np.ndarray:
    """Per-person Grad-CAM of the logit on the last conv layer(s), ``(n, C, K)``.

    Each last-layer tensor contributes its gradient-weighted sum of feature maps. A map
    that a whole-axis kernel collapsed to one row (``trajectory``) or one bin
    (``cross_section``) is spread along that axis: it says which variable, or which age,
    and nothing finer. The contributions are added before the positive part is taken.
    """
    model.eval()
    device = next(model.parameters()).device
    C, K = image.shape[2:]
    out = []
    for start in range(0, image.shape[0], chunk):
        features = model.features(image[start : start + chunk].to(device))
        logit = model.head_from(features, static[start : start + chunk].to(device))
        grads = torch.autograd.grad(logit.sum(), features)  # people are independent in eval mode
        total = 0
        for f, g in zip(features, grads):
            cam = (g.mean(dim=(2, 3), keepdim=True) * f).sum(dim=1)  # (B, rows, bins)
            total = total + cam.expand(-1, C, K) if cam.shape[1:] != (C, K) else total + cam
        out.append(F.relu(total).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# --- metrics --------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, p: np.ndarray, groups: int = 10) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.quantile(p, np.linspace(0, 1, groups + 1))
    group = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, groups - 1)
    gaps = [(abs(p[group == g].mean() - y[group == g].mean()), (group == g).sum()) for g in range(groups) if (group == g).any()]
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ici_decile": float(np.average([g for g, _ in gaps], weights=[w for _, w in gaps])),
        "prevalence": float(y.mean()),
    }


# --- one outer fold -------------------------------------------------------------------
@dataclass
class MapTotals:
    """Per-cell sums and observer counts, per map and group, summed over folds."""

    sums: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]

    @classmethod
    def empty(cls) -> MapTotals:
        return cls({}, {})

    def add(self, name: str, per_person: np.ndarray, observed: np.ndarray) -> None:
        total = (per_person * observed).sum(axis=0)
        count = observed.sum(axis=0)
        self.sums[name] = self.sums.get(name, 0) + total
        self.counts[name] = self.counts.get(name, 0) + count


def _grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def _split_for_stopping(data: ChronicDataset, rows: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """About ``fraction`` of the households in ``rows``, held out for early stopping only."""
    fit_idx, stop_idx = household_folds(data.y[rows], data.household[rows], max(2, round(1 / fraction)), seed)[0]
    return rows[fit_idx], rows[stop_idx]


def _fit_network(data: ChronicDataset, rows: np.ndarray, network: str, hyper: dict[str, Any], config: dict[str, Any],
                 seed: int, y_override: np.ndarray | None = None):
    """Scale on ``rows``, hold out households for early stopping, train."""
    scaler = fit_scaler(data, rows, image_scaling(config, network))
    fit_rows, stop_rows = _split_for_stopping(data, rows, config["training"]["early_stop_fraction"], seed)
    train_t, stop_t = tensors(data, fit_rows, scaler), tensors(data, stop_rows, scaler)
    if y_override is not None:  # shuffled labels, aligned to ``rows``
        lookup = dict(zip(rows, y_override))
        train_t = (*train_t[:2], torch.tensor([lookup[r] for r in fit_rows], dtype=torch.float32))
        stop_t = (*stop_t[:2], torch.tensor([lookup[r] for r in stop_rows], dtype=torch.float32))
    model, epochs, _ = fit_classifier(
        train_t, stop_t, network, config["networks"][network]["architecture"], hyper, config["training"], seed,
        resolve_device(config["device"]),
    )
    return model, scaler, epochs


def tune(data: ChronicDataset, rows: np.ndarray, network: str, config: dict[str, Any], seed: int) -> tuple[dict[str, Any], list]:
    """Pick hyperparameters by mean inner-fold log loss. Inner scores are never reported as results."""
    grid = _grid(config["networks"][network]["grid"])
    if len(grid) == 1:
        return grid[0], []
    inner = household_folds(data.y[rows], data.household[rows], config["inner_folds"], seed)
    scores = []
    for g, hyper in enumerate(grid):
        losses = []
        for k, (fit_idx, val_idx) in enumerate(inner):
            model, scaler, _ = _fit_network(data, rows[fit_idx], network, hyper, config, seed + 7919 * (g + 1) + k)
            val_t = tensors(data, rows[val_idx], scaler)
            losses.append(log_loss(val_t[2].numpy(), np.clip(predict(model, *val_t[:2]), 1e-6, 1 - 1e-6), labels=[0, 1]))
        scores.append({**hyper, "inner_log_loss": float(np.mean(losses))})
    best = min(range(len(grid)), key=lambda g: scores[g]["inner_log_loss"])
    return grid[best], scores


def run_fold(data: ChronicDataset, fold: int, train: np.ndarray, test: np.ndarray, config: dict[str, Any],
             maps: MapTotals) -> dict[str, Any]:
    """Fit every model on ``train``, score on ``test``, add each network's maps to ``maps``.

    Map names are ``<network>/<kind>/<group>``: kinds ``gradcam`` (CNN only),
    ``eg_signed`` and ``eg_abs``; groups ``cases``, ``controls`` and ``all``.
    """
    seed = seed_for(data.condition, data.status_bin, f"fold{fold}")
    y_train, y_test = data.y[train], data.y[test]
    out: dict[str, Any] = {"metrics": {}, "seconds": {}, "sanity": {}}

    if config.get("run_benchmarks", True):
        inner = household_folds(y_train, data.household[train], config["inner_folds"], seed + 7)
        predictions, out["benchmarks"], out["seconds"]["benchmarks"] = fit_benchmarks(data, train, test, config, seed, inner)
        for name, p in predictions.items():
            out["metrics"][name] = classification_metrics(y_test, p)

    # The same explained people and Expected Gradients baselines for every network.
    att = config["attribution"]
    rng = np.random.default_rng(seed + 2)
    explain = np.sort(rng.choice(len(test), size=min(att["n_explain"], len(test)), replace=False))
    baseline_rows = rng.choice(len(train), size=min(att["n_baselines"], len(train)), replace=False)
    observed = data.M[test][explain]
    cases = y_test[explain] == 1

    for n, network in enumerate(config["networks"]):
        start = time.perf_counter()
        net_seed = seed + 1000 * n
        hyper, inner_scores = tune(data, train, network, config, net_seed)
        model, scaler, epochs = _fit_network(data, train, network, hyper, config, net_seed + 1)
        image, static_t, _ = tensors(data, test, scaler)
        out["metrics"][network] = classification_metrics(y_test, predict(model, image, static_t))
        out[network] = {"hyperparameters": hyper, "best_epoch": epochs, "inner_scores": inner_scores}
        if scaler.image.method != "cell_z":  # per-channel; min and max are one person's values and stay local (D22)
            out[network]["image_scaling"] = {"method": scaler.image.method, **{
                k: v for k, v in scaler.image.stats().items() if k not in ("minimum", "maximum")}}
        out["seconds"][network] = time.perf_counter() - start

        start = time.perf_counter()
        baselines = tensors(data, train, scaler)[0][baseline_rows]

        def attributions(m) -> dict[str, np.ndarray]:
            eg = expected_gradients(m, image[explain], static_t[explain], baselines, att["n_samples"], net_seed + 3,
                                    target=torch.sigmoid)
            found = {"eg_signed": eg, "eg_abs": np.abs(eg)}
            if hasattr(m, "features"):  # every convolutional network: Grad-CAM needs conv layers
                found["gradcam"] = grad_cam(m, image[explain], static_t[explain])
            return found

        found = attributions(model)
        for kind, per_person in found.items():
            maps.add(f"{network}/{kind}/cases", per_person[cases], observed[cases])
            maps.add(f"{network}/{kind}/controls", per_person[~cases], observed[~cases])
            maps.add(f"{network}/{kind}/all", per_person, observed)

        def compared(found: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            """Population maps of |attribution| (``population_map``): ``gradcam`` and ``eg``."""
            return {("eg" if k == "eg_signed" else k): population_map(v, observed)[0]
                    for k, v in found.items() if k != "eg_abs"}

        reference = compared(found)

        def compare(tag: str, m) -> None:
            for kind, control_map in compared(attributions(m)).items():
                total = reference[kind].sum()
                out["sanity"][f"{network}_{tag}_{kind}_similarity"] = map_similarity(reference[kind], control_map)
                out["sanity"][f"{network}_{tag}_{kind}_mass_ratio"] = (
                    float(control_map.sum() / total) if total > 0 else float("nan")
                )

        if config["sanity"]["param_random"]:
            compare("param_random", randomised_copy(model, net_seed + 4))
        if fold in config["sanity"]["label_random_folds"]:
            shuffled = np.random.default_rng(seed + 5).permutation(y_train)  # one permutation for every network
            shuffled_model, _, _ = _fit_network(data, train, network, hyper, config, net_seed + 6, y_override=shuffled)
            compare("label_random", shuffled_model)
        out["seconds"][f"{network}_maps"] = time.perf_counter() - start
    return out


# --- the figure ---------------------------------------------------------------------------
def _panel_values(values) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in values], dtype=float)


def map_figure(condition_report: dict[str, Any], channel_names: list[str], style: dict[str, Any],
               channel_domains: list[str] | None = None):
    """Top: CNN Grad-CAM for cases, controls, difference. Bottom: Expected Gradients."""
    from matplotlib.figure import Figure

    maps = condition_report["maps"]
    dataset = condition_report["dataset"]
    ages = [50 + 2 * b for b in range(dataset["image_bins"])]
    panels = []
    if "cnn" in maps:
        cnn = maps["cnn"]
        panels += [
            ("CNN Grad-CAM, cases", cnn["gradcam"]["cases"], style["map_cmap"], False),
            ("CNN Grad-CAM, controls", cnn["gradcam"]["controls"], style["map_cmap"], False),
            ("CNN Grad-CAM, cases minus controls", cnn["gradcam"]["difference"], style["diverging_cmap"], True),
            ("CNN Expected Gradients on P(case), signed", cnn["eg_signed"]["all"], style["diverging_cmap"], True),
        ]
    if "rnn" in maps:
        rnn = maps["rnn"]
        panels += [
            ("RNN Expected Gradients on P(case), signed", rnn["eg_signed"]["all"], style["diverging_cmap"], True),
            ("RNN |Expected Gradients|, cases minus controls", rnn["eg_abs"]["difference"], style["diverging_cmap"], True),
        ]
    domains = channel_domains or [n for n in ("upstream", "midstream", "downstream") for _ in range(len(channel_names) // 3)]
    boundaries = [i - 0.5 for i in range(1, len(domains)) if domains[i] != domains[i - 1]]
    width, height = style["figsize"]
    fig = Figure(figsize=(width, height * max(1.0, len(channel_names) / 12) ** 0.6), dpi=style["dpi"])
    axes = fig.subplots(2, 3)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    for ax, (title, values, cmap, centred) in zip(axes.ravel(), panels):
        values = _panel_values(values)
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
        kwargs = {"vmin": -limit, "vmax": limit} if centred else {"vmin": 0}
        mesh = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=style["label_fs"], loc="left")
        ax.set_xticks(range(0, len(ages), 2))
        ax.set_xticklabels([str(a) for a in ages[::2]], fontsize=style["tick_fs"])
        ax.set_xlabel("age at start of bin", fontsize=style["tick_fs"])
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names, fontsize=style["tick_fs"])
        for boundary in boundaries:
            ax.axhline(boundary, color=style["separator_color"], linewidth=1.0)
        fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    lo, hi = dataset["status_ages"]
    fig.suptitle(
        f"{dataset['title']} age {lo}-{hi}: cases {dataset['cases']}, controls {dataset['controls']}",
        fontsize=style["title_fs"],
    )
    fig.text(
        0.01, 0.005,
        "Each cell: mean over the test people who observed it, all folds. Blank: fewer than the report's "
        "min_cell observers (suppressed) or never asked.\nRead the difference panels: raw maps also track "
        "feature magnitude, which cases and controls share.",
        fontsize=style["tick_fs"], color=style["roles"]["footer"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


# --- the average life image (D28) ---------------------------------------------------------
def average_image(data: ChronicDataset, min_cell: int) -> dict[str, np.ndarray]:
    """Per-cell totals and head counts for the peer-standardised mean image of each group.

    Each cell is standardised against everyone kept who observed it (same channel, same
    age bin), so a group's mean reads as SD from same-age peers, as in the simulation's
    images. Returns raw totals and counts: suppress through ``SecureReport.mean_map``.
    ``peer_sd`` is NaN where fewer than ``min_cell`` people observed the cell.
    """
    X = data.X.astype(np.float64)
    observed = data.M
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells nobody observed
        mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
    n_all = observed.sum(axis=0)
    sd = np.where((n_all >= min_cell) & (sd > 1e-8), sd, np.nan)
    z = (X - mean) / sd
    totals: dict[str, np.ndarray] = {
        "peer_mean/total": np.where(observed, X, 0.0).sum(axis=0),
        "peer_mean/count": n_all,
        "peer_sd": sd,
    }
    for group, rows in (("cases", data.y == 1), ("controls", data.y == 0), ("all", np.ones(data.n, dtype=bool))):
        seen = observed[rows] & np.isfinite(z[rows])
        totals[f"{group}/total"] = np.where(seen, z[rows], 0.0).sum(axis=0)
        totals[f"{group}/count"] = seen.sum(axis=0)
        totals[f"{group}/n"] = np.array(int(rows.sum()))
    return totals


def observed_share(count: np.ndarray, n: int, min_cell: int) -> np.ndarray:
    """Share of a group observing each cell; NaN where fewer than ``min_cell`` did."""
    count = np.asarray(count)
    return np.where(count >= min_cell, count / max(n, 1), np.nan)


def average_image_figure(condition_report: dict[str, Any], channel_names: list[str], style: dict[str, Any],
                         channel_domains: list[str] | None = None):
    """The average life image: observed share, cases, controls, cases minus controls."""
    import matplotlib
    from matplotlib.figure import Figure

    images = condition_report["images"]
    dataset = condition_report["dataset"]
    ages = [50 + 2 * b for b in range(dataset["image_bins"])]
    cases, controls = _panel_values(images["cases"]), _panel_values(images["controls"])
    limit = np.nanmax(np.abs(np.concatenate([cases, controls]))) if np.isfinite(cases).any() else 1.0
    difference = _panel_values(images["difference"])
    spread = np.nanmax(np.abs(difference)) if np.isfinite(difference).any() else 1.0
    panels = [
        ("Share observed, everyone", _panel_values(images["observed_share"]), style["share_cmap"], (0.0, 1.0),
         "share of people"),
        (f"Cases (n = {dataset['cases']})", cases, style["diverging_cmap"], (-limit, limit), "SD from same-age peers"),
        (f"Controls (n = {dataset['controls']})", controls, style["diverging_cmap"], (-limit, limit),
         "SD from same-age peers"),
        ("Cases minus controls", difference, style["diverging_cmap"], (-spread, spread), "difference in SD"),
    ]
    domains = channel_domains or [n for n in ("upstream", "midstream", "downstream") for _ in range(len(channel_names) // 3)]
    boundaries = [i - 0.5 for i in range(1, len(domains)) if domains[i] != domains[i - 1]]
    width, height = style["figsize_average"]
    fig = Figure(figsize=(width, height * max(1.0, len(channel_names) / 12) ** 0.6), dpi=style["dpi"])
    axes = fig.subplots(1, len(panels), sharey=True)
    for i, (ax, (title, values, cmap_name, (vmin, vmax), label)) in enumerate(zip(axes, panels)):
        cmap = matplotlib.colormaps[cmap_name].with_extremes(bad=style["missing_color"])
        mesh = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest", cmap=cmap,
                         vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=style["label_fs"], loc="left")
        ax.set_xticks(range(0, len(ages), 2))
        ax.set_xticklabels([str(a) for a in ages[::2]], fontsize=style["tick_fs"])
        ax.set_xlabel("age at start of bin", fontsize=style["tick_fs"])
        if i == 0:
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels([n.replace("_", " ") for n in channel_names], fontsize=style["tick_fs"])
        for boundary in boundaries:
            ax.axhline(boundary, color=style["separator_color"], linewidth=1.0)
        bar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
        bar.set_label(label, fontsize=style["tick_fs"])
        bar.ax.tick_params(labelsize=style["tick_fs"])
    lo, hi = dataset["status_ages"]
    fig.suptitle(f"Average life image, ages 50-{lo}: {dataset['title']} age {lo}-{hi}", fontsize=style["title_fs"])
    fig.text(
        0.01, 0.005,
        "Each cell is standardised against everyone observed in that channel at that age, then averaged over the "
        f"group's observers. Grey: fewer than {condition_report['images']['min_cell']} observers (suppressed) or "
        "never asked.",
        fontsize=style["tick_fs"], color=style["roles"]["footer"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


# --- the kernel experiment's figures (D30) -------------------------------------------------
def kernel_auc_figure(condition_report: dict[str, Any], style: dict[str, Any], references: dict[str, float] | None = None,
                      title: str = "kernel arms"):
    """Each arm's test AUC per fold (dots) and mean (bar), arms in config order."""
    from matplotlib.figure import Figure

    folds = condition_report["folds"]
    arms = list(condition_report["metrics_mean"])
    fig = Figure(figsize=(style["figsize"][0], style["figsize"][1] * 0.5), dpi=style["dpi"])
    ax = fig.subplots()
    means = [condition_report["metrics_mean"][a]["auc"] for a in arms]
    ax.bar(range(len(arms)), means, color=style["colors"][0], alpha=0.35, width=0.6)
    for i, arm in enumerate(arms):
        ax.scatter([i] * len(folds), [f["metrics"][arm]["auc"] for f in folds], s=14, color=style["colors"][0], zorder=3)
    for k, (name, value) in enumerate((references or {}).items()):
        ax.axhline(value, color=style["colors"][3 - k % 2], linewidth=1, linestyle="--", label=f"{name} {value:.3f}")
    if references:
        ax.legend(fontsize=style["legend_fs"], loc="lower right")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=45, ha="right", fontsize=style["tick_fs"])
    lo = min(f["metrics"][a]["auc"] for f in folds for a in arms)
    ax.set_ylim(lo - 0.01, max(means + list((references or {}).values())) + 0.01)
    ax.set_ylabel("test AUC (dots: folds, bar: mean)", fontsize=style["label_fs"])
    ax.set_title(f"{condition_report['dataset']['title']} age {'-'.join(map(str, condition_report['dataset']['status_ages']))}: "
                 f"{title}", fontsize=style["title_fs"], loc="left")
    fig.tight_layout()
    return fig


def kernel_map_figure(condition_report: dict[str, Any], channel_names: list[str], style: dict[str, Any],
                      channel_domains: list[str], kind: str = "gradcam"):
    """One small cases-minus-controls map per arm: ``gradcam`` or ``eg_abs`` (|Expected Gradients|)."""
    from matplotlib.figure import Figure

    maps = condition_report["maps"]
    arms = [a for a in condition_report["metrics_mean"] if a in maps and kind in maps[a]]
    cols = 5
    rows = -(-len(arms) // cols)
    fig = Figure(figsize=(style["figsize"][0] * 1.2, 4.2 * rows), dpi=style["dpi"])
    axes = np.atleast_2d(fig.subplots(rows, cols))
    boundaries = [i - 0.5 for i in range(1, len(channel_domains)) if channel_domains[i] != channel_domains[i - 1]]
    ages = [50 + 2 * b for b in range(condition_report["dataset"]["image_bins"])]
    for ax in axes.ravel()[len(arms):]:
        ax.set_axis_off()
    for i, (ax, arm) in enumerate(zip(axes.ravel(), arms)):
        values = _panel_values(maps[arm][kind]["difference"])
        limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() and np.nanmax(np.abs(values)) > 0 else 1.0
        ax.imshow(values, aspect="auto", interpolation="nearest", cmap=style["diverging_cmap"], vmin=-limit, vmax=limit)
        auc = condition_report["metrics_mean"][arm]["auc"]
        ax.set_title(f"{arm}  (AUC {auc:.3f})", fontsize=style["label_fs"], loc="left")
        ax.set_xticks(range(0, len(ages), 3))
        ax.set_xticklabels([str(a) for a in ages[::3]], fontsize=style["tick_fs"])
        ax.set_yticks(range(len(channel_names)) if i % cols == 0 else [])
        if i % cols == 0:
            ax.set_yticklabels(channel_names, fontsize=style["tick_fs"] - 2)
        for boundary in boundaries:
            ax.axhline(boundary, color=style["separator_color"], linewidth=0.8)
    label = "Grad-CAM" if kind == "gradcam" else "|Expected Gradients|"
    fig.suptitle(f"{label}, cases minus controls, per arm (each panel on its own symmetric scale)",
                 fontsize=style["title_fs"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
