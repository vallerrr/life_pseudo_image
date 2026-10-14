"""Figures for analysis 02: attribution against truth, calibration, the input planes.

Owned by src/data_analysis/02_pseudo_image_cnn.ipynb. Reads only what the job wrote
(shard ``.npz`` files) and the model-input file; trains nothing. Every colour and size
comes from the resolved style dict (conventions §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from utils.shared_evaluation import STREAM_NAMES


def mean_maps(shard_files: list[Path]) -> dict[str, np.ndarray]:
    """Average each map over the shards (folds) that carry it; truth is identical in all."""
    collected: dict[str, list[np.ndarray]] = {}
    for path in shard_files:
        with np.load(path) as archive:
            for key in archive.files:
                if key != "calibration":
                    collected.setdefault(key, []).append(archive[key])
    return {key: np.mean(values, axis=0) for key, values in collected.items()}


def calibration_rows(shard_files: list[Path]) -> np.ndarray:
    tables = []
    for path in shard_files:
        with np.load(path) as archive:
            tables.append(archive["calibration"])
    return np.vstack(tables)


def summary_table(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Spec §5: median and IQR across replicates x folds, per scenario and N. Never a mean."""
    chosen = results[results["metric"].isin(metrics)]
    grouped = chosen.groupby(["scenario", "n", "metric"])["value"]
    table = pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
            "fits": grouped.count(),
        }
    )
    return table.round(3)


def _heatmap(fig: Figure, ax, values: np.ndarray, title: str, style: dict[str, Any], channels_per_stream: int):
    mesh = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=style["map_cmap"], vmin=0)
    T = values.shape[1]
    ax.set_title(title, fontsize=style["label_fs"], loc="left")
    ax.set_xticks(np.arange(0, T, 4))
    ax.set_xticklabels([str(t + 1) for t in range(0, T, 4)], fontsize=style["tick_fs"])
    ax.set_yticks([channels_per_stream * (s + 0.5) - 0.5 for s in range(3)])
    ax.set_yticklabels([name.capitalize() for name in STREAM_NAMES], fontsize=style["tick_fs"])
    for s in range(1, 3):
        ax.axhline(s * channels_per_stream - 0.5, color=style["separator_color"], linewidth=1.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.ax.tick_params(labelsize=style["tick_fs"], length=2)
    bar.outline.set_visible(False)


def attribution_figure(
    maps: dict[str, np.ndarray], scenario: str, channels_per_stream: int, style: dict[str, Any]
) -> Figure:
    """Truth, the trained model's map, and the two Adebayo controls, side by side.

    Each map is on its own colour scale: they are compared by *shape* (the recovery
    metrics are rank-based), and their magnitudes are not on a common unit.
    """
    panels = [
        ("w_true", "Ground truth  |w_true|"),
        ("A_abs", "Trained CNN  mean |EG|"),
        ("A_param_random", "Control: weights re-drawn"),
        ("A_label_random", "Control: labels shuffled"),
    ]
    panels = [(key, title) for key, title in panels if key in maps]
    fig = Figure(figsize=style["figsize"])
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for i, (ax, (key, title)) in enumerate(zip(axes, panels)):
        _heatmap(fig, ax, np.abs(maps[key]), title, style, channels_per_stream)
        ax.set_xlabel("Age bin", fontsize=style["label_fs"])
        if i:
            ax.set_yticklabels([])  # streams are named once, on the left
    fig.suptitle(
        f"Attribution against truth — {scenario}", fontsize=style["title_fs"], fontweight="bold"
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.82, wspace=0.35)
    return fig


def calibration_figure(rows: np.ndarray, style: dict[str, Any]) -> Figure:
    """Predicted vs observed disease CIF at T, one point per risk decile per fold."""
    fig = Figure(figsize=style["figsize_square"])
    ax = fig.subplots()
    limit = float(max(rows[:, 0].max(), rows[:, 1].max()) * 1.08)
    ax.plot([0, limit], [0, limit], color=style["separator_color"], linewidth=1.0)
    ax.scatter(rows[:, 0], rows[:, 1], s=18, color=style["roles"]["disease"], alpha=0.8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Predicted disease CIF at age 90", fontsize=style["label_fs"])
    ax.set_ylabel("Observed (Aalen–Johansen)", fontsize=style["label_fs"])
    ax.set_title("Calibration by risk decile", fontsize=style["title_fs"], loc="left")
    ax.tick_params(labelsize=style["tick_fs"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
