"""Renders one DGP diagnostic page. Owned by src/data_analysis/01_dgp_diagnostics.py.

Kept out of the entry point because the page is fourteen axes of layout arithmetic and
the entry point should stay readable as a sequence of steps. Every colour, size, and
font here comes from the resolved style dict; every target line comes from the
simulation config, so the dotted "target" on a plot cannot disagree with the target
the calibrator actually solved for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from utils.shared_dgp import SCENARIO_DISPLAY_NAMES, DGPResult, SimulationConfig

FOOTER = "Ages 50-90 represented as 20 two-year bins | First deliverable: DGP only"


def cumulative_curves(result: DGPResult) -> tuple[np.ndarray, np.ndarray]:
    """Empirical cumulative incidence of disease and of death, by bin."""
    bins = np.arange(1, result.w_true.shape[1] + 1)
    disease = np.array(
        [np.mean((result.event_type == 1) & (result.event_time <= t)) for t in bins]
    )
    death = np.array(
        [np.mean((result.event_type == 2) & (result.event_time <= t)) for t in bins]
    )
    return disease, death


def _bin_ticks(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(0, n_bins, 2), np.arange(1, n_bins + 1, 2)


def build_figure(
    result: DGPResult,
    cfg: SimulationConfig,
    runtime: float,
    style: dict[str, Any],
) -> Figure:
    """Compose the one-page diagnostic figure for a single scenario."""
    roles = style["roles"]
    n_bins = result.w_true.shape[1]
    bins = np.arange(1, n_bins + 1)
    disease, death = cumulative_curves(result)
    attrition = np.array([np.mean(result.dropout_time <= t) for t in bins])
    xticks, xticklabels = _bin_ticks(n_bins)

    fig = Figure(figsize=style["figsize"])
    grid = fig.add_gridspec(
        2, 3, left=0.06, right=0.96, bottom=0.12, top=0.88, wspace=0.36, hspace=0.38
    )
    fig.suptitle(
        f"DGP diagnostic - {SCENARIO_DISPLAY_NAMES[result.metadata['scenario']]}",
        fontsize=style["title_fs"],
        fontweight="bold",
        y=0.96,
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(bins, disease, color=roles["disease"], label="Disease")
    ax.plot(bins, death, color=roles["death"], label="Death")
    ax.axhline(cfg.target_disease, color=roles["disease"], linestyle=":", alpha=0.7)
    ax.axhline(cfg.target_death, color=roles["death"], linestyle=":", alpha=0.7)
    ax.set(title="Cumulative incidence", xlabel="Bin", ylabel="Proportion")
    ax.set_ylim(0, max(cfg.target_disease, cfg.target_death) * 1.2)
    ax.legend(frameon=False, fontsize=style["legend_fs"])

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(bins, attrition, color=roles["attrition"])
    ax.axhline(cfg.attrition_target, color=roles["attrition"], linestyle=":", alpha=0.7)
    ax.set(title="Informative attrition", xlabel="Bin", ylabel="Proportion")
    ax.set_ylim(0, cfg.attrition_target * 1.2)

    ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow(
        result.M.mean(axis=0), aspect="auto", vmin=0, vmax=1, cmap=style["missing_cmap"]
    )
    ax.set(
        title="Observation rate by channel/bin",
        xlabel="Bin",
        ylabel="Channel",
        xticks=xticks,
        xticklabels=xticklabels,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 0])
    for person, alpha in zip(range(6), np.linspace(0.45, 0.95, 6)):
        ax.plot(
            bins,
            result.latent[person, 0, :],
            color=roles["disease"],
            alpha=alpha,
            linewidth=style["trajectory_linewidth"],
        )
    ax.set(title="Example upstream trajectories", xlabel="Bin", ylabel="Latent U")

    ax = fig.add_subplot(grid[1, 1])
    image = ax.imshow(
        result.w_true,
        aspect="auto",
        cmap=style["truth_cmap"],
        vmin=0,
        vmax=max(float(result.w_true.max()), 1e-3),
    )
    ax.set(
        title="Ground-truth structural support",
        xlabel="Bin",
        ylabel="Channel",
        xticks=xticks,
        xticklabels=xticklabels,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    metadata = result.metadata
    lines = [
        "Calibration and checks",
        "",
        f"N: {metadata['N']:,}",
        f"Seed: {metadata['seed']}",
        f"Disease: {metadata['realized_disease_incidence']:.3f} "
        f"(target {cfg.target_disease:.3f})",
        f"Death: {metadata['realized_mortality']:.3f} (target {cfg.target_death:.3f})",
        f"Attrition: {metadata['realized_attrition']:.3f} "
        f"(target {cfg.attrition_target:.3f})",
        f"Missing cells: {metadata['missing_fraction']:.3f}",
        f"Generation: {runtime:.2f} s",
        "",
        "Solid curves are realized empirical CIFs.",
        "Dotted lines mark pre-specified targets.",
        "Outcome follow-up continues after dropout.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=style["body_fs"],
        linespacing=1.35,
    )
    fig.text(
        0.5,
        0.025,
        FOOTER,
        ha="center",
        fontsize=style["footer_fs"],
        color=roles["footer"],
    )
    return fig


def write_report(
    result: DGPResult,
    cfg: SimulationConfig,
    runtime: float,
    style: dict[str, Any],
    destination: Path,
) -> None:
    """Render the diagnostic page for one scenario to a single-page PDF."""
    fig = build_figure(result, cfg, runtime, style)
    with PdfPages(destination) as pdf:
        pdf.savefig(fig, dpi=style["dpi"])
