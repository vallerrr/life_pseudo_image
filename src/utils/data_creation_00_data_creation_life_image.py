"""Renders the life image: one person's C x T pseudo-image beside their outcomes.

Owned by src/data_creation/00_data_creation.ipynb. Kept out of the notebook because
the page is nested grids, a secondary axis and a categorical status strip -- layout
arithmetic the notebook should call, not carry. Every colour, size and font comes
from the resolved style dict (conventions §6); the axis geometry comes from the
simulation config, so a figure cannot disagree with the grid it was drawn from.

Two presentational constants live here and nowhere in the DGP: the age a bin sits at,
and the calendar year a person's bin sits in. The DGP has no calendar time -- its
birth-cohort covariate is a standardised normal with no year attached -- so the year
axis is a *reading* of that covariate, not a quantity the simulation used. It is
declared here, once, so that every figure reads the same clock.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.image import imsave
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

from utils.shared_dgp import SCENARIO_DISPLAY_NAMES, DGPResult, SimulationConfig

# --- the clock (presentational; see the module docstring) -------------------
AGE_AT_FIRST_BIN = 50  # bin 1 opens on a 50th birthday
YEARS_PER_BIN = 2  # 20 bins therefore span ages 50-90
# The birth-cohort covariate S[:, 0] is standardised; these turn it into a year.
BIRTH_YEAR_MEAN = 1940
BIRTH_YEAR_SD = 5.0

# The social hierarchy, top to bottom. The names are the specification's (§1.1);
# how many channels sit in each stream comes from the config, never from here.
STREAM_NAMES = ("Upstream", "Midstream", "Downstream")
STREAM_GLOSS = (
    "structural / material",
    "psychosocial / relational",
    "behavioural / biological",
)
STREAM_INITIALS = ("U", "M", "D")
HIERARCHY_GLOSS = " · ".join(
    f"{name}: {gloss}" for name, gloss in zip(STREAM_NAMES, STREAM_GLOSS)
)

# Status strip codes, in the order their colours are listed in the style.
STATUS_AT_RISK, STATUS_ATTRITED, STATUS_DISEASE, STATUS_DEAD = range(4)
STATUS_LABELS = ("At risk, observed", "At risk, left the panel", "Disease", "Death")

FOOTER = (
    "Ages 50-90 as 20 two-year bins  |  the year axis reads the birth-cohort covariate "
    f"at {BIRTH_YEAR_MEAN} +/- {BIRTH_YEAR_SD:.0f} yr and is presentational only"
)


# --- geometry ---------------------------------------------------------------
def age_at(x: float | np.ndarray) -> np.ndarray:
    """Age at position ``x`` on the bin axis, where bin k spans [k, k+1)."""
    return AGE_AT_FIRST_BIN + YEARS_PER_BIN * np.asarray(x, dtype=float)


def bin_at_age(age: float | np.ndarray) -> np.ndarray:
    """Inverse of :func:`age_at`; the secondary axis needs the round trip."""
    return (np.asarray(age, dtype=float) - AGE_AT_FIRST_BIN) / YEARS_PER_BIN


def birth_years(result: DGPResult) -> np.ndarray:
    """Read each person's standardised birth cohort as a calendar year."""
    return np.rint(BIRTH_YEAR_MEAN + BIRTH_YEAR_SD * result.S[:, 0]).astype(int)


def stream_blocks(cfg: SimulationConfig) -> list[tuple[str, str, str, int, int]]:
    """(name, gloss, initial, first channel, one past last) per stream, top down."""
    width = cfg.channels_per_stream
    return [
        (STREAM_NAMES[s], STREAM_GLOSS[s], STREAM_INITIALS[s], s * width, (s + 1) * width)
        for s in range(len(STREAM_NAMES))
    ]


def channel_labels(cfg: SimulationConfig) -> list[str]:
    """Channel tick labels, e.g. U1..U4, M1..M4, D1..D4."""
    return [
        f"{initial}{i + 1}"
        for _, _, initial, start, stop in stream_blocks(cfg)
        for i in range(stop - start)
    ]


# --- what the image shows ---------------------------------------------------
# Two ways to colour a cell. Neither is more correct than the other; they answer
# different questions, so the colourbar always says which one is on screen.
DISPLAY_MODES = ("peer", "raw")
DISPLAY_LABELS = {
    "peer": "SD from peers of the same age",
    "raw": "Channel value (raw units)",
}
DISPLAY_TITLES = {"peer": "peer-standardised", "raw": "raw units"}


def display_values(result: DGPResult, mode: str = "peer") -> np.ndarray:
    """The array to colour, for display only. ``result.X`` is never modified.

    ``raw`` shows the observed channel value as the DGP emitted it. It is the
    honest picture of what a model is handed, and it is the one to look at when
    asking how big these numbers actually are -- but the three streams are not on a
    common scale. Each is AR(1) with an individual intercept, so the stationary
    variance grows down the hierarchy, and one colour limit therefore renders
    upstream as near-flat beside a saturated downstream. Worse, the between-person
    spread in the random intercept is large enough to dominate the limit, so most of
    one person's image sits near the midpoint.

    ``peer`` standardises every ``(channel, bin)`` cell across the population, so a
    cell reads as *where this person sits among peers of the same age*, in SD. That
    removes the level and the variance growth that make ``raw`` hard to read, and
    puts all three streams on one scale.

    A cell no one was asked (midstream in odd bins) has no population to centre on
    and stays NaN under ``peer``, which the colormap renders as "not observed" --
    the same as it was already rendered under ``raw``.
    """
    if mode not in DISPLAY_MODES:
        raise ValueError(f"mode must be one of {DISPLAY_MODES}, got {mode!r}")
    if mode == "raw":
        return result.X
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing cells
        mean = np.nanmean(result.X, axis=0, keepdims=True)
        sd = np.nanstd(result.X, axis=0, keepdims=True)
    return (result.X - mean) / np.where(sd > 1e-8, sd, np.nan)


def image_limit(values: np.ndarray, quantile: float = 0.99) -> float:
    """One symmetric colour limit for every person, so images stay comparable."""
    return float(np.nanquantile(np.abs(values), quantile))


# --- colour -----------------------------------------------------------------
def image_cmap(style: dict[str, Any]) -> LinearSegmentedColormap:
    """Diverging ramp for the signed pseudo-image: two hues, neutral midpoint."""
    cmap = LinearSegmentedColormap.from_list(
        "life_image", list(style["image_colors"]), N=256
    )
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))  # transparent: the hatch beneath shows through
    return cmap


# A cell nobody observed is hatched rather than filled. A flat grey is a colour the
# ramp itself passes through near zero, so "missing" and "average" would read alike.
MISSING_HATCH = "////"


def missing_patch(style: dict[str, Any]) -> Patch:
    """Legend handle for an unobserved cell, drawn the way the image draws one."""
    return Patch(
        facecolor="none",
        edgecolor=style["missing_color"],
        hatch=MISSING_HATCH,
        label="Cell not observed",
    )


# --- lives ------------------------------------------------------------------
def status_codes(result: DGPResult, person: int, n_bins: int) -> np.ndarray:
    """Per-bin life status for one person: at risk, out of panel, diseased, dead."""
    codes = np.full(n_bins, STATUS_AT_RISK, dtype=np.int8)
    bins = np.arange(1, n_bins + 1)
    codes[bins >= result.dropout_time[person]] = STATUS_ATTRITED
    event_bin, event_type = result.event_time[person], result.event_type[person]
    if event_type == 1:
        codes[bins >= event_bin] = STATUS_DISEASE
    elif event_type == 2:
        codes[bins >= event_bin] = STATUS_DEAD
    return codes


def exemplars(result: DGPResult, n_bins: int) -> dict[str, int]:
    """One person per life course worth showing, chosen deterministically.

    Picking the first qualifying index rather than sampling keeps the gallery stable:
    the same seed tuple always renders the same four lives. Each condition also asks
    for the event in mid-panel, because a life that ends in bin 1 has no life course
    left to look at.
    """
    midlife = (result.event_time >= n_bins // 3) & (result.event_time <= n_bins)
    mid_dropout = (result.dropout_time >= n_bins // 3) & (result.dropout_time <= n_bins)
    stayed = result.dropout_time > n_bins
    event_free = result.event_type == 0
    wanted = {
        "Disease onset": (result.event_type == 1) & midlife & stayed,
        "Death": (result.event_type == 2) & midlife,
        "Left the panel": event_free & mid_dropout,
        "Event-free, fully observed": event_free & stayed,
    }
    chosen: dict[str, int] = {}
    for label, mask in wanted.items():
        candidates = np.flatnonzero(mask)
        if candidates.size:
            chosen[label] = int(candidates[0])
    return chosen


def life_summary(result: DGPResult, person: int, n_bins: int) -> str:
    """One line naming how this life ended, in ages and years."""
    born = int(birth_years(result)[person])
    event_bin = int(result.event_time[person])
    event_type = int(result.event_type[person])
    if event_type == 0:
        ending = f"event-free through age {int(age_at(n_bins))}"
    else:
        name = "disease" if event_type == 1 else "death"
        age = int(age_at(event_bin - 1))
        ending = f"{name} at age {age} ({born + age})"
    dropout = int(result.dropout_time[person])
    if dropout <= n_bins:
        ending += f", last observed at age {int(age_at(dropout - 1))}"
    sex = "female" if result.S[person, 1] > 0 else "male"
    return f"born {born}, {sex} — {ending}"


# --- axes -------------------------------------------------------------------
def _despine(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


def _age_axis(ax: Axes, n_bins: int, style: dict[str, Any], label: bool) -> None:
    """The first x-axis: bins read as age, along the bottom."""
    ticks = np.arange(0, n_bins + 1, 2)
    ax.set_xlim(0, n_bins)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(a)}" for a in age_at(ticks)], fontsize=style["tick_fs"])
    ax.tick_params(axis="x", labelbottom=label, length=0)
    if label:
        ax.set_xlabel("Age (years)", fontsize=style["label_fs"])


def _year_axis(ax: Axes, born: int, style: dict[str, Any], label: bool) -> Any:
    """The second x-axis: the same bins read as calendar years, along the top.

    Calendar year is age plus a birth year, so this is the same axis in another
    unit -- never a second scale for a second quantity.
    """
    secondary = ax.secondary_xaxis(
        "top",
        functions=(lambda x: born + age_at(x), lambda y: bin_at_age(y - born)),
    )
    secondary.xaxis.set_major_locator(MultipleLocator(10))
    secondary.tick_params(labelsize=style["tick_fs"], colors=style["axis_color"], length=0)
    _despine(secondary)
    if label:
        secondary.set_xlabel(
            "Calendar year", fontsize=style["label_fs"], color=style["axis_color"]
        )
    return secondary


def _left_of(ax: Axes, points: float) -> Any:
    """x in axes fraction shifted ``points`` leftward, y in data: size-independent."""
    shift = ScaledTranslation(-points / 72.0, 0.0, ax.figure.dpi_scale_trans)
    return blended_transform_factory(ax.transAxes, ax.transData) + shift


def _hierarchy_axis(ax: Axes, cfg: SimulationConfig, style: dict[str, Any], compact: bool) -> None:
    """The y-axis: social factors bracketed and named by their stream.

    The bracket, the tick labels and the stream names are offset in points, not in
    axes fraction, so they clear each other at every figure width.
    """
    blocks = stream_blocks(cfg)
    colours = list(style["stream_colors"])
    trans = _left_of(ax, 5.0)

    for (_, _, _, start, stop), colour in zip(blocks, colours):
        ax.plot(
            [0.0, 0.0],
            [start + 0.08, stop - 0.08],
            transform=trans,
            clip_on=False,
            color=colour,
            linewidth=3.0,
            solid_capstyle="butt",
        )
    for _, _, _, start, _ in blocks[1:]:
        ax.axhline(start, color=style["separator_color"], linewidth=1.4)

    if compact:
        # Too little height for twelve labels: name the three streams instead.
        ax.set_yticks([(start + stop) / 2.0 for _, _, _, start, stop in blocks])
        ax.set_yticklabels(STREAM_NAMES, fontsize=style["tick_fs"])
        for tick, colour in zip(ax.get_yticklabels(), colours):
            tick.set_color(colour)
        ax.tick_params(axis="y", pad=12)  # clear the bracket bar
    else:
        ax.set_yticks(np.arange(cfg.C) + 0.5)
        ax.set_yticklabels(channel_labels(cfg), fontsize=style["tick_fs"])
        ax.tick_params(axis="y", pad=12)  # clear the bracket bar
        for (name, _, _, start, stop), colour in zip(blocks, colours):
            ax.text(
                0.0,
                (start + stop) / 2.0,
                name,
                transform=_left_of(ax, 40.0),
                rotation=90,
                ha="center",
                va="center",
                fontsize=style["annotation_fs"],
                color=colour,
            )
    ax.tick_params(length=0)
    _despine(ax)


def _draw_image(
    ax: Axes,
    values: np.ndarray,
    cfg: SimulationConfig,
    person: int,
    style: dict[str, Any],
    vlim: float,
    compact: bool = False,
) -> Any:
    """One person's C x T pseudo-image, hierarchy down, bins across."""
    n_bins = values.shape[2]
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=style["missing_color"],
            hatch=MISSING_HATCH,
            linewidth=0.0,
            zorder=-1,
        )
    )
    mesh = ax.imshow(
        np.ma.masked_invalid(values[person]),
        aspect="auto",
        interpolation="nearest",
        extent=(0, n_bins, cfg.C, 0),
        vmin=-vlim,
        vmax=vlim,
        cmap=image_cmap(style),
    )
    _hierarchy_axis(ax, cfg, style, compact)
    return mesh


def _mark_life_events(
    ax: Axes, result: DGPResult, person: int, style: dict[str, Any], annotate: bool
) -> None:
    """Dropout and the terminating event, drawn identically on every panel."""
    n_bins = result.X.shape[2]
    dropout = int(result.dropout_time[person])
    if dropout <= n_bins:
        ax.axvline(
            dropout - 1, color=style["separator_color"], linestyle="--", linewidth=1.3
        )
    event_bin = int(result.event_time[person])
    event_type = int(result.event_type[person])
    if event_type == 0:
        return
    colour = style["roles"]["disease" if event_type == 1 else "death"]
    ax.axvline(event_bin - 1, color=colour, linewidth=1.8)
    if annotate:
        ax.annotate(
            "Disease" if event_type == 1 else "Death",
            xy=(event_bin - 1, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=style["annotation_fs"],
            color=colour,
            fontweight="bold",
        )


def _section_label(fig: Figure, ax: Axes, text: str, style: dict[str, Any]) -> None:
    """Name a section in the left gutter, level with the axes it belongs to."""
    box = ax.get_position()
    fig.text(
        0.014,
        (box.y0 + box.y1) / 2.0,
        text,
        rotation=90,
        ha="left",
        va="center",
        fontsize=style["label_fs"],
        fontweight="bold",
    )


def _colorbar(fig: Figure, mesh: Any, label: str, style: dict[str, Any], **kwargs: Any) -> None:
    bar = fig.colorbar(mesh, **kwargs)
    bar.set_label(label, fontsize=style["annotation_fs"])
    bar.ax.tick_params(labelsize=style["tick_fs"], length=2)
    bar.outline.set_visible(False)


# --- the two figures --------------------------------------------------------
def build_life_image(
    result: DGPResult,
    cfg: SimulationConfig,
    person: int,
    style: dict[str, Any],
    values: np.ndarray | None = None,
    vlim: float | None = None,
    mode: str = "peer",
) -> Figure:
    """One life: the social-hierarchy image above, disease and death below."""
    n_bins = result.X.shape[2]
    values = display_values(result, mode) if values is None else values
    vlim = image_limit(values) if vlim is None else vlim
    born = int(birth_years(result)[person])
    bin_centres = np.arange(n_bins) + 0.5
    roles = style["roles"]

    fig = Figure(figsize=style["figsize"])
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=style["height_ratios"],
        left=0.14,
        right=0.90,
        bottom=0.16,
        top=0.80,
        hspace=0.16,
    )

    ax_image = fig.add_subplot(grid[0])
    mesh = _draw_image(ax_image, values, cfg, person, style, vlim)
    _year_axis(ax_image, born, style, label=True)
    _mark_life_events(ax_image, result, person, style, annotate=True)
    _colorbar(fig, mesh, DISPLAY_LABELS[mode], style, ax=ax_image, fraction=0.022, pad=0.015)

    ax_outcome = fig.add_subplot(grid[1], sharex=ax_image)
    # Hazards are conditional on still being at risk, so the curves stop at the event.
    at_risk = slice(0, min(int(result.event_time[person]), n_bins))
    for series, role in (
        (result.disease_probability, "disease"),
        (result.death_probability, "death"),
    ):
        ax_outcome.plot(
            bin_centres[at_risk],
            series[person][at_risk] * 100,
            color=roles[role],
            linewidth=2.0,
        )
    _mark_life_events(ax_outcome, result, person, style, annotate=False)
    ax_outcome.set_ylabel("Per-bin hazard (%)", fontsize=style["label_fs"])
    ax_outcome.set_ylim(bottom=0)
    ax_outcome.grid(axis="y", color=style["separator_color"], linewidth=0.6, alpha=0.5)
    ax_outcome.set_axisbelow(True)
    for side in ("top", "right"):
        ax_outcome.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_outcome.spines[side].set_color(style["separator_color"])
    ax_outcome.tick_params(labelsize=style["tick_fs"], length=2)

    _age_axis(ax_image, n_bins, style, label=False)
    _age_axis(ax_outcome, n_bins, style, label=True)
    _section_label(fig, ax_image, "Social factors, by hierarchy", style)
    _section_label(fig, ax_outcome, "Disease and death", style)

    ax_outcome.legend(
        handles=[
            Line2D([], [], color=roles["disease"], linewidth=2.0, label="Disease hazard"),
            Line2D([], [], color=roles["death"], linewidth=2.0, label="Death hazard"),
            Line2D(
                [],
                [],
                color=style["separator_color"],
                linestyle="--",
                linewidth=1.3,
                label="Left the panel",
            ),
            missing_patch(style),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.46),
        ncol=4,
        frameon=False,
        fontsize=style["legend_fs"],
    )

    scenario = SCENARIO_DISPLAY_NAMES[result.metadata["scenario"]]
    fig.suptitle(
        f"Life image — person {person} — {scenario} scenario, {DISPLAY_TITLES[mode]}",
        fontsize=style["title_fs"],
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.14,
        0.915,
        life_summary(result, person, n_bins),
        fontsize=style["body_fs"],
    )
    fig.text(
        0.14,
        0.878,
        HIERARCHY_GLOSS,
        fontsize=style["annotation_fs"],
        color=style["axis_color"],
    )
    fig.text(
        0.5, 0.025, FOOTER, ha="center", fontsize=style["footer_fs"], color=roles["footer"]
    )
    return fig


def build_life_image_gallery(
    result: DGPResult,
    cfg: SimulationConfig,
    people: dict[str, int],
    style: dict[str, Any],
    values: np.ndarray | None = None,
    vlim: float | None = None,
    mode: str = "peer",
) -> Figure:
    """Several lives stacked: each its own year axis, one shared age axis.

    Every panel is on one age axis, but each person's year axis starts at their own
    birth. Reading down the figure is therefore reading the same ages in different
    decades -- which is the whole reason the two axes are drawn separately.
    """
    n_bins = result.X.shape[2]
    values = display_values(result, mode) if values is None else values
    vlim = image_limit(values) if vlim is None else vlim
    labels = list(people)
    status_cmap = ListedColormap(list(style["status_colors"]))

    fig = Figure(figsize=style["figsize_gallery"])
    outer = fig.add_gridspec(
        len(labels), 1, left=0.14, right=0.88, bottom=0.11, top=0.855, hspace=0.62
    )

    mesh = None
    for row, label in enumerate(labels):
        person = people[label]
        born = int(birth_years(result)[person])
        last = row == len(labels) - 1
        inner = outer[row].subgridspec(2, 1, height_ratios=[cfg.C, 2.0], hspace=0.12)

        ax_image = fig.add_subplot(inner[0])
        mesh = _draw_image(ax_image, values, cfg, person, style, vlim, compact=True)
        _age_axis(ax_image, n_bins, style, label=False)
        _year_axis(ax_image, born, style, label=row == 0)
        _mark_life_events(ax_image, result, person, style, annotate=False)
        ax_image.text(
            0.0,
            1.42,
            f"{label} — {life_summary(result, person, n_bins)}",
            transform=ax_image.transAxes,
            fontsize=style["annotation_fs"],
            fontweight="bold",
        )

        ax_status = fig.add_subplot(inner[1])
        ax_status.imshow(
            status_codes(result, person, n_bins)[None, :],
            aspect="auto",
            interpolation="nearest",
            extent=(0, n_bins, 1, 0),
            cmap=status_cmap,
            vmin=-0.5,
            vmax=len(STATUS_LABELS) - 0.5,
        )
        ax_status.set_yticks([0.5])
        ax_status.set_yticklabels(["Life status"], fontsize=style["tick_fs"])
        ax_status.tick_params(length=0)
        _despine(ax_status)
        _age_axis(ax_status, n_bins, style, label=last)

    _colorbar(
        fig, mesh, DISPLAY_LABELS[mode], style, cax=fig.add_axes((0.905, 0.20, 0.011, 0.52))
    )

    handles = [
        Patch(facecolor=colour, label=label)
        for colour, label in zip(style["status_colors"], STATUS_LABELS)
    ] + [missing_patch(style)]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        frameon=False,
        fontsize=style["legend_fs"],
    )

    scenario = SCENARIO_DISPLAY_NAMES[result.metadata["scenario"]]
    fig.suptitle(
        f"{len(labels)} life images — {scenario} scenario, {DISPLAY_TITLES[mode]}",
        fontsize=style["title_fs"],
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.14,
        0.930,
        HIERARCHY_GLOSS,
        fontsize=style["annotation_fs"],
        color=style["axis_color"],
    )
    fig.text(
        0.5,
        0.018,
        FOOTER,
        ha="center",
        fontsize=style["footer_fs"],
        color=style["roles"]["footer"],
    )
    return fig


# --- the per-person export: pixels only --------------------------------------
# The export is the image itself -- no axes, labels, title or colourbar -- on one
# calendar-year grid shared by everyone, so the same column is the same year in
# every file. Rows, top to bottom: the social factors by stream, age, then the
# static covariates, all spanning the person's ages 50-90. Every value is in SD and
# shares one colour scale; a cell with no value (not observed, or a year outside
# this person's ages 50-90) is gray.
TRAIT_NAMES = ("Birth cohort", "Sex", "Frailty")  # columns of S, in order (spec §1.1)


def standardised_age(n_bins: int) -> np.ndarray:
    """Age 50, 51, ... 89 as SD over that window: one mapping shared by every person.

    It depends on the clock alone, never on who is in the data, so the same age is
    the same colour in every exported image.
    """
    ages = AGE_AT_FIRST_BIN + np.arange(n_bins * YEARS_PER_BIN)
    return (ages - ages.mean()) / ages.std()


def calendar_span(result: DGPResult, n_bins: int) -> tuple[int, int]:
    """[first, end) calendar years covering every person's panel, from the data."""
    born = birth_years(result)
    return int(born.min() + AGE_AT_FIRST_BIN), int(born.max() + age_at(n_bins))


def export_row_blocks(cfg: SimulationConfig, n_traits: int) -> list[tuple[str, int, int]]:
    """(name, first row, one past last) for every block of the exported image."""
    blocks = [(name, start, stop) for name, _, _, start, stop in stream_blocks(cfg)]
    blocks.append(("Age", cfg.C, cfg.C + 1))
    blocks.append(("Fixed traits", cfg.C + 1, cfg.C + 1 + n_traits))
    return blocks


def calendar_image(
    result: DGPResult,
    cfg: SimulationConfig,
    values: np.ndarray,
    person: int,
    span: tuple[int, int],
) -> np.ndarray:
    """One person's ``rows x years`` values on the shared calendar grid, NaN = gray.

    Each two-year bin fills two one-year columns, so no birth year is rounded onto
    a coarser grid. Age comes from :func:`standardised_age`; the static covariates
    are already standardised and are held constant across the same 50-90 window,
    so every row of a life starts and stops in the same column.
    """
    first, end = span
    n_traits = result.S.shape[1]
    image = np.full((cfg.C + 1 + n_traits, end - first), np.nan)

    born = int(birth_years(result)[person])
    columns = born + AGE_AT_FIRST_BIN - first + np.arange(cfg.T * YEARS_PER_BIN)
    image[: cfg.C, columns] = np.repeat(values[person], YEARS_PER_BIN, axis=1)
    image[cfg.C, columns] = standardised_age(cfg.T)
    image[cfg.C + 1 :, columns] = result.S[person][:, None]
    return image


def save_calendar_image(
    image: np.ndarray,
    cfg: SimulationConfig,
    destination: Path,
    style: dict[str, Any],
    vlim: float,
) -> Path:
    """Write the image pixel-exact: each cell a block, thin rules between blocks."""
    rgba = image_cmap(style)(Normalize(-vlim, vlim)(np.nan_to_num(image)))
    rgba[np.isnan(image)] = to_rgba(style["missing_color"])
    rgba = np.repeat(rgba, style["export_cell_height"], axis=0)
    rgba = np.repeat(rgba, style["export_year_width"], axis=1)

    rule = np.tile(to_rgba(style["separator_color"]), (style["export_rule_px"], rgba.shape[1], 1))
    starts = [start for _, start, _ in export_row_blocks(cfg, image.shape[0] - cfg.C - 1)[1:]]
    pieces = np.split(rgba, [start * style["export_cell_height"] for start in starts])
    stacked = [pieces[0]]
    for piece in pieces[1:]:
        stacked += [rule, piece]
    imsave(destination, np.vstack(stacked))
    return destination


def save_figure(fig: Figure, destination: Path, style: dict[str, Any]) -> Path:
    """Write one figure in every format the style asks for; return the last path."""
    written = destination
    for suffix in style["formats"]:
        written = destination.with_suffix(f".{suffix}")
        fig.savefig(written, dpi=style["dpi"])
    return written
