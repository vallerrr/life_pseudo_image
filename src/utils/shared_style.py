"""The style loader: one palette and one set of font sizes for the whole project.

Every figure used to read a ``STYLE = {...}`` dict declared in the module that drew
it, which is how palettes drift: nothing forces two scripts to agree, so eventually
they do not, and a restyle means hunting literals. The rule this file now enforces:
styling lives in ``universal_settings.yml`` and reaches a figure only through
``load_style()``. A hardcoded colour, font size, or figure size in plotting code is
a bug, not a shortcut.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

import yaml

from utils import shared_paths as P

# Keys whose value is a two-element list that matplotlib wants as a tuple.
_FIGSIZE_PREFIX = "figsize"


@lru_cache(maxsize=1)
def _raw_settings() -> dict[str, Any]:
    with P.UNIVERSAL_SETTINGS.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_colors(value: Any, palette: list[str]) -> Any:
    """An int indexes the palette; a string is a literal colour."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return palette[value % len(palette)]
    return value


def _resolve(style: dict[str, Any]) -> dict[str, Any]:
    palette = list(style.get("colors", []))
    resolved: dict[str, Any] = {}
    for key, value in style.items():
        if key.startswith(_FIGSIZE_PREFIX) and isinstance(value, list) and len(value) == 2:
            resolved[key] = tuple(value)
        elif (key == "roles" or key.endswith("_colors")) and isinstance(value, dict):
            resolved[key] = {k: _resolve_colors(v, palette) for k, v in value.items()}
        elif key.endswith("_colors") and isinstance(value, list):
            resolved[key] = [_resolve_colors(v, palette) for v in value]
        else:
            resolved[key] = value
    return resolved


def load_style(notebook: str | None = None) -> dict[str, Any]:
    """Return ``style.base`` overridden by ``style.notebooks.<notebook>``.

    ``notebook`` is the stem of the calling notebook or script, e.g.
    ``"01_dgp_diagnostics"``. An unknown name is an error rather than a silent
    fallback to base -- a typo that silently returns the wrong style is exactly the
    drift this module exists to stop.
    """
    settings = deepcopy(_raw_settings()["style"])
    style: dict[str, Any] = settings.get("base", {})
    if notebook is not None:
        sections = settings.get("notebooks") or {}
        if notebook not in sections:
            known = ", ".join(sorted(sections)) or "none"
            raise KeyError(
                f"universal_settings.yml has no style.notebooks.{notebook}; known: {known}"
            )
        style.update(sections[notebook] or {})
    return _resolve(style)


def savedir(style: dict[str, Any]) -> Any:
    """Absolute save directory for a resolved style, relative to the repo root."""
    return P.PROJECT_ROOT / style["savedir"]
