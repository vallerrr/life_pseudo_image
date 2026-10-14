"""The path registry: every directory and shared file named once, in one place.

Before this module existed the project addressed the same directory two ways --
``outputs/reports/`` written by the diagnostics script and ``outputs/`` assumed by
the documentation -- and the drift was silent, because each spelling was correct on
its own screen. The rule this file now enforces: two modules referring to the same
location refer to the same constant, and every constant is anchored on the repo root
rather than on the working directory, so the project runs identically from the repo
root, from a notebook two levels down, or from a cluster scratch directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Anchored on this file, never on the cwd: src/utils/shared_paths.py -> parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

# --- settings -------------------------------------------------------------
UNIVERSAL_SETTINGS = PROJECT_ROOT / "universal_settings.yml"
# The DGP's structural parameters, deliberately kept beside the module that reads
# them (see doc/conventions.md) rather than in universal_settings.yml, which owns
# plotting style only.
DGP_CONFIG = SRC_DIR / "utils" / "shared_dgp_config.yaml"
# The pseudo-image CNN's architecture, search grid and run profiles; same reasoning.
PSEUDO_IMAGE_CNN_CONFIG = SRC_DIR / "utils" / "data_analysis_02_pseudo_image_cnn_config.yaml"

# --- data -----------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
HRS_DIR = DATA_DIR / "hrs"  # source pull; immutable, never edited in place
ANALYSIS_DATA_DIR = DATA_DIR / "analysis"
DGP_DIAGNOSTICS_DATA_DIR = ANALYSIS_DATA_DIR / "dgp_diagnostics"
# Model-input arrays, one .npz per (scenario, N, replicate); written by 00_data_creation
# and by any model job that asks for a dataset not yet on disk. Every model reads here.
MODEL_INPUT_DIR = ANALYSIS_DATA_DIR / "data_creation"
PSEUDO_IMAGE_CNN_DATA_DIR = ANALYSIS_DATA_DIR / "pseudo_image_cnn"

# --- output ---------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DGP_DIAGNOSTICS_FIGURE_DIR = FIGURE_DIR / "data_analysis" / "01_dgp_diagnostics"
LIFE_IMAGE_FIGURE_DIR = FIGURE_DIR / "data_creation" / "00_data_creation"
PSEUDO_IMAGE_CNN_FIGURE_DIR = FIGURE_DIR / "data_analysis" / "02_pseudo_image_cnn"

LOG_DIR = PROJECT_ROOT / "logs"

# Matplotlib writes a font cache on first import; point it inside the repo so a
# read-only or shared HOME on a cluster cannot fail the import.
MPL_CACHE_DIR = PROJECT_ROOT / "tmp" / "matplotlib"

# Directories that code writes into. Read-only locations are deliberately absent:
# ensure_dirs() must never conjure an empty data source that then reads as "no rows".
_WRITABLE_DIRS = (
    ANALYSIS_DATA_DIR,
    FIGURE_DIR,
    TABLE_DIR,
    DGP_DIAGNOSTICS_FIGURE_DIR,
    LIFE_IMAGE_FIGURE_DIR,
    MODEL_INPUT_DIR,
    PSEUDO_IMAGE_CNN_DATA_DIR,
    PSEUDO_IMAGE_CNN_FIGURE_DIR,
    LOG_DIR,
    MPL_CACHE_DIR,
)

_BOOTSTRAPPED = False


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path``, chdir to the repo root, create writable dirs.

    Idempotent: safe to call from every notebook cell, module, and test.
    """
    global _BOOTSTRAPPED
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    if not _BOOTSTRAPPED:
        ensure_dirs()
        _BOOTSTRAPPED = True
    return PROJECT_ROOT


def ensure_dirs() -> None:
    """Create every directory the project writes into."""
    for directory in _WRITABLE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """Render a path relative to the repo root, for logs and manifests."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
