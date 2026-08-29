"""Put `src` on sys.path so tests import modules exactly as a notebook does."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import shared_paths as P  # noqa: E402

P.bootstrap()
