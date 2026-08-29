"""Seeds and provenance: making a run identifiable after the fact.

A dataset is identified by the tuple ``(master_seed, scenario, N, replicate)``. Using
``hash()`` or a counter for that would be reproducible only within one Python process
(``PYTHONHASHSEED`` is randomised per process), so a figure regenerated tomorrow would
quietly differ from the one in the report. The rule this file enforces: every seed is
derived deterministically from the identifying tuple by SHA-256, and every generated
artifact records the git hash of the code that produced it.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from utils import shared_paths as P


def stable_seed(master_seed: int, scenario: str, n: int, replicate: int) -> int:
    """Derive a platform-stable uint32 seed from a dataset's identifying tuple."""
    key = f"{master_seed}|{scenario}|{n}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def git_hash(project_root: Path | None = None) -> str:
    """Return the short git hash, or ``"unknown"`` outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root or P.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"
