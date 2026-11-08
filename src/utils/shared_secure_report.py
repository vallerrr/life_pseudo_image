"""Run and report: the only way a result from a blind source reaches an assistant.

HRS is ``blind`` (D07): no assistant reads its rows, and the user does not share
them. The assistant writes the code, the user runs it, and the run writes one JSON
report under ``P.SECURE_REPORT_DIR`` that the user hands back. This module is how
such a script writes that report, and it enforces the rules in
``doc/conventions.md`` ("Blind sources: run and report", D22):

* **Aggregates only.** Counts, means, quantiles and averaged maps; never a row, an id
  or a per-person array. The API has no call that takes a person-level table.
* **Small cells suppressed.** A count from 1 to ``min_cell - 1`` is written as
  ``"<11"``, and a summary or map cell over fewer than ``min_cell`` people is ``null``.
* **Errors leak nothing.** :func:`guarded` records the exception's type, file, line
  and function -- never its message or traceback, which can embed values -- and
  writes the report anyway, so a failed run still says where it failed.

A report is JSON with plain Python numbers, readable by the assistant and by eye.
"""

from __future__ import annotations

import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from utils import shared_paths as P
from utils.shared_reproducibility import git_hash

MIN_CELL = 11
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class SecureReport:
    """An aggregate report, built section by section and written once."""

    def __init__(self, name: str, source: str, out_dir: Path | None = None, min_cell: int = MIN_CELL) -> None:
        self.name, self.min_cell = name, min_cell
        self.out_dir = Path(out_dir) if out_dir is not None else P.SECURE_REPORT_DIR
        self.body: dict[str, Any] = {
            "report": name,
            "source": source,
            "created": datetime.now().isoformat(timespec="seconds"),
            "git_hash": git_hash(),
            "min_cell": min_cell,
        }

    # --- what may go in ------------------------------------------------------
    def count(self, n: int) -> int | str:
        """A head count, suppressed when it is small but not zero."""
        n = int(n)
        return n if n == 0 or n >= self.min_cell else f"<{self.min_cell}"

    def counts(self, mapping: dict[Any, int]) -> dict[str, int | str]:
        return {str(k): self.count(v) for k, v in mapping.items()}

    def summary(self, values: np.ndarray) -> dict[str, Any]:
        """n, mean, SD and the 5/25/50/75/95th percentiles of the finite values.

        No min or max: an extreme is one person's value.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < self.min_cell:
            return {"n": self.count(values.size)}
        quantiles = np.quantile(values, QUANTILES)
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "sd": float(values.std()),
            **{f"p{int(q * 100)}": float(v) for q, v in zip(QUANTILES, quantiles)},
        }

    def mean_map(self, total: np.ndarray, count: np.ndarray) -> list:
        """A per-cell mean from per-cell totals and head counts; thin cells are null."""
        total, count = np.asarray(total, dtype=float), np.asarray(count)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count >= self.min_cell, total / np.maximum(count, 1), np.nan)
        return _plain(mean)

    def add(self, path: str, value: Any) -> None:
        """Put ``value`` at ``a/b/c`` in the report, creating sections as needed."""
        *heads, last = path.split("/")
        node = self.body
        for head in heads:
            node = node.setdefault(head, {})
        node[last] = _plain(value)

    # --- out --------------------------------------------------------------------
    def write(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = self.out_dir / f"{self.name}_{stamp}.json"
        destination.write_text(json.dumps(self.body, indent=1, allow_nan=False))
        return destination


def _plain(value: Any) -> Any:
    """numpy -> Python, NaN -> None, recursively, so the report is strict JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def error_location(error: BaseException) -> dict[str, Any]:
    """Type, file, line and function of the innermost frame in this project's code."""
    frames = traceback.extract_tb(error.__traceback__)
    ours = [f for f in frames if str(P.PROJECT_ROOT) in f.filename] or frames
    frame = ours[-1] if ours else None
    return {
        "type": type(error).__name__,
        "file": P.relative(Path(frame.filename)) if frame else None,
        "line": frame.lineno if frame else None,
        "function": frame.name if frame else None,
    }


@contextmanager
def guarded(report: SecureReport, debug: bool = False) -> Iterator[SecureReport]:
    """Run a block on blind data; on failure record where, write, and exit 1.

    The traceback is not printed unless ``debug``: the terminal is often pasted
    whole, and a traceback can carry the offending values.
    """
    try:
        yield report
    except Exception as error:  # noqa: BLE001 -- every failure is reported the same way
        report.add("error", error_location(error))
        path = report.write()
        where = report.body["error"]
        print(f"FAILED: {where['type']} at {where['file']}:{where['line']} ({where['function']}).")
        print(f"Report written to {P.relative(path)} -- send that file, not the terminal output.")
        if debug:
            raise
        sys.exit(1)
    else:
        path = report.write()
        print(f"Report written to {P.relative(path)} -- this file is the one to send.")
