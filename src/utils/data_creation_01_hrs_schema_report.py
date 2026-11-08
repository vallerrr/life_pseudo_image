"""The RAND file's structure as an aggregate report: what the assistant needs, no rows.

Owned by src/data_creation/01_hrs_schema_report.py and run by the user (D22). Under
``blind`` the assistant writes ingestion code from the codebook; when the codebook is
not enough -- a storage type, the case of the names, which waves a variable really
has, which missing codes really occur -- this report answers from the file itself:

* **Header.** Row and column counts; every variable name collapsed to its wave stem
  (``r{w}bmi``) with the waves present, and the first wave's label.
* **Requested columns.** Storage type, variable label, value labels, valid count,
  counts of each Stata missing code, and for numeric columns the summary
  ``SecureReport`` allows (n, mean, SD, 5/25/50/75/95th percentiles -- never min or
  max); for a column with at most ``max_levels`` distinct values, the count per value.

Every count goes through ``SecureReport.count``, so a cell under 11 people is "<11".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.shared_hrs_ingest import read_rand
from utils.shared_secure_report import SecureReport

_WAVE_NAME = re.compile(r"^([a-z]+?)(\d{1,2})([a-z_].*)$")
# Identifiers: a percentile of an id column *is* someone's id, so these get counts only.
ID_COLUMNS = frozenset({"hhidpn", "hhid", "pn"})


def stem_of(name: str) -> tuple[str, int | None]:
    """``r12bmi`` -> (``r{w}bmi``, 12); a name with no wave number maps to itself."""
    match = _WAVE_NAME.match(name.lower())
    if not match:
        return name.lower(), None
    head, wave, tail = match.groups()
    return f"{head}{{w}}{tail}", int(wave)


def header(path: Path) -> dict[str, Any]:
    """Names, labels, storage types and value-label sets, from the file header only."""
    with pd.read_stata(path, iterator=True) as reader:
        variable_labels = reader.variable_labels()
        value_label_sets = reader.value_labels()
        names = list(variable_labels)
        # Storage types and label-set names are not public API in pandas; report what
        # is there rather than fail on a pandas upgrade.
        types = dict(zip(names, getattr(reader, "_typlist", []) or []))
        label_names = dict(zip(names, getattr(reader, "_lbllist", []) or []))
        n_rows = getattr(reader, "_nobs", None)
    return {
        "names": names,
        "variable_labels": variable_labels,
        "value_label_sets": value_label_sets,
        "types": types,
        "label_names": label_names,
        "n_rows": n_rows,
    }


def _storage(code: Any) -> str:
    """pandas' Stata type code to a Stata storage name."""
    names = {65530: "byte", 65529: "int", 65528: "long", 65527: "float", 65526: "double"}
    if isinstance(code, (int, np.integer)):
        return names.get(int(code), f"str{int(code)}" if int(code) < 2046 else str(code))
    return str(code)


def schema_report(path: Path, columns: list[str], report: SecureReport, max_levels: int = 20) -> None:
    head = header(path)
    lower = {n.lower(): n for n in head["names"]}
    report.add("file/name", Path(path).name)
    report.add("file/n_rows", report.count(head["n_rows"]) if head["n_rows"] is not None else None)
    report.add("file/n_columns", len(head["names"]))
    report.add("file/names_lowercase", all(n == n.lower() for n in head["names"]))

    stems: dict[str, dict[str, Any]] = {}
    for name in head["names"]:
        stem, wave = stem_of(name)
        entry = stems.setdefault(stem, {"waves": [], "label": head["variable_labels"].get(name, "")})
        if wave is not None:
            entry["waves"].append(wave)
    report.add("file/stems", {s: {"waves": sorted(e["waves"]), "label": e["label"]} for s, e in sorted(stems.items())})

    present = [c for c in columns if c in lower]
    report.add("requested/absent", [c for c in columns if c not in lower])
    frame = read_rand(path, columns=present)
    for column in present:
        name = lower[column]
        values = frame.values[column].to_numpy()
        codes = frame.codes[column].to_numpy()
        entry: dict[str, Any] = {
            "storage": _storage(head["types"].get(name)),
            "label": head["variable_labels"].get(name, ""),
            "n_valid": report.count(int(np.isfinite(values).sum())),
            "missing_codes": report.counts(pd.Series(codes[codes != ""]).value_counts().to_dict()),
        }
        label_set = head["label_names"].get(name)
        if label_set and label_set in head["value_label_sets"]:
            entry["value_labels"] = {str(k): v for k, v in head["value_label_sets"][label_set].items()}
        valid = values[np.isfinite(values)]
        levels = np.unique(valid)
        if column in ID_COLUMNS:
            entry["distinct"] = report.count(levels.size)
        elif 0 < levels.size <= max_levels:
            entry["levels"] = report.counts({_level(v): int((valid == v).sum()) for v in levels})
        else:
            entry["summary"] = report.summary(valid)
        report.add(f"requested/columns/{column}", entry)


def _level(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
