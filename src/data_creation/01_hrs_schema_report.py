"""Data creation 01 - the RAND HRS file's structure, as an aggregate report.

Run by the user, never by an assistant (HRS is blind, D07, D22). Reads the header and
the columns the ingestion uses (plus any ``--extra`` stems), and writes one JSON report
under ``output/reports/blind/``. Send that file; it holds no rows.

Run:
    python src/data_creation/01_hrs_schema_report.py
    python src/data_creation/01_hrs_schema_report.py --extra "r{w}cesd" "r{w}shlt"
    python src/data_creation/01_hrs_schema_report.py --rand-file tests/fixtures/randhrs_fixture.dta
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap (conventions §2) -------------------------------------------
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P  # noqa: E402

P.bootstrap()

import argparse  # noqa: E402

from utils.data_creation_01_hrs_schema_report import schema_report  # noqa: E402
from utils.shared_hrs_chronic import chronic_columns  # noqa: E402
from utils.shared_hrs_ingest import ALL_WAVES  # noqa: E402
from utils.shared_secure_report import SecureReport, guarded  # noqa: E402

NOTEBOOK = "01_hrs_schema_report"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rand-file", type=Path, default=P.RAND_HRS_FILE)
    parser.add_argument("--extra", nargs="*", default=[], help='more stems, e.g. "r{w}cesd" or "raeduc"')
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true", help="print the traceback on failure (do not paste it)")
    args = parser.parse_args()

    columns = chronic_columns()
    for stem in args.extra:
        names = [stem.format(w=w) for w in ALL_WAVES] if "{w}" in stem else [stem]
        columns += [c.lower() for c in names if c.lower() not in columns]

    report = SecureReport(NOTEBOOK, source="hrs", out_dir=args.report_dir)
    with guarded(report, debug=args.debug):
        schema_report(args.rand_file, columns, report)


if __name__ == "__main__":
    main()
