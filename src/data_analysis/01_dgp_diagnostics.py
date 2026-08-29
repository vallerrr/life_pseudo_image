"""Analysis 01 - DGP diagnostics.

Generates one dataset per scenario at the diagnostic sample size, writes a one-page
PDF per scenario, and records a manifest pinning the git hash, the master seed, and
the full simulation config against the realized calibration figures.

Run:  python src/data_analysis/01_dgp_diagnostics.py   (or `make diagnostics`)
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap (conventions §2) -------------------------------------------
# Anchor every path on the repo root regardless of where this is launched from.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P  # noqa: E402

P.bootstrap()

import json  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import yaml  # noqa: E402

from utils.data_analysis_01_dgp_diagnostics_report import write_report  # noqa: E402
from utils.shared_dgp import (  # noqa: E402
    MASTER_SEED,
    SCENARIOS,
    generate_dataset,
    load_config,
)
from utils.shared_reproducibility import git_hash  # noqa: E402
from utils.shared_style import load_style  # noqa: E402

STYLE = load_style("01_dgp_diagnostics")

NOTEBOOK = "01_dgp_diagnostics"
DIAGNOSTIC_N = 10_000
DIAGNOSTIC_REPLICATE = 0


def main() -> None:
    save_date = date.today().strftime("%Y%m%d")
    figure_dir = P.DGP_DIAGNOSTICS_FIGURE_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)
    P.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    manifest: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        start = time.perf_counter()
        result = generate_dataset(
            scenario,
            n=DIAGNOSTIC_N,
            replicate=DIAGNOSTIC_REPLICATE,
            config=cfg,
            master_seed=MASTER_SEED,
        )
        runtime = time.perf_counter() - start

        destination = figure_dir / f"dgp_{scenario}_{save_date}.pdf"
        write_report(result, cfg, runtime, STYLE, destination)

        manifest.append(
            {
                **result.metadata,
                "generation_seconds": runtime,
                "report": P.relative(destination),
            }
        )
        print(
            f"{scenario}: disease={result.metadata['realized_disease_incidence']:.3f}, "
            f"death={result.metadata['realized_mortality']:.3f}, "
            f"attrition={result.metadata['realized_attrition']:.3f}, "
            f"runtime={runtime:.2f}s"
        )

    payload = {
        "git_hash": git_hash(),
        "master_seed": MASTER_SEED,
        "config": yaml.safe_load(P.DGP_CONFIG.read_text(encoding="utf-8")),
        "runs": manifest,
    }
    manifest_path = P.TABLE_DIR / f"dgp_manifest_{save_date}.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {P.relative(manifest_path)}")


if __name__ == "__main__":
    main()
