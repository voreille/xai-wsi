"""DHMC LUAD slide metadata and predominant growth-pattern labels from the release table.

The DHMC lung adenocarcinoma release (``MetaData_Release_1.0.csv``) gives one predominant
pattern per slide (acinar, solid, lepidic, micropapillary, papillary) and the scan
resolution. It has no patient ids, so each slide is its own case (``case_id`` = slide id).
Labels are references for evaluation, never training targets.

    uv run python -m xaiwsi.preprocessing.dhmc.metadata \\
        --release-table data/external/dhmc-luad/MetaData_Release_1.0.csv \\
        --slides-dir data/external/dhmc-luad/DHMC_LUAD_corr \\
        --output-dir data/processed/dhmc-luad/metadata

Writes ``wsi_slides.csv`` (slide_id, case_id, svs_filename, svs_relpath, mpp, magnification;
the CPTAC/TCGA layout, so slide-level tools read it unchanged) and ``growth_patterns.csv``
(case_id, dominant_pattern, source). Slides missing from ``--slides-dir`` are an error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PATTERNS = {"acinar", "solid", "lepidic", "micropapillary", "papillary"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-table", type=Path, required=True)
    parser.add_argument("--slides-dir", type=Path, required=True, help="the WSIs that are tiled")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.release_table)
    filenames = table["File Name"].astype(str)
    missing = [f for f in filenames if not (args.slides_dir / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} slides of the table are not in {args.slides_dir}: {missing[:5]}")
    labels = table["Class"].str.strip().str.lower()
    unknown = set(labels) - PATTERNS
    if unknown:
        raise SystemExit(f"unknown pattern labels: {sorted(unknown)}")
    slide_ids = filenames.str.removesuffix(".tif")
    if slide_ids.duplicated().any():
        raise SystemExit("duplicated slide ids in the release table")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "slide_id": slide_ids,
            "case_id": slide_ids,
            "svs_filename": filenames,
            "svs_relpath": filenames,
            "mpp": table["Microns Per Pixel"],
            "magnification": table["Magnification"],
        }
    ).to_csv(args.output_dir / "wsi_slides.csv", index=False)
    pd.DataFrame(
        {"case_id": slide_ids, "dominant_pattern": labels, "source": f"DHMC {args.release_table.name}"}
    ).to_csv(args.output_dir / "growth_patterns.csv", index=False)
    print(f"{len(table)} slides; patterns: {labels.value_counts().to_dict()}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
