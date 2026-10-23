"""List the FFPE tumour-tissue slides of one CPTAC cohort, for `wsi-tile --include-slides`.

CPTAC folders also hold normal-tissue and frozen (OCT) slides. The slide table is the
TCIA pathology metadata (one row per slide with Slide_ID, Tumor, Embedding_Medium,
Specimen_Type), e.g. the CPTAC table of the TCIA pathology portal:
https://pathdb.cancerimagingarchive.net/system/files/collectionmetadata/202407/cptac_metadata_07-09-2024.csv

    uv run python scripts/filter_cptac_slide_id.py \\
        --metadata data/external/tcia/cptac_metadata_07-09-2024.csv --tumor LSCC \\
        --output data/processed/cptac-lusc/metadata/slide_filenames.json

The CPTAC-LUAD list (data/tiles/cptac/slide_filenames.json) was made from
/mnt/nas6/data/CPTAC/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv with the same filter, which
has no Tumor column (omit --tumor).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", type=Path, required=True, help="TCIA pathology table (.csv)")
    parser.add_argument("--tumor", default=None, help="value of the Tumor column, e.g. LSCC")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.metadata)
    keep = (table["Embedding_Medium"] == "FFPE") & (table["Specimen_Type"] == "tumor_tissue")
    if args.tumor is not None:
        keep &= table["Tumor"] == args.tumor
    slides = table[keep]

    filenames = [f"{slide_id}.svs" for slide_id in slides["Slide_ID"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filenames))
    print(f"{len(filenames)} slides from {slides['Case_ID'].nunique()} cases -> {args.output}")


if __name__ == "__main__":
    main()
