"""Slide -> specimen (tissue block) table from the TCIA CPTAC pathology table.

CPTAC slides and RNA samples are both cut from named specimens: slide ``C3L-03262-21`` is
from specimen ``C3L-03262-01``, which is also the GDC ``sample_id`` of the RNA aliquot taken
from that block. With this table, ``CohortSpec(pairing="specimen")`` pairs an RNA sample with
a slide of its own block instead of the first slide of the case.

    uv run python -m xaiwsi.preprocessing.cptac.slide_specimens \\
        --table data/raw/cptac-luad-clinical/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv \\
        --output data/processed/cptac-luad/metadata/slide_specimens.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def slide_specimens(table: pd.DataFrame) -> pd.DataFrame:
    out = table.rename(
        columns={
            "Slide_ID": "slide_id",
            "Case_ID": "case_id",
            "Specimen_ID": "sample_id",
            "Specimen_Type": "specimen_type",
        }
    )[["slide_id", "case_id", "sample_id", "specimen_type"]]
    if out["slide_id"].duplicated().any():
        dups = out.loc[out["slide_id"].duplicated(), "slide_id"].tolist()
        raise SystemExit(f"Slides listed twice in the TCIA table: {dups[:5]}")
    return out.sort_values("slide_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True, help="TCIA_CPTAC_*_Pathology_Data_Table.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out = slide_specimens(pd.read_csv(args.table))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    counts = out["specimen_type"].value_counts().to_dict()
    print(f"{len(out)} slides, {out['sample_id'].nunique()} specimens {counts} -> {args.output}")


if __name__ == "__main__":
    main()
