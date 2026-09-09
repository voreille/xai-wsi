from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xaiwsi.preprocessing.gdc.rna import build_metadata_mapping


def validate_tcga_metadata(df: pd.DataFrame) -> None:
    required = ["case_id", "sample_id", "aliquot_id"]
    problems = [
        f"{int(df[column].isna().sum())} row(s) missing {column}"
        for column in required
        if column not in df.columns or df[column].isna().any()
    ]
    if problems:
        raise ValueError(
            "TCGA RNA metadata validation failed:\n- "
            + "\n- ".join(problems)
        )

    invalid_cases = ~df["case_id"].astype(str).str.startswith("TCGA-")
    if invalid_cases.any():
        raise ValueError(
            "Non-TCGA case identifiers found:\n"
            + df.loc[invalid_cases, "case_id"].head(20).to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TCGA RNA metadata from a GDC manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    df = build_metadata_mapping(
        args.manifest,
        args.raw_root,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    validate_tcga_metadata(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    replicate_samples = int(
        (df.groupby("sample_id")["aliquot_id"].nunique() > 1).sum()
    )

    print("TCGA RNA metadata")
    print("-----------------")
    print(f"Files:                    {df['file_id'].nunique()}")
    print(f"Cases:                    {df['case_id'].nunique()}")
    print(f"Samples:                  {df['sample_id'].nunique()}")
    print(f"Aliquots:                 {df['aliquot_id'].nunique()}")
    print(f"Samples with >1 aliquot:  {replicate_samples}")
    print(f"Missing raw files:        {int((~df['raw_exists']).sum())}")
    print(f"Output:                   {args.output}")


if __name__ == "__main__":
    main()
