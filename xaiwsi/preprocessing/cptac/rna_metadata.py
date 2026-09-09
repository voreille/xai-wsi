from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xaiwsi.preprocessing.gdc.rna import build_metadata_mapping


def validate_cptac_metadata(df: pd.DataFrame) -> None:
    required = ["case_id", "sample_id"]
    problems = [
        f"{int(df[column].isna().sum())} row(s) missing {column}"
        for column in required
        if column not in df.columns or df[column].isna().any()
    ]
    if problems:
        raise ValueError(
            "CPTAC RNA metadata validation failed:\n- "
            + "\n- ".join(problems)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CPTAC RNA metadata from a GDC manifest."
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
    validate_cptac_metadata(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("CPTAC RNA metadata")
    print("------------------")
    print(f"Files:              {df['file_id'].nunique()}")
    print(f"Cases:              {df['case_id'].nunique()}")
    print(f"Samples:            {df['sample_id'].nunique()}")
    print(f"Missing raw files:  {int((~df['raw_exists']).sum())}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()
