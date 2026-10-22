from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xaiwsi.preprocessing.gdc.clinical import (
    flatten_cases,
    query_cases,
    summarize_clinical,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CPTAC case-level clinical/survival data from the GDC API."
    )
    parser.add_argument(
        "--cases-from",
        type=Path,
        nargs="+",
        required=True,
        help="metadata CSV(s) with a case_id column (e.g. rna_samples.csv, "
        "wsi_slides.csv); CPTAC-3 spans several cancer types",
    )
    parser.add_argument("--project-id", default="CPTAC-3")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip cases absent from GDC (e.g. slide-only cases from TCIA)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    case_ids = sorted(
        set().union(*(pd.read_csv(path)["case_id"].dropna() for path in args.cases_from))
    )
    hits = query_cases(
        args.project_id,
        case_ids,
        batch_size=args.batch_size,
        timeout=args.timeout,
        allow_missing=args.allow_missing,
    )
    df = flatten_cases(hits)
    df["source"] = f"GDC {args.project_id}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summarize_clinical(df, "CPTAC clinical")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()
