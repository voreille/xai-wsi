from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from xaiwsi.preprocessing.gdc.clinical import stage_group, summarize_clinical

# TCGA Clinical Data Resource (Liu et al., Cell 2018), Supplemental Table S1.
TCGA_CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
TCGA_CDR_SHEET = "TCGA-CDR"
ENDPOINTS = ["OS", "DSS", "DFI", "PFI"]


def download_cdr(path: Path, timeout: float = 300.0) -> None:
    response = requests.get(TCGA_CDR_URL, timeout=timeout)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)


def load_cdr(path: str | Path, cancer_type: str) -> pd.DataFrame:
    """Curated TCGA-CDR endpoints for one TCGA cancer type (``type`` column).

    Keeps OS, DSS, DFI and PFI as ``<endpoint>_time`` (days) and ``<endpoint>_event``
    (0/1), as recommended by the CDR paper. Redacted patients are dropped.
    """
    cdr = pd.read_excel(path, sheet_name=TCGA_CDR_SHEET, na_values=["#N/A"])
    cdr = cdr[(cdr["type"] == cancer_type) & cdr["Redaction"].isna()]
    if cdr.empty:
        raise ValueError(f"no TCGA-CDR rows with type == {cancer_type!r}")

    stage = cdr["ajcc_pathologic_tumor_stage"].where(
        cdr["ajcc_pathologic_tumor_stage"].str.startswith("Stage", na=False)
    )
    df = pd.DataFrame(
        {
            "case_id": cdr["bcr_patient_barcode"],
            "age": pd.to_numeric(
                cdr["age_at_initial_pathologic_diagnosis"], errors="coerce"
            ),
            "sex": cdr["gender"].str.lower(),
            "stage": stage,
            "stage_group": stage.map(stage_group),
            "histological_type": cdr["histological_type"],
            "vital_status": cdr["vital_status"],
        }
    )
    for endpoint in ENDPOINTS:
        df[f"{endpoint.lower()}_time"] = pd.to_numeric(
            cdr[f"{endpoint}.time"], errors="coerce"
        )
        df[f"{endpoint.lower()}_event"] = pd.to_numeric(cdr[endpoint], errors="coerce")
    df["source"] = "TCGA-CDR (Liu et al. 2018)"
    return df.replace({np.inf: np.nan}).sort_values("case_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TCGA case-level clinical/survival data from TCGA-CDR."
    )
    parser.add_argument(
        "--cdr",
        type=Path,
        required=True,
        help="TCGA-CDR-SupplementalTableS1.xlsx; downloaded from GDC if missing",
    )
    parser.add_argument(
        "--cancer-type", required=True, help="TCGA-CDR type, e.g. LUAD"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.cdr.exists():
        print(f"Downloading TCGA-CDR to {args.cdr}")
        download_cdr(args.cdr)

    df = load_cdr(args.cdr, args.cancer_type)
    if df["case_id"].duplicated().any():
        raise ValueError("duplicate case_id in TCGA-CDR")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summarize_clinical(df, f"TCGA-{args.cancer_type} clinical")
    print(f"PFI events:         {int(df['pfi_event'].sum())}")
    print(f"Output:             {args.output}")


if __name__ == "__main__":
    main()
