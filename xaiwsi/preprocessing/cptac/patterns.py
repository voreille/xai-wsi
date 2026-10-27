"""Download a sample-level histology label (e.g. dominant growth pattern) from cBioPortal.

Default: ``DOMINANT_HISTOLOGICAL_SUBTYPE`` of the ``luad_cptac_2020`` study (Gillette et
al., Cell 2020; 110 CPTAC-LUAD tumours). Labels are references for interpretation and
evaluation, never training targets.

    uv run python -m xaiwsi.preprocessing.cptac.patterns \\
        --output data/processed/cptac-luad/metadata/growth_patterns.csv

Writes ``case_id,<label column>,source`` (one row per case; cBioPortal patient id = CPTAC
case id).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

API = "https://www.cbioportal.org/api"


def fetch_clinical_attribute(study: str, attribute: str, timeout: float) -> pd.DataFrame:
    response = requests.get(
        f"{API}/studies/{study}/clinical-data",
        params={"clinicalDataType": "SAMPLE", "attributeId": attribute, "projection": "SUMMARY"},
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise SystemExit(f"No {attribute} values in cBioPortal study {study}.")
    return pd.DataFrame(rows)[["patientId", "sampleId", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="luad_cptac_2020")
    parser.add_argument("--attribute", default="DOMINANT_HISTOLOGICAL_SUBTYPE")
    parser.add_argument("--column", default="dominant_pattern", help="label column name in the output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    data = fetch_clinical_attribute(args.study, args.attribute, args.timeout)
    per_case = data.groupby("patientId")["value"].agg(lambda v: sorted(set(v)))
    conflicting = per_case[per_case.map(len) > 1]
    if not conflicting.empty:
        raise SystemExit(f"Cases with conflicting labels: {conflicting.to_dict()}")

    out = pd.DataFrame(
        {
            "case_id": per_case.index,
            args.column: per_case.map(lambda v: v[0].strip().lower()).to_numpy(),
            "source": f"cBioPortal {args.study} {args.attribute}",
        }
    ).sort_values("case_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out[args.column].value_counts().to_string())
    print(f"{len(out)} cases -> {args.output}")


if __name__ == "__main__":
    main()
