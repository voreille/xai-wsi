"""Download the GDC ``primary_diagnosis`` of every case of a project (e.g. mucinous LUAD in TCGA).

TCGA-CDR keeps only "Lung Adenocarcinoma"; GDC keeps the ICD-O morphology term of each
diagnosis (``Mucinous adenocarcinoma``, ``Bronchio-alveolar carcinoma, mucinous``, ...). A
case can have several diagnosis records (other primaries); all distinct terms are kept,
joined by ``; ``. Labels are references for interpretation and evaluation, never training
targets.

    uv run python -m xaiwsi.preprocessing.gdc.diagnoses --project TCGA-LUAD \\
        --output data/processed/tcga-luad/metadata/primary_diagnosis.csv

Writes ``case_id,primary_diagnosis,source``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

API = "https://api.gdc.cancer.gov/cases"


def fetch_diagnoses(project: str, timeout: float) -> pd.DataFrame:
    filters = {"op": "in", "content": {"field": "project.project_id", "value": [project]}}
    response = requests.get(
        API,
        params={
            "filters": json.dumps(filters),
            "fields": "submitter_id,diagnoses.primary_diagnosis",
            "size": 10000,
            "format": "JSON",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    hits = response.json()["data"]["hits"]
    if not hits:
        raise SystemExit(f"No cases for GDC project {project}.")
    rows = []
    for hit in hits:
        terms = {
            d.get("primary_diagnosis")
            for d in hit.get("diagnoses", [])
            if d.get("primary_diagnosis") and d["primary_diagnosis"] != "Not Reported"
        }
        rows.append({"case_id": hit["submitter_id"], "primary_diagnosis": "; ".join(sorted(terms))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="GDC project id, e.g. TCGA-LUAD")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = fetch_diagnoses(args.project, args.timeout)
    frame["source"] = f"GDC {args.project} diagnoses.primary_diagnosis"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("case_id").to_csv(args.output, index=False)
    print(f"{len(frame)} cases -> {args.output}")
    print(frame["primary_diagnosis"].replace("", "(none)").value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
