from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import requests

from xaiwsi.preprocessing.gdc.rna import chunks

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

GDC_CLINICAL_FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "disease_type",
    "primary_site",
    "demographic.gender",
    "demographic.sex_at_birth",  # replaces gender in newer GDC releases
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.primary_diagnosis",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.tumor_grade",
    "diagnoses.days_to_last_follow_up",
    "follow_ups.days_to_follow_up",
]

def stage_group(stage: object) -> str | float:
    """Collapse an AJCC stage ("Stage IIIA") to its group ("III"); NaN otherwise."""
    if not isinstance(stage, str):
        return np.nan
    match = re.fullmatch(r"\s*Stage\s+(IV|I{1,3})[A-C]?\d?\s*", stage)
    return match.group(1) if match else np.nan


def query_cases(
    project_id: str,
    submitter_ids: Sequence[str],
    *,
    batch_size: int = 200,
    timeout: float = 60.0,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve clinical fields for GDC cases by submitter ID (e.g. ``C3L-00001``).

    Cases not found in ``project_id`` raise, or are reported and skipped with
    ``allow_missing``.
    """
    submitter_ids = list(dict.fromkeys(submitter_ids))
    session = requests.Session()
    hits: list[dict[str, Any]] = []

    for batch in chunks(submitter_ids, batch_size):
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {
                        "op": "in",
                        "content": {
                            "field": "project.project_id",
                            "value": [project_id],
                        },
                    },
                    {
                        "op": "in",
                        "content": {"field": "submitter_id", "value": batch},
                    },
                ],
            },
            "format": "JSON",
            "fields": ",".join(GDC_CLINICAL_FIELDS),
            "size": len(batch),
        }
        response = session.post(GDC_CASES_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        hits.extend(response.json()["data"]["hits"])

    returned = {hit["submitter_id"] for hit in hits}
    missing = sorted(set(submitter_ids) - returned)
    if missing:
        message = f"GDC did not return {len(missing)} {project_id} case(s): " + ", ".join(missing)
        if not allow_missing:
            raise RuntimeError(message)
        print(f"Warning: {message}")
    return hits


def _primary_diagnosis(case: dict[str, Any]) -> dict[str, Any]:
    diagnoses = case.get("diagnoses") or []
    primary = [d for d in diagnoses if d.get("diagnosis_is_primary_disease")]
    return (primary or diagnoses or [{}])[0]


def flatten_cases(hits: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """One row per case with overall survival derived from GDC clinical fields.

    ``os_event`` is 1 if ``vital_status`` is Dead, 0 if Alive. ``os_time`` is
    ``days_to_death`` for deaths and the latest follow-up (over all diagnoses and
    follow-ups) otherwise. Both are NaN when the status or the time is unknown.
    """
    rows = []
    for case in hits:
        demographic = case.get("demographic") or {}
        diagnosis = _primary_diagnosis(case)
        follow_up = [
            d.get("days_to_last_follow_up") for d in case.get("diagnoses") or []
        ] + [f.get("days_to_follow_up") for f in case.get("follow_ups") or []]
        follow_up = [days for days in follow_up if days is not None]

        vital_status = demographic.get("vital_status")
        days_to_death = demographic.get("days_to_death")
        last_follow_up = max(follow_up) if follow_up else np.nan
        if vital_status == "Dead":
            os_event = 1.0
            os_time = np.nan if days_to_death is None else float(days_to_death)
        elif vital_status == "Alive":
            os_event = 0.0
            os_time = float(last_follow_up)
        else:
            os_event = os_time = np.nan

        age_days = diagnosis.get("age_at_diagnosis")
        stage = diagnosis.get("ajcc_pathologic_stage")
        rows.append(
            {
                "case_id": case["submitter_id"],
                "case_uuid": case["case_id"],
                "project_id": (case.get("project") or {}).get("project_id"),
                "os_time": os_time,
                "os_event": os_event,
                "age": np.nan if age_days is None else age_days / 365.25,
                "sex": demographic.get("sex_at_birth") or demographic.get("gender"),
                "stage": stage,
                "stage_group": stage_group(stage),
                "tumor_grade": diagnosis.get("tumor_grade"),
                "primary_diagnosis": diagnosis.get("primary_diagnosis"),
                "vital_status": vital_status,
                "days_to_death": days_to_death,
                "days_to_last_follow_up": last_follow_up,
            }
        )
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def summarize_clinical(df: pd.DataFrame, title: str) -> None:
    usable = df.dropna(subset=["os_time", "os_event"])
    print(title)
    print("-" * len(title))
    print(f"Cases:              {len(df)}")
    print(f"Usable OS:          {len(usable)}")
    print(f"OS events:          {int(usable['os_event'].sum())}")
    print(f"OS time <= 0:       {int((usable['os_time'] <= 0).sum())}")
    print(f"Median OS time (d): {usable['os_time'].median():.0f}")
    print(f"Stage groups:       {df['stage_group'].value_counts().sort_index().to_dict()}")
