from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

GDC_RNA_FIELDS = [
    "file_id",
    "file_name",
    "data_type",
    "data_category",
    "experimental_strategy",
    "analysis.workflow_type",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "cases.samples.tumor_descriptor",
    "cases.samples.portions.portion_id",
    "cases.samples.portions.submitter_id",
    "cases.samples.portions.analytes.analyte_id",
    "cases.samples.portions.analytes.submitter_id",
    "cases.samples.portions.analytes.analyte_type",
    "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
]


def _chunks(
    items: Sequence[str],
    size: int,
) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def read_gdc_manifest(path: str | Path) -> pd.DataFrame:
    """Read a GDC download manifest.

    Standard GDC manifests contain:
        id, filename, md5, size, state

    Headerless manifests are also accepted.
    """
    path = Path(path)

    df = pd.read_csv(path, sep="\t", dtype=str)

    if "id" not in df.columns:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["id", "filename", "md5", "size", "state"],
            dtype=str,
        )

    required = {"id", "filename"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {path} is missing required columns: {sorted(missing)}"
        )

    rename = {
        "id": "file_id",
        "filename": "file_name",
        "md5": "md5sum",
        "size": "file_size",
        "state": "file_state",
    }

    return df.rename(columns=rename)


def query_gdc_files(
    file_ids: Sequence[str],
    *,
    batch_size: int = 200,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Query GDC metadata for a collection of file UUIDs."""
    file_ids = list(dict.fromkeys(file_ids))
    if not file_ids:
        return []

    session = requests.Session()
    hits: list[dict[str, Any]] = []

    for batch in _chunks(file_ids, batch_size):
        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "files.file_id",
                    "value": batch,
                },
            },
            "format": "JSON",
            "fields": ",".join(GDC_RNA_FIELDS),
            "size": len(batch),
        }

        response = session.post(
            GDC_FILES_ENDPOINT,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        body = response.json()
        batch_hits = body["data"]["hits"]
        hits.extend(batch_hits)

        returned = {hit["file_id"] for hit in batch_hits}
        missing = sorted(set(batch) - returned)
        if missing:
            raise RuntimeError(
                "GDC did not return metadata for the following file IDs:\n"
                + "\n".join(missing)
            )

    return hits


def _base_row(
    hit: dict[str, Any],
    case: dict[str, Any] | None,
    sample: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = hit.get("analysis") or {}
    project = (case or {}).get("project") or {}

    return {
        "file_id": hit.get("file_id"),
        "file_name": hit.get("file_name"),
        "data_type": hit.get("data_type"),
        "data_category": hit.get("data_category"),
        "experimental_strategy": hit.get("experimental_strategy"),
        "workflow_type": analysis.get("workflow_type"),
        "project_id": project.get("project_id"),
        "case_uuid": (case or {}).get("case_id"),
        "case_id": (case or {}).get("submitter_id"),
        "sample_uuid": (sample or {}).get("sample_id"),
        "sample_id": (sample or {}).get("submitter_id"),
        "sample_type": (sample or {}).get("sample_type"),
        "tissue_type": (sample or {}).get("tissue_type"),
        "tumor_descriptor": (sample or {}).get("tumor_descriptor"),
    }


def flatten_gdc_rna_metadata(
    hits: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten file -> case -> sample -> portion -> analyte -> aliquot.

    The output intentionally keeps one row per file/aliquot relationship.
    Replicate aliquots are biological/provenance information and are not
    resolved at the metadata stage.
    """
    rows: list[dict[str, Any]] = []

    for hit in hits:
        cases = hit.get("cases") or [None]

        for case in cases:
            samples = (case or {}).get("samples") or [None]

            for sample in samples:
                base = _base_row(hit, case, sample)
                portions = (sample or {}).get("portions") or [None]

                emitted = False

                for portion in portions:
                    analytes = (portion or {}).get("analytes") or [None]

                    for analyte in analytes:
                        aliquots = (analyte or {}).get("aliquots") or [None]

                        for aliquot in aliquots:
                            rows.append(
                                {
                                    **base,
                                    "portion_uuid": (portion or {}).get("portion_id"),
                                    "portion_id": (portion or {}).get("submitter_id"),
                                    "analyte_uuid": (analyte or {}).get("analyte_id"),
                                    "analyte_id": (analyte or {}).get("submitter_id"),
                                    "analyte_type": (analyte or {}).get("analyte_type"),
                                    "aliquot_uuid": (aliquot or {}).get("aliquot_id"),
                                    "aliquot_id": (aliquot or {}).get("submitter_id"),
                                }
                            )
                            emitted = True

                if not emitted:
                    rows.append(
                        {
                            **base,
                            "portion_uuid": None,
                            "portion_id": None,
                            "analyte_uuid": None,
                            "analyte_id": None,
                            "analyte_type": None,
                            "aliquot_uuid": None,
                            "aliquot_id": None,
                        }
                    )

    columns = [
        "file_id",
        "file_name",
        "data_type",
        "data_category",
        "experimental_strategy",
        "workflow_type",
        "project_id",
        "case_uuid",
        "case_id",
        "sample_uuid",
        "sample_id",
        "sample_type",
        "tissue_type",
        "tumor_descriptor",
        "portion_uuid",
        "portion_id",
        "analyte_uuid",
        "analyte_id",
        "analyte_type",
        "aliquot_uuid",
        "aliquot_id",
    ]

    return pd.DataFrame(rows, columns=columns)


def build_rna_mapping(
    manifest_path: str | Path,
    raw_root: str | Path,
    *,
    batch_size: int = 200,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Build the authoritative GDC RNA file/biospecimen mapping."""
    manifest = read_gdc_manifest(manifest_path)
    raw_root = Path(raw_root)

    metadata = flatten_gdc_rna_metadata(
        query_gdc_files(
            manifest["file_id"].tolist(),
            batch_size=batch_size,
            timeout=timeout,
        )
    )

    # Avoid accidental duplicate rows returned from nested metadata.
    metadata = metadata.drop_duplicates(
        subset=["file_id", "aliquot_uuid", "aliquot_id"]
    )

    df = manifest.merge(
        metadata,
        on=["file_id", "file_name"],
        how="left",
        validate="one_to_many",
    )

    df["raw_relpath"] = df["file_id"].astype(str) + "/" + df["file_name"].astype(str)
    df["raw_exists"] = [(raw_root / relpath).is_file() for relpath in df["raw_relpath"]]

    sort_columns = [
        "case_id",
        "sample_id",
        "aliquot_id",
        "file_id",
    ]
    return df.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def validate_mapping(df: pd.DataFrame) -> None:
    """Fail on metadata problems that would make preprocessing ambiguous."""
    missing_case = df["case_id"].isna()
    missing_sample = df["sample_id"].isna()
    missing_aliquot = df["aliquot_id"].isna()

    problems: list[str] = []

    if missing_case.any():
        problems.append(f"{int(missing_case.sum())} row(s) have no TCGA case barcode")
    if missing_sample.any():
        problems.append(
            f"{int(missing_sample.sum())} row(s) have no TCGA sample barcode"
        )
    if missing_aliquot.any():
        problems.append(
            f"{int(missing_aliquot.sum())} row(s) have no TCGA aliquot barcode"
        )

    # A GDC expression file should resolve to one aliquot in this workflow.
    per_file = df.groupby("file_id", dropna=False)["aliquot_id"].nunique(dropna=True)
    ambiguous_files = per_file[per_file > 1]
    if not ambiguous_files.empty:
        problems.append(f"{len(ambiguous_files)} GDC file(s) map to multiple aliquots")

    if problems:
        raise ValueError("RNA metadata validation failed:\n- " + "\n- ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build TCGA RNA file -> case -> sample -> aliquot metadata "
            "from a GDC manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="GDC download manifest.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root containing <file_uuid>/<file_name> downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path, e.g. metadata/rna_samples.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Write metadata even if case/sample/aliquot validation fails.",
    )

    args = parser.parse_args()

    df = build_rna_mapping(
        args.manifest,
        args.raw_root,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )

    if not args.no_validate:
        validate_mapping(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_files = df["file_id"].nunique()
    n_cases = df["case_id"].nunique()
    n_samples = df["sample_id"].nunique()
    n_aliquots = df["aliquot_id"].nunique()
    n_replicate_samples = int(
        (df.groupby("sample_id")["aliquot_id"].nunique() > 1).sum()
    )
    n_missing_raw = int((~df["raw_exists"]).sum())

    print("TCGA RNA metadata")
    print("-----------------")
    print(f"Files:                    {n_files}")
    print(f"Cases:                    {n_cases}")
    print(f"Samples:                  {n_samples}")
    print(f"Aliquots:                 {n_aliquots}")
    print(f"Samples with >1 aliquot:  {n_replicate_samples}")
    print(f"Missing raw files:        {n_missing_raw}")
    print(f"Output:                   {args.output}")


if __name__ == "__main__":
    main()
