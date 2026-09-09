from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


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

GENE_ID_COL = "gene_id"
GENE_NAME_COL = "gene_name"
GENE_TYPE_COL = "gene_type"
COUNT_COL = "unstranded"
TPM_COL = "tpm_unstranded"


def chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def strip_gencode_version(gene_id: str) -> str:
    """Remove only the numeric GENCODE version, preserving suffixes.

    Examples
    --------
    ENSG00000002586.20 -> ENSG00000002586
    ENSG00000002586.20_PAR_Y -> ENSG00000002586_PAR_Y
    """
    import re

    return re.sub(r"\.\d+", "", gene_id, count=1)


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a standard or headerless GDC download manifest."""
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

    return df.rename(
        columns={
            "id": "file_id",
            "filename": "file_name",
            "md5": "md5sum",
            "size": "file_size",
            "state": "file_state",
        }
    )


def query_files(
    file_ids: Sequence[str],
    *,
    batch_size: int = 200,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Retrieve biospecimen metadata for GDC file UUIDs."""
    file_ids = list(dict.fromkeys(file_ids))
    if not file_ids:
        return []

    session = requests.Session()
    hits: list[dict[str, Any]] = []

    for batch in chunks(file_ids, batch_size):
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
        batch_hits = response.json()["data"]["hits"]
        hits.extend(batch_hits)

        returned = {hit["file_id"] for hit in batch_hits}
        missing = sorted(set(batch) - returned)
        if missing:
            raise RuntimeError(
                "GDC did not return metadata for:\n" + "\n".join(missing)
            )

    return hits


def flatten_file_biospecimens(
    hits: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Flatten file -> case -> sample -> portion -> analyte -> aliquot."""
    rows: list[dict[str, Any]] = []

    for hit in hits:
        cases = hit.get("cases") or [None]

        for case in cases:
            samples = (case or {}).get("samples") or [None]

            for sample in samples:
                project = (case or {}).get("project") or {}
                analysis = hit.get("analysis") or {}

                base = {
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

                portions = (sample or {}).get("portions") or [None]
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

    return pd.DataFrame(rows)


def build_metadata_mapping(
    manifest_path: str | Path,
    raw_root: str | Path,
    *,
    batch_size: int = 200,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Build a GDC RNA file/biospecimen mapping from a manifest."""
    manifest = read_manifest(manifest_path)
    raw_root = Path(raw_root)

    metadata = flatten_file_biospecimens(
        query_files(
            manifest["file_id"].tolist(),
            batch_size=batch_size,
            timeout=timeout,
        )
    )

    if not metadata.empty:
        metadata = metadata.drop_duplicates(
            subset=["file_id", "aliquot_uuid", "aliquot_id"]
        )

    df = manifest.merge(
        metadata,
        on=["file_id", "file_name"],
        how="left",
        validate="one_to_many",
    )

    df["raw_relpath"] = (
        df["file_id"].astype(str) + "/" + df["file_name"].astype(str)
    )
    df["raw_exists"] = [
        (raw_root / relpath).is_file()
        for relpath in df["raw_relpath"]
    ]

    sort_cols = [
        c
        for c in ["case_id", "sample_id", "aliquot_id", "file_id"]
        if c in df.columns
    ]
    return df.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def read_augmented_star_counts(path: str | Path) -> pd.DataFrame:
    """Read one harmonized GDC augmented STAR gene-count TSV."""
    path = Path(path)

    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype={
            GENE_ID_COL: str,
            GENE_NAME_COL: str,
            GENE_TYPE_COL: str,
        },
    )

    required = {
        GENE_ID_COL,
        GENE_NAME_COL,
        GENE_TYPE_COL,
        COUNT_COL,
        TPM_COL,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected GDC STAR columns: {sorted(missing)}"
        )

    df = df[df[GENE_ID_COL].str.startswith("ENSG", na=False)].copy()
    df["gene_id_version"] = df[GENE_ID_COL]
    df[GENE_ID_COL] = df[GENE_ID_COL].map(strip_gencode_version)

    duplicated = df[GENE_ID_COL].duplicated(keep=False)
    if duplicated.any():
        conflicts = df.loc[
            duplicated,
            [GENE_ID_COL, "gene_id_version", GENE_NAME_COL],
        ]
        raise ValueError(
            f"Duplicate Ensembl IDs after version removal in {path}:\n"
            f"{conflicts.head(20).to_string(index=False)}"
        )

    df[COUNT_COL] = pd.to_numeric(df[COUNT_COL], errors="raise").astype(np.int64)
    df[TPM_COL] = pd.to_numeric(df[TPM_COL], errors="raise").astype(np.float64)
    return df.reset_index(drop=True)


def resolve_raw_path(row: pd.Series, raw_root: str | Path) -> Path:
    raw_root = Path(raw_root)

    if "raw_relpath" in row.index and pd.notna(row["raw_relpath"]):
        return raw_root / str(row["raw_relpath"])

    return raw_root / str(row["file_id"]) / str(row["file_name"])


def load_expression_matrices(
    samples: pd.DataFrame,
    raw_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build gene x sample raw-count and TPM matrices."""
    count_series: list[pd.Series] = []
    tpm_series: list[pd.Series] = []

    gene_metadata: pd.DataFrame | None = None
    reference_gene_ids: pd.Index | None = None

    for row in tqdm(
        samples.to_dict(orient="records"),
        total=len(samples),
        desc="Reading GDC RNA-seq",
    ):
        row = pd.Series(row)
        path = resolve_raw_path(row, raw_root)

        if not path.is_file():
            raise FileNotFoundError(f"RNA file not found:\n{path}")

        expr = read_augmented_star_counts(path).set_index(GENE_ID_COL, drop=False)
        gene_ids = expr.index

        if reference_gene_ids is None:
            reference_gene_ids = gene_ids
            gene_metadata = expr[
                ["gene_id_version", GENE_NAME_COL, GENE_TYPE_COL]
            ].copy()
            gene_metadata.index.name = GENE_ID_COL
        else:
            missing = reference_gene_ids.difference(gene_ids)
            extra = gene_ids.difference(reference_gene_ids)
            if len(missing) or len(extra):
                raise ValueError(
                    "GENCODE gene set differs between GDC files.\n"
                    f"File: {path}\n"
                    f"Missing genes: {len(missing)}\n"
                    f"Extra genes: {len(extra)}"
                )
            expr = expr.loc[reference_gene_ids]

        sample_id = str(row["sample_id"])
        count_series.append(
            pd.Series(
                expr[COUNT_COL].to_numpy(dtype=np.int64),
                index=expr.index,
                name=sample_id,
            )
        )
        tpm_series.append(
            pd.Series(
                expr[TPM_COL].to_numpy(dtype=np.float64),
                index=expr.index,
                name=sample_id,
            )
        )

    if gene_metadata is None:
        raise ValueError("No RNA samples were loaded.")

    counts = pd.concat(count_series, axis=1)
    tpm = pd.concat(tpm_series, axis=1)

    counts.index.name = GENE_ID_COL
    tpm.index.name = GENE_ID_COL
    return counts, tpm, gene_metadata
