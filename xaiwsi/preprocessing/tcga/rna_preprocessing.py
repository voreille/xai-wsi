from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rnanorm import TMM
from tqdm import tqdm

GENE_ID_COL = "gene_id"
GENE_NAME_COL = "gene_name"
GENE_TYPE_COL = "gene_type"
COUNT_COL = "unstranded"
TPM_COL = "tpm_unstranded"

# Historical TCGA/Firehose precedence for RNA analyte replicates:
# H preferred to R, and both preferred to T.
RNA_ANALYTE_PRIORITY = {
    "H": 3,
    "R": 2,
    "T": 1,
}


def read_gdc_star_counts(path: str | Path) -> pd.DataFrame:
    """Read one GDC augmented STAR gene-count TSV."""
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

    # STAR summary rows are named N_unmapped, N_multimapping, ...
    # Actual GENCODE gene IDs begin with ENSG.
    df = df[df[GENE_ID_COL].str.startswith("ENSG", na=False)].copy()

    df["gene_id_version"] = df[GENE_ID_COL]
    # Remove only the Ensembl version number while preserving suffixes
    # such as "_PAR_Y".
    df[GENE_ID_COL] = df[GENE_ID_COL].str.replace(
        r"\.\d+",
        "",
        regex=True,
    )

    duplicated = df[GENE_ID_COL].duplicated(keep=False)
    if duplicated.any():
        conflicts = df.loc[
            duplicated,
            [
                GENE_ID_COL,
                "gene_id_version",
                GENE_NAME_COL,
            ],
        ]
        raise ValueError(
            f"Duplicate Ensembl IDs after version removal in {path}:\n"
            f"{conflicts.head(20).to_string(index=False)}"
        )

    df[COUNT_COL] = pd.to_numeric(
        df[COUNT_COL],
        errors="raise",
    ).astype(np.int64)

    df[TPM_COL] = pd.to_numeric(
        df[TPM_COL],
        errors="raise",
    ).astype(np.float64)

    return df.reset_index(drop=True)


def _parse_tcga_aliquot_barcode(
    aliquot_id: str,
) -> tuple[str, str]:
    """Return (analyte_code, plate) from a TCGA aliquot barcode.

    Example:
        TCGA-44-6147-01A-11R-A278-07
                         ^   ^^^^
                         R   plate
    """
    parts = aliquot_id.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected TCGA aliquot barcode: {aliquot_id!r}")

    portion_analyte = parts[4]
    plate = parts[5]

    if not portion_analyte:
        raise ValueError(f"Could not read analyte code from {aliquot_id!r}")

    analyte_code = portion_analyte[-1].upper()
    return analyte_code, plate


def _annotate_replicate_priority(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    metadata = metadata.copy()

    analyte_codes: list[str | None] = []
    plates: list[str | None] = []

    for aliquot_id in metadata["aliquot_id"]:
        if pd.isna(aliquot_id):
            analyte_codes.append(None)
            plates.append(None)
            continue

        analyte_code, plate = _parse_tcga_aliquot_barcode(str(aliquot_id))
        analyte_codes.append(analyte_code)
        plates.append(plate)

    metadata["analyte_code"] = analyte_codes
    metadata["plate"] = plates
    metadata["analyte_priority"] = (
        metadata["analyte_code"].map(RNA_ANALYTE_PRIORITY).fillna(0).astype(int)
    )

    return metadata


def select_rna_samples(
    metadata: pd.DataFrame,
    *,
    sample_type: str | None = "Primary Tumor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one RNA aliquot/file per TCGA sample.

    Replicate selection follows the historical TCGA/Firehose convention:
      1. prefer RNA analyte H > R > T;
      2. within the preferred analyte, prefer the later/higher plate;
      3. use the full aliquot barcode as a deterministic final tie-break.

    Returns:
        selected:
            One row per sample_id.
        audit:
            All eligible rows with rank/selected information.
    """
    required = {
        "file_id",
        "file_name",
        "case_id",
        "sample_id",
        "sample_type",
        "aliquot_id",
        "raw_relpath",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "RNA metadata is missing required columns: "
            f"{sorted(missing)}. Re-run rna_metadata.py."
        )

    df = metadata.copy()

    if sample_type is not None:
        df = df[df["sample_type"] == sample_type].copy()

    df = df.dropna(
        subset=[
            "file_id",
            "file_name",
            "case_id",
            "sample_id",
        ]
    )

    if df.empty:
        raise ValueError("No RNA samples remain after metadata/sample-type filtering.")

    # Remove exact repeats that might arise from nested API metadata.
    df = df.drop_duplicates(subset=["file_id", "sample_id", "aliquot_id"])

    # If a sample has replicates, an aliquot barcode is required to resolve
    # them scientifically rather than arbitrarily.
    n_files_per_sample = df.groupby("sample_id")["file_id"].nunique()
    replicate_samples = n_files_per_sample[n_files_per_sample > 1].index
    bad = df[df["sample_id"].isin(replicate_samples) & df["aliquot_id"].isna()]
    if not bad.empty:
        raise ValueError(
            "Replicate RNA samples are missing aliquot metadata. "
            "Re-run rna_metadata.py with aliquot fields.\n"
            + bad[["sample_id", "file_id", "file_name"]].to_string(index=False)
        )

    df = _annotate_replicate_priority(df)

    # A duplicate file for the exact same aliquot is not a biological
    # replicate. Do not silently choose between analysis files.
    per_aliquot = (
        df.dropna(subset=["aliquot_id"])
        .groupby(["sample_id", "aliquot_id"])["file_id"]
        .nunique()
    )
    duplicate_files = per_aliquot[per_aliquot > 1]
    if not duplicate_files.empty:
        keys = set(duplicate_files.index)
        conflict = df[
            [
                (sample_id, aliquot_id) in keys
                for sample_id, aliquot_id in zip(
                    df["sample_id"],
                    df["aliquot_id"],
                )
            ]
        ]
        raise ValueError(
            "Multiple GDC expression files map to the exact same aliquot. "
            "Resolve file versions explicitly:\n"
            + conflict[
                [
                    "sample_id",
                    "aliquot_id",
                    "file_id",
                    "file_name",
                ]
            ].to_string(index=False)
        )

    df["n_available_files"] = (
        df.groupby("sample_id")["file_id"].transform("nunique").astype(int)
    )
    df["n_available_aliquots"] = (
        df.groupby("sample_id")["aliquot_id"]
        .transform(lambda x: x.dropna().nunique())
        .astype(int)
    )

    # Missing values sort last. Descending order encodes the TCGA
    # replicate preference.
    df["_plate_sort"] = df["plate"].fillna("")
    df["_aliquot_sort"] = df["aliquot_id"].fillna("")

    df = df.sort_values(
        [
            "sample_id",
            "analyte_priority",
            "_plate_sort",
            "_aliquot_sort",
        ],
        ascending=[True, False, False, False],
        kind="stable",
    )

    df["selection_rank"] = df.groupby("sample_id").cumcount() + 1
    df["selected"] = df["selection_rank"] == 1
    df["selection_reason"] = np.where(
        df["n_available_files"] == 1,
        "unique",
        "tcga_firehose_replicate_rule",
    )

    audit = (
        df.drop(columns=["_plate_sort", "_aliquot_sort"])
        .sort_values(["case_id", "sample_id", "selection_rank"])
        .reset_index(drop=True)
    )

    selected = (
        audit[audit["selected"]]
        .copy()
        .sort_values(["case_id", "sample_id"])
        .reset_index(drop=True)
    )

    if selected["sample_id"].duplicated().any():
        raise RuntimeError(
            "Internal error: replicate selection did not produce one row per sample."
        )

    return selected, audit


def _resolve_raw_path(
    row: pd.Series,
    raw_root: Path,
) -> Path:
    relpath = row.get("raw_relpath")
    if pd.notna(relpath):
        return raw_root / str(relpath)

    return raw_root / str(row["file_id"]) / str(row["file_name"])


def load_expression_matrices(
    samples: pd.DataFrame,
    raw_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build gene x sample raw-count and TPM matrices."""
    raw_root = Path(raw_root)

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
        path = _resolve_raw_path(row, raw_root)

        if not path.is_file():
            raise FileNotFoundError(f"RNA file not found:\n{path}")

        expr = read_gdc_star_counts(path)
        expr = expr.set_index(GENE_ID_COL, drop=False)
        gene_ids = expr.index

        if reference_gene_ids is None:
            reference_gene_ids = gene_ids

            gene_metadata = expr[
                [
                    "gene_id_version",
                    GENE_NAME_COL,
                    GENE_TYPE_COL,
                ]
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

            # Reorder to the first file explicitly.
            expr = expr.loc[reference_gene_ids]

            # Gene annotations should also be harmonized across files.
            assert gene_metadata is not None
            current_names = expr[GENE_NAME_COL].fillna("")
            reference_names = gene_metadata[GENE_NAME_COL].fillna("")
            if not current_names.equals(reference_names):
                raise ValueError(
                    "Gene symbols differ between harmonized GDC files. "
                    f"First mismatch encountered in {path}"
                )

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


def compute_expression_filter(
    counts: pd.DataFrame,
    *,
    min_cpm: float,
    min_sample_fraction: float,
) -> pd.Series:
    """Flag genes expressed above min_cpm in enough samples."""
    library_sizes = counts.sum(axis=0)

    if (library_sizes <= 0).any():
        bad = library_sizes[library_sizes <= 0]
        raise ValueError("Samples with zero library size:\n" + bad.to_string())

    cpm = counts.divide(library_sizes, axis=1) * 1e6

    min_samples = max(
        1,
        math.ceil(counts.shape[1] * min_sample_fraction),
    )

    keep = (cpm >= min_cpm).sum(axis=1) >= min_samples
    keep.name = "keep_for_analysis"
    return keep


def tmm_normalize(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return TMM-normalized CPM and per-sample normalization factors.

    rnanorm expects samples x genes; our stored matrices are genes x samples.
    """
    x = counts.T

    normalizer = TMM().set_output(transform="pandas")
    normalized = normalizer.fit_transform(x)

    factors = normalizer.get_norm_factors(x)
    if isinstance(factors, pd.Series):
        factors = factors.reindex(x.index)
    else:
        factors = pd.Series(
            np.asarray(factors),
            index=x.index,
            name="tmm_norm_factor",
        )

    normalized = normalized.T
    normalized.index.name = counts.index.name
    normalized.columns.name = counts.columns.name

    factors.name = "tmm_norm_factor"
    return normalized, factors


def preprocess_rna(
    metadata_path: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    sample_type: str | None = "Primary Tumor",
    min_cpm: float = 1.0,
    min_sample_fraction: float = 0.1,
) -> None:
    """Assemble and normalize a TCGA RNA-seq cohort."""
    metadata_path = Path(metadata_path)
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path, dtype=str)

    samples, replicate_audit = select_rna_samples(
        metadata,
        sample_type=sample_type,
    )

    # Check selected paths before doing expensive matrix assembly.
    selected_paths = [_resolve_raw_path(row, raw_root) for _, row in samples.iterrows()]
    missing_paths = [path for path in selected_paths if not path.is_file()]
    if missing_paths:
        preview = "\n".join(str(p) for p in missing_paths[:20])
        raise FileNotFoundError(
            f"{len(missing_paths)} selected RNA files are missing:\n{preview}"
        )

    counts, tpm, genes = load_expression_matrices(
        samples,
        raw_root,
    )

    keep = compute_expression_filter(
        counts,
        min_cpm=min_cpm,
        min_sample_fraction=min_sample_fraction,
    )
    genes["keep_for_analysis"] = keep

    tmm_cpm, tmm_factors = tmm_normalize(counts)

    log2_tpm = np.log2(tpm + 1.0)
    log2_tmm_cpm = np.log2(tmm_cpm + 1.0)

    # Add useful sample-level QC/provenance.
    samples = samples.copy()

    library_sizes = counts.sum(axis=0)
    samples["library_size"] = samples["sample_id"].map(library_sizes).astype(np.int64)
    samples["tmm_norm_factor"] = samples["sample_id"].map(tmm_factors).astype(float)

    # Keep expression matrices gene x sample. This gives ~60k rows rather
    # than ~60k Parquet columns and works naturally for gene-set scoring.
    counts.to_parquet(output_dir / "counts.parquet")
    tpm.to_parquet(output_dir / "tpm.parquet")
    log2_tpm.to_parquet(output_dir / "log2_tpm.parquet")
    tmm_cpm.to_parquet(output_dir / "tmm_cpm.parquet")
    log2_tmm_cpm.to_parquet(output_dir / "log2_tmm_cpm.parquet")

    genes.reset_index().to_csv(
        output_dir / "genes.csv",
        index=False,
    )
    samples.to_csv(
        output_dir / "samples.csv",
        index=False,
    )
    replicate_audit.to_csv(
        output_dir / "rna_replicates.csv",
        index=False,
    )

    n_replicate_samples = int(
        (replicate_audit["n_available_files"] > 1)
        .groupby(replicate_audit["sample_id"])
        .any()
        .sum()
    )

    print("TCGA RNA preprocessing")
    print("----------------------")
    print(f"Selected samples:           {counts.shape[1]}")
    print(f"Genes:                      {counts.shape[0]}")
    print(f"Genes passing CPM filter:   {int(keep.sum())}")
    print(f"Samples with replicates:    {n_replicate_samples}")
    print(f"Sample type:                {sample_type or 'all'}")
    print(f"Output:                     {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Assemble and normalize GDC TCGA augmented STAR gene counts.")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="rna_samples.csv produced by rna_metadata.py.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root of the raw GDC RNA download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sample-type",
        default="Primary Tumor",
        help="GDC sample type to retain (default: Primary Tumor).",
    )
    parser.add_argument(
        "--all-sample-types",
        action="store_true",
        help="Do not filter by GDC sample_type.",
    )
    parser.add_argument(
        "--min-cpm",
        type=float,
        default=1.0,
        help="CPM threshold used for the analysis gene flag.",
    )
    parser.add_argument(
        "--min-sample-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of selected samples in which a gene must reach "
            "--min-cpm to set keep_for_analysis=True."
        ),
    )

    args = parser.parse_args()

    if args.min_cpm < 0:
        parser.error("--min-cpm must be >= 0")

    if not 0 < args.min_sample_fraction <= 1:
        parser.error("--min-sample-fraction must be in (0, 1]")

    sample_type = None if args.all_sample_types else args.sample_type

    preprocess_rna(
        metadata_path=args.metadata,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        sample_type=sample_type,
        min_cpm=args.min_cpm,
        min_sample_fraction=args.min_sample_fraction,
    )


if __name__ == "__main__":
    main()
