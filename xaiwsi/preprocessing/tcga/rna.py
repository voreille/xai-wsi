from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xaiwsi.preprocessing.gdc.rna import (
    load_expression_matrices,
    resolve_raw_path,
)
from xaiwsi.rna.filtering import expressed_gene_mask
from xaiwsi.rna.normalization import log2p1, tmm_normalize


RNA_ANALYTE_PRIORITY = {
    "H": 3,
    "R": 2,
    "T": 1,
}


def parse_tcga_aliquot_barcode(
    aliquot_id: str,
) -> tuple[str, str]:
    parts = aliquot_id.split("-")
    if len(parts) < 7:
        raise ValueError(
            f"Unexpected TCGA aliquot barcode: {aliquot_id!r}"
        )

    portion_analyte = parts[4]
    plate = parts[5]
    if not portion_analyte:
        raise ValueError(
            f"Could not read analyte code from {aliquot_id!r}"
        )

    return portion_analyte[-1].upper(), plate


def select_tcga_rna_samples(
    metadata: pd.DataFrame,
    *,
    sample_type: str | None = "Primary Tumor",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one RNA file per TCGA sample with an auditable rule."""
    df = metadata.copy()

    if sample_type is not None:
        df = df[df["sample_type"] == sample_type].copy()

    required = {
        "file_id",
        "file_name",
        "case_id",
        "sample_id",
        "aliquot_id",
        "raw_relpath",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"RNA metadata is missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["file_id", "file_name", "case_id", "sample_id"]
    )
    df = df.drop_duplicates(
        subset=["file_id", "sample_id", "aliquot_id"]
    )

    n_files = df.groupby("sample_id")["file_id"].nunique()
    replicate_samples = n_files[n_files > 1].index

    bad = df[
        df["sample_id"].isin(replicate_samples)
        & df["aliquot_id"].isna()
    ]
    if not bad.empty:
        raise ValueError(
            "Replicate TCGA samples are missing aliquot metadata:\n"
            + bad[["sample_id", "file_id"]].to_string(index=False)
        )

    analyte_codes = []
    plates = []
    for aliquot in df["aliquot_id"]:
        if pd.isna(aliquot):
            analyte_codes.append(None)
            plates.append(None)
        else:
            analyte, plate = parse_tcga_aliquot_barcode(str(aliquot))
            analyte_codes.append(analyte)
            plates.append(plate)

    df["analyte_code"] = analyte_codes
    df["plate"] = plates
    df["analyte_priority"] = (
        df["analyte_code"]
        .map(RNA_ANALYTE_PRIORITY)
        .fillna(0)
        .astype(int)
    )

    per_aliquot = (
        df.dropna(subset=["aliquot_id"])
        .groupby(["sample_id", "aliquot_id"])["file_id"]
        .nunique()
    )
    duplicate_files = per_aliquot[per_aliquot > 1]
    if not duplicate_files.empty:
        raise ValueError(
            "Multiple GDC expression files map to the exact same TCGA "
            "aliquot; resolve file versions explicitly."
        )

    df["n_available_files"] = (
        df.groupby("sample_id")["file_id"].transform("nunique").astype(int)
    )
    df["n_available_aliquots"] = (
        df.groupby("sample_id")["aliquot_id"]
        .transform(lambda x: x.dropna().nunique())
        .astype(int)
    )

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
    return selected, audit


def preprocess_tcga_rna(
    metadata_path: str | Path,
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    sample_type: str | None = "Primary Tumor",
    min_cpm: float = 1.0,
    min_sample_fraction: float = 0.1,
) -> None:
    metadata = pd.read_csv(metadata_path, dtype=str)
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, replicate_audit = select_tcga_rna_samples(
        metadata,
        sample_type=sample_type,
    )

    missing_paths = [
        resolve_raw_path(row, raw_root)
        for _, row in samples.iterrows()
        if not resolve_raw_path(row, raw_root).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} selected RNA files are missing:\n"
            + "\n".join(map(str, missing_paths[:20]))
        )

    counts, tpm, genes = load_expression_matrices(samples, raw_root)
    keep = expressed_gene_mask(
        counts,
        min_cpm=min_cpm,
        min_sample_fraction=min_sample_fraction,
    )
    genes["keep_for_analysis"] = keep

    tmm_cpm, factors = tmm_normalize(counts)
    log2_tpm = log2p1(tpm)
    log2_tmm_cpm = log2p1(tmm_cpm)

    samples = samples.copy()
    samples["library_size"] = (
        samples["sample_id"].map(counts.sum(axis=0)).astype(np.int64)
    )
    samples["tmm_norm_factor"] = (
        samples["sample_id"].map(factors).astype(float)
    )

    counts.to_parquet(output_dir / "counts.parquet")
    tpm.to_parquet(output_dir / "tpm.parquet")
    log2_tpm.to_parquet(output_dir / "log2_tpm.parquet")
    tmm_cpm.to_parquet(output_dir / "tmm_cpm.parquet")
    log2_tmm_cpm.to_parquet(output_dir / "log2_tmm_cpm.parquet")
    genes.reset_index().to_csv(output_dir / "genes.csv", index=False)
    samples.to_csv(output_dir / "samples.csv", index=False)
    replicate_audit.to_csv(output_dir / "rna_replicates.csv", index=False)

    print("TCGA RNA preprocessing")
    print("----------------------")
    print(f"Selected samples:          {counts.shape[1]}")
    print(f"Genes:                     {counts.shape[0]}")
    print(f"Genes passing CPM filter:  {int(keep.sum())}")
    print(f"Output:                    {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble and normalize TCGA RNA-seq."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-type", default="Primary Tumor")
    parser.add_argument("--all-sample-types", action="store_true")
    parser.add_argument("--min-cpm", type=float, default=1.0)
    parser.add_argument("--min-sample-fraction", type=float, default=0.1)
    args = parser.parse_args()

    preprocess_tcga_rna(
        args.metadata,
        args.raw_root,
        args.output_dir,
        sample_type=None if args.all_sample_types else args.sample_type,
        min_cpm=args.min_cpm,
        min_sample_fraction=args.min_sample_fraction,
    )


if __name__ == "__main__":
    main()
