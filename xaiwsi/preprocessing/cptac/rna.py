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


def select_cptac_rna_samples(
    metadata: pd.DataFrame,
    *,
    sample_type: str | None = "Primary Tumor",
) -> pd.DataFrame:
    """Select one GDC expression file per CPTAC sample.

    CPTAC-specific replicate rules are not assumed. Multiple metadata
    rows for the same file/sample can occur because expanded GDC
    biospecimen metadata contains multiple downstream entities; these
    are collapsed here.

    If genuinely different GDC files map to the same sample, fail and
    require explicit inspection.
    """
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

    # Expanded GDC biospecimen metadata may produce several rows for
    # the same expression file/sample because multiple aliquots or
    # other nested entities are attached to the sample.
    #
    # At the expression-file level this is still a single asset.
    df = df.drop_duplicates(
        subset=["file_id", "sample_id"],
        keep="first",
    )

    # A real replicate means DIFFERENT expression files for the same
    # biological sample.
    n_files = df.groupby("sample_id")["file_id"].nunique()

    duplicates = n_files[n_files > 1]

    if not duplicates.empty:
        conflict = df[df["sample_id"].isin(duplicates.index)]

        raise ValueError(
            "Multiple distinct CPTAC RNA files map to the same sample. "
            "Inspect these before defining a CPTAC replicate rule:\n"
            + conflict[
                [
                    "case_id",
                    "sample_id",
                    "file_id",
                    "file_name",
                ]
            ].to_string(index=False)
        )

    if df["sample_id"].duplicated().any():
        raise RuntimeError(
            "Internal error: sample_id is still non-unique after CPTAC file selection."
        )

    return df.sort_values(["case_id", "sample_id"]).reset_index(drop=True)


def preprocess_cptac_rna(
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

    samples = select_cptac_rna_samples(
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
    samples["tmm_norm_factor"] = samples["sample_id"].map(factors).astype(float)

    counts.to_parquet(output_dir / "counts.parquet")
    tpm.to_parquet(output_dir / "tpm.parquet")
    log2_tpm.to_parquet(output_dir / "log2_tpm.parquet")
    tmm_cpm.to_parquet(output_dir / "tmm_cpm.parquet")
    log2_tmm_cpm.to_parquet(output_dir / "log2_tmm_cpm.parquet")
    genes.reset_index().to_csv(output_dir / "genes.csv", index=False)
    samples.to_csv(output_dir / "samples.csv", index=False)

    print("CPTAC RNA preprocessing")
    print("-----------------------")
    print(f"Selected samples:          {counts.shape[1]}")
    print(f"Genes:                     {counts.shape[0]}")
    print(f"Genes passing CPM filter:  {int(keep.sum())}")
    print(f"Output:                    {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble and normalize CPTAC RNA-seq."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-type", default="Primary Tumor")
    parser.add_argument("--all-sample-types", action="store_true")
    parser.add_argument("--min-cpm", type=float, default=1.0)
    parser.add_argument("--min-sample-fraction", type=float, default=0.1)
    args = parser.parse_args()

    preprocess_cptac_rna(
        args.metadata,
        args.raw_root,
        args.output_dir,
        sample_type=None if args.all_sample_types else args.sample_type,
        min_cpm=args.min_cpm,
        min_sample_fraction=args.min_sample_fraction,
    )


if __name__ == "__main__":
    main()
