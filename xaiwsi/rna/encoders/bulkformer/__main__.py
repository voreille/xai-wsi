"""Precompute frozen BulkFormer sample embeddings for one cohort.

    uv run python -m xaiwsi.rna.encoders.bulkformer \\
        --config configs/rna/bulkformer_147M.yaml \\
        --counts data/processed/tcga-luad/rna/counts.parquet \\
        --output-dir data/processed/tcga-luad/rna-embeddings/bulkformer/147M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from xaiwsi.preprocessing.gdc.rna import strip_gencode_version

from .embed import extract_sample_embeddings
from .load import load_bulkformer
from .preprocess import align_to_vocabulary, counts_to_log_tpm

UPSTREAM = "https://github.com/KangBoming/BulkFormer@d63c881"
STAT_COLUMNS = ["stat_mask_ratio", "stat_expr_mean", "stat_nonzero_ratio"]
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--counts",
        type=Path,
        required=True,
        help="raw counts parquet, genes (Ensembl IDs) x samples",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregation", choices=["mean", "max", "median", "all"], default="mean"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", choices=list(AMP_DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gene_ids = pd.read_csv(cfg["gene_info"])["ensg_id"].tolist()
    gene_lengths = pd.read_csv(cfg["gene_lengths"]).set_index("ensg_id")["length"]

    counts = pd.read_parquet(args.counts).T
    counts.columns = [strip_gencode_version(c) for c in counts.columns]
    counts = counts.T.groupby(level=0).sum().T
    expression, mask_prob = align_to_vocabulary(
        counts_to_log_tpm(counts, gene_lengths), gene_ids
    )
    print(
        f"{len(expression)} samples, {mask_prob:.2%} of {len(gene_ids)} "
        "vocabulary genes missing"
    )

    model = load_bulkformer(
        cfg["checkpoint"],
        cfg["graph"],
        cfg["graph_weight"],
        gene_length=len(gene_ids),
        device=args.device,
        **cfg["model"],
    )
    emb = extract_sample_embeddings(
        model,
        expression.to_numpy(dtype="float32"),
        mask_prob,
        batch_size=args.batch_size,
        aggregation=args.aggregation,
        amp_dtype=AMP_DTYPES[args.amp],
    )

    dim = cfg["model"]["dim"]
    columns = [f"emb_{i}" for i in range(dim)] + STAT_COLUMNS
    out = pd.DataFrame(emb, index=expression.index, columns=columns)
    out.index.name = "sample_id"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_dir / "embeddings.parquet")
    meta = {
        "upstream": UPSTREAM,
        "config": str(args.config),
        **{k: str(cfg[k]) for k in ("checkpoint", "graph", "graph_weight")},
        "model": cfg["model"],
        "counts": str(args.counts),
        "normalization": "ln(TPM + 1), TPM from raw counts and upstream gene lengths",
        "n_samples": len(out),
        "mask_prob": mask_prob,
        "aggregation": args.aggregation,
        "amp": args.amp,
        "columns": f"emb_0..emb_{dim - 1} encoder output, then {STAT_COLUMNS}",
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out.shape} to {args.output_dir}")


if __name__ == "__main__":
    main()
