from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xaiwsi.rna.genes import expression_to_gene_symbols
from xaiwsi.rna.pathways.genesets import load_gmt
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import score_mean_z, score_ssgsea
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pathway/gene-set activity from an RNA matrix."
    )
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--gene-sets", type=Path, required=True)
    parser.add_argument(
        "--gene-set-type",
        choices=["gmt", "hallmarks", "tavernari"],
        default="gmt",
    )
    parser.add_argument(
        "--method",
        choices=["ssgsea", "mean-z"],
        default="ssgsea",
    )
    parser.add_argument(
        "--duplicate-symbols",
        choices=["sum", "mean", "max", "error"],
        default="sum",
    )
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    expression = pd.read_parquet(args.expression)
    genes = pd.read_csv(args.genes)

    expression = expression_to_gene_symbols(
        expression,
        genes,
        duplicate_strategy=args.duplicate_symbols,
    )

    if args.gene_set_type == "hallmarks":
        gene_sets = load_hallmarks(args.gene_sets)
    elif args.gene_set_type == "tavernari":
        gene_sets = load_tavernari_signatures(args.gene_sets)
    else:
        gene_sets = load_gmt(args.gene_sets)

    if args.method == "ssgsea":
        scores = score_ssgsea(
            expression,
            gene_sets,
            min_size=args.min_size,
            threads=args.threads,
        )
    else:
        scores = score_mean_z(
            expression,
            gene_sets,
            min_size=args.min_size,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(args.output)

    print(f"Samples:   {len(scores)}")
    print(f"Pathways:  {scores.shape[1]}")
    print(f"Output:    {args.output}")


if __name__ == "__main__":
    main()
