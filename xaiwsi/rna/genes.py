from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def expression_to_gene_symbols(
    expression: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    gene_id_col: str = "gene_id",
    gene_name_col: str = "gene_name",
    duplicate_strategy: str = "sum",
) -> pd.DataFrame:
    """Convert a gene-ID indexed expression matrix to gene symbols.

    Duplicate gene symbols can arise, including PAR genes. They are handled
    explicitly here instead of during GDC ingestion.

    duplicate_strategy:
        "sum"  - sum duplicate rows
        "mean" - average duplicate rows
        "max"  - maximum expression across duplicate rows
        "error" - fail if duplicate symbols exist
    """
    if gene_id_col in genes.columns:
        mapping = genes.set_index(gene_id_col)[gene_name_col]
    else:
        if genes.index.name != gene_id_col:
            raise ValueError(
                f"genes must contain {gene_id_col!r} or use it as index"
            )
        mapping = genes[gene_name_col]

    shared = expression.index.intersection(mapping.index)
    expr = expression.loc[shared].copy()
    symbols = mapping.loc[shared]

    valid = symbols.notna() & (symbols.astype(str).str.len() > 0)
    expr = expr.loc[valid]
    symbols = symbols.loc[valid].astype(str)

    expr.index = symbols
    expr.index.name = "gene_symbol"

    duplicated = expr.index.duplicated(keep=False)
    if duplicate_strategy == "error" and duplicated.any():
        dup = sorted(expr.index[duplicated].unique())
        raise ValueError(
            "Duplicate gene symbols after mapping: "
            + ", ".join(dup[:20])
        )

    if duplicate_strategy == "sum":
        return expr.groupby(level=0, sort=False).sum()
    if duplicate_strategy == "mean":
        return expr.groupby(level=0, sort=False).mean()
    if duplicate_strategy == "max":
        return expr.groupby(level=0, sort=False).max()
    if duplicate_strategy == "error":
        return expr

    raise ValueError(
        "duplicate_strategy must be one of: sum, mean, max, error"
    )
