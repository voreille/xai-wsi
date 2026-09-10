from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .genesets import filter_gene_sets


def fit_gene_zscore_stats(expression: pd.DataFrame) -> pd.DataFrame:
    """Fit per-gene mean/std across samples.

    expression: gene_symbol x sample
    """
    stats = pd.DataFrame(
        {
            "mean": expression.mean(axis=1),
            "std": expression.std(axis=1, ddof=0),
        }
    )
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats


def score_mean_z(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, set[str]],
    *,
    zscore_stats: pd.DataFrame | None = None,
    min_size: int = 5,
) -> pd.DataFrame:
    """Score gene sets by mean gene-wise z-score.

    Parameters
    ----------
    expression:
        gene_symbol x sample matrix.
    gene_sets:
        Mapping from gene-set name to gene symbols.
    zscore_stats:
        Optional DataFrame indexed by gene_symbol with columns
        ["mean", "std"]. If None, statistics are fitted from expression.

    Returns
    -------
    sample x pathway score matrix.
    """
    if zscore_stats is None:
        zscore_stats = fit_gene_zscore_stats(expression)

    shared = expression.index.intersection(zscore_stats.index)

    expression = expression.loc[shared]
    stats = zscore_stats.loc[shared]

    z = expression.sub(stats["mean"], axis=0).div(stats["std"], axis=0)

    gene_sets = filter_gene_sets(
        gene_sets,
        set(z.index),
        min_size=min_size,
    )

    scores = {
        name: z.loc[sorted(genes)].mean(axis=0) for name, genes in gene_sets.items()
    }

    out = pd.DataFrame(scores)
    out.index.name = "sample_id"
    return out


def score_ssgsea(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, set[str]],
    *,
    min_size: int = 5,
    max_size: int = 5000,
    threads: int = 1,
) -> pd.DataFrame:
    """Score gene sets with ssGSEA using optional dependency gseapy.

    expression must be gene_symbol x sample.
    Returns sample x pathway.
    """
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "ssGSEA requires gseapy. Install it with `uv add gseapy`."
        ) from exc

    gene_sets = filter_gene_sets(
        gene_sets,
        set(expression.index),
        min_size=min_size,
        max_size=max_size,
    )

    result = gp.ssgsea(
        data=expression,
        gene_sets={k: sorted(v) for k, v in gene_sets.items()},
        min_size=min_size,
        max_size=max_size,
        threads=threads,
        permutation_num=0,
        no_plot=True,
        verbose=False,
    )

    res2d = result.res2d.copy()

    # Current gseapy represents ssGSEA output in long format.
    required = {"Name", "Term", "NES"}
    if not required.issubset(res2d.columns):
        raise RuntimeError(
            f"Unexpected gseapy ssGSEA result columns: {list(res2d.columns)}"
        )

    out = res2d.pivot(
        index="Name",
        columns="Term",
        values="NES",
    )
    out.index.name = "sample_id"
    return out
