from __future__ import annotations

import numpy as np
import pandas as pd
from rnanorm import TMM


def counts_to_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """Convert a gene x sample raw-count matrix to CPM."""
    library_sizes = counts.sum(axis=0)
    if (library_sizes <= 0).any():
        bad = library_sizes[library_sizes <= 0]
        raise ValueError(
            "Samples with zero library size:\n" + bad.to_string()
        )
    return counts.divide(library_sizes, axis=1) * 1e6


def tmm_normalize(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return TMM-normalized CPM and sample normalization factors."""
    x = counts.T

    normalizer = TMM().set_output(transform="pandas")
    normalized = normalizer.fit_transform(x)

    factors = normalizer.get_norm_factors(x)
    if not isinstance(factors, pd.Series):
        factors = pd.Series(
            np.asarray(factors),
            index=x.index,
            name="tmm_norm_factor",
        )
    else:
        factors = factors.reindex(x.index)
        factors.name = "tmm_norm_factor"

    normalized = normalized.T
    normalized.index.name = counts.index.name
    return normalized, factors


def log2p1(expression: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log2(x + 1)."""
    return np.log2(expression + 1.0)
