from __future__ import annotations

import math

import pandas as pd

from .normalization import counts_to_cpm


def expressed_gene_mask(
    counts: pd.DataFrame,
    *,
    min_cpm: float = 1.0,
    min_sample_fraction: float = 0.1,
) -> pd.Series:
    """Flag genes reaching min_cpm in a minimum fraction of samples."""
    if min_cpm < 0:
        raise ValueError("min_cpm must be >= 0")
    if not 0 < min_sample_fraction <= 1:
        raise ValueError("min_sample_fraction must be in (0, 1]")

    cpm = counts_to_cpm(counts)
    min_samples = max(
        1,
        math.ceil(counts.shape[1] * min_sample_fraction),
    )
    keep = (cpm >= min_cpm).sum(axis=1) >= min_samples
    keep.name = "keep_for_analysis"
    return keep
