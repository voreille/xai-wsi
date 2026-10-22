from __future__ import annotations

import numpy as np
import pandas as pd

MASK_VALUE = -10.0
DEFAULT_GENE_LENGTH_BP = 1000


def counts_to_log_tpm(
    counts: pd.DataFrame,
    gene_lengths: pd.Series,
) -> pd.DataFrame:
    """Raw counts (samples x genes) to ``ln(TPM + 1)``, as upstream ``normalize_data``.

    Genes without a length get 1000 bp. TPM is computed over all genes in ``counts``.
    This uses the natural log, unlike our ``log2_tpm`` matrices.
    """
    lengths_kb = (
        gene_lengths.reindex(counts.columns).fillna(DEFAULT_GENE_LENGTH_BP).to_numpy()
        / 1000
    )
    rate = counts.to_numpy(dtype=np.float64) / lengths_kb
    totals = rate.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1e-6
    tpm = rate / totals * 1e6
    return pd.DataFrame(np.log1p(tpm), index=counts.index, columns=counts.columns)


def align_to_vocabulary(
    expression: pd.DataFrame,
    gene_ids: list[str],
) -> tuple[pd.DataFrame, float]:
    """Reorder columns to the model vocabulary and fill missing genes with -10.

    Returns the aligned matrix and the missing-gene fraction, which upstream passes
    to the model as ``mask_prob``.
    """
    present = pd.Index(gene_ids).isin(expression.columns)
    aligned = expression.reindex(columns=gene_ids, fill_value=MASK_VALUE)
    return aligned, float(1 - present.mean())
