from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .model import BulkFormer

Aggregation = Literal["mean", "max", "median", "all"]


@torch.no_grad()
def extract_sample_embeddings(
    model: BulkFormer,
    expression: np.ndarray,
    mask_prob: float,
    *,
    batch_size: int = 4,
    gene_idx: np.ndarray | None = None,
    aggregation: Aggregation = "mean",
    amp_dtype: torch.dtype | None = torch.float16,
) -> np.ndarray:
    """Sample-level embeddings, as upstream ``extract_feature(..., 'sample_level')``.

    ``expression`` is ``N x G`` in vocabulary order (see ``preprocess``). Each gene
    token is ``dim + 3`` wide: the encoder output followed by the mask ratio, mean
    expression and non-zero ratio. Tokens are aggregated over ``gene_idx`` (all genes
    by default). ``"all"`` is max + mean + median, as upstream.
    """
    device = next(model.parameters()).device
    out = []
    for start in tqdm(range(0, len(expression), batch_size), desc="bulkformer"):
        x = torch.as_tensor(
            expression[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.autocast(
            device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
        ):
            tokens = model(x, mask_prob=mask_prob, output_expr=False)
        tokens = tokens.float()
        if gene_idx is not None:
            tokens = tokens[:, torch.as_tensor(gene_idx, device=device)]
        if aggregation == "mean":
            emb = tokens.mean(dim=1)
        elif aggregation == "max":
            emb = tokens.amax(dim=1)
        elif aggregation == "median":
            emb = tokens.median(dim=1).values
        elif aggregation == "all":
            emb = tokens.amax(dim=1) + tokens.mean(dim=1) + tokens.median(dim=1).values
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_expression(
    model: BulkFormer,
    expression: np.ndarray,
    mask_prob: float,
    *,
    batch_size: int = 4,
    amp_dtype: torch.dtype | None = torch.float16,
) -> np.ndarray:
    """Reconstructed ``ln(TPM + 1)``. Masked (-10) genes are imputed."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(expression), batch_size):
        x = torch.as_tensor(
            expression[start : start + batch_size], dtype=torch.float32, device=device
        )
        with torch.autocast(
            device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None
        ):
            pred = model(x, mask_prob=mask_prob, output_expr=True)
        out.append(pred.float().cpu().numpy())
    return np.concatenate(out)
