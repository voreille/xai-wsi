"""Frozen BulkFormer bulk-RNA encoder (vendored model code plus loading and embedding)."""

from .embed import extract_sample_embeddings, predict_expression
from .load import load_bulkformer
from .preprocess import align_to_vocabulary, counts_to_log_tpm

__all__ = [
    "align_to_vocabulary",
    "counts_to_log_tpm",
    "extract_sample_embeddings",
    "load_bulkformer",
    "predict_expression",
]
