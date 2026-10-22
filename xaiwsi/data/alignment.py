from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from wsitools.storage.interfaces import EmbeddingStore


def load_rna_embeddings(path: str | Path, column_regex: str = r"^emb_") -> pd.DataFrame:
    """Precomputed RNA embeddings (samples x features), e.g. BulkFormer's parquet."""
    embeddings = pd.read_parquet(path).filter(regex=column_regex)
    if embeddings.empty:
        raise ValueError(f"no columns matching {column_regex!r} in {path}")
    return embeddings


class SlideRNADataset(Dataset):
    """Paired whole-slide tile bags and RNA embedding vectors, one item per case."""

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        pairs: pd.DataFrame,
        rna_embeddings: pd.DataFrame,
    ):
        self.embedding_store = embedding_store
        self.pairs = pairs.reset_index(drop=True)
        # copy=True: with pandas copy-on-write, to_numpy() can return a read-only view.
        self.rna = rna_embeddings.loc[self.pairs["sample_id"]].to_numpy(
            np.float32, copy=True
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.pairs.iloc[idx]
        tiles, _, _ = self.embedding_store.load(str(row["slide_id"]))
        return {
            "case_id": str(row["case_id"]),
            "slide_id": str(row["slide_id"]),
            "sample_id": str(row["sample_id"]),
            "tiles": torch.from_numpy(np.asarray(tiles)).float(),
            "rna": torch.from_numpy(self.rna[idx]),
        }


def collate_bags(items: list[dict[str, object]]) -> dict[str, object]:
    """Batch variable-size bags as a list (no padding, no tile sampling)."""
    return {
        "case_id": [item["case_id"] for item in items],
        "slide_id": [item["slide_id"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
        "tiles": [item["tiles"] for item in items],
        "rna": torch.stack([item["rna"] for item in items]),
    }
