from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from wsitools.storage.interfaces import EmbeddingStore

from xaiwsi.rna.genes import expression_to_gene_symbols
from xaiwsi.rna.normalization import log2p1


def load_log2_tpm_gene_symbols(rna_dir: str | Path) -> pd.DataFrame:
    """Load TPM, collapse duplicate symbols in linear space, then log2(TPM + 1)."""
    rna_dir = Path(rna_dir)
    tpm = pd.read_parquet(rna_dir / "tpm.parquet")
    genes = pd.read_csv(rna_dir / "genes.csv")
    tpm = expression_to_gene_symbols(tpm, genes, duplicate_strategy="sum")
    return log2p1(tpm)


def build_case_pairs(
    embedding_store: EmbeddingStore,
    wsi_metadata: pd.DataFrame,
    rna_samples: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Build one slide / one RNA sample per case for the first baseline."""
    available_slides = set(embedding_store.slide_ids())

    wsi = wsi_metadata[wsi_metadata["slide_id"].isin(available_slides)].copy()
    rna = rna_samples[rna_samples["sample_id"].isin(expression.columns)].copy()

    slides_per_case = wsi.groupby("case_id")["slide_id"].nunique()
    rna_per_case = rna.groupby("case_id")["sample_id"].nunique()

    print("Cases with multiple slides:", int((slides_per_case > 1).sum()))
    print("Cases with multiple RNA samples:", int((rna_per_case > 1).sum()))

    # Temporary deterministic baseline behavior.
    wsi = (
        wsi.sort_values(["case_id", "slide_id"])
        .drop_duplicates("case_id", keep="first")
    )
    rna = (
        rna.sort_values(["case_id", "sample_id"])
        .drop_duplicates("case_id", keep="first")
    )

    pairs = wsi[["case_id", "slide_id"]].merge(
        rna[["case_id", "sample_id"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    return pairs.sort_values("case_id").reset_index(drop=True)


def split_cases(
    pairs: pd.DataFrame,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))
    n_val = round(len(pairs) * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return (
        pairs.iloc[train_idx].reset_index(drop=True),
        pairs.iloc[val_idx].reset_index(drop=True),
    )


class PathwayEmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        pairs: pd.DataFrame,
        pathway_scores: pd.DataFrame,
    ):
        self.embedding_store = embedding_store
        self.pairs = pairs.reset_index(drop=True)
        self.pathway_scores = pathway_scores

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.pairs.iloc[idx]
        slide_id = str(row["slide_id"])
        sample_id = str(row["sample_id"])

        embeddings, _, _ = self.embedding_store.load(slide_id)
        embeddings = torch.from_numpy(np.asarray(embeddings)).float()
        target = torch.from_numpy(
            self.pathway_scores.loc[sample_id].to_numpy(dtype=np.float32)
        )

        return {
            "case_id": str(row["case_id"]),
            "slide_id": slide_id,
            "sample_id": sample_id,
            "embeddings": embeddings,
            "target": target,
        }
