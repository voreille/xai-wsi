from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.pathway import (
    PathwayEmbeddingDataset,
    build_case_pairs,
    load_log2_tpm_gene_symbols,
    split_cases,
)
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import fit_gene_zscore_stats, score_mean_z


class PathwayDataModule(L.LightningDataModule):
    """TCGA train/validation with CPTAC as an external test cohort."""

    def __init__(
        self,
        tcga_rna_dir: str,
        cptac_rna_dir: str,
        tcga_embedding_dir: str,
        cptac_embedding_dir: str,
        tcga_wsi_metadata: str,
        cptac_wsi_metadata: str,
        hallmark_gmt: str,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.tcga_rna_dir = Path(tcga_rna_dir)
        self.cptac_rna_dir = Path(cptac_rna_dir)
        self.tcga_embedding_dir = Path(tcga_embedding_dir)
        self.cptac_embedding_dir = Path(cptac_embedding_dir)
        self.tcga_wsi_metadata = Path(tcga_wsi_metadata)
        self.cptac_wsi_metadata = Path(cptac_wsi_metadata)
        self.hallmark_gmt = Path(hallmark_gmt)

        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if self.batch_size != 1:
            raise ValueError(
                "The baseline currently requires batch_size=1 because WSIs "
                "contain variable numbers of tiles."
            )

        self.pathway_names: list[str] = []
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        tcga_store = build_embedding_store_from_dir(root_dir=self.tcga_embedding_dir)
        cptac_store = build_embedding_store_from_dir(root_dir=self.cptac_embedding_dir)

        tcga_expression = load_log2_tpm_gene_symbols(self.tcga_rna_dir)
        cptac_expression = load_log2_tpm_gene_symbols(self.cptac_rna_dir)

        tcga_pairs = build_case_pairs(
            tcga_store,
            pd.read_csv(self.tcga_wsi_metadata),
            pd.read_csv(self.tcga_rna_dir / "samples.csv"),
            tcga_expression,
        )
        cptac_pairs = build_case_pairs(
            cptac_store,
            pd.read_csv(self.cptac_wsi_metadata),
            pd.read_csv(self.cptac_rna_dir / "samples.csv"),
            cptac_expression,
        )

        print(f"TCGA paired cases:  {len(tcga_pairs)}")
        print(f"CPTAC paired cases: {len(cptac_pairs)}")

        train_pairs, val_pairs = split_cases(
            tcga_pairs,
            val_fraction=self.val_fraction,
            seed=self.split_seed,
        )

        hallmarks = load_hallmarks(self.hallmark_gmt)
        self.pathway_names = list(hallmarks.keys())

        train_sample_ids = train_pairs["sample_id"].tolist()
        val_sample_ids = val_pairs["sample_id"].tolist()
        cptac_sample_ids = cptac_pairs["sample_id"].tolist()

        # Fit gene-wise z-score statistics on TCGA training samples only.
        zscore_stats = fit_gene_zscore_stats(tcga_expression[train_sample_ids])

        train_scores = score_mean_z(
            tcga_expression[train_sample_ids],
            hallmarks,
            zscore_stats=zscore_stats,
        )
        val_scores = score_mean_z(
            tcga_expression[val_sample_ids],
            hallmarks,
            zscore_stats=zscore_stats,
        )
        cptac_scores = score_mean_z(
            cptac_expression[cptac_sample_ids],
            hallmarks,
            zscore_stats=zscore_stats,
        )

        if list(train_scores.columns) != self.pathway_names:
            raise RuntimeError("Unexpected Hallmark ordering in training scores.")
        if list(val_scores.columns) != self.pathway_names:
            raise RuntimeError("TCGA validation Hallmark ordering differs.")
        if list(cptac_scores.columns) != self.pathway_names:
            raise RuntimeError("CPTAC Hallmark ordering differs.")

        self.train_dataset = PathwayEmbeddingDataset(tcga_store, train_pairs, train_scores)
        self.val_dataset = PathwayEmbeddingDataset(tcga_store, val_pairs, val_scores)
        self.test_dataset = PathwayEmbeddingDataset(cptac_store, cptac_pairs, cptac_scores)

        print(
            f"Training on {len(self.train_dataset)} TCGA cases, "
            f"validating on {len(self.val_dataset)} TCGA cases, "
            f"testing on {len(self.test_dataset)} CPTAC cases."
        )

        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
