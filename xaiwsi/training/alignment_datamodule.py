from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.alignment import SlideRNADataset, collate_bags, load_rna_embeddings
from xaiwsi.data.pathway import build_case_pairs, split_cases


class AlignmentDataModule(L.LightningDataModule):
    """Whole-slide tile bags paired with frozen RNA embeddings.

    TCGA train/val (same pairing and split as ``PathwayDataModule`` for the same
    ``val_fraction``/``split_seed``), CPTAC as the external test cohort. RNA features
    are standardized with TCGA-train statistics. With ``rna_center_per_cohort``, CPTAC
    is instead centered on its own mean (label-free, transductive), which targets the
    TCGA -> CPTAC offset shift seen in stage 2b.
    """

    def __init__(
        self,
        tcga_rna_dir: str,
        cptac_rna_dir: str,
        tcga_rna_embeddings: str,
        cptac_rna_embeddings: str,
        tcga_embedding_dir: str,
        cptac_embedding_dir: str,
        tcga_wsi_metadata: str,
        cptac_wsi_metadata: str,
        tcga_slides_root_dir: str | None = None,
        cptac_slides_root_dir: str | None = None,
        rna_column_regex: str = r"^emb_",
        rna_center_per_cohort: bool = False,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        batch_size: int = 64,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.tcga_rna_dir = Path(tcga_rna_dir)
        self.cptac_rna_dir = Path(cptac_rna_dir)
        self.tcga_rna_embeddings = Path(tcga_rna_embeddings)
        self.cptac_rna_embeddings = Path(cptac_rna_embeddings)
        self.tcga_embedding_dir = Path(tcga_embedding_dir)
        self.cptac_embedding_dir = Path(cptac_embedding_dir)
        self.tcga_wsi_metadata = Path(tcga_wsi_metadata)
        self.cptac_wsi_metadata = Path(cptac_wsi_metadata)
        # Unused for training; kept in the run config for tile-map visualization.
        self.tcga_slides_root_dir = tcga_slides_root_dir
        self.cptac_slides_root_dir = cptac_slides_root_dir
        self.rna_column_regex = rna_column_regex
        self.rna_center_per_cohort = rna_center_per_cohort
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.rna_dim: int | None = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        if self._is_setup:
            return

        tcga_store = build_embedding_store_from_dir(root_dir=self.tcga_embedding_dir)
        cptac_store = build_embedding_store_from_dir(root_dir=self.cptac_embedding_dir)
        tcga_rna = load_rna_embeddings(self.tcga_rna_embeddings, self.rna_column_regex)
        cptac_rna = load_rna_embeddings(self.cptac_rna_embeddings, self.rna_column_regex)
        if list(tcga_rna.columns) != list(cptac_rna.columns):
            raise ValueError("TCGA and CPTAC RNA embeddings have different columns.")

        # build_case_pairs keeps RNA samples present in the expression columns.
        tcga_pairs = build_case_pairs(
            tcga_store,
            pd.read_csv(self.tcga_wsi_metadata),
            pd.read_csv(self.tcga_rna_dir / "samples.csv"),
            tcga_rna.T,
        )
        cptac_pairs = build_case_pairs(
            cptac_store,
            pd.read_csv(self.cptac_wsi_metadata),
            pd.read_csv(self.cptac_rna_dir / "samples.csv"),
            cptac_rna.T,
        )
        train_pairs, val_pairs = split_cases(
            tcga_pairs, val_fraction=self.val_fraction, seed=self.split_seed
        )

        train_rna = tcga_rna.loc[train_pairs["sample_id"]]
        mean, std = train_rna.mean(), train_rna.std().replace(0.0, 1.0)
        cptac_mean = (
            cptac_rna.loc[cptac_pairs["sample_id"]].mean()
            if self.rna_center_per_cohort
            else mean
        )
        tcga_rna = (tcga_rna - mean) / std
        cptac_rna = (cptac_rna - cptac_mean) / std
        self.rna_dim = tcga_rna.shape[1]

        self.train_dataset = SlideRNADataset(tcga_store, train_pairs, tcga_rna)
        self.val_dataset = SlideRNADataset(tcga_store, val_pairs, tcga_rna)
        self.test_dataset = SlideRNADataset(cptac_store, cptac_pairs, cptac_rna)
        print(
            f"Alignment: {len(self.train_dataset)} TCGA train, "
            f"{len(self.val_dataset)} TCGA val, {len(self.test_dataset)} CPTAC test "
            f"cases; RNA dim {self.rna_dim}."
        )
        self._is_setup = True

    def _loader(self, dataset, *, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size if train else self.eval_batch_size,
            shuffle=train,
            # Keep the number of in-batch negatives fixed.
            drop_last=train,
            collate_fn=collate_bags,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, train=False)
