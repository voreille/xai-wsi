from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from xaiwsi.data.cohorts import (
    CohortSpec,
    TargetNormalization,
    build_pathway_splits,
    select_cohorts,
)
from xaiwsi.data.pathway import PathwayEmbeddingDataset
from xaiwsi.rna.pathways.hallmarks import load_hallmarks


class PathwayDataModule(L.LightningDataModule):
    """TCGA train/validation with CPTAC as an external test cohort (two-cohort configs).

    Kept as the CLI's base data module so that run configs without ``data.class_path``
    still load. New configs use ``CohortPathwayDataModule``. Here the cohorts are named
    ``tcga`` and ``cptac``.
    """

    def __init__(
        self,
        tcga_rna_dir: str,
        cptac_rna_dir: str,
        tcga_embedding_dir: str,
        cptac_embedding_dir: str,
        tcga_slides_root_dir: str,
        cptac_slides_root_dir: str,
        tcga_wsi_metadata: str,
        cptac_wsi_metadata: str,
        hallmark_gmt: str,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        train_fraction: float = 1.0,
        train_subset_seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self._configure(
            train_cohorts=[
                CohortSpec(
                    name="tcga",
                    rna_dir=tcga_rna_dir,
                    wsi_metadata=tcga_wsi_metadata,
                    embedding_dir=tcga_embedding_dir,
                    slides_root_dir=tcga_slides_root_dir,
                )
            ],
            test_cohorts=[
                CohortSpec(
                    name="cptac",
                    rna_dir=cptac_rna_dir,
                    wsi_metadata=cptac_wsi_metadata,
                    embedding_dir=cptac_embedding_dir,
                    slides_root_dir=cptac_slides_root_dir,
                    zscore_reference="tcga",
                )
            ],
            processed_root=None,
            embedding="",
            hallmark_gmt=hallmark_gmt,
            target_normalization="per_cohort",
            val_fraction=val_fraction,
            split_seed=split_seed,
            train_fraction=train_fraction,
            train_subset_seed=train_subset_seed,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _configure(
        self,
        *,
        train_cohorts: list[CohortSpec],
        test_cohorts: list[CohortSpec],
        processed_root: str | None,
        embedding: str,
        hallmark_gmt: str,
        target_normalization: TargetNormalization,
        val_fraction: float,
        split_seed: int,
        train_fraction: float,
        train_subset_seed: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        self.train_cohorts = [c.resolve(processed_root, embedding) for c in train_cohorts]
        self.test_cohorts = [c.resolve(processed_root, embedding) for c in test_cohorts]
        self.hallmark_gmt = Path(hallmark_gmt)
        self.target_normalization = target_normalization

        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.train_fraction = train_fraction
        self.train_subset_seed = train_subset_seed
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

        hallmarks = load_hallmarks(self.hallmark_gmt)
        self.pathway_names = list(hallmarks.keys())

        splits = build_pathway_splits(
            self.train_cohorts,
            self.test_cohorts,
            hallmarks,
            val_fraction=self.val_fraction,
            split_seed=self.split_seed,
            train_fraction=self.train_fraction,
            train_subset_seed=self.train_subset_seed,
            target_normalization=self.target_normalization,
        )

        def dataset(split: str) -> ConcatDataset:
            return ConcatDataset(
                [
                    PathwayEmbeddingDataset(part.store, part.pairs, part.scores, cohort=part.cohort)
                    for part in splits[split]
                ]
            )

        self.train_dataset = dataset("train")
        self.val_dataset = dataset("val")
        self.test_dataset = dataset("test")

        print(
            f"Training on {len(self.train_dataset)} cases, "
            f"validating on {len(self.val_dataset)} cases, "
            f"testing on {len(self.test_dataset)} cases."
        )

        self._is_setup = True

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)


class CohortPathwayDataModule(PathwayDataModule):
    """Train/validation on one or more cohorts, external test on others.

    Every training cohort is split into train/val (``split_cases``, same seed); test
    cohorts are used whole. Cohort paths follow ``<processed_root>/<name>/`` unless set in
    the ``CohortSpec``. See ``xaiwsi.data.cohorts`` for the target normalization options.

    ``use_train_cohorts`` trains on a subset of ``train_cohorts`` by name (one config for
    LUAD-only and LUAD+LUSC runs); test cohorts whose ``zscore_reference`` is dropped are
    skipped. ``"<name>:<n>"`` also caps that cohort's training split at n cases.
    """

    def __init__(
        self,
        train_cohorts: list[CohortSpec],
        test_cohorts: list[CohortSpec],
        hallmark_gmt: str,
        use_train_cohorts: list[str] | None = None,
        processed_root: str | None = None,
        embedding: str = "h-optimus-1",
        target_normalization: TargetNormalization = "per_cohort",
        val_fraction: float = 0.2,
        split_seed: int = 42,
        train_fraction: float = 1.0,
        train_subset_seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        L.LightningDataModule.__init__(self)
        if use_train_cohorts is not None:
            train_cohorts, test_cohorts = select_cohorts(
                train_cohorts, test_cohorts, use_train_cohorts
            )
        self._configure(
            train_cohorts=train_cohorts,
            test_cohorts=test_cohorts,
            processed_root=processed_root,
            embedding=embedding,
            hallmark_gmt=hallmark_gmt,
            target_normalization=target_normalization,
            val_fraction=val_fraction,
            split_seed=split_seed,
            train_fraction=train_fraction,
            train_subset_seed=train_subset_seed,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
