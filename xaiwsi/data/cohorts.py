"""Cohorts for WSI -> pathway training: paths, case pairing, splits and mean-z targets.

A cohort is one processed dataset under ``<processed_root>/<name>/`` with the layout

    rna/{tpm.parquet,genes.csv,samples.csv}
    metadata/wsi_slides.csv
    wsi-embeddings/<embedding>/

Explicit paths override the layout (older configs point at arbitrary directories).

Targets are gene-wise z-scored mean-z scores. The z-score statistics are always fit on
training-split samples only. With ``target_normalization="per_cohort"`` each training
cohort gets its own statistics, so targets are relative to that cohort and a model can't
lower the loss by telling the cohorts apart; test cohorts use the statistics of their
``zscore_reference`` training cohort. With ``"pooled"`` one set of statistics is fit on
all training samples together and used everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from wsitools.storage.factory import build_embedding_store_from_dir
from wsitools.storage.interfaces import EmbeddingStore

from xaiwsi.data.pathway import build_case_pairs, load_log2_tpm_gene_symbols, split_cases
from xaiwsi.rna.pathways.scoring import fit_gene_zscore_stats, score_mean_z

TargetNormalization = Literal["per_cohort", "pooled"]


@dataclass
class CohortSpec:
    """One cohort. Paths default to the processed-data layout under ``root_dir``."""

    name: str
    root_dir: str | None = None
    slides_root_dir: str | None = None
    # Test cohorts with per-cohort normalization: the training cohort whose z-score
    # statistics define their targets. Not needed with a single training cohort.
    zscore_reference: str | None = None
    rna_dir: str | None = None
    wsi_metadata: str | None = None
    embedding_dir: str | None = None

    def resolve(self, processed_root: str | Path | None, embedding: str) -> ResolvedCohort:
        root = Path(self.root_dir) if self.root_dir else None
        if root is None and processed_root is not None:
            root = Path(processed_root) / self.name

        def path(explicit: str | None, relative: str) -> Path:
            if explicit:
                return Path(explicit)
            if root is None:
                raise ValueError(
                    f"Cohort {self.name!r}: set root_dir, processed_root or an explicit "
                    f"path for {relative!r}."
                )
            return root / relative

        return ResolvedCohort(
            name=self.name,
            rna_dir=path(self.rna_dir, "rna"),
            wsi_metadata=path(self.wsi_metadata, "metadata/wsi_slides.csv"),
            embedding_dir=path(self.embedding_dir, f"wsi-embeddings/{embedding}"),
            slides_root_dir=Path(self.slides_root_dir) if self.slides_root_dir else None,
            zscore_reference=self.zscore_reference,
        )


@dataclass(frozen=True)
class ResolvedCohort:
    name: str
    rna_dir: Path
    wsi_metadata: Path
    embedding_dir: Path
    slides_root_dir: Path | None
    zscore_reference: str | None


@dataclass
class LoadedCohort:
    cohort: ResolvedCohort
    store: EmbeddingStore
    expression: pd.DataFrame  # gene_symbol x sample, log2(TPM + 1)
    pairs: pd.DataFrame  # case_id, slide_id, sample_id


@dataclass
class CohortSplit:
    """Pairs of one cohort in one split, with their targets (sample_id x pathway)."""

    cohort: str
    store: EmbeddingStore
    pairs: pd.DataFrame
    scores: pd.DataFrame


def select_cohorts(
    train_cohorts: list[CohortSpec],
    test_cohorts: list[CohortSpec],
    train_names: list[str],
) -> tuple[list[CohortSpec], list[CohortSpec]]:
    """Keep the training cohorts named in `train_names` (in that order) and the test
    cohorts whose `zscore_reference` is unset or one of them."""
    by_name = {c.name: c for c in train_cohorts}
    unknown = [n for n in train_names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown training cohorts {unknown}; have {list(by_name)}.")
    test = [c for c in test_cohorts if c.zscore_reference in (None, *train_names)]
    return [by_name[n] for n in train_names], test


def load_cohort(cohort: ResolvedCohort) -> LoadedCohort:
    store = build_embedding_store_from_dir(root_dir=cohort.embedding_dir)
    expression = load_log2_tpm_gene_symbols(cohort.rna_dir)
    print(f"[{cohort.name}]")
    pairs = build_case_pairs(
        store,
        pd.read_csv(cohort.wsi_metadata),
        pd.read_csv(cohort.rna_dir / "samples.csv"),
        expression,
    )
    print(f"[{cohort.name}] paired cases: {len(pairs)}")
    return LoadedCohort(cohort=cohort, store=store, expression=expression, pairs=pairs)


def build_pathway_splits(
    train_cohorts: list[ResolvedCohort],
    test_cohorts: list[ResolvedCohort],
    gene_sets: Mapping[str, set[str]],
    *,
    val_fraction: float,
    split_seed: int,
    train_fraction: float = 1.0,
    train_subset_seed: int = 0,
    target_normalization: TargetNormalization = "per_cohort",
) -> dict[str, list[CohortSplit]]:
    """Pair, split and score all cohorts. Returns {"train"|"val"|"test": [CohortSplit]}.

    Each training cohort is split into train/val with ``split_cases`` (same seed for all,
    so a cohort's split doesn't depend on which other cohorts are trained with it).
    ``train_fraction`` subsamples every training split (learning curves); val and test
    stay fixed and z-score statistics are fit on the subsample only.
    """
    if not train_cohorts:
        raise ValueError("At least one training cohort is required.")
    names = [c.name for c in train_cohorts + test_cohorts]
    if len(set(names)) != len(names):
        raise ValueError(f"Cohort names must be unique, got {names}.")
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}.")

    train_loaded = [load_cohort(c) for c in train_cohorts]
    test_loaded = [load_cohort(c) for c in test_cohorts]

    split_pairs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for loaded in train_loaded:
        train_pairs, val_pairs = split_cases(loaded.pairs, val_fraction=val_fraction, seed=split_seed)
        if train_fraction < 1.0:
            train_pairs = train_pairs.sample(
                n=max(int(round(train_fraction * len(train_pairs))), 1),
                random_state=train_subset_seed,
            )
        split_pairs[loaded.cohort.name] = (train_pairs, val_pairs)

    def train_expression(loaded: LoadedCohort) -> pd.DataFrame:
        return loaded.expression[split_pairs[loaded.cohort.name][0]["sample_id"].tolist()]

    if target_normalization == "per_cohort":
        stats = {loaded.cohort.name: fit_gene_zscore_stats(train_expression(loaded)) for loaded in train_loaded}
    elif target_normalization == "pooled":
        pooled = fit_gene_zscore_stats(
            pd.concat([train_expression(loaded) for loaded in train_loaded], axis=1, join="inner")
        )
        stats = {loaded.cohort.name: pooled for loaded in train_loaded}
    else:
        raise ValueError(f"Unknown target_normalization {target_normalization!r}.")

    def reference(cohort: ResolvedCohort) -> str:
        if cohort.zscore_reference is not None:
            if cohort.zscore_reference not in stats:
                raise ValueError(
                    f"zscore_reference {cohort.zscore_reference!r} of {cohort.name!r} is not "
                    f"a training cohort."
                )
            return cohort.zscore_reference
        if target_normalization == "pooled" or len(train_cohorts) == 1:
            return train_cohorts[0].name
        raise ValueError(
            f"Test cohort {cohort.name!r} needs zscore_reference (one of "
            f"{[c.name for c in train_cohorts]}) with per-cohort normalization."
        )

    pathway_names = list(gene_sets)

    def make_split(loaded: LoadedCohort, pairs: pd.DataFrame, reference_name: str) -> CohortSplit:
        scores = score_mean_z(
            loaded.expression[pairs["sample_id"].tolist()],
            gene_sets,
            zscore_stats=stats[reference_name],
        )
        if list(scores.columns) != pathway_names:
            raise RuntimeError(
                f"[{loaded.cohort.name}] pathway columns differ from the gene sets "
                f"(gene sets dropped by min_size?)."
            )
        return CohortSplit(
            cohort=loaded.cohort.name,
            store=loaded.store,
            pairs=pairs.reset_index(drop=True),
            scores=scores,
        )

    splits: dict[str, list[CohortSplit]] = {"train": [], "val": [], "test": []}
    for loaded in train_loaded:
        name = loaded.cohort.name
        train_pairs, val_pairs = split_pairs[name]
        splits["train"].append(make_split(loaded, train_pairs, name))
        splits["val"].append(make_split(loaded, val_pairs, name))
    for loaded in test_loaded:
        splits["test"].append(make_split(loaded, loaded.pairs, reference(loaded.cohort)))

    for split, parts in splits.items():
        print(f"{split}: " + ", ".join(f"{p.cohort} {len(p.pairs)}" for p in parts))
    return splits
