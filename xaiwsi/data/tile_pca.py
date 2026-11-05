"""Fit a PCA of tile embeddings on the training split, as a fixed projection for tile models.

Uses the cohorts and split of a ``CohortPathwayDataModule`` config
(``build_pathway_splits``), so only training-split slides of the chosen training cohorts are
seen (no val/test tiles). Up to ``--tiles-per-slide`` random tiles per slide are pooled.

    uv run python -m xaiwsi.data.tile_pca --config configs/training/pathway_cohorts.yaml \\
        --train-cohorts tcga-luad --components 64 \\
        --output data/processed/tile-pca/h-optimus-1-fp16/tcga-luad_pca64.npz

Writes the ``.npz`` (``mean``, ``components`` k x D, ``explained_variance``,
``explained_variance_ratio``) and a ``.json`` next to it with the slides and settings.
``xaiwsi.models.nca.FixedProjection`` reads the ``.npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--train-cohorts", nargs="+", required=True)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--tiles-per-slide", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, _ = select_cohorts(train_specs, test_specs, args.train_cohorts)
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )

    rng = np.random.default_rng(args.seed)
    tiles, slides = [], []
    for part in splits["train"]:
        for slide_id in tqdm(sorted(set(part.pairs["slide_id"].astype(str))), desc=part.cohort):
            embeddings = np.asarray(part.store.load(slide_id)[0], dtype=np.float32)
            n = min(args.tiles_per_slide, len(embeddings))
            tiles.append(embeddings[np.sort(rng.choice(len(embeddings), n, replace=False))])
            slides.append(f"{part.cohort}/{slide_id}")
    x = np.concatenate(tiles)
    pca = PCA(n_components=args.components, random_state=args.seed).fit(x)
    print(
        f"{len(x)} tiles from {len(slides)} slides; variance explained by {args.components} "
        f"components: {pca.explained_variance_ratio_.sum():.3f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "train_cohorts": args.train_cohorts,
                "embedding": embedding,
                "components": args.components,
                "tiles_per_slide": args.tiles_per_slide,
                "seed": args.seed,
                "n_tiles": len(x),
                "slides": slides,
            },
            indent=2,
        )
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
