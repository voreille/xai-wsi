"""Cluster a sample of tile embeddings across cohorts and render tiles per cluster.

First step of tile QC: find which embedding clusters are background, pen marks, blur,
folds and so on, by looking at the tiles, and how often they occur per cohort. Tiles are
sampled uniformly per slide from the cohorts of a ``CohortPathwayDataModule`` config,
reduced with PCA and clustered with k-means.

    uv run python -m xaiwsi.qc.tile_clusters \\
        --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad tcga-lusc tcga-brca cptac-luad cptac-lusc \\
        --output-dir outputs/tile_qc/clusters

Writes to ``--output-dir``:
- ``tiles.parquet``: one row per sampled tile (cohort, slide_id, tile index, level-0 x/y,
  cluster, distance to the centroid).
- ``model.npz``: PCA mean/components and k-means centroids, to assign every tile later.
- ``cluster_cohorts.csv``: fraction of each cohort's sampled tiles per cluster.
- ``overview.png``: one row of tiles per cluster; ``montages/cluster_<k>.png``: a grid of
  tiles from different slides for each cluster.

``--refine 18 36 --refine-k 10`` re-clusters the tiles of the listed clusters (artifact
candidates found in ``overview.png``) into finer sub-clusters: column ``sub`` of
``tiles.parquet`` (−1 elsewhere) and ``refine_overview.png``. The labels given to
clusters and sub-clusters after looking at the images go in a YAML file for
``xaiwsi.qc.tile_classifier``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.training.run_config import data_args


def cohort_specs(config: dict) -> dict[str, CohortSpec]:
    data = data_args(config)
    return {s["name"]: CohortSpec(**s) for s in data["train_cohorts"] + data["test_cohorts"]}


def sample_tiles(
    store, slide_ids: list[str], tiles_per_slide: int, rng: np.random.Generator
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings, rows = [], []
    for slide_id in slide_ids:
        emb, coords, attrs = store.load(slide_id)
        n = len(coords)
        idx = np.sort(rng.choice(n, size=min(tiles_per_slide, n), replace=False))
        embeddings.append(np.asarray(emb[idx], dtype=np.float32))
        rows.append(
            pd.DataFrame(
                {
                    "slide_id": slide_id,
                    "tile": idx,
                    "x": np.asarray(coords)[idx, 0],
                    "y": np.asarray(coords)[idx, 1],
                    "patch_size": int(attrs["patch_size"]),
                    "relative_wsi_path": str(attrs["relative_wsi_path"]),
                    "n_tiles": n,
                }
            )
        )
    return np.concatenate(embeddings), pd.concat(rows, ignore_index=True)


def read_tile(row, slides_root: Path, size: int) -> Image.Image:
    import openslide

    with openslide.OpenSlide(str(slides_root / row.relative_wsi_path)) as slide:
        region = slide.read_region((int(row.x), int(row.y)), 0, (row.patch_size, row.patch_size))
    return region.convert("RGB").resize((size, size), Image.BILINEAR)


def render(rows: pd.DataFrame, roots: dict[str, Path], size: int, columns: int, label: str | None = None):
    with ThreadPoolExecutor(16) as pool:
        tiles = list(pool.map(lambda r: read_tile(r, roots[r.cohort], size), rows.itertuples()))
    n_rows = max(1, int(np.ceil(len(tiles) / columns)))
    offset = 120 if label else 0
    canvas = Image.new("RGB", (offset + columns * size, n_rows * size), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (offset + (i % columns) * size, (i // columns) * size))
    if label:
        ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def pick_spread(group: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Up to n tiles of a cluster, at most one per slide when possible (random order)."""
    shuffled = group.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    first = shuffled.drop_duplicates("slide_id")
    if len(first) >= n:
        return first.head(n)
    return pd.concat([first, shuffled.drop(first.index)]).head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--slides-root", nargs="*", default=[], help="'<cohort>=<dir>' overrides")
    parser.add_argument("--slides-per-cohort", type=int, default=60)
    parser.add_argument("--tiles-per-slide", type=int, default=200)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--montage-tiles", type=int, default=48)
    parser.add_argument("--tile-px", type=int, default=96)
    parser.add_argument("--refine", type=int, nargs="*", default=[], help="clusters to sub-cluster")
    parser.add_argument("--refine-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    (args.output_dir / "montages").mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text())
    data = data_args(config)
    specs = cohort_specs(config)
    overrides = dict(entry.split("=", 1) for entry in args.slides_root)
    roots, embeddings, frames = {}, [], []
    for name in args.cohorts:
        cohort = specs[name].resolve(data.get("processed_root"), data.get("embedding", "h-optimus-1"))
        roots[name] = Path(overrides.get(name) or cohort.slides_root_dir)
        store = build_embedding_store_from_dir(root_dir=cohort.embedding_dir)
        slide_ids = sorted(store.slide_ids())
        chosen = list(rng.choice(slide_ids, size=min(args.slides_per_cohort, len(slide_ids)), replace=False))
        emb, rows = sample_tiles(store, chosen, args.tiles_per_slide, rng)
        rows.insert(0, "cohort", name)
        embeddings.append(emb)
        frames.append(rows)
        print(f"{name}: {len(chosen)} slides, {len(rows)} tiles")
    x = np.concatenate(embeddings)
    tiles = pd.concat(frames, ignore_index=True)

    pca = PCA(n_components=args.pca_dim, random_state=args.seed).fit(x)
    z = pca.transform(x)
    kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=5, batch_size=4096).fit(z)
    tiles["cluster"] = kmeans.labels_
    tiles["distance"] = np.linalg.norm(z - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    print(f"PCA {args.pca_dim}: {pca.explained_variance_ratio_.sum():.1%} of variance")
    tiles["sub"] = -1
    if args.refine:
        mask = tiles["cluster"].isin(args.refine).to_numpy()
        sub = KMeans(n_clusters=args.refine_k, random_state=args.seed, n_init=5).fit_predict(z[mask])
        tiles.loc[mask, "sub"] = sub

    tiles.to_parquet(args.output_dir / "tiles.parquet", index=False)
    np.savez(
        args.output_dir / "model.npz",
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        centroids=kmeans.cluster_centers_,
    )
    shares = pd.crosstab(tiles["cluster"], tiles["cohort"], normalize="columns").round(4)
    shares.insert(0, "n_tiles", tiles["cluster"].value_counts().sort_index())
    shares.insert(1, "n_slides", tiles.groupby("cluster")["slide_id"].nunique())
    shares.to_csv(args.output_dir / "cluster_cohorts.csv")
    print(shares.to_string())

    rows_png = []
    for k, group in tiles.groupby("cluster"):
        render(pick_spread(group, args.montage_tiles, rng), roots, args.tile_px, 8).save(
            args.output_dir / "montages" / f"cluster_{k:02d}.png"
        )
        label = f"c{k} n={len(group)}\nslides={group['slide_id'].nunique()}"
        rows_png.append(render(pick_spread(group, 10, rng), roots, 64, 10, label=label))
    overview = Image.new("RGB", (rows_png[0].width, sum(r.height for r in rows_png)), "white")
    top = 0
    for row in rows_png:
        overview.paste(row, (0, top))
        top += row.height
    overview.save(args.output_dir / "overview.png")

    if args.refine:
        refined = tiles[tiles["sub"] >= 0]
        rows_png = []
        for k, group in refined.groupby("sub"):
            parents = ",".join(f"c{c}:{n}" for c, n in group["cluster"].value_counts().items())
            label = f"s{k} n={len(group)}\nslides={group['slide_id'].nunique()}\n{parents}"
            rows_png.append(render(pick_spread(group, 12, rng), roots, 72, 12, label=label))
        overview = Image.new("RGB", (rows_png[0].width, sum(r.height for r in rows_png)), "white")
        top = 0
        for row in rows_png:
            overview.paste(row, (0, top))
            top += row.height
        overview.save(args.output_dir / "refine_overview.png")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
