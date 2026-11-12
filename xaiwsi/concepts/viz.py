"""Read SAE concepts: which concepts carry which Hallmarks, their top tiles and slide maps.

1. Concept ranking: the Hallmark ridge probe of ``cohort_probe`` (clipped concept bag,
   standardised, RidgeCV) is fitted on the training split of ``--condition``; per requested
   pathway the concepts with the largest positive and negative standardised coefficients
   are kept, with their training-split Spearman against that pathway. ``--site-concepts``
   adds the concepts with the highest TCGA-site F (``xaiwsi.evaluation.site_mask`` csv).
2. Top tiles: per concept, the most activated tiles of the training cohort and of each test
   cohort (per-tile codes from ``xaiwsi.concepts.bags``, at most one tile per slide), read from
   the WSIs.
3. Slide maps: tile activation of each pathway's top concept on the slides of each cohort with
   the highest slide activation ā.

    uv run python -m xaiwsi.concepts.viz --config configs/training/pathway_cohorts.yaml \\
        --condition tcga-luad --sae tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --pathways HALLMARK_E2F_TARGETS HALLMARK_INTERFERON_GAMMA_RESPONSE \\
        --site-concepts 4 --output-dir outputs/concept_viz/luad_valpick

Writes ``concepts.csv`` (the ranking), ``tiles/<group>.jpg`` (one row per concept and cohort)
and ``maps/<pathway>.jpg``. Reading aid only: rankings come from training data, and test
cohorts are only shown.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from tqdm import tqdm

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import TablePrep
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args
from xaiwsi.visualization.pathways import (
    load_thumbnail,
    rasterize_tile_values_to_thumbnail,
    render_heatmap_overlay,
)


def tile_layout(embedding_dir: Path, slide_id: str) -> tuple[np.ndarray, dict]:
    """Tile coordinates and store attributes without reading the features."""
    with h5py.File(embedding_dir / "h5_files" / f"{slide_id}.h5", "r") as f:
        return f["coords"][...].astype(np.int64), {k: f.attrs[k] for k in f.attrs}


def tile_activations(bag_dir: Path, slide_id: str, concept: int) -> np.ndarray:
    """Activation of one concept on every tile of a slide (0 where not in the top-k)."""
    codes = np.load(bag_dir / "tiles" / f"{slide_id}.npz")
    indices, values = codes["indices"].astype(np.int64), codes["values"].astype(np.float32)
    return np.where(indices == concept, values, 0.0).sum(axis=1)


def top_tiles(bag_dir: Path, slide_ids: list[str], concepts: list[int], n: int) -> dict[int, list]:
    """Per concept, the n most activated tiles over the slides, at most one per slide."""
    best: dict[int, list] = {c: [] for c in concepts}
    wanted = np.array(concepts)
    for slide_id in tqdm(slide_ids, desc=bag_dir.parent.parent.name):
        codes = np.load(bag_dir / "tiles" / f"{slide_id}.npz")
        indices, values = codes["indices"].astype(np.int64), codes["values"].astype(np.float32)
        for c in wanted:
            hit = indices == c
            if not hit.any():
                continue
            act = np.where(hit, values, 0.0).sum(axis=1)
            tile = int(act.argmax())
            best[int(c)].append((float(act[tile]), slide_id, tile))
    return {c: sorted(rows, reverse=True)[:n] for c, rows in best.items()}


def read_tiles(rows: list, embedding_dir: Path, slides_root: Path, size: int) -> list[Image.Image]:
    import openslide

    def one(row) -> Image.Image:
        _, slide_id, tile = row
        coords, attrs = tile_layout(embedding_dir, slide_id)
        patch = int(attrs["patch_size"])
        with openslide.OpenSlide(str(slides_root / str(attrs["relative_wsi_path"]))) as slide:
            region = slide.read_region((int(coords[tile, 0]), int(coords[tile, 1])), 0, (patch, patch))
        return region.convert("RGB").resize((size, size), Image.BILINEAR)

    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, rows))


def tile_panel(rows: list[tuple[str, list[Image.Image]]], size: int, columns: int) -> Image.Image:
    label_width = 260
    canvas = Image.new("RGB", (label_width + columns * size, len(rows) * (size + 4)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (label, tiles) in enumerate(rows):
        y = r * (size + 4)
        draw.multiline_text((4, y + 4), label, fill="black")
        for i, tile in enumerate(tiles[:columns]):
            canvas.paste(tile, (label_width + i * size, y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True, help="one training cohort of the config")
    parser.add_argument("--sae", required=True, help="bag folder name under <cohort dir>/concepts/ (with tiles/)")
    parser.add_argument("--pathways", nargs="+", required=True)
    parser.add_argument("--per-pathway", type=int, default=3, help="concepts per sign and pathway")
    parser.add_argument("--site-concepts", type=int, default=0)
    parser.add_argument("--tiles", type=int, default=8, help="tiles per concept and cohort")
    parser.add_argument("--tile-size", type=int, default=112)
    parser.add_argument("--map-slides", type=int, default=2, help="slides per cohort in the maps")
    parser.add_argument("--map-size", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (args.output_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "maps").mkdir(parents=True, exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, test_used = select_cohorts(train_specs, test_specs, [args.condition])
    gene_sets = load_hallmarks(data["hallmark_gmt"])
    names = list(gene_sets)
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [c.resolve(processed_root, embedding) for c in test_used],
        gene_sets,
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    cohorts = {}
    for spec in train_used + test_used:
        resolved = spec.resolve(processed_root, embedding)
        cohort_dir = resolved.embedding_dir.parent.parent
        bags = pd.read_parquet(cohort_dir / "concepts" / args.sae / "slides_mean.parquet")
        bags.index = bags.index.astype(str)
        cohorts[spec.name] = {
            "embedding_dir": resolved.embedding_dir,
            "slides_root": Path(spec.slides_root_dir),
            "bag_dir": cohort_dir / "concepts" / args.sae,
            "bags": bags,
        }

    # 1. Ranking on the training split.
    train = splits["train"][0]
    slide_ids = train.pairs["slide_id"].astype(str)
    x_raw = cohorts[train.cohort]["bags"].loc[slide_ids].to_numpy()
    y = train.scores.loc[train.pairs["sample_id"].astype(str), names].to_numpy()
    prep = TablePrep(x_raw, 0.0, "clip")
    model = make_probe().fit(prep(x_raw), y)
    coef = model[-1].coef_  # targets x concepts, standardised inputs
    concept_names = list(cohorts[train.cohort]["bags"].columns)
    rows = []
    for pathway in args.pathways:
        p = names.index(pathway)
        order = np.argsort(coef[p])
        for sign, picked in (("+", order[::-1][: args.per_pathway]), ("-", order[: args.per_pathway])):
            for rank, k in enumerate(picked, start=1):
                rho = spearmanr(x_raw[:, k], y[:, p]).statistic
                rows.append({"group": pathway, "sign": sign, "rank": rank, "concept": int(k),
                             "coef": coef[p, k], "train_spearman": rho})
    if args.site_concepts:
        site = pd.read_csv(cohorts[train.cohort]["bag_dir"] / "slides_mean_sitemask.csv")
        for rank, column in enumerate(site["column"].head(args.site_concepts), start=1):
            rows.append({"group": "site", "sign": "", "rank": rank, "concept": concept_names.index(column),
                         "coef": np.nan, "train_spearman": np.nan,
                         "site_f": float(site.loc[site["column"] == column, "f"].iloc[0])})
    table = pd.DataFrame(rows)
    table["active_train"] = [(x_raw[:, k] > 0).mean() for k in table["concept"]]
    table.to_csv(args.output_dir / "concepts.csv", index=False)
    print(table.round(3).to_string(index=False))

    # 2. Top tiles per concept and cohort (all slides of the cohort with bags).
    concepts = sorted(set(table["concept"]))
    best = {
        name: top_tiles(c["bag_dir"], list(c["bags"].index), concepts, args.tiles) for name, c in cohorts.items()
    }
    for group, frame in table.groupby("group", sort=False):
        panel_rows = []
        for r in frame.itertuples():
            tag = f"{r.sign}{r.rank}" if r.sign else f"#{r.rank}"
            detail = f"coef {r.coef:+.3f} rho {r.train_spearman:+.2f}" if group != "site" else f"site F {r.site_f:.1f}"
            for name, c in cohorts.items():
                tiles = read_tiles(best[name][r.concept], c["embedding_dir"], c["slides_root"], args.tile_size)
                panel_rows.append((f"c{r.concept} {tag}\n{detail}\n{name}", tiles))
        tile_panel(panel_rows, args.tile_size, args.tiles).save(args.output_dir / "tiles" / f"{group}.jpg", quality=88)

    # 3. Maps of each pathway's top positive concept on the highest-ā slides per cohort.
    for pathway in args.pathways:
        k = int(table[(table["group"] == pathway) & (table["sign"] == "+")].iloc[0]["concept"])
        maps = []
        for name, c in cohorts.items():
            column = c["bags"].iloc[:, k]
            for slide_id in column.sort_values(ascending=False).index[: args.map_slides]:
                coords, attrs = tile_layout(c["embedding_dir"], slide_id)
                act = tile_activations(c["bag_dir"], slide_id, k)
                thumbnail = load_thumbnail(c["slides_root"] / str(attrs["relative_wsi_path"]), args.map_size)
                heat = rasterize_tile_values_to_thumbnail(coords, act, attrs, thumbnail)
                maps.append((f"{name} {slide_id[:23]} mean act {column[slide_id]:.3f}", thumbnail, heat, act))
        vmax = float(np.percentile(np.concatenate([m[3][m[3] > 0] for m in maps]), 99))
        panels = []
        for label, thumbnail, heat, _ in maps:
            overlay = render_heatmap_overlay(thumbnail, heat, kind="weight", alpha=0.6, vmax=vmax).convert("RGB")
            pair = Image.new("RGB", (thumbnail.width * 2, thumbnail.height + 20), "white")
            pair.paste(thumbnail, (0, 20))
            pair.paste(overlay, (thumbnail.width, 20))
            ImageDraw.Draw(pair).text((4, 4), f"c{k} ({pathway.removeprefix('HALLMARK_')})  {label}", fill="black")
            panels.append(pair)
        canvas = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
        y0 = 0
        for p in panels:
            canvas.paste(p, (0, y0))
            y0 += p.height
        canvas.save(args.output_dir / "maps" / f"{pathway.removeprefix('HALLMARK_')}.jpg", quality=85)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
