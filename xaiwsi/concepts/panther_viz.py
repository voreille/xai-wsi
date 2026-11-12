"""Read PANTHER prototypes: what each prototype looks like, where it sits, and its probe maps.

Needs a PANTHER run with ``--save-tiles`` (``xaiwsi.concepts.panther``).

1. Prototype table: per prototype, mean mixture weight pi per cohort, and per requested pathway
   the standardised ridge coefficient of the pi probe (16-ish inputs, so readable) and the
   training-split Spearman of pi_c with the pathway (``prototypes.csv``).
2. Atlas: per prototype and cohort, the most typical tiles (assigned to the prototype, smallest
   distance to it, at most one per slide; ``tiles/typical.jpg``) and random assigned tiles
   (``tiles/random.jpg``), because K-means clusters in 1536 dimensions are broad and the most
   typical tiles can be the blandest. Assignment is the per-slide argmax responsibility, i.e.
   what component c pools on that slide: with the weak prior (tau = 1 tile) the components
   drift per slide, and only ~65% of tiles stay with their nearest global prototype (TCGA- and
   CPTAC-LUAD, C = 16, full dim), so the random rows are mixed.
3. Slide panels (``--map-slides`` per evaluation cohort, spread over the first pathway's true
   score): thumbnail, prototype assignment (argmax responsibility), then per pathway the tile
   maps of three ridge probes trained on the condition's training split: slide mean (w·x),
   PANTHER pimu (sum_c r_ic w_c·x_i) and PANTHER pi (sum_c r_ic w_c). All three are linear in
   tile features, so the tile average is the slide prediction (pimu and pi up to the GMM
   prior and the last M step; ``maps.csv`` gives both). Probes here are unclipped, so the
   decomposition is exact.

    uv run python -m xaiwsi.concepts.panther_viz --config configs/training/pathway_cohorts.yaml \\
        --condition tcga-luad --panther tcga-luad_c16_full_s0 \\
        --pathways HALLMARK_E2F_TARGETS HALLMARK_INTERFERON_GAMMA_RESPONSE \\
            HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION \\
        --slide-means-dir outputs/concept_sweep/tcga-luad_k1024_top64_s0 \\
        --output-dir outputs/panther_viz/tcga-luad_c16_full_s0

Reading aid only: rankings come from training data, test cohorts are only shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw
from scipy.stats import spearmanr
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.concepts.panther import Projection
from xaiwsi.concepts.viz import read_tiles, tile_layout, tile_panel, tile_scores
from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args
from xaiwsi.visualization.pathways import load_thumbnail, rasterize_tile_values_to_thumbnail, render_heatmap_overlay


def prototype_colors(n: int) -> np.ndarray:
    cmap = matplotlib.colormaps["tab20"] if n <= 20 else matplotlib.colormaps["gist_ncar"]
    return np.array([cmap(i / max(n - 1, 1) if n > 20 else i)[:3] for i in range(n)]) * 255


def assignment_overlay(thumbnail: Image.Image, coords, attrs, assign: np.ndarray, colors: np.ndarray) -> Image.Image:
    """Thumbnail with each tile painted in the colour of its prototype."""
    out = np.asarray(thumbnail.convert("RGB")).astype(np.float32).copy()
    for c in np.unique(assign):
        mask = rasterize_tile_values_to_thumbnail(coords, (assign == c).astype(float), attrs, thumbnail)
        hit = np.nan_to_num(mask) > 0.5
        out[hit] = 0.35 * out[hit] + 0.65 * colors[c]
    return Image.fromarray(out.astype(np.uint8))


def legend(colors: np.ndarray, labels: list[str], width: int) -> Image.Image:
    row = 14
    per_line = max(1, width // 120)
    canvas = Image.new("RGB", (width, row * int(np.ceil(len(labels) / per_line)) + 4), "white")
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(labels):
        x, y = (i % per_line) * 120 + 4, (i // per_line) * row + 2
        draw.rectangle((x, y + 2, x + 10, y + 12), fill=tuple(int(v) for v in colors[i]))
        draw.text((x + 14, y), label, fill="black")
    return canvas


def stack(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (max(im.width for im in images), sum(im.height for im in images)), "white")
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True, help="one training cohort of the config")
    parser.add_argument("--panther", required=True, help="run name under <cohort dir>/panther/ (with tiles/)")
    parser.add_argument("--pathways", nargs="+", required=True)
    parser.add_argument("--tiles", type=int, default=8, help="tiles per prototype and cohort")
    parser.add_argument("--tile-size", type=int, default=112)
    parser.add_argument("--map-slides", type=int, default=3, help="slides per evaluation cohort")
    parser.add_argument("--map-size", type=int, default=768)
    parser.add_argument(
        "--slide-means-dir", type=Path, required=True,
        help="dir with cohort_probe slide_means_<cohort>_<embedding>.parquet",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (args.output_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "maps").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

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

    def targets(part) -> pd.DataFrame:
        return part.scores.loc[part.pairs["sample_id"].astype(str), names].reset_index(drop=True)

    cohorts = {}
    for spec in train_used + test_used:
        resolved = spec.resolve(processed_root, embedding)
        run_dir = resolved.embedding_dir.parent.parent / "panther" / args.panther
        tables = {}
        for kind in ("pi", "pimu"):
            frame = pd.read_parquet(run_dir / f"slides_{kind}.parquet")
            frame.index = frame.index.astype(str)
            tables[kind] = frame
        means = pd.read_parquet(args.slide_means_dir / f"slide_means_{spec.name}_{resolved.embedding_dir.name}.parquet")
        means.index = means.index.astype(str)
        cohorts[spec.name] = {"embedding_dir": resolved.embedding_dir, "slides_root": Path(spec.slides_root_dir),
                              "run_dir": run_dir, "means": means, **tables}
    proto_dir = Path(json.loads((cohorts[args.condition]["run_dir"] / "prototypes.json").read_text())["prototypes"])
    settings = json.loads((proto_dir / "prototypes.json").read_text())["settings"]
    project = Projection(Path(settings["pca"]) if settings["pca"] else None, torch.device("cpu"))
    n_proto = int(settings["components"])
    colors = prototype_colors(n_proto)

    # 1. Probes on the training split and the prototype table.
    train = splits["train"][0]
    train_ids = train.pairs["slide_id"].astype(str)
    y = targets(train).to_numpy()
    c_train = cohorts[train.cohort]
    probes = {
        "mean": make_probe().fit(c_train["means"].loc[train_ids].to_numpy(), y),
        "pimu": make_probe().fit(c_train["pimu"].loc[train_ids].to_numpy(), y),
        "pi": make_probe().fit(c_train["pi"].loc[train_ids].to_numpy(), y),
    }
    pi_train = c_train["pi"].loc[train_ids].to_numpy()
    table = pd.DataFrame({"prototype": range(n_proto), "pi_train": pi_train.mean(axis=0)})
    for name, c in cohorts.items():
        table[f"pi_{name}"] = c["pi"].mean(axis=0).to_numpy()
    for pathway in args.pathways:
        p = names.index(pathway)
        short = pathway.removeprefix("HALLMARK_")
        table[f"coef_{short}"] = probes["pi"][-1].coef_[p]
        table[f"rho_{short}"] = [spearmanr(pi_train[:, k], y[:, p]).statistic for k in range(n_proto)]
    table.to_csv(args.output_dir / "prototypes.csv", index=False)
    print(table.round(3).to_string(index=False))

    # 2. Atlas: typical and random assigned tiles per prototype and cohort.
    typical, random_rows = {}, {}
    for name, c in cohorts.items():
        best = {k: [] for k in range(n_proto)}
        pool = {k: [] for k in range(n_proto)}
        for slide_id in tqdm(list(c["pi"].index), desc=f"{name} atlas"):
            tiles = np.load(c["run_dir"] / "tiles" / f"{slide_id}.npz")
            assign = tiles["resp"].astype(np.float32).argmax(axis=1)
            dist = tiles["dist"].astype(np.float32)
            for k in np.unique(assign):
                idx = np.flatnonzero(assign == k)
                t = idx[dist[idx, k].argmin()]
                best[k].append((-float(dist[t, k]), slide_id, int(t)))
                pool[k].append((0.0, slide_id, int(rng.choice(idx))))
        typical[name] = {k: sorted(rows, reverse=True)[: args.tiles] for k, rows in best.items()}
        random_rows[name] = {
            k: [rows[i] for i in rng.permutation(len(rows))[: args.tiles]] for k, rows in pool.items()
        }
    for label, picked in (("typical", typical), ("random", random_rows)):
        panel_rows = []
        for k in range(n_proto):
            for name, c in cohorts.items():
                tiles = read_tiles(picked[name][k], c["embedding_dir"], c["slides_root"], args.tile_size)
                share = table.loc[k, f"pi_{name}"]
                panel_rows.append((f"P{k}  {name}\nmean pi {share:.3f}", tiles))
        panel = tile_panel(panel_rows, args.tile_size, args.tiles)
        # Colour key next to each prototype's rows, matching the maps.
        draw = ImageDraw.Draw(panel)
        for r in range(len(panel_rows)):
            y0 = r * (args.tile_size + 4)
            draw.rectangle((240, y0 + 4, 254, y0 + 18), fill=tuple(int(v) for v in colors[r // len(cohorts)]))
        panel.save(args.output_dir / "tiles" / f"{label}.jpg", quality=88)

    # 3. Slide panels: assignment and three probe maps per pathway.
    first = names.index(args.pathways[0])
    records, panels = [], {pathway: [] for pathway in args.pathways}
    for part in splits["val"] + splits["test"]:
        c = cohorts[part.cohort]
        truth_frame = targets(part)
        truth = truth_frame[names[first]].to_numpy()
        picks = np.argsort(truth)[np.linspace(0, len(truth) - 1, args.map_slides).round().astype(int)]
        store = build_embedding_store_from_dir(root_dir=c["embedding_dir"])
        for i in picks[::-1]:
            slide_id = str(part.pairs["slide_id"].iloc[i])
            x_raw, coords, attrs = store.load(slide_id)
            x = project(x_raw).numpy()
            resp = np.load(c["run_dir"] / "tiles" / f"{slide_id}.npz")["resp"].astype(np.float32)
            assign = resp.argmax(axis=1)
            thumbnail = load_thumbnail(c["slides_root"] / str(attrs["relative_wsi_path"]), args.map_size // 2)
            assigned = assignment_overlay(thumbnail, coords, attrs, assign, colors)
            for pathway in args.pathways:
                p = names.index(pathway)
                s_mean = tile_scores(probes["mean"], np.asarray(x_raw, dtype=np.float32), p)
                scaler, ridge = probes["pimu"][0], probes["pimu"][-1]
                w = (ridge.coef_[p] / scaler.scale_).reshape(n_proto, -1)
                const = ridge.intercept_[p] - (scaler.mean_ / scaler.scale_) @ ridge.coef_[p]
                s_pimu = (resp * (x @ w.T)).sum(axis=1) + const
                scaler, ridge = probes["pi"][0], probes["pi"][-1]
                s_pi = resp @ (ridge.coef_[p] / scaler.scale_) + ridge.intercept_[p] - (
                    scaler.mean_ / scaler.scale_
                ) @ ridge.coef_[p]
                stored = {k: probes[k].predict(c[k].loc[[slide_id]].to_numpy())[0, p] for k in ("pimu", "pi")}
                records.append({
                    "pathway": pathway, "cohort": part.cohort, "slide_id": slide_id, "n_tiles": len(x),
                    "true": truth_frame.iloc[i][pathway], "pred_mean": s_mean.mean(),
                    "pred_pimu": stored["pimu"], "tile_avg_pimu": s_pimu.mean(),
                    "pred_pi": stored["pi"], "tile_avg_pi": s_pi.mean(),
                    "rho_mean_pimu": spearmanr(s_mean, s_pimu).statistic,
                    "rho_mean_pi": spearmanr(s_mean, s_pi).statistic,
                })
                vmax = float(np.percentile(np.abs(np.concatenate([s_mean, s_pimu, s_pi])), 99))
                images = [thumbnail, assigned]
                for scores in (s_mean, s_pimu, s_pi):
                    heat = rasterize_tile_values_to_thumbnail(coords, scores, attrs, thumbnail)
                    images.append(render_heatmap_overlay(thumbnail, heat, kind="score", alpha=0.6, vmax=vmax).convert("RGB"))
                row = Image.new("RGB", (sum(im.width for im in images), thumbnail.height + 20), "white")
                for j, im in enumerate(images):
                    row.paste(im, (j * thumbnail.width, 20))
                ImageDraw.Draw(row).text(
                    (4, 4),
                    f"{part.cohort} {slide_id[:23]}  true {truth_frame.iloc[i][pathway]:+.2f} | prototypes | "
                    f"slide mean {s_mean.mean():+.2f} | PANTHER pimu {stored['pimu']:+.2f} | "
                    f"PANTHER pi {stored['pi']:+.2f}  (+-{vmax:.2f})",
                    fill="black",
                )
                panels[pathway].append(row)
    for pathway, rows in panels.items():
        key = legend(colors, [f"P{k}" for k in range(n_proto)], rows[0].width)
        stack([key, *rows]).save(args.output_dir / "maps" / f"{pathway.removeprefix('HALLMARK_')}.jpg", quality=85)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "maps.csv", index=False)
    print(frame.round(3).to_string(index=False))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
