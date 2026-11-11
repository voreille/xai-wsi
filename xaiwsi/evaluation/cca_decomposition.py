"""How much of each Hallmark the slide explains through the shared WSI <-> RNA axes, with intervals and tile maps.

Linear CCA (``xaiwsi.evaluation.shared_structure.LinearCCA``, fit on TCGA train) between
slide-mean H-optimus-1 embeddings (PCA ``--pca-dim``) and one RNA representation:

- ``bulkformer``: BulkFormer embeddings (PCA ``--rna-dim``), as in ``shared_structure``;
- ``hallmarks``: the 50 mean-z Hallmark scores (PCA ``--hallmark-dim``).

Each canonical axis k pairs a slide score u_k and an RNA score v_k (unit variance on train).
From the slide alone, the best linear guess of v_k is rho_k * u_k, with variance 1 - rho_k^2
left; RNA directions beyond the first K axes are unknown (train mean, full variance). rho_k
is the canonical correlation on TCGA val (train correlations are inflated; negative values
set to 0), so val rows are partly in-sample and CPTAC is the clean test. Per Hallmark:

- ``cca-<rna>``: Hallmark predicted from the slide through the first K axes. For
  ``hallmarks`` the map back is the CCA inverse; for ``bulkformer`` a ridge Hallmarks <- v_1..K
  fit on TCGA train, applied to rho * u.
- ``shared-<rna>`` (needs RNA, diagnostic): the same Hallmark from the true RNA scores
  v_1..K: how much of the pathway lies in the shared RNA directions at all. The slide can
  explain at most this through these axes.
- ``ridge``: ridge on the full slide mean (the stage-1 baseline), same cases.

``r2`` is the squared Pearson correlation (variance explained up to a cohort offset; median
over Hallmarks in the summary) and ``spearman`` as in the stage-1 tables.

Gaussian intervals (``hallmarks`` only, closed form): prediction ± z * sigma, with
sigma_p^2 = sum_{k<=K} (1 - rho_k^2) L_kp^2 + sum_{k>K} L_kp^2 + the train variance the PCA
drops, L = the map from canonical scores to Hallmarks. Coverage and width at level
``--alpha`` against the marginal interval (train mean ± z * train sd). The predictions are
also written as per-case files for ``xaiwsi.evaluation.intervals`` (split conformal).

Site check: share of each axis' variance explained by the TCGA tissue source site (eta², TCGA
train + val), per side. Tile maps: the slide score is linear in the mean tile embedding, so
each tile gets s_i = a_k . (t_i standardized) with mean_i s_i = u_k exactly; the
``--map-slides`` highest and lowest slides of TCGA val per axis are drawn for
``--map-axes`` (needs the WSIs; ``--map-axes`` with no values skips the maps).

    uv run python -m xaiwsi.evaluation.cca_decomposition \\
        --config configs/training/alignment_baseline.yaml \\
        --pathway-config configs/training/pathway_cohorts.yaml \\
        --slide-means-dir outputs/cohort_probe --output-dir outputs/cca_decomposition

Writes ``shares.csv`` (method, K, split, pathway: r2, spearman), ``spectrum.csv``,
``intervals.csv``, ``axis_site.csv``, ``predictions/<method>_k<K>.parquet``, ``maps/`` and
``cca_decomposition.md``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr

from xaiwsi.evaluation.cca_hallmarks import SPLIT_NAMES, hallmark_targets
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.evaluation.shared_structure import SPLITS, LinearCCA, check_ranges, load_frozen, tcga_site
from xaiwsi.training.run_config import data_args

COHORT = {"val": "tcga-luad", "test": "cptac-luad"}


def column_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    denom = np.sqrt((a**2).sum(0) * (b**2).sum(0))
    return np.where(denom > 0, (a * b).sum(0) / np.where(denom > 0, denom, 1), 0.0)


def pathway_rows(method: str, k: int, split: str, y: np.ndarray, pred: np.ndarray, names: list[str]) -> list[dict]:
    r = column_r(y, pred)
    rho = [spearmanr(y[:, p], pred[:, p]).statistic for p in range(len(names))]
    return [
        {"method": method, "k": k, "split": SPLIT_NAMES[split], "pathway": n, "r2": r[p] ** 2, "spearman": rho[p]}
        for p, n in enumerate(names)
    ]


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance of ``values`` explained by group means."""
    frame = pd.DataFrame({"v": values, "g": groups})
    means = frame.groupby("g")["v"].transform("mean")
    total = ((frame["v"] - frame["v"].mean()) ** 2).sum()
    return float(((means - frame["v"].mean()) ** 2).sum() / total)


def tile_axis_scores(tiles: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, cca: LinearCCA, axes: list[int]) -> np.ndarray:
    """N x len(axes) per-tile slide scores; their mean over tiles is the slide's u_k."""
    standardized = (np.asarray(tiles, dtype=np.float64) - x_mean) / x_std
    return (standardized - cca.mx) @ cca.wx[:, [a - 1 for a in axes]]


def draw_maps(args, splits, cca, label: str, axes: list[int], u_val: np.ndarray, out_dir: Path) -> list[dict]:
    from PIL import Image, ImageDraw

    from xaiwsi.visualization.pathways import load_thumbnail, rasterize_tile_values_to_thumbnail, render_heatmap_overlay

    data = data_args(yaml.safe_load(args.config.read_text()))
    slides_root = Path(data["tcga_slides_root_dir"])
    store = splits["val"]["dataset"].embedding_store
    slide_ids = splits["val"]["pairs"]["slide_id"].astype(str).to_numpy()
    x_mean, x_std = splits["train"]["x_mean"], splits["train"]["x_std"]
    records = []
    for axis in axes:
        order = np.argsort(u_val[:, axis - 1])
        chosen = [("high", i) for i in order[::-1][: args.map_slides]] + [("low", i) for i in order[: args.map_slides]]
        maps = []
        for side, i in chosen:
            tiles, coords, attrs = store.load(slide_ids[i])
            scores = tile_axis_scores(tiles, x_mean, x_std, cca, [axis])[:, 0]
            slide_u = u_val[i, axis - 1]
            if not np.isclose(scores.mean(), slide_u, atol=1e-6 * max(1.0, abs(slide_u))):
                raise ValueError(f"{slide_ids[i]} axis {axis}: tile mean {scores.mean():.6f} != slide score {slide_u:.6f}")
            thumbnail = load_thumbnail(slides_root / str(attrs["relative_wsi_path"]), args.map_size)
            heatmap = rasterize_tile_values_to_thumbnail(np.asarray(coords), scores, attrs, thumbnail)
            maps.append((side, slide_ids[i], slide_u, thumbnail, heatmap, scores))
            records.append({"rna": label, "axis": axis, "side": side, "slide_id": slide_ids[i], "slide_score": slide_u,
                            "tile_p05": np.percentile(scores, 5), "tile_p95": np.percentile(scores, 95)})
        # Shared colour scale per axis (centred on 0 = the train mean slide), so slides compare.
        vmax = float(np.percentile(np.abs(np.concatenate([m[5] for m in maps])), 99))
        panels = []
        for side, slide_id, slide_u, thumbnail, heatmap, _ in maps:
            image = render_heatmap_overlay(thumbnail, heatmap, kind="score", alpha=0.6, vmax=vmax).convert("RGB")
            pair = Image.new("RGB", (thumbnail.width * 2, thumbnail.height + 24), "white")
            pair.paste(thumbnail, (0, 24))
            pair.paste(image, (thumbnail.width, 24))
            ImageDraw.Draw(pair).text((4, 4), f"{side}  {slide_id}  u={slide_u:+.2f}  (vmax {vmax:.2f})", fill="black")
            panels.append(pair)
        width = max(p.width for p in panels)
        montage = Image.new("RGB", (width, sum(p.height for p in panels)), "white")
        y = 0
        for p in panels:
            montage.paste(p, (0, y))
            y += p.height
        out_dir.mkdir(parents=True, exist_ok=True)
        montage.save(out_dir / f"{label}_axis{axis}.jpg", quality=88)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--pathway-config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--train-cohort", default="tcga-luad")
    parser.add_argument("--test-cohort", default="cptac-luad")
    parser.add_argument("--pca-dim", type=int, default=32, help="slide-mean PCA components")
    parser.add_argument("--rna-dim", type=int, default=32, help="BulkFormer PCA components")
    parser.add_argument("--hallmark-dim", type=int, default=16, help="Hallmark PCA components")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--map-axes", type=int, nargs="*", default=[1, 2, 3])
    parser.add_argument("--map-slides", type=int, default=2, help="highest and lowest TCGA-val slides per axis")
    parser.add_argument("--map-size", type=int, default=768)
    parser.add_argument(
        "--table-k", type=int, default=5,
        help="axes in the per-Hallmark table (the axes with a clear held-out rho; at large K the shared ceiling is trivial)",
    )
    parser.add_argument("--top", type=int, default=12, help="Hallmarks shown per end of the per-pathway table")
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache (default: output dir)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output_dir)
    targets, names = hallmark_targets(args.pathway_config, args.train_cohort, args.test_cohort)
    h = {}
    for split in SPLITS:
        cases = splits[split]["pairs"]["case_id"].astype(str)
        missing = set(cases) - set(targets[split].index)
        if missing:
            raise SystemExit(f"{split}: {len(missing)} alignment cases without Hallmark targets in that split.")
        h[split] = targets[split].loc[cases, names].to_numpy()
    x = {s: splits[s]["x"] for s in SPLITS}
    rna = {"bulkformer": {s: splits[s]["y"] for s in SPLITS}, "hallmarks": h}
    dims = {"bulkformer": args.rna_dim, "hallmarks": args.hallmark_dim}

    rows, spectrum, interval_rows, site_rows = [], [], [], []
    z_crit = norm.ppf(1 - args.alpha / 2)

    ridge = make_probe().fit(x["train"], h["train"])
    for split in ("val", "test"):
        rows += pathway_rows("ridge", 0, split, h[split], ridge.predict(x[split]), names)
    sites = np.concatenate([tcga_site(splits[s]["pairs"]["case_id"]).to_numpy() for s in ("train", "val")])

    fits = {}
    for label, y in rna.items():
        cca = LinearCCA(args.pca_dim, dims[label]).fit(x["train"], y["train"])
        z = {s: cca.transform(x[s], y[s]) for s in SPLITS}
        n_axes = min(args.pca_dim, dims[label])
        z = {s: (u[:, :n_axes], v[:, :n_axes]) for s, (u, v) in z.items()}
        canon = {s: column_r(*z[s]) for s in SPLITS}
        rho = np.clip(canon["val"], 0, None)
        fits[label] = (cca, z)
        for axis in range(n_axes):
            spectrum.append({"rna": label, "axis": axis + 1, **{f"rho_{SPLIT_NAMES[s]}": canon[s][axis] for s in SPLITS}})
            for side, idx in (("slide", 0), ("rna", 1)):
                values = np.concatenate([z[s][idx][:, axis] for s in ("train", "val")])
                site_rows.append({"rna": label, "axis": axis + 1, "side": side, "site_eta2": eta_squared(values, sites)})

        # Map from RNA canonical scores back to Hallmarks.
        if label == "hallmarks":
            loadings = np.linalg.pinv(cca.wy)[:n_axes]  # n_axes x 50
            pca_resid = h["train"] - cca.my - z["train"][1] @ loadings
            resid_var = pca_resid.var(axis=0, ddof=1)
        for k in [k for k in args.ks if k <= n_axes]:
            if label == "hallmarks":
                def back(v: np.ndarray, k=k) -> np.ndarray:
                    return cca.my + v[:, :k] @ loadings[:k]

                predict = {s: back(z[s][0][:, :k] * rho[:k]) for s in SPLITS}
                shared = {s: back(z[s][1]) for s in SPLITS}
                sigma = np.sqrt(((1 - rho[:k] ** 2)[:, None] * loadings[:k] ** 2).sum(0)
                                + (loadings[k:] ** 2).sum(0) + resid_var)
                marginal_sd = h["train"].std(axis=0, ddof=1)
                for split in ("val", "test"):
                    pred = predict[split]
                    inside = np.abs(h[split] - pred) <= z_crit * sigma
                    inside_marginal = np.abs(h[split] - h["train"].mean(0)) <= z_crit * marginal_sd
                    interval_rows.append({
                        "k": k, "split": SPLIT_NAMES[split], "n": len(pred),
                        "mean_coverage": inside.mean(0).mean(), "min_coverage": inside.mean(0).min(),
                        "marginal_mean_coverage": inside_marginal.mean(0).mean(),
                        "median_width": float(np.median(2 * z_crit * sigma)),
                        "width_ratio": float(np.median(sigma / marginal_sd)),
                    })
            else:
                head = make_probe().fit(z["train"][1][:, :k], h["train"])
                predict = {s: head.predict(z[s][0][:, :k] * rho[:k]) for s in SPLITS}
                shared = {s: head.predict(z[s][1][:, :k]) for s in SPLITS}
            frames = []
            for split in ("val", "test"):
                rows += pathway_rows(f"cca-{label}", k, split, h[split], predict[split], names)
                rows += pathway_rows(f"shared-{label}", k, split, h[split], shared[split], names)
                frames.append(prediction_frame(splits[split]["pairs"]["case_id"].astype(str), COHORT[split], split,
                                               h[split], predict[split], names))
            pd.concat(frames, ignore_index=True).to_parquet(args.output_dir / "predictions" / f"cca-{label}_k{k}.parquet")

    shares = pd.DataFrame(rows)
    check_ranges(shares, ["r2"], 0, 1)
    check_ranges(shares, ["spearman"], -1, 1)
    spectrum = pd.DataFrame(spectrum)
    check_ranges(spectrum, [c for c in spectrum.columns if c.startswith("rho_")], -1, 1)
    intervals = pd.DataFrame(interval_rows)
    check_ranges(intervals, ["mean_coverage", "min_coverage", "marginal_mean_coverage"], 0, 1)
    site = pd.DataFrame(site_rows)
    check_ranges(site, ["site_eta2"], 0, 1)
    shares.to_csv(args.output_dir / "shares.csv", index=False)
    spectrum.to_csv(args.output_dir / "spectrum.csv", index=False)
    intervals.to_csv(args.output_dir / "intervals.csv", index=False)
    site.to_csv(args.output_dir / "axis_site.csv", index=False)

    maps = []
    for label in ("bulkformer",) if args.map_axes else ():
        cca, z = fits[label]
        maps += draw_maps(args, splits, cca, label, args.map_axes, z["val"][0], args.output_dir / "maps")
    if maps:
        pd.DataFrame(maps).to_csv(args.output_dir / "maps" / "slides.csv", index=False)

    # --- Markdown --------------------------------------------------------------------
    summary = (
        shares.groupby(["method", "k", "split"], sort=False)[["r2", "spearman"]].median()
        .unstack("split").sort_index(axis=1, level=0)
    )
    summary.columns = [f"median_{metric}_{split}" for metric, split in summary.columns]
    summary = summary.reset_index()
    # K chosen on TCGA val (median Spearman) per route; CPTAC is not used to choose.
    best = {}
    for method in ("cca-bulkformer", "cca-hallmarks"):
        frame = summary[summary["method"] == method]
        best[method] = int(frame.loc[frame["median_spearman_tcga-val"].idxmax(), "k"])
    wide = shares[shares["split"] == "cptac"].pivot_table(index="pathway", columns=["method", "k"], values="r2")
    k = args.table_k
    per_pathway = pd.DataFrame({
        "ridge r2": wide[("ridge", 0)],
        f"cca-bulkformer K={k} r2": wide[("cca-bulkformer", k)],
        f"shared-bulkformer K={k} r2": wide[("shared-bulkformer", k)],
        f"cca-hallmarks K={k} r2": wide[("cca-hallmarks", k)],
        f"shared-hallmarks K={k} r2": wide[("shared-hallmarks", k)],
    }).sort_values("ridge r2", ascending=False)
    per_pathway.index = per_pathway.index.str.removeprefix("HALLMARK_")
    ends = pd.concat([per_pathway.head(args.top), per_pathway.tail(args.top)])
    spectrum_md = spectrum[spectrum["axis"] <= 8]
    site_md = site[site["axis"] <= 5].pivot_table(index=["rna", "axis"], columns="side", values="site_eta2")
    site_md = site_md.rename(columns=lambda c: f"{c}_side_eta2").rename_axis(columns=None).reset_index()

    n = {SPLIT_NAMES[s]: len(h[s]) for s in SPLITS}
    command = "uv run python -m xaiwsi.evaluation.cca_decomposition " + shlex.join(sys.argv[1:])
    text = (
        f"Linear CCA, slide means (PCA {args.pca_dim}) vs BulkFormer (PCA {args.rna_dim}) or Hallmarks "
        f"(PCA {args.hallmark_dim}), fit on TCGA train; rho from TCGA val; n = {n}. r2 = squared Pearson "
        f"per Hallmark (median over the 50). Generated by:\n\n```bash\n{command}\n```\n\n"
        f"Canonical correlations (first 8 axes):\n\n{markdown_table(spectrum_md)}\n\n"
        f"Hallmarks from the slide through the first K shared axes (`cca-*`), from the true RNA scores of "
        f"the same axes (`shared-*`, needs RNA), and the full slide-mean ridge:\n\n{markdown_table(summary)}\n\n"
        f"K with the best TCGA-val median Spearman: {best}. Per Hallmark on CPTAC (r2) at K = {args.table_k}, top and bottom {args.top} by the ridge:\n\n"
        f"{markdown_table(ends.reset_index().rename(columns={'pathway': 'hallmark'}))}\n\n"
        f"Gaussian intervals from the Hallmark CCA, level {1 - args.alpha:.0%} (width in z units; "
        f"width_ratio = median sigma / train sd, 1 = no information):\n\n{markdown_table(intervals)}\n\n"
        f"Site share of each axis (eta², TCGA site, train + val):\n\n{markdown_table(site_md)}\n"
    )
    if maps:
        text += (
            f"\nTile maps of the BulkFormer CCA slide axes {args.map_axes}: the {args.map_slides} highest and lowest "
            f"TCGA-val slides per axis, `maps/bulkformer_axis<k>.jpg` (tile scores average exactly to the slide score; "
            f"shared colour scale per axis, red = towards the high end).\n"
        )
    (args.output_dir / "cca_decomposition.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
