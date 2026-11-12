"""Tile score maps of the concept-bag and slide-mean probes: structure vs a slide-wide offset.

Both ridge probes (``cohort_probe``: slide mean; clipped concept bag) are linear in tile features,
so each gives tile scores whose slide average is the prediction. On every evaluation slide (val
of the training cohort, test cohorts) and every target, per probe:

- ``between_share``: share of the tile-score variance of a cohort that lies between slides
  (slide averages) rather than within slides. A probe whose maps look like a uniform tint per
  slide has a high share: the slide moves as a block.
- ``within_sd``: median within-slide standard deviation of the tile scores (map contrast).
- ``spearman``: the slide predictions against the truth (the usual metric).

and, between probes, ``tile_agreement``: median over slides of the tile-level Spearman of the two
maps. Medians over targets are in ``<output>.md``; per target in ``<output>.csv``.

Tissue breakdown (``--tissue-probe``, default IGNITE from ``xaiwsi.tissue.apply``): per slide, each
probe's tile scores averaged within tissue groups (``TISSUE_GROUPS``, tiles weighted by the
probe's soft fractions, or with ``--min-tile-fraction`` only the tiles whose group fraction reaches
it; a group with less than ``--min-group-weight`` tiles is missing on that slide). ``<output>_tissue.csv``: per cohort, probe, target and group, the Spearman across slides of
the group score with the truth and with the slide prediction. Background and blood carry no
Hallmark biology, so if their scores follow the prediction the slide-wide offset is technical.
Group scores per slide: ``<output>_tissue.npz``.

    uv run python -m xaiwsi.evaluation.tile_tint --config configs/training/pathway_cohorts.yaml \\
        --condition tcga-luad --sae tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --slide-means-dir outputs/concept_sweep/tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --output outputs/concept_viz/tint_luad_valpick
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.concepts.viz import dense_codes
from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import TablePrep, markdown_table
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args

# Tissue groups of the IGNITE classes: the compartments of configs/tissue/compartments.yaml plus
# tissue that should carry no tumour-RNA signal (background, alveolar lung, blood).
TISSUE_GROUPS = {
    "tumour": ["Tumor epithelium", "Keratinization"],
    "stroma": ["Stroma"],
    "immune": ["Inflammation", "Macrophages"],
    "necrosis": ["Necrotic tissue"],
    "background": ["Background"],
    "alveolar": ["Alveolar tissue"],
    "blood": ["Erythrocytes"],
}


def tile_matrix(model, x: np.ndarray) -> np.ndarray:
    """Tiles x targets outputs of a fitted StandardScaler + ridge pipeline."""
    scaler, ridge = model[0], model[-1]
    return ((x - scaler.mean_) / scaler.scale_) @ ridge.coef_.T + ridge.intercept_


def column_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = rankdata(a, axis=0), rankdata(b, axis=0)
    ra, rb = ra - ra.mean(axis=0), rb - rb.mean(axis=0)
    return (ra * rb).sum(axis=0) / (np.linalg.norm(ra, axis=0) * np.linalg.norm(rb, axis=0) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--sae", required=True, help="bag folder name under <cohort dir>/concepts/ (with tiles/)")
    parser.add_argument("--slide-means-dir", type=Path, required=True)
    parser.add_argument("--tissue-probe", default="ignite", help="applied tissue probe under <cohort dir>/tissue/")
    parser.add_argument(
        "--min-tile-fraction",
        type=float,
        default=0.0,
        help="count a tile in a group only if its group fraction is at least this (0 = soft weights)",
    )
    parser.add_argument("--min-group-weight", type=float, default=20.0, help="summed tile weight for a group score")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, test_used = select_cohorts(train_specs, test_specs, [args.condition])
    names = list(load_hallmarks(data["hallmark_gmt"]))
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [c.resolve(processed_root, embedding) for c in test_used],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    dirs, bag_dirs, bags, means = {}, {}, {}, {}
    for spec in train_used + test_used:
        resolved = spec.resolve(processed_root, embedding)
        dirs[spec.name] = resolved.embedding_dir
        bag_dirs[spec.name] = resolved.embedding_dir.parent.parent / "concepts" / args.sae
        bag = pd.read_parquet(bag_dirs[spec.name] / "slides_mean.parquet")
        bag.index = bag.index.astype(str)
        bags[spec.name] = bag
        mean = pd.read_parquet(args.slide_means_dir / f"slide_means_{spec.name}_{resolved.embedding_dir.name}.parquet")
        mean.index = mean.index.astype(str)
        means[spec.name] = mean

    def targets(part) -> np.ndarray:
        return part.scores.loc[part.pairs["sample_id"].astype(str), names].to_numpy()

    train = splits["train"][0]
    ids = train.pairs["slide_id"].astype(str)
    y_train = targets(train)
    prep = TablePrep(bags[train.cohort].loc[ids].to_numpy(), 0.0, "clip")
    models = {
        "concept": make_probe().fit(prep(bags[train.cohort].loc[ids].to_numpy()), y_train),
        "slide-mean": make_probe().fit(means[train.cohort].loc[ids].to_numpy(), y_train),
    }
    n_latents = bags[train.cohort].shape[1]
    tissue_dirs = {name: d.parent.parent / "tissue" / args.tissue_probe for name, d in dirs.items()}
    classes = json.loads((tissue_dirs[train.cohort] / "probe.json").read_text())["classes"]
    group_matrix = np.zeros((len(classes), len(TISSUE_GROUPS)))
    for g, members in enumerate(TISSUE_GROUPS.values()):
        for member in members:
            group_matrix[classes.index(member), g] = 1.0
    tissue_rows, tissue_arrays = [], {}

    rows, agreement_rows = [], []
    for split in ("val", "test"):
        for part in splits[split]:
            store = build_embedding_store_from_dir(root_dir=dirs[part.cohort])
            stats = {probe: {"mean": [], "var": [], "n": [], "groups": []} for probe in models}
            agreement = []
            for slide_id in tqdm(part.pairs["slide_id"].astype(str), desc=part.cohort):
                x_tiles = store.load(slide_id)[0]
                scores = {
                    "slide-mean": tile_matrix(models["slide-mean"], x_tiles),
                    "concept": tile_matrix(models["concept"], dense_codes(bag_dirs[part.cohort], slide_id, n_latents)),
                }
                weights = np.load(tissue_dirs[part.cohort] / "tiles" / f"{slide_id}.npy").astype(np.float64) @ group_matrix
                if args.min_tile_fraction > 0:
                    weights = (weights >= args.min_tile_fraction).astype(np.float64)
                if len(weights) != len(x_tiles):
                    raise ValueError(f"{slide_id}: {len(weights)} tissue rows for {len(x_tiles)} tiles")
                total = weights.sum(axis=0)
                for probe, s in scores.items():
                    group_scores = (weights.T @ s) / np.maximum(total, 1e-9)[:, None]  # groups x targets
                    group_scores[total < args.min_group_weight] = np.nan
                    stats[probe]["groups"].append(group_scores)
                    stats[probe]["mean"].append(s.mean(axis=0))
                    stats[probe]["var"].append(s.var(axis=0))
                    stats[probe]["n"].append(len(s))
                agreement.append(column_spearman(scores["slide-mean"], scores["concept"]))
            y = targets(part)
            for probe, st in stats.items():
                slide_means = np.array(st["mean"])  # slides x targets (the tile-score averages)
                within = np.average(np.array(st["var"]), axis=0, weights=st["n"])
                between = slide_means.var(axis=0)
                rho = np.array([spearmanr(slide_means[:, p], y[:, p]).statistic for p in range(len(names))])
                groups = np.array(st["groups"])  # slides x groups x targets
                tissue_arrays[f"{part.cohort}__{probe}"] = groups
                for g, group in enumerate(TISSUE_GROUPS):
                    for p, name in enumerate(names):
                        ok = np.isfinite(groups[:, g, p])
                        if ok.sum() < 20:
                            continue
                        tissue_rows.append({
                            "split": split, "cohort": part.cohort, "probe": probe, "target": name, "group": group,
                            "n_slides": int(ok.sum()),
                            "rho_truth": spearmanr(groups[ok, g, p], y[ok, p]).statistic,
                            "rho_prediction": spearmanr(groups[ok, g, p], slide_means[ok, p]).statistic,
                        })
                for p, name in enumerate(names):
                    rows.append({"split": split, "cohort": part.cohort, "probe": probe, "target": name,
                                 "spearman": rho[p], "between_share": between[p] / (between[p] + within[p]),
                                 "within_sd": float(np.median(np.sqrt(np.array(st["var"])[:, p])))})
            agreement = np.median(np.array(agreement), axis=0)
            for p, name in enumerate(names):
                agreement_rows.append({"split": split, "cohort": part.cohort, "target": name,
                                       "tile_agreement": agreement[p]})

    frame = pd.DataFrame(rows)
    if ((frame["spearman"].abs() > 1) | (frame["between_share"] < 0) | (frame["between_share"] > 1)).any():
        raise ValueError("Metric out of range")
    agree = pd.DataFrame(agreement_rows)
    frame.merge(agree, on=["split", "cohort", "target"]).to_csv(args.output.with_suffix(".csv"), index=False)
    summary = (
        frame.groupby(["split", "cohort", "probe"], sort=False)[["spearman", "between_share", "within_sd"]]
        .median()
        .reset_index()
    )
    summary = summary.merge(
        agree.groupby(["split", "cohort"], sort=False)["tile_agreement"].median().reset_index(), on=["split", "cohort"]
    )
    command = "uv run python -m xaiwsi.evaluation.tile_tint " + shlex.join(sys.argv[1:])
    args.output.with_suffix(".md").write_text(
        "Tile score maps of the slide-mean and concept-bag ridge probes, medians over the targets: Spearman of the "
        "slide predictions, share of tile-score variance between slides, within-slide SD of tile scores, and the "
        f"median tile-level Spearman between the two maps. Generated by:\n\n```bash\n{command}\n```\n\n"
        f"{markdown_table(summary)}\n"
    )
    print(markdown_table(summary))

    tissue = pd.DataFrame(tissue_rows)
    tissue.to_csv(args.output.parent / f"{args.output.name}_tissue.csv", index=False)
    np.savez(args.output.parent / f"{args.output.name}_tissue.npz", groups=list(TISSUE_GROUPS), targets=names,
             **tissue_arrays)
    wide = tissue.groupby(["cohort", "probe", "group"], sort=False)[["n_slides", "rho_truth", "rho_prediction"]].median()
    wide = wide.reset_index()
    (args.output.parent / f"{args.output.name}_tissue.md").write_text(
        "Probe tile scores averaged within IGNITE tissue groups, per slide; medians over targets of the Spearman "
        "across slides with the truth and with the slide prediction. Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{markdown_table(wide)}\n"
    )
    print(markdown_table(wide))


if __name__ == "__main__":
    main()
