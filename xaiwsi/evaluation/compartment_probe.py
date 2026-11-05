"""Ridge probe on compartment-pooled slide embeddings (tumour, stroma, immune, necrosis).

Same cohorts, splits, targets and ridge as ``xaiwsi.evaluation.cohort_probe``, but the slide
features are weighted means of the tile embeddings, with weights from the applied IGNITE
probe (``xaiwsi.tissue.apply``) as set in ``--spec`` (``configs/tissue/compartments.yaml``):

- ``slide-mean``: plain mean over all tiles (equals cohort_probe's slide mean; sanity check).
- ``no-background``: tiles weighted by 1 - p(background).
- ``tme``: tiles weighted by the tumour-microenvironment mass (background, healthy lung and
  other normal tissue left out), the baseline for the compartment variants.
- ``compartment:<name>`` (B1): that compartment's mean alone, each normalized by its own
  weight, so its area does not matter; one ridge per compartment.
- ``concat`` (B2): all compartment means, their shares of the slide and low-weight flags in
  one ridge, which can weight the compartments per pathway.

A compartment with less total weight than ``min_weight`` on a slide falls back to the
slide's TME mean with a ``low_<name>`` flag of 1.

    uv run python -m xaiwsi.evaluation.compartment_probe \\
        --config configs/training/pathway_cohorts.yaml --conditions tcga-luad \\
        --output-dir outputs/compartment_probe

``--tile-qc-threshold 0.5`` gives weight 0 to the tiles that ``xaiwsi.qc.tile_classifier``
flags as artifacts, in every pool (caches get a ``_qc<threshold>`` suffix).

Writes ``probe_metrics.{csv,md}``, ``per_pathway.csv`` (Spearman and R² per pathway, feature
set, split and cohort), ``per_pathway.md`` (per pathway: the best compartment on the
validation split next to the slide mean, val and test), ``low_weight.csv`` (fraction of slides per cohort below
``min_weight``) and ``predictions/<condition>__<normalization>__<features>.parquet`` for
``xaiwsi.evaluation.paired_bootstrap``. Pooled embeddings are cached per cohort as
``pools_<cohort>_<embedding>.parquet`` in the output directory.
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
from tqdm import tqdm

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import check_ranges, markdown_table
from xaiwsi.evaluation.geneset_vs_hallmarks import add_axis_scores
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.qc.mask import keep_mask, load_artifact_probs
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures
from xaiwsi.tissue.features import compartment_pools, tissue_dir
from xaiwsi.training.run_config import data_args


def pooled_embeddings(
    store, root: Path, slide_ids: list[str], spec: dict, cache: Path, tile_masks=None
) -> pd.DataFrame:
    """slide_id x (``<pool>__f<i>`` means, ``weight__<pool>``, ``n_tiles``), cached as parquet."""
    if cache.exists():
        pools = pd.read_parquet(cache)
        if set(slide_ids) <= set(pools.index):
            return pools.loc[slide_ids]
    probe_dir = root / spec["probe"]
    classes = json.loads((probe_dir / "probe.json").read_text())["classes"]
    rows = {}
    for slide_id in tqdm(slide_ids, desc=f"compartment pools ({cache.stem})"):
        tiles, _, _ = store.load(slide_id)
        fractions = np.load(probe_dir / "tiles" / f"{slide_id}.npy")
        keep = tile_masks(slide_id) if tile_masks is not None else None
        means, totals = compartment_pools(tiles, fractions, classes, spec, keep=keep)
        row = {f"{pool}__f{i}": v for pool, mean in means.items() for i, v in enumerate(mean)}
        row |= {f"weight__{pool}": total for pool, total in totals.items()}
        row["n_tiles"] = len(tiles)
        rows[slide_id] = row
    pools = pd.DataFrame.from_dict(rows, orient="index").astype(np.float32)
    pools.to_parquet(cache)
    return pools


def feature_sets(spec: dict) -> list[str]:
    compartments = list(spec["compartments"])
    return ["slide-mean", "no-background", "tme", *[f"compartment:{c}" for c in compartments], "concat"]


def build_inputs(pools: pd.DataFrame, features: str, spec: dict) -> np.ndarray:
    """Feature matrix of one feature set, low-weight compartments replaced by the TME mean."""

    def block(pool: str) -> pd.DataFrame:
        return pools.filter(regex=f"^{pool}__f")

    if features in ("slide-mean", "no-background", "tme"):
        pool = {"slide-mean": "all", "no-background": "no_background"}.get(features, features)
        return block(pool).to_numpy()
    tme = block("tme").to_numpy()
    if np.isnan(tme).any():
        raise ValueError("slides without TME weight; check the spec and the probe outputs")

    def compartment(name: str) -> tuple[np.ndarray, np.ndarray]:
        low = (pools[f"weight__{name}"] < spec["min_weight"]).to_numpy()
        mean = block(name).to_numpy().copy()
        mean[low] = tme[low]
        return mean, low.astype(float)[:, None]

    if features.startswith("compartment:"):
        mean, low = compartment(features.removeprefix("compartment:"))
        return np.concatenate([mean, low], axis=1)
    if features == "concat":
        blocks = []
        for name in spec["compartments"]:
            mean, low = compartment(name)
            share = (pools[f"weight__{name}"] / pools["n_tiles"]).to_numpy()[:, None]
            blocks += [mean, low, share]
        return np.concatenate(blocks, axis=1)
    raise ValueError(f"unknown feature set {features!r}")


def write_best_compartments(per_pathway: pd.DataFrame, spec: dict, path: Path, command: str) -> None:
    """Per pathway, the compartment (B1) with the highest val Spearman, next to the slide mean."""
    compartments = [f"compartment:{c}" for c in spec["compartments"]]
    sections = []
    for (condition, normalization), frame in per_pathway.groupby(["condition", "normalization"], sort=False):
        wide = frame.pivot_table(
            index="pathway", columns=["features", "split", "cohort"], values="spearman", sort=False
        )
        val_cohorts = [c for f, s, c in wide.columns if f == "slide-mean" and s == "val"]
        test_cohorts = [c for f, s, c in wide.columns if f == "slide-mean" and s == "test"]
        # Selection on the first (main) validation cohort only; test is reported, never used.
        main_val = val_cohorts[0]
        val = wide.loc[:, [(f, "val", main_val) for f in compartments]]
        val.columns = [f.removeprefix("compartment:") for f in compartments]
        table = pd.DataFrame({"best compartment (val)": val.idxmax(axis=1)})
        for features, label in (("slide-mean", "slide mean"), ("concat", "concat")):
            table[f"{label} val {main_val}"] = wide[(features, "val", main_val)]
        table[f"best val {main_val}"] = val.max(axis=1)
        for cohort in test_cohorts:
            best_test = [wide.loc[p, (f"compartment:{c}", "test", cohort)] for p, c in table["best compartment (val)"].items()]
            table[f"slide mean test {cohort}"] = wide[("slide-mean", "test", cohort)]
            table[f"best test {cohort}"] = best_test
        counts = table["best compartment (val)"].value_counts().to_dict()
        sections.append(
            f"### {condition}, {normalization}\n\nBest compartment counts: {counts}\n\n"
            + markdown_table(table.reset_index())
        )
    path.write_text(
        "Per-pathway Spearman: compartment with the highest validation Spearman (B1, one ridge "
        f"per compartment) vs the slide mean. Generated by:\n\n```bash\n{command}\n```\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--conditions", nargs="+", required=True, help="comma-separated training cohort sets")
    parser.add_argument(
        "--normalizations", nargs="+", default=["per_cohort"], choices=["per_cohort", "pooled"]
    )
    parser.add_argument(
        "--gene-sets", type=Path, default=None, help=".gmt/.json gene sets (default: config Hallmarks)"
    )
    parser.add_argument("--spec", type=Path, default=Path("configs/tissue/compartments.yaml"))
    parser.add_argument("--features", nargs="+", default=None, help="default: all feature sets")
    parser.add_argument(
        "--tile-qc-threshold",
        type=float,
        help="drop tiles with artifact probability at or above this (xaiwsi.qc.tile_classifier)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    spec = yaml.safe_load(args.spec.read_text())
    features_list = args.features or feature_sets(spec)
    unknown = set(features_list) - set(feature_sets(spec))
    if unknown:
        raise SystemExit(f"unknown feature sets {sorted(unknown)}; choose from {feature_sets(spec)}")

    data = data_args(yaml.safe_load(args.config.read_text()))
    if "train_cohorts" not in data:
        raise SystemExit(f"{args.config} is not a CohortPathwayDataModule config.")
    gene_sets = (
        load_hallmarks(data["hallmark_gmt"]) if args.gene_sets is None else load_tavernari_signatures(args.gene_sets)
    )
    names = list(gene_sets)
    output_names = list(add_axis_scores(pd.DataFrame(columns=names)).columns)

    def with_axes(values: np.ndarray) -> np.ndarray:
        return add_axis_scores(pd.DataFrame(values, columns=names))[output_names].to_numpy()

    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]

    rows, low_rows, pathway_rows = [], [], []
    for condition in args.conditions:
        train_used, test_used = select_cohorts(train_specs, test_specs, condition.split(","))
        for normalization in args.normalizations:
            splits = build_pathway_splits(
                [c.resolve(processed_root, embedding) for c in train_used],
                [c.resolve(processed_root, embedding) for c in test_used],
                gene_sets,
                val_fraction=data.get("val_fraction", 0.2),
                split_seed=data.get("split_seed", 42),
                target_normalization=normalization,
            )
            pools = {}
            by_cohort: dict[str, list] = {}
            for parts in splits.values():
                for part in parts:
                    by_cohort.setdefault(part.cohort, []).append(part)
            for cohort, parts in by_cohort.items():
                cohort_spec = next(c for c in train_specs + test_specs if c.name == cohort)
                embedding_dir = cohort_spec.resolve(processed_root, embedding).embedding_dir
                slide_ids = sorted({sid for p in parts for sid in p.pairs["slide_id"].astype(str)})
                masks, suffix = None, ""
                if args.tile_qc_threshold is not None:
                    probs = load_artifact_probs(embedding_dir)
                    masks = lambda sid, probs=probs: keep_mask(probs[sid], args.tile_qc_threshold)  # noqa: E731
                    suffix = f"_qc{args.tile_qc_threshold:g}"
                pools[cohort] = pooled_embeddings(
                    parts[0].store,
                    tissue_dir(embedding_dir),
                    slide_ids,
                    spec,
                    args.output_dir / f"pools_{cohort}_{embedding_dir.name}{suffix}.parquet",
                    tile_masks=masks,
                )
                for name in spec["compartments"]:
                    weight = pools[cohort].loc[slide_ids, f"weight__{name}"]
                    low_rows.append(
                        {
                            "cohort": cohort,
                            "compartment": name,
                            "n_slides": len(slide_ids),
                            "fraction_low": float((weight < spec["min_weight"]).mean()),
                            "median_weight": float(weight.median()),
                        }
                    )

            def part_pools(part) -> pd.DataFrame:
                return pools[part.cohort].loc[part.pairs["slide_id"].astype(str)]

            def targets(part) -> np.ndarray:
                return part.scores.loc[part.pairs["sample_id"].astype(str), names].to_numpy()

            y_train = np.concatenate([targets(p) for p in splits["train"]])
            for features in features_list:
                x_train = np.concatenate([build_inputs(part_pools(p), features, spec) for p in splits["train"]])
                model = make_probe().fit(x_train, y_train)
                predictions = []
                for split in ("val", "test"):
                    for part in splits[split]:
                        x = build_inputs(part_pools(part), features, spec)
                        y, y_pred = with_axes(targets(part)), with_axes(model.predict(x))
                        predictions.append(
                            prediction_frame(part.pairs["case_id"], part.cohort, split, y, y_pred, output_names)
                        )
                        summary, per_pathway = compute_pathway_metrics(y, y_pred, output_names)
                        pathway_rows.append(
                            per_pathway[["spearman", "r2"]].reset_index().assign(
                                condition=condition,
                                normalization=normalization,
                                features=features,
                                split=split,
                                cohort=part.cohort,
                            )
                        )
                        rows.append(
                            {
                                "condition": condition,
                                "normalization": normalization,
                                "features": features,
                                "n_features": x_train.shape[1],
                                "n_train": len(y_train),
                                "split": split,
                                "cohort": part.cohort,
                                "n": len(y),
                                "median_spearman": summary["median_spearman"],
                                "mean_spearman": summary["mean_spearman"],
                                "median_r2": summary["median_r2"],
                            }
                        )
                        print(
                            f"{condition} {normalization} {features} {split} {part.cohort}: "
                            f"median rho {summary['median_spearman']:.3f} "
                            f"median R² {summary['median_r2']:.3f}"
                        )
                stem = f"{condition}__{normalization}__{features.replace(':', '-')}"
                pd.concat(predictions, ignore_index=True).to_parquet(
                    args.output_dir / "predictions" / f"{stem}.parquet"
                )

    results = pd.DataFrame(rows)
    check_ranges(results)
    results.to_csv(args.output_dir / "probe_metrics.csv", index=False)
    per_pathway = pd.concat(pathway_rows, ignore_index=True)
    rho = per_pathway["spearman"].dropna()
    if ((rho < -1) | (rho > 1)).any() or (per_pathway["r2"].dropna() > 1).any():
        raise ValueError("per-pathway Spearman outside [-1, 1] or R² above 1")
    per_pathway.to_csv(args.output_dir / "per_pathway.csv", index=False)
    low = pd.DataFrame(low_rows).drop_duplicates(["cohort", "compartment"])
    low.to_csv(args.output_dir / "low_weight.csv", index=False)

    command = "uv run python -m xaiwsi.evaluation.compartment_probe " + shlex.join(sys.argv[1:])
    table = markdown_table(results)
    (args.output_dir / "probe_metrics.md").write_text(
        f"Ridge on compartment-pooled slide embeddings ({embedding}, spec {args.spec}). "
        f"Generated by:\n\n```bash\n{command}\n```\n\n{table}\n\n"
        f"Slides below min_weight = {spec['min_weight']} (fallback to the TME mean):\n\n"
        f"{markdown_table(low)}\n"
    )
    write_best_compartments(per_pathway, spec, args.output_dir / "per_pathway.md", command)
    print(table)
    print(markdown_table(low))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
