"""Ridge probe on slide-mean embeddings for sets of training cohorts (LUAD vs LUAD+LUSC).

Uses the cohorts, splits and mean-z targets of a ``CohortPathwayDataModule`` config
(``xaiwsi.data.cohorts.build_pathway_splits``), so the numbers are comparable with the
tile-level runs of that config. Each condition is a comma-separated set of training
cohorts taken from the config; test cohorts whose ``zscore_reference`` is not trained on
are skipped for that condition. The probe is the baseline the tile-level models must beat.

    uv run python -m xaiwsi.evaluation.cohort_probe \\
        --config configs/training/pathway_cohorts.yaml \\
        --conditions tcga-luad tcga-luad,tcga-lusc \\
        --normalizations per_cohort pooled \\
        --output-dir outputs/cohort_probe

Writes ``probe_metrics.csv`` (one row per condition, normalization, split and cohort),
``probe_metrics.md`` (Markdown table with the generating command, for progress.md) and
``predictions/<condition>__<normalization>.parquet``: per-case predictions and targets of
the val and test cohorts (see ``xaiwsi.evaluation.predictions``), the input of
``xaiwsi.evaluation.paired_bootstrap``.

``--tile-qc-threshold 0.5`` averages only the tiles that ``xaiwsi.qc.tile_classifier``
does not flag as artifacts (slide means cached with a ``_qc<threshold>`` suffix).

``--gene-sets <.gmt|.json>`` probes other gene sets than the config's Hallmarks (e.g. the
approximate Tavernari lepidic/solid sets); every ``LEPIDIC_<src>``/``SOLID_<src>`` pair also
gets the axis ``L2S_<src>`` = solid - lepidic, for targets and predictions alike.

``--features`` picks the slide features of the ridge, as ``+``-joined blocks: ``slide-mean``
(default), ``composition`` (the few tissue shares of ``--composition-spec``, from the applied
tile-composition probes of ``xaiwsi.tissue.apply``) and ``table:<path>``, a slide_id-indexed
parquet relative to each cohort's directory (e.g. the concept bags of ``xaiwsi.concepts.bags``,
``table:concepts/<sae>/slides_mean.parquet``). Each feature set adds a ``features`` column to
the metrics; non-default sets suffix the prediction files with ``__<features>`` (``/`` and
``:`` replaced, ``.parquet`` dropped). With composition features, it also writes
``composition_correlations.csv`` (Spearman of each share with each true target, per split
and cohort) and ``composition_correlations.md`` (the three targets most correlated with
each share on the training split).
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.geneset_vs_hallmarks import add_axis_scores
from xaiwsi.evaluation.learning_curve import make_probe, slide_means
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.qc.mask import keep_mask, load_artifact_probs
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures
from xaiwsi.tissue.features import composition_features, load_feature_spec, tissue_dir
from xaiwsi.training.run_config import data_args

# Blocks are joined by '+'; split only where a known block starts, so table paths may contain '+'.
BLOCK_SEPARATOR = re.compile(r"\+(?=slide-mean(?:\+|$)|composition(?:\+|$)|table:)")


def feature_blocks(features: str) -> list[str]:
    return BLOCK_SEPARATOR.split(features)


class TablePrep:
    """Preprocessing of one table block, fitted on its training rows.

    Drops columns active (> 0) in fewer than ``min_active`` of the training slides, then
    with ``transform="clip"`` clips each column to its training range. Both guard against
    columns that are near-constant on the training cohort: the probe's StandardScaler
    divides by their tiny spread, so values seen only in a test cohort blow up.
    ``transform="whiten"`` instead projects on the training principal components (all with
    non-negligible variance) scaled to unit variance: a control for how much of a table's
    effect comes from the probe's per-column standardisation alone.
    """

    def __init__(self, x_train: np.ndarray, min_active: float, transform: str) -> None:
        self.keep = (x_train > 0).mean(axis=0) >= min_active
        if not self.keep.any():
            raise SystemExit(f"min_active {min_active} drops every column")
        self.transform = transform
        kept = x_train[:, self.keep]
        self.low, self.high = kept.min(axis=0), kept.max(axis=0)
        if transform == "whiten":
            self.center = kept.mean(axis=0)
            _, singular, vt = np.linalg.svd(kept - self.center, full_matrices=False)
            rank = singular > singular[0] * 1e-6
            self.components = vt[rank].T / (singular[rank] / np.sqrt(len(kept)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self.keep]
        if self.transform == "clip":
            return np.clip(x, self.low, self.high)
        if self.transform == "whiten":
            return (x - self.center) @ self.components
        return x


def table_prep_label(min_active: float, transform: str) -> str:
    return f"min{min_active:g}_{transform}"


def check_ranges(results: pd.DataFrame) -> None:
    """Fail loudly on impossible metric values."""
    spearman = results[["median_spearman", "mean_spearman"]]
    if ((spearman < -1) | (spearman > 1)).any().any():
        raise ValueError(f"Spearman outside [-1, 1]:\n{results}")
    if (results["median_r2"] > 1).any():
        raise ValueError(f"R² above 1:\n{results}")


def markdown_table(results: pd.DataFrame) -> str:
    cells = results.astype(object).map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(results.columns) + " |", "|" + "---|" * len(results.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def write_correlations(correlations: pd.DataFrame, output_dir: Path, command: str) -> None:
    """Long table of share/target Spearman; Markdown with the top-3 targets per share on train."""
    rho = correlations["spearman"]
    if ((rho < -1) | (rho > 1)).any():
        raise ValueError(f"Spearman outside [-1, 1]:\n{correlations[(rho < -1) | (rho > 1)]}")
    correlations.to_csv(output_dir / "composition_correlations.csv", index=False)
    sections = []
    for (condition, normalization), frame in correlations.groupby(["condition", "normalization"], sort=False):
        wide = frame.pivot_table(
            index=["feature", "target"], columns=["split", "cohort"], values="spearman", sort=False
        )
        wide.columns = [f"{split} {cohort}" for split, cohort in wide.columns]
        train_columns = [c for c in wide.columns if c.startswith("train ")]
        # Rank on the mean over training cohorts, so pooled conditions pick shared targets.
        strength = wide[train_columns].mean(axis=1).abs()
        top = (
            strength.groupby(level="feature", sort=False, group_keys=False)
            .nlargest(3)
            .index
        )
        table = wide.loc[top].reset_index()
        sections.append(f"### {condition}, {normalization}\n\n{markdown_table(table)}")
    (output_dir / "composition_correlations.md").write_text(
        "Spearman of composition shares with the true targets (top 3 per share by |rho| on "
        f"the training split). Generated by:\n\n```bash\n{command}\n```\n\n"
        + "\n\n".join(sections)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="comma-separated training cohort sets, e.g. tcga-luad tcga-luad,tcga-lusc",
    )
    parser.add_argument(
        "--normalizations", nargs="+", default=["per_cohort"], choices=["per_cohort", "pooled"]
    )
    parser.add_argument(
        "--tile-qc-threshold",
        type=float,
        help="average only tiles with artifact probability below this (xaiwsi.qc.tile_classifier)",
    )
    parser.add_argument(
        "--gene-sets", type=Path, default=None, help=".gmt/.json gene sets (default: config Hallmarks)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=["slide-mean"],
        help="+-joined blocks: slide-mean, composition, table:<parquet path relative to the cohort dir>",
    )
    parser.add_argument(
        "--table-min-active",
        type=float,
        nargs="+",
        default=[0.0],
        help="for table blocks: drop columns active in fewer than this fraction of training slides "
        "(one probe per value)",
    )
    parser.add_argument(
        "--table-transforms",
        nargs="+",
        default=["none"],
        choices=["none", "clip", "whiten"],
        help="for table blocks: none, clip to the training range, or PCA-whiten on the training slides "
        "(one probe per value)",
    )
    parser.add_argument(
        "--composition-spec", type=Path, default=Path("configs/tissue/composition_features.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    if "train_cohorts" not in data:
        raise SystemExit(f"{args.config} is not a CohortPathwayDataModule config.")
    if args.gene_sets is None:
        gene_sets = load_hallmarks(data["hallmark_gmt"])
    else:
        gene_sets = load_tavernari_signatures(args.gene_sets)
    names = list(gene_sets)
    output_names = list(add_axis_scores(pd.DataFrame(columns=names)).columns)

    def with_axes(values: np.ndarray) -> np.ndarray:
        return add_axis_scores(pd.DataFrame(values, columns=names))[output_names].to_numpy()
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**spec) for spec in data["train_cohorts"]]
    test_specs = [CohortSpec(**spec) for spec in data["test_cohorts"]]

    for features in args.features:
        for block in feature_blocks(features):
            if block not in ("slide-mean", "composition") and not block.startswith("table:"):
                raise SystemExit(f"Unknown feature block {block!r} in {features!r}")
    table_paths = sorted(
        {b.removeprefix("table:") for f in args.features for b in feature_blocks(f) if b.startswith("table:")}
    )
    feature_spec = None
    if any("composition" in feature_blocks(f) for f in args.features):
        feature_spec = load_feature_spec(args.composition_spec)
    rows, correlations = [], []
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

            # One cache per cohort and store, covering all its paired slides.
            means, shares, tables = {}, {}, {}
            by_cohort: dict[str, list] = {}
            for parts in splits.values():
                for part in parts:
                    by_cohort.setdefault(part.cohort, []).append(part)
            for cohort, parts in by_cohort.items():
                slide_ids = sorted({sid for p in parts for sid in p.pairs["slide_id"].astype(str)})
                spec = next(c for c in train_specs + test_specs if c.name == cohort)
                embedding_dir = spec.resolve(processed_root, embedding).embedding_dir
                masks, suffix = None, ""
                if args.tile_qc_threshold is not None:
                    probs = load_artifact_probs(embedding_dir)
                    masks = lambda sid, probs=probs: keep_mask(probs[sid], args.tile_qc_threshold)  # noqa: E731
                    suffix = f"_qc{args.tile_qc_threshold:g}"
                means[cohort] = slide_means(
                    parts[0].store,
                    slide_ids,
                    args.output_dir / f"slide_means_{cohort}_{embedding}{suffix}.parquet",
                    tile_masks=masks,
                )
                if feature_spec is not None:
                    shares[cohort] = composition_features(tissue_dir(embedding_dir), feature_spec)
                for path in table_paths:
                    table = pd.read_parquet(embedding_dir.parent.parent / path)
                    missing = sorted(set(slide_ids) - set(table.index.astype(str)))
                    if missing:
                        raise SystemExit(f"{cohort}: {len(missing)} slides missing from {path}, e.g. {missing[:3]}")
                    table.index = table.index.astype(str)
                    tables[cohort, path] = table

            def targets(part) -> np.ndarray:
                return part.scores.loc[part.pairs["sample_id"].astype(str), names].to_numpy()

            def raw_blocks(part, features: str) -> list[np.ndarray]:
                slide_ids = part.pairs["slide_id"].astype(str)
                blocks = []
                for block in feature_blocks(features):
                    if block == "slide-mean":
                        blocks.append(means[part.cohort].loc[slide_ids].to_numpy())
                    elif block == "composition":
                        blocks.append(shares[part.cohort].loc[slide_ids].to_numpy())
                    else:
                        blocks.append(tables[part.cohort, block.removeprefix("table:")].loc[slide_ids].to_numpy())
                return blocks

            if feature_spec is not None:
                for split, parts in splits.items():
                    for part in parts:
                        x = shares[part.cohort].loc[part.pairs["slide_id"].astype(str)]
                        y = pd.DataFrame(with_axes(targets(part)), columns=output_names)
                        for feature in x.columns:
                            for target in output_names:
                                rho = spearmanr(x[feature].to_numpy(), y[target].to_numpy()).statistic
                                correlations.append(
                                    {
                                        "condition": condition,
                                        "normalization": normalization,
                                        "split": split,
                                        "cohort": part.cohort,
                                        "feature": feature,
                                        "target": target,
                                        "spearman": rho,
                                    }
                                )

            y_train = np.concatenate([targets(p) for p in splits["train"]])
            for features in args.features:
                is_table = [b.startswith("table:") for b in feature_blocks(features)]
                train_blocks = [raw_blocks(p, features) for p in splits["train"]]
                stacked = [np.concatenate([b[i] for b in train_blocks]) for i in range(len(is_table))]
                variants = (
                    list(product(args.table_min_active, args.table_transforms)) if any(is_table) else [None]
                )
                for variant in variants:
                    preps = [TablePrep(x, *variant) if t else None for t, x in zip(is_table, stacked)]

                    def inputs(part, features: str = features, preps: list = preps) -> np.ndarray:
                        blocks = raw_blocks(part, features)
                        return np.concatenate([p(x) if p else x for p, x in zip(preps, blocks)], axis=1)

                    prep_label = "-" if variant is None else table_prep_label(*variant)
                    x_train = np.concatenate([inputs(p) for p in splits["train"]])
                    model = make_probe().fit(x_train, y_train)
                    predictions = []
                    for split in ("val", "test"):
                        for part in splits[split]:
                            y, y_pred = with_axes(targets(part)), with_axes(model.predict(inputs(part)))
                            predictions.append(
                                prediction_frame(part.pairs["case_id"], part.cohort, split, y, y_pred, output_names)
                            )
                            summary, _ = compute_pathway_metrics(y, y_pred, output_names)
                            rows.append(
                                {
                                    "condition": condition,
                                    "normalization": normalization,
                                    "features": features,
                                    "table_prep": prep_label,
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
                                f"{condition} {normalization} {features} {prep_label} {split} {part.cohort}: "
                                f"median rho {summary['median_spearman']:.3f} "
                                f"median R² {summary['median_r2']:.3f}"
                            )
                    stem = f"{condition}__{normalization}"
                    if features != "slide-mean":
                        stem += "__" + features.replace(".parquet", "").replace("/", "-").replace(":", "-")
                    if variant is not None and variant != (0.0, "none"):
                        stem += "__" + prep_label
                    pd.concat(predictions, ignore_index=True).to_parquet(
                        args.output_dir / "predictions" / f"{stem}.parquet"
                    )

    results = pd.DataFrame(rows)
    check_ranges(results)
    results.to_csv(args.output_dir / "probe_metrics.csv", index=False)

    command = "uv run python -m xaiwsi.evaluation.cohort_probe " + shlex.join(sys.argv[1:])
    table = markdown_table(results)
    (args.output_dir / "probe_metrics.md").write_text(
        f"Ridge probe on slide features ({embedding}; see the features column). Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{table}\n"
    )
    print(table)
    if correlations:
        write_correlations(pd.DataFrame(correlations), args.output_dir, command)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
