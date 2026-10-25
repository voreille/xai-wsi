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
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.learning_curve import make_probe, slide_means
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.qc.mask import keep_mask, load_artifact_probs
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    if "train_cohorts" not in data:
        raise SystemExit(f"{args.config} is not a CohortPathwayDataModule config.")
    gene_sets = load_hallmarks(data["hallmark_gmt"])
    names = list(gene_sets)
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**spec) for spec in data["train_cohorts"]]
    test_specs = [CohortSpec(**spec) for spec in data["test_cohorts"]]

    rows = []
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
            features = {}
            by_cohort: dict[str, list] = {}
            for parts in splits.values():
                for part in parts:
                    by_cohort.setdefault(part.cohort, []).append(part)
            for cohort, parts in by_cohort.items():
                slide_ids = sorted({sid for p in parts for sid in p.pairs["slide_id"].astype(str)})
                masks, suffix = None, ""
                if args.tile_qc_threshold is not None:
                    spec = next(c for c in train_specs + test_specs if c.name == cohort)
                    probs = load_artifact_probs(spec.resolve(processed_root, embedding).embedding_dir)
                    masks = lambda sid, probs=probs: keep_mask(probs[sid], args.tile_qc_threshold)  # noqa: E731
                    suffix = f"_qc{args.tile_qc_threshold:g}"
                features[cohort] = slide_means(
                    parts[0].store,
                    slide_ids,
                    args.output_dir / f"slide_means_{cohort}_{embedding}{suffix}.parquet",
                    tile_masks=masks,
                )

            def xy(part) -> tuple[np.ndarray, np.ndarray]:
                x = features[part.cohort].loc[part.pairs["slide_id"].astype(str)].to_numpy()
                y = part.scores.loc[part.pairs["sample_id"].astype(str), names].to_numpy()
                return x, y

            x_train, y_train = map(np.concatenate, zip(*[xy(p) for p in splits["train"]], strict=True))
            model = make_probe().fit(x_train, y_train)
            predictions = []
            for split in ("val", "test"):
                for part in splits[split]:
                    x, y = xy(part)
                    y_pred = model.predict(x)
                    predictions.append(
                        prediction_frame(part.pairs["case_id"], part.cohort, split, y, y_pred, names)
                    )
                    summary, _ = compute_pathway_metrics(y, y_pred, names)
                    rows.append(
                        {
                            "condition": condition,
                            "normalization": normalization,
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
                        f"{condition} {normalization} {split} {part.cohort}: "
                        f"median rho {summary['median_spearman']:.3f} "
                        f"median R² {summary['median_r2']:.3f}"
                    )
            pd.concat(predictions, ignore_index=True).to_parquet(
                args.output_dir / "predictions" / f"{condition}__{normalization}.parquet"
            )

    results = pd.DataFrame(rows)
    check_ranges(results)
    results.to_csv(args.output_dir / "probe_metrics.csv", index=False)

    command = "uv run python -m xaiwsi.evaluation.cohort_probe " + shlex.join(sys.argv[1:])
    table = markdown_table(results)
    (args.output_dir / "probe_metrics.md").write_text(
        f"Ridge on slide-mean embeddings ({embedding}). Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{table}\n"
    )
    print(table)
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
