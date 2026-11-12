"""EXPOSE-style site masking: drop the slide-feature columns most associated with the TCGA site.

Scores each column of a slide feature table by its one-way ANOVA F across TCGA tissue-source
sites (second barcode field; sites with at least ``--min-site-cases`` cases), on the
**training split** of one TCGA cohort only (split of a ``CohortPathwayDataModule`` config), and
writes the table without the top fraction of columns, for the training cohort and every test
cohort that uses it as z-score reference. Baseline for the concept bags: EXPOSE (Witte et al.
2026) masks domain latents of an SAE; the same masking on the slide-mean dimensions is the
matched control.

    uv run python -m xaiwsi.evaluation.site_mask --config configs/training/pathway_cohorts.yaml \\
        --cohort tcga-luad --tables concepts/<sae>/slides_mean.parquet \\
        concepts/slide_means_h-optimus-1-fp16.parquet --fractions 0.05 0.1 0.25

Writes ``<table stem>_sitemask<fraction>.parquet`` next to each table (in every cohort dir), and
``<table stem>_sitemask.csv`` in the training cohort's dir (F per column, rank). Probe the
masked tables with ``xaiwsi.evaluation.cohort_probe --features table:<masked path>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import f_classif

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.shared_structure import tcga_site
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", required=True, help="a TCGA training cohort of the config")
    parser.add_argument("--tables", nargs="+", required=True, help="parquet paths relative to the cohort dirs")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25])
    parser.add_argument("--min-site-cases", type=int, default=10)
    args = parser.parse_args()

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, test_used = select_cohorts(train_specs, test_specs, [args.cohort])
    resolved = train_used[0].resolve(processed_root, embedding)
    splits = build_pathway_splits(
        [resolved],
        [],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    pairs = pd.concat([p.pairs for p in splits["train"]], ignore_index=True)
    sites = tcga_site(pairs["case_id"]).to_numpy()
    counts = pd.Series(sites).value_counts()
    keep = np.isin(sites, counts[counts >= args.min_site_cases].index)
    slide_ids = pairs["slide_id"].astype(str).to_numpy()[keep]
    sites = sites[keep]
    print(f"{args.cohort} train: {keep.sum()} cases from {len(set(sites))} sites with >= {args.min_site_cases} cases")
    cohort_dirs = [s.resolve(processed_root, embedding).embedding_dir.parent.parent for s in train_used + test_used]

    for path in args.tables:
        table = pd.read_parquet(cohort_dirs[0] / path)
        table.index = table.index.astype(str)
        f_stat, _ = f_classif(table.loc[slide_ids].to_numpy(np.float64), sites)
        f_stat = np.nan_to_num(f_stat, nan=0.0)  # constant columns carry no site signal
        order = np.argsort(-f_stat, kind="stable")
        stem = Path(path).with_suffix("")
        scores = pd.DataFrame({"column": table.columns[order], "f": f_stat[order], "rank": np.arange(len(order))})
        scores.to_csv(cohort_dirs[0] / f"{stem}_sitemask.csv", index=False)
        for fraction in args.fractions:
            dropped = set(table.columns[order[: round(fraction * len(order))]])
            kept = [c for c in table.columns if c not in dropped]
            for cohort_dir in cohort_dirs:
                frame = pd.read_parquet(cohort_dir / path)[kept]
                frame.to_parquet(cohort_dir / f"{stem}_sitemask{fraction:g}.parquet")
            print(f"{path}: dropped {len(dropped)}/{len(order)} columns at {fraction:g} (F >= {f_stat[order][len(dropped) - 1]:.2f})")


if __name__ == "__main__":
    main()
