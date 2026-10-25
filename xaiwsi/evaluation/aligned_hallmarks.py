"""Stage 2d: ridge probe Hallmarks <- aligned slide embeddings, vs frozen slide features.

For each alignment run (best checkpoint), embeds the TCGA train/val and CPTAC slides
with the run's slide encoder (L2-normalized ``slide`` output, WSI only) and fits a ridge
probe to the 50 Hallmark mean-z scores on TCGA train (``make_probe``, as in
``cohort_probe``). Baselines on the same cases: the frozen slide mean (the
``cohort_probe`` tcga-luad row) and the first ``--cca-axes`` slide-side axes of the
linear CCA of ``xaiwsi.evaluation.shared_structure``.

    uv run python -m xaiwsi.evaluation.aligned_hallmarks \\
        --config configs/training/alignment_baseline.yaml \\
        --pathway-config configs/training/pathway_cohorts.yaml \\
        --runs runs/alignment/xai-wsi/aln-1025-* \\
        --slide-means-dir outputs/cohort_probe --output-dir outputs/aligned_hallmarks

Runs without ``test_metrics.json`` (unfinished) are skipped. Writes ``probe_metrics.csv``
(one row per features x split), ``probe_metrics.md`` (per-setting mean and sd over seeds,
with the generating command) and ``predictions/<features>.parquet`` (per-case, the input
of ``xaiwsi.evaluation.paired_bootstrap``; seeds of a setting as a glob).
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from xaiwsi.data.alignment import cache_slide_tiles
from xaiwsi.evaluation.cca_hallmarks import SPLIT_NAMES, hallmark_targets
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.evaluation.shared_structure import SPLITS, LinearCCA, aligned_embeddings, check_ranges, load_frozen
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.training.run_config import data_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--pathway-config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="alignment run dirs")
    parser.add_argument("--train-cohort", default="tcga-luad")
    parser.add_argument("--test-cohort", default="cptac-luad")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--cca-axes", type=int, default=8)
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache (default: output dir)")
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    runs = [r for r in args.runs if (r / "test_metrics.json").exists()]
    for r in sorted(set(args.runs) - set(runs)):
        print(f"skip {r.name}: no test_metrics.json")
    if not runs:
        raise SystemExit("No finished runs.")

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output_dir)
    targets, names = hallmark_targets(args.pathway_config, args.train_cohort, args.test_cohort)
    cases = {s: splits[s]["pairs"]["case_id"].astype(str) for s in SPLITS}
    y = {}
    for split in SPLITS:
        missing = set(cases[split]) - set(targets[split].index)
        if missing:
            raise SystemExit(f"{split}: {len(missing)} alignment cases without Hallmark targets in that split.")
        y[split] = targets[split].loc[cases[split], names].to_numpy()
    cohorts = {"val": args.train_cohort, "test": args.test_cohort}

    rows = []

    def probe(features: dict[str, np.ndarray], label: str, setting: str) -> None:
        model = make_probe().fit(features["train"], y["train"])
        frames = []
        for split in ("val", "test"):
            pred = model.predict(features[split])
            frames.append(prediction_frame(cases[split], cohorts[split], split, y[split], pred, names))
            summary, _ = compute_pathway_metrics(y[split], pred, names)
            rows.append({"features": label, "setting": setting, "dim": features["train"].shape[1],
                         "split": SPLIT_NAMES[split], "n": len(y[split]),
                         "median_spearman": summary["median_spearman"],
                         "mean_spearman": summary["mean_spearman"], "median_r2": summary["median_r2"]})
        pd.concat(frames, ignore_index=True).to_parquet(args.output_dir / "predictions" / f"{label}.parquet")
        print(f"{label}: " + ", ".join(f"{r['split']} rho {r['median_spearman']:.3f}" for r in rows[-2:]))

    probe({s: splits[s]["x"] for s in SPLITS}, "slide-mean", "slide mean (frozen)")
    cca = LinearCCA(args.pca_dim).fit(splits["train"]["x"], splits["train"]["y"])
    probe({s: cca.transform(splits[s]["x"], splits[s]["y"])[0][:, : args.cca_axes] for s in SPLITS},
          f"cca{args.pca_dim}-axes{args.cca_axes}", f"linear CCA slide axes 1-{args.cca_axes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = {}
    for split in SPLITS:
        ids = splits[split]["pairs"]["slide_id"].astype(str).tolist()
        tiles.update(cache_slide_tiles(splits[split]["embedding_dir"], ids, workers=args.cache_workers))
    for run_dir in runs:
        emb = aligned_embeddings(run_dir, splits, tiles, device)
        probe({s: emb[s][0] for s in SPLITS}, run_dir.name, re.sub(r"-s\d+$", "", run_dir.name))

    results = pd.DataFrame(rows)
    check_ranges(results, ["median_spearman", "mean_spearman"], -1, 1)
    if (results["median_r2"] > 1).any():
        raise ValueError(f"R² above 1:\n{results}")
    results.to_csv(args.output_dir / "probe_metrics.csv", index=False)

    grouped = results.groupby(["setting", "split"], sort=False)
    summary = grouped[["median_spearman", "mean_spearman", "median_r2"]].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "n_runs", grouped.size())
    summary.insert(0, "dim", grouped["dim"].first())
    summary = summary.reset_index()
    command = "uv run python -m xaiwsi.evaluation.aligned_hallmarks " + shlex.join(sys.argv[1:])
    text = (
        "Ridge probe (RidgeCV, fit on TCGA train) Hallmarks <- slide features; aligned = "
        "L2-normalized slide embedding of each run's best checkpoint; mean and sd over seeds. "
        f"Generated by:\n\n```bash\n{command}\n```\n\n{markdown_table(summary)}\n"
    )
    (args.output_dir / "probe_metrics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
