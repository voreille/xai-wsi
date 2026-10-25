"""What the shared WSI <-> RNA axes are: Hallmark correlates of the linear CCA axes.

Fits the linear CCA of ``xaiwsi.evaluation.shared_structure`` (slide means vs BulkFormer
embeddings, PCA ``--pca-dim`` per modality, TCGA train) and relates its canonical
variates to the 50 Hallmark mean-z scores of the same cases (targets as in a
``CohortPathwayDataModule`` config, ``tcga-luad`` z-score statistics):

1. Spearman correlation of each axis (slide side and RNA side) with each Hallmark.
   Hallmarks are ranked on TCGA train (no Hallmarks are used in the CCA fit, but the
   slide side is in-sample there), and the correlations are replicated on TCGA val and
   CPTAC.
2. Ridge probe Hallmarks <- the first ``k`` slide-side canonical coordinates, against the
   full slide mean and its top-``pca-dim`` PCs: how much of the WSI-predictable Hallmark
   signal lives in the shared axes.

    uv run python -m xaiwsi.evaluation.cca_hallmarks \\
        --config configs/training/alignment_baseline.yaml \\
        --pathway-config configs/training/pathway_cohorts.yaml \\
        --slide-means-dir outputs/cohort_probe --output-dir outputs/shared_structure

Writes ``cca_axis_hallmarks.csv`` (axis x side x split x pathway), ``cca_hallmark_probe.csv``
and ``cca_hallmarks.md``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.learning_curve import make_probe
from xaiwsi.evaluation.shared_structure import SPLITS, LinearCCA, check_ranges, load_frozen
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args

SPLIT_NAMES = {"train": "tcga-train", "val": "tcga-val", "test": "cptac"}


def hallmark_targets(pathway_config: Path, train_cohort: str, test_cohort: str) -> tuple[dict, list[str]]:
    """{split: DataFrame case_id x pathway} of mean-z targets for one train/test cohort."""
    data = data_args(yaml.safe_load(pathway_config.read_text()))
    gene_sets = load_hallmarks(data["hallmark_gmt"])
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_used, test_used = select_cohorts(
        [CohortSpec(**s) for s in data["train_cohorts"]],
        [CohortSpec(**s) for s in data["test_cohorts"]],
        [train_cohort],
    )
    test_used = [c for c in test_used if c.name == test_cohort]
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [c.resolve(processed_root, embedding) for c in test_used],
        gene_sets,
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    out = {}
    for split, (part,) in splits.items():
        scores = part.scores.loc[part.pairs["sample_id"].astype(str)]
        scores.index = part.pairs["case_id"].astype(str)
        out[split] = scores
    return out, list(gene_sets)


def spearman_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of ``a`` with every column of ``b``."""
    rho = spearmanr(a, b).statistic
    return rho[: a.shape[1], a.shape[1] :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--pathway-config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--train-cohort", default="tcga-luad")
    parser.add_argument("--test-cohort", default="cptac-luad")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--axes", type=int, default=6, help="axes described in the Markdown")
    parser.add_argument("--probe-ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16, 32])
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache (default: output dir)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output_dir)
    targets, names = hallmark_targets(args.pathway_config, args.train_cohort, args.test_cohort)
    # Same case split in both configs (split_cases, same seed), checked here.
    y_hall = {}
    for split in SPLITS:
        cases = splits[split]["pairs"]["case_id"].astype(str)
        missing = set(cases) - set(targets[split].index)
        if missing:
            raise SystemExit(f"{split}: {len(missing)} alignment cases without Hallmark targets in that split.")
        y_hall[split] = targets[split].loc[cases, names].to_numpy()

    cca = LinearCCA(args.pca_dim).fit(splits["train"]["x"], splits["train"]["y"])
    z = {split: cca.transform(splits[split]["x"], splits[split]["y"]) for split in SPLITS}

    rows = []
    for split in SPLITS:
        for side, coords in zip(("slide", "rna"), z[split], strict=True):
            rho = spearman_columns(coords, y_hall[split])
            for axis in range(args.pca_dim):
                for p, name in enumerate(names):
                    rows.append({"axis": axis + 1, "side": side, "split": SPLIT_NAMES[split],
                                 "pathway": name, "spearman": rho[axis, p]})
    corr = pd.DataFrame(rows)
    check_ranges(corr, ["spearman"], -1, 1)
    corr.to_csv(args.output_dir / "cca_axis_hallmarks.csv", index=False)

    # Per axis: canonical correlation per split, and the top Hallmarks of the slide side
    # ranked on TCGA train, with their val / CPTAC replication.
    wide = corr.pivot_table(index=["axis", "side", "pathway"], columns="split", values="spearman")
    summary = []
    for axis in range(1, args.axes + 1):
        canon = {
            SPLIT_NAMES[s]: np.corrcoef(z[s][0][:, axis - 1], z[s][1][:, axis - 1])[0, 1] for s in SPLITS
        }
        slide = wide.loc[(axis, "slide")]
        rna = wide.loc[(axis, "rna")]
        top = slide["tcga-train"].abs().sort_values(ascending=False).index[:4]
        for name in top:
            summary.append(
                {
                    "axis": axis,
                    "canon r train / val / cptac": " / ".join(f"{canon[k]:.2f}" for k in SPLIT_NAMES.values()),
                    "hallmark": name.removeprefix("HALLMARK_"),
                    "slide rho train": slide.loc[name, "tcga-train"],
                    "slide rho val": slide.loc[name, "tcga-val"],
                    "slide rho cptac": slide.loc[name, "cptac"],
                    "rna rho val": rna.loc[name, "tcga-val"],
                    "rna rho cptac": rna.loc[name, "cptac"],
                }
            )
    summary = pd.DataFrame(summary)
    breadth = (
        corr[(corr["side"] == "slide") & (corr["axis"] <= args.axes)]
        .assign(strong=lambda d: d["spearman"].abs() >= 0.3)
        .groupby(["axis", "split"])["strong"].sum().unstack("split")[list(SPLIT_NAMES.values())]
        .reset_index()
    )

    # Ridge probe: Hallmarks from the first k slide-side canonical coordinates.
    def probe(features: dict[str, np.ndarray], label: str) -> list[dict]:
        model = make_probe().fit(features["train"], y_hall["train"])
        out = []
        for split in ("val", "test"):
            summary_m, _ = compute_pathway_metrics(y_hall[split], model.predict(features[split]), names)
            out.append({"features": label, "dim": features["train"].shape[1], "split": SPLIT_NAMES[split],
                        "n": len(y_hall[split]), "median_spearman": summary_m["median_spearman"],
                        "mean_spearman": summary_m["mean_spearman"], "median_r2": summary_m["median_r2"]})
        return out

    probes = probe({s: splits[s]["x"] for s in SPLITS}, "slide mean (all features)")
    probes += probe({s: (splits[s]["x"] - cca.mx) @ LinearCCA._whitener(splits["train"]["x"], args.pca_dim)[1]
                     for s in SPLITS}, f"slide mean PCA {args.pca_dim}")
    for k in args.probe_ks:
        probes += probe({s: z[s][0][:, :k] for s in SPLITS}, f"CCA slide axes 1-{k}")
    probes = pd.DataFrame(probes)
    check_ranges(probes, ["median_spearman", "mean_spearman"], -1, 1)
    if (probes["median_r2"] > 1).any():
        raise ValueError(f"R² above 1:\n{probes}")
    probes.to_csv(args.output_dir / "cca_hallmark_probe.csv", index=False)

    n = {SPLIT_NAMES[s]: len(y_hall[s]) for s in SPLITS}
    command = "uv run python -m xaiwsi.evaluation.cca_hallmarks " + shlex.join(sys.argv[1:])
    text = (
        f"Hallmark correlates of the linear CCA axes (PCA {args.pca_dim} per modality, fit on TCGA "
        f"train), Spearman with mean-z Hallmark scores ({args.train_cohort} statistics); n = {n}. "
        f"Top Hallmarks per axis are ranked by |rho| of the slide side on TCGA train. Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{markdown_table(summary)}\n\n"
        f"Number of Hallmarks with |rho| >= 0.3 (slide side):\n\n{markdown_table(breadth)}\n\n"
        f"Ridge probe (RidgeCV, fit on TCGA train) Hallmarks <- slide features:\n\n{markdown_table(probes)}\n"
    )
    (args.output_dir / "cca_hallmarks.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
