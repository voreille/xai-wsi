"""How much of an extra gene-set score do the 50 Hallmarks already explain?

Scores extra gene sets (e.g. the approximate Tavernari lepidic/solid sets) with mean-z on
the same TCGA pairs, split and train-only z-score statistics as the stage-1 WSI models.
Every "LEPIDIC_<src>" / "SOLID_<src>" pair also gets an axis score
"L2S_<src>" = solid - lepidic. For each extra score it reports gene coverage, the Hallmarks
with the largest gene overlap and correlation, and how well a ridge model on the 50
Hallmark scores (fit on TCGA-train) predicts it on TCGA-val and CPTAC. These are RNA-side
diagnostics of the targets; no WSI model is involved.

    uv run python -m xaiwsi.evaluation.geneset_vs_hallmarks \\
        --config configs/training/pathway_baseline.yaml \\
        --gene-sets data/external/tavernari/derived/tavernari_l2s_approx.json \\
        --output-dir outputs/tavernari_approx
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.pathway import build_case_pairs, load_log2_tpm_gene_symbols, split_cases
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import fit_gene_zscore_stats, score_mean_z
from xaiwsi.rna.pathways.tavernari import load_tavernari_signatures

RIDGE_ALPHAS = np.logspace(-3, 5, 17)
SPLITS = ("train", "val", "cptac")


def add_axis_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Add L2S_<src> = SOLID_<src> - LEPIDIC_<src> for every matching pair of columns."""
    scores = scores.copy()
    for column in list(scores.columns):
        match = re.search(r"SOLID_(\w+)$", column)
        lepidic = column.replace("SOLID", "LEPIDIC") if match else None
        if lepidic in scores.columns:
            scores[f"L2S_{match.group(1)}"] = scores[column] - scores[lepidic]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="stage-1 LightningCLI config; its data section defines pairs and split",
    )
    parser.add_argument("--gene-sets", type=Path, required=True, help=".gmt or .json")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = yaml.safe_load(args.config.read_text())["data"]
    expression, pairs = {}, {}
    for name in ("tcga", "cptac"):
        rna_dir = Path(data[f"{name}_rna_dir"])
        expression[name] = load_log2_tpm_gene_symbols(rna_dir)
        pairs[name] = build_case_pairs(
            build_embedding_store_from_dir(root_dir=data[f"{name}_embedding_dir"]),
            pd.read_csv(data[f"{name}_wsi_metadata"]),
            pd.read_csv(rna_dir / "samples.csv"),
            expression[name],
        )
    train_pairs, val_pairs = split_cases(
        pairs["tcga"], val_fraction=data["val_fraction"], seed=data["split_seed"]
    )
    ids = {
        "train": train_pairs["sample_id"].tolist(),
        "val": val_pairs["sample_id"].tolist(),
        "cptac": pairs["cptac"]["sample_id"].tolist(),
    }
    cohort = {"train": "tcga", "val": "tcga", "cptac": "cptac"}
    print(f"train {len(ids['train'])}, val {len(ids['val'])}, cptac {len(ids['cptac'])}")

    hallmarks = load_hallmarks(data["hallmark_gmt"])
    extra = load_tavernari_signatures(args.gene_sets)
    stats = fit_gene_zscore_stats(expression["tcga"][ids["train"]])
    hallmark_scores, extra_scores = {}, {}
    for split in SPLITS:
        expr = expression[cohort[split]][ids[split]]
        hallmark_scores[split] = score_mean_z(expr, hallmarks, zscore_stats=stats)[list(hallmarks)]
        extra_scores[split] = add_axis_scores(
            score_mean_z(expr, extra, zscore_stats=stats)[list(extra)]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        extra_scores[split].to_parquet(args.output_dir / f"scores_{split}.parquet")

    # Gene coverage and Hallmark gene overlap of each extra set.
    available = set(stats.index) & set(expression["cptac"].index)
    coverage = []
    for name, genes in extra.items():
        overlaps = sorted(
            ((len(genes & members), h) for h, members in hallmarks.items()), reverse=True
        )[: args.top]
        coverage.append(
            {
                "gene_set": name,
                "n_genes": len(genes),
                "n_scored": len(genes & available),
                "top_hallmark_overlap": "; ".join(
                    f"{h.removeprefix('HALLMARK_')} ({n})" for n, h in overlaps if n
                ),
            }
        )
    coverage = pd.DataFrame(coverage)

    # Redundancy with the Hallmark scores.
    x = {split: hallmark_scores[split].to_numpy() for split in SPLITS}
    tcga = pd.concat([hallmark_scores["train"], hallmark_scores["val"]])
    rows = []
    for name in extra_scores["train"].columns:
        y = {split: extra_scores[split][name].to_numpy() for split in SPLITS}
        target = pd.concat([extra_scores["train"][name], extra_scores["val"][name]])
        rho = tcga.apply(lambda column: spearmanr(column, target).statistic)
        best = rho.abs().sort_values(ascending=False).index[: args.top]
        model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)
        ).fit(x["train"], y["train"])
        row = {
            "score": name,
            "top_hallmark_spearman_tcga": "; ".join(
                f"{h.removeprefix('HALLMARK_')} {rho[h]:+.2f}" for h in best
            ),
        }
        for split in ("val", "cptac"):
            pred = model.predict(x[split])
            row[f"ridge50_r2_{split}"] = r2_score(y[split], pred)
            row[f"ridge50_spearman_{split}"] = spearmanr(y[split], pred).statistic
        row["cptac_mean_shift_sd"] = (y["cptac"].mean() - y["train"].mean()) / y["train"].std()
        rows.append(row)
    redundancy = pd.DataFrame(rows)

    axis = [c for c in extra_scores["train"].columns if c.startswith("L2S_")]
    agreement = {
        split: extra_scores[split][axis].corr(method="spearman").round(3) for split in SPLITS
    }

    coverage.to_csv(args.output_dir / "coverage.csv", index=False)
    redundancy.to_csv(args.output_dir / "redundancy.csv", index=False)
    (args.output_dir / "axis_agreement.json").write_text(
        json.dumps({split: frame.to_dict() for split, frame in agreement.items()}, indent=1)
    )
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print("\nCoverage and gene overlap with Hallmarks:")
        print(coverage.to_string(index=False))
        print("\nRedundancy with the 50 Hallmark scores (ridge fit on TCGA-train):")
        print(redundancy.round(3).to_string(index=False))
        for split in SPLITS:
            print(f"\nL2S axis agreement (Spearman), {split}:")
            print(agreement[split].to_string())
    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
