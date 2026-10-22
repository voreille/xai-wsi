"""Stage 2b: how much Hallmark information do frozen RNA representations carry?

Linear (ridge) probes from RNA representations to mean-z Hallmark scores, on the same
TCGA pairs, split and train-only z-score statistics as the stage-1 WSI models, plus a
TCGA-vs-CPTAC cohort classifier as a measure of cohort shift. These are diagnostics of
the RNA encoders, not prediction tasks: the targets are derived from the same RNA.

    uv run python -m xaiwsi.evaluation.rna_probes \\
        --config configs/training/pathway_baseline.yaml \\
        --bulkformer-dir rna-embeddings/bulkformer/147M \\
        --output-dir outputs/rna_probes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.pathway import build_case_pairs, load_log2_tpm_gene_symbols, split_cases
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import fit_gene_zscore_stats, score_mean_z

RIDGE_ALPHAS = np.logspace(-3, 5, 17)


def reduce(
    train: np.ndarray, others: list[np.ndarray], k: int | None
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Standardize on train, then optionally project on the top-k train PCs."""
    scaler = StandardScaler().fit(train)
    train, others = scaler.transform(train), [scaler.transform(x) for x in others]
    if k is None or k >= min(train.shape):
        return train, others
    pca = PCA(n_components=k, random_state=0).fit(train)
    return pca.transform(train), [pca.transform(x) for x in others]


def ridge_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: list[np.ndarray]
) -> list[np.ndarray]:
    """Multi-output ridge with a per-target alpha chosen by efficient LOO on train."""
    model = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(x_train, y_train)
    return [model.predict(x) for x in x_eval]


def cohort_auc(x_a: np.ndarray, x_b: np.ndarray, seed: int = 0) -> float:
    """5-fold CV ROC AUC of a logistic classifier separating two cohorts."""
    x = np.concatenate([x_a, x_b])
    y = np.r_[np.zeros(len(x_a)), np.ones(len(x_b))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc").mean())


def lowrank_hallmark_reference(
    y_train: np.ndarray, y_eval: np.ndarray, k: int
) -> np.ndarray:
    """Project Hallmark scores on the top-k train PCs and reconstruct (oracle ceiling
    of a rank-k Hallmark head)."""
    mean = y_train.mean(0)
    _, _, vt = np.linalg.svd(y_train - mean, full_matrices=False)
    basis = vt[:k]
    return (y_eval - mean) @ basis.T @ basis + mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="stage-1 LightningCLI config; its data section defines pairs and split",
    )
    parser.add_argument(
        "--bulkformer-dir",
        default="rna-embeddings/bulkformer/147M",
        help="embedding dir relative to each cohort's processed dir",
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--hallmark-ranks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = yaml.safe_load(args.config.read_text())["data"]
    cohorts = {}
    for name in ("tcga", "cptac"):
        rna_dir = Path(data[f"{name}_rna_dir"])
        expression = load_log2_tpm_gene_symbols(rna_dir)
        pairs = build_case_pairs(
            build_embedding_store_from_dir(root_dir=data[f"{name}_embedding_dir"]),
            pd.read_csv(data[f"{name}_wsi_metadata"]),
            pd.read_csv(rna_dir / "samples.csv"),
            expression,
        )
        bulkformer = pd.read_parquet(rna_dir.parent / args.bulkformer_dir / "embeddings.parquet")
        cohorts[name] = (expression, pairs, bulkformer.filter(regex=r"^emb_"))

    tcga_expr, tcga_pairs, _ = cohorts["tcga"]
    cptac_expr, cptac_pairs, _ = cohorts["cptac"]
    train_pairs, val_pairs = split_cases(
        tcga_pairs, val_fraction=data["val_fraction"], seed=data["split_seed"]
    )
    ids = {
        "train": train_pairs["sample_id"].tolist(),
        "val": val_pairs["sample_id"].tolist(),
        "cptac": cptac_pairs["sample_id"].tolist(),
    }
    print(f"train {len(ids['train'])}, val {len(ids['val'])}, cptac {len(ids['cptac'])}")

    hallmarks = load_hallmarks(data["hallmark_gmt"])
    names = list(hallmarks.keys())
    stats = fit_gene_zscore_stats(tcga_expr[ids["train"]])
    y = {
        split: score_mean_z(expr[ids[split]], hallmarks, zscore_stats=stats)[names].to_numpy()
        for split, expr in (("train", tcga_expr), ("val", tcga_expr), ("cptac", cptac_expr))
    }

    # Representations: rows = samples, in split order.
    genes = tcga_expr.index[tcga_expr[ids["train"]].std(axis=1) > 0]
    genes = genes.intersection(cptac_expr.index)
    reps = {
        "bulkformer": {
            s: cohorts["tcga" if s != "cptac" else "cptac"][2].loc[ids[s]].to_numpy()
            for s in ids
        },
        "log2tpm": {
            s: (tcga_expr if s != "cptac" else cptac_expr).loc[genes, ids[s]].T.to_numpy()
            for s in ids
        },
    }

    rows = []
    for rep_name, x in reps.items():
        for k in [*args.dims, None]:
            x_tr, (x_val, x_cp) = reduce(x["train"], [x["val"], x["cptac"]], k)
            dim = x_tr.shape[1]
            p_val, p_cp = ridge_probe(x_tr, y["train"], [x_val, x_cp])
            s_val, _ = compute_pathway_metrics(y["val"], p_val, names)
            s_cp, _ = compute_pathway_metrics(y["cptac"], p_cp, names)
            auc = cohort_auc(np.concatenate([x_tr, x_val]), x_cp)
            rows.append(
                {
                    "representation": rep_name if k else f"{rep_name} (all)",
                    "dim": dim,
                    "val_median_r2": s_val["median_r2"],
                    "val_median_spearman": s_val["median_spearman"],
                    "cptac_median_r2": s_cp["median_r2"],
                    "cptac_median_spearman": s_cp["median_spearman"],
                    "cohort_auc": auc,
                }
            )
            print(rows[-1])

    for k in args.hallmark_ranks:
        p_val = lowrank_hallmark_reference(y["train"], y["val"], k)
        p_cp = lowrank_hallmark_reference(y["train"], y["cptac"], k)
        s_val, _ = compute_pathway_metrics(y["val"], p_val, names)
        s_cp, _ = compute_pathway_metrics(y["cptac"], p_cp, names)
        rows.append(
            {
                "representation": "hallmark PCs (oracle)",
                "dim": k,
                "val_median_r2": s_val["median_r2"],
                "val_median_spearman": s_val["median_spearman"],
                "cptac_median_r2": s_cp["median_r2"],
                "cptac_median_spearman": s_cp["median_spearman"],
                "cohort_auc": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "rna_probes.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
