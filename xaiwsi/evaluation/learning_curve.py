"""Learning curve: does WSI -> Hallmark performance still grow with more training cases?

Fits a ridge probe on per-slide mean tile embeddings, on random fractions of the stage-1
TCGA training split, with the TCGA-val split fixed and CPTAC as external test. Mean-z
targets use gene z-score statistics fit on the training subset only. If median Spearman
is still rising at 100%, more training cases (e.g. other TCGA cohorts) are worth adding.
The probe is a proxy for the tile-level models: they say how data-limited the task is,
not how well the stage-1 models do.

    uv run python -m xaiwsi.evaluation.learning_curve \\
        --config configs/training/pathway_baseline.yaml \\
        --embedding-subdir wsi-embeddings/h-optimus-1-fp16 \\
        --output-dir outputs/learning_curve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.pathway import build_case_pairs, load_log2_tpm_gene_symbols, split_cases
from xaiwsi.metrics.pathways import compute_pathway_metrics
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import fit_gene_zscore_stats, score_mean_z

RIDGE_ALPHAS = np.logspace(-2, 6, 17)


def slide_means(store, slide_ids: list[str], cache: Path, tile_masks=None) -> pd.DataFrame:
    """Mean tile embedding per slide, cached as parquet. ``tile_masks``: optional
    callable slide_id -> boolean mask of the tiles to average (all tiles if None)."""
    if cache.exists():
        means = pd.read_parquet(cache)
        if set(slide_ids) <= set(means.index):
            return means.loc[slide_ids]
    rows = {}
    for slide_id in tqdm(slide_ids, desc=f"slide means ({cache.stem})"):
        tiles, _, _ = store.load(slide_id)
        tiles = np.asarray(tiles, dtype=np.float32)
        if tile_masks is not None:
            mask = tile_masks(slide_id)
            if len(mask) != len(tiles):
                raise ValueError(f"{slide_id}: tile mask has {len(mask)} entries, store {len(tiles)}.")
            tiles = tiles[mask]
        rows[slide_id] = tiles.mean(axis=0)
    means = pd.DataFrame.from_dict(rows, orient="index")
    means.columns = [f"f{i}" for i in range(means.shape[1])]
    means.to_parquet(cache)
    return means


def make_probe() -> object:
    # An sklearn MLP probe was tried and dropped: with early stopping on a few dozen
    # cases it was unstable (median R² down to -5), which says nothing about data needs.
    return make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="stage-1 LightningCLI config")
    parser.add_argument(
        "--embedding-subdir",
        default=None,
        help="tile store dir relative to each cohort's processed dir (default: config dirs)",
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(args.config.read_text())["data"]
    hallmarks = load_hallmarks(data["hallmark_gmt"])
    names = list(hallmarks)
    cohorts = {}
    for name in ("tcga", "cptac"):
        rna_dir = Path(data[f"{name}_rna_dir"])
        embedding_dir = (
            rna_dir.parent / args.embedding_subdir
            if args.embedding_subdir
            else Path(data[f"{name}_embedding_dir"])
        )
        store = build_embedding_store_from_dir(root_dir=embedding_dir)
        expression = load_log2_tpm_gene_symbols(rna_dir)
        pairs = build_case_pairs(
            store,
            pd.read_csv(data[f"{name}_wsi_metadata"]),
            pd.read_csv(rna_dir / "samples.csv"),
            expression,
        )
        means = slide_means(
            store, pairs["slide_id"].astype(str).tolist(), args.output_dir / f"slide_means_{name}.parquet"
        )
        cohorts[name] = (expression, pairs, means)

    tcga_expr, tcga_pairs, tcga_means = cohorts["tcga"]
    cptac_expr, cptac_pairs, cptac_means = cohorts["cptac"]
    train_pairs, val_pairs = split_cases(
        tcga_pairs, val_fraction=data["val_fraction"], seed=data["split_seed"]
    )
    print(f"train {len(train_pairs)}, val {len(val_pairs)}, cptac {len(cptac_pairs)}")

    def features(means: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
        return means.loc[pairs["slide_id"].astype(str)].to_numpy()

    x_val, x_cptac = features(tcga_means, val_pairs), features(cptac_means, cptac_pairs)
    rows = []
    for fraction in args.fractions:
        n = max(int(round(fraction * len(train_pairs))), 10)
        for seed in range(args.seeds if fraction < 1.0 else 1):
            subset = train_pairs.sample(n=n, random_state=seed) if n < len(train_pairs) else train_pairs
            stats = fit_gene_zscore_stats(tcga_expr[subset["sample_id"].tolist()])

            def targets(expr: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
                return score_mean_z(
                    expr[pairs["sample_id"].tolist()], hallmarks, zscore_stats=stats
                )[names].to_numpy()

            y_train = targets(tcga_expr, subset)
            y_eval = {"val": targets(tcga_expr, val_pairs), "cptac": targets(cptac_expr, cptac_pairs)}
            x_train = features(tcga_means, subset)
            model = make_probe().fit(x_train, y_train)
            row = {"fraction": fraction, "n_train": n, "seed": seed}
            for split, x in (("val", x_val), ("cptac", x_cptac)):
                summary, _ = compute_pathway_metrics(y_eval[split], model.predict(x), names)
                row[f"{split}_median_spearman"] = summary["median_spearman"]
                row[f"{split}_median_r2"] = summary["median_r2"]
            rows.append(row)
            print(
                f"frac {fraction:.1f} n {n:3d} seed {seed} "
                f"val rho {row['val_median_spearman']:.3f} cptac rho {row['cptac_median_spearman']:.3f}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "learning_curve.csv", index=False)
    summary = (
        results.groupby(["fraction", "n_train"])
        .agg(
            val_rho=("val_median_spearman", "mean"),
            val_rho_sd=("val_median_spearman", "std"),
            val_r2=("val_median_r2", "mean"),
            cptac_rho=("cptac_median_spearman", "mean"),
            cptac_rho_sd=("cptac_median_spearman", "std"),
            cptac_r2=("cptac_median_r2", "mean"),
        )
        .round(3)
    )
    summary.to_csv(args.output_dir / "learning_curve_summary.csv")
    print(summary.to_string())
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
