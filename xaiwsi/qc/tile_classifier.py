"""Tile artifact classifier on tile embeddings: train from labelled clusters, score all tiles.

``train``: logistic regression on the PCA features of ``xaiwsi.qc.tile_clusters`` (its
``model.npz``), with labels from a YAML file that maps the refined sub-clusters to
artifact / tissue / excluded (``configs/qc/tile_artifacts.yaml``). Reports slide-grouped
5-fold cross-validated AUC, then folds PCA + logistic weights into one linear map on the
raw embedding (``w``, ``b``) and saves it.

    uv run python -m xaiwsi.qc.tile_classifier train \\
        --labels configs/qc/tile_artifacts.yaml --output outputs/tile_qc/artifact_classifier.npz

``score``: artifact probability of every tile of every slide of the given cohorts, saved
as ``<processed_root>/<cohort>/tile-qc/<embedding>/artifact_prob.npz`` (one float16 array
per slide id, in the store's tile order), plus ``artifact_summary.csv`` (per slide: tiles,
fraction above 0.5) and a montage per probability bin of tiles from slides not used for
training (``<montage-dir>/<cohort>_p<lo>-<hi>.png``) to check the classifier by eye.

    uv run python -m xaiwsi.qc.tile_classifier score \\
        --classifier outputs/tile_qc/artifact_classifier.npz \\
        --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad tcga-lusc tcga-brca cptac-luad cptac-lusc \\
        --montage-dir outputs/tile_qc/score_montages
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.qc.tile_clusters import pick_spread, render
from xaiwsi.training.run_config import data_args

BINS = [(0.0, 0.1), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]


def labelled_tiles(labels: dict) -> pd.DataFrame:
    """Sampled tiles with label 1 (artifact) / 0 (tissue); excluded sub-clusters dropped."""
    tiles = pd.read_parquet(Path(labels["clusters_dir"]) / "tiles.parquet")
    artifact = set(labels["artifact_subclusters"])
    tissue = set(labels.get("tissue_subclusters", {}))
    excluded = set(labels.get("excluded_subclusters", {}))
    named = artifact | tissue | excluded
    refined = set(tiles.loc[tiles["sub"] >= 0, "sub"])
    if refined != named:
        raise SystemExit(f"Sub-clusters {sorted(refined)} but labels name {sorted(named)}.")
    tiles = tiles[~tiles["sub"].isin(excluded)].copy()
    tiles["label"] = tiles["sub"].isin(artifact).astype(int)
    return tiles


def load_tile_embeddings(tiles: pd.DataFrame, embedding_dirs: dict[str, Path]) -> np.ndarray:
    parts = {}
    for (cohort, slide_id), group in tiles.groupby(["cohort", "slide_id"], sort=False):
        store = build_embedding_store_from_dir(root_dir=embedding_dirs[cohort])
        emb, _, _ = store.load(slide_id)
        parts[(cohort, slide_id)] = (group.index, np.asarray(emb[group["tile"].to_numpy()], dtype=np.float32))
    dim = next(iter(parts.values()))[1].shape[1]
    out = np.empty((len(tiles), dim), dtype=np.float32)
    position = pd.Series(np.arange(len(tiles)), index=tiles.index)
    for index, emb in parts.values():
        out[position.loc[index].to_numpy()] = emb
    return out


def cohort_dirs(config_path: Path) -> tuple[dict, dict[str, CohortSpec]]:
    data = data_args(yaml.safe_load(config_path.read_text()))
    specs = {s["name"]: CohortSpec(**s) for s in data["train_cohorts"] + data["test_cohorts"]}
    return data, specs


def train(args) -> None:
    labels = yaml.safe_load(args.labels.read_text())
    clusters_dir = Path(labels["clusters_dir"])
    tiles = labelled_tiles(labels)
    data, specs = cohort_dirs(args.config)
    embedding = data.get("embedding", "h-optimus-1")
    dirs = {
        name: specs[name].resolve(data.get("processed_root"), embedding).embedding_dir
        for name in tiles["cohort"].unique()
    }
    x = load_tile_embeddings(tiles, dirs)
    pca = np.load(clusters_dir / "model.npz")
    z = (x - pca["pca_mean"]) @ pca["pca_components"].T
    y = tiles["label"].to_numpy()

    def model() -> LogisticRegression:
        return LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000)

    oof = np.zeros(len(y))
    for fit, held in GroupKFold(n_splits=5).split(z, y, groups=tiles["slide_id"]):
        oof[held] = model().fit(z[fit], y[fit]).predict_proba(z[held])[:, 1]
    auc = roc_auc_score(y, oof)
    pred = oof >= 0.5
    precision = (pred & (y == 1)).sum() / max(pred.sum(), 1)
    recall = (pred & (y == 1)).sum() / (y == 1).sum()
    print(
        f"{len(y)} tiles ({y.sum()} artifact), slide-grouped 5-fold: AUC {auc:.4f}, "
        f"precision {precision:.3f}, recall {recall:.3f} at 0.5"
    )
    final = model().fit(z, y)
    w = pca["pca_components"].T @ final.coef_[0]
    b = float(final.intercept_[0] - pca["pca_mean"] @ w)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        w=w.astype(np.float32),
        b=np.float32(b),
        embedding=embedding,
        cv_auc=auc,
        cv_precision=precision,
        cv_recall=recall,
        train_slides=np.asarray(sorted(tiles["slide_id"].unique())),
    )
    print(f"Classifier: {args.output}")


def score(args) -> None:
    clf = np.load(args.classifier)
    w, b = clf["w"], float(clf["b"])
    train_slides = set(clf["train_slides"].tolist())
    data, specs = cohort_dirs(args.config)
    embedding = str(clf["embedding"])
    if embedding != data.get("embedding", "h-optimus-1"):
        raise SystemExit(f"Classifier is for {embedding!r}, config uses {data.get('embedding')!r}.")
    overrides = dict(entry.split("=", 1) for entry in args.slides_root)
    rng = np.random.default_rng(args.seed)
    for name in args.cohorts:
        cohort = specs[name].resolve(data.get("processed_root"), embedding)
        store = build_embedding_store_from_dir(root_dir=cohort.embedding_dir)
        slide_ids = sorted(store.slide_ids())

        def one(slide_id: str):
            emb, coords, attrs = store.load(slide_id)
            logits = np.asarray(emb, dtype=np.float32) @ w + b
            prob = expit(logits)
            return slide_id, prob.astype(np.float16), np.asarray(coords), attrs

        probs, rows, tile_rows = {}, [], []
        with ThreadPoolExecutor(args.workers) as pool:
            for slide_id, prob, coords, attrs in pool.map(one, slide_ids):
                probs[slide_id] = prob
                rows.append(
                    {"slide_id": slide_id, "n_tiles": len(prob), "artifact_fraction": float((prob >= 0.5).mean())}
                )
                if slide_id not in train_slides:
                    pick = rng.choice(len(prob), size=min(50, len(prob)), replace=False)
                    flagged = np.flatnonzero(prob >= 0.3)
                    pick = np.union1d(pick, rng.choice(flagged, size=min(50, len(flagged)), replace=False))
                    tile_rows.append(
                        pd.DataFrame(
                            {
                                "cohort": name,
                                "slide_id": slide_id,
                                "x": coords[pick, 0],
                                "y": coords[pick, 1],
                                "patch_size": int(attrs["patch_size"]),
                                "relative_wsi_path": str(attrs["relative_wsi_path"]),
                                "prob": prob[pick].astype(np.float32),
                            }
                        )
                    )
        out_dir = cohort.embedding_dir.parent.parent / "tile-qc" / embedding
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "artifact_prob.npz", **probs)
        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "artifact_summary.csv", index=False)
        frac = summary["artifact_fraction"]
        print(
            f"{name}: {len(summary)} slides, artifact tiles {frac.mean():.1%} on average "
            f"(median {frac.median():.1%}, p90 {frac.quantile(0.9):.1%}, max {frac.max():.1%}); "
            f"{(frac > 0.2).sum()} slides > 20% -> {out_dir}"
        )
        if args.montage_dir:
            args.montage_dir.mkdir(parents=True, exist_ok=True)
            sample = pd.concat(tile_rows, ignore_index=True)
            root = Path(overrides.get(name) or cohort.slides_root_dir)
            for lo, hi in BINS:
                group = sample[(sample["prob"] >= lo) & (sample["prob"] < hi)]
                if len(group):
                    render(pick_spread(group, 48, rng), {name: root}, 96, 8).save(
                        args.montage_dir / f"{name}_p{lo:.1f}-{hi:.1f}.png"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/training/pathway_cohorts.yaml"))
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--classifier", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cohorts", nargs="+", required=True)
    p.add_argument("--slides-root", nargs="*", default=[], help="'<cohort>=<dir>' overrides")
    p.add_argument("--montage-dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
