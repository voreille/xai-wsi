"""Shared WSI <-> RNA structure: linear CCA on frozen embeddings, and the site shortcut.

Diagnostics for the common part of the two modalities, on the pairs and split of an
``AlignmentDataModule`` config (TCGA train/val, CPTAC test):

1. Linear CCA between slide-mean tile embeddings and the RNA embeddings (BulkFormer),
   after a PCA to ``--pca-dims`` per modality, fitted on TCGA train only. Reports the
   held-out canonical correlation of each component on TCGA val and CPTAC, with a
   permutation null (train pairs shuffled), and cross-modal retrieval in the top-``k``
   canonical coordinates. This is the linear baseline for the contrastive aligner.
2. The same with the tissue source site regressed out of both modalities (site means fit
   on TCGA train; CPTAC sites are unknown to the fit and left unchanged).
3. Site leakage: cross-validated logistic-regression probe for the TCGA site (``TCGA-XX``)
   on each representation (raw, CCA, and the aligned embeddings of ``--runs``), and
   site enrichment of the top-5 retrieved TCGA-val candidates.

CPTAC rows use TCGA-train statistics; ``cptac-centered`` also centers each CPTAC modality
on its own mean, which needs CPTAC RNA and is a diagnostic only.

    uv run python -m xaiwsi.evaluation.shared_structure \\
        --config configs/training/alignment_baseline.yaml \\
        --runs runs/alignment/xai-wsi/kq0d13qf runs/alignment/xai-wsi/orpg08id \\
        --slide-means-dir outputs/cohort_probe --output-dir outputs/shared_structure

Writes ``cca_spectrum.csv``, ``cca_retrieval.csv``, ``site_probe.csv``,
``site_enrichment.csv`` and ``shared_structure.md`` (tables with the generating command).
"""

from __future__ import annotations

import argparse
import importlib
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xaiwsi.data.alignment import cache_slide_tiles
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.learning_curve import slide_means
from xaiwsi.evaluation.predict_run import best_checkpoint
from xaiwsi.metrics.retrieval import retrieval_metrics
from xaiwsi.training.alignment_datamodule import AlignmentDataModule
from xaiwsi.training.run_config import data_args

SPLITS = ("train", "val", "test")
RETRIEVAL_KS = (2, 4, 8, 16, 32, 64)


# --- linear CCA ---------------------------------------------------------------------


class LinearCCA:
    """CCA after a per-modality PCA whitening to ``pca_dim`` components (fit on train)."""

    def __init__(self, pca_dim: int):
        self.pca_dim = pca_dim

    @staticmethod
    def _whitener(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
        mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
        scale = s[:dim] / np.sqrt(len(x) - 1)
        return mean, vt[:dim].T / scale

    def fit(self, x: np.ndarray, y: np.ndarray, y_perm: np.ndarray | None = None) -> LinearCCA:
        self.mx, wx = self._whitener(x, self.pca_dim)
        self.my, wy = self._whitener(y, self.pca_dim)
        self.set_pairing(x, y if y_perm is None else y_perm, wx, wy)
        return self

    def set_pairing(self, x, y, wx, wy) -> None:
        xw, yw = (x - self.mx) @ wx, (y - self.my) @ wy
        a, rho, bt = np.linalg.svd(xw.T @ yw / (len(x) - 1))
        self.train_corr = rho
        self.wx, self.wy = wx @ a, wy @ bt.T

    def transform(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (x - self.mx) @ self.wx, (y - self.my) @ self.wy


def column_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return (a * b).mean(0)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- data ---------------------------------------------------------------------------


def tcga_site(case_ids: pd.Series) -> pd.Series:
    """Tissue source site, the second field of a TCGA barcode (``TCGA-XX-YYYY``)."""
    return case_ids.astype(str).str.split("-").str[1]


def load_frozen(data: dict, slide_means_dir: Path) -> dict[str, dict]:
    """Per split: pairs, slide means (standardized with TCGA-train stats), RNA embeddings
    (standardized as in ``AlignmentDataModule``)."""
    dm = AlignmentDataModule(**{**data, "cache_tiles": False})
    dm.setup()
    datasets = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    out = {}
    for split, ds in datasets.items():
        embedding_dir = Path(dm._embedding_dirs[id(ds)])
        cohort = embedding_dir.parent.parent.name
        ids = ds.pairs["slide_id"].astype(str).tolist()
        means = slide_means(
            ds.embedding_store, ids, slide_means_dir / f"slide_means_{cohort}_{embedding_dir.name}.parquet"
        )
        out[split] = {
            "dataset": ds,
            "embedding_dir": embedding_dir,
            "pairs": ds.pairs,
            "x": means.loc[ids].to_numpy(np.float64),
            "y": ds.rna.astype(np.float64),
        }
    mean, std = out["train"]["x"].mean(0), out["train"]["x"].std(0) + 1e-8
    for split in SPLITS:
        out[split]["x"] = (out[split]["x"] - mean) / std
    for split in ("train", "val"):
        out[split]["site"] = tcga_site(out[split]["pairs"]["case_id"]).to_numpy()
    return out


def remove_site_means(splits: dict[str, dict]) -> dict[str, dict]:
    """Subtract each TCGA site's train mean offset (from the train grand mean); sites
    absent from train and CPTAC cases are left unchanged."""
    train = splits["train"]
    result = {}
    effects = {}
    for key in ("x", "y"):
        grand = train[key].mean(0)
        frame = pd.DataFrame(train[key]).groupby(train["site"]).mean()
        effects[key] = {site: row.to_numpy() - grand for site, row in frame.iterrows()}
    for split, part in splits.items():
        new = dict(part)
        for key in ("x", "y"):
            values = part[key].copy()
            if "site" in part:
                for i, site in enumerate(part["site"]):
                    values[i] -= effects[key].get(site, 0.0)
            new[key] = values
        result[split] = new
    return result


# --- aligned embeddings from checkpoints --------------------------------------------


def instantiate(spec):
    if isinstance(spec, dict) and "class_path" in spec:
        module, _, name = spec["class_path"].rpartition(".")
        kwargs = {k: instantiate(v) for k, v in (spec.get("init_args") or {}).items()}
        return getattr(importlib.import_module(module), name)(**kwargs)
    return spec


@torch.no_grad()
def aligned_embeddings(run_dir: Path, splits: dict[str, dict], tiles: dict, device) -> dict:
    """L2-normalized slide and RNA embeddings of each split from the run's best checkpoint."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    aligner = instantiate(config["model"]["aligner"])
    state = torch.load(best_checkpoint(run_dir), map_location="cpu", weights_only=False)["state_dict"]
    aligner.load_state_dict({k.removeprefix("aligner."): v for k, v in state.items() if k.startswith("aligner.")})
    aligner = aligner.to(device).eval()
    out = {}
    for split, part in splits.items():
        ds = part["dataset"]
        slides, rnas = [], []
        for start in range(0, len(ds), 16):
            idx = range(start, min(start + 16, len(ds)))
            batch = {
                "tiles": [tiles[str(ds.pairs.iloc[i]["slide_id"])].to(device) for i in idx],
                "rna": torch.from_numpy(ds.rna[list(idx)]).to(device),
            }
            result = aligner(batch)
            slides.append(result["slide"].cpu())
            rnas.append(result["rna"].cpu())
        out[split] = (torch.cat(slides).numpy(), torch.cat(rnas).numpy())
    return out


# --- site probes --------------------------------------------------------------------


def site_probe(features: np.ndarray, sites: np.ndarray, min_cases: int, seed: int = 0) -> dict:
    """5-fold CV balanced accuracy of a logistic-regression site classifier, on the
    cases of sites with at least ``min_cases`` cases."""
    counts = pd.Series(sites).value_counts()
    keep = np.isin(sites, counts[counts >= min_cases].index)
    x, y = features[keep], sites[keep]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, x, y, cv=cv)
    return {
        "n_cases": int(keep.sum()),
        "n_sites": int(y.size and len(set(y))),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "chance": 1.0 / len(set(y)),
    }


def site_enrichment(query: np.ndarray, gallery: np.ndarray, sites: np.ndarray, k: int = 5) -> dict:
    """Fraction of the top-k retrieved wrong candidates that share the query's site,
    against the fraction of all wrong candidates that do (the no-shortcut rate)."""
    sim = query @ gallery.T
    np.fill_diagonal(sim, -np.inf)  # the true match is not a wrong candidate
    same = sites[:, None] == sites[None, :]
    np.fill_diagonal(same, False)
    top = np.argsort(-sim, axis=1)[:, :k]
    observed = np.take_along_axis(same, top, axis=1).mean()
    expected = same.sum(1).mean() / (len(sites) - 1)
    return {"top_k": k, "same_site_top_k": float(observed), "same_site_base": float(expected)}


# --- main ---------------------------------------------------------------------------


def check_ranges(frame: pd.DataFrame, columns: list[str], low: float, high: float) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if np.any((values < low - 1e-9) | (values > high + 1e-9)):
        raise ValueError(f"{columns} outside [{low}, {high}]:\n{frame}")


def cca_rows(variant: str, splits: dict, pca_dims: list[int], n_perm: int, seed: int):
    spectrum, retrieval = [], []
    rng = np.random.default_rng(seed)
    tr = splits["train"]
    test_centered = {
        key: splits["test"][key] - splits["test"][key].mean(0) + tr[key].mean(0) for key in ("x", "y")
    }
    for p in pca_dims:
        cca = LinearCCA(p).fit(tr["x"], tr["y"])
        held = {
            "val": cca.transform(splits["val"]["x"], splits["val"]["y"]),
            "cptac": cca.transform(splits["test"]["x"], splits["test"]["y"]),
            "cptac-centered": cca.transform(test_centered["x"], test_centered["y"]),
        }
        # Correlations do not depend on centering, so cptac-centered matters only for retrieval.
        corr = {name: column_corr(*held[name]) for name in ("val", "cptac")}
        # Null: refit the CCA rotation on shuffled train pairs (the PCA does not depend
        # on the pairing), evaluate on the true val pairs.
        null = []
        wx = LinearCCA._whitener(tr["x"], p)[1]
        wy = LinearCCA._whitener(tr["y"], p)[1]
        for _ in range(n_perm):
            perm = LinearCCA(p)
            perm.mx, perm.my = cca.mx, cca.my
            perm.set_pairing(tr["x"], tr["y"][rng.permutation(len(tr["y"]))], wx, wy)
            null.append(column_corr(*perm.transform(splits["val"]["x"], splits["val"]["y"])))
        null95 = np.quantile(np.asarray(null), 0.95, axis=0)
        for c in range(p):
            spectrum.append(
                {
                    "variant": variant,
                    "pca_dim": p,
                    "component": c + 1,
                    "train_corr": cca.train_corr[c],
                    "val_corr": corr["val"][c],
                    "val_null95": null95[c],
                    "cptac_corr": corr["cptac"][c],
                }
            )
        for k in [k for k in RETRIEVAL_KS if k <= p]:
            for name, (zx, zy) in held.items():
                m = retrieval_metrics(l2(zx[:, :k]), l2(zy[:, :k]))
                retrieval.append(
                    {
                        "variant": variant,
                        "pca_dim": p,
                        "k": k,
                        "split": name,
                        "n": int(m["n_candidates"]),
                        "wsi2rna_mrr": m["q2g_mrr"],
                        "rna2wsi_mrr": m["g2q_mrr"],
                        "mean_mrr": m["mean_mrr"],
                        "mean_r5": m["mean_r5"],
                        "chance_r5": m["chance_r5"],
                    }
                )
    return spectrum, retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="alignment run dirs")
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--report-pca-dim", type=int, default=32, help="spectrum shown in the Markdown")
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--min-site-cases", type=int, default=10)
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache (default: output dir)")
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output_dir)
    variants = {"raw": splits, "site-removed": remove_site_means(splits)}

    spectrum, retrieval = [], []
    for variant, parts in variants.items():
        s, r = cca_rows(variant, parts, args.pca_dims, args.permutations, seed=0)
        spectrum += s
        retrieval += r
    spectrum = pd.DataFrame(spectrum)
    retrieval = pd.DataFrame(retrieval)
    check_ranges(spectrum, ["train_corr", "val_corr", "val_null95", "cptac_corr"], -1, 1)
    check_ranges(retrieval, ["wsi2rna_mrr", "rna2wsi_mrr", "mean_mrr", "mean_r5"], 0, 1)

    # Site probes on TCGA train + val.
    sites = np.concatenate([splits["train"]["site"], splits["val"]["site"]])
    reps: dict[str, np.ndarray] = {}
    for variant, parts in variants.items():
        reps[f"{variant} slide mean"] = np.concatenate([parts["train"]["x"], parts["val"]["x"]])
        reps[f"{variant} RNA embedding"] = np.concatenate([parts["train"]["y"], parts["val"]["y"]])
        cca = LinearCCA(args.report_pca_dim).fit(parts["train"]["x"], parts["train"]["y"])
        zx, zy = zip(*(cca.transform(parts[s]["x"], parts[s]["y"]) for s in ("train", "val")), strict=True)
        reps[f"{variant} CCA{args.report_pca_dim} slide"] = np.concatenate(zx)
        reps[f"{variant} CCA{args.report_pca_dim} RNA"] = np.concatenate(zy)
    enrichment = []
    cca = LinearCCA(args.report_pca_dim).fit(splits["train"]["x"], splits["train"]["y"])
    zx, zy = cca.transform(splits["val"]["x"], splits["val"]["y"])
    for k in (8, args.report_pca_dim):
        enrichment.append(
            {"representation": f"raw CCA{args.report_pca_dim} top-{k}", "direction": "wsi2rna",
             **site_enrichment(l2(zx[:, :k]), l2(zy[:, :k]), splits["val"]["site"])}
        )

    if args.runs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tiles = {}
        for split in SPLITS:
            ids = splits[split]["pairs"]["slide_id"].astype(str).tolist()
            tiles.update(cache_slide_tiles(splits[split]["embedding_dir"], ids, workers=args.cache_workers))
        for run_dir in args.runs:
            emb = aligned_embeddings(run_dir, splits, tiles, device)
            name = run_dir.name
            reps[f"{name} slide"] = np.concatenate([emb["train"][0], emb["val"][0]])
            reps[f"{name} RNA"] = np.concatenate([emb["train"][1], emb["val"][1]])
            m = retrieval_metrics(*emb["val"])
            print(f"{name}: recomputed TCGA-val mean MRR {m['mean_mrr']:.3f}")
            for direction, (q, g) in {"wsi2rna": emb["val"], "rna2wsi": emb["val"][::-1]}.items():
                enrichment.append(
                    {"representation": f"{name} (val MRR {m['mean_mrr']:.3f})", "direction": direction,
                     **site_enrichment(q, g, splits["val"]["site"])}
                )

    probes = pd.DataFrame(
        [{"representation": name, **site_probe(x, sites, args.min_site_cases)} for name, x in reps.items()]
    )
    enrichment = pd.DataFrame(enrichment)
    check_ranges(probes, ["balanced_accuracy", "chance"], 0, 1)
    check_ranges(enrichment, ["same_site_top_k", "same_site_base"], 0, 1)

    spectrum.to_csv(args.output_dir / "cca_spectrum.csv", index=False)
    retrieval.to_csv(args.output_dir / "cca_retrieval.csv", index=False)
    probes.to_csv(args.output_dir / "site_probe.csv", index=False)
    enrichment.to_csv(args.output_dir / "site_enrichment.csv", index=False)

    n = {split: len(splits[split]["pairs"]) for split in SPLITS}
    shown = spectrum[(spectrum["pca_dim"] == args.report_pca_dim) & (spectrum["component"] <= 12)]
    shown = shown.pivot(index="component", columns="variant",
                        values=["val_corr", "val_null95", "cptac_corr"])
    shown.columns = [f"{variant} {metric}" for metric, variant in shown.columns]
    shown = shown.reset_index()
    best = retrieval.loc[retrieval.groupby(["variant", "split"])["mean_mrr"].idxmax()]
    by_val = retrieval[retrieval["split"] == "val"].groupby("variant")["mean_mrr"].idxmax()
    chosen = retrieval.set_index(["variant", "pca_dim", "k"]).loc[
        [tuple(retrieval.loc[i, ["variant", "pca_dim", "k"]]) for i in by_val]
    ].reset_index()
    command = "uv run python -m xaiwsi.evaluation.shared_structure " + shlex.join(sys.argv[1:])
    (args.output_dir / "shared_structure.md").write_text(
        f"Linear CCA (PCA {args.pca_dims} per modality, fit on TCGA train, n = {n['train']}) "
        f"between slide-mean H-optimus-1 and BulkFormer embeddings; TCGA val n = {n['val']}, "
        f"CPTAC n = {n['test']}; null = 95th percentile over {args.permutations} shuffled-train "
        f"refits. Generated by:\n\n```bash\n{command}\n```\n\n"
        f"Held-out canonical correlations, PCA {args.report_pca_dim}:\n\n{markdown_table(shown)}\n\n"
        f"Retrieval in CCA coordinates, (PCA dim, k) chosen on TCGA val:\n\n{markdown_table(chosen)}\n\n"
        f"Best (PCA dim, k) per split (optimistic for CPTAC, for reference only):\n\n{markdown_table(best)}\n\n"
        f"Site probe (5-fold CV logistic regression, TCGA train + val, sites with >= "
        f"{args.min_site_cases} cases):\n\n{markdown_table(probes)}\n\n"
        f"Site enrichment of the top-5 wrong candidates (TCGA val):\n\n{markdown_table(enrichment)}\n"
    )
    print((args.output_dir / "shared_structure.md").read_text())


if __name__ == "__main__":
    main()
