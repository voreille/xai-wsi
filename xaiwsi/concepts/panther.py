"""PANTHER slide representations (Song et al., CVPR 2024, arXiv 2405.11643), reimplemented.

1. Prototypes: K-means (C clusters) on tiles of the training split of ``--train-cohorts``
   (``build_pathway_splits`` of a ``CohortPathwayDataModule`` config, like
   ``xaiwsi.data.tile_pca``), optionally after the fixed tile PCA of ``--pca``.
2. Per slide: a diagonal Gaussian mixture fitted by a few MAP-EM steps that start at the
   prototypes, with a prior of strength ``--tau`` (in tiles) towards the prototype mean and
   the global per-dimension variance, so component c stays tied to prototype c and an empty
   component falls back to its prototype.
3. Features per component: mixture weight ``pi``, mean ``mu``, diagonal variance ``sigma`` and
   the mass-weighted mean ``pimu`` = pi_c * mu_c (sum over c is the slide's mixture mean, so a
   linear head on ``pimu`` contains the slide mean in the projected space).

    uv run python -m xaiwsi.concepts.panther --config configs/training/pathway_cohorts.yaml \\
        --train-cohorts tcga-luad --components 16 \\
        --cohorts tcga-luad tcga-lusc cptac-luad cptac-lusc \\
        --output-name tcga-luad_c16

Writes the prototypes to ``data/processed/panther/<embedding>/<name>/prototypes.npz`` (with a
``.json`` of the settings) and per cohort ``<cohort dir>/panther/<name>/slides_{pi,mu,sigma,pimu}.parquet``
(slide_id x features), read by ``xaiwsi.evaluation.cohort_probe --features table:panther/<name>/slides_pi.parquet+...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.concepts.bags import embedding_dirs
from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


class Projection:
    """Identity, or the fixed PCA of ``xaiwsi.data.tile_pca``."""

    def __init__(self, pca: Path | None, device: torch.device) -> None:
        self.mean = self.components = None
        if pca is not None:
            state = np.load(pca)
            self.mean = torch.as_tensor(state["mean"], device=device)
            self.components = torch.as_tensor(state["components"], device=device)
        self.device = device

    def __call__(self, tiles: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(tiles), device=self.device).float()
        return x if self.components is None else (x - self.mean) @ self.components.T


def kmeans(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    """Lloyd's K-means with k-means++ initialisation, on the GPU."""
    gen = torch.Generator(device=x.device).manual_seed(seed)
    centers = [x[torch.randint(len(x), (1,), generator=gen, device=x.device)]]
    dist = torch.cdist(x, centers[0]).squeeze(1).pow(2)
    for _ in range(k - 1):
        centers.append(x[torch.multinomial(dist / dist.sum(), 1, generator=gen)])
        dist = torch.minimum(dist, torch.cdist(x, centers[-1]).squeeze(1).pow(2))
    centers = torch.cat(centers)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float().unsqueeze(1)
        centers = torch.where(counts > 0, sums / counts.clamp(min=1), centers)
    return centers


def slide_gmm(
    x: torch.Tensor, prototypes: torch.Tensor, variance: torch.Tensor, tau: float, steps: int
) -> dict[str, torch.Tensor]:
    """MAP-EM of a diagonal GMM on one slide's tiles (N x d), starting at the prototypes."""
    c = len(prototypes)
    pi = torch.full((c,), 1.0 / c, device=x.device)
    mu, sigma = prototypes.clone(), variance.expand(c, -1).clone()
    for _ in range(steps):
        # E step: log N(x | mu_c, diag sigma_c) + log pi_c.
        # Expanded square, so no N x C x d tensor is formed.
        inv = 1.0 / sigma
        sq = (x**2) @ inv.T - 2 * x @ (mu * inv).T + (mu**2 * inv).sum(-1)
        log_p = -0.5 * (sq + torch.log(sigma).sum(-1) + x.shape[1] * np.log(2 * np.pi)) + torch.log(pi)
        resp = torch.softmax(log_p, dim=1)  # N x C
        # M step with the prototype prior (tau pseudo-tiles at mu0 with the global variance).
        mass = resp.sum(0)  # C
        mu = (resp.T @ x + tau * prototypes) / (mass + tau).unsqueeze(1)
        second = resp.T @ (x**2)
        sigma = (second - 2 * mu * (resp.T @ x) + mass.unsqueeze(1) * mu**2 + tau * variance) / (mass + tau).unsqueeze(1)
        sigma = sigma.clamp(min=1e-4 * variance.min().item())
        pi = (mass + 1e-3) / (mass.sum() + 1e-3 * c)
    return {"pi": pi, "mu": mu, "sigma": sigma, "pimu": pi.unsqueeze(1) * mu}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--train-cohorts", nargs="+", required=True)
    parser.add_argument("--cohorts", nargs="+", required=True, help="cohorts to encode (<name> or <name>=<dir>)")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--pca", type=Path, default=None, help="tile PCA .npz (xaiwsi.data.tile_pca)")
    parser.add_argument("--tiles-per-slide", type=int, default=512, help="for K-means")
    parser.add_argument("--kmeans-iters", type=int, default=50)
    parser.add_argument("--em-steps", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0, help="prior strength in tiles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project = Projection(args.pca, device)

    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, _ = select_cohorts(train_specs, test_specs, args.train_cohorts)
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    rng = np.random.default_rng(args.seed)
    tiles, slides = [], []
    for part in splits["train"]:
        for slide_id in tqdm(sorted(set(part.pairs["slide_id"].astype(str))), desc=f"{part.cohort} k-means tiles"):
            embeddings = part.store.load(slide_id)[0]
            n = min(args.tiles_per_slide, len(embeddings))
            tiles.append(project(embeddings[np.sort(rng.choice(len(embeddings), n, replace=False))]))
            slides.append(f"{part.cohort}/{slide_id}")
    x = torch.cat(tiles)
    prototypes = kmeans(x, args.components, args.kmeans_iters, args.seed)
    variance = x.var(0)
    share = torch.bincount(torch.cdist(x, prototypes).argmin(1), minlength=args.components).float() / len(x)
    print(f"K-means on {len(x)} tiles ({len(slides)} slides, d={x.shape[1]}); cluster shares {np.round(share.cpu().numpy(), 3)}")

    proto_dir = Path(processed_root) / "panther" / embedding / args.output_name
    proto_dir.mkdir(parents=True, exist_ok=True)
    np.savez(proto_dir / "prototypes.npz", prototypes=prototypes.cpu().numpy(), variance=variance.cpu().numpy())
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (proto_dir / "prototypes.json").write_text(
        json.dumps({"settings": settings, "embedding": embedding, "n_tiles": len(x), "slides": slides}, indent=2)
    )

    d = x.shape[1]
    columns = {
        "pi": [f"pi{c}" for c in range(args.components)],
        "mu": [f"mu{c}_{j}" for c in range(args.components) for j in range(d)],
        "sigma": [f"sigma{c}_{j}" for c in range(args.components) for j in range(d)],
        "pimu": [f"pimu{c}_{j}" for c in range(args.components) for j in range(d)],
    }
    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        out_dir = embedding_dir.parent.parent / "panther" / args.output_name
        out_dir.mkdir(parents=True, exist_ok=True)
        store = build_embedding_store_from_dir(root_dir=embedding_dir)
        rows: dict[str, dict[str, np.ndarray]] = {kind: {} for kind in columns}
        for slide_id in tqdm(sorted(store.slide_ids()), desc=cohort):
            tiles_slide = store.load(slide_id)[0]
            if len(tiles_slide) == 0:
                continue
            with torch.no_grad():
                fit = slide_gmm(project(tiles_slide), prototypes, variance, args.tau, args.em_steps)
            for kind in columns:
                rows[kind][slide_id] = fit[kind].flatten().cpu().numpy()
        for kind, table in rows.items():
            frame = pd.DataFrame.from_dict(table, orient="index", columns=columns[kind]).astype(np.float32)
            frame.index.name = "slide_id"
            frame.to_parquet(out_dir / f"slides_{kind}.parquet")
        (out_dir / "prototypes.json").write_text(json.dumps({"prototypes": str(proto_dir)}, indent=2))
        pi = pd.DataFrame.from_dict(rows["pi"], orient="index")
        print(f"[{cohort}] {len(pi)} slides -> {out_dir}; mean pi {np.round(pi.mean().to_numpy(), 3)}")


if __name__ == "__main__":
    main()
