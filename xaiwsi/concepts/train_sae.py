"""Train a TopK SAE (``xaiwsi.models.sae``) on the tiles of the training split.

Uses the cohorts and split of a ``CohortPathwayDataModule`` config (``build_pathway_splits``),
like ``xaiwsi.data.tile_pca``: the SAE sees only tiles of training-split slides of
``--train-cohorts``. Reconstruction is also reported on up to ``--val-tiles-per-slide`` tiles
of each val-split slide (no targets are used). All tiles are held on the GPU in fp16, so no
DataLoader is involved (5.6M TCGA-LUAD training tiles, about 17 GB; LUAD+LUSC 11.8M, about 36 GB).

    uv run python -m xaiwsi.concepts.train_sae --config configs/training/pathway_cohorts.yaml \\
        --train-cohorts tcga-luad --latents 4096 --k 32 \\
        --output-dir data/processed/sae/h-optimus-1-fp16/tcga-luad_k4096_top32

Writes ``sae.pt`` (config, state dict; ``xaiwsi.models.sae.load_sae``), ``metrics.csv`` (per
epoch: train loss and FVU, val FVU, dead fraction on train and val) and ``train.json``
(settings, slides, final metrics).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.models.sae import TopKSAE
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


def load_tiles(parts, tiles_per_slide: int | None, rng: np.random.Generator) -> tuple[torch.Tensor, list[str]]:
    tiles, slides = [], []
    for part in parts:
        for slide_id in tqdm(sorted(set(part.pairs["slide_id"].astype(str))), desc=part.cohort):
            embeddings = np.asarray(part.store.load(slide_id)[0], dtype=np.float16)
            if tiles_per_slide is not None and len(embeddings) > tiles_per_slide:
                embeddings = embeddings[np.sort(rng.choice(len(embeddings), tiles_per_slide, replace=False))]
            tiles.append(torch.from_numpy(embeddings))
            slides.append(f"{part.cohort}/{slide_id}")
    return torch.cat(tiles), slides


@torch.no_grad()
def evaluate(model: TopKSAE, x: torch.Tensor, batch_size: int) -> dict[str, float]:
    """FVU against the mean of ``x`` and the fraction of latents never active on ``x``."""
    model.eval()
    mean = model.normalize(x[: min(len(x), 200_000)]).mean(0)
    sse = sst = 0.0
    active = torch.zeros(model.n_latents, dtype=torch.bool, device=x.device)
    for start in range(0, len(x), batch_size):
        x_norm = model.normalize(x[start : start + batch_size])
        values, indices = model.topk(model.pre_activations(x_norm))
        x_hat = model.decode_sparse(values, indices)
        sse += (x_norm - x_hat).pow(2).sum().item()
        sst += (x_norm - mean).pow(2).sum().item()
        active[indices[values > 0]] = True
    model.train()
    return {"fvu": sse / sst, "dead_fraction": 1 - active.float().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--train-cohorts", nargs="+", required=True)
    parser.add_argument("--latents", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--aux-k", type=int, default=512)
    parser.add_argument("--aux-coef", type=float, default=1 / 32)
    parser.add_argument("--dead-after", type=int, default=2_000_000, help="tiles without activation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tiles-per-slide", type=int, default=None, help="default: all tiles")
    parser.add_argument("--val-tiles-per-slide", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    x_train, train_slides = load_tiles(splits["train"], args.tiles_per_slide, rng)
    x_val, val_slides = load_tiles(splits["val"], args.val_tiles_per_slide, rng)
    x_train, x_val = x_train.to(device), x_val.to(device)
    print(f"{len(x_train)} train tiles ({len(train_slides)} slides), {len(x_val)} val tiles ({len(val_slides)} slides)")

    model = TopKSAE(
        x_train.shape[1], args.latents, args.k, aux_k=args.aux_k, aux_coef=args.aux_coef, dead_after=args.dead_after
    ).to(device)
    sample = x_train[torch.randperm(len(x_train), device=device)[:200_000]].float()
    mean = sample.mean(0)
    # One global scale so that E||x_norm||^2 = D (Gao et al.); affine, keeps slide means linear.
    scale = math.sqrt((sample - mean).pow(2).sum(1).mean().item() / x_train.shape[1])
    model.set_input_stats(mean, scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=6.25e-10)
    steps_per_epoch = math.ceil(len(x_train) / args.batch_size)
    total = args.epochs * steps_per_epoch
    # Constant learning rate, linear decay to 0 over the last 20% of steps.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (total - step) / (0.2 * total))
    )

    history = []
    for epoch in range(args.epochs):
        start = time.time()
        order = torch.randperm(len(x_train), device=device)
        sums = {"loss": 0.0, "fvu": 0.0, "aux": 0.0}
        for step in range(steps_per_epoch):
            out = model(x_train[order[step * args.batch_size : (step + 1) * args.batch_size]])
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            model.project_decoder_grad()
            optimizer.step()
            scheduler.step()
            model.normalize_decoder()
            for key in sums:
                sums[key] += out[key].item()
        train_eval = evaluate(model, x_train[: len(x_val) * 4], args.batch_size)
        val_eval = evaluate(model, x_val, args.batch_size)
        row = {
            "epoch": epoch + 1,
            "train_loss": sums["loss"] / steps_per_epoch,
            "train_fvu_batch": sums["fvu"] / steps_per_epoch,
            "train_aux": sums["aux"] / steps_per_epoch,
            "train_fvu": train_eval["fvu"],
            "val_fvu": val_eval["fvu"],
            "dead_fraction_counter": out["dead_fraction"].item(),
            "dead_fraction_train_subset": train_eval["dead_fraction"],
            "dead_fraction_val": val_eval["dead_fraction"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        if not 0 <= row["val_fvu"] < 10:
            raise RuntimeError(f"Val FVU out of range: {row}")

    torch.save({"config": model.config(), "state_dict": model.state_dict()}, args.output_dir / "sae.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "train.json").write_text(
        json.dumps(
            {
                "config": str(args.config),
                "train_cohorts": args.train_cohorts,
                "embedding": embedding,
                "settings": {k: v for k, v in vars(args).items() if k not in ("config", "output_dir")},
                "input_scale": scale,
                "n_train_tiles": len(x_train),
                "n_val_tiles": len(x_val),
                "final": history[-1],
                "train_slides": train_slides,
                "val_slides": val_slides,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
