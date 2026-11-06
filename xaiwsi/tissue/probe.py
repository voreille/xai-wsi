"""Linear tile-composition probe: frozen tile embedding -> class fractions of the tile.

Trained on the output of ``xaiwsi.tissue.annotated``. The head is one linear layer with a
softmax, fit with soft-label cross-entropy against the per-tile class fractions (so it
predicts a composition, not a single label), on standardized embeddings (train mean/std).
Weight decay is chosen by cross-validation over the ``validation_fold`` values of the
``train`` split (all train tiles for fitting, on-grid tiles for scoring); the final probe
is refit on the whole train split and evaluated once on the ``test`` split, on-grid
tiles only. Classes absent from the training tiles are reported and keep their logit.

Metrics on test:
- ``soft_ce``: mean soft cross-entropy (lower is better), and ``entropy``: the targets'
  own entropy, its lower bound;
- per class, on tile fractions: Pearson r (``tile_r``); r on mixed tiles only, true fraction
  in (``--mixed-low``, ``--mixed-high``) (``mixed_r``: composition, not presence/absence);
  MAE in fraction units and R² (both also penalize offset and scale, which r ignores);
  soft IoU = sum min(pred, true) / sum max(pred, true) over tiles (tile-level analog of pixel
  IoU; it cannot see errors inside a tile, so it is not equal to the segmenters' IoU);
- per class, on per-image mean fractions (image composition, closest to the slide fractions
  used downstream): Pearson r (``image_r``) and MAE (``image_mae``);
- pure tiles (true dominant fraction >= ``--pure``): accuracy and per-class F1 of argmax.

    uv run python -m xaiwsi.tissue.probe \\
        --data-dir data/processed/tissue/ignite/h-optimus-1 \\
        --output-dir outputs/tissue_probe/ignite

Writes ``probe.pt`` (weight, bias, mean, std, classes, settings), ``cv.csv``,
``test_metrics.json``, ``per_class.csv``, ``test_predictions.parquet`` (on-grid test tiles)
and ``metrics.md`` (tables with the generating command).

``--context ring1|ring2|image|shuffled`` concatenates each tile's embedding with the mean of
other tiles of its image (``xaiwsi.tissue.context``), to test whether neighbouring tissue
helps; ``shuffled`` is the control with as many random tiles as ``ring1``. Such probes need
the same context at application time, so ``xaiwsi.tissue.apply`` refuses them.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.tissue.context import context_features

WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1)


def fit_linear(
    x: torch.Tensor, y: torch.Tensor, weight_decay: float, epochs: int, lr: float, batch_size: int, seed: int
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    head = torch.nn.Linear(x.shape[1], y.shape[1]).to(x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * max(1, -(-len(x) // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=x.device).split(batch_size):
            loss = soft_ce(head(x[idx]), y[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return head.eval()


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1)


def predict(head: torch.nn.Linear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(head(x), dim=1).cpu().numpy()


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    total = ((true - true.mean()) ** 2).sum()
    return float(1 - ((true - pred) ** 2).sum() / total) if total > 0 else float("nan")


def evaluate(
    pred: np.ndarray,
    true: np.ndarray,
    images: np.ndarray,
    classes: list[str],
    pure: float,
    mixed: tuple[float, float] = (0.1, 0.9),
) -> tuple[dict, pd.DataFrame]:
    eps = 1e-12
    ce = -(true * np.log(pred + eps)).sum(axis=1)
    entropy = -(true * np.log(true + eps)).sum(axis=1)
    frame = pd.DataFrame(pred, columns=classes).assign(image=images)
    image_pred = frame.groupby("image").mean()
    image_true = pd.DataFrame(true, columns=classes).assign(image=images).groupby("image").mean()
    is_pure = true.max(axis=1) >= pure
    hard_true, hard_pred = true[is_pure].argmax(axis=1), pred[is_pure].argmax(axis=1)

    rows = []
    for k, name in enumerate(classes):
        tp = int(((hard_pred == k) & (hard_true == k)).sum())
        n_true, n_pred = int((hard_true == k).sum()), int((hard_pred == k).sum())
        f1 = 2 * tp / (n_true + n_pred) if n_true + n_pred else float("nan")
        p, t = pred[:, k], true[:, k]
        is_mixed = (t > mixed[0]) & (t < mixed[1])
        union = np.maximum(p, t).sum()
        rows.append(
            {
                "class": name,
                "true_fraction": float(t.mean()),
                "pred_fraction": float(p.mean()),
                "tile_r": safe_pearson(p, t),
                "mixed_tiles": int(is_mixed.sum()),
                "mixed_r": safe_pearson(p[is_mixed], t[is_mixed]),
                "tile_mae": float(np.abs(p - t).mean()),
                "tile_r2": r_squared(p, t),
                "soft_iou": float(np.minimum(p, t).sum() / union) if union > 0 else float("nan"),
                "image_r": safe_pearson(image_pred[name].to_numpy(), image_true[name].to_numpy()),
                "image_mae": float((image_pred[name] - image_true[name]).abs().mean()),
                "pure_tiles": n_true,
                "pure_f1": f1 if n_true else float("nan"),
            }
        )
    per_class = pd.DataFrame(rows)
    present = per_class["pure_tiles"] > 0
    summary = {
        "n_tiles": int(len(true)),
        "n_images": int(image_true.shape[0]),
        "soft_ce": float(ce.mean()),
        "entropy": float(entropy.mean()),
        "median_tile_r": float(per_class["tile_r"].median()),
        "median_mixed_r": float(per_class["mixed_r"].median()),
        "median_tile_r2": float(per_class["tile_r2"].median()),
        "mean_soft_iou": float(per_class.loc[per_class["true_fraction"] > 0, "soft_iou"].mean()),
        "median_image_r": float(per_class["image_r"].median()),
        "median_image_mae": float(per_class["image_mae"].median()),
        "n_pure_tiles": int(is_pure.sum()),
        "pure_accuracy": float((hard_pred == hard_true).mean()) if is_pure.any() else float("nan"),
        "pure_macro_f1": float(per_class.loc[present, "pure_f1"].mean()),
    }
    return summary, per_class


def check_ranges(summary: dict, per_class: pd.DataFrame) -> None:
    bounds = {
        "tile_r": (-1, 1), "mixed_r": (-1, 1), "image_r": (-1, 1), "tile_r2": (-np.inf, 1),
        "soft_iou": (0, 1), "tile_mae": (0, 1), "image_mae": (0, 1),
    }
    for column, (low, high) in bounds.items():
        values = per_class[column].dropna()
        if not values.between(low, high).all():
            raise SystemExit(f"{column} out of [{low}, {high}]:\n{per_class[['class', column]]}")
    for key in ("pure_accuracy", "pure_macro_f1"):
        if not 0 <= summary[key] <= 1:
            raise SystemExit(f"{key} out of [0, 1]: {summary[key]}")
    if summary["soft_ce"] < summary["entropy"] - 1e-6:
        raise SystemExit("soft cross-entropy below the target entropy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=list(WEIGHT_DECAYS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pure", type=float, default=0.8, help="dominant fraction for pure-tile metrics")
    parser.add_argument("--mixed-low", type=float, default=0.1)
    parser.add_argument("--mixed-high", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--context", default="none", choices=["none", "ring1", "ring2", "image", "shuffled"]
    )
    parser.add_argument("--tile-size", type=int, default=224, help="tile size in pixels (context offsets)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    x_all = np.load(args.data_dir / "embeddings.npy").astype(np.float32)
    y_all = np.load(args.data_dir / "fractions.npy")
    tiles = pd.read_parquet(args.data_dir / "tiles.parquet").reset_index(drop=True)
    x_all = context_features(tiles, x_all, args.context, tile_size=args.tile_size, seed=args.seed)
    classes = json.loads((args.data_dir / "classes.json").read_text())
    train = (tiles["split"] == "train").to_numpy()
    test = (tiles["split"] == "test").to_numpy() & tiles["on_grid"].to_numpy()
    if not train.any() or not test.any():
        raise SystemExit("Need train tiles and on-grid test tiles.")
    absent = [c for c, s in zip(classes, y_all[train].sum(axis=0), strict=True) if s == 0]
    if absent:
        print(f"Classes absent from training tiles: {absent}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = x_all[train].mean(axis=0), x_all[train].std(axis=0) + 1e-6
    x = torch.from_numpy((x_all - mean) / std).to(device)
    y = torch.from_numpy(y_all).to(device)
    fit = lambda mask, wd: fit_linear(  # noqa: E731
        x[torch.tensor(mask, device=device)], y[torch.tensor(mask, device=device)],
        wd, args.epochs, args.lr, args.batch_size, args.seed,
    )

    folds = sorted(tiles.loc[train, "validation_fold"].dropna().unique())
    cv_rows = []
    for wd in args.weight_decays:
        for fold in folds:
            in_fold = (tiles["validation_fold"] == fold).to_numpy()
            head = fit(train & ~in_fold, wd)
            held = train & in_fold & tiles["on_grid"].to_numpy()
            pred = torch.from_numpy(predict(head, x[torch.from_numpy(held).to(device)]))
            ce = soft_ce(torch.log(pred + 1e-12), torch.from_numpy(y_all[held])).mean().item()
            cv_rows.append({"weight_decay": wd, "fold": fold, "n": int(held.sum()), "soft_ce": ce})
        print(f"wd {wd:g}: cv soft CE {np.mean([r['soft_ce'] for r in cv_rows[-len(folds):]]):.4f}")
    cv = pd.DataFrame(cv_rows)
    cv_summary = cv.groupby("weight_decay")["soft_ce"].agg(["mean", "std"]).reset_index()
    best_wd = float(cv_summary.loc[cv_summary["mean"].idxmin(), "weight_decay"])

    head = fit(train, best_wd)
    pred = predict(head, x[torch.from_numpy(test).to(device)])
    images = tiles.loc[test, "sample_id"].to_numpy()
    summary, per_class = evaluate(
        pred, y_all[test], images, classes, args.pure, (args.mixed_low, args.mixed_high)
    )
    check_ranges(summary, per_class)
    summary.update({"weight_decay": best_wd, "n_train_tiles": int(train.sum()), "absent_in_train": absent})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": head.weight.detach().cpu(),
            "bias": head.bias.detach().cpu(),
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "classes": classes,
            "data_dir": str(args.data_dir),
            "settings": {k: v for k, v in vars(args).items() if k not in ("data_dir", "output_dir")}
            | {"weight_decay": best_wd},
        },
        args.output_dir / "probe.pt",
    )
    cv.to_csv(args.output_dir / "cv.csv", index=False)
    per_class.to_csv(args.output_dir / "per_class.csv", index=False)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    pd.concat(
        [
            tiles.loc[test].reset_index(drop=True),
            pd.DataFrame(y_all[test], columns=[f"true__{c}" for c in classes]),
            pd.DataFrame(pred, columns=[f"pred__{c}" for c in classes]),
        ],
        axis=1,
    ).to_parquet(args.output_dir / "test_predictions.parquet")

    command = "uv run python -m xaiwsi.tissue.probe " + shlex.join(sys.argv[1:])
    shown = {k: v for k, v in summary.items() if k != "absent_in_train"}
    text = (
        f"Linear tile-composition probe ({args.data_dir}), test split, on-grid tiles. "
        f"Generated by:\n\n```bash\n{command}\n```\n\n"
        f"Weight decay by CV on the train folds (soft CE):\n\n{markdown_table(cv_summary)}\n\n"
        f"{markdown_table(pd.DataFrame([shown]))}\n\nPer class:\n\n{markdown_table(per_class)}\n"
    )
    (args.output_dir / "metrics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
