"""Embed annotated segmentation images (IGNITE, ANORAK, ...) as tiles, with per-tile class fractions.

Each image of a ``pathseg``-style dataset directory (``metadata.csv`` with ``sample_id``,
``image_relpath``, ``mask_relpath``, ``mpp_x``; ``label_map.json`` name -> id, with
``Ignore`` = 255) is cut into ``tile_size`` tiles at ``stride``, and every tile gets:
- its embedding from the same frozen encoder and preprocessing as ``wsi-embed``
  (``wsitools.models.model_builder``), so it is comparable with the slide stores;
- the fraction of each class among its non-ignored pixels (soft target);
- ``on_grid``: True for the non-overlapping ``tile_size`` grid (evaluation uses those only;
  the overlapping tiles are extra training data).
Tiles with more than ``--max-ignore`` ignored pixels are dropped. Splits and folds are taken
from ``--split-file`` (``sample_id, split, validation_fold``) or from ``metadata.csv``.

    uv run python -m xaiwsi.tissue.annotated \\
        --dataset-root /home/valentin/data/fss/ignite \\
        --output-dir data/processed/tissue/ignite/h-optimus-1

Writes ``embeddings.npy`` (N x D float16), ``fractions.npy`` (N x K float32), ``tiles.parquet``
(one row per tile: sample_id, x, y, on_grid, split, validation_fold, ignored fraction, plus
the metadata columns ``patient_id``/``histological_subtype`` when present) and
``classes.json`` (class names in column order).
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from wsitools.models.model_builder import build_encoder, build_tx

IGNORE = 255
EXTRA_COLUMNS = ("patient_id", "histological_subtype", "source", "scanner")


def load_dataset(root: Path, split_file: Path | None) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(root / "metadata.csv")
    if split_file is not None:
        splits = pd.read_csv(split_file)[["sample_id", "split", "validation_fold"]]
        metadata = metadata.drop(columns=["split", "validation_fold"], errors="ignore").merge(
            splits, on="sample_id", how="left", validate="one_to_one"
        )
    missing = {"sample_id", "image_relpath", "mask_relpath", "split"} - set(metadata.columns)
    if missing:
        raise SystemExit(f"metadata is missing columns {sorted(missing)}")
    if metadata["split"].isna().any():
        raise SystemExit("Some images have no split.")
    label_map = json.loads((root / "label_map.json").read_text())
    ids = sorted(v for v in label_map.values() if v != IGNORE)
    if ids != list(range(len(ids))):
        raise SystemExit(f"Class ids must be 0..K-1 (plus {IGNORE}), got {ids}")
    names = {v: k for k, v in label_map.items()}
    return metadata, [names[i] for i in ids]


def tile_image(
    image: np.ndarray, mask: np.ndarray, n_classes: int, tile_size: int, stride: int, max_ignore: float
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    """Crops, class fractions and records of the kept tiles of one image."""
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape} differ")
    height, width = mask.shape
    crops, fractions, records = [], [], []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            labels = mask[y : y + tile_size, x : x + tile_size].ravel()
            counts = np.bincount(labels, minlength=256)
            ignored = counts[IGNORE] / labels.size
            valid = counts[:n_classes].sum()
            if counts[n_classes:IGNORE].any() or counts[IGNORE + 1 :].any():
                raise ValueError(f"unexpected label ids {np.nonzero(counts[n_classes:])[0] + n_classes}")
            if ignored > max_ignore or valid == 0:
                continue
            crops.append(image[y : y + tile_size, x : x + tile_size])
            fractions.append(counts[:n_classes] / valid)
            records.append(
                {"x": x, "y": y, "on_grid": x % tile_size == 0 and y % tile_size == 0, "ignored": ignored}
            )
    return crops, np.asarray(fractions, dtype=np.float32).reshape(-1, n_classes), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None, help="sample_id, split, validation_fold")
    parser.add_argument("--encoder", default="h-optimus-1")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112, help="< tile-size adds overlapping training tiles")
    parser.add_argument("--max-ignore", type=float, default=0.5)
    parser.add_argument("--mpp", type=float, default=0.5, help="expected image mpp (checked, not resampled)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.tile_size % args.stride:
        raise SystemExit("--stride must divide --tile-size so the grid tiles are included.")

    metadata, classes = load_dataset(args.dataset_root, args.split_file)
    if "mpp_x" in metadata and not np.allclose(metadata["mpp_x"], args.mpp, rtol=0.05):
        raise SystemExit(f"Images are not at {args.mpp} mpp: {sorted(metadata['mpp_x'].unique())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, dim, dtype = build_encoder(args.encoder)
    model.eval().to(device)
    tx = build_tx(mean, std, args.tile_size)
    autocast = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    embeddings, fractions, rows = [], [], []
    for i, image_row in enumerate(metadata.itertuples(index=False)):
        image = np.asarray(Image.open(args.dataset_root / image_row.image_relpath).convert("RGB"))
        mask = np.asarray(Image.open(args.dataset_root / image_row.mask_relpath))
        crops, image_fractions, records = tile_image(
            image, mask, len(classes), args.tile_size, args.stride, args.max_ignore
        )
        for start in range(0, len(crops), args.batch_size):
            batch = torch.stack([tx(Image.fromarray(c)) for c in crops[start : start + args.batch_size]])
            with torch.inference_mode(), autocast:
                feats = model(batch.to(device))
            embeddings.append(feats.float().cpu().numpy().astype(np.float16))
        fractions.append(image_fractions)
        extra = {c: getattr(image_row, c) for c in EXTRA_COLUMNS if c in metadata.columns}
        fold = getattr(image_row, "validation_fold", None)
        rows += [
            {"sample_id": image_row.sample_id, **r, "split": image_row.split, "validation_fold": fold, **extra}
            for r in records
        ]
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(metadata)} images, {len(rows)} tiles")

    embeddings = np.concatenate(embeddings) if embeddings else np.zeros((0, dim), np.float16)
    fractions = np.concatenate(fractions)
    tiles = pd.DataFrame(rows)
    if not (len(embeddings) == len(fractions) == len(tiles)):
        raise RuntimeError("embeddings, fractions and tiles are misaligned")
    if not np.allclose(fractions.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("class fractions do not sum to 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings)
    np.save(args.output_dir / "fractions.npy", fractions)
    tiles.to_parquet(args.output_dir / "tiles.parquet")
    (args.output_dir / "classes.json").write_text(json.dumps(classes, indent=2))
    print(
        f"{len(tiles)} tiles ({int(tiles['on_grid'].sum())} on grid) from {tiles['sample_id'].nunique()} "
        f"images -> {args.output_dir}\n{tiles.groupby('split').size().to_string()}"
    )


if __name__ == "__main__":
    main()
