"""Tile masks from the artifact probabilities of ``xaiwsi.qc.tile_classifier score``."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def artifact_prob_path(embedding_dir: Path) -> Path:
    """``<cohort dir>/tile-qc/<embedding>/artifact_prob.npz`` for a store at
    ``<cohort dir>/wsi-embeddings/<embedding>``."""
    embedding_dir = Path(embedding_dir)
    return embedding_dir.parent.parent / "tile-qc" / embedding_dir.name / "artifact_prob.npz"


def load_artifact_probs(embedding_dir: Path) -> dict[str, np.ndarray]:
    path = artifact_prob_path(embedding_dir)
    if not path.exists():
        raise FileNotFoundError(f"No tile artifact probabilities at {path} (run tile_classifier score).")
    with np.load(path) as probs:
        return {key: probs[key] for key in probs.files}


def keep_mask(prob: np.ndarray, threshold: float, min_tiles: int = 16) -> np.ndarray:
    """Tiles with artifact probability below ``threshold``. If fewer than ``min_tiles``
    would remain, all tiles are kept (the slide is then unfiltered, not dropped)."""
    keep = np.asarray(prob, dtype=np.float32) < threshold
    if keep.sum() < min_tiles:
        return np.ones_like(keep)
    return keep
