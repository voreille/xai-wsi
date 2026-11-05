"""Slide-level composition features and compartment pools from the applied tile-composition probes.

``xaiwsi.tissue.apply`` writes, per cohort and probe, ``<cohort dir>/tissue/<probe>/slides.parquet``
with the mean class fraction per slide. A feature spec (``configs/tissue/composition_features.yaml``)
names, per feature, the probe, the classes to sum and optionally the classes to renormalize over.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def tissue_dir(embedding_dir: Path) -> Path:
    """``<cohort dir>/tissue`` for a store at ``<cohort dir>/wsi-embeddings/<embedding>``."""
    return Path(embedding_dir).parent.parent / "tissue"


def load_feature_spec(path: Path) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text())["features"]


def composition_features(root: Path, spec: dict[str, dict]) -> pd.DataFrame:
    """slide_id x feature shares for one cohort; ``root`` is its ``tissue`` directory."""
    slides = {}
    columns = {}
    for name, feature in spec.items():
        probe = feature["probe"]
        if probe not in slides:
            slides[probe] = pd.read_parquet(root / probe / "slides.parquet").set_index("slide_id")
        frame = slides[probe]
        missing = set(feature["classes"]) | set(feature.get("renormalize_over", []))
        missing -= set(frame.columns)
        if missing:
            raise KeyError(f"{root / probe}: classes {sorted(missing)} not in slides.parquet")
        share = frame[feature["classes"]].sum(axis=1)
        if "renormalize_over" in feature:
            share = share / frame[feature["renormalize_over"]].sum(axis=1)
        columns[name] = share
    features = pd.DataFrame(columns)
    if features.isna().any().any():
        raise ValueError(f"{root}: NaN composition features (slides missing from a probe, or zero mass)")
    return features


def compartment_pools(
    tiles: np.ndarray,
    fractions: np.ndarray,
    classes: list[str],
    spec: dict,
    keep: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Weighted mean tile embeddings of one slide: ``all`` (plain mean), ``no_background``,
    ``tme`` and one per compartment of ``spec`` (``configs/tissue/compartments.yaml``).

    tiles: N x D embeddings; fractions: N x K probe fractions in the order of ``classes``.
    Each pool weights tile i by the summed fractions of its classes and divides by the
    total weight, so a compartment's mean does not depend on its area. ``keep``: optional
    boolean tile mask (e.g. artifact QC); dropped tiles get weight 0 in every pool. Returns the means
    and the total weights (tile units). Pools with no weight are NaN.
    """
    if len(tiles) != len(fractions):
        raise ValueError(f"{len(tiles)} tiles but {len(fractions)} fraction rows.")
    index = {name: k for k, name in enumerate(classes)}
    unknown = {c for group in _groups(spec).values() for c in group} - set(index)
    if unknown:
        raise KeyError(f"classes {sorted(unknown)} not in the probe classes {classes}")
    tiles = np.asarray(tiles, dtype=np.float64)
    fractions = np.asarray(fractions, dtype=np.float64)
    weights = {"all": np.ones(len(tiles))}
    weights["no_background"] = 1.0 - fractions[:, [index[c] for c in spec["background"]]].sum(axis=1)
    for name, group in _groups(spec).items():
        weights[name] = fractions[:, [index[c] for c in group]].sum(axis=1)
    if keep is not None:
        if len(keep) != len(tiles):
            raise ValueError(f"{len(tiles)} tiles but a keep mask of {len(keep)}.")
        weights = {name: w * np.asarray(keep, dtype=np.float64) for name, w in weights.items()}
    means, totals = {}, {}
    for name, w in weights.items():
        totals[name] = float(w.sum())
        means[name] = w @ tiles / totals[name] if totals[name] > 0 else np.full(tiles.shape[1], np.nan)
    return means, totals


def _groups(spec: dict) -> dict[str, list[str]]:
    return {"tme": spec["tme"], **spec["compartments"]}
