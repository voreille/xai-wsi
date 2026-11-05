"""Slide-level composition features from the applied tile-composition probes.

``xaiwsi.tissue.apply`` writes, per cohort and probe, ``<cohort dir>/tissue/<probe>/slides.parquet``
with the mean class fraction per slide. A feature spec (``configs/tissue/composition_features.yaml``)
names, per feature, the probe, the classes to sum and optionally the classes to renormalize over.
"""

from __future__ import annotations

from pathlib import Path

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
