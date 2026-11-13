"""Where data, fitted models and derived outputs live (layout of 2026-11-13).

``data/processed/<cohort>/`` holds data only:
    metadata/, rna/, rna-embeddings/
    wsi-tiles/<grid>/                      tiling stores (224px, 512px)
    wsi-embeddings/<store>/                tile embedding stores (h-optimus-1[-fp16], conch-v1.5[-fp16])
    wsi-embeddings/<slide encoder>/        slide embeddings, slides.parquet (titan)
    tissue/<store>/<probe>/                per-tile predictions on a store (ignite, anorak, tile-qc)
Reference tile sets of the tissue probes are datasets too: ``data/processed/{ignite,anorak}/``.

Fitted unsupervised models: ``models/<kind>/<store>/<name>/`` (sae, panther, tile-pca).
Slide features computed with them: ``outputs/<kind>-bags/<store>/<name>/<cohort>/`` (concept, panther),
caches: ``outputs/cache/<cohort>/``.

A cohort's store is at ``<cohort dir>/wsi-embeddings/<store>``; the helpers take that directory,
so the cohort and the store come from it.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
OUTPUTS = REPO / "outputs"


def cohort_dir(embedding_dir: Path) -> Path:
    return Path(embedding_dir).parent.parent


def cohort_name(embedding_dir: Path) -> str:
    return cohort_dir(embedding_dir).name


def store_name(embedding_dir: Path) -> str:
    return Path(embedding_dir).name


def tissue_dir(embedding_dir: Path) -> Path:
    """``<cohort dir>/tissue/<store>``: one folder per probe applied to that store."""
    return cohort_dir(embedding_dir) / "tissue" / store_name(embedding_dir)


def slide_embedding_dir(embedding_dir: Path, encoder: str) -> Path:
    """``<cohort dir>/wsi-embeddings/<encoder>`` for a slide encoder fed by this store."""
    return Path(embedding_dir).parent / encoder


def model_dir(kind: str, store: str, name: str = "") -> Path:
    """``models/<kind>/<store>/<name>`` (kind: sae, panther, tile-pca)."""
    return MODELS / kind / store / name if name else MODELS / kind / store


def bag_dir(kind: str, embedding_dir: Path, name: str) -> Path:
    """``outputs/<kind>-bags/<store>/<name>/<cohort>`` (kind: concept, panther)."""
    return OUTPUTS / f"{kind}-bags" / store_name(embedding_dir) / name / cohort_name(embedding_dir)


def cache_dir(embedding_dir: Path) -> Path:
    return OUTPUTS / "cache" / cohort_name(embedding_dir)


# Slide-feature tables named in ``cohort_probe --features``: ``<prefix><name>/<file>``, without .parquet.
TABLE_PREFIXES = {"bags:": "concept", "panther:": "panther"}


def is_table_block(block: str) -> bool:
    return block.startswith("table:") or any(block.startswith(p) for p in TABLE_PREFIXES)


def table_path(block: str, embedding_dir: Path) -> Path:
    """Parquet of a slide-feature block for one cohort.

    - ``bags:<sae name>/<file>``: concept bags, ``outputs/concept-bags/<store>/<sae name>/<cohort>/<file>.parquet``;
    - ``panther:<name>/<file>``: PANTHER tables, likewise under ``outputs/panther-bags``;
    - ``table:<path>``: relative to the cohort dir (e.g. ``wsi-embeddings/titan/slides.parquet``),
      or, if it contains ``{cohort}``, relative to the repo root after substituting the cohort.
    """
    for prefix, kind in TABLE_PREFIXES.items():
        if block.startswith(prefix):
            name, _, file = block.removeprefix(prefix).partition("/")
            return bag_dir(kind, embedding_dir, name) / f"{file}.parquet"
    path = block.removeprefix("table:")
    if "{cohort}" in path:
        return REPO / path.format(cohort=cohort_name(embedding_dir))
    return cohort_dir(embedding_dir) / path
