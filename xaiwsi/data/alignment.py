from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir
from wsitools.storage.interfaces import EmbeddingStore

_worker_store: EmbeddingStore | None = None


def _init_cache_worker(embedding_dir: str) -> None:
    global _worker_store
    _worker_store = build_embedding_store_from_dir(root_dir=embedding_dir)


def _load_tiles(slide_id: str, dtype: str) -> tuple[str, np.ndarray]:
    # features_dtype=None keeps the stored dtype (no fp32 round trip for fp16 stores).
    tiles, _, _ = _worker_store.load(slide_id, features_dtype=None)
    return slide_id, np.asarray(tiles).astype(dtype, copy=False)


def cache_slide_tiles(
    embedding_dir: str | Path,
    slide_ids: list[str],
    *,
    dtype: str = "float16",
    workers: int = 16,
) -> dict[str, torch.Tensor]:
    """Decode whole-slide tile embeddings once into RAM, in parallel processes.

    Legacy h5 stores (LZF, tiny auto-chunks) cost about 1-10 s of CPU per slide read;
    repacked fp16 stores (``wsi-repack``) are much faster to read, but caching once per
    run still makes later epochs GPU-bound.
    """
    slide_ids = list(dict.fromkeys(slide_ids))
    cache: dict[str, torch.Tensor] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_cache_worker,
        initargs=(str(embedding_dir),),
    ) as pool:
        futures = [pool.submit(_load_tiles, slide_id, dtype) for slide_id in slide_ids]
        for future in tqdm(futures, desc=f"caching {Path(embedding_dir).name} tiles"):
            slide_id, tiles = future.result()
            cache[slide_id] = torch.from_numpy(tiles)
    return cache


def load_rna_embeddings(path: str | Path, column_regex: str = r"^emb_") -> pd.DataFrame:
    """Precomputed RNA embeddings (samples x features), e.g. BulkFormer's parquet."""
    embeddings = pd.read_parquet(path).filter(regex=column_regex)
    if embeddings.empty:
        raise ValueError(f"no columns matching {column_regex!r} in {path}")
    return embeddings


class SlideRNADataset(Dataset):
    """Paired whole-slide tile bags and RNA embedding vectors, one item per case."""

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        pairs: pd.DataFrame,
        rna_embeddings: pd.DataFrame,
    ):
        self.embedding_store = embedding_store
        self.pairs = pairs.reset_index(drop=True)
        # copy=True: with pandas copy-on-write, to_numpy() can return a read-only view.
        self.rna = rna_embeddings.loc[self.pairs["sample_id"]].to_numpy(
            np.float32, copy=True
        )
        # Optional slide_id -> tiles cache (see cache_slide_tiles); tiles keep the
        # cache dtype, and the slide encoder casts them on the GPU.
        self.tile_cache: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.pairs.iloc[idx]
        slide_id = str(row["slide_id"])
        if self.tile_cache is not None:
            tiles = self.tile_cache[slide_id]
        else:
            loaded, _, _ = self.embedding_store.load(slide_id)
            tiles = torch.from_numpy(np.asarray(loaded)).float()
        return {
            "case_id": str(row["case_id"]),
            "slide_id": slide_id,
            "sample_id": str(row["sample_id"]),
            "tiles": tiles,
            "rna": torch.from_numpy(self.rna[idx]),
        }


def collate_bags(items: list[dict[str, object]]) -> dict[str, object]:
    """Batch variable-size bags as a list (no padding, no tile sampling)."""
    return {
        "case_id": [item["case_id"] for item in items],
        "slide_id": [item["slide_id"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
        "tiles": [item["tiles"] for item in items],
        "rna": torch.stack([item["rna"] for item in items]),
    }
