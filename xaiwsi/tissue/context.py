"""Context features for tile probes: each tile's embedding plus a mean of other tiles.

For tiles of annotated images (``xaiwsi.tissue.annotated``: ``sample_id``, ``x``, ``y`` in
pixels, tiles of ``tile_size`` at a stride that divides it), the context of a tile is the
mean embedding of a set of tiles from the same image:

- ``ring<r>``: the tiles at offsets of whole tile steps (``tile_size``) within Chebyshev
  distance r, self excluded: the non-overlapping neighbours around the tile;
- ``image``: all on-grid tiles of the image except those overlapping the tile;
- ``shuffled``: as many tiles as ``ring1`` would use, drawn at random (fixed seed) among the
  image's on-grid tiles that do not overlap the tile: same amount of context, no locality.

Tiles with no context tile get their own embedding as context. Returned features are
``[embedding, context]`` (N x 2D).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def context_features(
    tiles: pd.DataFrame, embeddings: np.ndarray, mode: str, tile_size: int = 224, seed: int = 0
) -> np.ndarray:
    if mode == "none":
        return embeddings
    radius = None
    if mode.startswith("ring"):
        radius = int(mode.removeprefix("ring"))
    elif mode not in ("image", "shuffled"):
        raise ValueError(f"unknown context mode {mode!r}")
    rng = np.random.default_rng(seed)
    context = np.empty_like(embeddings)
    for _, image in tiles.groupby("sample_id", sort=False):
        index = image.index.to_numpy()
        xy = image[["x", "y"]].to_numpy()
        position = {(int(x), int(y)): i for (x, y), i in zip(xy, index, strict=True)}
        grid = image.index[image["on_grid"].to_numpy()].to_numpy()
        grid_xy = tiles.loc[grid, ["x", "y"]].to_numpy()
        for (x, y), i in zip(xy, index, strict=True):
            if radius is not None:
                members = [
                    position[(x + dx * tile_size, y + dy * tile_size)]
                    for dx in range(-radius, radius + 1)
                    for dy in range(-radius, radius + 1)
                    if (dx, dy) != (0, 0) and (x + dx * tile_size, y + dy * tile_size) in position
                ]
            else:
                apart = (np.abs(grid_xy[:, 0] - x) >= tile_size) | (np.abs(grid_xy[:, 1] - y) >= tile_size)
                candidates = grid[apart]
                if mode == "image":
                    members = list(candidates)
                else:
                    n_ring = sum(
                        (x + dx * tile_size, y + dy * tile_size) in position
                        for dx in (-1, 0, 1)
                        for dy in (-1, 0, 1)
                        if (dx, dy) != (0, 0)
                    )
                    n = min(n_ring, len(candidates))
                    members = list(rng.choice(candidates, size=n, replace=False)) if n else []
            context[i] = embeddings[members].mean(axis=0) if members else embeddings[i]
    return np.concatenate([embeddings, context], axis=1)
