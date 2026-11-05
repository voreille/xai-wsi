"""Neural cellular automaton (NCA) on the tile grid of a slide, read out as a tile mean.

Tiles of a slide lie on a regular grid (level-0 coordinates, one tile step apart). Each
tile becomes a cell whose state is ``[v | h]``: ``v`` = the tile embedding projected to a few
channels (fixed during the updates), ``h`` = hidden channels that start at 0. For T steps
with shared weights, every tissue cell perceives itself and the mean of its tissue
neighbours (isotropic: no direction-specific filters, so the model is equivariant to
rotations by 90 degrees and flips of the grid) and adds a small 1x1 MLP update to ``h``.
Cells off tissue are dead: they hold zeros and are never averaged in, so information does
not cross glass. Each tile is then read out by a per-tile MLP on ``[v | h]``, and the slide
prediction is the mean over tiles, as in ``TileMLPMean`` (uniform tile weights 1/N).

With ``steps = 0`` (or ``train_steps = (0, 0)``) no information moves between tiles: the
model is a ``TileMLPMean`` on the projected embeddings, the control for spatial context.
``shuffle_positions`` permutes the tiles over the tissue cells of their slide (fixed per
slide), the control for spatial *layout* at equal capacity.

Projection: ``projection_path`` loads a fixed PCA (``xaiwsi.data.tile_pca``, no learned
parameters); without it, a learned ``Linear + LayerNorm``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from xaiwsi.models.pathway import PathwayPredictor

# Neighbour weights for the isotropic mean (edge neighbours count twice the corners).
_NEIGHBOURS = torch.tensor([[1.0, 2.0, 1.0], [2.0, 0.0, 2.0], [1.0, 2.0, 1.0]])


def grid_cells(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Occupied cells of a slide's grid and the cell of each tile.

    coords: N x 2 integer (x, y) grid positions in tiles (``xaiwsi.data.pathway.tile_grid``).
    Returns (rows, cols) of the C occupied cells, the cell index of each tile (N, values in
    [0, C)) and the grid height and width. Tiles can share a cell (overlapping regions).
    """
    offsets = coords - coords.min(dim=0).values
    width = int(offsets[:, 0].max()) + 1
    height = int(offsets[:, 1].max()) + 1
    flat = offsets[:, 1] * width + offsets[:, 0]
    cells, tile_cell = torch.unique(flat, return_inverse=True)
    return cells // width, cells % width, tile_cell, height, width


class FixedProjection(nn.Module):
    """x -> (x - mean) @ components.T / scale, from an ``xaiwsi.data.tile_pca`` file."""

    def __init__(self, path: str, output_dim: int, whiten: bool = True):
        super().__init__()
        with np.load(path) as pca:
            components = pca["components"][:output_dim]
            mean = pca["mean"]
            variance = pca["explained_variance"][:output_dim]
        if len(components) < output_dim:
            raise ValueError(f"{path} has {len(components)} components, need {output_dim}")
        scale = np.sqrt(variance) if whiten else np.ones_like(variance)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components.T / self.scale


class TileNCAMean(PathwayPredictor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        proj_dim: int = 32,
        hidden_channels: int = 16,
        update_dim: int = 64,
        readout_dim: int = 128,
        steps: int = 16,
        train_steps: tuple[int, int] | None = (8, 24),
        fire_rate: float = 0.5,
        dropout: float = 0.2,
        tile_dropout: float = 0.0,
        projection_path: str | None = None,
        whiten: bool = True,
        shuffle_positions: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not 0.0 < fire_rate <= 1.0:
            raise ValueError("fire_rate must be in (0, 1].")
        if not 0.0 <= tile_dropout < 1.0:
            raise ValueError("tile_dropout must be in [0, 1).")
        if train_steps is not None and not 0 <= train_steps[0] <= train_steps[1]:
            raise ValueError("train_steps must be (min, max) with 0 <= min <= max.")
        self.steps = steps
        self.train_steps = tuple(train_steps) if train_steps is not None else None
        self.fire_rate = fire_rate
        self.tile_dropout = tile_dropout
        self.shuffle_positions = shuffle_positions
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_channels = hidden_channels

        if projection_path is not None:
            self.projection = FixedProjection(projection_path, proj_dim, whiten=whiten)
        else:
            self.projection = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim))

        channels = proj_dim + hidden_channels
        # Perception = [state, neighbour mean - state] -> 2 * channels inputs per cell.
        self.update = nn.Sequential(
            nn.Conv2d(2 * channels, update_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(update_dim, hidden_channels, kernel_size=1),
        )
        # Zero-initialized last layer: at init the automaton does nothing (standard NCA).
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        self.register_buffer("neighbours", _NEIGHBOURS.expand(1, 1, 3, 3).clone())

        self.readout = nn.Sequential(
            nn.Linear(channels, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, output_dim),
        )

    def _n_steps(self) -> int:
        if self.training and self.train_steps is not None:
            low, high = self.train_steps
            return int(torch.randint(low, high + 1, (1,)).item())
        return self.steps

    def _neighbour_mean(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Weighted mean of the tissue neighbours of each cell (0 where there are none)."""
        channels = state.shape[1]
        kernel = self.neighbours.expand(channels, 1, 3, 3)
        total = F.conv2d(state * mask, kernel, padding=1, groups=channels)
        weight = F.conv2d(mask, self.neighbours, padding=1)
        return total / weight.clamp_min(1e-6) * (weight > 0)

    def _step(self, visible: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = torch.cat([visible, hidden], dim=1)
        perception = torch.cat([state, self._neighbour_mean(state, mask) - state], dim=1)
        delta = self.update(perception)
        if self.training and self.fire_rate < 1.0:
            fire = (torch.rand_like(mask) < self.fire_rate).to(delta.dtype)
        else:
            fire = torch.full_like(mask, self.fire_rate)  # expected update at eval
        return (hidden + delta * fire) * mask

    def _slide(self, embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """N x D tiles, N x 2 coords -> N x (proj_dim + hidden) final cell states."""
        rows, cols, tile_cell, height, width = grid_cells(coords)
        if self.shuffle_positions:
            generator = torch.Generator().manual_seed(len(embeddings))
            order = torch.randperm(len(rows), generator=generator).to(rows.device)
            rows, cols = rows[order], cols[order]
        projected = self.projection(embeddings)  # N x d
        n_cells, d = len(rows), projected.shape[1]
        # Mean of the tiles sharing a cell (usually one tile per cell).
        counts = torch.zeros(n_cells, device=projected.device).index_add_(
            0, tile_cell, torch.ones_like(tile_cell, dtype=projected.dtype)
        )
        cell_visible = projected.new_zeros(n_cells, d).index_add_(0, tile_cell, projected) / counts[:, None]
        visible = projected.new_zeros(1, d, height, width)
        visible[0][:, rows, cols] = cell_visible.T
        mask = projected.new_zeros(1, 1, height, width)
        mask[0, 0][rows, cols] = 1.0
        hidden = projected.new_zeros(1, self.hidden_channels, height, width)
        for _ in range(self._n_steps()):
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(self._step, visible, hidden, mask, use_reentrant=False)
            else:
                hidden = self._step(visible, hidden, mask)
        # Each tile reads out its own projection and the hidden state of its cell.
        return torch.cat([projected, hidden[0][:, rows, cols].T[tile_cell]], dim=1)

    def forward(
        self, embeddings: torch.Tensor, coords: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | None]:
        if coords is None:
            raise ValueError("TileNCAMean needs tile coords (B x N x 2).")
        states = []
        for x, c in zip(embeddings, coords, strict=True):
            if self.training and self.tile_dropout > 0.0:
                n_keep = max(1, round(len(x) * (1.0 - self.tile_dropout)))
                keep = torch.randperm(len(x), device=x.device)[:n_keep]
                x, c = x[keep], c[keep]
            states.append(self._slide(x, c))
        if len({len(s) for s in states}) != 1:
            raise ValueError("slides in a batch must have the same tile count (use batch_size 1)")
        tile_scores = self.readout(torch.stack(states))  # B x N x P
        return {"pred": tile_scores.mean(dim=1), "tile_scores": tile_scores}
