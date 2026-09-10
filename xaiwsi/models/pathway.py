from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class PathwayPredictor(nn.Module, ABC):
    """Base class for WSI -> pathway predictors."""

    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Predict pathway scores from B x N x D tile embeddings."""
        raise NotImplementedError


class TileLinearMean(PathwayPredictor):
    """Apply a linear predictor per tile, then mean-pool predictions."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor | None]:
        tile_scores = self.linear(embeddings)
        return {"pred": tile_scores.mean(dim=1), "tile_scores": tile_scores}


class TileMLPMean(PathwayPredictor):
    """Apply an MLP independently to each tile, then mean-pool predictions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor | None]:
        tile_scores = self.mlp(embeddings)
        return {"pred": tile_scores.mean(dim=1), "tile_scores": tile_scores}


class MeanMLP(PathwayPredictor):
    """Mean-pool tile embeddings first, then predict pathways with an MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor | None]:
        pooled = embeddings.mean(dim=1)
        return {"pred": self.mlp(pooled), "tile_scores": None}
