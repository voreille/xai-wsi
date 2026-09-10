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


class TileMLPGatedMean(PathwayPredictor):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        tile_dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if not 0.0 <= tile_dropout < 1.0:
            raise ValueError("tile_dropout must be in [0, 1).")

        self.tile_dropout = tile_dropout
        self.eps = eps

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Linear(hidden_dim, output_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def _drop_tiles(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.tile_dropout == 0.0:
            return embeddings

        n_tiles = embeddings.shape[1]

        n_keep = max(
            1,
            round(n_tiles * (1.0 - self.tile_dropout)),
        )

        keep = torch.randperm(
            n_tiles,
            device=embeddings.device,
        )[:n_keep]

        return embeddings[:, keep]

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # B x N x D
        embeddings = self._drop_tiles(embeddings)

        h = self.backbone(embeddings)

        # B x N x P
        tile_scores = self.score_head(h)

        # B x N x 1
        tile_gates = torch.sigmoid(self.gate_head(h))

        # Fraction of total relevance.
        tile_weights = tile_gates / (tile_gates.sum(dim=1, keepdim=True) + self.eps)

        # B x P
        pred = (tile_weights * tile_scores).sum(dim=1)

        return {
            "pred": pred,
            "tile_scores": tile_scores,
            "tile_gates": tile_gates,
            "tile_weights": tile_weights,
        }


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
