from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

Bags = torch.Tensor | list[torch.Tensor]


def _drop_tiles(tiles: torch.Tensor, fraction: float, training: bool) -> torch.Tensor:
    """Keep a random subset of an N x D bag (training-time augmentation)."""
    if not training or fraction == 0.0:
        return tiles
    n_keep = max(1, round(tiles.shape[0] * (1.0 - fraction)))
    return tiles[torch.randperm(tiles.shape[0], device=tiles.device)[:n_keep]]


class SlideEncoder(nn.Module, ABC):
    """Tile embeddings -> one slide embedding, projecting tiles before pooling.

    Each tile is mapped into the output space (``tile_embeddings``, ``h_i``) and the
    slide embedding is their weighted sum, ``embedding = sum_i tile_weights[i] * h_i``.
    Input is a list of ``N_i x D`` bags (variable tile counts) or a ``B x N x D``
    tensor. Per-tile outputs are returned as lists with one ``N_i x ...`` tensor per
    slide.
    """

    def __init__(self, tile_dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= tile_dropout < 1.0:
            raise ValueError("tile_dropout must be in [0, 1).")
        self.tile_dropout = tile_dropout

    @abstractmethod
    def encode_bag(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode one ``N x D`` bag; return ``embedding`` (E) and per-tile tensors."""
        raise NotImplementedError

    def forward(self, tiles: Bags) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        bags = list(tiles.unbind(0)) if isinstance(tiles, torch.Tensor) else tiles
        outputs = [
            self.encode_bag(_drop_tiles(bag, self.tile_dropout, self.training))
            for bag in bags
        ]
        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "embedding": torch.stack([out["embedding"] for out in outputs])
        }
        for key in outputs[0]:
            if key != "embedding":
                result[key] = [out[key] for out in outputs]
        return result


def _tile_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class GatedSlideEncoder(SlideEncoder):
    """Project tiles, then pool with sigmoid gates normalized to sum to one.

    Same gating as ``TileMLPGatedMean``: ``tile_weights = gate_i / sum_j gate_j``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.0,
        tile_dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__(tile_dropout)
        self.eps = eps
        self.backbone = _tile_mlp(input_dim, hidden_dim, dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)

    def encode_bag(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(tiles)
        tile_embeddings = self.proj(h)  # N x E
        tile_gates = torch.sigmoid(self.gate_head(h))  # N x 1
        tile_weights = tile_gates / (tile_gates.sum(dim=0, keepdim=True) + self.eps)
        return {
            "embedding": (tile_weights * tile_embeddings).sum(dim=0),
            "tile_embeddings": tile_embeddings,
            "tile_gates": tile_gates,
            "tile_weights": tile_weights,
        }


class MeanSlideEncoder(SlideEncoder):
    """Project tiles, then mean-pool (uniform weights 1/N). Baseline for the gate."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.0,
        tile_dropout: float = 0.0,
    ):
        super().__init__(tile_dropout)
        self.backbone = _tile_mlp(input_dim, hidden_dim, dropout)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def encode_bag(self, tiles: torch.Tensor) -> dict[str, torch.Tensor]:
        tile_embeddings = self.proj(self.backbone(tiles))
        return {
            "embedding": tile_embeddings.mean(dim=0),
            "tile_embeddings": tile_embeddings,
        }


class MLPProjector(nn.Module):
    """Projection head for a precomputed (frozen) embedding, e.g. BulkFormer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embedding": self.mlp(x)}


class ContrastiveAligner(nn.Module):
    """WSI <-> RNA alignment of a slide encoder and an RNA projector (CLIP-style).

    Takes a batch dict with ``tiles`` (bags) and ``rna`` (``B x R``). Returns
    L2-normalized ``slide`` and ``rna`` embeddings, the ``logit_scale`` and the slide
    encoder's per-tile outputs.
    """

    def __init__(
        self,
        slide_encoder: SlideEncoder,
        rna_encoder: MLPProjector,
        init_temperature: float = 0.07,
        learn_temperature: bool = True,
        max_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.slide_encoder = slide_encoder
        self.rna_encoder = rna_encoder
        self.max_logit_scale = max_logit_scale
        self.log_logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / init_temperature)),
            requires_grad=learn_temperature,
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        slide_out = self.slide_encoder(batch["tiles"])
        rna_out = self.rna_encoder(batch["rna"])
        out = {
            key: value for key, value in slide_out.items() if key != "embedding"
        }
        out["slide"] = F.normalize(slide_out["embedding"].float(), dim=-1)
        out["rna"] = F.normalize(rna_out["embedding"].float(), dim=-1)
        out["logit_scale"] = self.log_logit_scale.exp().clamp(max=self.max_logit_scale)
        return out


def symmetric_info_nce(
    slide: torch.Tensor,
    rna: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """CLIP loss: mean of WSI->RNA and RNA->WSI cross-entropy over the batch.

    Row ``i`` of ``slide`` and ``rna`` is the same patient; other rows are negatives.
    """
    logits = logit_scale * slide @ rna.T
    labels = torch.arange(len(slide), device=slide.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
