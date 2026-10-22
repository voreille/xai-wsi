from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import spmm

from .model import BulkFormer


class BatchedGCNConv(GCNConv):
    """``GCNConv`` on a ``torch.sparse`` adjacency with ``B x N x D`` node features.

    ``torch_sparse.SparseTensor`` (used upstream) broadcasts ``adj_t @ x`` over the
    batch, but ``torch.sparse.mm`` only takes 2-D inputs, so the batch is folded into
    the feature dimension. Parameters and state_dict keys are those of ``GCNConv``.
    """

    def message_and_aggregate(self, adj_t, x):
        if x.dim() != 3:
            return super().message_and_aggregate(adj_t, x)
        b, n, d = x.shape
        out = spmm(adj_t, x.transpose(0, 1).reshape(n, b * d), reduce=self.aggr)
        return out.reshape(n, b, d).transpose(0, 1)


def load_graph(
    graph_path: str | Path,
    weight_path: str | Path,
    n_genes: int,
) -> torch.Tensor:
    """Load BulkFormer's gene graph as a native torch sparse adjacency (``adj_t``).

    Upstream builds ``SparseTensor(row=g[1], col=g[0], value=w).t()``, which needs the
    compiled ``torch_sparse``. The transpose has rows ``g[0]`` and columns ``g[1]``, and
    PyG's ``GCNConv`` accepts the same matrix as a ``torch.sparse`` tensor.
    """
    edge_index = torch.load(graph_path, map_location="cpu", weights_only=True)
    edge_weight = torch.load(weight_path, map_location="cpu", weights_only=True)
    if edge_index.shape != (2, edge_weight.numel()):
        raise ValueError(
            f"graph {tuple(edge_index.shape)} does not match "
            f"{edge_weight.numel()} edge weights"
        )
    adj_t = torch.sparse_coo_tensor(
        torch.stack([edge_index[0], edge_index[1]]),
        edge_weight.float(),
        size=(n_genes, n_genes),
    )
    return adj_t.coalesce()


def load_state_dict(path: str | Path) -> OrderedDict[str, torch.Tensor]:
    """Load a BulkFormer checkpoint and strip the DataParallel ``module.`` prefix."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


def load_bulkformer(
    checkpoint: str | Path,
    graph: str | Path,
    graph_weight: str | Path,
    *,
    dim: int,
    gene_length: int,
    gb_repeat: int = 1,
    p_repeat: int = 1,
    bin_head: int = 12,
    full_head: int = 8,
    bins: int = 0,
    device: str | torch.device = "cpu",
) -> BulkFormer:
    """Build a frozen BulkFormer in eval mode from a checkpoint and its gene graph.

    The architecture arguments must match the checkpoint (see upstream
    ``model/config.py``); loading is strict.
    """
    adj_t = load_graph(graph, graph_weight, gene_length).to(device)
    model = BulkFormer(
        dim=dim,
        graph=adj_t,
        gene_emb=None,  # unused by the constructor
        gene_length=gene_length,
        bin_head=bin_head,
        full_head=full_head,
        bins=bins,
        gb_repeat=gb_repeat,
        p_repeat=p_repeat,
    )
    for block in model.gb_formers:
        block.g = BatchedGCNConv(dim, dim, cached=True, add_self_loops=False)
    model.load_state_dict(load_state_dict(checkpoint), strict=True)
    model.requires_grad_(False)
    return model.to(device).eval()
