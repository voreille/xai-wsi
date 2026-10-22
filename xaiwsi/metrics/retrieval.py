from __future__ import annotations

import numpy as np

RECALL_KS = (1, 5, 10)


def _ranks(similarity: np.ndarray) -> np.ndarray:
    """1-based rank of the matching column (the diagonal) in each row."""
    correct = np.diag(similarity)[:, None]
    return (similarity > correct).sum(axis=1) + 1


def retrieval_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    ks: tuple[int, ...] = RECALL_KS,
) -> dict[str, float]:
    """Cross-modal retrieval for paired, L2-normalized ``N x E`` embeddings.

    Row ``i`` of ``query`` (e.g. slides) matches row ``i`` of ``gallery`` (e.g. RNA).
    Reports both directions, ``q2g`` and ``g2q``: recall@k, median rank and mean
    reciprocal rank (MRR), and averages over the two directions (``mean_*``). Chance
    is recall@k = k / N, median rank about N / 2.
    """
    if query.shape != gallery.shape:
        raise ValueError(f"shape mismatch: {query.shape} vs {gallery.shape}")
    similarity = query @ gallery.T
    n = len(query)
    metrics: dict[str, float] = {"n_candidates": float(n)}
    for direction, ranks in (
        ("q2g", _ranks(similarity)),
        ("g2q", _ranks(similarity.T)),
    ):
        for k in ks:
            metrics[f"{direction}_r{k}"] = float((ranks <= k).mean())
        metrics[f"{direction}_median_rank"] = float(np.median(ranks))
        metrics[f"{direction}_mrr"] = float((1.0 / ranks).mean())
    for name in [f"r{k}" for k in ks] + ["median_rank", "mrr"]:
        metrics[f"mean_{name}"] = 0.5 * (metrics[f"q2g_{name}"] + metrics[f"g2q_{name}"])
    for k in ks:
        metrics[f"chance_r{k}"] = min(1.0, k / n)
    return metrics
