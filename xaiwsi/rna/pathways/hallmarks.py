from __future__ import annotations

from pathlib import Path

from .genesets import GeneSets, load_gmt


def load_hallmarks(path: str | Path) -> GeneSets:
    """Load MSigDB Hallmark gene sets from a GMT file.

    The caller supplies the GMT file so the project does not embed or
    redistribute MSigDB gene-set content.
    """
    gene_sets = load_gmt(path)

    hallmarks = {
        name: genes
        for name, genes in gene_sets.items()
        if name.startswith("HALLMARK_")
    }

    if not hallmarks:
        raise ValueError(
            f"No HALLMARK_* gene sets found in {Path(path)}"
        )

    return hallmarks
