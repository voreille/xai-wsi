from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

GeneSets = dict[str, set[str]]


def load_gmt(path: str | Path) -> GeneSets:
    """Load a GMT file as {gene_set_name: {gene symbols}}."""
    path = Path(path)
    gene_sets: GeneSets = {}

    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Invalid GMT row {lineno} in {path}: expected >= 3 fields"
                )

            name = fields[0]
            genes = {gene for gene in fields[2:] if gene}
            gene_sets[name] = genes

    return gene_sets


def filter_gene_sets(
    gene_sets: Mapping[str, set[str]],
    available_genes: set[str],
    *,
    min_size: int = 5,
    max_size: int | None = None,
) -> GeneSets:
    """Intersect gene sets with available genes and filter by resulting size."""
    out: GeneSets = {}

    for name, genes in gene_sets.items():
        overlap = set(genes) & available_genes
        if len(overlap) < min_size:
            continue
        if max_size is not None and len(overlap) > max_size:
            continue
        out[name] = overlap

    return out
