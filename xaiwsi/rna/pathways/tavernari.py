from __future__ import annotations

import json
from pathlib import Path

from .genesets import GeneSets, load_gmt


def load_tavernari_signatures(path: str | Path) -> GeneSets:
    """Load LUAD signatures curated from Tavernari et al.

    Supported formats
    -----------------
    .gmt:
        Standard GMT file.

    .json:
        Mapping from signature name to list of gene symbols, e.g.
        {
          "TAVERNARI_LEPIDIC": ["GENE1", "GENE2"],
          "TAVERNARI_SOLID": ["GENE3", "GENE4"]
        }

    The gene lists are intentionally supplied externally rather than
    hard-coded so they can be traced to the exact paper/supplement version
    used in an experiment.
    """
    path = Path(path)

    if path.suffix.lower() == ".gmt":
        return load_gmt(path)

    if path.suffix.lower() == ".json":
        with path.open() as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Tavernari JSON must contain an object mapping.")

        return {
            str(name): {str(gene) for gene in genes}
            for name, genes in payload.items()
        }

    raise ValueError(
        "Tavernari signatures must be provided as .gmt or .json"
    )
