"""Derive approximate lepidic/solid gene sets from the Tavernari et al. 2021 supplement.

The published lepidic-to-solid (L2S) signature genes are not released. These sets are
derived from the pattern-wise differential expression tables of the supplement (AACR
figshare), so they approximate the growth-pattern axis, not the published signature:

- Table S3, sheet "TCGA - Differential expression" (1,337 genes, FDR < 0.001)
- Table S6, sheet "CHUV - Differential expression" (216 genes, the Fig. 2c heatmap)

`LvsS` is the log2 fold change lepidic minus solid. Sets per source:

- TCGA / CHUV: genes with LvsS >= min_abs_lfc (lepidic) or <= -min_abs_lfc (solid).
- REPLICATED: genes in both tables with the same LvsS sign (no fold-change threshold).

The TCGA tables come from TCGA-LUAD samples, so the TCGA and REPLICATED sets were selected
partly on the same cohort that is used for training and validation here. The CHUV sets are
independent of TCGA.

    uv run python scripts/build_tavernari_approx_genesets.py \\
        --tcga-de data/external/tavernari/supplement/table_s3.xlsx \\
        --chuv-de data/external/tavernari/supplement/table_s6.xlsx \\
        --output data/external/tavernari/derived/tavernari_l2s_approx.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TCGA_SHEET = "TCGA - Differential expression"
CHUV_SHEET = "CHUV - Differential expression"


def load_lvss(path: Path, sheet: str) -> pd.Series:
    table = pd.read_excel(path, sheet_name=sheet)
    if table["genes"].duplicated().any():
        raise ValueError(f"duplicated genes in {path}:{sheet}")
    return table.set_index("genes")["LvsS"]


def split_by_sign(lvss: pd.Series, min_abs_lfc: float) -> dict[str, list[str]]:
    return {
        "LEPIDIC": sorted(lvss.index[lvss >= min_abs_lfc]),
        "SOLID": sorted(lvss.index[lvss <= -min_abs_lfc]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tcga-de", type=Path, required=True, help="supplementary Table S3 (.xlsx)")
    parser.add_argument("--chuv-de", type=Path, required=True, help="supplementary Table S6 (.xlsx)")
    parser.add_argument("--min-abs-lfc", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tcga = load_lvss(args.tcga_de, TCGA_SHEET)
    chuv = load_lvss(args.chuv_de, CHUV_SHEET)

    shared = tcga.index.intersection(chuv.index)
    agree = shared[(tcga[shared] > 0) == (chuv[shared] > 0)]
    sources = {
        "TCGA": split_by_sign(tcga, args.min_abs_lfc),
        "CHUV": split_by_sign(chuv, args.min_abs_lfc),
        "REPLICATED": split_by_sign(tcga[agree], 0.0),
    }
    gene_sets = {
        f"TAVERNARI_APPROX_{pattern}_{source}": genes
        for source, sets in sources.items()
        for pattern, genes in sets.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gene_sets, indent=1) + "\n")
    manifest = {
        "description": "Approximate growth-pattern gene sets, NOT the published L2S signature.",
        "paper": "Tavernari et al., Cancer Discovery 2021, doi:10.1158/2159-8290.CD-20-1274",
        "tcga_de": {"file": str(args.tcga_de), "sheet": TCGA_SHEET, "n_genes": len(tcga)},
        "chuv_de": {"file": str(args.chuv_de), "sheet": CHUV_SHEET, "n_genes": len(chuv)},
        "min_abs_lfc": args.min_abs_lfc,
        "shared_genes": len(shared),
        "shared_sign_agreement": len(agree) / max(len(shared), 1),
        "sizes": {name: len(genes) for name, genes in gene_sets.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for name, genes in gene_sets.items():
        print(f"{name:40s} {len(genes):4d}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
