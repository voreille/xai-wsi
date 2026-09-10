#!/bin/bash
set -euo pipefail

EXPRESSION_PATH=/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/log2_tpm.parquet
GENES_PATH=/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/genes.csv
GENE_SETS_PATH=/home/valentin/workspaces/xai-wsi/data/external/msigdb/h.all.v2026.1.Hs.symbols.gmt
OUTPUT_PATH=/home/valentin/workspaces/xai-wsi/data/processed/tcga-luad/rna/pathways/hallmarks-mean-z.parquet

mkdir -p "$(dirname "$OUTPUT_PATH")"

ARGS=(
    --expression "$EXPRESSION_PATH"
    --genes "$GENES_PATH"
    --gene-sets "$GENE_SETS_PATH"
    --gene-set-type hallmarks
    --method mean-z
    --output "$OUTPUT_PATH"
)

if [[ "${DEBUGPY:-0}" == "1" ]]; then
    python -m debugpy \
        --listen 127.0.0.1:5678 \
        --wait-for-client \
        -m xaiwsi.rna.pathways \
        "${ARGS[@]}"
else
    python -m xaiwsi.rna.pathways "${ARGS[@]}"
fi
