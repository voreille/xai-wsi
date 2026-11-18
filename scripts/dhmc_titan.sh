#!/bin/bash
# TITAN zero-shot predominant pattern on DHMC-LUAD, with tile-level attention and per-pattern maps.
#
# Stages:
#   embed     CONCH v1.5 on the 512 px grid (configs/tiling/tiling_512.yaml) + fp16 repack,
#             into data/processed/dhmc-luad/{wsi-tiles/512px,wsi-embeddings/conch-v1.5{,-fp16}}
#             (scripts/tile_embed_cohort.sh; skips slides already done)
#   titan     TITAN slide embeddings -> data/processed/dhmc-luad/wsi-embeddings/titan/slides.parquet
#   zeroshot  prompts configs/zeroshot/dhmc_luad_patterns.yaml (paper templates, our class names)
#             -> $OUT/zeroshot/{metrics,confusion,predictions,classifier}
#   maps      exact per-tile attention and pattern contributions for all slides, TITAN vs ANORAK
#             agreement per tumour cell, figures for RENDER_PER_CLASS slides per class
#             -> $OUT/maps/{tiles,figures,slides.csv,agreement.md}
#
# Timing: embed is the long part (143 slides; ~1 h on one GPU); the rest takes minutes.
#
# Usage (on the host, from the repo root):
#   GPU=0 bash scripts/dhmc_titan.sh
#   STAGES="zeroshot maps" RENDER_PER_CLASS=4 bash scripts/dhmc_titan.sh
set -euo pipefail

STAGES=${STAGES:-"embed titan zeroshot maps"}
GPU=${GPU:-0}
SLIDES=${SLIDES:-data/external/dhmc-luad/DHMC_LUAD_corr}
TILE_WORKERS=${TILE_WORKERS:-4}
NUM_WORKERS=${NUM_WORKERS:-8}  # 0 in claude-box (64 MB /dev/shm)
RENDER_PER_CLASS=${RENDER_PER_CLASS:-3}
PROMPTS=${PROMPTS:-configs/zeroshot/dhmc_luad_patterns.yaml}
OUT=${OUT:-outputs/titan_dhmc}

COHORT_DIR=data/processed/dhmc-luad
CONCH=$COHORT_DIR/wsi-embeddings/conch-v1.5-fp16
mkdir -p "$OUT"

if [[ " $STAGES " == *" embed "* ]]; then
    MODEL=conch-v1.5 TILES_GRID=512px TILING_CONFIG=configs/tiling/tiling_512.yaml \
        COHORT=dhmc-luad SLIDES="$SLIDES" RGLOB="*.tif" GPU="$GPU" TILE_WORKERS="$TILE_WORKERS" \
        NUM_WORKERS="$NUM_WORKERS" \
        bash scripts/tile_embed_cohort.sh
fi

if [[ " $STAGES " == *" titan "* ]]; then
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m xaiwsi.concepts.titan --cohorts "dhmc-luad=$CONCH"
fi

if [[ " $STAGES " == *" zeroshot "* ]]; then
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m xaiwsi.evaluation.titan_zeroshot \
        --embeddings "$COHORT_DIR/wsi-embeddings/titan/slides.parquet" \
        --labels "$COHORT_DIR/metadata/growth_patterns.csv" --label-column dominant_pattern \
        --slides "$COHORT_DIR/metadata/wsi_slides.csv" \
        --prompts "$PROMPTS" --output-dir "$OUT/zeroshot"
fi

if [[ " $STAGES " == *" maps "* ]]; then
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m xaiwsi.visualization.titan_maps \
        --embedding-dir "$CONCH" --zeroshot-dir "$OUT/zeroshot" \
        --anorak-dir "$COHORT_DIR/tissue/h-optimus-1-fp16/anorak" \
        --anorak-embedding-dir "$COHORT_DIR/wsi-embeddings/h-optimus-1-fp16" \
        --slides-root "$SLIDES" --render-per-class "$RENDER_PER_CLASS" \
        --output-dir "$OUT/maps"
fi
