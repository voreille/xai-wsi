#!/bin/bash
# Tile, embed (H-optimus-1) and repack (fp16) the diagnostic slides of one cohort into
# data/processed/<cohort>/{wsi-tiles,wsi-embeddings/h-optimus-1,wsi-embeddings/h-optimus-1-fp16}.
# Every step skips slides that are already done, so the script can be rerun after a crash.
#
# Usage (on the host, from the repo root):
#   COHORT=tcga-lusc SLIDES=/mnt/nas7/data/TCGA_Lung_svs/LUSC GPU=0 bash scripts/tile_embed_cohort.sh
#   # CONCH v1.5 on its own 512 px grid (for TITAN), into wsi-tiles-512 and wsi-embeddings/conch-v1.5{,-fp16}:
#   MODEL=conch-v1.5 TILES_NAME=wsi-tiles-512 TILING_CONFIG=configs/tiling/tiling_512.yaml COHORT=... SLIDES=... \
#       bash scripts/tile_embed_cohort.sh
#   STAGES="embed repack" COHORT=... SLIDES=... bash scripts/tile_embed_cohort.sh
#   # tile 8 slides in parallel (memory grows with it: one pyramid level per slide):
#   TILE_WORKERS=8 COHORT=... SLIDES=... bash scripts/tile_embed_cohort.sh
#   # CPTAC: no DX in the file names, and the folders hold normal/frozen slides too, so pass
#   # the FFPE tumour list from scripts/filter_cptac_slide_id.py:
#   COHORT=cptac-lusc SLIDES=/mnt/nas6/data/CPTAC/CPTAC-LSCC_v10/LSCC RGLOB="*.svs" \
#       INCLUDE_SLIDES=data/processed/cptac-lusc/metadata/slide_filenames.json GPU=1 \
#       bash scripts/tile_embed_cohort.sh
set -euo pipefail

: "${COHORT:?set COHORT, e.g. tcga-lusc}"
: "${SLIDES:?set SLIDES, the raw WSI root}"
RGLOB=${RGLOB:-"*DX*.svs"}
INCLUDE_SLIDES=${INCLUDE_SLIDES:-}
GPU=${GPU:-0}
TILE_WORKERS=${TILE_WORKERS:-1}
STAGES=${STAGES:-"tile embed repack"}
REPO=${REPO:-/home/valentin/workspaces/xai-wsi}
TILING_CONFIG=${TILING_CONFIG:-$REPO/configs/tiling/tiling.yaml}
MODEL=${MODEL:-h-optimus-1}
TILES_NAME=${TILES_NAME:-wsi-tiles}
BATCH_SIZE=${BATCH_SIZE:-256}

OUT=$REPO/data/processed/$COHORT
TILES=$OUT/$TILES_NAME
EMB=$OUT/wsi-embeddings
mkdir -p "$TILES" "$EMB"

if [[ " $STAGES " == *" tile "* ]]; then
    include=()
    if [[ -n "$INCLUDE_SLIDES" ]]; then
        include=(--include-slides "$INCLUDE_SLIDES")
    fi
    uv run wsi-tile \
        --source "$SLIDES" \
        --output "$TILES" \
        --config "$TILING_CONFIG" \
        --rglob-str "$RGLOB" \
        "${include[@]}" \
        --workers "$TILE_WORKERS" \
        --auto-skip
fi

if [[ " $STAGES " == *" embed "* ]]; then
    CUDA_VISIBLE_DEVICES=$GPU uv run wsi-embed \
        --tiles-rootdir "$TILES" \
        --slides-rootdir "$SLIDES" \
        --model-name "$MODEL" \
        --output-dir "$EMB/$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --num-workers 8
fi

if [[ " $STAGES " == *" repack "* ]]; then
    uv run python -m wsitools.storage.repack \
        --src "$EMB/$MODEL" \
        --dst "$EMB/$MODEL-fp16" \
        --dtype float16 --chunk-rows 1024 --workers 16
fi
