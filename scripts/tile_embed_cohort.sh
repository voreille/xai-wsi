#!/bin/bash
# Tile, embed (H-optimus-1) and repack (fp16) the diagnostic slides of one cohort into
# data/processed/<cohort>/{wsi-tiles,wsi-embeddings/h-optimus-1,wsi-embeddings/h-optimus-1-fp16}.
# Every step skips slides that are already done, so the script can be rerun after a crash.
#
# Usage (on the host, from the repo root):
#   COHORT=tcga-lusc SLIDES=/mnt/nas7/data/TCGA_Lung_svs/LUSC GPU=0 bash scripts/tile_embed_cohort.sh
#   STAGES="embed repack" COHORT=... SLIDES=... bash scripts/tile_embed_cohort.sh
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
STAGES=${STAGES:-"tile embed repack"}
REPO=${REPO:-/home/valentin/workspaces/xai-wsi}
TILING_CONFIG=${TILING_CONFIG:-$REPO/configs/tiling/tiling.yaml}

OUT=$REPO/data/processed/$COHORT
TILES=$OUT/wsi-tiles
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
        --auto-skip
fi

if [[ " $STAGES " == *" embed "* ]]; then
    CUDA_VISIBLE_DEVICES=$GPU uv run wsi-embed \
        --tiles-rootdir "$TILES" \
        --slides-rootdir "$SLIDES" \
        --model-name h-optimus-1 \
        --output-dir "$EMB/h-optimus-1" \
        --batch-size 256 \
        --num-workers 8
fi

if [[ " $STAGES " == *" repack "* ]]; then
    uv run python -m wsitools.storage.repack \
        --src "$EMB/h-optimus-1" \
        --dst "$EMB/h-optimus-1-fp16" \
        --dtype float16 --chunk-rows 1024 --workers 16
fi
