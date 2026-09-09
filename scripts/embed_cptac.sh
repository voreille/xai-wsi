#!/bin/bash
set -euo pipefail

SLIDES_ROOTDIR=/mnt/nas6/data/CPTAC/CPTAC-LUAD_v12/LUAD
TILES_DIR=/home/valentin/workspaces/xai-wsi/data/tiles/cptac
OUTPUT_DIR=/home/valentin/workspaces/xai-wsi/data/embeddings/cptac

mkdir -p "$(dirname "${OUTPUT_DIR}")"

CUDA_VISIBLE_DEVICES=0 wsi-embed --tiles-rootdir ${TILES_DIR} \
    --slides-rootdir ${SLIDES_ROOTDIR} \
    --model-name "h-optimus-1" \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 256 \
    --num-workers 8
