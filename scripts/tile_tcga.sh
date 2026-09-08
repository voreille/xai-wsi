#!/bin/bash
set -euo pipefail

TCGA_PATH=/mnt/nas7/data/TCGA_Lung_svs/LUAD
OUTPUT_PATH=/home/valentin/workspaces/xai-wsi/data/tiles/tcga
CONFIG_PATH=/home/valentin/workspaces/xai-wsi/configs/tiling/tiling.yaml

mkdir -p "$(dirname "${OUTPUT_PATH}")"

wsi-tile --source ${TCGA_PATH} --output ${OUTPUT_PATH} --config ${CONFIG_PATH} --rglob-str "*DX*.svs" --auto-skip


