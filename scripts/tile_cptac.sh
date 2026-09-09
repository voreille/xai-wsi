#!/bin/bash
set -euo pipefail

CPTAC_PATH=/mnt/nas6/data/CPTAC/CPTAC-LUAD_v12/LUAD
OUTPUT_PATH=/home/valentin/workspaces/xai-wsi/data/tiles/cptac
CONFIG_PATH=/home/valentin/workspaces/xai-wsi/configs/tiling/tiling.yaml
SLIDES_LIST=/home/valentin/workspaces/xai-wsi/data/tiles/cptac/slide_filenames.json

mkdir -p "$OUTPUT_PATH"

WST_CMD=(wsi-tile)

if [[ "${DEBUGPY:-0}" == "1" ]]; then
	WST_CMD=(
		python -m debugpy
		--listen 127.0.0.1:5678
		--wait-for-client
		"$(command -v wsi-tile)"
	)
fi

"${WST_CMD[@]}" \
	--source "$CPTAC_PATH" \
	--output "$OUTPUT_PATH" \
	--config "$CONFIG_PATH" \
    --include-slides "$SLIDES_LIST" \
	--rglob-str '*.svs' \
	--auto-skip
