#!/bin/bash
# NCA slide encoder (xaiwsi.models.nca.TileNCAMean) on Hallmarks, TCGA-LUAD -> CPTAC-LUAD,
# with the controls that say whether spatial context helps:
#   t0       steps 0: no communication between tiles (= TileMLPMean on the PCA features)
#   t16      16 steps at test, 8-24 at training (default overlay)
#   t16shuf  t16 with tiles shuffled over the tissue cells of their slide (layout control)
#   t32      32 steps at test, 16-48 at training (larger receptive field)
#   t16lp    t16 with a learned Linear+LayerNorm projection instead of the fixed PCA
#
# 1. PCA of TCGA-LUAD training-split tiles (skipped if the .npz exists; ~2-3 min, CPU).
# 2. Fit + test each variant and seed through scripts/pathway_cohorts.sh
#    (W&B ids coh-<tag>nca-<variant>-luad-percohort-s<seed>).
# 3. Collect and paired bootstrap on CPTAC-LUAD: each variant vs t0, and t0 vs the plain
#    tile-mean runs (BASELINE_TAG) -> outputs/nca_runs/.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/pathway_nca.sh
#   VARIANTS="t0 t16" SEEDS="0" bash scripts/pathway_nca.sh
set -euo pipefail

TAG=${TAG:-$(date +%m%d)}
VARIANTS=${VARIANTS:-"t0 t16 t16shuf t32 t16lp"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
BASELINE_TAG=${BASELINE_TAG:-1024}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
OUT=${OUT:-outputs/nca_runs}
PCA=${PCA:-data/processed/tile-pca/h-optimus-1-fp16/tcga-luad_pca64.npz}
CONFIG=configs/training/pathway_cohorts.yaml

if [[ ! -f "$PCA" ]]; then
    uv run python -m xaiwsi.data.tile_pca --config "$CONFIG" --train-cohorts tcga-luad \
        --components 64 --output "$PCA"
fi

variant_args() {
    local p=--model.predictor.init_args
    case $1 in
        t0) echo "$p.steps 0 $p.train_steps [0,0]" ;;
        t16) echo "" ;;
        t16shuf) echo "$p.shuffle_positions true" ;;
        t32) echo "$p.steps 32 $p.train_steps [16,48]" ;;
        t16lp) echo "$p.projection_path null" ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

for variant in $VARIANTS; do
    extra=$(variant_args "$variant")
    CONDITIONS=tcga-luad SEEDS="$SEEDS" GPUS="$GPUS" TAG="${TAG}nca-${variant}" RUN_ROOT="$RUN_ROOT" \
        NORMALIZATIONS=per_cohort STAGES=tiles CONFIG="$CONFIG" \
        EXTRA_ARGS="--config configs/training/overrides/nca.yaml --model.predictor.init_args.projection_path $PWD/$PCA $extra" \
        bash scripts/pathway_cohorts.sh
done

mkdir -p "$OUT/bootstrap"
uv run python -m xaiwsi.evaluation.collect_cohort_runs --run-root "$RUN_ROOT" --prefix "coh-${TAG}nca" \
    --output-dir "$OUT"
runs_of() { echo "$RUN_ROOT/coh-$1-luad-percohort-s*/test_predictions.parquet"; }
uv run python -m xaiwsi.evaluation.paired_bootstrap \
    --baseline "tilemean=$(runs_of "$BASELINE_TAG")" --candidates "t0=$(runs_of "${TAG}nca-t0")" \
    --cohort cptac-luad --split test --output "$OUT/bootstrap/t0_vs_tilemean"
candidates=()
for variant in $VARIANTS; do
    [[ $variant == t0 ]] || candidates+=("$variant=$(runs_of "${TAG}nca-${variant}")")
done
if (( ${#candidates[@]} )); then
    uv run python -m xaiwsi.evaluation.paired_bootstrap \
        --baseline "t0=$(runs_of "${TAG}nca-t0")" --candidates "${candidates[@]}" \
        --cohort cptac-luad --split test --output "$OUT/bootstrap/variants_vs_t0"
fi
