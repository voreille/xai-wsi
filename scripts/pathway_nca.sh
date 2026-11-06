#!/bin/bash
# NCA slide encoder (xaiwsi.models.nca.TileNCAMean) on Hallmarks, TCGA-LUAD -> CPTAC-LUAD,
# with the controls that say whether spatial context helps:
#   t0       steps 0: no communication between tiles (= TileMLPMean on the PCA features)
#   t16      16 steps at test, 8-24 at training (default overlay)
#   t16shuf  t16 with tiles shuffled over the tissue cells of their slide (layout control)
#   t32      32 steps at test, 16-48 at training (larger receptive field)
#   t16lp    t16 with a learned Linear+LayerNorm projection instead of the fixed PCA
#   t0lp, t16lpshuf  the t0 / shuffled controls for the learned projection
#   t0lr3, t16lr3, t16shuflr3  PCA variants at lr 1e-3 (at 1e-4 they still learn at epoch 30)
#
# 1. PCA of TCGA-LUAD training-split tiles (skipped if the .npz exists; ~2-3 min, CPU).
# 2. Fit + test each variant and seed through scripts/pathway_cohorts.sh
#    (W&B ids coh-<tag>nca-<variant>-luad-percohort-s<seed>).
# 3. Collect, learning curves from the local W&B files, and paired bootstraps on CPTAC-LUAD
#    for each PAIRS entry "<baseline>:<candidate>" whose runs exist (tilemean = BASELINE_TAG
#    tile-mean runs) -> outputs/nca_runs/.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/pathway_nca.sh
#   VARIANTS="t0 t16" SEEDS="0" bash scripts/pathway_nca.sh
#   VARIANTS="t0lp t16lpshuf t0lr3 t16lr3 t16shuflr3" GPUS="0 1" bash scripts/pathway_nca.sh
#   # only collect, curves and bootstraps of existing runs (no fit/test; runs in claude-box):
#   TAG=1105 STAGES=collect bash scripts/pathway_nca.sh
set -euo pipefail

TAG=${TAG:-$(date +%m%d)}
VARIANTS=${VARIANTS:-"t0 t16 t16shuf t32 t16lp"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
STAGES=${STAGES:-"train collect"}
BASELINE_TAG=${BASELINE_TAG:-1024}
# Bootstrap pairs (all are run, only those with runs for both sides are kept).
PAIRS=${PAIRS:-"tilemean:t0 t0:t16 t0:t16shuf t0:t32 t0:t16lp tilemean:t16lp t0lp:t16lp t16lp:t16lpshuf tilemean:t0lp t0lr3:t16lr3 t16lr3:t16shuflr3 t0:t0lr3 tilemean:t0lr3 tilemean:t16lr3"}
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
        t0lp) echo "$p.projection_path null $p.steps 0 $p.train_steps [0,0]" ;;
        t16lpshuf) echo "$p.projection_path null $p.shuffle_positions true" ;;
        t0lr3) echo "--model.lr 1e-3 $p.steps 0 $p.train_steps [0,0]" ;;
        t16lr3) echo "--model.lr 1e-3" ;;
        t16shuflr3) echo "--model.lr 1e-3 $p.shuffle_positions true" ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

for variant in $VARIANTS; do
    [[ " $STAGES " == *" train "* ]] || break
    extra=$(variant_args "$variant")
    CONDITIONS=tcga-luad SEEDS="$SEEDS" GPUS="$GPUS" TAG="${TAG}nca-${variant}" RUN_ROOT="$RUN_ROOT" \
        NORMALIZATIONS=per_cohort STAGES=tiles CONFIG="$CONFIG" \
        EXTRA_ARGS="--config configs/training/overrides/nca.yaml --model.predictor.init_args.projection_path $PWD/$PCA $extra" \
        bash scripts/pathway_cohorts.sh
done

[[ " $STAGES " == *" collect "* ]] || exit 0
mkdir -p "$OUT/bootstrap"
# Variants are looked up under any tag (a sweep can span several days); a variant found
# under two tags is an error, so the comparison is never silently mixed.
uv run python -m xaiwsi.evaluation.collect_cohort_runs --run-root "$RUN_ROOT" --prefix "coh-*nca" \
    --output-dir "$OUT"
run_prefix() {
    if [[ $1 == tilemean ]]; then echo "coh-$BASELINE_TAG-luad-percohort"; return; fi
    local tags
    tags=$(compgen -G "$RUN_ROOT/coh-*nca-$1-luad-percohort-s*" | sed -E "s|.*/(coh-[^/]*nca)-$1-luad-percohort-s[0-9]+$|\1|" | sort -u)
    if (( $(wc -w <<< "$tags") > 1 )); then
        echo "variant $1 exists under several tags: $tags" >&2; exit 1
    fi
    [[ -n $tags ]] && echo "$tags-$1-luad-percohort"
    return 0
}
runs_glob() {
    local prefix
    prefix=$(run_prefix "$1")
    [[ -n $prefix ]] && echo "$RUN_ROOT/$prefix-s*/test_predictions.parquet"
    return 0
}
# Fail here (not inside a later $(...)) if a variant exists under several tags.
for variant in t0 t16 t16shuf t32 t16lp t0lp t16lpshuf t0lr3 t16lr3 t16shuflr3; do
    prefix=$(run_prefix "$variant")
done
# Curves in two figures (at most 8 groups each), tile mean as the reference in both.
curves() {
    local name=$1; shift
    local groups=("tilemean=$(run_prefix tilemean)") variant
    for variant in "$@"; do
        [[ -n $(runs_glob "$variant") ]] && groups+=("$variant=$(run_prefix "$variant")")
    done
    (( ${#groups[@]} > 1 )) && uv run python -m xaiwsi.evaluation.wandb_curves "${groups[@]}" --output "$OUT/$name"
    return 0
}
curves curves_pca t0 t16 t16shuf t32 t0lr3 t16lr3 t16shuflr3
curves curves_lp t0lp t16lp t16lpshuf
for pair in $PAIRS; do
    base=${pair%%:*} cand=${pair##*:}
    if [[ -n $(runs_glob "$base") && -n $(runs_glob "$cand") ]]; then
        uv run python -m xaiwsi.evaluation.paired_bootstrap \
            --baseline "$base=$(runs_glob "$base")" --candidates "$cand=$(runs_glob "$cand")" \
            --cohort cptac-luad --split test --output "$OUT/bootstrap/${cand}_vs_${base}"
    fi
done
