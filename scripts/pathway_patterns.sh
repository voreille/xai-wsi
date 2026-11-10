#!/bin/bash
# Does the tile layout help a slide model read growth patterns? Train on TCGA-LUAD to predict
# the approximate Tavernari lepidic/solid scores (6 targets: LEPIDIC/SOLID x TCGA/CHUV/REPLICATED),
# then test zero-shot on DHMC LUAD (predominant pattern per slide, no RNA) through the predicted
# solid - lepidic axis (xaiwsi.evaluation.slide_pattern_runs). Variants:
#   tilemean   TileMLPMean (no grid)
#   t0lp       TileNCAMean, learned projection, 0 steps (no communication between tiles)
#   t16lp      16 steps at test, 8-24 at training
#   t16lpshuf  t16lp with tiles shuffled over the tissue cells of their slide (layout control)
#   t32lp      32 steps at test, 16-48 at training (larger receptive field; not in the default set)
# The spatial question is t16lp vs t16lpshuf (same capacity and averaging, no layout) and vs t0lp.
#
# 1. Fit + test each variant and seed through scripts/pathway_cohorts.sh (CPTAC-LUAD test on
#    the same 6 targets; W&B ids coh-<tag>tav-<variant>-luad-percohort-s<seed>).
# 2. Collect CPTAC/TCGA-val metrics -> $OUT/, and the DHMC zero-shot evaluation with paired
#    bootstraps -> $OUT/dhmc-luad/ (GPU if available; a few minutes per run).
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/pathway_patterns.sh
#   VARIANTS="t32lp" GPUS="0 1" bash scripts/pathway_patterns.sh
#   # only collect + DHMC evaluation of existing runs (no fit/test; runs in claude-box):
#   STAGES=collect bash scripts/pathway_patterns.sh
set -euo pipefail

TAG=${TAG:-$(date +%m%d)}
VARIANTS=${VARIANTS:-"tilemean t0lp t16lp t16lpshuf"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
STAGES=${STAGES:-"train collect"}
PAIRS=${PAIRS:-"t0lp:t16lp t16lpshuf:t16lp tilemean:t16lp tilemean:t0lp ridge:tilemean ridge:t16lp t0lp:t32lp t16lpshuf:t32lp"}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
OUT=${OUT:-outputs/pattern_runs}
GENE_SETS=${GENE_SETS:-data/external/tavernari/derived/tavernari_l2s_approx.json}
N_TARGETS=6
DHMC=${DHMC:-data/processed/dhmc-luad}
RIDGE=${RIDGE:-outputs/slide_patterns/dhmc-luad/slides.parquet}
CONFIG=configs/training/pathway_cohorts.yaml

variant_args() {
    local p=--model.predictor.init_args
    local nca="--config configs/training/overrides/nca.yaml $p.projection_path null"
    case $1 in
        tilemean) echo "" ;;
        t0lp) echo "$nca $p.steps 0 $p.train_steps [0,0]" ;;
        t16lp) echo "$nca" ;;
        t16lpshuf) echo "$nca $p.shuffle_positions true" ;;
        t32lp) echo "$nca $p.steps 32 $p.train_steps [16,48]" ;;
        *) echo "unknown variant $1" >&2; return 1 ;;
    esac
}

for variant in $VARIANTS; do
    [[ " $STAGES " == *" train "* ]] || break
    extra=$(variant_args "$variant")
    CONDITIONS=tcga-luad SEEDS="$SEEDS" GPUS="$GPUS" TAG="${TAG}tav-${variant}" RUN_ROOT="$RUN_ROOT" \
        NORMALIZATIONS=per_cohort STAGES=tiles CONFIG="$CONFIG" \
        EXTRA_ARGS="$extra --model.predictor.init_args.output_dim $N_TARGETS --data.init_args.gene_sets $PWD/$GENE_SETS" \
        bash scripts/pathway_cohorts.sh
done

[[ " $STAGES " == *" collect "* ]] || exit 0
uv run python -m xaiwsi.evaluation.collect_cohort_runs --run-root "$RUN_ROOT" --prefix "coh-*tav" \
    --output-dir "$OUT"
# Variants are looked up under any tag; a variant found under two tags is an error.
groups=()
for variant in tilemean t0lp t16lp t16lpshuf t32lp; do
    tags=$(compgen -G "$RUN_ROOT/coh-*tav-$variant-luad-percohort-s*" \
        | sed -E "s|.*/(coh-[^/]*tav)-$variant-luad-percohort-s[0-9]+$|\1|" | sort -u) || continue
    if (( $(wc -l <<< "$tags") > 1 )); then
        echo "variant $variant has runs under several tags: $(tr '\n' ' ' <<< "$tags")" >&2
        exit 1
    fi
    groups+=("$variant=$RUN_ROOT/$tags-$variant-luad-percohort-s*")
done
if (( ${#groups[@]} == 0 )); then
    echo "no coh-*tav-* runs in $RUN_ROOT" >&2
    exit 1
fi
reference=()
[[ -f "$RIDGE" ]] && reference=(--reference "ridge=$RIDGE")
# shellcheck disable=SC2086
uv run python -m xaiwsi.evaluation.slide_pattern_runs \
    --embedding-dir "$DHMC/wsi-embeddings/h-optimus-1-fp16" \
    --labels "$DHMC/metadata/growth_patterns.csv" \
    --groups "${groups[@]}" "${reference[@]}" \
    --pairs $PAIRS \
    --output-dir "$OUT/dhmc-luad"
