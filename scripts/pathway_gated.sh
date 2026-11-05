#!/bin/bash
# Option C: gated-mean tile model (TileMLPGatedMean) on Hallmarks, then where its gate looks
# compared with the IGNITE tissue classes (xaiwsi.evaluation.gate_compartments).
#
# 1. Fit + test through scripts/pathway_cohorts.sh with the gated overlay
#    (configs/training/overrides/gated.yaml); W&B ids coh-<tag>-<condition>-percohort-s<seed>.
# 2. Gate vs IGNITE classes on val + test slides of each condition's runs
#    -> outputs/gate_compartments/coh-<tag>-<condition>.{md,csv}.
# 3. Paired bootstrap vs the plain tile-mean runs of the same condition (BASELINE_TAG).
#
# Needs the applied IGNITE probe (xaiwsi.tissue.apply --name ignite) for all cohorts.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/pathway_gated.sh
#   CONDITIONS="tcga-luad tcga-luad,tcga-lusc" SEEDS="0 1 2" bash scripts/pathway_gated.sh
set -euo pipefail

TAG=${TAG:-$(date +%m%d)gated}
CONDITIONS=${CONDITIONS:-"tcga-luad"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
BASELINE_TAG=${BASELINE_TAG:-1024}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
OUT=${OUT:-outputs/gate_compartments}

CONDITIONS="$CONDITIONS" SEEDS="$SEEDS" GPUS="$GPUS" TAG="$TAG" RUN_ROOT="$RUN_ROOT" \
    NORMALIZATIONS=per_cohort STAGES=tiles \
    EXTRA_ARGS="--config configs/training/overrides/gated.yaml" \
    bash scripts/pathway_cohorts.sh

mkdir -p "$OUT/bootstrap"
first_gpu=${GPUS%% *}
for condition in $CONDITIONS; do
    short=${condition//tcga-/}
    short=${short//,/+}
    runs=()
    for seed in $SEEDS; do
        runs+=("$RUN_ROOT/coh-${TAG}-${short}-percohort-s${seed}")
    done
    CUDA_VISIBLE_DEVICES=$first_gpu uv run python -m xaiwsi.evaluation.gate_compartments \
        --run-dir "${runs[@]}" \
        --output "$OUT/coh-${TAG}-${short}"
    for cohort in $(uv run python -c "
import pandas as pd, sys
print(' '.join(sorted(pd.read_parquet(sys.argv[1])['cohort'].unique())))" "${runs[0]}/test_predictions.parquet"); do
        uv run python -m xaiwsi.evaluation.paired_bootstrap \
            --baseline "$RUN_ROOT/coh-${BASELINE_TAG}-${short}-percohort-s*/test_predictions.parquet" \
            --candidates "$RUN_ROOT/coh-${TAG}-${short}-percohort-s*/test_predictions.parquet" \
            --cohort "$cohort" --split test \
            --output "$OUT/bootstrap/${short}_${cohort}"
    done
done
uv run python -m xaiwsi.evaluation.collect_cohort_runs --run-root "$RUN_ROOT" --prefix "coh-$TAG" \
    --output-dir "$OUT/runs"
