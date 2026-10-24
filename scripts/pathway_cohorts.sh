#!/bin/bash
# Does adding training cohorts help WSI -> Hallmarks? LUAD-only vs LUAD+LUSC (+ BRCA).
# A condition is a comma-separated list of training cohorts; "<name>:<n>" caps that
# cohort's training split at n cases (subset fixed by data.train_subset_seed, default 0).
#
# 1. Ridge probe on slide-mean embeddings for every condition and normalization (CPU, a
#    few minutes; slide means are cached in outputs/cohort_probe/).
# 2. Tile-level model (CONFIG, a CohortPathwayDataModule config) for every condition,
#    normalization and seed. Checkpoints are selected on TCGA-LUAD val in all conditions.
#    Each run has a pinned W&B id coh-<tag>-<condition>-<normalization>-s<seed>, so its
#    directory is runs/pathways/xai-wsi/<id>/; it is then tested with its best checkpoint,
#    which writes <run-dir>/test_metrics.json.
#
# Needs the h-optimus-1-fp16 stores of all cohorts in CONDITIONS (tile_embed_cohort.sh).
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/pathway_cohorts.sh
#   STAGES=probes bash scripts/pathway_cohorts.sh
#   NORMALIZATIONS="per_cohort pooled" SEEDS="0 1 2" bash scripts/pathway_cohorts.sh
#   # far cohort, BRCA capped to the size of the LUSC training split:
#   CONDITIONS="tcga-luad,tcga-brca:381 tcga-luad,tcga-lusc,tcga-brca" bash scripts/pathway_cohorts.sh
set -euo pipefail

CONFIG=${CONFIG:-configs/training/pathway_cohorts.yaml}
TAG=${TAG:-$(date +%m%d)}
CONDITIONS=${CONDITIONS:-"tcga-luad tcga-luad,tcga-lusc"}
NORMALIZATIONS=${NORMALIZATIONS:-"per_cohort"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
STAGES=${STAGES:-"probes tiles"}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
LOG_DIR=${LOG_DIR:-outputs/cohort_runs/logs}

mkdir -p "$LOG_DIR"

if [[ " $STAGES " == *" probes "* ]]; then
    # shellcheck disable=SC2086
    uv run python -m xaiwsi.evaluation.cohort_probe \
        --config "$CONFIG" \
        --conditions $CONDITIONS \
        --normalizations $NORMALIZATIONS \
        --output-dir outputs/cohort_probe
fi

run_one() {
    local gpu=$1 condition=$2 normalization=$3 seed=$4
    local short=${condition//tcga-/}
    short=${short//:/}  # W&B ids can't contain ':'
    local id="coh-${TAG}-${short//,/+}-${normalization//_/}-s${seed}"
    local run_dir="$RUN_ROOT/$id"
    if compgen -G "$run_dir/checkpoints/epoch=*.ckpt" > /dev/null; then
        echo "skip $id (checkpoint exists)"
    else
        echo "fit  $id on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.cli fit \
            --config "$CONFIG" \
            --seed_everything "$seed" \
            --data.init_args.use_train_cohorts "[${condition}]" \
            --data.init_args.target_normalization "$normalization" \
            --trainer.logger.init_args.id "$id" \
            --trainer.logger.init_args.name "$id" \
            --trainer.logger.init_args.group "cohorts-$TAG" \
            > "$LOG_DIR/$id.fit.log" 2>&1
    fi
    local ckpt
    ckpt=$(ls "$run_dir"/checkpoints/epoch=*.ckpt | head -n 1)
    echo "test $id with $(basename "$ckpt")"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.cli test \
        --config "$run_dir/config.yaml" \
        --ckpt_path "$ckpt" \
        > "$LOG_DIR/$id.test.log" 2>&1
}

if [[ " $STAGES " == *" tiles "* ]]; then
    # One sequential queue per GPU; jobs are dealt round-robin.
    read -r -a gpus <<< "$GPUS"
    jobs=()
    for condition in $CONDITIONS; do
        for normalization in $NORMALIZATIONS; do
            for seed in $SEEDS; do
                jobs+=("$condition $normalization $seed")
            done
        done
    done
    for g in "${!gpus[@]}"; do
        (
            for j in "${!jobs[@]}"; do
                if (( j % ${#gpus[@]} == g )); then
                    run_one "${gpus[$g]}" ${jobs[$j]}
                fi
            done
        ) &
    done
    wait
    echo "Done. Collect with: uv run python -m xaiwsi.evaluation.collect_cohort_runs --run-root $RUN_ROOT --prefix coh-$TAG"
fi
