#!/bin/bash
# Learning curve for WSI -> Hallmarks: is performance still rising with more TCGA training
# cases? Decides whether downloading more TCGA/CPTAC cohorts is worth it.
#
# 1. Fast ridge probe on per-slide mean embeddings, CPU, a few minutes.
# 2. Stage-1 tile-level model (CONFIG) trained on fractions of TCGA-train, several subset
#    seeds per fraction; validation (TCGA-val) and CPTAC test sets stay fixed. Each run has
#    a pinned W&B id lc-<tag>-f<fraction>-s<seed>, so its directory is
#    runs/pathways/xai-wsi/<id>/, and it is tested on CPTAC with its best checkpoint.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/learning_curve.sh
#   STAGES=probes bash scripts/learning_curve.sh        # only step 1
#   STAGES=tiles TAG=gated CONFIG=... bash scripts/learning_curve.sh
set -euo pipefail

CONFIG=${CONFIG:-configs/training/pathway_baseline.yaml}
TAG=${TAG:-$(date +%m%d)}
FRACTIONS=${FRACTIONS:-"0.25 0.5 0.75 1.0"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
STAGES=${STAGES:-"probes tiles"}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
LOG_DIR=${LOG_DIR:-outputs/learning_curve/logs}

mkdir -p "$LOG_DIR"

if [[ " $STAGES " == *" probes "* ]]; then
    uv run python -m xaiwsi.evaluation.learning_curve \
        --config "$CONFIG" \
        --embedding-subdir wsi-embeddings/h-optimus-1-fp16 \
        --output-dir outputs/learning_curve
fi

run_one() {
    local gpu=$1 fraction=$2 seed=$3
    local id="lc-${TAG}-f${fraction/./}-s${seed}"
    local run_dir="$RUN_ROOT/$id"
    if compgen -G "$run_dir/checkpoints/epoch=*.ckpt" > /dev/null; then
        echo "skip $id (checkpoint exists)"
    else
        echo "fit  $id on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.cli fit \
            --config "$CONFIG" \
            --seed_everything "$seed" \
            --data.train_fraction "$fraction" \
            --data.train_subset_seed "$seed" \
            --trainer.logger.init_args.id "$id" \
            --trainer.logger.init_args.name "$id" \
            --trainer.logger.init_args.group "learning-curve-$TAG" \
            > "$LOG_DIR/$id.fit.log" 2>&1 \
            || { echo "FAIL fit  $id, see $LOG_DIR/$id.fit.log"; return 1; }
    fi
    local ckpt
    ckpt=$(compgen -G "$run_dir/checkpoints/epoch=*.ckpt" | head -n 1) || true
    if [[ -z "$ckpt" ]]; then
        echo "FAIL $id: no best checkpoint in $run_dir/checkpoints"
        return 1
    fi
    echo "test $id with $(basename "$ckpt")"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.cli test \
        --config "$run_dir/config.yaml" \
        --ckpt_path "$ckpt" \
        > "$LOG_DIR/$id.test.log" 2>&1 \
        || { echo "FAIL test $id, see $LOG_DIR/$id.test.log"; return 1; }
}

if [[ " $STAGES " == *" tiles "* ]]; then
    # One sequential queue per GPU; jobs are dealt round-robin. A failed job is reported
    # and the queue goes on; the script exits non-zero if any job failed.
    read -r -a gpus <<< "$GPUS"
    jobs=()
    for fraction in $FRACTIONS; do
        for seed in $SEEDS; do
            jobs+=("$fraction $seed")
        done
    done
    pids=()
    for g in "${!gpus[@]}"; do
        (
            failed=0
            for j in "${!jobs[@]}"; do
                if (( j % ${#gpus[@]} == g )); then
                    run_one "${gpus[$g]}" ${jobs[$j]} || failed=1
                fi
            done
            exit $failed
        ) &
        pids+=($!)
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    if (( status )); then
        echo "Some runs failed (FAIL lines above). Fix and rerun: runs with a checkpoint are skipped."
        exit 1
    fi
    echo "Done. Collect with: uv run python -m xaiwsi.evaluation.collect_learning_curve --run-root $RUN_ROOT --prefix lc-$TAG"
fi
