#!/bin/bash
# Stage 2c: regularization sweep of WSI <-> RNA contrastive alignment (TCGA-LUAD train/val,
# CPTAC-LUAD test). The earlier runs (kq0d13qf, orpg08id) peaked at epoch 13-15 and then
# overfit (train loss 0.05 vs val 3.2). Grid: slide encoder x tile dropout x weight decay
# x seed. Each run has a pinned W&B id aln-<tag>-<enc>-td<tile_dropout>-wd<weight_decay>-s<seed>,
# so its directory is runs/alignment/xai-wsi/<id>/; its best checkpoint is then tested on
# CPTAC, which writes <run-dir>/test_metrics.json.
#
# RAM: each run caches TCGA-LUAD slides (~26 GB fp16) plus CPTAC at test time.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/alignment_sweep.sh
#   ENCODERS="gated" TILE_DROPOUTS="0.5" WEIGHT_DECAYS="0.1" SEEDS="0" bash scripts/alignment_sweep.sh
# Collect:
#   uv run python -m xaiwsi.evaluation.collect_alignment_runs --run-root runs/alignment/xai-wsi \
#       --prefix aln-<tag> --output-dir outputs/alignment_runs
set -euo pipefail

CONFIG=${CONFIG:-configs/training/alignment_baseline.yaml}
TAG=${TAG:-$(date +%m%d)}
ENCODERS=${ENCODERS:-"mean gated"}
TILE_DROPOUTS=${TILE_DROPOUTS:-"0.0 0.5"}
WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.01 0.1"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
RUN_ROOT=${RUN_ROOT:-runs/alignment/xai-wsi}
LOG_DIR=${LOG_DIR:-outputs/alignment_runs/logs}

mkdir -p "$LOG_DIR"

encoder_class() {
    case $1 in
        mean) echo xaiwsi.models.alignment.MeanSlideEncoder ;;
        gated) echo xaiwsi.models.alignment.GatedSlideEncoder ;;
        *) echo "Unknown encoder '$1' (mean|gated)" >&2; return 2 ;;
    esac
}

run_one() {
    local gpu=$1 encoder=$2 tile_dropout=$3 weight_decay=$4 seed=$5
    local id="aln-${TAG}-${encoder}-td${tile_dropout}-wd${weight_decay}-s${seed}"
    id=${id//./p}  # W&B ids: no dots
    local run_dir="$RUN_ROOT/$id"
    if compgen -G "$run_dir/checkpoints/epoch=*.ckpt" > /dev/null; then
        echo "skip $id (checkpoint exists)"
    else
        echo "fit  $id on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.alignment_cli fit \
            --config "$CONFIG" \
            --seed_everything "$seed" \
            --model.aligner.init_args.slide_encoder.class_path "$(encoder_class "$encoder")" \
            --model.aligner.init_args.slide_encoder.init_args.tile_dropout "$tile_dropout" \
            --model.weight_decay "$weight_decay" \
            --trainer.logger.init_args.id "$id" \
            --trainer.logger.init_args.name "$id" \
            --trainer.logger.init_args.group "alignment-$TAG" \
            > "$LOG_DIR/$id.fit.log" 2>&1 \
            || { echo "FAIL fit  $id, see $LOG_DIR/$id.fit.log"; return 1; }
    fi
    local ckpt
    ckpt=$(compgen -G "$run_dir/checkpoints/epoch=*.ckpt" | head -n 1) || true
    if [[ -z "$ckpt" ]]; then
        echo "FAIL $id: no best checkpoint in $run_dir/checkpoints"
        return 1
    fi
    if [[ -f "$run_dir/test_metrics.json" ]]; then
        echo "skip test $id (test_metrics.json exists)"
        return 0
    fi
    echo "test $id with $(basename "$ckpt")"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.alignment_cli test \
        --config "$run_dir/config.yaml" \
        --ckpt_path "$ckpt" \
        > "$LOG_DIR/$id.test.log" 2>&1 \
        || { echo "FAIL test $id, see $LOG_DIR/$id.test.log"; return 1; }
}

# One sequential queue per GPU; jobs are dealt round-robin. A failed job is reported and
# the queue goes on; the script exits non-zero if any job failed.
read -r -a gpus <<< "$GPUS"
jobs=()
for encoder in $ENCODERS; do
    for tile_dropout in $TILE_DROPOUTS; do
        for weight_decay in $WEIGHT_DECAYS; do
            for seed in $SEEDS; do
                jobs+=("$encoder $tile_dropout $weight_decay $seed")
            done
        done
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
    echo "Some runs failed (FAIL lines above). Fix and rerun: finished runs are skipped."
    exit 1
fi
echo "Done. Collect with: uv run python -m xaiwsi.evaluation.collect_alignment_runs --run-root $RUN_ROOT --prefix aln-$TAG --output-dir outputs/alignment_runs"
