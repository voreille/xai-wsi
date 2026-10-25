#!/bin/bash
# Stage 2c: soft-target InfoNCE sweep (TCGA-LUAD train/val, CPTAC-LUAD test).
# SoftTargetInfoNCE (xaiwsi.models.alignment) replaces the one-hot InfoNCE targets with
# (1 - mass) on the true pair plus `mass` spread over the other cases of the batch by
# softmax(cos(rna_i, rna_j) / target_temperature) of the frozen BulkFormer inputs.
# mass 0 is the hard InfoNCE baseline (run once per setting, the temperature is unused).
# With 64-case batches, temperature 0.05 / 0.1 / 0.2 gives about 3 / 6 / 20 effective
# soft positives. Grid: off-diagonal mass x target temperature x embedding dim x seed, on
# one regularization setting (set ENCODER / TILE_DROPOUT / WEIGHT_DECAY from the
# aln-<tag> sweep). Embedding dim 16 tests a small shared space (linear CCA found about
# 3-8 shared axes, docs/progress.md 2c).
#
# Each run has a pinned W&B id alns-<tag>-<enc>-td..-wd..-e<dim>-m<mass>-tt<temp>-s<seed>,
# so its directory is runs/alignment/xai-wsi/<id>/; its best checkpoint is then tested on
# CPTAC, which writes <run-dir>/test_metrics.json. Finished runs are skipped on rerun.
#
# RAM: each run caches TCGA-LUAD slides (~26 GB fp16) plus CPTAC at test time.
#
# Usage (on the host, from the repo root, after the aln-<tag> sweep has finished):
#   GPUS="0 1" bash scripts/alignment_soft_sweep.sh
#   ENCODER=gated TILE_DROPOUT=0.5 WEIGHT_DECAY=0.1 GPUS="0 1" bash scripts/alignment_soft_sweep.sh
# Collect:
#   uv run python -m xaiwsi.evaluation.collect_alignment_runs --run-root runs/alignment/xai-wsi \
#       --prefix alns-<tag> --output-dir outputs/alignment_soft_runs
set -euo pipefail

CONFIG=${CONFIG:-configs/training/alignment_baseline.yaml}
TAG=${TAG:-$(date +%m%d)}
ENCODER=${ENCODER:-mean}
TILE_DROPOUT=${TILE_DROPOUT:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MASSES=${MASSES:-"0.0 0.25 0.5"}
TARGET_TEMPS=${TARGET_TEMPS:-"0.05 0.1"}
EMBED_DIMS=${EMBED_DIMS:-"256 16"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
RUN_ROOT=${RUN_ROOT:-runs/alignment/xai-wsi}
LOG_DIR=${LOG_DIR:-outputs/alignment_soft_runs/logs}

mkdir -p "$LOG_DIR"

encoder_class() {
    case $1 in
        mean) echo xaiwsi.models.alignment.MeanSlideEncoder ;;
        gated) echo xaiwsi.models.alignment.GatedSlideEncoder ;;
        *) echo "Unknown encoder '$1' (mean|gated)" >&2; return 2 ;;
    esac
}
encoder_class "$ENCODER" > /dev/null

run_one() {
    local gpu=$1 embed_dim=$2 mass=$3 temp=$4 seed=$5
    local id="alns-${TAG}-${ENCODER}-td${TILE_DROPOUT}-wd${WEIGHT_DECAY}-e${embed_dim}-m${mass}-tt${temp}-s${seed}"
    id=${id//./p}  # W&B ids: no dots
    local run_dir="$RUN_ROOT/$id"
    if compgen -G "$run_dir/checkpoints/epoch=*.ckpt" > /dev/null; then
        echo "skip $id (checkpoint exists)"
    else
        echo "fit  $id on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.training.alignment_cli fit \
            --config "$CONFIG" \
            --seed_everything "$seed" \
            --model.aligner.init_args.slide_encoder.class_path "$(encoder_class "$ENCODER")" \
            --model.aligner.init_args.slide_encoder.init_args.tile_dropout "$TILE_DROPOUT" \
            --model.aligner.init_args.slide_encoder.init_args.embed_dim "$embed_dim" \
            --model.aligner.init_args.rna_encoder.init_args.embed_dim "$embed_dim" \
            --model.loss.class_path xaiwsi.models.alignment.SoftTargetInfoNCE \
            --model.loss.init_args.off_diagonal_mass "$mass" \
            --model.loss.init_args.target_temperature "$temp" \
            --model.weight_decay "$WEIGHT_DECAY" \
            --trainer.logger.init_args.id "$id" \
            --trainer.logger.init_args.name "$id" \
            --trainer.logger.init_args.group "alignment-soft-$TAG" \
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
for embed_dim in $EMBED_DIMS; do
    for mass in $MASSES; do
        temps=$TARGET_TEMPS
        # mass 0 = hard InfoNCE: the temperature has no effect, run it once.
        if [[ $(awk -v m="$mass" 'BEGIN { print (m == 0) }') == 1 ]]; then
            temps=${TARGET_TEMPS%% *}
        fi
        for temp in $temps; do
            for seed in $SEEDS; do
                jobs+=("$embed_dim $mass $temp $seed")
            done
        done
    done
done
echo "${#jobs[@]} runs: $ENCODER, tile dropout $TILE_DROPOUT, weight decay $WEIGHT_DECAY, GPUs $GPUS"
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
echo "Done. Collect with: uv run python -m xaiwsi.evaluation.collect_alignment_runs --run-root $RUN_ROOT --prefix alns-$TAG --output-dir outputs/alignment_soft_runs"
