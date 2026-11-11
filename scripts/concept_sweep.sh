#!/bin/bash
# README 2f: SAE concept-bag sweep (dictionary size x sparsity x SAE training cohorts x seed).
# Each job trains a TopK SAE on the training-split tiles of TRAIN_COHORTS
# (xaiwsi.concepts.train_sae), writes the concept bags of the probe cohorts
# (xaiwsi.concepts.bags) and runs the Hallmark ridge probe on them for the TCGA-LUAD
# condition (xaiwsi.evaluation.cohort_probe; slide mean, mean activation, active fraction).
# Choose settings on TCGA-val only; CPTAC is reported, never used to pick.
#
# Names: <cohorts>_k<latents>_top<k>_s<seed> (cohorts joined by '+'), SAE under
# data/processed/sae/<embedding>/<name>/, bags under <cohort dir>/concepts/<name>/, probe under
# outputs/concept_sweep/<name>/. Finished steps are skipped on rerun.
#
# Time per job on an H100: about 3 min tile loading, 1-10 min training (K), 5 min bags, 3 min
# probe. GPU memory: training tiles in fp16 (LUAD ~5 GB, LUAD+LUSC ~10 GB) plus the K-wide
# pre-activations of a batch.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/concept_sweep.sh
#   LATENTS="4096" TOPK="32" SEEDS="0 1 2" GPUS="0" bash scripts/concept_sweep.sh
# Collect:
#   uv run python -m xaiwsi.evaluation.collect_concept_runs --sweep-dir outputs/concept_sweep \
#       --output outputs/concept_sweep/summary
set -euo pipefail

CONFIG=${CONFIG:-configs/training/pathway_cohorts.yaml}
EMBEDDING=${EMBEDDING:-h-optimus-1-fp16}
LATENTS=${LATENTS:-"1024 4096 16384"}
TOPK=${TOPK:-"16 32 64"}
TRAIN_COHORT_SETS=${TRAIN_COHORT_SETS:-"tcga-luad tcga-luad,tcga-lusc"}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-30}
PROBE_COHORTS=${PROBE_COHORTS:-"tcga-luad cptac-luad"}
GPUS=${GPUS:-"0"}
OUT_ROOT=${OUT_ROOT:-outputs/concept_sweep}
LOG_DIR=${LOG_DIR:-$OUT_ROOT/logs}

mkdir -p "$LOG_DIR"

run_one() {
    local gpu=$1 cohorts=$2 latents=$3 topk=$4 seed=$5
    local name="${cohorts//,/+}_k${latents}_top${topk}_s${seed}"
    local sae_dir="data/processed/sae/$EMBEDDING/$name"
    local probe_dir="$OUT_ROOT/$name"
    if [[ -f "$sae_dir/sae.pt" ]]; then
        echo "skip train $name"
    else
        echo "train $name on GPU $gpu"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.concepts.train_sae \
            --config "$CONFIG" --train-cohorts ${cohorts//,/ } \
            --latents "$latents" --k "$topk" --epochs "$EPOCHS" --seed "$seed" \
            --output-dir "$sae_dir" > "$LOG_DIR/$name.train.log" 2>&1 \
            || { echo "FAIL train $name, see $LOG_DIR/$name.train.log"; return 1; }
    fi
    if [[ -f "data/processed/cptac-luad/concepts/$name/slides_mean.parquet" ]]; then
        echo "skip bags $name"
    else
        echo "bags $name"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.concepts.bags \
            --config "$CONFIG" --cohorts $PROBE_COHORTS --sae "$sae_dir" \
            > "$LOG_DIR/$name.bags.log" 2>&1 \
            || { echo "FAIL bags $name, see $LOG_DIR/$name.bags.log"; return 1; }
    fi
    if [[ -f "$probe_dir/probe_metrics.csv" ]]; then
        echo "skip probe $name"
        return 0
    fi
    echo "probe $name"
    uv run python -m xaiwsi.evaluation.cohort_probe \
        --config "$CONFIG" --conditions tcga-luad \
        --features slide-mean \
            "table:concepts/$name/slides_mean.parquet" "table:concepts/$name/slides_frac.parquet" \
        --output-dir "$probe_dir" > "$LOG_DIR/$name.probe.log" 2>&1 \
        || { echo "FAIL probe $name, see $LOG_DIR/$name.probe.log"; return 1; }
}

# One sequential queue per GPU; jobs are dealt round-robin. A failed job is reported and
# the queue goes on; the script exits non-zero if any job failed.
read -r -a gpus <<< "$GPUS"
jobs=()
for cohorts in $TRAIN_COHORT_SETS; do
    for latents in $LATENTS; do
        for topk in $TOPK; do
            for seed in $SEEDS; do
                jobs+=("$cohorts $latents $topk $seed")
            done
        done
    done
done
echo "${#jobs[@]} jobs on GPUs $GPUS"
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
exit $status
