#!/bin/bash
# README 2f: PANTHER baseline sweep under the same protocol as the SAE concept bags
# (scripts/concept_sweep.sh): prototypes x projection x fitting cohorts x seed.
# Each job fits the K-means prototypes on the training-split tiles of TRAIN_COHORTS and the
# per-slide GMMs of the probe cohorts (xaiwsi.concepts.panther), then runs the Hallmark ridge
# probe for CONDITION (xaiwsi.evaluation.cohort_probe) on slide mean, pi, mu, pi+mu,
# pi+mu+sigma (the paper's representation) and pimu, with and without clipping to the training
# range. Choose settings on TCGA-val only; CPTAC is reported, never used to pick.
#
# PROJECTIONS: "full" (the 1536-d embedding, as in the paper; C=16 gives ~49k dims for
# pi+mu+sigma) and/or "pca<k>" (the fixed tile PCA of xaiwsi.data.tile_pca, fitted on the same
# training cohorts and created here if missing).
#
# Names: <cohorts>_c<C>_<projection>_s<seed> (cohorts joined by '+'), prototypes under
# data/processed/panther/<embedding>/<name>/, slide tables under <cohort dir>/panther/<name>/,
# probe under OUT_ROOT/<name>/. Finished steps are skipped on rerun.
#
# Time per job on an H100 (box check 2026-11-12, full, C=16, TCGA-LUAD + CPTAC-LUAD): 2.5 min
# K-means + slide GMMs, 1 min probe; the full-dim mu/sigma tables are ~75 MB per cohort.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/panther_sweep.sh
# LSCC replication (TCGA-LUSC condition, CPTAC-LSCC test):
#   CONDITION=tcga-lusc PROBE_COHORTS="tcga-lusc cptac-lusc" TRAIN_COHORT_SETS="tcga-lusc" \
#       OUT_ROOT=outputs/panther_sweep_lusc GPUS="0 1" bash scripts/panther_sweep.sh
# Collect:
#   uv run python -m xaiwsi.evaluation.collect_concept_runs --encoder panther \
#       --sweep-dir outputs/panther_sweep --output outputs/panther_sweep/summary
set -euo pipefail

CONFIG=${CONFIG:-configs/training/pathway_cohorts.yaml}
EMBEDDING=${EMBEDDING:-h-optimus-1-fp16}
COMPONENTS=${COMPONENTS:-"8 16 32"}
PROJECTIONS=${PROJECTIONS:-"full pca64"}
TRAIN_COHORT_SETS=${TRAIN_COHORT_SETS:-"tcga-luad"}
SEEDS=${SEEDS:-"0 1 2"}
CONDITION=${CONDITION:-tcga-luad}
PROBE_COHORTS=${PROBE_COHORTS:-"tcga-luad cptac-luad"}
TABLE_TRANSFORMS=${TABLE_TRANSFORMS:-"none clip"}
GPUS=${GPUS:-"0"}
OUT_ROOT=${OUT_ROOT:-outputs/panther_sweep}
LOG_DIR=${LOG_DIR:-$OUT_ROOT/logs}

mkdir -p "$LOG_DIR"

# The tile PCAs are shared by jobs, so fit them up front (sequentially, on the first GPU).
read -r -a gpus <<< "$GPUS"
for cohorts in $TRAIN_COHORT_SETS; do
    for projection in $PROJECTIONS; do
        [[ $projection == full ]] && continue
        pca="data/processed/tile-pca/$EMBEDDING/${cohorts//,/+}_${projection}.npz"
        [[ -f $pca ]] && continue
        echo "tile PCA $pca"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${gpus[0]} uv run python -m xaiwsi.data.tile_pca \
            --config "$CONFIG" --train-cohorts ${cohorts//,/ } --components "${projection#pca}" \
            --output "$pca" > "$LOG_DIR/${cohorts//,/+}_${projection}.pca.log" 2>&1
    done
done

run_one() {
    local gpu=$1 cohorts=$2 components=$3 projection=$4 seed=$5
    local name="${cohorts//,/+}_c${components}_${projection}_s${seed}"
    local probe_dir="$OUT_ROOT/$name"
    local pca_args=()
    [[ $projection == full ]] || pca_args=(--pca "data/processed/tile-pca/$EMBEDDING/${cohorts//,/+}_${projection}.npz")
    local missing=()
    for cohort in $PROBE_COHORTS; do
        [[ -f "data/processed/$cohort/panther/$name/slides_pimu.parquet" ]] || missing+=("$cohort")
    done
    if (( ${#missing[@]} == 0 )); then
        echo "skip encode $name"
    else
        echo "encode $name on GPU $gpu (missing: ${missing[*]})"
        # Refitting the prototypes is not bitwise reproducible on the GPU, so all probe cohorts
        # are re-encoded together.
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu uv run python -m xaiwsi.concepts.panther \
            --config "$CONFIG" --train-cohorts ${cohorts//,/ } --components "$components" \
            "${pca_args[@]}" --seed "$seed" --cohorts $PROBE_COHORTS --output-name "$name" \
            > "$LOG_DIR/$name.encode.log" 2>&1 \
            || { echo "FAIL encode $name, see $LOG_DIR/$name.encode.log"; return 1; }
    fi
    if [[ -f "$probe_dir/probe_metrics.csv" ]]; then
        echo "skip probe $name"
        return 0
    fi
    echo "probe $name"
    local t="table:panther/$name/slides"
    # shellcheck disable=SC2086
    uv run python -m xaiwsi.evaluation.cohort_probe \
        --config "$CONFIG" --conditions "$CONDITION" --table-transforms $TABLE_TRANSFORMS \
        --features slide-mean "$t"_pi.parquet "$t"_mu.parquet "$t"_pi.parquet+"$t"_mu.parquet \
            "$t"_pi.parquet+"$t"_mu.parquet+"$t"_sigma.parquet "$t"_pimu.parquet \
        --output-dir "$probe_dir" > "$LOG_DIR/$name.probe.log" 2>&1 \
        || { echo "FAIL probe $name, see $LOG_DIR/$name.probe.log"; return 1; }
}

# One sequential queue per GPU; jobs are dealt round-robin. A failed job is reported and
# the queue goes on; the script exits non-zero if any job failed.
jobs=()
for cohorts in $TRAIN_COHORT_SETS; do
    for components in $COMPONENTS; do
        for projection in $PROJECTIONS; do
            for seed in $SEEDS; do
                jobs+=("$cohorts $components $projection $seed")
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
