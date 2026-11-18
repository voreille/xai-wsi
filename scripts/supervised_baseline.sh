#!/bin/bash
# Supervised baseline for the 2f paper table: tile-level models trained end to end on the
# Hallmarks (TileMLPMean, TileMLPGatedMean, ABMILMean), against the slide-mean ridge and the SAE
# concept bags, with the same splits (split_seed 42), the same train-only per-cohort
# z-scores and paired bootstraps on the same cases.
#
# LUAD side: existing runs, selected on TCGA-LUAD val (coh-1024-luad*, coh-1105gated-luad).
# LSCC side: new runs selected on TCGA-LUSC val (configs/training/overrides/select_lusc.yaml),
#   conditions tcga-lusc and tcga-luad,tcga-lusc, ids coh-<LUSC_TAG>[gated]-<cond>-percohort-s<seed>.
#   coh-1024-luad+lusc (selected on LUAD val) is kept as a second LSCC row.
# ABMIL (gated attention MIL, configs/training/overrides/abmil.yaml), both sides, conditions
#   tcga-<side> and tcga-luad,tcga-lusc: coh-<ABMIL_TAG>-<cond>-percohort-s<seed> (LUAD val)
#   and coh-<ABMIL_TAG>lusc-<cond>-percohort-s<seed> (LUSC val).
# FIT_MODELS picks the models to fit (tile, gated, abmil); fitting skips runs with a
# checkpoint but re-tests them, so fit only what is new.
#
# Stages:
#   fit        train + test the LSCC-side runs (scripts/pathway_cohorts.sh; skips existing)
#   predict    val + test predictions of every run in the table (predict_run; skips existing)
#   bootstrap  paired bootstraps -> $OUT/{vs_mean,sae_vs_supervised}_<cohort>.{md,csv}
# Val numbers of the tile models are optimistic: their checkpoint is picked on that val.
#
# Timing (host, 2 GPUs): fit 20-40 min per run (ABMIL: 12 runs); predict ~5 min per run.
#
# Usage (on the host, from the repo root):
#   GPUS="0 1" bash scripts/supervised_baseline.sh
#   FIT_MODELS=abmil GPUS="0 1" bash scripts/supervised_baseline.sh   # add the ABMIL rows
#   STAGES=bootstrap SIDES=luad bash scripts/supervised_baseline.sh
set -euo pipefail

STAGES=${STAGES:-"fit predict bootstrap"}
SIDES=${SIDES:-"luad lscc"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0"}
LUSC_TAG=${LUSC_TAG:-1112lusc}
ABMIL_TAG=${ABMIL_TAG:-1118abmil}
FIT_MODELS=${FIT_MODELS:-"tile gated abmil"}
RUN_ROOT=${RUN_ROOT:-runs/pathways/xai-wsi}
OUT=${OUT:-outputs/supervised_baseline}
LUSC_CONDITIONS="tcga-lusc tcga-luad,tcga-lusc"

mkdir -p "$OUT"
first_gpu=${GPUS%% *}

fit_family() {  # conditions tag extra-args
    CONDITIONS="$1" SEEDS="$SEEDS" GPUS="$GPUS" TAG="$2" RUN_ROOT="$RUN_ROOT" \
        NORMALIZATIONS=per_cohort STAGES=tiles EXTRA_ARGS="$3" \
        LOG_DIR="$OUT/logs" \
        bash scripts/pathway_cohorts.sh
}

if [[ " $STAGES " == *" fit "* ]]; then
    for model in $FIT_MODELS; do
        case $model in
            tile)  overlay="" ; lusc_tag=$LUSC_TAG ;;
            gated) overlay="--config configs/training/overrides/gated.yaml" ; lusc_tag=${LUSC_TAG}gated ;;
            abmil) overlay="--config configs/training/overrides/abmil.yaml" ; lusc_tag=${ABMIL_TAG}lusc ;;
            *) echo "unknown model '$model' in FIT_MODELS" >&2; exit 2 ;;
        esac
        if [[ " $SIDES " == *" lscc "* ]]; then
            fit_family "$LUSC_CONDITIONS" "$lusc_tag" \
                "$overlay --config configs/training/overrides/select_lusc.yaml"
        fi
        # The LUAD-side tile and gated runs predate this script (coh-1024, coh-1105gated).
        if [[ $model == abmil && " $SIDES " == *" luad "* ]]; then
            fit_family "tcga-luad tcga-luad,tcga-lusc" "$ABMIL_TAG" "$overlay"
        fi
    done
fi

# Run families per side, as label=prefix (seeds are -s*).
luad_runs=(
    "tilemlp-luad=coh-1024-luad-percohort"
    "tilemlp-luad+lusc=coh-1024-luad+lusc-percohort"
    "gated-luad=coh-1105gated-luad-percohort"
    "abmil-luad=coh-${ABMIL_TAG}-luad-percohort"
    "abmil-luad+lusc=coh-${ABMIL_TAG}-luad+lusc-percohort"
)
lscc_runs=(
    "tilemlp-lusc=coh-${LUSC_TAG}-lusc-percohort"
    "tilemlp-luad+lusc=coh-${LUSC_TAG}-luad+lusc-percohort"
    "gated-lusc=coh-${LUSC_TAG}gated-lusc-percohort"
    "gated-luad+lusc=coh-${LUSC_TAG}gated-luad+lusc-percohort"
    "tilemlp-luad+lusc-selLUAD=coh-1024-luad+lusc-percohort"
    "abmil-lusc=coh-${ABMIL_TAG}lusc-lusc-percohort"
    "abmil-luad+lusc=coh-${ABMIL_TAG}lusc-luad+lusc-percohort"
)

if [[ " $STAGES " == *" predict "* ]]; then
    todo=()
    for side in $SIDES; do
        declare -n runs="${side}_runs"
        for entry in "${runs[@]}"; do
            for seed in $SEEDS; do
                run_dir="$RUN_ROOT/${entry#*=}-s$seed"
                [[ -f "$run_dir/predictions.parquet" ]] || todo+=("$run_dir")
            done
        done
        unset -n runs
    done
    if (( ${#todo[@]} )); then
        echo "predict ${#todo[@]} runs on GPU $first_gpu"
        CUDA_VISIBLE_DEVICES=$first_gpu uv run python -m xaiwsi.evaluation.predict_run \
            --test-cohorts-from configs/training/pathway_cohorts.yaml \
            --run-dir $(printf '%s\n' "${todo[@]}" | sort -u)
    fi
fi

if [[ " $STAGES " == *" bootstrap "* ]]; then
    for side in $SIDES; do
        declare -n runs="${side}_runs"
        if [[ $side == luad ]]; then
            mean=outputs/concept_sweep/tcga-luad_k1024_top64_s0/predictions/tcga-luad__per_cohort.parquet
            sae='sae-valpick-5seeds=outputs/concept_sweep/tcga-luad+tcga-lusc_k1024_top64_s*/predictions/tcga-luad__per_cohort__table-concepts-tcga-luad+tcga-lusc_k1024_top64_s*-slides_mean__min0_clip.parquet'
            sae_all='sae-all18=outputs/concept_sweep/*_s0/predictions/tcga-luad__per_cohort__table-concepts-*_s0-slides_mean__min0_clip.parquet'
            pairs="cptac-luad:test tcga-luad:val"
        else
            mean=outputs/concept_sweep_lusc/tcga-lusc_k1024_top32_s0/predictions/tcga-lusc__per_cohort.parquet
            sae='sae-valpick-5seeds=outputs/concept_sweep_lusc/tcga-lusc_k1024_top32_s*/predictions/tcga-lusc__per_cohort__table-concepts-tcga-lusc_k1024_top32_s*-slides_mean__min0_clip.parquet'
            sae_all='sae-all18=outputs/concept_sweep_lusc/*lusc_k*_s0/predictions/tcga-lusc__per_cohort__table-concepts-*lusc_k*_s0-slides_mean__min0_clip.parquet'
            pairs="cptac-lusc:test tcga-lusc:val"
        fi
        supervised=()
        for entry in "${runs[@]}"; do
            supervised+=("${entry%%=*}=$RUN_ROOT/${entry#*=}-s[0-9]/predictions.parquet")
        done
        for pair in $pairs; do
            cohort=${pair%%:*}
            split=${pair#*:}
            uv run python -m xaiwsi.evaluation.paired_bootstrap \
                --baseline "slide-mean=$mean" \
                --candidates "${supervised[@]}" "$sae" "$sae_all" \
                --cohort "$cohort" --split "$split" \
                --output "$OUT/vs_mean_${cohort}"
            uv run python -m xaiwsi.evaluation.paired_bootstrap \
                --baseline "${supervised[@]}" \
                --candidates "$sae" "$sae_all" \
                --cohort "$cohort" --split "$split" \
                --output "$OUT/sae_vs_supervised_${cohort}"
        done
        unset -n runs
    done
fi
