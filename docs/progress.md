# Progress and results

This file records results and decisions as they come in, newest entries at the top of
each section. The plan is in the README "Research roadmap". Record W&B run ids so every
number can be traced to `runs/pathways/xai-wsi/<id>/`.

Conventions:
- TCGA-val numbers use the fixed `split_cases` split. CPTAC is used for external testing
  only and is never used for tuning.
- Report 95% bootstrap CIs over patients whenever the comparison is close.

## Data status (2026-10-22)

| | TCGA-LUAD | CPTAC-LUAD |
|---|---|---|
| RNA samples (selected) | 529 | 228 |
| Slides with H-optimus-1 embeddings | 540 | 640 |
| Tiles per slide (median / p95 / max) | 15.7k / 36k / 59k | 9.6k / 29k / 42k |
| Cases with WSI + RNA | 474 | 224 |
| … with usable OS | 465 (163 events) | 203 (54 events) |
| OS time ≤ 0 (to drop or offset) | 4 | 1 |
| Missing BulkFormer vocabulary genes | 0.51% | 0.51% |

Notes:
- 15 CPTAC cases are slide-only (from TCIA) and absent from GDC, so they have no RNA and
  no clinical data.
- TCGA survival endpoints come from TCGA-CDR, CPTAC's from GDC CPTAC-3.

## Stage 1: WSI → 50 Hallmarks (mean-z targets)

Fill these in from W&B; the runs are on the other server.

| Date | W&B id | Model | Tiles enc. | TCGA-val median Spearman | TCGA-val median R² | CPTAC median Spearman | CPTAC median R² | Notes |
|---|---|---|---|---|---|---|---|---|
| | | `MeanMLP` | H-optimus-1 | | | | | |
| | | `TileLinearMean` | H-optimus-1 | | | | | |
| | | `TileMLPMean` | H-optimus-1 | | | | | |
| | | `TileMLPGatedMean` | H-optimus-1 | | | | | |

## Stage 2

### 2a. BulkFormer integration checks (2026-10-22)

All checks ran in the project uv env (torch 2.14.1+cu126, torch-geometric 2.8.1,
performer-pytorch 1.1.4) on an H100.

| Check | Result |
|---|---|
| 147M checkpoint, strict `load_state_dict` | passes |
| GCN on `torch.sparse` adjacency vs dense reference | max abs diff 4.8e-7 |
| Batched (`BatchedGCNConv`) vs per-sample GCN | max abs diff 2.4e-7 |
| Our `counts_to_log_tpm` vs upstream `normalize_data` (demo counts) | identical (0.0) |
| Sample embedding, fp16 vs fp32 autocast | cosine 0.9999998 |

Imputation check on 64 TCGA-LUAD samples, with 15% of observed genes masked and
predicted. PCC across genes is dominated by gene identity, so the per-gene metric is the
informative one.

| Predictor | PCC across genes (mean over samples) | Median per-gene PCC across samples |
|---|---|---|
| Per-gene mean (baseline) | 0.934 | 0.000 |
| BulkFormer-37M | 0.891 | 0.689 |
| BulkFormer-147M | 0.911 | 0.766 |

Embeddings written: `rna-embeddings/bulkformer/147M/embeddings.parquet` (TCGA 529 × 643,
CPTAC 228 × 643, no NaN).

### 2b. RNA sanity checks

| Date | RNA encoder | Dim | Hallmark probe median R² (TCGA-val) | Cohort classifier AUC | Notes |
|---|---|---|---|---|---|
| | BulkFormer-147M (mean) | 640 | | | |
| | PCA of log TPM | 640 | | | |

### 2c. WSI ↔ RNA alignment (retrieval)

Chance level is recall@k = k / n_candidates and median rank = n / 2.

| Date | W&B id | Slide encoder | RNA encoder | B | TCGA-val R@1 / R@5 / R@10 (WSI→RNA) | TCGA-val median rank | CPTAC R@1 / R@5 / R@10 | CPTAC median rank | Notes |
|---|---|---|---|---|---|---|---|---|---|

### 2d. Downstream probes on frozen encoders

Hallmarks from WSI only:

| Date | Slide embedding | Probe | TCGA-val median Spearman | CPTAC median Spearman | vs stage-1 best |
|---|---|---|---|---|---|

Overall survival (C-index, 95% CI):

| Date | Arm | Inputs | TCGA-val C-index | CPTAC C-index | Notes |
|---|---|---|---|---|---|
| | Clinical | age, stage | | | |
| | WSI only | slide embedding | | | |
| | RNA only | BulkFormer-147M | | | |
| | Fusion | concat | | | |

## Decisions log

- **2026-10-22:** BulkFormer is used frozen with upstream preprocessing (ln TPM+1, -10
  for missing genes). The model code is vendored; `torch_sparse` is replaced by native
  `torch.sparse`.
- **2026-10-22:** survival endpoints come from TCGA-CDR (TCGA) and GDC (CPTAC). The
  primary endpoint is OS, since CPTAC has no curated PFI.
- **2026-10-22:** the slide encoder for alignment is trained on lists of whole-slide bags
  (no tile sampling), B ≈ 64. Tile subsampling is an optional augmentation experiment.
- **2026-10-22:** Hallmarks remain a WSI-only target. Fused models are evaluated on
  survival or retrieval only.
