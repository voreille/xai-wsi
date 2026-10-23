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

TCGA-LUSC, added 2026-10-23 as extra training data (CPTAC-LSCC RNA downloaded, test only,
not processed yet): 501 RNA samples (primary tumour), 512 diagnostic slides from 478 cases,
476 cases with WSI + RNA. Tiling/embedding: `scripts/tile_embed_cohort.sh` (host, pending).

## Stage 1: WSI → 50 Hallmarks (mean-z targets)

Fill these in from W&B; the runs are on the other server.

| Date | W&B id | Model | Tiles enc. | TCGA-val median Spearman | TCGA-val median R² | CPTAC median Spearman | CPTAC median R² | Notes |
|---|---|---|---|---|---|---|---|---|
| | | `MeanMLP` | H-optimus-1 | | | | | |
| | | `TileLinearMean` | H-optimus-1 | | | | | |
| | | `TileMLPMean` | H-optimus-1 | | | | | |
| | | `TileMLPGatedMean` | H-optimus-1 | | | | | |

### Learning curve: is WSI → Hallmarks data-limited? (2026-10-23)

Ridge probe on per-slide mean H-optimus-1 embeddings, on random fractions of TCGA-train
(5 subsets per fraction; z-score statistics fit on each subset), TCGA-val and CPTAC fixed
(`xaiwsi.evaluation.learning_curve`, `outputs/learning_curve/`). This is a proxy for the
data needs, not a stage-1 model. Tile-level confirmation: `scripts/learning_curve.sh` on the
host (`data.train_fraction`, `data.train_subset_seed`).

| Train cases | TCGA-val median Spearman (sd) | CPTAC median Spearman (sd) |
|---|---|---|
| 38 (10%) | 0.167 (0.070) | 0.180 (0.033) |
| 76 (20%) | 0.218 (0.043) | 0.215 (0.051) |
| 113 (30%) | 0.248 (0.041) | 0.243 (0.042) |
| 189 (50%) | 0.227 (0.052) | 0.300 (0.053) |
| 265 (70%) | 0.252 (0.032) | 0.337 (0.027) |
| 378 (100%) | 0.283 | 0.364 |

- Still rising at 100%, with no plateau: CPTAC gains about 0.06 per doubling of the
  training set (0.300 → 0.364 from 189 to 378 cases). TCGA-val is noisier (95 cases) but
  follows the same direction.
- CPTAC median R² stays negative (−0.26 at 100%), the known offset between cohorts.
- An sklearn MLP probe was dropped: unstable with early stopping on few cases.
- Suggests more training cases would help, at least for this linear probe. Whether cases
  from other cancer types help LUAD as much as LUAD cases do is a separate question.

### Approximate Tavernari growth-pattern sets vs Hallmarks (2026-10-23)

The published L2S signature genes are not released. Approximate lepidic/solid sets were
derived from the supplement's pattern-wise DE tables (`scripts/build_tavernari_approx_genesets.py`,
sources in `data/external/tavernari/supplement/SOURCE.md`): TCGA (Table S3, |LvsS| ≥ 1:
476 lepidic / 322 solid), CHUV (Table S6, |LvsS| ≥ 1: 88 / 92), and REPLICATED (in both
tables with the same sign: 45 / 17; all 62 shared genes agree in sign). 17 TCGA-set genes with
outdated symbols are not scored. `L2S_<src>` = solid − lepidic mean-z, with train-only
z-score statistics (`xaiwsi.evaluation.geneset_vs_hallmarks`, outputs in
`outputs/tavernari_approx/`). The TCGA and REPLICATED sets were selected partly on TCGA-LUAD
samples, so they are not independent of our cohort; the CHUV sets are.

RNA side only: how well a ridge model on the 50 Hallmark scores (fit on TCGA-train)
predicts each score.

| Score | Top Hallmarks (Spearman, TCGA) | R² val | R² CPTAC | Spearman CPTAC |
|---|---|---|---|---|
| `L2S_TCGA` | E2F +0.70, G2M +0.66, MTORC1 +0.54 | 0.94 | 0.90 | 0.97 |
| `L2S_REPLICATED` | E2F +0.65, G2M +0.60, MTORC1 +0.51 | 0.88 | 0.77 | 0.95 |
| `L2S_CHUV` | bile acid −0.54, fatty acid −0.44, heme −0.43 | 0.75 | 0.80 | 0.92 |
| `SOLID_TCGA` | E2F +0.87, G2M +0.85, MTORC1 +0.74 | 0.98 | 0.95 | 0.99 |
| `SOLID_CHUV` | EMT +0.46, allograft rejection +0.45, inflammatory +0.37 | 0.64 | 0.68 | 0.85 |
| `LEPIDIC_TCGA` | bile acid +0.63, KRAS_DN +0.57, myogenesis +0.51 | 0.89 | 0.83 | 0.95 |

- The 50 Hallmarks explain 88–94% of the variance of the TCGA-derived axes on TCGA-val. The
  solid side is mostly proliferation (SOLID_TCGA shares 60 genes with G2M_CHECKPOINT and
  57 with E2F_TARGETS).
- The CHUV axis keeps the most unexplained variance (25%), but its solid set tracks EMT and
  immune Hallmarks: probably stromal/immune content of the CHUV regions. Tavernari et al.
  removed this by keeping only cancer-specific markers, which can't be redone from the
  tables. The lepidic sets track bile acid/fatty acid metabolism, KRAS_DN and myogenesis,
  probably normal alveolar and stromal content.
- The three L2S variants agree (Spearman 0.62–0.97). The CHUV variant agrees least.
- Conclusion: as RNA targets, these approximate sets add little beyond the Hallmarks. This
  says nothing yet about how predictable they are from the WSI.

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

### 2b. RNA sanity checks (2026-10-22)

Command: `python -m xaiwsi.evaluation.rna_probes --config configs/training/pathway_baseline.yaml --output-dir outputs/rna_probes`.

Setup:
- Ridge probes (per-target alpha chosen by LOO on train) to mean-z Hallmarks, on the
  stage-1 pairs and split (`split_seed` 42, `val_fraction` 0.2): 378 train, 95 val, 213
  CPTAC.
- Z-score statistics and PCA are fit on TCGA-train only.
- Targets are derived from the same RNA, so this measures how much information each
  representation keeps. It is not a prediction result.

| Representation | Dim | Val median R² | Val median Spearman | CPTAC median R² | CPTAC median Spearman |
|---|---|---|---|---|---|
| BulkFormer-147M | 8 | 0.798 | 0.891 | 0.240 | 0.843 |
| BulkFormer-147M | 32 | 0.919 | 0.957 | 0.801 | 0.920 |
| BulkFormer-147M | 128 | **0.956** | **0.976** | **0.876** | 0.946 |
| BulkFormer-147M | 640 (all) | **0.956** | 0.975 | 0.859 | **0.947** |
| PCA of log2 TPM | 8 | 0.827 | 0.902 | -1.790 | 0.787 |
| PCA of log2 TPM | 32 | 0.904 | 0.949 | 0.569 | 0.897 |
| PCA of log2 TPM | 128 | 0.934 | 0.964 | 0.727 | 0.927 |
| log2 TPM (all genes) | 56,109 | 0.951 | 0.974 | 0.793 | 0.933 |

Rank-k Hallmark oracle: project the true scores on the top-k TCGA-train Hallmark PCs and
reconstruct. This is the ceiling of a rank-k Hallmark head.

| k | Val median R² | Val median Spearman | CPTAC median R² | CPTAC median Spearman |
|---|---|---|---|---|
| 1 | 0.646 | 0.771 | 0.202 | 0.714 |
| 3 | 0.846 | 0.913 | 0.575 | 0.888 |
| 5 | 0.906 | 0.947 | 0.694 | 0.931 |
| 10 | 0.954 | 0.969 | 0.849 | 0.956 |
| 20 | 0.979 | 0.987 | 0.935 | 0.982 |

Findings:
- **BulkFormer passes.** It retains Hallmark information as well as the full 56k-gene
  matrix on TCGA-val (0.956 vs 0.951), is more compact at 32–128 dims, and transfers
  better to CPTAC (R² 0.86–0.88 vs 0.79; at dim 8, 0.24 vs -1.79).
  - Use BulkFormer-147M as the RNA encoder for 2c. Keep PCA-128 as the baseline arm.
- **Cohort shift is large in every representation.** A TCGA-vs-CPTAC logistic
  classifier reaches AUC 0.999–1.000 everywhere, so this metric saturates and can't rank
  encoders. The CPTAC R² drop, with Spearman staying high, points to a
  calibration/offset shift. Report Spearman for CPTAC, and consider per-cohort centering
  for retrieval.
- **A rank-5 to rank-10 Hallmark head loses little** (val R² 0.91–0.95), which supports
  the low-rank head experiment in 2d. The BulkFormer route's ceiling (about 0.956) is the
  same as rank 10.

### 2c. WSI ↔ RNA alignment (retrieval)

R@k values for the W&B runs are last-epoch values (the W&B summary). The best-epoch val
MRR comes from the checkpoint's `ModelCheckpoint.best_model_score`.


Chance level is recall@k = k / n_candidates and median rank = n / 2.

| Date | W&B id | Slide encoder | RNA encoder | B | TCGA-val R@1 / R@5 / R@10 (WSI→RNA) | TCGA-val median rank | CPTAC R@1 / R@5 / R@10 | CPTAC median rank | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-22 | `orpg08id` | Gated, 512→256 | BulkFormer-147M → MLP 256 | 64 | last epoch (45): 0.211 / 0.495 / 0.653 | 5 | best ckpt: 0.066 / 0.192 / 0.347; centered: 0.103 / 0.230 / 0.352 | 21; centered 19 | Best val MRR **0.386** at epoch 15 (`epoch=015-step=80.ckpt`), early-stopped at 45. Train loss 0.055 vs val 3.25: overfits after about epoch 15. fp16 stores + RAM cache. |
| 2026-10-22 | `kq0d13qf` | Mean, 512→256 | BulkFormer-147M → MLP 256 | 64 | last epoch (43): 0.242 / 0.521 / 0.668 | 5 | best ckpt: 0.042 / 0.164 / 0.338; centered: 0.047 / 0.169 / 0.333 | 20; centered 22 | Best val MRR **0.405** at epoch 13 (`epoch=013-step=70.ckpt`), early-stopped at 43. Same overfitting. Mean ≈ gated: the 0.02 MRR gap is within noise for 95 val cases. fp16 stores + RAM cache. |
| 2026-10-22 | smoke (CSV) | Gated, 512→256 | BulkFormer-147M → MLP 256 | 64 | 0.126 / 0.284 / 0.453 | 12 | 0.033 / 0.089 / 0.207 | 41 | **Pipeline check only:** 2 epochs (10 steps), not tuned. Chance: val R@5 0.053 (median rank 47.5), CPTAC R@5 0.023 (median rank 106). Peak GPU 15 GB. About 8 min/epoch with `num_workers 0` (I/O-bound) |

CPTAC external test (2026-10-23, best checkpoints, 213 candidates, chance R@5 0.023, median
rank 106). "Centered" means `data.rna_center_per_cohort true`: CPTAC RNA embeddings are
centered on their own mean (label-free but transductive) and scaled with the TCGA-train std.
Run in claude-box without a W&B key (`--trainer.logger false`), so these numbers are not in
the W&B runs yet.

| W&B id | RNA centering | WSI→RNA R@1 / R@5 / R@10, median rank, MRR | RNA→WSI R@1 / R@5 / R@10, median rank, MRR |
|---|---|---|---|
| `kq0d13qf` (mean) | TCGA-train | 0.042 / 0.164 / 0.338, 20, 0.128 | 0.019 / 0.103 / 0.174, 65, 0.076 |
| `kq0d13qf` (mean) | per cohort | 0.047 / 0.169 / 0.333, 22, 0.138 | 0.080 / 0.235 / 0.385, 17, 0.181 |
| `orpg08id` (gated) | TCGA-train | 0.066 / 0.192 / 0.347, 21, 0.147 | 0.028 / 0.080 / 0.169, 57, 0.076 |
| `orpg08id` (gated) | per cohort | 0.103 / 0.230 / 0.352, 19, 0.182 | 0.089 / 0.235 / 0.404, 17, 0.177 |

- Alignment transfers to CPTAC (WSI→RNA R@5 about 7–10× chance) but is far below TCGA-val
  (MRR 0.39–0.41 vs 0.13–0.18), consistent with overfitting plus a cohort shift.
- The cohort offset sits in the RNA embeddings and mainly breaks RNA→WSI (median rank
  57–65 → 17). A constant shift of the RNA queries changes which slides rank first, while a
  constant shift of all RNA candidates mostly preserves the ranking. Per-cohort centering
  fixes it and makes the two directions symmetric.
- Gated ≥ mean on CPTAC (centered MRR 0.182 vs 0.138 WSI→RNA), the reverse of TCGA-val. With
  213 cases and one seed this is suggestive, not conclusive.

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

- **2026-10-23:** exploration aim: learn from WSI/RNA pairs something more objective
  than pathologist labels. Pattern/grade labels are used only to interpret and evaluate.
  Public pattern labels are scarce: ANORAK (quality uncertain), DHMC (kept as test), CPTAC
  dominant subtype on cBioPortal (110 cases, mostly acinar), predominant pattern in about
  50 TCGA pathology reports (TCGA-Reports OCR). Grade: CPTAC GDC `tumor_grade` (219 of
  228 cases), TCGA report text (about 300 cases), LungHist700 (45 patients, image level).
- **2026-10-23:** final goal written down: WSI-only inference of the activity of the 50
  Hallmarks, as a continuous activity with uncertainty (not a high/low class),
  interpretable at tile level. The Tavernari et al. growth-pattern gene sets are not in the
  paper's released material, so Tavernari is on hold. Hallmarks are cross-cancer, so more
  TCGA/CPTAC cohorts are an option against overfitting. RNA is training-time
  only. So RNA-side per-cohort centering on CPTAC is a diagnostic (it shows the cohort shift
  sits in the RNA embeddings), not part of the method. Shifts on the slide side can be
  corrected at test time from slides alone. Still exploratory: alignment and fusion are
  judged by how much they help the WSI-only predictor.
- **2026-10-22:** H-optimus-1 stores repacked to fp16, uncompressed, with 1024-row chunks
  (`h-optimus-1-fp16/`, `wsi-repack` in wsitools, every slide verified). The alignment
  config uses them. Stage-1 configs still point at the original fp32 LZF stores.
  Alignment training also caches decoded slides in RAM (`cache_tiles`), which made 4
  epochs take about 80 s instead of several minutes per epoch.
- **2026-10-22:** the 2c slide encoder projects tiles into the shared space before
  pooling, so tile tokens can feed the stage-4 multimodal SAE. BulkFormer gene-level
  embeddings move up in priority.
- **2026-10-22:** Hallmark scores are low-rank (participation ratio 3.7, 5 PCs = 80% of
  variance, TCGA-LUAD, scores on all samples, descriptive only). Low-rank heads and a
  frozen BulkFormer decoder head are compared before any full-embedding regression.
- **2026-10-22:** BulkFormer is used frozen with upstream preprocessing (ln TPM+1, -10
  for missing genes). The model code is vendored; `torch_sparse` is replaced by native
  `torch.sparse`.
- **2026-10-22:** survival endpoints come from TCGA-CDR (TCGA) and GDC (CPTAC). The
  primary endpoint is OS, since CPTAC has no curated PFI.
- **2026-10-22:** the slide encoder for alignment is trained on lists of whole-slide bags
  (no tile sampling), B ≈ 64. Tile subsampling is an optional augmentation experiment.
- **2026-10-22:** Hallmarks remain a WSI-only target. Fused models are evaluated on
  survival or retrieval only.
