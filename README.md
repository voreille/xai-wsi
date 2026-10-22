# xai-wsi

Utilities and baselines for linking whole-slide histopathology representations with bulk RNA-derived pathway activity in lung adenocarcinoma (LUAD).

The current pipeline supports:

- TCGA-LUAD and CPTAC-LUAD RNA preprocessing from GDC augmented STAR gene-count files.
- TCGA/CPTAC WSI metadata preparation.
- Hallmark pathway scoring from bulk RNA.
- Frozen-VFM WSI pathway prediction baselines.
- TCGA train/validation with CPTAC external testing.
- LightningCLI-based training with swappable models, loggers, callbacks, and configs.
- Tile-level pathway score, relevance-weight, and contribution heatmaps.

The main use case is to study whether morphology encoded by pathology foundation models predicts RNA-derived biological programs, while retaining tile-level interpretability.

The longer-term goal is a multimodal (WSI + bulk RNA) model for LUAD that is interpretable at the tile and concept level, then possibly extending to other TCGA/CPTAC cancer types. See [Research roadmap](#research-roadmap).

---

## Repository layout

```text
xai-wsi/
├── configs/
│   ├── rna/
│   │   └── bulkformer_{37M,147M}.yaml
│   └── training/
│       └── pathway_baseline.yaml
│
├── models/
│   └── external/bulkformer/   # weights + gene graph (gitignored)
│
├── data/
│   ├── raw/
│   │   ├── tcga-luad-rna/
│   │   └── cptac-luad-rna/
│   ├── external/
│   │   ├── msigdb/
│   │   └── tcga-cdr/
│   └── processed/
│       ├── tcga-luad/
│       │   ├── metadata/
│       │   ├── rna/
│       │   ├── rna-embeddings/
│       │   ├── wsi-tiles/
│       │   └── wsi-embeddings/
│       └── cptac-luad/
│           ├── metadata/
│           ├── rna/
│           ├── rna-embeddings/
│           ├── wsi-tiles/
│           └── wsi-embeddings/
│
└── xaiwsi/
    ├── data/
    ├── metrics/
    ├── models/
    ├── preprocessing/
    │   ├── gdc/
    │   ├── tcga/
    │   └── cptac/
    ├── rna/
    │   ├── encoders/
    │   │   └── bulkformer/
    │   └── pathways/
    ├── training/
    └── visualization/
```

---

## Installation

The project uses `uv`. It depends on [wsi-tools](https://github.com/voreille/wsi-tools) (`wsitools` package), installed editable from a sibling checkout:

```bash
git clone git@github.com:voreille/wsi-tools.git ../wsi-tools
uv sync
```

All dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

OpenSlide binaries come from `openslide-bin`, so no system OpenSlide install is needed.

---

# Data preparation

## 1. Download GDC RNA data

The RNA pipeline expects GDC **augmented STAR gene counts** files.

Raw files can be downloaded with the GDC client using a manifest:

```bash
gdc-client download \
    -m manifest.txt \
    -d data/raw/tcga-luad-rna
```

and similarly for CPTAC-LUAD:

```bash
gdc-client download \
    -m manifest.txt \
    -d data/raw/cptac-luad-rna
```

The raw GDC filenames are kept untouched. Dataset-specific metadata tables provide the mapping from GDC file IDs to cases, samples, and aliquots.

---

## 2. Build RNA metadata

### TCGA-LUAD

```bash
python -m xaiwsi.preprocessing.tcga.rna_metadata
```

### CPTAC-LUAD

```bash
python -m xaiwsi.preprocessing.cptac.rna_metadata
```

The CPTAC metadata pipeline uses the same harmonized GDC expression format, but does not assume TCGA barcode semantics.

---

## 3. Preprocess RNA

### TCGA-LUAD

```bash
python -m xaiwsi.preprocessing.tcga.rna
```

### CPTAC-LUAD

```bash
python -m xaiwsi.preprocessing.cptac.rna
```

The canonical RNA outputs include:

```text
rna/
├── counts.parquet
├── tpm.parquet
├── log2_tpm.parquet
├── tmm_cpm.parquet
├── log2_tmm_cpm.parquet
├── genes.csv
└── samples.csv
```

For TCGA, replicate-selection information may additionally be stored in:

```text
rna_replicates.csv
```

The Hallmark mean-z training pipeline uses TPM as:

```text
TPM
→ collapse duplicate gene symbols in linear TPM space
→ log2(TPM + 1)
→ gene-wise z-score using training-set statistics
→ mean over genes in each Hallmark
```

The z-score reference statistics are intentionally fit **after the train/validation split**.

---

## 4. Build WSI metadata

### TCGA-LUAD

```bash
python -m xaiwsi.preprocessing.tcga.wsi_metadata
```

### CPTAC-LUAD

```bash
python -m xaiwsi.preprocessing.cptac.wsi_metadata
```

WSI/RNA matching is performed at the case/patient level.

For TCGA, slide IDs and RNA sample IDs are not expected to be identical. A diagnostic slide can use a `01Z` specimen identifier while the RNA sample uses `01A`; both still map to the same TCGA case.

---

## 5. Build clinical / survival data

Both steps write one row per case to `data/processed/<cohort>/metadata/clinical.csv`,
keyed by `case_id` (the same key used for WSI/RNA pairing). Shared columns: `os_time`
(days), `os_event` (1 = death), `age` (years), `sex`, `stage`, `stage_group` (I-IV).

### TCGA-LUAD

Endpoints come from the curated **TCGA Clinical Data Resource** (TCGA-CDR, Liu et al.,
*Cell* 2018), not from raw GDC follow-up. The table is downloaded from GDC if it is
missing. Besides OS, it keeps `dss_*`, `dfi_*` and `pfi_*`.

```bash
python -m xaiwsi.preprocessing.tcga.clinical \
    --cdr data/external/tcga-cdr/TCGA-CDR-SupplementalTableS1.xlsx \
    --cancer-type LUAD \
    --output data/processed/tcga-luad/metadata/clinical.csv
```

### CPTAC-LUAD

Clinical data comes from the GDC API. CPTAC-3 spans several cancer types, so it is
restricted to the cases in our metadata. OS is `days_to_death` for deaths and the latest
follow-up otherwise. A few slide-only cases (from TCIA) are not in GDC and are skipped
with `--allow-missing`.

```bash
python -m xaiwsi.preprocessing.cptac.clinical \
    --cases-from data/processed/cptac-luad/metadata/rna_samples.csv \
                 data/processed/cptac-luad/metadata/wsi_slides.csv \
    --allow-missing \
    --output data/processed/cptac-luad/metadata/clinical.csv
```

Current coverage of the WSI+RNA paired cases with usable OS: TCGA-LUAD 465 (163 events),
CPTAC-LUAD 203 (54 events). A few cases have `os_time <= 0`; survival models should
drop or offset them.

---

# WSI tiling and embedding

WSI tiling and embedding are performed with `wsitools`.

## Tiling

```bash
wsi-tile \
    --source /path/to/slides \
    --output data/processed/tcga-luad/wsi-tiles \
    --config configs/tiling/tiling.yaml \
    --rglob-str "*DX*.svs" \
    --auto-skip
```

Typical tiling-store outputs include:

```text
wsi-tiles/
├── .tiling_store.json
├── masks/
├── patches_json/
├── stitches/
├── tiling_jobs.yaml
└── tiling_manifest.yaml
```

## Embedding extraction

Example with H-optimus-1:

```bash
CUDA_VISIBLE_DEVICES=0 wsi-embed \
    --tiles-rootdir data/processed/tcga-luad/wsi-tiles \
    --slides-rootdir /path/to/slides \
    --model-name h-optimus-1 \
    --output-dir data/processed/tcga-luad/wsi-embeddings/h-optimus-1 \
    --batch-size 256 \
    --num-workers 8
```

Each slide keeps its own embedding file. The embedding store records geometry and slide metadata such as:

```text
relative_wsi_path
coord_space
level0_dim
patch_size
step_size
encoder_name
encoder_out_dim
```

This lets downstream visualization recover the original WSI and reconstruct tile-level maps.

---

# Pathway scoring CLI

Pathway utilities live under:

```text
xaiwsi.rna.pathways
```

The generic pathway CLI is:

```bash
python -m xaiwsi.rna.pathways \
    --expression <expression.parquet> \
    --genes <genes.csv> \
    --gene-sets <gene_sets.gmt> \
    --gene-set-type <gmt|hallmarks|tavernari> \
    --method <ssgsea|mean-z> \
    --output <scores.parquet>
```

## BulkFormer RNA embeddings

[BulkFormer](https://github.com/KangBoming/BulkFormer) is used as a frozen bulk-RNA encoder.
Its model code (MIT) is vendored unchanged in `xaiwsi/rna/encoders/bulkformer/`
(upstream commit `d63c881`). We don't use the authors' conda environment: inference only
needs `torch-geometric` and `performer-pytorch==1.1.4`. Their compiled `torch_sparse`
`SparseTensor` graph is replaced by the same adjacency as a native `torch.sparse` tensor.
A `BatchedGCNConv` subclass handles batched inputs.

Files in `models/external/bulkformer/` (all from
[Zenodo 10.5281/zenodo.15744294](https://doi.org/10.5281/zenodo.15744294) except the
checkpoints, which come from the Google Drive links in the upstream README):

```text
BulkFormer_147M.pt  BulkFormer_37M.pt  G_tcga.pt  G_tcga_weight.pt
bulkformer_gene_info.csv  gene_length_df.csv
```

Input preprocessing follows upstream exactly:
- raw `unstranded` counts are converted to TPM with the upstream gene lengths
  (1000 bp if a gene has no length);
- then `ln(TPM + 1)` (natural log, **not** our `log2_tpm`);
- genes are aligned to the 20,010-gene vocabulary, and missing genes are set to -10;
- `mask_prob` is the missing-gene fraction.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m xaiwsi.rna.encoders.bulkformer \
    --config configs/rna/bulkformer_147M.yaml \
    --counts data/processed/tcga-luad/rna/counts.parquet \
    --output-dir data/processed/tcga-luad/rna-embeddings/bulkformer/147M
```

The output is `embeddings.parquet`, with one row per RNA sample (`sample_id`, as in
`samples.csv`). Columns `emb_0..emb_{dim-1}` are the gene tokens averaged over genes
(`--aggregation mean|max|median|all`). The three `stat_*` columns are the upstream
sample statistics appended to every token: mask ratio, mean expression and non-zero
ratio. `meta.json` records the inputs and settings.

Sanity check (TCGA-LUAD, 64 samples, 15% of observed genes masked): imputation reaches a
median per-gene Pearson correlation across samples of 0.77 for 147M and 0.69 for 37M.
A per-gene-mean baseline gets 0.

## MSigDB Hallmarks

Store the human Hallmark gene-symbol GMT for example as:

```text
data/external/msigdb/h.all.v2026.1.Hs.symbols.gmt
```

### ssGSEA

For exploratory precomputed scores:

```bash
python -m xaiwsi.rna.pathways \
    --expression data/processed/tcga-luad/rna/log2_tpm.parquet \
    --genes data/processed/tcga-luad/rna/genes.csv \
    --gene-sets data/external/msigdb/h.all.v2026.1.Hs.symbols.gmt \
    --gene-set-type hallmarks \
    --method ssgsea \
    --output data/processed/tcga-luad/rna/pathways/hallmarks_ssgsea.parquet
```

### Mean-z

For pathway `P` and sample `s`:

```text
1. log2(TPM + 1)
2. z-score each gene across reference samples
3. average gene z-scores over genes belonging to P
```

For ML experiments, mean-z scores are normally computed inside the training pipeline so that gene means and standard deviations are fit on the **training samples only**.

The standalone CLI remains useful for exploratory analysis:

```bash
python -m xaiwsi.rna.pathways \
    --expression data/processed/tcga-luad/rna/log2_tpm.parquet \
    --genes data/processed/tcga-luad/rna/genes.csv \
    --gene-sets data/external/msigdb/h.all.v2026.1.Hs.symbols.gmt \
    --gene-set-type hallmarks \
    --method mean-z \
    --output data/processed/tcga-luad/rna/pathways/hallmarks_mean_z.parquet
```

---

# Pathway prediction training

Training uses PyTorch Lightning and LightningCLI.

The current baseline setup is:

```text
TCGA-LUAD
├── training
└── validation

CPTAC-LUAD
└── external test
```

Gene-wise z-score statistics are fit on the TCGA training split only and reused for TCGA validation and CPTAC.

## Models

The current pure-PyTorch pathway predictors include:

```text
TileLinearMean
TileMLPMean
TileMLPGatedMean
MeanMLP
```

### `TileMLPMean`

```text
tile embedding
→ MLP
→ 50 local pathway scores
→ uniform mean over tiles
→ 50 slide-level predictions
```

### `TileMLPGatedMean`

```text
tile embedding
→ shared MLP
├── pathway-score head → 50 local pathway scores
└── sigmoid gate head  → tile relevance

normalize gates by their sum
→ weighted average of local pathway scores
→ 50 slide-level predictions
```

The model returns:

```text
pred
tile_scores
tile_gates
tile_weights
```

The exact contribution of tile `i` to pathway `k` is:

```text
tile_weights[i] * tile_scores[i, k]
```

### `MeanMLP`

```text
mean tile embeddings
→ MLP
→ 50 pathway predictions
```

This model does not naturally provide tile-level pathway maps.

---

## Lightning config

Example:

```yaml
seed_everything: 42

model:
  predictor:
    class_path: xaiwsi.models.pathway.TileMLPGatedMean
    init_args:
      input_dim: 1536
      hidden_dim: 128
      output_dim: 50
      dropout: 0.2
      tile_dropout: 0.0

  lr: 1.0e-4
  weight_decay: 1.0e-4

data:
  tcga_rna_dir: /path/to/data/processed/tcga-luad/rna
  cptac_rna_dir: /path/to/data/processed/cptac-luad/rna

  tcga_embedding_dir: /path/to/data/processed/tcga-luad/wsi-embeddings/h-optimus-1
  cptac_embedding_dir: /path/to/data/processed/cptac-luad/wsi-embeddings/h-optimus-1

  tcga_slides_root_dir: /path/to/tcga/slides
  cptac_slides_root_dir: /path/to/cptac/slides

  tcga_wsi_metadata: /path/to/data/processed/tcga-luad/metadata/wsi_slides.csv
  cptac_wsi_metadata: /path/to/data/processed/cptac-luad/metadata/wsi_slides.csv

  hallmark_gmt: /path/to/data/external/msigdb/h.all.v2026.1.Hs.symbols.gmt

  val_fraction: 0.2
  split_seed: 42
  batch_size: 1
  num_workers: 8
  pin_memory: true

trainer:
  accelerator: auto
  devices: 1
  max_epochs: 30
  default_root_dir: /path/to/runs/pathways

  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: xai-wsi
      save_dir: /path/to/runs/pathways
      log_model: false

  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val/median_spearman
        mode: max
        save_top_k: 1
        save_last: true

    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val/median_spearman
        mode: max
        patience: 5
```

The logger is entirely config-driven and can be replaced with another Lightning logger, for example `CSVLogger`.

---

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m xaiwsi.training.cli fit \
    --config configs/training/pathway_baseline.yaml
```

Model selection is based on:

```text
val/median_spearman
```

Typical logged metrics include:

```text
train/loss
val/mse
val/zero_mse
val/skill
val/median_r2
val/median_spearman
val/mean_spearman
val/n_rho_gt_0
val/n_rho_gt_03
```

---

## Test on CPTAC

Use the saved run config and selected checkpoint:

```bash
python -m xaiwsi.training.cli test \
    --config runs/pathways/xai-wsi/<run-id>/config.yaml \
    --ckpt_path runs/pathways/xai-wsi/<run-id>/checkpoints/<checkpoint>.ckpt
```

Typical output includes:

```text
MSE
zero-baseline MSE
skill vs zero
median R²
median Spearman
mean Spearman
number of Hallmarks with rho > 0
number of Hallmarks with rho > 0.3
top Hallmarks by Spearman
```

CPTAC is intended as an **external test cohort**. Architecture and hyperparameter choices should therefore be made using TCGA training/validation only.

---

# Run artifacts and logging

A run is organized as:

```text
runs/pathways/
├── wandb/                       # W&B local cache / bookkeeping
└── xai-wsi/
    └── <run-id>/
        ├── config.yaml
        └── checkpoints/
            ├── <best>.ckpt
            └── last.ckpt
```

The resolved LightningCLI config is saved next to the checkpoints so downstream analysis can reconstruct the model and dataset paths from the run itself.

With `WandbLogger`, `<run-id>` is the W&B run id (the last part of the run URL), and:

- the saved `config.yaml` pins that id (`trainer.logger.init_args.id`), so `test` with it logs CPTAC metrics into the same W&B run and writes nothing new under `runs/`;
- the W&B run config has `run_dir`, `config_path` and `host`; add them as columns in the W&B runs table to copy the local paths of a run.

---

# Tile-level pathway visualization

Tile-level maps are produced with:

```bash
python -m xaiwsi.visualization.pathways
```

The CLI loads the saved Lightning config, checkpoint, embedding store, original WSI, and Hallmark definitions.

## New run config

If the training config contains the WSI roots:

```yaml
data:
  tcga_slides_root_dir: /path/to/tcga/slides
  cptac_slides_root_dir: /path/to/cptac/slides
```

then:

```bash
python -m xaiwsi.visualization.pathways \
    --config runs/pathways/xai-wsi/<run-id>/config.yaml \
    --checkpoint runs/pathways/xai-wsi/<run-id>/checkpoints/<checkpoint>.ckpt \
    --dataset tcga \
    --slide-id <slide-id> \
    --pathway HALLMARK_E2F_TARGETS \
    --map-type score \
    --map-type contribution \
    --map-type weight
```

## Backward compatibility with older runs

Older configs may not contain the WSI root directory. Pass it explicitly:

```bash
python -m xaiwsi.visualization.pathways \
    --config runs/pathways/xai-wsi/<old-run>/config.yaml \
    --checkpoint runs/pathways/xai-wsi/<old-run>/checkpoints/<checkpoint>.ckpt \
    --dataset tcga \
    --slides-root-dir /path/to/tcga/slides \
    --slide-id <slide-id> \
    --pathway HALLMARK_E2F_TARGETS \
    --map-type contribution
```

An explicit `--embedding-dir` can similarly override the saved embedding path.

---

## Visualization map types

### `score`

```text
tile_scores[i, pathway]
```

Local pathway-associated prediction for each tile.

### `weight`

For `TileMLPMean`:

```text
weight = 1 / N_tiles
```

For `TileMLPGatedMean`:

```text
gate_i = sigmoid(logit_i)
weight_i = gate_i / sum_j gate_j
```

### `gate`

Raw sigmoid relevance gate before normalization. Available only for gated models.

### `contribution`

```text
contribution[i, pathway]
    = weight[i] * tile_scores[i, pathway]
```

Therefore:

```text
sum_i contribution[i, pathway]
    = slide prediction[pathway]
```

up to floating-point error.

---

## Visualization outputs

For each slide:

```text
outputs/pathway-maps/<slide-id>/
├── metadata.json
├── predictions.csv
├── thumbnail.png
├── tile_outputs.npz
├── HALLMARK_E2F_TARGETS_score.png
├── HALLMARK_E2F_TARGETS_contribution.png
├── tile_weights.png
└── tile_gates.png
```

`tile_outputs.npz` stores the raw arrays needed for later analysis or re-rendering:

```text
coords
pred
tile_scores
tile_weights
tile_contributions
tile_gates        # gated models only
pathway_names
```

The embedding metadata are used to locate the original WSI as:

```text
slide_path = slides_root_dir / relative_wsi_path
```

The raw heatmap uses one raster cell per tile sampling position. Missing tissue is represented by `NaN`, while zero remains a valid pathway value.

---

# Metrics

## MSE

Absolute prediction error.

## Zero-baseline skill

Because gene z-score statistics are fitted on TCGA train, zero corresponds approximately to the TCGA-training pathway mean:

```text
skill = 1 - model_mse / zero_baseline_mse
```

## R²

Standard coefficient of determination, computed per pathway.

## Spearman correlation

The current primary model-selection metric is:

```text
median Spearman across the 50 Hallmarks
```

Spearman is especially useful for CPTAC external evaluation because the model may preserve pathway ordering while absolute calibration shifts between cohorts.

---

# Current assumptions and limitations

- One RNA sample is selected per patient.
- One WSI is currently selected per patient when multiple slides are available.
- CPTAC contains many patients with multiple slides, so patient-level multi-slide aggregation is an important next step.
- Hallmark targets use bulk RNA and therefore provide only patient-level supervision.
- Tile pathway scores are learned morphological surrogates and should not be interpreted as directly measured local RNA pathway activity.
- The gated model currently uses one relevance gate shared by all 50 Hallmarks.
- CPTAC is intended as an external test cohort and should not be used for routine hyperparameter tuning.

---

# Research roadmap

Each stage depends on the previous one producing a usable result. Stages 3 and 4 only make sense if the LUAD multimodal model from stage 2 works well.

## Stage 1: WSI → RNA programs (current)

- Frozen-VFM WSI baselines predicting the 50 Hallmarks (done).
- Add the **Tavernari et al. (2021)** LUAD signature as a target. Tavernari et al., *Nongenetic evolution drives lung adenocarcinoma spatial heterogeneity and progression*, Cancer Discovery 2021 ([PMID 33563664](https://pubmed.ncbi.nlm.nih.gov/33563664/)), derived a transcriptional signature that tracks the histological pattern progression (lepidic → acinar/papillary → solid/micropapillary). Because the signature is tied to growth pattern, it is a natural target for a morphology model, and tile maps can be checked against pattern annotations.
  - Loader and scoring CLI already exist (`xaiwsi.rna.pathways.tavernari`, `--gene-set-type tavernari`). The gene list must come from the paper's supplement as a `.gmt`/`.json` file under `data/external/tavernari/`, with the supplement version recorded.
- Patient-level aggregation over multiple WSIs (needed before trusting CPTAC numbers).

## Stage 2: RNA encoders and WSI + RNA fusion

- **BulkFormer** as a frozen bulk-RNA foundation model, mirroring how tile embeddings are used:
  - done: one cached embedding per RNA sample under `data/processed/<cohort>/rna-embeddings/bulkformer/<variant>/`, using BulkFormer's own normalization (see "BulkFormer RNA embeddings");
  - open: gene-level embeddings (for SAEs), and restricting aggregation to gene subsets.
- Unimodal baselines first (WSI-only, RNA-only), then fusion (early concat, late fusion, cross-attention between tiles and the RNA embedding).
- **Target leakage.** When RNA is a model input, targets computed from that same RNA (Hallmarks, Tavernari score) are no longer valid prediction targets. The task becomes reconstructing a function of the input. Multimodal experiments need targets independent of the RNA input, for example:
  - histological growth pattern / IASLC grade, stage, survival;
  - cross-modal alignment (contrastive WSI ↔ RNA), evaluated by retrieval.
- **Querying the Tavernari signature.** With a shared or aligned WSI–RNA space, the signature can be used as a query rather than a target: score it on RNA, then find which tiles, latent directions or fused features align with it, and test whether they correspond to the expected growth patterns.

## Stage 3: Other cancer types (conditional on stage 2)

- Download additional TCGA/CPTAC cohorts (RNA + diagnostic WSIs) with the same GDC/STAR pipeline.
- Requires generalizing code that is currently LUAD-specific: cohort names and paths in configs, the TCGA/CPTAC preprocessing entry points, and any hard-coded output dimension (50 Hallmarks).
- Cancer-specific signatures (like Tavernari for LUAD) will not transfer; Hallmarks remain the cross-cancer common target.

## Stage 4: Explainability with sparse autoencoders

- Train SAEs on frozen representations: VFM tile embeddings, BulkFormer embeddings, then the fused multimodal representation.
- Interpret SAE concepts by association with Hallmarks, the Tavernari signature and histological patterns.
- Spatial analysis of pathway- and pattern-associated concepts on the WSI.
- Compare with the built-in tile-level explanations (scores, gates, contributions) and with pathway-specific relevance weights / ABMIL.

## Other open items

- Linear vs nonlinear pathway decoders.
- Uniform mean vs sigmoid-gated mean vs ABMIL.
- Additional RNA/pathway targets and robustness analyses.
