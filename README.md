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

**Final goal:** at inference, use the WSI only and estimate the activity of the 50 MSigDB Hallmark pathways, with tile- and concept-level interpretability. The output is a continuous activity with its uncertainty (a predictive distribution or interval), because the estimate from morphology won't be certain. It is not a high/low class. The original motivation was the growth-pattern-linked pathways of Tavernari et al. (2021), but their gene sets are not released. Hallmarks are therefore the target, and because they are shared across cancer types, more TCGA/CPTAC data can be added to fight overfitting. RNA is a training-time signal: it supplies the targets and can shape the slide representation (alignment, fusion, distillation), but a deployed model must not need RNA. The work is still exploratory: multimodal (WSI + bulk RNA) models, alignment and other cancer types are tested to learn what this WSI-only predictor needs. See [Research roadmap](#research-roadmap).

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

Manifests come from the GDC `files` endpoint (`return_type: manifest`) with open-access
`data_type = Gene Expression Quantification` and `analysis.workflow_type = STAR - Counts`, plus:

| Cohort | Extra filters | Files |
|---|---|---|
| TCGA-LUAD | `cases.project.project_id = TCGA-LUAD` | 601 |
| TCGA-LUSC | `cases.project.project_id = TCGA-LUSC` | 562 |
| CPTAC-LUAD | `CPTAC-3`, `cases.primary_site = Bronchus and lung`, `cases.disease_type = Adenomas and Adenocarcinomas` | 446 |
| CPTAC-LSCC | `CPTAC-3`, `cases.primary_site = Bronchus and lung`, `cases.disease_type = Squamous Cell Neoplasms` | 206 |

Counts as of 2026-10-23. Each manifest is stored as `data/raw/<cohort>-rna/manifest.txt`.

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

### TCGA-LUSC (extra training cohort)

The TCGA modules take the cohort through their arguments:

```bash
P=data/processed/tcga-lusc
python -m xaiwsi.preprocessing.tcga.rna_metadata \
    --manifest data/raw/tcga-lusc-rna/manifest.txt --raw-root data/raw/tcga-lusc-rna \
    --output $P/metadata/rna_samples.csv
python -m xaiwsi.preprocessing.tcga.rna \
    --metadata $P/metadata/rna_samples.csv --raw-root data/raw/tcga-lusc-rna --output-dir $P/rna
python -m xaiwsi.preprocessing.tcga.wsi_metadata \
    --slides-root /mnt/nas7/data/TCGA_Lung_svs/LUSC --output $P/metadata/wsi_slides.csv
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

### Repacked fp16 stores

`wsi-repack` (wsitools) rewrites a store with fp16 features, no compression and
1024-row chunks. It writes atomically per slide, is resumable, and verifies each slide
against the source. Reads take about 3× less time with half the disk space (TCGA 55.0 →
28.3 GB, CPTAC 45.5 → 23.7 GB). The alignment config uses these stores; the original
stores are kept for older run configs.

```bash
for c in tcga-luad cptac-luad; do
  E=data/processed/$c/wsi-embeddings
  uv run python -m wsitools.storage.repack --src $E/h-optimus-1 \
      --dst $E/h-optimus-1-fp16 --dtype float16 --chunk-rows 1024 --workers 16
done
```


For a whole cohort, `scripts/tile_embed_cohort.sh` runs tiling, H-optimus-1 embedding
and the fp16 repack into `data/processed/<cohort>/` (run it on the host):

```bash
COHORT=tcga-lusc SLIDES=/mnt/nas7/data/TCGA_Lung_svs/LUSC GPU=0 bash scripts/tile_embed_cohort.sh
```

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

# WSI ↔ RNA alignment training (stage 2c)

The slide encoder (frozen H-optimus-1 tiles, trained with project-then-pool) and an MLP
on frozen BulkFormer embeddings are aligned with a symmetric InfoNCE (CLIP) loss. It
uses the same TCGA pairs and split as the Hallmark models, with CPTAC as the external
test.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m xaiwsi.training.alignment_cli fit \
    --config configs/training/alignment_baseline.yaml

uv run python -m xaiwsi.training.alignment_cli test \
    --config runs/alignment/xai-wsi/<run-id>/config.yaml \
    --ckpt_path runs/alignment/xai-wsi/<run-id>/checkpoints/<best>.ckpt
```

- **Batches** are lists of whole-slide bags (`collate_bags`): no padding and no tile
  sampling. Each bag is pooled separately, and the B slide embeddings are stacked, so B
  is the number of in-batch negatives. Training uses `drop_last` so B stays fixed.
- **Models** (`xaiwsi/models/alignment.py`):
  - `GatedSlideEncoder` (sigmoid gates normalized to sum to one, as in
    `TileMLPGatedMean`) and `MeanSlideEncoder` (uniform weights), wrapped in
    `ContrastiveAligner` with an `MLPProjector` for RNA;
  - both encoders return `tile_embeddings` `h_i` in the shared space, with
    `embedding = Σ_i tile_weights[i] · h_i`.
- **RNA features** are standardized with TCGA-train statistics.
  `data.rna_center_per_cohort: true` centers CPTAC on its own mean instead. This is
  label-free but transductive, and targets the cohort offset seen in stage 2b.
- **Metrics** (`xaiwsi/metrics/retrieval.py`), computed over the whole split:
  - `wsi2rna_*` and `rna2wsi_*`: recall@1/5/10, median rank and MRR, plus their averages
    `mean_*` and the chance levels `chance_r{k}`;
  - model selection uses `val/mean_mrr` (max).
- **Data loading:** with `data.cache_tiles: true` (the default), each split's slides are
  decoded once into RAM as fp16 by `cache_workers` processes. That takes about 45 s for
  TCGA-train and about 26 GB of RAM for TCGA. After that, epochs are GPU-bound (4 epochs
  in about 80 s in total).
  - The original h5 stores are LZF-compressed in 64×96 auto-chunks (1.00×
    compression), so every read decompresses on the CPU. Under load (several runs with
    16 workers each, plus other users) that took 1–12 s per slide.
  - Without the cache (`cache_tiles: false`), DataLoader workers pass about 95 MB bags
    through `/dev/shm`. In containers with a small `/dev/shm` (Docker's default is
    64 MB), loading stalls.

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

## Stage 1: WSI → RNA programs

- Frozen-VFM WSI baselines predicting the 50 Hallmarks (done).
- Add the **Tavernari et al. (2021)** LUAD signature as a target. Tavernari et al., *Nongenetic evolution drives lung adenocarcinoma spatial heterogeneity and progression*, Cancer Discovery 2021 ([PMID 33563664](https://pubmed.ncbi.nlm.nih.gov/33563664/)), derived a transcriptional signature that tracks the histological pattern progression (lepidic → acinar/papillary → solid/micropapillary). Because the signature is tied to growth pattern, it is a natural target for a morphology model, and tile maps can be checked against pattern annotations.
  - Loader and scoring CLI already exist (`xaiwsi.rna.pathways.tavernari`, `--gene-set-type tavernari`). The gene list must come from the paper's supplement as a `.gmt`/`.json` file under `data/external/tavernari/`, with the supplement version recorded.
  - **Status (2026-10-23): on hold.** The marker genes of the lepidic-to-solid (L2S) signature are not released. Checked: the supplement on AACR figshare (Tables S1–S9; S8 lists only GO enrichment of the solid/lepidic markers), Zenodo 10.5281/zenodo.3941896 (CHUV molecular data only), and no code repository was found. Available instead: pattern-wise differential expression in TCGA (Table S3, 1,337 genes, pairwise log-FCs L/P/A/S) and CHUV (Table S6), and the 216-gene heatmap of Fig. 2c (Table S7). Approximate lepidic/solid sets are derived from S3/S6 with `scripts/build_tavernari_approx_genesets.py` (raw tables and checksums in `data/external/tavernari/supplement/`). `python -m xaiwsi.evaluation.geneset_vs_hallmarks` scores them with train-only statistics and measures how much the Hallmarks explain. On the RNA side, the 50 Hallmarks explain 75–94% of the approximate L2S axis on TCGA-val (see `docs/progress.md`). Hallmarks remain the target.
- Patient-level aggregation over multiple WSIs (needed before trusting CPTAC numbers).

## Stage 2: RNA encoders and WSI + RNA fusion (current)

Results are tracked in [`docs/progress.md`](docs/progress.md).

**Target leakage.** When RNA is a model input, targets computed from that same RNA
(Hallmarks, Tavernari score) are no longer valid prediction targets: the task becomes
reconstructing a function of the input. RNA can be an input only for targets that don't
come from it (survival, stage, growth pattern) or for alignment/retrieval. Hallmarks
stay a **WSI-only** target throughout.

**Splits.** Every stage-2 experiment uses the same fixed TCGA train/val split
(`split_cases`), and CPTAC is used only for external testing. Alignment pretraining sees
only TCGA-train patients, so later probes on TCGA-val are not transductive. Report
bootstrap 95% CIs over patients (CPTAC has only 54 OS events).

### 2a. Inputs (done)
- BulkFormer-147M sample embeddings (frozen) for both cohorts, see "BulkFormer RNA
  embeddings". Use the 640 encoder dims and drop the `stat_*` columns.
- Clinical / survival tables (`metadata/clinical.csv`): TCGA endpoints from TCGA-CDR,
  CPTAC from GDC.
- Open: gene-level BulkFormer embeddings (for SAEs), and aggregation over gene subsets.

### 2b. RNA sanity checks (before any fusion)
- Linear probes from the BulkFormer embedding to Hallmarks. This checks how much
  information the embedding carries; it is not a prediction task.
- Compare against **PCA of log TPM** at the same dimension. If BulkFormer doesn't beat
  PCA, keep PCA as the RNA encoder or as a second arm.
- Measure the TCGA → CPTAC shift in RNA space (cohort classifier AUC, UMAP).

### 2c. Slide encoder + WSI ↔ RNA alignment
- **Slide encoder, project then pool:**
  - each frozen H-optimus-1 tile is mapped into the shared space, `h_i = proj(tile_i)`;
  - gated-attention pooling (like `TileMLPGatedMean`, with no BatchNorm) then gives
    `slide = Σ_i w_i h_i`;
  - the attention weights give tile-level maps.
  - Projecting before pooling puts every tile token in the same space as the slide and
    RNA embeddings, which the multimodal SAE in stage 4 needs.
- **RNA side:** frozen BulkFormer embedding, then an MLP projection. Later the same
  project-then-pool is applied to gene tokens: `g_j = proj(gene_j)`, `rna = mean_j g_j`.
- **Loss:** symmetric InfoNCE with a learnable temperature over B patients per step.
- **Batching:** each batch is a list of whole-slide bags (all tiles, no padding or tile
  sampling). Each bag is pooled separately and the B slide embeddings are stacked. Slides
  hold about 10k–60k tiles; B=64 fits in an H100's memory, using bf16 and an early
  1536→256/512 projection if needed. Random tile subsets are an optional later experiment
  (augmentation), not a requirement.
- **Metrics:** WSI → RNA and RNA → WSI retrieval (recall@1/5/10, median rank) on
  TCGA-val, then CPTAC once.
- **Baselines:** chance level, mean-pooled tiles, and PCA as the RNA encoder.
- **Later:** patient-level bags (concatenate one patient's slides), which matters for
  CPTAC.

### 2d. Downstream evaluation of the frozen encoders
- **Hallmarks from WSI** (no RNA input):
  - probe (ridge / small MLP) the frozen aligned slide embedding to the 50 Hallmarks;
  - compare with the supervised stage-1 models on the same split, the same train-only
    z-score statistics and the same metrics (median Spearman, R²);
  - this tests whether alignment transfers RNA-program information into the slide
    encoder;
  - optional: a multitask variant, alignment loss plus an auxiliary Hallmark head on the
    WSI branch. It stays leakage-free because predictions use only the WSI.
- **Hallmarks via RNA structure:** the 50 Hallmarks are low-rank. On TCGA-LUAD, PC1
  explains 48% of the variance, 5 PCs reach 80% and the participation ratio is 3.7. So
  the bottleneck is likely morphological visibility, not the number of targets. Compare:
  - **low-rank heads without BulkFormer:** predict the top-k Hallmark PCs (k = 5–10, fit
    on TCGA-train) and reconstruct, or use reduced-rank regression;
  - **frozen-decoder head:** fit g, BulkFormer → Hallmarks, on RNA and freeze it; the
    slide encoder outputs z in BulkFormer space and the loss is on g(z). Plain MSE to the
    full BulkFormer embedding is not recommended, because most of it is not visible in
    morphology;
  - **alignment then probe** (above).
  - The BulkFormer → Hallmark probe from 2b is the ceiling of any BulkFormer route.
- **Overall survival:**
  - Cox models on fixed vectors (full batches, no tile batching), with three arms: WSI
    only (slide embedding), RNA only (BulkFormer), and fusion (concatenation, late
    fusion);
  - clinical covariates (age, stage) as a reference arm, and with each arm to show added
    value;
  - an end-to-end reference: a discrete-time survival loss from tiles at batch size 1 (as
    in MCAT/PORPOISE);
  - metrics: C-index on TCGA-val, CPTAC once, and a stratified log-rank test (KM split
    at the TCGA-train median risk);
  - PFI and DSS (TCGA only) as secondary endpoints.
- **Querying the Tavernari signature:**
  - score it on RNA, then find the tiles, latent directions or fused features that align
    with it in the shared space;
  - test whether they correspond to the expected growth patterns.

### 2e. Fusion models (if 2c/2d show complementary signal)
- Cross-attention between tiles and the RNA embedding, with survival or other
  non-RNA-derived targets only.

## Stage 3: Other cancer types (conditional on stage 2)

Update 2026-10-23: more TCGA/CPTAC cohorts are also a lever against the overfitting of the LUAD models (about 470 TCGA training/validation cases), since Hallmarks are a cross-cancer target. They can be pulled in earlier than stage 2 completion if regularization alone does not suffice.


- Download additional TCGA/CPTAC cohorts (RNA + diagnostic WSIs) with the same GDC/STAR pipeline.
- Requires generalizing code that is currently LUAD-specific: cohort names and paths in configs, the TCGA/CPTAC preprocessing entry points, and any hard-coded output dimension (50 Hallmarks).
- Cancer-specific signatures (like Tavernari for LUAD) will not transfer; Hallmarks remain the cross-cancer common target.

## Stage 4: Explainability with (multimodal) sparse autoencoders

**Data scale decides the level.** At patient level there are only about 470 TCGA and 220
CPTAC patients, too few for an SAE with a dictionary larger than the embedding dimension.
At token level there is plenty of data: about 8M TCGA tiles and about 10M BulkFormer gene
tokens (529 samples × 20,010 genes). SAEs are trained on tokens, and patient-level
analysis comes afterwards.

1. **Unimodal tile SAE** on H-optimus-1 tile embeddings, the baseline:
   - use a TopK SAE (Gao et al. 2024) for stable sparsity;
   - interpret features through their top tiles, and through patient-level correlations
     of each feature's active-tile fraction with Hallmarks, BulkFormer embeddings,
     survival and the Tavernari score.
2. **Shared-space SAE** on projected tile tokens plus projected gene tokens, using the 2c
   projection heads:
   - each feature is read through its top tiles (morphology) and top genes (biology);
   - **modality gap:** contrastive spaces put the two modalities in separate regions
     (Liang et al. 2022), so a naive SAE learns "tile vs gene" features. Center each
     modality separately, sample them equally, and report which features fire for both;
   - shared features are identified from paired patients: a feature's tile and gene
     activations should correlate across patients.
3. **Patient-level crosscoder** on the pair (slide embedding, RNA embedding):
   - it reads both modalities and reconstructs both, with per-feature decoder norms per
     modality;
   - norm in both modalities means a shared concept; norm in only one means
     modality-exclusive;
   - this answers which transcriptomic programs are visible in morphology;
   - small n means a small dictionary, so this is analysis rather than large-scale
     training.

**Spatial maps.** With project-then-pool, `h_i ≈ Σ_k a_ik d_k` gives
`slide ≈ Σ_k (Σ_i w_i a_ik) d_k`. A slide-level concept activation is the
attention-weighted sum of tile activations, exact up to reconstruction error. This extends
the `pred = Σ_i w_i · tile_score_i` invariant to concepts.

**Evaluation:**
- association with Hallmarks, Tavernari and histological patterns;
- spatial coherence of active tiles;
- **faithfulness:** ablating a feature in the shared space should move the Hallmark probe
  or the survival prediction in the expected direction;
- pathologist review of the top tiles for a few features;
- comparison with built-in tile explanations (scores, gates, contributions).

**Prerequisites:** BulkFormer gene-level embeddings (about 13.5 GB fp16 for TCGA) and the
2c projection heads.

## Other open items

- Linear vs nonlinear pathway decoders.
- Uniform mean vs sigmoid-gated mean vs ABMIL.
- Additional RNA/pathway targets and robustness analyses.
