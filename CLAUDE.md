# CLAUDE.md

Research code that predicts bulk-RNA Hallmark pathway scores (50 MSigDB Hallmarks) from
frozen pathology foundation-model tile embeddings (H-optimus-1, dim 1536) in lung
adenocarcinoma, and keeps tile-level interpretability. TCGA-LUAD is used for train/val
and CPTAC-LUAD is the external test cohort. `README.md` has the full pipeline,
commands, output formats and the staged research roadmap; read it before changing data
flow or CLIs.

## Research direction

Final goal: WSI-only inference that estimates the activity of the 50 Hallmark pathways,
as a continuous activity with its uncertainty (predictive distribution/interval, not a
high/low class), with tile-level interpretability. Tavernari et al. 2021 growth-pattern
gene sets were the motivation but are not released, so they are on hold. Because Hallmarks
are cross-cancer, more TCGA/CPTAC cohorts are an option against overfitting.
Exploration aim: mine from WSI/RNA pairs a description of morphology that is more objective
than pathologist labels. Pathologist labels (growth pattern, grade, ANORAK/DHMC
annotations) are references for interpretation and evaluation, not training targets. RNA is used only at
training time (targets, alignment, fusion or distillation). The project is exploratory: try
variants to learn what that predictor needs, but judge them by what a WSI-only model can
use at test time. For example, a test-time step that needs CPTAC RNA (such as centering the
RNA embeddings per cohort) is a diagnostic only. Slide-only steps, such as per-cohort
normalization of slide embeddings, are allowed at deployment.

Planned stages (details in README "Research roadmap"); later stages depend on earlier ones
working:
1. WSI → RNA programs: Hallmarks (done), Tavernari et al. 2021 LUAD growth-pattern
   signature, multi-slide patient aggregation.
2. BulkFormer as a frozen RNA encoder, then WSI + RNA fusion.
3. Other TCGA/CPTAC cancer types: either after stage 2 works, or earlier as extra
   training data against overfitting.
4. XAI with sparse autoencoders on tile, RNA and fused representations.

What this means for code written now:
- Keep encoders and heads swappable through config, like `PathwayPredictor`. A model
  should take its modalities as a batch dict, so RNA embeddings can be added without
  rewriting the training module.
- Precompute frozen-encoder embeddings once and load them from disk (tiles via
  `wsitools` stores; BulkFormer under `data/processed/<cohort>/rna-embeddings/bulkformer/`).
  Don't run foundation models inside the training loop.
- Don't hard-code LUAD, the two cohorts, or `output_dim=50` in library code. Put them in
  configs/arguments.
- Tavernari gene lists are supplied externally (`.gmt`/`.json`, from the paper's
  supplement). Never invent or hard-code the genes.

## Environment

- Experiments: quick checks (minutes, small probes, smoke runs) can run in claude-box.
  For bigger experiments (training sweeps, long GPU runs), write the code plus a
  `scripts/*.sh` launcher, and the user runs it on the host (W&B, DataLoader workers).
- Python >= 3.12. Set up with `uv sync`, run everything through `uv run ...`.
- `wsitools` (provides `wsi-tile`, `wsi-embed` and the `wsitools.storage` embedding
  stores) is a sibling repo installed editable from `../wsi-tools` via
  `[tool.uv.sources]`. That checkout must exist next to this repo. Changes to
  wsitools belong in that repo.
- On Linux, torch/torchvision come from the PyTorch `cu126` index (`[tool.uv.index]`),
  because the GPU servers' driver supports CUDA <= 12.9 and PyPI torch is built for
  CUDA 13. Don't remove this until the driver is upgraded. Check
  `torch.cuda.is_available()` after changing torch versions.
- There is no test suite (`tests/` is referenced in pyproject but does not exist).
- `data/`, `runs/`, `outputs/` are gitignored and host-specific. Configs and scripts use
  absolute paths under `/home/valentin/workspaces/xai-wsi` and NAS mounts
  (`/mnt/nas6`, `/mnt/nas7`) that may not be visible in every environment.
- `*.sh` is gitignored, so the files in `scripts/` are force-added. New scripts need
  `git add -f`.

## Layout

- `xaiwsi/preprocessing/{gdc,tcga,cptac}`: GDC STAR-count RNA, WSI metadata and clinical prep
  (TCGA survival from TCGA-CDR, CPTAC from the GDC API) -> `metadata/clinical.csv`.
  Each step runs as a module (`python -m xaiwsi.preprocessing.tcga.rna`, ...).
- `xaiwsi/rna/`: gene handling, normalization, pathway scoring (`rna/pathways`, CLI in
  `__main__.py`; methods `ssgsea` and `mean-z`; gene sets: GMT, hallmarks, tavernari).
- `xaiwsi/rna/encoders/bulkformer/`: frozen BulkFormer (vendored upstream model code, kept
  unchanged; our code is in `load.py`/`preprocess.py`/`embed.py`), configs in `configs/rna/`.
  Uses native `torch.sparse` instead of `torch_sparse`; don't add torch-scatter/sparse.
- Tile embedding stores: `wsi-embeddings/h-optimus-1/` (original fp32, LZF, tiny chunks)
  and `h-optimus-1-fp16/` (repacked with `wsi-repack`, used by alignment). Alignment
  training caches decoded slides in RAM (`data.cache_tiles`).
- `xaiwsi/data/pathway.py`: case pairing of RNA and slides, splits, embedding dataset.
- `xaiwsi/models/pathway.py`: `PathwayPredictor` subclasses (`TileLinearMean`,
  `TileMLPMean`, `TileMLPGatedMean`, `MeanMLP`).
- `xaiwsi/metrics/pathways.py`: MSE, zero-baseline skill, per-pathway R² and Spearman.
- `xaiwsi/training/`: LightningCLI entry point (`cli.py`), `PathwayTraining` module,
  `PathwayDataModule`, `RunSaveConfigCallback`.
- `xaiwsi/training/baseline.py`: legacy pre-Lightning script with hardcoded paths. It
  duplicates logic now in `data/`, `models/`, `metrics/`. Don't extend it.
- `xaiwsi/visualization/pathways.py`: tile-level heatmaps from a run config and checkpoint.
- `configs/training/*.yaml`: LightningCLI configs (`pathway_baseline_csv.yaml` is the
  CSVLogger variant, for runs without W&B).

## Commands

```bash
# train (TCGA train/val)
CUDA_VISIBLE_DEVICES=0 uv run python -m xaiwsi.training.cli fit \
    --config configs/training/pathway_baseline.yaml

# external test on CPTAC with the saved run config
uv run python -m xaiwsi.training.cli test \
    --config runs/pathways/xai-wsi/<run-id>/config.yaml \
    --ckpt_path runs/pathways/xai-wsi/<run-id>/checkpoints/<best>.ckpt

# tile-level maps
uv run python -m xaiwsi.visualization.pathways --config ... --checkpoint ... \
    --dataset tcga --slide-id <id> --pathway HALLMARK_E2F_TARGETS --map-type contribution
```

## Design rules

- Scientific code (data, models, metrics, rna) stays plain Python/PyTorch with no
  Lightning or W&B imports. Lightning only orchestrates training, validation,
  checkpointing, logging and the CLI. `PathwayTraining` is a thin wrapper.
- Models, loggers and callbacks are swapped via `class_path`/`init_args` in YAML. Don't
  add a model-ID registry or builder.
- Predictors take `B x N x D` tile embeddings and return a dict with `pred`, plus
  `tile_scores` (`None` for `MeanMLP`). Only the gated model also returns `tile_gates` and
  `tile_weights`; the mean models imply uniform weights `1/N`. The visualization depends
  on these keys. Invariant: `sum_i tile_weights[i] * tile_scores[i, k] == pred[k]`.
- `RunSaveConfigCallback` writes the resolved `config.yaml` to
  `<save_dir>/<logger name>/<version>/`, next to `checkpoints/`. Downstream tools
  rebuild model and data paths from it, so keep it complete. With W&B, it pins the run
  id so `test` reuses the run, is only written once (non-fit stages keep the fit-time
  file), and stores `run_dir`/`config_path`/`host` in the W&B run config. Older run configs may lack
  `*_slides_root_dir`, so visualization accepts `--slides-root-dir` / `--embedding-dir`
  overrides.
- Model selection metric: `val/median_spearman` (mode max).
- Result tables in `docs/progress.md` are generated by scripts from run outputs, not
  typed by hand (see the conventions at the top of that file).
- `batch_size: 1`: slides have variable tile counts and there is no padding/collation yet.

## Scientific constraints (don't break these)

- Mean-z targets: TPM, collapse duplicate symbols in linear space, `log2(TPM+1)`,
  gene-wise z-score, mean over pathway genes. Z-score statistics are fit on the TCGA
  **training split only**, after the split, and reused for val and CPTAC. No leakage.
- CPTAC is external test only. Don't tune hyperparameters or architecture on it.
- If RNA is a model input, targets derived from that same RNA (Hallmarks, Tavernari score)
  are leakage. Multimodal targets must be independent of the RNA input (growth
  pattern/grade, stage, survival) or the task must be alignment/retrieval. Signatures can
  still be used as queries/probes for interpretation.
- WSI/RNA matching is at case level. TCGA slide (`01Z`) and RNA (`01A`) sample codes differ.
- Current limitation: when a patient has several slides or RNA samples, one of each is
  picked deterministically. Patient-level multi-slide aggregation is the next planned
  step (it matters most for CPTAC).
- Tile scores are learned morphological surrogates, not measured local RNA activity.
