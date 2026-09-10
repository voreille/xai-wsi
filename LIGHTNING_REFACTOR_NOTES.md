# Lightning refactor

The scientific pieces remain plain Python/PyTorch; Lightning only orchestrates
training, validation, checkpointing, logging and the CLI.

```text
xaiwsi/
├── data/pathway.py
├── metrics/pathways.py
├── models/pathway.py
└── training/
    ├── cli.py
    ├── pathway_datamodule.py
    └── pathway_module.py

configs/training/
├── pathway_baseline.yaml
└── pathway_baseline_csv.yaml
```

## Fit

```bash
uv add wandb "jsonargparse[signatures]"
CUDA_VISIBLE_DEVICES=0 python -m xaiwsi.training.cli fit \
    --config configs/training/pathway_baseline.yaml
```

The checkpoint callback monitors `val/median_spearman`, saves the best model and
`last.ckpt`, and LightningCLI saves the resolved `config.yaml` with the run.

## Test

```bash
python -m xaiwsi.training.cli test \
    --config <run-dir>/config.yaml \
    --ckpt_path <run-dir>/checkpoints/<best>.ckpt
```

## Swap predictor

For the linear additive baseline:

```yaml
model:
  predictor:
    class_path: xaiwsi.models.pathway.TileLinearMean
    init_args:
      input_dim: 1536
      output_dim: 50
```

For pooling before the MLP:

```yaml
model:
  predictor:
    class_path: xaiwsi.models.pathway.MeanMLP
    init_args:
      input_dim: 1536
      hidden_dim: 128
      output_dim: 50
      dropout: 0.0
```

No model-ID registry or builder is required.

## Swap logger

The model contains no W&B-specific code. Replace only `trainer.logger` in the
YAML. `pathway_baseline_csv.yaml` shows the CSVLogger version.

## Current limitation

The data pairing still deterministically selects one slide and one RNA sample
per patient when multiples are present, matching the initial baseline. The next
step should be explicit patient-level aggregation across multiple slides.
