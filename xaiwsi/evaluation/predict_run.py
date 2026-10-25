"""Per-case val and test predictions of a trained tile-level run, without Lightning.

Rebuilds the run's cohorts, splits and targets from its saved ``config.yaml`` (same
``build_pathway_splits`` call as the data module), builds the predictor from
``model.predictor`` and loads the ``predictor.*`` weights of the checkpoint. The val rows
are what calibrates prediction intervals (``xaiwsi.evaluation.intervals``); the test rows
equal the run's ``test_predictions.parquet`` (checked when that file exists).

    uv run python -m xaiwsi.evaluation.predict_run \\
        --run-dir runs/pathways/xai-wsi/coh-1024-luad+lusc-percohort-s0 \\
        --test-cohorts-from configs/training/pathway_cohorts.yaml

``--test-cohorts-from`` replaces the run's test cohorts by those of another config (e.g. to
add CPTAC-LSCC to runs trained before it existed); as in training, test cohorts whose
``zscore_reference`` is not a training cohort are skipped. Writes
``<run-dir>/predictions.parquet`` (columns of ``xaiwsi.evaluation.predictions``).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.predictions import prediction_frame
from xaiwsi.models.pathway import PathwayPredictor
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


def build_predictor(config: dict) -> PathwayPredictor:
    spec = config["model"]["predictor"]
    module, _, name = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(module), name)(**spec.get("init_args", {}))


def load_predictor(config: dict, ckpt_path: Path, device: torch.device) -> PathwayPredictor:
    predictor = build_predictor(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k.removeprefix("predictor."): v for k, v in state.items() if k.startswith("predictor.")}
    predictor.load_state_dict(weights)
    return predictor.to(device).eval()


def best_checkpoint(run_dir: Path) -> Path:
    """The ModelCheckpoint best file (`epoch=*.ckpt`), not `last.ckpt`."""
    ckpts = sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
    if len(ckpts) != 1:
        raise SystemExit(f"{run_dir}: expected one epoch=*.ckpt, found {[c.name for c in ckpts]}.")
    return ckpts[0]


def run_splits(config: dict, test_specs: list[CohortSpec] | None = None) -> tuple[dict, list[str]]:
    """The run's {"train"|"val"|"test": [CohortSplit]} and pathway names."""
    data = data_args(config)
    if "train_cohorts" not in data:
        raise SystemExit("Only CohortPathwayDataModule run configs are supported.")
    train_specs = [CohortSpec(**spec) for spec in data["train_cohorts"]]
    if test_specs is None:
        test_specs = [CohortSpec(**spec) for spec in data["test_cohorts"]]
    if data.get("use_train_cohorts"):
        train_specs, test_specs = select_cohorts(train_specs, test_specs, data["use_train_cohorts"])
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    gene_sets = load_hallmarks(data["hallmark_gmt"])
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_specs],
        [c.resolve(processed_root, embedding) for c in test_specs],
        gene_sets,
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
        train_fraction=data.get("train_fraction", 1.0),
        train_subset_seed=data.get("train_subset_seed", 0),
        target_normalization=data.get("target_normalization", "per_cohort"),
    )
    return splits, list(gene_sets)


@torch.no_grad()
def predict_part(predictor: PathwayPredictor, part, names: list[str], device: torch.device) -> tuple:
    preds, targets = [], []
    for row in part.pairs.itertuples(index=False):
        embeddings, _, _ = part.store.load(str(row.slide_id))
        x = torch.from_numpy(np.asarray(embeddings)).float().to(device)[None]
        preds.append(predictor(x)["pred"][0].cpu().numpy())
        targets.append(part.scores.loc[str(row.sample_id), names].to_numpy(dtype=np.float32))
    return np.stack(targets), np.stack(preds)


def check_against_test_predictions(frame: pd.DataFrame, path: Path) -> None:
    """Test rows must match what the Lightning test stage wrote for the same cohorts."""
    ref = pd.read_parquet(path).set_index(["cohort", "case_id"]).sort_index()
    mine = frame[frame["split"] == "test"].set_index(["cohort", "case_id"]).sort_index()
    mine = mine.loc[mine.index.intersection(ref.index)]
    if mine.empty:
        return
    columns = [c for c in ref.columns if c.startswith(("true__", "pred__"))]
    diff = np.abs(mine[columns].to_numpy() - ref.loc[mine.index, columns].to_numpy()).max()
    if diff > 1e-4:
        raise RuntimeError(f"Test predictions differ from {path} (max abs diff {diff:.2e}).")
    print(f"  test rows match {path.name} ({len(mine)} cases, max abs diff {diff:.1e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--test-cohorts-from", type=Path, help="config whose data.test_cohorts to use")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    test_specs = None
    if args.test_cohorts_from:
        other = data_args(yaml.safe_load(args.test_cohorts_from.read_text()))
        test_specs = [CohortSpec(**spec) for spec in other["test_cohorts"]]

    for run_dir in args.run_dir:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        ckpt = best_checkpoint(run_dir)
        predictor = load_predictor(config, ckpt, device)
        splits, names = run_splits(config, test_specs)
        frames = []
        for split in args.splits:
            for part in splits[split]:
                y, y_pred = predict_part(predictor, part, names, device)
                frames.append(prediction_frame(part.pairs["case_id"], part.cohort, split, y, y_pred, names))
        frame = pd.concat(frames, ignore_index=True)
        print(f"{run_dir.name} ({ckpt.name}): {frame.groupby(['split', 'cohort']).size().to_dict()}")
        if (run_dir / "test_predictions.parquet").exists():
            check_against_test_predictions(frame, run_dir / "test_predictions.parquet")
        frame.to_parquet(run_dir / "predictions.parquet", index=False)


if __name__ == "__main__":
    main()
