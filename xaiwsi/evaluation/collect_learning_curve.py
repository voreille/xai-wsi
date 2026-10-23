"""Collect the tile-level learning-curve runs started by scripts/learning_curve.sh.

For each run directory `<run-root>/<prefix>-*`: training fraction and subset seed from the
saved config, best TCGA-val median Spearman from the checkpoint's ModelCheckpoint state,
and CPTAC test medians parsed from the test log.

    uv run python -m xaiwsi.evaluation.collect_learning_curve \\
        --run-root runs/pathways/xai-wsi --prefix lc-1023
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import yaml

from xaiwsi.training.run_config import data_args

# Separator: spaces or table borders, never the sign of the value.
TEST_METRIC = re.compile(r"(test/median_(?:spearman|r2))[\s│|]+([-+]?\d[\d.eE+-]*)")


def best_val_score(run_dir: Path) -> float | None:
    for ckpt in sorted((run_dir / "checkpoints").glob("epoch=*.ckpt")):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        for key, callback in state.get("callbacks", {}).items():
            if key.startswith("ModelCheckpoint") and callback.get("best_model_score") is not None:
                return float(callback["best_model_score"])
    return None


def test_metrics(log: Path) -> dict[str, float]:
    if not log.exists():
        return {}
    return {name: float(value) for name, value in TEST_METRIC.findall(log.read_text())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=Path("runs/pathways/xai-wsi"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/learning_curve/logs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/learning_curve/tile_runs.csv"))
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.run_root.glob(f"{args.prefix}-*")):
        data = data_args(yaml.safe_load((run_dir / "config.yaml").read_text()))
        test = test_metrics(args.log_dir / f"{run_dir.name}.test.log")
        rows.append(
            {
                "run": run_dir.name,
                "fraction": data.get("train_fraction", 1.0),
                "subset_seed": data.get("train_subset_seed", 0),
                "val_median_spearman": best_val_score(run_dir),
                "cptac_median_spearman": test.get("test/median_spearman"),
                "cptac_median_r2": test.get("test/median_r2"),
            }
        )
    if not rows:
        raise SystemExit(f"No runs matching {args.run_root}/{args.prefix}-*")

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary = results.groupby("fraction")[
        ["val_median_spearman", "cptac_median_spearman", "cptac_median_r2"]
    ].agg(["mean", "std", "count"])
    with pd.option_context("display.width", 200):
        print(results.to_string(index=False))
        print(summary.round(3).to_string())
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
