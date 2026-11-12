"""Do SAE reconstruction scores rank per-slide Hallmark prediction error? (README 2f, uncertainty)

For each probe prediction file (``cohort_probe`` ``predictions/*.parquet``) and each split/cohort
in it, takes the per-case error (mean absolute error over the Hallmarks) and asks whether a
per-slide score of the paired slide ranks it:

- ``mean_rel_error``, ``tile_fvu``, ``n_tiles``: from the SAE's bag table
  (``<cohort dir>/concepts/<sae>/slides.parquet``, ``xaiwsi.concepts.bags``);
- ``residual_norm``: norm of the mean residual (``slides_residual.parquet``,
  ``xaiwsi.concepts.residual``), when present.

Reports the Spearman of score and error with a case bootstrap interval, and a risk-coverage
summary: the mean error of the cases kept when the highest-scoring ones are rejected, against
the error over all cases. Slides are paired to cases as in ``cohort_probe`` (same config split).

    uv run python -m xaiwsi.evaluation.ood_score --config configs/training/pathway_cohorts.yaml \\
        --condition tcga-luad --sae tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --predictions sae-mean-clip=outputs/.../predictions/<file>.parquet \\
        --output outputs/concept_residual/ood_luad

Writes ``<output>.csv`` and ``<output>.md`` (with the generating command).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from xaiwsi.data.cohorts import CohortSpec, build_pathway_splits, select_cohorts
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.run_config import data_args


def case_slides(config: Path, condition: str) -> tuple[dict[tuple[str, str], str], dict[str, Path]]:
    """(cohort, case_id) -> slide_id of the probe's pairing, and cohort -> cohort dir."""
    data = data_args(yaml.safe_load(config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    train_specs = [CohortSpec(**s) for s in data["train_cohorts"]]
    test_specs = [CohortSpec(**s) for s in data["test_cohorts"]]
    train_used, test_used = select_cohorts(train_specs, test_specs, condition.split(","))
    splits = build_pathway_splits(
        [c.resolve(processed_root, embedding) for c in train_used],
        [c.resolve(processed_root, embedding) for c in test_used],
        load_hallmarks(data["hallmark_gmt"]),
        val_fraction=data.get("val_fraction", 0.2),
        split_seed=data.get("split_seed", 42),
    )
    pairs, dirs = {}, {}
    for parts in splits.values():
        for part in parts:
            for case, slide in zip(part.pairs["case_id"].astype(str), part.pairs["slide_id"].astype(str)):
                pairs[part.cohort, case] = slide
    for spec in train_used + test_used:
        dirs[spec.name] = spec.resolve(processed_root, embedding).embedding_dir.parent.parent
    return pairs, dirs


def slide_scores(cohort_dir: Path, sae: str) -> pd.DataFrame:
    bag_dir = cohort_dir / "concepts" / sae
    scores = pd.read_parquet(bag_dir / "slides.parquet")
    scores.index = scores.index.astype(str)
    residual_path = bag_dir / "slides_residual.parquet"
    if residual_path.exists():
        residual = pd.read_parquet(residual_path)
        residual.index = residual.index.astype(str)
        scores["residual_norm"] = np.linalg.norm(residual.loc[scores.index].to_numpy(), axis=1)
    return scores


def risk_at(error: np.ndarray, score: np.ndarray, coverage: float) -> float:
    keep = np.argsort(score, kind="stable")[: max(1, round(coverage * len(score)))]
    return float(error[keep].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", required=True, help="training cohorts of the probe, comma-separated")
    parser.add_argument("--sae", required=True, help="bag folder name under <cohort dir>/concepts/")
    parser.add_argument("--predictions", nargs="+", required=True, help="<label>=<predictions parquet>")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.9, 0.8, 0.7, 0.5])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs, dirs = case_slides(args.config, args.condition)
    rng = np.random.default_rng(args.seed)
    rows = []
    for item in args.predictions:
        label, _, path = item.partition("=")
        frame = pd.read_parquet(path)
        pathways = [c.removeprefix("true__") for c in frame.columns if c.startswith("true__HALLMARK_")]
        error = (
            frame[[f"pred__{p}" for p in pathways]].to_numpy() - frame[[f"true__{p}" for p in pathways]].to_numpy()
        )
        frame["error"] = np.abs(error).mean(axis=1)
        for (split, cohort), group in frame.groupby(["split", "cohort"], sort=False):
            scores = slide_scores(dirs[cohort], args.sae)
            slides = [pairs[cohort, str(case)] for case in group["case_id"]]
            err = group["error"].to_numpy()
            for name in scores.columns:
                score = scores.loc[slides, name].to_numpy(np.float64)
                rho = spearmanr(score, err).statistic
                boot = []
                for _ in range(args.bootstrap):
                    idx = rng.integers(0, len(err), len(err))
                    boot.append(spearmanr(score[idx], err[idx]).statistic)
                low, high = np.nanpercentile(boot, [2.5, 97.5])
                if not -1 <= rho <= 1:
                    raise ValueError(f"Spearman out of range: {label} {cohort} {name} {rho}")
                row = {
                    "predictions": label,
                    "split": split,
                    "cohort": cohort,
                    "score": name,
                    "n": len(err),
                    "spearman": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "error_all": err.mean(),
                }
                for coverage in args.coverages:
                    row[f"error_keep{coverage:g}"] = risk_at(err, score, coverage)
                rows.append(row)
                print(f"{label} {split} {cohort} {name}: rho {rho:.3f} [{low:.3f}, {high:.3f}]")

    results = pd.DataFrame(rows)
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    command = "uv run python -m xaiwsi.evaluation.ood_score " + shlex.join(sys.argv[1:])
    args.output.with_suffix(".md").write_text(
        "Per-slide SAE scores vs per-case Hallmark error (mean |pred - true| over pathways); Spearman with "
        f"case-bootstrap 95% interval; error_keep<c> = mean error of the fraction c of cases with the lowest "
        f"score. Generated by:\n\n```bash\n{command}\n```\n\n{markdown_table(results)}\n"
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
