"""Paired bootstrap over cases: is condition B better than condition A on one cohort?

Inputs are per-case prediction files (``cohort_probe`` writes them under
``<output-dir>/predictions/``): one row per case with ``case_id``, ``cohort``, ``split``,
``true__<pathway>`` and ``pred__<pathway>`` columns. Both conditions are scored on the
same resampled cases, so the confidence interval is for the difference, which is much
tighter than comparing two separate intervals. Targets must be identical in A and B (same
z-score reference), otherwise the comparison is refused.

    uv run python -m xaiwsi.evaluation.paired_bootstrap \\
        --baseline outputs/cohort_probe_brca/predictions/tcga-luad,tcga-lusc__per_cohort.parquet \\
        --candidates outputs/cohort_probe_brca/predictions/tcga-luad,tcga-lusc,tcga-brca__per_cohort.parquet \\
        --cohort cptac-luad --split test \\
        --output outputs/cohort_probe_brca/bootstrap_cptac-luad

A condition can also be a quoted glob over the runs (seeds) of a tile-level model, whose
test stage writes ``<run-dir>/test_predictions.parquet``; its metric is then the mean over
runs, on the same resampled cases:

    uv run python -m xaiwsi.evaluation.paired_bootstrap \\
        --baseline 'runs/pathways/xai-wsi/coh-1024-luad+lusc-percohort-s*/test_predictions.parquet' \\
        --candidates 'runs/pathways/xai-wsi/coh-1024-luad+lusc+brca-percohort-s*/test_predictions.parquet' \\
        --cohort cptac-luad --split test --output outputs/cohort_runs/bootstrap/vs_luadlusc_cptac-luad

Writes ``<output>.csv`` and ``<output>.md`` (Markdown table with the generating command).
For each metric (median over pathways of the per-pathway Spearman or R²): A, B, B − A,
the 95% percentile interval of B − A and the fraction of resamples with B − A ≤ 0.
"""

from __future__ import annotations

import argparse
import glob
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.metrics.pathways import compute_pathway_metrics


def load_predictions(path: Path, cohort: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(targets, predictions), both case_id x pathway, sorted by case_id."""
    frame = pd.read_parquet(path)
    frame = frame[(frame["cohort"] == cohort) & (frame["split"] == split)]
    if frame.empty:
        raise SystemExit(f"{path}: no rows for cohort {cohort!r}, split {split!r}.")
    if frame["case_id"].duplicated().any():
        raise SystemExit(f"{path}: duplicated case_id in {cohort}/{split}.")
    frame = frame.set_index("case_id").sort_index()
    true = frame.filter(like="true__").rename(columns=lambda c: c.removeprefix("true__"))
    pred = frame.filter(like="pred__").rename(columns=lambda c: c.removeprefix("pred__"))
    if list(true.columns) != list(pred.columns):
        raise SystemExit(f"{path}: true__/pred__ pathway columns differ.")
    return true, pred


def median_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Median over pathways of Spearman and R², for a batch of resamples.

    y_true, y_pred: (B, N, P). Returns (B, 2): [median_spearman, median_r2]. Spearman is
    the Pearson correlation of average ranks (as scipy's spearmanr). A pathway with a
    constant column has no Spearman/R²; compute_pathway_metrics would drop it from the
    median, which would silently change what the median is over, so this raises instead.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rt = rankdata(y_true, axis=1)
        rp = rankdata(y_pred, axis=1)
        rt -= rt.mean(axis=1, keepdims=True)
        rp -= rp.mean(axis=1, keepdims=True)
        rho = (rt * rp).sum(axis=1) / np.sqrt((rt**2).sum(axis=1) * (rp**2).sum(axis=1))
        ss_res = ((y_true - y_pred) ** 2).sum(axis=1)
        ss_tot = ((y_true - y_true.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    undefined = np.isnan(rho) | np.isnan(r2)
    if undefined.any():
        raise ValueError(
            f"Spearman/R² undefined (constant column or NaN) for {int(undefined.sum())} "
            f"pathway value(s) over {len(rho)} resample(s)."
        )
    return np.stack([np.median(rho, axis=1), np.median(r2, axis=1)], axis=1)


METRICS = ["median_spearman", "median_r2"]


def check_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str], path: Path) -> None:
    """The vectorized metrics must match the reference implementation."""
    mine = median_metrics(y_true[None], y_pred[None])[0]
    ref, _ = compute_pathway_metrics(y_true, y_pred, names)
    if not np.allclose(mine, [ref["median_spearman"], ref["median_r2"]]):
        raise RuntimeError(f"{path}: bootstrap metrics {mine} differ from compute_pathway_metrics {ref}.")


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    batch: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimates of A and B, shape (2,), and bootstrap differences B − A, (n_boot, 2).

    pred_a, pred_b: (S, N, P), the predictions of S runs (seeds) of a condition. A
    condition's metric is the mean over its runs of the median metrics, on the same
    resampled cases for all runs of both conditions. With S = 1 this is the plain paired
    bootstrap."""
    n = len(y_true)

    def score(idx: np.ndarray | None, preds: np.ndarray) -> np.ndarray:
        t = y_true[None] if idx is None else y_true[idx]
        return np.mean(
            [median_metrics(t, p[None] if idx is None else p[idx]) for p in preds], axis=0
        )

    point_a, point_b = score(None, pred_a)[0], score(None, pred_b)[0]
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, n_boot, batch):
        idx = rng.integers(0, n, size=(min(batch, n_boot - start), n))
        deltas.append(score(idx, pred_b) - score(idx, pred_a))
    return point_a, point_b, np.concatenate(deltas)


def expand(pattern: str) -> list[Path]:
    """A prediction file, or a glob over the runs (seeds) of one condition."""
    if not glob.has_magic(pattern):
        return [Path(pattern)]
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No prediction files match {pattern!r}.")
    return paths


def label(pattern: str) -> str:
    """Condition name: the file stem, or the run directory for `test_predictions.parquet`."""
    path = Path(pattern)
    return path.parent.name if path.name == "test_predictions.parquet" else path.stem


def load_condition(
    pattern: str, cohort: str, split: str
) -> tuple[pd.DataFrame, np.ndarray, list[Path]]:
    """(targets, predictions of shape (S, N, P), files) of one condition's runs."""
    paths = expand(pattern)
    true, first = load_predictions(paths[0], cohort, split)
    preds = [first]
    for path in paths[1:]:
        true_k, pred_k = load_predictions(path, cohort, split)
        check_same_targets(true, true_k, path)
        preds.append(pred_k)
    return true, np.stack([p.to_numpy() for p in preds]), paths


def check_same_targets(reference: pd.DataFrame, other: pd.DataFrame, path: Path) -> None:
    if not (reference.index.equals(other.index) and reference.columns.equals(other.columns)):
        raise SystemExit(f"{path}: cases or pathways differ from the baseline.")
    if not np.allclose(reference.to_numpy(), other.to_numpy()):
        raise SystemExit(f"{path}: targets differ from the baseline (other z-score reference?).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline", required=True, help="predictions of condition A (file or quoted glob over seeds)"
    )
    parser.add_argument("--candidates", nargs="+", required=True, help="conditions B (files or globs)")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="writes <output>.csv/.md")
    args = parser.parse_args()

    true_a, preds_a, paths_a = load_condition(args.baseline, args.cohort, args.split)
    names = list(true_a.columns)
    y = true_a.to_numpy()
    for path, pred in zip(paths_a, preds_a, strict=True):
        check_metrics(y, pred, names, path)

    rows = []
    for pattern in args.candidates:
        true_b, preds_b, paths_b = load_condition(pattern, args.cohort, args.split)
        check_same_targets(true_a, true_b, paths_b[0])
        for path, pred in zip(paths_b, preds_b, strict=True):
            check_metrics(y, pred, names, path)

        point_a, point_b, deltas = paired_bootstrap(y, preds_a, preds_b, n_boot=args.n_boot, seed=args.seed)
        for k, metric in enumerate(METRICS):
            rows.append(
                {
                    "baseline": label(args.baseline),
                    "candidate": label(pattern),
                    "runs": f"{len(paths_a)}/{len(paths_b)}",
                    "cohort": args.cohort,
                    "split": args.split,
                    "n": len(y),
                    "metric": metric,
                    "baseline_value": point_a[k],
                    "candidate_value": point_b[k],
                    "delta": point_b[k] - point_a[k],
                    "ci_low": float(np.percentile(deltas[:, k], 2.5)),
                    "ci_high": float(np.percentile(deltas[:, k], 97.5)),
                    "p_delta_le_0": float((deltas[:, k] <= 0).mean()),
                }
            )

    results = pd.DataFrame(rows)
    spearman = results["metric"] == "median_spearman"
    values = results.loc[spearman, ["baseline_value", "candidate_value"]]
    if ((values < -1) | (values > 1)).any().any():
        raise ValueError(f"Spearman outside [-1, 1]:\n{results}")
    if (results.loc[~spearman, ["baseline_value", "candidate_value"]] > 1).any().any():
        raise ValueError(f"R² above 1:\n{results}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    command = "uv run python -m xaiwsi.evaluation.paired_bootstrap " + shlex.join(sys.argv[1:])
    table = markdown_table(results)
    args.output.with_suffix(".md").write_text(
        f"Paired bootstrap over {args.cohort} {args.split} cases ({args.n_boot} resamples, "
        f"seed {args.seed}); delta = candidate − baseline, 95% percentile interval; "
        f"runs = baseline/candidate runs averaged per resample. "
        f"Generated by:\n\n```bash\n{command}\n```\n\n{table}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
