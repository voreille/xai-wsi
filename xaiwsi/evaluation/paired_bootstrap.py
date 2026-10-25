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

Writes ``<output>.csv`` and ``<output>.md`` (Markdown table with the generating command).
For each metric (median over pathways of the per-pathway Spearman or R²): A, B, B − A,
the 95% percentile interval of B − A and the fraction of resamples with B − A ≤ 0.
"""

from __future__ import annotations

import argparse
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
    the Pearson correlation of average ranks (as scipy's spearmanr); pathways with a
    constant column are NaN and skipped by the median, as in compute_pathway_metrics.
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
    return np.stack([np.nanmedian(rho, axis=1), np.nanmedian(r2, axis=1)], axis=1)


METRICS = ["median_spearman", "median_r2"]


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    batch: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimates of A and B, shape (2,), and bootstrap differences B − A, (n_boot, 2)."""
    n = len(y_true)
    stack = np.stack([y_true, pred_a, pred_b])  # (3, N, P)
    point = [median_metrics(y_true[None], p[None])[0] for p in (pred_a, pred_b)]
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, n_boot, batch):
        idx = rng.integers(0, n, size=(min(batch, n_boot - start), n))
        t, a, b = (stack[k][idx] for k in range(3))  # (b, N, P)
        deltas.append(median_metrics(t, b) - median_metrics(t, a))
    return point[0], point[1], np.concatenate(deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True, help="predictions of condition A")
    parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="conditions B")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="writes <output>.csv/.md")
    args = parser.parse_args()

    def label(path: Path) -> str:
        return path.stem

    true_a, pred_a = load_predictions(args.baseline, args.cohort, args.split)
    rows = []
    for path in args.candidates:
        true_b, pred_b = load_predictions(path, args.cohort, args.split)
        if not (true_a.index.equals(true_b.index) and true_a.columns.equals(true_b.columns)):
            raise SystemExit(f"{path}: cases or pathways differ from the baseline.")
        if not np.allclose(true_a.to_numpy(), true_b.to_numpy()):
            raise SystemExit(f"{path}: targets differ from the baseline (other z-score reference?).")
        y, a, b = true_a.to_numpy(), pred_a.to_numpy(), pred_b.to_numpy()

        # The vectorized metrics must match the reference implementation.
        names = list(true_a.columns)
        for pred, mine in zip((a, b), median_metrics(y[None].repeat(2, 0), np.stack([a, b])), strict=True):
            ref, _ = compute_pathway_metrics(y, pred, names)
            if not np.allclose(mine, [ref["median_spearman"], ref["median_r2"]]):
                raise RuntimeError(f"Bootstrap metrics {mine} differ from compute_pathway_metrics {ref}.")

        point_a, point_b, deltas = paired_bootstrap(y, a, b, n_boot=args.n_boot, seed=args.seed)
        for k, metric in enumerate(METRICS):
            rows.append(
                {
                    "baseline": label(args.baseline),
                    "candidate": label(path),
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
        f"seed {args.seed}); delta = candidate − baseline, 95% percentile interval. "
        f"Generated by:\n\n```bash\n{command}\n```\n\n{table}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
