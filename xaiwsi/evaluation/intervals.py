"""Split-conformal prediction intervals for WSI → Hallmark predictions: coverage and width.

A condition is a quoted glob of per-case prediction files with val and test rows (one
file per seed: ``predict_run`` writes ``<run-dir>/predictions.parquet``; ``cohort_probe``
writes ridge predictions). Its point prediction is the mean over files. Intervals are
calibrated on the TCGA val split of a test cohort's z-score reference cohort
(``--pairs cptac-luad:tcga-luad``) and evaluated on that test cohort. All inputs at
prediction time are the slide only; RNA gives the calibration and evaluation targets.

Methods (``xaiwsi.uncertainty.conformal``):
- ``marginal``: no model; the val mean of each pathway ± a conformal quantile. The width
  a model has to beat (``width_ratio`` = width / marginal width, < 1 is informative).
- ``constant``: point prediction ± a per-pathway constant.
- ``linear``: as ``constant`` after a per-pathway linear recalibration y ≈ a + b * pred
  fit on the calibration cases (fixes a model whose ranking is better than its scale).
  a, b and q come from the same cases, so the coverage guarantee is approximate.
- ``ensemble``: point prediction ± q * (sd over seeds + its val median); needs ≥ 2 files.
  The val median keeps the scale away from 0 when the seeds agree.

Each condition is evaluated on the test cohort and, as an in-distribution check, on its
calibration cohort by 2-fold cross-fitting within val (``eval`` = ``<cohort> val 2-fold``).

    uv run python -m xaiwsi.evaluation.intervals \\
        --conditions 'tile luad+lusc=runs/pathways/xai-wsi/coh-1024-luad+lusc-percohort-s*/predictions.parquet' \\
        --pairs cptac-luad:tcga-luad cptac-lusc:tcga-lusc --alpha 0.1 \\
        --output outputs/intervals/conformal

Writes ``<output>.csv`` and ``<output>.md`` (Markdown table with the generating command),
and ``<output>_per_pathway.csv`` (coverage, width_ratio and abs_bias per pathway).
Columns: per-pathway coverage averaged and minimum over pathways; median width (z units of
the targets); width_ratio; mean interval score (lower is better); median |mean residual|
(a cohort offset that intervals calibrated on TCGA can't absorb); for ``ensemble``,
sigma_rho = median over pathways of Spearman(seed sd, |residual|).
"""

from __future__ import annotations

import argparse
import glob
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.uncertainty.conformal import fit_intervals, interval_metrics


def load_condition(pattern: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(targets, mean prediction, sd over files or NaN), indexed by (split, cohort, case_id)."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No prediction files match {pattern!r}.")
    trues, preds = [], []
    for path in paths:
        frame = pd.read_parquet(path).set_index(["split", "cohort", "case_id"]).sort_index()
        if frame.index.duplicated().any():
            raise SystemExit(f"{path}: duplicated (split, cohort, case_id) rows.")
        trues.append(frame.filter(like="true__").rename(columns=lambda c: c.removeprefix("true__")))
        preds.append(frame.filter(like="pred__").rename(columns=lambda c: c.removeprefix("pred__")))
    for path, true in zip(paths[1:], trues[1:], strict=True):
        if not (true.index.equals(trues[0].index) and true.columns.equals(trues[0].columns)):
            raise SystemExit(f"{path}: cases or pathways differ from {paths[0]}.")
        if not np.allclose(true.to_numpy(), trues[0].to_numpy()):
            raise SystemExit(f"{path}: targets differ from {paths[0]}.")
    stack = np.stack([p.to_numpy() for p in preds])
    mean = pd.DataFrame(stack.mean(axis=0), index=trues[0].index, columns=trues[0].columns)
    sd = stack.std(axis=0, ddof=1) if len(paths) > 1 else np.full(stack.shape[1:], np.nan)
    return trues[0], mean, pd.DataFrame(sd, index=mean.index, columns=mean.columns)


def select(frames, split: str, cohort: str) -> tuple[np.ndarray, ...]:
    key = (split, cohort)
    for frame in frames:
        if key not in frame.index.droplevel("case_id"):
            raise SystemExit(f"No {split} rows for cohort {cohort!r}.")
    return tuple(frame.xs(key, level=["split", "cohort"]).to_numpy() for frame in frames)


def linear_recalibration(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column least squares y ≈ a + b * pred; returns (a, b), each (P,)."""
    pred_c = pred - pred.mean(axis=0)
    b = (pred_c * (y - y.mean(axis=0))).sum(axis=0) / (pred_c**2).sum(axis=0)
    return y.mean(axis=0) - b * pred.mean(axis=0), b


def predict_intervals(method: str, cal, test, alpha: float):
    """(lower, upper, scale_or_None) on `test` from calibration data `cal`; both are
    (y, pred, sd) tuples."""
    y_cal, pred_cal, sd_cal = cal
    y_test, pred_test, sd_test = test
    if method == "marginal":
        center = y_cal.mean(axis=0)
        q = fit_intervals(y_cal, np.broadcast_to(center, y_cal.shape), 1.0, alpha)
        center = np.broadcast_to(center, y_test.shape)
        return center - q, center + q, None
    if method == "constant":
        q = fit_intervals(y_cal, pred_cal, 1.0, alpha)
        return pred_test - q, pred_test + q, None
    if method == "linear":
        a, b = linear_recalibration(y_cal, pred_cal)
        q = fit_intervals(y_cal, a + b * pred_cal, 1.0, alpha)
        center = a + b * pred_test
        return center - q, center + q, None
    if method == "ensemble":
        floor = np.median(sd_cal, axis=0)
        q = fit_intervals(y_cal, pred_cal, sd_cal + floor, alpha)
        scale = sd_test + floor
        return pred_test - q * scale, pred_test + q * scale, sd_test
    raise ValueError(method)


def evaluate(method: str, cal, test, alpha: float) -> dict:
    lower, upper, sd = predict_intervals(method, cal, test, alpha)
    y, pred, _ = test
    metrics = interval_metrics(y, lower, upper, alpha)
    result = {k: v for k, v in metrics.items()}
    result["abs_bias"] = np.abs((y - (lower + upper) / 2).mean(axis=0))
    if sd is not None:
        rho = [spearmanr(sd[:, k], np.abs(y[:, k] - pred[:, k])).statistic for k in range(y.shape[1])]
        result["sigma_rho"] = np.asarray(rho)
    return result


def two_fold(method: str, cal, alpha: float, seed: int) -> dict:
    """Calibrate on one half of val and evaluate on the other, both ways; per-case
    outcomes of the two halves are pooled (means weighted by half size)."""
    n = len(cal[0])
    perm = np.random.default_rng(seed).permutation(n)
    halves = [perm[: n // 2], perm[n // 2 :]]
    parts = []
    for a, b in ((0, 1), (1, 0)):
        pick = lambda idx: tuple(x[idx] for x in cal)  # noqa: E731
        parts.append((len(halves[b]), evaluate(method, pick(halves[a]), pick(halves[b]), alpha)))
    total = sum(n_b for n_b, _ in parts)
    return {k: sum(n_b * r[k] for n_b, r in parts) / total for k in parts[0][1]}


def summarize(result: dict, marginal_width: np.ndarray) -> dict:
    row = {
        "mean_coverage": float(result["coverage"].mean()),
        "min_coverage": float(result["coverage"].min()),
        "median_width": float(np.median(result["width"])),
        "width_ratio": float(np.median(result["width"] / marginal_width)),
        "interval_score": float(result["interval_score"].mean()),
        "median_abs_bias": float(np.median(result["abs_bias"])),
        "sigma_rho": float(np.median(result["sigma_rho"])) if "sigma_rho" in result else np.nan,
    }
    return row


def check_ranges(results: pd.DataFrame) -> None:
    coverage = results[["mean_coverage", "min_coverage"]]
    if ((coverage < 0) | (coverage > 1)).any().any():
        raise ValueError(f"Coverage outside [0, 1]:\n{results}")
    if (results[["median_width", "width_ratio", "interval_score"]] < 0).any().any():
        raise ValueError(f"Negative width or interval score:\n{results}")
    rho = results["sigma_rho"].dropna()
    if ((rho < -1) | (rho > 1)).any():
        raise ValueError(f"sigma_rho outside [-1, 1]:\n{results}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conditions", nargs="+", required=True, help="'<label>=<glob>' entries")
    parser.add_argument("--pairs", nargs="+", required=True, help="'<test cohort>:<calibration cohort>'")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--methods", nargs="+", default=["marginal", "constant", "linear", "ensemble"])
    parser.add_argument("--seed", type=int, default=0, help="val 2-fold split")
    parser.add_argument("--output", type=Path, required=True, help="writes <output>.csv/.md")
    args = parser.parse_args()

    rows, per_pathway = [], []
    for entry in args.conditions:
        label, sep, pattern = entry.partition("=")
        if not sep:
            raise SystemExit(f"Condition {entry!r} is not '<label>=<glob>'.")
        frames = load_condition(pattern)
        names = list(frames[0].columns)
        n_runs = len(glob.glob(pattern))
        cohorts = set(frames[0].index.droplevel("case_id"))
        for pair in args.pairs:
            test_cohort, _, cal_cohort = pair.partition(":")
            if ("test", test_cohort) not in cohorts or ("val", cal_cohort) not in cohorts:
                print(f"skip {label} {pair}: cohort not in the predictions")
                continue
            cal = select(frames, "val", cal_cohort)
            test = select(frames, "test", test_cohort)
            evals = {
                f"{cal_cohort} val 2-fold": lambda m: two_fold(m, cal, args.alpha, args.seed),
                test_cohort: lambda m: evaluate(m, cal, test, args.alpha),
            }
            for eval_name, run in evals.items():
                if eval_name.endswith("2-fold") and any(
                    r["condition"] == label and r["eval"] == eval_name for r in rows
                ):
                    continue  # same calibration cohort as an earlier pair
                marginal_width = run("marginal")["width"]
                for method in args.methods:
                    if method == "ensemble" and n_runs < 2:
                        continue
                    n_eval = len(test[0]) if eval_name == test_cohort else len(cal[0])
                    result = run(method)
                    keys = {"condition": label, "eval": eval_name, "method": method}
                    rows.append(
                        {
                            "condition": label,
                            "runs": n_runs,
                            "calibration": f"{cal_cohort} val ({len(cal[0])})",
                            "eval": eval_name,
                            "n": n_eval,
                            "method": method,
                            **summarize(result, marginal_width),
                        }
                    )
                    per_pathway.append(
                        pd.DataFrame(
                            {
                                **keys,
                                "pathway": names,
                                "coverage": result["coverage"],
                                "width_ratio": result["width"] / marginal_width,
                                "abs_bias": result["abs_bias"],
                            }
                        )
                    )
                    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[-1].items()))

    results = pd.DataFrame(rows)
    check_ranges(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    pd.concat(per_pathway, ignore_index=True).to_csv(
        args.output.with_name(args.output.name + "_per_pathway.csv"), index=False
    )
    command = "uv run python -m xaiwsi.evaluation.intervals " + shlex.join(sys.argv[1:])
    table = markdown_table(results)
    args.output.with_suffix(".md").write_text(
        f"Split-conformal intervals, target coverage {1 - args.alpha:.0%} per pathway; widths in "
        f"z units of the mean-z targets. Generated by:\n\n```bash\n{command}\n```\n\n{table}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
