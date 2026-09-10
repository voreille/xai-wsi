from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_pathway_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pathway_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute aggregate and per-pathway regression metrics."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )
    if y_true.ndim != 2:
        raise ValueError(f"Expected N x P arrays, got {y_true.shape}.")
    if y_true.shape[1] != len(pathway_names):
        raise ValueError(
            f"Got {y_true.shape[1]} outputs but {len(pathway_names)} pathway names."
        )

    rows: list[dict[str, float | str]] = []
    for i, name in enumerate(pathway_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = float(np.mean((yt - yp) ** 2))
        zero_mse = float(np.mean(yt**2))
        skill = 1.0 - mse / zero_mse if zero_mse > 0 else np.nan

        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if np.std(yt) > 0 and np.std(yp) > 0:
            rho = float(spearmanr(yt, yp).statistic)
        else:
            rho = np.nan

        rows.append(
            {
                "pathway": name,
                "mse": mse,
                "zero_mse": zero_mse,
                "skill": skill,
                "r2": r2,
                "spearman": rho,
            }
        )

    by_pathway = pd.DataFrame(rows).set_index("pathway")
    global_mse = float(np.mean((y_true - y_pred) ** 2))
    global_zero_mse = float(np.mean(y_true**2))

    summary = {
        "mse": global_mse,
        "zero_mse": global_zero_mse,
        "skill": (
            1.0 - global_mse / global_zero_mse
            if global_zero_mse > 0
            else np.nan
        ),
        "median_r2": float(by_pathway["r2"].median()),
        "median_spearman": float(by_pathway["spearman"].median()),
        "mean_spearman": float(by_pathway["spearman"].mean()),
        "n_rho_gt_0": float((by_pathway["spearman"] > 0).sum()),
        "n_rho_gt_03": float((by_pathway["spearman"] > 0.3).sum()),
    }
    return summary, by_pathway


def format_pathway_metrics(
    name: str,
    summary: dict[str, float],
    by_pathway: pd.DataFrame,
    top_k: int = 10,
) -> str:
    lines = [
        "",
        name,
        "-" * len(name),
        f"MSE:                 {summary['mse']:.4f}",
        f"Zero baseline MSE:   {summary['zero_mse']:.4f}",
        f"Skill vs zero:       {summary['skill']:+.3f}",
        f"Median R²:           {summary['median_r2']:+.3f}",
        f"Median Spearman:     {summary['median_spearman']:+.3f}",
        f"Mean Spearman:       {summary['mean_spearman']:+.3f}",
        f"Hallmarks rho > 0:   {int(summary['n_rho_gt_0'])}/{len(by_pathway)}",
        f"Hallmarks rho > 0.3: {int(summary['n_rho_gt_03'])}/{len(by_pathway)}",
        "",
        f"Top {top_k} Hallmarks by Spearman:",
        by_pathway.sort_values("spearman", ascending=False)
        .head(top_k)
        .to_string(
            columns=["spearman", "r2", "mse", "skill"],
            float_format=lambda x: f"{x:+.3f}",
        ),
    ]
    return "\n".join(lines)
