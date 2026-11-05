"""Where does a learned tile gate look? Gate of ``TileMLPGatedMean`` runs vs the IGNITE tissue classes.

For each run (saved ``config.yaml`` + best checkpoint), each val/test slide is passed
through the predictor, and its ``tile_gates`` / ``tile_weights`` are compared with the
per-tile fractions of the applied tile-composition probe (``xaiwsi.tissue.apply``,
``<cohort dir>/tissue/<probe>/tiles``), class by class and for the pools of ``--spec``:

- ``enrichment``: gated share / area share = (sum_i w_i f_ic) / mean_i f_ic. Above 1: the
  gate puts more weight on class c than its area; 1 is the plain mean.
- ``spearman``: Spearman of the gate with f_ic over the tiles of the slide.
- ``effective_fraction`` (per slide): (1 / sum_i w_i^2) / N, the share of tiles the gate
  effectively averages (1 = uniform).

Per run, split and cohort the median over slides is taken; the Markdown table reports the
mean and range over runs (seeds).

    uv run python -m xaiwsi.evaluation.gate_compartments \\
        --run-dir runs/pathways/xai-wsi/coh-1105gated-luad-percohort-s{0,1,2} \\
        --output outputs/gate_compartments/coh-1105gated-luad

Writes ``<output>_slides.parquet`` (one row per run, slide and class), ``<output>.csv``
(run-level medians) and ``<output>.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.data.pathway import tile_grid
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.predict_run import best_checkpoint, load_predictor, run_splits
from xaiwsi.tissue.features import tissue_dir
from xaiwsi.training.run_config import data_args


def slide_rows(gates: np.ndarray, weights: np.ndarray, fractions: np.ndarray, groups: dict) -> list[dict]:
    """One row per class/pool: enrichment and gate Spearman on one slide."""
    rows = []
    for name, idx in groups.items():
        f = fractions[:, idx].sum(axis=1)
        area = f.mean()
        enrichment = float(weights @ f / area) if area > 0 else np.nan
        rho = spearmanr(gates, f).statistic if np.std(f) > 0 and np.std(gates) > 0 else np.nan
        rows.append({"group": name, "area_share": float(area), "enrichment": enrichment, "spearman": rho})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/tissue/compartments.yaml"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    spec = yaml.safe_load(args.spec.read_text())

    records = []
    for run_dir in args.run_dir:
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        ckpt = best_checkpoint(run_dir)
        predictor = load_predictor(config, ckpt, device)
        data = data_args(config)
        processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
        cohort_specs = {c["name"]: CohortSpec(**c) for c in data["train_cohorts"] + data["test_cohorts"]}
        splits, _ = run_splits(config)
        for split in args.splits:
            for part in splits[split]:
                embedding_dir = cohort_specs[part.cohort].resolve(processed_root, embedding).embedding_dir
                probe_dir = tissue_dir(embedding_dir) / spec["probe"]
                classes = json.loads((probe_dir / "probe.json").read_text())["classes"]
                groups = {c: [k] for k, c in enumerate(classes)}
                groups |= {
                    f"pool:{name}": [classes.index(c) for c in members]
                    for name, members in {"tme": spec["tme"], **spec["compartments"]}.items()
                }
                for slide_id in part.pairs["slide_id"].astype(str):
                    embeddings, coords, attrs = part.store.load(slide_id)
                    x = torch.from_numpy(np.asarray(embeddings)).float().to(device)[None]
                    c = torch.from_numpy(tile_grid(coords, attrs)).to(device)[None]
                    with torch.no_grad():
                        out = predictor(x, coords=c)
                    if "tile_gates" not in out:
                        raise SystemExit(f"{run_dir}: predictor returns no tile_gates (not a gated model).")
                    gates = out["tile_gates"][0, :, 0].cpu().numpy().astype(np.float64)
                    weights = out["tile_weights"][0, :, 0].cpu().numpy().astype(np.float64)
                    fractions = np.load(probe_dir / "tiles" / f"{slide_id}.npy").astype(np.float64)
                    if len(fractions) != len(gates):
                        raise ValueError(f"{slide_id}: {len(fractions)} probe rows, {len(gates)} tiles")
                    effective = float(1.0 / (weights**2).sum() / len(weights))
                    for row in slide_rows(gates, weights, fractions, groups):
                        records.append(
                            {
                                "run": run_dir.name,
                                "split": split,
                                "cohort": part.cohort,
                                "slide_id": slide_id,
                                "effective_fraction": effective,
                                **row,
                            }
                        )
        print(f"{run_dir.name} ({ckpt.name}): done")

    slides = pd.DataFrame(records)
    rho = slides["spearman"].dropna()
    if ((rho < -1) | (rho > 1)).any():
        raise ValueError("gate Spearman outside [-1, 1]")
    if not slides["effective_fraction"].between(0, 1 + 1e-6).all():
        raise ValueError("effective fraction outside [0, 1]")
    if (slides["enrichment"].dropna() < 0).any():
        raise ValueError("negative enrichment")
    slides.to_parquet(f"{args.output}_slides.parquet", index=False)

    keys = ["run", "split", "cohort", "group"]
    per_run = slides.groupby(keys, sort=False)[["area_share", "enrichment", "spearman"]].median().reset_index()
    per_run.to_csv(f"{args.output}.csv", index=False)

    sections = []
    for (split, cohort), frame in per_run.groupby(["split", "cohort"], sort=False):
        stats = frame.groupby("group", sort=False).agg(
            area_share=("area_share", "mean"),
            enrichment=("enrichment", "mean"),
            enrichment_min=("enrichment", "min"),
            enrichment_max=("enrichment", "max"),
            gate_spearman=("spearman", "mean"),
            gate_spearman_min=("spearman", "min"),
            gate_spearman_max=("spearman", "max"),
        )
        stats = stats.sort_values("enrichment", ascending=False).reset_index()
        effective = (
            slides[(slides["split"] == split) & (slides["cohort"] == cohort)]
            .drop_duplicates(["run", "slide_id"])
            .groupby("run")["effective_fraction"]
            .median()
        )
        sections.append(
            f"### {split} {cohort} ({frame['run'].nunique()} runs)\n\n"
            f"Median effective fraction of tiles per run: "
            f"{', '.join(f'{v:.2f}' for v in effective.to_numpy())}\n\n{markdown_table(stats)}"
        )
    command = "uv run python -m xaiwsi.evaluation.gate_compartments " + shlex.join(sys.argv[1:])
    text = (
        f"Learned tile gate vs probe `{spec['probe']}` classes (median over slides per run; mean "
        "and min/max over runs). enrichment = gated share / area share (1 = plain mean); "
        "gate_spearman = Spearman of the gate with the class fraction over a slide's tiles. "
        f"Generated by:\n\n```bash\n{command}\n```\n\n" + "\n\n".join(sections) + "\n"
    )
    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
