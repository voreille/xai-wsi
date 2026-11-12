"""Collect the SAE concept-bag sweep (``scripts/concept_sweep.sh``) or the PANTHER sweep into one table.

Reads ``<sweep-dir>/<name>/probe_metrics.csv`` (``cohort_probe``) and the SAE's
``train.json`` (``data/processed/sae/<embedding>/<name>/``; with ``--encoder panther`` the
``prototypes.json`` of ``scripts/panther_sweep.sh`` under ``data/processed/panther/<embedding>/<name>/``),
keeps its val and test rows, and writes ``<output>.csv`` and ``<output>.md`` (one row per SAE and
feature kind and table preprocessing (``--table-min-active``/``--table-transforms``), plus the slide-mean baseline once, with the generating command). Fails on
out-of-range metrics and on runs whose slide-mean baseline differs (they would not share
targets).

    uv run python -m xaiwsi.evaluation.collect_concept_runs --sweep-dir outputs/concept_sweep \\
        --output outputs/concept_sweep/summary
    uv run python -m xaiwsi.evaluation.collect_concept_runs --encoder panther \\
        --sweep-dir outputs/panther_sweep --output outputs/panther_sweep/summary
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

from xaiwsi.evaluation.cohort_probe import check_ranges, feature_blocks, markdown_table


def bag_label(features: str) -> str:
    """Table kinds of a feature string, e.g. ``pi+mu`` for ``table:.../slides_pi.parquet+table:.../slides_mu.parquet``."""
    return "+".join(b.rsplit("slides_", 1)[-1].removesuffix(".parquet") for b in feature_blocks(features))


def encoder_settings(encoder: str, root: Path, name: str) -> dict:
    if encoder == "sae":
        train = json.loads((root / name / "train.json").read_text())
        settings = train["settings"]
        return {"sae": name, "train_cohorts": "+".join(train["train_cohorts"]), "latents": settings["latents"],
                "k": settings["k"], "seed": settings["seed"], "val_fvu": train["final"]["val_fvu"]}
    settings = json.loads((root / name / "prototypes.json").read_text())["settings"]
    return {"panther": name, "train_cohorts": "+".join(settings["train_cohorts"]),
            "components": settings["components"], "projection": Path(settings["pca"]).stem.rsplit("_", 1)[-1]
            if settings["pca"] else "full", "seed": settings["seed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=["sae", "panther"], default="sae")
    parser.add_argument(
        "--sae-root", type=Path, default=None,
        help="encoder dir (default data/processed/sae/h-optimus-1-fp16, or panther/h-optimus-1-fp16)",
    )
    parser.add_argument("--bags", nargs="+", default=None, help="keep only these bag kinds (mean, frac)")
    parser.add_argument("--preps", nargs="+", default=None, help="keep only these table preprocessings, e.g. min0_clip")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="keep only these encoder seeds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.sae_root or Path("data/processed") / args.encoder / "h-optimus-1-fp16"

    rows, baselines = [], []
    for path in sorted(args.sweep_dir.glob("*/probe_metrics.csv")):
        name = path.parent.name
        metrics = pd.read_csv(path)
        if "table_prep" not in metrics:  # written before cohort_probe had table preprocessing
            metrics["table_prep"] = "min0_none"
        if "probe" not in metrics:  # written before cohort_probe had --probes
            metrics["probe"] = "ridge"
        check_ranges(metrics)
        settings = encoder_settings(args.encoder, root, name)
        for _, row in metrics.iterrows():
            key = f"{row['split']}/{row['cohort']}"
            value = {"median_spearman": row["median_spearman"], "median_r2": row["median_r2"]}
            if row["features"] == "slide-mean":
                baselines.append({"run": name, "key": key, **value})
                continue
            rows.append(
                {
                    **settings,
                    "bag": bag_label(row["features"]),
                    "prep": row["table_prep"],
                    "probe": row["probe"],
                    "key": key,
                    **value,
                }
            )
    if not rows:
        raise SystemExit(f"No probe_metrics.csv under {args.sweep_dir}")
    base = pd.DataFrame(baselines)
    spread = base.groupby("key")["median_spearman"].agg(["min", "max"])
    if ((spread["max"] - spread["min"]) > 1e-9).any():
        raise ValueError(f"Slide-mean baselines differ between runs:\n{spread}")

    long = pd.DataFrame(rows)
    for column, values in (("bag", args.bags), ("prep", args.preps), ("seed", args.seeds)):
        if values is not None:
            long = long[long[column].isin(values)]
    if long.empty:
        raise SystemExit("No rows left after --bags/--preps/--seeds")
    index = list(settings) + ["bag", "prep", "probe"]
    wide = long.pivot_table(index=index, columns="key", values=["median_spearman", "median_r2"])
    wide.columns = [f"{metric} {key}" for metric, key in wide.columns]
    wide = wide.reset_index()
    first = base.drop_duplicates("key").set_index("key")
    baseline_row = {c: "" for c in index} | {index[0]: "slide mean", "bag": "-", "prep": "-", "probe": "ridge"}
    for metric in ("median_spearman", "median_r2"):
        for key in first.index:
            baseline_row[f"{metric} {key}"] = first.loc[key, metric]
    table = pd.concat([pd.DataFrame([baseline_row]), wide], ignore_index=True)
    sort_col = next(c for c in table.columns if c.startswith("median_spearman val/"))
    table = pd.concat([table.iloc[:1], table.iloc[1:].sort_values(sort_col, ascending=False)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    command = "uv run python -m xaiwsi.evaluation.collect_concept_runs " + shlex.join(sys.argv[1:])
    args.output.with_suffix(".md").write_text(
        f"{'SAE concept-bag' if args.encoder == 'sae' else 'PANTHER'} sweep, ridge probe, sorted by val median "
        f"Spearman. Generated by:\n\n```bash\n{command}\n```\n\n{markdown_table(table)}\n"
    )
    print(markdown_table(table))


if __name__ == "__main__":
    main()
