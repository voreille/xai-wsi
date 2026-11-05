"""What a tile-composition probe predicts on artifact tiles, and how much pooled weight they carry.

Two views, both on the probe outputs of ``xaiwsi.tissue.apply``:

- Labelled tile-QC sub-clusters (``outputs/tile_qc/clusters/artifact_subclusters.parquet``,
  names from ``configs/qc/tile_artifacts.yaml``): mean class fractions per sub-cluster, its
  top classes and the TME mass (summed fractions of the ``tme`` classes of ``--spec``), i.e.
  the weight such a tile adds to the TME pool of ``xaiwsi.evaluation.compartment_probe``.
- All tiles of each cohort, split by the artifact probability of ``xaiwsi.qc.tile_classifier``
  (flagged = at or above ``--threshold``): fraction of flagged tiles per slide, share of the
  TME and of each compartment's weight on flagged tiles, and top classes on flagged tiles.

    uv run python -m xaiwsi.evaluation.tissue_artifacts \\
        --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad tcga-lusc cptac-luad cptac-lusc \\
        --output outputs/tissue_checks/ignite_artifacts

Writes ``<output>_subclusters.csv``, ``<output>_cohorts.csv`` and ``<output>.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.qc.mask import load_artifact_probs
from xaiwsi.tissue.features import tissue_dir
from xaiwsi.training.run_config import data_args


def top_classes(mean: np.ndarray, classes: list[str], k: int = 3) -> str:
    return ", ".join(f"{classes[i]} {mean[i]:.2f}" for i in np.argsort(mean)[::-1][:k])


def check_fraction(values: pd.Series, name: str) -> None:
    if ((values < 0) | (values > 1 + 1e-6)).any():
        raise ValueError(f"{name} outside [0, 1]:\n{values}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="CohortPathwayDataModule config")
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/tissue/compartments.yaml"))
    parser.add_argument(
        "--subclusters", type=Path, default=Path("outputs/tile_qc/clusters/artifact_subclusters.parquet")
    )
    parser.add_argument("--labels", type=Path, default=Path("configs/qc/tile_artifacts.yaml"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(args.spec.read_text())
    data = data_args(yaml.safe_load(args.config.read_text()))
    processed_root, embedding = data.get("processed_root"), data.get("embedding", "h-optimus-1")
    specs = {c["name"]: CohortSpec(**c) for c in data["train_cohorts"] + data["test_cohorts"]}
    embedding_dirs = {}
    for name in args.cohorts:
        if name not in specs:
            raise SystemExit(f"cohort {name!r} not in {args.config}")
        embedding_dirs[name] = specs[name].resolve(processed_root, embedding).embedding_dir
    probe_dirs = {name: tissue_dir(d) / spec["probe"] for name, d in embedding_dirs.items()}
    classes = json.loads((next(iter(probe_dirs.values())) / "probe.json").read_text())["classes"]
    groups = {"tme": spec["tme"], **spec["compartments"]}
    group_idx = {g: [classes.index(c) for c in members] for g, members in groups.items()}

    # Labelled sub-clusters (tiles from any cohort with probe outputs).
    labels = yaml.safe_load(args.labels.read_text())
    names = {}
    for key, prefix in (
        ("artifact_subclusters", "artifact"),
        ("tissue_subclusters", "tissue"),
        ("excluded_subclusters", "excluded"),
    ):
        names |= {k: f"{prefix}: {v}" for k, v in labels.get(key, {}).items()}
    subclusters = pd.read_parquet(args.subclusters)
    fractions: dict[int, list[np.ndarray]] = {}
    for (cohort, slide_id), tiles in subclusters.groupby(["cohort", "slide_id"]):
        path = tissue_dir(Path(processed_root) / cohort / "wsi-embeddings" / embedding) / spec["probe"]
        path = path / "tiles" / f"{slide_id}.npy"
        if not path.exists():
            continue
        slide = np.load(path).astype(np.float32)
        for sub, row in zip(tiles["sub"], slide[tiles["tile"].to_numpy()], strict=True):
            fractions.setdefault(int(sub), []).append(row)
    sub_rows = []
    for sub in sorted(fractions):
        mean = np.mean(fractions[sub], axis=0)
        sub_rows.append(
            {
                "subcluster": sub,
                "label": names.get(sub, "?"),
                "n_tiles": len(fractions[sub]),
                "top_classes": top_classes(mean, classes),
                "tme_mass": float(mean[group_idx["tme"]].sum()),
                **{f"{g}_mass": float(mean[idx].sum()) for g, idx in group_idx.items() if g != "tme"},
            }
        )
    sub_table = pd.DataFrame(sub_rows)
    for column in sub_table.filter(like="_mass").columns:
        check_fraction(sub_table[column], column)

    # All tiles, by artifact probability.
    cohort_rows = []
    for cohort, embedding_dir in embedding_dirs.items():
        probs = load_artifact_probs(embedding_dir)
        flagged_sum = np.zeros(len(classes))
        weight_flagged = dict.fromkeys(groups, 0.0)
        weight_all = dict.fromkeys(groups, 0.0)
        flagged_fraction = []
        for slide_id, prob in probs.items():
            path = probe_dirs[cohort] / "tiles" / f"{slide_id}.npy"
            if not path.exists():
                continue
            slide = np.load(path).astype(np.float64)
            if len(slide) != len(prob):
                raise ValueError(f"{cohort}/{slide_id}: {len(slide)} probe rows, {len(prob)} QC scores")
            flagged = np.asarray(prob) >= args.threshold
            flagged_fraction.append(flagged.mean())
            flagged_sum += slide[flagged].sum(axis=0)
            for g, idx in group_idx.items():
                w = slide[:, idx].sum(axis=1)
                weight_flagged[g] += w[flagged].sum()
                weight_all[g] += w.sum()
        if not flagged_fraction:
            raise SystemExit(f"{cohort}: no slides with both QC scores and probe outputs")
        flagged_mean = flagged_sum / max(flagged_sum.sum(), 1e-12)
        cohort_rows.append(
            {
                "cohort": cohort,
                "n_slides": len(flagged_fraction),
                "flagged_tiles_mean": float(np.mean(flagged_fraction)),
                "flagged_tiles_median": float(np.median(flagged_fraction)),
                **{f"{g}_weight_on_flagged": weight_flagged[g] / weight_all[g] for g in groups},
                "top_classes_flagged": top_classes(flagged_mean, classes, k=4),
            }
        )
    cohort_table = pd.DataFrame(cohort_rows)
    for column in cohort_table.select_dtypes("number").columns.drop("n_slides"):
        check_fraction(cohort_table[column], column)

    sub_table.to_csv(f"{args.output}_subclusters.csv", index=False)
    cohort_table.to_csv(f"{args.output}_cohorts.csv", index=False)
    command = "uv run python -m xaiwsi.evaluation.tissue_artifacts " + shlex.join(sys.argv[1:])
    text = (
        f"Probe `{spec['probe']}` on artifact tiles. Generated by:\n\n```bash\n{command}\n```\n\n"
        "Labelled tile-QC sub-clusters (mean probe fractions; *_mass = summed fractions of the "
        f"classes of each pool in {args.spec}):\n\n{markdown_table(sub_table)}\n\n"
        f"All tiles, flagged = artifact probability >= {args.threshold:g} "
        f"(*_weight_on_flagged = share of the pool's total weight on flagged tiles):\n\n"
        f"{markdown_table(cohort_table)}\n"
    )
    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
