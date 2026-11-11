"""Do the shared WSI <-> RNA CCA axes separate labelled histological subtypes (e.g. mucinous LUAD)?

Refits the linear CCA of ``xaiwsi.evaluation.cca_axis_reading`` (slide means vs BulkFormer,
PCA ``--pca-dim``, TCGA train; same fit as the tile maps) and, for each label group, gives
the AUC of each axis (slide score u_k and RNA score v_k) for the positive cases against the
other labelled cases, with a stratified bootstrap CI. ``--genes`` adds single genes (log2
TPM+1) as references, to judge the labels themselves (e.g. NKX2-1, lost in invasive
mucinous adenocarcinoma). The TCGA train rows are in-sample for the CCA (not for the
labels); val and CPTAC are held out.

A group is ``<name>::<csv>::<column>::<regex>``: cases of the CSV (``case_id`` column) whose
``<column>`` matches the regex (case-insensitive) are positive, the other non-empty values
negative.

    uv run python -m xaiwsi.evaluation.cca_axis_labels \\
        --config configs/training/alignment_baseline.yaml \\
        --group 'tcga mucinous::data/processed/tcga-luad/metadata/primary_diagnosis.csv::primary_diagnosis::(?<!non-)mucinous|signet' \\
        --genes NKX2-1 MUC5AC --slide-means-dir outputs/cohort_probe --output outputs/cca_axis_reading/labels

``--map-axes`` draws tile maps of those axes (``cca_decomposition.draw_maps``) on the highest
and lowest slides of ``--map-split``, each annotated with its labels from the groups.

Writes ``<output>.csv``, ``<output>_scores.csv``, ``<output>.md`` and ``<output>_maps/``.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from xaiwsi.data.pathway import load_log2_tpm_gene_symbols
from xaiwsi.evaluation.cca_hallmarks import SPLIT_NAMES
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.shared_structure import SPLITS, LinearCCA, check_ranges, load_frozen
from xaiwsi.training.run_config import data_args


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(roc_auc_score(y[idx], score[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--group", nargs="+", required=True, help="<name>::<csv>::<column>::<regex>")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--genes", nargs="*", default=[])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--map-axes", type=int, nargs="*", default=[], help="draw tile maps of these axes")
    parser.add_argument("--map-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--map-high", type=int, default=8)
    parser.add_argument("--map-low", type=int, default=3)
    parser.add_argument("--map-size", type=int, default=512)
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output.parent)
    cca = LinearCCA(args.pca_dim).fit(splits["train"]["x"], splits["train"]["y"])
    expression = {}
    if args.genes:
        expression = {
            "tcga": load_log2_tpm_gene_symbols(data["tcga_rna_dir"]),
            "cptac": load_log2_tpm_gene_symbols(data["cptac_rna_dir"]),
        }
    scores = []
    for split in SPLITS:
        u, v = cca.transform(splits[split]["x"], splits[split]["y"])
        pairs = splits[split]["pairs"]
        frame = pd.DataFrame({"case_id": pairs["case_id"].astype(str).to_numpy(), "split": SPLIT_NAMES[split]})
        for axis in args.axes:
            frame[f"slide axis {axis}"] = u[:, axis - 1]
            frame[f"rna axis {axis}"] = v[:, axis - 1]
        source = "cptac" if split == "test" else "tcga"
        for gene in args.genes:
            frame[f"gene {gene}"] = expression[source].loc[gene, pairs["sample_id"].astype(str)].to_numpy()
        scores.append(frame)
    scores = pd.concat(scores, ignore_index=True)
    features = [c for c in scores.columns if c not in ("case_id", "split")]

    rng = np.random.default_rng(0)
    rows = []
    notes: dict[str, list[str]] = {}
    for spec in args.group:
        parts = spec.split("::")
        if len(parts) != 4:
            raise SystemExit(f"expected <name>::<csv>::<column>::<regex>, got {spec!r}")
        name, path, column, pattern = parts
        labels = pd.read_csv(path).astype({"case_id": str}).set_index("case_id")[column].fillna("")
        labels = labels[labels.str.strip() != ""]
        positive = labels.str.contains(pattern, flags=re.IGNORECASE, regex=True)
        for case, value in labels.items():
            notes.setdefault(case, []).append(f"{name}: {value}{' [+]' if positive.loc[case] else ''}")
        cases = scores[scores["case_id"].isin(labels.index)]
        groups = list(cases.groupby("split", sort=False))
        if len(groups) > 1:
            groups.append(("all labelled", cases))
        for split, frame in groups:
            y = positive.loc[frame["case_id"]].to_numpy()
            if y.sum() == 0 or y.all():
                continue
            for feature in features:
                values = frame[feature].to_numpy()
                low, high = bootstrap_auc(y, values, args.bootstrap, rng)
                rows.append({"group": name, "split": split, "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                             "feature": feature, "auc": roc_auc_score(y, values), "ci_low": low, "ci_high": high})
    results = pd.DataFrame(rows)
    check_ranges(results, ["auc", "ci_low", "ci_high"], 0, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{args.output}.csv", index=False)
    scores.to_csv(f"{args.output}_scores.csv", index=False)

    compact = results.assign(
        value=results["auc"].map("{:.2f}".format)
        + " [" + results["ci_low"].map("{:.2f}".format) + ", " + results["ci_high"].map("{:.2f}".format) + "]"
    ).pivot_table(index=["group", "split", "n_pos", "n_neg"], columns="feature", values="value",
                  aggfunc="first", sort=False)
    compact = compact[[f for f in features if f in compact.columns]].rename_axis(columns=None).reset_index()
    command = "uv run python -m xaiwsi.evaluation.cca_axis_labels " + shlex.join(sys.argv[1:])
    text = (
        f"AUC of the CCA axes (slide means vs BulkFormer, PCA {args.pca_dim}, TCGA train) and of single genes "
        f"for each label group (positives vs the other labelled cases; 95% stratified bootstrap CI, "
        f"{args.bootstrap} draws), as `AUC [low, high]`. AUC > 0.5 = higher in the positives. Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{markdown_table(compact)}\n"
    )
    if args.map_axes:
        from xaiwsi.evaluation.cca_decomposition import draw_maps

        split = args.map_split
        u = cca.transform(splits[split]["x"], splits[split]["y"])[0]
        args.map_slides = args.map_high
        maps = draw_maps(
            args, splits, cca, "bulkformer", args.map_axes, u, Path(f"{args.output}_maps"), split=split,
            n_high=args.map_high, n_low=args.map_low, notes={k: " | ".join(v) for k, v in notes.items()},
        )
        pd.DataFrame(maps).to_csv(f"{args.output}_maps/slides.csv", index=False)
        text += (
            f"\nTile maps of axes {args.map_axes} on the {args.map_high} highest and {args.map_low} lowest "
            f"{SPLIT_NAMES[split]} slides, annotated with the labels above ([+] = positive): "
            f"`{Path(f'{args.output}_maps').name}/` (`slides.csv` lists them).\n"
        )
    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
