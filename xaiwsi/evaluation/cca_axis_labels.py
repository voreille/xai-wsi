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

Writes ``<output>.csv`` and ``<output>.md``.
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
    for spec in args.group:
        parts = spec.split("::")
        if len(parts) != 4:
            raise SystemExit(f"expected <name>::<csv>::<column>::<regex>, got {spec!r}")
        name, path, column, pattern = parts
        labels = pd.read_csv(path).astype({"case_id": str}).set_index("case_id")[column].fillna("")
        labels = labels[labels.str.strip() != ""]
        positive = labels.str.contains(pattern, flags=re.IGNORECASE, regex=True)
        cases = scores[scores["case_id"].isin(labels.index)]
        for split, frame in list(cases.groupby("split", sort=False)) + [("all labelled", cases)]:
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

    command = "uv run python -m xaiwsi.evaluation.cca_axis_labels " + shlex.join(sys.argv[1:])
    text = (
        f"AUC of the CCA axes (slide means vs BulkFormer, PCA {args.pca_dim}, TCGA train) and of single genes "
        f"for each label group (positives vs the other labelled cases; 95% stratified bootstrap CI, "
        f"{args.bootstrap} draws). AUC > 0.5 = higher in the positives. Generated by:\n\n```bash\n{command}\n```\n\n"
        f"{markdown_table(results)}\n"
    )
    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
