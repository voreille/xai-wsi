"""Zero-shot slide classification with TITAN's text tower, e.g. the predominant LUAD pattern on
DHMC-LUAD (TITAN paper, Supplementary Table 103: balanced accuracy 0.585 ± 0.020, weighted F1
0.489, AUROC 0.859, with class names that were not published).

Slide embeddings come from ``xaiwsi.concepts.titan`` (``<cohort dir>/wsi-embeddings/titan/
slides.parquet``, CONCH v1.5 512 px tiles); the classifier from a prompt YAML
(``configs/zeroshot/*.yaml``: the paper's templates and our class names). Labels are only used to
score. Metrics as in the paper: balanced accuracy, weighted F1 and macro one-vs-rest AUROC of the
softmax over cosines (``Titan.zero_shot``), with a slide bootstrap (mean ± sd and 95% interval),
plus per-class AUROC and the confusion matrix.

    uv run python -m xaiwsi.evaluation.titan_zeroshot \\
        --embeddings data/processed/dhmc-luad/wsi-embeddings/titan/slides.parquet \\
        --labels data/processed/dhmc-luad/metadata/growth_patterns.csv --label-column dominant_pattern \\
        --slides data/processed/dhmc-luad/metadata/wsi_slides.csv \\
        --prompts configs/zeroshot/dhmc_luad_patterns.yaml --output-dir outputs/titan_zeroshot/dhmc-luad

Writes ``metrics.csv/.md`` (with the command), ``confusion.csv``, ``predictions.parquet``
(slide_id, case_id, label, predicted, ``logit__<class>``, ``prob__<class>``) and
``classifier.pt`` (labels and text weights, reused by ``xaiwsi.visualization.titan_maps``).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from xaiwsi.concepts.titan import load_titan, text_classifier, zero_shot_logits
from xaiwsi.evaluation.cohort_probe import markdown_table


def scores(y: np.ndarray, prob: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = prob.argmax(1)
    out = {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=range(n_classes), zero_division=0),
    }
    present = np.unique(y)
    if len(present) == n_classes:
        out["macro_auroc"] = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=range(n_classes))
    else:
        out["macro_auroc"] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embeddings", type=Path, required=True, help="TITAN slides.parquet")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with case_id and the label column")
    parser.add_argument("--label-column", default="dominant_pattern")
    parser.add_argument("--slides", type=Path, required=True, help="wsi_slides.csv (slide_id, case_id)")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_titan(device)
    classes, weights = text_classifier(model, args.prompts, device)

    embeddings = pd.read_parquet(args.embeddings)
    embeddings.index = embeddings.index.astype(str)
    slides = pd.read_csv(args.slides, dtype=str)[["slide_id", "case_id"]]
    labels = pd.read_csv(args.labels, dtype=str)[["case_id", args.label_column]].rename(columns={args.label_column: "label"})
    table = slides.merge(labels, on="case_id").query("slide_id in @embeddings.index").sort_values("slide_id")
    unknown = sorted(set(table["label"]) - set(classes))
    if unknown:
        raise SystemExit(f"Labels without a prompt class: {unknown}")
    if table["case_id"].duplicated().any():
        raise SystemExit("Several slides per case: aggregate first (not implemented).")
    missing = len(labels) - len(table)
    print(f"{len(table)} slides with label and TITAN embedding ({missing} labelled cases without one)")

    x = torch.as_tensor(embeddings.loc[table["slide_id"]].to_numpy(), device=device)
    with torch.inference_mode():
        logits = zero_shot_logits(model, x, weights)
        prob = torch.softmax(logits, dim=-1).cpu().numpy()
    logits = logits.cpu().numpy()
    y = table["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    n_classes = len(classes)

    point = scores(y, prob, n_classes)
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(y), len(y))
        boot.append(scores(y[idx], prob[idx], n_classes))
    boot = pd.DataFrame(boot)
    rows = []
    for metric, value in point.items():
        b = boot[metric].dropna()
        rows.append({"metric": metric, "value": value, "boot_mean": b.mean(), "boot_sd": b.std(),
                     "ci_low": b.quantile(0.025), "ci_high": b.quantile(0.975), "n": len(y)})
    for i, c in enumerate(classes):
        auc = roc_auc_score(y == i, prob[:, i]) if 0 < (y == i).sum() < len(y) else np.nan
        rows.append({"metric": f"auroc_{c}", "value": auc, "n": int((y == i).sum())})
    metrics = pd.DataFrame(rows)
    in_range = metrics["value"].dropna()
    if ((in_range < 0) | (in_range > 1)).any():
        raise ValueError(f"metric outside [0, 1]:\n{metrics}")
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    confusion = pd.DataFrame(
        confusion_matrix(y, prob.argmax(1), labels=range(n_classes)),
        index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes],
    )
    confusion.to_csv(args.output_dir / "confusion.csv")
    out = table.reset_index(drop=True).assign(predicted=[classes[i] for i in prob.argmax(1)])
    for i, c in enumerate(classes):
        out[f"logit__{c}"] = logits[:, i]
        out[f"prob__{c}"] = prob[:, i]
    out.to_parquet(args.output_dir / "predictions.parquet", index=False)
    torch.save({"classes": classes, "weights": weights.cpu(), "prompts": str(args.prompts)}, args.output_dir / "classifier.pt")

    command = "uv run python -m xaiwsi.evaluation.titan_zeroshot " + shlex.join(sys.argv[1:])
    (args.output_dir / "metrics.md").write_text(
        f"TITAN zero-shot ({args.prompts}), {len(y)} slides; bootstrap over slides ({args.bootstrap}, seed "
        f"{args.seed}): mean ± sd and 95% percentile interval. Generated by:\n\n```bash\n{command}\n```\n\n"
        f"{markdown_table(metrics)}\n\nConfusion matrix (rows true, columns predicted):\n\n"
        f"{markdown_table(confusion.reset_index(names='class'))}\n"
    )
    print(markdown_table(metrics))
    print(confusion)


if __name__ == "__main__":
    main()
