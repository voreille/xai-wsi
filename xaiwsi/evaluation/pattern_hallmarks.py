"""Hallmark scores vs histology labels (dominant growth pattern, grade) on a test cohort.

Each Hallmark is tested against the labels three ways: on the true RNA scores (does the
pathway differ by pattern at all?) and on the WSI-only predictions of each condition (does
the slide model recover that difference?). Labels are references for interpretation, not
training targets; the models never saw them.

Tests per score source and pathway:
- ``<group> vs rest``: AUC = P(score in group > score outside), Mann-Whitney p, for each
  pattern with at least ``--min-group`` labelled cases. AUC is scale-free, so true and
  predicted (shrunk) scores are comparable; ``mean_diff`` is in z units of the targets.
- ``pattern grade``: Spearman with an ordinal pattern grade (IASLC-style: lepidic 1,
  acinar/papillary 2, solid/micropapillary 3; ``--pattern-grade``); other patterns
  (mucinous, sarcomatoid, ...) are left out.
- ``histologic grade``: Spearman with G1/G2/G3 from the cohort's ``clinical.csv``
  (``--grade-clinical``; GX/unknown left out). Independent of the pattern labels and
  available for more cases.
q = Benjamini-Hochberg over the pathways within one source and test.

A condition is ``name=glob`` of per-case prediction files (``test_predictions.parquet`` of
the tile runs, or a ``cohort_probe`` predictions file); its prediction is the mean over
files. The true scores come from the files' ``true__`` columns (identical across files).

    uv run python -m xaiwsi.evaluation.pattern_hallmarks \\
        --labels data/processed/cptac-luad/metadata/growth_patterns.csv \\
        --grade-clinical data/processed/cptac-luad/metadata/clinical.csv \\
        --conditions 'tile=runs/pathways/xai-wsi/coh-1024-luad+lusc+brca-percohort-s*/test_predictions.parquet' \\
            'ridge=outputs/cohort_probe_lscc/predictions/tcga-luad,tcga-lusc,tcga-brca__per_cohort.parquet' \\
        --cohort cptac-luad --output outputs/pattern_hallmarks/cptac-luad

Writes ``<output>_tests.csv`` (long: source, test, pathway, n, effect, p, q),
``<output>_cases.parquet`` (labels + true + predicted scores per case, for picking example
slides), and ``<output>.md`` (tables with the generating command).
"""

from __future__ import annotations

import argparse
import glob
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu, spearmanr

from xaiwsi.evaluation.cohort_probe import markdown_table

DEFAULT_PATTERN_GRADE = "lepidic:1,acinar:2,papillary:2,solid:3,micropapillary:3"


def load_condition(pattern: str, cohort: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(true scores, mean prediction over files), indexed by case_id."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No prediction files match {pattern!r}.")
    trues, preds = [], []
    for path in paths:
        frame = pd.read_parquet(path)
        frame = frame[(frame["cohort"] == cohort) & (frame["split"] == split)].set_index("case_id")
        if frame.empty:
            raise SystemExit(f"{path}: no {split} rows for {cohort!r}.")
        if frame.index.duplicated().any():
            raise SystemExit(f"{path}: duplicated case_id rows.")
        frame = frame.sort_index()
        trues.append(frame.filter(like="true__").rename(columns=lambda c: c.removeprefix("true__")))
        preds.append(frame.filter(like="pred__").rename(columns=lambda c: c.removeprefix("pred__")))
    for path, true in zip(paths[1:], trues[1:], strict=True):
        if not (true.index.equals(trues[0].index) and true.columns.equals(trues[0].columns)):
            raise SystemExit(f"{path}: cases or pathways differ from {paths[0]}.")
        if not np.allclose(true.to_numpy(), trues[0].to_numpy()):
            raise SystemExit(f"{path}: targets differ from {paths[0]}.")
    mean = sum(p.to_numpy() for p in preds) / len(preds)
    return trues[0], pd.DataFrame(mean, index=trues[0].index, columns=trues[0].columns)


def parse_pattern_grade(spec: str) -> dict[str, int]:
    return {k.strip(): int(v) for k, v in (item.split(":") for item in spec.split(","))}


def one_vs_rest(scores: pd.DataFrame, labels: pd.Series, group: str) -> pd.DataFrame:
    inside = labels == group
    rows = []
    for pathway in scores.columns:
        a, b = scores.loc[inside, pathway], scores.loc[~inside, pathway]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "pathway": pathway,
                "n": f"{len(a)}/{len(b)}",
                "effect": u / (len(a) * len(b)),
                "mean_diff": a.mean() - b.mean(),
                "p": p,
            }
        )
    return pd.DataFrame(rows)


def ordinal(scores: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    rows = []
    for pathway in scores.columns:
        rho, p = spearmanr(scores[pathway], values)
        rows.append({"pathway": pathway, "n": str(len(values)), "effect": rho, "mean_diff": np.nan, "p": p})
    return pd.DataFrame(rows)


def check_ranges(tests: pd.DataFrame) -> None:
    auc = tests["test"].str.endswith("vs rest")
    bad = (
        (auc & ~tests["effect"].between(0, 1))
        | (~auc & ~tests["effect"].between(-1, 1))
        | ~tests["p"].between(0, 1)
        | tests[["effect", "p", "q"]].isna().any(axis=1)
    )
    if bad.any():
        raise SystemExit(f"Out-of-range statistics:\n{tests[bad]}")


def summary_tables(tests: pd.DataFrame, sources: list[str], top: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Agreement of true vs predicted effects per test, and top pathways per test."""
    wide = tests.pivot_table(index=["test", "pathway"], columns="source", values=["effect", "q"], aggfunc="first")
    agreement, tops = [], {}
    for test, block in wide.groupby(level="test", sort=False):
        block = block.droplevel("test")
        null = 0.5 if test.endswith("vs rest") else 0.0
        row = {"test": test, "n": tests.loc[tests["test"] == test, "n"].iloc[0]}
        for source in sources:
            row[f"q<0.1 {source}"] = int((block[("q", source)] < 0.1).sum())
        for source in sources[1:]:
            row[f"rho(true, {source})"] = spearmanr(block[("effect", "true")], block[("effect", source)])[0]
        agreement.append(row)

        order = (block[("effect", "true")] - null).abs().sort_values(ascending=False).index[:top]
        table = pd.DataFrame({"pathway": [p.removeprefix("HALLMARK_") for p in order]})
        for source in sources:
            table[f"{source} effect"] = block.loc[order, ("effect", source)].to_numpy()
            table[f"{source} q"] = block.loc[order, ("q", source)].to_numpy()
        tops[test] = table
    return pd.DataFrame(agreement), tops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True, help="CSV with case_id and the label column")
    parser.add_argument("--label-column", default="dominant_pattern")
    parser.add_argument("--conditions", nargs="+", required=True, help="name=glob of prediction files")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-group", type=int, default=5, help="minimum cases for a group-vs-rest test")
    parser.add_argument("--pattern-grade", default=DEFAULT_PATTERN_GRADE, help="pattern:rank,... ('' = skip)")
    parser.add_argument("--grade-clinical", type=Path, default=None, help="clinical.csv with tumor_grade")
    parser.add_argument(
        "--exclude-cases-from",
        type=Path,
        default=None,
        help="CSV with case_id; its labelled cases are dropped (pattern tests on new cases only)",
    )
    parser.add_argument("--top", type=int, default=10, help="pathways per test in the Markdown tables")
    parser.add_argument("--output", type=Path, required=True, help="output prefix")
    args = parser.parse_args()

    names, true, preds = [], None, {}
    for spec in args.conditions:
        name, _, pattern = spec.partition("=")
        if not pattern or name in ("true", *names):
            raise SystemExit(f"Bad or duplicate condition {spec!r}; use name=glob.")
        cond_true, cond_pred = load_condition(pattern, args.cohort, args.split)
        if true is None:
            true = cond_true
        elif not (cond_true.index.equals(true.index) and np.allclose(cond_true.to_numpy(), true.to_numpy())):
            raise SystemExit(f"Condition {name!r}: cases or targets differ from the first condition.")
        names.append(name)
        preds[name] = cond_pred
    sources = {"true": true, **preds}

    labels = pd.read_csv(args.labels).set_index("case_id")[args.label_column].dropna()
    labels = labels[labels.index.isin(true.index)]
    if args.exclude_cases_from is not None:
        excluded = pd.read_csv(args.exclude_cases_from).set_index("case_id")
        excluded = excluded[excluded.get(args.label_column, excluded.index.to_series()).notna()].index
        labels = labels[~labels.index.isin(excluded)]
    counts = labels.value_counts()
    print(f"{len(labels)} labelled cases with predictions:\n{counts.to_string()}")
    groups = [g for g, n in counts.items() if n >= args.min_group]

    tests = []
    for source, scores in sources.items():
        for group in groups:
            tests.append(one_vs_rest(scores.loc[labels.index], labels, group).assign(test=f"{group} vs rest"))
        if args.pattern_grade:
            ranks = labels.map(parse_pattern_grade(args.pattern_grade)).dropna()
            tests.append(ordinal(scores.loc[ranks.index], ranks).assign(test="pattern grade"))
        if args.grade_clinical is not None:
            grade = pd.read_csv(args.grade_clinical).set_index("case_id")["tumor_grade"]
            grade = grade.map({"G1": 1, "G2": 2, "G3": 3}).dropna()
            grade = grade[grade.index.isin(scores.index)]
            tests.append(ordinal(scores.loc[grade.index], grade).assign(test="histologic grade"))
        for frame in tests[-(len(groups) + bool(args.pattern_grade) + (args.grade_clinical is not None)) :]:
            frame["source"] = source
    tests = pd.concat(tests, ignore_index=True)
    tests["q"] = tests.groupby(["source", "test"])["p"].transform(false_discovery_control)
    tests = tests[["source", "test", "pathway", "n", "effect", "mean_diff", "p", "q"]]
    check_ranges(tests)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(f"{args.output}_tests.csv", index=False)
    cases = pd.concat(
        [labels.reindex(true.index).rename(args.label_column)]
        + [scores.add_prefix(f"{source}__") for source, scores in sources.items()],
        axis=1,
    )
    cases.to_parquet(f"{args.output}_cases.parquet")

    agreement, tops = summary_tables(tests, list(sources), args.top)
    command = "uv run python -m xaiwsi.evaluation.pattern_hallmarks " + shlex.join(sys.argv[1:])
    text = (
        f"Hallmarks vs histology on {args.cohort} {args.split}. Effect = AUC for `<group> vs rest` "
        f"(0.5 = no difference), Spearman for grades; q = BH over the {len(true.columns)} scores per source and test; "
        f"`rho(true, x)` = Spearman across pathways of true vs predicted effects. Generated by:\n\n"
        f"```bash\n{command}\n```\n\n{markdown_table(agreement)}\n"
    )
    for test, table in tops.items():
        text += f"\n{test}, top {args.top} pathways by true effect:\n\n{markdown_table(table)}\n"
    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
