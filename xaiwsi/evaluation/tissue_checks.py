"""External checks of the tile-composition probes (``xaiwsi.tissue.apply``) on a cohort without masks.

1. Compartments vs pathology percentages (``--pathology-table``, TCIA CPTAC format: one row
   per slide with ``Slide_ID``, ``Case_ID``, ``Percent_Tumor_Nuclei``,
   ``Percent_Total_Cellularity``, ``Percent_Necrosis``; -1 = missing). Each predicted share of
   the tissue (class / (1 - background)) is correlated (Spearman) with each percentage, per
   slide and per case (means over the case's slides). The matched pairs (tumour epithelium vs
   tumour nuclei, necrosis vs necrosis) should stand out from the off-diagonal ones.
2. Growth-pattern shares vs dominant-pattern labels (``--labels name=csv``, e.g. cBioPortal
   and TCIA from ``xaiwsi.preprocessing.cptac``). Pattern shares are renormalized over the
   pattern classes (background left out, which the ANORAK probe under-predicts), on the slide
   the models use for each case (first slide_id per case, as in ``build_case_pairs``). Per
   label source: AUC of each pattern's share for "label == pattern vs rest" (groups with at
   least ``--min-group`` cases), and argmax agreement with the label vs the majority-class rate.

    uv run python -m xaiwsi.evaluation.tissue_checks --cohort-dir data/processed/cptac-luad \\
        --pathology-table data/raw/cptac-luad-clinical/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv \\
        --labels cbio=data/processed/cptac-luad/metadata/growth_patterns.csv \\
            tcia=data/processed/cptac-luad/metadata/growth_patterns_tcia.csv \\
        --output outputs/tissue_checks/cptac-luad

Writes ``<output>_pathology.csv``, ``<output>_patterns.csv`` and ``<output>.md``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from xaiwsi.evaluation.cohort_probe import markdown_table

PERCENTAGES = ["Percent_Tumor_Nuclei", "Percent_Total_Cellularity", "Percent_Necrosis"]
COMPARTMENTS = ["Tumor epithelium", "Stroma", "Inflammation", "Necrotic tissue", "Alveolar tissue"]
PATTERNS = ["lepidic", "acinar", "papillary", "micropapillary", "cribriform", "solid"]


def tissue_shares(slides: pd.DataFrame, classes: list[str], background: str) -> pd.DataFrame:
    tissue = 1 - slides[background]
    return slides[classes].div(tissue, axis=0)


def pathology_check(compartments: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    table = table[table["Slide_ID"].isin(compartments.index)].set_index("Slide_ID")
    pct = table[PERCENTAGES].where(table[PERCENTAGES] >= 0)
    rows = []
    for level in ("slide", "case"):
        shares, values = compartments.loc[table.index], pct
        if level == "case":
            shares = shares.groupby(table["Case_ID"]).mean()
            values = pct.groupby(table["Case_ID"]).mean()
        for compartment in shares.columns:
            for percentage in PERCENTAGES:
                ok = values[percentage].notna()
                rho, p = spearmanr(shares.loc[ok, compartment], values.loc[ok, percentage])
                rows.append(
                    {"level": level, "compartment": compartment, "percentage": percentage,
                     "n": int(ok.sum()), "spearman": rho, "p": p}
                )
    return pd.DataFrame(rows)


def pattern_check(shares: pd.DataFrame, labels: pd.Series, min_group: int) -> pd.DataFrame:
    labels = labels[labels.index.isin(shares.index)]
    shares = shares.loc[labels.index]
    rows = []
    for pattern, n in labels.value_counts().items():
        if pattern not in shares.columns or n < min_group:
            continue
        inside = labels == pattern
        a, b = shares.loc[inside, pattern], shares.loc[~inside, pattern]
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {"pattern": pattern, "n": f"{len(a)}/{len(b)}", "auc": u / (len(a) * len(b)), "p": p,
             "median_share_in": a.median(), "median_share_out": b.median()}
        )
    in_patterns = labels.isin(shares.columns)
    agreement = (shares.loc[in_patterns].idxmax(axis=1) == labels[in_patterns]).mean()
    majority = labels[in_patterns].value_counts(normalize=True).iloc[0]
    rows.append(
        {"pattern": "argmax == label", "n": str(int(in_patterns.sum())), "auc": np.nan, "p": np.nan,
         "median_share_in": agreement, "median_share_out": majority}
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--compartments", default="ignite", help="tissue/<name> with the compartment probe")
    parser.add_argument("--patterns", default="anorak", help="tissue/<name> with the pattern probe")
    parser.add_argument("--pathology-table", type=Path, default=None)
    parser.add_argument("--labels", nargs="*", default=[], help="name=csv with case_id, dominant_pattern")
    parser.add_argument("--min-group", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = "uv run python -m xaiwsi.evaluation.tissue_checks " + shlex.join(sys.argv[1:])
    text = f"Tile-composition probes on {args.cohort_dir.name}, external checks. Generated by:\n\n```bash\n{command}\n```\n"

    if args.pathology_table is not None:
        slides = pd.read_parquet(args.cohort_dir / "tissue" / args.compartments / "slides.parquet").set_index("slide_id")
        compartments = tissue_shares(slides, COMPARTMENTS, "Background")
        result = pathology_check(compartments, pd.read_csv(args.pathology_table))
        if not result["spearman"].dropna().between(-1, 1).all():
            raise SystemExit("Spearman out of range")
        result.to_csv(f"{args.output}_pathology.csv", index=False)
        for level, block in result.groupby("level", sort=False):
            wide = block.pivot(index="compartment", columns="percentage", values="spearman").loc[COMPARTMENTS]
            text += (
                f"\nCompartment share of tissue vs pathology %, Spearman per {level} "
                f"(n = {block['n'].min()}–{block['n'].max()}):\n\n{markdown_table(wide.reset_index())}\n"
            )

    if args.labels:
        slides = pd.read_parquet(args.cohort_dir / "tissue" / args.patterns / "slides.parquet").set_index("slide_id")
        meta = pd.read_csv(args.cohort_dir / "metadata" / "wsi_slides.csv")
        used = meta[meta["slide_id"].isin(slides.index)].sort_values(["case_id", "slide_id"]).drop_duplicates("case_id")
        shares = slides.loc[used["slide_id"], PATTERNS]
        shares = shares.div(shares.sum(axis=1), axis=0).set_axis(used["case_id"].to_numpy())
        results = []
        for spec in args.labels:
            name, _, path = spec.partition("=")
            labels = pd.read_csv(path).set_index("case_id")["dominant_pattern"].dropna()
            result = pattern_check(shares, labels, args.min_group).assign(labels=name)
            results.append(result)
            text += (
                f"\nPattern shares (among pattern classes) vs {name} dominant pattern; last row: "
                f"argmax agreement (median_share_in) vs majority-class rate (median_share_out):\n\n"
                f"{markdown_table(result.drop(columns='labels'))}\n"
            )
        pd.concat(results).to_csv(f"{args.output}_patterns.csv", index=False)

    Path(f"{args.output}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
