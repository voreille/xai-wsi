"""Case-level dominant growth pattern from the TCIA CPTAC pathology table (free text).

``Tumor_Histological_Type`` is free text per slide ("Acinar adenocarcinoma", "Adenocarcinoma,
70% solid subtype, 30% acinic subtype", "Mixed adenocarcinoma, acinar and papillary
sub-types", ...). It is reduced to one pattern per case with the same names as the cBioPortal
labels (``xaiwsi.preprocessing.cptac.patterns``):
- a single named pattern, or "<pattern> predominant" -> that pattern;
- percentages -> the pattern with the largest share;
- other histologies (adenosquamous, squamous, colloid, enteric, neuroendocrine) -> kept as
  their own label;
- mixed without percentages or a predominant pattern, "Adenocarcinoma" without subtype,
  missing -> no label (left out).
Only tumour slides are used; a case whose slides disagree is an error. cBioPortal is the
curated reference; this label is a second, noisier source.

    uv run python -m xaiwsi.preprocessing.cptac.tcia_patterns \\
        --table data/raw/cptac-luad-clinical/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv \\
        --output data/processed/cptac-luad/metadata/growth_patterns_tcia.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PATTERNS = "micropapillary|papillary|acinar|solid|lepidic|invasive mucinous|mucinous"
OTHER_HISTOLOGIES = ["adenosquamous", "squamous", "colloid", "enteric", "neuroendocrine"]


def _name(pattern: str) -> str:
    return "invasive-mucinous" if "mucinous" in pattern else pattern


def normalize(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    s = text.lower().replace("acinic", "acinar").replace("caricnoma", "carcinoma")
    shares = re.findall(rf"(\d+)%\s*({PATTERNS})", s)
    shares += [(p, n) for n, p in re.findall(rf"({PATTERNS})[^,(]*?\((\d+)%\)", s)]
    if shares:
        return _name(max(shares, key=lambda x: int(x[0]))[1])
    for histology in OTHER_HISTOLOGIES:
        if histology in s:
            return histology
    predominant = re.search(rf"({PATTERNS})\s+predominant", s)
    named = set(re.findall(PATTERNS, s))
    if predominant and len(re.findall(r"predominant", s)) == 1 and " and " not in s:
        return _name(predominant.group(1))
    if len(named) == 1 and "mixed" not in s:
        return _name(named.pop())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--column", default="dominant_pattern", help="label column name in the output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    table = table[table["Specimen_Type"] == "tumor_tissue"].copy()
    table["label"] = table["Tumor_Histological_Type"].map(normalize)

    per_case = table.groupby("Case_ID").agg(
        label=("label", lambda v: sorted(set(v.dropna()))),
        raw=("Tumor_Histological_Type", lambda v: " | ".join(sorted(set(v.dropna())))),
    )
    conflicting = per_case[per_case["label"].map(len) > 1]
    if not conflicting.empty:
        raise SystemExit(f"Cases whose tumour slides disagree:\n{conflicting}")

    out = pd.DataFrame(
        {
            "case_id": per_case.index,
            args.column: per_case["label"].map(lambda v: v[0] if v else None).to_numpy(),
            "raw": per_case["raw"].to_numpy(),
            "source": f"TCIA {args.table.name} Tumor_Histological_Type",
        }
    ).sort_values("case_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out[args.column].value_counts(dropna=False).to_string())
    print(f"{len(out)} cases -> {args.output}")


if __name__ == "__main__":
    main()
