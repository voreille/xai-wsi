from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_CPTAC_CASE_RE = re.compile(r"(C3[NL]-\d+)", re.IGNORECASE)


def extract_cptac_case_id(filename: str) -> str:
    """Extract a CPTAC case ID such as C3L-xxxxx or C3N-xxxxx."""
    match = _CPTAC_CASE_RE.search(filename)
    if match is None:
        raise ValueError(
            f"Could not extract CPTAC case ID from: {filename}"
        )
    return match.group(1).upper()


def build_wsi_mapping(
    slides_root: str | Path,
    *,
    rglob: str = "*.svs",
) -> pd.DataFrame:
    """Build a conservative CPTAC WSI table.

    Only the case ID is inferred from filenames. More specific sample/slide
    biospecimen IDs should be added later from authoritative CPTAC metadata
    rather than guessed from filename structure.
    """
    slides_root = Path(slides_root)
    rows = []

    for path in sorted(slides_root.rglob(rglob)):
        rows.append(
            {
                "slide_id": path.stem,
                "case_id": extract_cptac_case_id(path.name),
                "svs_filename": path.name,
                "svs_relpath": str(path.relative_to(slides_root)),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["case_id", "slide_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CPTAC WSI metadata from filenames."
    )
    parser.add_argument("--slides-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rglob", default="*.svs")
    args = parser.parse_args()

    df = build_wsi_mapping(args.slides_root, rglob=args.rglob)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Slides:  {len(df)}")
    print(f"Cases:   {df['case_id'].nunique() if not df.empty else 0}")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
