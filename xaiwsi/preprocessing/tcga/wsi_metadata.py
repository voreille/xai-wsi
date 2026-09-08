from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_TCGA_SLIDE_RE = re.compile(
    r"^(?P<case_id>TCGA-[^-]+-[^-]+)"
    r"-(?P<sample_vial>[^-]+)"
    r"-(?P<portion>[^-]+)"
    r"-(?P<slide_designator>[^.]+)"
    r"(?:\.(?P<source_suffix>.+))?$",
    re.IGNORECASE,
)


def parse_tcga_slide_id(stem: str) -> dict[str, str | None]:
    match = _TCGA_SLIDE_RE.match(stem)

    if match is None:
        raise ValueError(f"Could not parse TCGA slide filename: {stem}")

    data = match.groupdict()

    case_id = data["case_id"]
    sample_vial = data["sample_vial"]

    sample_type_code = (
        sample_vial[:2] if len(sample_vial) >= 2 and sample_vial[:2].isdigit() else None
    )

    slide_barcode = (
        f"{case_id}-{sample_vial}-{data['portion']}-{data['slide_designator']}"
    )

    return {
        "slide_id": stem,
        "slide_barcode": slide_barcode,
        "case_id": case_id,
        "sample_id": f"{case_id}-{sample_vial}",
        "sample_type_code": sample_type_code,
        "vial": sample_vial[2:] or None,
        "portion": data["portion"],
        "slide_designator": data["slide_designator"],
        "source_suffix": data["source_suffix"],
    }


def build_wsi_mapping(
    slides_root: str | Path,
    *,
    rglob: str = "*DX*.svs",
) -> pd.DataFrame:
    slides_root = Path(slides_root)

    rows = []

    for path in sorted(slides_root.rglob(rglob)):
        metadata = parse_tcga_slide_id(path.stem)

        rows.append(
            {
                **metadata,
                "svs_filename": path.name,
                "svs_relpath": str(path.relative_to(slides_root)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["case_id", "sample_id", "slide_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slides-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rglob",
        default="*DX*.svs",
    )

    args = parser.parse_args()

    df = build_wsi_mapping(
        args.slides_root,
        rglob=args.rglob,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(args.output, index=False)

    print(f"Slides: {len(df)}")
    print(f"Cases:  {df['case_id'].nunique()}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
