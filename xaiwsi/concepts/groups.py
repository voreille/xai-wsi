"""Compartment-level slide features from concept bags, with concepts attributed to tissue groups.

Concepts get the dominant tissue group of their profile on the reference cohort
(``xaiwsi.concepts.tissue`` ``concepts.csv``; tissue probe outputs only, no RNA). Per slide, from
the concept bag ā (``slides_mean.parquet``), writes next to it:

- ``slides_groupsum.parquet``: summed ā per group, an SAE version of the composition shares;
- ``slides_mean_tissue.parquet``: ā without the concepts of ``--drop-groups`` (default background
  and other), to test whether those concepts carry the bag's signal;
- ``slides_within.parquet``: within each group of ``--within-groups``, ā of its concepts divided by
  the group's sum: the make-up of a compartment independent of how much of it the slide has.

    uv run python -m xaiwsi.concepts.groups --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad cptac-luad --sae tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --profiles outputs/concept_tissue/luad_valpick/concepts.csv --reference tcga-luad

Probe with ``cohort_probe --features table:concepts/<sae>/slides_<kind>.parquet``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xaiwsi.concepts.bags import embedding_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--profiles", type=Path, required=True, help="xaiwsi.concepts.tissue concepts.csv")
    parser.add_argument("--reference", required=True, help="cohort whose profiles assign the concepts")
    parser.add_argument("--drop-groups", nargs="+", default=["background", "other"])
    parser.add_argument("--within-groups", nargs="+", default=["tumour", "stroma", "immune", "necrosis"])
    args = parser.parse_args()

    profiles = pd.read_csv(args.profiles)
    profiles = profiles[profiles["cohort"] == args.reference].set_index("concept")
    groups = profiles["dominant"]
    print(groups.value_counts().to_string())
    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        bag_dir = embedding_dir.parent.parent / "concepts" / args.sae
        bags = pd.read_parquet(bag_dir / "slides_mean.parquet")
        columns = np.array([int(c.removeprefix("c")) for c in bags.columns])
        assigned = groups.loc[columns].to_numpy()
        groupsum = pd.DataFrame(
            {g: bags.loc[:, assigned == g].sum(axis=1) for g in sorted(set(assigned))}, index=bags.index
        )
        groupsum.to_parquet(bag_dir / "slides_groupsum.parquet")
        bags.loc[:, ~np.isin(assigned, args.drop_groups)].to_parquet(bag_dir / "slides_mean_tissue.parquet")
        within = []
        for g in args.within_groups:
            part = bags.loc[:, assigned == g]
            within.append(part.div(part.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).add_prefix(f"{g}_"))
        pd.concat(within, axis=1).to_parquet(bag_dir / "slides_within.parquet")
        print(f"[{cohort}] {bag_dir}: groupsum {groupsum.shape[1]}, without {args.drop_groups} "
              f"{(~np.isin(assigned, args.drop_groups)).sum()}, within {sum(len(w.columns) for w in within)} columns")


if __name__ == "__main__":
    main()
