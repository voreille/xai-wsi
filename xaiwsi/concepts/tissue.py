"""Tissue profile of each SAE concept from an applied tissue probe (IGNITE by default).

Over all tiles of each cohort, per concept k: the activation-weighted share of each tissue group
(``sum_t a_tk p_tg / sum_t a_tk``, with ``p_tg`` the probe's soft fraction of group g on tile t,
groups as ``xaiwsi.evaluation.tile_tint.TISSUE_GROUPS`` plus ``other``), its enrichment over the
cohort's tile share of that group, and the tile-level Pearson correlation of the activation with
each group fraction. The dominant group and its share (purity) attribute the concept to a
compartment; the correlation of a concept's group shares between cohorts says whether it fires
on the same tissue in both.

    uv run python -m xaiwsi.concepts.tissue --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad cptac-luad --sae tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --output-dir outputs/concept_tissue/luad_valpick

Writes ``concepts.csv`` (one row per concept and cohort), ``consistency.csv`` (per concept, between
the first cohort and each other one) and ``summary.md`` (concepts per dominant group and purity).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from xaiwsi.concepts.bags import embedding_dirs
from xaiwsi.concepts.viz import dense_codes
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.tile_tint import TISSUE_GROUPS


def group_matrix(classes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Classes x groups 0/1 matrix; classes in no group go to ``other``."""
    groups = [*TISSUE_GROUPS, "other"]
    matrix = np.zeros((len(classes), len(groups)))
    for c, name in enumerate(classes):
        hit = [g for g, members in enumerate(TISSUE_GROUPS.values()) if name in members]
        matrix[c, hit[0] if hit else len(groups) - 1] = 1.0
    return matrix, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True, help="first one is the reference for consistency")
    parser.add_argument("--sae", required=True, help="bag folder name under <cohort dir>/concepts/ (with tiles/)")
    parser.add_argument("--tissue-probe", default="ignite")
    parser.add_argument("--purity", type=float, default=0.6, help="dominant-group share to call a concept pure")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, shares = [], {}
    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        cohort_dir = embedding_dir.parent.parent
        bag_dir = cohort_dir / "concepts" / args.sae
        tissue_dir = cohort_dir / "tissue" / args.tissue_probe
        classes = json.loads((tissue_dir / "probe.json").read_text())["classes"]
        to_groups, groups = group_matrix(classes)
        slide_ids = list(pd.read_parquet(bag_dir / "slides_mean.parquet").index.astype(str))
        n_latents = pd.read_parquet(bag_dir / "slides_mean.parquet").shape[1]
        sum_ap = np.zeros((n_latents, len(groups)))
        sum_a, sum_a2 = np.zeros(n_latents), np.zeros(n_latents)
        sum_p, sum_p2 = np.zeros(len(groups)), np.zeros(len(groups))
        n = 0
        for slide_id in tqdm(slide_ids, desc=cohort):
            a = dense_codes(bag_dir, slide_id, n_latents).astype(np.float64)
            p = np.load(tissue_dir / "tiles" / f"{slide_id}.npy").astype(np.float64) @ to_groups
            if len(p) != len(a):
                raise ValueError(f"{cohort} {slide_id}: {len(p)} tissue rows for {len(a)} tiles")
            sum_ap += a.T @ p
            sum_a += a.sum(axis=0)
            sum_a2 += (a**2).sum(axis=0)
            sum_p += p.sum(axis=0)
            sum_p2 += (p**2).sum(axis=0)
            n += len(a)
        share = sum_ap / np.maximum(sum_a, 1e-12)[:, None]
        base = sum_p / n
        cov = sum_ap / n - np.outer(sum_a / n, base)
        sd_a = np.sqrt(np.maximum(sum_a2 / n - (sum_a / n) ** 2, 0))
        sd_p = np.sqrt(np.maximum(sum_p2 / n - base**2, 0))
        corr = cov / np.maximum(np.outer(sd_a, sd_p), 1e-12)
        if (np.abs(corr) > 1 + 1e-6).any() or (share < -1e-9).any() or (share > 1 + 1e-6).any():
            raise ValueError(f"{cohort}: tissue share or correlation out of range")
        shares[cohort] = share
        dominant = share.argmax(axis=1)
        for k in range(n_latents):
            row = {"cohort": cohort, "concept": k, "mean_activation": sum_a[k] / n,
                   "dominant": groups[dominant[k]], "purity": share[k, dominant[k]]}
            row |= {f"share_{g}": share[k, j] for j, g in enumerate(groups)}
            row |= {f"enrich_{g}": share[k, j] / base[j] for j, g in enumerate(groups)}
            row |= {f"corr_{g}": corr[k, j] for j, g in enumerate(groups)}
            rows.append(row)
        print(f"[{cohort}] {n} tiles; tile shares " + ", ".join(f"{g} {b:.3f}" for g, b in zip(groups, base)))

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "concepts.csv", index=False)
    reference = args.cohorts[0]
    consistency = []
    for cohort in args.cohorts[1:]:
        a, b = shares[reference], shares[cohort]
        a_c, b_c = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
        r = (a_c * b_c).sum(axis=1) / np.maximum(np.linalg.norm(a_c, axis=1) * np.linalg.norm(b_c, axis=1), 1e-12)
        same = a.argmax(axis=1) == b.argmax(axis=1)
        consistency.append(pd.DataFrame({"concept": range(len(r)), "cohort": cohort, "profile_r": r,
                                         "same_dominant": same}))
    consistency = pd.concat(consistency, ignore_index=True) if consistency else pd.DataFrame()
    consistency.to_csv(args.output_dir / "consistency.csv", index=False)

    table["pure"] = table["purity"] >= args.purity
    counts = (
        table.groupby(["cohort", "dominant"], sort=False)
        .agg(concepts=("concept", "size"), pure=("pure", "sum"), median_purity=("purity", "median"))
        .reset_index()
    )
    lines = [markdown_table(counts)]
    if len(consistency):
        summary = consistency.groupby("cohort").agg(
            median_profile_r=("profile_r", "median"), same_dominant=("same_dominant", "mean")
        ).reset_index()
        lines.append(f"Profile consistency with {reference} (per-concept correlation of group shares; share of "
                     f"concepts with the same dominant group):\n\n{markdown_table(summary)}")
    command = "uv run python -m xaiwsi.concepts.tissue " + shlex.join(sys.argv[1:])
    (args.output_dir / "summary.md").write_text(
        f"Tissue profile of SAE concepts ({args.tissue_probe} groups; dominant group = largest activation-weighted "
        f"share, pure = share >= {args.purity}). Generated by:\n\n```bash\n{command}\n```\n\n" + "\n\n".join(lines) + "\n"
    )
    print("\n\n".join(lines))


if __name__ == "__main__":
    main()
