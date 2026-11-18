"""Re-pool concept bags from the saved per-tile codes, optionally without artifact tiles.

Reads ``tiles/<slide_id>.npz`` of an existing bag folder (``xaiwsi.concepts.bags``; per-tile
top-k ``indices`` and float16 ``values`` in store order) and the tile-QC artifact probabilities
(``<cohort dir>/tissue/<store>/tile-qc/artifact_prob.npz``, same order), and writes
``slides_mean.parquet`` / ``slides_frac.parquet`` like ``bags`` plus ``slides.parquet``
(``n_tiles``, ``n_kept``, ``flagged_frac``: tiles at or above the threshold, before the
``keep_mask`` fallback that keeps every tile of a slide with fewer than 16 left).

Without ``--tile-qc-threshold`` all tiles are pooled: the same bag as ``bags`` up to the
float16 rounding of the saved codes, so it is the matched baseline for the masked bag.

    uv run python -m xaiwsi.concepts.qc_bags --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad cptac-luad --bags tcga-luad+tcga-lusc_k1024_top64_s0 \\
        --tile-qc-threshold 0.5 --name tcga-luad+tcga-lusc_k1024_top64_s0_qc0.5

Output: ``outputs/concept-bags/<store>/<name>/<cohort>/``; probe with
``cohort_probe --features bags:<name>/slides_mean``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from xaiwsi import paths
from xaiwsi.concepts.bags import embedding_dirs
from xaiwsi.qc.mask import keep_mask, load_artifact_probs


def pool(indices: np.ndarray, values: np.ndarray, keep: np.ndarray, n_latents: int) -> tuple[np.ndarray, np.ndarray]:
    """(mean activation, active fraction) per concept over the kept tiles; zeros included."""
    idx = indices[keep].astype(np.int64).ravel()
    val = values[keep].astype(np.float32).ravel()
    n = int(keep.sum())
    mean = np.bincount(idx, weights=val, minlength=n_latents) / n
    frac = np.bincount(idx, weights=(val > 0).astype(np.float32), minlength=n_latents) / n
    return mean.astype(np.float32), frac.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="config whose cohorts are named in --cohorts")
    parser.add_argument("--cohorts", nargs="+", required=True, help="<name> from --config, or <name>=<embedding dir>")
    parser.add_argument("--bags", required=True, help="existing bag folder name (with tiles/)")
    parser.add_argument("--tile-qc-threshold", type=float, help="drop tiles with artifact probability >= this")
    parser.add_argument("--name", required=True, help="output folder name")
    args = parser.parse_args()
    if args.name == args.bags:
        raise SystemExit("--name must differ from --bags (it would overwrite the source bags).")

    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        src = paths.bag_dir("concept", embedding_dir, args.bags)
        sae_info = json.loads((src / "sae.json").read_text())
        n_latents = int(sae_info["config"]["n_latents"])
        probs = load_artifact_probs(embedding_dir) if args.tile_qc_threshold is not None else None
        out_dir = paths.bag_dir("concept", embedding_dir, args.name)
        out_dir.mkdir(parents=True, exist_ok=True)

        means, fracs, rows = {}, {}, []
        for path in tqdm(sorted((src / "tiles").glob("*.npz")), desc=cohort):
            slide_id = path.stem
            with np.load(path) as codes:
                indices, values = codes["indices"], codes["values"]
            n = len(indices)
            if probs is None:
                keep, flagged = np.ones(n, dtype=bool), 0.0
            else:
                prob = probs[slide_id]
                if len(prob) != n:
                    raise SystemExit(f"{cohort} {slide_id}: {len(prob)} QC probabilities, {n} tile codes.")
                keep = keep_mask(prob, args.tile_qc_threshold).astype(bool)
                flagged = float((prob >= args.tile_qc_threshold).mean())
            means[slide_id], fracs[slide_id] = pool(indices, values, keep, n_latents)
            rows.append({"slide_id": slide_id, "n_tiles": n, "n_kept": int(keep.sum()), "flagged_frac": flagged})

        columns = [f"c{k}" for k in range(n_latents)]
        for kind, table in (("mean", means), ("frac", fracs)):
            frame = pd.DataFrame.from_dict(table, orient="index", columns=columns)
            frame.index.name = "slide_id"
            frame.to_parquet(out_dir / f"slides_{kind}.parquet")
        slides = pd.DataFrame(rows).set_index("slide_id")
        slides.to_parquet(out_dir / "slides.parquet")
        (out_dir / "sae.json").write_text(
            json.dumps(
                {**sae_info, "pooled_from": str(src), "tile_qc_threshold": args.tile_qc_threshold}, indent=2
            )
        )
        print(
            f"[{cohort}] {len(slides)} slides -> {out_dir}; flagged tiles mean "
            f"{slides['flagged_frac'].mean():.3f}, median {slides['flagged_frac'].median():.3f}; "
            f"kept {slides['n_kept'].sum() / slides['n_tiles'].sum():.3f} of tiles"
        )


if __name__ == "__main__":
    main()
