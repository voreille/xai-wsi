"""Panel of tile maps: one row per slide, thumbnail + one score map per pathway.

Reads the per-slide folders written by ``xaiwsi.visualization.pathways`` (``thumbnail.png``,
``tile_outputs.npz``) and redraws the score maps on one shared colour scale, so rows are
comparable. Row labels come from ``--label slide_id=text``.

    uv run python -m xaiwsi.visualization.montage --maps-dir outputs/pathway-maps/cptac-patterns \\
        --slide-id C3N-00203-21 C3L-00001-21 --label 'C3N-00203-21=solid' \\
        --pathway HALLMARK_E2F_TARGETS HALLMARK_COAGULATION --vmax 1.5 --output panel.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from xaiwsi.visualization.pathways import render_heatmap_overlay, rasterize_tile_values_to_thumbnail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--slide-id", nargs="+", required=True)
    parser.add_argument("--pathway", nargs="+", required=True)
    parser.add_argument("--label", nargs="*", default=[], help="slide_id=text row labels")
    parser.add_argument("--vmax", type=float, required=True, help="shared colour scale ±vmax")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = dict(item.split("=", 1) for item in args.label)
    ncols = 1 + len(args.pathway)
    fig, axes = plt.subplots(
        len(args.slide_id), ncols, figsize=(3.2 * ncols, 2.9 * len(args.slide_id)), squeeze=False
    )
    for row, slide_id in enumerate(args.slide_id):
        folder = args.maps_dir / slide_id
        thumbnail = Image.open(folder / "thumbnail.png").convert("RGB")
        outputs = np.load(folder / "tile_outputs.npz")
        attrs = json.loads((folder / "metadata.json").read_text())["attrs"]
        names = list(outputs["pathway_names"])
        axes[row, 0].imshow(thumbnail)
        axes[row, 0].set_ylabel(f"{slide_id}\n{labels.get(slide_id, '')}", fontsize=9)
        for col, pathway in enumerate(args.pathway, start=1):
            k = names.index(pathway)
            grid = rasterize_tile_values_to_thumbnail(
                outputs["coords"], outputs["tile_scores"][:, k], attrs, thumbnail
            )
            overlay = render_heatmap_overlay(thumbnail, grid, kind="score", alpha=args.alpha, vmax=args.vmax)
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f"{pathway.removeprefix('HALLMARK_')}  pred {outputs['pred'][k]:+.2f}", fontsize=8
            )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("H&E", fontsize=8)
    mappable = plt.cm.ScalarMappable(
        norm=TwoSlopeNorm(vmin=-args.vmax, vcenter=0.0, vmax=args.vmax), cmap="coolwarm"
    )
    fig.colorbar(mappable, ax=axes, shrink=0.4, label="tile score (z units)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
