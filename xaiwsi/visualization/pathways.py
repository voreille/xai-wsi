from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.training.pathway_module import PathwayTraining


def _as_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _resolve_dataset_paths(
    config: dict[str, Any],
    dataset: str,
    *,
    embedding_dir: Path | None = None,
    slides_root_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Resolve embedding and WSI paths from the saved Lightning config.

    Explicit CLI values override the config. This makes old run configs that
    don't contain *_slides_root_dir usable with --slides-root-dir.
    """
    data = config.get("data", {})

    embedding_key = f"{dataset}_embedding_dir"
    slides_key = f"{dataset}_slides_root_dir"

    if embedding_dir is None:
        value = data.get(embedding_key)
        if value is None:
            raise click.ClickException(
                f"{embedding_key!r} is missing from the run config. "
                f"Pass --embedding-dir explicitly."
            )
        embedding_dir = _as_path(value)

    if slides_root_dir is None:
        value = data.get(slides_key)
        if value is None:
            raise click.ClickException(
                f"{slides_key!r} is missing from this run config. "
                f"This is expected for older runs; pass "
                f"--slides-root-dir explicitly."
            )
        slides_root_dir = _as_path(value)

    return embedding_dir, slides_root_dir


def _resolve_hallmark_gmt(
    config: dict[str, Any],
    hallmark_gmt: Path | None,
) -> Path:
    if hallmark_gmt is not None:
        return hallmark_gmt

    value = config.get("data", {}).get("hallmark_gmt")

    if value is None:
        raise click.ClickException(
            "hallmark_gmt is missing from the run config. "
            "Pass --hallmark-gmt explicitly."
        )

    return _as_path(value)


def predict_slide(
    model: PathwayTraining,
    embedding_store,
    slide_id: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Run tile-level pathway inference for one WSI."""
    embeddings, coords, attrs = embedding_store.load(slide_id)

    embeddings = (
        torch.from_numpy(np.asarray(embeddings)).float().unsqueeze(0).to(device)
    )

    with torch.inference_mode():
        output = model(embeddings)

    tile_scores = output.get("tile_scores")

    if tile_scores is None:
        raise RuntimeError(
            f"{model.predictor.__class__.__name__} does not produce "
            "tile-level pathway scores."
        )

    n_tiles = tile_scores.shape[1]

    tile_weights = output.get("tile_weights")

    # TileMLPMean: uniform weighting.
    if tile_weights is None:
        tile_weights = torch.full(
            (1, n_tiles, 1),
            1.0 / n_tiles,
            dtype=tile_scores.dtype,
            device=tile_scores.device,
        )

    tile_contributions = tile_weights * tile_scores

    result = {
        "pred": output["pred"].squeeze(0).cpu().numpy(),
        "tile_scores": tile_scores.squeeze(0).cpu().numpy(),
        "tile_weights": tile_weights.squeeze(0).cpu().numpy(),
        "tile_contributions": (tile_contributions.squeeze(0).cpu().numpy()),
    }

    tile_gates = output.get("tile_gates")
    if tile_gates is not None:
        result["tile_gates"] = tile_gates.squeeze(0).cpu().numpy()

    return result, np.asarray(coords), attrs


def rasterize_tile_values_to_thumbnail(
    coords: np.ndarray,
    values: np.ndarray,
    attrs: dict,
    thumbnail: Image.Image,
) -> np.ndarray:
    """Rasterize one scalar value per tile directly to thumbnail coordinates.

    Each tile is drawn as its full patch footprint, not as a single cell on a
    coarse grid. This avoids the spatial shift introduced by coarse-grid
    resizing.
    """
    coord_space = str(attrs["coord_space"])
    if coord_space != "level0":
        raise ValueError(f"Expected level0 coordinates, got {coord_space!r}.")

    slide_w, slide_h = map(int, np.asarray(attrs["level0_dim"]))
    patch_size = int(attrs["patch_size"])

    thumb_w, thumb_h = thumbnail.size

    sx = thumb_w / slide_w
    sy = thumb_h / slide_h

    accum = np.zeros((thumb_h, thumb_w), dtype=np.float64)
    counts = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    for (x, y), v in zip(coords, values, strict=False):
        if not np.isfinite(v):
            continue

        x = float(x)
        y = float(y)

        x0 = int(np.floor(x * sx))
        y0 = int(np.floor(y * sy))
        x1 = int(np.ceil((x + patch_size) * sx))
        y1 = int(np.ceil((y + patch_size) * sy))

        # Clip to image bounds
        x0 = max(0, min(x0, thumb_w))
        x1 = max(0, min(x1, thumb_w))
        y0 = max(0, min(y0, thumb_h))
        y1 = max(0, min(y1, thumb_h))

        if x1 <= x0 or y1 <= y0:
            continue

        accum[y0:y1, x0:x1] += float(v)
        counts[y0:y1, x0:x1] += 1.0

    heatmap = np.full((thumb_h, thumb_w), np.nan, dtype=np.float32)
    mask = counts > 0
    heatmap[mask] = (accum[mask] / counts[mask]).astype(np.float32)

    return heatmap


def load_thumbnail(
    slide_path: Path,
    max_size: int,
) -> Image.Image:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required for WSI thumbnails.") from exc

    slide = openslide.OpenSlide(str(slide_path))

    try:
        thumbnail = slide.get_thumbnail((max_size, max_size)).convert("RGB")
    finally:
        slide.close()

    return thumbnail


def render_heatmap_overlay(
    thumbnail: Image.Image,
    heatmap: np.ndarray,
    *,
    kind: str,
    alpha: float,
    percentile: float = 99.0,
) -> Image.Image:
    valid = np.isfinite(heatmap)

    if not valid.any():
        return thumbnail.copy()

    values = heatmap[valid]

    if kind in {"score", "contribution"}:
        vmax = float(np.percentile(np.abs(values), percentile))
        if vmax <= 0:
            vmax = 1.0

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = matplotlib.colormaps["coolwarm"]

    elif kind in {"weight", "gate"}:
        vmax = float(np.percentile(values, percentile))
        if vmax <= 0:
            vmax = 1.0

        norm = Normalize(vmin=0.0, vmax=vmax)
        cmap = matplotlib.colormaps["viridis"]

    else:
        raise ValueError(f"Unknown heatmap kind: {kind}")

    rgba = np.zeros((*heatmap.shape, 4), dtype=np.float32)
    rgba[valid] = cmap(norm(heatmap[valid]))

    # transparent background where no tiles exist
    rgba[..., 3] = valid.astype(np.float32) * alpha

    heatmap_image = Image.fromarray(
        (rgba * 255).astype(np.uint8),
        mode="RGBA",
    )

    return Image.alpha_composite(
        thumbnail.convert("RGBA"),
        heatmap_image,
    )


def _safe_filename(name: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        name,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Saved Lightning run config.yaml.",
)
@click.option(
    "--checkpoint",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--dataset",
    type=click.Choice(
        ["tcga", "cptac"],
        case_sensitive=False,
    ),
    required=True,
)
@click.option(
    "--slide-id",
    required=True,
)
@click.option(
    "--pathway",
    "pathways",
    multiple=True,
    help=(
        "Hallmark to render. May be repeated. Example: --pathway HALLMARK_E2F_TARGETS"
    ),
)
@click.option(
    "--all-pathways",
    is_flag=True,
    help="Render all Hallmarks.",
)
@click.option(
    "--map-type",
    "map_types",
    type=click.Choice(["score", "contribution", "weight", "gate"]),
    multiple=True,
    default=("contribution",),
    show_default=True,
)
@click.option(
    "--embedding-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Override embedding directory from the run config.",
)
@click.option(
    "--slides-root-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        path_type=Path,
    ),
    default=None,
    help=(
        "Override WSI root. Also provides backward compatibility "
        "for old run configs without *_slides_root_dir."
    ),
)
@click.option(
    "--hallmark-gmt",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    default=None,
    help="Override Hallmark GMT from the run config.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    default=Path("outputs/pathway-maps"),
    show_default=True,
)
@click.option(
    "--thumbnail-size",
    type=int,
    default=2048,
    show_default=True,
)
@click.option(
    "--alpha",
    type=click.FloatRange(0.0, 1.0),
    default=0.55,
    show_default=True,
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
)
def main(
    config_path: Path,
    checkpoint: Path,
    dataset: str,
    slide_id: str,
    pathways: tuple[str, ...],
    all_pathways: bool,
    map_types: tuple[str, ...],
    embedding_dir: Path | None,
    slides_root_dir: Path | None,
    hallmark_gmt: Path | None,
    output_dir: Path,
    thumbnail_size: int,
    alpha: float,
    device: str,
) -> None:
    dataset = dataset.lower()

    if all_pathways and pathways:
        raise click.ClickException("Use either --all-pathways or --pathway, not both.")

    if not all_pathways and not pathways:
        raise click.ClickException(
            "Specify at least one --pathway or use --all-pathways."
        )

    config = _load_config(config_path)

    embedding_dir, slides_root_dir = _resolve_dataset_paths(
        config,
        dataset,
        embedding_dir=embedding_dir,
        slides_root_dir=slides_root_dir,
    )

    hallmark_gmt = _resolve_hallmark_gmt(
        config,
        hallmark_gmt,
    )

    hallmarks = load_hallmarks(hallmark_gmt)
    pathway_names = list(hallmarks.keys())

    if all_pathways:
        selected_pathways = pathway_names
    else:
        selected_pathways = list(pathways)

    unknown = set(selected_pathways) - set(pathway_names)

    if unknown:
        raise click.ClickException("Unknown pathway(s): " + ", ".join(sorted(unknown)))

    embedding_store = build_embedding_store_from_dir(root_dir=embedding_dir)

    available_slide_ids = set(embedding_store.slide_ids())

    if slide_id not in available_slide_ids:
        raise click.ClickException(
            f"Slide {slide_id!r} is not present in {embedding_dir}."
        )

    torch_device = torch.device(device)

    model = PathwayTraining.load_from_checkpoint(
        checkpoint,
        map_location=torch_device,
    )
    model = model.to(torch_device)
    model.eval()

    result, coords, attrs = predict_slide(
        model,
        embedding_store,
        slide_id,
        torch_device,
    )

    if result["tile_scores"].shape[1] != len(pathway_names):
        raise RuntimeError(
            "Model output dimension does not match the Hallmark GMT: "
            f"{result['tile_scores'].shape[1]} vs "
            f"{len(pathway_names)}."
        )

    relative_wsi_path = Path(str(attrs["relative_wsi_path"]))

    slide_path = slides_root_dir / relative_wsi_path

    if not slide_path.exists():
        raise click.ClickException(
            f"WSI not found:\n{slide_path}\n\n"
            f"slides_root_dir={slides_root_dir}\n"
            f"relative_wsi_path={relative_wsi_path}"
        )

    slide_output_dir = output_dir / _safe_filename(slide_id)
    slide_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thumbnail = load_thumbnail(
        slide_path,
        thumbnail_size,
    )
    thumbnail.save(slide_output_dir / "thumbnail.png")

    predictions = pd.DataFrame(
        {
            "pathway": pathway_names,
            "prediction": result["pred"],
        }
    )
    predictions.to_csv(
        slide_output_dir / "predictions.csv",
        index=False,
    )

    npz_data = {
        "coords": coords,
        "pred": result["pred"],
        "tile_scores": result["tile_scores"],
        "tile_weights": result["tile_weights"],
        "tile_contributions": result["tile_contributions"],
        "pathway_names": np.asarray(
            pathway_names,
            dtype=str,
        ),
    }

    if "tile_gates" in result:
        npz_data["tile_gates"] = result["tile_gates"]

    np.savez_compressed(
        slide_output_dir / "tile_outputs.npz",
        **npz_data,
    )

    metadata = {
        "slide_id": slide_id,
        "dataset": dataset,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "embedding_dir": str(embedding_dir),
        "slides_root_dir": str(slides_root_dir),
        "slide_path": str(slide_path),
        "attrs": _jsonable(attrs),
    }

    with (slide_output_dir / "metadata.json").open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Pathway-independent maps
    # ---------------------------------------------------------

    if "weight" in map_types:
        weight_grid = rasterize_tile_values_to_thumbnail(
            coords, result["tile_weights"][:, 0], attrs, thumbnail
        )

        overlay = render_heatmap_overlay(
            thumbnail,
            weight_grid,
            kind="weight",
            alpha=alpha,
        )

        overlay.save(slide_output_dir / "tile_weights.png")

    if "gate" in map_types:
        if "tile_gates" not in result:
            click.echo("Model has no tile_gates; skipping gate map.")
        else:
            gate_grid = rasterize_tile_values_to_thumbnail(
                coords, result["tile_gates"][:, 0], attrs, thumbnail
            )

            overlay = render_heatmap_overlay(
                thumbnail,
                gate_grid,
                kind="gate",
                alpha=alpha,
            )

            overlay.save(slide_output_dir / "tile_gates.png")

    # ---------------------------------------------------------
    # Pathway-specific maps
    # ---------------------------------------------------------

    for pathway in selected_pathways:
        pathway_idx = pathway_names.index(pathway)
        filename = _safe_filename(pathway)

        click.echo(f"{pathway}: {result['pred'][pathway_idx]:+.4f}")

        if "score" in map_types:
            score_grid = rasterize_tile_values_to_thumbnail(
                coords, result["tile_scores"][:, pathway_idx], attrs, thumbnail
            )

            overlay = render_heatmap_overlay(
                thumbnail,
                score_grid,
                kind="score",
                alpha=alpha,
            )

            overlay.save(slide_output_dir / f"{filename}_score.png")

        if "contribution" in map_types:
            contribution_grid = rasterize_tile_values_to_thumbnail(
                coords, result["tile_contributions"][:, pathway_idx], attrs, thumbnail
            )

            overlay = render_heatmap_overlay(
                thumbnail,
                contribution_grid,
                kind="contribution",
                alpha=alpha,
            )

            overlay.save(slide_output_dir / f"{filename}_contribution.png")

    click.echo(f"\nSaved to {slide_output_dir}")


if __name__ == "__main__":
    main()
