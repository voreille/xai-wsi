"""Concept bags: apply a trained TopK SAE (``xaiwsi.concepts.train_sae``) to every slide of the
embedding stores and pool its codes per slide.

Per slide and concept k: ``mean`` = mean activation over tiles (ā_k, zeros included) and
``frac`` = fraction of tiles where k is among the top-k active latents. The slide mean of the
tiles is recovered linearly from ā: ``mean_i x_i ≈ (ā D + b_pre) * scale + input_mean``;
``slides.parquet`` records how well per slide (``mean_rel_error``, relative to the slide mean's
distance from the SAE's training mean), next to the tile-level FVU against that training mean
(``tile_fvu``, a shift diagnostic for test cohorts).

    uv run python -m xaiwsi.concepts.bags --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad tcga-lusc cptac-luad cptac-lusc \\
        --sae data/processed/sae/h-optimus-1-fp16/tcga-luad_k4096_top32

Cohorts are named as in ``--config`` or given as ``<name>=<embedding dir>``, like
``xaiwsi.tissue.apply``. Writes under ``<cohort dir>/concepts/<sae name>/``:
``slides_mean.parquet`` and ``slides_frac.parquet`` (slide_id x ``c<k>``), ``slides.parquet``
(n_tiles, mean_rel_error, tile_fvu), ``sae.json`` (SAE path and settings) and, unless
``--no-tiles``, ``tiles/<slide_id>.npz`` with the per-tile top-k ``indices`` (int16/int32) and
``values`` (float16) in store order, for concept maps and top tiles.
``xaiwsi.evaluation.cohort_probe --features table:concepts/<sae name>/slides_mean.parquet``
reads the bags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.models.sae import TopKSAE, load_sae
from xaiwsi.training.run_config import data_args


def embedding_dirs(config: Path | None, cohorts: list[str]) -> dict[str, Path]:
    """``<name>`` from a ``CohortPathwayDataModule`` config, or ``<name>=<embedding dir>``."""
    dirs = {}
    named = [c for c in cohorts if "=" not in c]
    if named:
        if config is None:
            raise SystemExit(f"--config is needed for cohorts given by name: {named}")
        data = data_args(yaml.safe_load(config.read_text()))
        specs = {s["name"]: CohortSpec(**s) for s in data.get("train_cohorts", []) + data.get("test_cohorts", [])}
        unknown = set(named) - set(specs)
        if unknown:
            raise SystemExit(f"Cohorts not in {config}: {sorted(unknown)}")
        for name in named:
            dirs[name] = specs[name].resolve(data.get("processed_root"), data.get("embedding", "h-optimus-1")).embedding_dir
    for item in cohorts:
        if "=" in item:
            name, path = item.split("=", 1)
            dirs[name] = Path(path)
    return dirs


@torch.no_grad()
def encode_slide(model: TopKSAE, tiles: np.ndarray, chunk: int, device: torch.device) -> dict:
    n = len(tiles)
    sums = torch.zeros(model.n_latents, device=device)
    counts = torch.zeros(model.n_latents, device=device)
    x_sum = torch.zeros(model.input_dim, device=device)
    sse = sst = 0.0
    all_values, all_indices = [], []
    for start in range(0, n, chunk):
        x = torch.as_tensor(tiles[start : start + chunk], device=device).float()
        x_norm = model.normalize(x)
        values, indices = model.topk(model.pre_activations(x_norm))
        sums.index_add_(0, indices.flatten(), values.flatten())
        counts.index_add_(0, indices.flatten(), (values > 0).float().flatten())
        x_hat = model.decode_sparse(values, indices)
        sse += (x_norm - x_hat).pow(2).sum().item()
        sst += x_norm.pow(2).sum().item()
        x_sum += x.sum(0)
        all_values.append(values.half().cpu())
        all_indices.append(indices.cpu())
    mean_act = sums / n
    x_mean = x_sum / n
    recovered = (mean_act @ model.decoder + model.pre_bias) * model.input_scale + model.input_mean
    rel_error = ((x_mean - recovered).norm() / (x_mean - model.input_mean).norm()).item()
    return {
        "mean": mean_act.cpu().numpy(),
        "frac": (counts / n).cpu().numpy(),
        "mean_rel_error": rel_error,
        "tile_fvu": sse / sst,
        "values": torch.cat(all_values).numpy(),
        "indices": torch.cat(all_indices).numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="config whose cohorts are named in --cohorts")
    parser.add_argument("--cohorts", nargs="+", required=True, help="<name> from --config, or <name>=<embedding dir>")
    parser.add_argument("--sae", type=Path, required=True, help="train_sae output dir (with sae.pt)")
    parser.add_argument("--name", default=None, help="output folder name (default: SAE dir name)")
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--no-tiles", action="store_true", help="do not write per-tile codes")
    args = parser.parse_args()
    name = args.name or args.sae.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sae(args.sae / "sae.pt").to(device)
    index_dtype = np.int16 if model.n_latents <= np.iinfo(np.int16).max else np.int32
    columns = [f"c{k}" for k in range(model.n_latents)]

    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        out_dir = embedding_dir.parent.parent / "concepts" / name
        if not args.no_tiles:
            (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        store = build_embedding_store_from_dir(root_dir=embedding_dir)
        means, fracs, rows = {}, {}, []
        for slide_id in tqdm(sorted(store.slide_ids()), desc=cohort):
            tiles = store.load(slide_id)[0]
            if len(tiles) == 0:
                continue
            bag = encode_slide(model, tiles, args.chunk, device)
            means[slide_id], fracs[slide_id] = bag["mean"], bag["frac"]
            rows.append(
                {
                    "slide_id": slide_id,
                    "n_tiles": len(tiles),
                    "mean_rel_error": bag["mean_rel_error"],
                    "tile_fvu": bag["tile_fvu"],
                }
            )
            if not args.no_tiles:
                np.savez(
                    out_dir / "tiles" / f"{slide_id}.npz",
                    indices=bag["indices"].astype(index_dtype),
                    values=bag["values"],
                )
        for kind, table in (("mean", means), ("frac", fracs)):
            frame = pd.DataFrame.from_dict(table, orient="index", columns=columns).astype(np.float32)
            frame.index.name = "slide_id"
            frame.to_parquet(out_dir / f"slides_{kind}.parquet")
        slides = pd.DataFrame(rows).set_index("slide_id")
        slides.to_parquet(out_dir / "slides.parquet")
        (out_dir / "sae.json").write_text(
            json.dumps({"sae": str(args.sae), "config": model.config(), "embedding_dir": str(embedding_dir)}, indent=2)
        )
        active = (pd.DataFrame.from_dict(fracs, orient="index") > 0).any(axis=0).mean()
        print(
            f"[{cohort}] {len(slides)} slides -> {out_dir}; median mean_rel_error "
            f"{slides['mean_rel_error'].median():.3f}, median tile FVU {slides['tile_fvu'].median():.3f}, "
            f"concepts active in any slide {active:.3f}"
        )


if __name__ == "__main__":
    main()
