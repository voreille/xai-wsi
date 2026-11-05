"""Apply a tile-composition probe (``xaiwsi.tissue.probe``) to every slide of the embedding stores.

For each cohort named in a ``CohortPathwayDataModule`` config (train and test cohorts),
all slides of its embedding store get per-tile class fractions from the probe (same tile
order as the store). The store's tiles must be at the probe's resolution: the effective
tile mpp (``patch_mpp * patch_size / target_patch_size``) is recorded per slide and slides
off by more than ``--mpp-tolerance`` are reported and skipped.

    uv run python -m xaiwsi.tissue.apply --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad tcga-lusc cptac-luad cptac-lusc \\
        --probe outputs/tissue_probe/ignite/probe.pt --name ignite

Writes under ``<processed_root>/<cohort>/tissue/<name>/``: ``tiles/<slide_id>.npy`` (n_tiles x
K float16 fractions, store order), ``slides.parquet`` (per slide: n_tiles, effective mpp,
mean fraction per class) and ``probe.json`` (probe path, classes, embedding dir).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.data.cohorts import CohortSpec
from xaiwsi.training.run_config import data_args


class CompositionProbe:
    def __init__(self, path: Path, device: torch.device) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.classes: list[str] = state["classes"]
        self.weight = state["weight"].to(device)
        self.bias = state["bias"].to(device)
        self.mean = state["mean"].to(device)
        self.std = state["std"].to(device)
        self.device = device

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = ((x - self.mean) / self.std) @ self.weight.T + self.bias
            return torch.softmax(logits, dim=1).cpu().numpy()


def effective_mpp(attrs: dict) -> float:
    return float(attrs["patch_mpp"]) * int(attrs["patch_size"]) / int(attrs["target_patch_size"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--name", required=True, help="output folder name, e.g. ignite")
    parser.add_argument("--mpp", type=float, default=0.5, help="probe training resolution")
    parser.add_argument("--mpp-tolerance", type=float, default=0.1, help="relative")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = data_args(yaml.safe_load(args.config.read_text()))
    specs = {s["name"]: CohortSpec(**s) for s in data.get("train_cohorts", []) + data.get("test_cohorts", [])}
    unknown = set(args.cohorts) - set(specs)
    if unknown:
        raise SystemExit(f"Cohorts not in {args.config}: {sorted(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = CompositionProbe(args.probe, device)

    for name in args.cohorts:
        cohort = specs[name].resolve(data.get("processed_root"), data.get("embedding", "h-optimus-1"))
        out_dir = cohort.embedding_dir.parent.parent / "tissue" / args.name
        (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
        store = build_embedding_store_from_dir(root_dir=cohort.embedding_dir)
        rows, skipped = [], []
        for i, slide_id in enumerate(sorted(store.slide_ids())):
            embeddings, _, attrs = store.load(slide_id)
            mpp = effective_mpp(attrs)
            if abs(mpp / args.mpp - 1) > args.mpp_tolerance:
                skipped.append((slide_id, mpp))
                continue
            path = out_dir / "tiles" / f"{slide_id}.npy"
            if path.exists() and not args.overwrite:
                fractions = np.load(path).astype(np.float32)
            else:
                fractions = probe(embeddings)
                np.save(path, fractions.astype(np.float16))
            if len(fractions) != len(embeddings):
                raise RuntimeError(f"{slide_id}: {len(fractions)} fractions for {len(embeddings)} tiles")
            rows.append(
                {"slide_id": slide_id, "n_tiles": len(fractions), "effective_mpp": mpp}
                | dict(zip(probe.classes, fractions.mean(axis=0).tolist(), strict=True))
            )
            if (i + 1) % 200 == 0:
                print(f"[{name}] {i + 1} slides")
        slides = pd.DataFrame(rows)
        slides.to_parquet(out_dir / "slides.parquet")
        (out_dir / "probe.json").write_text(
            json.dumps(
                {"probe": str(args.probe), "classes": probe.classes, "embedding_dir": str(cohort.embedding_dir)},
                indent=2,
            )
        )
        mpps = slides["effective_mpp"]
        print(
            f"[{name}] {len(slides)} slides -> {out_dir} (mpp {mpps.min():.3f}-{mpps.max():.3f}); "
            f"skipped {len(skipped)} off-resolution slides{': ' + str(skipped[:5]) if skipped else ''}"
        )


if __name__ == "__main__":
    main()
