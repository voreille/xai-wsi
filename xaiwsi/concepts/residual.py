"""Split each slide mean into the part an SAE's concepts reconstruct and the mean residual.

In the SAE's normalised input space, the slide mean of the tiles decomposes exactly as
``mean_i x̃_i = (ā D + b_pre) + r̄``: ``ā`` is the slide's concept bag
(``xaiwsi.concepts.bags``), ``D`` the decoder, ``b_pre`` the pre-bias and ``r̄`` the mean
reconstruction residual. This writes both parts per slide, so a probe can ask which of them
carries the Hallmark signal and which the cohort shift (README 2f, residual test).

    uv run python -m xaiwsi.concepts.residual --config configs/training/pathway_cohorts.yaml \\
        --cohorts tcga-luad cptac-luad \\
        --sae data/processed/sae/h-optimus-1-fp16/tcga-luad_k4096_top32_s0

Needs the bags of the same SAE. Writes under ``<cohort dir>/concepts/<sae name>/``:
``slides_recon.parquet`` (``ā D + b_pre``) and ``slides_residual.parquet`` (``r̄``), slide_id x
``f<j>``. The slide means of all tiles are cached per cohort in
``<cohort dir>/concepts/slide_means_<embedding>.parquet`` and shared between SAEs. Fails if
the reconstruction error disagrees with the one the bags recorded (``slides.parquet``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi.concepts.bags import embedding_dirs
from xaiwsi.evaluation.learning_curve import slide_means
from xaiwsi.models.sae import load_sae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="config whose cohorts are named in --cohorts")
    parser.add_argument("--cohorts", nargs="+", required=True, help="<name> from --config, or <name>=<embedding dir>")
    parser.add_argument("--sae", type=Path, required=True, help="train_sae output dir (with sae.pt)")
    parser.add_argument("--name", default=None, help="bag folder name (default: SAE dir name)")
    args = parser.parse_args()
    name = args.name or args.sae.name
    model = load_sae(args.sae / "sae.pt")
    decoder = model.decoder.detach().double().numpy()
    pre_bias = model.pre_bias.detach().double().numpy()
    input_mean = model.input_mean.double().numpy()
    scale = float(model.input_scale)

    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        concepts_dir = embedding_dir.parent.parent / "concepts"
        bag_dir = concepts_dir / name
        bags = pd.read_parquet(bag_dir / "slides_mean.parquet")
        bags.index = bags.index.astype(str)
        slide_ids = list(bags.index)
        store = build_embedding_store_from_dir(root_dir=embedding_dir)
        means = slide_means(store, slide_ids, concepts_dir / f"slide_means_{embedding_dir.name}.parquet")
        x_norm = (means.loc[slide_ids].to_numpy(np.float64) - input_mean) / scale
        recon = bags.to_numpy(np.float64) @ decoder + pre_bias
        residual = x_norm - recon

        # Same definition as bags.encode_slide, in normalised units (the scale cancels).
        rel_error = np.linalg.norm(residual, axis=1) / np.linalg.norm(x_norm, axis=1)
        recorded = pd.read_parquet(bag_dir / "slides.parquet")["mean_rel_error"]
        recorded.index = recorded.index.astype(str)
        gap = np.abs(rel_error - recorded.loc[slide_ids].to_numpy()).max()
        if gap > 1e-2:
            raise ValueError(f"{cohort}: reconstruction error differs from the bags' by up to {gap:.3g}")

        columns = [f"f{j}" for j in range(recon.shape[1])]
        for kind, values in (("recon", recon), ("residual", residual)):
            frame = pd.DataFrame(values.astype(np.float32), index=slide_ids, columns=columns)
            frame.index.name = "slide_id"
            frame.to_parquet(bag_dir / f"slides_{kind}.parquet")
        print(
            f"[{cohort}] {len(slide_ids)} slides -> {bag_dir}; median |r̄|/|mean| {np.median(rel_error):.3f}, "
            f"median |recon| {np.median(np.linalg.norm(recon, axis=1)):.3f}, "
            f"median |r̄| {np.median(np.linalg.norm(residual, axis=1)):.3f}"
        )


if __name__ == "__main__":
    main()
