"""TITAN slide embeddings (Ding et al. 2024, MahmoodLab/TITAN) from CONCH v1.5 tile features.

TITAN is a pretrained slide encoder (Mass-340K, no TCGA/CPTAC, no RNA), so it is a frozen
unsupervised slide representation like the slide mean, PANTHER and the SAE concept bags, and
is probed the same way (``xaiwsi.evaluation.cohort_probe --features table:wsi-embeddings/titan/slides.parquet``).
It needs CONCH v1.5 features of 512 px tiles at 20x: the ``conch-v1.5`` encoder of ``wsi-embed``
on the ``configs/tiling/tiling_512.yaml`` grid (``scripts/tile_embed_cohort.sh``), so the config's
``data.embedding`` must be a CONCH v1.5 store (e.g. ``conch-v1.5-fp16``). The tile grid step at
level 0 (``step_size`` x ``downsample``) is TITAN's ``patch_size_lv0``.

    uv run python -m xaiwsi.concepts.titan --config outputs/conch_titan/config.yaml \\
        --cohorts tcga-luad tcga-lusc cptac-luad cptac-lusc --reference

Writes ``<cohort dir>/wsi-embeddings/titan/slides.parquet`` (slide_id x 768, float32; slides already in
it are kept unless ``--overwrite``), a slide-level
embedding next to the tile stores. ``--reference`` compares
the TCGA slides with the embeddings MahmoodLab released (``TCGA_TITAN_features.pkl``, same
model, their own tiling) and writes the cosine similarity to ``--check-dir``: an end-to-end check of our tiling and
CONCH features, expected close to but below 1 since the tile grids differ.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi import paths
from xaiwsi.concepts.bags import embedding_dirs

REPO = "MahmoodLab/TITAN"


def load_titan(device: torch.device) -> torch.nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(REPO, trust_remote_code=True).to(device).eval()


def load_prompts(path: Path) -> tuple[list[str], list[list[str]], list[str]]:
    """(class labels, class names per class, templates) from a zero-shot prompt YAML
    (``configs/zeroshot/*.yaml``: ``templates`` with ``CLASSNAME``, ``classes: {label: [names]}``)."""
    import yaml

    spec = yaml.safe_load(Path(path).read_text())
    templates = list(spec["templates"])
    if not all("CLASSNAME" in t for t in templates):
        raise SystemExit(f"{path}: every template needs CLASSNAME")
    labels = list(spec["classes"])
    return labels, [list(spec["classes"][label]) for label in labels], templates


def text_classifier(model: torch.nn.Module, path: Path, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """Class labels and TITAN's zero-shot weights (embedding dim x classes, unit columns): the
    mean normalized text embedding over a class's names and the templates (``zero_shot_classifier``)."""
    labels, names, templates = load_prompts(path)
    with torch.inference_mode():
        weights = model.zero_shot_classifier(names, templates, device=device)
    return labels, weights.float()


def zero_shot_logits(model: torch.nn.Module, slide_embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Cosine of the projected slide embeddings (``slides.parquet``, before ``proj``) with each class,
    as in ``Titan.zero_shot`` (which softmaxes these without a temperature)."""
    projected = slide_embeddings.float() @ model.vision_encoder.proj.float()
    return torch.nn.functional.normalize(projected, dim=-1) @ weights


def patch_size_lv0(attrs: dict) -> int:
    """Tile grid step in level-0 pixels (coords are level 0)."""
    if attrs.get("coord_space", "level0") != "level0":
        raise ValueError(f"expected level-0 coords, got {attrs.get('coord_space')!r}")
    return int(round(float(attrs["step_size"]) * float(np.asarray(attrs["downsample"]).ravel()[0])))


def encode_slide(model: torch.nn.Module, tiles: np.ndarray, coords: np.ndarray, step: int, device: torch.device) -> np.ndarray:
    features = torch.as_tensor(np.asarray(tiles), dtype=torch.float32, device=device)[None]
    xy = torch.as_tensor(np.asarray(coords), dtype=torch.int64, device=device)[None]
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        embedding = model.encode_slide_from_patch_features(features, xy, step)
    return embedding.float().cpu().numpy().ravel()


def reference_cosine(frame: pd.DataFrame) -> pd.Series:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(REPO, "TCGA_TITAN_features.pkl"), "rb") as f:
        reference = pickle.load(f)
    ref = pd.DataFrame(np.asarray(reference["embeddings"]), index=[str(n) for n in reference["filenames"]])
    shared = frame.index.intersection(ref.index)
    if len(shared) == 0:
        return pd.Series(dtype=float)
    a = frame.loc[shared].to_numpy()
    b = ref.loc[shared].to_numpy()
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return pd.Series(cos, index=shared)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="config whose data.embedding is a CONCH v1.5 store (for cohorts by name)")
    parser.add_argument("--cohorts", nargs="+", required=True, help="<name> from --config, or <name>=<embedding dir>")
    parser.add_argument("--reference", action="store_true", help="compare TCGA slides with the released TITAN embeddings")
    parser.add_argument("--check-dir", type=Path, default=Path("outputs/titan_check"), help="for <cohort>_reference_cosine.csv")
    parser.add_argument("--overwrite", action="store_true", help="re-encode slides already in slides.parquet")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_titan(device)

    for cohort, embedding_dir in embedding_dirs(args.config, args.cohorts).items():
        out_dir = paths.slide_embedding_dir(embedding_dir, "titan")
        out_dir.mkdir(parents=True, exist_ok=True)
        store = build_embedding_store_from_dir(root_dir=embedding_dir)
        rows, done = {}, pd.DataFrame()
        if (out_dir / "slides.parquet").exists() and not args.overwrite:
            done = pd.read_parquet(out_dir / "slides.parquet")
        todo = sorted(set(store.slide_ids()) - set(done.index.astype(str)))
        print(f"[{cohort}] {len(done)} slides already encoded, {len(todo)} to encode")
        for slide_id in tqdm(todo, desc=cohort):
            tiles, coords, attrs = store.load(slide_id)
            if len(tiles) == 0:
                continue
            if attrs.get("encoder_name") != "conch-v1.5":
                raise SystemExit(f"{embedding_dir}: encoder {attrs.get('encoder_name')!r}, TITAN needs conch-v1.5 features")
            rows[slide_id] = encode_slide(model, tiles, coords, patch_size_lv0(attrs), device)
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=[f"titan{j}" for j in range(768)]).astype(np.float32)
        frame = pd.concat([done, frame]).sort_index() if len(done) else frame
        frame.index.name = "slide_id"
        if not np.isfinite(frame.to_numpy()).all():
            raise SystemExit(f"[{cohort}] non-finite TITAN embeddings")
        frame.to_parquet(out_dir / "slides.parquet")
        print(f"[{cohort}] {len(frame)} slides -> {out_dir / 'slides.parquet'}")
        if args.reference:
            cos = reference_cosine(frame)
            if len(cos):
                print(f"[{cohort}] cosine vs released TITAN embeddings, {len(cos)} slides: "
                      f"median {cos.median():.4f}, 5th percentile {cos.quantile(0.05):.4f}, min {cos.min():.4f}")
                args.check_dir.mkdir(parents=True, exist_ok=True)
                cos.rename("cosine").to_csv(args.check_dir / f"{cohort}_reference_cosine.csv")


if __name__ == "__main__":
    main()
