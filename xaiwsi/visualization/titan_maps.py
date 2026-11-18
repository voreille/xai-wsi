"""Tile-level TITAN maps: attention and exact per-pattern zero-shot contributions per 512 px tile.

TITAN's slide encoder is a 6-layer transformer over one token per CONCH v1.5 tile (plus a class
token), followed by a single-query attention pooler (``attn_pool_contrastive``; the released
config pools in ``parallel``, so this pooler reads the tile tokens directly). For one slide:

    pooled = sum_j c_j + b_out,   c_j = W_out [concat_h a_hj (W_v,h LN(x_j) + b_v,h)]

with a_hj the attention of head h on token j (each head sums to one over the tokens). The
zero-shot logit of class k is ``(pooled @ proj) . t_k / ||pooled @ proj||``, so

    logit_k = sum_tiles contrib_jk + contrib_cls,k + contrib_bias,k

exactly, with ``contrib_jk = (c_j @ proj) . t_k / ||pooled @ proj||``. A tile's token already
carries its neighbourhood (the transformer mixes tiles; ALiBi positions), so a tile's
contribution is what TITAN reads at that location, not the tile in isolation. The recomputed
pooling is checked against TITAN's own ``forward_attn_pool`` output and the logits against the sum.

Per slide this writes ``tiles/<slide_id>.npz`` (store order: ``attention`` = mean over heads,
``contrib`` tiles x classes, ``cls``/``bias`` per class, ``logits``, grid ``cells``). With
``--anorak-dir`` it also compares, per 512 px cell, TITAN's winning pattern (largest contrast
``contrib_k - mean_k contrib``) with the ANORAK tile probe's winning pattern (224 px tiles
binned into the cell, tumour cells only: ANORAK background < 0.5) -> ``agreement.csv/.md``.
Figures (``figures/<slide_id>.png``) for ``--render-per-class`` slides of each true class (first
by slide id) or ``--render-slides``: thumbnail, attention, TITAN and ANORAK winning pattern, and
one contrast map per pattern (red = pushes the slide toward that pattern).

    uv run python -m xaiwsi.visualization.titan_maps \\
        --embedding-dir data/processed/dhmc-luad/wsi-embeddings/conch-v1.5-fp16 \\
        --zeroshot-dir outputs/titan_zeroshot/dhmc-luad \\
        --anorak-dir data/processed/dhmc-luad/tissue/h-optimus-1-fp16/anorak \\
        --anorak-embedding-dir data/processed/dhmc-luad/wsi-embeddings/h-optimus-1-fp16 \\
        --slides-root data/external/dhmc-luad/DHMC_LUAD_corr \\
        --output-dir outputs/titan_maps/dhmc-luad
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from wsitools.storage.factory import build_embedding_store_from_dir

from xaiwsi import paths
from xaiwsi.concepts.titan import load_titan, patch_size_lv0
from xaiwsi.evaluation.cohort_probe import markdown_table

# Categorical slots 1-5 of the dataviz default palette, fixed order = the prompt YAML's class order.
PATTERN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]


def grid_cells(coords: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """(cell (column 0, column 1) per tile, level-0 origin), as TITAN's ``preprocess_features``."""
    coords = np.asarray(coords, dtype=np.int64)
    origin = coords.min(axis=0)
    cells = np.floor_divide(coords - origin, step)
    cells = cells - cells.min(axis=0)
    return cells, origin


def token_positions(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index of each tile's token in TITAN's sequence, number of tiles sharing that token). The
    class token is at 0, then occupied grid cells in row-major order of (column 0, column 1).
    Tiles of different tissue contours sit on grids with different origins, so a few can fall
    into one cell; ``preprocess_features`` sums their features into one token, and here they
    share it."""
    width = int(cells[:, 1].max()) + 1
    flat = cells[:, 0] * width + cells[:, 1]
    occupied, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return inverse + 1, counts[inverse]


@torch.no_grad()
def decompose(model, tiles: np.ndarray, coords: np.ndarray, step: int, weights: torch.Tensor, device) -> dict:
    encoder = model.vision_encoder
    if getattr(encoder, "attn_pool_type", None) != "parallel":
        raise SystemExit(f"expected TITAN's parallel attention pooler, got {getattr(encoder, 'attn_pool_type', None)!r}")
    preprocess = sys.modules[type(encoder).__module__].preprocess_features
    features = torch.as_tensor(np.asarray(tiles), dtype=torch.float32, device=device)[None]
    xy = torch.as_tensor(np.asarray(coords), dtype=torch.int64, device=device)[None]
    grid, grid_coords, bg_mask = preprocess(features, xy, step)
    tokens = encoder.forward_features(grid, coords=grid_coords, bg_mask=bg_mask)  # 1 x L x E
    reference, _ = encoder.forward_attn_pool(tokens, bg_mask=bg_mask)

    pooler = encoder.attn_pool_contrastive
    attn = pooler.attn
    kv = pooler.ln_k(tokens[0])  # L x E
    query = pooler.ln_q(pooler.query)  # 1 x E
    n_tokens, dim = kv.shape
    heads, head_dim = attn.num_heads, dim // attn.num_heads
    w_q, w_k, w_v = attn.in_proj_weight.chunk(3)
    b_q, b_k, b_v = attn.in_proj_bias.chunk(3)
    q = (query @ w_q.T + b_q).view(heads, head_dim)
    k = (kv @ w_k.T + b_k).view(n_tokens, heads, head_dim)
    v = (kv @ w_v.T + b_v).view(n_tokens, heads, head_dim)
    a = torch.softmax(torch.einsum("hd,lhd->hl", q, k) / math.sqrt(head_dim), dim=-1)  # H x L
    per_token = torch.einsum("hl,lhd->lhd", a, v).reshape(n_tokens, dim) @ attn.out_proj.weight.T
    pooled = per_token.sum(0) + attn.out_proj.bias
    if not torch.allclose(pooled, reference[0], rtol=1e-3, atol=1e-3):
        raise RuntimeError(f"recomputed pooling differs from TITAN's: max {(pooled - reference[0]).abs().max():.2e}")

    proj = encoder.proj
    norm = (pooled @ proj).norm()
    contrib = (per_token @ proj) @ weights / norm  # L x C
    bias = (attn.out_proj.bias @ proj) @ weights / norm
    logits = (pooled @ proj) / norm @ weights
    if not torch.allclose(contrib.sum(0) + bias, logits, atol=1e-4):
        raise RuntimeError("per-token contributions do not sum to the logits")
    return {
        "attention": a.mean(0).cpu().numpy(),
        "contrib": contrib.cpu().numpy(),
        "bias": bias.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "pooled": pooled.cpu().numpy(),
    }


def anorak_cells(anorak_dir: Path, store, slide_id: str, origin: np.ndarray, step: int, n_cells: tuple[int, int],
                 classes: list[str], anorak_classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(mean ANORAK pattern probabilities over ``classes``, background share) per TITAN cell."""
    probs = np.load(anorak_dir / "tiles" / f"{slide_id}.npy")
    _, coords, attrs = store.load(slide_id)
    if len(coords) != len(probs):
        raise ValueError(f"{slide_id}: {len(probs)} ANORAK rows, {len(coords)} tiles")
    size = patch_size_lv0(attrs)
    centers = np.asarray(coords, dtype=np.int64) + size // 2
    cells = np.floor_divide(centers - origin, step)
    inside = (cells >= 0).all(1) & (cells[:, 0] < n_cells[0]) & (cells[:, 1] < n_cells[1])
    flat = cells[inside, 0] * n_cells[1] + cells[inside, 1]
    cols = [anorak_classes.index(c) for c in classes]
    total = np.zeros((n_cells[0] * n_cells[1], len(classes) + 1))
    count = np.bincount(flat, minlength=total.shape[0]).astype(float)
    sub = probs[inside]
    for j, col in enumerate(cols + [anorak_classes.index("background")]):
        total[:, j] = np.bincount(flat, weights=sub[:, col], minlength=total.shape[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count[:, None]
    return mean[:, :-1], mean[:, -1]


def render_figure(path: Path, slide_id: str, title: str, cells: np.ndarray, origin: np.ndarray, step: int, result: dict,
           classes: list[str], anorak_win: np.ndarray | None, tumour: np.ndarray | None, thumbnail) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    n0, n1 = cells.max(0) + 1
    extent = [origin[0], origin[0] + n0 * step, origin[1] + n1 * step, origin[1]]

    def image(values: np.ndarray) -> np.ndarray:
        img = np.full((n1, n0), np.nan)
        img[cells[:, 1], cells[:, 0]] = values
        return img

    contrast = result["contrib"] - result["contrib"].mean(1, keepdims=True)
    win = contrast.argmax(1)
    pattern_cmap = ListedColormap(PATTERN_COLORS[: len(classes)])
    diverging = LinearSegmentedColormap.from_list("blue_gray_red", DIVERGING)
    fig, axes = plt.subplots(2, len(classes), figsize=(4 * len(classes), 8.6), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1]), ax.set_ylim(extent[2], extent[3])
        for side in ax.spines.values():
            side.set_visible(False)

    ax = axes[0, 0]
    if thumbnail is not None:
        ax.imshow(thumbnail[0], extent=thumbnail[1])
    ax.set_title("H&E", fontsize=10)

    ax = axes[0, 1]
    rel = np.log2(result["attention"] * len(result["attention"]))
    im = ax.imshow(image(rel), extent=extent, cmap="Blues", interpolation="nearest",
                   vmin=np.percentile(rel, 1), vmax=np.percentile(rel, 99))
    fig.colorbar(im, ax=ax, shrink=0.7, label="log2 attention / uniform")
    ax.set_title("TITAN attention (mean over heads)", fontsize=10)

    ax = axes[0, 2]
    # The model barely reads low-attention tiles, so their winner is faded by attention rank.
    rank = result["attention"].argsort().argsort() / max(len(win) - 1, 1)
    alpha = 0.1 + 0.9 * rank
    rgba = pattern_cmap(win)
    rgba[:, 3] = alpha
    img = np.ones((n1, n0, 4))
    img[..., 3] = 0
    img[cells[:, 1], cells[:, 0]] = rgba
    ax.imshow(img, extent=extent, interpolation="nearest")
    ax.set_title("TITAN winning pattern (faded: low attention)", fontsize=10)

    ax = axes[0, 3]
    if anorak_win is not None:
        shown = np.where(np.isnan(anorak_win), np.nan, anorak_win)
        ax.imshow(image(shown), extent=extent, cmap=pattern_cmap, vmin=-0.5, vmax=len(classes) - 0.5,
                  interpolation="nearest")
        ax.set_title("ANORAK winning pattern (tumour cells)", fontsize=10)
    ax = axes[0, 4] if len(classes) > 4 else None
    if ax is not None:
        ax.axis("off")
        ax.legend(handles=[Patch(color=PATTERN_COLORS[i], label=c) for i, c in enumerate(classes)],
                  loc="center", frameon=False, fontsize=10, title="pattern")

    limit = np.percentile(np.abs(contrast), 99)
    for i, c in enumerate(classes):
        ax = axes[1, i]
        im = ax.imshow(image(contrast[:, i]), extent=extent, cmap=diverging, vmin=-limit, vmax=limit,
                       interpolation="nearest")
        ax.set_title(f"{c}: logit {result['logits'][i]:.3f}", fontsize=10)
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.7, label="tile contribution − mean over patterns")
    fig.suptitle(f"{slide_id}  {title}", fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-dir", type=Path, required=True, help="CONCH v1.5 512 px store")
    parser.add_argument("--zeroshot-dir", type=Path, required=True, help="titan_zeroshot output (classifier.pt, predictions.parquet)")
    parser.add_argument("--anorak-dir", type=Path, help="tissue/<store>/anorak of the cohort (224 px tiles)")
    parser.add_argument("--anorak-embedding-dir", type=Path, help="the store the ANORAK probe was applied to (coords)")
    parser.add_argument("--slides-root", type=Path, help="raw slides, for thumbnails")
    parser.add_argument("--render-per-class", type=int, default=2)
    parser.add_argument("--render-slides", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    (args.output_dir / "tiles").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_titan(device).float()
    saved = torch.load(args.zeroshot_dir / "classifier.pt")
    classes, weights = saved["classes"], saved["weights"].to(device).float()
    predictions = pd.read_parquet(args.zeroshot_dir / "predictions.parquet").set_index("slide_id")
    store = build_embedding_store_from_dir(root_dir=args.embedding_dir)
    anorak_store, anorak_classes = None, None
    if args.anorak_dir:
        anorak_store = build_embedding_store_from_dir(root_dir=args.anorak_embedding_dir)
        anorak_classes = json.loads((args.anorak_dir / "probe.json").read_text())["classes"]

    to_render = set(args.render_slides)
    for label in classes:
        to_render |= set(sorted(predictions.index[predictions["label"] == label])[: args.render_per_class])

    rows, pooled_titan, pooled_ours = [], [], []
    stored = pd.read_parquet(paths.slide_embedding_dir(args.embedding_dir, "titan") / "slides.parquet")
    for slide_id in tqdm(sorted(predictions.index), desc="slides"):
        tiles, coords, attrs = store.load(slide_id)
        step = patch_size_lv0(attrs)
        cells, origin = grid_cells(coords, step)
        positions, shared = token_positions(cells)
        result = decompose(model, tiles, coords, step, weights, device)
        # A shared token's attention and contribution are split equally over its tiles, so the
        # per-tile values still sum to the token totals (and the logits).
        per_tile = {
            "attention": result["attention"][positions] / shared,
            "contrib": result["contrib"][positions] / shared[:, None],
            "shared": shared,
            "cls": result["contrib"][0],
            "bias": result["bias"],
            "logits": result["logits"],
        }
        pooled_titan.append(stored.loc[slide_id].to_numpy())
        pooled_ours.append(result["pooled"])
        np.savez(args.output_dir / "tiles" / f"{slide_id}.npz", cells=cells, origin=origin, step=step, **per_tile)

        anorak_win = tumour = None
        row = {"slide_id": slide_id, "label": predictions.loc[slide_id, "label"],
               "predicted": predictions.loc[slide_id, "predicted"], "n_tiles": len(tiles)}
        contrast = per_tile["contrib"] - per_tile["contrib"].mean(1, keepdims=True)
        titan_win = contrast.argmax(1)
        if anorak_store is not None:
            n_cells = tuple(int(v) for v in cells.max(0) + 1)
            mean, background = anorak_cells(args.anorak_dir, anorak_store, slide_id, origin, step, n_cells,
                                            classes, anorak_classes)
            flat = cells[:, 0] * n_cells[1] + cells[:, 1]
            tile_mean, tile_bg = mean[flat], background[flat]
            tumour = np.nan_to_num(tile_bg, nan=1.0) < 0.5
            anorak_tile_win = np.where(tumour, np.nan_to_num(tile_mean, nan=-1).argmax(1), np.nan)
            anorak_win = anorak_tile_win
            row.update({
                "tumour_cells": int(tumour.sum()),
                "agreement": float((titan_win[tumour] == anorak_tile_win[tumour]).mean()) if tumour.any() else np.nan,
            })
            att = per_tile["attention"][tumour]
            same = titan_win[tumour] == anorak_tile_win[tumour]
            row["agreement_attention_weighted"] = float((att * same).sum() / att.sum()) if tumour.any() else np.nan
            np.savez(args.output_dir / "tiles" / f"{slide_id}_anorak.npz", tumour=tumour,
                     anorak_win=np.nan_to_num(anorak_tile_win, nan=-1).astype(int), titan_win=titan_win,
                     attention=per_tile["attention"])
        # Slide-level read-outs of the map: most frequent winning pattern over (tumour) tiles.
        keep = tumour if tumour is not None and tumour.any() else np.ones(len(titan_win), bool)
        row["map_majority"] = classes[np.bincount(titan_win[keep], minlength=len(classes)).argmax()]
        rows.append(row)

        if slide_id in to_render:
            thumbnail = None
            if args.slides_root is not None:
                import openslide

                files = list(args.slides_root.glob(f"{slide_id}.*"))
                if files:
                    with openslide.OpenSlide(str(files[0])) as wsi:
                        w, h = wsi.dimensions
                        thumbnail = (np.asarray(wsi.get_thumbnail((2048, 2048))), [0, w, h, 0])
            title = f"true {row['label']}, zero-shot {row['predicted']}"
            render_args = (cells, origin, step, per_tile, classes, anorak_win, tumour, thumbnail)
            render_figure(args.output_dir / "figures" / f"{slide_id}.png", slide_id, title, *render_args)

    a, b = np.asarray(pooled_titan), np.asarray(pooled_ours)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "slides.csv", index=False)
    lines = [f"Recomputed fp32 pooled embedding vs stored TITAN slide embedding (fp16 autocast): cosine "
             f"median {np.median(cos):.5f}, min {cos.min():.5f}, {len(cos)} slides."]
    summary = {"slides": len(table), "map_majority_accuracy": float((table["map_majority"] == table["label"]).mean())}
    if anorak_store is not None:
        tiles_t, tiles_a, tiles_w = [], [], []
        for slide_id in table["slide_id"]:
            z = np.load(args.output_dir / "tiles" / f"{slide_id}_anorak.npz")
            tiles_t.append(z["titan_win"][z["tumour"]])
            tiles_a.append(z["anorak_win"][z["tumour"]])
            w = z["attention"][z["tumour"]]
            tiles_w.append(w / w.sum() if w.sum() > 0 else w)  # each slide weighs one in total
        t, an, w = np.concatenate(tiles_t), np.concatenate(tiles_a), np.concatenate(tiles_w)
        summary.update({
            "tumour_cells": len(t),
            "cell_agreement": float((t == an).mean()),
            "cell_kappa": float(cohen_kappa_score(an, t, labels=list(range(len(classes))))),
            "cell_agreement_attention_weighted": float((w * (t == an)).sum() / w.sum()),
            "cell_kappa_attention_weighted": float(
                cohen_kappa_score(an, t, labels=list(range(len(classes))), sample_weight=w)
            ),
            "median_slide_agreement": float(table["agreement"].median()),
        })
        crosstab = pd.crosstab(pd.Series([classes[i] for i in an], name="anorak"),
                               pd.Series([classes[i] for i in t], name="titan"))
        crosstab.to_csv(args.output_dir / "agreement_crosstab.csv")
        lines += ["", "TITAN vs ANORAK winning pattern per tumour cell (rows ANORAK, columns TITAN):", "",
                  markdown_table(crosstab.reset_index())]
    command = "uv run python -m xaiwsi.visualization.titan_maps " + shlex.join(sys.argv[1:])
    (args.output_dir / "agreement.md").write_text(
        f"TITAN tile maps ({args.zeroshot_dir}). Generated by:\n\n```bash\n{command}\n```\n\n"
        + markdown_table(pd.DataFrame([summary])) + "\n\n" + "\n".join(lines) + "\n"
    )
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
