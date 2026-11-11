"""What a shared WSI <-> RNA CCA axis is: tissue composition and single genes.

Fits the linear CCA of ``xaiwsi.evaluation.shared_structure`` (slide means vs BulkFormer,
PCA ``--pca-dim`` per modality, TCGA train; the same fit as the tile maps of
``xaiwsi.evaluation.cca_decomposition``) and reads the axes ``--axes``:

1. Composition. Slide level: Spearman of each axis (slide score u_k, RNA score v_k) with the
   slide shares of the applied tile probes (``xaiwsi.tissue.apply``: IGNITE classes, ANORAK
   patterns renormalized over the tumour patterns), per split. Tile level: within each
   TCGA-val and CPTAC slide, Spearman of the tile axis scores (which average exactly to u_k)
   with the tile class fractions; median and quartiles over slides.
2. Genes. Spearman of each axis with every gene (log2 TPM+1, genes with mean >= ``--min-log2tpm``
   on TCGA train), ranked on TCGA train by the RNA side, with the slide side and the
   val / CPTAC replication.

    uv run python -m xaiwsi.evaluation.cca_axis_reading \\
        --config configs/training/alignment_baseline.yaml \\
        --slide-means-dir outputs/cohort_probe --output-dir outputs/cca_axis_reading

Writes ``composition_slides.csv``, ``composition_tiles.csv``, ``genes.csv`` and
``cca_axis_reading.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm

from xaiwsi.data.pathway import load_log2_tpm_gene_symbols
from xaiwsi.evaluation.cca_decomposition import tile_axis_scores
from xaiwsi.evaluation.cca_hallmarks import SPLIT_NAMES
from xaiwsi.evaluation.cohort_probe import markdown_table
from xaiwsi.evaluation.shared_structure import SPLITS, LinearCCA, check_ranges, load_frozen
from xaiwsi.tissue.features import tissue_dir
from xaiwsi.training.run_config import data_args

ANORAK_PATTERNS = ["cribriform", "micropapillary", "solid", "papillary", "acinar", "lepidic"]


def rank_corr(scores: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Spearman of a vector (n) with every column of ``matrix`` (n x m), vectorized."""
    a = rankdata(scores)
    b = rankdata(matrix, axis=0)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean(0)) / np.where(b.std(0) > 0, b.std(0), np.inf)
    return (a[:, None] * b).mean(0)


def slide_shares(embedding_dir: Path, slide_ids: list[str]) -> pd.DataFrame:
    root = tissue_dir(embedding_dir)
    ignite = pd.read_parquet(root / "ignite" / "slides.parquet").set_index("slide_id")
    anorak = pd.read_parquet(root / "anorak" / "slides.parquet").set_index("slide_id")
    ignite_classes = json.loads((root / "ignite" / "probe.json").read_text())["classes"]
    patterns = anorak[ANORAK_PATTERNS].div(anorak[ANORAK_PATTERNS].sum(axis=1), axis=0)
    shares = pd.concat(
        [ignite[ignite_classes].add_prefix("ignite:"), patterns.add_prefix("anorak:")], axis=1
    )
    return shares.loc[slide_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="AlignmentDataModule config")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--axes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--min-log2tpm", type=float, default=1.0)
    parser.add_argument("--top-genes", type=int, default=25)
    parser.add_argument("--slide-means-dir", type=Path, help="slide-mean parquet cache (default: output dir)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = data_args(yaml.safe_load(args.config.read_text()))
    splits = load_frozen(data, args.slide_means_dir or args.output_dir)
    cca = LinearCCA(args.pca_dim).fit(splits["train"]["x"], splits["train"]["y"])
    z = {s: cca.transform(splits[s]["x"], splits[s]["y"]) for s in SPLITS}
    axes = [a - 1 for a in args.axes]
    x_mean, x_std = splits["train"]["x_mean"], splits["train"]["x_std"]

    # --- 1. composition, slide level --------------------------------------------------
    slide_rows = []
    shares = {}
    for split in SPLITS:
        ids = splits[split]["pairs"]["slide_id"].astype(str).tolist()
        shares[split] = slide_shares(splits[split]["embedding_dir"], ids)
        if shares[split].isna().any().any():
            raise ValueError(f"{split}: slides without tile-probe shares")
        for side, idx in (("slide", 0), ("rna", 1)):
            for axis in axes:
                rho = rank_corr(z[split][idx][:, axis], shares[split].to_numpy())
                for name, value in zip(shares[split].columns, rho, strict=True):
                    slide_rows.append({"axis": axis + 1, "side": side, "split": SPLIT_NAMES[split],
                                       "feature": name, "spearman": value})
    comp = pd.DataFrame(slide_rows)
    check_ranges(comp, ["spearman"], -1, 1)
    comp.to_csv(args.output_dir / "composition_slides.csv", index=False)

    # --- 1. composition, tile level -----------------------------------------------------
    tile_rows = []
    for split in ("val", "test"):
        store = splits[split]["dataset"].embedding_store
        root = tissue_dir(splits[split]["embedding_dir"])
        classes = {p: json.loads((root / p / "probe.json").read_text())["classes"] for p in ("ignite", "anorak")}
        for i, slide_id in enumerate(tqdm(splits[split]["pairs"]["slide_id"].astype(str), desc=f"tiles {split}")):
            tiles, _, _ = store.load(slide_id)
            scores = tile_axis_scores(tiles, x_mean, x_std, cca, args.axes)
            for a, axis in enumerate(axes):
                if not np.isclose(scores[:, a].mean(), z[split][0][i, axis], atol=1e-5):
                    raise ValueError(f"{slide_id}: tile scores do not average to the slide score")
            for probe, names in classes.items():
                fractions = np.load(root / probe / "tiles" / f"{slide_id}.npy").astype(np.float64)
                if len(fractions) != len(scores):
                    raise ValueError(f"{slide_id}: {len(fractions)} {probe} rows, {len(scores)} tiles")
                for a, axis in enumerate(axes):
                    rho = rank_corr(scores[:, a], fractions)
                    for name, value in zip(names, rho, strict=True):
                        tile_rows.append({"axis": axis + 1, "split": SPLIT_NAMES[split], "slide_id": slide_id,
                                          "feature": f"{probe}:{name}", "spearman": value})
    tiles = pd.DataFrame(tile_rows)
    check_ranges(tiles.dropna(), ["spearman"], -1, 1)
    tile_summary = (
        tiles.groupby(["axis", "split", "feature"])["spearman"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    tile_summary.to_csv(args.output_dir / "composition_tiles.csv", index=False)

    # --- 2. genes ---------------------------------------------------------------------
    expression = {
        "tcga": load_log2_tpm_gene_symbols(data["tcga_rna_dir"]),
        "cptac": load_log2_tpm_gene_symbols(data["cptac_rna_dir"]),
    }
    train_expr = expression["tcga"][splits["train"]["pairs"]["sample_id"].astype(str)]
    genes = train_expr.index[train_expr.mean(axis=1) >= args.min_log2tpm].intersection(expression["cptac"].index)
    gene_rows = []
    for split in SPLITS:
        source = "cptac" if split == "test" else "tcga"
        matrix = expression[source].loc[genes, splits[split]["pairs"]["sample_id"].astype(str)].to_numpy().T
        for side, idx in (("slide", 0), ("rna", 1)):
            for axis in axes:
                rho = rank_corr(z[split][idx][:, axis], matrix)
                gene_rows.append(pd.DataFrame({"axis": axis + 1, "side": side, "split": SPLIT_NAMES[split],
                                               "gene": genes, "spearman": rho}))
    gene_corr = pd.concat(gene_rows, ignore_index=True)
    check_ranges(gene_corr, ["spearman"], -1, 1)
    gene_corr.to_csv(args.output_dir / "genes.csv", index=False)

    # --- Markdown -----------------------------------------------------------------------
    sections = []
    comp_wide = comp.pivot_table(index=["axis", "feature"], columns=["side", "split"], values="spearman")
    comp_wide.columns = [f"{side} {split}" for side, split in comp_wide.columns]
    tile_wide = tile_summary.pivot_table(index=["axis", "feature"], columns="split", values="median")
    tile_wide.columns = [f"tile median {c}" for c in tile_wide.columns]
    gene_wide = gene_corr.pivot_table(index=["axis", "gene"], columns=["side", "split"], values="spearman")
    gene_wide.columns = [f"{side} {split}" for side, split in gene_wide.columns]
    for axis in args.axes:
        table = comp_wide.loc[axis].join(tile_wide.loc[axis])
        table = table.reindex(table["slide cptac"].abs().sort_values(ascending=False).index)
        cols = ["slide tcga-train", "slide tcga-val", "slide cptac", "rna tcga-train", "rna cptac",
                "tile median tcga-val", "tile median cptac"]
        genes_axis = gene_wide.loc[axis].sort_values("rna tcga-train")
        gene_cols = ["rna tcga-train", "rna tcga-val", "rna cptac", "slide tcga-train", "slide cptac"]
        top = genes_axis.tail(args.top_genes).iloc[::-1]
        bottom = genes_axis.head(args.top_genes)
        sections.append(
            f"#### Axis {axis}\n\nComposition (Spearman; slide shares vs slide/RNA score, and median within-slide "
            f"tile Spearman), 12 strongest on CPTAC:\n\n{markdown_table(table[cols].head(12).reset_index())}\n\n"
            f"Top {args.top_genes} genes, positive end (ranked on the RNA side, TCGA train):\n\n"
            f"{markdown_table(top[gene_cols].reset_index())}\n\n"
            f"Top {args.top_genes} genes, negative end:\n\n{markdown_table(bottom[gene_cols].reset_index())}"
        )
    canon = {SPLIT_NAMES[s]: [np.corrcoef(z[s][0][:, a], z[s][1][:, a])[0, 1].round(2) for a in axes] for s in SPLITS}
    command = "uv run python -m xaiwsi.evaluation.cca_axis_reading " + shlex.join(sys.argv[1:])
    text = (
        f"Reading of the linear CCA axes {args.axes} (slide means vs BulkFormer, PCA {args.pca_dim}, TCGA train; "
        f"canonical r per split {canon}). {len(genes)} genes with mean log2(TPM+1) >= {args.min_log2tpm} on TCGA "
        f"train. Generated by:\n\n```bash\n{command}\n```\n\n" + "\n\n".join(sections) + "\n"
    )
    (args.output_dir / "cca_axis_reading.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
