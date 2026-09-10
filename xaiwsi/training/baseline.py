# TODO: add embed dim to the embeding store interface and remove this constant.
# TODO: save weights
# TODO: add wandbl, use pl ?


from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from wsitools.storage.factory import build_embedding_store_from_dir
from wsitools.storage.interfaces import EmbeddingStore

from xaiwsi.rna.genes import expression_to_gene_symbols
from xaiwsi.rna.normalization import log2p1
from xaiwsi.rna.pathways.hallmarks import load_hallmarks
from xaiwsi.rna.pathways.scoring import (
    fit_gene_zscore_stats,
    score_mean_z,
)

DATA_DIR = Path("/home/valentin/workspaces/xai-wsi/data")

HALLMARK_GMT = DATA_DIR / "external/msigdb/h.all.v2026.1.Hs.symbols.gmt"

TCGA_RNA_DIR = DATA_DIR / "processed/tcga-luad/rna"
CPTAC_RNA_DIR = DATA_DIR / "processed/cptac-luad/rna"

TCGA_EMBEDDING_DIR = DATA_DIR / "processed/tcga-luad/wsi-embeddings/h-optimus-1"
CPTAC_EMBEDDING_DIR = DATA_DIR / "processed/cptac-luad/wsi-embeddings/h-optimus-1"

TCGA_WSI_METADATA = DATA_DIR / "processed/tcga-luad/metadata/wsi_slides.csv"
CPTAC_WSI_METADATA = DATA_DIR / "processed/cptac-luad/metadata/wsi_slides.csv"

EMBEDDING_DIM = 1536


def load_log2_tpm_gene_symbols(rna_dir: Path) -> pd.DataFrame:
    """Return gene_symbol x sample log2(TPM + 1).

    Collapse duplicate gene symbols in TPM space BEFORE taking the log.
    """
    tpm = pd.read_parquet(rna_dir / "tpm.parquet")
    genes = pd.read_csv(rna_dir / "genes.csv")

    tpm = expression_to_gene_symbols(
        tpm,
        genes,
        duplicate_strategy="sum",
    )

    return log2p1(tpm)


def build_case_pairs(
    embedding_store: EmbeddingStore,
    wsi_metadata: pd.DataFrame,
    rna_samples: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Build one slide / one RNA sample per case for the baseline."""

    available_slides = set(embedding_store.slide_ids())

    wsi = wsi_metadata[wsi_metadata["slide_id"].isin(available_slides)].copy()

    rna = rna_samples[rna_samples["sample_id"].isin(expression.columns)].copy()

    # Report ambiguities before resolving them.
    slides_per_case = wsi.groupby("case_id")["slide_id"].nunique()
    rna_per_case = rna.groupby("case_id")["sample_id"].nunique()

    print(
        "Cases with multiple slides:",
        int((slides_per_case > 1).sum()),
    )
    print(
        "Cases with multiple RNA samples:",
        int((rna_per_case > 1).sum()),
    )

    # Simple baseline:
    # deterministic one-slide / one-RNA-sample selection.
    #
    # Later this should be replaced by an explicit biological/technical
    # selection rule or patient-level aggregation.
    wsi = wsi.sort_values(["case_id", "slide_id"]).drop_duplicates(
        "case_id", keep="first"
    )

    rna = rna.sort_values(["case_id", "sample_id"]).drop_duplicates(
        "case_id", keep="first"
    )

    pairs = wsi[["case_id", "slide_id"]].merge(
        rna[["case_id", "sample_id"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    return pairs.sort_values("case_id").reset_index(drop=True)


def split_cases(
    pairs: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    indices = rng.permutation(len(pairs))
    n_val = round(len(pairs) * val_fraction)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        pairs.iloc[train_idx].reset_index(drop=True),
        pairs.iloc[val_idx].reset_index(drop=True),
    )


class TileMLPMean(nn.Module):
    """Predict each tile independently and average tile predictions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # B x N x D -> B x N x P
        tile_scores = self.mlp(embeddings)

        # B x N x P -> B x P
        slide_scores = tile_scores.mean(dim=1)

        return slide_scores, tile_scores


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        pairs: pd.DataFrame,
        pathway_scores: pd.DataFrame,
    ):
        self.embedding_store = embedding_store
        self.pairs = pairs.reset_index(drop=True)
        self.pathway_scores = pathway_scores

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        slide_id = row["slide_id"]
        sample_id = row["sample_id"]

        embeddings, coords, _ = self.embedding_store.load(slide_id)

        embeddings = torch.from_numpy(np.asarray(embeddings)).float()

        target = torch.from_numpy(
            self.pathway_scores.loc[sample_id].to_numpy(dtype=np.float32)
        )

        return {
            "case_id": row["case_id"],
            "slide_id": slide_id,
            "sample_id": sample_id,
            "embeddings": embeddings,
            "target": target,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pathway_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute global and per-pathway regression metrics."""

    rows = []

    for i, name in enumerate(pathway_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = np.mean((yt - yp) ** 2)

        # Since the z-score statistics were fitted on TCGA train,
        # zero corresponds approximately to the TCGA-training mean target.
        zero_mse = np.mean(yt**2)
        skill = 1.0 - mse / zero_mse if zero_mse > 0 else np.nan

        # Standard R²: baseline is the mean of THIS evaluation split.
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if np.std(yt) > 0 and np.std(yp) > 0:
            rho = spearmanr(yt, yp).statistic
        else:
            rho = np.nan

        rows.append(
            {
                "pathway": name,
                "mse": mse,
                "zero_mse": zero_mse,
                "skill": skill,
                "r2": r2,
                "spearman": rho,
            }
        )

    by_pathway = pd.DataFrame(rows).set_index("pathway")

    global_mse = np.mean((y_true - y_pred) ** 2)
    global_zero_mse = np.mean(y_true**2)

    summary = {
        "mse": global_mse,
        "zero_mse": global_zero_mse,
        "skill": 1.0 - global_mse / global_zero_mse,
        "median_r2": by_pathway["r2"].median(),
        "median_spearman": by_pathway["spearman"].median(),
        "mean_spearman": by_pathway["spearman"].mean(),
        "n_rho_gt_0": int((by_pathway["spearman"] > 0).sum()),
        "n_rho_gt_03": int((by_pathway["spearman"] > 0.3).sum()),
    }

    return summary, by_pathway


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pathway_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()

    all_targets = []
    all_predictions = []

    for batch in dataloader:
        embeddings = batch["embeddings"].to(device)
        targets = batch["target"].to(device)

        predictions, _ = model(embeddings)

        all_targets.append(targets.cpu())
        all_predictions.append(predictions.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_predictions, dim=0).numpy()

    return compute_metrics(
        y_true,
        y_pred,
        pathway_names,
    )


def print_metrics(
    name: str,
    summary: dict[str, float],
    by_pathway: pd.DataFrame,
    top_k: int = 10,
):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MSE:                 {summary['mse']:.4f}")
    print(f"Zero baseline MSE:   {summary['zero_mse']:.4f}")
    print(f"Skill vs zero:       {summary['skill']:+.3f}")
    print(f"Median R²:           {summary['median_r2']:+.3f}")
    print(f"Median Spearman:     {summary['median_spearman']:+.3f}")
    print(f"Mean Spearman:       {summary['mean_spearman']:+.3f}")
    print(f"Hallmarks rho > 0:   {summary['n_rho_gt_0']}/{len(by_pathway)}")
    print(f"Hallmarks rho > 0.3: {summary['n_rho_gt_03']}/{len(by_pathway)}")

    print(f"\nTop {top_k} Hallmarks by Spearman:")
    print(
        by_pathway.sort_values("spearman", ascending=False)
        .head(top_k)
        .to_string(
            columns=["spearman", "r2", "mse", "skill"],
            float_format=lambda x: f"{x:+.3f}",
        )
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------

    tcga_store = build_embedding_store_from_dir(root_dir=TCGA_EMBEDDING_DIR)
    cptac_store = build_embedding_store_from_dir(root_dir=CPTAC_EMBEDDING_DIR)

    # ------------------------------------------------------------
    # Load RNA
    # ------------------------------------------------------------

    tcga_expression = load_log2_tpm_gene_symbols(TCGA_RNA_DIR)
    cptac_expression = load_log2_tpm_gene_symbols(CPTAC_RNA_DIR)

    # Use exactly the same gene universe in both cohorts.
    common_genes = tcga_expression.index.intersection(cptac_expression.index)

    tcga_expression = tcga_expression.loc[common_genes]
    cptac_expression = cptac_expression.loc[common_genes]

    # ------------------------------------------------------------
    # Match WSI <-> RNA at patient/case level
    # ------------------------------------------------------------

    tcga_pairs = build_case_pairs(
        tcga_store,
        pd.read_csv(TCGA_WSI_METADATA),
        pd.read_csv(TCGA_RNA_DIR / "samples.csv"),
        tcga_expression,
    )

    cptac_pairs = build_case_pairs(
        cptac_store,
        pd.read_csv(CPTAC_WSI_METADATA),
        pd.read_csv(CPTAC_RNA_DIR / "samples.csv"),
        cptac_expression,
    )

    print(f"TCGA paired cases:  {len(tcga_pairs)}")
    print(f"CPTAC paired cases: {len(cptac_pairs)}")

    # ------------------------------------------------------------
    # TCGA train / validation split
    # ------------------------------------------------------------

    train_pairs, val_pairs = split_cases(
        tcga_pairs,
        val_fraction=0.2,
        seed=42,
    )

    # ------------------------------------------------------------
    # Compute Hallmark targets
    # ------------------------------------------------------------

    hallmarks = load_hallmarks(HALLMARK_GMT)
    pathway_names = list(hallmarks.keys())

    train_sample_ids = train_pairs["sample_id"].tolist()
    val_sample_ids = val_pairs["sample_id"].tolist()
    cptac_sample_ids = cptac_pairs["sample_id"].tolist()

    # IMPORTANT:
    # gene statistics are fitted ONLY on TCGA training patients.
    zscore_stats = fit_gene_zscore_stats(tcga_expression[train_sample_ids])

    train_scores = score_mean_z(
        tcga_expression[train_sample_ids],
        hallmarks,
        zscore_stats=zscore_stats,
    )

    val_scores = score_mean_z(
        tcga_expression[val_sample_ids],
        hallmarks,
        zscore_stats=zscore_stats,
    )

    cptac_scores = score_mean_z(
        cptac_expression[cptac_sample_ids],
        hallmarks,
        zscore_stats=zscore_stats,
    )

    assert train_scores.shape[1] == 50
    assert list(train_scores.columns) == list(val_scores.columns)
    assert list(train_scores.columns) == list(cptac_scores.columns)

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------

    train_dataset = EmbeddingDataset(
        tcga_store,
        train_pairs,
        train_scores,
    )
    val_dataset = EmbeddingDataset(
        tcga_store,
        val_pairs,
        val_scores,
    )
    cptac_dataset = EmbeddingDataset(
        cptac_store,
        cptac_pairs,
        cptac_scores,
    )

    # Slides have variable numbers of tiles, so batch_size=1 for now.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )
    cptac_loader = DataLoader(
        cptac_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = TileMLPMean(
        input_dim=EMBEDDING_DIM,
        hidden_dim=128,
        output_dim=len(hallmarks),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print(
        f"Training on {len(train_dataset)} TCGA cases, "
        f"validating on {len(val_dataset)} TCGA cases, "
        f"testing on {len(cptac_dataset)} CPTAC cases."
    )

    best_val_mse = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(10):
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            embeddings = batch["embeddings"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            predictions, _ = model(embeddings)

            loss = nn.functional.mse_loss(
                predictions,
                targets,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_summary, _ = evaluate(
            model,
            val_loader,
            device,
            pathway_names,
        )

        if val_summary["mse"] < best_val_mse:
            best_val_mse = val_summary["mse"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train={train_loss:.4f} | "
            f"val={val_summary['mse']:.4f} | "
            f"rho={val_summary['median_spearman']:+.3f} | "
            f"skill={val_summary['skill']:+.3f}"
        )
    # ------------------------------------------------------------
    # Evaluate best TCGA-validation checkpoint
    # ------------------------------------------------------------

    assert best_state is not None
    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")

    val_summary, val_by_pathway = evaluate(
        model,
        val_loader,
        device,
        pathway_names,
    )

    cptac_summary, cptac_by_pathway = evaluate(
        model,
        cptac_loader,
        device,
        pathway_names,
    )

    print_metrics(
        "TCGA validation",
        val_summary,
        val_by_pathway,
    )

    print_metrics(
        "CPTAC external test",
        cptac_summary,
        cptac_by_pathway,
    )


if __name__ == "__main__":
    main()
