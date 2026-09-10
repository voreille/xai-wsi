from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from xaiwsi.metrics.pathways import compute_pathway_metrics, format_pathway_metrics
from xaiwsi.models.pathway import PathwayPredictor


class PathwayTraining(L.LightningModule):
    """Thin Lightning wrapper around a framework-agnostic predictor."""

    def __init__(
        self,
        predictor: PathwayPredictor,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        log_per_pathway_metrics: bool = False,
        top_k_to_print: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.predictor = predictor
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_per_pathway_metrics = log_per_pathway_metrics
        self.top_k_to_print = top_k_to_print

        self._val_targets: list[torch.Tensor] = []
        self._val_predictions: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_predictions: list[torch.Tensor] = []

    def forward(self, embeddings: torch.Tensor):
        return self.predictor(embeddings)

    def training_step(self, batch, batch_idx):
        output = self(batch["embeddings"])
        loss = F.mse_loss(output["pred"], batch["target"])
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["target"].shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_targets.clear()
        self._val_predictions.clear()

    def validation_step(self, batch, batch_idx):
        output = self(batch["embeddings"])
        pred = output["pred"]
        target = batch["target"]

        loss = F.mse_loss(pred, target)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=target.shape[0],
        )

        self._val_targets.append(target.detach().cpu())
        self._val_predictions.append(pred.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_targets:
            return

        summary, by_pathway = self._compute_epoch_metrics(
            self._val_targets,
            self._val_predictions,
        )
        self._log_summary("val", summary, prog_bar=True)
        self._log_per_pathway("val", by_pathway)

    def on_test_epoch_start(self) -> None:
        self._test_targets.clear()
        self._test_predictions.clear()

    def test_step(self, batch, batch_idx):
        output = self(batch["embeddings"])
        pred = output["pred"]
        target = batch["target"]

        self._test_targets.append(target.detach().cpu())
        self._test_predictions.append(pred.detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_targets:
            return

        summary, by_pathway = self._compute_epoch_metrics(
            self._test_targets,
            self._test_predictions,
        )
        self._log_summary("test", summary, prog_bar=True)
        self._log_per_pathway("test", by_pathway)

        if self.trainer.is_global_zero:
            print(
                format_pathway_metrics(
                    "CPTAC external test",
                    summary,
                    by_pathway,
                    top_k=self.top_k_to_print,
                )
            )

    def _pathway_names(self, n_outputs: int) -> list[str]:
        datamodule = self.trainer.datamodule
        if datamodule is not None:
            names = getattr(datamodule, "pathway_names", None)
            if names:
                return list(names)
        return [f"pathway_{i}" for i in range(n_outputs)]

    def _compute_epoch_metrics(
        self,
        targets: list[torch.Tensor],
        predictions: list[torch.Tensor],
    ):
        y_true = torch.cat(targets, dim=0).numpy()
        y_pred = torch.cat(predictions, dim=0).numpy()
        return compute_pathway_metrics(
            y_true,
            y_pred,
            self._pathway_names(y_true.shape[1]),
        )

    def _log_summary(
        self,
        prefix: str,
        summary: dict[str, float],
        *,
        prog_bar: bool,
    ) -> None:
        for name, value in summary.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar and name in {"mse", "median_spearman"},
                sync_dist=True,
            )

    def _log_per_pathway(self, prefix: str, by_pathway) -> None:
        if not self.log_per_pathway_metrics:
            return

        for pathway, row in by_pathway.iterrows():
            for metric in ("mse", "skill", "r2", "spearman"):
                value = row[metric]
                if np.isfinite(value):
                    self.log(
                        f"{prefix}/pathway/{pathway}/{metric}",
                        float(value),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
