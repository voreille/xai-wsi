from __future__ import annotations

import json
import math
from pathlib import Path

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
        self.save_hyperparameters(ignore=["predictor"])

        self.predictor = predictor
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_per_pathway_metrics = log_per_pathway_metrics
        self.top_k_to_print = top_k_to_print

        self._val_targets: list[torch.Tensor] = []
        self._val_predictions: list[torch.Tensor] = []
        self._val_cohorts: list[str] = []
        self._test_targets: list[torch.Tensor] = []
        self._test_predictions: list[torch.Tensor] = []
        self._test_cohorts: list[str] = []

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
        self._val_cohorts.clear()

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
        self._val_cohorts.extend(_batch_cohorts(batch, len(target)))

    def on_validation_epoch_end(self) -> None:
        if not self._val_targets:
            return

        self._evaluate("val", self._val_targets, self._val_predictions, self._val_cohorts)

    def on_test_epoch_start(self) -> None:
        self._test_targets.clear()
        self._test_predictions.clear()
        self._test_cohorts.clear()

    def test_step(self, batch, batch_idx):
        output = self(batch["embeddings"])
        pred = output["pred"]
        target = batch["target"]

        self._test_targets.append(target.detach().cpu())
        self._test_predictions.append(pred.detach().cpu())
        self._test_cohorts.extend(_batch_cohorts(batch, len(target)))

    def on_test_epoch_end(self) -> None:
        if not self._test_targets:
            return

        results = self._evaluate(
            "test", self._test_targets, self._test_predictions, self._test_cohorts
        )

        if self.trainer.is_global_zero:
            for name, (summary, by_pathway, _) in results.items():
                if name is None and len(results) == 2:
                    continue  # one test cohort: the pooled table repeats it
                title = "All test cohorts" if name is None else f"{name} external test"
                print(format_pathway_metrics(title, summary, by_pathway, top_k=self.top_k_to_print))
            self._write_test_metrics(results)

    def _evaluate(
        self,
        prefix: str,
        targets: list[torch.Tensor],
        predictions: list[torch.Tensor],
        cohorts: list[str],
    ) -> dict[str | None, tuple[dict[str, float], object, int]]:
        """Metrics over all cases (key None, logged as `<prefix>/<metric>`) and per cohort
        (logged as `<prefix>/<cohort>/<metric>`). Values: (summary, by_pathway, n)."""
        y_true = torch.cat(targets, dim=0).numpy()
        y_pred = torch.cat(predictions, dim=0).numpy()
        names = self._pathway_names(y_true.shape[1])
        labels = np.asarray(cohorts)

        results = {None: (*compute_pathway_metrics(y_true, y_pred, names), len(y_true))}
        for cohort in dict.fromkeys(cohorts):
            if not cohort:
                continue
            mask = labels == cohort
            results[cohort] = (*compute_pathway_metrics(y_true[mask], y_pred[mask], names), int(mask.sum()))

        for cohort, (summary, by_pathway, _) in results.items():
            key = prefix if cohort is None else f"{prefix}/{cohort}"
            self._log_summary(key, summary, prog_bar=cohort is None)
            self._log_per_pathway(key, by_pathway)
        return results

    def _write_test_metrics(self, results) -> None:
        """Write `test_metrics.json` into the run directory (the parent of the
        checkpoint's `checkpoints/` dir, else the trainer's log dir)."""
        ckpt_path = self.trainer.ckpt_path
        if ckpt_path and Path(ckpt_path).parent.name == "checkpoints":
            run_dir = Path(ckpt_path).parent.parent
        else:
            run_dir = Path(self.trainer.log_dir or self.trainer.default_root_dir)

        def clean(value: float) -> float | None:
            return None if not math.isfinite(value) else float(value)

        def entry(summary, by_pathway, n) -> dict:
            return {
                "n_cases": n,
                "summary": {k: clean(v) for k, v in summary.items()},
                "per_pathway": {
                    pathway: {k: clean(v) for k, v in row.items()}
                    for pathway, row in by_pathway.to_dict(orient="index").items()
                },
            }

        payload = {
            "ckpt_path": str(ckpt_path) if ckpt_path else None,
            "all": entry(*results[None]),
            "cohorts": {name: entry(*value) for name, value in results.items() if name is not None},
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "test_metrics.json"
        path.write_text(json.dumps(payload, indent=2))
        print(f"Test metrics: {path}")

    def _pathway_names(self, n_outputs: int) -> list[str]:
        datamodule = self.trainer.datamodule
        if datamodule is not None:
            names = getattr(datamodule, "pathway_names", None)
            if names:
                return list(names)
        return [f"pathway_{i}" for i in range(n_outputs)]

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


def _batch_cohorts(batch, n: int) -> list[str]:
    """Cohort names of a batch; empty names for datasets without a cohort label."""
    cohorts = batch.get("cohort")
    if cohorts is None:
        return [""] * n
    return [str(c) for c in cohorts]
