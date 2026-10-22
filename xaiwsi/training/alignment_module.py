from __future__ import annotations

import lightning as L
import torch

from xaiwsi.metrics.retrieval import retrieval_metrics
from xaiwsi.models.alignment import ContrastiveAligner, symmetric_info_nce

# retrieval_metrics directions: query = slides, gallery = RNA.
DIRECTION_NAMES = {"q2g": "wsi2rna", "g2q": "rna2wsi"}


class AlignmentTraining(L.LightningModule):
    """Thin Lightning wrapper around a framework-agnostic ``ContrastiveAligner``.

    Validation/test embed the whole split and report retrieval over all of its cases,
    plus the InfoNCE loss with every case as a candidate.
    """

    def __init__(
        self,
        aligner: ContrastiveAligner,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["aligner"])
        self.aligner = aligner
        self.lr = lr
        self.weight_decay = weight_decay
        self._slides: list[torch.Tensor] = []
        self._rnas: list[torch.Tensor] = []

    def forward(self, batch: dict):
        return self.aligner(batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = symmetric_info_nce(out["slide"], out["rna"], out["logit_scale"])
        batch_size = len(batch["rna"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/logit_scale", out["logit_scale"], on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def _collect(self, batch) -> None:
        out = self(batch)
        self._slides.append(out["slide"].detach().cpu())
        self._rnas.append(out["rna"].detach().cpu())

    def _log_epoch(self, prefix: str) -> dict[str, float]:
        slide = torch.cat(self._slides)
        rna = torch.cat(self._rnas)
        self._slides.clear()
        self._rnas.clear()

        logit_scale = self.aligner.log_logit_scale.exp().clamp(
            max=self.aligner.max_logit_scale
        ).detach().cpu()
        metrics = retrieval_metrics(slide.numpy(), rna.numpy())
        metrics["loss"] = float(symmetric_info_nce(slide, rna, logit_scale))
        for short, name in DIRECTION_NAMES.items():
            metrics = {key.replace(short, name): value for key, value in metrics.items()}
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"loss", "mean_mrr", "mean_r5"},
            )
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._slides.clear()
        self._rnas.clear()

    def validation_step(self, batch, batch_idx):
        self._collect(batch)

    def on_validation_epoch_end(self) -> None:
        if self._slides:
            self._log_epoch("val")

    def on_test_epoch_start(self) -> None:
        self._slides.clear()
        self._rnas.clear()

    def test_step(self, batch, batch_idx):
        self._collect(batch)

    def on_test_epoch_end(self) -> None:
        if not self._slides:
            return
        metrics = self._log_epoch("test")
        if self.trainer.is_global_zero:
            n = int(metrics["n_candidates"])
            print(f"\nCPTAC external test retrieval ({n} candidates)")
            for name in ("wsi2rna", "rna2wsi"):
                print(
                    f"  {name}: R@1 {metrics[f'{name}_r1']:.3f}  R@5 {metrics[f'{name}_r5']:.3f}  "
                    f"R@10 {metrics[f'{name}_r10']:.3f}  median rank {metrics[f'{name}_median_rank']:.0f}  "
                    f"MRR {metrics[f'{name}_mrr']:.3f}"
                )
            print(f"  chance: R@1 {metrics['chance_r1']:.3f}  R@5 {metrics['chance_r5']:.3f}  "
                  f"R@10 {metrics['chance_r10']:.3f}  median rank {n / 2:.0f}")

    def configure_optimizers(self):
        # As in CLIP: no weight decay on biases, gains and the logit scale.
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {"params": [p for p in params if p.ndim >= 2]},
                {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
