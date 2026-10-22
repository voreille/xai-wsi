"""LightningCLI entry point for WSI <-> RNA contrastive alignment (stage 2c)."""

from __future__ import annotations

from xaiwsi.training.alignment_datamodule import AlignmentDataModule
from xaiwsi.training.alignment_module import AlignmentTraining
from xaiwsi.training.cli import run_cli


def main() -> None:
    run_cli(AlignmentTraining, AlignmentDataModule)


if __name__ == "__main__":
    main()
