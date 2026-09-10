from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

from xaiwsi.training.pathway_datamodule import PathwayDataModule
from xaiwsi.training.pathway_module import PathwayTraining


def main() -> None:
    LightningCLI(
        model_class=PathwayTraining,
        datamodule_class=PathwayDataModule,
        auto_configure_optimizers=False,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": False,
        },
    )


if __name__ == "__main__":
    main()
