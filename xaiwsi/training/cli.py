from __future__ import annotations

import lightning as L
from lightning.pytorch.cli import LightningCLI

from xaiwsi.training.callbacks import RunSaveConfigCallback
from xaiwsi.training.pathway_datamodule import PathwayDataModule
from xaiwsi.training.pathway_module import PathwayTraining


def run_cli(
    model_class: type[L.LightningModule],
    datamodule_class: type[L.LightningDataModule],
) -> LightningCLI:
    return LightningCLI(
        model_class=model_class,
        datamodule_class=datamodule_class,
        auto_configure_optimizers=False,
        save_config_callback=RunSaveConfigCallback,
        save_config_kwargs={
            "config_filename": "config.yaml",
            "overwrite": False,
        },
    )


def main() -> None:
    run_cli(PathwayTraining, PathwayDataModule)


if __name__ == "__main__":
    main()
