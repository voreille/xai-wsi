from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import SaveConfigCallback


class RunSaveConfigCallback(SaveConfigCallback):
    """Save the LightningCLI config next to the run checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            save_to_log_dir=False,
            **kwargs,
        )

    def save_config(self, trainer, pl_module, stage) -> None:
        logger = trainer.logger

        if logger is None:
            run_dir = Path(trainer.default_root_dir)
        else:
            # Important for WandB:
            # make sure the run exists so logger.version contains the W&B run id.
            _ = logger.experiment

            save_dir = logger.save_dir or trainer.default_root_dir
            name = logger.name
            version = logger.version

            if version is None:
                raise RuntimeError("Logger has no run/version id after initialization.")

            run_dir = Path(save_dir) / str(name) / str(version)

        run_dir.mkdir(parents=True, exist_ok=True)

        config_path = run_dir / self.config_filename

        self.parser.save(
            self.config,
            config_path,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )
