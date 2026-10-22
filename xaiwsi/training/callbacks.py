from __future__ import annotations

import socket
from pathlib import Path

from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger


class RunSaveConfigCallback(SaveConfigCallback):
    """Save the LightningCLI config next to the run checkpoints.

    With a W&B logger, the saved config pins the W&B run id, so running
    `test` with it logs into the same W&B run and run directory. The local
    run paths are also stored in the W&B run config.
    """

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

            if isinstance(logger, WandbLogger):
                run_dir = Path(save_dir) / str(name) / str(version)
            else:
                # CSV/TensorBoard: checkpoints go to <save_dir>/<name>/version_<N>/.
                run_dir = Path(logger.log_dir)

        config_path = run_dir / self.config_filename

        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(
                {
                    "run_dir": str(run_dir.resolve()),
                    "config_path": str(config_path.resolve()),
                    "host": socket.gethostname(),
                },
                allow_val_change=True,
            )

        # test/validate/predict from a saved run config reuse the run directory:
        # keep the config written at fit time.
        if stage != "fit" and config_path.exists():
            return

        config = self.config.clone()
        if isinstance(logger, WandbLogger):
            _pin_wandb_run_id(config, logger.version)

        run_dir.mkdir(parents=True, exist_ok=True)

        self.parser.save(
            config,
            config_path,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )


def _pin_wandb_run_id(config, run_id: str) -> None:
    """Set the W&B run id in the trainer logger config."""
    loggers = config.trainer.logger
    if not isinstance(loggers, list):
        loggers = [loggers]

    for logger_config in loggers:
        class_path = getattr(logger_config, "class_path", "")
        if class_path.endswith("WandbLogger"):
            logger_config.init_args.id = run_id
            logger_config.init_args.version = None
            return
