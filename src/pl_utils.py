from typing import Any

from pytorch_lightning.loggers import (CSVLogger, LightningLoggerBase,
                                       TensorBoardLogger, WandbLogger)
from pytorch_lightning.utilities.cli import LightningArgumentParser


class MyLightningArgumentParser(LightningArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_logger_args()

    def add_logger_args(self) -> None:
        # Common args
        self.add_argument(
            "--logger_type",
            type=str,
            help="Name of logger",
            default="csv",
            choices=["csv", "wandb", "tensorboard"],
        )
        self.add_argument(
            "--save_path",
            type=str,
            help="Save path of outputs",
            default="output/",
        )
        self.add_argument("--name", type=str, help="name of run", default="default")

        # Wandb
        self.add_argument(
            "--project", type=str, help="Name of wandb project", default="default"
        )


def init_logger(args: dict) -> LightningLoggerBase:  # type:ignore
    """Initialize logger from arguments

    Args:
        args: parsed argument dictionary

    Returns:
        logger: intialized logger object
    """
    if args["logger_type"] == "wandb":
        return WandbLogger(
            project=args["project"],
            name=args["name"],
            save_dir=args["save_path"],
        )
    elif args["logger"] == "tensorboard":
        return TensorBoardLogger(name=args["name"], save_dir=args["save_path"])
    elif args["logger"] == "csv":
        return CSVLogger(name=args["name"], save_dir=args["save_path"])
    else:
        ValueError(
            f"{args['logger']} is not an available logger. Should be one of ['cvs', 'wandb', 'tensorboard']"
        )
