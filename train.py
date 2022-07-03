from jsonargparse import lazy_instance
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.cli import (LightningArgumentParser,
                                             LightningCLI)

from src.data import SimpleDataModule
from src.model import DecoderDenoisingModel


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output/", name="default"
                )
            }
        )


cli = MyLightningCLI(
    DecoderDenoisingModel, SimpleDataModule, run=False, save_config_overwrite=True
)

cli.trainer.tune(cli.model, cli.datamodule)
cli.trainer.fit(cli.model, cli.datamodule)
