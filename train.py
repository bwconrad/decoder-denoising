from argparse import Namespace

import pytorch_lightning as pl

from src.data import SimpleDataModule
from src.model import DecoderDenoisingModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = DecoderDenoisingModel
dm_class = SimpleDataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data")
parser.add_lightning_class_args(model_class, "model")

args = parser.parse_args()

# Setup trainer
logger = init_logger(args)
dm = dm_class(**args["data"])
model = model_class(**args["model"])
trainer = pl.Trainer.from_argparse_args(Namespace(**args), logger=logger)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)
