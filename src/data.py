import os
from glob import glob
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Lambda, RandomCrop,
                                    RandomHorizontalFlip, Resize, ToTensor)


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        size: int = 256,
        crop: int = 224,
        num_val: int = 1000,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """Basic data module

        Args:
            root: Path to image directory
            size: Size of resized image
            crop: Size of image crop
            num_val: Number of validation samples
            batch_size: Number of batch samples
            workers: Number of data workers
        """
        super().__init__()
        self.root = root
        self.num_val = num_val
        self.batch_size = batch_size
        self.workers = workers

        self.transforms = Compose(
            [
                Resize(size),
                RandomCrop(crop),
                RandomHorizontalFlip(),
                ToTensor(),
                Lambda(lambda t: (t * 2) - 1),  # Scale to [-1, 1]
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SimpleDataset(self.root, self.transforms)

            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val, self.num_val],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
        )


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = [
            f for f in glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)
        ]
        self.transforms = transforms

        print(f"Loaded {len(self.paths)} images from {root}")

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
