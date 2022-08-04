import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DecoderDenoisingModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adam",
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
        momentum: float = 0.9,
        arch: str = "unet",
        encoder: str = "resnet18",
        in_channels: int = 3,
        mode: str = "decoder",
        noise_type: str = "scaled",
        noise_std: float = 0.22,
        loss_type: str = "l2",
        channel_last: bool = False,
    ):
        """Decoder Denoising Pretraining Model

        Args:
            lr: Learning rate
            optimizer: Name of optimizer (adam | adamw | sgd)
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            arch: Segmentation model architecture
            encoder: Segmentation model encoder architecture
            in_channels: Number of channels of input image
            mode: Denoising pretraining mode (encoder | encoder+decoder)
            noise_type: Type of noising process (scaled | simple)
            noise_std: Standard deviation/magnitude of gaussian noise
            loss_type: Loss function type (l1 | l2 | huber)
            channel_last: Change to channel last memory format for possible training speed up
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.channel_last = channel_last

        # Initialize loss function
        self.loss_fn = self.get_loss_fn(loss_type)

        # Initalize network
        self.net = smp.create_model(
            arch,
            encoder_name=encoder,
            in_channels=in_channels,
            classes=in_channels,
            encoder_weights="imagenet" if mode == "decoder" else None,
        )

        # Freeze encoder when doing decoder only pretraining
        if mode == "decoder":
            for child in self.net.encoder.children():  # type:ignore
                for param in child.parameters():
                    param.requires_grad = False
        elif mode != "encoder+decoder":
            raise ValueError(
                f"{mode} is not an available training mode. Should be one of ['decoder', 'encoder+decoder']"
            )

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channel_last:
            self = self.to(memory_format=torch.channels_last)

    @staticmethod
    def get_loss_fn(loss_type: str):
        if loss_type == "l1":
            return F.l1_loss
        elif loss_type == "l2":
            return F.mse_loss
        elif loss_type == "huber":
            return F.smooth_l1_loss
        else:
            raise ValueError(
                f"{loss_type} is not an available loss function. Should be one of ['l1', 'l2', 'huber']"
            )

    @torch.no_grad()
    def add_noise(self, x):
        # Sample noise
        noise = torch.randn_like(x)

        # Add noise to x
        if self.noise_type == "simple":
            x_noise = x + noise * self.noise_std
        elif self.noise_type == "scaled":
            x_noise = ((1 + self.noise_std**2) ** -0.5) * (x + noise * self.noise_std)
        else:
            raise ValueError(
                f"{self.noise_type} is not an available noise type. Should be one of ['simple', 'scaled']"
            )

        return x_noise, noise

    def denoise_step(self, x, mode="train"):
        if self.channel_last:
            x = x.to(memory_format=torch.channels_last)

        # Add noise to x
        x_noise, noise = self.add_noise(x)

        # Predict noise
        pred_noise = self.net(x_noise)

        # Calculate loss
        loss = self.loss_fn(pred_noise, noise)

        # Log
        self.log(f"{mode}_loss", loss)

        return loss

    def training_step(self, x, _):
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],  # type:ignore
            prog_bar=True,
        )
        return self.denoise_step(x, mode="train")

    def validation_step(self, x, _):
        return self.denoise_step(x, mode="val")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches  # type:ignore
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
