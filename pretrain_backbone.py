"""
Waste of time, detection only gets worse.
"""

import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
import torch
from torchvision.utils import make_grid
from src.data import DenoisingDataModule
from src.models import ResNet50AutoEncoder

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--lr", "--learning-rate", type=float, default=1e-3)
@click.option("--batch-size", "--bs", type=int, default=8)
@click.option("--epochs", type=int, default=20)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=3)
@click.option("--dropout", type=(click.FloatRange(0, 1, max_open=True)),
              default=0.0)
def main(seed: int, split_seed: int, lr: float, batch_size: int, epochs: int,
         trainable_bb_layers: int, dropout: float | None):
    pl.seed_everything(seed)
    aug_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomCrop(224, 224),
        A.HueSaturationValue(10, 20, 10)
    ])
    noise_transform = A.Compose([
        A.GaussNoise(var_limit=(10**2, 100**2), always_apply=True),
    ])
    model = LitAutoEncoder(
        learning_rate=lr,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers,
        dropout=dropout
    )
    dm = DenoisingDataModule(
        root="data",
        aug_transform=aug_transform,
        noise_transform=noise_transform,
        split_seed=split_seed,
        test_size=0.2,
        batch_size=batch_size,
        num_workers=8
    )
    save_best = ModelCheckpoint(monitor="val_loss", mode="min")
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs",
                                          name="denoising")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        callbacks=[save_best],
        logger=logger,
        accumulate_grad_batches=2
    )
    trainer.fit(model, datamodule=dm)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float, **kwargs):
        super().__init__()
        self.lr = learning_rate
        self.model = ResNet50AutoEncoder(**kwargs)
        self.loss_fn = nn.L1Loss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        noisy_image, target = batch
        output = self(noisy_image)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_image, target = batch
        output = self(noisy_image)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0:
            images = torch.cat(
                [target, noisy_image.clip(0, 1), output.clip(0, 1)],
                dim=0
            )
            grid = make_grid(images, nrow=len(target))
            self.logger.experiment.add_image("outputs", grid,
                                             self.current_epoch)


if __name__ == "__main__":
    main()
