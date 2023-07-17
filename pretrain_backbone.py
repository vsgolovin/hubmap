"""
Waste of time, detection only gets worse.
"""

from typing import Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch import nn, optim, Tensor
from torch.nn.functional import interpolate
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from src.data import ImageDataModule
from src.models import ResNet50AutoEncoder

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--test-size", type=float, default=0.2)
@click.option("--lr", "--learning-rate", type=float, default=3e-4)
@click.option("--batch-size", "--bs", type=int, default=8)
@click.option("--epochs", type=int, default=20)
@click.option("--masking-ratio", default=0.4,
              type=click.FloatRange(0, 1, min_open=False, max_open=False))
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=3)
@click.option("--dropout", type=(click.FloatRange(0, 1, max_open=True)),
              default=0.0)
def main(seed: int, split_seed: int, test_size: float, lr: float,
         batch_size: int, epochs: int, masking_ratio: float,
         trainable_bb_layers: int, dropout: float):
    pl.seed_everything(seed)
    model = MaskedAutoEncoder(
        learning_rate=lr,
        masking_ratio=masking_ratio,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers,
        dropout=dropout
    )
    dm = ImageDataModule(
        root="data",
        train_transform=get_transform(train=True),
        val_transform=get_transform(train=False),
        split_seed=split_seed,
        test_size=test_size,
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


def get_transform(train: bool) -> Callable:
    if train:
        transforms = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomCrop(224, 224),
            A.HueSaturationValue(10, 20, 10)
        ]
    else:
        transforms = [
            A.CenterCrop(224, 224),
        ]
    transforms.extend([
        A.ToFloat(max_value=255),
        ToTensorV2()
    ])
    return A.Compose(transforms)


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float, masking_ratio: float, **kwargs):
        super().__init__()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        self.lr = learning_rate
        self.mask_p = masking_ratio
        self.mask_xy = 32
        self.model = ResNet50AutoEncoder(**kwargs)
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_mask(self, b: int, h: int, w: int) -> Tensor:
        # randomly select masking regions
        assert h % self.mask_xy == 0 and w % self.mask_xy == 0
        hm, wm = h // 32, w // 32
        # number of masked squares
        n = int(round(self.mask_p * hm * wm))
        # indices of masked squares for every image
        inds = torch.rand((b, hm * wm)).argsort(1)[:, :n]
        # convert indices to boolean mask
        mask = torch.ones((b, hm * wm), dtype=torch.float32)
        mask[torch.arange(b).repeat_interleave(n), inds.ravel()] = 0.0
        # reshape and upscale mask
        mask = mask.reshape((b, 1, hm, wm))
        mask = interpolate(mask, size=(h, w), mode="nearest")
        return mask.repeat((1, 3, 1, 1)).to(self.device)

    def training_step(self, batch, batch_idx):
        images = batch
        b, _, h, w = images.shape
        mask = self._create_mask(b, h, w)
        output = self(self.normalize(images) * mask)
        mask_b = torch.logical_not(mask.bool())
        loss = self.loss_fn(output[mask_b], images[mask_b])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch
        b, _, h, w = images.shape
        mask = self._create_mask(b, h, w)
        output = self(self.normalize(images) * mask)
        mask_b = torch.logical_not(mask.bool())
        loss = self.loss_fn(output[mask_b], images[mask_b])
        self.log("val_loss", loss, prog_bar=True)
        if batch_idx == 0:
            all_images = torch.cat(
                [images, images * mask, output.clip(0, 1)],
                dim=0
            )
            grid = make_grid(all_images, nrow=len(images))
            self.logger.experiment.add_image("outputs", grid,
                                             self.current_epoch)


if __name__ == "__main__":
    main()
