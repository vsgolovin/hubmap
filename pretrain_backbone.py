"""
Small improvement compared to ImageNet weights.
"""

from typing import Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import ImageDataModule
from src.models import MaskedAutoEncoder

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--test-size", type=float, default=0.2)
@click.option("--lr", "--learning-rate", type=float, default=3e-4)
@click.option("--wd", "--weight-decay", type=float, default=1e-3)
@click.option("--batch-size", "--bs", type=int, default=8)
@click.option("--epochs", type=int, default=20)
@click.option("--masking-ratio", default=0.4,
              type=click.FloatRange(0, 1, min_open=False, max_open=False))
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=2)
def main(seed: int, split_seed: int, test_size: float, lr: float, wd: float,
         batch_size: int, epochs: int, masking_ratio: float,
         trainable_bb_layers: int):
    pl.seed_everything(seed)
    model = MaskedAutoEncoder(
        masking_ratio=masking_ratio,
        mask_patch_size=32,
        learning_rate=lr,
        weight_decay=wd,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers,
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


if __name__ == "__main__":
    main()
