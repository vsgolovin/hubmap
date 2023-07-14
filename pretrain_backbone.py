import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torchvision.utils import make_grid
from src.data import DenoisingDataModule
from src.models import ResNet50AutoEncoder

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--lr", "--learning-rate", type=float, default=3e-4)
@click.option("--batch-size", "--bs", type=int, default=8)
@click.option("--epochs", type=int, default=20)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=3)
def main(seed: int, split_seed: int, lr: float, batch_size: int, epochs: int,
         trainable_bb_layers: int):
    pl.seed_everything(seed)
    aug_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomCrop(224, 224),
        A.HueSaturationValue(10, 20, 10)
    ])
    noise_transform = A.Compose([
        A.CoarseDropout(max_height=32, max_width=32, p=1.0),
        A.RandomBrightnessContrast()
    ])
    model = LitAutoEncoder(
        learning_rate=lr,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers,
        latent_size=512
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
            grid = make_grid(output.clip(0, 1))
            self.logger.experiment.add_image("outputs", grid,
                                             self.current_epoch)


if __name__ == "__main__":
    main()
