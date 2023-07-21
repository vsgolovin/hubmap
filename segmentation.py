import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torchvision.utils import make_grid
from src.data import SegmentationDataModule
from src.models import ResUNet

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=394039)
@click.option("--split-seed", type=int, default=75389)
@click.option("--bs", "--batch-size", type=int, default=3)
@click.option("--accumulate-grad-batches", type=int, default=4)
@click.option("--lr", "--learning-rate", type=float, default=4e-3)
@click.option("--wd", "--weight-decay", type=float, default=0.0)
@click.option("--epochs", type=int, default=50)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=0)
def main(seed: int, split_seed: int, bs: int, accumulate_grad_batches: int,
         lr: float, wd: float, epochs: int, trainable_bb_layers: int):
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    dm = SegmentationDataModule(
        root="./data",
        target_class="glomerulus",
        dataset_ids=[2],
        train_transform=get_transform(train=True),
        val_transform=get_transform(train=False),
        split_seed=split_seed,
        val_size=0.1,
        batch_size=bs,
        num_workers=min(bs, 12)
    )
    model = LightningSegmentationModel(
        lr=lr,
        weight_decay=wd,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers
    )
    save_best = ModelCheckpoint(monitor="val/loss", save_top_k=1)
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs",
                                          name="segmentation")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[save_best],
        logger=logger
    )
    trainer.fit(model, dm)


def get_transform(train: bool):
    if train:
        transforms = [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.HueSaturationValue(10, 20, 10),
            A.GaussianBlur(),
        ]
    else:
        transforms = []
    transforms.extend([
        A.ToFloat(),
        ToTensorV2()
    ])
    return A.Compose(transforms)


class LightningSegmentationModel(pl.LightningModule):
    def __init__(self, lr: float, weight_decay: float = 0.0,
                 pretrained: bool = True, trainable_backbone_layers: int = 3,
                 dice_loss_eps: float = 1.0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = ResUNet(
            pretrained=pretrained,
            out_channels=1,
            trainable_backbone_layers=trainable_backbone_layers
        )
        self.dice_loss_eps = dice_loss_eps
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze(1)  # logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

    def dice_loss(self, pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        dice = (2 * (pred * target).sum(1) + self.dice_loss_eps) \
            / (pred.sum(1) + target.sum(1) + self.dice_loss_eps)
        loss = 1 - dice
        return loss.mean()

    def iou(self, preds, targets):
        # uses boolean masks instead of original floats
        assert preds.ndim == 3 and targets.ndim == 3
        preds = (preds > 0.5).flatten(start_dim=1)
        targets = (targets > 0.5).flatten(start_dim=1)
        intersection = torch.logical_and(preds, targets).sum(1)
        union = torch.logical_or(preds, targets).sum(1)
        return (intersection / union).mean()

    def _forward_step(self, batch):
        images, masks = batch
        logits = self(images)
        predictions = torch.sigmoid(logits)
        loss = self.dice_loss(predictions, masks)
        iou = self.iou(predictions, masks)
        return loss, iou

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        predictions = torch.sigmoid(logits)
        loss = self.dice_loss(predictions, masks)
        self.log("train/loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True)
        iou = self.iou(predictions, masks)
        self.log("train/iou", iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        predictions = torch.sigmoid(logits)
        loss = self.dice_loss(predictions, masks)
        self.log("val/loss", loss, prog_bar=True)
        iou = self.iou(predictions, masks)
        self.log("val/iou", iou)
        if batch_idx == 0:  # visualize results
            img_stack = torch.cat(
                [images, masks.unsqueeze(1).repeat((1, 3, 1, 1)),
                 predictions.unsqueeze(1).repeat((1, 3, 1, 1))],
                dim=0
            )
            grid = make_grid(img_stack, nrow=len(images))
            self.logger.experiment.add_image("predictions", grid,
                                             self.current_epoch)


if __name__ == "__main__":
    main()
