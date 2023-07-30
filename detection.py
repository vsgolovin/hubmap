from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import DetectionDataModule, MyDetectionDataModule
from src.models import get_maskrcnn

import click


BACKBONE_DIR = Path("models/backbones")


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--bs", "--batch-size", type=int, default=1)
@click.option("--accumulate-grad-batches", type=int, default=16)
@click.option("--lr", "--learning-rate", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--epochs", type=int, default=20)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=2)
@click.option("-w", "--backbone-weights", type=str, default="default")
@click.option("--lr-find", is_flag=True, default=False)
@click.option("-T", "--cosine-annealing-periods", type=int, default=1)
@click.option("--confidence-threshold", default=0.0,
              type=click.FloatRange(0, 1, max_open=True))
def pretrain(seed: int, split_seed: int, bs: int, accumulate_grad_batches: int,
             lr: float, weight_decay: float, epochs: int,
             trainable_bb_layers: int, backbone_weights: str, lr_find: bool,
             cosine_annealing_periods: int, confidence_threshold: float):
    "Train model on dataset 2 and 3 images annotated with previous model"
    pl.seed_everything(seed)

    # load data
    dm = MyDetectionDataModule(
        root="./data/hubmap",
        annotation_dir="./data/annotations",
        train_transform=get_transform(train=True),
        val_transform=get_transform(train=False),
        split_seed=split_seed,
        batch_size=bs,
        num_workers=min(bs, 8),
        confidence_threshold=confidence_threshold
    )

    # use cosine annealing with warm restarts
    if cosine_annealing_periods:
        ca_steps = epochs // cosine_annealing_periods
    else:
        ca_steps = 0

    # mask r-cnn wrapper
    model = LitMaskRCNN(
        lr=lr,
        weight_decay=weight_decay,
        ca_steps=ca_steps,
        pretrained=True,
        trainable_backbone_layers=trainable_bb_layers,
        num_classes=2,
    )

    # load pretrained backbone weights
    if backbone_weights != "default":
        model.model.backbone.body.load_state_dict(
            torch.load(BACKBONE_DIR / f"{backbone_weights}.pth"),
            strict=False
        )

    # training settings
    save_best = ModelCheckpoint(monitor="val_mAP", mode="max",
                                save_top_k=2)
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs",
                                          name="detection/pretraining")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[save_best],
        logger=logger,
    )

    # find initial learning rate with pytorch-lightning
    if lr_find:
        tuner = pl.tuner.Tuner(trainer)
        tuner.lr_find(model, dm)

    # train the model
    trainer.fit(model, dm)


@cli.command(context_settings={"show_default": True})
@click.option("--ckpt", type=click.Path())
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--val-images", type=click.Path(),
              default="data/val_images_ds1.txt")
@click.option("--bs", "--batch-size", type=int, default=1)
@click.option("--accumulate-grad-batches", type=int, default=16)
@click.option("--lr", "--learning-rate", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--epochs", type=int, default=15)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5), default=2)
@click.option("--lr-find", is_flag=True, default=False)
@click.option("-T", "--cosine-annealing-periods", type=int, default=1)
def finetune(ckpt: str, seed: int, split_seed: int, val_images: str, bs: int,
             accumulate_grad_batches: int, lr: float, weight_decay: float,
             epochs: int, trainable_bb_layers: int, lr_find: bool,
             cosine_annealing_periods: int):
    "Fine-tune pretrained model on dataset 1"
    pl.seed_everything(seed)

    # load validation images ids
    vi = []
    if val_images:
        with open(val_images, "r") as fin:
            for line in fin:
                vi.append(line.rstrip())

    # load data
    dm = DetectionDataModule(
        root="./data/hubmap",
        target_class="blood_vessel",
        dataset_ids=[1],
        train_transform=get_transform(train=True),
        val_transform=get_transform(train=False),
        split_seed=split_seed,
        val_images=vi,
        batch_size=bs,
        num_workers=min(bs, 8)
    )

    # use cosine annealing (> 0) with warm restarts (> 1)
    if cosine_annealing_periods:
        ca_steps = epochs // cosine_annealing_periods
    else:
        ca_steps = 0

    # load checkpoint and override some hyperparameters
    if ckpt:
        model = LitMaskRCNN.load_from_checkpoint(
            checkpoint_path=ckpt,
            lr=lr,
            weight_decay=weight_decay,
            ca_steps=ca_steps,
            trainable_backbone_layers=trainable_bb_layers,
        )
    else:
        model = LitMaskRCNN(
            lr=lr,
            weight_decay=weight_decay,
            ca_steps=ca_steps,
            pretrained=True,
            trainable_backbone_layers=trainable_bb_layers,
            num_classes=2,
        )

    # training settings
    save_best = ModelCheckpoint(monitor="val_mAP", mode="max",
                                save_top_k=2)
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs",
                                          name="detection/dset1")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[save_best],
        logger=logger,
    )

    # find initial learning rate with pytorch-lightning
    if lr_find:
        tuner = pl.tuner.Tuner(trainer)
        tuner.lr_find(model, dm)

    # train the model
    trainer.fit(model, dm)


def get_transform(train: bool = True):
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
        A.ToFloat(max_value=255),
        ToTensorV2()
    ])
    return A.Compose(transforms)


class LitMaskRCNN(pl.LightningModule):
    LOSS_NAMES = ("classifier", "box_reg", "mask", "objectness", "rpn_box_reg")

    def __init__(self, lr: float, weight_decay: float, ca_steps: int,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.wd = weight_decay
        self.ca_steps = ca_steps
        self.model = get_maskrcnn(**kwargs)
        self.val_preds = []
        self.val_targets = []
        self.mean_ap_values = []
        self.mean_ap_counts = []
        self.save_hyperparameters()

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        if self.ca_steps == 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=self.ca_steps)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }}

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        loss_dict = self(batch[0], batch[1])
        for ln in self.LOSS_NAMES:
            self.log(f"train_loss/{ln}", loss_dict[f"loss_{ln}"],
                     batch_size=batch_size, on_step=False, on_epoch=True)
        loss = sum(loss_dict.values())
        self.log("train_loss/total", loss, prog_bar=True,
                 batch_size=batch_size, on_step=False, on_epoch=True)
        return loss

    def _compute_mean_ap(self):
        mean_ap = MeanAveragePrecision(iou_type="segm")
        results = mean_ap(self.val_preds, self.val_targets)
        self.mean_ap_values.append(results["map"])
        self.mean_ap_counts.append(len(self.val_preds))
        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        for pred in preds:
            pred["masks"] = pred["masks"].round().to(torch.uint8).squeeze(1)
        self.val_preds.extend(preds)
        self.val_targets.extend(targets)
        if len(self.val_preds) >= 50:
            self._compute_mean_ap()

    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            self._compute_mean_ap()
        weights = np.array(self.mean_ap_counts) / sum(self.mean_ap_counts)
        mean_ap = (np.array(self.mean_ap_values) * weights).sum()
        self.log("val_mAP", mean_ap, prog_bar=True)
        self.mean_ap_values = []
        self.mean_ap_counts = []


if __name__ == "__main__":
    cli()
