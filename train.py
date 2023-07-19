from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import DetectionDataModule
from src.models import get_maskrcnn

from pytorch_lightning.profilers import AdvancedProfiler

import click

MODEL_DIR = Path("models")


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--split-seed", type=int, default=4277)
@click.option("--lr", "--learning-rate", type=float, default=8e-5)
@click.option("--epochs", type=int, default=12)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=3)
@click.option("-w", "--backbone-weights", type=str, default="default")
def main(seed: int, split_seed: int, lr: float, epochs: int,
         trainable_bb_layers: int, backbone_weights: str):
    pl.seed_everything(seed)

    # datasets and dataloaders
    dm = DetectionDataModule(
        root="./data",
        dataset_id=2,
        train_transform=get_transform(train=True),
        val_transform=get_transform(train=False),
        drop_unsure=True,
        split_seed=split_seed,
        stratify_policy="glomerulus",
        batch_size=2,
        num_workers=2
    )

    # create and train the model
    model = LitMaskRCNN(lr=lr, pretrained=True,
                        trainable_backbone_layers=trainable_bb_layers)
    if backbone_weights != "default":
        model.model.backbone.body.load_state_dict(
            torch.load(MODEL_DIR / f"{backbone_weights}.pth"),
            strict=False
        )
    save_best = ModelCheckpoint(monitor="val_loss/total", mode="min")
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs",
                                          name="detection")
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=8,
        log_every_n_steps=8,
        callbacks=[save_best],
        logger=logger,
        profiler=profiler,
    )
    trainer.fit(model, dm)

    # compute mean average precision on test (== val)
    model = LitMaskRCNN.load_from_checkpoint(save_best.best_model_path)
    trainer.test(model, dm.val_dataloader())
    # export state dict
    torch.save(model.model.state_dict(), MODEL_DIR / "mask_r-cnn.pth")


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

    def __init__(self, lr: float = 3e-4, **kwargs):
        super().__init__()
        self.lr = lr
        self.model = get_maskrcnn(num_classes=3, **kwargs)
        self.test_preds = []
        self.test_targets = []
        self.save_hyperparameters()

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        return self.model(images)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        loss_dict = self(batch[0], batch[1])
        for ln in self.LOSS_NAMES:
            self.log(f"train_loss/{ln}", loss_dict[f"loss_{ln}"],
                     batch_size=batch_size)
        loss = sum(loss_dict.values())
        self.log("train_loss/total", loss, prog_bar=True,
                 batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        self.train()
        loss_dict = self.model(batch[0], batch[1])
        for ln in self.LOSS_NAMES:
            self.log(f"val_loss/{ln}", loss_dict[f"loss_{ln}"],
                     batch_size=batch_size)
        loss = sum(loss_dict.values())
        self.log("val_loss/total", loss, prog_bar=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        for pred in preds:
            pred["masks"] = pred["masks"].round().to(torch.uint8).squeeze(1)
        self.test_preds.extend(preds)
        self.test_targets.extend(targets)

    def on_test_epoch_end(self):
        mean_ap = MeanAveragePrecision(iou_type="segm")
        results = mean_ap(self.test_preds, self.test_targets)
        self.log("test_mAP", results["map"], )
        self.tests_preds = []
        self.test_targets = []


if __name__ == "__main__":
    main()
