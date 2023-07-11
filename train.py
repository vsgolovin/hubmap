import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import DetectionDataset1
from src.models import get_maskrcnn
from src import utils


def main():
    transform = A.Compose([
        A.HorizontalFlip(),
        A.ToFloat(max_value=255),
        ToTensorV2()
    ])

    dset = DetectionDataset1("./data", transform)
    train_idx, val_idx = train_test_split(np.arange(len(dset)), test_size=0.1)
    train_ds = Subset(dset, train_idx)
    val_ds = Subset(dset, val_idx)
    train_dl = DataLoader(train_ds, 2, shuffle=True, collate_fn=collate_fn,
                          num_workers=2)
    val_dl = DataLoader(val_ds, 2, shuffle=False, collate_fn=collate_fn,
                        num_workers=2)

    model = LitMaskRCNN(pretrained=True)
    save_best = ModelCheckpoint(monitor="val_loss/total", mode="min")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=12,
        accumulate_grad_batches=8,
        log_every_n_steps=1,
        callbacks=[save_best]
    )
    trainer.fit(model, train_dl, val_dl)
    model = LitMaskRCNN.load_from_checkpoint(save_best.best_model_path)
    torch.save(model.model.state_dict(), "models/mask_r-cnn.pth")


def collate_fn(samples: list) -> tuple[list[Tensor], list[dict]]:
    images, targets = [], []
    for image, masks in samples:
        images.append(image)
        masks = torch.stack(masks)
        boxes = [utils.mask2bbox(mask) for mask in masks]
        boxes = torch.tensor(np.array(boxes), dtype=torch.float32)
        labels = torch.ones(len(masks), dtype=torch.int64)
        d = {"masks": masks, "boxes": boxes, "labels": labels}
        targets.append(d)
    return images, targets


class LitMaskRCNN(pl.LightningModule):
    LOSS_NAMES = ("classifier", "box_reg", "mask", "objectness", "rpn_box_reg")

    def __init__(self, **kwargs):
        super().__init__()
        self.model = get_maskrcnn(**kwargs)
        self.save_hyperparameters()

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        return self.model(images)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=3e-4)

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


if __name__ == "__main__":
    main()
