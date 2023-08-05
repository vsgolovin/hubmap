from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader
from src.data import parse_annotations
from src import utils
from detection import LitMaskRCNN, get_transform

import click


@click.command(context_settings={"show_default": True})
@click.option("--seed", type=int, default=5511)
@click.option("--bs", "--batch-size", type=int, default=1)
@click.option("--accumulate-grad-batches", type=int, default=16)
@click.option("--lr", "--learning-rate", type=float, default=1e-4)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--epochs", type=int, default=20)
@click.option("--trainable-bb-layers", type=click.IntRange(0, 5),
              default=3)
@click.option("-T", "--cosine-annealing-periods", type=int, default=1)
def main(seed: int, bs: int, accumulate_grad_batches: int, lr: float,
         weight_decay: float, epochs: int, trainable_bb_layers: int,
         cosine_annealing_periods: int):
    "Train model on dataset 2 and 3 images annotated with previous model"
    pl.seed_everything(seed)

    # load validation images ids
    vi = []
    for i in (1, 2):
        with open(f"./data/val_images_ds{i}.txt", "r") as fin:
            vi.extend(fin.read().split())

    # load data
    dm = CombinedDetectionDatamodule(
        root="./data/hubmap",
        annotation_dir="./data/ds2_annotations",
        train_transform=get_transform(True),
        val_transform=get_transform(False),
        val_images=vi,
        batch_size=bs,
        num_workers=min(bs, 8)
    )

    # use cosine annealing (> 0) with warm restarts (> 1)
    if cosine_annealing_periods:
        ca_steps = epochs // cosine_annealing_periods
    else:
        ca_steps = 0

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
                                          name="detection/new")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[save_best],
        logger=logger,
    )

    # train the model
    trainer.fit(model, dm)


class CombinedDetectionDataset(Dataset):
    def __init__(self, image_dir: Path | str, image_ids: list[str],
                 annotations: pd.DataFrame, annotation_dir: Path | str,
                 transform: Callable):
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.annotations = annotations
        self.ann_dir = Path(annotation_dir)
        self.transform = transform

    def __getitem__(self, idx):
        # load image
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.tif"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load mask
        if image_id in self.annotations.index:
            mask_dict = parse_annotations(
                self.annotations.loc[image_id, "annotations"],
                class_names=["blood_vessel"]
            )
            masks = mask_dict["blood_vessel"]
        else:
            npz_file = self.ann_dir / f"{image_id}.npz"
            mask_outlines = np.load(npz_file)
            masks = np.zeros((len(mask_outlines), 512, 512), dtype=np.uint8)
            for i, coords in enumerate(mask_outlines.values()):
                masks[i] = cv2.fillPoly(np.zeros((512, 512), dtype=np.uint8),
                                        pts=[coords], color=1)
        # apply transform
        transformed = self.transform(image=image, masks=masks)
        image = transformed["image"]
        masks = transformed["masks"]
        return image, masks

    def __len__(self) -> int:
        return len(self.image_ids)


class CombinedDetectionDatamodule(pl.LightningDataModule):
    def __init__(self, root: Path | str, annotation_dir: Path | str,
                 train_transform: Callable, val_transform: Callable,
                 val_images: list[str], batch_size: int = 2,
                 num_workers: int = 2):
        super().__init__()
        self.root = Path(root)
        self.ann_dir = Path(annotation_dir)
        for p in (self.root, self.ann_dir):
            assert p.exists() and p.is_dir()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_images = val_images
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # dataset 1
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        polygons = polygons.set_index("id")
        tile_meta = pd.read_csv(self.root / "tile_meta.csv")
        self.dset1_ids = []
        for image_id in tile_meta.loc[tile_meta["dataset"] == 1, "id"]:
            annotations = polygons.loc[image_id, "annotations"]
            if any([d["type"] == "blood_vessel" for d in annotations]):
                self.dset1_ids.append(image_id)  # has blood vessel detections
        self.polygons = polygons.loc[self.dset1_ids, :]
        # dataset 2
        self.dset2_ids = []
        for file in self.ann_dir.iterdir():
            if file.suffix == ".npz":
                self.dset2_ids.append(file.stem)

    def setup(self, stage: str):
        val_img_set = set(self.val_images)
        full_img_set = set(self.dset1_ids + self.dset2_ids)
        train_image_ids = list(full_img_set.difference(val_img_set))
        val_image_ids = list(full_img_set.intersection(val_img_set))
        self.train_dset = CombinedDetectionDataset(
            image_dir=self.root / "train",
            image_ids=train_image_ids,
            annotations=self.polygons,
            annotation_dir=self.ann_dir,
            transform=self.train_transform
        )
        self.val_dset = CombinedDetectionDataset(
            image_dir=self.root / "train",
            image_ids=val_image_ids,
            annotations=self.polygons,
            annotation_dir=self.ann_dir,
            transform=self.val_transform
        )

    @staticmethod
    def collate_fn(samples: list) -> tuple[list[torch.Tensor], list[dict]]:
        images, targets = [], []
        for image, masks in samples:
            images.append(image)
            masks = torch.stack(masks)
            boxes = [utils.mask2bbox(mask) for mask in masks]
            boxes = torch.tensor(np.array(boxes), dtype=torch.float32)
            labels = torch.ones(size=(len(masks),), dtype=torch.int64)
            d = {"masks": masks, "boxes": boxes, "labels": labels}
            targets.append(d)
        return images, targets

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)


if __name__ == "__main__":
    main()
