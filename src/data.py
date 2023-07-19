from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
import pytorch_lightning as pl
from src import utils


CLASS_NAMES = ("blood_vessel", "glomerulus", "unsure")


class DetectionDataset(Dataset):
    def __init__(self, root: Path, images: list[str], masks: list[np.ndarray],
                 transform: Callable | None):
        self.root = root
        self.images = images
        self.masks = masks
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.root / f"{self.images[idx]}.tif"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.masks[idx]
        if self.transform:
            out = self.transform(image=image, masks=masks)
            image = out["image"]
            masks = out["masks"]
        return image, masks

    def __len__(self) -> int:
        return len(self.images)


class DetectionDataModule(pl.LightningDataModule):
    "Uses images either from dataset 1 or dataset 2"
    def __init__(self, root: Path | str, target_class: str,
                 dataset_ids: list[int], train_transform: Callable,
                 val_transform: Callable, split_seed: int | None = None,
                 val_size: float = 0.1, batch_size: int = 2,
                 num_workers: int = 2):
        super().__init__()
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        assert target_class in CLASS_NAMES[:-1]
        self.target_class = target_class
        assert set(dataset_ids).issubset({1, 2})
        self.dataset_ids = dataset_ids
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.split_seed = split_seed
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # read tile meta data to select only images from dataset 1
        df = pd.read_csv(self.root / "tile_meta.csv")
        df = df.set_index("id")  # set id as key
        # read polygons.jsonl -- file with object masks
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        self.images = []
        self.masks = []
        for _, row in polygons.iterrows():
            img_id = row["id"]
            # skip images from other dataset
            if df.loc[img_id, "dataset"] not in self.dataset_ids:
                continue
            # read masks
            masks = []
            for d in row["annotations"]:
                if d["type"] != self.target_class:
                    continue
                # convert polygon coordinates to mask
                coords = np.array(d["coordinates"][0])
                mask = cv2.fillPoly(np.zeros((512, 512), dtype=np.uint8),
                                    pts=[coords], color=1)
                masks.append(mask)
            # image has objects of target class
            if len(masks) > 0:
                self.images.append(img_id)
                self.masks.append(np.array(masks))

    def setup(self, stage: str):
        # stratify by WSI
        df = pd.read_csv(self.root / "tile_meta.csv").set_index("id")
        stratify = [df.loc[img_id, "source_wsi"] for img_id in self.images]
        # split data
        train_idx, val_idx = train_test_split(
            np.arange(len(self.images)),
            test_size=self.val_size,
            random_state=self.split_seed,
            stratify=stratify
        )
        self.train_dset = DetectionDataset(
            root=self.root / "train",
            images=[self.images[ind] for ind in train_idx],
            masks=[self.masks[ind] for ind in train_idx],
            transform=self.train_transform
        )
        self.val_dset = DetectionDataset(
            root=self.root / "train",
            images=[self.images[ind] for ind in val_idx],
            masks=[self.masks[ind] for ind in val_idx],
            transform=self.val_transform
        )

    @staticmethod
    def collate_fn(samples: list) -> tuple[list[Tensor], list[dict]]:
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


class ImageDataset(Dataset):
    "Images for unsupervised or self-supervised learning"
    def __init__(self, root: Path | str, image_ids: list[str],
                 transform: Callable):
        "`transform` should have `albumentations` interface"
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.image_ids = tuple(image_ids)
        self.transform = transform

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / f"{self.image_ids[idx]}.tif"
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        return image

    def __len__(self) -> int:
        return len(self.image_ids)


class ImageDataModule(pl.LightningDataModule):
    "Datamodule for self-supervised learning, returns only uncorrupted images"
    def __init__(self, root: Path | str, train_transform: Callable,
                 val_transform: Callable, split_seed: int | None = None,
                 test_size: float = 0.1, batch_size: int = 16,
                 num_workers: int = 4):
        super().__init__()
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.split_seed = split_seed
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.tile_meta = pd.read_csv(self.root / "tile_meta.csv")

    def setup(self, stage: str):
        mask = self.tile_meta["dataset"] == 3
        img_ids = self.tile_meta.loc[mask, "id"].to_list()
        wsi = self.tile_meta.loc[mask, "source_wsi"].to_numpy()
        train_ids, val_ids = train_test_split(
            img_ids, test_size=self.test_size, random_state=self.split_seed,
            stratify=wsi)
        self.train_dset = ImageDataset(
            root=self.root / "train",
            image_ids=train_ids,
            transform=self.train_transform
        )
        self.val_dset = ImageDataset(
            root=self.root / "train",
            image_ids=val_ids,
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
