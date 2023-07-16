from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import to_tensor
import pytorch_lightning as pl


class DetectionDataset(torch.utils.data.Dataset):
    "Detections from dataset 1"
    def __init__(self, root: Path | str, transform: Callable | None):
        self.root = Path(root)
        self.img_dir = self.root / "train"
        assert self.root.exists() and self.root.is_dir()

        # albumentations transform
        self.transform = transform

        # read tile meta data to select only images from dataset 1
        df = pd.read_csv(self.root / "tile_meta.csv")
        df = df.set_index("id")  # set id as key

        # read polygons.jsonl -- file with object masks
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        # read detection masks
        self.ids = []
        self.masks = []
        for _, row in polygons.iterrows():
            img_id = row["id"]
            # skip images not from dataset 1
            if df.loc[img_id, "dataset"] != 1:
                continue
            # check that image exists
            p = self._get_img_path(img_id)
            assert p.exists() and p.suffix == ".tif"
            # read masks
            ann_lst = row["annotations"]
            img_masks = []
            for d in ann_lst:
                # only use verified blood vessel masks
                if d["type"] != "blood_vessel":
                    continue
                assert len(d["coordinates"]) == 1
                # convert polygon coordinates to mask
                coords = np.array(d["coordinates"][0])
                mask = cv2.fillPoly(np.zeros((512, 512), dtype=np.uint8),
                                    pts=[coords], color=1)
                img_masks.append(mask)
            if img_masks:
                self.ids.append(img_id)
                self.masks.append(img_masks)

    def __getitem__(self, idx: int) -> tuple:
        p = self._get_img_path(self.ids[idx])
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_RGB2BGR)
        masks = self.masks[idx]
        if self.transform:
            out = self.transform(image=img, masks=masks)
            img = out["image"]
            masks = out["masks"]
        return img, masks

    def __len__(self) -> int:
        return len(self.ids)

    def _get_img_path(self, img_id: str) -> Path:
        return self.img_dir / f"{img_id}.tif"


class DetectionSubset(Dataset):
    def __init__(self, dset: DetectionDataset, indices: np.ndarray,
                 transform: Callable | None):
        assert dset.transform is None or transform is None
        self.dset = Subset(dset, indices=indices)
        self.transform = transform

    def __getitem__(self, idx):
        image, masks = self.dset[idx]
        if self.transform is not None:
            out = self.transform(image=image, masks=masks)
            image = out["image"]
            masks = out["masks"]
        return image, masks

    def __len__(self) -> int:
        return len(self.dset)


class ImageDenoisingDataset(Dataset):
    "Images for unsupervised or self-supervised learning"
    def __init__(self, root: Path | str, image_ids: list[str],
                 aug_transform: Callable, noise_transform: Callable):
        "Both transforms should have albumentations interface"
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.image_ids = tuple(image_ids)
        self.aug_transform = aug_transform
        self.noise_transform = noise_transform

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / f"{self.image_ids[idx]}.tif"
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        image = self.aug_transform(image=image)["image"]
        noisy_image = self.noise_transform(image=image)["image"]
        return to_tensor(noisy_image), to_tensor(image)

    def __len__(self) -> int:
        return len(self.image_ids)


class DenoisingDataModule(pl.LightningDataModule):
    def __init__(self, root: Path | str, aug_transform: Callable,
                 noise_transform: Callable, split_seed: int | None = None,
                 test_size: float = 0.1, batch_size: int = 16,
                 num_workers: int = 4):
        super().__init__()
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.aug_transform = aug_transform
        self.noise_transform = noise_transform
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
        self.train_dset = ImageDenoisingDataset(
            root=self.root / "train",
            image_ids=train_ids,
            aug_transform=self.aug_transform,
            noise_transform=self.noise_transform
        )
        self.val_dset = ImageDenoisingDataset(
            root=self.root / "train",
            image_ids=val_ids,
            aug_transform=self.aug_transform,
            noise_transform=self.noise_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
