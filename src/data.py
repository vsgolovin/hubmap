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


def parse_annotations(annotations: list[dict], class_names: tuple[str],
                      duplicate_thresh: float = 0.9) -> dict:
    "Also removes duplicates"
    out = dict.fromkeys(class_names, [])

    # iterate over target classes
    for cls_name in class_names:

        # transform polygons to masks
        masks = []
        for dct in annotations:
            if dct["type"] != cls_name:
                continue
            # convert polygon coordinates to mask
            coords = np.array(dct["coordinates"][0])
            mask = cv2.fillPoly(np.zeros((512, 512), dtype=np.uint8),
                                pts=[coords], color=1)
            masks.append(mask)

        # remove duplicate or strongly overlapping masks
        masks = np.array(masks)
        if len(masks) > 1:
            seen = masks[0].astype(bool)
            keep = [0]
            for j, mask in enumerate(masks[1:]):
                mb = mask.astype(bool)
                if (mb & seen).sum() / mb.sum() < duplicate_thresh:
                    keep.append(j + 1)
                    seen |= mb
            masks = masks[keep]

        # add masks to dictionary
        out[cls_name] = masks

    return out


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
        # read tile meta to get dataset numbers and WSI
        df = pd.read_csv(self.root / "tile_meta.csv")
        df = df.set_index("id")  # set id as key
        # read polygons.jsonl -- file with object masks
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        self.images = []
        self.masks = []
        for _, row in polygons.iterrows():
            img_id = row["id"]
            # skip images from not from selected datasets
            if df.loc[img_id, "dataset"] not in self.dataset_ids:
                continue
            # read masks
            masks_dict = parse_annotations(row["annotations"],
                                           class_names=(self.target_class,))
            masks = masks_dict[self.target_class]
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

    def __getitem__(self, idx) -> Tensor:
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


class SegmentationDataset(Dataset):
    def __init__(self, root: Path | str, image_ids: list[str],
                 masks: np.ndarray, transform: Callable):
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.image_ids = image_ids
        assert masks.ndim == 3 and masks.shape[0] == len(image_ids)
        self.masks = masks
        self.transform = transform

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        image_path = self.root / f"{self.image_ids[idx]}.tif"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.masks[idx]
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        if not isinstance(image, Tensor):
            image = to_tensor(image)
        mask = transformed["mask"]
        if not isinstance(mask, Tensor):
            mask = torch.tensor(mask)
        # use float masks as segmentation targets
        return image, mask.to(torch.float32)

    def __len__(self) -> int:
        return len(self.image_ids)


class SegmentationDataModule(pl.LightningDataModule):
    "Intended to do glomerulus semantic segmentation"
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
        # read tile meta data to get dataset numbers and WSI
        df = pd.read_csv(self.root / "tile_meta.csv")
        df = df.set_index("id")  # set id as key
        # read polygons.jsonl -- file with object masks
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        self.images = []
        self.masks = []
        for _, row in polygons.iterrows():
            img_id = row["id"]
            # skip images not from selected datasets
            if df.loc[img_id, "dataset"] not in self.dataset_ids:
                continue
            # read masks
            masks_dict = parse_annotations(row["annotations"],
                                           class_names=(self.target_class,))
            masks = masks_dict[self.target_class]
            # merge masks
            if len(masks) > 0:
                self.images.append(img_id)
                masks = np.stack(masks, axis=0)
                assert masks.ndim == 3
                mask = masks.sum(0).clip(0, 1).astype(np.uint8)
                self.masks.append(mask)
        # cast masks to array
        self.masks = np.stack(self.masks, axis=0)

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
        self.train_dset = SegmentationDataset(
            root=self.root / "train",
            image_ids=[self.images[ind] for ind in train_idx],
            masks=self.masks[train_idx],
            transform=self.train_transform
        )
        self.val_dset = SegmentationDataset(
            root=self.root / "train",
            image_ids=[self.images[ind] for ind in val_idx],
            masks=self.masks[val_idx],
            transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


class MyDetectionDataset(Dataset):
    "Unannotated images with annotations by Mask R-CNN"
    def __init__(self, image_dir: Path | str, image_ids: list[str],
                 annotations: pd.DataFrame, annotation_dir: Path | str,
                 transform: Callable,
                 confidence_threshold: float | None = None,
                 overlap_threshold: float = 0.9):
        """
        Parameters
        ----------
            image_dir: Path | str
                Path to dataset images, i.e. "data/hubmap/train".
            image_ids: list[str]
                Filenames (without extenstion) of used images.
            annotations: pd.DataFrame
                Slice of `polygons.jsonl` indexed by image id.
            annotation_dir: Path | str
                Path to directory with image annotations created by previously
                trained detector.
            transform: Callable
                `albumentations` transform.
            confidence_threshold: float | None
                Only return detections with confidence exceeding this value.
            overlap_theshold: float
                Threshold for removing strongly overlapping detections.
        """
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.annotations = annotations
        self.ann_dir = Path(annotation_dir)
        self.transform = transform
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # load image
        image_path = self.image_dir / f"{image_id}.tif"
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load masks
        # if image is from dataset 2, load original annotations
        ann_lst = self.annotations["annotations"].get(image_id, [])
        if ann_lst:
            masks = parse_annotations(ann_lst, ("blood_vessel",),
                                      duplicate_thresh=self.overlap_threshold
                                      )["blood_vessel"]
            seen = masks.sum(0).astype(bool)
            masks = list(masks)
        else:
            masks = []
            seen = np.zeros((512, 512), dtype=bool)
        # now add detections from previously trained model
        npz_file = self.ann_dir / f"{image_id}.npz"
        if npz_file.exists():
            masks_npz = np.load(npz_file)
            for score_str, coords in masks_npz.items():
                score = float(score_str)
                if (self.confidence_threshold
                        and score < self.confidence_threshold):
                    break  # keys are sorted
                mask = cv2.fillPoly(np.zeros((512, 512), dtype=np.uint8),
                                    pts=[coords], color=1)
                mask_b = mask.astype(bool)
                if ((mask_b & seen).sum() / mask_b.sum() <
                        self.overlap_threshold):
                    masks.append(mask)
                    seen |= mask_b

        # apply transform
        out = self.transform(image=image, masks=masks)
        return out["image"], out["masks"]

    def __len__(self) -> int:
        return len(self.image_ids)


class MyDetectionDataModule(pl.LightningDataModule):
    def __init__(self, root: Path | str, annotation_dir: Path | str,
                 train_transform: Callable, val_transform: Callable,
                 split_seed: int | None = None, val_size: float = 0.15,
                 batch_size: int = 2, num_workers: int = 2,
                 confidence_threshold: float | None = None,
                 overlap_threshold: float = 0.9):
        super().__init__()
        self.root = Path(root)
        assert self.root.exists() and self.root.is_dir()
        self.ann_dir = Path(annotation_dir)
        assert self.ann_dir.exists() and self.ann_dir.is_dir()
        self.image_ids_ds2 = []
        self.image_ids_ds3 = []
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.split_seed = split_seed
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def prepare_data(self):
        # read tile meta data to get dataset numbers and WSI
        self.tile_meta = pd.read_csv(self.root / "tile_meta.csv")  \
            .set_index("id")
        # all dataset 2 images
        self.image_ids_ds2 = self.tile_meta.index[
            self.tile_meta["dataset"] == 2].to_list()

        # images from dataset 3
        self.image_ids_ds3 = []
        for file in self.ann_dir.iterdir():
            if file.suffix != ".npz":
                continue
            image_id = file.stem
            assert (self.root / "train" / f"{image_id}.tif").exists()
            if self.tile_meta.loc[image_id, "dataset"] == 3:
                if self.confidence_threshold:
                    masks_npz = np.load(self.ann_dir / f"{image_id}.npz")
                    scores = [float(score) for score in masks_npz.keys()]
                    if all(s < self.confidence_threshold for s in scores):
                        continue  # no confident predictions
                self.image_ids_ds3.append(image_id)

        # find ground-truth (sort of) annotations of dataset 2 images
        polygons = pd.read_json(self.root / "polygons.jsonl", lines=True)
        keep = [False] * len(polygons)
        for i, row in polygons.iterrows():
            # check if we use this image in current dataset
            if self.tile_meta.loc[row["id"], "dataset"] != 2:
                continue
            # next check for blood vessel detections
            keep[i] = any(d["type"] == "blood_vessel"
                          for d in row["annotations"])
            if not keep[i]:
                masks_npz = np.load(self.ann_dir / f"{row['id']}.npz")
                scores = [float(score) for score in masks_npz.keys()]
                if all(s < self.confidence_threshold for s in scores):
                    self.image_ids_ds2.remove(row["id"])
        self.annotations = polygons[keep].set_index("id")
        assert len(self.annotations) == 1206

    def setup(self, stage: str):
        # stratify by dataset and WSI
        # first split dataset 2 images
        wsi = self.tile_meta.loc[self.image_ids_ds2, "source_wsi"].to_numpy()
        train_ds2, val_ds2 = train_test_split(
            self.image_ids_ds2,
            test_size=self.val_size,
            random_state=self.split_seed,
            stratify=wsi
        )
        # now split dataset 3 images
        wsi = self.tile_meta.loc[self.image_ids_ds3, "source_wsi"].to_numpy()
        train_ds3, val_ds3 = train_test_split(
            self.image_ids_ds3,
            test_size=self.val_size,
            random_state=self.split_seed,
            stratify=wsi
        )
        # create dataset objects
        train_in_df = self.annotations.index.intersection(train_ds2)
        self.train_dset = MyDetectionDataset(
            image_dir=self.root / "train",
            image_ids=train_ds2 + train_ds3,
            annotations=self.annotations.loc[train_in_df, :],
            annotation_dir=self.ann_dir,
            transform=self.train_transform,
            confidence_threshold=self.confidence_threshold,
            overlap_threshold=self.overlap_threshold
        )
        val_in_df = self.annotations.index.intersection(val_ds2)
        self.val_dset = MyDetectionDataset(
            image_dir=self.root / "train",
            image_ids=val_ds2 + val_ds3,
            annotations=self.annotations.loc[val_in_df, :],
            annotation_dir=self.ann_dir,
            transform=self.train_transform,
            confidence_threshold=self.confidence_threshold,
            overlap_threshold=self.overlap_threshold
        )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=DetectionDataModule.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=DetectionDataModule.collate_fn)
