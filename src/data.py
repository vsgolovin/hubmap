from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import pandas as pd
import torch


class DetectionDataset1(torch.utils.data.Dataset):
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
            self.ids.append(img_id)
            self.masks.append(np.array(img_masks))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        p = self._get_img_path(self.ids[idx])
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_RGB2BGR)
        masks = self.masks[idx]
        if self.transform:
            out = self.transform(image=img, masks=masks)
            img = out["image"]
            masks = np.array(out["masks"])
        return img, masks

    def __len__(self) -> int:
        return len(self.ids)

    def _get_img_path(self, img_id: str) -> Path:
        return self.img_dir / f"{img_id}.tif"
