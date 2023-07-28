from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import click


@click.group()
def cli():
    pass


@cli.command(context_settings={"show_default": True})
@click.option("--input-dir", type=click.Path(), default="./data/hubmap",
              help="Path to HuBMAP dataset")
@click.option("--output-dir", type=click.Path(), default="./data/yolo-clean")
@click.option("--val-images", type=click.Path(),
              default="./data/val_images_ds1.txt")
def clean(input_dir: str, output_dir: str, val_images: str):
    "Convert dataset 1 to YOLO format."
    input_dir = Path(input_dir)
    assert input_dir.exists() and input_dir.is_dir()
    output_dir = Path(output_dir)
    if output_dir.exists():
        assert output_dir.is_dir() and not any(output_dir.iterdir()), \
            f"{output_dir} is not an empty directory"
    else:
        output_dir.mkdir()

    # read validation image ids
    with open(val_images, "r") as fin:
        val_image_ids = fin.read().split()
    val_image_ids = set(val_image_ids)

    # get object polygons for dataset 1 images
    polygons = pd.read_json(input_dir / "polygons.jsonl", lines=True)
    polygons = polygons.set_index("id")
    dset1_ids = get_image_ids(input_dir, (1,))[1]
    polygons = polygons.loc[dset1_ids, :]

    # create output directories
    train_dir = output_dir / "train"
    train_dir.mkdir()
    val_dir = output_dir / "val"
    val_dir.mkdir()
    for subset_dir in [train_dir, val_dir]:
        for dir_name in ["images", "labels"]:
            (subset_dir / dir_name).mkdir()

    # create dataset
    for _, row in tqdm(polygons.iterrows()):
        # check if image exists
        image_id = row.name
        image_path = input_dir / "train" / f"{image_id}.tif"
        assert image_path.exists()
        # get bounding boxes and mask boundary coordinates
        annotations = parse_hubmap_annotations(row["annotations"])

        # write to dataset
        if annotations:
            subset_dir = val_dir if image_id in val_image_ids else train_dir
            shutil.copy(image_path, subset_dir / "images" / image_path.name)
            ann_path = subset_dir / "labels" / f"{image_path.stem}.txt"
            with open(ann_path, "w") as fout:
                for d in annotations:
                    fout.write("0 ")  # class label
                    fout.write(" ".join([str(num) for num in d["bbox"]]))
                    for x, y in d["coordinates"]:
                        fout.write(f" {x} {y}")
                    fout.write("\n")


@cli.command(context_settings={"show_default": True})
@click.option("--hubmap-dir", type=click.Path(), default="./data/hubmap",
              help="Path to HuBMAP dataset")
@click.option("--annotation-dir", type=click.Path(),
              default="./data/annotations",
              help="Path to annotations created by a previously traned model")
@click.option("--output-dir", type=click.Path(), default="./data/yolo-noisy")
@click.option("--val-images", type=click.Path(),
              default="./data/val_images_ds2-3.txt")
@click.option("--confidence-threshold", default=0.7,
              type=click.FloatRange(0, 1, max_open=True))
def noisy(hubmap_dir: str, annotation_dir: str, output_dir: str,
          val_images: str, confidence_threshold: float):
    """
    Convert datasets 2 and 3 (original partial + Mask R-CNN annotations) to
    YOLO format.
    """
    hubmap_dir = Path(hubmap_dir)
    annotation_dir = Path(annotation_dir)
    for p in (hubmap_dir, annotation_dir):
        assert p.exists() and p.is_dir()
    output_dir = Path(output_dir)
    if output_dir.exists():
        assert output_dir.is_dir() and not any(output_dir.iterdir()), \
            f"{output_dir} is not an empty directory"
    else:
        output_dir.mkdir()

    # read validation image ids
    with open(val_images, "r") as fin:
        val_image_ids = fin.read().split()
    val_image_ids = set(val_image_ids)

    # get object polygons for dataset 2 and 3
    polygons = pd.read_json(hubmap_dir / "polygons.jsonl", lines=True)
    polygons = polygons.set_index("id")
    image_ids_dict = get_image_ids(hubmap_dir, (2, 3))

    # create output directories
    train_dir = output_dir / "train"
    train_dir.mkdir()
    val_dir = output_dir / "val"
    val_dir.mkdir()
    for subset_dir in [train_dir, val_dir]:
        for dir_name in ["images", "labels"]:
            (subset_dir / dir_name).mkdir()

    # dataset 2
    for dataset_id in (2, 3):
        image_ids = image_ids_dict[dataset_id]
        for image_id in tqdm(image_ids):
            # check if image exists
            image_path = hubmap_dir / "train" / f"{image_id}.tif"
            assert image_path.exists()
            # get bounding boxes and mask boundary coordinates
            annotations = []
            if dataset_id == 2:
                annotations += parse_hubmap_annotations(
                    polygons.loc[image_id, "annotations"],
                    yolo_format=False
                )
            annotations += parse_maskrcnn_annotations(
                npz_file=annotation_dir / f"{image_id}.txt",
                threshold=confidence_threshold
            )
            if annotations:
                keep = filter_annotations(annotations)
                annotations = [a for a, k in zip(annotations, keep) if k]

            # write to dataset
            if annotations:
                subset_dir = val_dir if image_id in val_image_ids \
                             else train_dir
                shutil.copy(image_path, subset_dir / "images"
                            / image_path.name)
                ann_path = subset_dir / "labels" / f"{image_path.stem}.txt"
                with open(ann_path, "w") as fout:
                    for d in annotations:
                        bbox = pascalvoc_bbox_to_yolo(d["bbox"])
                        coords = d["coordinates"].astype(np.float64) / 512
                        fout.write("0 ")  # class label
                        fout.write(" ".join([str(num) for num in bbox]))
                        for x, y in coords:
                            fout.write(f" {x} {y}")
                        fout.write("\n")


def get_image_ids(data_dir: Path, dataset_ids: tuple[int] = (1,)) -> dict:
    df = pd.read_csv(data_dir / "tile_meta.csv")
    out = {}
    for n in dataset_ids:
        mask = (df["dataset"] == n)
        out[n] = df.loc[mask, "id"].to_numpy()
    return out


def parse_hubmap_annotations(ann: list[dict],
                             target_class: str = "blood_vessel",
                             image_size: tuple[int] = (512, 512),
                             yolo_format: bool = True
                             ) -> list[dict]:
    "Select only target class, convert to arrays, remove duplicates"
    out = []
    bbox_set = set()
    for d in ann:
        if d["type"] != target_class:
            continue
        # polygon coordinates
        coords = np.array(d["coordinates"][0])
        # bbox coordinates in Pascal VOC format
        bbox = coords_to_bbox(coords)  # (x1, y1, x2, y2)
        # same bounding box => duplicate
        # not ideal, but probably fine
        if bbox not in bbox_set:
            bbox_set.add(bbox)
            if yolo_format:
                coords = coords.astype(np.float64)
                coords[:, 0] /= image_size[1]
                coords[:, 1] /= image_size[0]
                bbox = pascalvoc_bbox_to_yolo(bbox, image_size)
            out.append({"coordinates": coords, "bbox": bbox})
    return out


def parse_maskrcnn_annotations(npz_file: Path, threshold: float = 0.0):
    if not npz_file.exists():
        return []
    npz = np.load(npz_file)
    out = []
    for k, v in npz.items():
        confidence = float(k)
        if confidence > threshold:
            coords = v.squeeze(1)
            bbox = coords_to_bbox(coords)
            out.append({"coordinates": coords, "bbox": bbox})
    return out


def coords_to_bbox(coords: np.ndarray) -> tuple[int]:
    "Pascal VOC format"
    x1, x2 = coords[:, 0].min(), coords[:, 0].max()
    y1, y2 = coords[:, 1].min(), coords[:, 1].max()
    return (x1, y1, x2, y2)


def pascalvoc_bbox_to_yolo(bbox: tuple[int],
                           image_size: tuple[int] = (512, 512)
                           ) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    xc = (x1 + (x2 - x1) / 2) / image_size[1]
    yc = (y1 + (y2 - y1) / 2) / image_size[0]
    w = (x2 - x1) / image_size[1]
    h = (y2 - y1) / image_size[0]
    return np.array([xc, yc, w, h], dtype=np.float64)


def filter_annotations(ann: list[dict], overlap_threshold: float = 0.5,
                       image_size: tuple[int] = (512, 512)
                       ) -> list[bool]:
    mask = np.zeros(image_size, dtype=bool)
    keep = []
    for d in ann:
        mask_i = cv2.fillPoly(np.zeros(image_size, dtype=np.uint8),
                              pts=[d["coordinates"]], color=1).astype(bool)
        overlap = np.logical_and(mask_i, mask).sum() \
            / np.logical_or(mask_i, mask).sum()
        if overlap < overlap_threshold:
            mask &= mask_i
            keep.append(True)
        else:
            keep.append(False)
    return keep


if __name__ == "__main__":
    cli()
