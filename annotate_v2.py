from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from src.data import parse_annotations
from src.models import get_maskrcnn

import click


@click.command(context_settings={"show_default": True})
@click.argument("weights", type=click.Path())
@click.option("--input-dir", type=click.Path(), default="./data/hubmap",
              help="Path to HuBMAP dataset")
@click.option("--output-dir", type=click.Path(),
              default="./data/ds2_annotations",
              help="Directory to save annotations to")
@click.option("--min-confidence", default=0.5,
              type=click.FloatRange(0, 1, max_open=True))
@click.option("--min-iou", default=0.33,
              type=click.FloatRange(0, 1, min_open=True, max_open=True))
@click.option("--keep-confidence", default=0.85,
              type=click.FloatRange(0, 1))
def main(weights: str, input_dir: str, output_dir: str, min_confidence: float,
         min_iou: float, keep_confidence: float):
    # check inputs
    input_dir = Path(input_dir)
    assert input_dir.exists()
    output_dir = Path(output_dir)
    if output_dir.exists():
        assert output_dir.is_dir() and not any(output_dir.iterdir()), \
            f"{output_dir} is not an empty directory"
    else:
        output_dir.mkdir()
    weights_path = Path(weights)
    assert weights_path.exists() and weights_path.suffix == ".pth"
    assert keep_confidence > min_confidence

    # select dataset 2 image ids
    tile_meta = pd.read_csv(input_dir / "tile_meta.csv")
    image_ids = tile_meta.loc[tile_meta["dataset"] == 2, "id"].to_list()

    # load file with original unverified annotations
    polygons = pd.read_json(input_dir / "polygons.jsonl", lines=True)
    polygons = polygons.set_index("id")

    # load detection model
    model = get_maskrcnn(pretrained=False)
    model.load_state_dict(torch.load(weights))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # process every image
    image_count = 0
    mask_count = 0
    for image_id in tqdm(image_ids):
        # load image
        p = input_dir / "train" / f"{image_id}.tif"
        image = cv2.imread(str(p))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load original masks
        masks_dct = parse_annotations(polygons.loc[image_id, "annotations"],
                                      class_names=["blood_vessel", "unsure"])
        masks = np.concatenate([
            masks_dct["blood_vessel"].reshape(-1, 512, 512),
            masks_dct["unsure"].reshape(-1, 512, 512)
        ], axis=0)  # reshape needed for dealing with empty arrays
        assert masks.shape[0] != 0

        # get blood vessel detections
        with torch.no_grad():
            image_t = to_tensor(image).to(device)
            preds = model(image_t.unsqueeze(0))[0]
        n = 0  # remove low confidence detections
        while n < len(preds["scores"]) and preds["scores"][n] > min_confidence:
            n += 1
        # could have done it in the function below but whatever
        preds = {"masks": preds["masks"].cpu()[:n, 0],
                 "scores": preds["scores"].cpu()[:n]}

        # select masks for export
        masks_out = filter_masks(masks, preds, min_iou=min_iou,
                                 conf_thresh=keep_confidence)

        # export them as polygons
        if masks_out.shape[0] > 0:
            mask_count += masks_out.shape[0]
            image_count += 1
            outputs = []
            for mask in masks_out:
                contours, _ = cv2.findContours(
                    image=mask.astype(np.uint8),
                    mode=cv2.RETR_EXTERNAL,
                    method=cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) == 1:
                    contour = contours[0]
                else:
                    areas = [cv2.contourArea(c) for c in contours]
                    ind = np.argmax(areas)
                    contour = contours[ind]
                outputs.append(contour)
            fout = output_dir / f"{image_id}.npz"
            np.savez(fout, *outputs)

    print(f"Found {mask_count} detections in",
          f"{image_count} / {len(image_ids)} images")


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    return (m1 * m2).sum() / (m1 + m2).clip(0, 1).sum()


def filter_masks(masks: np.ndarray, predictions: dict,
                 min_iou: float = 0.33, conf_thresh: float = 0.85):
    """
    1. Keep only masks that have matching (iou > iou_thresh) predictions
    2. Add unused masks with confidence > conf_thresh
    """
    pred = predictions["masks"].numpy().round().astype(np.uint8)
    scores = predictions["scores"].numpy()
    keep = np.zeros(len(masks), dtype=bool)
    used = np.zeros(len(pred), dtype=bool)

    for i, mask in enumerate(masks):
        iou = np.array([mask_iou(mask, pred_i) for pred_i in pred])
        if (iou > min_iou).any():
            keep[i] = True
            used[np.argmax(iou)] = True
            # pred[used] can be reused but will not be added at the end

    unused_confident = np.logical_and(~used, scores > conf_thresh)
    if mask.any():
        return np.concatenate([masks[keep], pred[unused_confident]], axis=0)
    return masks[keep]


if __name__ == "__main__":
    main()
