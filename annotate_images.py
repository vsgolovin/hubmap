from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from src.data import ImageDataset
from src.models import get_maskrcnn
from src.utils import mask2bbox

import click


@click.command(context_settings={"show_default": True})
@click.argument("weights", type=click.Path())
@click.option("--input-dir", type=click.Path(), default="./data/hubmap",
              help="Path to HubMAP dataset")
@click.option("--output-dir", type=click.Path(), default="./data/annotations",
              help="Directory to save annotations to")
@click.option("--threshold", default=0.7,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Minimum confidence score of saved detections")
def main(weights: str, input_dir: str, output_dir: str, threshold: float):
    "Create annotations for images from datasets 2 and 3."
    # check arguments
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

    # find dataset 2 and 3 images
    df = pd.read_csv(input_dir / "tile_meta.csv")
    mask = df["dataset"] > 1
    image_ids = df.loc[mask, "id"].to_list()
    dset = ImageDataset(
        root="./data/hubmap/train",
        image_ids=image_ids,
        transform=A.Compose([A.ToFloat(), ToTensorV2()])
    )

    # load the pretrained model
    model = get_maskrcnn(pretrained=False, trainable_backbone_layers=0)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # detect blood vessels image by image
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    skipped_count = 0
    detections_count = 0
    for i in trange(len(dset)):
        # detect objects with mask r-cnn
        image = dset[i]
        image_id = dset.image_ids[i]
        with torch.no_grad():
            preds = model(image.to(device).unsqueeze(0))[0]

        # select only detections with confidence above threshold
        n = 0
        while len(preds["scores"]) > n and preds["scores"][n] > threshold:
            n += 1
        if n == 0:  # skip files without confident detections
            skipped_count += 1
            continue
        detections_count += n

        # export masks to npz file
        masks = preds["masks"][:n, 0].cpu().numpy().round()
        scores = preds["scores"][:n].cpu().numpy()
        output = {}
        for score, mask in zip(scores, masks):
            x1, y1, x2, y2 = mask2bbox(mask)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
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
            output[str(score)] = contour
        if output:
            np.savez(output_dir / f"{image_id}.npz", **output)

    # simple summary
    print(f"Found {detections_count} detections in",
          f"{len(dset) - skipped_count} / {len(dset)} images")


if __name__ == "__main__":
    main()
