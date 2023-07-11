from pathlib import Path
import cv2
import numpy as np
import torch
from src.models import get_maskrcnn
from src.utils import encode_binary_mask


model = get_maskrcnn(pretrained=False)
model.load_state_dict(torch.load("models/mask_r-cnn.pth", map_location="cpu"))
model.eval()

fout = open("submission.csv", "w")
D = ","
fout.write(D.join(["id", "height", "width", "prediction_string"]) + "\n")

image_folder = Path("data/test")
for f in image_folder.iterdir():
    img = cv2.imread(str(f))
    fout.write(
        D.join([f.stem, str(img.shape[0]), str(img.shape[1])])
        + D
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = torch.tensor(img.transpose(2, 0, 1))
    with torch.no_grad():
        preds = model([img])[0]

    for score, mask in zip(preds["scores"], preds["masks"]):
        mask = mask.numpy().round().astype(bool)
        code = encode_binary_mask(mask).decode("utf-8")
        fout.write(" ".join(["0", str(score.item()), code]) + " ")

    fout.write("\n")

fout.close()
