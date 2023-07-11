import base64
import numpy as np
import torch
from pycocotools import _mask as coco_mask
from typing import Text
import zlib


def mask2bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    "Convert 2d object mask to COCO-style bounding box (x1, y1, x2, y2)"
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    assert mask.any()
    yy, xx = np.nonzero(mask)
    return (xx.min(), yy.min(), xx.max(), yy.max())


def encode_binary_mask(mask: np.ndarray) -> Text:
    "Converts a binary mask into OID challenge encoding ascii text."
    # check input mask --
    if mask.dtype is not np.dtype("bool"):
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str
