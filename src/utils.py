import numpy as np


def mask2bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    "Convert 2d object mask to COCO-style bounding box (x1, y1, x2, y2)"
    assert mask.any()
    yy, xx = np.nonzero(mask)
    return (xx.min(), yy.min(), xx.max(), yy.max())
