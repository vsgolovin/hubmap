import pytest
import numpy as np
from src import utils


@pytest.mark.parametrize("h,w", [(1, 8), (32, 8), (32, 224)])
def test_mask2bbox(h: int, w: int):
    mask = np.random.randint(2, size=(h, w), dtype=np.uint8)
    while not mask.any():
        mask = np.random.randint(2, size=(h, w), dtype=np.uint8)

    # simple iterative solution
    x_min, x_max = w, -1
    y_min, y_max = h, -1
    for i in range(h):
        for j in range(w):
            if mask[i][j] != 0:
                x_min = min(j, x_min)
                x_max = max(j, x_max)
                y_min = min(i, y_min)
                y_max = max(i, y_max)
    ans = np.array([x_min, y_min, x_max, y_max])

    # vectorized solution
    out = utils.mask2bbox(mask)

    assert np.allclose(out, ans)
