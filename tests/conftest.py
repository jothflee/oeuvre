"""Shared test helpers."""
import numpy as np


def star_image(coords, shape=(256, 256), amp=1.0, bg=0.1, noise=0.004, seed=0):
    """Render a synthetic star field: 3x3 bright dots on a faint, well-defined
    gaussian background (so detection thresholds behave realistically)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, noise, shape).astype(np.float32)
    for (x, y) in coords:
        img[y - 1:y + 2, x - 1:x + 2] += amp
    return img


# A spread-out, well-separated set of star positions for matching tests.
STARS = [(40, 50), (200, 60), (120, 40), (70, 150), (210, 180),
         (150, 210), (90, 110), (180, 120), (50, 200), (230, 90),
         (110, 170), (160, 80)]
