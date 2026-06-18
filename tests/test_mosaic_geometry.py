"""Tests for nebula-robust detection and chained mosaic registration.

These guard the two fixes for "use all the data" on IC 1805:
  - detect_stars must survive a bright diffuse-nebula background
  - compute_mosaic_geometry must place panels that only overlap a *neighbour*
    (A->B->anchor chain), not just panels that overlap a single reference.
"""
import numpy as np

from oeuvre.star_match import detect_stars
from oeuvre.natural_narrowband import compute_mosaic_geometry


def _field(stars, shape, amp=1.0, bg=0.05, noise=0.004, seed=0, nebula=0.0):
    """Render a star field, optionally with a smooth bright nebula gradient."""
    rng = np.random.default_rng(seed)
    img = rng.normal(bg, noise, shape).astype(np.float32)
    if nebula > 0:
        ys, xs = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        cy, cx = shape[0] * 0.5, shape[1] * 0.5
        r2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (shape[1] * 0.35) ** 2)
        img += (nebula * np.exp(-r2)).astype(np.float32)
    for (x, y) in stars:
        xi, yi = int(round(x)), int(round(y))
        if 1 <= xi < shape[1] - 1 and 1 <= yi < shape[0] - 1:
            img[yi - 1:yi + 2, xi - 1:xi + 2] += amp
    return img


# A dense, well-separated global star catalogue spanning a wide field.
_RNG = np.random.default_rng(42)
_GLOBAL = [(float(x), float(y))
           for x, y in _RNG.uniform([20, 20], [680, 280], size=(70, 2))]


def _panel(x0, shape=(300, 300), **kw):
    """Crop the global catalogue into a panel whose origin is at x0."""
    local = [(gx - x0, gy) for (gx, gy) in _GLOBAL
             if x0 <= gx < x0 + shape[1] and 0 <= gy < shape[0]]
    return _field(local, shape, **kw), local


def test_detect_stars_survives_nebula():
    """High-pass detection finds stars buried under bright nebula that the
    raw (no-prefilter) path misses."""
    stars = [(float(x), float(y))
             for x, y in _RNG.uniform([20, 20], [236, 236], size=(30, 2))]
    img = _field(stars, (256, 256), amp=0.6, nebula=0.9, seed=1)

    n_hp = len(detect_stars(img)[0])                       # default high-pass on
    n_raw = len(detect_stars(img, highpass_sigma=0.0)[0])  # legacy behaviour

    assert n_hp >= 20, f"high-pass found only {n_hp} stars under nebula"
    assert n_hp > n_raw, f"high-pass ({n_hp}) should beat raw ({n_raw})"


def test_mosaic_places_chained_panel():
    """Panel C overlaps only panel B, which overlaps the anchor. All three
    must be placed (the old single-reference matcher dropped C)."""
    anchor, _ = _panel(0)      # x 0..300
    mid, _ = _panel(200)       # x 200..500  (overlaps anchor)
    far, _ = _panel(400)       # x 400..700  (overlaps mid only, NOT anchor)

    geo = compute_mosaic_geometry([anchor, mid, far], log=lambda *a: None)
    kept = geo['kept_indices']

    assert kept == [0, 1, 2], f"expected all panels placed, got {kept}"

    # Transforms map panel-local -> anchor-local (anchor = whichever panel has
    # the most stars), so check the anchor-independent invariant: consecutive
    # panels are 200 px apart in x and aligned in y.
    tx = [geo['transforms'][i][0, 2] for i in range(3)]
    ty = [geo['transforms'][i][1, 2] for i in range(3)]
    assert abs((tx[1] - tx[0]) - 200) < 3, f"anchor->mid dx={tx[1] - tx[0]}"
    assert abs((tx[2] - tx[1]) - 200) < 3, f"mid->far dx={tx[2] - tx[1]}"
    assert abs(ty[1] - ty[0]) < 3 and abs(ty[2] - ty[1]) < 3, f"ty={ty}"


def test_mosaic_drops_only_truly_isolated_panel():
    """A panel sharing no stars with any other is still dropped."""
    anchor, _ = _panel(0)
    mid, _ = _panel(180)
    isolated = _field([(float(x), float(y))
                       for x, y in _RNG.uniform([20, 20], [280, 280], (25, 2))],
                      (300, 300), seed=99)  # unrelated catalogue

    geo = compute_mosaic_geometry([anchor, mid, isolated], log=lambda *a: None)
    assert set(geo['kept_indices']) == {0, 1}
