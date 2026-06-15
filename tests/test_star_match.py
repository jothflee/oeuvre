"""Tests for the shared star-centroid + asterism matcher."""
import numpy as np
import cv2

from oeuvre.star_match import detect_stars, match_and_solve
from conftest import star_image, STARS


def test_detect_stars_finds_injected():
    img = star_image(STARS)
    xy, flux = detect_stars(img)
    assert len(xy) == len(STARS)
    # every injected star should have a detected centroid within ~1px
    for (sx, sy) in STARS:
        d = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)
        assert d.min() < 1.5


def test_detect_stars_empty_on_flat():
    xy, flux = detect_stars(np.zeros((128, 128), np.float32))
    assert len(xy) == 0


def test_match_identity():
    img = star_image(STARS)
    M, inliers = match_and_solve(img, img)
    assert M is not None and inliers > 0
    assert abs(np.hypot(M[0, 0], M[1, 0]) - 1.0) < 0.01     # scale ~1
    assert abs(M[0, 2]) < 0.5 and abs(M[1, 2]) < 0.5         # translation ~0


def test_match_recovers_translation():
    tx, ty = 10, -7
    dst = star_image(STARS, seed=1)
    src = star_image([(x + tx, y + ty) for (x, y) in STARS], seed=2)
    M, inliers = match_and_solve(src, dst)
    assert M is not None and inliers >= 8
    assert abs(np.hypot(M[0, 0], M[1, 0]) - 1.0) < 0.02          # scale ~1
    ang = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    assert ang < 0.5                                             # rotation ~0
    # M maps src->dst, i.e. translates by -(tx, ty)
    assert abs(M[0, 2] + tx) < 1.5 and abs(M[1, 2] + ty) < 1.5


def test_match_too_few_stars_returns_none():
    a = star_image([(60, 60)])
    b = star_image([(80, 80)])
    M, inliers = match_and_solve(a, b)
    assert M is None and inliers == 0
