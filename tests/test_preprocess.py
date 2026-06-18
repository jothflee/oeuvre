"""Tests for pure-Python calibration, registration, and stacking."""
import numpy as np

from oeuvre import preprocess
from oeuvre.preprocess import (
    _sigma_clip_mean, _winsorized_sigma_stack, make_bad_pixel_map,
    calibrate_light, register_frames, stack_frames,
)
from conftest import star_image, STARS


def test_sigma_clip_mean_rejects_outlier():
    # Many frames so a lone extreme outlier is genuinely rejected (mirrors a
    # cosmic-ray hit in a master-dark stack).
    rng = np.random.default_rng(0)
    base = (10.0 + rng.normal(0, 0.3, (30, 8, 8))).astype(np.float32)
    base[7, 4, 4] = 1000.0
    out = _sigma_clip_mean(base)
    assert abs(out[4, 4] - 10.0) < 1.0     # outlier rejected, not averaged in


def test_winsorized_stack_nan_aware():
    # NaN (out-of-frame) pixels are ignored, not propagated.
    stack = np.stack([np.full((4, 4), 5.0, np.float32) for _ in range(4)])
    stack[0, 0, 0] = np.nan
    stack[1, 1, 1] = np.nan
    out = _winsorized_sigma_stack(stack)
    assert np.isfinite(out).all()
    assert np.allclose(out, 5.0, atol=1e-4)


def test_winsorized_stack_reduces_outlier():
    rng = np.random.default_rng(0)
    stack = (10.0 + rng.normal(0, 0.3, (30, 4, 4))).astype(np.float32)
    stack[5, 2, 2] = 1000.0
    naive = float(stack[:, 2, 2].mean())
    out = _winsorized_sigma_stack(stack)
    assert np.isfinite(out).all()
    assert out[2, 2] < naive                # outlier influence reduced


def test_bad_pixel_map_flags_hot_pixel():
    dark = np.full((16, 16), 2.0, np.float32)
    dark[8, 8] = 100.0
    bpm = make_bad_pixel_map(dark)
    assert bpm[8, 8]
    assert bpm.sum() <= 3                   # essentially just the hot pixel


def test_calibrate_light_subtracts_and_repairs():
    dark = np.full((16, 16), 2.0, np.float32)
    dark[8, 8] = 100.0                       # hot pixel
    bpm = make_bad_pixel_map(dark)
    light = np.full((16, 16), 12.0, np.float32)
    light[8, 8] = 110.0
    cal = calibrate_light(light, dark, bpm)
    assert abs(cal[0, 0] - 10.0) < 1e-4      # 12 - 2
    assert cal[8, 8] < 50.0                  # hot pixel repaired to neighbours


def test_register_and_stack_roundtrip():
    f1 = star_image(STARS, seed=1)
    f2 = star_image([(x + 4, y - 3) for (x, y) in STARS], seed=2)
    f3 = star_image(STARS, seed=3)
    aligned, ref_idx = register_frames([f1, f2, f3])
    assert len(aligned) == 3
    assert 0 <= ref_idx < 3
    for a in aligned:
        assert a.shape == f1.shape
    master = stack_frames(aligned, ref_idx)
    assert np.isfinite(master).all()
    assert master.min() >= 0.0 and master.max() <= 1.0


def test_sum_livetime_handles_missing(tmp_path):
    # No EXPTIME headers / unreadable -> 0, no crash
    assert preprocess._sum_livetime([]) == 0


def test_reject_low_quality_drops_cloudy_subs():
    from oeuvre.preprocess import _reject_low_quality
    counts = [150, 160, 140, 8, 155, 150, 145, 9]   # idx 3,7 cloudy (few stars)
    bgs = [0.01] * 8
    bgs[3] = 0.05                                    # idx 3 also high background
    keep = _reject_low_quality(counts, bgs, [f's{i}' for i in range(8)],
                               log=lambda *a: None)
    assert 3 not in keep and 7 not in keep
    assert set(keep) == {0, 1, 2, 4, 5, 6}


def test_reject_low_quality_keep_floor():
    from oeuvre.preprocess import _reject_low_quality
    # All look bad relative to a high-count outlier, but the floor must hold.
    counts = [3, 3, 3, 3, 100]
    bgs = [0.01] * 5
    keep = _reject_low_quality(counts, bgs, [f's{i}' for i in range(5)],
                               log=lambda *a: None)
    assert len(keep) >= 5  # min_keep floor → nothing dropped below it
