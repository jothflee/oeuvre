"""Tests for config, filter classification, clustering, FITS I/O, color math."""
import os
import numpy as np
import pytest

from oeuvre import config
from oeuvre.pipeline import classify_filter, _auto_star_consensus_from_stats
from oeuvre.mosaic_prep import cluster_frames
from oeuvre.natural_narrowband import (
    load_fits, save_fits, estimate_background, arcsinh_stretch,
    find_arcsinh_beta, screen_blend,
)


# ── config / workspace ────────────────────────────────────────────────────

def test_workspace_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv('OEUVRE_WORKSPACE', str(tmp_path))
    assert config.workspace() == str(tmp_path)


def test_workspace_default(monkeypatch):
    monkeypatch.delenv('OEUVRE_WORKSPACE', raising=False)
    assert config.workspace() == os.path.expanduser('~/oeuvre-astro')


def test_ensure_workspace_creates(monkeypatch, tmp_path):
    ws = tmp_path / 'made'
    monkeypatch.setenv('OEUVRE_WORKSPACE', str(ws))
    config.ensure_workspace()
    assert ws.is_dir()


# ── filter classification / consensus ─────────────────────────────────────

@pytest.mark.parametrize('name,expected', [
    ('Ha-7nm', 'ha'), ('SII-7nm', 'sii'), ('OIII-7nm', 'oiii'),
    ('S2', 'sii'), ('O3', 'oiii'), ('Lum', None),
])
def test_classify_filter(name, expected):
    assert classify_filter(name) == expected


def test_auto_star_consensus():
    assert _auto_star_consensus_from_stats({'a': (5,), 'b': (50,)}) == 'strict'
    assert _auto_star_consensus_from_stats({'a': (50,), 'b': (50,)}) == 'off'
    assert _auto_star_consensus_from_stats({}) == 'soft'


# ── clustering (scipy replacement for sklearn) ─────────────────────────────

def test_cluster_frames_two_groups():
    frames = [('a', 10.0, 20.0), ('b', 10.04, 20.03), ('c', 10.02, 20.01),
              ('d', 11.0, 21.0), ('e', 11.03, 21.02)]
    clusters = cluster_frames(frames, 0.15)
    assert len(clusters) == 2
    assert sorted(len(c[2]) for c in clusters) == [2, 3]


def test_cluster_frames_edges():
    assert cluster_frames([], 0.15) == []
    one = cluster_frames([('a', 1.0, 2.0)], 0.15)
    assert len(one) == 1 and len(one[0][2]) == 1


# ── FITS round-trip ────────────────────────────────────────────────────────

def test_fits_roundtrip_2d(tmp_path):
    data = np.random.default_rng(0).random((32, 48)).astype(np.float32)
    p = str(tmp_path / 'm.fit')
    save_fits(data, p)
    back, hdr = load_fits(p)
    assert back.shape == data.shape
    assert np.allclose(back, data, atol=1e-5)


def test_fits_roundtrip_3d(tmp_path):
    data = np.random.default_rng(1).random((3, 16, 20)).astype(np.float32)
    p = str(tmp_path / 'rgb.fit')
    save_fits(data, p)
    back, _ = load_fits(p)
    assert back.shape == data.shape
    assert np.allclose(back, data, atol=1e-5)


# ── color / stretch math ───────────────────────────────────────────────────

def test_estimate_background():
    rng = np.random.default_rng(0)
    img = rng.normal(0.1, 0.005, (200, 200)).astype(np.float32)
    img[::50, ::50] = 0.9                       # sparse bright stars
    bg, sigma = estimate_background(img)
    assert abs(bg - 0.1) < 0.01
    assert sigma > 0


def test_arcsinh_stretch_endpoints_monotonic():
    x = np.linspace(0, 1, 50).astype(np.float32)
    y = arcsinh_stretch(x, beta=10.0)
    assert abs(y[0]) < 1e-6 and abs(y[-1] - 1.0) < 1e-5
    assert np.all(np.diff(y) >= -1e-7)          # monotonic increasing


def test_find_arcsinh_beta_roundtrip():
    beta = find_arcsinh_beta(0.05, 0.25)
    assert arcsinh_stretch(np.array([0.05]), beta)[0] == pytest.approx(0.25, abs=1e-3)


def test_screen_blend():
    a = np.array([[0.0, 0.5, 1.0]], np.float32)
    b = np.array([[0.0, 0.5, 1.0]], np.float32)
    out = screen_blend(a, b)
    # 1 - (1-a)(1-b)
    assert np.allclose(out, [0.0, 0.75, 1.0], atol=1e-6)
