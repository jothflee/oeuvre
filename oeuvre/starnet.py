#!/usr/bin/env python3
"""
StarNet star removal — pure-local replacement for Siril's `starnet` command.

Shells out to the bundled StarNet++ v2 CLI (StarNetv2CLI_MacOS/starnet++),
which operates on 16-bit TIFF only and loads its weights (`starnet2_weights.pb`)
plus the tensorflow dylibs from its own directory — so the binary MUST be run
with cwd set to that directory.

Public API mirrors the old `siril_star_removal`:
    remove_stars(rgb_fits_path, work_dir, log) -> (starless, stars)  # [H,W,3] float32
"""

import os
import subprocess

import numpy as np
import cv2

from .config import workspace


def find_starnet():
    """Locate the StarNet++ CLI directory and binary.

    Returns (binary_path, starnet_dir) or (None, None) if not found.
    Honors the STARNET_DIR environment variable as an override, then looks for
    ``StarNetv2CLI_MacOS/`` under the workspace and the current directory.
    """
    candidates = []
    env_dir = os.environ.get('STARNET_DIR')
    if env_dir:
        candidates.append(env_dir)
    candidates += [
        os.path.join(workspace(), 'StarNetv2CLI_MacOS'),
        os.path.join(os.getcwd(), 'StarNetv2CLI_MacOS'),
    ]
    for d in candidates:
        binp = os.path.join(d, 'starnet++')
        if os.path.isfile(binp) and os.access(binp, os.X_OK):
            return binp, d
    return None, None


def _to_hwc(data):
    """Convert FITS [3,H,W] (or [H,W]) to [H,W,3] float32."""
    if data.ndim == 3 and data.shape[0] == 3:
        return np.transpose(data, (1, 2, 0)).astype(np.float32)
    if data.ndim == 2:
        return np.stack([data] * 3, axis=-1).astype(np.float32)
    return data.astype(np.float32)


def _write_tiff16(rgb_hwc, path):
    """Write [H,W,3] float [0,1] as a 16-bit RGB TIFF (cv2 wants BGR)."""
    u16 = np.clip(rgb_hwc, 0.0, 1.0)
    u16 = np.rint(u16 * 65535.0).astype(np.uint16)
    if not cv2.imwrite(path, u16[:, :, ::-1]):  # RGB -> BGR
        raise RuntimeError(f"Failed to write 16-bit TIFF: {path}")


def _read_tiff16(path):
    """Read a 16-bit TIFF back to [H,W,3] float32 in [0,1] (cv2 gives BGR)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"StarNet output not readable: {path}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    rgb = img[:, :, ::-1].astype(np.float32)  # BGR -> RGB
    return rgb / 65535.0


def remove_stars(rgb_fits_path, work_dir, log=print, timeout=1800):
    """Remove stars from a stretched RGB FITS using local StarNet++.

    Args:
        rgb_fits_path: path to a [3,H,W] (or [H,W]) FITS, values in [0,1].
        work_dir: scratch directory for the intermediate TIFFs.
        log: logging callable.
        timeout: seconds before giving up on the StarNet process.

    Returns:
        (starless, stars) as float32 [H,W,3] arrays. stars = original - starless.
    """
    # Late import avoids a circular dependency with natural_narrowband.
    from .natural_narrowband import load_fits

    binp, sn_dir = find_starnet()
    if binp is None:
        raise RuntimeError(
            "StarNet++ binary not found. Expected "
            f"{os.path.join(workspace(), 'StarNetv2CLI_MacOS', 'starnet++')} "
            "or set STARNET_DIR."
        )

    original = _to_hwc(load_fits(rgb_fits_path)[0])

    in_tif = os.path.join(work_dir, '_starnet_in.tif')
    out_tif = os.path.join(work_dir, '_starnet_out.tif')
    _write_tiff16(original, in_tif)

    log(f"  Running StarNet++ ({original.shape[1]}x{original.shape[0]})...")
    # Run from the StarNet dir so it finds its weights + tensorflow dylibs.
    result = subprocess.run(
        [binp, in_tif, out_tif],
        cwd=sn_dir,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0 or not os.path.exists(out_tif):
        err = (result.stderr or result.stdout)[:500]
        raise RuntimeError(f"StarNet failed (exit {result.returncode}):\n{err}")

    starless = _read_tiff16(out_tif)

    # Match shapes defensively (StarNet preserves dimensions, but be safe).
    if starless.shape != original.shape:
        h = min(starless.shape[0], original.shape[0])
        w = min(starless.shape[1], original.shape[1])
        starless = starless[:h, :w]
        original = original[:h, :w]

    stars = np.clip(original - starless, 0.0, None).astype(np.float32)

    log(f"  Starless range: [{np.min(starless):.4f}, {np.max(starless):.4f}]")
    log(f"  Stars range:    [{np.min(stars):.4f}, {np.max(stars):.4f}]")

    # Clean up intermediates.
    for p in (in_tif, out_tif):
        try:
            os.unlink(p)
        except OSError:
            pass

    return starless, stars
