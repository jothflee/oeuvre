#!/usr/bin/env python3
"""
StarNet star removal — pure-local replacement for Siril's `starnet` command.

The app can manage a local StarNet v2 install under ``~/oeuvre`` but never
rehosts the binary. The StarNet CLI operates on 16-bit TIFF only and loads its
weights (``starnet2_weights.pb``) plus the tensorflow dylibs from its own
directory — so the binary MUST be run with cwd set to that directory.

Public API mirrors the old `siril_star_removal`:
    remove_stars(rgb_fits_path, work_dir, log) -> (starless, stars)  # [H,W,3] float32
"""

import os
import platform
import shutil
import subprocess
import threading

import numpy as np
import cv2


def _rosetta_note(binp):
    """Warn string if a non-arm64 binary will run under Rosetta on this Mac."""
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return ''
    try:
        archs = subprocess.run(['/usr/bin/file', '-b', binp],
                               capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ''
    if 'arm64' not in archs and 'x86_64' in archs:
        return ('  NOTE: this is an x86_64 StarNet build running under Rosetta '
                'on Apple Silicon — much slower. Use an arm64 build for speed.')
    return ''

from .config import workspace, app_home, starnet_dir_setting

STARNET_DIR_NAME = 'StarNetv2CLI_MacOS'  # legacy default folder, still auto-found
# Known StarNet CLI executables, in preference order. 'starnet2' is the modern
# StarNet v2 (ONNX Runtime) build; 'starnet++' is the legacy binary. We prefer
# v2 wherever both exist.
STARNET_BINARY_NAMES = ('starnet2', 'starnet++')
STARNET_OFFICIAL_URL = 'https://starnetastro.com/cli-tools/starnet/'


def managed_starnet_dir():
    """Default local install directory under ~/oeuvre."""
    return os.path.join(app_home(), STARNET_DIR_NAME)


def starnet_binary_path(starnet_dir, names=None):
    """Return the path to a usable StarNet binary inside *starnet_dir*.

    Checks the known executable names (``names`` overrides, default preference
    order = StarNet v2 first), returning the first one that exists and is
    executable, or ``None`` if the folder has no usable StarNet binary.
    """
    if not starnet_dir:
        return None
    for cand in (names or STARNET_BINARY_NAMES):
        p = os.path.join(starnet_dir, cand)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def strip_quarantine(path, log=None):
    """Remove the macOS com.apple.quarantine xattr from *path* (recursively).

    Browser-downloaded StarNet builds arrive quarantined; on first real use the
    CLI loads its bundled runtime dylibs and Gatekeeper stalls verifying each
    quarantined file, which looks exactly like a hang. Clearing the attribute up
    front avoids that. No-op off macOS / on failure.
    """
    import sys
    import subprocess
    if sys.platform != 'darwin' or not path or not os.path.exists(path):
        return
    try:
        r = subprocess.run(['xattr', '-dr', 'com.apple.quarantine', path],
                           capture_output=True, timeout=30)
        if log and r.returncode == 0:
            log(f"  Cleared macOS quarantine on {os.path.basename(path)}")
    except Exception:
        pass


def _starnet_argv(binp, in_tif, out_tif):
    """Build the CLI argv for whichever StarNet binary *binp* is.

    StarNet v2 ('starnet2', ONNX Runtime) takes named --input/--output options;
    the legacy 'starnet++' takes positional input/output. Both load their model
    weights from their own directory, so callers run with cwd set there.
    """
    name = os.path.basename(binp).lower()
    if name.startswith('starnet2'):
        return [binp, '--input', in_tif, '--output', out_tif]
    return [binp, in_tif, out_tif]


def _is_valid_starnet_dir(path):
    """True when *path* contains a usable StarNet binary (v2 or legacy)."""
    if not path or not os.path.isdir(path):
        return False
    return starnet_binary_path(path) is not None


def resolve_starnet_source(path):
    """Resolve a user-selected path to a usable StarNet folder.

    Accepts the folder itself, a path to a known StarNet binary, or a parent
    that contains a StarNetv2CLI_MacOS subfolder. Returns the folder, or None.
    """
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        parent = os.path.dirname(path)
        if os.path.basename(path) in STARNET_BINARY_NAMES:
            return parent if _is_valid_starnet_dir(parent) else None
        return None
    if _is_valid_starnet_dir(path):
        return path
    nested = os.path.join(path, STARNET_DIR_NAME)
    if _is_valid_starnet_dir(nested):
        return nested
    return None


def installed_starnet_dir():
    """Return the local managed StarNet directory if present."""
    path = managed_starnet_dir()
    return path if _is_valid_starnet_dir(path) else None


def starnet_status():
    """Summarize the current StarNet discovery state."""
    binp, sn_dir = find_starnet()
    managed_dir = installed_starnet_dir()
    chosen = starnet_dir_setting() or None
    return {
        'available': bool(binp),
        'binary': binp,
        'binary_name': os.path.basename(binp) if binp else None,
        'dir': sn_dir,
        'managed_dir': managed_dir,
        'managed_root': managed_starnet_dir(),
        'chosen_dir': chosen,
        # True when the active install is the one the user explicitly chose.
        'chosen_active': bool(chosen and sn_dir
                              and os.path.abspath(chosen) == os.path.abspath(sn_dir)),
    }


def uninstall_managed_starnet():
    """Remove the managed local StarNet install, if one exists."""
    dest = managed_starnet_dir()
    if os.path.islink(dest) or os.path.isfile(dest):
        os.unlink(dest)
        return True
    if os.path.isdir(dest):
        shutil.rmtree(dest)
        return True
    return False


def find_starnet():
    """Locate a StarNet CLI directory and binary.

    Returns (binary_path, starnet_dir) or (None, None).

    Resolution order:
      1. Explicit overrides — the STARNET_DIR env var, then the user-chosen
         folder from settings — win and use whatever binary they contain.
      2. Otherwise search the managed install, the legacy workspace/cwd folders,
         and any ``*starnet*`` folder under the workspace / app home (so a
         versioned download like ``starnet2_macos-..._cli`` is auto-discovered),
         preferring a StarNet v2 ('starnet2') binary over the legacy 'starnet++'.
    """
    # 1. Explicit overrides — honor the user's exact choice (any binary).
    for d in (os.environ.get('STARNET_DIR'), starnet_dir_setting()):
        if d:
            binp = starnet_binary_path(d)
            if binp:
                return binp, d

    # 2. Known + discovered locations.
    candidates = [
        managed_starnet_dir(),
        os.path.join(workspace(), STARNET_DIR_NAME),
        os.path.join(os.getcwd(), STARNET_DIR_NAME),
    ]
    for root in (workspace(), app_home()):
        try:
            candidates += [
                os.path.join(root, e) for e in sorted(os.listdir(root))
                if 'starnet' in e.lower()
                and os.path.isdir(os.path.join(root, e))
            ]
        except OSError:
            pass

    seen, uniq = set(), []
    for d in candidates:
        ad = os.path.abspath(d)
        if ad not in seen:
            seen.add(ad)
            uniq.append(d)

    # Prefer StarNet v2 anywhere before falling back to the legacy binary.
    for prefer in (('starnet2',), ('starnet++',)):
        for d in uniq:
            binp = starnet_binary_path(d, names=prefer)
            if binp:
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
            "StarNet binary not found (looked for "
            f"{' / '.join(STARNET_BINARY_NAMES)}). Choose a StarNet folder in "
            "Oeuvre's settings or set STARNET_DIR."
        )

    # Clear quarantine so loading the bundled runtime dylibs can't stall.
    strip_quarantine(sn_dir, log=log)

    original = _to_hwc(load_fits(rgb_fits_path)[0])

    # Absolute paths so they resolve regardless of the StarNet cwd.
    in_tif = os.path.abspath(os.path.join(work_dir, '_starnet_in.tif'))
    out_tif = os.path.abspath(os.path.join(work_dir, '_starnet_out.tif'))
    _write_tiff16(original, in_tif)

    mp = original.shape[0] * original.shape[1] / 1e6
    log(f"  Running {os.path.basename(binp)} "
        f"({original.shape[1]}x{original.shape[0]}, {mp:.1f} MP) — "
        f"large images can take several minutes...")
    note = _rosetta_note(binp)
    if note:
        log(note)

    # Heartbeat so a long (silent, captured) run doesn't look hung.
    done = threading.Event()

    def _heartbeat():
        secs = 0
        while not done.wait(30):
            secs += 30
            log(f"  ...{os.path.basename(binp)} still running "
                f"({secs // 60}m{secs % 60:02d}s)")

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()
    # Run from the StarNet dir so it finds its bundled weights + runtime libs.
    try:
        result = subprocess.run(
            _starnet_argv(binp, in_tif, out_tif),
            cwd=sn_dir,
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        done.set()
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
