#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SHO Hubble Palette Processor v2.0                       ║
║          Mathematically Rigorous Narrowband Image Processing               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Takes 3 unstretched narrowband masters (SII, Ha, OIII) and produces a
properly stretched, balanced Hubble-palette image with professional
star handling.

Pipeline:
  1. Load & inspect channels
  2. Star alignment  (Siril 2-pass deep sky + COG framing)
  3. Channel normalization (background-subtract + peak-normalize)
  4. SHO → RGB mapping  (S→R, H→G, O→B)
  5. Linked arcsinh stretch (calibrated from luminance statistics)
  6. Star removal  (Siril StarNet)
  7. SCNR green removal on starless (luminosity-preserving)
  8. Color balance + Hubble palette tuning on starless
  9. Star desaturation + anti-purple
 10. Screen-blend recombination
 11. Final output (FITS + TIFF + PNG)

Each step shows an interactive cv2 preview in a montage window.

Math foundations:
  - Arcsinh stretch:  f(x) = arcsinh(β·x) / arcsinh(β)
    β solved numerically so that median → target_median
  - SCNR Maximum Neutral:  G' = G − max(0, G − max(R,B))·amount
    with luminosity restoration: scale all channels by L_before / L_after
  - Screen blend:  result = 1 − (1−base)·(1−overlay)

Usage:
  python natural_narrowband.py \\
      --sii sii.fit --ha ha.fit --oiii oiii.fit \\
      [--output-dir ./output] [--no-preview] [--interactive]

Requirements: numpy, opencv-python (cv2)
Optional:     astropy (used if available, otherwise built-in FITS I/O)
"""

import os
import math
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2

from .mosaic_prep import read_fits_header
from .starnet import remove_stars

VERSION = "2.0.0"


def _parallel_map(fn, items, max_workers=None):
    """Order-preserving threaded map for CPU/IO work that releases the GIL.

    The heavy primitives here (cv2 warps/blurs, numpy reductions, FITS reads)
    release the GIL, so threads give real speedup. Falls back to a serial map
    for 0/1 items. Workers are capped to keep memory bounded.
    """
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    workers = max_workers or min(len(items), (os.cpu_count() or 4))
    workers = max(1, min(workers, 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def _smooth_lowpass(img, sigma):
    """Large-sigma Gaussian low-pass, fast.

    The seam-correction field is intentionally very smooth, so a full-res blur
    with a huge kernel (sigma can be hundreds of px) is wasteful. Downsampling,
    blurring with a proportionally smaller kernel, and upsampling is visually
    identical here at a tiny fraction of the cost.
    """
    h, w = img.shape[:2]
    f = max(1, int(sigma // 8))  # downsample factor
    if f > 1:
        small = cv2.resize(img, (max(1, w // f), max(1, h // f)),
                           interpolation=cv2.INTER_AREA)
        s = max(0.8, sigma / f)
        k = max(3, int(s * 4) | 1)
        small = cv2.GaussianBlur(small, (k, k), s)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(img, (k, k), sigma)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Constants                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

PREVIEW_MAX_DIM   = 1400        # max pixel dimension for preview window

STRETCH_TARGET    = 0.18        # target median after stretch (0.20–0.30)
SCNR_AMOUNT       = 1.00       # green removal strength (0.0–1.0)
STAR_DESAT        = 0.70        # star desaturation amount (0.0–1.0)
SAT_BOOST         = 1.25        # nebula saturation boost factor
BG_SAMPLE_FRAC    = 0.15        # fraction of darkest pixels for bg estimation

# CIE 1931 luminance coefficients
LR, LG, LB = 0.2126, 0.7152, 0.0722


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FITS I/O  (pure-Python fallback, astropy used if available)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _parse_fits_header(f):
    """Parse FITS header from an open binary file. Returns dict."""
    header = {}
    while True:
        block = f.read(2880)
        if len(block) < 2880:
            raise ValueError("Truncated FITS header")
        done = False
        for i in range(0, 2880, 80):
            card = block[i:i+80].decode('ascii', errors='replace')
            key = card[:8].strip()
            if key == 'END':
                done = True
                break
            if '=' in card[8:10]:
                val_comment = card[10:]
                # Handle string values
                if "'" in val_comment:
                    start = val_comment.index("'")
                    end = val_comment.index("'", start + 1)
                    header[key] = val_comment[start+1:end].strip()
                else:
                    val = val_comment.split('/')[0].strip()
                    if val == 'T':
                        header[key] = True
                    elif val == 'F':
                        header[key] = False
                    else:
                        try:
                            header[key] = int(val)
                        except ValueError:
                            try:
                                header[key] = float(val)
                            except ValueError:
                                header[key] = val
        if done:
            break
    return header


def load_fits(path):
    """Load a FITS file. Returns (data: float32 ndarray, header: dict).

    For 2D mono: shape = (H, W)
    For 3D RGB:  shape = (3, H, W)  (Siril convention)

    Data is returned as-stored — no row-order flipping.
    """
    path = str(path)
    if not os.path.exists(path):
        if os.path.exists(path + '.fit'):
            path = path + '.fit'
        elif os.path.exists(path + '.fits'):
            path = path + '.fits'
        else:
            raise FileNotFoundError(f"FITS file not found: {path}")

    # Try astropy first
    try:
        from astropy.io import fits as astropy_fits
        with astropy_fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = dict(hdul[0].header)

        # Standard FITS is BOTTOM-UP (first row = bottom of image).
        # Flip vertically so row[0] = top, matching display/processing convention.
        if header.get('ROWORDER', 'BOTTOM-UP').upper() == 'BOTTOM-UP':
            data = np.flip(data, axis=-2).copy()

        return data, header
    except ImportError:
        pass

    # Pure-Python FITS reader
    with open(path, 'rb') as f:
        header = _parse_fits_header(f)

        bitpix = header.get('BITPIX', -32)
        naxis = header.get('NAXIS', 2)

        if naxis == 2:
            shape = (header['NAXIS2'], header['NAXIS1'])
        elif naxis == 3:
            shape = (header['NAXIS3'], header['NAXIS2'], header['NAXIS1'])
        else:
            raise ValueError(f"Unsupported NAXIS={naxis}")

        n_pixels = 1
        for s in shape:
            n_pixels *= s

        # Seek to start of data (next 2880-byte boundary)
        pos = f.tell()
        data_start = ((pos + 2879) // 2880) * 2880
        f.seek(data_start)

        dtype_map = {-32: '>f4', -64: '>f8', 16: '>i2', 32: '>i4', 8: '>u1'}
        if bitpix not in dtype_map:
            raise ValueError(f"Unsupported BITPIX={bitpix}")

        dt = np.dtype(dtype_map[bitpix])
        raw = f.read(n_pixels * dt.itemsize)
        data = np.frombuffer(raw, dtype=dt).reshape(shape).astype(np.float32)

        # Apply BZERO / BSCALE
        bscale = header.get('BSCALE', 1.0)
        bzero = header.get('BZERO', 0.0)
        if isinstance(bscale, (int, float)) and isinstance(bzero, (int, float)):
            if bscale != 1.0 or bzero != 0.0:
                data = data * float(bscale) + float(bzero)

        # For 16-bit unsigned (BITPIX=16, BZERO=32768), normalize to [0,1]
        if bitpix == 16 and bzero == 32768:
            data = data / 65535.0

        # Standard FITS is BOTTOM-UP — flip so row[0] = top of image.
        if header.get('ROWORDER', 'BOTTOM-UP').upper() == 'BOTTOM-UP':
            data = np.flip(data, axis=-2).copy()

    return data, header


def save_fits(data, path, header_extra=None):
    """Save a numpy array as a FITS file.

    data: float32, shape (H, W) for mono or (3, H, W) for RGB.
    Data is written as-stored — no row-order flipping.
    """
    path = str(path)
    data = data.astype(np.float32)

    roworder = 'TOP-DOWN'

    # Try astropy first
    try:
        from astropy.io import fits as astropy_fits
        hdu = astropy_fits.PrimaryHDU(data=data)
        hdu.header['ROWORDER'] = roworder
        hdu.header['PROGRAM'] = f'SHO_Pipeline v{VERSION}'
        _MANDATORY = frozenset({
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
            'BZERO', 'BSCALE', 'ROWORDER', 'PROGRAM', 'EXTEND',
        })
        if header_extra:
            for k, v in header_extra.items():
                if str(k).upper().strip() in _MANDATORY:
                    continue
                try:
                    hdu.header[k] = v
                except Exception:
                    pass
        hdu.writeto(path, overwrite=True)
        return
    except ImportError:
        pass

    # Pure-Python FITS writer
    ndim = len(data.shape)
    cards = []
    cards.append(("SIMPLE", True))
    cards.append(("BITPIX", -32))
    cards.append(("NAXIS", ndim))
    cards.append(("NAXIS1", data.shape[-1]))
    cards.append(("NAXIS2", data.shape[-2]))
    if ndim == 3:
        cards.append(("NAXIS3", data.shape[0]))
    cards.append(("BZERO", 0.0))
    cards.append(("BSCALE", 1.0))
    cards.append(("ROWORDER", roworder))
    cards.append(("PROGRAM", f"SHO_Pipeline v{VERSION}"))

    # Keywords already written as mandatory cards — skip if in header_extra
    _MANDATORY = frozenset({
        'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
        'BZERO', 'BSCALE', 'ROWORDER', 'PROGRAM', 'EXTEND',
    })
    if header_extra:
        for k, v in header_extra.items():
            if str(k).upper().strip() in _MANDATORY:
                continue
            cards.append((str(k)[:8], v))

    # Build header block
    header_bytes = b''
    for key, val in cards:
        key_s = key.ljust(8)[:8]
        if isinstance(val, bool):
            val_s = ('T' if val else 'F').rjust(20)
        elif isinstance(val, int):
            val_s = str(val).rjust(20)
        elif isinstance(val, float):
            val_s = f'{val:20.10E}'
        elif isinstance(val, str):
            # FITS string value: must be enclosed in single quotes, left-justified
            # within an 8-68 character field
            val_s = ("'" + val + "'").ljust(20)
        else:
            val_s = str(val).rjust(20)
        card = f"{key_s}= {val_s}".ljust(80)[:80]
        header_bytes += card.encode('ascii', errors='replace')

    header_bytes += 'END'.ljust(80).encode('ascii')

    # Pad to 2880 boundary
    pad_len = (2880 - len(header_bytes) % 2880) % 2880
    header_bytes += b' ' * pad_len

    # Write file
    with open(path, 'wb') as f:
        f.write(header_bytes)
        f.write(data.astype('>f4').tobytes())
        # Pad data to 2880 boundary
        data_len = data.nbytes
        pad_len = (2880 - data_len % 2880) % 2880
        if pad_len:
            f.write(b'\x00' * pad_len)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Preview System (cv2)                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PipelinePreview:
    """Single-window grid preview that builds up as panels are processed.

    Phases shown in the grid:
      1. Unstretched RGB composite (per panel)
      2. Fully processed panel (per panel, updated in-place)
      3. Mosaic result (full window)
      4. Final product (full window)
    """

    def __init__(self, enabled=True, interactive=False):
        self.enabled = enabled
        self.interactive = interactive
        self._win = "SHO Pipeline"
        self._n_panels = 0
        self._grid_cols = 0
        self._grid_rows = 0
        self._panel_images = []   # BGR uint8 per cell
        self._grid_mode = True    # False when showing full image
        if enabled:
            cv2.namedWindow(self._win,
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # ── public API ──

    def init_grid(self, n_panels):
        """Initialize the panel grid layout."""
        self._n_panels = n_panels
        self._grid_mode = True
        if n_panels <= 1:
            self._grid_cols, self._grid_rows = 1, 1
        elif n_panels <= 2:
            self._grid_cols, self._grid_rows = 2, 1
        elif n_panels <= 4:
            self._grid_cols, self._grid_rows = 2, 2
        elif n_panels <= 6:
            self._grid_cols, self._grid_rows = 3, 2
        else:
            self._grid_cols = 3
            self._grid_rows = (n_panels + 2) // 3
        self._panel_images = [None] * n_panels
        if self.enabled:
            cv2.resizeWindow(self._win,
                             min(1600, self._grid_cols * 700),
                             min(1200, self._grid_rows * 550))

    def update_panel(self, panel_idx, image, label=""):
        """Update a single panel cell in the grid and refresh."""
        if not self.enabled:
            return
        bgr = self._to_bgr_uint8(image, is_rgb=True)
        labeled = self._add_label(bgr, label)
        self._panel_images[panel_idx] = labeled
        self._grid_mode = True
        self._refresh_grid()

    def show_full(self, image, label=""):
        """Show a single full-window image (mosaic or final)."""
        if not self.enabled:
            return
        self._grid_mode = False
        bgr = self._to_bgr_uint8(image, is_rgb=True)
        labeled = self._add_label(bgr, label)
        scaled = self._scale(labeled, PREVIEW_MAX_DIM)
        cv2.imshow(self._win, scaled)
        if self.interactive:
            cv2.waitKey(0)
        else:
            cv2.waitKey(200)

    def finish(self):
        """Keep window open until user closes it (X button or any key)."""
        if not self.enabled:
            return
        print("\n  Preview window open \u2014 close it or press any key to exit.")
        try:
            while True:
                key = cv2.waitKey(500)
                if key != -1:
                    break
                try:
                    if cv2.getWindowProperty(self._win,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()

    # ── internals ──

    def _refresh_grid(self):
        """Rebuild and display the panel grid."""
        filled = [img for img in self._panel_images if img is not None]
        if not filled:
            return
        cell_w = max(img.shape[1] for img in filled)
        cell_h = max(img.shape[0] for img in filled)

        # Scale cells so grid fits on screen
        max_w, max_h = 2400, 1600
        scale = min(1.0, max_w / (cell_w * self._grid_cols),
                    max_h / (cell_h * self._grid_rows))
        cell_w = int(cell_w * scale)
        cell_h = int(cell_h * scale)

        canvas = np.zeros((cell_h * self._grid_rows,
                           cell_w * self._grid_cols, 3), dtype=np.uint8)
        for i, img in enumerate(self._panel_images):
            if img is None:
                continue
            r = i // self._grid_cols
            c = i % self._grid_cols
            resized = cv2.resize(img, (cell_w, cell_h),
                                 interpolation=cv2.INTER_AREA)
            canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

        cv2.imshow(self._win, canvas)
        if self.interactive:
            cv2.waitKey(0)
        else:
            cv2.waitKey(200)

    @staticmethod
    def _to_bgr_uint8(img, is_rgb):
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            if len(img.shape) == 3 and img.shape[2] == 3:
                return img
            elif len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        out = np.clip(img, 0, 1)
        out = (out * 255).astype(np.uint8)
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif is_rgb and len(out.shape) == 3 and out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out

    @staticmethod
    def _scale(img, max_dim):
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        s = max_dim / max(h, w)
        return cv2.resize(img, (int(w*s), int(h*s)),
                          interpolation=cv2.INTER_AREA)

    @staticmethod
    def _add_label(img, text, bar_h=32):
        h, w = img.shape[:2]
        bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 210, 210), 2, cv2.LINE_AA)
        return np.vstack([bar, img])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Display Helpers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def preview_stretch_mono(data, target_median=0.25):
    """Quick gamma stretch of mono data for preview display only."""
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data, dtype=np.float32)
    med = float(np.median(valid))
    if med <= 0 or med >= 1:
        return np.clip(data, 0, 1).astype(np.float32)
    gamma = math.log(max(target_median, 0.01)) / math.log(med)
    gamma = max(0.05, min(gamma, 20.0))
    return np.clip(np.power(np.clip(data, 0, 1), gamma), 0, 1).astype(np.float32)


def preview_stretch_rgb(r, g, b, target_median=0.25):
    """Linked gamma stretch of RGB for preview display only. Returns [H,W,3]."""
    L = LR * r + LG * g + LB * b
    valid = L[L > 0]
    if len(valid) == 0:
        return np.stack([r, g, b], axis=-1).astype(np.float32)
    med = float(np.median(valid))
    if med <= 0 or med >= 1:
        return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1),
                         np.clip(b, 0, 1)], axis=-1).astype(np.float32)
    gamma = math.log(max(target_median, 0.01)) / math.log(med)
    gamma = max(0.05, min(gamma, 20.0))
    rs = np.clip(np.power(np.clip(r, 0, 1), gamma), 0, 1)
    gs = np.clip(np.power(np.clip(g, 0, 1), gamma), 0, 1)
    bs = np.clip(np.power(np.clip(b, 0, 1), gamma), 0, 1)
    return np.stack([rs, gs, bs], axis=-1).astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Math: Stretch Functions                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def estimate_background(data):
    """Estimate background level using iterative sigma-clipped median.

    Returns (bg_level, sigma) where sigma = MAD * 1.4826.
    The MAD-to-sigma conversion uses the Normal distribution constant:
        sigma = MAD / Phi^{-1}(3/4) = MAD * 1.4826
    """
    valid = data[data > 0]
    if len(valid) == 0:
        return 0.0, 0.0
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    sigma = mad * 1.4826
    # Iterative 2.5-sigma clipping (3 rounds for convergence)
    for _ in range(3):
        mask = np.abs(valid - med) < 2.5 * sigma
        if np.sum(mask) < 10:
            break
        med = float(np.median(valid[mask]))
        mad = float(np.median(np.abs(valid[mask] - med)))
        sigma = mad * 1.4826
    return med, sigma


def find_arcsinh_beta(current_median, target_median, tol=1e-6, max_iter=200):
    """Numerically solve for beta in:  target = arcsinh(beta * m) / arcsinh(beta)

    Uses geometric bisection for fast convergence across many decades.

    Parameters
    ----------
    current_median : float  -- median of the normalized data in (0, 1)
    target_median  : float  -- desired output median (e.g. 0.25)

    Returns
    -------
    beta : float  -- the arcsinh stretch parameter
    """
    if current_median <= 0 or current_median >= 1:
        return 1.0
    if target_median <= current_median:
        return 0.01

    lo, hi = 0.01, 500000.0
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)  # geometric mean for log-scale convergence
        result = math.asinh(mid * current_median) / math.asinh(mid)
        if abs(result - target_median) < tol:
            return mid
        if result < target_median:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def arcsinh_stretch(data, beta):
    """Apply arcsinh stretch:  f(x) = arcsinh(beta * x) / arcsinh(beta).

    Properties:
      - Monotonically increasing
      - f(0) = 0,  f(1) = 1
      - For beta >> 1, compresses highlights and boosts shadows
      - Preserves relative channel ratios when applied identically
    """
    if beta < 0.01:
        return data.copy()
    denom = math.asinh(beta)
    return (np.arcsinh(beta * data) / denom).astype(np.float32)


def linked_stretch_rgb(r, g, b, target_median=0.25, shadow_clip=1.5, log=print):
    """Linked arcsinh stretch: compute beta from luminance, apply to all channels.

    This is the mathematically correct way to stretch narrowband SHO data:
    a single stretch function is applied to all three channels, preserving
    the color ratios that encode the emission-line spatial distribution.

    Algorithm:
      1. Compute synthetic luminance L = 0.2126*R + 0.7152*G + 0.0722*B
      2. Estimate background bg and noise sigma from L
      3. Black point = max(bg - 2.8*sigma, 0)  (PixInsight convention)
      4. Normalize:  x' = (x - bp) / (peak - bp)
      5. Solve for beta:  target = arcsinh(beta * median') / arcsinh(beta)
      6. Apply f(x') = arcsinh(beta * x') / arcsinh(beta)  to each channel

    Returns (R', G', B') all float32 in [0, 1].
    """
    log("  Computing luminance for stretch calibration...")
    L = LR * r + LG * g + LB * b

    bg, sigma = estimate_background(L)
    log(f"  Background: {bg:.6f}  sigma: {sigma:.6f}")

    # Black point: clip shadows. Lower shadow_clip keeps more faint
    # extended structure (looks brighter/fuller); higher buries it.
    black_point = max(bg - shadow_clip * sigma, 0.0)

    # Peak: 99.95th percentile (reject hot-pixel outliers)
    valid_L = L[L > 0]
    peak = float(np.percentile(valid_L, 99.95)) if len(valid_L) > 0 else 1.0
    scale = peak - black_point
    if scale <= 0:
        scale = 1.0

    log(f"  Black point: {black_point:.6f}  Peak: {peak:.6f}  Scale: {scale:.6f}")

    # Normalized luminance median
    L_norm = np.clip((L - black_point) / scale, 0, 1)
    valid_Ln = L_norm[L_norm > 0]
    median_norm = float(np.median(valid_Ln)) if len(valid_Ln) > 0 else 0.5
    log(f"  Normalized median: {median_norm:.6f}")

    # Solve for beta
    beta = find_arcsinh_beta(median_norm, target_median)
    log(f"  Solved beta = {beta:.2f}")

    # Apply linked stretch to each channel
    def stretch_ch(ch):
        x = np.clip((ch - black_point) / scale, 0, None)
        return np.clip(arcsinh_stretch(x, beta), 0, 1)

    r_s = stretch_ch(r)
    g_s = stretch_ch(g)
    b_s = stretch_ch(b)

    # Verify
    L_out = LR * r_s + LG * g_s + LB * b_s
    valid_out = L_out[L_out > 0.001]
    out_median = float(np.median(valid_out)) if len(valid_out) > 0 else 0
    log(f"  Stretched median: {out_median:.4f}  (target: {target_median:.4f})")

    return r_s, g_s, b_s


def local_adaptive_stretch(r, g, b, target_median=0.18, strength=0.6, grid=14,
                           clip_lo=0.10, clip_hi=2.5, log=print):
    """Linked arcsinh stretch with a SMOOTH locally-varying scale.

    A single global stretch sets its white point from the brightest region
    (the nebula core), so faint outer structure is under-stretched and washes
    out across a large mosaic. This recovers the per-region local contrast that
    the old per-cluster pipeline got for free — by dividing, before the stretch,
    by a smoothly-interpolated *local* brightness field instead of one global
    scale. `strength` blends local↔global (0 = global, 1 = fully local); the
    field is heavily smoothed so there are no tile seams. Colour ratios are
    preserved (the same scale + beta apply to all channels).
    """
    if strength <= 0:
        return linked_stretch_rgb(r, g, b, target_median=target_median, log=log)

    L = (LR * r + LG * g + LB * b).astype(np.float32)
    bg, sigma = estimate_background(L)
    bp = max(bg - 2.8 * sigma, 0.0)
    valid = L[L > 0]
    gpeak = float(np.percentile(valid, 99.95)) if valid.size else 1.0

    # Per-cell local white level (90th pct of above-floor pixels), bicubic to
    # full res, then heavily blurred → a seamless local-brightness field.
    H, W = L.shape
    chy, cwx = max(1, H // grid), max(1, W // grid)
    cells = np.full((grid, grid), gpeak, np.float32)
    for gy in range(grid):
        for gx in range(grid):
            cell = L[gy * chy:(gy + 1) * chy, gx * cwx:(gx + 1) * cwx].ravel()
            v = cell[cell > bp + sigma]
            if v.size > 50:
                cells[gy, gx] = float(np.percentile(v, 90))
    S = cv2.resize(cells, (W, H), interpolation=cv2.INTER_CUBIC)
    S = cv2.GaussianBlur(S, (0, 0), sigmaX=max(chy, cwx) * 0.6)
    S = np.clip(S, gpeak * clip_lo, gpeak * clip_hi)

    # Geometric blend of global and local scale, then linked arcsinh.
    divisor = (gpeak ** (1.0 - strength)) * (S ** strength)
    scale = np.maximum(divisor - bp, 1e-6)
    Ln = np.clip((L - bp) / scale, 0, 1)
    med = float(np.median(Ln[Ln > 0])) if np.any(Ln > 0) else 0.5
    beta = find_arcsinh_beta(med, target_median)
    log(f"  Local-adaptive stretch: strength={strength:.2f} grid={grid} "
        f"beta={beta:.2f}")

    def st(c):
        return np.clip(arcsinh_stretch(np.clip((c - bp) / scale, 0, None), beta),
                       0, 1).astype(np.float32)

    return st(r), st(g), st(b)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Math: Color Processing                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def scnr_green(r, g, b, amount=0.85, method='maximum_neutral', log=print):
    """Subtractive Chromatic Noise Reduction with luminosity preservation.

    The Ha -> Green mapping in SHO creates excessive green. SCNR removes the
    green excess while preserving the luminosity of every pixel.

    Methods
    -------
    'maximum_neutral'  -- G' = min(G, max(R, B))    [standard for SHO]
    'average_neutral'  -- G' = min(G, (R+B)/2)      [gentler variant]

    Luminosity preservation
    -----------------------
      1.  L_before = 0.2126*R + 0.7152*G + 0.0722*B
      2.  Apply SCNR -> G_new
      3.  L_after  = 0.2126*R + 0.7152*G_new + 0.0722*B
      4.  Scale all channels by (L_before / L_after)

    This redistributes the lost green luminosity proportionally across
    all channels, maintaining total brightness without green cast.
    """
    log(f"  SCNR method: {method}, amount: {amount:.2f}")

    # 1. Record luminosity
    L_before = LR * r + LG * g + LB * b

    # 2. Compute green limit
    if method == 'maximum_neutral':
        neutral_limit = np.maximum(r, b)
    else:
        neutral_limit = (r + b) / 2.0

    # 3. Reduce green
    g_excess = np.maximum(g - neutral_limit, 0)
    g_new = g - g_excess * amount

    # Statistics
    n_affected = int(np.sum(g_excess > 0))
    total = g.size
    log(f"  Pixels with green excess: {n_affected} / {total} "
        f"({100*n_affected/total:.1f}%)")
    if n_affected > 0:
        log(f"  Mean excess removed: {float(np.mean(g[g_excess>0] - g_new[g_excess>0])):.5f}")

    # 4. Luminosity preservation
    L_after = LR * r + LG * g_new + LB * b
    safe_L = np.maximum(L_after, 1e-10)
    lum_scale = np.where(L_after > 1e-10, L_before / safe_L, 1.0)

    r_out = np.clip(r * lum_scale, 0, 1).astype(np.float32)
    g_out = np.clip(g_new * lum_scale, 0, 1).astype(np.float32)
    b_out = np.clip(b * lum_scale, 0, 1).astype(np.float32)

    # Verify preservation
    L_final = LR * r_out + LG * g_out + LB * b_out
    lum_err = float(np.mean(np.abs(L_before - L_final)))
    log(f"  Luminosity preservation error: {lum_err:.8f}")

    return r_out, g_out, b_out


def flatten_background_channel(ch, grid=24, sample_pct=5, hard_pedestal=True,
                               log=print):
    """Remove a spatial background gradient from a single channel.

    Pass 1 — bicubic gradient surface:
      Divides the image into a grid, takes the low-percentile sky level in
      each cell (5th pct by default — much more aggressive than 20th),
      bicubic-interpolates a full-resolution gradient surface, subtracts it.

    Pass 2 — hard pedestal clip (when hard_pedestal=True):
      After the spatial gradient is gone, any remaining uniform glow is
      driven to zero by subtracting the global 2nd-percentile of what's left.
      This forces the true sky floor to black.

    Args:
        ch            : 2-D float32 array [0, 1]
        grid          : cells per axis (default 24 — finer than before)
        sample_pct    : percentile used as sky estimator per cell (default 5)
        hard_pedestal : if True, apply a second global floor subtraction
    """
    H, W = ch.shape
    cell_h = H // grid
    cell_w = W // grid
    if cell_h < 4 or cell_w < 4:
        return ch  # image too small for this grid size

    # ── Pass 1: spatial gradient ──────────────────────────────────────
    samples = np.zeros((grid, grid), dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w
            cell = ch[y0:y1, x0:x1].ravel()
            valid = cell[cell > 1e-6]
            if len(valid) < 10:
                samples[gy, gx] = 0.0
            else:
                samples[gy, gx] = float(np.percentile(valid, sample_pct))

    gradient = cv2.resize(samples, (W, H), interpolation=cv2.INTER_CUBIC)
    gradient = np.clip(gradient, 0, None).astype(np.float32)
    out = np.clip(ch - gradient, 0, 1).astype(np.float32)
    removed_grad = float(np.mean(gradient))

    # ── Pass 2: hard global pedestal ─────────────────────────────────
    removed_ped = 0.0
    if hard_pedestal:
        flat = out.ravel()
        valid_flat = flat[flat > 1e-6]
        if len(valid_flat) > 100:
            pedestal = float(np.percentile(valid_flat, 2))
            out = np.clip(out - pedestal, 0, 1).astype(np.float32)
            removed_ped = pedestal

    log(f"    channel: gradient={removed_grad:.5f}  pedestal={removed_ped:.5f}")
    return out


def flatten_background_rgb(r, g, b, grid=24, sample_pct=5, log=print):
    """Apply aggressive gradient + pedestal background flattening to all channels.

    Each channel is processed independently so colour casts from
    clouds/vignetting are removed along with the spatial gradient.
    Intended for the stars-only layer where the background should be black.
    """
    log(f"  Background flattening (grid={grid}, sample_pct={sample_pct}, "
        f"hard_pedestal=True)...")
    r_f = flatten_background_channel(r, grid=grid, sample_pct=sample_pct,
                                     hard_pedestal=True, log=log)
    g_f = flatten_background_channel(g, grid=grid, sample_pct=sample_pct,
                                     hard_pedestal=True, log=log)
    b_f = flatten_background_channel(b, grid=grid, sample_pct=sample_pct,
                                     hard_pedestal=True, log=log)
    bg_r = float(np.mean(r - r_f))
    bg_g = float(np.mean(g - g_f))
    bg_b = float(np.mean(b - b_f))
    log(f"  Total removed — R={bg_r:.5f}  G={bg_g:.5f}  B={bg_b:.5f}")
    return r_f, g_f, b_f


def neutralize_background_rgb(r, g, b, sample_fraction=BG_SAMPLE_FRAC, log=print):
    """Neutralize background color cast.

    Measures per-channel median in the darkest pixels (true sky background)
    and subtracts to make the background color-neutral.
    """
    log("  Neutralizing background...")
    L = LR * r + LG * g + LB * b

    # Sample the darkest fraction of REAL pixels only. On a mosaic the image is
    # padded with zeros outside the all-channel footprint; including those would
    # make the "background" sample ≈0 so neutralisation does nothing and a true
    # sky colour-cast (e.g. an OIII blue halo) survives. Restrict to L>0.
    real = L > 0.0
    Lr = L[real]
    if Lr.size < 100:
        log("  Warning: too few background pixels, skipping")
        return r, g, b
    threshold = float(np.percentile(Lr, sample_fraction * 100))
    bg_mask = real & (L <= threshold)
    n_bg = int(np.sum(bg_mask))
    if n_bg < 100:
        log("  Warning: too few background pixels, skipping")
        return r, g, b

    bg_r = float(np.median(r[bg_mask]))
    bg_g = float(np.median(g[bg_mask]))
    bg_b = float(np.median(b[bg_mask]))
    log(f"  Background: R={bg_r:.5f} G={bg_g:.5f} B={bg_b:.5f} "
        f"({n_bg} pixels sampled)")

    r_n = np.clip(r - bg_r, 0, 1).astype(np.float32)
    g_n = np.clip(g - bg_g, 0, 1).astype(np.float32)
    b_n = np.clip(b - bg_b, 0, 1).astype(np.float32)

    return r_n, g_n, b_n


def background_gradient_neutralize(r, g, b, grid=32, bg_pct=20, neb_pct=60,
                                   log=print):
    """Remove a smoothly-varying background colour cast from a finished mosaic.

    When per-cluster panels are mosaicked, each panel carries a slightly
    different residual sky floor (and seam-equalisation can lift it), so the
    assembled background develops a soft colour gradient — typically a reddish
    (SII) cast over part of the frame. A single uniform subtraction
    (neutralize_background_rgb) can't fix a *spatially-varying* cast, and on a
    nebula-filled mosaic its "darkest pixels" sample is contaminated by faint
    signal, so it over-subtracts and turns the image green.

    This estimates a per-channel background *surface* from background pixels
    only (grid of low-luminance medians, NaN-filled where nebula covers the
    cell, bicubic + low-pass smoothed), then subtracts it weighted by a
    luminance taper so the bright nebula keeps its colour. Subtracting the
    smooth surface from faint nebula is correct — it removes the pedestal the
    nebula sits on — so the gold/teal stay intact while the sky goes neutral.
    """
    log("  Background gradient neutralize (nebula-safe)...")
    L = (LR * r + LG * g + LB * b).astype(np.float32)
    real = L > 0.0
    if int(np.sum(real)) < 1000:
        log("  Warning: too few real pixels, skipping")
        return r, g, b
    Lr = L[real]
    lo = float(np.percentile(Lr, bg_pct))
    hi = float(np.percentile(Lr, neb_pct))
    bg_mask = real & (L <= lo)
    h, w = L.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)

    def surface(ch):
        cell = np.full((grid, grid), np.nan, dtype=np.float32)
        for iy in range(grid):
            for ix in range(grid):
                m = bg_mask[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
                if int(m.sum()) > 30:
                    cell[iy, ix] = np.median(
                        ch[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]][m])
        # Nebula-covered cells (no background) → global background median.
        gmed = float(np.nanmedian(cell)) if np.isfinite(cell).any() else 0.0
        cell[~np.isfinite(cell)] = gmed
        up = cv2.resize(cell, (w, h), interpolation=cv2.INTER_CUBIC)
        return _smooth_lowpass(up, sigma=max(h, w) / 12.0)

    sr, sg, sb = _parallel_map(surface, [r, g, b])
    log(f"  Background surface medians: R={float(np.median(sr[bg_mask])):.4f} "
        f"G={float(np.median(sg[bg_mask])):.4f} "
        f"B={float(np.median(sb[bg_mask])):.4f}")
    # Taper: full subtraction in background (L≤lo), none in nebula (L≥hi).
    wt = (np.clip((hi - L) / (hi - lo + 1e-6), 0, 1) * real).astype(np.float32)
    return (np.clip(r - sr * wt, 0, 1).astype(np.float32),
            np.clip(g - sg * wt, 0, 1).astype(np.float32),
            np.clip(b - sb * wt, 0, 1).astype(np.float32))


def boost_saturation(r, g, b, factor=1.25, signal_floor=0.03, log=print):
    """Boost color saturation in nebula regions while preserving luminosity.

    Only enhances pixels above signal_floor to avoid amplifying noise.

    Math:
      ch_saturated = L + (ch - L) * factor
      Then rescale so L is preserved.
    """
    log(f"  Saturation boost: {factor:.2f}x (floor={signal_floor:.3f})")
    L = LR * r + LG * g + LB * b
    mask = L > signal_floor

    r_b = np.where(mask, L + (r - L) * factor, r)
    g_b = np.where(mask, L + (g - L) * factor, g)
    b_b = np.where(mask, L + (b - L) * factor, b)

    # Preserve luminosity
    L_new = LR * r_b + LG * g_b + LB * b_b
    safe = np.maximum(L_new, 1e-10)
    scale = np.where(L_new > 1e-10, L / safe, 1.0)

    return (np.clip(r_b * scale, 0, 1).astype(np.float32),
            np.clip(g_b * scale, 0, 1).astype(np.float32),
            np.clip(b_b * scale, 0, 1).astype(np.float32))


def hubble_color_refine(r, g, b, strength=0.3, oiii_factor=0.38, log=print):
    """Hubble palette refinement using hue curve transformation.

    Uses HSV hue rotation to shift yellow-green hues toward orange,
    while pushing OIII regions toward deeper blue (away from cyan).
    This is more natural than RGB channel manipulation because it
    targets specific hue ranges.

    In HSV (0-360):
      Yellow = 60°, Green = 120°, Orange = 30°, Blue = 210-240°
      We shift hues in the 40°-130° range toward orange (24°-39°),
      and OIII hues (170-220°) toward deeper blue (~215°).
    """
    log(f"  Hubble palette refinement via hue curve (strength={strength:.2f})")
    import cv2

    # Build RGB image for HSV conversion
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # OpenCV HSV: H=[0,360), S=[0,1], V=[0,1] for float32

    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Only modify pixels with actual signal and saturation
    L = LR * r + LG * g + LB * b
    signal = (L > 0.05) & (s > 0.05)

    # Hue curve: shift yellow-green range (40-130°) toward orange (25-40°)
    # The shift is strongest at pure green (120°) and tapers off
    #   40° (yellow-orange) -> small shift toward 30°
    #   60° (yellow)        -> shift toward 35°
    #   90° (yellow-green)  -> shift toward 35°
    #  120° (green)         -> shift toward 40°
    #  130°+               -> no shift (preserve teal/cyan)
    hue_min, hue_max = 40.0, 130.0

    in_range = signal & (h >= hue_min) & (h <= hue_max)

    # Normalized position in the source range [0, 1]
    t = np.where(in_range, (h - hue_min) / (hue_max - hue_min), 0)

    # Target hue: map entire range toward orange, with slight spread
    # hue_min(40°) -> 24°, hue_max(130°) -> 39°
    target = 24.0 + t * 15.0  # 24° to 39° (more orange)

    # Blend between original hue and target based on strength
    h_new = np.where(in_range,
                     h * (1.0 - strength) + target * strength, h)

    n_shifted = int(np.sum(in_range))
    log(f"  Hue-shifted pixels: {n_shifted} (yellow-green -> orange)")

    # OIII blue enhancement: shift cyan (160-215°) toward deep blue (235°)
    # Cyan sits at 180°; 235° is a rich indigo-blue.
    # Factor is independent of the yellow-green strength so it can be
    # tuned aggressively without over-rotating the gold tones.
    oiii_range = signal & (h >= 160) & (h <= 215)
    h_new = np.where(oiii_range,
                     h * (1.0 - oiii_factor) + 235.0 * oiii_factor,
                     h_new)
    n_oiii = int(np.sum(oiii_range))
    log(f"  OIII blue pixels: {n_oiii}  (cyan→deep-blue factor={oiii_factor})")

    hsv_out = np.stack([h_new.astype(np.float32),
                        s.astype(np.float32),
                        v.astype(np.float32)], axis=-1)
    rgb_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)

    return (np.clip(rgb_out[:,:,0], 0, 1).astype(np.float32),
            np.clip(rgb_out[:,:,1], 0, 1).astype(np.float32),
            np.clip(rgb_out[:,:,2], 0, 1).astype(np.float32))


def hubbleize_with_skimage(r, g, b, strength=0.55, log=print):
    """Advanced artistic Hubble recolor using scikit-image.

    Goals:
      - Push cyan structures toward deeper blue
      - Warm yellow/green transition zones toward orange-gold
      - Tame excessive green dominance from strong Ha signal

    Returns updated (r, g, b) float32 channels.
    """
    try:
        from skimage import color, filters
    except Exception:
        log("  [HUBBLEIZE] scikit-image not available; skipping advanced recolor")
        return r, g, b

    log(f"  [HUBBLEIZE] Advanced recolor (scikit-image), strength={strength:.2f}")

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb = np.clip(rgb, 0, 1)

    # ── Pass 1: cyan -> blue in HSV space ─────────────────────────────
    hsv = color.rgb2hsv(rgb)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    signal = (v > 0.04) & (s > 0.04)

    cyan = signal & (h >= (155.0 / 360.0)) & (h <= (215.0 / 360.0))
    cyan_w = filters.gaussian(cyan.astype(np.float32), sigma=2.0,
                              preserve_range=True)
    cyan_strength = np.clip(cyan_w * (0.50 * strength), 0, 1)
    target_blue = 235.0 / 360.0
    h2 = h * (1.0 - cyan_strength) + target_blue * cyan_strength

    hsv2 = np.stack([h2.astype(np.float32), s.astype(np.float32),
                     v.astype(np.float32)], axis=-1)
    rgb2 = np.clip(color.hsv2rgb(hsv2), 0, 1).astype(np.float32)

    # ── Pass 2: warm edge hues toward orange/gold in LAB ──────────────
    hsv_mid = color.rgb2hsv(rgb2)
    hm, sm, vm = hsv_mid[:, :, 0], hsv_mid[:, :, 1], hsv_mid[:, :, 2]
    warm_zone = ((hm >= (40.0 / 360.0)) & (hm <= (130.0 / 360.0))
                 & (sm > 0.05) & (vm > 0.05))
    warm_w = filters.gaussian(warm_zone.astype(np.float32), sigma=2.5,
                              preserve_range=True)

    lab = color.rgb2lab(rgb2)
    # LAB a*: +red / -green, b*: +yellow / -blue
    lab[:, :, 1] = lab[:, :, 1] + (18.0 * strength) * warm_w
    lab[:, :, 2] = lab[:, :, 2] + (14.0 * strength) * warm_w
    rgb3 = np.clip(color.lab2rgb(lab), 0, 1).astype(np.float32)

    # ── Pass 3: mild green taming where G strongly dominates ───────────
    L_before = LR * rgb3[:, :, 0] + LG * rgb3[:, :, 1] + LB * rgb3[:, :, 2]
    g_dom = (rgb3[:, :, 1] > 1.08 * np.maximum(rgb3[:, :, 0], rgb3[:, :, 2]))
    g_w = filters.gaussian(g_dom.astype(np.float32), sigma=2.0,
                           preserve_range=True)
    g_scale = np.clip(1.0 - (0.14 * strength) * g_w, 0.75, 1.0)
    rgb3[:, :, 1] = np.clip(rgb3[:, :, 1] * g_scale, 0, 1)

    # Preserve luminance after color manipulations
    L_after = LR * rgb3[:, :, 0] + LG * rgb3[:, :, 1] + LB * rgb3[:, :, 2]
    safe = np.maximum(L_after, 1e-10)
    scale = np.where(L_after > 1e-10, L_before / safe, 1.0)
    rgb3 = np.clip(rgb3 * scale[:, :, None], 0, 1).astype(np.float32)

    n_cyan = int(np.count_nonzero(cyan))
    n_warm = int(np.count_nonzero(warm_zone))
    self_msg = (f"  [HUBBLEIZE] cyan->blue px={n_cyan}, "
                f"gold-warm px={n_warm}")
    log(self_msg)

    return (rgb3[:, :, 0].astype(np.float32),
            rgb3[:, :, 1].astype(np.float32),
            rgb3[:, :, 2].astype(np.float32))


def process_stars(r, g, b, desat_amount=0.70, log=print):
    """Process stars for natural appearance in narrowband.

    In SHO mapping, stars get unnatural colors (green from Ha, blue from OIII).
    This function:
      1. Desaturates toward neutral white
      2. Adds very slight warmth (natural star color)
      3. Anti-purple guard: ensures G >= 0.75 * min(R, B) in bright pixels
         Purple/magenta occurs when G is deficient relative to R and B.
    """
    log(f"  Star desaturation: {desat_amount:.2f}")

    L = LR * r + LG * g + LB * b

    # If a star is strongly blue-dominant (common when OIII is much
    # stronger than SII/Ha in the star layer), increase desaturation
    # locally so stellar color stays neutral.
    desat_map = np.full_like(L, float(desat_amount), dtype=np.float32)
    blue_dominant = (L > 0.10) & (b > 1.25 * np.maximum(r, g))
    desat_map = np.where(blue_dominant,
                         np.minimum(1.0, desat_map + 0.25),
                         desat_map)
    n_blue_dom = int(np.count_nonzero(blue_dominant))
    if n_blue_dom > 0:
        log(f"  Blue-dominant star pixels: {n_blue_dom} "
            f"(extra desat applied)")

    # Desaturate toward luminance
    r_d = r * (1 - desat_map) + L * desat_map
    g_d = g * (1 - desat_map) + L * desat_map
    b_d = b * (1 - desat_map) + L * desat_map

    # Slight warmth for natural stellar color
    warmth = 0.015
    r_d = r_d + warmth * L
    b_d = b_d - warmth * 0.5 * L

    # Anti-purple: in bright pixels, prevent G from falling below threshold
    bright = L > 0.15
    rb_min = np.minimum(r_d, b_d)
    g_floor = rb_min * 0.75
    g_d = np.where(bright, np.maximum(g_d, g_floor), g_d)

    # Anti-blue halo guard: for bright stars, cap B relative to R/G.
    # This prevents oversized blue stars when OIII dominates.
    rg_max = np.maximum(r_d, g_d)
    b_cap = rg_max * 1.10
    b_capped = bright & (b_d > b_cap)
    n_b_capped = int(np.count_nonzero(b_capped))
    if n_b_capped > 0:
        log(f"  Blue cap applied to {n_b_capped} bright star pixels")
    b_d = np.where(bright, np.minimum(b_d, b_cap), b_d)

    # Report star color balance
    star_mask = L > 0.3
    if np.any(star_mask):
        mr = float(np.mean(r_d[star_mask]))
        mg = float(np.mean(g_d[star_mask]))
        mb = float(np.mean(b_d[star_mask]))
        log(f"  Bright star mean color: R={mr:.3f} G={mg:.3f} B={mb:.3f}")

    return (np.clip(r_d, 0, 1).astype(np.float32),
            np.clip(g_d, 0, 1).astype(np.float32),
            np.clip(b_d, 0, 1).astype(np.float32))


def tame_blue_stars_pre_starnet(r, g, b, strength=0.85, log=print):
    """Tame blue-dominant stellar cores before StarNet separation.

    This operates on stretched RGB and targets only star-like compact
    bright structures, leaving diffuse nebula colour mostly untouched.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return r, g, b

    L = LR * r + LG * g + LB * b

    # Star-likeness from high-frequency luminance detail.
    L_blur = cv2.GaussianBlur(L.astype(np.float32), (0, 0), sigmaX=2.2, sigmaY=2.2)
    detail = np.clip(L - L_blur, 0, None)
    if np.count_nonzero(detail > 0) < 64:
        return r, g, b

    thr = float(np.percentile(detail[detail > 0], 92.0))
    star_like = (detail > thr) & (L > 0.08)

    # Focus only where blue strongly dominates R/G.
    blue_dom = star_like & (b > 1.20 * np.maximum(r, g))
    n_blue = int(np.count_nonzero(blue_dom))
    if n_blue == 0:
        return r, g, b

    mask = cv2.GaussianBlur(blue_dom.astype(np.float32), (0, 0), sigmaX=1.4, sigmaY=1.4)
    mask = np.clip(mask * strength, 0, 1)

    rg_max = np.maximum(r, g)
    b_target = np.minimum(b, rg_max * 1.04)
    b_out = b * (1.0 - mask) + b_target * mask

    # Mild green floor in strongest stellar cores to avoid magenta bias.
    bright = L > 0.18
    g_floor = np.minimum(r, b_out) * 0.72
    g_out = np.where(bright, np.maximum(g, g_floor), g)

    log(f"  Pre-StarNet blue-star tame: adjusted {n_blue} px (strength={strength:.2f})")
    return (np.clip(r, 0, 1).astype(np.float32),
            np.clip(g_out, 0, 1).astype(np.float32),
            np.clip(b_out, 0, 1).astype(np.float32))


def apply_star_channel_consensus(r, g, b, mode='soft', log=print):
    """Suppress stars not supported across channels.

    Modes:
      - off:    no filtering
      - soft:   strongly suppress single-channel stars, keep 2/3+ stars
            - strict: keep only stars present in at least 2 channels
    """
    mode = (mode or 'off').lower().strip()
    if mode == 'off':
        return r, g, b

    channels = [r, g, b]
    present = []
    for ch in channels:
        vals = ch[ch > 0]
        if vals.size < 100:
            thresh = 0.02
        else:
            thresh = max(0.02, float(np.percentile(vals, 88.0)) * 0.22)
        present.append(ch > thresh)

    present_count = (present[0].astype(np.uint8)
                     + present[1].astype(np.uint8)
                     + present[2].astype(np.uint8))

    if mode == 'strict':
        # Preserve physically plausible multi-channel stars (e.g. OIII+Ha)
        # while rejecting single-channel artifacts.
        weight = (present_count >= 2).astype(np.float32)
    else:
        # soft: remove most single-channel stars, keep 2/3 channel stars
        weight = np.where(present_count >= 3, 1.0,
                 np.where(present_count == 2, 0.85,
                 np.where(present_count == 1, 0.12, 0.0))).astype(np.float32)

    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=1.3, sigmaY=1.3)
    weight = np.clip(weight, 0, 1)

    kept = int(np.count_nonzero(weight > 0.5))
    reduced = int(np.count_nonzero((present_count == 1) & (weight < 0.3)))
    n_2plus = int(np.count_nonzero(present_count >= 2))
    log(f"  Star consensus ({mode}): kept={kept} px, "
        f"multi-channel={n_2plus} px, reduced single-channel={reduced} px")

    return (np.clip(r * weight, 0, 1).astype(np.float32),
            np.clip(g * weight, 0, 1).astype(np.float32),
            np.clip(b * weight, 0, 1).astype(np.float32))


def suppress_blue_star_halos(final_rgb, stars_rgb, amount=0.90, log=print):
    """Suppress blue-dominant star halos in the final artistic blend.

    Uses the star layer as a spatial prior so only star regions are affected.
    This avoids tinting nebula structures while taming oversized blue stars.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return final_rgb

    r = final_rgb[:, :, 0]
    g = final_rgb[:, :, 1]
    b = final_rgb[:, :, 2]

    star_signal = np.max(stars_rgb, axis=2)
    star_mask = (star_signal > 0.015).astype(np.float32)
    if np.count_nonzero(star_mask) == 0:
        return final_rgb

    # Soft mask so transitions are smooth and don't create hard rims.
    soft = cv2.GaussianBlur(star_mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
    soft = np.clip(soft * amount, 0, 1)

    rg_max = np.maximum(r, g)
    b_cap = rg_max * 1.05
    over = b > b_cap
    n_over = int(np.count_nonzero(over & (soft > 0.05)))
    if n_over > 0:
        log(f"  Blue-star halo guard adjusted {n_over} pixels")

    b_tamed = np.minimum(b, b_cap)
    b_out = b * (1.0 - soft) + b_tamed * soft

    out = np.stack([r, g, np.clip(b_out, 0, 1)], axis=-1).astype(np.float32)
    return out


def screen_blend(base, overlay):
    """Screen blend mode:  result = 1 - (1 - base) * (1 - overlay).

    Properties:
      - Commutative: blend(A, B) = blend(B, A)
      - Neutral element: overlay=0 returns base unchanged
      - Never darkens: result >= max(base, overlay)
      - Naturally handles star re-integration without clipping
    """
    return (1.0 - (1.0 - base) * (1.0 - overlay)).astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Mosaic (Star-Aligned Panel Stitching)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _feather_weight(shape, border=100):
    """Create a distance-based weight map with cosine taper to 0 at edges.

    Uses a cosine ease-in/out curve for smooth blending with zero
    derivative at both the edge (weight=0) and interior (weight=1).
    This eliminates the harsh transition visible with linear ramps
    when nebula structure crosses the panel boundary.
    """
    h, w = shape
    # Distance from each edge
    y_dist = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    x_dist = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)

    # 2D minimum distance from any edge, normalized to [0, 1]
    t = np.clip(np.minimum(y_dist[:, None], x_dist[None, :]) / max(border, 1),
                0, 1)

    # Cosine taper: smooth S-curve with zero derivative at both ends
    weight = (0.5 * (1 - np.cos(np.pi * t))).astype(np.float32)
    return weight


def compute_mosaic_geometry(panel_images, log=print):
    """Compute star-asterism affine transforms and canvas layout for a mosaic.

    Matches stars between panels with the shared asterism matcher
    (oeuvre.star_match) and computes partial-affine transforms (translate +
    rotate + uniform scale = 4 DOF), the correct model for astronomical panels
    from the same telescope session (no perspective distortion in the sky).

    Args:
        panel_images: list of 2D numpy arrays (float32), one per panel.
                      These should be well-stretched images with good
                      star visibility (e.g. quick SHO luminance).
        log: logging function

    Returns:
        dict with keys:
          - transforms: list of 2x3 affine matrices (one per panel)
          - canvas_w, canvas_h: output canvas dimensions
          - offset: 2x3 offset matrix to shift into positive coordinates
          - feather_border: feather blend width in pixels
    """

    n = len(panel_images)
    ref = panel_images[0]
    h_ref, w_ref = ref.shape[:2]

    identity_2x3 = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    if n == 1:
        return {
            'transforms': [identity_2x3.copy()],
            'canvas_w': w_ref,
            'canvas_h': h_ref,
            'offset': identity_2x3.copy(),
            'feather_border': max(50, int(np.sqrt(h_ref**2 + w_ref**2)) // 5),
        }

    # Register panels with the shared asterism matcher (oeuvre.star_match) — the
    # same star-geometry registration used for sub stacking and SHO channel
    # alignment. Two robustness properties matter for "use all the data":
    #   1. Detection is nebula-robust (high-pass prefilter in detect_stars), so
    #      nebula-heavy panels still expose enough real stars to match.
    #   2. Each panel is matched against *any already-placed panel*, not just a
    #      single fixed reference. A strip/grid mosaic chains together (panel A
    #      may only overlap panel B, which overlaps the anchor) instead of
    #      dropping panels that simply don't touch panel 0.
    from .star_match import build_star_model

    # Detect plenty of stars per panel: with only weak (~40%) overlap, each
    # panel's brightest few-hundred stars sit mostly OUTSIDE the shared strip,
    # so a low cap leaves too few common stars to match. A high cap keeps enough
    # stars in the overlap region for weak-overlap panels to register.
    MOSAIC_MAX_STARS = 800

    # Detect stars + build asterism invariants ONCE per panel. The incremental
    # placement below does up to O(n²) pairwise matches; rebuilding star models
    # (which run the high-pass blur + detection) inside that loop would be
    # prohibitively slow on full-res panels, so we precompute and reuse them.
    models = _parallel_map(
        lambda p: build_star_model(p, max_stars=MOSAIC_MAX_STARS), panel_images)
    star_counts = [len(m[0]) for m in models]

    # Anchor = the panel exposing the most stars (most reliable to match against).
    anchor = int(np.argmax(star_counts))
    log(f"  Anchor panel: {anchor+1} "
        f"({panel_images[anchor].shape[1]}x{panel_images[anchor].shape[0]}, "
        f"{star_counts[anchor]} stars); per-panel stars={star_counts}")

    transforms = [None] * n            # panel coords -> anchor coords (2x3) or None
    transforms[anchor] = identity_2x3.copy()
    placed = {anchor}

    # Iterate until no further panel can be attached (handles A->B->anchor chains).
    progress = True
    while progress and len(placed) < n:
        progress = False
        for i in range(n):
            if i in placed:
                continue
            best = None  # (inliers, M, j)
            for j in list(placed):
                # Bidirectional: the matcher is asymmetric under weak overlap,
                # so a pair can solve (placed → unplaced) even when the
                # (unplaced → placed) query fails. Try both, invert as needed.
                M, inliers = _solve_models_either(models[i], models[j], log=log)
                if M is not None and (best is None or inliers > best[0]):
                    best = (inliers, M, j)
            if best is None:
                continue
            inliers, M, j = best
            transforms[i] = _compose_affine(transforms[j], M)
            placed.add(i)
            progress = True
            sx = np.hypot(M[0, 0], M[1, 0])
            angle_deg = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            log(f"  Panel {i+1}: matched to panel {j+1}, {inliers} inliers  "
                f"dx={M[0, 2]:.1f}px dy={M[1, 2]:.1f}px "
                f"rot={angle_deg:.3f}° scale={sx:.4f}")

    # ── Filter out panels that couldn't be attached to anything ───────────
    kept_indices = [i for i, M in enumerate(transforms) if M is not None]
    if len(kept_indices) < n:
        dropped = [i + 1 for i, M in enumerate(transforms) if M is None]
        log(f"  WARNING: panel(s) {dropped} could not be matched to any other "
            f"panel (insufficient star overlap). Dropping from mosaic.")
    transforms = [transforms[i] for i in kept_indices]
    panel_images = [panel_images[i] for i in kept_indices]

    if len(panel_images) == 0:
        raise ValueError("No panels could be matched — cannot mosaic")
    if len(panel_images) == 1:
        log(f"  Only 1 panel remaining — returning as-is")
        return {
            'transforms': [identity_2x3.copy()],
            'canvas_w': panel_images[0].shape[1],
            'canvas_h': panel_images[0].shape[0],
            'offset': identity_2x3.copy(),
            'feather_border': max(50, int(np.sqrt(
                panel_images[0].shape[0]**2 + panel_images[0].shape[1]**2)) // 5),
            'kept_indices': kept_indices,
        }

    # ── Compute canvas bounds ───────────────────────────────────────────
    # Transform corners of each panel through its 2×3 affine
    all_corners = []
    for i, img in enumerate(panel_images):
        h_i, w_i = img.shape[:2]
        corners = np.float64([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]])
        M = transforms[i]
        # Apply 2×3 affine: dst = M @ [x, y, 1]^T
        ones = np.ones((4, 1), dtype=np.float64)
        pts_h = np.hstack([corners, ones])  # [4, 3]
        warped = (M @ pts_h.T).T  # [4, 2]
        all_corners.append(warped)

    all_corners = np.concatenate(all_corners, axis=0)  # [N, 2]
    x_min = int(np.floor(all_corners[:, 0].min()))
    y_min = int(np.floor(all_corners[:, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0].max()))
    y_max = int(np.ceil(all_corners[:, 1].max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    log(f"  Canvas size: {canvas_w} x {canvas_h}")

    # Offset matrix (2×3): just adds translation
    offset = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min]
    ], dtype=np.float64)

    diag = int(np.sqrt(h_ref**2 + w_ref**2))
    feather_border = max(50, diag // 5)
    log(f"  Feather border: {feather_border} pixels (cosine taper)")

    return {
        'transforms': transforms,
        'canvas_w': canvas_w,
        'canvas_h': canvas_h,
        'offset': offset,
        'feather_border': feather_border,
        'kept_indices': kept_indices,
    }


def _compose_affine(A, B):
    """Compose two 2×3 affine matrices: result = A ∘ B.

    Equivalent to multiplying the 3×3 augmented forms and extracting
    the top 2 rows.
    """
    # Augment to 3×3
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    C3 = A3 @ B3
    return C3[:2, :].copy()


def _invert_affine(M):
    """Invert a 2×3 affine matrix. M maps a→b; result maps b→a."""
    M3 = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(M3)[:2, :].copy()


def _solve_models_either(model_i, model_j, log=print):
    """Solve the i→j affine, robust to the matcher's directional asymmetry.

    star_match.solve_from_models is not symmetric: with weak overlap it
    frequently solves one direction (e.g. P1→P4) but returns None for the
    reverse (P4→P1), because the triangle-hash query uses the source panel's
    asterisms. The mosaic placement loop always queries (unplaced → placed),
    so a perfectly good pair can be dropped purely because that *direction*
    failed. Try the reverse and invert when the forward solve fails, keeping
    whichever yields more inliers.

    Returns (M_i_to_j, inliers) or (None, 0).
    """
    from .star_match import solve_from_models
    M_fwd, inl_fwd = solve_from_models(model_i, model_j, log=log)
    M_rev, inl_rev = solve_from_models(model_j, model_i, log=log)
    cand = []
    if M_fwd is not None:
        cand.append((inl_fwd, M_fwd))
    if M_rev is not None:
        cand.append((inl_rev, _invert_affine(M_rev)))
    if not cand:
        return None, 0
    inliers, M = max(cand, key=lambda c: c[0])
    return M, inliers


def apply_mosaic_rgb(panel_images, geometry, log=print):
    """Apply pre-computed mosaic geometry to warp and blend RGB panels.

    Three-pass blend with spatially-varying overlap equalization:
      Pass 1: Warp panels, compute initial feather blend.
      Pass 2: For each panel, compute a per-channel spatially-varying
              correction from the overlap zone (difference between panel
              and blend, heavily smoothed + extrapolated beyond overlap).
              Then apply a global background-only gain+offset on top.
      Pass 3: Final feather blend with corrected panels.

    This eliminates both the brightness seam and background color mismatch
    between panels.

    Args:
        panel_images: list of [H,W,3] float32 arrays to mosaic
        geometry: dict from compute_mosaic_geometry()
        log: logging function

    Returns:
        Single mosaicked [H,W,3] array (float32)
    """
    import cv2

    transforms = geometry['transforms']
    canvas_w = geometry['canvas_w']
    canvas_h = geometry['canvas_h']
    offset_mat = geometry['offset']
    feather_border = geometry['feather_border']

    n = len(panel_images)
    if n == 1:
        return panel_images[0]

    # ── Pass 1: warp all panels, store warped images + masks ───────
    warped_panels = []
    warped_weights = []
    panel_masks = []   # binary: where panel has valid data

    for i, img in enumerate(panel_images):
        M_full = _compose_affine(offset_mat, transforms[i])  # 2×3

        warped = cv2.warpAffine(
            img.astype(np.float32), M_full, (canvas_w, canvas_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        h_i, w_i = img.shape[:2]
        weight = _feather_weight((h_i, w_i), border=feather_border)
        warped_weight = cv2.warpAffine(
            weight, M_full, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        warped_panels.append(warped)
        warped_weights.append(warped_weight)
        panel_masks.append(warped_weight > 0.01)
        log(f"  Panel {i+1} warped")

    # ── Build overlap map ──────────────────────────────────────────
    contrib_count = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    for pm in panel_masks:
        contrib_count += pm.astype(np.int32)
    overlap_mask = contrib_count >= 2
    n_overlap = int(np.sum(overlap_mask))
    log(f"  Overlap region: {n_overlap} pixels "
        f"({100*n_overlap/(canvas_h*canvas_w):.1f}% of canvas)")

    # ── Initial feather blend (reference truth) ────────────────────
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    for wp, ww in zip(warped_panels, warped_weights):
        w3 = ww[:, :, np.newaxis].astype(np.float64)
        canvas_sum += wp.astype(np.float64) * w3
        weight_sum += ww.astype(np.float64)

    blend_mask = weight_sum > 1e-10
    blend = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    for c in range(3):
        blend[:, :, c][blend_mask] = (
            canvas_sum[:, :, c][blend_mask] / weight_sum[blend_mask]
        ).astype(np.float32)

    # ── Pass 2: spatially-varying + background equalization ────────
    if n_overlap > 500:
        log(f"\n  Equalizing panels using overlap zone...")

        # Smoothing kernel size: ~1/6 of image diagonal, must be odd
        diag = int(np.sqrt(canvas_h**2 + canvas_w**2))
        ksize = max(101, (diag // 6) | 1)  # ensure odd
        sigma = ksize / 4.0

        for i in range(n):
            pm = panel_masks[i]
            po = pm & overlap_mask  # this panel's overlap pixels
            n_po = int(np.sum(po))
            if n_po < 200:
                log(f"    Panel {i+1}: too few overlap pixels ({n_po}), skipping")
                continue

            for c in range(3):
                # ── Spatially-varying correction ───────────────────
                # Compute difference: blend - panel in overlap zone
                diff_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
                diff_map[po] = blend[:, :, c][po] - warped_panels[i][:, :, c][po]

                # Weight map: 1 in overlap, 0 elsewhere
                diff_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
                diff_weight[po] = 1.0

                # Smooth both numerator and denominator → normalized smooth diff
                # This extrapolates the correction beyond the overlap zone.
                # _smooth_lowpass (downscale→blur→upscale) is ~the same result
                # as a huge-kernel GaussianBlur but far faster on big mosaics.
                smooth_num = _smooth_lowpass(diff_map, sigma)
                smooth_den = _smooth_lowpass(diff_weight, sigma)
                smooth_den = np.maximum(smooth_den, 1e-6)
                correction = smooth_num / smooth_den

                # Apply spatially-varying correction to entire panel
                warped_panels[i][:, :, c][pm] += correction[pm]

                # ── Background-level fine-tune ─────────────────────
                # Use darkest 30% of overlap pixels for bg alignment
                panel_vals = warped_panels[i][:, :, c][po]
                blend_vals = blend[:, :, c][po]
                lum_panel = (LR * warped_panels[i][:, :, 0][po]
                           + LG * warped_panels[i][:, :, 1][po]
                           + LB * warped_panels[i][:, :, 2][po])
                thresh = np.percentile(lum_panel, 30)
                bg_mask_local = lum_panel <= thresh

                if np.sum(bg_mask_local) > 100:
                    bg_panel = panel_vals[bg_mask_local]
                    bg_blend = blend_vals[bg_mask_local]
                    bg_offset = float(np.median(bg_blend) - np.median(bg_panel))
                    warped_panels[i][:, :, c][pm] += bg_offset

            log(f"    Panel {i+1}: spatial + bg equalization "
                f"(ksize={ksize}, {n_po} overlap px)")
    else:
        log(f"  Overlap too small for equalization ({n_overlap} px)")

    # ── Pass 3: final blend with equalized panels ──────────────────
    canvas_sum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    for wp, ww in zip(warped_panels, warped_weights):
        w3 = ww[:, :, np.newaxis].astype(np.float64)
        canvas_sum += wp.astype(np.float64) * w3
        weight_sum += ww.astype(np.float64)

    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    final_mask = weight_sum > 1e-10
    for c in range(3):
        result[:, :, c][final_mask] = (
            canvas_sum[:, :, c][final_mask] / weight_sum[final_mask]
        ).astype(np.float32)

    result = np.clip(result, 0, 1)

    log(f"  RGB Mosaic complete: {canvas_w} x {canvas_h}, "
        f"range=[{np.min(result):.4f}, {np.max(result):.4f}]")

    return result


def refine_channel_translation(ref_ha, moving, label, log=print,
                               min_shift_px=0.25, max_shift_px=20.0):
    """Refine residual translational misalignment of a channel to Ha.

    Uses phase correlation on star-emphasized planes, then applies the
    inverse shift to the moving channel when the shift is plausible.
    """
    ref = _alignment_plane(ref_ha)
    mov = _alignment_plane(moving)

    (dx, dy), response = cv2.phaseCorrelate(ref, mov)
    shift_mag = float(np.hypot(dx, dy))

    if response < 0.02:
        log(f"  Alignment refine ({label}): low confidence (resp={response:.3f}), skipped")
        return moving
    if shift_mag < min_shift_px:
        return moving
    if shift_mag > max_shift_px:
        log(f"  Alignment refine ({label}): shift {shift_mag:.2f}px too large, skipped")
        return moving

    M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    h, w = moving.shape[:2]
    corrected = cv2.warpAffine(
        moving.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    log(f"  Alignment refine ({label}): applied dX={-dx:.2f}px dY={-dy:.2f}px "
        f"(raw {dx:.2f},{dy:.2f}, resp={response:.3f})")
    return corrected.astype(np.float32)


def normalize_channels(sii, ha, oiii, log=print):
    """Normalize SHO channels for palette combination.

    Strategy:
      1. Sigma-clipped background estimation per channel
      2. Background subtraction
      3. Scale each channel so 99th-percentile of signal = 1.0

    This ensures each emission line contributes proportionally to the
    final palette. The spatial distribution of emission is preserved
    while overall brightness levels are equalized -- which is essential
    for a balanced Hubble palette.
    """
    log("  Normalizing channels...")
    channels = {'SII': sii, 'Ha': ha, 'OIII': oiii}
    normalized = {}

    for name, data in channels.items():
        bg, sigma = estimate_background(data)

        # Subtract background
        data_sub = np.clip(data - bg, 0, None)

        # 99th percentile of above-background signal
        signal_pixels = data_sub[data_sub > sigma]
        if len(signal_pixels) > 100:
            signal_peak = float(np.percentile(signal_pixels, 99.0))
        else:
            mx = float(np.max(data_sub))
            signal_peak = mx if mx > 0 else 1.0

        if signal_peak > 0:
            data_norm = data_sub / signal_peak
        else:
            data_norm = data_sub

        data_norm = np.clip(data_norm, 0, 1).astype(np.float32)

        valid = data_norm[data_norm > 0]
        med = float(np.median(valid)) if len(valid) > 0 else 0
        log(f"    {name:5s}: bg={bg:.6f} sigma={sigma:.6f} "
            f"peak={signal_peak:.6f} -> norm_median={med:.6f}")

        normalized[name] = data_norm

    return normalized['SII'], normalized['Ha'], normalized['OIII']


def _alignment_plane(data):
    """Build a star-emphasized plane for translation estimation."""
    x = np.asarray(data, dtype=np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=2.0, sigmaY=2.0)
    hp = np.clip(x - blur, 0, None)
    p = float(np.percentile(hp, 99.5)) if np.any(hp > 0) else 0.0
    if p > 0:
        hp = hp / p
    return np.clip(hp, 0, 1).astype(np.float32)


def align_channels(sii_path, ha_path, oiii_path, work_dir, log=print):
    """Align SHO channels with pure-cv2 2-pass registration (Ha = reference).

    Replaces the former Siril 2-pass + COG step. Ha is forced as the reference
    so it is never resampled; SII and OIII are warped onto it. Residual
    sub-pixel translation is polished afterward by refine_channel_translation
    in the caller. Returns (sii_aligned, ha_aligned, oiii_aligned) as 2D float32.
    """
    from .preprocess import register_frames

    def mono(a):
        return a if a.ndim == 2 else a[0]

    ha = mono(load_fits(ha_path)[0].astype(np.float32))
    sii = mono(load_fits(sii_path)[0].astype(np.float32))
    oiii = mono(load_fits(oiii_path)[0].astype(np.float32))

    # Order [Ha, SII, OIII] and force Ha (index 0) as the reference.
    aligned, _ = register_frames(
        [ha, sii, oiii], log=log,
        labels=['Ha (ref)', 'SII', 'OIII'], ref_idx=0)
    ha_a, sii_a, oiii_a = aligned

    # Out-of-frame pixels from warping are NaN; zero them for downstream math.
    sii_a = np.nan_to_num(sii_a, nan=0.0).astype(np.float32)
    oiii_a = np.nan_to_num(oiii_a, nan=0.0).astype(np.float32)
    ha_a = ha_a.astype(np.float32)

    for name, data in [("Ha (ref)", ha_a), ("SII", sii_a), ("OIII", oiii_a)]:
        log(f"    {name:10s}: shape={data.shape}  "
            f"range=[{np.min(data):.6f}, {np.max(data):.6f}]")
    return sii_a, ha_a, oiii_a


class SHOPipeline:
    """Professional SHO Hubble Palette processing pipeline.

    Takes three unstretched narrowband masters and produces a properly
    stretched, balanced Hubble-palette image with per-step cv2 previews.
    """

    def __init__(self, sii_paths, ha_paths, oiii_paths,
                 output_dir=None, preview=True, interactive=False,
                 stretch_target=STRETCH_TARGET, scnr_amount=SCNR_AMOUNT,
                 star_desat=STAR_DESAT, sat_boost=SAT_BOOST,
                 recolor_only=False, flatten_background=False,
                 hue_strength=0.40, oiii_factor=0.32,
                 truthful_mode=False,
                 hubbleize=True, hubbleize_strength=0.45,
                 sii_boost=1.0, oiii_boost=1.0, star_scale=0.70,
                 local_stretch_strength=0.0,
                 star_consensus='auto'):

        # Always work with lists of paths (mosaic is the default mode)
        self.sii_paths  = [os.path.abspath(p) for p in sii_paths]
        self.ha_paths   = [os.path.abspath(p) for p in ha_paths]
        self.oiii_paths = [os.path.abspath(p) for p in oiii_paths]

        if not self.sii_paths or not self.ha_paths or not self.oiii_paths:
            raise ValueError("No FITS files found for one or more channels")

        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = os.path.dirname(self.ha_paths[0])
        os.makedirs(self.output_dir, exist_ok=True)

        self.work_dir = os.path.join(self.output_dir, '_sho_work')
        os.makedirs(self.work_dir, exist_ok=True)

        self.stretch_target = stretch_target
        self.scnr_amount    = scnr_amount
        self.star_desat     = star_desat
        self.sat_boost      = sat_boost
        self.recolor_only        = recolor_only
        self.flatten_background  = flatten_background
        self.hue_strength        = hue_strength
        self.oiii_factor    = oiii_factor
        self.sii_boost      = float(sii_boost)
        self.oiii_boost     = float(oiii_boost)
        self.star_scale     = float(star_scale)
        self.local_stretch_strength = float(local_stretch_strength)
        self.truthful_mode  = bool(truthful_mode)
        self.hubbleize = bool(hubbleize)
        self.hubbleize_strength = float(hubbleize_strength)
        scm = (star_consensus or 'auto').lower().strip()
        if scm not in ('auto', 'off', 'soft', 'strict'):
            scm = 'auto'
        if scm == 'auto':
            scm = 'off' if self.truthful_mode else 'soft'
        self.star_consensus = scm

        self.preview = PipelinePreview(enabled=preview, interactive=interactive)
        self.phase_dir = self.work_dir  # root dir for phase PNGs
        self.step = 0

    def log(self, msg):
        print(msg)

    def _save_phase_png(self, panel_idx, phase, rgb_hwc):
        """Save a phase PNG to the work directory."""
        filename = f"panel_{panel_idx + 1}_phase_{phase}.png"
        path = os.path.join(self.phase_dir, filename)
        img8 = np.clip(rgb_hwc * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(path, img8[:, :, ::-1])  # RGB -> BGR for cv2
        self.log(f"  [{filename}] saved")

    def _save_star_debug(self, panel_idx, starless_rgb, stars_rgb):
        """Save explicit debug products for starless and stars layers."""
        panel = panel_idx + 1
        base = f"panel_{panel}"

        starless_png = os.path.join(self.phase_dir, f"{base}_debug_starless.png")
        stars_png = os.path.join(self.phase_dir, f"{base}_debug_stars.png")
        starless_fit = os.path.join(self.phase_dir, f"{base}_debug_starless.fit")
        stars_fit = os.path.join(self.phase_dir, f"{base}_debug_stars.fit")

        cv2.imwrite(starless_png,
                    np.clip(starless_rgb * 255, 0, 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(stars_png,
                    np.clip(stars_rgb * 255, 0, 255).astype(np.uint8)[:, :, ::-1])
        save_fits(np.transpose(starless_rgb.astype(np.float32), (2, 0, 1)), starless_fit)
        save_fits(np.transpose(stars_rgb.astype(np.float32), (2, 0, 1)), stars_fit)

        self.log(f"  [debug] starless: {starless_png}")
        self.log(f"  [debug] stars:    {stars_png}")

    def _crop_to_full_coverage(self, rgb, coverage=None):
        """Crop to where all 3 channels have data, blacking out the rest.

        `coverage` (bool [H, W]) is the authoritative all-3-channel mask taken
        from the mosaic warp — preferred, because after colour processing a
        covered pixel can still be near-zero (and vice-versa). Pixels outside it
        are zeroed and the image is cropped to its bounding box, so partial tiles
        keep only the area where every channel is present. Without a mask,
        coverage is inferred from the RGB values (legacy single-image path).
        """
        if coverage is None:
            threshold = 0.002  # ignore feathered near-zero edges
            coverage = np.all(rgb > threshold, axis=2)  # [H, W] bool
        elif coverage.shape != rgb.shape[:2]:
            # Defensive: StarNet normally preserves dims, but if it cropped,
            # align the mask to the (possibly smaller) processed image.
            h = min(coverage.shape[0], rgb.shape[0])
            w = min(coverage.shape[1], rgb.shape[1])
            coverage = coverage[:h, :w]
            rgb = rgb[:h, :w]

        if not np.any(coverage):
            self.log("  WARNING: No pixels with full 3-channel coverage!")
            return rgb
        if np.all(coverage):
            self.log("  All pixels have full 3-channel coverage — no crop needed")
            return rgb

        # Black out anything lacking all-3 coverage so partial edges don't show.
        rgb = rgb.copy()
        rgb[~coverage] = 0.0

        rows = np.any(coverage, axis=1)
        cols = np.any(coverage, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        h_orig, w_orig = rgb.shape[:2]
        cropped = rgb[r_min:r_max + 1, c_min:c_max + 1]
        h_new, w_new = cropped.shape[:2]

        pct = 100 * (1.0 - (h_new * w_new) / (h_orig * w_orig))
        self.log(f"  Cropped to 3-channel coverage: "
                 f"{w_orig}×{h_orig} → {w_new}×{h_new}  "
                 f"({pct:.1f}% removed)")
        self.log(f"    Margins removed — top={r_min}  bottom={h_orig - 1 - r_max}  "
                 f"left={c_min}  right={w_orig - 1 - c_max}")
        return cropped

    def run(self):
        """Execute the full SHO pipeline (hybrid).

        Per-cluster processing for STRUCTURE: each complete pointing is aligned
        and run through the full SHO pipeline independently — its stretch adapts
        to that region, drawing out local detail — then the finished RGB panels
        are mosaicked (feather + seam-equalised). A final GLOBAL BALANCE pass
        cleans the assembled mosaic (background gradient flatten + zero-aware
        neutralise + gentle colour/saturation unify) — the clean-background
        benefit of the per-channel approach, applied once at the end.
        """
        self._banner()
        t0 = time.time()
        try:
            sii_panels = _parallel_map(self._load_mono, self.sii_paths)
            ha_panels = _parallel_map(self._load_mono, self.ha_paths)
            oiii_panels = _parallel_map(self._load_mono, self.oiii_paths)
            n = min(len(sii_panels), len(ha_panels), len(oiii_panels))
            self.preview.init_grid(n)

            self._single_panel = (n == 1)
            if n == 1:
                sii, ha, oiii = self._step_align(
                    sii_panels[0], ha_panels[0], oiii_panels[0])
                final = self._process_sho(sii, ha, oiii, panel_idx=0)
            else:
                final = self._step_mosaic_processed(
                    sii_panels[:n], ha_panels[:n], oiii_panels[:n])

            # Full global balance (gradient flatten + uniform neutralize) is
            # disabled — it crushed the gold (0.25→0.12) and over-blued the
            # result. Instead, a gentle nebula-safe background gradient
            # neutralize removes the soft red sky cast that builds up when
            # per-cluster panels are mosaicked, while leaving the nebula colour
            # untouched. Single-panel runs have no inter-panel cast, so skip it.
            if not getattr(self, '_single_panel', False):
                final = self._step_background_neutralize(final)
            final = self._crop_to_full_coverage(final)

            output_path = self._step_save(final)
            elapsed = time.time() - t0
            self._finish(output_path, elapsed)
            return output_path
        except Exception as e:
            self.log(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _step_background_neutralize(self, final_rgb):
        """Remove the soft inter-panel background colour gradient (mosaic only).

        A targeted, nebula-safe replacement for the old global balance: it
        neutralises only the smoothly-varying sky cast that appears once
        per-cluster panels are mosaicked, and leaves the nebula colour alone.
        """
        self.log("\n" + "=" * 70)
        self.log("  STEP: Background gradient neutralize (mosaic seam cast)")
        self.log("=" * 70)
        r, g, b = (final_rgb[:, :, 0], final_rgb[:, :, 1], final_rgb[:, :, 2])
        r, g, b = background_gradient_neutralize(r, g, b, log=self.log)
        return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)

    def _load_mono(self, path):
        """Load a 2D float32 master from a (mono or [3,H,W]) FITS path."""
        data, _ = load_fits(path)
        return (data if data.ndim == 2 else data[0]).astype(np.float32)

    def _cache_path(self, filename):
        """Get path inside the current work dir's cache directory."""
        cache_dir = os.path.join(self.work_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, filename)

    def _cache_exists(self, *filenames):
        """Check if all named cache files exist."""
        return all(os.path.isfile(self._cache_path(f)) for f in filenames)

    def _cache_save_rgb(self, name, rgb_hwc):
        """Save [H,W,3] float32 RGB to cache as [3,H,W] FITS + PNG preview."""
        path = self._cache_path(name)
        save_fits(np.transpose(rgb_hwc, (2, 0, 1)), path)
        # PNG preview (RGB already stretched to [0,1])
        png_path = os.path.splitext(path)[0] + '.png'
        img8 = np.clip(rgb_hwc * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(png_path, img8[:, :, ::-1])  # RGB → BGR for cv2
        return path

    def _cache_load_rgb(self, name):
        """Load [3,H,W] FITS from cache, return as [H,W,3] float32."""
        path = self._cache_path(name)
        data, _ = load_fits(path)
        if len(data.shape) == 3 and data.shape[0] == 3:
            return np.transpose(data, (1, 2, 0))
        return data

    def _cache_save_mono(self, name, data_2d):
        """Save 2D float32 array to cache + PNG preview."""
        path = self._cache_path(name)
        save_fits(data_2d, path)
        # PNG preview (gamma-stretch linear data for visibility)
        png_path = os.path.splitext(path)[0] + '.png'
        stretched = preview_stretch_mono(data_2d)
        img8 = np.clip(stretched * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(png_path, img8)

    def _cache_load_mono(self, name):
        """Load 2D float32 array from cache."""
        data, _ = load_fits(self._cache_path(name))
        return data


    # ── Core SHO processing ─────────────────────────────────────────────

    def _step_mosaic_processed(self, sii_panels, ha_panels, oiii_panels):
        """Process each panel through full SHO, then SIFT-mosaic the results.

        Strategy:
          1. For each panel: align channels → full SHO pipeline → RGB
          2. SIFT on processed RGB luminance → affine geometry (4 DOF)
          3. Mosaic finished RGB panels with feather blending

        This gives excellent SIFT matches (processed images are high
        contrast) and avoids the seam issues of raw-mosaic-then-process.
        """
        n = len(ha_panels)
        orig_work_dir = self.work_dir
        panel_rgbs = []

        for i in range(n):
            self.log(f"\n{'#' * 70}")
            self.log(f"  PANEL {i+1}/{n}")
            self.log(f"{'#' * 70}")

            # Per-panel work directory (for Siril temp files)
            panel_work = os.path.join(orig_work_dir, f'_panel_{i+1}')
            os.makedirs(panel_work, exist_ok=True)
            self.work_dir = panel_work
            self.step = 0

            # Align channels for this panel (cached)
            sii_a, ha_a, oiii_a = self._step_align(
                sii_panels[i], ha_panels[i], oiii_panels[i])

            # Full SHO pipeline (cached at multiple checkpoints)
            rgb = self._process_sho(sii_a, ha_a, oiii_a, panel_idx=i)
            panel_rgbs.append(rgb)

        # Restore work_dir
        self.work_dir = orig_work_dir

        # ── SIFT on processed RGB luminance ─────────────────────────
        self.step += 1
        self.log(f"\n{'=' * 70}")
        self.log(f"  STEP {self.step}: Mosaic {n} Processed Panels "
                 f"(SIFT on RGB Luminance, Affine 4-DOF)")
        self.log(f"{'=' * 70}")

        luminances = []
        for i, rgb in enumerate(panel_rgbs):
            lum = (LR * rgb[:, :, 0] + LG * rgb[:, :, 1]
                   + LB * rgb[:, :, 2]).astype(np.float32)
            med = float(np.median(lum[lum > 0.01]))
            self.log(f"  Panel {i+1}: luminance median={med:.4f}  "
                     f"range=[{np.min(lum):.4f}, {np.max(lum):.4f}]")
            luminances.append(lum)

        self.log("\n  Computing SIFT affine geometry...")
        geometry = compute_mosaic_geometry(luminances, log=self.log)

        # Filter to only the panels that were successfully matched
        kept = geometry.get('kept_indices')
        if kept is not None and len(kept) < n:
            dropped = [i + 1 for i in range(n) if i not in kept]
            self.log(f"  Panel(s) {dropped} excluded from mosaic")
            panel_rgbs = [panel_rgbs[i] for i in kept]

        self.log("\n  Mosaicking processed RGB panels...")
        final = apply_mosaic_rgb(panel_rgbs, geometry, log=self.log)

        # Phase 3: mosaic result
        phase3_path = os.path.join(self.phase_dir, 'phase_3_mosaic.png')
        img8 = np.clip(final * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(phase3_path, img8[:, :, ::-1])
        self.log(f"  [phase_3_mosaic.png] saved")
        self.preview.show_full(final, "Phase 3 \u2014 Mosaic")

        return final

    def _step_align(self, sii, ha, oiii):
        self.step = 2
        self.log("\n" + "=" * 70)
        self.log("  STEP 2: Channel Alignment (cv2 2-Pass, Ha reference)")
        self.log("=" * 70)

        # Check cache first
        if self._cache_exists('aligned_sii.fit', 'aligned_ha.fit',
                              'aligned_oiii.fit'):
            self.log("  [CACHE] Aligned channels found, loading from cache")
            sii_a = self._cache_load_mono('aligned_sii.fit')
            ha_a = self._cache_load_mono('aligned_ha.fit')
            oiii_a = self._cache_load_mono('aligned_oiii.fit')

            # Even cached alignments can become stale/noisy with changing
            # input selections; refine residual translation each run.
            sii_a = refine_channel_translation(ha_a, sii_a, 'SII', log=self.log)
            oiii_a = refine_channel_translation(ha_a, oiii_a, 'OIII', log=self.log)

            for name, data in [("Ha (ref)", ha_a),
                               ("SII", sii_a), ("OIII", oiii_a)]:
                self.log(f"    {name:10s}: shape={data.shape}  "
                         f"range=[{np.min(data):.6f}, {np.max(data):.6f}]")

            self._cache_save_mono('aligned_sii.fit', sii_a)
            self._cache_save_mono('aligned_oiii.fit', oiii_a)
            return sii_a, ha_a, oiii_a

        self.log("  Method: cv2 2-pass star registration (asterism + similarity)")
        self.log("  Reference: Ha (kept un-resampled)")

        # Write channels to temp FITS so align_channels reads via load_fits.
        align_sii  = os.path.join(self.work_dir, '_align_sii.fit')
        align_ha   = os.path.join(self.work_dir, '_align_ha.fit')
        align_oiii = os.path.join(self.work_dir, '_align_oiii.fit')
        save_fits(sii, align_sii)
        save_fits(ha, align_ha)
        save_fits(oiii, align_oiii)

        sii_a, ha_a, oiii_a = align_channels(
            align_sii, align_ha, align_oiii,
            self.work_dir, log=self.log)

        # Refine residual per-channel translation to Ha reference.
        sii_a = refine_channel_translation(ha_a, sii_a, 'SII', log=self.log)
        oiii_a = refine_channel_translation(ha_a, oiii_a, 'OIII', log=self.log)

        # Ensure matching shapes after alignment
        shapes = [sii_a.shape, ha_a.shape, oiii_a.shape]
        if len(set(shapes)) > 1:
            self.log("  Post-alignment shape mismatch, cropping to common size")
            min_h = min(s[-2] for s in shapes)
            min_w = min(s[-1] for s in shapes)
            sii_a = sii_a[..., :min_h, :min_w]
            ha_a = ha_a[..., :min_h, :min_w]
            oiii_a = oiii_a[..., :min_h, :min_w]

        # Save to cache
        self._cache_save_mono('aligned_sii.fit', sii_a)
        self._cache_save_mono('aligned_ha.fit', ha_a)
        self._cache_save_mono('aligned_oiii.fit', oiii_a)
        self.log("  [CACHE] Saved aligned channels to cache")

        return sii_a, ha_a, oiii_a

    def _process_sho(self, sii, ha, oiii, panel_idx=0):
        """Run the full SHO colour pipeline on aligned channels.

        Caches intermediate results in self.work_dir/cache/.
        On re-run, skips steps whose cached outputs already exist.
        Updates the preview grid at phase boundaries.

        Returns [H, W, 3] float32 RGB.
        """
        pn = panel_idx + 1  # 1-based for display

        # ── Check for final panel RGB cache ──────────────────────────
        if (self.recolor_only or self.flatten_background) and self._cache_exists('panel_rgb.fit'):
            self.log("  [RECOLOR] Deleting panel_rgb.fit to force color-balance re-run")
            os.remove(self._cache_path('panel_rgb.fit'))
            png = self._cache_path('panel_rgb.png')
            if os.path.isfile(png):
                os.remove(png)

        if self._cache_exists('panel_rgb.fit'):
            self.log("  [CACHE] panel_rgb.fit found \u2014 skipping SHO pipeline")
            final = self._cache_load_rgb('panel_rgb.fit')
            self.log(f"  [CACHE] Loaded {final.shape[1]}x{final.shape[0]} RGB, "
                     f"range=[{np.min(final):.4f}, {np.max(final):.4f}]")
            self._save_phase_png(panel_idx, 2, final)
            self.preview.update_panel(
                panel_idx, final, f"Panel {pn} \u2014 Processed (cached)")
            return final

        # ── Check for starless + stars cache ─────────────────────────
        if self._cache_exists('starless.fit', 'stars.fit'):
            self.log("  [CACHE] starless.fit + stars.fit found "
                     "\u2014 skipping steps 3-6")
            starless = self._cache_load_rgb('starless.fit')
            stars = self._cache_load_rgb('stars.fit')
            self.log(f"  [CACHE] Starless: {starless.shape}, "
                     f"Stars: {stars.shape}")
            self._save_star_debug(panel_idx, starless, stars)
        else:
            sii_n, ha_n, oiii_n = self._step_normalize(sii, ha, oiii)
            r, g, b             = self._step_map_sho(sii_n, ha_n, oiii_n)

            # Phase 1: unstretched RGB composite (preview-stretched)
            rgb_phase1 = preview_stretch_rgb(r, g, b, target_median=0.20)
            self._save_phase_png(panel_idx, 1, rgb_phase1)
            self.preview.update_panel(
                panel_idx, rgb_phase1, f"Panel {pn} \u2014 Phase 1")

            r_s, g_s, b_s       = self._step_stretch(r, g, b)
            starless, stars     = self._step_star_removal(r_s, g_s, b_s)

            # Cache star removal outputs (most expensive step)
            self._cache_save_rgb('starless.fit', starless)
            self._cache_save_rgb('stars.fit', stars)
            self.log("  [CACHE] Saved starless.fit + stars.fit")
            self._save_star_debug(panel_idx, starless, stars)

        sl_r, sl_g, sl_b        = self._step_scnr(starless)
        if self.flatten_background:
            st_r_tmp, st_g_tmp, st_b_tmp = self._step_flatten_background(
                                      stars[:,:,0], stars[:,:,1], stars[:,:,2],
                                      label='stars')
            stars = np.stack([st_r_tmp, st_g_tmp, st_b_tmp], axis=-1)
        sl_r, sl_g, sl_b        = self._step_color_balance(sl_r, sl_g, sl_b)
        st_r, st_g, st_b        = self._step_process_stars(stars)
        final                   = self._step_recombine(
                                      sl_r, sl_g, sl_b,
                                      st_r, st_g, st_b)

        # Phase 2: fully processed panel
        self._save_phase_png(panel_idx, 2, final)
        self.preview.update_panel(
            panel_idx, final, f"Panel {pn} \u2014 Processed")

        # Cache final panel RGB
        self._cache_save_rgb('panel_rgb.fit', final)
        self.log("  [CACHE] Saved panel_rgb.fit")

        return final

    # ── Step 1: Load ────────────────────────────────────────────────────────


    # ── Step 2: Align ───────────────────────────────────────────────────────


    # ── Step 3: Normalize ───────────────────────────────────────────────────

    def _step_normalize(self, sii, ha, oiii):
        self.step = 3
        self.log("\n" + "=" * 70)
        self.log("  STEP 3: Channel Normalization")
        self.log("=" * 70)
        self.log("  Strategy: background-subtract + peak-normalize to equalize signal")

        sii_n, ha_n, oiii_n = normalize_channels(sii, ha, oiii, log=self.log)

        return sii_n, ha_n, oiii_n

    # ── Step 4: SHO mapping ─────────────────────────────────────────────────

    def _step_map_sho(self, sii, ha, oiii):
        self.step = 4
        self.log("\n" + "=" * 70)
        self.log("  STEP 4: SHO -> RGB Mapping")
        self.log("=" * 70)
        self.log("  Mapping:  S-II -> Red    Ha -> Green    O-III -> Blue")

        r, g, b = sii.copy(), ha.copy(), oiii.copy()

        for ch_name, ch_data in [("R (SII)", r), ("G (Ha)", g), ("B (OIII)", b)]:
            v = ch_data[ch_data > 0]
            med = float(np.median(v)) if len(v) > 0 else 0
            self.log(f"    {ch_name}: median={med:.6f}")

        return r, g, b

    # ── Step 5: Stretch ─────────────────────────────────────────────────────

    def _step_stretch(self, r, g, b):
        self.step = 5
        self.log("\n" + "=" * 70)
        self.log("  STEP 5: Linked Arcsinh Stretch")
        self.log("=" * 70)
        self.log(f"  Target median: {self.stretch_target:.3f}")

        if self.local_stretch_strength > 0:
            self.log(f"  Locally-adaptive stretch "
                     f"(strength={self.local_stretch_strength:.2f}) — recovers "
                     f"local contrast across the mosaic")
            r_s, g_s, b_s = local_adaptive_stretch(
                r, g, b, target_median=self.stretch_target,
                strength=self.local_stretch_strength, log=self.log)
        else:
            r_s, g_s, b_s = linked_stretch_rgb(
                r, g, b, target_median=self.stretch_target, log=self.log)

        if not self.truthful_mode:
            r_s, g_s, b_s = tame_blue_stars_pre_starnet(
                r_s, g_s, b_s,
                strength=0.85,
                log=self.log,
            )

        return r_s, g_s, b_s

    # ── Step 6: Star removal ────────────────────────────────────────────────

    def _step_star_removal(self, r, g, b):
        self.step = 6
        self.log("\n" + "=" * 70)
        self.log("  STEP 6: Star Removal (local StarNet++)")
        self.log("=" * 70)

        # Save stretched RGB as FITS [3, H, W]
        rgb_3hw = np.stack([r, g, b], axis=0)
        rgb_fits = os.path.join(self.work_dir, 'stretched_rgb.fit')
        save_fits(rgb_3hw, rgb_fits)
        self.log(f"  Saved stretched RGB: {rgb_fits}")

        try:
            starless, stars = remove_stars(
                rgb_fits, self.work_dir, log=self.log)
        except Exception as e:
            self.log(f"  StarNet failed: {e}")
            self.log("  Falling back: no star removal (processing full image)")
            rgb_hwc = np.stack([r, g, b], axis=-1)
            starless = rgb_hwc.copy()
            stars = np.zeros_like(rgb_hwc)

        return starless, stars

    # ── Step 7: SCNR ───────────────────────────────────────────────────────

    def _step_scnr(self, starless):
        self.step = 7
        self.log("\n" + "=" * 70)
        self.log("  STEP 7: SCNR Green Removal (Luminosity-Preserving)")
        self.log("=" * 70)
        if self.truthful_mode:
            self.log("  Truthful mode: skipping SCNR (preserve direct SHO channel balance)")
            return starless[:, :, 0], starless[:, :, 1], starless[:, :, 2]
        self.log("  Ha -> Green creates excessive green in SHO.")
        self.log("  SCNR removes green excess; luminosity restored via L scaling.")

        r = starless[:, :, 0]
        g = starless[:, :, 1]
        b = starless[:, :, 2]

        r_out, g_out, b_out = scnr_green(
            r, g, b, amount=self.scnr_amount, log=self.log)

        return r_out, g_out, b_out

    # ── Step 7b: Background flatten (optional) ───────────────────────────

    def _step_flatten_background(self, r, g, b, label='image'):
        self.log("\n" + "=" * 70)
        self.log(f"  STEP 7b: Background Flatten ({label})")
        self.log("=" * 70)
        return flatten_background_rgb(r, g, b, grid=24, sample_pct=5,
                                      log=self.log)

    # ── Step 8: Color balance ───────────────────────────────────────────────

    def _step_color_balance(self, r, g, b):
        self.step = 8
        self.log("\n" + "=" * 70)
        self.log("  STEP 8: Color Balance & Hubble Palette Tuning")
        self.log("=" * 70)

        r, g, b = neutralize_background_rgb(r, g, b, log=self.log)
        if self.truthful_mode:
            self.log("  Truthful mode: skipping hue-shift and saturation boost")
            return r, g, b
        # Per-channel SHO balance: lift SII→Red (SII rarely exceeds Ha after a
        # global normalise, so without this everything collapses to gold) and
        # OIII→Blue (to restore teal/blue) — the warm/teal SHO variety.
        if abs(self.sii_boost - 1.0) > 1e-3:
            self.log(f"  SII red-boost: R x{self.sii_boost:.2f}")
            r = np.clip(r * self.sii_boost, 0, 1).astype(np.float32)
        if abs(self.oiii_boost - 1.0) > 1e-3:
            self.log(f"  OIII blue-boost: B x{self.oiii_boost:.2f}")
            b = np.clip(b * self.oiii_boost, 0, 1).astype(np.float32)
        r, g, b = hubble_color_refine(r, g, b, strength=self.hue_strength,
                                      oiii_factor=self.oiii_factor, log=self.log)
        r, g, b = boost_saturation(r, g, b, factor=self.sat_boost, log=self.log)
        if self.hubbleize:
            r, g, b = hubbleize_with_skimage(
                r, g, b,
                strength=self.hubbleize_strength,
                log=self.log,
            )

        return r, g, b

    # ── Step 9: Star processing ─────────────────────────────────────────────

    def _step_process_stars(self, stars):
        self.step = 9
        self.log("\n" + "=" * 70)
        self.log("  STEP 9: Star Processing (Desaturation + Anti-Purple)")
        self.log("=" * 70)

        r, g, b = stars[:, :, 0], stars[:, :, 1], stars[:, :, 2]

        if self.truthful_mode:
            self.log("  Truthful mode: skipping star color manipulation")
            return r, g, b

        if np.max(stars) < 0.01:
            self.log("  No significant star signal detected, skipping.")
            return r, g, b

        r, g, b = apply_star_channel_consensus(
            r, g, b,
            mode=self.star_consensus,
            log=self.log,
        )

        r_p, g_p, b_p = process_stars(
            r, g, b, desat_amount=self.star_desat, log=self.log)

        return r_p, g_p, b_p

    # ── Step 10: Recombine ──────────────────────────────────────────────────

    def _step_recombine(self, sl_r, sl_g, sl_b, st_r, st_g, st_b):
        self.step = 10
        self.log("\n" + "=" * 70)
        self.log("  STEP 10: Recombine (Screen Blend)")
        self.log("=" * 70)
        self.log("  Screen blend: result = 1 - (1-starless)*(1-stars)")

        starless = np.stack([sl_r, sl_g, sl_b], axis=-1).astype(np.float32)
        stars    = np.stack([st_r, st_g, st_b], axis=-1).astype(np.float32)

        # Scale stars down to reduce their intensity in the blend (also limits
        # how much the screen-blend brightens the median, so the nebula stretch
        # can lift faint structure without the whole image over-exposing).
        star_scale = 1.00 if self.truthful_mode else float(self.star_scale)
        stars = stars * star_scale
        self.log(f"  Star intensity scaled to {star_scale:.0%}")

        final = screen_blend(starless, stars)

        if not self.truthful_mode:
            final = suppress_blue_star_halos(
                final,
                stars,
                amount=0.90,
                log=self.log,
            )

        self.log(f"  Final stats: median={np.median(final):.4f} "
                 f"max={np.max(final):.4f}")

        return final

    # ── Step 11: Save ───────────────────────────────────────────────────────

    def _step_save(self, final):
        self.step += 1
        self.log("\n" + "=" * 70)
        self.log(f"  STEP {self.step}: Save Output")
        self.log("=" * 70)

        ha_stem = Path(self.ha_paths[0]).stem
        base_name = f"SHO_Hubble_{ha_stem}"

        # FITS (Siril-compatible [3,H,W])
        fits_path = os.path.join(self.output_dir, f"{base_name}.fit")
        fits_data = np.transpose(final, (2, 0, 1))  # [H,W,3] -> [3,H,W]

        # Propagate astrometry from source Ha file(s)
        fits_extra = {'IMAGETYP': 'SHO Hubble Palette'}
        fits_extra['CHANMAP'] = 'SII>R HA>G OIII>B'
        fits_extra['TRUTHMOD'] = bool(self.truthful_mode)
        fits_extra['SCNRUSED'] = not bool(self.truthful_mode)
        fits_extra['HUESHIFT'] = not bool(self.truthful_mode)
        fits_extra['STARPROC'] = not bool(self.truthful_mode)
        fits_extra['STARSCL'] = 1.00 if self.truthful_mode else 0.70
        fits_extra['SATBOOST'] = 1.00 if self.truthful_mode else float(self.sat_boost)
        fits_extra['HUBBLIZE'] = bool(self.hubbleize)
        fits_extra['HUBSTR'] = float(self.hubbleize_strength)
        fits_extra['STARCONS'] = self.star_consensus
        try:
            # Read all Ha panel headers for coordinate computation
            ha_headers = []
            for hp in self.ha_paths:
                try:
                    ha_headers.append(read_fits_header(hp))
                except Exception:
                    pass

            if ha_headers:
                src_hdr = ha_headers[0]

                # Copy non-positional metadata from first panel
                for key in ('RADECSYS', 'FOCALLEN', 'XPIXSZ', 'YPIXSZ',
                            'OBJECT'):
                    if key in src_hdr:
                        val = src_hdr[key]
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            pass
                        fits_extra[key] = val

                # Compute scale (arcsec/pixel) — same for all panels
                scale_aspp = None
                if 'SECPIX1' in src_hdr:
                    scale_aspp = float(src_hdr['SECPIX1'])
                elif 'SCALE' in src_hdr:
                    scale_aspp = float(src_hdr['SCALE'])
                elif 'CDELT1' in src_hdr:
                    scale_aspp = abs(float(src_hdr['CDELT1'])) * 3600.0

                if scale_aspp:
                    fits_extra['SCALE'] = scale_aspp
                    fits_extra['SECPIX1'] = scale_aspp
                    fits_extra['SECPIX2'] = scale_aspp

                # Compute centre RA/DEC for the output mosaic
                if len(ha_headers) == 1:
                    # Single panel — use its coordinates directly
                    for key in ('RA', 'DEC', 'OBJCTRA', 'OBJCTDEC',
                                'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                                'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2',
                                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
                        if key in src_hdr:
                            val = src_hdr[key]
                            try:
                                val = float(val)
                            except (ValueError, TypeError):
                                pass
                            fits_extra[key] = val
                else:
                    # Multi-panel mosaic — compute combined centre
                    ras, decs = [], []
                    for hdr in ha_headers:
                        ra = dec = None
                        for rk in ('RA', 'CRVAL1'):
                            if rk in hdr:
                                try:
                                    ra = float(hdr[rk])
                                    break
                                except (ValueError, TypeError):
                                    pass
                        for dk in ('DEC', 'CRVAL2'):
                            if dk in hdr:
                                try:
                                    dec = float(hdr[dk])
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if ra is not None and dec is not None:
                            ras.append(ra)
                            decs.append(dec)

                    if ras:
                        centre_ra = sum(ras) / len(ras)
                        centre_dec = sum(decs) / len(decs)

                        fits_extra['RA'] = centre_ra
                        fits_extra['DEC'] = centre_dec

                        # CRVAL = centre, CRPIX = centre pixel
                        fits_extra['CRVAL1'] = centre_ra
                        fits_extra['CRVAL2'] = centre_dec
                        out_h, out_w = final.shape[:2]
                        fits_extra['CRPIX1'] = out_w / 2.0
                        fits_extra['CRPIX2'] = out_h / 2.0
                        if scale_aspp:
                            cdelt = scale_aspp / 3600.0  # deg/pixel
                            fits_extra['CDELT1'] = -cdelt  # RA increases W
                            fits_extra['CDELT2'] = cdelt
                        fits_extra['CTYPE1'] = 'RA---TAN'
                        fits_extra['CTYPE2'] = 'DEC--TAN'

                        # OBJCTRA / OBJCTDEC in HMS/DMS
                        ra_h = centre_ra / 15.0
                        h = int(ra_h)
                        m = int((ra_h - h) * 60)
                        s = (ra_h - h - m / 60) * 3600
                        fits_extra['OBJCTRA'] = f"{h} {m:02d} {s:05.2f}"
                        d = int(centre_dec)
                        dm = int(abs(centre_dec - d) * 60)
                        ds = (abs(centre_dec - d) * 60 - dm) * 60
                        fits_extra['OBJCTDEC'] = f"{d} {dm:02d} {ds:05.2f}"

                        self.log(f"  Mosaic centre: RA={centre_ra:.4f}° "
                                 f"DEC={centre_dec:.4f}° "
                                 f"({len(ras)} panels)")

        except Exception:
            pass  # Non-fatal: output just won't have astrometry

        fits_extra['ROWORDER'] = 'BOTTOM-UP'
        save_fits(fits_data, fits_path, header_extra=fits_extra)
        fits_size = os.path.getsize(fits_path) / (1024 * 1024)
        self.log(f"  FITS: {fits_path}  ({fits_size:.1f} MB)")

        # Plate-solve the output to embed full WCS (rotation, distortion)
        try:
            from .plate_solve import plate_solve_if_needed as _plate_solve
            from .plate_solve import update_fits_wcs as _update_wcs
            self.log(f"  Checking output WCS ...")
            # Best-effort + time-bounded: stamping WCS on the full mosaic is a
            # nicety, not essential, and can stall on a busy local solver — keep
            # it from blocking the rest of the save.
            wcs = _plate_solve(fits_path, log_fn=self.log, timeout=90)
            if wcs:
                _update_wcs(fits_path, wcs, log_fn=self.log)
            elif not os.path.isfile(fits_path):
                self.log(f"  (plate-solve unavailable — WCS not added)")
            elif not wcs:
                self.log(f"  (plate-solve unavailable — WCS not added)")
        except Exception as e:
            self.log(f"  (plate-solve skipped: {e})")

        # 16-bit TIFF
        tiff_path = os.path.join(self.output_dir, f"{base_name}.tiff")
        tiff_u16 = np.clip(final * 65535, 0, 65535).astype(np.uint16)
        tiff_bgr = cv2.cvtColor(tiff_u16, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tiff_path, tiff_bgr)
        tiff_size = os.path.getsize(tiff_path) / (1024 * 1024)
        self.log(f"  TIFF: {tiff_path}  ({tiff_size:.1f} MB)")

        # 16-bit PNG \u2014 full precision, no 8-bit truncation (reuses the TIFF
        # array). Zooming no longer shows flat, posterized pixels.
        png_path = os.path.join(self.output_dir, f"{base_name}.png")
        cv2.imwrite(png_path, tiff_bgr)
        self.log(f"  PNG (16-bit): {png_path}")

        # 8-bit quick-look copy for the on-screen preview / phase debug only.
        png8_bgr = cv2.cvtColor(
            np.clip(final * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        phase4_path = os.path.join(self.phase_dir, 'phase_4_final.png')
        cv2.imwrite(phase4_path, png8_bgr)
        self.log(f"  [phase_4_final.png] saved")
        self.preview.show_full(final, "Phase 4 \u2014 Final")

        # Sidecar provenance for transparency in downstream sharing/publication
        prov_path = os.path.join(self.output_dir, f"{base_name}_provenance.txt")
        with open(prov_path, 'w', encoding='utf-8') as pf:
            pf.write("SHO Processing Provenance\n")
            pf.write("========================\n")
            pf.write(f"Mode: {'truthful' if self.truthful_mode else 'artistic'}\n")
            pf.write("Channel mapping: SII->R, Ha->G, OIII->B\n")
            pf.write(f"SCNR applied: {not self.truthful_mode}\n")
            pf.write(f"Hue shift applied: {not self.truthful_mode}\n")
            pf.write(f"Saturation boost: {1.00 if self.truthful_mode else float(self.sat_boost):.2f}\n")
            pf.write(f"Star color processing: {not self.truthful_mode}\n")
            pf.write(f"Star intensity scale: {1.00 if self.truthful_mode else 0.70:.2f}\n")
            pf.write(f"Advanced hubbleize (scikit-image): {self.hubbleize}\n")
            pf.write(f"Advanced hubbleize strength: {self.hubbleize_strength:.2f}\n")
            pf.write(f"Star consensus mode: {self.star_consensus}\n")
            pf.write("Astrometry: output is plate-solved when solver is available\n")
        self.log(f"  Provenance: {prov_path}")

        return fits_path

    # ── Helpers ──

    def _banner(self):
        w = 62
        self.log("")
        self.log("+" + "=" * w + "+")
        self.log("|" + "SHO Hubble Palette Processor v2.0".center(w) + "|")
        self.log("|" + "Mathematically Rigorous Narrowband Processing".center(w) + "|")
        self.log("+" + "-" * w + "+")
        self.log(f"|  SII:  {len(self.sii_paths)} panel(s)".ljust(w + 1) + "|")
        self.log(f"|  Ha:   {len(self.ha_paths)} panel(s)".ljust(w + 1) + "|")
        self.log(f"|  OIII: {len(self.oiii_paths)} panel(s)".ljust(w + 1) + "|")
        self.log(f"|  Mode: {'TRUTHFUL' if self.truthful_mode else 'ARTISTIC'}".ljust(w + 1) + "|")
        if self.hubbleize:
            self.log(f"|  Style: HUBBLEIZE (skimage {self.hubbleize_strength:.2f})".ljust(w + 1) + "|")
        self.log("|  Out:  " + self.output_dir[:w-8].ljust(w - 8) + "|")
        self.log("+" + "=" * w + "+")

    def _finish(self, output_path, elapsed):
        w = 62
        self.log("")
        self.log("+" + "=" * w + "+")
        self.log("|" + "PROCESSING COMPLETE".center(w) + "|")
        self.log("+" + "-" * w + "+")
        self.log("|  Output: " + os.path.basename(output_path).ljust(w - 10) + "|")
        self.log("|  Time:   " + f"{elapsed:.1f}s".ljust(w - 10) + "|")
        self.log("+" + "=" * w + "+")
        self.preview.finish()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLI Entry Point                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='SHO Hubble Palette Processor v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sii sii.fit --ha ha.fit --oiii oiii.fit
  %(prog)s --sii sii.fit --ha ha.fit --oiii oiii.fit --output-dir ./results
  %(prog)s --sii sii.fit --ha ha.fit --oiii oiii.fit --no-preview
  %(prog)s --sii sii.fit --ha ha.fit --oiii oiii.fit --interactive

Pipeline steps (each with cv2 preview):
  1. Load & inspect channels
  2. Star alignment (Siril 2-pass deep sky + COG framing)
  3. Channel normalization (background-subtract + peak-normalize)
  4. SHO -> RGB mapping (S->R, H->G, O->B)
  5. Linked arcsinh stretch (beta auto-calibrated from luminance)
  6. Star removal (Siril StarNet)
  7. SCNR green removal (luminosity-preserving)
  8. Color balance + Hubble palette tuning
  9. Star desaturation + anti-purple
 10. Screen-blend recombination
 11. Save (FITS + TIFF + PNG)
        """)

    parser.add_argument('--version', action='version', version=VERSION)

    # Channel inputs: directory or file(s)
    # If a directory, globs for result_*.fit inside it.
    # If file(s), uses them directly. Multiple panels are mosaicked.
    parser.add_argument('--sii', required=True, nargs='+',
                        help='S-II channel: directory or FITS file(s)')
    parser.add_argument('--ha', required=True, nargs='+',
                        help='H-alpha channel: directory or FITS file(s)')
    parser.add_argument('--oiii', required=True, nargs='+',
                        help='O-III channel: directory or FITS file(s)')

    # Output
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as first Ha file)')

    # Processing parameters
    parser.add_argument('--stretch-target', type=float, default=STRETCH_TARGET,
                        help=f'Target median after stretch (default: {STRETCH_TARGET})')
    parser.add_argument('--scnr-amount', type=float, default=SCNR_AMOUNT,
                        help=f'SCNR green removal 0-1 (default: {SCNR_AMOUNT})')
    parser.add_argument('--star-desat', type=float, default=STAR_DESAT,
                        help=f'Star desaturation 0-1 (default: {STAR_DESAT})')
    parser.add_argument('--sat-boost', type=float, default=SAT_BOOST,
                        help=f'Nebula saturation boost (default: {SAT_BOOST})')
    parser.set_defaults(truthful_mode=False, hubbleize=True)
    parser.add_argument('--truthful-mode', dest='truthful_mode', action='store_true',
                        help='Use transparent SHO mapping (strict/diagnostic mode)')
    parser.add_argument('--artistic-mode', dest='truthful_mode', action='store_false',
                        help='Enable artistic color/star manipulations (default)')
    parser.add_argument('--hubbleize', action='store_true',
                        help='Apply advanced scikit-image Hubble recolor pass (artistic mode)')
    parser.add_argument('--hubbleize-strength', type=float, default=0.45,
                        help='Strength of advanced hubbleize pass (default: 0.45)')
    parser.add_argument('--star-consensus', choices=['auto', 'off', 'soft', 'strict'],
                        default='auto',
                        help='Star-channel agreement filter (default: auto)')

    # Preview control
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable cv2 preview windows')
    parser.add_argument('--interactive', action='store_true',
                        help='Wait for keypress between steps')

    args = parser.parse_args()

    import glob as _glob

    def resolve_channel(args_list):
        """Resolve a channel argument to a list of FITS file paths.

        Accepts:
          - One or more directories → globs for result_*.fit in each
          - One or more files       → uses directly
          - Comma-separated         → splits and resolves each
          - Mix of dirs and files   → resolves each appropriately
        """
        # First flatten any comma-separated entries
        flat = []
        for a in args_list:
            flat.extend(a.split(','))
        flat = [f.strip() for f in flat if f.strip()]

        def _glob_dir(d):
            """Find result FITS in a directory."""
            for pattern in ['result_*.fit', 'result_*.fits',
                            '*.fit', '*.fits']:
                hits = sorted(_glob.glob(os.path.join(d, pattern)))
                if hits:
                    return hits
            return []

        result = []
        for entry in flat:
            if os.path.isdir(entry):
                fits = _glob_dir(entry)
                if not fits:
                    raise FileNotFoundError(
                        f"No FITS files found in directory: {entry}")
                print(f"  Found {len(fits)} file(s) in {entry}")
                result.extend(fits)
            elif os.path.isfile(entry):
                result.append(entry)
            else:
                raise FileNotFoundError(f"Not found: {entry}")

        if not result:
            raise FileNotFoundError(
                f"No FITS files resolved from: {args_list}")
        return result

    sii_paths  = resolve_channel(args.sii)
    ha_paths   = resolve_channel(args.ha)
    oiii_paths = resolve_channel(args.oiii)

    pipeline = SHOPipeline(
        sii_paths=sii_paths,
        ha_paths=ha_paths,
        oiii_paths=oiii_paths,
        output_dir=args.output_dir,
        preview=not args.no_preview,
        interactive=args.interactive,
        stretch_target=args.stretch_target,
        scnr_amount=args.scnr_amount,
        star_desat=args.star_desat,
        sat_boost=args.sat_boost,
        truthful_mode=(args.truthful_mode and not args.hubbleize),
        hubbleize=args.hubbleize,
        hubbleize_strength=args.hubbleize_strength,
        star_consensus=args.star_consensus,
    )

    output = pipeline.run()
    print(f"\nDone. Output: {output}")


if __name__ == '__main__':
    main()
