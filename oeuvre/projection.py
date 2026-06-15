#!/usr/bin/env python3
"""
Projection engine — places FITS images on a sky map.

Uses a gnomonic (tangent-plane) projection centred on the
mean position of all loaded frames.  Each image is rendered
as a thumbnail at its sky position, and the full-resolution
composite can be exported as a PNG/TIFF.
"""

import math
import os

import numpy as np
from PIL import Image

from .mosaic_prep import read_fits_header
from .natural_narrowband import load_fits
from .plate_solve import plate_solve, update_fits_wcs


# ── Data structures ─────────────────────────────────────────────────────────

class SkyFrame:
    """One FITS image with its sky metadata."""

    __slots__ = (
        'path', 'label',
        'ra_deg', 'dec_deg',
        'scale_aspp',          # arcsec per pixel
        'width_px', 'height_px',
        'fov_w_deg', 'fov_h_deg',
        'rotation_deg',        # position angle (N through E)
        'n_channels',
        'thumbnail',           # PIL Image or None
        '_pil_cache',          # cached full-res PIL image
        '_has_wcs',            # True if CROTA2 or CD matrix found in header
    )

    def __init__(self, path):
        self.path = path
        self.label = os.path.basename(path)
        self.thumbnail = None
        self._pil_cache = None
        self._read_header()

    def _read_header(self):
        hdr = read_fits_header(self.path)

        # ── Position ────────────────────────────────────────────────────
        if 'RA' in hdr and 'DEC' in hdr:
            self.ra_deg = float(hdr['RA'])
            self.dec_deg = float(hdr['DEC'])
        elif 'CRVAL1' in hdr and 'CRVAL2' in hdr:
            self.ra_deg = float(hdr['CRVAL1'])
            self.dec_deg = float(hdr['CRVAL2'])
        else:
            raise ValueError(
                f"{self.label}: no RA/DEC or CRVAL1/CRVAL2 in header"
            )

        # ── Pixel scale ─────────────────────────────────────────────────
        if 'SECPIX1' in hdr:
            self.scale_aspp = float(hdr['SECPIX1'])
        elif 'SCALE' in hdr:
            self.scale_aspp = float(hdr['SCALE'])
        elif 'CDELT1' in hdr:
            self.scale_aspp = abs(float(hdr['CDELT1'])) * 3600.0
        else:
            self.scale_aspp = 1.865  # fallback (common narrowband setup)

        # ── Dimensions ──────────────────────────────────────────────────
        self.width_px = int(float(hdr.get('NAXIS1', 3010)))
        self.height_px = int(float(hdr.get('NAXIS2', 3010)))
        naxis = int(float(hdr.get('NAXIS', 2)))
        self.n_channels = int(float(hdr.get('NAXIS3', 1))) if naxis >= 3 else 1

        # ── Rotation ────────────────────────────────────────────────────
        # Try CROTA2, then CD matrix, default 0
        if 'CROTA2' in hdr:
            self.rotation_deg = float(hdr['CROTA2'])
            self._has_wcs = True
        elif 'CD1_1' in hdr and 'CD1_2' in hdr:
            cd11 = float(hdr['CD1_1'])
            cd12 = float(hdr['CD1_2'])
            self.rotation_deg = math.degrees(math.atan2(cd12, cd11))
            self._has_wcs = True
        else:
            self.rotation_deg = 0.0
            self._has_wcs = False

        # ── FOV ─────────────────────────────────────────────────────────
        self.fov_w_deg = self.width_px * self.scale_aspp / 3600.0
        self.fov_h_deg = self.height_px * self.scale_aspp / 3600.0

    def _load_pil(self):
        """Load FITS pixel data and convert to a PIL Image.

        Data is returned as-stored from the file — no row-order
        flipping.  Rotation is handled separately in get_image().
        """
        data, _ = load_fits(self.path)
        if data is None:
            return None
        # load_fits returns (3,H,W) for RGB or (H,W) for mono
        if data.ndim == 3 and data.shape[0] in (3, 4):
            data = np.moveaxis(data, 0, -1)  # → (H,W,C)

        return _array_to_pil(data)

    def load_thumbnail(self, max_size=256):
        """Load image data and create a PIL thumbnail."""
        try:
            pil = self._load_pil()
            if pil is None:
                return
            self._pil_cache = pil  # cache full-res for rendering
            thumb = pil.copy()
            thumb.thumbnail((max_size, max_size), Image.LANCZOS)
            self.thumbnail = thumb
        except Exception:
            self.thumbnail = None

    def get_image(self, target_w, target_h):
        """Return a PIL Image scaled and rotated for sky map placement.

        Args:
            target_w, target_h: desired pixel dimensions on the canvas

        Returns:
            PIL RGBA Image properly scaled and rotated, or None.
        """
        pil = self._pil_cache
        if pil is None:
            try:
                pil = self._load_pil()
                if pil is None:
                    return None
                self._pil_cache = pil
            except Exception:
                return None

        # Scale to target size
        tw = max(int(target_w), 2)
        th = max(int(target_h), 2)
        scaled = pil.resize((tw, th), Image.LANCZOS)

        # Convert to RGBA for rotation with transparency
        if scaled.mode != 'RGBA':
            scaled = scaled.convert('RGBA')

        # Rotate to align North up.
        # CROTA2 = angle of North from +NAXIS2 (N-through-E).
        # To undo the camera rotation we rotate by -CROTA2.
        # PIL rotate() is counter-clockwise, so rotate(-CROTA2) is correct.
        if abs(self.rotation_deg) > 0.1:
            scaled = scaled.rotate(-self.rotation_deg, resample=Image.BICUBIC,
                                   expand=True, fillcolor=(0, 0, 0, 0))

        return scaled


def _array_to_pil(arr):
    """Convert a float32 numpy array to an 8-bit PIL Image.

    Handles both linear (raw) and already-stretched (processed)
    data.  If the data looks already stretched (max ≤ 1.1 and
    median > 0.05) it skips the percentile re-normalisation to
    preserve the pipeline's colour balance.
    """
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)

    vmax = float(np.max(arr))
    valid = arr[arr > 0]
    med = float(np.median(valid)) if len(valid) > 0 else 0.0

    if vmax <= 1.1 and med > 0.05:
        # Already stretched to [0,1] — just clip and scale
        arr = np.clip(arr, 0, 1)
    else:
        # Raw linear data — percentile stretch
        vmin, vmax = np.percentile(arr, [1, 99.5])
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        arr = np.clip(arr, 0, 1)

    arr = (arr * 255).astype(np.uint8)
    if len(arr.shape) == 2:
        return Image.fromarray(arr, 'L')
    if arr.shape[2] == 3:
        return Image.fromarray(arr, 'RGB')
    # Fallback: take first channel
    return Image.fromarray(arr[:, :, 0], 'L')


# ── Gnomonic projection ────────────────────────────────────────────────────

def gnomonic_project(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic (tangent-plane) projection.

    Returns (x, y) in degrees on the tangent plane centred at
    (ra0, dec0).  +x = West (increasing RA), +y = North.
    """
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    ra0 = math.radians(ra0_deg)
    dec0 = math.radians(dec0_deg)

    cos_c = (math.sin(dec0) * math.sin(dec) +
             math.cos(dec0) * math.cos(dec) * math.cos(ra - ra0))
    if cos_c <= 0:
        return (0.0, 0.0)  # behind tangent point

    x = math.cos(dec) * math.sin(ra - ra0) / cos_c
    y = (math.cos(dec0) * math.sin(dec) -
         math.sin(dec0) * math.cos(dec) * math.cos(ra - ra0)) / cos_c

    return (math.degrees(x), math.degrees(y))


# ── Sky map renderer ───────────────────────────────────────────────────────

class SkyMap:
    """Manages a collection of SkyFrames and renders them on a canvas."""

    def __init__(self):
        self.frames: list[SkyFrame] = []
        self.ra0 = 0.0   # projection centre
        self.dec0 = 0.0

    def add_file(self, path, auto_solve=True, log_fn=None):
        """Add a FITS file and return the SkyFrame.

        If the file lacks rotation info (CROTA2 / CD matrix) and
        *auto_solve* is True, attempt to plate-solve it via the
        local astrometry.net Docker solver.
        """
        sf = SkyFrame(path)

        # Auto plate-solve only if header lacks WCS rotation entirely
        if auto_solve and not sf._has_wcs:
            _log = log_fn or print
            _log(f"  No WCS in header — attempting plate solve...")
            wcs = plate_solve(sf.path, log_fn=_log)
            if wcs:
                update_fits_wcs(sf.path, wcs, log_fn=_log)
                # Re-read header with fresh WCS
                sf._pil_cache = None
                sf._read_header()
            else:
                _log(f"  Plate solve failed — using rotation=0")

        sf.load_thumbnail()
        self.frames.append(sf)
        self._update_centre()
        return sf

    def remove_frame(self, index):
        """Remove frame at index."""
        if 0 <= index < len(self.frames):
            del self.frames[index]
            self._update_centre()

    def _update_centre(self):
        """Set projection centre to mean RA/DEC of all frames."""
        if not self.frames:
            return
        self.ra0 = sum(f.ra_deg for f in self.frames) / len(self.frames)
        self.dec0 = sum(f.dec_deg for f in self.frames) / len(self.frames)

    def project_frame(self, frame):
        """Return (cx, cy, w, h) in degrees on the tangent plane."""
        cx, cy = gnomonic_project(frame.ra_deg, frame.dec_deg,
                                  self.ra0, self.dec0)
        return (cx, cy, frame.fov_w_deg, frame.fov_h_deg)

    def get_bounds(self, padding_factor=1.15):
        """Get (min_x, min_y, max_x, max_y) in degrees covering all frames."""
        if not self.frames:
            return (-1, -1, 1, 1)

        xs, ys = [], []
        for f in self.frames:
            cx, cy, fw, fh = self.project_frame(f)
            xs.extend([cx - fw/2, cx + fw/2])
            ys.extend([cy - fh/2, cy + fh/2])

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Add padding
        dx = (max_x - min_x) * (padding_factor - 1) / 2
        dy = (max_y - min_y) * (padding_factor - 1) / 2
        # Minimum extent
        dx = max(dx, 0.1)
        dy = max(dy, 0.1)

        return (min_x - dx, min_y - dy, max_x + dx, max_y + dy)

    def render_map(self, canvas_w=800, canvas_h=600, show_thumbnails=True):
        """Render the sky map to a PIL Image.

        Composites image data with feathered edges on a black background
        for a seamless deep-sky mosaic look.

        Args:
            canvas_w, canvas_h: output image size in pixels
            show_thumbnails: if True, render actual image data

        Returns:
            PIL Image (RGB)
        """
        from PIL import ImageDraw, ImageFont

        img = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))

        if not self.frames:
            draw = ImageDraw.Draw(img)
            draw.text((canvas_w // 2 - 60, canvas_h // 2),
                      'No images loaded', fill=(108, 112, 134))
            return img

        bounds = self.get_bounds()
        min_x, min_y, max_x, max_y = bounds
        span_x = max_x - min_x
        span_y = max_y - min_y

        # Uniform scaling — same deg/pixel for both axes
        scale = min(canvas_w / span_x, canvas_h / span_y)
        ox = (canvas_w - span_x * scale) / 2.0
        oy = (canvas_h - span_y * scale) / 2.0

        def sky_to_px(sx, sy):
            px = int(canvas_w - ox - (sx - min_x) * scale)
            py = int(canvas_h - oy - (sy - min_y) * scale)
            return (px, py)

        # ── Composite frames with feathered alpha ───────────────────────
        for i, frame in enumerate(self.frames):
            cx, cy, fw, fh = self.project_frame(frame)

            x1, y1 = sky_to_px(cx - fw/2, cy - fh/2)
            x2, y2 = sky_to_px(cx + fw/2, cy + fh/2)
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            rect_w = right - left
            rect_h = bottom - top

            if rect_w < 4 or rect_h < 4:
                continue

            rendered = frame.get_image(rect_w, rect_h) if show_thumbnails else None
            if rendered is None:
                continue

            # Build a feather mask — smooth falloff at the edges
            feather_px = max(8, min(rect_w, rect_h) // 6)
            mask = _feather_mask(rendered.size[0], rendered.size[1], feather_px)

            # Multiply rendered alpha by feather mask
            if rendered.mode != 'RGBA':
                rendered = rendered.convert('RGBA')
            r, g, b, a = rendered.split()
            # Combine existing alpha (rotation transparency) with feather
            a = Image.fromarray(
                np.minimum(np.array(a), np.array(mask)).astype(np.uint8)
            )
            rendered = Image.merge('RGBA', (r, g, b, a))

            # Paste centred on the frame's sky position
            cxp, cyp = sky_to_px(cx, cy)
            rw, rh = rendered.size
            paste_x = cxp - rw // 2
            paste_y = cyp - rh // 2
            img.paste(rendered, (paste_x, paste_y), rendered)

        # ── Subtle info overlay (bottom-right corner) ───────────────────
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/SFNSMono.ttf", 11)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(img)
        ra_h = self.ra0 / 15.0
        h = int(ra_h)
        m = int((ra_h - h) * 60)
        s = (ra_h - h - m / 60) * 3600
        info = (f"RA {h}h{m:02d}m{s:04.1f}s  "
                f"DEC {self.dec0:+.3f}°  "
                f"{span_x:.1f}°×{span_y:.1f}°")
        draw.text((8, canvas_h - 18), info,
                  fill=(60, 60, 80), font=font)

        return img

    def export_png(self, output_path, width=4000, height=3000):
        """Export a high-resolution sky map PNG."""
        img = self.render_map(canvas_w=width, canvas_h=height,
                              show_thumbnails=True)
        img.save(output_path, 'PNG')
        return output_path


def _feather_mask(w, h, border):
    """Create a feathered alpha mask — full opacity in the centre,
    smooth falloff to transparent at the edges.

    Returns a PIL 'L' mode Image (0 = transparent, 255 = opaque).
    """
    # 1-D ramps for each axis
    x_ramp = np.ones(w, dtype=np.float32)
    y_ramp = np.ones(h, dtype=np.float32)

    for i in range(min(border, w // 2)):
        t = i / max(border, 1)
        # Smooth cosine ease-in
        v = 0.5 - 0.5 * math.cos(t * math.pi)
        x_ramp[i] = v
        x_ramp[w - 1 - i] = v

    for i in range(min(border, h // 2)):
        t = i / max(border, 1)
        v = 0.5 - 0.5 * math.cos(t * math.pi)
        y_ramp[i] = v
        y_ramp[h - 1 - i] = v

    # Outer product → 2-D mask
    mask_2d = np.outer(y_ramp, x_ramp)
    mask_2d = (mask_2d * 255).astype(np.uint8)
    return Image.fromarray(mask_2d, 'L')
