#!/usr/bin/env python3
"""Procedural star-field background + frosted-glass panels for the GUI.

Rendered once with numpy + Pillow (no GPU, no per-frame cost) and cached, so it
is cheap on a laptop. The starfield is static, so a frosted-glass panel — a
blurred, tinted crop of the stars behind it — is visually identical to a live
backdrop-blur (which Tkinter can't do natively).

    render_starfield(w, h, seed) -> PIL.Image           # background
    glass_card(base, box, ...)   -> PIL.Image           # frost a panel region
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Deep-space gradient endpoints (top -> bottom).
_SPACE_TOP = (5, 7, 15)
_SPACE_BOTTOM = (11, 13, 26)


def _nebula(w, h, rng):
    """Faint patchy colour clouds for depth (teal + magenta, very subtle)."""
    def cloud(color, thresh):
        sw, sh = max(2, w // 48), max(2, h // 48)
        base = (rng.random((sh, sw)) * 255).astype(np.uint8)
        im = Image.fromarray(base).resize((w, h), Image.BICUBIC)
        im = im.filter(ImageFilter.GaussianBlur(max(w, h) // 18))
        n = np.asarray(im, np.float32) / 255.0
        n = np.clip((n - thresh) * 2.4, 0, 1)        # patchy
        return n[:, :, None] * np.array(color, np.float32)[None, None, :]
    return cloud((12, 22, 40), 0.52) * 1.0 + cloud((30, 12, 38), 0.56) * 0.75


def _vignette(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    return np.clip(1.0 - 0.22 * np.clip(r - 0.4, 0, 1) ** 2, 0, 1)[:, :, None]


def render_starfield(width, height, seed=7, n_stars=None):
    """Render an RGB starfield image (space gradient + nebula + stars)."""
    rng = np.random.default_rng(seed)
    w, h = int(width), int(height)

    top = np.array(_SPACE_TOP, np.float32)
    bot = np.array(_SPACE_BOTTOM, np.float32)
    grad = top + (bot - top) * np.linspace(0, 1, h)[:, None]      # (h,3)
    img = np.repeat(grad[:, None, :], w, axis=1).copy()           # (h,w,3)
    img += _nebula(w, h, rng)

    # Stars: dense field of tight pinpoints — power-law brightness (mostly
    # faint single pixels, a few brighter), slight colour. No bloom or spikes,
    # like real narrowband sub-frames.
    n = n_stars or max(1500, (w * h) // 850)
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    bright = rng.random(n) ** 3.0                                 # skew faint
    hue = rng.random(n)
    tint = np.ones((n, 3), np.float32)
    tint[hue < 0.22] = (0.72, 0.83, 1.0)                          # blue-white
    tint[hue > 0.90] = (1.0, 0.86, 0.70)                          # warm
    stars = np.zeros((h, w, 3), np.float32)
    np.add.at(stars, (ys, xs), bright[:, None] * 255.0 * tint)

    # Just a whisper of core glow so the brightest points read; faint stars
    # stay as crisp single pixels.
    halo = np.asarray(
        Image.fromarray(np.clip(stars, 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(1.1)), np.float32)
    img += stars + halo * 0.4

    img *= _vignette(w, h)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')


def frost_crop(base, box, tint=(20, 24, 34), tint_alpha=16, blur=2):
    """Return a box-sized frosted-glass tile: a blurred crop of the starfield
    under a light neutral tint. Used as a live panel background so the stars
    read straight through the glass.
    """
    crop = base.crop(box).filter(ImageFilter.GaussianBlur(blur)).convert('RGBA')
    card = Image.alpha_composite(
        crop, Image.new('RGBA', crop.size, (*tint, tint_alpha)))
    return card.convert('RGB')


def _rounded_mask(w, h, radius):
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    return m


def glass_card(base, box, radius=18, tint=(38, 42, 50), tint_alpha=24,
               blur=7, border=(180, 190, 205, 80)):
    """Composite a near-clear glass card into `base` over region `box`.

    Just the glass effect: a soft frost (blurred starfield crop) under a very
    light neutral tint, with a faint hairline edge and rounded corners. No white
    sheen — the starfield reads straight through.
    """
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    crop = base.crop(box).filter(ImageFilter.GaussianBlur(blur)).convert('RGBA')

    card = Image.alpha_composite(crop, Image.new('RGBA', (cw, ch), (*tint, tint_alpha)))
    ImageDraw.Draw(card).rounded_rectangle(
        [0, 0, cw - 1, ch - 1], radius=radius, outline=border, width=1)

    out = base.convert('RGBA')
    out.paste(card, (x0, y0), _rounded_mask(cw, ch, radius))
    return out.convert('RGB')


def make_icon(size=1024):
    """Render the Oeuvre app icon: a glowing ✦ star over a deep-space squircle."""
    S = size
    ss = 2                                   # supersample for smooth edges
    w = S * ss
    img = render_starfield(w, w, seed=3, n_stars=w * w // 1400).convert('RGB')
    arr = np.asarray(img, np.float32)

    # Central 4-point star (✦): outer points N/E/S/W, inner points between.
    cx = cy = w / 2.0
    R, r = w * 0.34, w * 0.34 * 0.30
    pts = []
    for i in range(8):
        ang = np.radians(-90 + i * 45)
        rad = R if i % 2 == 0 else r
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    star = Image.new('L', (w, w), 0)
    ds = ImageDraw.Draw(star)
    ds.polygon(pts, fill=255)
    ds.ellipse([cx - w / 18, cy - w / 18, cx + w / 18, cy + w / 18], fill=255)

    s = np.asarray(star, np.float32) / 255.0
    glow = np.asarray(star.filter(ImageFilter.GaussianBlur(w // 36)),
                      np.float32) / 255.0
    arr = np.clip(arr
                  + glow[:, :, None] * np.array([95, 135, 205], np.float32)
                  + s[:, :, None] * np.array([185, 212, 255], np.float32),
                  0, 255)

    icon = Image.fromarray(arr.astype(np.uint8), 'RGB').convert('RGBA')
    # Squircle mask + subtle edge.
    mask = Image.new('L', (w, w), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, w - 1, w - 1], radius=int(w * 0.22), fill=255)
    icon.putalpha(mask)
    ImageDraw.Draw(icon).rounded_rectangle(
        [0, 0, w - 1, w - 1], radius=int(w * 0.22),
        outline=(150, 180, 225, 90), width=max(1, w // 256))
    return icon.resize((S, S), Image.LANCZOS)


def _demo(path, w=1200, h=780):
    """Render a mock app frame to preview the aesthetic."""
    img = render_starfield(w, h, seed=7)
    pad = 18
    # header + left preview + right controls cards
    img = glass_card(img, (pad, pad, w - pad, 78))
    img = glass_card(img, (pad, 92, int(w * 0.62), h - pad))
    img = glass_card(img, (int(w * 0.62) + 14, 92, w - pad, h - pad))

    d = ImageDraw.Draw(img)
    d.text((pad + 22, 30), "✦  Oeuvre", fill=(180, 210, 255))
    d.text((pad + 130, 33), "SHO Hubble Palette Pipeline", fill=(120, 140, 180))
    d.text((int(w * 0.62) + 36, 112), "Target:  NGC6888", fill=(205, 215, 245))
    d.text((int(w * 0.62) + 36, 150), "[ Run Pipeline ]", fill=(160, 200, 255))
    d.text((pad + 200, h // 2), "preview", fill=(90, 110, 150))
    img.save(path)
    return path


if __name__ == '__main__':
    print(_demo('/tmp/oeuvre_starfield_demo.png'))
