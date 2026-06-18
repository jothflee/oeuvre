#!/usr/bin/env python3
"""Compare the SHO colour balance of two result images (crop/size-invariant).

We can't pixel-align two differently-cropped mosaics, so we compare the
*colour distribution* of the nebula signal: where the hues sit (green vs gold
vs teal/blue), how saturated they are, and how neutral the background is — the
things that define the SHO "look". Prints per-image stats and a similarity
score (1.0 = identical colour balance) so a re-processed result can be scored
against a known-good reference.

Usage: python tools/color_compare.py REFERENCE.png CANDIDATE.png
"""
import sys
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Hue bands in degrees [0,360). SHO "good" = mostly GOLD + TEAL/BLUE, little GREEN.
BANDS = {
    'red':   [(0, 20), (340, 360)],
    'gold':  [(20, 65)],          # SII/Ha gold-orange-yellow
    'green': [(65, 150)],         # Ha green — should be SMALL after SCNR
    'teal_blue': [(150, 265)],    # OIII teal→blue
    'magenta': [(265, 340)],      # over-blue/purple — should be SMALL
}


def _load_rgb(path, max_dim=1600):
    im = Image.open(path).convert('RGB')
    s = min(1.0, max_dim / max(im.size))
    if s < 1.0:
        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def _stats(path):
    rgb = _load_rgb(path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H[0,360) S[0,1] V[0,1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    signal = lum > 0.12              # nebula/structure (excludes black sky)
    bg = (lum > 0.005) & (lum < 0.06)  # faint real sky (excludes pure padding)

    sig_s, sig_h = s[signal], h[signal]
    w = sig_s  # weight hue by saturation so near-grey pixels don't vote
    wsum = float(w.sum()) or 1.0
    bands = {}
    for name, ranges in BANDS.items():
        m = np.zeros_like(sig_h, dtype=bool)
        for lo, hi in ranges:
            m |= (sig_h >= lo) & (sig_h < hi)
        bands[name] = float(w[m].sum()) / wsum

    # Saturation-weighted hue histogram (36 bins) for distribution comparison.
    hist, _ = np.histogram(sig_h, bins=36, range=(0, 360), weights=w)
    hist = hist / (hist.sum() or 1.0)

    return {
        'signal_frac': float(np.mean(signal)),
        'mean_sat_signal': float(sig_s.mean()) if sig_s.size else 0.0,
        'mean_sat_bg': float(s[bg].mean()) if bg.any() else 0.0,
        'bands': bands,
        'hist': hist,
    }


def compare(ref_path, cand_path):
    R, C = _stats(ref_path), _stats(cand_path)
    # Hue-distribution similarity (histogram intersection) + saturation match.
    hue_sim = float(np.minimum(R['hist'], C['hist']).sum())
    sat_ratio = (C['mean_sat_signal'] / R['mean_sat_signal']
                 if R['mean_sat_signal'] else 0.0)
    sat_sim = max(0.0, 1.0 - abs(1.0 - sat_ratio))
    # Band-fraction L1 distance → similarity.
    band_l1 = sum(abs(R['bands'][k] - C['bands'][k]) for k in BANDS)
    band_sim = max(0.0, 1.0 - band_l1)
    score = 0.5 * hue_sim + 0.25 * sat_sim + 0.25 * band_sim

    def fmt(d):
        b = d['bands']
        return (f"signal={d['signal_frac']*100:4.1f}%  sat_sig={d['mean_sat_signal']:.3f}  "
                f"sat_bg={d['mean_sat_bg']:.3f}  | gold={b['gold']:.2f} "
                f"teal_blue={b['teal_blue']:.2f} green={b['green']:.2f} "
                f"red={b['red']:.2f} magenta={b['magenta']:.2f}")

    print(f"REF  {fmt(R)}")
    print(f"CAND {fmt(C)}")
    print(f"\nhue_sim={hue_sim:.3f}  sat_sim={sat_sim:.3f} (ratio {sat_ratio:.2f})  "
          f"band_sim={band_sim:.3f}")
    print(f"SCORE = {score:.3f}   (1.0 = identical colour balance)")
    # Hints
    if C['bands']['green'] > R['bands']['green'] + 0.05:
        print("  → CAND has more GREEN than ref (stronger SCNR / hue shift needed)")
    if sat_ratio < 0.85:
        print("  → CAND is LESS saturated (washed out) — raise saturation/contrast")
    if C['mean_sat_bg'] > R['mean_sat_bg'] + 0.03:
        print("  → CAND background has more colour cast (neutralise/flatten harder)")
    if C['bands']['teal_blue'] > R['bands']['teal_blue'] + 0.07:
        print("  → CAND has more TEAL/BLUE than ref (OIII over-weighted)")
    if C['bands']['gold'] < R['bands']['gold'] - 0.07:
        print("  → CAND has less GOLD than ref (SII under-weighted)")
    return score


if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2])
