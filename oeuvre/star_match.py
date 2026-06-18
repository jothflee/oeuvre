#!/usr/bin/env python3
"""
Shared star-centroid + asterism (triangle) matcher.

This is the single registration primitive used everywhere stars must be matched
between two frames:
  - per-filter sub stacking          (preprocess.register_frames)
  - SHO channel alignment            (natural_narrowband.align_channels)
  - mosaic panel alignment           (natural_narrowband.compute_mosaic_geometry)

Why asterisms instead of SIFT: on star fields every star is a near-identical
blob, so SIFT descriptors are non-distinctive and most matches get rejected.
Triangle invariants built from star *geometry* are highly distinctive and exactly
invariant to translation, rotation, and scale — the approach used by
astrometry.net / Siril / astroalign. We implement it with numpy/scipy/cv2 only
(no new dependencies), feeding the matched correspondences to the same
cv2.estimateAffinePartial2D + RANSAC solver used before.

Public API:
    detect_stars(image, max_stars) -> (xy [N,2], flux [N])
    match_and_solve(src_img, dst_img, log) -> (M 2x3, n_inliers) or (None, 0)
"""

import numpy as np
import cv2
from scipy.spatial import cKDTree


# Diffuse-nebula suppression for star detection. Subtracting a large-kernel
# blur removes the smooth emission-nebula background, whose bright gradients
# otherwise inflate the MAD background sigma (starving the threshold) and seed
# false centroids from nebula texture. Standard practice for star detection on
# emission-line fields; on real IC 1805 panels it is the difference between a
# nebula-heavy panel detecting 9 stars vs 200+, and is what lets both the
# mosaic matcher and the SHO channel aligner lock onto real stars.
_HIGHPASS_SIGMA = 8.0


def _bg_sigma(img):
    """Robust background level and noise sigma (MAD-based)."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, (mad if mad > 0 else (float(v.std()) or 1e-6))


def detect_stars(image, max_stars=200, thresh_sigma=5.0,
                 min_area=2, max_area=120, highpass_sigma=_HIGHPASS_SIGMA):
    """Detect stars and return flux-weighted sub-pixel centroids.

    Works on linear subs or stretched luminance alike (threshold is relative to
    background + k·sigma). Compact-source area filtering rejects nebula blobs.

    A high-pass prefilter (``highpass_sigma`` > 0) subtracts a large-kernel
    Gaussian blur first, removing diffuse nebula so the threshold tracks the
    star noise rather than the nebula gradient. Pass ``highpass_sigma=0`` to
    detect on the raw image.

    Returns (xy [N,2] float64 as (x,y), flux [N]) sorted brightest-first.
    """
    img = np.nan_to_num(np.asarray(image, dtype=np.float32))
    if highpass_sigma and highpass_sigma > 0:
        blur = cv2.GaussianBlur(img, (0, 0),
                                sigmaX=float(highpass_sigma),
                                sigmaY=float(highpass_sigma))
        img = img - blur
    bg, sig = _bg_sigma(img)
    mask = img > (bg + thresh_sigma * sig)
    if not mask.any():
        return np.empty((0, 2)), np.empty((0,))

    n, lbl = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.empty((0, 2)), np.empty((0,))

    flat = lbl.ravel()
    w = np.clip(img - bg, 0, None).ravel()
    ys, xs = np.indices(img.shape)
    area = np.bincount(flat, minlength=n)
    fsum = np.bincount(flat, weights=w, minlength=n)
    wx = np.bincount(flat, weights=(xs.ravel() * w), minlength=n)
    wy = np.bincount(flat, weights=(ys.ravel() * w), minlength=n)

    # Drop background label 0; guard against zero-flux labels.
    area, fsum, wx, wy = area[1:], fsum[1:], wx[1:], wy[1:]
    valid = (area >= min_area) & (area <= max_area) & (fsum > 0)
    if not valid.any():
        return np.empty((0, 2)), np.empty((0,))
    cx = wx[valid] / fsum[valid]
    cy = wy[valid] / fsum[valid]
    flux = fsum[valid]

    order = np.argsort(flux)[::-1][:max_stars]
    xy = np.column_stack([cx[order], cy[order]]).astype(np.float64)
    return xy, flux[order]


def _build_invariants(xy, n_neighbors=6, max_elong=10.0):
    """Build triangle invariants from each star and its nearest neighbors.

    Returns (inv [M,2], tris [M,3]) where each tri lists vertex indices ordered
    canonically by ascending opposite-side length, so matched triangles yield
    direct point correspondences.
    """
    n = len(xy)
    if n < 3:
        return np.empty((0, 2)), []
    tree = cKDTree(xy)
    k = min(n_neighbors + 1, n)
    invs, tris, seen = [], [], set()
    for i in range(n):
        _, idxs = tree.query(xy[i], k=k)
        nbrs = [int(j) for j in np.atleast_1d(idxs) if j != i][:n_neighbors]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                tri = tuple(sorted((i, nbrs[a], nbrs[b])))
                if tri in seen:
                    continue
                seen.add(tri)
                p = xy[list(tri)]
                d01 = np.linalg.norm(p[0] - p[1])
                d12 = np.linalg.norm(p[1] - p[2])
                d20 = np.linalg.norm(p[2] - p[0])
                opp = np.array([d12, d20, d01])  # side opposite each vertex
                L = np.sort([d01, d12, d20])
                if L[0] < 1e-3 or L[2] / L[0] > max_elong:
                    continue
                invs.append((L[2] / L[1], L[1] / L[0]))
                order = np.argsort(opp)  # vertices by ascending opposite side
                tris.append(tuple(np.array(tri)[order]))
    return np.array(invs), tris


def build_star_model(image, max_stars=200):
    """Detect stars and precompute their asterism invariants for an image.

    Returns an opaque model (xy, invariants, triangles) reusable across many
    pairwise matches — detection (incl. the high-pass blur) and invariant
    construction are the expensive parts, so callers that match one image
    against many others should build the model once via this and feed it to
    solve_from_models, rather than calling match_and_solve repeatedly (which
    re-detects both images every time).
    """
    xy, _ = detect_stars(image, max_stars=max_stars)
    inv, tri = _build_invariants(xy)
    return (xy, inv, tri)


def solve_from_models(src_model, dst_model, log=print, min_inliers=8,
                      inv_tol=0.05):
    """Solve a 4-DOF similarity (translate + rotate + uniform scale) from two
    precomputed star models (see build_star_model).

    Returns (M 2x3 mapping src->dst, n_inliers), or (None, 0) on failure.
    """
    src_xy, inv_s, tri_s = src_model
    dst_xy, inv_d, tri_d = dst_model
    if len(src_xy) < 3 or len(dst_xy) < 3:
        return None, 0
    if len(inv_s) == 0 or len(inv_d) == 0:
        return None, 0

    tree = cKDTree(inv_d)
    src_pts, dst_pts = [], []
    for ks, inv in enumerate(inv_s):
        dist, jd = tree.query(inv)
        if dist < inv_tol:
            for a in range(3):
                src_pts.append(src_xy[tri_s[ks][a]])
                dst_pts.append(dst_xy[tri_d[jd][a]])
    if len(src_pts) < min_inliers:
        return None, 0

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if M is None or inliers < min_inliers:
        return None, 0

    # Reject degenerate/collapsed similarities: RANSAC can occasionally lock
    # onto a coincidental cluster of false matches and return a scale≈0 (or
    # wildly large) transform. Same-instrument frames are always ~unit scale.
    if not np.all(np.isfinite(M)):
        return None, 0
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.8 < scale < 1.25):
        log(f"    rejected degenerate transform (scale={scale:.4f})")
        return None, 0
    return M, inliers


def match_and_solve(src_img, dst_img, log=print, min_inliers=8,
                    inv_tol=0.05, max_stars=200):
    """Match stars between two images via asterism invariants and solve a
    4-DOF similarity transform (translate + rotate + uniform scale).

    Convenience wrapper that builds both star models then solves. For matching
    one image against many, build models once and call solve_from_models.

    Returns (M 2x3 mapping src->dst, n_inliers), or (None, 0) on failure.
    """
    return solve_from_models(
        build_star_model(src_img, max_stars=max_stars),
        build_star_model(dst_img, max_stars=max_stars),
        log=log, min_inliers=min_inliers, inv_tol=inv_tol)
