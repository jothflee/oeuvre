#!/usr/bin/env python3
"""
Pure-Python preprocessing — replaces Siril's calibrate/register/stack pipeline.

Reproduces, with numpy/scipy/cv2 only, what the Siril `.ssf` scripts did:
  - master dark      (Siril: `stack dark rej 3 3 -nonorm`)
  - dark calibration (Siril: `calibrate light -dark=master -cc=dark`)
  - 2-pass star registration  (Siril: `register -2pass` + `seqapplyreg`)
  - Winsorized-sigma stacking with addscale normalization
                     (Siril: `stack rej 3 3 -norm=addscale -output_norm -32b`)

Design notes for parity with Siril:
  - Registration uses a 4-DOF similarity model (translate + rotate + uniform
    scale) via cv2.SIFT + estimateAffinePartial2D + RANSAC — the same machinery
    the mosaic path already trusts. Detection runs on a stretched proxy; the
    transform is applied to the LINEAR data with Lanczos interpolation.
  - Frames that fail feature matching fall back to phase-correlation
    translation; any frame whose transform can't be verified (too few stars
    land on the reference) is dropped rather than stacked unregistered.
  - Stacking is NaN-masked: pixels outside a warped frame are excluded per-pixel,
    so dithered frames don't darken the master edges (Siril's COG framing
    equivalent).

The registration primitives here are also reused for SHO channel alignment
(see natural_narrowband.align_channels).
"""

import os

import numpy as np
import cv2
from scipy import ndimage


# ── lazy imports (avoid a circular dependency with natural_narrowband) ────────

def _io():
    from .natural_narrowband import load_fits, save_fits, estimate_background
    return load_fits, save_fits, estimate_background


# ── registration ──────────────────────────────────────────────────────────

def _star_proxy_8bit(linear):
    """Star-emphasized 8-bit proxy (high-pass) for the phase-correlation
    fallback only."""
    x = np.asarray(linear, dtype=np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=2.0, sigmaY=2.0)
    hp = np.clip(x - blur, 0, None)
    p = float(np.percentile(hp, 99.5)) if np.any(hp > 0) else 0.0
    if p > 0:
        hp = hp / p
    return (np.clip(hp, 0, 1) * 255).astype(np.uint8)


def _phasecorr_translation(ref_lin, img_lin, max_shift=200.0):
    """Translation-only last-resort fallback via phase correlation."""
    rf = _star_proxy_8bit(ref_lin).astype(np.float32) / 255.0
    mf = _star_proxy_8bit(img_lin).astype(np.float32) / 255.0
    (dx, dy), resp = cv2.phaseCorrelate(rf, mf)
    if resp < 0.02 or np.hypot(dx, dy) > max_shift:
        return None
    return np.float32([[1, 0, -dx], [0, 1, -dy]])


def _hough_translation(src_xy, ref_xy, max_shift=400.0, bin_px=3.0, min_votes=15):
    """Robust translation estimate via pairwise-offset voting (a 2-D Hough).

    For dense, nebula-dominated fields (e.g. deep Rosette masters) where the
    brightest detections are nebula knots, the asterism triangles are ambiguous
    and phase-correlation locks onto wrong peaks - but the real common stars
    still agree on one offset. Every src star votes its offset to each nearby
    ref star; the dominant bin is the true translation. Needs no consistent
    triangles, just enough common stars, so it survives where the others fail.

    Returns a 2x3 translation matrix (src->ref) or None.
    """
    src_xy = np.asarray(src_xy, dtype=np.float64)
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    if len(src_xy) < min_votes or len(ref_xy) < min_votes:
        return None
    chunks = []
    for s in src_xy:
        d = ref_xy - s
        m = (np.abs(d[:, 0]) < max_shift) & (np.abs(d[:, 1]) < max_shift)
        if m.any():
            chunks.append(d[m])
    if not chunks:
        return None
    offs = np.vstack(chunks)
    nb = max(1, int(np.ceil(2 * max_shift / bin_px)))
    Hh, xe, ye = np.histogram2d(offs[:, 0], offs[:, 1], bins=nb,
                                range=[[-max_shift, max_shift]] * 2)
    ix, iy = np.unravel_index(int(np.argmax(Hh)), Hh.shape)
    cx = 0.5 * (xe[ix] + xe[ix + 1])
    cy = 0.5 * (ye[iy] + ye[iy + 1])
    in_peak = ((np.abs(offs[:, 0] - cx) <= bin_px)
               & (np.abs(offs[:, 1] - cy) <= bin_px))
    votes = int(np.sum(in_peak))
    if votes < min_votes:
        return None
    dx, dy = np.median(offs[in_peak], axis=0)
    return np.float32([[1, 0, dx], [0, 1, dy]])


def _refine_transform(M, src_xy, ref_xy, iters=5):
    """ICP-style sub-pixel polish of a coarse src→ref transform.

    A RANSAC asterism fit (3px inlier threshold) or a phase-correlation shift
    can sit ~1–2px off — invisible to a loose inlier test but enough to smear
    the stacked stars. Re-pair each src star with its nearest reference star and
    re-fit a similarity, tightening the match tolerance each pass, so the
    transform converges toward sub-pixel where the data allows. Frames that
    cannot converge are caught by the residual check in verification.
    """
    from scipy.spatial import cKDTree
    if M is None or len(src_xy) < 6 or len(ref_xy) < 6:
        return M
    M = np.asarray(M, dtype=np.float64).copy()
    src = np.asarray(src_xy, dtype=np.float32)
    ref = np.asarray(ref_xy, dtype=np.float32)
    tree = cKDTree(ref)
    tol = 3.0
    for _ in range(iters):
        pts = np.column_stack([src, np.ones(len(src))])
        proj = (M @ pts.T).T[:, :2]
        dist, idx = tree.query(proj)
        keep = dist < tol
        if int(keep.sum()) < 6:
            break
        Mr, _ = cv2.estimateAffinePartial2D(
            src[keep], ref[idx[keep]], method=cv2.RANSAC,
            ransacReprojThreshold=max(0.5, tol * 0.4))
        if Mr is None:
            break
        M = Mr.astype(np.float64)
        tol = max(0.6, tol * 0.5)
    return M.astype(np.float32)


def _verify_registration(M, src_xy, ref_xy, tol=2.0):
    """Measure how well a transform lands src stars on reference stars.

    Returns ``(n_matched_within_tol, median_residual_px)``. Independent
    confirmation that a transform is both *real* and *tight* — not just that
    some solver returned one. A correct transform lands many common stars at a
    sub-pixel median; a bogus one lands almost none, and a merely-coarse one
    (asterism RANSAC slop, or a frame that genuinely drifted) lands stars but
    with a large median residual that would smear the stack. Callers drop on
    both counts: too few matches, or too loose a median.
    """
    from scipy.spatial import cKDTree
    if M is None or len(src_xy) == 0 or len(ref_xy) == 0:
        return 0, float('inf')
    pts = np.column_stack([src_xy, np.ones(len(src_xy))])
    proj = (np.asarray(M, dtype=np.float64) @ pts.T).T[:, :2]
    dist, _ = cKDTree(np.asarray(ref_xy)).query(proj)
    near = dist[dist < tol]
    if len(near) == 0:
        return 0, float('inf')
    return int(len(near)), float(np.median(near))


def _reject_low_quality(counts, bgs, labels, log, min_keep_frac=0.6,
                        min_keep=5, count_frac=0.5, bg_sigma=3.0):
    """Flag subs that are too poor to help the stack and return kept indices.

    Two cheap, robust signals (both already available from registration):
      - star count far below the median  → cloud / poor transparency,
      - background far above the median   → cloud glow / moonlight.
    Conservative by design: never drops below a keep-floor (the worst are kept
    if too many are flagged), and logs every rejection with its reason.
    """
    n = len(counts)
    counts = np.asarray(counts, dtype=float)
    bgs = np.asarray(bgs, dtype=float)
    cmed = float(np.median(counts))
    bmed = float(np.median(bgs))
    bmad = float(np.median(np.abs(bgs - bmed))) * 1.4826 or 1e-9

    low_stars = counts < count_frac * cmed
    high_bg = bgs > bmed + bg_sigma * bmad
    bad = low_stars | high_bg

    floor = max(min_keep, int(np.ceil(min_keep_frac * n)))
    keep = [i for i in range(n) if not bad[i]]
    if len(keep) < floor:
        # Too many flagged — keep the best `floor` by a combined quality score.
        score = counts / (cmed or 1.0) - (bgs - bmed) / bmad
        keep = sorted(int(i) for i in np.argsort(score)[::-1][:floor])

    keepset = set(keep)
    for i in range(n):
        if i in keepset:
            continue
        why = []
        if low_stars[i]:
            why.append(f"{int(counts[i])} stars (median {int(cmed)})")
        if high_bg[i]:
            why.append(f"bg {bgs[i]:.4f} (median {bmed:.4f})")
        log(f"    REJECT {labels[i]}: {', '.join(why) or 'low quality'}")
    if len(keep) < n:
        log(f"  Sub rejection: kept {len(keep)}/{n} "
            f"(dropped {n - len(keep)} low-quality)")
    return keep


def _star_elongation(frame, xy, n=120, half=11):
    """Median star elongation (major/minor axis ratio) of a frame.

    1.0 = round; larger = trailed/streaked. Measured from the flux second
    moments of the brightest detected stars. Returns 1.0 (treated as clean) when
    too few stars are usable, so a sparse frame is never rejected on shape.
    """
    f = np.asarray(frame, dtype=np.float32)
    H, W = f.shape[:2]
    es = []
    for x, y in xy[:n]:
        xi, yi = int(round(x)), int(round(y))
        if xi < half + 1 or yi < half + 1 or xi >= W - half - 1 or yi >= H - half - 1:
            continue
        p = f[yi - half:yi + half + 1, xi - half:xi + half + 1]
        p = np.clip(p - np.median(p), 0, None)
        s = float(p.sum())
        if s <= 0:
            continue
        ys, xs = np.indices(p.shape)
        cx = (xs * p).sum() / s
        cy = (ys * p).sum() / s
        sxx = ((xs - cx) ** 2 * p).sum() / s
        syy = ((ys - cy) ** 2 * p).sum() / s
        sxy = ((xs - cx) * (ys - cy) * p).sum() / s
        ev = np.linalg.eigvalsh(np.array([[sxx, sxy], [sxy, syy]]))
        ev = np.clip(ev, 1e-6, None)
        es.append(float((ev[1] / ev[0]) ** 0.5))
    return float(np.median(es)) if len(es) >= 10 else 1.0


def _reject_streaked_frames(elongs, labels, log, abs_floor=1.55, sigma=3.0,
                            min_keep_frac=0.7, min_keep=5):
    """Flag subs whose stars are streaked (bad guiding/tracking) and return kept.

    A frame is rejected only if its median star elongation is BOTH above an
    absolute floor (so tight frames are never touched) AND a robust outlier
    above the session median (med + sigma·MAD). The AND is deliberate: a session
    that is *uniformly* mildly trailed has no outliers, so none are dropped
    (culling them would gut the panel) — only genuinely worse frames than their
    own session go. Keep-floor guards against over-rejection.
    """
    n = len(elongs)
    e = np.asarray(elongs, dtype=float)
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med))) * 1.4826 or 1e-9
    bad = (e > abs_floor) & (e > med + sigma * mad)

    floor = max(min_keep, int(np.ceil(min_keep_frac * n)))
    keep = [i for i in range(n) if not bad[i]]
    if len(keep) < floor:
        # Too many flagged — keep the roundest `floor`.
        keep = sorted(int(i) for i in np.argsort(e)[:floor])

    keepset = set(keep)
    for i in range(n):
        if i in keepset:
            continue
        log(f"    REJECT {labels[i]}: streaked stars "
            f"(elongation {e[i]:.2f}, session median {med:.2f})")
    if len(keep) < n:
        log(f"  Streak rejection: kept {len(keep)}/{n} "
            f"(dropped {n - len(keep)} streaked)")
    return keep


def register_frames(frames, log=print, labels=None, ref_idx=None, reject=True):
    """2-pass registration of 2D linear frames to a common reference, using the
    shared star-centroid + asterism matcher (oeuvre.star_match).

    Pass 1: detect stars in every frame; optionally reject low-quality subs
            (cloud/poor-transparency: few stars or high background), then pick
            the frame with the most stars as the reference (unless ref_idx is
            given). Sub rejection is on for auto-reference stacking only.
    Pass 2: match each frame's asterisms to the reference and warp the LINEAR
            data with Lanczos. Fallback is phase-correlation translation; a
            frame whose transform can't be verified against the reference stars
            is dropped (never stacked unregistered).

    Returns (aligned_frames, ref_idx) where ref_idx indexes the returned list
    (which may be shorter than the input if frames were dropped). Pixels
    outside a warped frame are NaN.
    """
    from .star_match import build_star_model, solve_from_models
    from .natural_narrowband import _parallel_map, estimate_background

    n = len(frames)
    if labels is None:
        labels = [f"frame{i+1}" for i in range(n)]

    # Pass 1: detect stars + build asterism models ONCE per frame (parallel).
    # Reused for both reference selection and every pairwise match below, so the
    # expensive detection/high-pass runs once per frame rather than per match.
    models = _parallel_map(build_star_model, frames)
    counts = [len(m[0]) for m in models]

    # Sub-quality rejection (cloud/transparency) before alignment + stacking.
    if reject and ref_idx is None and n >= 4:
        bgs = _parallel_map(lambda fr: estimate_background(fr)[0], frames)
        keep = _reject_low_quality(counts, bgs, labels, log)
        if len(keep) < n:
            frames = [frames[i] for i in keep]
            models = [models[i] for i in keep]
            counts = [counts[i] for i in keep]
            labels = [labels[i] for i in keep]
            n = len(frames)

    # Streak rejection (bad-guiding frames) — drop subs whose stars are
    # outlier-trailed relative to the session, so they don't smear the stack.
    if reject and ref_idx is None and n >= 4:
        elongs = _parallel_map(
            lambda fm: _star_elongation(fm[0], fm[1][0]),
            list(zip(frames, models)))
        keep = _reject_streaked_frames(elongs, labels, log)
        if len(keep) < n:
            frames = [frames[i] for i in keep]
            models = [models[i] for i in keep]
            counts = [counts[i] for i in keep]
            labels = [labels[i] for i in keep]
            n = len(frames)

    if ref_idx is None:
        ref_idx = int(np.argmax(counts))
    h, w = frames[ref_idx].shape[:2]
    log(f"  Registration: reference = {labels[ref_idx]} "
        f"({counts[ref_idx]} stars); {n} frames")

    ref = frames[ref_idx]
    ref_model = models[ref_idx]

    # Pass 2: match each frame to the reference and warp the LINEAR data. Each
    # frame is independent (matched against the fixed reference model, warped on
    # its own output buffer), so run the whole pass in a thread pool. Results and
    # log lines are reassembled in frame order for deterministic output.
    #
    # A frame is registered only if its transform is *verified tight* — enough
    # of its stars land on reference stars AND their median residual is
    # sub-pixel. A frame with no transform, a bogus one, or one that only aligns
    # coarsely (~1px+, e.g. a sub that drifted mid-session) is DROPPED: stacking
    # it would offset/smear its stars and blur every star in the master.
    ref_xy = ref_model[0]
    min_match = max(8, int(0.05 * len(ref_xy)))
    MAX_RESID = 0.8  # px — median star residual above this smears the stack

    def _align(i):
        if i == ref_idx:
            return frames[i].astype(np.float32), None, True
        M, inliers = solve_from_models(models[i], ref_model, log=lambda *a: None)
        method = f"asterism ({inliers} inliers)" if M is not None else None
        if M is None:
            M = _phasecorr_translation(ref, frames[i])
            if M is not None:
                method = "phasecorr"
        if M is not None:
            M = _refine_transform(M, models[i][0], ref_xy)  # sub-pixel polish
        matched, med = _verify_registration(M, models[i][0], ref_xy)
        if M is None or matched < min_match or med > MAX_RESID:
            why = ("no transform found" if M is None
                   else f"unverified ({matched} stars, {med:.2f}px median)")
            return None, f"    DROP {labels[i]}: registration failed — {why}", False
        warped = cv2.warpAffine(
            frames[i].astype(np.float32), M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        return warped, (f"    {labels[i]}: {method}  "
                        f"dx={M[0,2]:.1f} dy={M[1,2]:.1f} "
                        f"rot={ang:.3f}° scale={sc:.4f} "
                        f"verified={matched}@{med:.2f}px"), True

    aligned = []
    new_ref_idx = 0
    dropped = 0
    for i, (frame_out, msg, ok) in enumerate(_parallel_map(_align, range(n))):
        if msg:
            log(msg)
        if not ok:
            dropped += 1
            continue
        if i == ref_idx:
            new_ref_idx = len(aligned)
        aligned.append(frame_out)

    if dropped:
        log(f"  Registration: dropped {dropped} unregisterable frame(s); "
            f"stacking {len(aligned)}")
    return aligned, new_ref_idx


# ── calibration ─────────────────────────────────────────────────────────────

def _sigma_clip_mean(stack, low=3.0, high=3.0, iters=3):
    """Sigma-clipped mean along axis 0 (NaN-aware). stack: (N,H,W)."""
    data = stack.astype(np.float32, copy=True)
    for _ in range(iters):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        lo = mean - low * std
        hi = mean + high * std
        data = np.where((data < lo) | (data > hi), np.nan, data)
    return np.nanmean(data, axis=0)


def build_master_dark(dark_paths, log=print):
    """Sigma-clipped master dark (Siril: stack dark rej 3 3 -nonorm)."""
    load_fits, _, _ = _io()
    log(f"  Master dark: stacking {len(dark_paths)} darks (3σ rejection)")
    stack = np.stack([load_fits(p)[0].astype(np.float32) for p in dark_paths],
                     axis=0)
    master = _sigma_clip_mean(stack, 3.0, 3.0)
    return master.astype(np.float32)


def make_bad_pixel_map(master_dark, sigma=5.0):
    """Hot/cold pixel map from the master dark (Siril's -cc=dark cosmetic).

    Flags pixels deviating from a 3x3 median by more than `sigma` robust σ.
    """
    med = ndimage.median_filter(master_dark, size=3)
    dev = master_dark - med
    mad = float(np.median(np.abs(dev - np.median(dev)))) * 1.4826
    if mad <= 0:
        mad = float(np.std(dev)) or 1e-6
    return np.abs(dev) > sigma * mad


def calibrate_light(light, master_dark, bad_pixel_map):
    """Dark-subtract and cosmetic-correct one light frame."""
    cal = (light.astype(np.float32) - master_dark)
    if bad_pixel_map is not None and np.any(bad_pixel_map):
        med = ndimage.median_filter(cal, size=3)
        cal = np.where(bad_pixel_map, med, cal)
    return cal.astype(np.float32)


# ── stacking ────────────────────────────────────────────────────────────────

def _addscale_normalize(frames, ref_idx, log=print):
    """Additive + multiplicative normalization to the reference (Siril addscale).

    Matches each frame's background pedestal and noise scale to the reference:
        norm = (frame - bg_i) * (sigma_ref / sigma_i) + bg_ref
    NaN-aware (ignores out-of-frame pixels).
    """
    _, _, estimate_background = _io()

    def robust_bg_sigma(a):
        v = a[np.isfinite(a)]
        v = v[v > 0]
        if v.size < 10:
            return 0.0, 1.0
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        return med, (mad if mad > 0 else 1.0)

    bg_ref, sig_ref = robust_bg_sigma(frames[ref_idx])
    out = []
    for i, fr in enumerate(frames):
        if i == ref_idx:
            out.append(fr)
            continue
        bg_i, sig_i = robust_bg_sigma(fr)
        scale = sig_ref / sig_i if sig_i > 0 else 1.0
        out.append((fr - bg_i) * scale + bg_ref)
    return out


def _winsorized_sigma_stack(stack, low=3.0, high=3.0, iters=3):
    """Winsorized sigma-clipped mean along axis 0 (Siril rej 3 3), NaN-aware.

    Winsorizing clamps outliers to the ±kσ boundary (rather than discarding),
    then averages — Siril's default rejection for deep-sky stacks.
    """
    data = stack.astype(np.float32, copy=True)
    for _ in range(iters):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        lo = mean - low * std
        hi = mean + high * std
        # Clamp (winsorize), preserving NaNs.
        data = np.clip(data, lo, hi)
    return np.nanmean(data, axis=0)


def stack_frames(aligned, ref_idx, log=print):
    """Normalize (addscale) + Winsorized-σ reject + mean + output normalize."""
    log(f"  Stacking {len(aligned)} frames (addscale + Winsorized 3σ)")
    norm = _addscale_normalize(aligned, ref_idx, log=log)
    stack = np.stack(norm, axis=0).astype(np.float32)
    master = _winsorized_sigma_stack(stack, 3.0, 3.0)

    # output_norm: scale to [0,1] using a robust range, clip negatives.
    finite = master[np.isfinite(master)]
    if finite.size:
        lo = float(np.nanpercentile(master, 0.01))
        hi = float(np.nanmax(master))
        if hi > lo:
            master = (master - lo) / (hi - lo)
    master = np.nan_to_num(master, nan=0.0)
    return np.clip(master, 0.0, 1.0).astype(np.float32)


# ── orchestration ───────────────────────────────────────────────────────────

def _sum_livetime(light_paths):
    """Total integration time (s) from EXPTIME/EXPOSURE headers."""
    from .mosaic_prep import read_fits_header
    total = 0.0
    for p in light_paths:
        try:
            hdr = read_fits_header(p)
            for k in ('EXPTIME', 'EXPOSURE', 'EXP'):
                if k in hdr:
                    total += float(hdr[k])
                    break
        except Exception:
            pass
    return int(round(total))


def preprocess_filter(light_paths, dark_paths, out_path, log=print,
                      master_dark=None):
    """Full calibrate → register → stack for one filter; writes out_path.

    Returns out_path. If master_dark (array) is provided it is reused (so a
    panel's shared darks are only stacked once).
    """
    load_fits, save_fits, _ = _io()
    log(f"  Preprocessing {len(light_paths)} lights, "
        f"{len(dark_paths)} darks -> {os.path.basename(out_path)}")

    if master_dark is None and dark_paths:
        master_dark = build_master_dark(dark_paths, log=log)
    bpm = make_bad_pixel_map(master_dark) if master_dark is not None else None

    cal_frames, labels = [], []
    for p in light_paths:
        light = load_fits(p)[0].astype(np.float32)
        if master_dark is not None and light.shape == master_dark.shape:
            light = calibrate_light(light, master_dark, bpm)
        cal_frames.append(light)
        labels.append(os.path.basename(p)[:24])

    aligned, ref_idx = register_frames(
        cal_frames, log=log, labels=labels)
    master = stack_frames(aligned, ref_idx, log=log)

    # Save TOP-DOWN (save_fits default); no mirror needed — we own orientation.
    save_fits(master, out_path)
    log(f"  Wrote master: {out_path}  shape={master.shape}")
    return out_path
