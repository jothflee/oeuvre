#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Mosaic Panel Grouper for Siril                           ║
║        Groups light frames by telescope pointing for mosaic prep           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Reads RA/DEC from FITS headers of all light frames in a Siril workspace,
clusters them by pointing position, and creates a new directory structure
with symlinks so that each panel group can be independently preprocessed
with Siril's Mono_Preprocessing.ssf script.

Input structure (existing):
    <target>/siril_workspace/<filter>/
        lights/     *.fits raw frames (mixed pointings)
        darks/      dark frames
        flats/      flat frames (optional)
        biases/     bias frames (optional)
        masters/    pre-stacked masters (optional)

Output structure (created):
    <target>/siril_workspace_mosaic/
        panel_1/
            <filter>/
                lights/     -> symlinks to grouped lights
                darks/      -> symlinks to original darks
                flats/      -> symlinks to original flats
                biases/     -> symlinks to original biases
                masters/    -> symlinks to original masters
        panel_2/
            <filter>/
                ...

Usage:
    python mosaic_prep.py <target_dir>
    python mosaic_prep.py IC_1805
    python mosaic_prep.py IC_1805 --no-siril   # only create panel dirs
"""

import os
import sys
import math
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ── FITS Header Reader ──────────────────────────────────────────────────────

def read_fits_header(path):
    """Read FITS header key-value pairs (pure Python, no astropy)."""
    headers = {}
    with open(path, 'rb') as f:
        while True:
            block = f.read(2880)
            if len(block) < 2880:
                break
            for i in range(0, 2880, 80):
                card = block[i:i+80].decode('ascii', errors='replace')
                key = card[:8].strip()
                if key == 'END':
                    return headers
                if '=' in card[8:10]:
                    val = card[10:].strip()
                    # Strip inline comment
                    if '/' in val:
                        val = val[:val.index('/')].strip()
                    val = val.strip("' ")
                    headers[key] = val
    return headers


# ── Clustering ───────────────────────────────────────────────────────────────

def angular_dist_deg(ra1, dec1, ra2, dec2):
    """Great-circle angular distance in degrees."""
    ra1, dec1, ra2, dec2 = [math.radians(x) for x in [ra1, dec1, ra2, dec2]]
    dlat = dec2 - dec1
    dlon = ra2 - ra1
    a = math.sin(dlat/2)**2 + math.cos(dec1)*math.cos(dec2)*math.sin(dlon/2)**2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))


def get_fov_deg(header):
    """Estimate field-of-view in degrees from FITS header."""
    # Try pixel scale + image size
    scale = None
    npix = None
    if 'CDELT1' in header:
        scale = abs(float(header['CDELT1']))  # deg/pixel
    elif 'SECPIX1' in header:
        scale = float(header['SECPIX1']) / 3600.0  # arcsec -> deg
    elif 'SCALE' in header:
        scale = float(header['SCALE']) / 3600.0
    if 'NAXIS1' in header:
        npix = int(float(header['NAXIS1']))
    if scale and npix:
        return scale * npix
    # Fallback: typical narrowband FOV ~ 1.5 deg
    return 1.5


def _pairwise_haversine_deg(ras_deg, decs_deg):
    """Vectorised pairwise great-circle distances in degrees.

    Properly accounts for cos(DEC) compression of RA at high
    declinations, which is critical for correct clustering of
    mosaic panels near the celestial poles.
    """
    ra = np.deg2rad(ras_deg)
    dec = np.deg2rad(decs_deg)
    dlat = dec[:, None] - dec[None, :]
    dlon = ra[:, None] - ra[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(dlon / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.minimum(1.0, np.sqrt(a))))


def cluster_frames(frame_list, radius_deg):
    """Cluster (filename, ra, dec) tuples by pointing proximity.

    Uses complete-linkage hierarchical clustering on the pairwise
    haversine distance matrix.  Complete linkage guarantees that
    every pair of frames within a cluster is within *radius_deg*,
    preventing chain-through merges that occur with DBSCAN when
    a few bridge frames sit between two distinct pointings.

    Args:
        frame_list: list of (filename, ra_deg, dec_deg)
        radius_deg: maximum pairwise angular distance within a cluster

    Returns list of (center_ra, center_dec, [(fname, ra, dec), ...]).
    """
    if not frame_list:
        return []
    if len(frame_list) == 1:
        f = frame_list[0]
        return [(f[1], f[2], [f])]

    ras = np.array([f[1] for f in frame_list])
    decs = np.array([f[2] for f in frame_list])
    dist = _pairwise_haversine_deg(ras, decs)

    # Complete-linkage hierarchical clustering, cutting the dendrogram so every
    # pair within a cluster is within radius_deg (scipy; no scikit-learn dep).
    Z = linkage(squareform(dist, checks=False), method='complete')
    labels = fcluster(Z, t=radius_deg, criterion='distance')

    clusters = []
    for label in sorted(set(labels)):
        mask = labels == label
        members = [frame_list[i] for i in np.where(mask)[0]]
        cra = float(ras[mask].mean())
        cdec = float(decs[mask].mean())
        clusters.append((cra, cdec, members))

    return clusters


def reject_outlier_clusters(clusters, fov_deg, max_dist_fov=3.0):
    """Remove clusters whose center is far from the primary cluster.

    Frames near the celestial pole or at wildly different positions are
    likely from mount parking/homing and should be excluded.

    The primary cluster is chosen as the *most central* — the one with
    the smallest mean angular distance to all other clusters.  This is
    robust even when outlier clusters happen to contain many frames
    (e.g. parked-mount exposures at the pole that all land in one spot).

    Args:
        clusters: list of (center_ra, center_dec, [members])
        fov_deg: field of view in degrees
        max_dist_fov: maximum distance in multiples of FOV to keep

    Returns:
        (kept, rejected) tuple of cluster lists
    """
    if len(clusters) <= 1:
        return clusters, []

    threshold = fov_deg * max_dist_fov

    # If we have a mix of near-pole and non-pole clusters, prefer the
    # non-pole set as the primary candidate.  Pole clusters are commonly
    # mount parking/homing exposures and can otherwise win by accident.
    pole_dec = 85.0
    non_pole_idx = [i for i, c in enumerate(clusters) if abs(c[1]) < pole_dec]

    if non_pole_idx and len(non_pole_idx) < len(clusters):
        candidate_idx = non_pole_idx
    else:
        candidate_idx = list(range(len(clusters)))

    # Primary = most central candidate (smallest mean distance to others).
    # Ties are broken by cluster size to keep this deterministic.
    mean_dists = []
    for i in candidate_idx:
        ci = clusters[i]
        total = sum(
            angular_dist_deg(ci[0], ci[1], clusters[j][0], clusters[j][1])
            for j in candidate_idx if j != i
        )
        denom = max(len(candidate_idx) - 1, 1)
        mean_dists.append(total / denom)

    best_mean = min(mean_dists)
    eps = 1e-9
    tied = [
        idx for idx, md in zip(candidate_idx, mean_dists)
        if abs(md - best_mean) <= eps
    ]
    primary_idx = max(tied, key=lambda idx: len(clusters[idx][2]))
    primary = clusters[primary_idx]
    pra, pdec = primary[0], primary[1]

    kept = []
    rejected = []
    for c in clusters:
        d = angular_dist_deg(c[0], c[1], pra, pdec)
        if d <= threshold:
            kept.append(c)
        else:
            rejected.append(c)

    return kept, rejected


def merge_small_clusters(clusters, min_frames=3):
    """Merge clusters with fewer than min_frames into the nearest larger cluster.

    Small clusters (e.g. 1-2 straggler frames) are typically from brief
    guide-star acquisition or incomplete dithers.  Merging them into
    the nearest viable group is better than creating a panel from them.

    Returns cleaned list of clusters.
    """
    if not clusters:
        return clusters

    large = [c for c in clusters if len(c[2]) >= min_frames]
    small = [c for c in clusters if len(c[2]) < min_frames]

    if not small:
        return clusters
    if not large:
        # All clusters are small — keep them all
        return clusters

    for sc in small:
        sra, sdec, smembers = sc
        # Find nearest large cluster
        best_idx = 0
        best_dist = float('inf')
        for i, (cra, cdec, _) in enumerate(large):
            d = angular_dist_deg(sra, sdec, cra, cdec)
            if d < best_dist:
                best_dist = d
                best_idx = i
        # Merge
        cra, cdec, cmembers = large[best_idx]
        cmembers.extend(smembers)
        n = len(cmembers)
        new_ra = sum(m[1] for m in cmembers) / n
        new_dec = sum(m[2] for m in cmembers) / n
        large[best_idx] = (new_ra, new_dec, cmembers)

    return large


# ── Directory Builder ────────────────────────────────────────────────────────

def scan_fits_by_filter(search_dir, log=print):
    """Scan a directory tree for FITS files and group by FILTER header.

    Searches search_dir and one level of subdirectories for .fits/.fit
    files, reads FILTER/RA/DEC headers, and groups frames by filter.

    Args:
        search_dir: root directory to scan
        log: logging callable

    Returns:
        (filter_frames, fov, filter_exptime, source_lookup) where:
        - filter_frames: {filter_name: [(fname, ra, dec), ...]}
        - fov: estimated field of view in degrees
        - filter_exptime: {filter_name: exptime_seconds}
        - source_lookup: {(filter_name, fname): absolute_path}
    """
    fits_files = []

    # Collect FITS files from search_dir and immediate subdirectories
    for entry in sorted(os.listdir(search_dir)):
        full = os.path.join(search_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(('.fits', '.fit')):
            fits_files.append(full)
        elif os.path.isdir(full) and not entry.startswith('.'):
            for sub in sorted(os.listdir(full)):
                sub_full = os.path.join(full, sub)
                if os.path.isfile(sub_full) and sub.lower().endswith(('.fits', '.fit')):
                    fits_files.append(sub_full)

    if not fits_files:
        return {}, 1.5, {}, {}

    filter_frames = {}    # filter_name -> [(fname, ra, dec)]
    filter_exptime = {}   # filter_name -> seconds
    source_lookup = {}    # (filter_name, fname) -> absolute_path
    fov = 1.5

    for path in fits_files:
        h = read_fits_header(path)

        filt = h.get('FILTER', '').strip()
        if not filt:
            continue  # silently skip files without FILTER header

        # Get RA/DEC
        ra = dec = None
        for rk in ('RA', 'CRVAL1'):
            if rk in h:
                try:
                    ra = float(h[rk])
                    break
                except ValueError:
                    pass
        for dk in ('DEC', 'CRVAL2'):
            if dk in h:
                try:
                    dec = float(h[dk])
                    break
                except ValueError:
                    pass

        if ra is None or dec is None:
            log(f"  WARNING: No RA/DEC in {os.path.basename(path)}, skipping")
            continue

        # Handle duplicate basenames from different subdirectories
        basename = os.path.basename(path)
        name_key = (filt, basename)
        if name_key in source_lookup:
            parent = os.path.basename(os.path.dirname(path))
            basename = f"{parent}_{basename}"
            name_key = (filt, basename)

        filter_frames.setdefault(filt, []).append((basename, ra, dec))
        source_lookup[name_key] = os.path.abspath(path)

        # Get FOV and exposure from first frame of each filter
        if filt not in filter_exptime:
            fov = get_fov_deg(h)
            for ek in ('EXPTIME', 'EXPOSURE'):
                if ek in h:
                    try:
                        filter_exptime[filt] = float(h[ek])
                        break
                    except ValueError:
                        pass
            if filt not in filter_exptime:
                filter_exptime[filt] = 0

    return filter_frames, fov, filter_exptime, source_lookup


def scan_filter_dir(filter_dir):
    """Scan a filter directory and return (frames, fov_deg, exptime).

    frames: list of (filename, ra, dec) for each light frame
    fov_deg: estimated field of view from first frame
    exptime: exposure time per frame in seconds (from EXPTIME header)

    Handles two layouts:
      - filter_dir/lights/*.fits   (siril workspace layout)
      - filter_dir/*.fits          (raw Light/ layout)
    """
    lights_dir = os.path.join(filter_dir, 'lights')
    if not os.path.isdir(lights_dir):
        # Fall back to scanning the dir itself
        lights_dir = filter_dir

    files = sorted([
        f for f in os.listdir(lights_dir)
        if f.lower().endswith(('.fits', '.fit'))
    ])
    if not files:
        return [], 1.5, 0

    frames = []
    fov = 1.5
    exptime = 0

    for i, fname in enumerate(files):
        path = os.path.join(lights_dir, fname)
        h = read_fits_header(path)

        # Get RA/DEC (try multiple header keys)
        ra = dec = None
        for rk in ('RA', 'CRVAL1'):
            if rk in h:
                try:
                    ra = float(h[rk])
                    break
                except ValueError:
                    pass
        for dk in ('DEC', 'CRVAL2'):
            if dk in h:
                try:
                    dec = float(h[dk])
                    break
                except ValueError:
                    pass

        if ra is None or dec is None:
            print(f"  WARNING: No RA/DEC in {fname}, skipping")
            continue

        frames.append((fname, ra, dec))

        # Get FOV and exposure from first frame
        if i == 0:
            fov = get_fov_deg(h)
            for ek in ('EXPTIME', 'EXPOSURE'):
                if ek in h:
                    try:
                        exptime = float(h[ek])
                        break
                    except ValueError:
                        pass

    return frames, fov, exptime


def symlink_dir_contents(src_dir, dst_dir):
    """Create symlinks in dst_dir pointing to all files in src_dir."""
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            dst = os.path.join(dst_dir, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
                count += 1
    return count


def build_panel_dirs(target_dir, cluster_radius_deg=0.15, output_suffix='_mosaic',
                     darks_dir=None):
    """Build the panel directory structure for mosaic preprocessing.

    Clusters light frames by pointing position and creates
    <target>/siril_workspace_mosaic/panel_N/<filter>/lights/
    with symlinks to the original FITS files.

    Supports two input layouts:
      1. Structured: <target>/Light/<filter>/*.fits
         Filters are inferred from subdirectory names.
      2. Flexible:   <target>/*.fits  (or one level of subdirs)
         Filters are read from each file's FILTER FITS header.

    Panels are matched across filters by spatial proximity — each panel
    must have data from ALL filters.  Panels missing any filter are
    discarded with a warning.

    Args:
        target_dir: path to target (e.g. 'IC_1805')
        cluster_radius_deg: cluster radius in degrees (default 0.15°)
        output_suffix: suffix for the output workspace dir
        darks_dir: path to shared darks directory (optional)

    Returns:
        (panel_map, out_workspace, stats) where panel_map is
        {panel_name: {filter: panel_filter_dir}} and stats is
        {filter: (n_frames, total_seconds)}
    """
    light_dir = os.path.join(target_dir, 'Light')
    flexible_mode = not os.path.isdir(light_dir)

    # source_lookup: (filter_name, fname) -> absolute source path
    source_lookup = {}
    filter_clusters = {}
    filter_exptime = {}
    fov = 1.5

    if flexible_mode:
        # ── Flexible mode: auto-detect filters from FITS headers ─────
        filter_frames, fov, filter_exptime, source_lookup = \
            scan_fits_by_filter(target_dir)
        if not filter_frames:
            print(f"ERROR: No FITS files with FILTER/RA/DEC headers in "
                  f"{target_dir}")
            sys.exit(1)

        filters = sorted(filter_frames.keys())
        print(f"Auto-detected filters: {', '.join(filters)}")

        for filt in filters:
            frames = filter_frames[filt]
            if not frames:
                continue

            clusters = cluster_frames(frames, cluster_radius_deg)
            kept, rejected = reject_outlier_clusters(clusters, fov)
            if rejected:
                for rc in rejected:
                    print(f"\n  {filt}: EXCLUDED {len(rc[2])} frames at "
                          f"({rc[0]:.1f}, {rc[1]:.1f}) — outlier")
                clusters = kept

            before_merge = len(clusters)
            clusters = merge_small_clusters(clusters, min_frames=3)
            if len(clusters) < before_merge:
                merged_count = before_merge - len(clusters)
                print(f"  {filt}: merged {merged_count} tiny group(s) "
                      f"into nearest panels")

            filter_clusters[filt] = clusters
            print(f"\n  {filt}: {len(frames)} frames, FOV={fov:.2f}, "
                  f"radius={cluster_radius_deg:.2f} → "
                  f"{len(clusters)} group(s)")
            for i, (cra, cdec, members) in enumerate(clusters):
                print(f"    Group {i+1}: ({cra:.3f}, {cdec:.3f}) "
                      f"[{len(members)} frames]")
    else:
        # ── Structured mode: Light/<filter>/ directories ─────────────
        filters = sorted([
            d for d in os.listdir(light_dir)
            if os.path.isdir(os.path.join(light_dir, d))
            and not d.startswith('.')
        ])
        print(f"Found filters: {', '.join(filters)}")

        for filt in filters:
            filt_dir = os.path.join(light_dir, filt)
            frames, filt_fov, exptime = scan_filter_dir(filt_dir)
            fov = filt_fov
            if not frames:
                print(f"  {filt}: no light frames with RA/DEC, skipping")
                continue

            filter_exptime[filt] = exptime

            # Build source_lookup for this filter's frames
            scan_dir = os.path.join(filt_dir, 'lights')
            if not os.path.isdir(scan_dir):
                scan_dir = filt_dir
            for fname, ra, dec in frames:
                source_lookup[(filt, fname)] = \
                    os.path.abspath(os.path.join(scan_dir, fname))

            radius = cluster_radius_deg
            clusters = cluster_frames(frames, radius)

            kept, rejected = reject_outlier_clusters(clusters, fov)
            if rejected:
                for rc in rejected:
                    print(f"\n  {filt}: EXCLUDED {len(rc[2])} frames at "
                          f"({rc[0]:.1f}, {rc[1]:.1f}) — "
                          f"outlier (>{fov * 3:.1f} from primary)")
                clusters = kept

            before_merge = len(clusters)
            clusters = merge_small_clusters(clusters, min_frames=3)
            if len(clusters) < before_merge:
                merged_count = before_merge - len(clusters)
                print(f"  {filt}: merged {merged_count} tiny group(s) "
                      f"into nearest panels")

            filter_clusters[filt] = clusters

            print(f"\n  {filt}: {len(frames)} frames, FOV={fov:.2f}, "
                  f"radius={radius:.2f} → {len(clusters)} group(s)")
            for i, (cra, cdec, members) in enumerate(clusters):
                print(f"    Group {i+1}: ({cra:.3f}, {cdec:.3f}) "
                      f"[{len(members)} frames]")

    if not filter_clusters:
        print("ERROR: No frames found in any filter")
        sys.exit(1)

    # ── Match panels across filters by spatial proximity ─────────────
    # Build master panel positions from the filter with the most clusters
    # (typically Ha), then match each other filter's clusters to them.
    ref_filt = max(filter_clusters, key=lambda f: len(filter_clusters[f]))
    master_panels = []  # [(ra, dec)] — one per panel
    for cra, cdec, _ in filter_clusters[ref_filt]:
        master_panels.append((cra, cdec))

    # For each filter, use the Hungarian algorithm to optimally assign
    # its clusters to master panels (minimising total angular distance).
    match_threshold = fov * 0.5  # clusters must be within half FOV to match
    panel_assignments = [{} for _ in master_panels]  # [{filter: cluster_index}]

    for filt, clusters in filter_clusters.items():
        n_panels_m = len(master_panels)
        n_clusters = len(clusters)
        if n_clusters == 0:
            continue

        # Build cost matrix (panels × clusters), using angular distance.
        # Entries beyond the match threshold are set to a large sentinel
        # so the Hungarian algorithm avoids them.
        BIG = 1e6
        cost = np.full((n_panels_m, n_clusters), BIG)
        for pi, (pra, pdec) in enumerate(master_panels):
            for ci, (cra, cdec, _) in enumerate(clusters):
                d = angular_dist_deg(pra, pdec, cra, cdec)
                if d <= match_threshold:
                    cost[pi, ci] = d

        # Solve optimal assignment (handles rectangular matrices)
        row_ind, col_ind = linear_sum_assignment(cost)
        for pi, ci in zip(row_ind, col_ind):
            if cost[pi, ci] < BIG:
                panel_assignments[pi][filt] = ci

    # Keep only panels where ALL filters matched — drop incomplete panels
    required_filters = set(filter_clusters.keys())
    complete_panels = []
    for pi, assignment in enumerate(panel_assignments):
        matched = set(assignment.keys())
        missing = required_filters - matched
        if not missing:
            complete_panels.append(assignment)
        else:
            pra, pdec = master_panels[pi]
            print(f"\n  Dropping panel at ({pra:.3f}, {pdec:.3f}) "
                  f"— missing {', '.join(sorted(missing))}")

    n_panels = len(complete_panels)
    print(f"\n  Total panels: {n_panels} "
          f"(all {len(required_filters)} filters present)")

    if n_panels == 0:
        print("ERROR: No panels have data from all filters")
        sys.exit(1)

    # Create output structure
    out_workspace = os.path.join(target_dir, f'siril_workspace{output_suffix}')
    os.makedirs(out_workspace, exist_ok=True)

    panel_map = {}

    for panel_idx, assignment in enumerate(complete_panels):
        panel_name = f'panel_{panel_idx + 1}'
        panel_dir = os.path.join(out_workspace, panel_name)
        panel_map[panel_name] = {}

        print(f"\n{'='*60}")
        print(f"  Building {panel_name}")
        print(f"{'='*60}")

        for filt in filters:
            if filt not in filter_clusters:
                continue

            ci = assignment[filt]
            cluster = filter_clusters[filt][ci]
            cra, cdec, members = cluster

            filt_dst = os.path.join(panel_dir, filt)
            os.makedirs(filt_dst, exist_ok=True)
            panel_map[panel_name][filt] = filt_dst

            # Symlink lights for this group
            lights_dst = os.path.join(filt_dst, 'lights')
            os.makedirs(lights_dst, exist_ok=True)

            for fname, _, _ in members:
                src = source_lookup[(filt, fname)]
                dst = os.path.join(lights_dst, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
            print(f"  {filt}: {len(members)} lights "
                  f"@ ({cra:.3f}, {cdec:.3f})")

            # Symlink darks
            if darks_dir and os.path.isdir(darks_dir):
                darks_dst = os.path.join(filt_dst, 'darks')
                n = symlink_dir_contents(darks_dir, darks_dst)
                if n > 0:
                    print(f"  {filt}: {n} darks linked")

    print(f"\n{'='*60}")
    print(f"  Panel structure created in: {out_workspace}")
    print(f"{'='*60}")

    # Compute integration stats per filter (used frames only)
    stats = {}  # filter -> (n_frames, total_seconds)
    used_indices = {}  # filter -> set of cluster indices used
    for assignment in complete_panels:
        for filt, ci in assignment.items():
            used_indices.setdefault(filt, set()).add(ci)

    for filt, clusters in filter_clusters.items():
        n_frames = sum(
            len(clusters[ci][2]) for ci in used_indices.get(filt, set())
        )
        exp = filter_exptime.get(filt, 0)
        stats[filt] = (n_frames, n_frames * exp)

    total_frames = sum(s[0] for s in stats.values())
    total_secs = sum(s[1] for s in stats.values())
    total_mins = total_secs / 60

    print(f"\n  Integration summary:")
    for filt in filters:
        if filt in stats:
            nf, ts = stats[filt]
            print(f"    {filt:>10}: {nf:3d} frames x "
                  f"{filter_exptime.get(filt, 0):.0f}s = "
                  f"{ts/60:.0f} min")
    print(f"    {'Total':>10}: {total_frames} frames, "
          f"{total_mins:.0f} min ({total_mins/60:.1f} hr)")

    return panel_map, out_workspace, stats


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Group FITS light frames by pointing for mosaic preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mosaic_prep.py IC_1805
  python mosaic_prep.py IC_1805 --radius 0.4
  python mosaic_prep.py IC_1805 --no-preprocess
        """)

    parser.add_argument('target_dir',
                        help='Target directory containing Light/<filter>/')
    parser.add_argument('--radius', type=float, default=0.15,
                        help='Cluster radius in degrees (default: 0.15)')
    parser.add_argument('--output-suffix', default='_mosaic',
                        help='Suffix for output workspace dir (default: _mosaic)')
    parser.add_argument('--darks-dir', default=None,
                        help='Path to shared darks directory')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Skip preprocessing (only create panel dirs)')

    args = parser.parse_args()

    target = os.path.abspath(args.target_dir)
    if not os.path.isdir(target):
        print(f"ERROR: Directory not found: {target}")
        sys.exit(1)

    print(f"Target: {target}")
    print(f"Cluster radius: {args.radius:.2f} deg\n")

    panel_map, out_workspace, stats = build_panel_dirs(
        target,
        cluster_radius_deg=args.radius,
        output_suffix=args.output_suffix,
        darks_dir=args.darks_dir,
    )

    # Run pure-Python preprocessing on each panel (unless --no-preprocess)
    if not args.no_preprocess:
        from . import preprocess
        import glob as _glob

        for panel_name in sorted(panel_map.keys()):
            print(f"\n{'='*60}")
            print(f"  Preprocessing {panel_name}...")
            print(f"{'='*60}")
            for filt, filt_dir in sorted(panel_map[panel_name].items()):
                lights = sorted(
                    _glob.glob(os.path.join(filt_dir, 'lights', '*.fit')) +
                    _glob.glob(os.path.join(filt_dir, 'lights', '*.fits')))
                darks = sorted(
                    _glob.glob(os.path.join(filt_dir, 'darks', '*.fit')) +
                    _glob.glob(os.path.join(filt_dir, 'darks', '*.fits')))
                if not lights:
                    print(f"  WARNING: {filt}: no lights — skipping")
                    continue
                lt = preprocess._sum_livetime(lights)
                out_path = os.path.join(filt_dir, f"result_{filt}_{lt}s.fit")
                preprocess.preprocess_filter(lights, darks, out_path)
                print(f"  OK: {os.path.relpath(out_path, out_workspace)}")

        print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
