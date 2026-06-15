#!/usr/bin/env python3
"""
Pipeline orchestrator — extracted from process.py.

Runs the full SHO Narrowband Processing Pipeline:
  1. Scan Light/<filter>/ frames, cluster by RA/DEC pointing
  2. Create panel dirs with symlinked lights + darks
  3. Pure-Python preprocessing (calibrate / register / stack) per panel/filter
  4. Mosaic panel results + SHO Hubble palette processing
"""

import os
import glob
import time
import shutil
from dataclasses import dataclass

from . import config
from .mosaic_prep import build_panel_dirs
from .natural_narrowband import (
    SHOPipeline, STRETCH_TARGET, SCNR_AMOUNT, STAR_DESAT, SAT_BOOST,
)

# Workspace (data root) is resolved at call time from OEUVRE_WORKSPACE / cwd —
# see oeuvre.config. The code no longer assumes it lives inside the data dir.


# ── Config dataclass ────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tuneable parameters for a pipeline run."""
    target: str = ''
    cluster_radius: float = 0.15
    no_preprocess: bool = False
    output_dir: str = None
    stretch_target: float = STRETCH_TARGET
    scnr_amount: float = SCNR_AMOUNT
    star_desat: float = STAR_DESAT
    sat_boost: float = SAT_BOOST
    no_preview: bool = False
    interactive: bool = False
    recolor_only: bool = False
    clear_cache: bool = False
    flatten_background: bool = False
    hue_strength: float = 0.40
    oiii_factor: float = 0.32
    truthful_mode: bool = False
    hubbleize: bool = True
    hubbleize_strength: float = 0.45
    star_consensus: str = 'auto'
    log_callback: object = None  # callable(str) for GUI log sink
    preview_object: object = None  # optional TkPreview (or compatible) instance

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_filter(name):
    """Map a filter directory name (e.g. 'Ha-7nm') to a channel key."""
    up = name.upper()
    if 'SII' in up or 'S2' in up:
        return 'sii'
    if 'HA' in up:
        return 'ha'
    if 'OIII' in up or 'O3' in up:
        return 'oiii'
    return None


def _gather_frames(subdir):
    """Sorted FITS files inside a lights/ or darks/ subdirectory."""
    out = []
    for pattern in ('*.fit', '*.fits', '*.fts'):
        out += glob.glob(os.path.join(subdir, pattern))
    return sorted(out)


def has_results(filt_dir):
    """True if the filter directory already contains result_*.fit."""
    return len(glob.glob(os.path.join(filt_dir, 'result_*.fit'))) > 0


def glob_result_files(filt_dir):
    """Return sorted list of result FITS files in a filter directory."""
    for pattern in ('result_*.fit', 'result_*.fits', '*.fit', '*.fits'):
        hits = sorted(glob.glob(os.path.join(filt_dir, pattern)))
        if hits:
            return hits
    return []


# ── Preprocessing runner (pure Python) ───────────────────────────────────────

def _run_preprocess(panel_map, out_workspace, cfg):
    """Pure-Python calibrate/register/stack for panels lacking results.

    Mirrors the old Siril runner's contract: for each panel/filter it reads
    lights/ and darks/ and writes result_<FILTER>_<LIVETIME>s.fit into the
    filter directory. Master darks are cached per unique dark set within a run.
    """
    from . import preprocess

    dark_cache = {}  # frozenset(dark_paths) -> master dark array

    for panel_name in sorted(panel_map.keys()):
        filters = panel_map[panel_name]
        if all(has_results(d) for d in filters.values()):
            cfg.log(f"\n  {panel_name}: all result files exist — skipping")
            continue

        cfg.log(f"\n{'=' * 60}")
        cfg.log(f"  Preprocessing {panel_name}...")
        cfg.log(f"{'=' * 60}")

        for filt_name, filt_dir in sorted(filters.items()):
            if has_results(filt_dir):
                cfg.log(f"  {filt_name}: result exists — skipping")
                continue

            lights = _gather_frames(os.path.join(filt_dir, 'lights'))
            darks = _gather_frames(os.path.join(filt_dir, 'darks'))
            if not lights:
                cfg.log(f"  WARNING: {filt_name}: no lights found — skipping")
                continue

            # Reuse a master dark across filters sharing the same dark set.
            master_dark = None
            if darks:
                key = frozenset(os.path.realpath(d) for d in darks)
                if key not in dark_cache:
                    dark_cache[key] = preprocess.build_master_dark(
                        darks, log=cfg.log)
                master_dark = dark_cache[key]

            livetime = preprocess._sum_livetime(lights)
            out_path = os.path.join(
                filt_dir, f"result_{filt_name}_{livetime}s.fit")
            cfg.log(f"\n  → {filt_name}  ({len(lights)} lights, "
                    f"{len(darks)} darks, {livetime}s)")
            try:
                preprocess.preprocess_filter(
                    lights, darks, out_path, log=cfg.log,
                    master_dark=master_dark)
            except Exception as e:
                cfg.log(f"  ERROR preprocessing {filt_name}: {e}")

    # Summary
    cfg.log(f"\n  Preprocessing results:")
    for panel_name in sorted(panel_map.keys()):
        for filt_name, filt_dir in sorted(panel_map[panel_name].items()):
            files = glob.glob(os.path.join(filt_dir, 'result_*.fit'))
            status = f"{len(files)} file(s)" if files else "MISSING"
            cfg.log(f"    {panel_name}/{filt_name}: {status}")


# ── Channel collection ───────────────────────────────────────────────────────

def _collect_channels(panel_map, cfg):
    """Collect result FITS file paths grouped by channel across all panels."""
    channels = {'sii': [], 'ha': [], 'oiii': []}

    for panel_name in sorted(panel_map.keys()):
        for filt_name, filt_dir in panel_map[panel_name].items():
            ch = classify_filter(filt_name)
            if not ch:
                cfg.log(f"  WARNING: Cannot classify filter '{filt_name}' — skipping")
                continue
            files = glob_result_files(filt_dir)
            if files:
                channels[ch].extend(files)
            else:
                cfg.log(f"  WARNING: No result files in {filt_dir}")

    return channels


def _auto_star_consensus_from_stats(stats):
    """Choose star consensus strictness from available data quality.

    As channel depth/balance improves, remove fewer stars:
      - severe imbalance or very low weak-channel depth -> strict
      - moderate imbalance/depth -> soft
      - strong, balanced data -> off
    """
    if not stats:
        return 'soft'

    counts = [int(v[0]) for v in stats.values() if int(v[0]) > 0]
    if not counts:
        return 'soft'

    weak = min(counts)
    strong = max(counts)
    ratio = weak / max(strong, 1)

    if weak < 10 or ratio < 0.35:
        return 'strict'
    if weak < 24 or ratio < 0.65:
        return 'soft'
    return 'off'


# ── Target scanning ──────────────────────────────────────────────────────────

def scan_targets(base_dir=None):
    """Find all target directories that have processable FITS data.

    Discovers targets in two modes:
      1. Structured: directories containing Light/<filter>/ subdirectories
      2. Flexible:   directories containing .fits/.fit files directly
         (filters will be auto-detected from FITS FILTER headers)

    Returns list of (name, path, filters) sorted by name.
    filters is a list of filter names (from subdirs or FITS headers);
    an empty list means filters will be auto-detected at processing time.
    """
    if base_dir is None:
        base_dir = config.ensure_workspace()

    # Directories to skip (internal / non-target)
    skip_dirs = {
        'oeuvre', '__pycache__', '.venv', '.git', '.claude', 'darks', 'siril',
        'scripts', 'tools', 'misses', 'DeepSkyWorkflowScripts',
        'StarNetv2CLI_MacOS', 'Sequences',
        'astrometry.net', 'astrometry_indexes', '_solve_tmp',
    }

    targets = []
    for entry in sorted(os.listdir(base_dir)):
        if entry in skip_dirs or entry.startswith('.'):
            continue
        full = os.path.join(base_dir, entry)
        if not os.path.isdir(full):
            continue

        # Mode 1: Structured Light/<filter>/ layout
        light_dir = os.path.join(full, 'Light')
        if os.path.isdir(light_dir):
            filters = sorted([
                d for d in os.listdir(light_dir)
                if os.path.isdir(os.path.join(light_dir, d))
                and not d.startswith('.')
            ])
            if filters:
                targets.append((entry, full, filters))
                continue

        # Mode 2: Flexible — directory has .fits/.fit files (or subdirs do)
        has_fits = any(
            f.lower().endswith(('.fits', '.fit'))
            for f in os.listdir(full)
            if os.path.isfile(os.path.join(full, f))
        )
        if not has_fits:
            # Check one level of subdirectories
            for sub in os.listdir(full):
                sub_full = os.path.join(full, sub)
                if os.path.isdir(sub_full) and not sub.startswith('.'):
                    has_fits = any(
                        f.lower().endswith(('.fits', '.fit'))
                        for f in os.listdir(sub_full)
                        if os.path.isfile(os.path.join(sub_full, f))
                    )
                    if has_fits:
                        break
        if has_fits:
            # Quick-scan a few files to discover filter names
            from .mosaic_prep import read_fits_header
            found_filters = set()
            count = 0
            for f in os.listdir(full):
                fp = os.path.join(full, f)
                if os.path.isfile(fp) and f.lower().endswith(('.fits', '.fit')):
                    h = read_fits_header(fp)
                    filt = h.get('FILTER', '').strip()
                    if filt:
                        found_filters.add(filt)
                    count += 1
                    if count >= 5:
                        break
            targets.append((entry, full, sorted(found_filters) or ['auto-detect']))

    return targets


# ── Main pipeline entry point ────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig):
    """Execute the full SHO pipeline end-to-end.

    Args:
        cfg: PipelineConfig with all parameters

    Returns:
        output_path: path to final output file

    Raises:
        FileNotFoundError: if Siril script or target directory not found
        ValueError: if required channels are missing
    """
    # Ensure the workspace exists, then resolve a bare target name
    # (e.g. "NGC6888") under it so it works from any cwd.
    ws = config.ensure_workspace()
    target = cfg.target
    if not os.path.isabs(target) and not os.path.isdir(target):
        target = os.path.join(ws, target)
    target = os.path.abspath(target)
    if not os.path.isdir(target):
        raise FileNotFoundError(f"Not a directory: {target}")

    cfg.target = target
    target_name = os.path.basename(target)
    t0 = time.time()

    cfg.log("")
    cfg.log("=" * 60)
    cfg.log(f"  SHO Pipeline — {target_name}")
    cfg.log("=" * 60)

    # Clear cache (complete): all derived state — panel workspace, stacked
    # masters, and the SHO work cache — lives under _sho_work, so one wipe is a
    # full reprocess from the raw subs. Must happen before Step 1.
    if cfg.clear_cache:
        output_dir = cfg.output_dir or target
        for d in {os.path.join(target, '_sho_work'),
                  os.path.join(output_dir, '_sho_work')}:
            if os.path.isdir(d):
                cfg.log(f"  [CLEAR CACHE] Removing {d}")
                shutil.rmtree(d)
        cfg.log("  [CLEAR CACHE] Done — full reprocess from raw subs\n")

    # ── Step 1: Group frames by pointing ────────────────────────────────
    cfg.log(f"\n▸ Step 1/4: Grouping frames by RA/DEC pointing...\n")

    panel_map, out_workspace, stats = build_panel_dirs(
        target,
        cluster_radius_deg=cfg.cluster_radius,
        darks_dir=config.darks_dir(),
    )

    n_panels = len(panel_map)
    n_filters = sum(len(v) for v in panel_map.values())
    cfg.log(f"\n  → {n_panels} panel(s), {n_filters} filter group(s)")

    # ── Step 2: Preprocessing (calibrate / register / stack) ────────────
    if cfg.no_preprocess or cfg.recolor_only:
        skip_reason = "RECOLOR ONLY" if cfg.recolor_only else "SKIPPED"
        cfg.log(f"\n▸ Step 2/4: Preprocessing — {skip_reason}\n")
    else:
        cfg.log(f"\n▸ Step 2/4: Preprocessing (calibrate / register / stack)...\n")
        _run_preprocess(panel_map, out_workspace, cfg)

    # ── Step 3: Collect result channels ─────────────────────────────────
    cfg.log(f"\n▸ Step 3/4: Collecting result files...\n")

    channels = _collect_channels(panel_map, cfg)

    for ch_name in ('sii', 'ha', 'oiii'):
        files = channels[ch_name]
        cfg.log(f"  {ch_name.upper():>4}: {len(files)} result file(s)")
        for f in files:
            cfg.log(f"        {os.path.relpath(f, out_workspace)}")

    missing = [ch for ch in ('sii', 'ha', 'oiii') if not channels[ch]]
    if missing:
        raise ValueError(
            f"No result files for: {', '.join(c.upper() for c in missing)}. "
            "Run without --no-preprocess to generate them."
        )

    # ── Step 4: SHO Hubble palette processing ───────────────────────────
    cfg.log(f"\n▸ Step 4/4: SHO Hubble palette processing...\n")

    output_dir = cfg.output_dir or target
    star_consensus = (cfg.star_consensus or 'auto').strip().lower()
    if star_consensus == 'auto':
        star_consensus = _auto_star_consensus_from_stats(stats)
        counts_s = ', '.join(
            f"{k}:{int(v[0])}" for k, v in sorted(stats.items())
        )
        cfg.log(f"  [AUTO] star consensus: {star_consensus} "
                f"(from frames: {counts_s})")

    pipeline = SHOPipeline(
        sii_paths=channels['sii'],
        ha_paths=channels['ha'],
        oiii_paths=channels['oiii'],
        output_dir=output_dir,
        preview=not cfg.no_preview,
        interactive=cfg.interactive,
        stretch_target=cfg.stretch_target,
        scnr_amount=cfg.scnr_amount,
        star_desat=cfg.star_desat,
        sat_boost=cfg.sat_boost,
        recolor_only=cfg.recolor_only,
        flatten_background=cfg.flatten_background,
        hue_strength=cfg.hue_strength,
        oiii_factor=cfg.oiii_factor,
        truthful_mode=cfg.truthful_mode,
        hubbleize=cfg.hubbleize,
        hubbleize_strength=cfg.hubbleize_strength,
        star_consensus=star_consensus,
    )

    # Inject Tk-based preview if provided (replaces cv2 PipelinePreview)
    if cfg.preview_object is not None:
        pipeline.preview = cfg.preview_object

    output_path = pipeline.run()

    elapsed = time.time() - t0
    mins, secs = divmod(elapsed, 60)

    total_frames = sum(s[0] for s in stats.values())
    total_mins = sum(s[1] for s in stats.values()) / 60

    cfg.log("")
    cfg.log("=" * 60)
    cfg.log(f"  DONE — {target_name}")
    cfg.log(f"  Output: {output_path}")
    cfg.log(f"  Mosaic: {total_frames} frames, {total_mins:.0f} min "
            f"({total_mins/60:.1f} hr)")
    cfg.log(f"  Time:   {int(mins)}m {secs:.0f}s")
    cfg.log("=" * 60)

    return output_path
