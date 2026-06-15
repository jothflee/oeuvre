#!/usr/bin/env python3
"""
Plate-solve FITS files using a local astrometry.net Docker service.

Runs ``solve-field`` inside the running ``astrometry`` container
via ``docker compose exec``, then reads the WCS solution back.
"""

import math
import os
import subprocess
import shutil

from .mosaic_prep import read_fits_header
from .config import workspace

# WCS keywords that solve-field writes into the .wcs file
_WCS_KEYS = (
    'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
    'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
    'EQUINOX', 'IMAGEW', 'IMAGEH',
    'A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER',
)


def _host_to_container(host_path):
    """Translate a host path under the workspace to /astro/<relative>."""
    host_path = os.path.abspath(host_path)
    ws = workspace()
    rel = os.path.relpath(host_path, ws)
    if rel.startswith('..'):
        raise ValueError(f"Path {host_path} is outside workspace {ws}")
    return '/astro/' + rel


def plate_solve(fits_path, *, timeout=300, scale_low=None, scale_high=None,
                ra_hint=None, dec_hint=None, radius=5.0,
                downsample=2, log_fn=None):
    """Plate-solve a FITS file using the running astrometry.net container.

    Runs ``solve-field`` via ``docker compose exec`` inside the container
    that already has index files at ``/index`` and the workspace at ``/astro``.

    Args:
        fits_path:      Path to the FITS file (host or relative).
        timeout:        Max seconds for the solve attempt.
        scale_low:      Lower bound for image scale (arcsec/pixel).
        scale_high:     Upper bound for image scale (arcsec/pixel).
        ra_hint:        Approx RA in degrees.
        dec_hint:       Approx DEC in degrees.
        radius:         Search radius around hint (degrees).
        downsample:     Downsample factor for source detection (2 is good
                        for large images).
        log_fn:         Callable for log messages; defaults to print.

    Returns:
        dict of WCS keywords on success, or None on failure.
    """
    if log_fn is None:
        log_fn = print

    fits_path = os.path.abspath(fits_path)
    if not os.path.isfile(fits_path):
        log_fn(f"  plate_solve: file not found: {fits_path}")
        return None

    base = os.path.basename(fits_path)
    stem = os.path.splitext(base)[0]

    # Try to read hints from header if not provided
    if ra_hint is None or dec_hint is None:
        try:
            hdr = read_fits_header(fits_path)
            if ra_hint is None:
                ra_hint = float(hdr.get('RA', hdr.get('CRVAL1', 0)))
            if dec_hint is None:
                dec_hint = float(hdr.get('DEC', hdr.get('CRVAL2', 0)))
            if scale_low is None and 'SCALE' in hdr:
                sc = float(hdr['SCALE'])
                scale_low = sc * 0.8
                scale_high = sc * 1.2
            elif scale_low is None and 'SECPIX1' in hdr:
                sc = float(hdr['SECPIX1'])
                scale_low = sc * 0.8
                scale_high = sc * 1.2
            elif scale_low is None and 'CDELT1' in hdr:
                sc = abs(float(hdr['CDELT1'])) * 3600.0
                scale_low = sc * 0.8
                scale_high = sc * 1.2
        except Exception:
            pass

    # Create temp output directory inside the workspace so the container
    # can see it at /astro/_solve_tmp/
    work_dir = os.path.join(workspace(), '_solve_tmp')
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Container paths
        container_fits = _host_to_container(fits_path)
        container_work = '/astro/_solve_tmp'

        # Build solve-field args
        sf_args = [
            'solve-field',
            container_fits,
            '--backend-config', '/index/docker.cfg',
            '--dir', container_work,
            '--overwrite',
            '--no-plots',
            '--no-verify',
            '--crpix-center',
            '--tweak-order', '2',
            '--downsample', str(downsample),
        ]

        if scale_low is not None and scale_high is not None:
            sf_args += ['--scale-units', 'arcsecperpix',
                        '--scale-low', str(scale_low),
                        '--scale-high', str(scale_high)]

        if ra_hint is not None and dec_hint is not None:
            sf_args += ['--ra', str(ra_hint),
                        '--dec', str(dec_hint),
                        '--radius', str(radius)]

        # Run via docker compose exec in the astrometry service
        cmd = [
            'docker', 'compose', 'exec', '-T', 'astrometry',
        ] + sf_args

        log_fn(f"  Plate-solving {base} ...")

        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            cwd=workspace(),  # so docker compose finds docker-compose.yml
        )

        # Check for solution (output files are in _solve_tmp on host)
        solved_file = os.path.join(work_dir, f'{stem}.solved')
        wcs_file = os.path.join(work_dir, f'{stem}.wcs')

        if not os.path.exists(solved_file):
            log_fn(f"  Plate-solve FAILED for {base}")
            # Show useful output lines
            output = result.stdout or result.stderr or ''
            for line in output.strip().split('\n')[-8:]:
                if line.strip():
                    log_fn(f"    {line}")
            return None

        # Read WCS from the .wcs file
        wcs_hdr = read_fits_header(wcs_file)
        wcs_dict = {}
        for key in wcs_hdr:
            if key in _WCS_KEYS or key.startswith(('A_', 'B_', 'AP_', 'BP_')):
                # Convert numeric values from strings to floats.
                # read_fits_header returns everything as strings; FITS
                # WCS keywords must be numeric for Siril compatibility.
                val = wcs_hdr[key]
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass  # keep as string (e.g. CTYPE1 = 'RA---TAN')
                wcs_dict[key] = val

        # Compute CROTA2 from CD matrix if not directly present
        if 'CROTA2' not in wcs_dict and 'CD1_1' in wcs_dict and 'CD1_2' in wcs_dict:
            cd11 = float(wcs_dict['CD1_1'])
            cd12 = float(wcs_dict['CD1_2'])
            wcs_dict['CROTA2'] = math.degrees(math.atan2(cd12, -cd11))

        ra_s = f"{float(wcs_dict.get('CRVAL1', 0)):.4f}"
        dec_s = f"{float(wcs_dict.get('CRVAL2', 0)):.4f}"
        rot_s = f"{float(wcs_dict.get('CROTA2', 0)):.2f}"
        log_fn(f"  Solved! RA={ra_s}  DEC={dec_s}  rot={rot_s}°")

        return wcs_dict

    except subprocess.TimeoutExpired:
        log_fn(f"  Plate-solve TIMEOUT for {base}")
        return None
    except Exception as e:
        log_fn(f"  Plate-solve ERROR: {e}")
        return None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def update_fits_wcs(fits_path, wcs_dict, log_fn=None):
    """Merge WCS keywords into an existing FITS file's header.

    Uses the pure-Python FITS header editor to avoid requiring astropy.
    Falls back to astropy if available.
    """
    if log_fn is None:
        log_fn = print

    fits_path = os.path.abspath(fits_path)

    try:
        from astropy.io import fits as astropy_fits
        with astropy_fits.open(fits_path, mode='update') as hdul:
            for k, v in wcs_dict.items():
                try:
                    hdul[0].header[k] = v
                except Exception:
                    pass
            hdul.flush()
        log_fn(f"  WCS written to {os.path.basename(fits_path)}")
        return True
    except ImportError:
        pass

    # Pure-python fallback: re-read, update header, re-write
    try:
        from .natural_narrowband import load_fits, save_fits
        data, header = load_fits(fits_path)
        header.update(wcs_dict)
        save_fits(data, fits_path, header_extra=header)
        log_fn(f"  WCS written to {os.path.basename(fits_path)}")
        return True
    except Exception as e:
        log_fn(f"  Failed to write WCS: {e}")
        return False
