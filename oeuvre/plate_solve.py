#!/usr/bin/env python3
"""
Plate-solve FITS files using an astrometry.net API endpoint.

The endpoint is configurable in Oeuvre settings. By default it targets the
public Nova.astrometry.net API, but self-hosted compatible endpoints work too.
"""

import json
import math
import mimetypes
import os
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import plate_solve_settings
from .mosaic_prep import read_fits_header

# WCS keywords that astrometry.net writes into the .wcs file
_WCS_KEYS = (
    'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
    'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
    'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2',
    'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
    'EQUINOX', 'IMAGEW', 'IMAGEH',
    'A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER',
)

_DEFAULT_API_BASE = 'https://nova.astrometry.net/api/'


def _normalize_api_base(endpoint):
    endpoint = (endpoint or _DEFAULT_API_BASE).strip()
    if not endpoint:
        endpoint = _DEFAULT_API_BASE
    if not endpoint.endswith('/'):
        endpoint += '/'
    if '/api/' not in endpoint:
        endpoint = endpoint.rstrip('/') + '/api/'
    return endpoint


def _api_root(api_base):
    return api_base.replace('/api/', '/', 1)


def _request_json(url, payload=None, headers=None, method=None, timeout=60):
    req_headers = {'Accept': 'application/json'}
    if headers:
        req_headers.update(headers)
    if payload is None:
        req = Request(url, headers=req_headers, method='GET')
    else:
        body = urlencode(payload).encode('utf-8')
        req_headers.setdefault('Content-Type', 'application/x-www-form-urlencoded')
        req = Request(url, data=body, headers=req_headers, method=method or 'POST')
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode('utf-8')
    return json.loads(raw)


def _encode_multipart(fields, files):
    boundary = '----oeuvre-' + hex(int(time.time() * 1000000))[2:]
    chunks = []

    for name, value in fields.items():
        chunks.append(f'--{boundary}\r\n'.encode('utf-8'))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode('utf-8'))
        chunks.append(str(value).encode('utf-8'))
        chunks.append(b'\r\n')

    for name, (filename, content) in files.items():
        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        chunks.append(f'--{boundary}\r\n'.encode('utf-8'))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{os.path.basename(filename)}"\r\n'.encode('utf-8'))
        chunks.append(f'Content-Type: {content_type}\r\n\r\n'.encode('utf-8'))
        chunks.append(content)
        chunks.append(b'\r\n')

    chunks.append(f'--{boundary}--\r\n'.encode('utf-8'))
    return boundary, b''.join(chunks)


def _upload_file(api_base, session, fits_path, payload, timeout=300):
    url = api_base + 'upload'
    with open(fits_path, 'rb') as f:
        data = f.read()
    fields = {
        'request-json': json.dumps(dict(payload, session=session)),
    }
    boundary, body = _encode_multipart(fields, {'file': (fits_path, data)})
    req = Request(
        url,
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Referer': api_base + 'login',
            'Accept': 'application/json',
        },
        method='POST',
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _download_binary(url, path, timeout=120):
    req = Request(url, headers={'Referer': url.rsplit('/', 1)[0] + '/login'})
    with urlopen(req, timeout=timeout) as resp, open(path, 'wb') as out:
        shutil.copyfileobj(resp, out)


def _solve_scale_hints(fits_path):
    try:
        hdr = read_fits_header(fits_path)
    except Exception:
        return {}, None, None

    payload = {}
    ra_hint = None
    dec_hint = None
    if 'RA' in hdr or 'CRVAL1' in hdr:
        try:
            ra_hint = float(hdr.get('RA', hdr.get('CRVAL1')))
        except Exception:
            ra_hint = None
    if 'DEC' in hdr or 'CRVAL2' in hdr:
        try:
            dec_hint = float(hdr.get('DEC', hdr.get('CRVAL2')))
        except Exception:
            dec_hint = None

    scale_low = scale_high = None
    if 'SCALE' in hdr:
        try:
            sc = float(hdr['SCALE'])
            scale_low, scale_high = sc * 0.8, sc * 1.2
        except Exception:
            pass
    elif 'SECPIX1' in hdr:
        try:
            sc = float(hdr['SECPIX1'])
            scale_low, scale_high = sc * 0.8, sc * 1.2
        except Exception:
            pass
    elif 'CDELT1' in hdr:
        try:
            sc = abs(float(hdr['CDELT1'])) * 3600.0
            scale_low, scale_high = sc * 0.8, sc * 1.2
        except Exception:
            pass

    if scale_low is not None and scale_high is not None:
        payload.update(
            scale_units='arcsecperpix',
            scale_type='ul',
            scale_lower=scale_low,
            scale_upper=scale_high,
        )
    if ra_hint is not None and dec_hint is not None:
        payload.update(
            center_ra=ra_hint,
            center_dec=dec_hint,
            radius=5.0,
        )
    payload.update(
        downsample_factor=2,
        publicly_visible='n',
        allow_commercial_use='d',
        allow_modifications='d',
    )
    return payload, ra_hint, dec_hint


def fits_has_wcs(fits_path):
    """Return True when the FITS header already contains usable WCS data."""
    try:
        hdr = read_fits_header(fits_path)
    except Exception:
        return False

    return any(
        key in hdr
        for key in (
            'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'CDELT1', 'CDELT2', 'CROTA2',
        )
    )


def plate_solve_if_needed(fits_path, **kwargs):
    """Solve only when the FITS does not already carry WCS metadata."""
    if fits_has_wcs(fits_path):
        log_fn = kwargs.get('log_fn') or print
        log_fn(f"  WCS already present in {os.path.basename(fits_path)}; skipping plate solve")
        return None
    return plate_solve(fits_path, **kwargs)


def plate_solve(fits_path, *, timeout=300, api_endpoint=None, api_key=None,
                log_fn=None):
    """Plate-solve a FITS file using an astrometry.net API endpoint.

    The solver uploads the FITS, polls the resulting submission/job, and then
    downloads the job's WCS FITS file.

    Returns:
        dict of WCS keywords on success, or None on failure.
    """
    if log_fn is None:
        log_fn = print

    fits_path = os.path.abspath(fits_path)
    if not os.path.isfile(fits_path):
        log_fn(f"  plate_solve: file not found: {fits_path}")
        return None

    settings = plate_solve_settings()
    api_base = _normalize_api_base(api_endpoint or settings['endpoint'])
    api_key = (api_key if api_key is not None else settings['api_key']).strip()

    if not api_key:
        log_fn('  Plate-solving skipped: no Astrometry.net API key configured')
        return None

    base = os.path.basename(fits_path)
    stem = os.path.splitext(base)[0]
    work_dir = os.path.join(os.path.dirname(fits_path), '_sho_work', '_solve_tmp')
    os.makedirs(work_dir, exist_ok=True)

    referer = api_base + 'login'
    try:
        log_fn(f"  Plate-solving {base} via {api_base} ...")
        login = _request_json(
            api_base + 'login',
            payload={'request-json': json.dumps({'apikey': api_key})},
            headers={'Referer': referer},
            timeout=timeout,
        )
        if login.get('status') != 'success' or not login.get('session'):
            log_fn(f"  Plate-solve login failed: {login}")
            return None

        upload_payload, ra_hint, dec_hint = _solve_scale_hints(fits_path)
        upload = _upload_file(api_base, login['session'], fits_path,
                              upload_payload, timeout=timeout)
        if upload.get('status') != 'success' or 'subid' not in upload:
            log_fn(f"  Plate-solve upload failed: {upload}")
            return None

        subid = int(upload['subid'])
        jobid = None
        deadline = time.monotonic() + timeout
        last_status = None

        while time.monotonic() < deadline:
            sub = _request_json(api_base + f'submissions/{subid}', timeout=60)
            jobs = sub.get('jobs', []) or []
            jobid = next((j for j in jobs if j is not None), None)
            if jobid is not None:
                job = _request_json(api_base + f'jobs/{jobid}', timeout=60)
                last_status = job.get('status')
                if last_status == 'success':
                    break
                if last_status in {'failure', 'failed', 'cancelled'}:
                    log_fn(f"  Plate-solve failed: {job}")
                    return None
            else:
                last_status = sub.get('status')
            time.sleep(5)

        if jobid is None:
            log_fn(f"  Plate-solve timed out waiting for a job (submission {subid})")
            return None

        if last_status != 'success':
            log_fn(f"  Plate-solve timed out waiting for success (job {jobid})")
            return None

        wcs_file = os.path.join(work_dir, f'{stem}.wcs')
        wcs_url = _api_root(api_base).rstrip('/') + f'/wcs_file/{jobid}'
        _download_binary(wcs_url, wcs_file, timeout=timeout)

        wcs_hdr = read_fits_header(wcs_file)
        wcs_dict = {}
        for key, val in wcs_hdr.items():
            if key in _WCS_KEYS or key.startswith(('A_', 'B_', 'AP_', 'BP_')):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
                wcs_dict[key] = val

        if 'CROTA2' not in wcs_dict and 'CD1_1' in wcs_dict and 'CD1_2' in wcs_dict:
            cd11 = float(wcs_dict['CD1_1'])
            cd12 = float(wcs_dict['CD1_2'])
            wcs_dict['CROTA2'] = math.degrees(math.atan2(cd12, -cd11))

        ra_s = f"{float(wcs_dict.get('CRVAL1', 0)):.4f}"
        dec_s = f"{float(wcs_dict.get('CRVAL2', 0)):.4f}"
        rot_s = f"{float(wcs_dict.get('CROTA2', 0)):.2f}"
        log_fn(f"  Solved! RA={ra_s}  DEC={dec_s}  rot={rot_s}°")
        return wcs_dict

    except (HTTPError, URLError, OSError, json.JSONDecodeError, ValueError) as e:
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
