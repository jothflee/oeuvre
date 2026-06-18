"""Tests for the projection loader."""

import numpy as np

from oeuvre import projection


def test_add_file_blind_solves_when_header_lacks_pointing(monkeypatch):
    headers = [
        {
            'NAXIS': '2',
            'NAXIS1': '10',
            'NAXIS2': '10',
            'SECPIX1': '1.5',
            'CROTA2': '0',
        },
        {
            'NAXIS': '2',
            'NAXIS1': '10',
            'NAXIS2': '10',
            'SECPIX1': '1.5',
            'RA': '123.4',
            'DEC': '-22.5',
            'CRVAL1': '123.4',
            'CRVAL2': '-22.5',
            'CROTA2': '0',
        },
    ]
    solved = {'done': False}
    plate_solve_calls = []

    def fake_read_header(path):
        return headers[1] if solved['done'] else headers[0]

    def fake_plate_solve_if_needed(path, log_fn=None):
        plate_solve_calls.append(path)
        solved['done'] = True
        return {'CRVAL1': 123.4, 'CRVAL2': -22.5, 'CROTA2': 0.0}

    monkeypatch.setattr(projection, 'read_fits_header', fake_read_header)
    monkeypatch.setattr(
        projection,
        'plate_solve_if_needed',
        fake_plate_solve_if_needed,
    )
    monkeypatch.setattr(
        projection,
        'update_fits_wcs',
        lambda path, wcs, log_fn=None: None,
    )
    monkeypatch.setattr(
        projection,
        'load_fits',
        lambda path: (np.zeros((10, 10), dtype=np.float32), {}),
    )

    sky_map = projection.SkyMap()
    frame = sky_map.add_file('/tmp/unsolved.fit')

    assert plate_solve_calls == ['/tmp/unsolved.fit']
    assert frame.ra_deg == 123.4
    assert frame.dec_deg == -22.5
    assert len(sky_map.frames) == 1


def test_add_file_skips_when_no_pointing_and_no_api_key(monkeypatch):
    monkeypatch.setattr(
        projection,
        'read_fits_header',
        lambda path: {
            'NAXIS': '2',
            'NAXIS1': '10',
            'NAXIS2': '10',
            'SECPIX1': '1.5',
        },
    )
    monkeypatch.setattr(
        projection,
        'plate_solve_if_needed',
        lambda path, log_fn=None: None,
    )

    sky_map = projection.SkyMap()
    frame = sky_map.add_file('/tmp/unsolved.fit')

    assert frame is None
    assert len(sky_map.frames) == 0