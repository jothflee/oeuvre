"""Tests for the plate-solve helper logic."""

from oeuvre import plate_solve


def test_fits_has_wcs_detects_existing_keywords(monkeypatch):
    monkeypatch.setattr(
        plate_solve,
        'read_fits_header',
        lambda path: {'CRVAL1': 12.3, 'CRVAL2': -45.6},
    )

    assert plate_solve.fits_has_wcs('/tmp/already_solved.fit')


def test_fits_has_wcs_false_when_header_is_empty(monkeypatch):
    monkeypatch.setattr(
        plate_solve,
        'read_fits_header',
        lambda path: {},
    )

    assert not plate_solve.fits_has_wcs('/tmp/unsolved.fit')


def test_plate_solve_if_needed_skips_when_wcs_present(monkeypatch):
    calls = []

    monkeypatch.setattr(plate_solve, 'fits_has_wcs', lambda path: True)
    monkeypatch.setattr(
        plate_solve,
        'plate_solve',
        lambda *args, **kwargs: calls.append('called'),
    )

    result = plate_solve.plate_solve_if_needed('/tmp/already_solved.fit')

    assert result is None
    assert calls == []


def test_plate_solve_if_needed_delegates_when_wcs_missing(monkeypatch):
    monkeypatch.setattr(plate_solve, 'fits_has_wcs', lambda path: False)
    monkeypatch.setattr(
        plate_solve,
        'plate_solve',
        lambda path, **kwargs: {'CRVAL1': 1.0},
    )

    result = plate_solve.plate_solve_if_needed('/tmp/unsolved.fit')

    assert result == {'CRVAL1': 1.0}


def test_plate_solve_allows_empty_api_key(monkeypatch):
    captured = {}

    monkeypatch.setattr(plate_solve, 'plate_solve_settings', lambda: {
        'endpoint': 'http://localhost:8000/api/',
        'api_key': '',
    })
    monkeypatch.setattr(plate_solve, 'os', plate_solve.os)
    monkeypatch.setattr(plate_solve, '_solve_scale_hints', lambda path: ({}, None, None))
    monkeypatch.setattr(plate_solve, '_request_json', lambda *args, **kwargs: {
        'status': 'success',
        'session': 'local-session',
    })

    def fake_upload(api_base, session, fits_path, payload, timeout=300):
        captured['session'] = session
        return {'status': 'success', 'subid': 42}

    monkeypatch.setattr(plate_solve, '_upload_file', fake_upload)
    monkeypatch.setattr(plate_solve.time, 'monotonic', lambda: 0.0)
    monkeypatch.setattr(plate_solve.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(plate_solve, '_download_binary', lambda *args, **kwargs: None)
    monkeypatch.setattr(plate_solve, 'read_fits_header', lambda path: {
        'CRVAL1': '123.4',
        'CRVAL2': '-22.5',
        'CROTA2': '0',
    })
    monkeypatch.setattr(
        plate_solve,
        'shutil',
        plate_solve.shutil,
    )

    original_isfile = plate_solve.os.path.isfile
    monkeypatch.setattr(
        plate_solve.os.path,
        'isfile',
        lambda path: True if path == '/tmp/unsolved.fit' else original_isfile(path),
    )
    monkeypatch.setattr(
        plate_solve,
        '_request_json',
        lambda url, payload=None, headers=None, method=None, timeout=60: (
            {'status': 'success', 'session': 'local-session'}
            if url.endswith('/login')
            else {'jobs': [42], 'status': 'solving'}
            if '/submissions/' in url
            else {'status': 'success'}
        ),
    )
    monkeypatch.setattr(
        plate_solve,
        '_upload_file',
        fake_upload,
    )

    result = plate_solve.plate_solve('/tmp/unsolved.fit', log_fn=lambda msg: None)

    assert captured['session'] == 'local-session'
    assert result == {'CRVAL1': 123.4, 'CRVAL2': -22.5, 'CROTA2': 0.0}