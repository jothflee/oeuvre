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