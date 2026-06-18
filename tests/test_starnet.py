"""StarNet resolution: multiple binary names, v2 preference, chosen-folder override."""
import os
import pytest

from oeuvre import starnet, config


def _make_starnet(dirpath, names):
    os.makedirs(dirpath, exist_ok=True)
    for n in names:
        p = os.path.join(dirpath, n)
        with open(p, 'w') as f:
            f.write('#!/bin/sh\n')
        os.chmod(p, 0o755)
    return os.path.abspath(dirpath)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Sandbox find_starnet's search roots + settings file; no real installs."""
    monkeypatch.delenv('STARNET_DIR', raising=False)
    ws, home = tmp_path / 'ws', tmp_path / 'home'
    ws.mkdir(); home.mkdir()
    monkeypatch.setattr(starnet, 'workspace', lambda: str(ws))
    monkeypatch.setattr(starnet, 'app_home', lambda: str(home))
    monkeypatch.setattr(config, 'settings_path', lambda: str(tmp_path / 's.json'))
    monkeypatch.chdir(tmp_path)
    return ws, home


def test_binary_name_preference(tmp_path):
    both = _make_starnet(str(tmp_path / 'both'), ['starnet++', 'starnet2'])
    assert os.path.basename(starnet.starnet_binary_path(both)) == 'starnet2'
    legacy = _make_starnet(str(tmp_path / 'legacy'), ['starnet++'])
    assert os.path.basename(starnet.starnet_binary_path(legacy)) == 'starnet++'
    assert starnet.starnet_binary_path(str(tmp_path / 'empty')) is None


def test_argv_per_binary():
    assert starnet._starnet_argv('/x/starnet2', 'i.tif', 'o.tif') == \
        ['/x/starnet2', '--input', 'i.tif', '--output', 'o.tif']
    assert starnet._starnet_argv('/x/starnet++', 'i.tif', 'o.tif') == \
        ['/x/starnet++', 'i.tif', 'o.tif']


def test_resolve_source(tmp_path):
    d = _make_starnet(str(tmp_path / 'sn'), ['starnet2'])
    assert starnet.resolve_starnet_source(d) == d
    assert starnet.resolve_starnet_source(os.path.join(d, 'starnet2')) == d
    assert starnet.resolve_starnet_source(str(tmp_path / 'nope')) is None


def test_find_prefers_v2_then_chosen_overrides(isolated):
    ws, _ = isolated
    legacy = _make_starnet(str(ws / 'StarNetv2CLI_MacOS'), ['starnet++'])
    v2 = _make_starnet(str(ws / 'starnet2_macos-x64_2.5.2_cli'), ['starnet2'])

    # Auto-discovery prefers the v2 binary even though the legacy folder is a
    # known/earlier candidate.
    binp, d = starnet.find_starnet()
    assert os.path.basename(binp) == 'starnet2'
    assert os.path.abspath(d) == v2

    # An explicit choice wins, using whatever binary that folder has.
    config.save_starnet_dir_setting(legacy)
    binp, d = starnet.find_starnet()
    assert os.path.abspath(d) == legacy
    assert os.path.basename(binp) == 'starnet++'

    st = starnet.starnet_status()
    assert st['available'] and st['chosen_active']
    assert st['binary_name'] == 'starnet++'


def test_status_autodetect_not_chosen(isolated):
    ws, _ = isolated
    _make_starnet(str(ws / 'starnet2_cli'), ['starnet2'])
    st = starnet.starnet_status()
    assert st['available'] and st['binary_name'] == 'starnet2'
    assert not st['chosen_active'] and st['chosen_dir'] is None
