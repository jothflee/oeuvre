#!/usr/bin/env python3
"""
Runtime configuration — decouples the oeuvre code from any fixed data location.

The "workspace" is the data root: where target directories, shared ``darks/``,
and the ``StarNetv2CLI_MacOS/`` binary live.

The local app home is separate: ``~/oeuvre`` is used for user-managed helpers
like an optional StarNet install, so they are easy to set up and remove without
touching the data workspace.
"""

import json
import os

DEFAULT_WORKSPACE = os.path.expanduser('~/oeuvre-astro')
DEFAULT_APP_HOME = os.path.expanduser('~/oeuvre')
DEFAULT_SETTINGS_FILE = 'oeuvre_settings.json'

DEFAULT_SETTINGS = {
    'plate_solve_endpoint': 'https://nova.astrometry.net/api/',
    'plate_solve_api_key': '',
}


def workspace():
    """Absolute path to the data/workspace root (env override, else default)."""
    return os.path.abspath(os.environ.get('OEUVRE_WORKSPACE') or DEFAULT_WORKSPACE)


def ensure_workspace():
    """Resolve the workspace and guarantee the directory exists (mkdir -p)."""
    ws = workspace()
    os.makedirs(ws, exist_ok=True)
    return ws


def darks_dir():
    """Default shared darks directory under the workspace."""
    return os.path.join(workspace(), 'darks')


def app_home():
    """Local app-managed home for optional helpers like StarNet."""
    return os.path.abspath(DEFAULT_APP_HOME)


def ensure_app_home():
    """Resolve the app home and guarantee the directory exists (mkdir -p)."""
    home = app_home()
    os.makedirs(home, exist_ok=True)
    return home


def settings_path():
    """Path to the persistent app settings file."""
    return os.path.join(app_home(), DEFAULT_SETTINGS_FILE)


def load_settings():
    """Load persistent settings, returning defaults if the file is missing."""
    data = dict(DEFAULT_SETTINGS)
    try:
        with open(settings_path(), 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data.update({k: v for k, v in raw.items() if v is not None})
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return data


def save_settings(settings):
    """Persist settings to the local app home."""
    ensure_app_home()
    data = dict(DEFAULT_SETTINGS)
    if isinstance(settings, dict):
        data.update({k: v for k, v in settings.items() if v is not None})
    with open(settings_path(), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')
    return data


def plate_solve_settings():
    """Return the current plate-solve settings merged with defaults."""
    settings = load_settings()
    return {
        'endpoint': settings.get('plate_solve_endpoint', ''),
        'api_key': settings.get('plate_solve_api_key', ''),
    }


def save_plate_solve_settings(*, endpoint=None, api_key=None):
    """Update and persist the plate-solve settings section."""
    settings = load_settings()
    if endpoint is not None:
        settings['plate_solve_endpoint'] = endpoint
    if api_key is not None:
        settings['plate_solve_api_key'] = api_key
    return save_settings(settings)
