#!/usr/bin/env python3
"""
Runtime configuration — decouples the oeuvre code from any fixed data location.

The "workspace" is the data root: where target directories, shared ``darks/``,
the ``StarNetv2CLI_MacOS/`` binary, and (optionally) ``docker-compose.yaml`` for
plate solving live. It is resolved, in order:

  1. the ``OEUVRE_WORKSPACE`` environment variable, if set
  2. the default data directory, ``~/oeuvre-astro``

The CLI ``--workspace`` flag simply sets ``OEUVRE_WORKSPACE`` before the
pipeline runs, so a single source of truth flows everywhere.
"""

import os

DEFAULT_WORKSPACE = os.path.expanduser('~/oeuvre-astro')


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
