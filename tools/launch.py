#!/usr/bin/env python3
"""Single-file launch entry point (also used as the PyInstaller bundle entry).

Kept out of the repo root so it doesn't shadow the ``oeuvre`` package on import.
Run directly with ``python tools/launch.py`` or via the ``oeuvre`` console script.
"""
import os
import sys

# In a windowed PyInstaller .app there is no console, so sys.stdout/stderr can
# be None. A bare print() then raises ("lost sys.stdout"), which would crash any
# code path that logs to stdout. Logging is routed to the GUI panel, but guard
# the streams so a stray print() can never take the app down.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

from oeuvre.__main__ import main

if __name__ == '__main__':
    main()
