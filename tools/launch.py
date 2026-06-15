#!/usr/bin/env python3
"""Single-file launch entry point (also used as the PyInstaller bundle entry).

Kept out of the repo root so it doesn't shadow the ``oeuvre`` package on import.
Run directly with ``python tools/launch.py`` or via the ``oeuvre`` console script.
"""
from oeuvre.__main__ import main

if __name__ == '__main__':
    main()
