"""
Oeuvre — SHO Narrowband Processing Pipeline
============================================

End-to-end: raw FITS lights → finished Hubble palette image.

Modules:
  mosaic_prep          – Frame clustering & panel directory builder
  natural_narrowband   – SHO Hubble palette image processing
  pipeline             – Unified pipeline orchestrator
  projection           – Sky map projection & visualisation
  gui                  – Tkinter GUI for target selection
"""

from .natural_narrowband import (
    SHOPipeline,
    STRETCH_TARGET,
    SCNR_AMOUNT,
    STAR_DESAT,
    SAT_BOOST,
)
from .mosaic_prep import build_panel_dirs, scan_fits_by_filter
from .pipeline import run_pipeline, PipelineConfig
from .projection import SkyMap, SkyFrame

__version__ = "0.2.0"
__all__ = [
    "SHOPipeline",
    "build_panel_dirs",
    "scan_fits_by_filter",
    "run_pipeline",
    "PipelineConfig",
    "SkyMap",
    "SkyFrame",
    "STRETCH_TARGET",
    "SCNR_AMOUNT",
    "STAR_DESAT",
    "SAT_BOOST",
]
