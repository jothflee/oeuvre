# Oeuvre

[![CI](https://github.com/jothflee/oeuvre/actions/workflows/ci.yml/badge.svg)](https://github.com/jothflee/oeuvre/actions/workflows/ci.yml)

Point Oeuvre at a night's worth of subs and quickly get a nice SHO image out —
**fully automated, with only light touch-up** (no heavy manual editing). It
calibrates, registers, and stacks your per-filter light frames, then produces a
stretched, star-balanced Hubble-palette (SHO) image. Pure Python, no Siril
dependency.

**Goal:** hands-off **full-sky SHO mosaics** — process every target captured in
a session and assemble them into a wide-field mosaic automatically.

Registration everywhere (sub stacking, channel alignment, mosaic panels) uses a
single star-centroid + asterism matcher. Star removal shells out to a local
StarNet++ binary; plate solving (optional) uses a local astrometry.net Docker
service.

## Requirements

- Python ≥ 3.11
- Runtime deps (resolved automatically): numpy, scipy, opencv-python, astropy, pillow
- [StarNet++ v2 CLI](https://www.starnetastro.com/) for star removal (external binary, not bundled)

## Install

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # create the venv and install deps
```

Or with pip: `pip install .`

**Desktop app (macOS):** `bash tools/make_macos_app.sh` builds a double-clickable
`Oeuvre.app` (with icon) — drag it to `/Applications`. Pre-built standalone
bundles for macOS and Linux are also attached to each
[GitHub Release](https://github.com/jothflee/oeuvre/releases).

## Data location (workspace)

Code and data are separate. The **workspace** is the data root — target
directories, shared `darks/`, the `StarNetv2CLI_MacOS/` binary, and (optionally)
`docker-compose.yaml` for plate solving. Resolved in order:

1. `OEUVRE_WORKSPACE` env var, or the `--workspace` flag
2. default: `~/oeuvre-astro` (created automatically if missing)

```
<workspace>/
  NGC6888/Light/Ha-7nm/*.fits     # target frames, grouped by filter
  darks/*.fits                    # shared dark frames
  StarNetv2CLI_MacOS/starnet++    # star removal binary
  docker-compose.yaml             # optional, for plate solving
```

## Usage

```bash
uv run oeuvre                       # GUI, pick a target
uv run oeuvre NGC6888               # GUI pre-selected on a target
uv run oeuvre NGC6888 --headless    # CLI-only, full pipeline
uv run oeuvre NGC6888 --headless --workspace ~/oeuvre-astro
uv run oeuvre NGC6888 --headless --no-preprocess   # reuse existing masters
```

(After `pip install .`, the `oeuvre` command and `python -m oeuvre` work too.)

## Pipeline stages

1. Group frames by RA/DEC pointing into panels (`mosaic_prep`)
2. Per-filter preprocessing: master dark, dark calibration + cosmetic
   correction, 2-pass asterism registration, Winsorized-σ stacking (`preprocess`)
3. SHO Hubble-palette processing: channel alignment, linked arcsinh stretch,
   star removal, SCNR, color, recombine (`natural_narrowband`)

## Output files

Each run writes three files to the target directory. The pipeline stays
floating-point throughout and is preserved as far as each format allows —
nothing is truncated to 8-bit:

- **`SHO_*.fits`** — 32-bit float + WCS. Full-precision archival master / for reprocessing.
- **`SHO_*.tiff`** — 16-bit RGB. Fidelity-first deliverable for editing and printing.
- **`SHO_*.png`** — 16-bit RGB. High-fidelity preview; zooms cleanly (no 8-bit banding). Down-convert from this for 8-bit-only sharing targets.

A `SHO_*_provenance.txt` sidecar records the processing parameters.

## Development

```bash
uv sync          # install runtime + dev dependencies
uv run pytest    # run the test suite
```

CI runs the tests on Python 3.11–3.13 via GitHub Actions.

## License

MIT — see [LICENSE](LICENSE).
