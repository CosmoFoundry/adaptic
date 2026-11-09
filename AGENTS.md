# AGENTS.md

This file provides guidance to LLM coding assistants when working with code in this repository.

## Project Overview

**adaptic** is a lightweight Python package providing PyTorch wrappers for cosmology datasets, currently focused on DESI (Dark Energy Spectroscopic Instrument) spectroscopic data. The package is designed to be usable by copy-pasting individual module files into projects.

Future development will include wrappers for Legacy Survey imaging and HATS catalogs, following a similar design philosophy.

## Commands

```bash
# Install in development mode
pip install -e py/

# Run tests
pytest py/adaptic/test/

# Run a single test file
pytest py/adaptic/test/test_desi.py

# Run a single test class or method
pytest py/adaptic/test/test_desi.py::TestDESI::test_healpix
```

There are no separate lint/format commands configured; CI runs only pytest. CI (`.github/workflows/test.yaml`) runs bare `pytest` from the repo root on Python 3.12, 3.13, and 3.14.

## Architecture

### Package layout

```
py/adaptic/          # Main package (installed from py/ subdirectory)
├── desi.py          # Primary module: DESIDataset and all DESI-specific logic
└── test/
    ├── test_desi.py    # unittest-based tests, run via pytest
    ├── test_uniqpix.py # unittest-based tests, run via pytest
    ├── util.py         # Synthetic FITS file generators for tests (no real DESI data needed)
    └── data/           # Small FITS fixtures checked into the repo (summary tables)
```

### Core class: `DESIDataset`

`DESIDataset` is a PyTorch `IterableDataset` in `desi.py`. It supports three on-disk data layout styles:

- **HEALPix**: `healpix/{survey}/{program}/{healpix//100}/{healpix}/`
- **Tile**: `tiles/cumulative/{tileid}/{night}/`
- **UniqPix**: `spectra/{survey}/{program}/{uniqpix//100}/{uniqpix}/`

Each layout reads paired FITS files: coadded spectra (`coadd-*.fits`) and redshift catalogs (`redrock-*.fits` or equivalent). The dataset optionally coadds B/R/Z camera spectra using inverse-variance weighting and returns dictionaries with `FLUX`, `IVAR`, `MASK` arrays and metadata columns.

### Summary tables and auto-discovery

`DESIDataset` needs a summary table enumerating the groups to load, supplied via `summary_table=`. Alternatively, pass `coadd_type="HEALPIX"` or `"TILE"` and it auto-discovers the table via `autodiscover_summary_table` (desi.py):

- HEALPIX: prefers a UniqPix specprod (`spectra/` dir → concatenates all `uniqpix-{survey}-{program}.fits` via `find_and_concat_uniqpix_tables`), else falls back to `healpix/tilepix.fits`
- TILE: expects `tiles-{specprod}.fits` at the specprod root, where `specprod` is the lowest directory name

An explicit `summary_table` overrides auto-discovery.

Key design decisions:
- Worker load balancing: healpix/tile groups are distributed across DataLoader workers deterministically
- Train/val splits use a seeded RNG applied per-target for strict exclusivity
- Column selection (`columns=`) auto-detects whether each column comes from the spectra or redshift file
- Normalization is optional (weighted mean=0, std=1 per spectrum)

### Test utilities

`py/adaptic/test/util.py` contains functions that generate synthetic DESI FITS files in memory — tests never require real DESI data on disk. When adding new features that read new columns or file structures, add corresponding synthetic data generation to `util.py`.

### Dependencies

Runtime: `numpy`, `fitsio`, `torch`
Dev: `pytest`

### Versioning & changelog

Version is in `py/adaptic/_version.py`. Changelog is tracked in `docs/changes.rst` in RST format. CI publishes to PyPI on `v*.*.*` tags via GitHub Actions trusted publishing.
