import string
from pathlib import Path

import fitsio
import numpy as np

fibermap_dtype = np.dtype([('TARGETID', '>i8'), ('COADD_FIBERSTATUS', '>i4'),
                  ('TARGET_RA', '>f8'), ('TARGET_DEC', '>f8'), ('DESINAME', '<U22'),
                  ('PMRA', '>f4'), ('PMDEC', '>f4'), ('REF_EPOCH', '>f4'),
                  ('FA_TARGET', '>i8'), ('FA_TYPE', 'u1'), ('OBJTYPE', '<U3'),
                  ('SUBPRIORITY', '>f8'), ('OBSCONDITIONS', '>i4'),
                  ('RELEASE', '>i2'), ('BRICKNAME', '<U8'), ('BRICKID', '>i4'),
                  ('BRICK_OBJID', '>i4'), ('MORPHTYPE', '<U4'), ('EBV', '>f4'),
                  ('FLUX_G', '>f4'), ('FLUX_R', '>f4'), ('FLUX_Z', '>f4'),
                  ('FLUX_W1', '>f4'), ('FLUX_W2', '>f4'), ('FLUX_IVAR_G', '>f4'),
                  ('FLUX_IVAR_R', '>f4'), ('FLUX_IVAR_Z', '>f4'), ('FLUX_IVAR_W1', '>f4'),
                  ('FLUX_IVAR_W2', '>f4'), ('FIBERFLUX_G', '>f4'), ('FIBERFLUX_R', '>f4'),
                  ('FIBERFLUX_Z', '>f4'), ('FIBERTOTFLUX_G', '>f4'),
                  ('FIBERTOTFLUX_R', '>f4'), ('FIBERTOTFLUX_Z', '>f4'), ('MASKBITS', '>i2'),
                  ('SERSIC', '>f4'), ('SHAPE_R', '>f4'), ('SHAPE_E1', '>f4'),
                  ('SHAPE_E2', '>f4'), ('REF_ID', '>i8'), ('REF_CAT', '<U2'),
                  ('GAIA_PHOT_G_MEAN_MAG', '>f4'), ('GAIA_PHOT_BP_MEAN_MAG', '>f4'),
                  ('GAIA_PHOT_RP_MEAN_MAG', '>f4'), ('PARALLAX', '>f4'), ('PHOTSYS', '<U1'),
                  ('PRIORITY_INIT', '>i8'), ('NUMOBS_INIT', '>i8'), ('DESI_TARGET', '>i8'),
                  ('BGS_TARGET', '>i8'), ('MWS_TARGET', '>i8'), ('SCND_TARGET', '>i8'),
                  ('PLATE_RA', '>f8'), ('PLATE_DEC', '>f8'), ('COADD_NUMEXP', '>i2'),
                  ('COADD_EXPTIME', '>f4'), ('COADD_NUMNIGHT', '>i2'), ('COADD_NUMTILE', '>i2'),
                  ('MEAN_DELTA_X', '>f4'), ('RMS_DELTA_X', '>f4'), ('MEAN_DELTA_Y', '>f4'),
                  ('RMS_DELTA_Y', '>f4'), ('MEAN_PSF_TO_FIBER_SPECFLUX', '>f4'),
                  ('MEAN_FIBER_RA', '>f8'), ('STD_FIBER_RA', '>f4'), ('MEAN_FIBER_DEC', '>f8'),
                  ('STD_FIBER_DEC', '>f4'), ('MIN_MJD', '>f8'), ('MAX_MJD', '>f8'),
                  ('MEAN_MJD', '>f8')])

redshifts_dtype = np.dtype([('TARGETID', '>i8'), ('Z', '>f8'), ('ZERR', '>f8'),
                            ('ZWARN', '>i8'), ('CHI2', '>f8'), ('COEFF', '>f8', (10,)),
                            ('FITMETHOD', '<U4'), ('NPIXELS', '>i8'), ('SPECTYPE', '<U6'),
                            ('SUBTYPE', '<U20'), ('NCOEFF', '>i8'), ('DELTACHI2', '>f8')])

wmin, wmax, wdelta = 3600, 9824, 0.8
desi_wave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)

# What subsection of the full wavelength grid is covered by each of the
# three DESI cameras.
cam_slice = {'B': slice(0, 2751), 'R': slice(2700, 5026), 'Z': slice(4900, 7781)}


def random_structured_array(num_spec, rng, dtype):
    arr = np.empty(num_spec, dtype=dtype)

    for col in dtype.names:
        # Store this so we don't have to keep accessing it
        dt = dtype[col]

        # Single letter describing type
        # https://numpy.org/devdocs/reference/generated/numpy.dtype.kind.html
        kind = dt.kind
        if kind in ["i", "u"]: # Integer
            arr[col] = rng.uniform(0, 256, size=num_spec)
        elif kind in ["f"]: # Float
            if dt.shape != ():
                arr[col] = rng.random(size=(num_spec, dt.shape)) * 256
            else:
                arr[col] = rng.random(size=num_spec) * 256
        elif kind in ["U", "S"]: # Strings
            U_idx = dt.str.index("U")
            # Find out how many characters this string is
            num_chars = int(dt.str[(U_idx + 1):])

            # Sample an array of num_spec by num_chars, sampling from all uppercase
            # letters. Then we'll use a list comprehension to concatenate the
            # chars into single strings along that axis.
            strings = rng.choice(list(string.ascii_uppercase), size=(num_spec, num_chars))
            arr[col] = np.array(["".join(row) for row in strings], dtype=arr[col].dtype)

    return arr

def random_desi_coadd(path, coadd_name, num_spec, rng, redshift_name=None, targetid_start=0):
    fmap = random_structured_array(num_spec, rng, dtype=fibermap_dtype)

    # We need these to be unique.
    fmap["TARGETID"] = np.arange(len(fmap)) + targetid_start

    with fitsio.FITS(path / coadd_name, "rw") as h:
        h.write(fmap, extname="FIBERMAP")

        # They don't really need be in the same order as the real data
        # because we index them by name, but it is better to be so.
        for cam in cam_slice.keys():
            cam_wave = desi_wave[cam_slice[cam]]
            h.write(cam_wave, extname=f"{cam}_WAVELENGTH")

            for arr_type in ["FLUX", "IVAR", "MASK"]:
                if arr_type == "MASK": # Needs to be integers
                    data = rng.integers(0, 8, size=(num_spec, len(cam_wave)))
                else: # Flux and IVAR can be floats
                    data = rng.random(size=(num_spec, len(cam_wave)))
                h.write(data, extname=f"{cam}_{arr_type}")

    if redshift_name is not None:
        with fitsio.FITS(path / redshift_name, "rw") as h:
            redshifts = random_structured_array(num_spec, rng, dtype=redshifts_dtype)
            redshifts["TARGETID"] = fmap["TARGETID"]
            h.write(redshifts, extname="REDSHIFTS")
            # We don't need any of the other HDUs in the redshifts file.

uniqpix_summary_dtype = np.dtype([('UNIQPIX', '>i8'), ('NTARGETS', '>i4')])

def make_uniqpix_table(specprod_dir, survey, program, uniqpix_values, ntargets_values):
    """
        Write a minimal ``uniqpix-{survey}-{program}.fits`` summary file under
        ``{specprod_dir}/spectra/{survey}/{program}/``, suitable for use as
        input to :func:`~adaptic.desi.find_and_concat_uniqpix_tables`.

        Parameters
        ----------
        specprod_dir : str or :class:`~pathlib.Path`
            The base directory of the spectroscopic production.  The file is
            written to ``{specprod_dir}/spectra/{survey}/{program}/``; any
            missing directories are created automatically.

        survey : str
            Survey identifier (e.g. ``'main'``, ``'sv1'``).

        program : str
            Program identifier (e.g. ``'dark'``, ``'bright'``).

        uniqpix_values : array-like of int
            UNIQPIX pixel indices to store in the table.

        ntargets_values : array-like of int
            Number of targets per pixel, parallel to ``uniqpix_values``.

        Returns
        -------
        :class:`~pathlib.Path`
            Absolute path to the written FITS file.
    """
    dirpath = Path(specprod_dir) / "spectra" / survey / program
    dirpath.mkdir(parents=True, exist_ok=True)
    fname = dirpath / f"uniqpix-{survey}-{program}.fits"
    tbl = np.empty(len(uniqpix_values), dtype=uniqpix_summary_dtype)
    tbl['UNIQPIX'] = uniqpix_values
    tbl['NTARGETS'] = ntargets_values
    with fitsio.FITS(fname, 'rw') as h:
        h.write(tbl)
    return fname
