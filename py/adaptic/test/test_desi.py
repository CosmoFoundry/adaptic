from pathlib import Path
import tempfile
import unittest

import fitsio
import numpy as np

from .util import *
from adaptic.desi import DESIDataset


class TestDESIDataset(unittest.TestCase):
    def setUp(self):
        # We will generate a dummy spectrum file on the fly.
        # The data is not important, really, just the file format
        # (although we will test that we actually loaded the right data)
        self.tempdir = tempfile.TemporaryDirectory()
        self.datadir = Path(self.tempdir.name)

        self.seed = 91701
        self.rng = np.random.default_rng(self.seed)

        healpix_file = Path(__file__).resolve().parent / "data" / "healpix_table.fits"
        with fitsio.FITS(healpix_file) as h:
            self.healpix_table = h[1].read()

        # Generate the dummy directories and files corresponding to this table
        for row in self.healpix_table:
            hpx = row['HEALPIX']
            srvy = row['SURVEY']
            prgrm = row['PROGRAM']
            fname = self.datadir / "healpix" / srvy / prgrm
            fname = fname / str(hpx // 100) / str(hpx)
            coaddname = fname / f"coadd-{srvy}-{prgrm}-{hpx}.fits"
            rrname = fname / f"redrock-{srvy}-{prgrm}-{hpx}.fits"

            # Make the directory first if necessary, because
            # fitsio doesn't gracefully handle that.
            fname.mkdir(exist_ok=True, parents=True)
            num_spec = row['NUMTARGETS']

            random_desi_coadd(fname, coaddname, num_spec, self.rng, redshift_name=rrname)

        self.total_spec = np.sum(self.healpix_table["NUMTARGETS"])

        # Additional returns that are not fibermap columns but are returned.
        self.spectra_set = set(["MU", "SIGMA", "FLUX", "IVAR", "MASK"])

    # We expect this one to fail with StopIteration.
    @unittest.expectedFailure
    def test_dataset_stop_loop(self):
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed)
        data_iter = iter(dataset)
        for _ in range(self.total_spec + 1):
            next(data_iter)

    def test_dataset_loop(self):
        # Generic initial
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, autoloop=True)
        data_iter = iter(dataset)
        for _ in range(self.total_spec):
            next(data_iter)

        # If this fails we have a problem. We should be able to loop back around
        # to the first spec this way. So if this test fails here the autoloop
        # is failing.
        data = next(data_iter)

        # This is probably silly to test since it constructs from DEFAULT_COLUMNS
        # but you never know what might happen. We cast to sets because
        # we don't care about order.
        expected = set(DESIDataset.DEFAULT_COLUMNS) | self.spectra_set
        self.assertEqual(set(data.keys()), expected)

    def tearDown(self):
        self.tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()
