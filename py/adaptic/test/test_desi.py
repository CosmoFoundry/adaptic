from pathlib import Path
import tempfile
import unittest

import fitsio
import numpy as np

from .util import *


class TestDESIDataset(unittest.TestCase):
    def setUp(self):
        # We will generate a dummy spectrum file on the fly.
        # The data is not important, really, just the file format
        # (although we will test that we actually loaded the right data)
        self.tempdir = tempfile.TemporaryDirectory()
        self.datadir = Path(self.tempdir.name)

        self.rng = np.random.default_rng(91701)

        healpix_file = Path("./data/healpix_table.fits")
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



    def tearDown(self):
        self.tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()
