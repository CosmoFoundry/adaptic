from pathlib import Path
import tempfile
import unittest

from .util import *
from adaptic.desi import DESIDataset

from torch.utils.data import DataLoader
import fitsio
import numpy as np

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
        self.batchsize = 13

    # We expect this one to fail with StopIteration.
    # Normally you wouldn't iterate over it this way, you'd do, e.g., for
    # item in dataset which would handle that StopIteration gracefully.
    @unittest.expectedFailure
    def test_dataset_stop_loop(self):
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed)
        data_iter = iter(dataset)
        for _ in range(self.total_spec + 1):
            next(data_iter)

    @unittest.expectedFailure
    def test_dataset_stop_loop_dataloader(self):
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed)
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)

    def test_dataset_loop(self):
        # This test tests autolooping, so shouldn't ever fail out when it runs out of data.

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

        # Test iterating with a DataLoader to batch in the main process
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)
        del train_dl

        # Test iterating with a DataLoader to batch with one subprocess
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=1))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)
        del train_dl

        # Test iterating with a DataLoader to batch with multiple subprocess
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=4))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)
        del train_dl

    def test_change_columns(self):
        # First test adding extra columns
        extra_columns = ["MEAN_PSF_TO_FIBER_SPECFLUX", "STD_FIBER_RA", "STD_FIBER_DEC",
                         "MEAN_FIBER_DEC", "MEAN_FIBER_RA"]
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, extra_cols=extra_columns)
        data_iter = iter(dataset)

        expected = set(DESIDataset.DEFAULT_COLUMNS) | self.spectra_set | set(extra_columns)
        data = next(data_iter)
        self.assertEqual(set(data.keys()), expected)

        # Then test replacing the entire set of columns.
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, return_cols=extra_columns)
        data_iter = iter(dataset)

        expected = self.spectra_set | set(extra_columns)
        data = next(data_iter)
        self.assertEqual(set(data.keys()), expected)

    def tearDown(self):
        self.tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()
