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
        curr_tid = 0
        for row in self.healpix_table:
            hpx = row['HEALPIX']
            srvy = row['SURVEY']
            prgrm = row['PROGRAM']
            dirname = self.datadir / "healpix" / srvy / prgrm
            dirname = dirname / str(hpx // 100) / str(hpx)
            coaddname = dirname / f"coadd-{srvy}-{prgrm}-{hpx}.fits"
            rrname = dirname / f"redrock-{srvy}-{prgrm}-{hpx}.fits"

            # Make the directory first if necessary, because
            # fitsio doesn't gracefully handle that.
            dirname.mkdir(exist_ok=True, parents=True)
            num_spec = row['NUMTARGETS']

            random_desi_coadd(dirname, coaddname, num_spec, self.rng, redshift_name=rrname,
                              targetid_start=curr_tid)
            curr_tid += num_spec

        self.total_spec = np.sum(self.healpix_table['NUMTARGETS'])

        tiles_file = Path(__file__).resolve().parent / "data" / "tiles-fake.fits"
        with fitsio.FITS(tiles_file) as h:
            self.tiles_table = h[1].read()

        # Generate the dummy directories and files corresponding to this table
        spec_per_petal = self.total_spec // np.sum(self.tiles_table['NUMPETALS'])
        curr_tid = 0
        for row in self.tiles_table:
            tileid = row['TILEID']
            night = row['LASTNIGHT']

            for petal in range(row['NUMPETALS']):
                dirname = self.datadir / "tiles" / "cumulative" / str(tileid) / str(night)
                coaddname = dirname / f"coadd-{petal}-{tileid}-thru{night}.fits"
                rrname = dirname / f"redrock-{petal}-{tileid}-thru{night}.fits"

                # Make the directory first if necessary, because
                # fitsio doesn't gracefully handle that.
                dirname.mkdir(exist_ok=True, parents=True)
                num_spec = spec_per_petal

                random_desi_coadd(dirname, coaddname, num_spec, self.rng, redshift_name=rrname,
                                targetid_start=curr_tid)
                curr_tid += spec_per_petal

        # Additional returns that are not fibermap columns but are returned.
        self.spectra_set = set(["MU", "SIGMA", "FLUX", "IVAR", "MASK"])
        self.batchsize = 13

    # We expect these tests to fail with StopIteration.
    # Normally you wouldn't iterate over it this way, you'd do, e.g., for
    # item in dataset which would handle that StopIteration gracefully.
    @unittest.expectedFailure
    def test_dataset_stop_loop_healpix(self):
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed)
        data_iter = iter(dataset)
        for _ in range(self.total_spec + 1):
            next(data_iter)

    @unittest.expectedFailure
    def test_dataset_stop_loop_healpix_dataloader(self):
        dataset = DESIDataset(self.datadir, self.healpix_table, seed=self.seed)
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)

    @unittest.expectedFailure
    def test_dataset_stop_loop_tiles(self):
        dataset = DESIDataset(self.datadir, self.tiles_table, seed=self.seed)
        data_iter = iter(dataset)
        for _ in range(self.total_spec + 1):
            next(data_iter)

    @unittest.expectedFailure
    def test_dataset_stop_loop_tiles_dataloader(self):
        dataset = DESIDataset(self.datadir, self.tiles_table, seed=self.seed)
        train_dl = iter(DataLoader(dataset, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl)

    def test_dataset_loop(self):
        # This test tests autolooping, so shouldn't ever fail out when it runs out of data.
        # Generic initial
        dataset_hp = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, autoloop=True)
        data_iter_hp = iter(dataset_hp)

        dataset_tiles = DESIDataset(self.datadir, self.tiles_table, seed=self.seed, autoloop=True)
        data_iter_tiles = iter(dataset_tiles)
        for _ in range(self.total_spec):
            next(data_iter_hp)
            next(data_iter_tiles)

        # If this fails we have a problem. We should be able to loop back around
        # to the first spec this way. So if this test fails here the autoloop
        # is failing.
        data_hp = next(data_iter_hp)
        data_tiles = next(data_iter_tiles)

        # This is probably silly to test since it constructs from DEFAULT_COLUMNS
        # but you never know what might happen. We cast to sets because
        # we don't care about order.
        expected = set(DESIDataset.DEFAULT_COLUMNS) | self.spectra_set
        self.assertEqual(set(data_hp.keys()), expected)
        self.assertEqual(set(data_tiles.keys()), expected)

        # Test iterating with a DataLoader to batch in the main process
        train_dl_hp = iter(DataLoader(dataset_hp, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_hp)
        del train_dl_hp

        train_dl_tiles = iter(DataLoader(dataset_tiles, batch_size=self.batchsize, num_workers=0))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_tiles)
        del train_dl_tiles


        # Test iterating with a DataLoader to batch with one subprocess
        train_dl_hp = iter(DataLoader(dataset_hp, batch_size=self.batchsize, num_workers=1))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_hp)
        del train_dl_hp

        train_dl_tiles = iter(DataLoader(dataset_tiles, batch_size=self.batchsize, num_workers=1))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_tiles)
        del train_dl_tiles

        # Test iterating with a DataLoader to batch with multiple subprocess
        train_dl_hp = iter(DataLoader(dataset_hp, batch_size=self.batchsize, num_workers=4))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_hp)
        del train_dl_hp

        train_dl_tiles = iter(DataLoader(dataset_tiles, batch_size=self.batchsize, num_workers=4))
        for _ in range((self.total_spec // self.batchsize) + 2):
            next(train_dl_tiles)
        del train_dl_tiles

    def test_change_columns(self):
        # First test adding extra columns
        extra_columns = ["MEAN_PSF_TO_FIBER_SPECFLUX", "STD_FIBER_RA", "STD_FIBER_DEC",
                         "MEAN_FIBER_DEC", "MEAN_FIBER_RA"]
        dataset_hp = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, extra_cols=extra_columns)
        data_iter_hp = iter(dataset_hp)

        dataset_tiles = DESIDataset(self.datadir, self.tiles_table, seed=self.seed, extra_cols=extra_columns)
        data_iter_tiles = iter(dataset_tiles)

        expected = set(DESIDataset.DEFAULT_COLUMNS) | self.spectra_set | set(extra_columns)
        data_hp = next(data_iter_hp)
        self.assertEqual(set(data_hp.keys()), expected)
        data_tiles = next(data_iter_tiles)
        self.assertEqual(set(data_tiles.keys()), expected)

        # Then test replacing the entire set of columns.
        dataset_hp = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, return_cols=extra_columns)
        data_iter_hp = iter(dataset_hp)

        dataset_tiles = DESIDataset(self.datadir, self.tiles_table, seed=self.seed, return_cols=extra_columns)
        data_iter_tiles = iter(dataset_tiles)

        expected = self.spectra_set | set(extra_columns)
        data_hp = next(data_iter_hp)
        self.assertEqual(set(data_hp.keys()), expected)
        data_tiles = next(data_iter_tiles)
        self.assertEqual(set(data_tiles.keys()), expected)


    def test_train_valid_split(self):
        # Test to ensure that all the data in the validation set is exclusive of the training set and fice versa
        ## Healpixel variables
        dataset_train_hp = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, train_frac=0.5, train_data=True, autoloop=True)
        train_iter_hp = iter(dataset_train_hp)

        dataset_valid_hp = DESIDataset(self.datadir, self.healpix_table, seed=self.seed, train_frac=0.5, train_data=False, autoloop=True)
        valid_iter_hp = iter(dataset_valid_hp)

        seen_train_tids_hp = []
        seen_valid_tids_hp = []

        ## Tile based variables
        dataset_train_tiles = DESIDataset(self.datadir, self.tiles_table, seed=self.seed, train_frac=0.5, train_data=True, autoloop=True)
        train_iter_tiles = iter(dataset_train_tiles)

        dataset_valid_tiles = DESIDataset(self.datadir, self.tiles_table, seed=self.seed, train_frac=0.5, train_data=False, autoloop=True)
        valid_iter_tiles = iter(dataset_valid_tiles)

        seen_train_tids_tiles = []
        seen_valid_tids_tiles = []

        for _ in range(self.total_spec):
            train = next(train_iter_hp)
            valid = next(valid_iter_hp)

            seen_train_tids_hp.append(train["TARGETID"])
            seen_valid_tids_hp.append(valid["TARGETID"])

            train = next(train_iter_tiles)
            valid = next(valid_iter_tiles)

            seen_train_tids_tiles.append(train["TARGETID"])
            seen_valid_tids_tiles.append(valid["TARGETID"])

        # These functionally check the same thing but its good to be sure.
        assert len(set(seen_train_tids_hp)) + len(set(seen_valid_tids_hp)) == len(set(seen_train_tids_hp + seen_valid_tids_hp))
        self.assertFalse(np.any(np.isin(seen_train_tids_hp, seen_valid_tids_hp)))
        self.assertFalse(np.any(np.isin(seen_valid_tids_hp, seen_train_tids_hp)))

        assert len(set(seen_train_tids_tiles)) + len(set(seen_valid_tids_tiles)) == len(set(seen_train_tids_tiles + seen_valid_tids_tiles))
        self.assertFalse(np.any(np.isin(seen_train_tids_tiles, seen_valid_tids_tiles)))
        self.assertFalse(np.any(np.isin(seen_valid_tids_tiles, seen_train_tids_tiles)))


    def tearDown(self):
        self.tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()
