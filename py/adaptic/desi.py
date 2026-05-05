import fitsio
import numpy as np
from numpy.lib.recfunctions import merge_arrays # To join the fibermap and redshifts headers
from torch.utils.data import IterableDataset, get_worker_info

from pathlib import Path
from copy import deepcopy

class DESIDataset(IterableDataset):
    def __init__(self, specprod_dir, summary_table, seed=123, shuffle_files=True,
                 transform=None, normalize=False, train_frac=None, train_data=True,
                 coadd_spectra=True, filter_func=None):
        """
            Initialize the dataset object.

            Parameters
            ----------
            specprod_dir : str or :class:`~pathlib.Path`
                The base directory of the spectroscopic production. Can be in any location,
                as long as the specprod follows main DESI data release conventions.
                That is, the healpix coadded spectra are stored in
                {specprod_dir}/healpix/{survey}/{program}/{healpix // 100}/{healpix}

            summary_table : :class:`~numpy.array` or :class:`~astropy.table.Table`
                A numpy record array, or alternatively, an astropy table if
                installed. At minimum needs to include the columns ["SURVEY", "PROGRAM",
                "HEALPIX", "NSIDE", "NUMTARGETS"], although can contain additional
                columns that will be ignored.

            seed : int, optional
                Seed to use for any randomness. Randomness is done through a
                numpy RNG object. Defaults to 123.

            shuffle_files : bool, optional
                Whether or not the summary_table needs to be randomly shuffled or not.
                Defaults to True. NOTE: Table order is not preserved if
                this dataset is used with a pytorch data loader with num_workers > 1.
                In that case, the rows are split such that each worker gets approximately
                the same amount of spectra, without regard for file ordering.

            transform : callable, optional
                A pytorch transform object to apply to the output FLUX. Defaults
                to None, which applies no transform.

            normalize : bool, optional
                Whether or not we should normalize the spectral data to have
                a weighted mean of zero and a weighted standard deviation of 1.
                Normalization is done using inverse variance weighting. Defaults to False.

            train_frac : float, optional
                If not None, the fraction of the dataset to use as training
                data. The data loaded will be split randomly, using the given
                seed. Whether this specific dataset object returns the
                fraction of the data denoted as train versus validation
                is dependent on the train_data parameter. Defaults to None,
                which returns everything. NOTE: Best practice if using train_frac
                != None is to use the same seed for both the training and
                validation Dataset objects, to ensure that the two correctly
                keep out the same data.

            train_data : bool, optional
                Whether this dataset represents training data or validation data.
                If True, return the given fraction of data defined by
                train_frac. If False, return the remaining fraction of data.

            coadd_spectra : bool, optional
                If True, coadd the spectra on loading. If False, do not
                coadd the spectra, and instead return the data as dictionaries
                of the individual cameras.
                Defaults to True, which does the coaddition. NOTE: Normalizing
                is not currently supported when coadd_spectra is False.

            filter_func : callable, optional
                A filtering function that operates on the spectra FIBERMAP,
                determining which spectra to return (or not). The filter
                function should take in a numpy rec array (the fibermap) and
                return a boolean array of items to return. For example,
                filter could be a function that checks DESITARGET and only
                returns science spectra. Defaults to None, which returns
                all spectra in every file.
        """
        super(DESIDataset).__init__()
        self.base_dir = Path(specprod_dir)
        self.summary = deepcopy(summary_table) # Don't want to mutate the input

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if shuffle_files:
            self.rng.shuffle(self.summary)

        # Nominally used for things like transforming to tensor,
        # So that the dataset object can handle all the transforms on the fly.
        self.transform = transform

        self.filter_func = filter_func

        wmin, wmax, wdelta = 3600, 9824, 0.8
        self.desi_wave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)

        # What subsection of the full wavelength grid is covered by each of the
        # three DESI cameras.
        self.cam_slice = {'B': slice(0, 2751), 'R': slice(2700, 5026), 'Z': slice(4900, 7781)}

        # Used to store values after loading each spectrum file.
        self._flux = None
        self._ivar = None
        self._mask = None
        self._details = None

        # Normaliation/standardization variables.
        self._mu = None
        self._std = None

        # Whether or not to normalize the spectra to mean 0, std_dev = 1.
        # this is done with a weighted normalization (i.e. weighted means etc)
        self.normalize = normalize
        self.is_train = train_data

        # TODO: think of a better way to handly not coadding things.
        self.coadd_spectra = coadd_spectra

        if train_frac is not None:
            assert (train_frac >= 0) and (train_frac <= 1), "train_frac must be between 0 and 1!"

        self.train_frac = train_frac

        self._return_cols = ['TARGETID',
                            'COADD_FIBERSTATUS',
                            'TARGET_RA',
                            'TARGET_DEC',
                            'PMRA',
                            'PMDEC',
                            'REF_EPOCH',
                            # 'FA_TARGET',
                            # 'FA_TYPE',
                            'OBJTYPE',
                            # 'SUBPRIORITY',
                            # 'OBSCONDITIONS',
                            # 'RELEASE',
                            # 'BRICKNAME',
                            # 'BRICKID',
                            # 'BRICK_OBJID',
                            # 'MORPHTYPE',
                            'EBV',
                            'FLUX_G',
                            'FLUX_R',
                            'FLUX_Z',
                            'FLUX_W1',
                            'FLUX_W2',
                            'FLUX_IVAR_G',
                            'FLUX_IVAR_R',
                            'FLUX_IVAR_Z',
                            'FLUX_IVAR_W1',
                            'FLUX_IVAR_W2',
                            'FIBERFLUX_G',
                            'FIBERFLUX_R',
                            'FIBERFLUX_Z',
                            'FIBERTOTFLUX_G',
                            'FIBERTOTFLUX_R',
                            'FIBERTOTFLUX_Z',
                            "Z",
                            "ZERR",
                            "ZWARN",
                            "SPECTYPE",
                            "SUBTYPE"]

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is not None:
            # IF worker info is not none there are multiple workers.
            # If the length of the summary table is less than the number of workers
            #  we have to handle that specially, so that every worker actually gets
            # files to load.
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            num_files = len(self.summary)

            if num_workers > num_files:
                if worker_id == 0:  print("Num Workers > Num Files. Will duplicate files across workers!")

                # This worker gets worker_id % num_files, which seems reasonably
                # the most straightforward decision on what file to keep.
                this_ids = np.asarray([worker_id % num_files])

            else:
                ntargs_per_bin = np.zeros(num_workers)
                idcs_per_bin = {k: [] for k in range(num_workers)}

                # Fast and decently close to perfect algorithm to split the
                # files into num_workers "roughly equal sized" bins
                # based on the number of targets.
                # See Section 5 of Graham 1969 "Bounds on Multiprocessing Timing Anomalies"
                # for more details.
                # This algorithm is entirely
                # deterministic so every worker should generate the exact same
                # result, however, it does require an ordered version of the summary
                # plot. We will shuffle the indices allocated to each worker
                # later to regain the randomnes across the sky.
                sort_idcs = np.argsort(self.summary["NUMTARGETS"])[::-1]
                self.summary = self.summary[sort_idcs]
                for i, row in enumerate(self.summary):
                    # Find the lowest bin, add the next number of targets
                    add_bin = np.argmin(ntargs_per_bin)
                    ntargs_per_bin[add_bin] += row["NUMTARGETS"]
                    idcs_per_bin[add_bin].append(i)

                this_ids = np.array(idcs_per_bin[worker_id])
                # Interestingly since each worker has its own rng  with the same seed
                # they'll shufflle their idcs in the same way, maintaining rough parity
                # in file size across workers.
        # If it's None we're in the main process so we can use the whole summary table for this process.
        else:
            this_ids = np.arange(len(self.summary))

        # Loop forver.
        j = 0
        while True:
            row = self.summary[this_ids][j]
            # print(f"{worker_info.id}, {row}")  # Left for debugging.
            self.load_and_coadd(self._filenames_from_row(row))

            for i in range(self._details.shape[0]):
                # Parse the example as a dictionary
                example = {k: self._details[k][i] for k in self._return_cols}
                example["MU"] = self._mu[i]
                example["SIGMA"] = self._sigma[i]
                if self.coadd_spectra:
                    example["FLUX"] = self._flux[i, :]
                    example["IVAR"] = self._ivar[i, :]
                    example["MASK"] = self._mask[i, :]
                else:
                    example["FLUX"] = {c: self._flux[c][i, :] for c in self._flux}
                    example["IVAR"] = {c: self._ivar[c][i, :] for c in self._ivar}
                    example["MASK"] = {c: self._mask[c][i, :] for c in self._mask}

                if self.transform:
                    if self.coadd_spectra:
                        example["FLUX"] = self.transform(example["FLUX"])
                    else:
                        for c in self._flux.keys():
                            example["FLUX"][c] = self.transform(example["FLUX"][c])

                yield np.int64(example["TARGETID"]), example

            # Loop back to the start of the files at the end.
            j +=1
            if j == len(self.summary[this_ids]):
                j = 0

    def _filenames_from_row(self, row):
        hpx = row["HEALPIX"]
        srvy = row["SURVEY"]
        prgrm = row["PROGRAM"]
        fname = self.base_dir / "healpix" / srvy / prgrm
        fname = fname / str(hpx // 100) / str(hpx)
        coaddname = fname / f"coadd-{srvy}-{prgrm}-{hpx}.fits"
        rrname = fname / f"redrock-{srvy}-{prgrm}-{hpx}.fits"
        return [coaddname, rrname]

    def load_and_coadd(self, fnames):
        with fitsio.FITS(fnames[0]) as h_coadd:
            # Reading the header should be fast, so this shouldn't be a problem.
            nspec = h_coadd["FIBERMAP"].read_header()["NAXIS2"]

            fmap = h_coadd["FIBERMAP"].read(columns=self._return_cols[:-5]) # The last 5 columns are from the redrock file.

            with fitsio.FITS(fnames[1]) as h_rr:
                # We don't need to load all the columns, especially not COEFF which is quite large.
                rr_cols = self._return_cols[-5:]
                rr_map = h_rr["REDSHIFTS"].read(columns=rr_cols)
            self._details = merge_arrays([fmap, rr_map], asrecarray=True, flatten=True)

            # self._details = fmap

            if self.filter_func is not None:
                keep_spec = self.filter_func(fmap)
            else:
                keep_spec = np.ones(nspec, dtype=bool)

            nkeep = np.sum(keep_spec)
            self._details = self._details[keep_spec] # Don't forget to trim the details.

            # If we don't coadd we'll store dictionary of the individual cameras.
            if self.coadd_spectra:
                flux = np.zeros((nkeep, len(self.desi_wave)), dtype=np.float32)
                ivar = np.zeros_like(flux, dtype=np.float32)
                mask = np.zeros_like(flux, dtype=np.int32)
            else:
                flux = {c: np.zeros((nkeep, len(self.desi_wave[self.cam_slice[c]])), dtype=np.float32) for c in self.cam_slice.keys()}
                ivar = {c: np.zeros_like(flux[c], dtype=np.float32) for c in self.cam_slice.keys()}
                mask = {c: np.zeros_like(flux[c], dtype=np.int32) for c in self.cam_slice.keys()}


            # Fast ish coadd cameras because we're going to exploit
            # the fact that we already know what overlaps what.
            for c in self.cam_slice.keys():
                fl = h_coadd[f"{c}_FLUX"].read()[keep_spec, :]
                iv = h_coadd[f"{c}_IVAR"].read()[keep_spec, :]
                m = h_coadd[f"{c}_MASK"].read()[keep_spec, :]

                # Extremely basic ivar weighted coadd
                if self.coadd_spectra:
                    flux[:, self.cam_slice[c]] += fl * iv #* mask
                    ivar[:, self.cam_slice[c]] += iv #* mask
                    mask[:, self.cam_slice[c]] += m #* mask
                else:
                    flux[c] = fl
                    ivar[c] = iv
                    mask[c] = m > 0

            nz = ivar != 0
            # We set these if we do any normalization, otherwise they remain at
            # 0 and 1. Leaving them at 0 and 1 means if we always return it, and the
            # user always uses them, they get the right answer whether they set
            # normalize to true or false.
            self._mu = np.zeros(nspec, dtype=np.float32)
            self._sigma = np.ones(nspec, dtype=np.float32)
            # Handle the normalization/standardization and the coadding.
            if self.coadd_spectra:
                if self.normalize:
                    flux_bar = np.sum(flux, axis=1) / np.sum(ivar, axis=1) # Weighted Average
                    self._mu = flux_bar
                    flux[nz] /= ivar[nz]
                    flux -= flux_bar[:, None] # Zero the mean

                    std = np.sqrt(np.sum(ivar * (flux) ** 2, axis=1) / np.sum(ivar, axis=1))
                    self._sigma = std
                    # If every pixel is masked we don't bother returning
                    # it, the flux is going to be nonsense anyway.
                    dont_return = np.sum(ivar, axis=1) == 0 # Calculate this before rescaling ivar.

                    flux /= std[:, None] # St. Dev -> 1
                    ivar *= std[:, None] ** 2 # Rescale ivar to match the new flux values.

                    flux = flux[~dont_return]
                    ivar = ivar[~dont_return]
                    mask = mask[~dont_return]
                    self._mu = self._mu[~dont_return]
                    self._sigma = self._sigma[~dont_return]
                else: # Don't forget to divide out the ivar when not normalizing
                    flux[nz] /= ivar[nz]
            # Store data in the object
            self._flux = flux
            self._ivar = ivar

            # Mask is normally an array of integers where each integer
            # is a bit set of reasons to mask. In our case we will consider
            # anything with Mask > 0 (any integer to be "it should be masked")
            if self.coadd_spectra:
                # For "dont_coadd" we already set the mask to be a boolean array
                # in the dictionaries since no coaddition is done.
                self._mask = mask > 0
            else:
                self._mask = mask

        if self.normalize:
            self._details = self._details[~dont_return]

        # Chunk out training data if necessary.
        if self.train_frac is not None:
            # Will select a random train_frac percentage of indices to save.
            nspec = self._details.shape[0]

            idcs = np.arange(nspec)
            choice = self.rng.choice(idcs, size=int(nspec * self.train_frac), replace=False)

            keep_idcs = np.isin(idcs, choice)
            if not self.is_train:
                keep_idcs = ~keep_idcs

            if self.coadd_spectra:
                self._flux = self._flux[keep_idcs]
                self._ivar = self._ivar[keep_idcs]
                self._details = self._details[keep_idcs]
            else:
                for c in self.cam_slice.keys():
                    self._flux[c] = self._flux[c][keep_idcs]
                    self._ivar[c] = self._ivar[c][keep_idcs]
                    self._mask[c] = self._mask[c][keep_idcs]

    def set_train(self):
        self.is_train = True

    def set_validation(self):
        self.is_train = False

    def __copy__(self):
        return DESIDataset(specprod_dir=self.base_dir, summary_table=self.summary,
                           seed=self.seed, shuffle_files=False, # If shuffle is true, we shuffled the table already. We don't want to shuffle it again.
                           transform=self.transform, normalize=self.normalize,
                           train_frac=self.train_frac, train_data=self.is_train,
                           coadd_spectra=self.coadd_spectra, filter_func=self.filter_func)
