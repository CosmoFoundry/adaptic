import os
import fitsio
import numpy as np
from numpy.lib.recfunctions import merge_arrays, append_fields # To join the fibermap and redshifts headers
from torch.utils.data import IterableDataset, get_worker_info

import hashlib
from pathlib import Path
from copy import deepcopy

class DESIDataset(IterableDataset):
    # class-level tuple of default columns to return
    DEFAULT_COLUMNS = (
        'TARGETID',
        'COADD_FIBERSTATUS',
        'TARGET_RA',
        'TARGET_DEC',
        'PMRA',
        'PMDEC',
        'REF_EPOCH',
        'OBJTYPE',
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
        'Z',
        'ZERR',
        'ZWARN',
        'SPECTYPE',
        'SUBTYPE',
        )

    def __init__(self, specprod_dir, summary_table=None, seed=123, shuffle_files=True,
                 transform=None, normalize=False, train_frac=None, train_data=True,
                 coadd_spectra=True, filter_func=None, autoloop=False,
                 extra_cols=None, return_cols=None):
        """
            Initialize the dataset object.

            Parameters
            ----------
            specprod_dir : str or :class:`~pathlib.Path`
                The base directory of the spectroscopic production. Can be in any location,
                as long as the specprod follows main DESI data release conventions.
                That is, the healpix coadded spectra are stored in
                {specprod_dir}/healpix/{survey}/{program}/{healpix // 100}/{healpix}

            summary_table : :class:`~numpy.array` or :class:`~astropy.table.Table`, optional
                A numpy record array, or alternatively, an astropy table if
                installed. If this table is for HEALPIX based coadds,
                at minimum needs to include the columns ["SURVEY", "PROGRAM",
                "HEALPIX", "NSIDE", "NUMTARGETS"]. If this table is for tile-based
                coadds, it must include at minimum the columns ["TILEID", "LASTNIGHT"].
                In both cases the table is allowed to contain additional
                columns that will be ignored. Optional, if not passed the Dataset
                will attempt to auto-discover the necessary file in the given `specprod_dir`.
                If passed, override any auto-discovery.

            seed : int, optional
                Seed to use for any randomness. Randomness is done through a
                numpy RNG object. Defaults to 123.

            shuffle_files : bool, optional
                Whether or not the summary_table should be randomly shuffled or not.
                Defaults to True. NOTE: Setting shuffle_files=False does not
                guarantee preservation of table order if this dataset
                is used with a pytorch data loader with num_workers > 1.

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

            autoloop : bool, optional
                If True, automatically reloop back to the start of the dataset
                once all spectra are loaded and exhausted. Otherwise terminate
                the iteration at the end of the dataset. Defaults to False.

            extra_cols : list of str, optional
                Additional column names to include in each yielded example dict,
                beyond the default set. Adaptic determines automatically whether
                each column comes from FIBERMAP or REDSHIFTS. Cannot be used
                together with return_cols.

            return_cols : list of str, optional
                Exact set of column names to include in each yielded example dict,
                replacing the default set entirely. Adaptic determines automatically
                whether each column comes from FIBERMAP or REDSHIFTS. Cannot be
                used together with extra_cols.
        """
        super(DESIDataset).__init__()
        self.base_dir = Path(specprod_dir)

        if summary_table is not None:
            self.summary = deepcopy(np.asarray(summary_table)) # Don't want to mutate the input
        else:
            # Try auto discover a healpix summary file.
            specprod = self.base_dir.name
            summary_loc = self.base_dir / f"healpix-{specprod}.fits"
            assert summary_loc.exists(), f"attempted auto discovery of {summary_loc}, but file not found!"

            with fitsio.FITS(summary_loc) as h:
                self.summary = h[1].read()

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._standardize_summary()
        self._known_missing_files = set()
        self.shuffle_files = shuffle_files
        if self.shuffle_files:
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

        self.coadd_spectra = coadd_spectra

        if train_frac is not None:
            if (train_frac < 0) or (train_frac > 1):
                raise ValueError("train_frac must be between 0 and 1!")

        self.train_frac = train_frac

        self._return_cols = self.DEFAULT_COLUMNS
        self.autoloop = autoloop

        if return_cols is not None and extra_cols is not None:
            raise ValueError("Specify at most one of extra_cols and return_cols, not both.")
        if return_cols is not None:
            if len(return_cols) == 0:
                raise ValueError("return_cols cannot be empty.")
            self._return_cols = tuple(return_cols)
        elif extra_cols is not None:
            self._return_cols = self.DEFAULT_COLUMNS + tuple(extra_cols)

    def __iter__(self):
        this_ids = self._balance_workers(worker_info=get_worker_info())

        # Use a while loop automatically handle the autoloop=True case,
        # and break at the end of one pass through if autoloop=False.
        j = 0
        num_reads = 0
        while True:
            row = self.summary[this_ids][j]
            filenames = self._filenames_from_row(row)
            if filenames is not None:    # could be None if files don't exist
                num_reads += 1
                self._load_and_coadd(filenames)

                for i in range(self._details.shape[0]):
                    # Parse the example as a dictionary
                    example = {k: self._details[k][i] for k in self._return_cols}
                    example['MU'] = self._mu[i]
                    example['SIGMA'] = self._sigma[i]
                    if self.coadd_spectra:
                        example['FLUX'] = self._flux[i, :]
                        example['IVAR'] = self._ivar[i, :]
                        example['MASK'] = self._mask[i, :]
                    else:
                        example['FLUX'] = {c: self._flux[c][i, :] for c in self._flux}
                        example['IVAR'] = {c: self._ivar[c][i, :] for c in self._ivar}
                        example['MASK'] = {c: self._mask[c][i, :] for c in self._mask}

                    if self.transform:
                        if self.coadd_spectra:
                            example['FLUX'] = self.transform(example['FLUX'])
                        else:
                            for c in self._flux.keys():
                                example['FLUX'][c] = self.transform(example['FLUX'][c])

                    yield example

            # Loop back to the start of the files at the end.
            j += 1
            if (j == len(self.summary[this_ids])):
                j = 0
                if num_reads == 0:
                    # something went wrong; looped through all options without finding anything to read
                    raise RuntimeError("Looped through files without finding any to read! Check that the summary table is correct and that the files exist.")
                if not self.autoloop:
                    break

    def _balance_workers(self, worker_info):
        """
            Determines the set of files that belong to this worker, if running with
            more than a single worker. Should only be called by DESIDataset.__iter__.
            Handles shuffling a worker's files if necessary.
        """
        if worker_info is not None:
            # If worker info is not none there are multiple workers.
            # If the length of the summary table is less than the number of workers
            # we have to handle that specially, so that every worker actually gets
            # files to load.
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            num_files = len(self.summary)

            if num_workers > num_files:
                if worker_id == 0:  print("Num Workers > Num Files. Will duplicate files across workers!")
                # Every worker gets every file, but each worker will shuffle them
                # differently. This maintains both randomness of the files plus
                # uniformity of access for each spectrum.
                this_ids = np.arange(num_files)

            else:
                ntargs_per_bin = np.zeros(num_workers)
                idcs_per_bin = {k: [] for k in range(num_workers)}

                # Fast and decently close to perfect algorithm to split the
                # files into num_workers "roughly equal sized" bins
                # based on the number of targets.
                # See Section 5 of Graham 1969 "Bounds on Multiprocessing Timing Anomalies"
                # This algorithm is entirely deterministic so every worker will
                # generate the exact same result, however, it does require an
                # ordered version of the summary table.
                sort_idcs = np.argsort(self.summary['NUMTARGETS'])[::-1]
                self.summary = self.summary[sort_idcs]
                for i, row in enumerate(self.summary):
                    # Find the lowest bin, add the next number of targets
                    add_bin = np.argmin(ntargs_per_bin)
                    ntargs_per_bin[add_bin] += row['NUMTARGETS']
                    idcs_per_bin[add_bin].append(i)

                # An array of indices that will *look* random but which, within
                # each worker, are actually the indices corresponding to largest
                # to smallest file size.
                this_ids = np.array(idcs_per_bin[worker_id])

            # We then reshuffle thise indices to regain randomness in file
            # size and location. Shuffle each worker with a different seed
            # to ensure relative to file size each worker is shuffled uniquely.
            # I.e. with the same seed and same number of files the shuffed order
            # will be the same in terms of file size, and this avoids that.
            # TODO: consider if the user passes shuffle=False, do they have a specific ordering in mind, and if so, reorder to match that order across the workers?
            if self.shuffle_files:
                worker_rng = np.random.default_rng(self.seed + worker_id)
                worker_rng.shuffle(this_ids)

        # If it's None we're in the main process so we can use the whole summary table for this process.
        else:
            this_ids = np.arange(len(self.summary))
        return this_ids

    def _standardize_summary(self):
        """
            Auto detect tiles-based vs. healpix-based summary table and update columns as needed.
            Modifies self.summary in-place. Should be called only by DESIDataset constructor.
        """
        # Confirm either HEALPIX or TILEID
        if 'HEALPIX' in self.summary.dtype.names:
            for col in ('SURVEY', 'PROGRAM'):
                assert col in self.summary.dtype.names, f'{col} missing from HEALPIX-based summary table'
        elif 'TILEID' in self.summary.dtype.names:
            for col in ('LASTNIGHT',):
                assert col in self.summary.dtype.names, f'{col} missing from TILEID-based summary table'
        else:
            raise ValueError(f"summary must have HEALPIX,SURVEY,PROGRAM or TILEID,LASTNIGHT columns; found {self.summary.dtype.names}")

        # Trim to unique SURVEY, PROGRAM, HEALPIX if needed;
        # tilepix.fits files map tiles:healpix and have multiple entries per healpix
        # Note: this section can be removed if we standardize on a different healpix summary
        #       file for each production
        if 'HEALPIX' in self.summary.dtype.names:
            ii = np.unique(self.summary[['SURVEY', 'PROGRAM', 'HEALPIX']], return_index=True)[1]
            self.summary = self.summary[ii]

        # Add default NUMTARGETS if needed; load-balancing may be off, but at least don't crash
        if 'NUMTARGETS' not in self.summary.dtype.names:
            numtargets = 500*np.ones(len(self.summary))
            self.summary = append_fields(self.summary, 'NUMTARGETS', numtargets, usemask=False)

        # promote bytestring SURVEY, PROGRAM columns to unicode columns
        description = self.summary.dtype.descr
        change_dtype = False
        for i, (name, dtype) in enumerate(description):
            if name in ('SURVEY', 'PROGRAM') and 'S' in dtype:
                change_dtype = True
                description[i] = (name, dtype.replace('S', 'U'))

        if change_dtype:
            self.summary = self.summary.astype(np.dtype(description))

        # if tiles-based (not healpix), expand to one row per PETAL and add NUMTARGETS=500 column
        if (('TILEID' in self.summary.dtype.names) and
            ('HEALPIX' not in self.summary.dtype.names) and
            ('PETAL' not in self.summary.dtype.names)
            ):
            summary = np.repeat(self.summary, 10)
            petal = np.arange(len(summary)) % 10
            self.summary = append_fields(summary, 'PETAL', petal, usemask=False)


    def _filenames_from_row(self, row):
        """
            Given the row of a summary table, determine the filenames of the associated
            coadd and redrock output files. Should only be called internally by a DESIDataset.

            Parameters
            ----------
            row : :class:`~numpy.array`
                A single row of the summary table stored in this DESIDataset.

            Returns
            -------
            list of :class:`~pathlib.Path`
                List of determined path names. The first element is the coadd
                path and the second is the redrock path.
        """
        if 'HEALPIX' in row.dtype.names:
            hpx = row['HEALPIX']
            srvy = row['SURVEY']
            prgrm = row['PROGRAM']
            fname = self.base_dir / "healpix" / srvy / prgrm
            fname = fname / str(hpx // 100) / str(hpx)
            coaddname = fname / f"coadd-{srvy}-{prgrm}-{hpx}.fits"
            rrname = fname / f"redrock-{srvy}-{prgrm}-{hpx}.fits"
        elif 'TILEID' in row.dtype.names:
            tileid = row['TILEID']
            night = row['LASTNIGHT']
            petal = row['PETAL']
            dirname = self.base_dir / "tiles" / "cumulative" / str(tileid) / str(night)
            coaddname = dirname / f"coadd-{petal}-{tileid}-thru{night}.fits"
            rrname = dirname / f"redrock-{petal}-{tileid}-thru{night}.fits"
            # check if files for this petal actually exist;
            # assume redrock exists if coadd does to minimize I/O
            if coaddname in self._known_missing_files:
                return None
            elif not os.path.isfile(coaddname):
                self._known_missing_files.add(coaddname)
                return None
        else:
            raise ValueError(f"row doesn't have HEALPIX or TILEID: {row=}")

        return [coaddname, rrname]

    def _load_and_coadd(self, fnames):
        """
            Given the a list of filenames corresponding to the coadd and the
            redrock files, load the spectra and their corresponding
            details. Should only be called internally by the DESIDataset,
            as most of the behaviour of loading is controlled
            by internal parameters, and the loaded elements are stored
            internally in the object instead of returned.

            Parameters
            ----------
            fnames : list of :class:`~pathlib.Path`
                A list of filenames to be loaded by this DESIDataset. The first
                elemnet is expected to be the filename of the coadd file, while
                the second is expected to be the redrock file.
        """
        with fitsio.FITS(fnames[0]) as h_coadd:
            # Reading the header should be fast, so this shouldn't be a problem.
            nspec = h_coadd['FIBERMAP'].read_header()['NAXIS2']

            with fitsio.FITS(fnames[1]) as h_rr:
                fmap_available = set(h_coadd['FIBERMAP'].get_colnames())
                rr_available   = set(h_rr['REDSHIFTS'].get_colnames())

                fmap_read, rr_read = [], []
                for col in self._return_cols:
                    if col in fmap_available:
                        fmap_read.append(col)
                    elif col in rr_available:
                        rr_read.append(col)
                    else:
                        raise ValueError(
                            f"Column {col!r} not found in FIBERMAP of {fnames[0]} "
                            f"or REDSHIFTS of {fnames[1]}"
                        )

                fmap   = h_coadd['FIBERMAP'].read(columns=fmap_read)
                rr_map = h_rr['REDSHIFTS'].read(columns=rr_read) if rr_read else None

            self._details = (merge_arrays([fmap, rr_map], asrecarray=True, flatten=True)
                             if rr_map is not None else fmap)

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
                fl = h_coadd[f'{c}_FLUX'].read()[keep_spec, :]
                iv = h_coadd[f'{c}_IVAR'].read()[keep_spec, :]
                m = h_coadd[f'{c}_MASK'].read()[keep_spec, :]

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

            # Determine train/validation split using a newly initialized rng
            # with a file specific rng so that the random choice is reproducible
            # every time the file is loaded.
            file_seed = hashlib.sha1((str(fnames[0]) + str(self.seed)).encode()).hexdigest()
            file_seed = int(file_seed, 16)
            file_rng = np.random.default_rng(file_seed)
            choice = file_rng.choice(idcs, size=int(nspec * self.train_frac), replace=False)

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
        """
            Set this DESIDataset to training mode, and return the training
            subset of each file.
        """
        self.is_train = True

    def set_validation(self):
        """
            Set this DESIDataset to validation mode, and return the validation
            subset of each file.
        """
        self.is_train = False

    def __copy__(self):
        return DESIDataset(specprod_dir=self.base_dir, summary_table=self.summary,
                           seed=self.seed, shuffle_files=False, # If shuffle is true, we shuffled the table already. We don't want to shuffle it again.
                           transform=self.transform, normalize=self.normalize,
                           train_frac=self.train_frac, train_data=self.is_train,
                           coadd_spectra=self.coadd_spectra, filter_func=self.filter_func,
                           autoloop=self.autoloop, return_cols=self._return_cols)
