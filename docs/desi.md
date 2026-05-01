`adaptic` is a performant data wrapper designed to be extensible for a multitude of cosmology datasets, enabling their use in ML and AI workflows through pytorch. In this file we document how to use the `desi` wrapper.

`adaptic` wrappers are designed to be used in two separate ways:
1) Installation into an environment through pypi (`pip install adaptic`) or through this repository (`pip install .`).
2) Copy pasting `adaptic.desi` into your project.

The only difference between the two install methods for the purposes of this documentation is the import method. If installed through pip, use
```python
from adaptic.desi import DESIDataset
```
otherwise use
```python
from desi import DESIDataset
```

The `DESIDataset` object inherits from the pytorch [IterableDataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), which means it streams the data from disk to your ML training loop. This Dataset object can then be passed to a pytorch `DataLoader` to handle batching and parallelism.

In this document we will first document a few useful object parameters before providing a minimum viable example demonstrating the wrapper's use.

## Picking Data to Load
The `DESIDataset` has a variety of optional arguments, but two *required* arguments: `specprod_dir` and `summary_table`. These two arguments tell the dataset where to load data from, and what data to load, respectively.

The `specprod_dir` can be any directory, so long as the data is stored according to the desi data model. `DESIDataset` currently only supports healpix based coadded spectra, so expects all data to live in directories of the form `{specprod_dir}/healpix/{survey}/{program}/{healpix // 100}/{healpix}`.

The `summary_table` should be a numpy rec array (or optionally, for the astronomy familiar users, an astropy table) that provides the columns `["SURVEY", "PROGRAM", "HEALPIX", "NSIDE", "NUMTARGETS"]`. Each file to be loaded should correspond to a single row in the summary_table. The necessary columns (`"SURVEY", "PROGRAM", "HEALPIX"`) are used to generate file paths to each spectra file, while `"NUMTARGETS"` is used to balance the worker loads when running with more than 1 parallel worker for loading.

**Note:** The `summary_table` does not need to contain *every* spectra file in the `specprod_dir`, and the `DESIDataset` will only load files represented by a row in the `summary_table`. The end user can preemptively subselect while files to load by trimming out the corresponding rows of the `summary_table`. For example, one could use only main survey spectra by generating a summary table with only the rows where `tbl[SURVEY] == main`.


## Splitting Data by Training or Validation

The `DESIDataset` provides two arguments that control the training/validation split: `train_data` and `train_frac`:

- `train_data` is a boolean that indicates that this specific instantiation of the `DESIDataset` object represents training data (if True) or validation (if False).
- `train_frac` is a float value (between 0 and 1) that determines what fraction of each file is to be considered training data.

Broadly the internal logic of the `DESIDataset` is:
- If `train_frac` is not provided, return all spectra in every file
- If  `train_frac` is passed, randomly subdivide each file upon load and designate `train_frac` of the spectra as training spectra and the remainder as validation. If `train_data` is True, return the training selection, if not then return the validation selection.

**Note:** In order to avoid duplicating spectra between the training and validation sets, make sure you initialize the training and validation `DESIDataset`s with the exact same parameters, especially the `train_frac` and `seed` values. For example:
```python
specprod_dir = {dir here}
data_table = {table here}
train_frac = 0.7
dataset_train = DESIDataset(specprod_dir=specprod_dir, summary_table=data_table, seed=91701, train_frac=train_frac, train_data=True)
dataset_valid = DESIDataset(specprod_dir=specprod_dir, summary_table=data_table, seed=91701, train_frac=train_frac, train_data=False)
```

Since the `DESIDataset` will only load spectra files included in `data_table`, an alternative way to generate a training and validation split is to create two mutually exlcusive summary tables, and define the training and validation set by which *entire* file is included in each:
```python
dataset_train = DESIDataset(specprod_dir=specprod_dir, summary_table=training_table)
dataset_valid = DESIDataset(specprod_dir=specprod_dir, summary_table=validation_table)
```

## Filtering Spectra
The `DESIDataset` provides some functionality to subselect spectra based on some data model based criterion. This allows the subselection to be done upon loading, before batching, ensuring that every batch is the same size while also providing the benefit that the subselection is parallelized over files.

This filtering is done through the `filter_func` argument, which must be a callable which acts upon the table of columns defined by the fibermap + redrock columns (see Datamodel below for a full list of columns) and returns a numpy boolean array where the element is `True` if that spectra should be returned or `False` otherwise.

The following sample function subselects only actual science objects, removing sky spectra from the dataset.
```python
def get_science_only(fmap):
    tgt_type = fmap["OBJTYPE"]
    return (tgt_type == "TGT")
```

## Datamodel
Calling `next()` on a `DESIDataset` returns a (targetid, dictionary) pair, where the keys of the dictionary are the original column names, and each value is the value of that column for that spectrum. The targetid is a unique targetid for each spectrum defined in DESI.

If the dataset is used in conjunction with a `DataLoader` to perform batching, PyTorch will automatically stack each dictionary value. For example, a DESI spectrum has 7781 pixels, and is returned as `data["FLUX"]`. If used in a DataLoader, `data["FLUX"]` is a (batchsize, 7781) array of spectra.

The following is a list of returned values for a single spectrum:

| NAME               | TYPE           |
| ------------------ | -------------- |
| FLUX               | FLOAT32[7781]  |
| IVAR               | FLOAT32[7781]  |
| MASK               | BOOL[7781]     |
| TARGETID           | INT64          |
| COADD_FIBERSTATUS  | INT32          |
| TARGET_RA          | FLOAT64        |
| TARGET_DEC         | FLOAT64        |
| PMRA               | FLOAT32        |
| PMDEC              | FLOAT32        |
| REF_EPOCH          | FLOAT32        |
| OBJTYPE            | CHAR[3]        |
| EBV                | FLOAT32        |
| FLUX_G             | FLOAT32        |
| FLUX_R             | FLOAT32        |
| FLUX_Z             | FLOAT32        |
| FLUX_W1            | FLOAT32        |
| FLUX_W2            | FLOAT32        |
| FLUX_IVAR_G        | FLOAT32        |
| FLUX_IVAR_R        | FLOAT32        |
| FLUX_IVAR_Z        | FLOAT32        |
| FLUX_IVAR_W1       | FLOAT32        |
| FLUX_IVAR_W2       | FLOAT32        |
| FIBERFLUX_G        | FLOAT32        |
| FIBERFLUX_R        | FLOAT32        |
| FIBERFLUX_Z        | FLOAT32        |
| FIBERTOTFLUX_G     | FLOAT32        |
| FIBERTOTFLUX_R     | FLOAT32        |
| FIBERTOTFLUX_Z     | FLOAT32        |
| Z                  | FLOAT64        |
| ZERR               | FLOAT64        |
| ZWARN              | INT64          |
| SPECTYPE           | CHAR[6]        |
| SUBTYPE            | CHAR[20]       |

## Minimum Viable Example
This is a small example that uses the `DESIDataset` to train a small fully-connected autoencoder. First, we instantiate two datasets, one for training and one for validation, using the public dr1 data:
```python
import fitsio
from adaptic.desi import DESIDataset

specprod_dir = "/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/"
summary_file = "/global/cfs/cdirs/desi/public/dr1/spectro/redux/iron/healpix-iron.fits"
with fitsio.FITS(summary_file) as h:
    summary_table = h[1].read()

train_frac = 0.7
dataset_train = DESIDataset(specprod_dir=specprod_dir, summary_table=data_table, seed=91701, train_frac=train_frac, train_data=True, normalize=True, filter_func=get_science_only)
dataset_valid = DESIDataset(specprod_dir=specprod_dir, summary_table=data_table, seed=91701, train_frac=train_frac, train_data=False, normalize=True, filter_func=get_science_only)
```
Note that we're using the get_science_only function defined above as a filtering function.

We will then use PyTorch `DataLoader` objects to handle batching our data, and use parallelism to speed up dataloading (in this case using 8 workers to load 8 files in parallel):
```python
from torch.utils.data import DataLoader

batchsize = 1024
train_dl = iter(DataLoader(dataset_train, batch_size=batchsize, num_workers=8))
valid_dl = iter(DataLoader(dataset_valid, batch_size=batchsize, num_workers=8))
```

We will instantite our autoencoder and use the Adam optimizer, makign sure to move our model to the GPU. Then we train:

```python
model = DESIEncoder()
model.cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)


# curr_loss and curr_valid track a 5 epoch average
# of the loss.
curr_loss = 0
curr_valid = 0
all_loss = []
valid_loss = []
max_iter = 1000

i = 0
while i < max_iter:
    model.train() # Ensure the model is in training mode to track gradients.
    data = next(train_dl)
    tids, details = data
    fl = details["FLUX"].cuda()
    # Use the mask to mask out pixels when computing loss.
    mask = ~details["MASK"].cuda()

    opt.zero_grad()
    outputs = model(fl)

    loss = loss_fn(mask * outputs, mask * fl)
    loss.backward()
    opt.step()

    all_loss.append(loss.item())
    curr_loss += loss.item()

    # Compute validation loss.
    model.eval()
    data_v = next(valid_dl)
    tids, details_v = data_v
    fl_v = details_v["FLUX"].cuda()
    # Use the mask to mask out pixels when computing loss.
    mask_v = ~details_v["MASK"].cuda()
    outputs = model(fl_v)
    loss_v = loss_fn(mask_v * outputs, mask_v * fl_v)

    valid_loss.append(loss.item())
    curr_valid += loss.item()

    if (i % 5) == 0:
        print(f"Batch {i}, average loss: {curr_loss / 5}, valid_loss = {curr_valid / 5}")
        curr_loss = 0
        curr_valid = 0

    i += 1
```
