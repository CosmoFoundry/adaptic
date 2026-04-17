# adaptic
Adapting cosmology datasets for AI uses.

Broadly, `adaptic` is a package that provides lightweight wrappers for some cosmology datasets for AI use. For example, code is provided to load DESI data from a public data release using PyTorch that handles minor nuisance tasks at the same time (like coadding across camera arms).

The wrappers in `adaptic` are designed to be used either by installing the entire package or by copy pasting the individual wrapper files into your project. For the DESI example, this would be the `adaptic/desi/dataset.py` file.