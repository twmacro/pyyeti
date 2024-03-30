<p align="center">
<a href="https://travis-ci.com/twmacro/pyyeti"><img alt="Build Status" src="https://travis-ci.com/twmacro/pyyeti.svg?branch=master"></a>
<a href="https://pyyeti.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/pyyeti/badge/?version=latest"></a>
<a href="https://coveralls.io/github/twmacro/pyyeti?branch=master"><img alt="Coverage Status" src="https://coveralls.io/repos/github/twmacro/pyyeti/badge.svg?branch=master"></a>
<a href="https://github.com/twmacro/pyyeti/blob/master/LICENSE.txt"><img alt="License: BSD 3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
<a href="https://pypi.org/project/pyyeti/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pyyeti"></a>
<a href="https://github.com/twmacro/pyyeti"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


# pyYeti

pyYeti has tools mostly related to structural dynamics:


* Solve matrix equations of motion in the time and frequency domains
* Shock response spectrum (SRS)
* Fatigue damage equivalent power spectral densities (PSD)
* Hurty-Craig-Bampton model checks
* Coupled loads analysis tools
* Statistics tools for computing k-factors (for tolerance bounds and intervals) and for order statistics
* A data-cursor for interacting with 2D/3D matplotlib plots
* Eigensystem Realization Algorithm
* Force limiting analysis tools
* Read/write Nastran output4 (.op4) files
* Limited capability to read Nastran output2 (.op2) files
* Rainflow cycle counting
* Resample data with the Lanczos method
* A "vectorized" writing module
* Eigensolution with the subspace iteration method
* Tools for working with the "nas2cam" Nastran DMAP
* Other miscellaneous tools


## Installation

pyYeti runs on Python 3.10 or later. The dependencies are NumPy, SciPy,
Matplotlib, pandas and setuptools. These are all conveniently provided
by the Anaconda Python distribution:
https://www.anaconda.com/distribution/.

You can install pyYeti via `pip`:

    python -m pip install pyyeti
    
or:

    python -m pip install --user --upgrade pyyeti

You can also install from source, doing something like this:

    git clone https://github.com/twmacro/pyyeti.git
    cd pyyeti
    python -m build -n
    cd dist
    python -m pip install <filename>.whl

Note that for the C version of the rainflow cycle counter, you also
need a C compiler installed. However, the speed of the pure Python
version is on par with the C version if you have Numba installed. In
fact, the C version may be removed in the future after making Numba a
required dependency.


## Documentation

pyYeti documentation is here:

   http://pyyeti.readthedocs.org/


## Tutorials

The documentation contains several tutorials in the documentation.
These are also available (in their original form) as Jupyter
notebooks:

   https://github.com/twmacro/pyyeti/tree/master/docs/tutorials


## License

BSD. See [LICENSE.txt](LICENSE.txt)


## Contributing to pyYeti

Contributions are much appreciated. Bug reports, documentation
updates, feature requests, and code enhancements are all great
ways to contribute.
