

pyYeti
======

pyYeti has tools mostly related to structural dynamics:

    * Matrix equations of motion solver in the time and frequency
      domains
    * Shock response spectra (SRS)
    * Rainflow cycle counting
    * Fatigue damage equivalent power spectral densities (PSD)
    * Resample data with the "Lanczos" method
    * A "vectorized" writing module
    * A data-cursor for interacting with 2D x-y plots
    * Statistics tools for computing k-factors (for tolerance
      bounds and intervals) and for order statistics
    * Force limiting analysis tools
    * Eigensolution with the subspace iteration method
    * Read/write Nastran output4 (.op4) files
    * Limited capability to read Nastran output2 (.op2) files
    * Hurty-Craig-Bampton model checks
    * Tools for working with the "nas2cam" Nastran DMAP
    * Other miscellaneous tools

More features are planned in the near future.


Installation
------------
pyYeti runs on Python 3.5 or later. The dependencies are NumPy, SciPy,
pandas and setuptools. These are all conveniently provided by the
Anaconda Python distribution: https://www.continuum.io/downloads.

The easiest way to install is to try `pip`::

  pip install pyyeti

You can also download manually from GitHub (link below) or from PyPi
(https://pypi.python.org/pypi/pyyeti), unpack it, and run::

  python setup.py install

For the C version of the rainflow cycle counter, you also need a C
compiler installed.


Development version
-------------------
The most current development version is here:

    https://github.com/twmacro/pyyeti


Documentation
-------------
pyYeti documentation is here:

    http://pythonhosted.org/pyyeti


Tutorials
---------
The documentation contains several tutorials. These are also available
(in their original form) as Jupyter notebooks:

    https://github.com/twmacro/pyyeti/tree/master/docs/tutorials
