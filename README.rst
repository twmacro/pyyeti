pyYeti is a package for
=======================

    * Solving matrix equations of motion in the time and frequency
      domains.
    * Computing the shock response spectra (SRS).
    * Doing "rainflow" cycle counting.
    * Calculating power spectral densities (PSD) base on fatigue damage
      equivalence.
    * Resampling data with the "Lanczos" method.
    * A "vectorized" writing module.
    * A data-cursor for interacting with 2D x-y plots.
    * Statistics tools for computing k-factors (for tolerance bounds
      and intervals) and for order statistics.
    * Force limiting analysis tools.
    * Calculating eigensolution with the subspace iteration method.
    * Reading/writing Nastran output4 (.op4) files.
    * Limited capability to read Nastran output2 (.op2) files.
    * Doing checks on Hurty-Craig-Bampton models.
    * Tools for working with the "nas2cam" Nastran DMAP.
    * Other miscellaneous tools.


To install
----------
pyYeti runs on Python 3.5 or later. The dependencies are NumPy, SciPy,
pandas and setuptools. These are all conveniently provided by the
Anaconda Python distribution: https://www.continuum.io/downloads.

If those are installed, download pyYeti as a zip from Github, unpack
it, and run::

  python setup.py install

For the C version of the rainflow cycle counter, you also need a C
compiler installed.

I plan to have pyYeti available via `pip` soon.


Development version
-------------------
The most current development version is here:

    http://github.com/twmacro/pyyeti


Documentation
-------------
pyYeti documentation will be here:

   http://readthedocs.org/pyyeti

See also the notebooks in the docs/tutorials directory.
