
|Build Status| |Coverage Status| |PyPI Status| |Docs Status|


pyYeti
======

pyYeti has tools mostly related to structural dynamics:

    * Solve matrix equations of motion in the time and
      frequency domains
    * Shock response spectrum (SRS)
    * Rainflow cycle counting
    * Fatigue damage equivalent power spectral densities (PSD)
    * Resample data with the Lanczos method
    * A "vectorized" writing module
    * A data-cursor for interacting with 2D x-y plots
    * Statistics tools for computing k-factors (for tolerance
      bounds and intervals) and for order statistics
    * Force limiting analysis tools
    * Eigensolution with the subspace iteration method
    * Read/write Nastran output4 (.op4) files
    * Limited capability to read Nastran output2 (.op2) files
    * Hurty-Craig-Bampton model checks
    * Coupled loads analysis tools
    * Tools for working with the "nas2cam" Nastran DMAP
    * Other miscellaneous tools

More features are planned in the near future.


Installation
------------
pyYeti runs on Python 3.5 or later. The dependencies are NumPy, SciPy,
Matplotlib, pandas and setuptools. These are all conveniently provided
by the Anaconda Python distribution:
https://www.continuum.io/downloads.

The easiest way to install is to try `pip`::

  pip install pyyeti

You can also download the source distribution from PyPI
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

    http://pyyeti.readthedocs.org/


Tutorials
---------
The documentation contains several tutorials. These are also available
(in their original form) as Jupyter notebooks:

    https://github.com/twmacro/pyyeti/tree/master/docs/tutorials


.. |Build Status| image:: https://travis-ci.org/twmacro/pyyeti.svg?branch=master
    :target: https://travis-ci.org/twmacro/pyyeti/

.. |Coverage Status| image:: https://coveralls.io/repos/github/twmacro/pyyeti/badge.svg?branch=master
    :target: https://coveralls.io/github/twmacro/pyyeti?branch=master 

.. |PyPI Status| image:: https://img.shields.io/pypi/v/pyyeti.svg
    :target: https://pypi.python.org/pypi/pyyeti

.. |Docs Status| image:: https://readthedocs.org/projects/pyyeti/badge/?version=latest
    :target: https://pyyeti.readthedocs.org
