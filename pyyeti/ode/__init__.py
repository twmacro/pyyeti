# -*- coding: utf-8 -*-
"""
Time and frequency domain ODE solvers for matrix equations. Adapted
and enhanced from the Yeti versions (which were adapted and enhanced
from the original CAM versions). Note that some features depend on the
equations being in modal space (particularly important where there are
distinctions between the rigid-body modes and the elastic modes).

.. note::
    Some features of this module are demonstrated in the pyYeti
    :ref:`tutorial`: :doc:`/tutorials/ode`. There is also a link to
    the source Jupyter notebook at the top of the tutorial.
"""

from ._utilities import *
from .freqdirect import FreqDirect
from .frf_mode_participation import getmodepart, modeselect
from .solvecdf import SolveCDF
from .solveexp1 import SolveExp1
from .solveexp2 import SolveExp2
from .solvenewmark import SolveNewmark
from .solveunc import SolveUnc
