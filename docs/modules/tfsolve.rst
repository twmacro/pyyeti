Time and frequency domain equation of motion solvers (tfsolve)
==============================================================

.. automodule:: pyyeti.tfsolve
.. currentmodule:: pyyeti.tfsolve

The TFSolve Class
-----------------
.. autosummary::
    :toctree: generated/

    TFSolve

Time domain solvers:

.. autosummary::
    :toctree: generated/

    TFSolve.se1
    TFSolve.se2
    TFSolve.su
    TFSolve.su_generator
    TFSolve.su_finalize

Frequency domain solvers:

.. autosummary::
    :toctree: generated/

    TFSolve.fsd
    TFSolve.fsu

Optional direct setup routines (usually called via
:func:`TFSolve.__init__`):

.. autosummary::
    :toctree: generated/

    TFSolve.mkfsdparams
    TFSolve.mkse1params
    TFSolve.mkse2params
    TFSolve.mksuparams

Other main routines
-------------------
.. autosummary::
    :toctree: generated/

    getmodepart
    modeselect
    solvepsd

Utility routines
----------------
.. autosummary::
    :toctree: generated/

    addconj
    delconj
    eigss
    finddups
    get_su_coef
    make_A
