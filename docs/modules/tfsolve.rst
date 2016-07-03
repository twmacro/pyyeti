Time and frequency domain equation of motion solvers (tfsolve)
==============================================================

.. automodule:: pyyeti.tfsolve
.. currentmodule:: pyyeti.tfsolve

1st Order ODE Solver `se1`
--------------------------
.. autosummary::
    :toctree: generated/

    se1
    se1.tsolve

2nd Order ODE Solver `su`
-------------------------
.. autosummary::
    :toctree: generated/

    su
    su.tsolve
    su.fsolve
    su.generator
    su.finalize
    fsu
    eigsu
    eigfsu

2nd Order ODE Solver `se2`
--------------------------
.. autosummary::
    :toctree: generated/

    se2
    se2.tsolve
    se2.generator
    se2.finalize
    eigse2

2nd Order ODE Frequency Domain Solver `fsd`
-------------------------------------------
.. autosummary::
    :toctree: generated/

    fsd
    fsd.fsolve

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
