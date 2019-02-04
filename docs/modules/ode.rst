Time and frequency domain equation of motion solvers
====================================================

.. automodule:: pyyeti.ode
.. currentmodule:: pyyeti.ode

1st Order ODE Solver `SolveExp1`
--------------------------------
.. autosummary::
    :toctree: generated/

    SolveExp1
    SolveExp1.tsolve

2nd Order ODE Solver `SolveUnc`
-------------------------------
.. autosummary::
    :toctree: generated/

    SolveUnc
    SolveUnc.tsolve
    SolveUnc.fsolve
    SolveUnc.generator
    SolveUnc.finalize
    SolveUnc.get_f2x

2nd Order ODE Solver `SolveCDF`
-------------------------------
.. autosummary::
    :toctree: generated/

    SolveCDF
    SolveCDF.tsolve
    SolveCDF.fsolve
    SolveCDF.generator
    SolveCDF.finalize
    SolveCDF.get_f2x

2nd Order ODE Solver `SolveExp2`
--------------------------------
.. autosummary::
    :toctree: generated/

    SolveExp2
    SolveExp2.tsolve
    SolveExp2.generator
    SolveExp2.finalize
    SolveExp2.get_f2x

2nd Order ODE Solver `SolveNewmark`
-----------------------------------
.. autosummary::
    :toctree: generated/

    SolveNewmark
    SolveNewmark.tsolve
    SolveNewmark.def_nonlin

2nd Order ODE Frequency Domain Solver `FreqDirect`
--------------------------------------------------
.. autosummary::
    :toctree: generated/

    FreqDirect
    FreqDirect.fsolve

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
    get_su_coef
    make_A
