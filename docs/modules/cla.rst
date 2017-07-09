Tools for coupled loads analysis
================================

.. automodule:: pyyeti.cla
.. currentmodule:: pyyeti.cla

Class for defining data recovery categories
-------------------------------------------
.. autosummary::
    :toctree: edited/

    DR_Def
    DR_Def.add
    DR_Def.addcat
    DR_Def.add_0rb
    DR_Def.copycat
    DR_Def.excel_summary

Class for getting data recovery ready for running a specific event
------------------------------------------------------------------
.. autosummary::
    :toctree: generated/

    DR_Event
    DR_Event.add
    DR_Event.prepare_results
    DR_Event.apply_uf
    DR_Event.frf_apply_uf

Class for storing and working with CLA results
----------------------------------------------
.. autosummary::
    :toctree: edited/

    DR_Results
    DR_Results.add_maxmin
    DR_Results.delete_extreme
    DR_Results.form_extreme
    DR_Results.form_stat_ext
    DR_Results.init
    DR_Results.init_extreme_cat
    DR_Results.merge
    DR_Results.psd_data_recovery
    DR_Results.resp_plots
    DR_Results.rptext
    DR_Results.rptpct
    DR_Results.rpttab
    DR_Results.solvepsd
    DR_Results.srs_plots
    DR_Results.strip_hists
    DR_Results.time_data_recovery

Utility routines
----------------
.. autosummary::
    :toctree: generated/

    extrema
    freq3_augment
    get_drfunc
    get_marker_cycle
    magpct
    maxmin
    mk_plots
    nan_absmax
    nan_argmax
    nan_argmin
    PrintCLAInfo
    PSD_consistent_rss
    rdext
    rptext1
    rptpct1
    rpttab1
