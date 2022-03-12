# -*- coding: utf-8 -*-
"""
DR_Results: manipulating data recovery results
"""
import os
import copy
from collections import OrderedDict
from types import SimpleNamespace
import warnings
import copyreg
import numpy as np
import xlsxwriter
from pyyeti import locate, srs
from ._utilities import maxmin, extrema, get_drfunc
from ._rptext1 import rptext1
from ._rptpct1 import rptpct1
from ._rpttab1 import rpttab1
from .dr_results_plots import mk_plots


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _is_eqsine(opts):
    """
    Checks to see if 'eqsine' option is set to true

    Parameters
    ----------
    opts : dict
        Dictionary of :func:`pyyeti.srs.srs` options; can be empty.

    Returns
    -------
    flag : bool
        True if the eqsine option is set to true.
    """
    if "eqsine" in opts:
        return opts["eqsine"]
    return False


class DR_Results(OrderedDict):
    """
    Subclass of :class:`collections.OrderedDict` that contains data
    recovery results

    Notes
    -----
    This data structure is designed to contain and operate on a
    hierarchy of event results, not unlike a directory tree on a
    computer. For example, it could be just one level deep and contain
    only the 'Liftoff' results. Or, it could contain all events from a
    CLA and each event could have sub-events under them.

    You can be creative in your organization to simplify otherwise
    cumbersome tasks. For example, say you want to form SRS envelope
    plots for all Stage-1 events separately from all Stage-2
    events. This is trivial if you organize your structure
    accordingly; for example::

        Top Level
            Stage-1
               Liftoff
               Transonics
               Max-Q
            Stage-2
               S2 Engine Start
               S2 Engine Cutoff

    For illustration, the following shows how this task might be
    done. To perhaps make this example more useful, it is assumed that
    we have five separate :class:`DR_Results` instances and we have to
    organize them. After that, we can make the SRS plots very
    easily. Note that we could also write out various results and
    comparison tables at the different levels. For this example, we'll
    form three separate SRS plots: the first showing the Stage-1 and
    Stage-2 enveloping curves together, the second showing the Stage-1
    Liftoff, Transonics, and Max-Q curves together, and the third
    showing the Stage-2 Engine Start and Cutoff curves together::

        # first, organize the results as shown above:
        results = cla.DR_Results()

        results['Stage-1'] = cla.DR_Results()
        results['Stage-1']['Liftoff'] = lo_results
        results['Stage-1']['Transonics'] = trans_results
        results['Stage-1']['Max-Q'] = maxq_results

        results['Stage-2'] = cla.DR_Results()
        results['Stage-2']['S2 Engine Start'] = ses_results
        results['Stage-2']['S2 Engine Cutoff'] = seco_results

        # next, compute extreme results at all levels:
        results.form_extreme()

        # srs plot number 1; will have 2 curves:
        results['extreme'].srs_plots(
            direc='S1_S2_srs', Q=20, showall=True)

        # srs plot number 2; will have 3 curves:
        results['Stage-1']['extreme'].srs_plots(
            direc='S1_events_srs', Q=20, showall=True)

        # srs plot number 3; will have 2 curves:
        results['Stage-2']['extreme'].srs_plots(
            direc='S2_events_srs', Q=20, showall=True)

    Below are a couple example :class:`DR_Results` instances named
    `results`. Note that this structure contains all the data recovery
    matrix information originally collected by calls to
    :func:`DR_Def.add` (in a "prepare_4_cla.py" script, for example).

    The first example is from a PSD buffet run (after running
    ``results.form_extreme()``):

    PP(results):

    .. code-block:: none

        <class 'cla.DR_Results'>[n=2]
            'MaxQ' : <class 'cla.DR_Results'>[n=12]
            'extreme': <class 'cla.DR_Results'>[n=12]

    PP(results['MaxQ']):

    .. code-block:: none

        <class 'cla.DR_Results'>[n=12]
            'PAF_ifatm'    : <class 'types.SimpleNamespace'>[n=17]
            'PAF_ifatm_0rb': <class 'types.SimpleNamespace'>[n=17]
            'PAF_ifltm'    : <class 'types.SimpleNamespace'>[n=17]
            'SC_atm'       : <class 'types.SimpleNamespace'>[n=17]
            'SC_dtm'       : <class 'types.SimpleNamespace'>[n=15]
            'SC_ifl'       : <class 'types.SimpleNamespace'>[n=17]
            'SC_ltma'      : <class 'types.SimpleNamespace'>[n=15]
            'SC_ltmd'      : <class 'types.SimpleNamespace'>[n=15]
            'SC_cg'        : <class 'types.SimpleNamespace'>[n=15]
            'SC_ifa'       : <class 'types.SimpleNamespace'>[n=17]
            'SC_ifa_0rb'   : <class 'types.SimpleNamespace'>[n=17]
            'Box_CG'       : <class 'types.SimpleNamespace'>[n=17]

    PP(results['MaxQ']['SC_ifa'], 3):

    .. code-block:: none

        <class 'types.SimpleNamespace'>[n=17]
            .cases  : [n=1]: ['MaxQ']
            .domain : 'freq'
            .drminfo: <class 'types.SimpleNamespace'>[n=20]
                .desc      : 'S/C Interface Accelerations'
                .drfile    : '/loads/CLA/Rocket/missions/...funcs.py'
                .drfunc    : 'SC_ifa'
                .filterval : 1e-06
                .histlabels: [n=12]: ['I/F Axial Accel     X sc', ...]
                .histpv    : slice(None, 12, None)
                .histunits : 'G, rad/sec^2'
                .ignorepv  : None
                .labels    : [n=12]: ['I/F Axial Accel     X sc', ...]
                .misc      : None
                .se        : 500
                .srsQs     : [n=2]: (25, 50)
                .srsconv   : 1.0
                .srsfrq    : float64 ndarray 990 elems: (990,)
                .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', ...']
                .srsopts   : <class 'dict'>[n=2]
                    'eqsine': 1
                    'ic'    : 'steady'
                .srspv     : slice(None, 12, None)
                .srsunits  : 'G, rad/sec^2'
                .uf_reds   : [n=4]: (1, 1, 1.25, 1)
                .units     : 'G, rad/sec^2'
            .event  : 'MaxQ Buffet'
            .ext    : float64 ndarray 24 elems: (12, 2)
            .ext_x  : float64 ndarray 24 elems: (12, 2)
            .freq   : float64 ndarray 2332 elems: (2332,)
            .maxcase: [n=12]: ['MaxQ', 'MaxQ', 'MaxQ' ...'MaxQ']
            .mx_x   : float64 ndarray 12 elems: (12, 1)
            .mincase: [n=12]: ['MaxQ', 'MaxQ', 'MaxQ' ...'MaxQ']
            .mn_x   : float64 ndarray 12 elems: (12, 1)
            .mission: 'Rocket / Spacecraft VLC'
            .mn     : float64 ndarray 12 elems: (12, 1)
            .mx     : float64 ndarray 12 elems: (12, 1)
            .psd    : float64 ndarray 27984 elems: (1, 12, 2332)
            .rms    : float64 ndarray 12 elems: (12, 1)
            .srs    : <class 'types.SimpleNamespace'>[n=5]
                .ext  : <class 'dict'>[n=2]
                    25: float64 ndarray 11880 elems: (12, 990)
                    50: float64 ndarray 11880 elems: (12, 990)
                .frq  : float64 ndarray 990 elems: (990,)
                .srs  : <class 'dict'>[n=2]
                    25: float64 ndarray 11880 elems: (1, 12, 990)
                    50: float64 ndarray 11880 elems: (1, 12, 990)
                .type : 'eqsine'
                .units: 'G, rad/sec^2'

    Here is another example from a time-domain case where the extreme
    values are computed statistically from all cases:

    PP(results['SC_ifa'], 4):

    .. code-block:: none

        <class 'types.SimpleNamespace'>[n=16]
            .cases  : [n=21]: ['SECO2 1', 'SECO2 2', 'SECO2 3 ... 21']
            .domain : 'time'
            .drminfo: <class 'types.SimpleNamespace'>[n=20]
                .desc      : 'S/C Interface Accelerations'
                .drfile    : '/loads/CLA/Rocket/missions/...funcs.py'
                .drfunc    : 'SC_ifa'
                .filterval : 1e-06
                .histlabels: [n=12]: ['I/F Axial Accel     X sc', ...]
                .histpv    : slice(None, 12, None)
                .histunits : 'G, rad/sec^2'
                .ignorepv  : None
                .labels    : [n=12]: ['I/F Axial Accel     X sc', ...]
                .misc      : None
                .se        : 500
                .srsQs     : [n=2]: (25, 50)
                .srsconv   : 1.0
                .srsfrq    : float64 ndarray 990 elems: (990,)
                .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', ...']
                .srsopts   : <class 'dict'>[n=2]
                    'eqsine': 1
                    'ic'    : 'steady'
                .srspv     : slice(None, 12, None)
                .srsunits  : 'G, rad/sec^2'
                .uf_reds   : [n=4]: (1, 1, 1.25, 1)
                .units     : 'G, rad/sec^2'
            .event  : 'SECO2'
            .ext    : float64 ndarray 24 elems: (12, 2)
            .ext_x: None
            .hist   : float64 ndarray 2520252 elems: (21, 12, 10001)
            .maxcase: None
            .mx_x   : float64 ndarray 252 elems: (12, 21)
            .mincase: None
            .mn_x   : float64 ndarray 252 elems: (12, 21)
            .mission: 'Rocket / Spacecraft VLC'
            .mn     : float64 ndarray 252 elems: (12, 21)
            .mx     : float64 ndarray 252 elems: (12, 21)
            .srs    : <class 'types.SimpleNamespace'>[n=5]
                .ext  : <class 'dict'>[n=2]
                    25: float64 ndarray 11880 elems: (12, 990)
                    50: float64 ndarray 11880 elems: (12, 990)
                .frq  : float64 ndarray 990 elems: (990,)
                .srs  : <class 'dict'>[n=2]
                    25: float64 ndarray 249480 elems: (21, 12, 990)
                    50: float64 ndarray 249480 elems: (21, 12, 990)
                .type : 'eqsine'
                .units: 'G, rad/sec^2'
            .time   : float32 ndarray 10001 elems: (10001,)
    """

    def __repr__(self):
        cats = ", ".join(f"'{name}'" for name in self)
        return (
            f"{type(self).__name__} ({hex(id(self))}) "
            f"with {len(self)} keys: [{cats}]"
        )

    def init(self, Info, mission, event, cats=None):
        r"""
        Build initial results data structure.

        Parameters
        ----------
        Info : :class:`collections.OrderedDict` or :class:`DR_Def` instance
            Contains data recovery information for each category. The
            category names are the keys. Either the `Info` attribute
            of :class:`DR_Event` or a :class:`DR_Def` instance.
            (`Info` is a copy of the information in one or more
            `DR_Def` instances with the possibly different `uf_reds`
            values.)
        mission : str
            Identifies the CLA
        event : str
            Name of event
        cats : iterable of strings or None
            If iterable, contains names of categories to include in
            results; other names are quietly skipped.

        Notes
        -----
        The name "_vars" is quietly skipped if present in `Info`.
        """
        for name in Info:
            if name == "_vars" or (cats and name not in cats):
                continue
            self[name] = SimpleNamespace(
                ext=None,
                maxcase=None,
                mincase=None,
                mission=mission,
                event=event,
                drminfo=copy.copy(Info[name]),
            )

    def merge(self, results_iter, rename_dict=None):
        """
        Merge CLA results together into a larger :class:`DR_Results`
        hierarchy.

        Parameters
        ----------
        results_iter : iterable
            Iterable of :class:`DR_Results` items to merge. (Can be
            list, tuple, generator expression, or other Python
            iterable.)
        rename_dict : dict; optional
            Used to rename entries in final `self` results structure.
            The key is old name and the value is the new name. See
            example below.

        Returns
        -------
        events : list
            List of event names in the order provided by
            `results_iter`.

        Notes
        -----
        This routine is sort of an inverse of :func:`split`. The name
        of each event is extracted from the :class:`DR_Results` data
        structure. The approach to get the name is as follows. Note
        that 'SC_atm' is just an example; this routine will simply use
        first element it finds. Let `results` be the current instance
        of :class:`DR_Results`:

            1. If any element of `results` is a SimpleNamespace:
               ``event = results['SC_atm'].event``

            2. Else, if any element from `results` is another
               :class:`DR_Results` structure, then:

               a. If ``results['extreme']`` exists:
                  ``event = results['extreme']['SC_atm'].event``

               b. Else ``event = ', '.join(key for key in results)``

            3. Else, raise TypeError

        Example usage::

            # merge "liftoff" and "meco" results and rename the
            # liftoff results from "LO" to "Liftoff":

            from pyyeti import cla

            results = cla.DR_Results()
            results.merge(
                (
                    cla.load(fn)
                    for fn in [
                        "../liftoff/results.pgz",
                        "../meco/results.pgz",
                    ]
                ),
                {"LO": "Liftoff"},
            )
            results.strip_hists()
            results.form_extreme()
            results["extreme"].rpttab()
            cla.save("results.pgz", results)
        """

        def _get_event_name(results):
            # get any value from dict:
            v = next(iter(results.values()))
            if isinstance(v, SimpleNamespace):
                return v.event
            if isinstance(v, DR_Results):
                try:
                    v2 = results["extreme"]
                except KeyError:
                    return ", ".join(key for key in results)
                else:
                    return next(iter(v2.values())).event
            raise TypeError(f"unexpected type: {type(results)}")

        events = []
        for results in results_iter:
            event = _get_event_name(results)
            if rename_dict is not None:
                event = rename_dict.get(event, event)

            if event in self.keys():
                raise ValueError(f"Event with name {event} already exists! Aborting.")
            else:
                events.append(event)
                self[event] = results

        return events

    def split(self):
        """
        Split results apart into a new :class:`DR_Results` structure

        Returns
        -------
        results : :class:`DR_Results` instance
            This is a new, two-level instance of :class:`DR_Results`
            (each entry value is a base-level instance of
            :class:`DR_Results`). For example, if::

                self['SC_atm'].cases == ['ff1', 'ff2', 'ff3']

            then `results` would have::

                results['ff1']['SC_atm'].cases = ['ff1']
                results['ff2']['SC_atm'].cases = ['ff2']
                results['ff3']['SC_atm'].cases = ['ff3']

        Notes
        -----
        This routine is sort of an inverse of :func:`merge`. Splitting
        allows easy comparisons of one sub-case to another. It also
        allows the deletion of some cases: split the results, delete
        the entries you don't want, re-merge the results back
        together, and form a new set of extreme values (with
        :func:`form_extreme`).

        To illustrate what this routine does, consider this 'SC_ifa'
        entry in an initial results structure. In this case, the CLA
        event is "Stage 1 / Stage 2 Separation" and there are 3 stage
        one / stage two separation cases.

        PP(results['SC_ifa']):

        .. code-block:: none

            <class 'types.SimpleNamespace'>[n=16]
                .cases  : [n=3]: ['Sep 1', 'Sep 2', 'Sep 3']
                .domain : 'time'
                .drminfo: <class 'types.SimpleNamespace'>[n=20]
                .event  : 'Stage 1 / Stage 2 Separation'
                .ext    : float64 ndarray: (12, 2)
                .ext_x  : None
                .hist   : float64 ndarray: (3, 12, 10001)
                .maxcase: None
                .mx_x   : float64 ndarray: (12, 3)
                .mincase: None
                .mn_x   : float64 ndarray: (12, 3)
                .mission: 'Rocket / Spacecraft VLC'
                .mn     : float64 ndarray: (12, 3)
                .mx     : float64 ndarray: (12, 3)
                .srs    : <class 'types.SimpleNamespace'>[n=5]
                    .ext  : <class 'dict'>[n=2]
                        25: float64 ndarray: (12, 990)
                        50: float64 ndarray: (12, 990)
                    .frq  : float64 ndarray: (990,)
                    .srs  : <class 'dict'>[n=2]
                        25: float64 ndarray: (3, 12, 990)
                        50: float64 ndarray: (3, 12, 990)
                    .type : 'eqsine'
                    .units: 'G, rad/sec^2'
                .time   : float32 ndarray: (10001,)

        This routine would return a new structure with the three cases
        split up into three events. For example, ``sp_res =
        results.split()``, would give the following. Note that there
        would typically be more categories than just 'SC_ifa', and
        they would all be split in the same way.

        .. code-block:: none

            sp_res['Sep 1']['SC_ifa']
            sp_res['Sep 2']['SC_ifa']
            sp_res['Sep 3']['SC_ifa']

        And ``sp_res['Sep 2']['SC_ifa']`` would be (notice the 3's
        become 1's):

        .. code-block:: none

            <class 'types.SimpleNamespace'>[n=16]
                .cases  : [n=1]: ['Sep 2']
                .domain : 'time'
                .drminfo: <class 'types.SimpleNamespace'>[n=20]
                .event  : 'Sep 2'
                .ext    : float64 ndarray: (12, 2)
                .ext_x  : None
                .hist   : float64 ndarray: (1, 12, 10001)
                .maxcase: None
                .mx_x   : float64 ndarray: (12, 1)
                .mincase: None
                .mn_x   : float64 ndarray: (12, 1)
                .mission: 'Rocket / Spacecraft VLC'
                .mn     : float64 ndarray: (12, 1)
                .mx     : float64 ndarray: (12, 1)
                .srs    : <class 'types.SimpleNamespace'>[n=5]
                    .ext  : <class 'dict'>[n=2]
                        25: float64 ndarray: (12, 990)
                        50: float64 ndarray: (12, 990)
                    .frq  : float64 ndarray: (990,)
                    .srs  : <class 'dict'>[n=2]
                        25: float64 ndarray: (1, 12, 990)
                        50: float64 ndarray: (1, 12, 990)
                    .type : 'eqsine'
                    .units: 'G, rad/sec^2'
                .time   : float32 ndarray: (10001,)

        Example usage 1::

            # compare sub-cases 'MECO  1' and 'MECO 10':
            sp = results.split()
            sp['MECO  1'].rptpct(
                sp['MECO 10'], names=('MECO  1', 'MECO 10'),
                direc='m1_vs_m10')

        Example usage 2::

            # delete case 'MECO 15' from the results and form a new
            # set of statistical extreme results:
            from pyyeti.stats import ksingle

            # split results and delete 'MECO 15':
            sp = results.split()
            del sp['MECO 15']

            # merge remaining cases back together:
            new_res = cla.DR_Results()
            new_res.merge(sp.values())

            # form non-statistical extrema (completes the merge):
            new_res.form_extreme()

            # change to statistical extrema:
            ncases = len(new_res['extreme'].cases)
            new_res['extreme'].calc_stat_ext(
                stats.ksingle(0.99, 0.90, ncases))

            # compare new extrema to original:
            new_res['extreme'].rptpct(
                results, names=('W/O MECO 15', 'Original'),
                direc='no_m15_vs_all')

        Raises
        ------
        TypeError
            When ``self[cat]`` is not a SimpleNamespace. This usually
            happens when `self` is a multiple level
            :class:`DR_Results` instance. In that case, instead of
            ``res.split()``, try something like
            ``res[event].split()``.
        """
        value = next(iter(self.values()))
        if not isinstance(value, SimpleNamespace):
            raise TypeError(
                ":func:`split` only works with base-level "
                ":class:`DR_Results` instances (eg: "
                "instead of ``res.split()``, try "
                "something like ``res[event].split()``)."
            )
        res = DR_Results()
        cases = value.cases
        for j, case in enumerate(cases):
            res[case] = DR_Results()
            # copy "case" results (j'th) into res[case]:
            for cat, sns in self.items():
                newsns = DR_Results.init_extreme_cat(
                    [case], sns, ext_name=case, domain=sns.domain  # sns.event,
                )

                # copy j'th results:
                newsns.mx[:, 0] = sns.mx[:, j]
                newsns.mn[:, 0] = sns.mn[:, j]
                newsns.ext = np.column_stack((newsns.mx, newsns.mn))
                newsns.mx_x[:, 0] = sns.mx_x[:, j]
                newsns.mn_x[:, 0] = sns.mn_x[:, j]
                newsns.ext_x = np.column_stack((newsns.mx_x, newsns.mn_x))

                # check for hist, time, psd, freq
                for item in ("hist", "time", "psd", "freq"):
                    try:
                        v = getattr(sns, item)
                    except AttributeError:
                        pass
                    else:
                        if v.ndim > 1:
                            v = v[[j]]
                        setattr(newsns, item, v)

                # handle SRS if present:
                try:
                    osrs = sns.srs
                except AttributeError:
                    pass
                else:
                    try:
                        osrs_srs = sns.srs.srs
                    except AttributeError:
                        del newsns.srs
                    else:
                        for q in osrs.ext:
                            newsns.srs.ext[q] = osrs_srs[q][j]
                            newsns.srs.srs[q][:] = osrs_srs[q][j]

                res[case][cat] = newsns
        return res

    def add_maxmin(
        self, cat, mxmn, maxcase, mincase=None, mxmn_xvalue=None, domain=None
    ):
        """
        Add maximum and minimum values from an external source

        Parameters
        ----------
        cat : string
            Data recovery category, eg: 'SC_atm'
        mxmn : 2d array_like
            2 column matrix of [max, min]
        maxcase : string or list of strings
            String or list of strings identifying the load case(s) for
            the maximum values.
        mincase : string or list of strings or None; optional
            Analogous to `maxcase` for the minimum values or None. If
            None, it is a copy of the `maxcase` values.
        mxmn_xvalue : 2d array_like or None; optional
            2 column matrix of [mx_xvalue, mn_xvalue]. Use None to
            not set the x-values (typically times or frequencies) of
            max/min values.
        domain : string or None; optional
            Typically 'time' or 'freq', but can be any string or
            None. Use None to not define a domain.

        Returns
        -------
        None

        Notes
        -----
        This routine is not normally needed. It is only here for
        situations where the results were calculated elsewhere but you
        want to use these tools (eg, for reports or doing
        comparisons). This routine does not check to see if `cat`
        already exists; if it does, it is overriden.

        Examples
        --------
        Here is a simple but complete example. CLA results are made up
        for an "ATM" and an "LTM" for 3 events:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from pyyeti import cla
        >>>
        >>> # make up some "external source" CLA results:
        >>> events = ('Liftoff', 'Transonics', 'MECO')
        >>> rows = {'ATM': 34, 'LTM': 29}
        >>> ext_results = {i: {} for i in rows}
        >>> t = np.arange(200)/200
        >>> for event in events:
        ...     for drm, nrows in rows.items():
        ...         resp = np.random.randn(nrows, len(t))
        ...         mxmn = cla.maxmin(resp, t)
        ...         ext_results[drm][event] = mxmn.ext
        >>>
        >>> # setup CLA parameters:
        >>> mission = "Rocket / Spacecraft VLC"
        >>> duf = 1.2
        >>> suf = 1.0
        >>>
        >>> # defaults for data recovery
        >>> defaults = dict(se=0,
        ...                 uf_reds=(1, 1, duf, suf))
        >>> drdefs = cla.DR_Def(defaults)
        >>>
        >>> def _get_labels(name):
        ...     return [f'{name} Row {i+1:6d}'
        ...             for i in range(rows[name])]
        >>>
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     name = 'ATM'
        ...     desc = 'S/C Internal Accelerations'
        ...     units = 'm/sec^2, rad/sec^2'
        ...     labels = _get_labels(name)
        ...     drfunc = 'no func'
        ...     drdefs.add(**locals())
        >>>
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     name = 'LTM'
        ...     desc = 'S/C Internal Loads'
        ...     units = 'N, N-m'
        ...     labels = _get_labels(name)
        ...     drfunc = 'no func'
        ...     drdefs.add(**locals())
        >>>
        >>> # for checking, make a pandas DataFrame to summarize data
        >>> # recovery definitions (but skip the excel file for this
        >>> # demo)
        >>> pd.options.display.max_colwidth = 25
        >>> drdefs.excel_summary(None)
                                         ATM                       LTM
        active                           yes                         -
        desc        S/C Internal Accelera...        S/C Internal Loads
        drfile                          None                         -
        drfunc                       no func                         -
        filterval                      1e-06                         -
        histlabels                      None                         -
        histpv                          None                         -
        histunits                       None                         -
        ignorepv                        None                         -
        labels      34: ['ATM Row      1'...  29: ['LTM Row      1'...
        misc                            None                         -
        se                                 0                         -
        srsQs                           None                         -
        srsconv                         None                         -
        srsfrq                          None                         -
        srslabels                       None                         -
        srsopts                         None                         -
        srspv                           None                         -
        srsunits                        None                         -
        uf_reds          4: (1, 1, 1.2, 1.0)                         -
        units             m/sec^2, rad/sec^2                    N, N-m
        >>>
        >>> # prepare results data structure:
        >>> DR = cla.DR_Event()
        >>> DR.add(None, drdefs)
        >>> results = cla.DR_Results()
        >>> for event in events:
        ...     results[event] = DR.prepare_results(mission, event)
        ...     for drm in rows:
        ...         results[event].add_maxmin(
        ...             drm, ext_results[drm][event], event)
        >>>
        >>> # Done with setup; now we can use the standard cla tools:
        >>> results.form_extreme()
        >>> # To write an extreme 'Results.xlsx' file, uncomment the
        >>> # following line:
        >>> # results['extreme'].rpttab(excel='Results')
        """
        self[cat].ext = np.atleast_2d(mxmn)
        if mxmn_xvalue is not None:
            mxmn_xvalue = np.atleast_2d(mxmn_xvalue)
        self[cat].ext_x = mxmn_xvalue
        self[cat].domain = domain

        # process maxcase, mincase:
        r = self[cat].ext.shape[0]
        if isinstance(maxcase, str):
            self[cat].maxcase = r * [maxcase]
        else:
            self[cat].maxcase = maxcase[:]

        if mincase is None:
            self[cat].mincase = self[cat].maxcase[:]
        elif isinstance(mincase, str):
            self[cat].mincase = r * [mincase]
        else:
            self[cat].mincase = mincase[:]

    def _store_maxmin(self, res, mm, j, case):
        try:
            res.cases.index(case)
        except ValueError:
            pass
        else:
            raise ValueError(f"case '{case}' already defined!")
        res.mx[:, j] = mm.ext[:, 0]
        res.mx_x[:, j] = mm.ext_x[:, 0]
        res.mn[:, j] = mm.ext[:, 1]
        res.mn_x[:, j] = mm.ext_x[:, 1]
        res.cases[j] = case

    def _check_labels_len(self, name, res, m=None):
        if m is None:
            m = res.ext.shape[0]
        lbllen = len(res.drminfo.labels)
        if lbllen != m:
            raise ValueError(
                f"for {name}, length of `labels` ({lbllen}) does "
                f"not match number of data recovery items ({m})"
            )

    def _init_mxmn(self, name, res, domain, mm, n):
        m = mm.ext.shape[0]
        self._check_labels_len(name, res, m)
        res.domain = domain
        res.mx = np.zeros((m, n))
        res.mn = np.zeros((m, n))
        res.mx_x = np.zeros((m, n))
        res.mn_x = np.zeros((m, n))
        res.cases = n * [[]]
        return m

    def _init_results_cat(
        self, name, dr, resp, respname, x, xname, mm, n, dohist, dosrs
    ):
        # dohist is here for 3rd party use cases
        res = self[name]
        m = self._init_mxmn(name, res, xname, mm, n)
        if dr.histpv is not None and dohist:
            m = len(resp[dr.histpv, 0])
            setattr(res, xname, x)
            setattr(res, respname, np.zeros((n, m, len(x)), resp.dtype))
        if dr.srspv is not None and dosrs:
            res.srs = SimpleNamespace(frq=dr.srsfrq, units=dr.srsunits, srs={}, ext={})
            m = len(resp[dr.srspv, 0])
            sh = (n, m, (len(res.srs.frq)))
            for q in dr.srsQs:
                res.srs.srs[q] = np.zeros(sh)

    def _compute_srs(self, res, dr, resp, respname, x, j, first, sr=None, pf=None):
        if _is_eqsine(dr.srsopts):
            res.srs.type = "eqsine"
            eqsine = True
        else:
            res.srs.type = "srs"
            eqsine = False

        rr = resp[dr.srspv].T
        for q in dr.srsQs:
            fact = dr.srsconv

            # compute the srs:
            if respname == "hist":
                srs_cur = fact * srs.srs(rr, sr, dr.srsfrq, q, **dr.srsopts).T
            elif respname == "frf":
                if eqsine:
                    fact /= q
                srs_cur = fact * srs.srs_frf(rr, x, dr.srsfrq, q).T
            elif respname == "psd":
                fact *= pf
                if eqsine:
                    fact /= q
                srs_cur = fact * srs.vrs((x, rr), x, q, Fn=dr.srsfrq, linear=True).T
            else:  # pragma: no cover
                raise ValueError(
                    "`respname` must be one of: " '"hist", "frf", or "psd"'
                )

            # store results and keep track of extreme srs:
            res.srs.srs[q][j] = srs_cur
            if first:
                res.srs.ext[q] = srs_cur
            else:
                res.srs.ext[q] = np.fmax(res.srs.ext[q], srs_cur)

    def time_data_recovery(self, sol, nas, case, DR, n, j, dosrs=True):
        """
        Time-domain data recovery function

        Parameters
        ----------
        sol : dict
            SimpleNamespace containing the modal solution as output
            from :func:`DR_Event.apply_uf`.
        dr_object : any object
            Any object that is useful for data recovery. Can be None
            if not needed: it is not used in this routine; it is only
            passed to the data recovery routines (the `drfunc` setting
            in :func:`DR_Def.add`). Historically, this was the nas2cam
            dictionary (``nas = pyyeti.nastran.op2.rdnas2cam()``).
        case : string
            Unique string identifying the case; stored in, for
            example, the ``self['SC_atm'].cases`` and the `.mincase`
            and `.maxcase` lists
        DR : instance of :class:`DR_Event`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.DR_Event()``). It is an event specific version
            of all combined :class:`DR_Def` objects with all ULVS
            matrices applied.
        n : integer
            Total number of load cases
        j : integer
            Current load case number starting at zero
        dosrs : bool; optional
            If False, do not calculate SRSs; default is to calculate
            them.

        Returns
        -------
        None

        Notes
        -----
        The `self` results dictionary is updated (see
        :class:`DR_Results` for an example).
        """
        for name, res in self.items():
            first = res.ext is None
            dr = DR.Info[name]  # record with: .desc, .labels, ...
            uf_reds = dr.uf_reds
            SOL = sol[uf_reds]
            drfunc = get_drfunc(dr.drfile, dr.drfunc)
            resp = drfunc(SOL, nas, DR.Vars, dr.se)

            mm = maxmin(resp, SOL.t)
            extrema(res, mm, case)

            if first:
                self._init_results_cat(
                    name,
                    dr,
                    resp,
                    "hist",
                    SOL.t,
                    "time",
                    mm,
                    n,
                    dohist=True,
                    dosrs=dosrs,
                )

            self._store_maxmin(res, mm, j, case)

            if dr.histpv is not None:
                res.hist[j] = resp[dr.histpv]

            if dr.srspv is not None and dosrs:
                sr = 1 / SOL.h if SOL.h else None
                self._compute_srs(res, dr, resp, "hist", SOL.t, j, first, sr=sr)

    def frf_data_recovery(self, sol, nas, case, DR, n, j, dosrs=True):
        """
        Frequency response data recovery function

        Parameters
        ----------
        sol : dict
            SimpleNamespace containing the modal solution as output
            from :func:`DR_Event.apply_uf`.
        nas : dictionary
            Typically, this is the nas2cam dictionary:
            ``nas = pyyeti.nastran.op2.rdnas2cam()``.
            Can be None if not needed: it is not used in this routine;
            it is only passed to the data recovery routines (the
            `drfunc` setting in :func:`DR_Def.add`).
        case : string
            Unique string identifying the case; stored in, for
            example, the ``self['SC_atm'].cases`` and the `.mincase`
            and `.maxcase` lists
        DR : instance of :class:`DR_Event`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.DR_Event()``). It is an event specific version
            of all combined :class:`DR_Def` objects with all ULVS
            matrices applied.
        n : integer
            Total number of load cases
        j : integer
            Current load case number starting at zero
        dosrs : bool; optional
            If False, do not calculate SRSs; default is to calculate
            them.

        Returns
        -------
        None

        Notes
        -----
        The `self` results dictionary is updated (see
        :class:`DR_Results` for an example).
        """
        for name, res in self.items():
            first = res.ext is None
            dr = DR.Info[name]  # record with: .desc, .labels, ...
            uf_reds = dr.uf_reds
            SOL = sol[uf_reds]
            drfunc = get_drfunc(dr.drfile, dr.drfunc)
            resp = drfunc(SOL, nas, DR.Vars, dr.se)

            mm = maxmin(abs(resp), SOL.f)
            mm.ext[:, 1] = -mm.ext[:, 0]
            mm.ext_x[:, 1] = mm.ext_x[:, 0]
            extrema(res, mm, case)

            if first:
                self._init_results_cat(
                    name,
                    dr,
                    resp,
                    "frf",
                    SOL.f,
                    "freq",
                    mm,
                    n,
                    dohist=True,
                    dosrs=dosrs,
                )

            self._store_maxmin(res, mm, j, case)

            if dr.histpv is not None:
                res.frf[j] = resp[dr.histpv]

            if dr.srspv is not None and dosrs:
                self._compute_srs(res, dr, resp, "frf", SOL.f, j, first)

    def solvepsd(
        self,
        nas,
        case,
        DR,
        fs,
        forcepsd,
        t_frc,
        freq,
        verbose=False,
        allow_force_trimming=False,
        **kwargs,
    ):
        """
        Solve equations of motion in frequency domain with PSD forces

        See also :func:`pyyeti.ode.solvepsd` for a very similar
        routine, but one that is independent of the :mod:`cla` module.

        Parameters
        ----------
        nas : dictionary
            Typically, this is the nas2cam dictionary:
            ``nas = pyyeti.nastran.op2.rdnas2cam()``. However, only
            the "nrb" member is needed directly by this routine (for
            uncertainty factor application). It is also passed to the
            data recovery routines (the `drfunc` setting in
            :func:`DR_Def.add`).
        case : string
            Unique string identifying the case; stored in, for
            example, the ``self['SC_atm'].cases`` and the `.mincase`
            and `.maxcase` lists. Also used to index the temporary
            dictionary ``self['SC_atm']._psd`` (see Notes below).
        DR : instance of :class:`DR_Event`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.DR_Event()``). It is an event specific version
            of all combined :class:`DR_Def` objects with all ULVS
            matrices applied.
        fs : class instance
            An instance of :class:`pyyeti.ode.SolveUnc` or
            :class:`pyyeti.ode.FreqDirect` (or similar ... must have
            ``.fsolve`` method)
        forcepsd : 2d array_like
            Matrix of force psds; each row is a force
        t_frc : 2d array_like
            Transform to put `forcepsd` into the coordinates of the
            equations of motion: ``t_frc @ forcepsd``. Commonly,
            `t_frc` is simply the transpose of a row-partition of the
            mode shape matrix (phi) and the conversion of `forcepsd`
            is from physical space to modal space. In that case, the
            row-partition is from the full set down to just the forced
            DOF. However, `t_frc` can also have force mappings (as
            from the TLOAD capability in Nastran); in that case,
            ``t_frc = phi.T @ mapping_vectors``. In any case, the
            number of columns in `t_frc` is the number of rows in
            `forcepsd`: ``t_frc.shape[1] == forcepsd.shape[0]``
        freq : 1d array_like
            Frequency vector at which solution will be computed
        verbose : bool; optional
            If True, print status messages and timer results.
        allow_force_trimming : bool; optional
            If True, zero forces will be trimmed off to save
            time. Since this can cause trouble during data recovery if
            you have any force-dependent data recovery matrices
            ("drmf"), the default is False. It is advisable to trim
            off zero forces before calling this routine and trim the
            corresponding columns off any "drmf" matrices.
        **kwargs : keyword arguments for ``fs.fsolve``; optional
            Currently, there are two arguments available:

            ============  ============================================
              argument    brief description
            ============  ============================================
            incrb         specifies how to handle rigid-body responses
            rf_disp_only  specifies how to handle residual-flexibility
                          modes
            ============  ============================================

            See, for example, :func:`pyyeti.ode.SolveUnc.fsolve`.

        Notes
        -----
        The `self` results dictionary is updated (see
        :class:`DR_Results` for an example).

        This routine calls :func:`DR_Event.frf_apply_uf` to apply the
        uncertainty factors.

        The response PSDs are stored in a temporary dictionary index
        by the case; eg: ``self['SC_atm'][case]._psd``. The routine
        :func:`psd_data_recovery` operates on this variable and
        deletes it after processing that last case. Note that some or
        all rows of this variable might be saved in
        ``self['SC_atm'].psd``; this is determined according to the
        `histpv` setting defined through the call to
        :func:`DR_Def.add`.

        The solution loop is essentially as follows, written in a
        python-like pseudo-code::

            for cat in categories:
                cat[case]._psd = 0.0

            for i in range(forcepsd.shape[0]):
                # solve for unit frequency response function for i'th
                # force:
                genforce = t_frc[:, [i]] @ unitforce;
                sol = fs.fsolve(genforce, freq, **kwargs)
                sol.pg[:] = 0.0
                sol.pg[i] = unitforce
                sol = DR.frf_apply_uf(sol, nas["nrb"])

                # compute the unit frf's for all categories:
                for cat in categories:
                    # call the drfunc set in cla.DR_Def.add for
                    # current category
                    resp = cat_drfunc(sol[uf_reds], nas, DR.Vars, se)
                    # ex: resp = (ltma @ sol.a + ltmd @ sol.d
                    #             + ltmf @ sol.pg)

                    # compute psd response and add it on:
                    cat._psd[case] += forcepsd[i] * abs(resp) ** 2

        .. note::
            If you have force-dependent data recovery matrices (which
            is typically from using the mode-acceleration method of
            data recovery for fundamentally displacement-based
            quantities), you may need to trim/reorder its columns to
            match `forcepsd`. Note that while looping over each PSD in
            `forcepsd`, this routine creates ``sol.pg`` sized
            compatibly with `forcepsd` and has only one row of 1.0
            values. See also `allow_force_trimming` above.
        """
        forcepsd, t_frc = np.atleast_2d(forcepsd, t_frc)
        if t_frc.shape[1] != forcepsd.shape[0]:
            raise ValueError(
                "`forcepsd` and `t_frc` are incompatibly "
                f"sized: {forcepsd.shape} vs {t_frc.shape}"
            )

        nonzero_forces = np.any(forcepsd, axis=1).nonzero()[0]
        nzero = forcepsd.shape[0] - nonzero_forces.size
        if nzero > 0:
            if allow_force_trimming:
                if verbose:
                    print(f"Trimming off {nzero} " "zero forces")
                forcepsd = forcepsd[nonzero_forces]
                t_frc = t_frc[:, nonzero_forces]
            else:
                # if verbose:
                warnings.warn(
                    f"There are {nzero} zero forces that are NOT being trimmed "
                    "off. See the parameter `allow_force_trimming`.",
                    RuntimeWarning,
                )

        freq = np.atleast_1d(freq)
        rpsd = forcepsd.shape[0]
        unitforce = np.ones(freq.shape)

        # initialize categories for data recovery
        drfuncs = {}
        for key, value in self.items():
            if "_psd" not in value.__dict__:
                value.freq = freq
                value._psd = {}
            else:
                if not np.allclose(value.freq, freq):
                    raise ValueError("`freq` must match `freq` from previous case")
            value._psd[case] = 0.0
            # get data recovery functions just once, outside of main
            # loop; returns tuple: (func, func_psd) ... func_psd will
            # be None if no special function defined for PSD
            # recovery):
            drfuncs[key] = get_drfunc(
                value.drminfo.drfile, value.drminfo.drfunc, get_psd=True
            )

        import time

        pg = np.zeros((t_frc.shape[1], freq.size))
        timers = [0, 0, 0]
        for i in range(rpsd):
            if verbose:
                print(f"{case}: processing force {i + 1} of {rpsd}")
            # solve for unit FRF for i'th force:
            genforce = t_frc[:, [i]] * unitforce
            t1 = time.time()
            sol = fs.fsolve(genforce, freq, **kwargs)

            # zeroing line not needed on first loop, but is okay (last
            # row gets re-zeroed)
            pg[i - 1] = 0.0
            pg[i] = unitforce
            sol.pg = pg
            timers[0] += time.time() - t1

            # apply uncertainty factors:
            t1 = time.time()
            sol = DR.frf_apply_uf(sol, nas["nrb"])
            # sol = DR.apply_uf(sol, *mbk, nas['nrb'], rfmodes)
            timers[1] += time.time() - t1

            # perform data recovery:
            t1 = time.time()
            for key, value in self.items():
                uf_reds = value.drminfo.uf_reds
                se = value.drminfo.se
                if drfuncs[key][1]:
                    # use PSD recovery function if present:
                    drfuncs[key][1](
                        sol[uf_reds], nas, DR.Vars, se, freq, forcepsd, value, case, i
                    )
                else:
                    # otherwise, use normal recovery function:
                    resp = drfuncs[key][0](sol[uf_reds], nas, DR.Vars, se)
                    value._psd[case] += forcepsd[i] * abs(resp) ** 2
            timers[2] += time.time() - t1
        if verbose:
            print("timers =", timers)

    def psd_data_recovery(
        self, case, DR, n, j, dosrs=True, peak_factor=3.0, resp_time=None
    ):
        """
        PSD data recovery function

        Parameters
        ----------
        case : string
            Unique string identifying the case; stored in the
            ``self[name].cases`` and the `.mincase` and `.maxcase`
            lists
        DR : instance of :class:`DR_Event`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.DR_Event()``). It is an event specific version
            of all combined :class:`DR_Def` objects with all ULVS
            matrices applied.
        n : integer
            Total number of load cases. This is the number of times
            :func:`solvepsd` and this routine get called, not the
            number of forces in a particular PSD force matrix.
        j : integer
            Current load case number starting at zero
        dosrs : bool; optional
            If False, do not calculate SRSs; default is to calculate
            them.
        peak_factor : scalar; optional
            Factor to multiply each RMS by to get a peak value. See
            also `resp_time`. RMS stands for root-mean-square: the
            square-root of the area under the PSD curve, and the area
            is the mean-square value. The default value of 3.0 is
            often used to get a 3-sigma value (for zero mean
            responses, the RMS value is the same as the standard
            deviation).
        resp_time : scalar or None; optional
            If not None, used to compute frequency-dependent peak
            factors for SRS from: ``sqrt(2*log(resp_time*f))``. See
            :func:`pyyeti.fdepsd.fdepsd` for the derivation of this
            factor.

        Returns
        -------
        None

        Notes
        -----
        The `self` results dictionary is updated (see
        :class:`DR_Results` for an example).

        Note: the x-value entries (eg, `mx_x`, `ext_x`) are really
        the "apparent frequency" values, an estimate for the number of
        positive slope zero crossings per second [#rand1]_, [#rand2]_.

        References
        ----------
        .. [#rand1] Wirsching, Paez, Ortiz, "Random Vibrations: Theory
                    and Practice", Dover Publications, Inc., 2006.

        .. [#rand2] Bendat, Julius S., "Probability Functions for
                    Random Responses: Prediction of Peaks, Fatigue
                    Damage, and Catastrophic Failures", NASA
                    Contractor Report 33 (NASA CR-33), 1964.
        """

        def _calc_rms(df, p):
            sumpsd = p[:, :-1] + p[:, 1:]
            return np.sqrt((df * sumpsd).sum(axis=1) / 2)

        for name, res in self.items():
            first = res.ext is None
            dr = DR.Info[name]  # record with: .desc, .labels, ...
            # compute area under curve (rms):
            freq = res.freq
            freqstep = np.diff(freq)
            psd = res._psd[case]
            rms = _calc_rms(freqstep, psd)

            # Need "velocity" rms to estimate number of positive slope
            # zero crossings (apparent frequency (af)).
            #   af1 = vel_rms/disp_rms                  (rad/sec)
            #   af = vel_rms/disp_rms * (1/(2 pi))      (Hz)
            # vel_psd = (2 pi f)**2 * PSD
            # - note that the 2 pi factor cancels after square root
            vrms = _calc_rms(freqstep, freq ** 2 * psd)

            pk = peak_factor * rms
            pk_freq = vrms / rms
            mm = SimpleNamespace(
                ext=np.column_stack((pk, -pk)),
                ext_x=np.column_stack((pk_freq, pk_freq)),
            )

            extrema(res, mm, case)

            if first:
                self._init_results_cat(
                    name,
                    dr,
                    psd,
                    "psd",
                    res.freq,
                    "freq",
                    mm,
                    n,
                    dohist=True,
                    dosrs=dosrs,
                )
                res.rms = np.zeros((rms.shape[0], n))

            self._store_maxmin(res, mm, j, case)
            res.rms[:, j] = rms

            if dr.histpv is not None:
                res.psd[j] = psd[dr.histpv]

            # srs:
            if dr.srspv is not None and dosrs:
                if resp_time is not None:
                    pf = np.sqrt(2 * np.log(resp_time * dr.srsfrq))
                else:
                    pf = peak_factor
                # spec = (freq, psd[dr.srspv].T)
                self._compute_srs(res, dr, psd, "psd", freq, j, first, pf=pf)

        if j == n - 1:
            del res._psd

    def calc_ext(self):
        """
        Calculate the .ext attribute from the .mx and .mn attributes

        Notes
        -----
        Each results SimpleNamespace (eg, ``self['SECO1']['SC_ifa']``)
        is expected to have `.mx` and `.mn` members. Each of these is
        data-recovery rows x load cases. This routine will calculate a
        new `.ext` member by::

           .ext = [max(mx), min(mn)]

        If ``.srs.srs[q]`` is present, a new ``srs.ext[q]`` will be
        calculated as well. Each ``.srs.srs[q]`` is assumed to be
        cases x rows x freq.

        The `.maxcase` and `.mincase` members are set to the
        corresponding maximizing and minimizing label from
        ``.cases``. The `.ext_x` member is set to None.

        Note that this method is not needed in typical situations,
        since the extreme values are computed automatically (during
        the call to :func:`time_data_recovery`, for example).
        """
        for res in self.values():
            mx = res.mx.max(axis=1)
            mn = res.mn.min(axis=1)
            argmx = res.mx.argmax(axis=1)
            argmn = res.mn.argmin(axis=1)
            res.ext = np.column_stack((mx, mn))
            cases = res.cases
            res.maxcase = [cases[i] for i in argmx]
            res.mincase = [cases[i] for i in argmn]
            res.ext_x = None

            # handle SRS if it is there:
            if "srs" in res.__dict__:
                for Q in res.srs.srs:
                    arr = res.srs.srs[Q]
                    res.srs.ext[Q] = arr.max(axis=0)

    def calc_stat_ext(self, k):
        """
        Calculate statistical extreme response for event results

        Parameters
        ----------
        k : scalar
            The statistical k-factor: extreme = mean + k*sigma

        Notes
        -----
        Each results SimpleNamespace (eg, ``self['SECO1']['SC_ifa']``)
        is expected to have `.mx` and `.mn` members. Each of these is
        data-recovery rows x load cases. This routine will calculate a
        new `.ext` member by::

           .ext = [mean(mx) + k*std(mx), mean(mn) - k*std(mn)]

        If ``.srs.srs[q]`` is present, a new ``srs.ext[q]`` will be
        calculated as well. Each ``.srs.srs[q]`` is assumed to be
        cases x rows x freq.

        The `.maxcase` and `.mincase` members are set to 'Statistical'
        and the `.ext_x` member is set to None.

        To compute k-factors, see :func:`pyyeti.stats.ksingle` and
        :func:`pyyeti.stats.kdouble`.
        """
        for res in self.values():
            mx = res.mx.mean(axis=1) + k * res.mx.std(ddof=1, axis=1)
            mn = res.mn.mean(axis=1) - k * res.mn.std(ddof=1, axis=1)
            res.ext = np.column_stack((mx, mn))
            res.maxcase = res.mincase = ["Statistical"] * mx.shape[0]
            res.ext_x = None

            # handle SRS if it is there:
            if "srs" in res.__dict__:
                for Q in res.srs.srs:
                    arr = res.srs.srs[Q]
                    res.srs.ext[Q] = arr.mean(axis=0) + k * arr.std(ddof=1, axis=0)

    def all_base_events(self, top_level_name="Top Level"):
        """
        A generator for looping over all base events

        Parameters
        ----------
        top_level_name : str; optional
            This is the name of the event at the top level of the
            results structure

        Yields
        ------
        name : str
            The next base-level dictionary key.
        base : :class:`DR_Results` instance
            The next base-level :class:`DR_Results` item. Its entries
            are the SimpleNamespace data structures for each category.

        Notes
        -----
        Entries that are neither a :class:`DR_Results` nor a
        SimpleNamespace are quietly ignored.

        Examples
        --------
        Since this routine doesn't actually operate on the results, we
        can make up an otherwise useless and fake results structure
        for demonstration. Here, we'll have two non-base events: the
        top level one and "NonBase", and two base events: "Base1" and
        "Base2". "Base1" is under "NonBase" while "Base2" is under the
        top level. We'll also have two data recovery categories for
        each base event: 'SC_atm' and 'SC_ltm'.

        The following demonstrates three generators available for
        :class:`DR_Results`: :func:`all_base_events`,
        :func:`all_nonbase_events`, and :func:`all_categories`:

        >>> from types import SimpleNamespace
        >>> from pyyeti import cla
        >>> res = cla.DR_Results()
        >>> res['NonBase'] = cla.DR_Results()
        >>> res['NonBase']['Base1'] = cla.DR_Results()
        >>> res['NonBase']['Base1']['SC_atm'] = SimpleNamespace()
        >>> res['NonBase']['Base1']['SC_ltm'] = SimpleNamespace()
        >>> res['Base2'] = cla.DR_Results()
        >>> res['Base2']['SC_atm'] = SimpleNamespace()
        >>> res['Base2']['SC_ltm'] = SimpleNamespace()

        Show the base events:

        >>> for name, base in res.all_base_events():
        ...     print(name, ':', base)   # doctest: +ELLIPSIS
        Base1 : DR_Results (...) with 2 keys: ['SC_atm', 'SC_ltm']
        Base2 : DR_Results (...) with 2 keys: ['SC_atm', 'SC_ltm']

        Show the non-base events:

        >>> for name, nonbase in res.all_nonbase_events():
        ...     print(name, ':', nonbase)   # doctest: +ELLIPSIS
        Top Level : DR_Results (...) with 2 keys: ['NonBase', 'Base2']
        NonBase : DR_Results (...) with 1 keys: ['Base1']

        Show all the data recovery categories:

        >>> for name, cat in res.all_categories():
        ...     print(name, ':', cat)
        SC_atm : namespace()
        SC_ltm : namespace()
        SC_atm : namespace()
        SC_ltm : namespace()
        """

        def _all_bases(dct, topname):
            value = next(iter(dct.values()))
            if isinstance(value, SimpleNamespace):
                yield topname, dct
            elif isinstance(value, DR_Results):
                for name, value in dct.items():
                    yield from _all_bases(value, name)

        yield from _all_bases(self, top_level_name)

    def all_nonbase_events(self, top_level_name="Top Level"):
        """
        A generator for looping over all non-base events

        Parameters
        ----------
        top_level_name : str; optional
            This is the name of the event at the top level of the
            results structure

        Yields
        ------
        name : str
            The next non-base-level dictionary key.
        nonbase : :class:`DR_Results` instance
            The next non-base-level :class:`DR_Results` item. Its
            entries are more :class:`DR_Results` items.

        Notes
        -----
        Entries that are neither a :class:`DR_Results` nor a
        SimpleNamespace are quietly ignored.

        Examples
        --------
        See :func:`all_base_events` for an example.
        """

        def _all_nonbases(dct, topname):
            value = next(iter(dct.values()))
            if isinstance(value, DR_Results):
                yield topname, dct
                for name, value in dct.items():
                    yield from _all_nonbases(value, name)

        yield from _all_nonbases(self, top_level_name)

    def all_categories(self):
        """
        A generator for looping over all data recovery categories

        Yields
        ------
        name : str
            The next data recovery category name (the dictionary keys
            in the base-level :class:`DR_Results` items).
        cat : SimpleNamespace
            The next data recovery category SimpleNamespace with
            ``.ext``, ``.cases``, etc; see
            ``results['MaxQ']['SC_ifa']`` in :class:`DR_Results` for
            an example.

        Notes
        -----
        Entries that are neither a :class:`DR_Results` nor a
        SimpleNamespace are quietly ignored.

        Examples
        --------
        See :func:`all_base_events` for an example.
        """

        def _all_cats(dct):
            for name, value in dct.items():
                if isinstance(value, DR_Results):
                    yield from _all_cats(value)
                elif isinstance(value, SimpleNamespace):
                    yield name, value

        yield from _all_cats(self)

    def delete_extreme(self):
        """
        Delete any 'extreme' entries

        Notes
        -----
        This routine will delete the 'extreme' dictionaries at all
        levels. Those entries are typically added by
        :func:`form_extreme`. Note that :func:`form_extreme` calls
        this routine to delete any stale 'extreme' entries before
        forming any new entries.
        """
        self.pop("extreme", None)
        for value in self.values():
            if isinstance(value, DR_Results):
                value.delete_extreme()
            else:
                return

    @staticmethod
    def init_extreme_cat(cases, oldcat, ext_name="Envelope", domain="X-Value"):
        """
        Initialize an "extrema" data recovery category

        Parameters
        ----------
        cases : list
            List of strings, each naming a column for the `.mx` and
            `.mn` members.
        oldcat : SimpleNamespace
            Results data structure with attributes `.ext`, `.mx`, etc
            (see example in :class:`DR_Results`). Only used for
            determining sizing information.
        ext_name : string; optional
            Name to use for extreme results (stored in, for example,
            ``self['extreme']['SC_atm'].event``)
        domain : string or None; optional
            Typically 'time' or 'freq', but can be any string or
            None. Use None to not define a domain.

        Returns
        -------
        newcat : SimpleNamespace
            New results data structure with attributes `.ext`, `.mx`,
            etc (see example in :class:`DR_Results`).

            The `.cases` attribute is set to the input `cases` and the
            `.event` attribute is set to the input `ext_name`.

            `.drminfo` is a copy of the one in `oldcat`.

            `.mission` is a reference to the one in `oldcat`.

            The `.ext`, `.ext_x`, `.maxcase`, `.mincase` attributes
            are all set to None.

            The `.mx`, `.mn`, `.mx_x`, `.mn_x`. `.srs.srs` (if
            present) are all filled with ``np.nan``.

            The `.srs.ext` dictionary is a deep copy of the one in
            `oldcat`.

        Notes
        -----
        This is normally called indirectly via the
        :func:`form_extreme` routine. However, it can also be handy
        when implementing combination equations, for example.

        Here is an example combination equation implementation. Both
        `ss` and `noise` are instances of the :class:`DR_Results` and
        those analyses are complete. The combination equation is
        simply to add the peaks of these two components::

            comb = cla.DR_Results()
            for cat, ss_sns in ss.items():
                sns = comb.init_extreme_cat(
                    ['ss', 'noise'], ss_sns, domain='combination')
                ssext = ss[cat].mx
                noiseext = abs(noise[cat].ext).max(axis=1)[:, None]
                sns.mx = np.column_stack((ssext, noise[cat].mx))
                sns.mn = np.column_stack((ssext, noise[cat].mn))
                sns.ext = np.column_stack((ssext+noiseext,
                                           ssext-noiseext))
                sns.maxcase = ['Combination']*ssext.shape[0]
                sns.mincase = sns.maxcase

                # srs:
                if getattr(sns, 'srs', None):
                    _srs = sns.srs
                    for Q, ss_srs in ss[cat].srs.ext.items():
                        _srs.srs[Q][0] = ss_srs
                        _srs.srs[Q][1] = noise[cat].srs.ext[Q]
                        _srs.ext[Q][:] = _srs.srs[Q].sum(axis=0)
                comb[cat] = sns
        """
        ncases = len(cases)
        nrows = oldcat.ext.shape[0]
        mx = np.empty((nrows, ncases))
        mn = np.empty((nrows, ncases))
        mx_x = np.empty((nrows, ncases))
        mn_x = np.empty((nrows, ncases))
        mx[:] = np.nan
        mn[:] = np.nan
        mx_x[:] = np.nan
        mn_x[:] = np.nan
        drminfo = copy.copy(oldcat.drminfo)

        ret = SimpleNamespace(
            cases=cases,
            drminfo=drminfo,
            mission=oldcat.mission,
            event=ext_name,
            ext=None,
            ext_x=None,
            maxcase=None,
            mincase=None,
            mx=mx,
            mn=mn,
            mx_x=mx_x,
            mn_x=mn_x,
            domain=domain,
        )

        # handle SRS if present:
        osrs = getattr(oldcat, "srs", None)
        if osrs is not None:
            srs_ns = copy.copy(osrs)
            srs_ns.ext = copy.deepcopy(osrs.ext)
            srs_ns.srs = {}
            ndof, nf = next(iter(osrs.ext.values())).shape
            for q in srs_ns.ext:
                srs_ns.srs[q] = np.empty((ncases, ndof, nf))
                srs_ns.srs[q][:] = np.nan
            ret.srs = srs_ns

        return ret

    def form_extreme(self, ext_name="Envelope", case_order=None, doappend=2):
        """
        Form extreme response over sets of results

        Parameters
        ----------
        ext_name : string; optional
            Name to use for extreme results (stored in, for example,
            ``self['extreme']['SC_atm'].event``)
        case_order : list or None; optional
            List of cases (or events) in desired order. Can be subset
            of cases available. If None, the order is determined by
            the order in which results were inserted
            (:class:`DR_Results` is a
            :class:`collections.OrderedDict`). Note that `case_order`
            is used for highest level only. `case_order` defines the
            'cases' member variable (for example,
            ``self['extreme']['SC_atm'].cases``).
        doappend : integer; optional
            Flag that defines how to build the extreme `.maxcase` and
            `.mincase` values. Both the lower level `.maxcase` and
            `.mincase` values and the dictionary key (which is
            typically the event name) can be used/combined. See the
            notes section below for an example of the different
            options. The options are:

            ==========  ==============================================
            `doappend`  Description
            ==========  ==============================================
                 0      Ignore all lower level `.maxcase` & `.mincase`
                        and just use higher level key.
                 1      Keep all lower level `.maxcase` & `.mincase`
                        and prepend higher level keys as levels are
                        traversed.
                 2      Ignore lowest level `.maxcase` & `.mincase`,
                        but prepend higher level keys after that.
                 3      Keep only lowest level `.maxcase` & `.mincase`
                        (do not append any keys).
            ==========  ==============================================

        Notes
        -----
        This routine will create 'extreme' dictionaries at all
        appropriate levels. Any old 'extreme' dictionaries (at all
        levels) are deleted before anything else is done.

        The extreme values from the events (eg,
        ``self['Liftoff']['SC_atm'].ext``) are collected into the
        max/min attributes in the new 'extreme' category (eg, into
        ``self['extreme']['SC_atm'].mx`` and ``<...>.mn``). The
        ``.cases`` attribute lists the events in the order they are
        assembled.

        To demonstrate the different `doappend` options, here is an
        example :class:`DR_Results` structure showing the extreme
        `.maxcase` values for each setting. This :class:`DR_Results`
        structure has two-levels: 'Gust' at the highest level and
        'Yaw' at the lower level. The 'extreme' entries are added by
        this routine. The first `.maxcase` setting shown in this
        example could be shorthand for, for example,
        ``self['Gust']['Yaw']['SC_atm'].maxcase[0]``.

        .. code-block:: none

            'Gust'   :
                'Yaw'    : .maxcase = 'Frq 3'
                'extreme': .maxcase = 'Yaw'        if doappend = 0
                           .maxcase = 'Yaw,Frq 3'  if doappend = 1
                           .maxcase = 'Yaw'        if doappend = 2
                           .maxcase = 'Frq 3'      if doappend = 3
            'extreme': .maxcase = 'Gust'             if doappend = 0
                       .maxcase = 'Gust,Yaw,Frq 3'   if doappend = 1
                       .maxcase = 'Gust,Yaw'         if doappend = 2
                       .maxcase = 'Frq 3'            if doappend = 3
        """
        DEFDOMAIN = "X-Value"

        def _mk_case_lbls(case, val, use_ext, doappend):
            case = str(case)
            if use_ext and doappend == 2:
                doappend = 1
            maxcase = mincase = case
            # handle 1 and 3 settings:
            if "maxcase" in val.__dict__:  # always true?
                if doappend == 1:
                    maxcase = [case + "," + i for i in val.maxcase]
                    mincase = [case + "," + i for i in val.mincase]
                elif doappend == 3:
                    maxcase = val.maxcase
                    mincase = val.mincase
            return maxcase, mincase

        def _expand(ext_old, labels, pv):
            # Expand:
            #   ext, ext_x, maxcase, mincase,
            #   mx, mn, mx_x, mn_x
            # Note: this function only called when expansion is needed
            n = len(labels)
            ext_new = copy.copy(ext_old)
            ext_new.drminfo = copy.copy(ext_old.drminfo)
            ext_new.drminfo.labels = labels
            for name in ["ext", "ext_x", "mx", "mn", "mx_x", "mn_x"]:
                old = ext_old.__dict__[name]
                if old is not None:
                    new = np.empty((n, old.shape[1]))
                    new[:] = np.nan
                    new[pv] = old
                    ext_new.__dict__[name] = new
            if ext_old.maxcase is not None:
                maxcase = ["n/a" for i in range(n)]
                mincase = ["n/a" for i in range(n)]
                for i, j in enumerate(pv):
                    maxcase[j] = ext_old.maxcase[i]
                    mincase[j] = ext_old.mincase[i]
                ext_new.maxcase = maxcase
                ext_new.mincase = mincase
            return ext_new

        def _check_row_compatibility(ext1, ext2):
            # if row labels differ, expand them to accommodate
            # each other
            l1 = ext1.drminfo.labels
            l2 = ext2.drminfo.labels
            if l1 == l2:
                return ext1, ext2
            for lbls in (l1, l2):
                if len(lbls) != len(set(lbls)):
                    msg = (
                        f'Row labels for "{ext1.drminfo.desc}" (event "{ext1.event}") '
                        f'are not all unique. Cannot compare to event "{ext2.event}".'
                    )
                    raise ValueError(msg)
            # for both ext1 and ext2, expand:
            #   [ext, ext_x, maxcase, mincase, mx, mn, mx_x, mn_x]
            l3, pv1, pv2 = locate.merge_lists(l1, l2)
            return (_expand(ext1, l3, pv1), _expand(ext2, l3, pv2))

        def _calc_extreme(dct, ext_name, case_order, doappend):
            if case_order is None:
                cases = list(dct)
            else:
                cases = [str(i) for i in case_order]
            new_ext = DR_Results()
            domain = None
            for j, case in enumerate(cases):
                try:
                    curext = dct[case]["extreme"]
                    use_ext = True
                except KeyError:
                    curext = dct[case]
                    use_ext = False
                domain = None
                for drm, val in curext.items():
                    if drm not in new_ext:
                        new_ext[drm] = new_ext.init_extreme_cat(
                            cases, val, ext_name, DEFDOMAIN
                        )
                    else:
                        new_ext[drm], val = _check_row_compatibility(new_ext[drm], val)
                    if domain is not None:
                        if domain != val.domain:
                            domain = DEFDOMAIN
                    else:
                        domain = val.domain
                    maxcase, mincase = _mk_case_lbls(
                        case, val, use_ext, doappend=doappend
                    )
                    extrema(new_ext[drm], val, maxcase, mincase, j)

                    osrs = getattr(val, "srs", None)
                    if osrs is not None:
                        _ext = new_ext[drm].srs.ext
                        _srs = new_ext[drm].srs.srs
                        for Q, S in osrs.ext.items():
                            _ext[Q] = np.fmax(_ext[Q], S)
                            _srs[Q][j] = S

            if domain != DEFDOMAIN:
                for val in new_ext.values():
                    val.domain = domain
            return new_ext

        def _add_extreme(dct, ext_name, case_order, doappend):
            for name, value in list(dct.items()):
                if isinstance(value, SimpleNamespace):
                    # one level too deep ... just return quietly
                    return
                # use ext_name, case_order only at the top level
                _add_extreme(value, name, None, doappend)
            dct["extreme"] = _calc_extreme(dct, ext_name, case_order, doappend)

        # main routine:
        self.delete_extreme()
        _add_extreme(self, ext_name, case_order, doappend)

    def strip_hists(self):
        """
        Strips out response histories and non-extreme srs data

        Notes
        -----
        This is typically to reduce file size of a summary results
        structure where the time and frequency domain histories are
        already saved in other files. Run this before
        :func:`DR_Results.form_extreme` (or rerun
        :func:`DR_Results.form_extreme` afterward).

        See example usage in :func:`DR_Results.merge`.
        """
        for name, cat in self.all_categories():
            for attr in ("hist", "time", "psd", "freq"):
                try:
                    delattr(cat, attr)
                except AttributeError:
                    pass
            if hasattr(cat, "srs"):
                try:
                    delattr(cat.srs, "srs")
                except AttributeError:  # pragma: no cover
                    pass

    def rptext(
        self,
        event=None,
        drms=None,
        direc="ext",
        doabsmax=False,
        numform="{:13.5e}",
        perpage=-1,
    ):
        """
        Writes .ext files for all max/min results.

        Parameters
        ----------
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[name].event``
        drms : list of data recovery categories or None; optional
            Data recovery categories to compare. If None, compare all
            available.
        direc : string; optional
            Name of directory to put tables; will be created if
            doesn't exist
        doabsmax : bool; optional
            If True, report only absolute maximums. Note that signs
            are retained.
        numform : string; optional
            Format of the max & min numbers.
        perpage : integer; optional
            The number of lines to write perpage. If < 0, there is no
            limit (one page).

        Notes
        -----
        The output files contain the maximums, minimums and cases as
        applicable. The file names are determined from the category
        names.
        """
        if not os.path.exists(direc):
            os.mkdir(direc)
        if drms is None:
            drms = self.keys()
        for drm in drms:
            res = self[drm]
            self._check_labels_len(drm, res)
            mission = res.mission
            if event is None:
                event = res.event
            title = f"{mission} - {event} Extrema Results"
            filename = os.path.join(direc, drm + ".ext")
            rptext1(
                res,
                filename,
                title=title,
                doabsmax=doabsmax,
                numform=numform,
                perpage=perpage,
            )

    def rpttab(
        self, event=None, drms=None, direc="tab", count_filter=1e-6, excel=False
    ):
        """
        Write results tables with bin count information.

        Parameters
        ----------
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[name].event``
        drms : list of data recovery categories or None; optional
            Data recovery categories to compare. If None, compare all
            available.
        direc : string; optional
            Name of directory to put tables; will be created if
            doesn't exist
        count_filter : scalar; optional
            Filter to use for the bin count; only numbers larger (in
            the absolute value sense) than the filter are counted
        excel : bool or string; optional
            If True, a Microsoft Excel file (with the '.xlsx'
            extension) is written instead of a normal text file for
            each data recovery category. If a string, a single '.xlsx'
            file named ``excel + '.xlsx'`` is created with all data
            recovery categories in it.

        Notes
        -----
        The output files contain the maximums, minimums, abs-max
        tables. The extrema value is also included along with the case
        that produced it.

        After those three tables, a table of bin counts is printed
        showing the number of extrema values produced by each event.

        The file names are determined from the category names (unless
        `excel` provides the single '.xlsx' filename).
        """
        if not os.path.exists(direc):
            os.mkdir(direc)
        if isinstance(excel, str):
            # create a single excel file
            filename = os.path.join(direc, excel + ".xlsx")
            opts = {"nan_inf_to_errors": True}
            workbook = xlsxwriter.Workbook(filename, opts)
            filename = workbook
        else:
            workbook = None
        try:
            if drms is None:
                drms = self.keys()
            for drm in drms:
                res = self[drm]
                self._check_labels_len(drm, res)
                mission = res.mission
                if event is None:
                    event = res.event
                ttl = f"{mission} - {event} Extrema Results and Bin Count Tables"
                if excel:
                    if not isinstance(excel, str):
                        filename = os.path.join(direc, drm + ".xlsx")
                else:
                    filename = os.path.join(direc, drm + ".tab")
                rpttab1(res, filename, title=ttl, count_filter=count_filter, name=drm)
        finally:
            if workbook is not None:
                workbook.close()

    def rptpct(
        self,
        refres,
        names=("Self", "Reference"),
        event=None,
        drms=None,
        fileext=".cmp",
        direc="compare",
        keyconv=None,
        **rptpct1_args,
    ):
        """
        Write comparison files for all max/min data in results.

        Parameters
        ----------
        refres : dictionary
            Dictionary of reference results to compare to. Keys are
            the category names and values are either:

              1. A 2-column array_like of ``[max, min]``, or
              2. A SimpleNamespace with the ``.ext`` (``[max, min]``)
                 member and, optionally, the ``.drminfo.labels``
                 member. If present, the labels will be used to
                 compare only the rows with the same labels.

            Notes:

              1. If the keys are different than those in ``self`` (eg,
                 'SC_ifa', 'SC_atm', etc), then the input `keyconv` is
                 necessary.

              2. If a key in ``self`` is not found in `refres`, a
                 message is printed and that item is skipped.

        names : list/tuple; optional
            2-element list or tuple identifying the two sets of
            results that are being compared. The first is for `self`
            and the second is for `refres`.
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[category].event``. Used for titling.
        drms : list of data recovery categories or None; optional
            Data recovery categories to compare. If None, compare all
            available.
        fileext : string; optional
            String to attach on to the DRM name for filename creation
        direc : string; optional
            Name of directory to put tables; will be created if
            doesn't exist
        keyconv : dictionary or None; optional
            Dictionary to map ``self[event]`` keys to the keys of
            `refres`. The keys are those of ``self[event]`` and the
            values are the keys in `refres`. Only those keys that need
            converting are required. For example, if ``self[event]``
            uses the name 'SC_atm', but `refres` uses 'SCATM', then,
            assuming all other names match,
            ``keyconv = {'SC_atm': 'SCATM'}`` would be sufficient.
            Note: if a key is in `keyconv`, it is used even if
            ``refres[key]`` exists.
        rptpct1_args : dict
            All remaining named args are passed to :func:`rptpct1`
        """
        if not os.path.exists(direc):
            os.mkdir(direc)
        skipdrms = []
        if keyconv is None:
            keyconv = {}
        if drms is None:
            drms = self.keys()
        for drm in drms:
            refdrm = keyconv[drm] if drm in keyconv else drm
            if refdrm not in refres:
                skipdrms.append(drm)
            else:
                res = self[drm]
                self._check_labels_len(drm, res)
                mission = res.mission
                if event is None:
                    event = res.event
                title = f"{mission}, {event} - {names[0]} vs. {names[1]}"
                filename = os.path.join(direc, drm + fileext)
                rptpct1(
                    res,
                    refres[refdrm],
                    filename,
                    title=title,
                    names=names,
                    **rptpct1_args,
                )
        if len(skipdrms) > 0:
            warnings.warn(
                "Some comparisons were skipped (not found in `refres`):\n"
                f"{str(skipdrms)}",
                RuntimeWarning,
            )

    def srs_plots(
        self,
        event=None,
        Q="auto",
        drms=None,
        inc0rb=True,
        fmt="pdf",
        onepdf=True,
        layout=(2, 3),
        figsize=(11, 8.5),
        showall=None,
        showboth=False,
        direc="srs_plots",
        tight_layout_args=None,
        plot="plot",
        show_figures=False,
    ):
        """
        Make SRS plots with optional printing to .pdf or .png files.

        Parameters
        ----------
        event : string or None; optional
            String for plot titles and file names (eg: 'Liftoff'). If
            None, `event` is determined from, for example,
            ``self['SC_atm'].event``
        Q : scalar or iterable or 'auto'; optional
            The Q value(s) to plot. If 'auto', all the Q values for
            each category are plotted. Must be a scalar if `showall`
            is True (see below).
        drms : list of data recovery categories or None; optional
            Data recovery categories to plot. If None, plot all
            available. See also input `inc0rb`.
        inc0rb : bool; optional
            If True, the '_0rb' versions of each data recovery
            category are automatically included.
        fmt : string or None; optional
            If `fmt` == "pdf", all plots are written to one PDF file,
            unless `onepdf` is set to False.  If `fmt` is some other
            string, it is used as the `format` parameter in
            :meth:`matplotlib.figure.Figure.savefig`. If None, no
            figures will be saved. Typical values for `fmt` are (from
            ``fig.canvas.get_supported_filetypes()``)::

                'eps': 'Encapsulated Postscript',
                'jpeg': 'Joint Photographic Experts Group',
                'jpg': 'Joint Photographic Experts Group',
                'pdf': 'Portable Document Format',
                'pgf': 'PGF code for LaTeX',
                'png': 'Portable Network Graphics',
                'ps': 'Postscript',
                'raw': 'Raw RGBA bitmap',
                'rgba': 'Raw RGBA bitmap',
                'svg': 'Scalable Vector Graphics',
                'svgz': 'Scalable Vector Graphics',
                'tif': 'Tagged Image File Format',
                'tiff': 'Tagged Image File Format'

            File naming conventions: if 'SC_atm' is a category, then
            example output filenames could be::

                'SC_atm_srs.pdf'
                'SC_atm_eqsine.pdf'
                'SC_atm_srs_0.png', 'SC_atm_srs_1.png', ...
                'SC_atm_eqsine_0.png', 'SC_atm_eqsine_1.png', ...

        onepdf : bool or string; optional
            If `onepdf` evaluates to True and `fmt` is 'pdf', all
            plots are written to one PDF file where the name is:

              ========   ========================================
              `onepdf`   PDF file name
              ========   ========================================
               string    All plots saved in: `onepdf` + ".pdf"
               True      All plots saved in: `event` + "_srs.pdf"
              ========   ========================================

            If False, each figure is saved to its own file named
            as described above (see `fmt`).
        layout : 2-element tuple/list; optional
            Subplot layout, eg: (2, 3) for 2 rows by 3 columns
        figsize : 2-element tuple/list; optional
            Define page size in inches.
        showall : bool or None; optional
            If True, show all SRS curves for all cases; otherwise just
            plot envelope. If None and `showboth` is True, `showall`
            is set to True.
        showboth : bool; optional
            If True, shows all SRS curves and the envelope; otherwise
            just plot which ever `showall` indicates.
        direc : string; optional
            Directory name to put all output plot files; will be
            created if it doesn't exist.
        tight_layout_args : dict or None; optional
            Arguments for
            :meth:`matplotlib.figure.Figure.tight_layout`. If None,
            defaults to::

                {'pad': 3.0,
                 'w_pad': 2.0,
                 'h_pad': 2.0,
                 'rect': (0.3 / figsize[0],
                          0.3 / figsize[1],
                          1.0 - 0.3 / figsize[0],
                          1.0 - 0.3 / figsize[1])}

        plot : string; optional
            The name of a function in :class:`matplotlib.axes.Axes`
            that will draw each curve. Defaults to "plot". Common
            options:

                +------------+
                | `plot`     |
                +============+
                | "plot"     |
                +------------+
                | "loglog"   |
                +------------+
                | "semilogx" |
                +------------+
                | "semilogy" |
                +------------+

        show_figures : bool; optional
            If True, plot figures will be displayed on the screen for
            interactive viewing. Warning: there may be many figures.

        Returns
        -------
        figs : list
            List of figure handles created by this routine.

        Notes
        -----
        This routine is an interface to the :func:`mk_plots` routine.

        Set the `onepdf` parameter to a string to specify the name of
        the PDF file.

        For example::

            # write a pdf file to 'srs_plots/':
            results.srs_plots()
            # write png file(s) to 'png/':
            results.srs_plots(fmt='png', direc='png')
        """
        return mk_plots(
            self,
            issrs=True,
            event=event,
            Q=Q,
            drms=drms,
            inc0rb=inc0rb,
            fmt=fmt,
            onepdf=onepdf,
            layout=layout,
            figsize=figsize,
            showall=showall,
            showboth=showboth,
            direc=direc,
            tight_layout_args=tight_layout_args,
            cases=None,
            plot=plot,
            show_figures=show_figures,
        )

    def resp_plots(
        self,
        event=None,
        drms=None,
        inc0rb=True,
        fmt="pdf",
        onepdf=True,
        layout=(2, 3),
        figsize=(11, 8.5),
        cases=None,
        direc="resp_plots",
        tight_layout_args=None,
        plot="plot",
        show_figures=False,
    ):
        """
        Make time or frequency domain responses plots.

        Parameters
        ----------
        event : string or None; optional
            String for plot titles and file names (eg: 'Liftoff'). If
            None, `event` is determined from, for example,
            ``self['SC_atm'].event``
        drms : list of data recovery categories or None; optional
            Data recovery categories to plot. If None, plot all
            available. See also input `inc0rb`.
        inc0rb : bool; optional
            If True, the '_0rb' versions of each data recovery
            category are automatically included.
        fmt : string or None; optional
            If `fmt` == "pdf", all plots are written to one PDF file,
            unless `onepdf` is set to False.  If `fmt` is some other
            string, it is used as the `format` parameter in
            :meth:`matplotlib.figure.Figure.savefig`. If None, no
            figures will be saved. Typical values for `fmt` are (from
            ``fig.canvas.get_supported_filetypes()``)::

                'eps': 'Encapsulated Postscript',
                'jpeg': 'Joint Photographic Experts Group',
                'jpg': 'Joint Photographic Experts Group',
                'pdf': 'Portable Document Format',
                'pgf': 'PGF code for LaTeX',
                'png': 'Portable Network Graphics',
                'ps': 'Postscript',
                'raw': 'Raw RGBA bitmap',
                'rgba': 'Raw RGBA bitmap',
                'svg': 'Scalable Vector Graphics',
                'svgz': 'Scalable Vector Graphics',
                'tif': 'Tagged Image File Format',
                'tiff': 'Tagged Image File Format'

            File naming conventions: if 'SC_atm' is a category, then
            example output filenames could be::

                'SC_atm_srs.pdf'
                'SC_atm_eqsine.pdf'
                'SC_atm_srs_0.png', 'SC_atm_srs_1.png', ...
                'SC_atm_eqsine_0.png', 'SC_atm_eqsine_1.png', ...

        onepdf : bool or string; optional
            If `onepdf` evaluates to True and `fmt` is 'pdf', all
            plots are written to one PDF file where the name is:

              ========   =========================================
              `onepdf`   PDF file name
              ========   =========================================
               string    All plots saved in: `onepdf` + ".pdf"
               True      All plots saved in: `event` + "_??.pdf",
                         where "??" is either 'hist', 'psd', or
                         'frf' as appropriate.
              ========   =========================================

            If False, each figure is saved to its own file named
            as described above (see `fmt`).
        layout : 2-element tuple/list; optional
            Subplot layout, eg: (2, 3) for 2 rows by 3 columns
        figsize : 2-element tuple/list; optional
            Define page size in inches.
        cases : tuple/list of case names to plot or None; optional
            If None, all cases are plotted.
        direc : string; optional
            Directory name to put all output plot files; will be
            created if it doesn't exist.
        tight_layout_args : dict or None; optional
            Arguments for
            :meth:`matplotlib.figure.Figure.tight_layout`. If None,
            defaults to::

                {'pad': 3.0,
                 'w_pad': 2.0,
                 'h_pad': 2.0,
                 'rect': (0.3 / figsize[0],
                          0.3 / figsize[1],
                          1.0 - 0.3 / figsize[0],
                          1.0 - 0.3 / figsize[1])}

        plot : string; optional
            The name of a function in :class:`matplotlib.axes.Axes`
            that will draw each curve. Defaults to "plot". Common
            options:

                +------------+
                | `plot`     |
                +============+
                | "plot"     |
                +------------+
                | "loglog"   |
                +------------+
                | "semilogx" |
                +------------+
                | "semilogy" |
                +------------+

        show_figures : bool; optional
            If True, plot figures will be displayed on the screen for
            interactive viewing. Warning: there may be many figures.

        Returns
        -------
        figs : list
            List of figure handles created by this routine.

        Notes
        -----
        This routine is an interface to the :func:`mk_plots` routine.

        Set the `onepdf` parameter to a string to specify the name of
        the PDF file.

        For example::

            # write a pdf file to 'resp_plots/':
            results.resp_plots()
            # write png file(s) to 'png/':
            results.resp_plots(fmt='png', direc='png')
        """
        return mk_plots(
            self,
            issrs=False,
            event=event,
            drms=drms,
            inc0rb=inc0rb,
            fmt=fmt,
            onepdf=onepdf,
            layout=layout,
            figsize=figsize,
            cases=cases,
            direc=direc,
            tight_layout_args=tight_layout_args,
            Q="auto",
            showall=None,
            showboth=False,
            plot=plot,
            show_figures=show_figures,
        )


# setup pickling for a little bit of future-proofing:
def unpickle_drresults(kwargs):
    # pickle_version is not used yet
    pickle_version = kwargs.pop("__pickle_version", 0)
    new_drresults = DR_Results()
    for k, v in kwargs.items():
        new_drresults[k] = v
    return new_drresults


def pickle_drresults(drresults):
    odct = OrderedDict(drresults)
    odct["__pickle_version"] = 1
    return unpickle_drresults, (odct,)


copyreg.pickle(DR_Results, pickle_drresults)
