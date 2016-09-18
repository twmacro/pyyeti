# -*- coding: utf-8 -*-
"""
Collection of tools used for CLA - coupled loads analysis
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import pandas as pd
import importlib
import os
import sys
import copy
import datetime
import pickle
import gzip
import numbers
import inspect
from collections import abc, OrderedDict
from io import StringIO
from types import SimpleNamespace
from warnings import warn
import xlsxwriter
from pyyeti import ytools, locate, srs, n2p, writer, ode
from pyyeti.ytools import save, load


def magpct(M1, M2, Ref=None, ismax=None, symbols=None):
    """
    Plot percent differences in two sets of values vs magnitude.

    Parameters
    ----------
    M1, M2 : 1d or 2d array_like
        The two sets of values to compare. Must have the same shape.
        If 2d, each column is compared.
    Ref : 1d or 2d array_like or None; optional
        Same size as `M1` and `M2` and is used as the reference
        values. If None, ``Ref = M2``.
    ismax : bool or None; optional
        If None, the sign of the percent differences is determined by
        ``M1 - M2`` (the normal way).  Otherwise, the sign is set to
        be positive where `M1` is more extreme than `M2`. More extreme
        is higher if `ismax` is True (comparing maximums), and lower
        if `ismax` is False (comparing minimums).
    symbols : iterable or None; optional
        Plot marker iterable (eg: string, list, tuple) that specifies
        the marker for each column. Values in `symbols` are reused if
        necessary. If None, :func:`get_marker_cycle` is used.

    Returns
    -------
    pds : list
        List of 1d percent differences, one numpy array for each
        column in `M1` and `M2`. Each 1d array contains only the
        percent differences where ``M2 != 0.0``. If `M2` is all zero
        for a column, the corresponding entry in `pds` is None.

    Notes
    -----
    The percent differences, ``(M1-M2)/Ref*100``, are plotted against
    the magnitude of `Ref`. If `ismax` is not None, signs are set as
    defined above so that positive percent differences indicate where
    `M1` is more extreme than `M2`.

    If desired, setup the plot axes before calling this routine.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti import cla
        >>> m1 = np.arange(200) + np.random.randn(200)
        >>> m2 = m1 + np.random.randn(200)
        >>> pds = cla.magpct(m1, m2)

    """
    if Ref is None:
        M1, M2 = np.atleast_1d(M1, M2)
        Ref = M2
    else:
        M1, M2, Ref = np.atleast_1d(M1, M2, Ref)

    if M1.shape != M2.shape or M1.shape != Ref.shape:
        raise ValueError('`M1`, `M2` and `Ref` must all have the same shape')

    def _get_next_pds(M1, M2, Ref, ismax):
        def _getpds(m1, m2, ref, ismax):
            pv = ref != 0
            if not np.any(pv):
                return None, None
            a = m1[pv]
            b = m2[pv]
            pdiff = (a - b)/ref[pv]*100.0
            if ismax is not None:
                pdiff = abs(pdiff)
                neg = a < b if ismax else a > b
                pdiff[neg] *= -1.0
            return pdiff, ref[pv]

        if M1.ndim == 1:
            yield _getpds(M1, M2, Ref, ismax)
        else:
            for c in range(M1.shape[1]):
                yield _getpds(M1[:, c], M2[:, c], Ref[:, c], ismax)

    marker = get_marker_cycle()
    pds = []
    for curpd, ref in _get_next_pds(M1, M2, Ref, ismax):
        pds.append(curpd)
        if curpd is None:
            continue
        apd = abs(curpd)
        _marker = next(marker)
        for pv, c in [((apd <= 5).nonzero()[0], 'b'),
                      (((apd > 5) & (apd <= 10)).nonzero()[0], 'm'),
                      ((apd > 10).nonzero()[0], 'r')]:
            plt.plot(ref[pv], curpd[pv], c+_marker)
    plt.xlabel('Reference Magnitude')
    plt.ylabel('% Difference')
    return pds


def rdext(name):
    """
    Function to read file written by rptext.cam. Probably temporary

    Parameters
    ----------
    name : string
        Name of .ext file. It is assumed to be one page.

    Returns
    -------
    ext : 2d ndarray
        Matrix of results: [row, max, time, min, time]

    Notes
    -----
    Reads one of three formats::

        Row Description Maximum Time Case Minimum Time Case
        --- ----------- ------- ---- ---- ------- ---- ----

        Row Description Maximum Time Minimum Time
        --- ----------- ------- ---- ------- ----

        Row Description Maximum Minimum
        --- ----------- ------- -------

    """
    with open(name, 'r') as f:
        for line in f:
            if -1 != line.find(' ----'):
                break
        p = 0
        starts = [p]
        p = line.find(' -', p+1)
        while p != -1:
            p += 1
            starts.append(p)
            p = line.find(' -', p+1)

        starts.append(len(line))
        arr = []
        if len(starts) >= 8:
            for line in f:
                nums = [int(line[starts[0]:starts[1]]),
                        float(line[starts[2]:starts[3]]),
                        float(line[starts[3]:starts[4]]),
                        float(line[starts[5]:starts[6]]),
                        float(line[starts[6]:starts[7]])]
                arr.append(nums)
        elif len(starts) >= 6:
            for line in f:
                nums = [int(line[starts[0]:starts[1]]),
                        float(line[starts[2]:starts[3]]),
                        float(line[starts[3]:starts[4]]),
                        float(line[starts[4]:starts[5]]),
                        float(line[starts[5]:starts[6]])]
                arr.append(nums)
        else:
            for line in f:
                nums = [int(line[starts[0]:starts[1]]),
                        float(line[starts[2]:starts[3]]),
                        0.0,
                        float(line[starts[3]:starts[4]]),
                        0.0]
                arr.append(nums)
        return np.array(arr)


def PrintCLAInfo(mission, event):
    "PrintCLAInfo Print CLA event info, typically for the log file"
    print('Mission:  {}'.format(mission))
    print('Event:    {}'.format(event))


def freq3_augment(freq1, lam, tol=1.e-5):
    """
    Mimic Nastran's FREQ3 augmentation of a frequency vector.

    Parameters
    ----------
    freq1 : 1d array_like
        Initial frequencies (Hz)
    lam : 1d array_like
        System eigenvalues (rad/sec)^2
    tol : scalar; optional
        Tolerance used for deleting near-duplicate frequencies

    Returns
    -------
    freq : 1d ndarray
        The modified frequency vector

    Notes
    -----
    Only natural frequencies in the range of `freq1` will be added.

    Examples
    --------
    >>> from pyyeti import cla
    >>> freq1 = np.arange(5., 11.)
    >>> sysHz = np.array([3.3, 6.7, 8.9, 9.00001, 12.4])
    >>> lam = (2*np.pi*sysHz)**2
    >>> cla.freq3_augment(freq1, lam)
    array([  5. ,   6. ,   6.7,   7. ,   8. ,   8.9,   9. ,  10. ])
    """
    freq1, lam = np.atleast_1d(freq1, lam)
    sysfreqs = np.sqrt(abs(lam))/(2*np.pi)
    pv = np.nonzero(np.logical_and(sysfreqs > freq1[0],
                                   sysfreqs < freq1[-1]))[0]
    freq3 = sysfreqs[pv]
    freq = np.sort(np.hstack((freq1, freq3)))
    uniq = locate.find_unique(freq, tol*(freq[-1] - freq[0]))
    return freq[uniq]


def get_marker_cycle():
    """
    Return an ``itertools.cycle`` of plot markers.

    The list is taken from `matplotlib.markers`.
    """
    return itertools.cycle([
        'o',          # circle marker
        'v',          # triangle_down marker
        '^',          # triangle_up marker
        '<',          # triangle_left marker
        '>',          # triangle_right marker
        '1',          # tri_down marker
        '2',          # tri_up marker
        '3',          # tri_left marker
        '4',          # tri_right marker
        's',          # square marker
        'p',          # pentagon marker
        '*',          # star marker
        'h',          # hexagon1 marker
        'H',          # hexagon2 marker
        '+',          # plus marker
        'x',          # x marker
        'D',          # diamond marker
        'd',          # thin_diamond marker
    ])


def _ensure_iter(obj):
    try:
        iter(obj)
    except TypeError:
        obj = (obj,)
    return obj


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
    if 'eqsine' in opts:
        return opts['eqsine']
    return False


def maxmin(response, time):
    """
    Compute max & min of a response matrix.

    Parameters
    ----------
    response : 2d ndarray
        Matrix where each row is a response signal.
    time: 1d ndarray
        Time vector; ``len(time) = response.shape[1]``

    Returns
    -------
    mxmn : 2d ndarray
        Four column matrix: ``[max, time, min, time]``
    """
    r, c = np.shape(response)
    if c != len(time):
        raise ValueError('# of cols in `response` is not compatible '
                         'with time.')
    jx = np.nanargmax(response, axis=1)
    jn = np.nanargmin(response, axis=1)
    ind = np.arange(r)
    mx = response[ind, jx]
    mn = response[ind, jn]
    return SimpleNamespace(
        ext=np.column_stack((mx, mn)),
        exttime=np.column_stack((time[jx], time[jn])))
    # mxmn = np.vstack((mx, time[jx], mn, time[jn])).T
    #    cols = ['max', 'maxtime', 'min', 'mintime']
    #    return pd.DataFrame(mxmn, columns=cols)
    # return mxmn


def extrema(curext, mm, maxcase, mincase=None, casenum=None):
    """
    Update extrema values in 'curext'

    Parameters
    ----------
    curext : SimpleNamespace
        Has extrema information (members may be None on first call)::

            .ext     = 1 or 2 columns: [max, min]
            .exttime = 1 or 2 columns: [max_time, min_time]
            .maxcase = list of strings identifying maximum case
            .mincase = list of strings identifying minimum case

        Also has these members if casenum > 0::

            .mx      = [case1_max, case2_max, ...]
            .mn      = [case1_min, case2_min, ...]
            .maxtime = [case1_max_time, case2_max_time, ...]
            .mintime = [case1_min_time, case2_min_time, ...]

    mm : SimpleNamespace
        Has min/max information for a case (or over cases)::

            .ext     = 1 or 2 columns: [max, min]
            .exttime = 1 or 2 columns: [max_time, min_time]

    maxcase : string or list of strings
        String identifying new case, or list of strings analogous to
        `curext.maxcase` but pertaining to `mm` (handy if `mm` is
        from another extrema data set)
    mincase : string or list of strings or None; optional
        Similar to `maxcase` or None. If None, it is set to `maxcase`.
    casenum : integer or None; optional
        If integer, it is case number (starting at 0); `curext` will
        have the `.mx`, `.mn`, `.maxtime`, `.mintime` members and the
        `casenum` column of each of these will be set to the data from
        `mm`. Zeros will be used for time if `mm` is 1 or 2 columns.
        If None, `.mx`, `.mn`, `.maxtime`, `.mintime` are not updated
        (and need not be present).

    Returns
    -------
    None.

    Notes
    -----
    This routine updates the `curext` variable.
    """
    r, c = mm.ext.shape
    if c not in [1, 2]:
        raise ValueError('mm.ext has {} cols, but must have 1 or 2.'
                         .format(c))

    # expand current case information to full size if necessary
    if isinstance(maxcase, str):
        maxcase = r*[maxcase]
    else:
        maxcase = maxcase[:]

    if c == 1:
        if casenum is not None:  # record current results
            curext.mx[:, casenum] = mm.ext
            curext.mn[:, casenum] = mm.ext
            curext.maxtime = copy.copy(mm.exttime)  # okay if None
            curext.mintime = copy.copy(mm.exttime)

        if curext.ext is None:
            curext.ext = mm.ext.copy()
            curext.exttime = copy.copy(mm.exttime)
            curext.maxcase = maxcase
            curext.mincase = maxcase[:]
            return

        # keep sign but compare based on absolute
        j = abs(mm.ext) > abs(curext.ext)
        if j.any():
            curext.maxcase[j] = maxcase[j]
            curext.ext[j] = mm.ext[j]
        return

    if mincase is None:
        mincase = maxcase[:]
    elif isinstance(mincase, str):
        mincase = r*[mincase]
    else:
        mincase = mincase[:]

    if casenum is not None:  # record current results
        curext.mx[:, casenum] = mm.ext[:, 0]
        curext.mn[:, casenum] = mm.ext[:, 1]
        if mm.exttime is not None:
            curext.maxtime[:, casenum] = mm.exttime[:, 0]
            curext.mintime[:, casenum] = mm.exttime[:, 1]
        else:
            curext.maxtime[:, casenum] = np.nan
            curext.mintime[:, casenum] = np.nan

    if curext.ext is None:
        curext.ext = mm.ext.copy()
        curext.exttime = copy.copy(mm.exttime)
        curext.maxcase = maxcase
        curext.mincase = mincase
        return

    def put_time(curext, mm, j, col):
        if mm.exttime is not None:
            if curext.exttime is None:
                curext.exttime = copy.copy(mm.exttime)
            else:
                curext.exttime[j, col] = mm.exttime[j, col]
        elif curext.exttime is not None:
            curext.exttime[j, col] = np.nan

    j = (mm.ext[:, 0] > curext.ext[:, 0]).nonzero()[0]
    if j.size > 0:
        for i in j:
            curext.maxcase[i] = maxcase[i]
        curext.ext[j, 0] = mm.ext[j, 0]
        put_time(curext, mm, j, 0)

    j = (mm.ext[:, 1] < curext.ext[:, 1]).nonzero()[0]
    if j.size > 0:
        for i in j:
            curext.mincase[i] = mincase[i]
        curext.ext[j, 1] = mm.ext[j, 1]
        put_time(curext, mm, j, 1)


class DR_Def(object):
    """
    Data recovery definitions.

    Parameters
    ----------
    defaults : dict or None; optional
        Sets the `defaults` attribute; see below.

    Attributes
    ----------
    defaults : dict or None; optional
        Dictionary with any desired defaults for the parameters
        listed in :func:`add`. If None, it is initialized to an
        empty dictionary.
    dr_def : dict
        This is a dictionary that defines how data recovery will be
        done. This is created through calls to member function
        :func:`DR_Def.add` (typically from a "prepare_4_cla.py"
        script). See the notes section below for an example showing
        what is in this dict.
    ncats : integer
        The number of data recovery categories defined.

    Notes
    -----
    Rather than try to describe the `dr_def` dictionary, we'll use
    :func:`pyyeti.pp.PP` to display sections of it for an example
    mission:

    PP(claparams['drdefs'].dr_def)::

        <class 'dict'>[n=9]
            'SC_atm'    : <class 'types.SimpleNamespace'>[n=19]
            'SC_dtm'    : <class 'types.SimpleNamespace'>[n=19]
            'SC_ifl'    : <class 'types.SimpleNamespace'>[n=19]
            'SC_ltma'   : <class 'types.SimpleNamespace'>[n=19]
            'SC_ltmd'   : <class 'types.SimpleNamespace'>[n=19]
            'SC_cg'     : <class 'types.SimpleNamespace'>[n=19]
            'SC_ifa'    : <class 'types.SimpleNamespace'>[n=19]
            'SC_ifa_0rb': <class 'types.SimpleNamespace'>[n=19]
            '_vars'     : <class 'types.SimpleNamespace'>[n=2]

    PP(claparams['drdefs'].dr_def['SC_ifa'])::

        <class 'types.SimpleNamespace'>[n=20]
            .desc      : 'S/C Interface Accelerations'
            .drfile    : '/loads/CLA/Rocket/missions/.../drfuncs.py'
            .drfunc    : 'SC_ifa'
            .filterval : 1e-06
            .histlabels: [n=12]: ['I/F Axial Accel     X sc', ... lv']
            .histpv    : slice(None, 12, None)
            .histunits : 'G, rad/sec^2'
            .labels    : [n=12]: ['I/F Axial Accel     X sc', ... lv']
            .misc      : None
            .napv      : None
            .se        : 500
            .srsQs     : [n=2]: (25, 50)
            .srsconv   : 1
            .srsfrq    : float64 ndarray 990 elems: (990,)
            .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', '$Z_ ...}$']
            .srsopts   : <class 'dict'>[n=2]
                'eqsine': 1
                'ic'    : 'steady'
            .srspv     : slice(None, 12, None)
            .srsunits  : 'G, rad/sec^2'
            .uf_reds   : [n=4]: (1, 1, 1.25, 1)
            .units     : 'G, rad/sec^2'

    PP(claparams['drdefs'].dr_def['_vars'], 3)

        <class 'types.SimpleNamespace'>[n=2]
            .drms   : <class 'dict'>[n=1]
                500: <class 'dict'>[n=7]
                    'SC_ifl_drm': float64 ndarray 1296 ... (12, 108)
                    'SC_cg_drm' : float64 ndarray 1512 ... (14, 108)
                    'SC_ifa_drm': float64 ndarray 1296 ... (12, 108)
                    'scatm'     : float64 ndarray 28944 ... (268, 108)
                    'scdtmd'    : float64 ndarray 9396 ... (87, 108)
                    'scltma'    : float64 ndarray 15768 ... (146, 108)
                    'scltmd'    : float64 ndarray 15768 ... (146, 108)
            .nondrms: <class 'dict'>[n=1]
                500: <class 'dict'>[n=0]

    """
    # static variable:
    ncats = 0

    @staticmethod
    def addcat(func):
        """
        Decorator to ensure function is called to add category.

        Notes
        -----
        Example of typical usage::

            @cla.DR_Def.addcat
            def _():
                name = 'SC_atm'
                desc = 'S/C Internal Accelerations'
                units = 'G'
                # ... other variables defined; see :func:`add`.
                drdefs.add(**locals())

        """
        nc = DR_Def.ncats
        func()
        if DR_Def.ncats <= nc:
            msg = ('function must call the `DR_Def.add` '
                   'method (eg: ``drdefs.add(**locals())``)')
            raise RuntimeError(msg)

    def __init__(self, defaults=None):
        self.defaults = {} if defaults is None else defaults
        self.dr_def = {}
        self.dr_def['_vars'] = SimpleNamespace(drms={}, nondrms={})
        self._drfilemap = {}
        return

    # add drms and nondrms to self.dr_def:
    def _add_vars(self, name, drms, nondrms):
        def _add_drms(d1, d2):
            for key in d2:
                if key in d1:
                    # print warning or error out:
                    msg = ('"{}" already included for a previously '
                           'defined data recovery matrix'
                           .format(key))
                    if d1[key] is d2[key]:
                        warn(msg, RuntimeWarning)
                    else:
                        raise ValueError('A different '+msg)
                else:
                    d1[key] = d2[key]

        _vars = self.dr_def['_vars']
        se = self.dr_def[name].se
        for curdrms, newdrms in zip((_vars.drms, _vars.nondrms),
                                    (drms, nondrms)):
            if se not in curdrms:
                curdrms[se] = {}
            _add_drms(curdrms[se], newdrms)

    def _check_for_drfunc(self, filename, funcname):
        modpath, modfile = os.path.split(filename)
        if modfile.endswith('.py'):
            modfile = modfile[:-3]
        try:
            sys.path.insert(0, modpath)
            drmod = importlib.import_module(modfile)
            if funcname not in dir(drmod):
                msg = ('"{}" not found in: {}.'
                       .format(funcname, filename))
                warn(msg, RuntimeWarning)
        finally:
            sys.path.pop(0)

    def _handle_defaults(self, ns):
        """Handle default values and take default actions"""
        # first, the defaults:
        if ns.drfile is None:
            ns.drfile = self.defaults

        if ns.se is None:
            ns.se = self.defaults

        if ns.uf_reds is None:
            ns.uf_reds = self.defaults

        if ns.srsQs is None:
            for k, v in ns.__dict__.items():
                if k.startswith('srs') and v is not None:
                    ns.srsQs = self.defaults
                    break

        if ns.srsfrq is None and ns.srsQs is not None:
            ns.srsfrq = self.defaults

        dct = ns.__dict__
        for key in dct:
            if dct[key] is self.defaults:
                try:
                    dct[key] = self.defaults[key]
                except KeyError:
                    print('{} set to `defaults` but is not found in '
                          '`defaults`!'.format(key))
                    raise

        # add path to `drfile` if needed:
        try:
            ns.drfile = self._drfilemap[ns.drfile]
        except KeyError:
            drfile = ns.drfile
            ns.drfile = os.path.realpath(drfile)
            self._drfilemap[drfile] = ns.drfile
        self._check_for_drfunc(ns.drfile, ns.drfunc)

        # next, the default actions:
        if ns.desc is None:
            ns.desc = ns.name

        if isinstance(ns.labels, numbers.Integral):
            ns.labels = ['Row {:6d}'.format(i+1)
                         for i in range(ns.labels)]

        # check filter value:
        try:
            nf = len(ns.filterval)
        except TypeError:
            pass
        else:
            if nf != len(ns.labels):
                raise ValueError('length of `filterval` ({}) does '
                                 'not match length of labels ({})'
                                 .format(nf, len(ns.labels)))
            ns.filterval = np.atleast_1d(ns.filterval)

        def _get_pv(pv, name, length):
            if pv is None:
                return pv
            if isinstance(pv, str) and pv == 'all':
                return slice(length)
            if isinstance(pv, slice):
                return pv
            if (not isinstance(pv, str) and
                    isinstance(pv, abc.Sequence)):
                pv = np.atleast_1d(pv)
            if isinstance(pv, np.ndarray) and pv.dtype == bool:
                if len(pv) != length:
                    raise ValueError('length of `{}` ({}) does not '
                                     'match length of labels ({})'
                                     .format(name, len(pv), length))
                return pv.nonzero()[0]
            raise TypeError('`{}` input not understood'.format(name))

        ns.napv = _get_pv(ns.napv, 'napv', len(ns.labels))

        if ns.srsconv is None:
            ns.srsconv = 1

        if ns.srsQs is not None:
            ns.srsQs = _ensure_iter(ns.srsQs)
            if ns.srspv is None:
                ns.srspv = slice(len(ns.labels))
                # ns.srspv = np.ones(len(ns.labels), bool)

        if ns.srspv is not None and ns.srsopts is None:
            ns.srsopts = {}

        def _get_labels(pv, labels):
            if isinstance(pv, slice):
                return labels[pv]
            return [labels[i] for i in pv]

        # fill in hist-labels/units and srs-labels/units if needed:
        for i in ('hist', 'srs'):
            pv = i + 'pv'
            dct[pv] = _get_pv(dct[pv], pv, len(ns.labels))
            if dct[pv] is not None:
                lbl = i + 'labels'
                unt = i + 'units'
                if dct[lbl] is None:
                    dct[lbl] = _get_labels(dct[pv], ns.labels)
                if dct[unt] is None:
                    dct[unt] = ns.units

    def add(self, *, name, labels, drms=None, drfile=None, se=None,
            desc=None, units='Not specified', uf_reds=None,
            filterval=1.e-6, histlabels=None, histpv=None,
            histunits=None, misc=None, napv=None, nondrms=None,
            srsQs=None, srsfrq=None, srsconv=None, srslabels=None,
            srsopts=None, srspv=None, srsunits=None, **kwargs):
        """
        Adds a data recovery category.

        All inputs to :func:`add` must be named.

        .. note::

            Any of the inputs can be set to `self.defaults`. In this
            case, the value for the parameter will be taken from the
            `self.defaults` dictionary (which is defined during
            instantiation). An error is raised if `self.defaults` does
            not contain a value for the parameter.

        .. note::

            Using None for some inputs (which is default for many)
            will cause a default action to be taken. These are listed
            below as applicable after "DA:".

        Parameters
        ----------
        name : string
            Short name of data recovery category, eg: 'SC_atm'. This
            is also the name of the function in `drfile` that is
            called to do data recovery and so it must be a valid
            Python variable name.
        labels : string or integer
            List of strings, describing each row. Can also be an
            integer specifying number of rows being recovered; in this
            case, the list is ``['Row 1', 'Row 2', ...]``. This input
            is used to determine number of rows being recovered.
        drms : dict or None
            Dictionary of data recovery matrices for this category;
            keys are matrix names and must match what is used in the
            data recovery function in `drfile`. For example: ``drms =
            {'scatm': scatm}``. If `se` is greater than 0, each matrix
            will be multiplied by the appropriate "ULVS" matrix (see
            :func:`pyyeti.op2.rdnas2cam`) during event simulation. If
            `se` is 0, it is used as is and is likely added during
            event simulation since system modes are often
            needed. (Also, when `se` is 0, using `drms` is equivalent
            to using `nondrms`.)
        drfile : string or None; optional
            Name of file that contains the data recovery function
            named `name`. It can optionally also have the PSD data
            recovery function named `name`_psd. See notes below for
            more info. This file is imported during event simulation.
            DA: get value from `self.defaults`.
        se : integer or None; optional
            The superelement number.
            DA: get value from `self.defaults`.
        desc : string or None; optional
            One line description of category; eg:
            ``desc = 'S/C Internal Accelerations'``.
            DA: set to `name`.
        units : string; optional
            Specifies the units.
        uf_reds : 4-element tuple or None; optional
            The uncertainty factors in "reds" order:
            [rigid, elastic, dynamic, static]. Examples::

                (1, 1, 1.25, 1)  - Typical; DUF = 1.25
                (0, 1, 1.25, 1)  - Same, but without rigid-body part
                (1, 1, 0, 1)     - Recover static part only
                (1, 1, 1, 0)     - Recover dynamic part only

            DA: get value from `self.defaults`.

        filterval : scalar or 1d array_like; optional
            Response values smaller than `filterval` will be skipped
            during comparison to another set of results. If 1d
            array_like, length must be ``len(labels)`` allowing for a
            unique filter value for each row.
        histlabels : list or None
            Analogous to `labels` but just for the `histpv` rows.
            DA: derive from `labels` according to `histpv` if
            needed.
        histpv : 1d bool array_like or 'all' or slice or None
            Boolean vector specifying which rows to save the response
            histories of. If 'all', it is reset internally to
            ``slice(len(labels))``. It can be entered as a slice
            directly. Set to None to save no histories. The stored
            version is either a slice or an index vector (uses
            0-offset for standard Python indexing).
        histunits : string or None
            Units string for the `histpv`.
            DA: set to `units`.
        misc : any object
            Available for storing miscellaneous information for the
            category. It is not used within this module.
        napv : 1d bool array_like or 'all' or slice or None
            "Not applicable" boolean partition vector specifying which
            rows that should be skipped in a comparison to another set
            of results. If 'all', it is reset internally to
            ``slice(len(labels))``. It can also be entered as a slice
            directly. If None, all rows are considered applicable. The
            stored version is either a slice or an index vector (uses
            0-offset for standard Python indexing).
        nondrms : dict or None
            With one important exception, this input is used
            identically to `drms`. The exception is that the values in
            `nondrms` are not multiplied by ULVS. Therefore, `nondrms`
            can contain any variables you need for data recovery. An
            alternative option is to include data you need in
            `drfile` with the data recovery functions.
        srsQs : scalar or 1d array_like or None
            Q values for SRS calculation. If None, no SRS's will be
            calculated.
            DA: set to `self.defaults` if any other `srs*` option is
            not None; otherwise, leave it None.
        srsfrq : 1d array_like or None
            Frequency vector for SRS.
            DA: get value from `self.defaults` if `srsQs` is not None.
        srsconv : scalar or 1d array_like or None
            Conversion factor scalar or vector same length as
            `srspv`. None is the same as 1.0.
        srslabels : list or None
            Analogous to `labels` but just for the `srspv` rows.
            DA: derive from `labels` according to `srspv` if needed.
        srsopts : dict or None
            Dictionary of options for SRS; eg:
            ``dict(eqsine=True, ic='steady')``
            DA: set to ``{}`` if `srspv` is not None.
        srspv : 1d bool array_like or 'all' or slice or None
            Boolean vector specifying which rows to compute SRS
            for. If 'all', it is reset internally to
            ``slice(len(labels))``. It can be entered as a slice
            directly. If None, the default action is taken. The stored
            version is either a slice or an index vector (uses
            0-offset for standard Python indexing).
            DA: if `srsQs` is not None, internally set to
            ``np.arange(len(labels))``; otherwise, leave it None.
        srsunits : string or None
            Units string for the `srspv`.
            DA: set to `units`.
        **kwargs : dict
            All other inputs are quietly ignored.

        Returns
        -------
        None. Updates the attribute `dr_def` (see :class:`DR_Def`).

        Notes
        -----
        The `drfile` must contain the appropriate data recovery
        function(s) named ``name`` and, optionally, ``name_psd``. For
        a typical data recovery category, only one data recovery
        function would be needed. Here are some examples:

        For a typical ATM::

            def SC_atm(sol, nrb, Vars, se):
                return Vars[se]['atm'] @ sol.a

        For a typical mode-displacement DTM::

            def SC_dtm(sol, nrb, Vars, se):
                return Vars[se]['dtm'] @ sol.d

        For a typical mode-acceleration LTM::

            def SC_ltm(sol, nrb, Vars, se):
                return (Vars[se]['ltma'] @ sol.a +
                        Vars[se]['ltmd'] @ sol.d)

        For a more complicated data recovery category, you might need
        to include a special PSD version also. This function has the
        same name except with "_psd" tacked on. For example, to
        compute a time-consistent (or phase-consistent)
        root-sum-square (RSS), you need to provide both
        functions. Here is a pair of functions that recover 3 rows
        (for `name` "x_y_rss"): acceleration response in the 'x' and
        'y' directions and the consistent RSS between them for the 3rd
        row. In this example, it is assumed the data recovery matrix
        has 3 rows where the 3rd row could be all zeros::

            def x_y_rss(sol, nrb, Vars, se):
                resp = Vars[se]['xyrss'] @ sol.a
                xr = 0             # 'x' row(s)
                yr = 1             # 'y' row(s)
                rr = 2             # rss  rows
                resp[rr] = np.sqrt(resp[xr]**2 + resp[yr]**2)

            def x_y_rss_psd(sol, nrb, Vars, se, freq, forcepsd,
                            drmres, case, i):
                # drmres is a results namespace, eg:
                #   drmres = results['x_y_rss']
                # i is psd force number
                resp = Vars[se]['xyrss'] @ sol.a
                xr = 0             # 'x' row(s)
                yr = 1             # 'y' row(s)
                rr = 2             # rss  rows
                cla.PSD_consistent_rss(
                    resp, xr, yr, rr, freq, forcepsd,
                    drmres, case, i)

        Ultimately, the PSD recovery function must provide the final
        PSD solution in ``drmres[case]`` after the final force is
        analyzed (when ``i == forcepsd.shape[0]-1``). In the example
        above, :func:`PSD_consistent_rss` takes care of that job.

        Recovery function inputs:

            =====  ===================================================
            Input  Description
            =====  ===================================================
            sol    ODE modal solution namespace with uncertainty
                   factors applied. Typically has at least .a, .v and
                   .d members (modal accelerations, velocities and
                   displacements). See
                   :func:`pyyeti.ode.SolveUnc.tsolve`.
            nrb    Number of rigid-body modes
            DR     Defines data recovery for an event simulation (and
                   is created in the simulation script via
                   ``DR = cla.Event_DR()``). It is an event specific
                   version of all combined :class:`DR_Def` objects
                   with all ULVS matrices applied.
            se     Superelement number. Used as key into `DR`.
            etc    See :func:`PSD_consistent_rss` for description of
                   `freq`, `forcepsd`, etc.
            =====  ===================================================
        """
        # get dictionary of inputs and trim out specially handled
        # entries:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        dr_def_args = {i: values[i] for i in args
                       if i not in ('self', 'name', 'drms',
                                    'nondrms', 'kwargs')}
        #dr_def_args = locals()
        #for item in ('self', 'name', 'drms', 'nondrms', 'kwargs'):
        #    del dr_def_args[item]
        #
        #print(dr_def_args.keys())
        #
        if name in self.dr_def:
            raise ValueError('data recovery for "{}" already defined'
                             .format(name))

        if drms is None:
            drms = {}

        if nondrms is None:
            nondrms = {}

        # check for overlapping keys in drms and nondrms:
        overlapping_names = set(nondrms.keys()) & set(drms.keys())
        if len(overlapping_names) > 0:
            raise ValueError('`drms` and `nondrms` have overlapping '
                             'names: {}'.format(overlapping_names))

        # define dr_def[name] entry:
        dr_def_args['drfunc'] = name
        self.dr_def[name] = SimpleNamespace(**dr_def_args)

        # hand defaults and default actions:
        self._handle_defaults(self.dr_def[name])

        # add drms and nondrms to self.dr_def:
        self._add_vars(name, drms, nondrms)

        # increment static variable ncats for error checking:
        DR_Def.ncats += 1

    def copycat(self, categories, name_addon, **kwargs):
        """
        Copy a category with optional modifications.

        Parameters
        ----------
        categories : string or list
            Name or names of category(s) to copy.
        name_addon : string or list
            If string, it will be appended to all category names to
            make the new names. If list, contains the new names with
            no regard to old names.
        **kwargs :
            Any options to modify. See :func:`DR_Def.add` for
            complete list (except `drfunc` is also available ... that
            is the name of the data recovery function inside
            `drfile`).

        Returns
        -------
        None. Updates the member `dr_def`.

        Notes
        -----
        The data recovery categories are copies of the originals with
        the `name` changed (has `name_addon` appended) and new values
        set according to `**kwargs`.

        One very common usage is to add a zero-rigid-body version of a
        category; for example::

            drdefs.copycat(['SC_ifa', 'SC_atm'], '_0rb',
                             uf_reds=(0, 1, 1.25, 1))

        would add 'SC_ifa_0rb' and 'SC_atm_0rb' copy categories
        without the rigid-body component.

        For another example, recover the 'SC_cg' (cg load factors)
        in "static" and "dynamic" pieces::

            drdefs.copycat('SC_cg', '_static',
                             uf_reds=(1, 1, 0, 1))
            drdefs.copycat('SC_cg', '_dynamic',
                             uf_reds=(1, 1, 1.25, 0))

        As a final example to show the alternate use of `name_addon`,
        here is an equivalent call for the static example::

            drdefs.copycat('SC_cg', ['SC_cg_static'],
                             uf_reds=(1, 1, 0, 1))

        Raises
        ------
        ValueError
            When the new category name already exists.
        """
        if isinstance(categories, str):
            categories = [categories]

        for name in categories:
            if name not in self.dr_def:
                raise ValueError('{} not found in `dr_def`'
                                 .format(name))

        for key in kwargs:
            if key not in self.dr_def[categories[0]].__dict__:
                raise ValueError('{} not found in `dr_def["{}"]`'
                                 .format(key, categories[0]))

        for i, name in enumerate(categories):
            if isinstance(name_addon, str):
                new_name = name + name_addon
            else:
                new_name = name_addon[i]
            if new_name in self.dr_def:
                raise ValueError('"{}" category already defined'
                                 .format(new_name))
            self.dr_def[new_name] = copy.copy(self.dr_def[name])
            cat = self.dr_def[new_name]
            for key, value in kwargs.items():
                cat.__dict__[key] = value

    def add_0rb(self, *args):
        """
        Add zero-rigid-body versions of selected categories.

        Parameters
        ----------
        *args : strings
            Category names for which to make '_0rb' versions.

        Notes
        -----
        This is a convenience function that uses :func:`copycat` to do
        the work::

            copycat(args, '_0rb', uf_reds=(0, e, d, s),
                    desc=desc+' w/o RB')

        where `e`, `d`, `s` and `desc` are the current values. See
        :func:`copycat` for more information.

        For example::

            drdefs.add_0rb('SC_ifa', 'SC_atm')

        would add 'SC_ifa_0rb' and 'SC_atm_0rb' categories with the
        first element in `uf_reds` set to 0.
        """
        if args[0] not in self.dr_def:
            raise ValueError('{} not found in `dr_def`'
                             .format(args[0]))
        r, e, d, s = self.dr_def[args[0]].uf_reds
        new_uf_reds = 0, e, d, s
        desc = self.dr_def[args[0]].desc + ' w/o RB'
        self.copycat(args, '_0rb', uf_reds=new_uf_reds, desc=desc)

    def excel_summary(self, excel_file='dr_summary.xlsx'):
        """
        Make excel file with summary of data recovery information.

        Parameters
        ----------
        excel_file : string or None; optional
            Name of excel file to create; if None, no file will be
            created.

        Returns
        -------
        drinfo : panda DataFrame
            Summary table of data recovery information. The index
            values are those set via :func:`DR_Def.add` (name, desc,
            etc). The columns are the categores (eg: 'SC_atm',
            'SC_ltm', etc).
        """
        cats = sorted([i for i in self.dr_def.keys()
                       if not i.startswith('_')])
        if len(cats) == 0:
            raise RuntimeError('add data recovery categories first')
        vals = sorted(self.dr_def[cats[0]].__dict__.keys())
        df = pd.DataFrame(index=vals, columns=cats)

        def _issame(old, new):
            if new is old:
                return True
            if type(new) != type(old):
                return False
            if isinstance(new, np.ndarray):
                if new.shape != old.shape:
                    return False
                return (new == old).all()
            return new == old

        fill_char = '-'
        # fill in DataFrame, use `fill_char` for "same as previous"
        for i, cat in enumerate(cats):
            for val in vals:
                new = self.dr_def[cat].__dict__[val]
                s = None
                if i > 0:
                    old = self.dr_def[cats[i-1]].__dict__[val]
                    if _issame(old, new):
                        self.dr_def[cat].__dict__[val] = old
                        s = fill_char
                if s is None:
                    s = str(new)
                    if len(s) > 30:
                        s = s[:10] + ' ... ' + s[-10:]
                    if not isinstance(new, str):
                        try:
                            slen = len(new)
                        except TypeError:
                            pass
                        else:
                            s = '{:d}: {:s}'.format(slen, s)
                df[cat].loc[val] = s

        if excel_file is not None:
            with pd.ExcelWriter(excel_file,
                                engine='xlsxwriter') as writer:
                shname = 'DR_Def'
                df.to_excel(writer, sheet_name=shname)
                workbook = writer.book
                worksheet = writer.sheets[shname]
                worksheet.set_column('A:A', 10)
                worksheet.set_column(1, len(cats), 25)
                frm = workbook.add_format({'border': 1})
                worksheet.conditional_format(
                    0, 0, df.shape[0], df.shape[1],
                    {'type': 'no_errors', 'format': frm})
                bold = workbook.add_format({'bold': True})
                worksheet.write(df.shape[0]+2, 1, 'Notes:', bold)
                tab = '    '
                msg = fill_char+' = same as previous category'
                worksheet.write(df.shape[0]+3, 1, tab+msg)
                msg = ('The partition vector variables (*pv) '
                       'use 0-offset (or are slices)')
                worksheet.write(df.shape[0]+4, 1, tab+msg)
        return df


# call this DR_Modes?
class Event_DR(object):
    """
    Setup data recovery for a specific event or set of modes.

    Attributes
    ----------
    Info : dict
        Contains data recovery information for each category. The
        category names are the keys. This is a copy of the information
        created during data recovery setup; eg, in a
        "prepare_4_cla.py" script. See description of `dr_def` in
        :class:`DR_Def`.
    UF_reds : list
        List of all unique 4-element uncertainty factor tuples. See
        :func:`DR_Def.add` for more information.
    Vars : dict
        Contains the data recovery matrices and possibly other data
        needed for data recovery. This is derived from the
        ``dr_def['_vars']`` dict and the current system modes. See the
        notes section below for an example showing what is in this
        dict.

    Notes
    -----
    Here is an example `Vars`:

    PP(DR.Vars)::

        <class 'dict'>[n=2]
            0  : <class 'dict'>[n=1]
                'Tnode4': float64 ndarray 732 elems: (3, 244)
            100: <class 'dict'>[n=3]
                'ifatm' : float64 ndarray 46896 elems: (48, 977)
                'ifltma': float64 ndarray 17586 elems: (18, 977)
                'ifltmd': float64 ndarray 17586 elems: (18, 977)
            500: <class 'dict'>[n=7]
                'SC_ifl_drm': float64 ndarray 11724 elems: (12, 977)
                'SC_cg_drm' : float64 ndarray 13678 elems: (14, 977)
                'SC_ifa_drm': float64 ndarray 11724 elems: (12, 977)
                'scatm'     : float64 ndarray 261836 elems: (268, 977)
                'scdtmd'    : float64 ndarray 84999 elems: (87, 977)
                'scltma'    : float64 ndarray 142642 elems: (146, 977)
                'scltmd'    : float64 ndarray 142642 elems: (146, 977)

    In that example, all the variables except 'Tnode4' are
    multiplied by the appropriate ULVS matrix. That is, the SE 100 and
    500 matrices all came from the ``dr_def['_vars'].drms`` entry and
    none came from the ``dr_def['_vars'].nondrms`` entry. The SE 0
    matrix could come from either the `.drms` or `.nondrms` entry.
    """

    def __init__(self):
        self.Info = {}
        self.UF_reds = []
        self.Vars = {}

    def add(self, nas, drdefs, uf_reds=None):
        """
        Add data recovery definitions for an event or set of modes.

        Parameters
        ----------
        nas : dictionary
            This is the nas2cam dictionary:
            ``nas = pyyeti.op2.rdnas2cam()``
        drdefs : DR_Def instance or None
            Contains the data recovery definitions for, typically, one
            superelement. See :class:`DR_Def`. If None, this routine
            does nothing.
        uf_reds : 4-element tuple or None; optional
            If not None, this is the uncertainty factors in "reds"
            order: [rigid, elastic, dynamic, static]. In that case,
            this entry overrides any `uf_reds` settings already
            defined in `drdefs`. Set any of the four value to None to
            keep the original value (often used for "rigid" since that
            is typically either 0 or 1). This `uf_reds` option can be
            useful when uncertainty factors are event specific rather
            than data recovery category specific.

            For example, to reset the dynamic uncertainty factor to
            1.1 while leaving the other values unchanged::

                uf_reds=(None, None, 1.1, None)

        Notes
        -----
        Typically called once for each superelement where data
        recovery is requested. The attributes `Info`, `UF_reds` and
        `Vars` are all updated on each call.
        """
        if drdefs is None:
            return
        dr_def = drdefs.dr_def
        for name in dr_def:
            if name == '_vars':
                continue
            if name in self.Info:
                raise ValueError('"{}" data recovery category already'
                                 ' defined'.format(name))

            # variables for how to do data recovery:
            self.Info[name] = copy.copy(dr_def[name])

            # reset uf_reds if input:
            if uf_reds is not None:
                old_uf_reds = self.Info[name].uf_reds
                new_uf_reds = (i if i is not None else j
                               for i, j in zip(uf_reds, old_uf_reds))
                self.Info[name].uf_reds = tuple(new_uf_reds)

            # collect all sets of uncertainty factors together for the
            # apply_uf routine later:
            uf_reds = self.Info[name].uf_reds
            if uf_reds not in self.UF_reds:
                self.UF_reds.append(uf_reds)

        se_last = -2
        # apply ULVS to all drms and put in DR:
        for se, dct in dr_def['_vars'].drms.items():
            if se not in self.Vars:
                self.Vars[se] = {}
            if se != 0 and se != se_last:
                ulvs = nas['ulvs'][se]
                uset = nas['uset'][se]
                # Want bset partition from aset.  But note that in the
                # .asm, .pch approach to SE's, it is valid in Nastran
                # to just put all b-set & q-set in a generic a-set.
                # If that's the case, find q-set by finding the
                # spoints:
                bset = n2p.mksetpv(uset, 'a', 'b')   # bool type
                if bset.all():
                    # manually check for q-set in the a-set:
                    aset = np.nonzero(n2p.mksetpv(uset, 'p', 'a'))[0]
                    qset = uset[aset, 1] == 0   # spoints
                    bset = ~qset
                bset = np.nonzero(bset)[0]
                Lb = len(bset)
            se_last = se

            for name, mat in dct.items():
                if name in self.Vars[se]:
                    raise ValueError('"{}" is already in Vars[{}]'
                                     .format(name, se))
                if se == 0:
                    self.Vars[se][name] = mat
                elif mat.shape[1] > Lb:
                    self.Vars[se][name] = mat @ ulvs
                else:
                    self.Vars[se][name] = mat @ ulvs[bset]

        # put all nondrms in DR:
        for se, dct in dr_def['_vars'].nondrms.items():
            if se not in self.Vars:
                self.Vars[se] = {}
            for name, mat in dct.items():
                if name in self.Vars[se]:
                    raise ValueError('"{}" is already in Vars[{}]'
                                     .format(name, se))
                    self.Vars[se][name] = mat

    def prepare_results(self, mission, event):
        """
        Returns an instance of the class:`DR_Results` class.

        Parameters
        ----------
        mission : str
            Identifies the CLA
        event : str
            Name of event
        """
        results = DR_Results()
        results.init(self.Info, mission, event)
        return results

    def apply_uf(self, sol, m, b, k, nrb, rfmodes):
        """
        Applies the uncertainty factors to the solution

        Parameters
        ----------
        sol : SimpleNamespace
            Solution, input only; expected to have::

                .a = modal acceleration time-history matrix
                .v = modal velocity time-history matrix
                .d = modal displacement time-history matrix
                .pg = g-set forces; optional

        m : 1d or 2d ndarray or None
            Modal mass; can be vector or matrix or None (for identity)
        b : 1d or 2d ndarray
            Modal damping; vector or matrix
        k : 1d or 2d ndarray
            Modal stiffness; vector or matrix
        nrb : scalar
            Number of rigid-body modes
        rfmodes : index vector or None
            Specifies where the res-flex modes are; if None, no
            resflex

        Returns
        -------
        solout : dict
            Dictionary of solution namespaces with scaled versions
            of `.a`, `.v`, `.d` and `.pg`. The keys are all the
            "uf_reds" values. Additionally, the displacement member is
            separated into static and dynamic parts: `.d_static`,
            `.d_dynamic`. On output, ``.d = .d_static + d_dynamic``.
            For example, if one of the "uf_reds" tuples is:
            ``(1, 1, 1.25, 1)``, then::

                solout[(1, 1, 1.25, 1)].a
                solout[(1, 1, 1.25, 1)].v
                solout[(1, 1, 1.25, 1)].d
                solout[(1, 1, 1.25, 1)].d_static
                solout[(1, 1, 1.25, 1)].d_dynamic
                solout[(1, 1, 1.25, 1)].pg (optional)

        Notes
        -----
        Uncertainty factors are applied as follows (rb=rigid-body,
        el=elastic, rf=residual-flexibility)::

           ruf = rb uncertainty factor
           euf = el uncertainty factor
           duf = dynamic uncertainty factor
           suf = static uncertainty factor

           .a_rb and .v_rb - scaled by ruf*suf
           .a_el and .v_el - scaled by euf*duf
           .a_rf and .v_rf - zeroed out

           .d_rb - zeroed out
           .d_el - static part:  scaled by euf*suf
                 - dynamic part: scaled by euf*duf
           .d_rf - scaled by euf*suf

           .pg   - scaled by suf

        Note that d_el is written out as::

              d_el = euf*inv(k_el)*(suf*F_el - duf*(a_el+b_el*v_el))

        where::

              F = m*a + b*v + k*d

        """
        n = k.shape[0]
        use_velo = 1

        # genforce = m*a+b*v+k*d
        if m is None:
            genforce = sol.a.copy()
        elif m.ndim == 1:
            genforce = m[:, None] * sol.a
        else:
            genforce = m @ sol.a

        if b.ndim == 1:
            genforce += b[:, None] * sol.v
        else:
            genforce += b @ sol.v

        if k.ndim == 1:
            genforce += k[:, None] * sol.d
        else:
            genforce += k @ sol.d

        solout = {}
        for item in self.UF_reds:
            ruf, euf, duf, suf = item
            solout[item] = copy.deepcopy(sol)
            SOL = solout[item]
            SOL.d_static = np.zeros_like(sol.d)
            SOL.d_dynamic = np.zeros_like(sol.d)

            # apply ufs:
            if nrb > 0:
                SOL.a[:nrb] *= (ruf*suf)
                SOL.v[:nrb] *= (ruf*suf)

            if rfmodes is not None:
                SOL.a[rfmodes] = 0
                SOL.v[rfmodes] = 0

            if 'pg' in SOL.__dict__:
                SOL.pg *= suf

            if nrb < n:
                SOL.a[nrb:] *= (euf*duf)
                SOL.v[nrb:] *= (euf*duf)

                if m is None:
                    avterm = SOL.a[nrb:].copy()
                elif m.ndim == 1:
                    avterm = m[nrb:, None] * SOL.a[nrb:]
                else:
                    avterm = m[nrb:, nrb:] @ SOL.a[nrb:]

                if not use_velo:
                    msg = ('not including velocity term in mode-'
                           'acceleration formulation for displacements.')
                    warn(msg, RuntimeWarning)
                else:
                    if b.ndim == 1:
                        avterm += b[nrb:, None] * SOL.v[nrb:]
                    else:
                        avterm += b[nrb:, nrb:] @ SOL.v[nrb:]

                # in case there is mass coupling between rfmodes and
                # other modes
                if rfmodes is not None:
                    avterm[rfmodes-nrb] = 0

                gf = (euf*suf)*genforce[nrb:]
                if k.ndim == 1:
                    invk = (1/k[nrb:])[:, None]
                    SOL.d_static[nrb:] = invk * gf
                    SOL.d_dynamic[nrb:, :] = -invk * avterm
                else:
                    lup = la.lu_factor(k[nrb:, nrb:])
                    SOL.d_static[nrb:] = la.lu_solve(lup, gf)
                    SOL.d_dynamic[nrb:, :] = la.lu_solve(lup, -avterm)

                SOL.d = SOL.d_static + SOL.d_dynamic
        return solout

    def psd_apply_uf(self, sol, nrb):
        """
        Applies the uncertainty factors to the PSD solution

        Parameters
        ----------
        sol : SimpleNamespace
            Solution, input only; expected to have::

                .a = modal acceleration PSD matrix
                .v = modal velocity PSD matrix
                .d = modal displacement PSD matrix
                .pg = g-set forces; optional

        nrb : scalar
            Number of rigid-body modes

        Returns
        -------
        solout : dict
            Dictionary of solution namespaces with scaled versions
            of `.a`, `.v`, `.d` and `.pg`. The keys are all the
            "uf_reds" values. For example, if one of the "uf_reds"
            tuples is: ``(1, 1, 1.25, 1)``, then::

                solout[(1, 1, 1.25, 1)].a
                solout[(1, 1, 1.25, 1)].v
                solout[(1, 1, 1.25, 1)].d
                solout[(1, 1, 1.25, 1)].pg (optional)

        Notes
        -----
        Uncertainty factors are applied as follows (rb=rigid-body,
        el=elastic, rf=residual-flexibility)::

           ruf = rb uncertainty factor
           euf = el uncertainty factor
           duf = dynamic uncertainty factor
           suf = static uncertainty factor

           .a_rb, .v_rb, d_rb - scaled by ruf*suf
           .a_el, .v_el, d_el - scaled by euf*duf

        """
        solout = {}
        for item in self.UF_reds:
            ruf, euf, duf, suf = item
            solout[item] = copy.deepcopy(sol)
            SOL = solout[item]
            if nrb > 0:
                SOL.a[:nrb] *= (ruf*suf)
                SOL.v[:nrb] *= (ruf*suf)
                SOL.d[:nrb] *= (ruf*suf)
                SOL.a[nrb:] *= (euf*duf)
                SOL.v[nrb:] *= (euf*duf)
                SOL.d[nrb:] *= (euf*duf)
            else:
                SOL.a *= (euf*duf)
                SOL.v *= (euf*duf)
                SOL.d *= (euf*duf)
            if 'pg' in SOL.__dict__:
                SOL.pg *= suf
        return solout


class DR_Results(OrderedDict):
    """
    Subclass of OrderedDict that contains data recovery results for
    events.

    Notes
    -----
    Below are a couple example :class:`DR_Results` instances named
    `results`. Note that this structure contains all the data
    recovery matrix information originally collected in a
    "prepare_4_cla.py" script.

    The first example is from a PSD buffet run (after running
    ``results.form_extreme()``):

    PP(results)::

        <class 'cla.DR_Results'>[n=2]
            'MaxQ' : <class 'cla.DR_Results'>[n=12]
            'extreme': <class 'cla.DR_Results'>[n=12]

    PP(results['MaxQ'])::

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

    PP(results['MaxQ']['SC_ifa'], 3)::

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
                .labels    : [n=12]: ['I/F Axial Accel     X sc', ...]
                .misc      : None
                .napv      : None
                .se        : 500
                .srsQs     : [n=2]: (25, 50)
                .srsconv   : 1
                .srsfrq    : float64 ndarray 990 elems: (990,)
                .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', ...LV}$']
                .srsopts   : <class 'dict'>[n=2]
                    'eqsine': 1
                    'ic'    : 'steady'
                .srspv     : slice(None, 12, None)
                .srsunits  : 'G, rad/sec^2'
                .uf_reds   : [n=4]: (1, 1, 1.25, 1)
                .units     : 'G, rad/sec^2'
            .event  : 'MaxQ Buffet'
            .ext    : float64 ndarray 24 elems: (12, 2)
            .exttime: float64 ndarray 24 elems: (12, 2)
            .freq   : float64 ndarray 2332 elems: (2332,)
            .maxcase: [n=12]: ['MaxQ', 'MaxQ', 'MaxQ' ...'MaxQ']
            .maxtime: float64 ndarray 12 elems: (12, 1)
            .mincase: [n=12]: ['MaxQ', 'MaxQ', 'MaxQ' ...'MaxQ']
            .mintime: float64 ndarray 12 elems: (12, 1)
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

    PP(results['SC_ifa'], 4)::

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
                .labels    : [n=12]: ['I/F Axial Accel     X sc', ...]
                .misc      : None
                .napv      : None
                .se        : 500
                .srsQs     : [n=2]: (25, 50)
                .srsconv   : 1
                .srsfrq    : float64 ndarray 990 elems: (990,)
                .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', ...LV}$']
                .srsopts   : <class 'dict'>[n=2]
                    'eqsine': 1
                    'ic'    : 'steady'
                .srspv     : slice(None, 12, None)
                .srsunits  : 'G, rad/sec^2'
                .uf_reds   : [n=4]: (1, 1, 1.25, 1)
                .units     : 'G, rad/sec^2'
            .event  : 'SECO2'
            .ext    : float64 ndarray 24 elems: (12, 2)
            .exttime: None
            .hist   : float64 ndarray 2520252 elems: (21, 12, 10001)
            .maxcase: None
            .maxtime: float64 ndarray 252 elems: (12, 21)
            .mincase: None
            .mintime: float64 ndarray 252 elems: (12, 21)
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

    def init(self, Info, mission, event):
        """
        Build initial results data structure.

        Parameters
        ----------
        Info : dict
            Contains data recovery information for each category. The
            category names are the keys.
        mission : str
            Identifies the CLA
        event : str
            Name of event
        """
        for name in Info:
            self[name] = SimpleNamespace(
                ext=None,
                maxcase=None,
                mincase=None,
                mission=mission,
                event=event,
                drminfo=copy.copy(Info[name]))

    def merge(self, results_iter, rename_dict=None):
        """
        Merge CLA results together

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
        The name of each event is extracted from the
        :class:`DR_Results` data structure. The approach to get the
        name is as follows. Note that 'SC_atm' is just an example;
        this routine will simply use first element it finds. Let
        `results` be the current instance of :class:`DR_Results`:

            1. If any element of `results` is a SimpleNamespace:
               ``return results['SC_atm'].event``

            2. Else, if any element from `results` is another
               :class:`DR_Results` structure, then:

               a. If ``results['extreme']`` exists:
                  ``return results['extreme']['SC_atm'].event``

               b. Else ``return ', '.join(key for key in results)``

            3. Else, raise TypeError

        Example usage::

           # merge "liftoff" and "meco" results and rename the
           # liftoff results from "LO" to "Liftoff":

           from pyyeti import cla
           results = cla.DR_Results()
           results.merge(
               (cla.load(fn) for fn in ['../liftoff/results.pgz',
                                        '../meco/results.pgz']),
               {'LO': 'Liftoff'}
               )
           results.strip_hists()
           results.form_extreme()
           results['extreme'].rpttab()
           cla.save('results.pgz', results)

        """
        def _get_event_name(results):
            # get any value from dict:
            v = next(iter(results.values()))
            if isinstance(v, SimpleNamespace):
                return v.event
            if isinstance(v, DR_Results):
                try:
                    v2 = results['extreme']
                except KeyError:
                    return ', '.join(key for key in results)
                else:
                    return next(iter(v2.values())).event
            raise TypeError('unexpected type: {}'
                            .format(str(type(results))))

        events = []
        for results in results_iter:
            event = _get_event_name(results)
            if rename_dict is not None:
                try:
                    newname = rename_dict[event]
                except KeyError:
                    pass
                else:
                    event = newname
            events.append(event)
            self[event] = results
        return events

    def add_maxmin(self, cat, mxmn, mxmn_xvalue=None, domain=None):
        """
        Add maximum and minimum values from an external source

        Parameters
        ----------
        cat : string
            Data recovery category, eg: 'SC_atm'
        mxmn : 2d array_like
            2 column matrix of [max, min]
        mxmn_xvalue : 2d array_like or None; optional
            2 column matrix of [max_xvalue, min_xvalue]. Use None to
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
        comparisons).

        Examples
        --------
        Here is a simple but complete example. CLA results are made up
        for an "ATM" and an "LTM" for 3 events:

        >>> import numpy as np
        >>> from pyyeti import cla
        >>>
        >>> # make up some CLA results:
        >>> events = ('Liftoff', 'Transonics', 'MECO')
        >>> rows = {'ATM': 34, 'LTM': 29}
        >>> ext_results = {'ATM': {}, 'LTM': {}}
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
        >>> defaults = dict(
        ...     se = 0,
        ...     uf_reds = (1, 1, duf, 1),
        ...     # drfile is required, but can just use __name__ for
        ...     # a dummy placeholder (will see a warning)
        ...     drfile = __name__
        ...     )
        >>>
        >>> drdefs = cla.DR_Def(defaults)
        >>>
        >>> def _get_labels(name):
        ...     return ['{} Row {:6d}'.format(name, i+1)
        ...             for i in range(rows[name])]
        >>>
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     name = 'ATM'
        ...     desc = 'S/C Internal Accelerations'
        ...     units = 'm/sec^2, rad/sec^2'
        ...     labels = _get_labels(name)
        ...     drdefs.add(**locals())
        >>>
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     name = 'LTM'
        ...     desc = 'S/C Internal Loads'
        ...     units = 'N, N-m'
        ...     labels = _get_labels(name)
        ...     drdefs.add(**locals())
        >>>
        >>> # for checking, make a pandas DataFrame to summarize data
        >>> # recovery definitions (but skip the excel file for this
        >>> # demo)
        >>> drdefs.excel_summary(None)   # doctest: +ELLIPSIS
                                              ATM ...            LTM
        desc           S/C Internal Accelerations ... Internal Loads
        drfile          ...
        drfunc                                ATM ...            LTM
        filterval                           1e-06 ...              -
        histlabels                           None ...              -
        histpv                               None ...              -
        histunits                            None ...              -
        labels      34: ['ATM Row  ... w     34'] ... ... w     29']
        misc                                 None ...              -
        napv                                 None ...              -
        se                                      0 ...              -
        srsQs                                None ...              -
        srsconv                                 1 ...              -
        srsfrq                               None ...              -
        srslabels                            None ...              -
        srsopts                              None ...              -
        srspv                                None ...              -
        srsunits                             None ...              -
        uf_reds                 4: (1, 1, 1.2, 1) ...              -
        units                  m/sec^2, rad/sec^2 ...         N, N-m
        >>>
        >>> # prepare results data structure:
        >>> DR = cla.Event_DR()
        >>> DR.add(None, drdefs)
        >>> results = cla.DR_Results()
        >>> for event in events:
        ...     results[event] = DR.prepare_results(mission, event)
        ...     for drm in rows:
        ...         results[event].add_maxmin(
        ...             drm, ext_results[drm][event])
        >>>
        >>> # Done with setup; now we can use the standard cla tools:
        >>> results.form_extreme()
        >>> # To write an extreme 'Results.xlsx' file, uncomment the
        >>> # following line:
        >>> # results['extreme'].rpttab(excel='Results')
        """
        self[cat].ext = mxmn
        self[cat].exttime = mxmn_xvalue
        self[cat].domain = domain

    def _get_drfunc(self, drinfo, get_psd=False):
        spec = importlib.util.spec_from_file_location("has_drfuncs",
                                                      drinfo.drfile)
        drmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drmod)
        func = eval('drmod.' + drinfo.drfunc)
        if get_psd:
            try:
                psdfunc = eval('drmod.' + drinfo.drfunc + '_psd')
            except AttributeError:
                psdfunc = None
            return func, psdfunc
        return func

    def _store_maxmin(self, res, mm, j, case):
        res.mx[:, j] = mm.ext[:, 0]
        res.maxtime[:, j] = mm.exttime[:, 0]
        res.mn[:, j] = mm.ext[:, 1]
        res.mintime[:, j] = mm.exttime[:, 1]
        res.cases[j] = case

    def time_data_recovery(self, sol, nas, case, DR, n, j,
                           dosrs=True):
        """
        Time-domain data recovery function

        Parameters
        ----------
        sol : dict
            Keys are the "uf_reds" values and the values are a
            solution namespace. For example, if one of the "uf_reds"
            tuples is: ``(1, 1, 1.25, 1)``, then `sol` would look like
            this for a time domain event::

                sol[(1, 1, 1.25, 1)].a
                sol[(1, 1, 1.25, 1)].v
                sol[(1, 1, 1.25, 1)].d
                sol[(1, 1, 1.25, 1)].d_static
                sol[(1, 1, 1.25, 1)].d_dynamic
                sol[(1, 1, 1.25, 1)].pg        # optional
                sol[(1, 1, 1.25, 1)].h         # time step
                sol[(1, 1, 1.25, 1)].t         # time vector

        nas : dictionary
            This is the nas2cam dictionary:
            ``nas = pyyeti.op2.rdnas2cam()``
        case : string
            String identifying the case; stored in the
            ``self.cat[name].cases`` and the `.mincase` and `.maxcase`
            lists
        DR : instance of :class:`Event_DR`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.Event_DR()``). It is an event specific version
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
            drfunc = self._get_drfunc(dr)
            resp = drfunc(SOL, nas, DR.Vars, dr.se)

            mm = maxmin(resp, SOL.t)
            extrema(res, mm, case)

            if first:
                res.domain = 'time'
                m = mm.ext.shape[0]
                res.mx = np.zeros((m, n))
                res.mn = np.zeros((m, n))
                res.maxtime = np.zeros((m, n))
                res.mintime = np.zeros((m, n))
                res.cases = n*[[]]
                if dr.histpv is not None:
                    res.time = SOL.t
                    m = len(resp[dr.histpv, 0])
                    res.hist = np.zeros((n, m, len(res.time)))
                if dr.srspv is not None and dosrs:
                    res.srs = SimpleNamespace(frq=dr.srsfrq,
                                              units=dr.srsunits,
                                              srs={}, ext={})
                    m = len(resp[dr.srspv, 0])
                    sh = (n, m, (len(res.srs.frq)))
                    for q in dr.srsQs:
                        res.srs.srs[q] = np.zeros(sh)

            self._store_maxmin(res, mm, j, case)

            if dr.histpv is not None:
                res.hist[j] = resp[dr.histpv]

            if dr.srspv is not None and dosrs:
                res.srs.type = ('eqsine' if _is_eqsine(dr.srsopts)
                                else 'srs')
                rr = resp[dr.srspv]
                sr = 1/SOL.h
                for q in dr.srsQs:
                    srs_cur = dr.srsconv*srs.srs(rr.T, sr, dr.srsfrq,
                                                 q, **dr.srsopts).T
                    res.srs.srs[q][j] = srs_cur
                    if first:
                        res.srs.ext[q] = srs_cur
                    else:
                        res.srs.ext[q] = np.fmax(res.srs.ext[q],
                                                 srs_cur)

    def solvepsd(self, nas, case, DR, m, b, k, forcepsd, t_frc, freq,
                 rfmodes=None):
        """
        Solve equations of motion in frequency domain with PSD forces

        Parameters
        ----------
        nas : dictionary
            This is the nas2cam dictionary:
            ``nas = pyyeti.op2.rdnas2cam()``
        case : string
            String identifying the case; stored in the
            ``self.cat[name].cases`` and the `.mincase` and `.maxcase`
            lists
        DR : instance of :class:`Event_DR`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.Event_DR()``). It is an event specific version
            of all combined :class:`DR_Def` objects with all ULVS
            matrices applied.
        m, b, k : 2d array_like
            The mass, damping, stiffness matrices suitable for use in
            :class:`pyyeti.ode.SolveUnc`.
        forcepsd : 2d array_like
            Matrix of force psds; each row is a force
        t_frc : 2d array_like
            Transformation from system modal DOF to forced DOF; number
            of rows is the number of PSDs:
            ``t_frc.shape[0] == forcepsd.shape[0]``
        freq : 1d array_like
            Frequency vector at which solution will be computed
        rfmodes : 1d array_like or None; optional
            Specifies where the res-flex modes are; if None, no
            resflex

        Notes
        -----
        The `self` results dictionary is updated (see
        :class:`DR_Results` for an example).
        """
        forcepsd, t_frc = np.atleast_2d(forcepsd, t_frc)
        nonzero_forces = np.any(forcepsd, axis=1).nonzero()[0]
        if nonzero_forces.size:
            print('Trimming off {} zero forces'
                  .format(forcepsd.shape[0]-nonzero_forces.size))
            forcepsd = forcepsd[nonzero_forces]
            t_frc = t_frc[nonzero_forces]
        freq = np.atleast_1d(freq)

        # decompose system equations for uncoupled solver:
        fs = ode.SolveUnc(m, b, k, rf=rfmodes)
        rpsd = forcepsd.shape[0]
        unitforce = np.ones_like(freq)
        freqw = 2*np.pi*freq

        # initialize categories for data recovery
        drfuncs = {}
        for key, value in self.items():
            value.freq = freq
            value._psd = {}
            value._psd[case] = 0.0
            # get data recovery functions just once, outside of main
            # loop; returns tuple: (func, func_psd) ... func_psd will
            # be None if no special function defined for PSD
            # recovery):
            drfuncs[key] = self._get_drfunc(value.drminfo,
                                            get_psd=True)

        import time
        timers = [0, 0, 0]
        for i in range(rpsd):
            print('Processing force {} of {}'.format(i+1, rpsd))
            # solve for unit FRF for i'th force:
            genforce = t_frc[i][:, None] * unitforce
            t1 = time.time()
            sol = fs.fsolve(genforce, freq)
            sol.pg = unitforce
            timers[0] += time.time() - t1

            # apply uncertainty factors:
            t1 = time.time()
            sol = DR.psd_apply_uf(sol, nas['nrb'])
            # sol = DR.apply_uf(sol, *mbk, nas['nrb'], rfmodes)
            timers[1] += time.time() - t1

            # perform data recovery:
            t1 = time.time()
            for key, value in self.items():
                uf_reds = value.drminfo.uf_reds
                se = value.drminfo.se
                if drfuncs[key][1]:
                    # use PSD recovery function if present:
                    drfuncs[key][1](sol[uf_reds], nas, DR.Vars, se,
                                    freq, forcepsd, value, case, i)
                else:
                    # otherwise, use normal recovery function:
                    resp = drfuncs[key][0](sol[uf_reds], nas,
                                           DR.Vars, se)
                    value._psd[case] += forcepsd[i] * abs(resp)**2
            timers[2] += time.time() - t1
        print('timers =', timers)

    def psd_data_recovery(self, case, DR, n, j, dosrs=True,
                          peak_factor=3.0):
        """
        PSD data recovery function

        Parameters
        ----------
        case : string
            String identifying the case; stored in the
            ``self.cat[name].cases`` and the `.mincase` and `.maxcase`
            lists
        DR : instance of :class:`Event_DR`
            Defines data recovery for an event simulation (and is
            created in the simulation script via
            ``DR = cla.Event_DR()``). It is an event specific version
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
        :class:`DR_Results` for an example). Note that the "time"
        entries (eg, `maxtime`, `exttime`) are really the "apparent
        frequency" values, an estimate for the number of positive
        slope zero crossings per second.
        """
        def _calc_rms(df, p):
            sumpsd = p[:, :-1] + p[:, 1:]
            return np.sqrt((df * sumpsd).sum(axis=1)/2)

        for name, res in self.items():
            first = res.ext is None
            dr = DR.Info[name]  # record with: .desc, .labels, ...
            # compute area under curve (rms):
            freq = res.freq
            freqstep = np.diff(freq)
            psd = res._psd[case]
            rms = _calc_rms(freqstep, psd)

            # Need "velocity" rms to estimate number of positive slope
            # zero crossings (apparent frequency).
            # `vrms` = (velocity rms)/2/pi ... (2 pi factor would cancel
            # later, so just leave it out):
            vrms = _calc_rms(freqstep, freq**2 * psd)

            pk = peak_factor * rms
            pk_freq = vrms/rms
            mm = SimpleNamespace(
                ext=np.column_stack((pk, -pk)),
                exttime=np.column_stack((pk_freq, pk_freq)))

            extrema(res, mm, case)

            if first:
                res.domain = 'freq'
                m = mm.ext.shape[0]
                res.rms = np.zeros((m, n))
                res.mx = np.zeros((m, n))
                res.mn = np.zeros((m, n))
                res.maxtime = np.zeros((m, n))
                res.mintime = np.zeros((m, n))
                res.cases = n*[[]]

                if dr.histpv is not None:
                    m = len(psd[dr.histpv, 0])
                    res.psd = np.zeros((n, m, len(res.freq)))
                if dr.srspv is not None and dosrs:
                    res.srs = SimpleNamespace(frq=dr.srsfrq,
                                              units=dr.srsunits,
                                              srs={}, ext={})
                    m = len(psd[dr.srspv, 0])
                    sh = (n, m, (len(res.srs.frq)))
                    for q in dr.srsQs:
                        res.srs.srs[q] = np.zeros(sh)

            self._store_maxmin(res, mm, j, case)

            if dr.histpv is not None:
                res.psd[j] = psd[dr.histpv]

            # srs:
            if dr.srspv is not None and dosrs:
                if _is_eqsine(dr.srsopts):
                    res.srs.type = 'eqsine'
                    eqsine = True
                else:
                    res.srs.type = 'srs'
                    eqsine = False
                spec = np.hstack((freq[:, None], psd[dr.srspv].T))
                for q in dr.srsQs:
                    fact = peak_factor * dr.srsconv
                    if eqsine:
                        fact /= q
                    srs_cur = fact * srs.vrs(spec, dr.srsfrq, q,
                                             linear=True).T
                    res.srs.srs[q][j] = srs_cur
                    if first:
                        res.srs.ext[q] = srs_cur
                    else:
                        res.srs.ext[q] = np.fmax(res.srs.ext[q],
                                                 srs_cur)
            del res._psd

    def form_stat_ext(self, k):
        """
        Form statistical extreme response for event results

        Parameters
        ----------
        k : scalar
            The statistical k-factor: extreme = mean + k*sigma

        Notes
        -----
        Each results record (eg, ``self['SECO1']['SC_ifa']``) is
        expected to have `.mx` and `.mn` members. Each of these is
        data-recovery rows x load cases. This routine will calculate a
        new `.ext` member by::

           .ext = [mean(mx) + k*std(mx), mean(mn) - k*std(mn)]

        If ``.srs.srs[q]`` is present, a new ``srs.ext[q]`` will be
        calculated as well. Each ``.srs.srs[q]`` is assumed to be
        cases x rows x freq.

        The `.maxcase`, `.mincase` and `.exttime` members are all set
        to None (since all cases contribute to the extrema).
        """
        for name, res in self.items():
            mx = res.mx.mean(axis=1) + k*res.mx.std(ddof=1, axis=1)
            mn = res.mn.mean(axis=1) - k*res.mn.std(ddof=1, axis=1)
            res.ext = np.vstack([mx, mn]).T
            res.maxcase = res.mincase = res.exttime = None

            # handle SRS if it is there:
            if 'srs' in res.__dict__:
                for Q in res.srs.srs:
                    arr = res.srs.srs[Q]
                    res.srs.ext[Q] = (arr.mean(axis=0) +
                                      k*arr.std(ddof=1, axis=0))

    def form_extreme(self, ext_name='Envelope', case_order=None,
                     doappend=2):
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
            (:class:`DR_Results` is an OrderedDict). Note that
            `case_order` is used for highest level only. `case_order`
            defines the 'cases' member variable (for example,
            ``self['extreme']['SC_atm'].cases``).
        doappend : integer; optional
            Flag that defines how to keep track of lower level
            `.maxcase` and `.mincase` ids as extreme envelopes are
            computed:

            ==========  ==============================================
            `doappend`  Description
            ==========  ==============================================
                 0      ignore lower level `.maxcase` & `.mincase` and
                        just use higher level key for case
                 1      keep lower level `.maxcase` & `.mincase` and
                        prepend higher level keys as levels are
                        traversed
                 2      ignore lowest level `.maxcase` & `.mincase`,
                        but prepend after that
                 3      keep only lowest level `.maxcase` & `.mincase`
                        (do not append any keys)
            ==========  ==============================================

        Notes
        -----
        This routine will 'extreme' dictionaries at all appropriate
        levels. Any old 'extreme' dictionaries (at all levels) are
        deleted before anything else is done.
        """
        DEFDOMAIN = 'X-Value'

        def _init_drm_ext(cases, val, ext_name):
            ncases = len(cases)
            nrows = val.ext.shape[0]
            mx = np.empty((nrows, ncases))
            mn = np.empty((nrows, ncases))
            maxtime = np.empty((nrows, ncases))
            mintime = np.empty((nrows, ncases))
            mx[:] = np.nan
            mn[:] = np.nan
            maxtime[:] = np.nan
            mintime[:] = np.nan
            drminfo = copy.copy(val.drminfo)

            ret = SimpleNamespace(
                cases=cases, drminfo=drminfo, mission=val.mission,
                event=ext_name, ext=None, exttime=None, maxcase=None,
                mincase=None, mx=mx, mn=mn, maxtime=maxtime,
                mintime=mintime, domain=DEFDOMAIN)

            # handle SRS if present:
            osrs = getattr(val, 'srs', None)
            if osrs is not None:
                srs_ns = copy.copy(osrs)
                srs_ns.ext = copy.deepcopy(osrs.ext)
                srs_ns.srs = {}
                ndof, nf = list(osrs.ext.values())[0].shape
                for q in srs_ns.ext:
                    srs_ns.srs[q] = np.empty((ncases, ndof, nf))
                    srs_ns.srs[q][:] = np.nan
                ret.srs = srs_ns

            return ret

        def _mk_case_lbls(case, val, use_ext, doappend):
            case = str(case)
            if use_ext and doappend == 2:
                doappend = 1
            if 'maxcase' in val.__dict__:   # always true?
                if doappend == 1:
                    maxcase = [case+','+i for i in val.maxcase]
                    mincase = [case+','+i for i in val.mincase]
                elif doappend == 3:
                    maxcase = val.maxcase
                    mincase = val.mincase
                elif doappend == 0 or doappend == 2:
                    maxcase = mincase = case
            else:
                maxcase = mincase = case
            return maxcase, mincase

        def _expand(ext_old, labels, pv):
            if ext_old.drminfo.labels == labels:
                return ext_old
            # Expand:
            #   ext, exttime, maxcase, mincase,
            #   mx, mn, maxtime, mintime
            n = len(labels)
            ext_new = copy.copy(ext_old)
            ext_new.drminfo = copy.copy(ext_old.drminfo)
            ext_new.drminfo.labels = labels
            for name in ['ext', 'exttime', 'mx', 'mn',
                         'maxtime', 'mintime']:
                old = ext_old.__dict__[name]
                if old is not None:
                    new = np.empty((n, old.shape[1]))
                    new[:] = np.nan
                    new[pv] = old
                    ext_new.__dict__[name] = new
            if ext_old.maxcase is not None:
                maxcase = ['n/a' for i in range(n)]
                mincase = ['n/a' for i in range(n)]
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
            for lbls, extA, extB in ((l1, ext1, ext2),
                                     (l2, ext2, ext1)):
                if len(lbls) != len(set(lbls)):
                    msg = ('Row labels for "{}" (event "{}") are not '
                           'all unique. Cannot compare to event "{}".'
                           .format(ext1.drminfo.desc,
                                   ext1.event,
                                   ext2.event))
                    raise ValueError(msg)
            # for both ext1 and ext2, expand:
            #   [ext, exttime, maxcase, mincase,
            #    mx, mn, maxtime, mintime]
            l3, pv1, pv2 = locate.merge_lists(l1, l2)
            return (_expand(ext1, l3, pv1),
                    _expand(ext2, l3, pv2))

        def _calc_extreme(dct, ext_name, case_order, doappend):
            if case_order is None:
                # cases = sorted(list(dct.keys()), key=str)
                cases = list(dct.keys())
                # eg: 'Liftoff', 'MECO', etc
            else:
                cases = [str(i) for i in case_order]
            new_ext = DR_Results()
            domain = None
            for j, case in enumerate(cases):
                try:
                    curext = dct[cases[j]]['extreme']
                    use_ext = True
                except KeyError:
                    curext = dct[cases[j]]
                    use_ext = False
                domain = None
                for drm, val in curext.items():
                    if drm not in new_ext:
                        new_ext[drm] = _init_drm_ext(
                            cases, val, ext_name)
                    else:
                        new_ext[drm], val = _check_row_compatibility(
                            new_ext[drm], val)
                    if domain is not None:
                        if domain != val.domain:
                            domain = DEFDOMAIN
                    else:
                        domain = val.domain
                    maxcase, mincase = _mk_case_lbls(
                        case, val, use_ext, doappend=doappend)
                    extrema(new_ext[drm], val, maxcase, mincase, j)

                    osrs = getattr(val, 'srs', None)
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

        def _delete_extreme(dct):
            dct.pop('extreme', None)
            for value in dct.values():
                if isinstance(value, DR_Results):
                    _delete_extreme(value)
                else:
                    return

        def _add_extreme(dct, ext_name, case_order, doappend):
            for name, value in list(dct.items()):
                if isinstance(value, SimpleNamespace):
                    # one level too deep ... just return quietly
                    return
                else:
                    # use ext_name, case_order only at the top level
                    _add_extreme(value, name, None, doappend)
            dct['extreme'] = _calc_extreme(
                dct, ext_name, case_order, doappend)

        # main routine:
        _delete_extreme(self)
        _add_extreme(self, ext_name, case_order, doappend)

    def strip_hists(self):
        """
        Strips out time and frequency domain histories and non-extreme
        srs data

        Notes
        -----
        This is typically to reduce file size of a summary results
        structure where the histories are already saved in other
        files. Run this before :func:`DR_Results.form_extreme`.

        See example usage in :func:`DR_Results.merge`.

        """
        def _delete_hists(dct):
            for value in dct.values():
                if isinstance(value, DR_Results):
                    _delete_hists(value)
                elif isinstance(value, SimpleNamespace):
                    for attr in ('hist', 'time', 'psd', 'freq'):
                        try:
                            delattr(value, attr)
                        except AttributeError:
                            pass
                    if hasattr(value, 'srs'):
                        try:
                            delattr(value.srs, 'srs')
                        except AttributeError:
                            pass
                else:
                    raise TypeError('unexpected type: {}'
                                    .format(str(type(value))))

        # main routine:
        _delete_hists(self)

    def rptext(self, event=None, direc='ext', doabsmax=False,
               numform='{:13.5e}', perpage=-1):
        """
        Writes .ext files for all max/min results.

        Parameters
        ----------
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[name].event``
        direc : string; optional
            Name of directory to put tables; will be created if
            doesn't exist
        doabsmax : bool; optional
            If True, compare only absolute maximums.
        numform : string; optional
            Format of the max & min numbers.
        perpage : integer; optional
            The number of lines to write perpage. If < 0, there is no
            limit (one page).

        Notes
        -----
        The output files contain the maximums, minimums and cases as
        applicable.
        """
        if not os.path.exists(direc):
            os.mkdir(direc)
        for name, res in self.items():
            mission = res.mission
            if event is None:
                event = res.event
            title = '{} - {} Extrema Results'.format(mission, event)
            filename = os.path.join(direc, name+'.ext')
            rptext1(res, filename, title=title,
                    doabsmax=doabsmax, numform=numform,
                    perpage=perpage)

    def rpttab(self, event=None, direc='tab', count_filter=1e-6,
               excel=False):
        """
        Write results tables with bin count information.

        Parameters
        ----------
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[name].event``
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
        """
        if not os.path.exists(direc):
            os.mkdir(direc)
        if isinstance(excel, str):
            # create a single excel file
            filename = os.path.join(direc, excel + '.xlsx')
            opts = {'nan_inf_to_errors': True}
            workbook = xlsxwriter.Workbook(filename, opts)
            filename = workbook
        else:
            workbook = None
        try:
            for name in sorted(self):
                res = self[name]
                mission = res.mission
                if event is None:
                    event = res.event
                ttl = ('{} - {} Extrema Results and Bin Count Tables'
                       .format(mission, event))
                if excel:
                    if not isinstance(excel, str):
                        filename = os.path.join(direc, name+'.xlsx')
                else:
                    filename = os.path.join(direc, name+'.tab')
                rpttab1(res, filename, title=ttl,
                        count_filter=count_filter, name=name)
        finally:
            if workbook is not None:
                workbook.close()

    def rptpct(self, refres, names=('Self', 'Reference'),
               event=None, fileext='.cmp', direc='compare',
               keyconv=None, **rptpct1_args):
        """
        Write comparison files for all max/min data in results.

        Parameters
        ----------
        refres : dictionary
            Dictionary of reference results to compare to. Keys are
            the event names and values are either:

              1. A 2-column matrix of [max, min], or
              2. A namespace with the .ext member = [max, min]

            Notes:

              1. If the keys are different than those in
                 ``self[event]`` (eg, 'SC_ifa', 'SC_atm', etc), then
                 the input `keyconv` is necessary.
              2. If a key in ``self[event]`` is not found in `refres`,
                 a message is printed and that item is skipped

        names : list/tuple; optional
            2-element list or tuple identifying the two sets of
            results that are being compared. The first is for `self`
            and the second is for `refres`.
        event : string or None; optional
            String identifying the event; if None, event is taken from
            each ``self[name].event``
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
        for drm in self:
            refdrm = keyconv[drm] if drm in keyconv else drm
            if refdrm not in refres:
                skipdrms.append(drm)
            else:
                res = self[drm]
                mission = res.mission
                if event is None:
                    event = res.event
                title = ('{}, {} - {} vs. {}'
                         .format(mission, event, *names))
                filename = os.path.join(direc, drm+fileext)
                try:
                    refext = refres[refdrm].ext
                except AttributeError:
                    refext = refres[refdrm]
                rptpct1(res, refext, filename, title=title,
                        names=names, **rptpct1_args)
        if len(skipdrms) > 0:
            warn('Some comparisons were skipped (not found in'
                 ' `refres`):\n{}'
                 .format(str(skipdrms)), RuntimeWarning)

    def srs_plots(self, event=None, Q='auto', drms=None,
                  inc0rb=True, fmt='pdf', onepdf=True, layout=[2, 3],
                  figsize=[11, 8.5], showall=None, showboth=False,
                  direc='srs_plots', sub_right=None):
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
            :func:`matplotlib.pyplot.savefig`. If None, no figures
            will be saved. Typical values for `fmt` are (from
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

        onepdf : bool; optional
            If True (and `fmt` == "pdf") write all plots to one PDF
            file; otherwise, each figure is put in its own file.
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
        sub_right : scalar or None; optional
            Used in: ``plt.subplots_adjust(right=sub_right)`` when
            a legend is placed outside the plots. If None, this
            routine tries to make an educated guess from the longest
            label.

        Returns
        -------
        figs : list
            List of figure handles created by this routine.

        Notes
        -----
        Example::

            # write a pdf file to 'srs_plots/':
            results.srs_plots()

            # write png file(s) to 'png/':
            results.srs_plots(fmt='png', direc='png')

        """
        return mk_plots(self, issrs=True, event=event, Q=Q, drms=drms,
                        inc0rb=inc0rb, fmt=fmt, onepdf=onepdf,
                        layout=layout, figsize=figsize,
                        showall=showall, showboth=showboth,
                        direc=direc, sub_right=sub_right,
                        cases=None)

    def resp_plots(self, event=None, drms=None, inc0rb=True,
                   fmt='pdf', onepdf=True, layout=[2, 3],
                   figsize=[11, 8.5], cases=None, direc='resp_plots',
                   sub_right=None):
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
            :func:`matplotlib.pyplot.savefig`. If None, no figures
            will be saved. Typical values for `fmt` are (from
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

        onepdf : bool; optional
            If True (and `fmt` == "pdf") write all plots to one PDF
            file; otherwise, each figure is put in its own file.
        layout : 2-element tuple/list; optional
            Subplot layout, eg: (2, 3) for 2 rows by 3 columns
        figsize : 2-element tuple/list; optional
            Define page size in inches.
        cases : tuple/list of case names to plot or None; optional
            If None, all cases are plotted.
        direc : string; optional
            Directory name to put all output plot files; will be
            created if it doesn't exist.
        sub_right : scalar or None; optional
            Used in: ``plt.subplots_adjust(right=sub_right)`` when
            a legend is placed outside the plots. If None, this
            routine tries to make an educated guess from the longest
            label.

        Returns
        -------
        figs : list
            List of figure handles created by this routine.

        Notes
        -----
        Example::

            # write a pdf file to 'resp_plots/':
            results.resp_plots()

            # write png file(s) to 'png/':
            results.resp_plots(fmt='png', direc='png')

        """
        return mk_plots(self, issrs=False, event=event, drms=drms,
                        inc0rb=inc0rb, fmt=fmt, onepdf=onepdf,
                        layout=layout, figsize=figsize,
                        cases=cases, direc=direc,
                        sub_right=sub_right, Q='auto',
                        showall=None, showboth=False)


#def Time_Add_SRS(results,DR):
#    """Adds SRSs to the results structure; see Time_DR_cols() for more information
#
#Normally not need since this is typically done during Time_DR_cols()"""
#    DRNames = keys(results)
#    for name in DRNames:
#        dr = DR['Info'][name]  % record with: .drfunc, .psd_drfunc, .drminfo, .srsinfo
#        if dr.srsinfo.rows:
#            r = results[name]
#            frq = dr.srsinfo.frq
#            r.srs.frq = frq
#            r.srs.units = dr.srsinfo.units
#            sr = (length(r.time)-1)/(r.time(end)-r.time(1))
#            rows = dr.srsinfo.rows
#            conv = dr.srsinfo.conv
#            if rows != dr.drminfo.histpv:
#                prterr.error("Time_Add_SRS:  Cannot add SRS because 'histpv' != 'srspv'.\n")
#            opts = dr.srsinfo.opts
#            eqsine = _is_eqsine(opts)
#            if eqsine:  r.srs.type = 'eqsine'
#            else:       r.srs.type = 'srs'
#            n = length(r.cases)
#            for q in dr.srsinfo.Qs:
#                if rows == 'all':
#                    L = size(r.hist,2)
#                    hist = r.hist(:,:)  % flatten to 2-D for SRS
#                else:
#                    L = length(rows)
#                    hist = r.hist(:,rows,:)(:,:)  % flatten to 2-D for SRS routine
#                r.srs.srs[q] = zeros(length(frq),L,n)
#                r.srs.srs[q](:) = conv .* srs.srs(hist,sr,frq,q,**opts)
#                r.srs.ext[q] = max(r.srs.srs[q],,3)


#def FRF_DR_cols(results,sol,nrb,case,DR,n,j,dosrs=1):
#    """Frequency response function data recovery function
#
#-- beta -- written but never used
#
#Parameters
#----------
#  results = results dictionary as described in form_extreme; results[key] must be the
#            data recovery items (eg, results['SC_atm'])
#  sol     = solution dictionary with each uf_reds vector as keys:
#            - each sol[uf_reds] is a record, input only, expected to have (at least):
#                .a = modal acceleration frequency response matrix (complex)
#                .v = modal velocity frequency response matrix (complex)
#                .d = modal displacement frequency response matrix (complex)
#                .f = frequency-vector
#  nrb     = scalar, number of rigid-body modes
#  case  = string identifying the case; stored in the results[drm].cases list and in
#            the results[drm].mincase & .maxcase lists
#  DR      = dictionary containing data recovery info; see description in DRsetup()
#  n       = total number of load cases
#  j       = current load case number
#  dosrs   = flag; if false, skip the calculation SRSs
#            - still need to add: FRF_Add_SRS() for similar functionality to Time_Add_SRS()
#
#The results dictionary is updated; for example:
#
#    results['SC_atm'] members:
#      .mx, .mn, .maxtime, .mintime <-- all r x cases
#      .cases (list of cases)
#      .freq (freq x 1)
#      .FRF (freq x r x cases)
#      .srs.srs[q] (srsfreq x r x cases)
#      .srs.ext[q] (each srsfreq x r)
#      .srs.frq (srsfreq x 1) (same as sol[uf_reds].f)
#      .srs.type (string, either 'srs' or 'eqsine')
#      .srs.units (string, eg: 'G, rad/sec^2')
#      .ext (r x 2), .maxcase (length r), .mincase (length r)
#"""
#    DRNames = keys(results)
#    for name in DRNames:
#        r = results[name]
#        first = r.ext == []
#        dr = DR['Info'][name]  % record with: .drfunc, .psd_drfunc, .drminfo, .srsinfo
#        uf_reds = dr.drminfo.uf_reds
#        SOL = sol[uf_reds]
#        resp = dr.drfunc(SOL,nrb,DR,dr.drminfo.se)
#        mm = maxmin(abs(resp),SOL.f)
#
#        extrema(r,mm,case)
#%        maxima(r,mm(:,:2),scase)
#
#        if first:
#            r.domain = 'freq'
#            r.mx = r.mn = r.maxtime = r.mintime = zeros(size(mm,1),n)
#%            r.mx = r.maxtime = zeros(size(mm,1),n)
#            r.cases = n*[[]]
#            if dr.drminfo.histpv:
#                if dr.drminfo.histpv == 'all':
#                    r.FRF = zeros(length(SOL.f),size(resp,1),n)
#                else:
#                    r.FRF = zeros(length(SOL.f),length(dr.drminfo.histpv),n)
#                r.freq = SOL.f
#            if dr.srsinfo.rows and dosrs:
#                frq = dr.srsinfo.frq
#                r.srs.frq = frq
#                r.srs.units = dr.srsinfo.units
#                for q in dr.srsinfo.Qs:
#                    if dr.srsinfo.rows == 'all':
#                        r.srs.srs[q] = zeros(length(frq),size(resp,1),n)
#                    else:
#                        r.srs.srs[q] = zeros(length(frq),length(dr.srsinfo.rows),n)
#
#        r.mx[:, j] = mm[:, 0]
#        r.maxtime[:, j] = mm[:, 1]
#        r.mn[:, j] = mm[:, 2]
#        r.mintime[:, j] = mm[:, 3]
#        r.cases(j) = case
#
#        if dr.drminfo.histpv:  % save histories
#            if dr.drminfo.histpv == 'all':
#                r.FRF(:,:,j) = resp.'
#            else:
#                r.FRF(:,:,j) = resp(dr.drminfo.histpv,:).'
#
#        if dr.srsinfo.rows and dosrs:
#            rows = dr.srsinfo.rows
#            frq = dr.srsinfo.frq
#            conv = dr.srsinfo.conv
#            eqsine = _is_eqsine(dr.srsinfo.opts)
#            if eqsine:  r.srs.type = 'eqsine'
#            else:       r.srs.type = 'srs'
#            for q in dr.srsinfo.Qs:
#                if eqsine:  fact = conv/q
#                else:       fact = conv
#                if rows == 'all':
#                    srs_cur = fact .* srs.srs_frf([SOL.f(:),resp.'],frq,q)
#                else:
#                    srs_cur = fact .* srs.srs_frf([SOL.f(:),resp(rows,:).'],frq,q)
#                r.srs.srs[q](:,:,j) = srs_cur
#                if first: r.srs.ext[q] = srs_cur
#                else:     r.srs.ext[q] = max(r.srs.ext[q], srs_cur)



#def PSD_Add_SRS(results,DR):
#    """Adds SRSs to the results structure; see PSD_DR_cols() for more information
#
#Normally not need since this is typically done during PSD_DR_cols()"""
#    DRNames = keys(results)
#    for name in DRNames:
#        dr = DR['Info'][name]  % record with: .drfunc, .psd_drfunc, .drminfo, .srsinfo
#        if dr.srsinfo.rows:
#            r = results[name]
#            frq = dr.srsinfo.frq
#            r.srs.frq = frq
#            r.srs.units = dr.srsinfo.units
#            rows = dr.srsinfo.rows
#            conv = dr.srsinfo.conv
#            if rows != dr.drminfo.histpv:
#                prterr.error("PSD_Add_SRS:  Cannot add SRS because 'histpv' != 'srspv'.\n")
#            eqsine = _is_eqsine(dr.srsinfo.opts)
#            if eqsine:  r.srs.type = 'eqsine'
#            else:       r.srs.type = 'srs'
#            n = length(r.cases)
#            for q in dr.srsinfo.Qs:
#                if eqsine: fact = (3/q)*conv
#                else:      fact = 3*conv
#                if rows == 'all':
#                    L = size(r.PSD,2)
#                    psd = r.PSD(:,:)  % flatten to 2-D for VRS
#                else:
#                    L = length(rows)
#                    psd = r.PSD(:,rows,:)(:,:)  % flatten to 2-D for VRS
#                r.srs.srs[q] = zeros(length(frq),L,n)
#                r.srs.srs[q](:) = fact .* srs.vrs([r.freq,psd],frq,q)
#                r.srs.ext[q] = max(r.srs.srs[q],,3)


def PSD_consistent_rss(resp, xr, yr, rr, freq, forcepsd, drmres,
                       case, i):
    """
    Compute phase-consistent (time-correlated) root-sum-square (RSS)
    responses in a PSD analysis. Each RSS is of two rows.

    In the following, 'x' denotes DOF 1 for the RSS and 'y' denotes
    the perpendicular direction: resp = sqrt(x*x + y*y).

    Parameters
    ----------
    resp : 2d ndarray
        Frequency response curves: DOF x len(freq)
    xr : 1d array_like
        Index vector of the 'x' `resp` rows (input to RSS)
    yr : 1d array_like
        Index vector of the 'y' `resp` rows (input to RSS)
    rr : 1d array_like or None
        If RSS responses are to be stored in the same matrix
        (category) as `resp`, then `rr` is an index vector specifying
        which rows will hold the RSS. For example, `resp` could have
        3 rows with ``xr = 0``, ``yr = 1``, and ``rr = 2``. The 3rd
        row would be the RSS of the first two.

        One the other hand, if the RSS is to be stored alone in its
        category, set `rr` to None. For example, `resp` could have 2
        rows with ``xr = 0``, ``yr = 1``. Then, the category would
        only have 1 data recovery item
    freq : 1d array_like
        Frequency vector (Hz)
    forcepsd : 2d ndarray
        Matrix of force PSDs; nforces x len(freq)
    drmres : input/output SimpleNamespace
        Results for a DRM; eg if drmres = results['ifa']:

           .rms (r x cases)
           .cases (list of cases)
           .freq (freq x 1)
           .psd (cases x r x freq)
           ._psd (temp dict, psd[case] is freq x r)
           .srs.srs[q] (cases x r x freq)
           .srs.ext[q] (each r x freq)
           .srs.frq (freq,)
           .srs.type (string, either 'srs' or 'eqsine')
           .srs.units (string, eg: 'G, rad/sec^2')
           .ext (r x 2), .maxcase (length r), .mincase ([])

    case : string
        Case identifier (like 'MaxQ') for storing the PSD results,
        eg::

            results['ifa']._psd[case]

    i : integer
        Current force index; starts at 0 and goes to nforces-1

    Returns
    -------
    None

    Notes
    -----
    The `drmres` input is modified on each call::

         ._psd[case] is updated (size = drm rows x freq)

    On the last call the RSS is computed and stored in ``._psd[case]``
    according to `rr`.

    These members are created to keep track of needed values (but
    deleted on last call)::

         .tmp.varx  = 'x' variance
         .tmp.vary  = 'y' variance
         .tmp.covar = covariance
         .tmp.xresp = list of 'x' response matrices;
                       - list is nforces long
                       - each response matrix is len(xr) x freq
         .tmp.yresp = list of 'y' response matrices
    """
    # drmres is a SimpleNamespace: drmres = results[drm]
    # i is psd force index
    F = forcepsd[i]
    # normal, non-rss data recovery:
    if rr is not None:
        drmres._psd[case] += F * abs(resp)**2
    N = forcepsd.shape[0]
    if i == 0:
        drmres.tmp = SimpleNamespace(
            varx=0,
            vary=0,
            covar=0,
            xresp=[0] * N,
            yresp=[0] * N)
    x = resp[xr]
    y = resp[yr]
    tmp = drmres.tmp
    tmp.varx += F * abs(x)**2
    tmp.vary += F * abs(y)**2
    tmp.covar += F * np.real(x*np.conj(y))
    tmp.xresp[i] = x
    tmp.yresp[i] = y

    if i == N-1:
        varx = np.trapz(tmp.varx, freq)
        vary = np.trapz(tmp.vary, freq)
        covar = np.trapz(tmp.covar, freq)
        term = (vary-varx) / (2*covar)
        th = np.arctan(term - np.sqrt(term**2 + 1))

        # check for covar == 0:
        pv = np.nonzero(abs(covar) < 1e-12)[0]
        if pv.size > 0:
            # assume zero degrees and check for 90:
            th[pv] = 0
            th[pv[vary[pv] > varx[pv]]] = np.pi/2
        c = np.cos(th)
        s = np.sin(th)

        # where the 2nd derivative is > 0, we got the angle wrong:
        second = 2*(vary - varx)*(c*c - s*s) - 8*covar*s*c
        pv = np.nonzero(second > 0)[0]
        if pv.size > 0:
            th[pv] = np.arctan(term[pv] + np.sqrt(term[pv]**2 + 1))
            c[pv] = np.cos(th[pv])
            s[pv] = np.sin(th[pv])

        # now have all sines/cosines, compute consistent results:
        rss_resp = 0.0
        for j in range(N):
            respxy = (c[:, None] * tmp.xresp[j] +
                      s[:, None] * tmp.yresp[j])
            rss_resp += forcepsd[j] * abs(respxy)**2

        if rr is not None:
            drmres._psd[case][rr] = rss_resp
        else:
            drmres._psd[case] = rss_resp

        # delete 'extra' info:
        del drmres.tmp


#def PSD_nonconsistent_rss(resp,xr,yr,freq,forcepsd,drmres,case,i,ar,rssfactor,afactor):
#    """For PSD analysis, compute NON - 'time-correlated' RSS + addon:  rssfactor.*RSS(xr,yr) + afactor.*ar
#
#This routine only exists as a easy way to do non-consistent RSS'ing using the same
#call format as for the consistent RSS'ing routine.  Otherwise, works just like PSD_consistent_rss().
#"""
#    % drmres is a record:  drmres = results[drm]
#    % i is psd force number
#    F = forcepsd(i,:)
#    drmres.psd[case] += F .* abs(resp)**2  % for the normal, non-rss rows ... and creating space
#    N = size(forcepsd,1)
#    if !locate(names(drmres),'xresp'):
#        drmres.xresp = 0
#        drmres.yresp = 0
#        if ar:  drmres.aresp = 0
#    drmres.xresp += F .* abs(resp(xr,:))**2
#    drmres.yresp += F .* abs(resp(yr,:))**2
#    if ar:  drmres.aresp += F .* abs(resp(ar,:))**2
#
#    if i == N:
#        if ar:  resp = rssfactor .* sqrt(drmres.xresp**2 + drmres.yresp**2) + afactor.*drmres.aresp
#        else:   resp = sqrt(drmres.xresp**2 + drmres.yresp**2)
#        % to delete 'extra' info from the results record:
#        del(drmres,['xresp','yresp','aresp'])
#        return resp


#def expand_results(results,bynum):
#    """Expands results so more cases can be added efficiently
#
#Parameters
#----------
#  results = results dictionary as described in form_extreme; results[key] must be the
#            data recovery items (eg, results['SC_atm'])
#  bynum   = number of cases to add room for in results
#
#Expands:  results[drm].mx, .mn, mxt, .mintime, .cases, .hist, .PSD, .FRF, .srs members.
#"""
#    DRNames = keys(results)
#    for name in DRNames:
#        r = results[name]
#        if !locate(names(r),'mx'):
#            prterr.error('expand_results:  results inputs does not have member .mx\n')
#        old = size(r.mx,2)
#        n = old+bynum
#        r.mx(1,n)=0
#        r.mn(1,n)=0
#        r.maxtime(1,n)=0
#        r.mintime(1,n)=0
#        r.cases(n) = []
#        if locate(names(r),'hist'):  r.hist(1,1,n) = 0
#        if locate(names(r),'PSD'):  r.PSD(1,1,n) = 0
#        if locate(names(r),'FRF'):  r.FRF(1,1,n) = 0
#        if locate(names(r),'srs'):
#            for q in keys(r.srs):  r.srs.srs[q](1,1,n) = 0
#    return [old+1,n]


def _get_rpt_headers(res=None, desc=None, uf_reds=None,
                     units=None, misc=''):
    if res is not None:
        desc = res.drminfo.desc
        uf_reds = res.drminfo.uf_reds
        units = res.drminfo.units
    descline = 'Description: {}\n'.format(desc)
    if uf_reds is None:
        unceline = 'Uncertainty: Not specified\n'
    else:
        unceline = ('Uncertainty: [Rigid, Elastic, Dynamic, Static] '
                    '= [{}, {}, {}, {}]\n'
                    .format(*uf_reds))
    unitline = 'Units:       {}\n'.format(units)
    currdate = datetime.date.today().strftime('%d-%b-%Y')
    dateline = 'Date:        {}\n'.format(currdate)
    return descline + unceline + unitline + misc + dateline


def rptext1(res, filename,
            title='M A X / M I N  S U M M A R Y',
            doabsmax=False, numform='{:13.5e}', perpage=-1):
    """
    Writes .ext file for max/min results.

    Parameters
    ----------
    res : SimpleNamespace
        Results data structure with attributes `.ext`, `.cases`, etc
        (see example in :class:`DR_Results`)
    filename : string, file handle or 1
        If a string, it is a filename that gets created. `filename`
        can also be a file handle or 1 to write to standard output
        (normally, the screen).
    title : string; optional
        Title for report
    doabsmax : bool; optional
        If True, compare only absolute maximums.
    numform : string; optional
        Format of the max & min numbers.
    perpage : integer; optional
        The number of lines to write perpage. If < 0, there is no
        limit (one page).

    Notes
    -----
    The output files contain the maximums, minimums and cases as
    applicable.
    """
    headers = []
    formats = []
    printargs = []
    widths = []
    seps = []
    justs = []

    def _add_column(hdr, frm, arg, width, sep, just):
        headers.append(hdr)
        formats.append(frm)
        printargs.append(arg)
        widths.append(width)
        seps.append(sep)
        justs.append(just)

    # row
    nrows = res.ext.shape[0]
    _add_column('Row', '{:7d}', np.arange(1, nrows+1), 7, 0, 'c')

    # labels
    labels = res.drminfo.labels
    if labels is not None:
        w = max(11, len(max(labels, key=len)))
        frm = '{{:{:d}}}'.format(w)
        _add_column('Description', frm, labels, w, 2, 'l')

    # max, time, case, min, time, case
    domain = getattr(res, 'domain', None)
    if domain:
        domain = domain.capitalize()
    else:
        domain = 'X-Value'
    if res.maxcase is not None:
        casewidth = max(4, len(max(res.maxcase, key=len)),
                        len(max(res.mincase, key=len)))
        casefrm = '{{:{:d}}}'.format(casewidth)

    one_col = res.ext.ndim == 1 or res.ext.shape[1] == 1
    for col, hdr, case in zip((0, 1),
                              ('Maximum', 'Minimum'),
                              (res.maxcase, res.mincase)):
        # maximum or minimum
        w = len(numform.format(np.pi))
        if doabsmax and not one_col:
            mx = res.ext.max(axis=1)
        else:
            mx = res.ext if one_col else res.ext[:, col]
        _add_column(hdr, numform, mx, w, 4, 'c')

        # time
        if res.exttime is not None:
            t = res.exttime if one_col else res.exttime[:, col]
            _add_column(domain, '{:8.3f}', t, 8, 2, 'c')

        # case
        if case is not None:
            _add_column('Case', casefrm, case, casewidth, 2, 'l')

        if doabsmax or res.ext.ndim == 1 or res.ext.shape[1] == 1:
            break

    hu, frm = writer.formheader(headers, widths, formats,
                                seps, justs)

    # format page header:
    header = title + '\n\n' + _get_rpt_headers(res) + '\n' + hu

    # write results
    def _wtext(f, header, frm, printargs, perpage):
        if perpage < 1:
            # one additional in case nrows is zero
            perpage = nrows + 1
        pages = (nrows + perpage - 1) // perpage
        for p in range(pages):
            if p > 0:
                f.write(chr(12))
            f.write(header)
            b = p * perpage
            e = b + perpage
            writer.vecwrite(f, frm, *printargs, so=slice(b, e))

    ytools.wtfile(filename, _wtext, header, frm,
                  printargs, perpage)


def _get_numform(mxmn1, excel=False):
    # excel logic is different than text:
    # - it avoids scientific notation since the actual values are
    #   there ... the user can just change the format
    pv = (mxmn1 != 0.0) & np.isfinite(mxmn1)
    if not np.any(pv):
        return '{:13.0f}' if not excel else '#,##0.'
    pmx = int(np.floor(np.log10(abs(mxmn1[pv]).max())))
    if excel:
        numform = '#,##0.' + '0'*(5 - pmx)
    else:
        pmn = int(np.floor(np.log10(abs(mxmn1[pv]).min())))
        if pmx-pmn < 6 and pmn > -3:
            if pmn < 5:
                numform = '{{:13.{}f}}'.format(5-pmn)
            else:
                numform = '{:13.0f}'
        else:
            numform = '{:13.6e}'
    return numform


def rpttab1(res, filename, title, count_filter=1e-6, name=None):
    """
    Write results tables with bin count information.

    Parameters
    ----------
    res : SimpleNamespace
        Results data structure with attributes `.ext`, `.cases`, etc
        (see example in :class:`DR_Results`)
    filename : string, file handle or 1
        If a string, it is a filename that gets created. `filename`
        can also be a file handle or 1 to write to standard output
        (normally, the screen). If the filename ends with '.xlsx', a
        Microsoft Excel file is written.
    title : string
        Title for report
    count_filter : scalar; optional
        Filter to use for the bin count; only numbers larger (in
        the absolute value sense) than the filter are counted
    name : string or None; optional
        For '.xlsx' files, this string is used for sheet naming. If
        None and writing an '.xlsx' file, a ValueError is raised.

    Notes
    -----
    The output files contain the maximums, minimums, abs-max
    tables. The extrema value is also included along with the case
    that produced it.

    After those three tables, a table of bin counts is printed
    showing the number of extrema values produced by each event.
    """
    def _event_count(mat, j, filter_=1e-6):
        n = mat.shape[1]
        count = np.zeros(n, np.int64)
        peaks = mat[np.arange(mat.shape[0]), j]
        pv = (abs(peaks) > filter_).nonzero()[0]
        if pv.size > 0:
            j = j[pv]
            for k in range(n):
                count[k] = np.sum(j == k)
        return count, count/len(pv) * 100

    def _add_zero_case(mat, cases):
        pv = (mat == 0).all(axis=1).nonzero()[0]
        if cases is None:
            new_cases = ['Statistical']*mat.shape[0]
        else:
            new_cases = copy.copy(cases)
        for i in pv:
            new_cases[i] = 'zero row'
        return new_cases

    def _get_absmax(res):
        # form abs-max matrix keeping sign:
        amx = res.mx.copy()
        pv = abs(res.mn) > abs(amx)
        amx[pv] = res.mn[pv]
        amxcase = res.maxcase
        if amxcase is not None:
            pv = abs(res.ext[:, 1]) > abs(res.ext[:, 0])
            pv = pv.nonzero()[0]
            for j in pv:
                amxcase[j] = res.mincase[j]
        aext = res.ext[:, 0].copy()
        amn = res.ext[:, 1]
        pv = abs(amn) > abs(aext)
        aext[pv] = amn[pv]
        return amx, amxcase, aext

    def _add_max_plus_min(ec):
        count = ec['Maximum'][0] + ec['Minimum'][0]
        sumcount = count.sum()
        if sumcount > 0:
            countperc = 100*count/sumcount
        else:
            countperc = count
        ec['Max+Min'] = {0: count, 1: countperc}

    def _rowlbls_table(lbl, ec, j):
        rowlabels = ['Maxima ' + lbl,
                     'Minima ' + lbl,
                     'Max+Min ' + lbl,
                     'Abs-Max ' + lbl]
        table = np.vstack((ec['Maximum'][j],
                           ec['Minimum'][j],
                           ec['Max+Min'][j],
                           ec['Abs-Max'][j]))
        return rowlabels, table

    def _wttab_eventcount(f, header, res, ec, count_filter):
        # extrema count
        f.write('Extrema Count\nFilter: {}\n\n'
                .format(count_filter))
        widths = [desclen, *([caselen]*n)]
        headers = ['Description', *res.cases]
        for j, frm, lbl in zip((0, 1),
                               ('{{:{}d}}'.format(caselen),
                                '{{:{}.1f}}'.format(caselen)),
                               ('Count', 'Percent')):
            formats = [descfrm, *([frm]*n)]
            hu_, frm_ = writer.formheader(
                headers, widths, formats, sep=[7, 1],
                just='c', ulchar='=')
            f.write(hu_)
            rowlabels, table = _rowlbls_table(lbl, ec, j)
            writer.vecwrite(f, frm_, rowlabels, table)
            if j == 0:
                f.write('\n')

    def _wtxlsx_eventcount(workbook, header, bold, hform, res, ec,
                           name, count_filter):
        # extrema count
        sheet = '{} Count'.format(name)
        worksheet = workbook.add_worksheet(sheet)

        title = header.split('\n')
        for i, ln in enumerate(title):
            worksheet.write(i, 0, ln, bold)

        n = len(title)
        worksheet.write(n, 0, 'Extrema Count', bold)
        worksheet.write(
            n+1, 0, 'Filter: {}'.format(count_filter), bold)
        worksheet.write(n+1, 1, count_filter)
        n += 2
        ncases = len(res.cases)
        headers = ['Description', *res.cases]
        chart_positions = ((25, 0), (25, 8), (55, 0), (55, 8))
        data_labels = {'category': True,
                       'percentage': True,
                       'separator': '\n'}
        chart_opts = {'x_offset': 25, 'y_offset': 10}
        chart_size = {'width': 600, 'height': 500}

        for j, frm, lbl in zip((0, 1),
                               ('#,##', '#,##0.0'),
                               ('Count', 'Percent')):
            number = workbook.add_format({'num_format': frm})
            worksheet.write_row(n, 0, headers, hform)
            rowlabels, table = _rowlbls_table(lbl, ec, j)

            # write table and pie charts:
            n += 1
            for i, (rowlbl, chpos) in enumerate(
                    zip(rowlabels, chart_positions)):
                worksheet.write(n+i, 0, rowlbl)
                worksheet.write_row(n+i, 1, table[i], number)
                if j == 1:
                    chart = workbook.add_chart({'type': 'pie'})
                    chart.add_series(
                        {'name': rowlbl,
                         'categories': [sheet, n-1, 1, n-1, ncases],
                         'values': [sheet, n+i, 1, n+i, ncases],
                         'data_labels': data_labels})
                    chart.set_title({'name': rowlbl})
                    chart.set_size(chart_size)
                    worksheet.insert_chart(*chpos, chart, chart_opts)
            n += len(rowlabels) + 1

        # adjust column widths
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, len(headers)-1, 14)

    def _wttab(f, header, hu, frm, res, loop_vars):
        f.write(header)
        ec = {}   # event counter
        for lbl, vals, case_pv, ext, extcases in loop_vars:
            f.write('{} Responses\n\n'.format(lbl))
            hu_ = hu.replace('Maximum', lbl)
            f.write(hu_)
            ec[lbl] = _event_count(vals, case_pv, count_filter)
            extcases = _add_zero_case(vals, extcases)
            writer.vecwrite(f, frm, rows, res.drminfo.labels,
                            vals, ext, extcases)
            f.write('\n\n')
        _add_max_plus_min(ec)
        _wttab_eventcount(f, header, res, ec, count_filter)

    def _wtxlsx(workbook, header, headers, res, loop_vars, name):
        bold = workbook.add_format({'bold': True})
        hform = workbook.add_format({'bold': True,
                                     'align': 'center'})
        frm = _get_numform(res.ext, excel=True)
        number = workbook.add_format({'num_format': frm})
        ec = {}
        for lbl, vals, case_pv, ext, extcases in loop_vars:
            worksheet = workbook.add_worksheet(
                '{} {}'.format(name, lbl))
            title = header.split('\n')
            for i, ln in enumerate(title):
                worksheet.write(i, 0, ln, bold)
            h = [i.replace('Maximum', lbl) for i in headers]
            worksheet.write_row(len(title), 0, h, hform)

            ec[lbl] = _event_count(vals, case_pv, count_filter)
            extcases = _add_zero_case(vals, extcases)

            # write table:
            n = len(title) + 1
            labels = res.drminfo.labels
            ncases = vals.shape[1]
            for i in range(len(rows)):
                worksheet.write(n+i, 0, rows[i])
                worksheet.write(n+i, 1, labels[i])
                worksheet.write_row(
                    n+i, 2, vals[i], number)
                worksheet.write(
                    n+i, 2+ncases, ext[i], number)
                worksheet.write(
                    n+i, 3+ncases, extcases[i])

            # adjust column widths and freeze row and col panes
            worksheet.set_column(1, 1, 20)  # description
            worksheet.set_column(2, 3+ncases, 14)
            worksheet.freeze_panes(n, 2)

        _add_max_plus_min(ec)
        _wtxlsx_eventcount(workbook, header, bold, hform, res, ec,
                           name, count_filter)
    rows = np.arange(res.mx.shape[0]) + 1
    headers = ['Row', 'Description', *res.cases,
               'Maximum', 'Case']
    header = title + '\n\n' + _get_rpt_headers(res) + '\n'

    amx, amxcase, aext = _get_absmax(res)
    # order of tables: max, min, abs-max with sign:
    loop_vars = (('Maximum', res.mx, np.nanargmax(res.mx, axis=1),
                  res.ext[:, 0], res.maxcase),
                 ('Minimum', res.mn, np.nanargmin(res.mn, axis=1),
                  res.ext[:, 1], res.mincase),
                 ('Abs-Max', amx, np.nanargmax(abs(amx), axis=1),
                  aext, amxcase))

    if isinstance(filename, xlsxwriter.workbook.Workbook):
        excel = 'old'
    elif filename.endswith('.xlsx'):
        excel = 'new'
    else:
        excel = ''

    if not excel:
        desclen = max(15, len(max(res.drminfo.labels, key=len)))
        caselen = max(13, len(max(res.cases, key=len)))
        n = len(res.cases)
        widths = [6, desclen, *([caselen]*n), caselen, caselen]
        descfrm = '{{:{:d}}}'.format(desclen)
        numform = '{{:{}.6e}}'.format(caselen)
        formats = ['{:6d}', descfrm, *([numform]*n),
                   numform, '{}']
        hu, frm = writer.formheader(
            headers, widths, formats, sep=[0, 1],
            just='c', ulchar='=')
        ytools.wtfile(filename, _wttab, header, hu, frm,
                      res, loop_vars)
    else:
        if not name:
            raise ValueError('`name` must be input when writing'
                             ' ".xlsx" files')
        if excel == 'new':
            opts = {'nan_inf_to_errors': True}
            with xlsxwriter.Workbook(filename, opts) as workbook:
                _wtxlsx(workbook, header, headers, res,
                        loop_vars, name)
        else:
            _wtxlsx(filename, header, headers, res, loop_vars, name)


def rptpct1(mxmn1, mxmn2, filename, *,
            title='PERCENT DIFFERENCE REPORT',
            names=('Self', 'Reference'),
            desc=None, filterval=None, labels=None,
            units=None, napv=None, uf_reds=None,
            use_range=True, numform=None,
            prtbad=None, prtbadh=None, prtbadl=None,
            flagbad=None, flagbadh=None, flagbadl=None,
            dohistogram=True, histogram_inc=1.0,
            domagpct=True, doabsmax=False, shortabsmax=False,
            roundvals=-1.0, rowhdr='Row', deschdr='Description',
            maxhdr='Maximum', minhdr='Minimum', absmhdr='Abs-Max',
            perpage=-1):
    """
    Write a percent difference report between 2 sets of max/min data.

    Parameters
    ----------
    mxmn1 : 2d ndarray or SimpleNamespace
        If 2-column ndarray, its columns are: [max, min]. If
        SimpleNamespace, it must be as defined in :class:`DR_Results`
        and have these members::

            .ext = [max, min]
            .drminfo = SimpleNamespace which has (at least):
               .desc      = one line description of category
               .filterval = the filter value; (see `filterval`
                            description below)
               .labels    = a list of descriptions; one per row
               .units     = string with units
               .napv      = these rows will get 'n/a' for % diff
               .uf_reds   = uncertainty factors

        Note that the inputs `desc`, `labels`, etc, override the
        values above.

    mxmn2 : 2d ndarray
        2-column matrix of reference max, min data:  [max, min]
    filename : string, file handle or 1
        If a string, it is a filename that gets created. `filename`
        can also be a file handle or 1 to write to standard output
        (normally, the screen).
    title : string; must be named; optional
        Title for the report
    names : list/tuple; must be named; optional
        Two (short) strings identifying the two sets of data
    desc : string or None; must be named; optional
        A one line description of the table. Overrides
        `mxmn1.drminfo.desc`. If neither are input,
        'No description provided' is used.
    filterval : scalar or 1d ndarray or None; must be named; optional
        Numbers <= than `filterval` will get a 'n/a' % diff. If
        vector, length must match number of rows in `mxmn1` and
        `mxmn2` data. Overrides `mxmn1.drminfo.filterval`. If neither
        are input, `filterval` is set to 1.e-6.
    labels : list or None; must be named; optional
        A list of strings briefly describing each row. Overrides
        `mxmn1.drminfo.labels`. If neither are input,
        ``['Row 1','Row 2',...]`` is used.
    units : string or None; must be named; optional
        Specifies the units. Overrides `mxmn1.drminfo.units`. If
        neither are input, 'Not specified' is used.
    napv : 1d array or None; must be named; optional
        0-offset index vector specifying which rows to ignore (they
        get the 'n/a' % diff). Overrides `mxmn1.drminfo.units`. If
        neither are input, no rows are ignored (though `filterval` is
        still used).
    uf_reds : 1d array or None; must be named; optional
        Uncertainty factors: [rigid, elastic, dynamic, static].
        Overrides `mxmn1.drminfo.uf_reds`. If neither is input,
        'Not specified' is used.
    use_range : bool
        If True, the denominator of the % diff calc for both the max
        & min for each row is the absolute maximum of the reference
        max & min for that row. If False, the denominator is the
        applicable reference max or min. A quick example shows why
        ``use_range=True`` might be useful::

            If [max1, min1] = [12345, -10] and
               [max2, min2] = [12300,  50]
            Then:
               % diff = [0.37%,   0.49%]  if use_range is True
               % diff = [0.37%, 120.00%]  if use_range is False

        Note that the sign of the % diff is defined such that a
        positive % diff means an exceedance: where ``max1 > max2`` or
        ``min1 < min2``.

        `use_range` is ignored if `doabsmax` is True.

    numform : string or None; must be named; optional
        Format of the max & min numbers. If None, it is set internally
        to be 13 chars wide and depends on the range of numbers to
        print:

           - if range is "small", numform='{:13.xf}' where "x" ranges
             from 0 to 7
           - if range is "large", numform='{:13.6e}'

    prtbad : scalar or None; must be named; optional
        Only print rows where ``abs(%diff) > prtbad``. For example, to
        print rows off by more than 5%, use ``prtbad=5``. `prtbad`
        takes precedence over `prtbadh` and `prtbadl`.
    prtbadh : scalar or None; must be named; optional
        Only print rows where ``%diff > prtbadh``. Handy for showing
        just the exceedances. `prtbadh` takes precedence over
        `prtbadl`.
    prtbadl : scalar or None; must be named; optional
        Only print rows where ``%diff < prtbadl``. Handy for showing
        where reference rows are higher.
    flagbad : scalar or None; must be named; optional
        Flag % diffs where ``abs(%diff) > flagbad``. Works similar to
        `prtbad`. The flag is an asterisk (*).
    flagbadh : scalar or None; must be named; optional
        Flag % diffs where ``%diff > flagbadh``. Works similar to
        `prtbadh`. Handy for flagging exceedances. `flagbadh` takes
        precedence over `flagbadl`.
    flagbadl : scalar or None; must be named; optional
        Flag % diffs where ``%diff < flagbadl``. Works similar to
        `prtbadl`.
    dohistogram : bool; must be named; optional
        If True, plot the histograms. Plots will be written to
        "`filename`.histogram.png".
    histogram_inc : scalar; must be named; optional
        The histogram increment; defaults to 1.0 (for 1%).
    domagpct : bool; must be named; optional
        If True, plot the % diffs versus magnitude via :func:`magpct`.
        Plots will be written to "`filename`.magpct.png". No filters
        are applied but `napv` is used.
    doabsmax : bool; must be named; optional
        If True, compare only absolute maximums.
    shortabsmax : bool; must be named; optional
        If True, set ``doabsmax=True`` and do not print the max1 and
        min1 columns.
    roundvals : integer; must be named; optional
        Round max & min numbers at specified decimal. If negative, no
        rounding.
    rowhdr : string; must be named; optional
        Header for row number column
    deschdr : string; must be named; optional
        Header for description column
    maxhdr : string; must be named; optional
        Header for the column 1 data
    minhdr : string; must be named; optional
        Header for the column 2 data
    absmhdr : string; must be named; optional
        Header for abs-max column
    perpage : integer; must be named; optional
        The number of lines to write perpage. If < 1, there is no
        limit (one page).

    Returns
    -------
    pdiff_info : dict
        Dictionary with 'amx' (abs-max), 'mx' (max), and 'mn' keys::

            <class 'dict'>[n=3]
                'amx': <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 33 elems: (11, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['  -2.46', '  -1.50', ...
                'mn' : <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 33 elems: (11, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['   1.55', '   1.53', ...
                'mx' : <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 27 elems: (9, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['  -2.46', '  -1.50', ...

        Where::

            'hsto'  : output of :func:`histogram`: [center, count, %]
            'mag'   : inputs to :func:`magpct`
            'pct'   : percent differences
            'prtpv' : rows to print partition vector
            'spct'  : string version of 'pct'

    Examples
    --------
    Compare some gaussian random data and show histogram and magpct
    plots with minimal inputs:

    >>> import numpy as np
    >>> from numpy.random import randn
    >>> from pyyeti import cla
    >>> ext1 = [[120.0, -8.0],
    ...         [8.0, -120.0]]
    >>> ext2 = [[115.0, -5.0],
    ...         [10.0, -125.0]]

    Run :func:`rptpct1` multiple times to get a more complete picture
    of all the output (the table is very wide). Also, the plots will
    be turned off for this example.

    First, the header:

    >>> opts = {'domagpct': False, 'dohistogram': False}
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    <BLANKLINE>
    Description: No description provided
    Uncertainty: Not specified
    Units:       Not specified
    Filter:      1e-06
    Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference...
                 Sign set such that positive % differences indicate...
    Date:        ...
    ...

    Then, the max/min/absmax percent difference table in 3 calls:

    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                                 Self        Reference             ...
      Row    Description       Maximum        Maximum      % Diff  ...
    -------  -----------    -------------  -------------  -------  ...
          1  Row      1         120.00000      115.00000     4.35  ...
          2  Row      2           8.00000       10.00000    -1.60  ...
    ...
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                         ...     Self        Reference             ...
      Row    Description ...   Minimum        Minimum      % Diff  ...
    -------  ----------- ...-------------  -------------  -------  ...
          1  Row      1  ...     -8.00000       -5.00000     2.61  ...
          2  Row      2  ...   -120.00000     -125.00000    -4.00  ...
    ...
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                         ...     Self        Reference
      Row    Description ...   Abs-Max        Abs-Max      % Diff
    -------  ----------- ...-------------  -------------  -------
          1  Row      1  ...    120.00000      115.00000     4.35
          2  Row      2  ...    120.00000      125.00000    -4.00
    ...

    Finally, the histogram summaries:

    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
        No description provided - Maximum Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -2.00          1     50.00
             4.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        50.0% of values are within 2%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-2.00, 4.00,...
    <BLANKLINE>
    <BLANKLINE>
        No description provided - Minimum Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -4.00          1     50.00
             3.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-4.00, 3.00,...
    <BLANKLINE>
    <BLANKLINE>
        No description provided - Abs-Max Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -4.00          1     50.00
             4.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-4.00, 4.00,...
    """
    def _get_filtline(filterval):
        if isinstance(filterval, np.ndarray):
            if filterval.size == 1:
                filterval = filterval.ravel()[0]
        if isinstance(filterval, np.ndarray):
            filtline = 'Filter:      <defined row-by-row>\n'
        else:
            filtline = 'Filter:      {}\n'.format(filterval)
        return filtline

    def _get_noteline(use_range, names, prtbads, flagbads):
        noteline = 'Notes:       '
        tab = '             '
        if not use_range:
            noteline += ('% Diff = +/- abs(({0}-{1})/'
                         '{1})*100\n'.format(*names))
        else:
            noteline += ('% Diff = +/- abs({0}-{1})/'
                         'max(abs({1}(max,min)))*100\n'
                         .format(*names))

        noteline += (tab + 'Sign set such that positive % '
                     'differences indicate exceedances\n')
        prtbad, prtbadh, prtbadl = prtbads
        flagbad, flagbadh, flagbadl = flagbads
        if (prtbad is not None or prtbadh is not None or
                prtbadl is not None):
            if prtbad is not None:
                prtbad = abs(prtbad)
                noteline += (tab + 'Printing rows where abs(% Diff)'
                             ' > {}%\n'.format(prtbad))
            elif prtbadh is not None:
                noteline += (tab + 'Printing rows where % Diff > '
                             '{}%\n'.format(prtbadh))
            else:
                noteline += (tab + 'Printing rows where % Diff < '
                             '{}%\n'.format(prtbadl))

        if (flagbad is not None or flagbadh is not None or
                flagbadl is not None):
            if flagbad is not None:
                flagbad = abs(flagbad)
                noteline += (tab + 'Flagging (*) rows where '
                             'abs(% Diff) > {}%\n'.format(flagbad))
            elif flagbadh is not None:
                noteline += (tab + 'Flagging (*) rows where '
                             '% Diff > {}%\n'.format(flagbadh))
            else:
                noteline += (tab + 'Flagging (*) rows where '
                             '% Diff < {}%\n'.format(flagbadl))
        return noteline

    def _get_badpv(pct, pv, bad, badh, badl, defaultpv=False):
        if (bad is not None or badh is not None or
                badl is not None):
            badpv = pv.copy()
            if bad is not None:
                badpv &= abs(pct) > bad
            elif badh is not None:
                badpv &= pct > badh
            else:
                badpv &= pct < badl
        else:
            badpv = np.empty(len(pct), bool)
            badpv[:] = defaultpv
        return badpv

    def _get_pct_diff(a, b, filt, pv, mxmn_b=None, ismax=True,
                      flagbads=None):
        # either can pass filter to be kept:
        pv &= (abs(a) > filt) | (abs(b) > filt)

        if mxmn_b is not None:
            denom = abs(mxmn_b).max(axis=1)
        else:
            denom = abs(b)

        # put 1's in for filtered values ... this is temporary
        a = a.copy()
        b = b.copy()
        a[~pv] = 1.0
        b[~pv] = 1.0

        z = denom == 0.0
        denom[z] = 1.0
        pct = 100*abs(a-b)/denom
        pct[z] = 100.0    # np.inf

        # make less extreme values negative
        neg = a < b if ismax else a > b
        pct[neg] *= -1.0

        # put nan's in for the filtered or n/a rows:
        pct[~pv] = np.nan

        # make 7 char version:
        spct = ['{:7.2f}'.format(p) for p in pct]
        badpv = _get_badpv(pct, pv, *flagbads, False)
        for j in badpv.nonzero()[0]:
            spct[j] += '*'
        for j in (~pv).nonzero()[0]:
            spct[j] = nastring

        return pct, spct

    def _get_histogram_str(desc, hdr, pctcount):
        s = [('\n\n    {} - {} Comparison Histogram\n\n'
              .format(desc, hdr)),
             ('      % Diff      Count    Percent\n'
              '     --------   --------   -------\n')]
        with StringIO() as f:
            writer.vecwrite(f, '     {:8.2f}   {:8.0f}   {:7.2f}\n',
                            pctcount)
            s.append(f.getvalue())
            s.append('\n')

        last = -1.0
        for pdiff in [1, 2, 5, 10, 15, 20, 25, 50, 100, 500]:
            pvdiff = abs(pctcount[:, 0]) <= pdiff
            num = pctcount[pvdiff, 2].sum()
            if num > last:
                s.append('    {:.1f}% of values are within {:d}%\n'
                         .format(num, pdiff))
            if np.round(num*10) == 1000:
                break
            last = num

        n = pctcount[:, 1].sum()
        A = pctcount[:, 0] * pctcount[:, 1]
        meanval = A.sum()/n
        if n == 1:
            stdval = 0
        else:
            a = pctcount[:, 1] * (pctcount[:, 0] - meanval)**2
            stdval = np.sqrt(a.sum()/(n-1))
        s.append('\n    % Diff Statistics: [Min, Max, Mean, StdDev]'
                 ' = [{:.2f}, {:.2f}, {:.4f}, {:.4f}]\n'
                 .format(pctcount[:, 0].min(),
                         pctcount[:, 0].max(),
                         meanval, stdval))
        return ''.join(s)

    def _proc_pct(ext1, ext2, filterval, *, names, mxmn1,
                  not_napv, mxmn_b, ismax, histogram_inc,
                  prtbads, flagbads, numform, valhdr, maxhdr,
                  minhdr, absmhdr, pdhdr):
        pv = not_napv.copy()
        mag = ext1[not_napv], ext2[not_napv]  # good here?
        pct, spct = _get_pct_diff(ext1, ext2, filterval, pv,
                                  mxmn_b=mxmn_b, ismax=ismax,
                                  flagbads=flagbads)
        pct_ret = pct[pv]
        hsto = ytools.histogram(pct_ret, histogram_inc)

        # for trimming down if prtbad set:
        prtpv = _get_badpv(pct, pv, *prtbads, True)
        pctlen = max(len(pdhdr),
                     len(max(spct, key=len)))
        sformatpd = '{{:{}}}'.format(pctlen)

        # for writer.formheader:
        numlen = max(13,
                     len(max(names, key=len)),
                     len(numform.format(np.pi)))
        if not doabsmax:
            headers1.extend([*names, ''])
            headers2.extend([valhdr, valhdr, pdhdr])
            formats.extend([numform, numform, sformatpd])
            printargs.extend([ext1, ext2, spct])
            widths.extend([numlen, numlen, pctlen])
            seps.extend([4, 2, 2])
            justs.extend(['c', 'c', 'c'])
        elif shortabsmax:
            headers1.extend([*names, ''])
            headers2.extend([absmhdr, absmhdr, pdhdr])
            formats.extend([numform, numform, sformatpd])
            printargs.extend([mx1, mx2, spct])
            widths.extend([numlen, numlen, pctlen])
            seps.extend([4, 2, 2])
            justs.extend(['c', 'c', 'c'])
        else:
            headers1.extend([names[0], names[0],
                             names[0], names[1], ''])
            headers2.extend([maxhdr, minhdr, absmhdr,
                             absmhdr, pdhdr])
            formats.extend([numform, numform, numform,
                            numform, sformatpd])
            printargs.extend([mxmn1[:, 0], mxmn1[:, 1],
                              mx1, mx2, spct])
            widths.extend([numlen, numlen, numlen,
                           numlen, pctlen])
            seps.extend([4, 2, 2, 2, 2])
            justs.extend(['c', 'c', 'c', 'c', 'c'])
        return dict(pct=pct_ret, spct=spct, hsto=hsto,
                    prtpv=prtpv, mag=mag)

    def _plot_magpct(pctinfo, names, desc, doabsmax, filename):
        ptitle = '{} - {{}} Comparison vs Magnitude'.format(desc)
        xl = '{} Magnitude'.format(names[1])
        yl = '% Diff of {} vs {}'.format(*names)
        figsize = [8.5, 11.0]
        if doabsmax:
            figsize[1] /= 3.0
        plt.figure('Magpct - '+desc, figsize=figsize)
        plt.clf()
        for lbl, hdr, sp, ismax in (('mx', maxhdr, 311, True),
                                    ('mn', minhdr, 312, False),
                                    ('amx', absmhdr, 313, True)):
            if 'mx' in pctinfo:
                plt.subplot(sp)
            if lbl in pctinfo:
                magpct(pctinfo[lbl]['mag'][0],
                       pctinfo[lbl]['mag'][1],
                       pctinfo['amx']['mag'][1],
                       ismax=ismax)
                plt.title(ptitle.format(hdr))
                plt.xlabel(xl)
                plt.ylabel(yl)
            plt.grid(True)
        plt.tight_layout(pad=3)
        if isinstance(filename, str):
            plt.savefig(filename+'.magpct.png')

    def _plot_histogram(pctinfo, names, desc, doabsmax, filename):
        ptitle = '{} - {{}} Comparison Histogram'.format(desc)
        xl = '% Diff of {} vs {}'.format(*names)
        yl = 'Percent Occurrence (%)'
        figsize = [8.5, 11.0]
        if doabsmax:
            figsize[1] /= 3.0
        plt.figure('Histogram - '+desc, figsize=figsize)
        plt.clf()
        for lbl, hdr, sp in (('mx', maxhdr, 311),
                             ('mn', minhdr, 312),
                             ('amx', absmhdr, 313)):
            if 'mx' in pctinfo:
                plt.subplot(sp)
            if lbl in pctinfo:
                width = histogram_inc
                x = pctinfo[lbl]['hsto'][:, 0]
                y = pctinfo[lbl]['hsto'][:, 2]
                colors = ['b']*len(x)
                ax = abs(x)
                pv1 = ((ax > 5) & (ax <= 10)).nonzero()[0]
                pv2 = (ax > 10).nonzero()[0]
                for pv, c in ((pv1, 'm'),
                              (pv2, 'r')):
                    for i in pv:
                        colors[i] = c
                plt.bar(x-width/2, y,
                        width=width, color=colors)
                plt.title(ptitle.format(hdr))
                plt.xlabel(xl)
                plt.ylabel(yl)
                x = abs(max(plt.xlim(), key=abs))
                if x < 5:
                    plt.xlim(-5, 5)
            plt.grid(True)
        plt.tight_layout(pad=3)
        if isinstance(filename, str):
            plt.savefig(filename+'.histogram.png')

    # check mxmn1:
    if isinstance(mxmn1, SimpleNamespace):
        if desc is None:
            desc = mxmn1.drminfo.desc
        if filterval is None:
            filterval = mxmn1.drminfo.filterval
        if labels is None:
            labels = mxmn1.drminfo.labels
        if units is None:
            units = mxmn1.drminfo.units
        if napv is None:
            napv = mxmn1.drminfo.napv
        if uf_reds is None:
            uf_reds = mxmn1.drminfo.uf_reds
        mxmn1 = mxmn1.ext

    mxmn1, mxmn2 = np.atleast_2d(mxmn1, mxmn2)
    R = mxmn1.shape[0]
    if R != mxmn2.shape[0]:
        raise ValueError('`mxmn1` and `mxmn2` have a different'
                         ' number of rows')
    if desc is None:
        desc = 'No description provided'
    if filterval is None:
        filterval = 1.e-6
    if labels is None:
        labels = ['Row {:6d}'.format(i+1)
                  for i in range(R)]
    if units is None:
        units = 'Not specified'
    if numform is None:
        numform = _get_numform(mxmn1)

    pdhdr = '% Diff'
    nastring = 'n/a '
    not_napv = np.ones(R, bool)
    if napv is not None:
        not_napv[napv] = False

    # for row labels:
    w = max(11, len(max(labels, key=len)))
    frm = '{{:{:d}}}'.format(w)

    # start preparing for writer.formheader:
    headers1 = ['', '']
    headers2 = [rowhdr, deschdr]
    formats = ['{:7d}', frm]
    printargs = [np.arange(1, R+1), labels]
    widths = [7, w]
    seps = [0, 2]
    justs = ['c', 'l']

    if shortabsmax:
        doabsmax = True
    if doabsmax:
        use_range = False
    if roundvals > -1:
        mxmn1 = np.round(mxmn1, roundvals)
        mxmn2 = np.round(mxmn2, roundvals)

    prtbads = (prtbad, prtbadh, prtbadl)
    flagbads = (flagbad, flagbadh, flagbadl)

    # compute percent differences
    pctinfo = {}
    kwargs = dict(names=names, mxmn1=mxmn1, not_napv=not_napv,
                  histogram_inc=histogram_inc, numform=numform,
                  prtbads=prtbads, flagbads=flagbads,
                  maxhdr=maxhdr, minhdr=minhdr, absmhdr=absmhdr,
                  pdhdr=pdhdr)
    mx1 = abs(mxmn1).max(axis=1)
    mx2 = abs(mxmn2).max(axis=1)
    if not doabsmax:
        max1, min1 = (*mxmn1.T,)
        max2, min2 = (*mxmn2.T,)
        mxmn_b = mxmn2 if use_range else None
        for i in zip(('mx', 'mn', 'amx'),
                     (max1, min1, mx1),
                     (max2, min2, mx2),
                     (True, False, True),
                     (maxhdr, minhdr, absmhdr)):
            lbl, ext1, ext2, ismax, valhdr = i
            pctinfo[lbl] = _proc_pct(ext1, ext2, filterval,
                                     mxmn_b=mxmn_b, ismax=ismax,
                                     valhdr=valhdr, **kwargs)
    else:
        pctinfo['amx'] = _proc_pct(mx1, mx2, filterval,
                                   mxmn_b=None, ismax=True,
                                   valhdr=absmhdr, **kwargs)
    prtpv = pctinfo['amx']['prtpv']
    hu, frm = writer.formheader([headers1, headers2], widths,
                                formats, sep=seps, just=justs)

    # format page header:
    misc = (_get_filtline(filterval) +
            _get_noteline(use_range, names, prtbads, flagbads))
    header = (title + '\n\n' +
              _get_rpt_headers(desc=desc, uf_reds=uf_reds,
                               units=units, misc=misc) +
              '\n')

    if domagpct:
        _plot_magpct(pctinfo, names, desc, doabsmax, filename)

    if dohistogram:
        _plot_histogram(pctinfo, names, desc, doabsmax, filename)

    # write results
    def _wtcmp(f, header, hu, frm, printargs, perpage,
               prtpv, pctinfo, desc):
        prtpv = prtpv.nonzero()[0]
        if perpage < 1:
            # one additional in case size is zero
            perpage = prtpv.size + 1
        pages = (prtpv.size + perpage - 1) // perpage
        if prtpv.size < len(printargs[0]):
            for i, item in enumerate(printargs):
                printargs[i] = [item[j] for j in prtpv]
        tabhead = header + hu
        pager = '\n'  # + chr(12)
        for p in range(pages):
            if p > 0:
                f.write(pager)
            f.write(tabhead)
            b = p * perpage
            e = b + perpage
            writer.vecwrite(f, frm, *printargs, so=slice(b, e))
        f.write(pager)
        # f.write(header)
        for lbl, hdr in zip(('mx', 'mn', 'amx'),
                            (maxhdr, minhdr, absmhdr)):
            if lbl in pctinfo:
                f.write(_get_histogram_str(
                        desc, hdr, pctinfo[lbl]['hsto']))

    ytools.wtfile(filename, _wtcmp, header, hu, frm,
                  printargs, perpage, prtpv, pctinfo, desc)
    return pctinfo


def mk_plots(res, event=None, issrs=True, Q='auto', drms=None,
             inc0rb=True, fmt='pdf', onepdf=True, layout=[2, 3],
             figsize=[11, 8.5], showall=None, showboth=False,
             cases=None, direc='srs_plots', sub_right=None):
    """
    Make SRS or response history plots

    Parameters
    ----------
    res : :class:`DR_Results` instance
        Subclass of dict containing categories with results (see
        :class:`DR_Results`). For example, ``results['SC_atm'].ext``.
    event : string or None; optional
        String for plot titles and file names (eg: 'Liftoff'). If
        None, `event` is determined from `res[drm].event`.
    issrs : bool; optional
        True if plotting SRS data; False otherwise.
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
        :func:`matplotlib.pyplot.savefig`. If None, no figures
        will be saved. Typical values for `fmt` are (from
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

    onepdf : bool; optional
        If True (and `fmt` == "pdf") write all plots to one PDF
        file; otherwise, each figure is put in its own file.
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
    sub_right : scalar or None; optional
        Used in: ``plt.subplots_adjust(right=sub_right)`` when
        a legend is placed outside the plots. If None, this
        routine tries to make an educated guess from the longest
        label.
    """
    def _get_Qs(Q, srsQs, showall, name):
        if Q == 'auto':
            Qs = srsQs
        else:
            Q_in = _ensure_iter(Q)
            Qs = []
            for q in Q_in:
                if q in srsQs:
                    Qs.append(q)
                else:
                    warn('no Q={} SRS data for {}'.
                         format(q, name), RuntimeWarning)
            if len(Qs) == 0:
                return None
        Qs = _ensure_iter(Qs)
        if len(Qs) > 1 and showall:
            raise ValueError('`Q` must be a scalar if `showall` '
                             'is true')
        return Qs

    def _set_vars(res, name, event, showall, showboth, cases):
        curres = res[name]
        if issrs:
            labels = curres.drminfo.srslabels
            rowpv = curres.drminfo.srspv
            lbl = srstype = curres.srs.type
            units = curres.drminfo.srsunits
            if showall:
                cases = curres.cases
                if showboth:
                    lbl = lbl + '_all_env'
                else:
                    lbl = lbl + '_all'
            else:
                cases = None
        else:
            labels = curres.drminfo.histlabels
            rowpv = curres.drminfo.histpv
            lbl = 'resp'
            srstype = None
            units = curres.drminfo.histunits
            if cases is None:
                cases = curres.cases
            else:
                for case in cases:
                    if case not in curres.cases:
                        raise ValueError('case {} not found for'
                                         ' {}'.format(case, name))

        if isinstance(rowpv, slice):
            rowpv = np.arange(len(curres.drminfo.labels))[rowpv]
        maxlen = len(max(labels, key=len))
        if event is not None:
            sname = event
        else:
            sname = curres.event

        return (labels, rowpv, maxlen, sname,
                srstype, lbl, units, cases)

    def _get_figname(nplots, perpage, fmt, onepdf,
                     name, lbl, sname):
        if nplots > perpage:
            if fmt == 'pdf' and onepdf:
                prefix = '{}_{}'.format(name, lbl)
                figname = '{} {}_{}'.format(sname, prefix,
                                            filenum)
            else:
                prefix = '{}_{}_{}'.format(name, lbl, filenum)
                figname = '{} {}'.format(sname, prefix)
        else:
            prefix = '{}_{}'.format(name, lbl)
            figname = '{} {}'.format(sname, prefix)
        return prefix, figname

    def _prep_subplot(rows, cols, sub, perpage, filenum, nplots,
                      fmt, name, lbl, sname, figsize, prefix):
        sub += 1
        if sub > perpage:
            sub = 1
            filenum += 1
            prefix, figname = _get_figname(
                nplots, perpage, fmt, onepdf, name, lbl, sname)
            plt.figure(figname, figsize=figsize)
            plt.clf()
        ax = plt.subplot(rows, cols, sub)
        ax.ticklabel_format(useOffset=False,
                            style='sci', scilimits=(-3, 4))
        txt = ax.get_yaxis().get_offset_text()
        txt.set_x(-.22)
        txt.set_va('bottom')
        plt.grid(True)
        return sub, filenum, prefix

    def _add_title(name, label, maxlen, sname, row, cols,
                   q=None):
        def _add_q(ttl, q):
            if q is not None:
                ttl = '{}, Q={}'.format(ttl, q)
            return ttl

        if cols == 1:
            small = 'medium'
            big = 'large'
        elif cols == 2:
            small = 10
            big = 'large'
        else:
            small = 8
            big = 12

        if maxlen > 10:
            ttl = '{} {}\nRow {}'.format(name, sname, row)
            plt.annotate(label, xy=(0, 1),
                         xycoords='axes fraction',
                         fontsize=small,
                         xytext=(3, -3),
                         textcoords='offset points',
                         ha='left', va='top')
        else:
            ttl = '{} {}\n{}'.format(name, sname, label)
        ttl = _add_q(ttl, q)
        plt.title(ttl, fontsize=big)

    def _plot_all(curres, q, frq, hist, showboth, cases, sub,
                  cols, maxcol, name, label, maxlen,
                  sname, rowpv, j, adjust4legend):
        # legspace = matplotlib.rcParams['legend.labelspacing']
        if issrs:
            srsall = curres.srs.srs[q]
            srsext = curres.srs.ext[q]
            # srsall (cases x rows x freq)
            # srsext (each rows x freq)
            every = len(frq) // 25
            h = []
            marker = get_marker_cycle()
            for n, case in enumerate(cases):
                h += plt.plot(x, srsall[n, j],
                              linestyle='-',
                              marker=next(marker),
                              markevery=every,
                              label=case)
                every += 1
            if showboth:
                h.insert(0,
                         plt.plot(x, srsext[j], 'k-', lw=2,
                                  label='Envelope', zorder=-1)[0])
        else:
            # hist (cases x rows x time | freq)
            h = []
            for n, case in enumerate(cases):
                h += plt.plot(x, hist[n, j],
                              linestyle='-',
                              label=case)
        if sub == maxcol:
            lbls = [_.get_label() for _ in h]
            lbllen = len(max(lbls, key=len))
            leg = plt.legend(h, lbls)
            plt.legend(loc='upper left',
                       bbox_to_anchor=(1.02, 1.),
                       borderaxespad=0.,
                       fontsize='small',
                       framealpha=0.5,
                       # labelspacing=legspace*.9,
                       )
            if sub == cols:
                adjust4legend.append(leg)
                adjust4legend.append(lbllen)
        _add_title(name, label, maxlen, sname,
                   rowpv[j]+1, cols, q)

    def _plot_ext(curres, q, frq, sub, cols, maxcol, name,
                  label, maxlen, sname, rowpv, j):
        srsext = curres.srs.ext[q]
        # srsext (each rows x freq)
        if sub == maxcol:
            plt.plot(frq, srsext[j], label='Q={}'.format(q))
            plt.legend(loc='best', fontsize='small',
                       fancybox=True, framealpha=0.5)
        else:
            plt.plot(frq, srsext[j])
        if q == Qs[0]:
            _add_title(name, label, maxlen, sname,
                       rowpv[j]+1, cols)

    def _add_xy_labels(uj, xlab, ylab, srstype):
        if isinstance(units, str):
            u = units
        else:
            if uj > len(units):
                uj = 0
            u = units[uj]
            if len(units) < nplots:
                if sub == 1 or sub == 4:
                    uj += 1   # each label goes to 3 rows
            else:
                uj += 1
        if issrs:
            if srstype == 'eqsine':
                plt.ylabel('EQ-Sine ({})'.format(u))
            else:
                plt.ylabel('SRS ({})'.format(u))
        else:
            plt.ylabel(ylab + ' ({})'.format(u))
        plt.xlabel(xlab)
        return uj

    # main routine
    if showboth and showall is None:
        showall = True

    if not os.path.exists(direc):
        os.mkdir(direc)

    rows = layout[0]
    cols = layout[1]
    perpage = rows*cols
    if cols > rows:
        orientation = 'landscape'
        if figsize[0] < figsize[1]:
            figsize = figsize[1], figsize[0]
    else:
        orientation = 'portrait'
        if figsize[0] > figsize[1]:
            figsize = figsize[1], figsize[0]

    if drms is None:
        alldrms = sorted(list(res.keys()))
    else:
        alldrms = copy.copy(drms)
        if inc0rb:
            for name in drms:
                if name+'_0rb' in res:
                    alldrms.append(name+'_0rb')

    pdffile = None
    try:
        for J, name in enumerate(alldrms):
            if name not in res:
                raise ValueError('category {} does not exist.'
                                 .format(name))
            if event is None:
                _event = res[name].event
            if issrs:
                if 'srs' not in res[name].__dict__:
                    if drms and name in drms:
                        warn('no SRS data for {}'.format(name),
                             RuntimeWarning)
                    continue
                Qs = _get_Qs(Q, res[name].drminfo.srsQs,
                             showall, name)
                if Qs is None:
                    continue
                x = res[name].srs.frq
                y = None
                xlab = 'Frequency (Hz)'
                ylab = None
            else:
                if 'hist' in res[name].__dict__:
                    x = res[name].time
                    y = res[name].hist
                    xlab = 'Time (s)'
                    ylab = 'Time Response'
                elif 'PSD' in res[name].__dict__:
                    x = res[name].freq
                    y = res[name].psd
                    xlab = 'Frequency (Hz)'
                    ylab = 'PSD Response'
                elif 'FRF' in res[name].__dict__:
                    x = res[name].freq
                    y = abs(res[name].FRF)
                    xlab = 'Frequency (Hz)'
                    ylab = 'FRF Response'
                else:
                    if drms and name in drms:
                        warn('no response data for {}'.
                             format(name), RuntimeWarning)
                    continue

            (labels, rowpv, maxlen,
             sname, srstype, lbl,
             units, _cases) = _set_vars(res, name, _event, showall,
                                        showboth, cases)

            if fmt == 'pdf' and onepdf and pdffile is None:
                fname = os.path.join(direc, sname+'.'+fmt)
                pdffile = PdfPages(fname)

            filenum = 0
            uj = 0   # units index
            # nplots = res[name].srs.ext[Qs[0]].shape[0]
            nplots = len(rowpv)
            maxcol = cols if nplots > cols else nplots
            sub = perpage
            prefix = None
            adjust4legend = []
            for j in range(nplots):
                sub, filenum, prefix = _prep_subplot(
                    rows, cols, sub, perpage, filenum, nplots,
                    fmt, name, lbl, sname, figsize, prefix)
                label = ' '.join(labels[j].split())
                if issrs:
                    for q in Qs:
                        if showall:
                            _plot_all(res[name], q, x, y,
                                      showboth, _cases, sub, cols,
                                      maxcol, name, label,
                                      maxlen, sname, rowpv, j,
                                      adjust4legend)
                        else:
                            _plot_ext(res[name], q, x, sub, cols,
                                      maxcol, name, label,
                                      maxlen, sname, rowpv, j)
                else:
                    _plot_all(res[name], None, x, y, showboth,
                              _cases, sub, cols, maxcol,
                              name, label, maxlen, sname,
                              rowpv, j, adjust4legend)
                _add_xy_labels(uj, xlab, ylab, srstype)

                if j+1 == nplots or (j+1) % perpage == 0:
                    plt.tight_layout(pad=3, w_pad=2.0, h_pad=2.0)
                    if len(adjust4legend) > 0:
                        if sub_right is None:
                            # try every 4 chars = 0.03
                            n4s = adjust4legend[1] // 4 + 1
                            _sub_right = 0.94 - 0.03 * n4s
                            if _sub_right < 0.5:
                                _sub_right = 0.8
                        else:
                            _sub_right = sub_right
                        plt.subplots_adjust(right=_sub_right)
                        adjust4legend = []
                    if fmt == 'pdf' and onepdf:
                        pdffile.savefig()
                        # orientation=orientation,
                        # papertype='letter')
                    elif fmt:
                        fname = os.path.join(direc,
                                             prefix+'.'+fmt)
                        plt.savefig(fname, format=fmt,
                                    orientation=orientation,
                                    dpi=200, bbox_inches='tight')
    finally:
        if pdffile:
            pdffile.close()
