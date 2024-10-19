# -*- coding: utf-8 -*-
import itertools
import copy
from collections import OrderedDict
from types import SimpleNamespace
from keyword import iskeyword
import datetime
import importlib
import numpy as np
from pyyeti import locate, ytools


__all__ = [
    "_is_valid_identifier",
    "_compile_strfunc",
    "get_drfunc",
    "_merge_uf_reds",
    "_get_rpt_headers",
    "_get_numform",
    "get_marker_cycle",
    "_proc_filterval",
    "PrintCLAInfo",
    "freq3_augment",
    "maxmin",
    "nan_argmax",
    "nan_argmin",
    "nan_absmax",
    "extrema",
    # "reorder",
    # "_calc_covariance_sine_cosine",
    "PSD_consistent_rss",
]


# temporary patch for numpy < 2.0
try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _is_valid_identifier(name):
    "Return True if `name` is valid Python identifier"
    return name.isidentifier() and not iskeyword(name)


def _compile_strfunc(s, get_psd):
    # build function and return it:
    strfunc = "def _func(sol, nas, Vars, se):\n    return " + s.strip()
    g = globals()
    exec(strfunc, g)
    if get_psd:
        return g["_func"], None
    return g["_func"]


def get_drfunc(filename, funcname, get_psd=False):
    """
    Get data recovery function(s)

    Parameters
    ----------
    filename : string
        Name of Python file containing data recovery function
    funcname : string
        Name of data recovery function
    get_psd : bool; optional
        If True, return a tuple of (`drfunc`, `psd_drfunc`).
        Otherwise, just return `drfunc`.

    Returns
    -------
    drfunc : function
        The data recovery function
    psd_drfunc : function or None; optional
        The PSD-specific data recovery function or None if no such
        function was defined; only returned if `get_psd` is True.
    """
    if _is_valid_identifier(funcname):
        # force a proper exception if file doesn't exist:
        with open(filename, "r") as _:
            pass
        spec = importlib.util.spec_from_file_location("has_drfuncs", filename)
        drmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(drmod)
        func = getattr(drmod, funcname)
        if get_psd:
            psdfunc = getattr(drmod, funcname + "_psd", None)
            return func, psdfunc
        return func

    return _compile_strfunc(funcname, get_psd)


def _merge_uf_reds(old, new, method="replace"):
    if method == "replace":
        merged = (n if n is not None else o for o, n in zip(old, new))
    elif method == "multiply":
        merged = (o * n if n is not None else o for o, n in zip(old, new))
    elif callable(method):
        merged = (method(o, n) if n is not None else o for o, n in zip(old, new))
    else:
        raise ValueError(
            '`method` value must be either "replace", '
            '"multiply", or a function (a callable)'
        )
    return tuple(merged)


def _get_rpt_headers(res=None, desc=None, uf_reds=None, units=None, misc=""):
    if res is not None:
        desc = res.drminfo.desc
        uf_reds = res.drminfo.uf_reds
        units = res.drminfo.units
    descline = f"Description: {desc}\n"
    if uf_reds is None:
        unceline = "Uncertainty: Not specified\n"
    else:
        unceline = (
            "Uncertainty: [Rigid, Elastic, Dynamic, Static] "
            "= [{}, {}, {}, {}]\n".format(*uf_reds)
        )
    unitline = f"Units:       {units}\n"
    currdate = datetime.date.today().strftime("%d-%b-%Y")
    dateline = f"Date:        {currdate}\n"
    return descline + unceline + unitline + misc + dateline


def _get_numform(mxmn1, excel=False):
    # excel logic is different than text:
    # - it avoids scientific notation since the actual values are
    #   there ... the user can just change the format
    pv = (mxmn1 != 0.0) & np.isfinite(mxmn1)
    if not np.any(pv):
        return "{:13.0f}" if not excel else "#,##0."
    pmx = int(np.floor(np.log10(abs(mxmn1[pv]).max())))
    if excel:
        numform = "#,##0." + "0" * (5 - pmx)
    else:
        pmn = int(np.floor(np.log10(abs(mxmn1[pv]).min())))
        if pmx - pmn < 6 and pmn > -3:
            if pmn < 5:
                numform = f"{{:13.{5 - pmn}f}}"
            else:
                numform = "{:13.0f}"
        else:
            numform = "{:13.6e}"
    return numform


def get_marker_cycle():
    """
    Return an ``itertools.cycle`` of plot markers.

    The list is taken from `matplotlib.markers`. Currently::

        'o',          # circle
        'v',          # triangle_down
        '^',          # triangle_up
        '<',          # triangle_left
        '>',          # triangle_right
        '1',          # tri_down
        '2',          # tri_up
        '3',          # tri_left
        '4',          # tri_right
        '8',          # octagon
        's',          # square
        'p',          # pentagon
        'P',          # plus (filled)
        '*',          # star
        'h',          # hexagon1
        'H',          # hexagon2
        '+',          # plus
        'x',          # x
        'X',          # x (filled)
        'D',          # diamond
        'd',          # thin_diamond

    """
    return itertools.cycle(
        [
            "o",  # circle
            "v",  # triangle_down
            "^",  # triangle_up
            "<",  # triangle_left
            ">",  # triangle_right
            "1",  # tri_down
            "2",  # tri_up
            "3",  # tri_left
            "4",  # tri_right
            "8",  # octagon
            "s",  # square
            "p",  # pentagon
            "P",  # plus (filled)
            "*",  # star
            "h",  # hexagon1
            "H",  # hexagon2
            "+",  # plus
            "x",  # x
            "X",  # x (filled)
            "D",  # diamond
            "d",  # thin_diamond
        ]
    )


def _proc_filterval(filterval, nrows, name="filterval"):
    if filterval is None:
        return None
    filterval = np.atleast_1d(filterval)
    if filterval.ndim > 1:
        raise ValueError(f"`{name}` must be 1-D (is {filterval.ndim}-D)")
    nfilt = len(filterval)
    if nfilt > 1 and nfilt != nrows:
        raise ValueError(
            f"`{name}` has incorrect length:"
            f" expected {nrows} elements, but got {nfilt}"
        )
    return filterval


def PrintCLAInfo(mission, event):
    "Print CLA event info, typically for the log file"
    print(f"Mission:  {mission}")
    print(f"Event:    {event}")


def freq3_augment(freq1, lam, tol=1.0e-5):
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
    >>> import numpy as np
    >>> from pyyeti import cla
    >>> freq1 = np.arange(5., 11.)
    >>> sysHz = np.array([3.3, 6.7, 8.9, 9.00001, 12.4])
    >>> lam = (2*np.pi*sysHz)**2
    >>> np.set_printoptions(linewidth=80)
    >>> cla.freq3_augment(freq1, lam)
    array([  5. ,   6. ,   6.7,   7. ,   8. ,   8.9,   9. ,  10. ])
    """
    freq1, lam = np.atleast_1d(freq1, lam)
    sysfreqs = np.sqrt(abs(lam)) / (2 * np.pi)
    pv = np.nonzero(np.logical_and(sysfreqs > freq1[0], sysfreqs < freq1[-1]))[0]
    freq3 = sysfreqs[pv]
    freq = np.sort(np.hstack((freq1, freq3)))
    uniq = locate.find_unique(freq, tol * (freq[-1] - freq[0]))
    return freq[uniq]


def maxmin(response, x):
    """
    Compute max & min of a response matrix.

    Parameters
    ----------
    response : 2d ndarray
        Matrix where each row is a response signal.
    x: 1d ndarray
        X-axis vector (eg, time or frequency);
        ``len(x) = response.shape[1]``

    Returns
    -------
    A SimpleNamespace with the members:

    ext : 2d ndarray
        Two column matrix: ``[max, min]``
    ext_x : 2d ndarray
        Two column matrix: ``[x_of_max, x_of_min]``
    """
    r, c = np.shape(response)
    if c != len(x):
        raise ValueError("# of cols in `response` is not compatible with `x`.")
    jx = np.nanargmax(response, axis=1)
    jn = np.nanargmin(response, axis=1)
    ind = np.arange(r)
    mx = response[ind, jx]
    mn = response[ind, jn]
    return SimpleNamespace(
        ext=np.column_stack((mx, mn)), ext_x=np.column_stack((x[jx], x[jn]))
    )


def nan_argmax(v1, v2):
    """
    Find where `v2` is greater than `v1` ignoring NaNs.

    Parameters
    ----------
    v1 : ndarray
        First set of values to compare
    v2 : ndarray
        Second set of values to compare; must be broadcast-compatible
        with `v1`.

    Returns
    -------
    pv : bool ndarray
        Contains True where `v2` is greater than `v1` ignoring NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import cla
    >>> v1 = np.array([1.0, np.nan, 2.0, np.nan])
    >>> v2 = np.array([-2.0, 3.0, np.nan, np.nan])
    >>> cla.nan_argmax(v1, v2)
    array([False,  True, False, False], dtype=bool)
    >>> cla.nan_argmin(v1, v2)
    array([ True,  True, False, False], dtype=bool)
    """
    with np.errstate(invalid="ignore"):
        return (v2 > v1) | (np.isnan(v1) & ~np.isnan(v2))


def nan_argmin(v1, v2):
    """
    Find where `v2` is less than `v1` ignoring NaNs.

    Parameters
    ----------
    v1 : ndarray
        First set of values to compare
    v2 : ndarray
        Second set of values to compare; must be broadcast-compatible
        with `v1`.

    Returns
    -------
    pv : bool ndarray
        Contains True where `v2` is less than `v1` ignoring NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import cla
    >>> v1 = np.array([1.0, np.nan, 2.0, np.nan])
    >>> v2 = np.array([-2.0, 3.0, np.nan, np.nan])
    >>> cla.nan_argmax(v1, v2)
    array([False,  True, False, False], dtype=bool)
    >>> cla.nan_argmin(v1, v2)
    array([ True,  True, False, False], dtype=bool)
    """
    with np.errstate(invalid="ignore"):
        return (v2 < v1) | (np.isnan(v1) & ~np.isnan(v2))


def nan_absmax(v1, v2):
    """
    Get absolute maximum values between `v1` and `v2` while
    retaining signs and ignoring NaNs.

    Parameters
    ----------
    v1 : ndarray
        First set of values to compare
    v2 : ndarray
        Second set of values to compare; must be broadcast-compatible
        with `v1`.

    Returns
    -------
    amax : ndarray
        The absolute maximum values over `v1` and `v2`. The signs are
        retained and NaNs are ignored as much as possible.
    pv : bool ndarray
        Contains True where ``abs(v2)`` is greater than ``abs(v1)``
        ignoring NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import cla
    >>> v1 = np.array([[1, -4], [3, 5]])
    >>> v2 = np.array([[-2, 3], [4, np.nan]])
    >>> cla.nan_absmax(v1, v2)
    (array([[-2, -4],
           [ 4,  5]]), array([[ True, False],
           [ True, False]], dtype=bool))
    """
    amx = v1.copy()
    pv = nan_argmax(abs(v1), abs(v2))
    amx[pv] = v2[pv]
    return amx, pv


def extrema(curext, mm, maxcase, mincase=None, casenum=None):
    """
    Update extrema values in 'curext'

    Parameters
    ----------
    curext : SimpleNamespace
        Has extrema information (members may be None on first call)::

            .ext     = 1 or 2 columns: [max, min]
            .ext_x   = None or 1 or 2 columns: [x_of_max, x_of_min]
            .maxcase = list of strings identifying maximum case
            .mincase = list of strings identifying minimum case

        Also has these members if casenum is an integer::

            .mx      = [case1_max, case2_max, ...]
            .mn      = [case1_min, case2_min, ...]
            .mx_x    = [case1_x_of_max, case2_x_of_max, ...]
            .mn_x    = [case1_x_of_min, case2_x_of_min, ...]

        The `mx_x` and `mn_x` will have ``np.nan`` values if
        `ext_x` is None.
    mm : SimpleNamespace
        Has min/max information for a case (or over cases)::

            .ext     = 1 or 2 columns: [max, min]
            .ext_x   = None or 1 or 2 columns: [x_of_max, x_of_min]

    maxcase : string or list of strings
        String or list of strings identifying the load case(s) for the
        maximum values. This is analogous to `curext.maxcase` but
        pertaining to `mm` (handy if `mm` is from another extrema data
        set).
    mincase : string or list of strings or None; optional
        Analogous to `maxcase` for the minimum values or None. If
        None, it is a copy of the `maxcase` values.
    casenum : integer or None; optional
        If integer, it is case number (starting at 0); `curext` will
        have the `.mx`, `.mn`, `.mx_x`, `.mn_x` members and the
        `casenum` column of each of these will be set to the data from
        `mm`. Zeros will be used for "x" if `mm` is 1 or 2 columns.
        If None, `.mx`, `.mn`, `.mx_x`, `.mn_x` are not updated
        (and need not be present).

    Returns
    -------
    None

    Notes
    -----
    This routine updates the `curext` variable. This routine is
    typically called indirectly via
    :func:`DR_Results.time_data_recovery`,
    :func:`DR_Results.frf_data_recovery`, and
    :func:`DR_Results.psd_data_recovery`.
    """

    def _put_time(curext, mm, j, col_lhs, col_rhs):
        if mm.ext_x is not None:
            if curext.ext_x is None:
                curext.ext_x = copy.copy(mm.ext_x)
            else:
                curext.ext_x[j, col_lhs] = mm.ext_x[j, col_rhs]
        elif curext.ext_x is not None:  # pragma: no cover
            curext.ext_x[j, col_lhs] = np.nan

    r, c = mm.ext.shape
    if c not in [1, 2]:
        raise ValueError(f"mm.ext has {c} cols, but must have 1 or 2.")

    # expand current case information to full size if necessary
    if isinstance(maxcase, str):
        maxcase = r * [maxcase]
    else:
        maxcase = maxcase[:]

    if c == 1:
        if casenum is not None:  # record current results
            curext.mx[:, casenum] = mm.ext[:, 0]
            curext.mn[:, casenum] = mm.ext[:, 0]
            if mm.ext_x is not None:
                curext.mx_x[:, casenum] = mm.ext_x[:, 0]
                curext.mn_x[:, casenum] = mm.ext_x[:, 0]
            else:
                curext.mx_x[:, casenum] = np.nan
                curext.mn_x[:, casenum] = np.nan

        if curext.ext is None:
            curext.ext = mm.ext @ [[1, 1]]
            if mm.ext_x is not None:
                curext.ext_x = mm.ext_x @ [[1, 1]]
            else:
                curext.ext_x = None
            curext.maxcase = maxcase
            curext.mincase = maxcase[:]
            return

        # keep sign but compare based on absolute
        j = nan_argmax(abs(curext.ext), abs(mm.ext)).nonzero()[0]
        if j.size > 0:
            for i in j:
                curext.maxcase[i] = maxcase[i]
            curext.ext[j, 0] = mm.ext[j, 0]
            _put_time(curext, mm, j, 0, 0)

        j = nan_argmin(abs(curext.ext), abs(mm.ext)).nonzero()[0]
        if j.size > 0:
            for i in j:
                curext.mincase[i] = maxcase[i]
            curext.ext[j, 1] = mm.ext[j, 0]
            _put_time(curext, mm, j, 1, 0)
        return

    if mincase is None:
        mincase = maxcase[:]
    elif isinstance(mincase, str):
        mincase = r * [mincase]
    else:
        mincase = mincase[:]

    if casenum is not None:  # record current results
        curext.mx[:, casenum] = mm.ext[:, 0]
        curext.mn[:, casenum] = mm.ext[:, 1]
        if mm.ext_x is not None:
            curext.mx_x[:, casenum] = mm.ext_x[:, 0]
            curext.mn_x[:, casenum] = mm.ext_x[:, 1]
        else:
            curext.mx_x[:, casenum] = np.nan
            curext.mn_x[:, casenum] = np.nan

    if curext.ext is None:
        curext.ext = mm.ext.copy()
        curext.ext_x = copy.copy(mm.ext_x)
        curext.maxcase = maxcase
        curext.mincase = mincase
        return

    j = nan_argmax(curext.ext[:, 0], mm.ext[:, 0]).nonzero()[0]
    if j.size > 0:
        for i in j:
            curext.maxcase[i] = maxcase[i]
        curext.ext[j, 0] = mm.ext[j, 0]
        _put_time(curext, mm, j, 0, 0)

    j = nan_argmin(curext.ext[:, 1], mm.ext[:, 1]).nonzero()[0]
    if j.size > 0:
        for i in j:
            curext.mincase[i] = mincase[i]
        curext.ext[j, 1] = mm.ext[j, 1]
        _put_time(curext, mm, j, 1, 1)


def PSD_consistent_rss(resp, xr, yr, rr, freq, forcepsd, drmres, case, i):
    """
    Compute phase-consistent (time-correlated) root-sum-square (RSS)
    responses in a PSD analysis; each RSS is of two rows.

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

        On the other hand, if the RSS is to be stored alone in its
        category, set `rr` to None. For example, `resp` could have 2
        rows with ``xr = 0``, ``yr = 1``. Then, the category would
        only have 1 data recovery item
    freq : 1d array_like
        Frequency vector (Hz)
    forcepsd : 2d ndarray
        Matrix of force PSDs; nforces x len(freq)
    drmres : input/output SimpleNamespace
        Results for a DRM; eg if drmres = results['ifa']:

        .. code-block:: none

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
        Unique case identifier (like 'MaxQ') for storing the PSD
        results, eg::

            results['ifa']._psd[case]

    i : integer
        Current force index; starts at 0 and goes to nforces-1

    Returns
    -------
    None

    Notes
    -----
    This function is typically called by a "drfunc" specified in a
    call to :func:`DR_Def.add`. An example function that uses this
    routine is provided there.

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

    This routine works by computing the maximum length principal axis
    (eigenvector) of the covariance matrix. The RSS results are then
    calculated along the eigenvector direction. See comments in source
    code of :func:`pyyeti.ytools._calc_covariance_sine_cosine` for
    more details. Also see, for example, reference [#cov]_.

    References
    ----------
    .. [#cov] http://www.visiondummy.com/2014/04/
              draw-error-ellipse-representing-covariance-matrix/
    """
    # drmres is a SimpleNamespace: drmres = results[drm]
    # i is psd force index
    F = forcepsd[i]
    # normal, non-rss data recovery:
    if rr is not None:
        drmres._psd[case] += F * abs(resp) ** 2
    N = forcepsd.shape[0]
    if i == 0:
        drmres.tmp = SimpleNamespace(
            varx=0, vary=0, covar=0, xresp=[0] * N, yresp=[0] * N
        )

    x = resp[xr]
    y = resp[yr]
    tmp = drmres.tmp
    # after area is computed, these will be the variance/covariance:
    tmp.varx += F * abs(x) ** 2
    tmp.vary += F * abs(y) ** 2
    tmp.covar += F * np.real(x * np.conj(y))
    tmp.xresp[i] = x
    tmp.yresp[i] = y

    if i == N - 1:
        varx = np.trapezoid(tmp.varx, freq)
        vary = np.trapezoid(tmp.vary, freq)
        covar = np.trapezoid(tmp.covar, freq)
        s, c = ytools._calc_covariance_sine_cosine(varx, vary, covar)

        # now have all sines/cosines, compute consistent vector sum
        # results:
        rss_resp = 0.0
        for j in range(N):
            respxy = c[:, None] * tmp.xresp[j] + s[:, None] * tmp.yresp[j]
            rss_resp += forcepsd[j] * abs(respxy) ** 2

        if rr is not None:
            drmres._psd[case][rr] = rss_resp
        else:
            drmres._psd[case] = rss_resp

        # delete 'extra' info:
        del drmres.tmp
