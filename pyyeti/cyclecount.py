# -*- coding: utf-8 -*-
"""
Tools for cycle counting. Adapted and enhanced from the Yeti version.
"""

import warnings
import numpy as np
import pandas as pd
from pyyeti import locate

try:
    import numba
except ImportError:
    HAVE_NUMBA = False
    numba_bool = bool
else:
    HAVE_NUMBA = True
    numba_bool = numba.types.bool_

try:
    import pyyeti.rainflow.c_rain as rain
except ImportError:
    if not HAVE_NUMBA:
        warnings.warn(
            "Compiled C version of rainflow algorithm failed to "
            "import. Using plain Python version, which can be very "
            "slow. Recommend installing 'numba': pyYeti will use that "
            "automatically to speed up the Python version to be on "
            "par with the C version.",
            RuntimeWarning,
        )
    import pyyeti.rainflow.py_rain as rain

# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def rainflow(peaks, getoffsets=False, use_pandas=True):
    """
    Rainflow cycle counting.

    Parameters
    ----------
    peaks : 1d array_like
        Vector of alternating peaks (as returned by :func:`findap`,
        for example)
    getoffsets : bool; optional
        If True, the tuple ``(rf, os)`` is returned; otherwise, only
        `rf` is returned.
    use_pandas : bool; optional
        If True, this routine will return pandas DataFrames

    Returns
    -------
    rf : pandas DataFrame or 2d ndarray
        n x 3 matrix with the rainflow cycle count information with
        the index going from 0 to n-1 and the columns being ['amp',
        'mean', 'count']:

            - amp is the cycle amplitude (half the peak-to-peak range)
            - mean is mean of the cycle
            - count is either 0.5 or 1.0 depending on whether it's
              half or full cycle

    os : pandas DataFrame or 2d ndarray
        Only returned if `getoffsets` is True. n x 2 matrix of cycle
        offsets with index going from 0 to n-1 and the columns being
        ['start', 'stop']:

            - start is the offset into `peaks` for start of cycle
            - stop is the offset into `peaks` for end of cycle

    Notes
    -----
    This algorithm is derived from reference [#cc1]_. This routine is
    a wrapper for either the C version
    (:func:`pyyeti.rainflow.c_rain`) or the Python version
    (:func:`pyyeti.rainflow.py_rain`) of the algorithm. Both versions
    give identical results. They also run in comparable times if
    :func:`numba.jit` is available (if ``import numba`` works). If
    :func:`numba.jit` is not available, the C version is *much*
    faster.

    References
    ----------
    .. [#cc1] "Standard Practices for Cycle Counting in Fatigue
           Analysis", ASTM E 1049 - 85 (Reapproved 2005).

    Examples
    --------
    Run the example from the ASTM paper:

    >>> from pyyeti.cyclecount import rainflow
    >>> rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
       amp  mean  count
    0  1.5  -0.5    0.5
    1  2.0  -1.0    0.5
    2  2.0   1.0    1.0
    3  4.0   1.0    0.5
    4  4.5   0.5    0.5
    5  4.0   0.0    0.5
    6  3.0   1.0    0.5

    With offsets:

    >>> rf, os = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2],
    ...                   getoffsets=True)
    >>> rf
       amp  mean  count
    0  1.5  -0.5    0.5
    1  2.0  -1.0    0.5
    2  2.0   1.0    1.0
    3  4.0   1.0    0.5
    4  4.5   0.5    0.5
    5  4.0   0.0    0.5
    6  3.0   1.0    0.5
    >>> os
       start  stop
    0      0     1
    1      1     2
    2      4     5
    3      2     3
    4      3     6
    5      6     7
    6      7     8

    If not using pandas:

    >>> rf, os = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2],
    ...                   getoffsets=True, use_pandas=False)
    >>> rf
    array([[ 1.5, -0.5,  0.5],
           [ 2. , -1. ,  0.5],
           [ 2. ,  1. ,  1. ],
           [ 4. ,  1. ,  0.5],
           [ 4.5,  0.5,  0.5],
           [ 4. ,  0. ,  0.5],
           [ 3. ,  1. ,  0.5]])
    >>> os                              # doctest: +ELLIPSIS
    array([[0, 1],
           [1, 2],
           [4, 5],
           [2, 3],
           [3, 6],
           [6, 7],
           [7, 8]]...)
    """
    if getoffsets:
        rf, os = rain.rainflow(peaks, getoffsets)
        if use_pandas:
            rf = pd.DataFrame(rf, columns=["amp", "mean", "count"])
            os = pd.DataFrame(os, columns=["start", "stop"])
        return rf, os

    rf = rain.rainflow(peaks, getoffsets)
    if use_pandas:
        rf = pd.DataFrame(rf, columns=["amp", "mean", "count"])
    return rf


if not HAVE_NUMBA:

    def _findap_worker(y, tol):
        # example: [1, 2, 3, 4, 4, -2, -2, -2]

        if y.size == 1:
            return np.array([True])

        # first, find unique values (1st of series is unique)
        u = locate.find_unique(y, tol)
        # [ True,  True,  True,  True, False,  True, False, False]

        # work with unique values only:
        if np.all(u):
            yu = y
            allu = True
        else:
            yu = y[u]
            # [ 1,  2,  3,  4, -2]
            allu = False

        # find signs of slopes to locate local max/mins:
        s = np.sign(np.diff(yu))
        # [ 1,  1,  1, -1]

        # locate local max/mins:
        pv = np.ones(yu.size, bool)
        pv[1:-1] = np.abs(np.diff(s)) == 2
        if yu.size > 2:
            pv[-1] = yu[-1] != yu[-2]
        # [ True, False, False,  True,  True]

        if allu:
            return pv

        # expand to full size:
        PV = np.zeros(y.size, bool)  # non-uniques are not peaks
        PV[u] = pv
        # [ True, False, False,  True, False,  True, False, False]
        return PV

else:

    def _findap_worker(y, tol):
        if y.size == 1:
            return np.array([True])

        if y.size == 2:
            if y[1] == y[0]:
                return np.array([True, False])
            return np.array([True, True])

        stol = np.abs(tol * np.abs(np.diff(y)).max())

        PV = np.zeros(y.size, numba_bool)
        PV[0] = True
        prv = y[0]
        i = 1
        while i < y.size:
            if np.abs(y[i] - prv) > stol:
                break
            i += 1

        if i == y.size:
            return PV

        j = i
        cur = y[j]
        if cur > prv:
            mountain = True  # find mountain peak
        else:
            mountain = False  # find valley floor

        for i in range(i + 1, y.size):
            nxt = y[i]
            if np.abs(nxt - cur) > stol:
                if mountain:
                    if nxt < cur:
                        PV[j] = True
                        mountain = False
                else:
                    if nxt > cur:
                        PV[j] = True
                        mountain = True
                cur = nxt
                j = i

        if np.abs(nxt - y[-2]) > stol:
            PV[-1] = True
        else:
            PV[j] = True

        return PV


def findap(y, tol=1e-6, nan_check=True):
    """
    Find alternating local maximum and minimum points in a vector.

    Parameters
    ----------
    y : ndarray
        y-axis data vector
    tol : scalar; optional
        Tolerance value for detecting unique values; see
        :func:`pyyeti.locate.find_unique`
    nan_check : bool; optional
        Whether to check for NaNs. Set to ``False`` for a performance
        gain if you know there are no NaNs.

    Returns
    -------
    pv : ndarray
        Boolean vector for the alternating peaks in `y`.

    Notes
    -----
    `y` is flattened to one dimension before operations.

    When `y` has a series of equal points, the first of the series is
    considered the peak. The first value in `y` is always considered a
    peak. The last point is a peak if and only if it is not equal to
    the point before it.

    This routine is typically used to prepare a signal for cycle
    counting.

    NaNs are not considered to be peaks and treated as if they are not
    present when `nan_check` is True (the default).

    Examples
    --------
    >>> from pyyeti import cyclecount
    >>> import numpy as np
    >>> cyclecount.findap(np.array([1]))
    array([ True], dtype=bool)
    >>> cyclecount.findap(np.array([1, 1, 1, 1]))
    array([ True, False, False, False], dtype=bool)
    >>> tf = cyclecount.findap(np.array([1, 2, 3, 4, 4, -2, -2, 0]))
    >>> np.nonzero(tf)[0]               # doctest: +ELLIPSIS
    array([0, 3, 5, 7]...)
    >>> tf = cyclecount.findap(np.array([1, 2, 3, 4, 4, -2, -2, -2]))
    >>> np.nonzero(tf)[0]               # doctest: +ELLIPSIS
    array([0, 3, 5]...)
    >>> tf = cyclecount.findap(np.array([1, 2, 3, 4, -2]))
    >>> np.nonzero(tf)[0]               # doctest: +ELLIPSIS
    array([0, 3, 4]...)
    >>> cyclecount.findap(np.array([np.nan, 1, 2, np.nan]))
    array([False,  True,  True, False], dtype=bool)
    """
    y = y.ravel()
    if nan_check:
        nans = np.isnan(y)
        if np.any(nans):
            full_pv = np.zeros(y.size, bool)
            non_nans = ~nans
            y = y[non_nans]
            if y.size == 0:
                return full_pv

            pv = _findap_worker(y, tol)

            full_pv[non_nans] = pv

            return full_pv

    return _findap_worker(y, tol)


def getbins(bins, mx, mn, right=True, check_bounds=False):
    """
    Utility routine used by :func:`sigcount` to get bin boundaries.

    Parameters
    ----------
    bins : scalar integer or 1d array_like
        Used to define bin boundaries. If array_like, must be
        monotonically increasing. See description below.
    mx, mn : scalar
        Maximum and minimum values of data to be put in bins. `mx` and
        `mn` may be input in either order. If `bins` is a 1d
        array_like, `mx` and `mn` are ignored unless `check_bounds`
        is True; see below.
    right : bool; optional
        Indicates whether the bins include the rightmost edge or the
        leftmost edge. If right == True (the default), then the bins
        [1,2,3,4] indicate (1,2], (2,3], (3,4].
    check_bounds : bool; optional
        If True, routine will check to see if the final bin boundaries
        cover the `mn` to `mx` range. Note that the range is guaranteed to be
        covered if `bins` is a scalar.

    Returns
    -------
    bb : 1d ndarray
        The bin boundaries; ``length = bins + 1``
    out_of_bounds : bool; optional
        True if either `mx` or `mn` will fall out of range of the bins
        (according to `right`). Only returned if `check_bounds` is
        True.

    Notes
    -----
    If `mx` and `mn` are equal, they are reset::

        mx = mx + 0.5
        mn = mn - 0.5

    If `bins` is a scalar, this routine tries to mimic the behavior of
    the :func:`pandas.cut` routine::

        bb = np.linspace(mn, mx, bins+1)
        p = 0.001 * (mx - mn)
        if right:
            bb[0] -= p
        else:
            bb[-1] += p

    If `bins` is a vector, it defines the boundaries directly and is
    returned as is::

        bb = np.array(bins)

    Raises
    ------
    ValueError
        If `bins` is a vector but is not monotonically increasing

    Examples
    --------
    >>> from pyyeti import cyclecount
    >>> cyclecount.getbins(4, 12, 4)
    array([  3.992,   6.   ,   8.   ,  10.   ,  12.   ])
    >>> cyclecount.getbins(4, 12, 4, right=False)
    array([  4.   ,   6.   ,   8.   ,  10.   ,  12.008])
    >>> cyclecount.getbins([1, 2, 3, 4], 12, 4, check_bounds=True)
    (array([1, 2, 3, 4]), True)
    """
    bins = np.atleast_1d(bins)
    if mx < mn:
        mx, mn = mn, mx
    elif mx == mn:
        mx = mx + 0.5
        mn = mn - 0.5
        # raise ValueError("`mx` and `mn` must not be equal")

    if bins.size == 1:
        bins = int(bins[0])
        bb = np.linspace(mn, mx, bins + 1)
        p = 0.001 * (mx - mn)
        if right:
            bb[0] -= p
        else:
            bb[-1] += p
        out_of_bounds = False
    elif bins.ndim == 1:
        if np.any(np.diff(bins) <= 0):
            raise ValueError(
                "when `bins` is input as a vector, it "
                "must be monotonically increasing"
            )
        bb = np.array(bins)
        if check_bounds:
            if right:
                if mn <= bb[0] or mx > bb[-1]:
                    out_of_bounds = True
                else:
                    out_of_bounds = False
            else:
                if mn < bb[0] or mx >= bb[-1]:
                    out_of_bounds = True
                else:
                    out_of_bounds = False

    if check_bounds:
        return bb, out_of_bounds

    return bb


def _getlabels(form, bins):
    return [form.format(i, j) for i, j in zip(bins[:-1], bins[1:])]


# @njit(cache=True)
def _binify(cycles, bins_range, bins_mean, right, ensure_boundaries=True):
    # Many thanks to Daniel Kowollik for contributing the original
    # implementation of this routine! :)
    num_bins_mean = len(bins_mean) - 1
    num_bins_range = len(bins_range) - 1

    bin_indices_range = np.digitize(cycles[:, 0], bins_range, right=right) - 1
    bin_indices_mean = np.digitize(cycles[:, 1], bins_mean, right=right) - 1

    markov_matrix = np.zeros((num_bins_mean, num_bins_range), np.float64)
    if ensure_boundaries:
        for i in range(len(cycles)):
            bim = bin_indices_mean[i]
            bir = bin_indices_range[i]
            if (0 <= bim < num_bins_mean) and (0 <= bir < num_bins_range):
                markov_matrix[bim, bir] += cycles[i, 2]
    else:
        for i in range(len(cycles)):
            markov_matrix[bin_indices_mean[i], bin_indices_range[i]] += cycles[i, 2]

    return markov_matrix


if HAVE_NUMBA:
    _findap_worker = numba.njit(cache=True)(_findap_worker)
    _binify = numba.njit(cache=True)(_binify)

    @numba.njit(cache=True)
    def maxmin(x):
        mn = mx = x[0]
        for v in x[1:]:
            if v > mx:
                mx = v
            elif v < mn:
                mn = v
        return mx, mn

else:

    def maxmin(x):
        return x.max(), x.min()


def binify(
    rf,
    ampbins=10,
    meanbins=1,
    right=True,
    precision=3,
    retbins=False,
    use_pandas=True,
    check_bounds=True,
):
    """
    Summarize cycle count results (as from :func:`rainflow`) into
    bins.

    Parameters
    ----------
    rf : 2d array_like
        2d matrix with (at least) 3 columns: [amp, mean, count]. See
        :func:`rainflow`.
    ampbins : scalar or 1d array_like; optional
        Defines the cycle amplitude bins; see :func:`getbins` for
        more complete description on how bins are defined.

        ==============   =============================================
        `ampbins` type   Brief description (see also :func:`getbins`)
        ==============   =============================================
           scalar        Defines number of bins to use
           vector        Defines lower bin boundaries
        ==============   =============================================

    meanbins : scalar or 1d array_like; optional
        Defines the cycle mean-value bins; see `ampbins` description
        and :func:`getbins` for more complete description on how bins
        are defined.
    right : bool; optional
        Indicates whether the bins include the rightmost edge or the
        leftmost edge. If right == True (the default), then the bins
        [1,2,3,4] indicate (1,2], (2,3], (3,4].
    precision : integer; optional
        Precision to use for DataFrame labels; only used if
        `use_pandas` is True.
    retbins : bool; optional
        If True, return the `ampbins` and `meanbins` vectors.
    use_pandas : bool; optional
        If True, this routine will return `table` as a pandas
        DataFrame.
    check_bounds : bool; optional
        If True, routine will check to see if the final bin boundaries
        for both the amplitude and mean cover the range of the
        data. Note that the range is guaranteed to be covered if the
        `ampbins` and `meanbins` inputs are scalars.

    Returns
    -------
    table : pandas DataFrame or 2d ndarray
        A cycle count table (len(`meanbins`) x len(`ampbins`)). Each
        value in `table` entry is the number of cycles in the bin
        (see below).
    ampbins : 1d ndarray; optional
        Boundaries of amplitude bins used; length = # of bins + 1.
        Only returned if `retbins` is True.
    meanbins : 1d ndarray; optional
        Boundaries of mean-value bins used; length = # of bins + 1.
        Only returned if `retbins` is True.

    Notes
    -----
    Algorithm works as follows:

    For each mean value bin (i'th):
       1. `rf` is filtered down to only those rows where the mean value
          falls in the mean-bin (call this subset of `rf` `rffilt`).
       2. For each amplitude bin (j'th): count number of cycles in
          `rffilt` with amplitude that falls in the amp-bin and store in
          ``table[i, j]``

    A value falls in bin `i` if::

        bin[i] < value <= bin[i+1]        } if right is True
        bin[i] <= value < bin[i+1]        } if right is not True

    Examples
    --------
    >>> from pyyeti import cyclecount
    >>> rf = cyclecount.rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    >>> rf
       amp  mean  count
    0  1.5  -0.5    0.5
    1  2.0  -1.0    0.5
    2  2.0   1.0    1.0
    3  4.0   1.0    0.5
    4  4.5   0.5    0.5
    5  4.0   0.0    0.5
    6  3.0   1.0    0.5
    >>> format = lambda x: f"{x:.1f}"
    >>> df = cyclecount.binify(rf, 3, 2)
    >>> df.map(format)                # doctest: +ELLIPSIS
    Amp             (1.497, 2.500] (2.500, 3.500] (3.500, 4.500]
    Mean...
    (-1.002, 0.000]            1.0            0.0            0.5
    (0.000, 1.000]             1.0            0.5            1.0
    >>> df = cyclecount.binify(rf, 3, 2, right=0)
    >>> df.map(format)                # doctest: +ELLIPSIS
    Amp             [1.500, 2.500) [2.500, 3.500) [3.500, 4.503)
    Mean...
    [-1.000, 0.000)            1.0            0.0            0.0
    [0.000, 1.002)             1.0            0.5            1.5

    If not using pandas:

    >>> cyclecount.binify(rf, 3, 2, use_pandas=False)
    array([[ 1. ,  0. ,  0.5],
           [ 1. ,  0.5,  1. ]])
    """
    rf = np.atleast_2d(rf)
    # rf = [amp, mean, count] columns

    ampb = getbins(ampbins, *maxmin(rf[:, 0]), right, check_bounds)
    aveb = getbins(meanbins, *maxmin(rf[:, 1]), right, check_bounds)

    if check_bounds:
        ampb, out_amp = ampb
        aveb, out_ave = aveb
        out = out_amp or out_ave
    else:
        out = False

    table = _binify(rf, ampb, aveb, right, out)

    if use_pandas:
        f = "{:." + str(precision) + "f}"
        f = f + ", " + f
        if right:
            form = "(" + f + "]"
        else:
            form = "[" + f + ")"

        index = _getlabels(form, aveb)
        columns = _getlabels(form, ampb)
        table = pd.DataFrame(table, index=index, columns=columns)
        table.columns.name = "Amp"
        table.index.name = "Mean"

    if retbins:
        return table, ampb, aveb

    return table


def sigcount(
    sig, ampbins=10, meanbins=1, right=True, precision=3, retbins=False, use_pandas=True
):
    """
    Do rainflow cycle counting on a signal.

    Parameters
    ----------
    sig : 1d array_like
        Signal (vector) to do cycle counting on.
    ampbins : scalar or 1d array_like
        Defines the cycle amplitude bins; see :func:`getbins` for
        more complete description on how bins are defined.

        ==============   =============================================
        `ampbins` type   Brief description (see also :func:`getbins`)
        ==============   =============================================
           scalar        Defines number of bins to use
           vector        Defines lower bin boundaries
        ==============   =============================================

    meanbins : scalar or 1d array_like
        Defines the cycle mean-value bins; see `ampbins` description
        and :func:`getbins` for more complete description on how bins
        are defined.
    right : bool; optional
        Indicates whether the bins include the rightmost edge or the
        leftmost edge. If right == True (the default), then the bins
        [1,2,3,4] indicate (1,2], (2,3], (3,4].
    precision : integer; optional
        Precision to use for DataFrame labels.
    retbins : bool; optional
        If True, return the `ampbins` and `meanbins` vectors.
    use_pandas : bool; optional
        If True, this routine will return `table` as a pandas
        DataFrame.

    Returns
    -------
    table : pandas DataFrame or 2d ndarray
        A cycle count table (len(`meanbins`) x len(`ampbins`)). Each
        value in `table` entry is the number of cycles in the bin
        (see below).
    ampbins : 1d ndarray
        Boundaries of amplitude bins used; length = # of bins + 1.
        Only returned if `retbins` is True.
    meanbins : 1d ndarray
        Boundaries of mean-value bins used; length = # of bins + 1.
        Only returned if `retbins` is True.

    Notes
    -----
    Steps:
      1.  Calls :func:`findap` to find all local minima and maxima:
          ``peaks = sig[findap(sig)]``
      2.  Calls :func:`rainflow` to do the cycle counting:
          ``rf = rainflow(peaks)``
      3.  Summarizes the rainflow counting by putting the counts in
          bins. For each mean value bin (i'th):

          a. `rf` is filtered down to only those rows where the mean
             value falls in the mean-bin (call this subset of `rf`
             `rffilt`).
          b. For each amplitude bin (j'th): count number of cycles in
             `rffilt` with amplitude that falls in the amp-bin and
             store in ``table[i, j]``

    A value falls in bin `i` if::

        bin[i] < value <= bin[i+1]        } if right is True
        bin[i] <= value < bin[i+1]        } if right is not True

    Examples
    --------
    >>> from pyyeti import cyclecount
    >>> import numpy as np
    >>> sig = np.arange(100)
    >>> sig[::2] *= -1   # [0, 1, -2, 3, -4, ..., 99]
    >>> # `sig` has 99 half-cycles; amplitude grows from 0.5 up to
    >>> #  98.5; mean of each is either 0.5 or -0.5
    >>> table = cyclecount.sigcount(sig, 2, 2)
    >>> table                              # doctest: +ELLIPSIS
    Amp              (0.402, 49.500]  (49.500, 98.500]
    Mean...
    (-0.501, 0.000]             12.5              12.0
    (0.000, 0.500]              12.5              12.5

    We can focus on amplitudes only (ignoring the mean value bins):

    >>> table.sum(axis=0)
    Amp
    (0.402, 49.500]     25.0
    (49.500, 98.500]    24.5
    dtype: float64

    Focus on only the mean value bins:

    >>> table.sum(axis=1)
    Mean
    (-0.501, 0.000]    24.5
    (0.000, 0.500]     25.0
    dtype: float64

    If not using pandas:

    >>> cyclecount.sigcount(sig, 2, 2, use_pandas=False)
    array([[ 12.5,  12. ],
           [ 12.5,  12.5]])
    """
    rf = rainflow(sig[findap(sig)], use_pandas=False)
    return binify(rf, ampbins, meanbins, right, precision, retbins, use_pandas)
