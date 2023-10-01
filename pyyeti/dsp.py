# -*- coding: utf-8 -*-
"""
Digital signal processing tools.
"""

import math
import itertools
from collections import abc
from warnings import warn
from types import SimpleNamespace
import numbers
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interp
import matplotlib.patches as mpatches
from pyyeti.ytools import _check_makeplot


try:
    import numba
except ImportError:
    HAVE_NUMBA = False
else:
    HAVE_NUMBA = True


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def resample(data, p, q, *, axis=-1, beta=14, pts=10, t=None, getfir=False):
    """
    Change sample rate of data by a rational factor using Lanczos
    resampling.

    Parameters
    ----------
    data : nd array_like
        Data to be resampled. The resampling is done along axis
        `axis`.
    p : integer
        The upsample factor.
    q : integer
        The downsample factor.
    axis : int, optional
        Axis along which to operate.
    beta : scalar
        The beta value for the Kaiser window. See
        :func:`scipy.signal.windows.kaiser`.
    pts : integer
        Number of points in data to average from each side of current
        data point. For example, if ``pts == 10``, a total 21 points
        of original data are used for averaging.
    t : array_like
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `data` and the new
        (resampled) positions are returned.
    getfir : bool
        If True, the FIR filter coefficients are returned.

    Returns
    -------
    rdata : nd ndarray
        The resampled data. If the signal(s) in `data` have `n`
        samples, the signal(s) in `rdata` have ``ceil(n*p/q)``
        samples.
    tnew : 1d ndarray; optional
        The resampled positions, same length as `rdata`. Only
        returned if `t` is input.
    fir : 1d ndarray; optional
        The FIR filter coefficients.
        ``len(fir) = 2*pts*max(p, q)/gcd(p, q) + 1``. ``gcd(p, q)`` is
        the greatest common denominator of `p` and `q`. `fir` is only
        returned if `getfir` is True.

    Notes
    -----
    This routine takes care not to introduce new frequency content
    when upsampling and not to alias higher frequency content into the
    lower frequencies when downsampling. It performs these basic
    steps:

        0. Removes the mean from `data`: ``mdata = data-mean(data)``.

        1. Inserts ``p-1`` zeros after every point in `mdata`.

        2. Forms an averaging, anti-aliasing FIR filter based on the
           'sinc' function and the Kaiser window to filter `mdata`.

           a. Each original point gets retained as-is (it gets
              multiplied by 1.0 and the other original data points
              get multiplied by 0.0).

           b. The new zero points are a weighted average (by distance)
              of the original points. The averaging coefficients (in
              the FIR filter) sum to 1.0

           c. The frequency cutoff for the filter is the minimum of
              the original sample rate divided by two, or the final
              sample rate divided by two. This ensures there is no
              aliasing.

        3. Downsamples by selecting every `q` points in filtered
           data.

        4. Adds the mean value(s) back on to final result.

    Using more points in the averaging results in more accuracy at the
    cost of run-time. From tests, upsampling with this routine
    approaches the output of :func:`scipy.signal.resample` as `pts` is
    increased except near the end points, where the periodic nature of
    the FFT (used in :func:`scipy.signal.resample`) becomes evident.
    See the first example below.

    When upsampling by a factor, the original points in `data` are
    retained.

    See also
    --------
    :func:`scipy.signal.resample`

    Examples
    --------
    The first example upsamples a hand-picked data vector to show
    how this routine compares and contrasts with the FFT method
    in :func:`scipy.signal.resample`.

    .. plot::
        :context: close-figs

        >>> from pyyeti import dsp
        >>> import scipy.signal as signal
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import numpy.fft as fft
        >>> p = 3
        >>> q = 1
        >>> data = [0., -0.08, 0.8,  1.6, -1.7, -1.8, 2, 0., 0.7, -1.5]
        >>> n = len(data)
        >>> x = np.arange(n)
        >>> upx = np.arange(n*p)/p
        >>> _ = plt.figure('Example', figsize=(10, 8), clear=True,
        ...                layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(x, data, 'o-', label='Original')
        >>> res = {}
        >>> for pts, m in zip([3, 5, 7, 10],
        ...                   ['^', 'v', '<', '>']):
        ...     res[pts], up2 = dsp.resample(data, p, q, pts=pts, t=x)
        ...     lab = f'Resample, pts={pts}'
        ...     _ = plt.plot(upx, res[pts], '-', label=lab, marker=m)
        >>> resfft = signal.resample(data, p*n)
        >>> _ = plt.plot(upx, resfft, '-D', label='scipy.signal.resample')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Signal')
        >>> _ = plt.xlabel('Time')
        >>> _ = plt.subplot(212)
        >>> n2 = len(upx)
        >>> frq = fft.rfftfreq(n, 1)
        >>> frqup = fft.rfftfreq(n2, 1/p)
        >>> _ = plt.plot(frq,   2*np.abs(fft.rfft(data))/n,
        ...          label='Original')
        >>> _ = plt.plot(frqup, 2*np.abs(fft.rfft(res[5]))/n2,
        ...          label='Resample, pts=5')
        >>> _ = plt.plot(frqup, 2*np.abs(fft.rfft(res[10]))/n2,
        ...          label='Resample, pts=10')
        >>> _ = plt.plot(frqup, 2*np.abs(fft.rfft(resfft))/n2,
        ...          label='scipy.signal.resample')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('FFT Mag.')
        >>> _ = plt.xlabel('Frequency - fraction of original sample rate')
        >>> np.allclose(up2, upx)
        True
        >>> np.allclose(res[5][::p], data)  # original data still here
        True

    .. plot::
        :context: close-figs

        For another example, downsample some random data:

        >>> p = 1
        >>> q = 5
        >>> n = 530
        >>> rng = np.random.default_rng(seed=10)
        >>> data = rng.normal(size=n)
        >>> x = np.arange(n)
        >>> dndata, dnx = dsp.resample(data, p, q, t=x)
        >>> fig = plt.figure('Example 2', clear=True,
        ...                  layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(x, data, 'o-', label='Original', alpha=0.3)
        >>> _ = plt.plot(dnx, dndata, label='Resample', lw=2)
        >>> resfft = signal.resample(data, int(np.ceil(n/q)))
        >>> _ = plt.plot(
        ...         dnx, resfft, label='scipy.signal.resample', lw=2
        ...     )
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Signal')
        >>> _ = plt.xlabel('Time')
        >>> _ = plt.subplot(212)
        >>> n2 = len(dnx)
        >>> frq = fft.rfftfreq(n, 1)
        >>> frqdn = fft.rfftfreq(n2, q)
        >>> _ = plt.plot(
        ...         frq, 2*np.abs(fft.rfft(data))/n, 'o-', alpha=.3
        ...     )
        >>> _ = plt.plot(frqdn, 2*np.abs(fft.rfft(dndata))/n2, lw=2)
        >>> _ = plt.plot(frqdn, 2*np.abs(fft.rfft(resfft))/n2, lw=2)
        >>> _ = plt.title('FFT Mag.')
        >>> xlbl = 'Frequency - fraction of original sample rate'
        >>> _ = plt.xlabel(xlbl)
        >>> _ = plt.xlim(-0.01, 0.13)
    """
    data = np.atleast_1d(data)
    ln = data.shape[axis]

    if not (axis == -1 or axis == data.ndim - 1):
        # Move the axis containing the data to the end
        data = np.swapaxes(data, axis, data.ndim - 1)

    # setup FIR filter for upsampling given the following parameters:
    gf = math.gcd(p, q)
    p = p // gf
    q = q // gf

    M = 2 * pts * max(p, q)
    w = signal.windows.kaiser(M + 1, beta)
    # w = signal.hann(M+1)
    n = np.arange(M + 1)

    # compute cutoff relative to highest sample rate (P*sr where sr=1)
    #  eg, if Q = 1, cutoff = 0.5 of old sample rate = 0.5/P of new
    #      if P = 1, cutoff = 0.5 of new sample rate = 0.5/Q of old
    cutoff = min(1 / q, 1 / p) / 2
    # sinc(x) = sin(pi*x)/(pi*x)
    s = 2 * cutoff * np.sinc(2 * cutoff * (n - M / 2))
    fir = p * w * s
    m = np.mean(data, axis=-1, keepdims=True)

    # insert zeros
    shape = [*data.shape]
    if p > 1:
        shape[-1] = ln * p
        updata1 = np.zeros(shape)
        updata1[..., ::p] = data - m
    else:
        updata1 = data - m

    # take care of lag by shifting with zeros:
    nz = M // 2
    shape[-1] = nz
    z = np.zeros(shape)
    updata1 = np.concatenate((z, updata1, z), axis=-1)
    updata = signal.lfilter(fir, 1, updata1, axis=-1)
    updata = updata[..., M:]

    # downsample:
    n = int(np.ceil(ln * p / q))
    if q > 1:
        shape[-1] = n
        RData = np.zeros(shape)
        RData = updata[..., ::q] + m
    else:
        RData = updata + m

    if not (axis == -1 or axis == data.ndim - 1):
        # Move the axis back to where it was
        RData = np.swapaxes(RData, axis, data.ndim - 1)

    if t is None:
        if getfir:
            return RData, fir
        return RData
    tnew = np.arange(n) * (t[1] - t[0]) * ln / n + t[0]
    if getfir:
        return RData, tnew, fir
    return RData, tnew


def _get_timedata(data):
    """
    Check for value time/data input for :func:`fixtime` and
    :func:`aligntime`
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                "incorrectly sized ndarray for "
                "time/data input (must be 2d with 2 "
                "columns)"
            )
        t = data[:, 0]
        d = data[:, 1]
        isndarray = True
    else:
        if len(data) != 2:
            raise ValueError("incorrectly defined time/data input")
        t, d = np.atleast_1d(*data)
        if len(t) != len(d):
            raise ValueError("time and data vectors are incompatibly sized")
        isndarray = False
    return t, d, isndarray


def _get_prev_index(vec, val):
    """Finds previous index for scalar `val`"""
    p = np.searchsorted(vec, val) - 1
    if p < 0:
        return 0
    return p


def exclusive_sgfilter(x, n, exclude_point="first", axis=-1):
    """
    1-d moving average that excludes selected point

    More specifically, this is a 0th order 1-d Savitzky-Golay FIR
    filter that has been modified such that it excludes the selected
    point. This is helpful to find outliers.

    Parameters
    ----------
    x : nd array_like
        Array to filter
    n : odd integer
        Number of points for filter; if even, it is reset to ``n+1``
    exclude_point : string or int or None; optional
        Defines which point to exclude in each moving average window.
        If integer, it must be in [0, n), specifying the point to
        exclude. If string, it must be 'first', 'middle', or 'last'
        (which is the same as ``0``, ``n // 2``, and ``n-1``,
        respectively). If None, no point will be excluded (this is
        primarily for testing and should match a standard 0th order
        Savitzky-Golay filter).
    axis : integer; optional
        Axis along which to apply filter; each subarray along this
        axis is filtered.  For example, to filter each column in a 2d
        array, set `axis` to 0.

    Returns
    -------
    x_f : nd ndarray
        Filtered version of `x`

    Notes
    -----
    The end windows cannot all exclude the selected point. For
    example, if the excluded point is the first point in each window,
    the last ``n-1`` windows cannot follow this rule. To illustrate,
    let signal `x` be ``np.arange(9)``, `n` be 5, and `exclude_point`
    be 'first' (or 0). In this scenario, the last 4 windows cannot
    exclude the first point (the "-" denotes the excluded point for
    each window)::

              x = [0, 1, 2, 3, 4, 5, 6, 7, 8]

         1st ave:  -  +  +  +  +
         2nd ave:     -  +  +  +  +
         3rd ave:        -  +  +  +  +
         4th ave:           -  +  +  +  +
         5th ave:              -  +  +  +  +
         6th ave:              +  -  +  +  +
         7th ave:              +  +  -  +  +
         8th ave:              +  +  +  -  +
         9th ave:              +  +  +  +  -

    If `exclude_point` is 'middle' or ``n // 2``::

              x = [0, 1, 2, 3, 4, 5, 6, 7, 8]

         1st ave:  -  +  +  +  +
         2nd ave:  +  -  +  +  +
         3rd ave:  +  +  -  +  +
         4th ave:     +  +  -  +  +
         5th ave:        +  +  -  +  +
         6th ave:           +  +  -  +  +
         7th ave:              +  +  -  +  +
         8th ave:              +  +  +  -  +
         9th ave:              +  +  +  +  -

    If `exclude_point` is 'last' or ``n-1``::

              x = [0, 1, 2, 3, 4, 5, 6, 7, 8]

         1st ave:  -  +  +  +  +
         2nd ave:  +  -  +  +  +
         3rd ave:  +  +  -  +  +
         4th ave:  +  +  +  -  +
         5th ave:  +  +  +  +  -
         6th ave:     +  +  +  +  -
         7th ave:        +  +  +  +  -
         8th ave:           +  +  +  +  -
         9th ave:              +  +  +  +  -

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti.dsp import exclusive_sgfilter
    >>> x = np.arange(6.)
    >>> x[3] *= 2
    >>> x
    array([ 0.,  1.,  2.,  6.,  4.,  5.])
    >>> for point in ('first', 'middle', 'last'):
    ...     print(exclusive_sgfilter(x, 3, exclude_point=point))
    [ 1.5  4.   5.   4.5  5.5  5. ]
    [ 1.5  1.   3.5  3.   5.5  5. ]
    [ 1.5  1.   0.5  1.5  4.   5. ]

    Equivalent run using indexes:

    >>> for point in (0, 1, 2):
    ...     print(exclusive_sgfilter(x, 3, exclude_point=point))
    [ 1.5  4.   5.   4.5  5.5  5. ]
    [ 1.5  1.   3.5  3.   5.5  5. ]
    [ 1.5  1.   0.5  1.5  4.   5. ]

    If `exclude_point` is None, this is the same as a normal 0th
    order Savitzky-Golay filter:

    >>> from scipy.signal import savgol_filter
    >>> savgol_filter(x, 3, polyorder=0)
    array([ 1.,  1.,  3.,  4.,  5.,  5.])
    >>> exclusive_sgfilter(x, 3, exclude_point=None)
    array([ 1.,  1.,  3.,  4.,  5.,  5.])
    """
    x = np.atleast_1d(x)
    n = min(x.size - 1, n) | 1
    b = np.empty(n)
    if isinstance(exclude_point, str):
        if exclude_point == "first":
            n_pt = 0
        elif exclude_point == "middle":
            n_pt = n // 2
        elif exclude_point == "last":
            n_pt = n - 1
        else:
            raise ValueError("invalid `exclude_point` string")
    else:
        n_pt = exclude_point
    if n_pt is None:
        b[:] = 1 / n
        n_pt = n // 2
    else:
        if not 0 <= n_pt <= n - 1:
            raise ValueError("invalid `exclude_point` integer")
        b[:] = 1 / (n - 1)
        # b is applied in reverse: y[1] = b[0]*x[1] + b[1]*x[0]
        b[n - n_pt - 1] = 0.0

    if not (axis == -1 or axis == x.ndim - 1):
        # Move the axis containing the data to the end
        x = np.swapaxes(x, axis, x.ndim - 1)

    # Append pieces of x onto the front and back so the averages on
    # the ends work out properly. For example, if n is 5 and x is:
    #      x = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # and n_pt is 2, then x2 is:
    #      x2 = [3, 4,  0, 1, 2, 3, 4, 5, 6, 7, 8,  4, 5]
    # 1st ave:   +  +   -  +  +
    # 2nd ave:      +   +  -  +  +
    # 3rd ave:          +  +  -  +  +
    # ... last ave:                       +  +  -   +  +

    # and n_pt is 1, then x2 is:
    #      x2 = [4,  0, 1, 2, 3, 4, 5, 6, 7, 8,  4, 5, 6]
    # 1st ave:   +   -  +  +  +
    # 2nd ave:       +  -  +  +  +
    # 3rd ave:          +  -  +  +  +
    # ... last ave:                       +  -   +  +  +
    x2 = np.concatenate((x[..., n - n_pt : n], x, x[..., -n : -(n_pt + 1)]), axis=-1)
    d = signal.lfilter(b, 1, x2)[..., n - 1 :]
    if not (axis == -1 or axis == x.ndim - 1):
        # Move the axis back to where it was
        d = np.swapaxes(d, axis, x.ndim - 1)
    return d


def _get_min_limit(x, n, threshold_sigma, threshold_value):
    if threshold_value is not None:
        return threshold_value
    ave = exclusive_sgfilter(x, n, exclude_point=None)
    return threshold_sigma * np.std(x - ave)


def _sweep_out_priors(y, i, limit, ave):
    # see if we can consider points before the detected outlier
    # also as outliers:
    pv = [i]
    lim = limit[i]
    av = ave[i]
    for k in range(i - 1, -1, -1):
        if abs(y[k] - av) <= lim:
            break
        pv.append(k)
    pv.reverse()
    return pv


def _sweep_out_nexts(y, i, limit, ave):
    # see if we can consider points after the detected outlier
    # also as outliers:
    pv = [i]
    lim = limit[i]
    av = ave[i]
    for k in range(i + 1, y.size):
        if abs(y[k] - av) <= lim:
            break
        pv.append(k)
    return pv


def _get_stats_full(y, n, sigma, min_limit, xp):
    ave = exclusive_sgfilter(y, n, exclude_point=xp)
    y_delta = abs(y - ave)
    var = exclusive_sgfilter(y**2, n, exclude_point=xp) - ave**2
    # use abs to care of negative numerical zeros:
    std = np.sqrt(abs(var))
    limit = np.fmax(sigma * std, min_limit)
    return ave, y_delta, var, std, limit


def _outs_first(y, n, sigma, min_limit, xp, ave, y_delta, var, std, limit):
    PV = y_delta > limit
    while True:
        pv = PV.nonzero()[0]
        if pv.size == 0:
            yield None, ave + limit, ave - limit  # we're done
        # keep only last one ... previous ones can change
        pv = _sweep_out_priors(y, pv[-1], limit, ave)
        yield pv, ave + limit, ave - limit
        i, j = pv[0], pv[-1]
        if i == 0:
            yield None, ave + limit, ave - limit  # we're done
        PV[i : j + 1] = False
        # To determine if point before i is a spike, need n-1
        # valid points after j:
        k = min(y.size, j + n)
        count = k - (j + 1)  # n-1 if away from end
        # shift good points backward in time to get rid of spikes:
        #            <---
        # ......ssss+++++   ==>  ......+++++
        #       i  j
        y[i : i + count] = y[j + 1 : k]

        # update only sections that need it: from i-n to i
        j = i
        i = max(i - n, 0)
        ave[i:j] = exclusive_sgfilter(y[i:k], n, exclude_point=xp)[: j - i]
        y_delta[i:j] = abs(y[i:j] - ave[i:j])
        avsq = exclusive_sgfilter(y[i:k] ** 2, n, exclude_point=xp)[: j - i]
        var[i:j] = avsq - ave[i:j] ** 2
        # use abs to care of negative numerical zeros:
        std[i:j] = np.sqrt(abs(var[i:j]))
        limit[i:j] = np.fmax(sigma * std[i:j], min_limit)
        PV[i:j] = y_delta[i:j] > limit[i:j]


def _outs_last(y, n, sigma, min_limit, xp, ave, y_delta, var, std, limit):
    PV = y_delta > limit
    while True:
        pv = PV.nonzero()[0]
        if pv.size == 0:
            yield None, ave + limit, ave - limit  # we're done
        # keep only first one ... later ones can change
        pv = _sweep_out_nexts(y, pv[0], limit, ave)
        yield pv, ave + limit, ave - limit
        i, j = pv[0], pv[-1]
        if j == y.size - 1:
            yield None, ave + limit, ave - limit  # we're done
        PV[i : j + 1] = False
        # To determine if point after j is a spike, need n-1
        # valid points before i:
        k = max(0, i - n + 1)
        count = i - k  # n-1 if away from start
        # shift good points forward in time to get rid of spikes:
        #  --->
        # ......ssss+++++   ==>  ......+++++
        #       i  j
        y[j - count + 1 : j + 1] = y[k:i]

        # update only sections that need it: from j to j+n
        i = j
        j = min(j + n, y.size)
        m = i - j  # -(j-i) ... keep last j-i points
        ave[i:j] = exclusive_sgfilter(y[k:j], n, exclude_point=xp)[m:]
        y_delta[i:j] = abs(y[i:j] - ave[i:j])
        avsq = exclusive_sgfilter(y[k:j] ** 2, n, exclude_point=xp)[m:]
        var[i:j] = avsq - ave[i:j] ** 2
        # use abs to care of negative numerical zeros:
        std[i:j] = np.sqrt(abs(var[i:j]))
        limit[i:j] = np.fmax(sigma * std[i:j], min_limit)
        PV[i:j] = y_delta[i:j] > limit[i:j]


def _outs_gen(y, n, sigma, min_limit, xp, ave, y_delta, limit):
    PV = np.zeros(y.size, bool)
    hi = ave + limit
    lo = ave - limit
    while True:
        pv = y_delta > limit
        if not pv.any():
            yield None, hi, lo  # we're done
        PV[~PV] = pv
        yield PV.nonzero()[0], hi, lo
        y = y[~pv]
        ave, y_delta, var, std, limit = _get_stats_full(y, n, sigma, min_limit, xp)
        hi[~PV] = ave + limit
        lo[~PV] = ave - limit


def _find_outlier_peaks(y, n, sigma, min_limit, xp):
    ave, y_delta, var, std, limit = _get_stats_full(y, n, sigma, min_limit, xp)
    if xp in ("first", 0):
        y = y.copy()
        yield from _outs_first(
            y, n, sigma, min_limit, xp, ave, y_delta, var, std, limit
        )
    elif xp in ("last", n - 1):
        y = y.copy()
        yield from _outs_last(y, n, sigma, min_limit, xp, ave, y_delta, var, std, limit)
    else:
        yield from _outs_gen(y, n, sigma, min_limit, xp, ave, y_delta, limit)


def despike(
    x,
    n,
    sigma=8.0,
    maxiter=-1,
    threshold_sigma=2.0,
    threshold_value=None,
    exclude_point="first",
    **kwargs,
):
    """
    Delete outlier data points from signal

    Parameters
    ----------
    x : 1d array_like
        Signal to de-spike.
    n : odd integer
        Number of points for moving average; if even, it is reset to
        ``n+1``. If greater than the dimension of `x`, it is reset to
        the dimension or 1 less.
    sigma : real scalar; optional
        Number of standard deviations beyond which a point is
        considered an outlier. The default value is quite high; this
        is possible because the point itself is excluded from the
        calculations.
    maxiter : integer; optional
        Maximum number of iterations of outlier removal allowed. If
        `exclude_point` is 'first', only the last spike is removed on
        each iteration; if it is 'last', only the first spike is
        removed on each iteration. It is done this way because
        removing a spike can expose other points as spikes (but didn't
        appear to be because the removed spike was present). If <= 0,
        there is no set limit and the looping will stop when no more
        outliers are detected. Routine will always run at least 1 loop
        (setting `maxiter` to 0 is the same as setting it to 1).
    threshold_sigma : scalar; optional
        Number of standard deviations below which all data is kept.
        This standard deviation is of the entire input signal minus
        the moving average (using a window of `n` size). This value
        exists to avoid deleting small deviations such as bit
        toggles. Set to 0.0 to not use a threshold. `threshold_value`
        overrides `threshold_sigma` if it is not None.
    threshold_value : scalar or None; optional
        Optional method for specifying a minimum threshold. If not
        None, this scalar is used as an absolute minimum deviation
        from the moving average for a value to be considered a spike.
        Overrides `threshold_sigma`. Set to 0.0 to not use a
        threshold.
    exclude_point : string or int or None; optional
        Defines where, within each window, the point that is being
        considered as a potential outlier is. For example, 'first'
        compares the first point in each window the rest in that
        window to test if it is an outlier. This option is passed
        directly to :func:`exclusive_sgfilter`. If integer, it must be
        in [0, n), specifying the point to exclude. If string, it must
        be 'first', 'middle', or 'last' (which is the same as ``0``,
        ``n // 2``, and ``n-1``, respectively). If None, the point
        will be in the middle of the window and will not be excluded
        from the statistics (this is not recommended).
    **kwargs : other args are ignored
        This is here to accommodate :func:`fixtime`.

    Returns
    -------
    A SimpleNamespace with the members:

    x : 1d ndarray
        Despiked version of input `x`. Will be shorter than input `x`
        if any spikes were deleted; otherwise, it will equal input
        `x`.
    pv : bool 1d ndarray; same size as input `x`
        Has True where an outlier was detected
    hilim : 1d ndarray; same size as input `x`
        This is the upper limit: ``mean + sigma*std``
    lolim : 1d ndarray; same size as input `x`
        This is the lower limit: ``mean - sigma*std``
    niter : integer
        Number of iterations executed

    Notes
    -----
    Uses :func:`exclusive_sgfilter` to exclude the point being tested
    from the moving average and the moving standard deviation
    calculations. Each point is tested. The points near the ends of
    the signal may not be at the requested position in the window (see
    :func:`exclusive_sgfilter` for more information on this).

    To not use a threshold, set `threshold_sigma` to 0.0 (or set
    `threshold_value` to 0.0).

    .. note::
        If you plan to use both :func:`fixtime` and :func:`despike`,
        it is recommended that you let :func:`fixtime` call
        :func:`despike` (via the `delspikes` option) instead of
        calling it directly. This is preferable because the ideal time
        to run :func:`despike` is in the middle of :func:`fixtime`:
        after drop-outs have been deleted but before gaps are filled.

    Examples
    --------
    Compare `exclude_point` 'first' and 'middle' options. An
    explanation follows:

    >>> import numpy as np
    >>> from pyyeti import dsp
    >>> x = [1, 1, 1, 1, 5, 5, 1, 1, 1, 1]
    >>> s = dsp.despike(x, n=5, exclude_point='first')
    >>> s.x
    array([1, 1, 1, 1, 1, 1, 1, 1])
    >>> s = dsp.despike(x, n=5, exclude_point='middle')
    >>> s.x
    array([1, 1, 1, 1, 5, 5, 1, 1, 1, 1])

    The two 5 points get deleted when using 'first' but not when using
    'middle'. This is logical because, when using 'first', the second
    5 is compared to following four 1 values (the window is
    ``[5, 1, 1, 1, 1]``. The second loop then catches the other 5. But
    when 'middle' is used, the window for the first 5 is
    ``[1, 1, 5, 5, 1]`` and the window for the second 5 is
    ``[1, 5, 5, 1, 1]``. For both points, the other 5 in the window
    prevents the center 5 from being considered an outlier.

    For another example, make up some data and, with carefully chosen
    inputs, demonstrate how the routine runs by plotting one iteration
    at a time:

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> np.set_printoptions(linewidth=65)
        >>> x = [100, 2, 3, -4, 25, -6, 6, 3, -2, 4, -2, -100]
        >>> _ = plt.figure('Example', figsize=(8, 11), clear=True,
        ...                layout='constrained')
        >>> for i in range(5):
        ...     s = dsp.despike(x, n=9, sigma=2, maxiter=1,
        ...                     threshold_sigma=0.1,
        ...                     exclude_point='middle')
        ...     _ = plt.subplot(5, 1, i+1)
        ...     _ = plt.plot(x)
        ...     _ = plt.plot(s.hilim, 'k--')
        ...     _ = plt.plot(s.lolim, 'k--')
        ...     _ = plt.title(f'Iteration {i+1}')
        ...     x = s.x
        >>> s.x
        array([ 2,  3,  6,  3, -2,  4, -2])

        Run all iterations at once to see what ``s.pv`` looks like:

        >>> x = [100, 2, 3, -4, 25, -6, 6, 3, -2, 4, -2, -100]
        >>> s = dsp.despike(x, n=9, sigma=2,
        ...                 threshold_sigma=0.1,
        ...                 exclude_point='middle')
        >>> s.x
        array([ 2,  3,  6,  3, -2,  4, -2])
        >>> s.pv
        array([ True, False, False,  True,  True,  True, False, False,
               False, False, False,  True], dtype=bool)
    """
    x = np.atleast_1d(x)
    if x.ndim > 1:
        raise ValueError("`x` must be 1d")
    if n > x.size:
        n = x.size - 1
    min_limit = _get_min_limit(x, n, threshold_sigma, threshold_value)

    PV = np.zeros(x.shape, bool)
    # start generator:
    gen = _find_outlier_peaks(x, n, sigma, min_limit, xp=exclude_point)
    for i, (pv, hi, lo) in zip(itertools.count(1), gen):
        if pv is None:
            break
        PV[pv] = True
        if maxiter > 0 and i >= maxiter:
            break
    return SimpleNamespace(x=x[~PV], pv=PV, hilim=hi, lolim=lo, niter=i)


def _sweep_out_priors_diff(y, i, limit, ave):
    # see if we can consider points before the detected outlier
    # also as outliers:
    pv = [i]
    lim = limit[i]
    av = ave[i]
    next_y = y[i + 1]
    for k in range(i - 1, -1, -1):
        new_dy = next_y - y[k]
        if abs(new_dy - av) <= lim:
            break
        pv.append(k)
    pv.reverse()
    return pv


def _sweep_out_nexts_diff(y, i, limit, ave):
    # see if we can consider points after the detected outlier
    # also as outliers:
    pv = [i]
    lim = limit[i - 1]
    av = ave[i - 1]
    prev_y = y[i - 1]
    for k in range(i + 1, y.size):
        new_dy = y[k] - prev_y
        if abs(new_dy - av) <= lim:
            break
        pv.append(k)
    return pv


def _outs_first_diff(
    y, dy, n, sigma, min_limit, xp, ave, dy_delta, var, std, limit, dpv
):
    while True:
        if dpv.any():
            # keep only last one ... previous ones can change
            i = dpv.nonzero()[0][-1]
            # since we're grabbing last spike in dy, that index
            # is also what we need for y:
            # dy -> y
            # 0 -> 1-0
            # 1 -> 2-1
            # 2 -> 3-2
            # say 1 of dy is last spike ... 2 isn't. So 3-2
            # of original is okay. spike in original has to be 1.
            pv = _sweep_out_priors_diff(y, i, limit, ave)
        else:
            pv = None
        yield pv

        i, j = pv[0], pv[-1]
        if i == 0:
            yield None  # we're done
        dpv[i : j + 1] = False
        # To determine if point before i is a spike, need n-1
        # valid points after j:
        k = min(y.size, j + n)
        count = k - (j + 1)  # n-1 if away from end
        # shift good points backward in time to get rid of
        # spikes:
        #            <---
        # ......ssss+++++   ==>  ......+++++
        #       i  j
        y[i : i + count] = y[j + 1 : k]

        # update only sections that need it: from i-n to i
        j = i
        i = max(i - n, 0)
        dy[i:k] = np.diff(y[i : k + 1])
        ave[i:j] = exclusive_sgfilter(dy[i:k], n, exclude_point=xp)[: j - i]
        dy_delta[i:j] = abs(dy[i:j] - ave[i:j])
        avsq = exclusive_sgfilter(dy[i:k] ** 2, n, exclude_point=xp)[: j - i]
        var[i:j] = avsq - ave[i:j] ** 2
        # use abs to care of negative numerical zeros:
        std[i:j] = np.sqrt(abs(var[i:j]))
        limit[i:j] = np.fmax(sigma * std[i:j], min_limit)
        dpv[i:j] = dy_delta[i:j] > limit[i:j]


def _outs_last_diff(
    y, dy, n, sigma, min_limit, xp, ave, dy_delta, var, std, limit, dpv
):
    while True:
        if dpv.any():
            # keep only first one ... later ones can change
            i = dpv.nonzero()[0][0]
            # since we're grabbing first spike in dy, that index
            # plus 1 is what we need for y:
            # dy -> y
            # 0 -> 1-0
            # 1 -> 2-1
            # 2 -> 3-2
            # say 1 of dy is first spike ... 0 isn't. So 1-0
            # of original is okay. spike in original has to be 2.
            pv = _sweep_out_nexts_diff(y, i + 1, limit, ave)
        else:
            pv = None
        yield pv

        i, j = pv[0], pv[-1]
        if j == dy.size:
            yield None  # we're done
        dpv[i - 1 : j] = False
        # To determine if point after j is a spike, need n-1
        # valid points before i:
        k = max(0, i - n + 1)
        count = i - k  # n-1 if away from start
        # shift good points forward in time to get rid of spikes:
        #  --->
        # ......ssss+++++   ==>  ......+++++
        #       i  j
        y[j - count + 1 : j + 1] = y[k:i]

        # update only sections that need it: from j to j+n
        i = j
        j = min(j + n, dy.size)
        m = i - j  # -(j-i) ... keep last j-i points
        dy[k:j] = np.diff(y[k : j + 1])
        ave[i:j] = exclusive_sgfilter(dy[k:j], n, exclude_point=xp)[m:]
        dy_delta[i:j] = abs(dy[i:j] - ave[i:j])
        avsq = exclusive_sgfilter(dy[k:j] ** 2, n, exclude_point=xp)[m:]
        var[i:j] = avsq - ave[i:j] ** 2
        # use abs to care of negative numerical zeros:
        std[i:j] = np.sqrt(abs(var[i:j]))
        limit[i:j] = np.fmax(sigma * std[i:j], min_limit)
        dpv[i:j] = dy_delta[i:j] > limit[i:j]


def _find_outlier_peaks_diff(y, dy, n, sigma, min_limit, xp):
    ave = exclusive_sgfilter(dy, n, exclude_point=xp)
    dy_delta = abs(dy - ave)
    var = exclusive_sgfilter(dy**2, n, exclude_point=xp) - ave**2
    # use abs to care of negative numerical zeros:
    std = np.sqrt(abs(var))
    limit = np.fmax(sigma * std, min_limit)
    dpv = dy_delta > limit
    if xp in ("first", 0):
        yield from _outs_first_diff(
            y, dy, n, sigma, min_limit, xp, ave, dy_delta, var, std, limit, dpv
        )
    elif xp in ("last", n - 1):
        yield from _outs_last_diff(
            y, dy, n, sigma, min_limit, xp, ave, dy_delta, var, std, limit, dpv
        )
    else:
        raise ValueError("invalid `exclude_point` for :func:`despike_diff` routine")


def despike_diff(
    x,
    n,
    sigma=8.0,
    maxiter=-1,
    threshold_sigma=2.0,
    threshold_value=None,
    exclude_point="first",
    **kwargs,
):
    """
    Delete outlier data points from signal based on level changes

    Parameters
    ----------
    x : 1d array_like
        Signal to de-spike.
    n : odd integer
        Number of points for moving average; if even, it is reset to
        ``n+1``. If greater than the dimension of `x`, it is reset to
        the dimension or 1 less.
    sigma : real scalar; optional
        Number of standard deviations beyond which a point is
        considered an outlier. The default value is quite high; this
        is possible because the point itself is excluded from the
        calculations.
    maxiter : integer; optional
        Maximum number of iterations of outlier removal allowed. If
        `exclude_point` is 'first', only the last spike is removed on
        each iteration; if it is 'last', only the first spike is
        removed on each iteration. It is done this way because
        removing a spike can expose other points as spikes (but didn't
        appear to be because the removed spike was present). If <= 0,
        there is no set limit and the looping will stop when no more
        outliers are detected. Routine will always run at least 1 loop
        (setting `maxiter` to 0 is the same as setting it to 1).
    threshold_sigma : scalar; optional
        Number of standard deviations below which all data is kept.
        This standard deviation is computed from `x`. Let ``dx =
        np.diff(x)``, the standard deviation is ``std(dx -
        moving_average(dx))``. The moving average uses a window of `n`
        size. This value exists to avoid deleting small deviations
        such as bit toggles. Set to 0.0 to not use a
        threshold. `threshold_value` overrides `threshold_sigma` if it
        is not None.
    threshold_value : scalar or None; optional
        Optional method for specifying a minimum threshold. If not
        None, this scalar is used as an absolute minimum deviation
        from the moving average for a value to be considered a spike.
        Overrides `threshold_sigma`. Set to 0.0 to not use a
        threshold.
    exclude_point : string or int; optional
        Defines where, within each window, the point that is being
        considered as a potential outlier is. For this routine,
        `exclude_point` must be either 'first' (or 0) or 'last' (or
        ``n-1``). For example, 'first' compares the first point in
        each window the rest in that window to test if it is an
        outlier.
    **kwargs : other args are ignored
        This is here to accommodate :func:`fixtime`.

    Returns
    -------
    A SimpleNamespace with the members:

    x : 1d ndarray
        Despiked version of input `x`. Will be shorter than input `x`
        if any spikes were deleted; otherwise, it will equal input
        `x`.
    pv : bool 1d ndarray; same size as input `x`
        Has True where an outlier was detected
    niter : integer
        Number of iterations executed

    Notes
    -----
    Uses :func:`exclusive_sgfilter` to exclude the point being tested
    from the moving average and the moving standard deviation
    calculations. Each point is tested. The points near the ends of
    the signal may not be at the requested position in the window (see
    :func:`exclusive_sgfilter` for more information on this).

    To not use a threshold, set `threshold_sigma` to 0.0 (or set
    `threshold_value` to 0.0).

    .. note::
        If you plan to use both :func:`fixtime` and
        :func:`despike_diff`, it is recommended that you let
        :func:`fixtime` call :func:`despike_diff` (via the `delspikes`
        option) instead of calling it directly. This is preferable
        because the ideal time to run :func:`despike_diff` is in the
        middle of :func:`fixtime`: after drop-outs have been deleted
        but before gaps are filled.

    Examples
    --------
    Set `threshold_value` to catch second spike but not first. The
    threshold is based on differences:

    >>> import numpy as np
    >>> from pyyeti import dsp
    >>> x = [2, 2, 2, 2, 5, 2, 2, 2, 2, 7, 2, 2, 2, 2, 2]
    >>> s = dsp.despike_diff(x, n=5, threshold_value=4)
    >>> s.x
    array([2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    >>> s.pv
    array([False, False, False, False, False, False, False, False,
           False,  True, False, False, False, False, False], dtype=bool)
    >>> s.niter
    2
    """
    x = np.atleast_1d(x)
    if x.ndim > 1:
        raise ValueError("`x` must be 1d")
    dx = np.diff(x)
    if n > dx.size:
        n = dx.size - 1
    min_limit = _get_min_limit(dx, n, threshold_sigma, threshold_value)

    PV = np.zeros(x.shape, bool)
    # start generator:
    gen = _find_outlier_peaks_diff(x.copy(), dx, n, sigma, min_limit, xp=exclude_point)
    for i, pv in zip(itertools.count(1), gen):
        if pv is None:
            break
        PV[pv] = True
        if maxiter > 0 and i >= maxiter:
            break
    return SimpleNamespace(x=x[~PV], pv=PV, niter=i)


def _chk_negsteps(t, data, negmethod):
    difft = np.diff(t)
    negs = difft < 0
    if negs.any():
        nneg = np.count_nonzero(negs)
        npos = difft.size - nneg
        if npos == 0:
            raise ValueError(
                "there are no positive steps in the entire time vector. "
                "Cannot fix this."
            )
        if negmethod == "stop":
            raise ValueError(f"There are {nneg:d} negative time steps. Stopping.")
        if negmethod == "sort":
            warn(
                f"there are {nneg} negative time steps. Sorting the data. "
                "This may be a poor way to handle the current data, so "
                "please check the results carefully.",
                RuntimeWarning,
            )
            j = t.argsort()
            # unsort = j.argsort()
            t = t[j]
            data = data[j]
            difft = np.diff(t)
    else:
        j = None
    return t, data, difft, j


def _find_drops(d, dropval):
    dropouts = np.logical_or(np.isnan(d), np.isinf(d))
    if np.isfinite(dropval):
        d = d[~dropouts]
        dropouts[~dropouts] = abs(d - dropval) < abs(dropval) / 100
        # dropouts = np.logical_or(
        #    dropouts, abs(d-dropval) < abs(dropval)/100)
    return dropouts


def _del_loners(dropouts, n, nz=3):
    """Delete "loner-ish" points amongst dropouts.
    dropouts : 1d bool ndarray of dropouts; True for drops
    n : integer; window size
    nz : integer; number of points in `n` point range that will
        cause all points between to True values to be turned to
        True
    """
    pv = dropouts.nonzero()[0]
    if pv.size > 2:
        # delete 1-of loners first (this is necessary if
        # method="despike_diff" because a middle spike
        # will be left behind if it is smaller than the two
        # surrounding spikes)
        loners = (np.diff(pv) == 2).nonzero()[0]
        if loners.size > 0:
            dropouts[pv[loners] + 1] = True
        s = dropouts.size
        ind = []
        for i in pv[: -(nz - 1)]:
            j = i + n
            j = j if j < s else s
            d_ij = dropouts[i:j]
            if d_ij.sum() >= nz:
                while not dropouts[j - 1]:
                    j -= 1
                ind.append(slice(i, j))
        for ij in ind:
            dropouts[ij] = True


def _del_drops(olddata, dropval, delspikes):
    dropouts = _find_drops(olddata, dropval)
    if dropouts.any():
        if delspikes:
            _del_loners(dropouts, delspikes["n"])
    keep = ~dropouts
    keep = keep.nonzero()[0]
    dropouts = dropouts.nonzero()[0]
    if keep.size == 0:
        warn("there are only drop-outs!", RuntimeWarning)
    return keep, dropouts


def _del_outtimes(told, keep, delouttimes):
    t = told[keep]
    mn = t.mean()
    sig = 3 * t.std(ddof=1)
    pv = np.logical_or(t < mn - sig, t > mn + sig)
    outtimes = keep[pv]
    if pv.any():
        if delouttimes:
            warn(
                f"there are {pv.sum()} outlier times being deleted. These are"
                " times more than 3-sigma away from the mean. "
                "This may be a poor way to handle the current data, so "
                "please check the results carefully.",
                RuntimeWarning,
            )
            keep = keep[~pv]
        else:
            warn(
                f"there are {pv.sum()} outlier times that are NOT being deleted"
                " because `delouttimes` is False. These are times more than "
                "3-sigma away from the mean.",
                RuntimeWarning,
            )
    return keep, outtimes


def _sr_calcs(difft, sr, verbose):
    min_ts = difft.min()
    max_ts = difft.max()
    ave_ts = difft.mean()

    max_sr = 1 / min_ts
    min_sr = 1 / max_ts
    ave_sr = 1 / ave_ts

    # get mode of all sample rates:
    Ldiff = len(difft)
    difft2 = difft[difft != 0]
    sr_all = 1 / difft2
    sr1 = sr_all.min()
    if sr1 > 5:
        dsr = 5
    else:
        dsr = round(10 * max(sr1, 0.1)) / 10

    # pandas is very fast for computing mode:
    counts = pd.Series(np.round(sr_all / dsr)).value_counts()
    mode_sr = counts.index[0] * dsr
    mode_pct = counts.iat[0] / Ldiff * 100
    mode_ts = 1 / mode_sr

    sr_stats = np.array([max_sr, min_sr, ave_sr, mode_sr, mode_pct])
    if not sr:  # pragma: no cover
        verbose = True
    if verbose:
        print("==> Info: [min, max, ave, count (% occurrence)] time step:")
        print(
            f"==>           [{min_ts:g}, {max_ts:g}, {ave_ts:g}, "
            f"{mode_ts:g} ({mode_pct:.1f}%)]"
        )
        print("==>       Corresponding sample rates:")
        print("==>           [{:g}, {:g}, {:g}, {:g} ({:.1f}%)]".format(*sr_stats))
        print('==>       Note: "count" shows most frequent sample rate to')
        print(f"          nearest {dsr} samples/sec.")

    if mode_pct > 90 or abs(mode_sr - ave_sr) < dsr:
        defsr = round(mode_sr / dsr) * dsr
    else:
        defsr = round(ave_sr / dsr) * dsr
    if sr == "auto":
        sr = defsr
    elif not sr:  # pragma: no cover
        ssr = input(f"==> Enter desired sample rate [{defsr:g}]: ")
        if not ssr:
            sr = defsr
        else:
            sr = float(ssr)
    if verbose:
        print(f"==> Using sample rate = {sr:g}")
    return sr, sr_stats


def _prep_delspikes(delspikes):
    def _dict_default(dct, **kwargs):
        for k, v in kwargs.items():
            if k not in dct:
                dct[k] = v

    if not isinstance(delspikes, abc.MutableMapping):
        delspikes = dict()
    else:
        delspikes = dict(delspikes)  # make a copy
    _dict_default(delspikes, sigma=8, n=15, method="despike_diff", maxiter=-1)
    return delspikes


def _post_despike(pv, keep, delspikes, niter):
    if pv.any():
        _del_loners(pv, delspikes["n"])
    spikes = keep[pv]
    keep = keep[~pv]
    despike_info = SimpleNamespace(delspikes=delspikes, niter=niter)
    return keep, spikes, despike_info


def _simple_filter(olddata, keep, delspikes):
    d = olddata[keep]
    PV = np.ones(d.size, bool)
    n = delspikes["n"]
    sigma = delspikes["sigma"]
    maxiter = delspikes["maxiter"]
    for i in itertools.count(1):
        ave = exclusive_sgfilter(
            d[PV],
            n,
            # exclude_point='middle')
            exclude_point=None,
        )
        delta = d[PV] - ave
        pv = abs(delta) > sigma * np.std(delta)
        if pv.any():
            PV[PV] = ~pv
        else:
            break
        if maxiter > 0 and i >= maxiter:
            break
    return _post_despike(~PV, keep, delspikes, i + 1)


def _del_spikes(olddata, keep, delspikes):
    method = delspikes["method"]
    if method == "despike_diff":
        s = despike_diff(olddata[keep], **delspikes)
        return _post_despike(s.pv, keep, delspikes, s.niter)
    elif method == "despike":
        s = despike(olddata[keep], **delspikes)
        return _post_despike(s.pv, keep, delspikes, s.niter)
    elif method == "simple":
        return _simple_filter(olddata, keep, delspikes)
    else:
        raise ValueError(f"unknown `method` ({method})")


def _get_alldrops(
    told, olddata, sortvec, dropouts, outtimes, spikes, despike_info, delouttimes
):
    alldrops = np.zeros(told.size, bool)
    if dropouts is not None:
        alldrops[dropouts] = True
    if delouttimes:
        alldrops[outtimes] = True
    if spikes is not None:
        alldrops[spikes] = True
        _del_loners(alldrops, despike_info.delspikes["n"], 3)
    else:
        spikes = None
    keep = ~alldrops
    t = told[keep]
    data = olddata[keep]
    alldrops = alldrops.nonzero()[0]

    def _apply_sortvec(sortvec, *args):
        args = list(args)
        for i, arg in enumerate(args):
            if arg is not None:
                args[i] = np.sort(sortvec[arg])
        return args

    if sortvec is not None:
        alldrops, dropouts, outtimes, spikes = _apply_sortvec(
            sortvec, alldrops, dropouts, outtimes, spikes
        )

    return (
        t,
        data,
        SimpleNamespace(
            dropouts=dropouts, outtimes=outtimes, spikes=spikes, alldrops=alldrops
        ),
    )


def _check_dt_size(difft, dt):
    n = len(difft)
    nsmall = (difft < 0.93 * dt).sum() / n
    nlarge = (difft > 1.07 * dt).sum() / n
    for n, s1, s2 in zip((nsmall, nlarge), ("smaller", "larger"), ("low", "high")):
        if n > 0.01:
            warn(
                f"there are a large ({n * 100:.2f}%) number of time "
                f"steps {s1:s} than {dt:g} by more than 7%. Double "
                f"check the sample rate; it might be too {s2:s}.",
                RuntimeWarning,
            )


def _get_time_shifts(told, dt):
    tp = np.empty(len(told), bool)
    tp[0] = True
    tp[1:] = abs(np.diff(told) - dt) > dt / 4
    tp[:-1] |= tp[1:]
    tp[-1] = True
    tp = np.nonzero(tp)[0]

    if len(tp) - 2 > len(told) // 2:  # -2 to ignore ends
        align = False
        p = (len(tp) - 2) / len(told) * 100
        msg = (
            "there are too many time-step changes ('turning points')"
            f" ({p:.2f}%) to align the largest section."
        )
        warn(msg, RuntimeWarning)
    else:
        align = True
    return tp, align


def _mk_initial_tnew(told, sr, dt, difft):
    L = int(round((told[-1] - told[0]) * sr)) + 1
    tnew = np.arange(L) / sr + told[0]

    # get turning points and see if we should try to align:
    tp, align = _get_time_shifts(told, dt)

    if align:
        # align with the largest "good" range in `told`:
        j = np.argmax(np.diff(tp))
        told_good = told[tp[j] : tp[j + 1] + 1]
        lold = len(told_good)

        p = _get_prev_index(tnew, told_good[0] + dt / 2)
        n = _get_prev_index(tnew, told_good[-1] + dt / 2)
        tnew_good = tnew[p : n + 1]
        if (lnew := len(tnew_good)) != lold:
            lohi = "low" if lnew < lold else "high"
            warn(
                "when trying to align best sections of data, lengths of old"
                f" time vector and new time vector do not match ({lold} vs "
                f"{lnew}); sample rate used too {lohi}? Only aligning on "
                "first data point of 'good' section.",
                RuntimeWarning,
            )
            delt = told_good[0] - tnew_good[0]
        else:
            delt = np.mean(told_good - tnew_good)
        tnew += delt
    return tnew, tp


if not HAVE_NUMBA:

    def _find_closest_times(told, tnew):
        # - note this simple method doesn't work if there are large
        #   gaps in `told`:
        #       index = np.searchsorted(told, tnew - dt / 2)

        index = np.searchsorted(told, tnew)
        # told[index] <= tnew (except possibly at ends)

        lold = len(told)
        index[index == lold] = lold - 1

        delta = told[index] - tnew

        # check 1 time-step earlier to see if it's closer:
        delta_1 = told[index - 1] - tnew

        pv = abs(delta_1) <= abs(delta)
        index[pv] -= 1
        return index

    def _find_closest_previous_times(told, tnew):
        index = np.searchsorted(told, tnew) - 1
        index[index < 0] = 0
        return index

else:

    @numba.njit(cache=True)
    def _find_closest_times(told, tnew):
        # both vectors are monotonically ascending
        lold = len(told)
        lnew = len(tnew)

        # find first i such that told[i] >= tnew[0]:
        v = tnew[0]
        for i in range(lold):
            if told[i] >= v:
                break
        else:
            return np.zeros(lnew, np.int64)

        index = np.empty(lnew, np.int64)
        if i > 0 and v - told[i - 1] <= told[i] - v:
            index[0] = i - 1
        else:
            index[0] = i

        for j in range(1, lnew):
            v = tnew[j]

            for i in range(i, lold):
                if told[i] >= v:
                    break

            if i > 0 and v - told[i - 1] <= told[i] - v:
                index[j] = i - 1
            else:
                index[j] = i

        return index

    @numba.njit(cache=True)
    def _find_closest_previous_times(told, tnew):
        # both vectors are monotonically ascending
        lold = len(told)
        lnew = len(tnew)

        # find first i such that told[i] > tnew[0]:
        v = tnew[0]
        for i in range(lold):
            if told[i] > v:
                break
        else:
            return np.zeros(lnew, np.int64)

        index = np.empty(lnew, np.int64)
        if i > 0:
            index[0] = i - 1
        else:
            index[0] = i

        for j in range(1, lnew):
            v = tnew[j]

            for i in range(i, lold):
                if told[i] > v:
                    break
            else:
                i = lold

            if i > 0:
                index[j] = i - 1
            else:
                index[j] = i

        return index


def _return(t, data, alldrops, sr_stats, tp, getall, return_ndarray, despike_info):
    if return_ndarray:
        newdata = np.vstack((t, data)).T
    else:
        newdata = (t, data)
    if getall:
        fixinfo = SimpleNamespace(
            sr_stats=sr_stats, tp=tp, alldrops=alldrops, despike_info=despike_info
        )
        return newdata, fixinfo
    return newdata


def fixtime(
    olddata,
    sr=None,
    *,
    negmethod="sort",
    deldrops=True,
    dropval=-1.40130e-45,
    delouttimes=True,
    delspikes=False,
    base=None,
    hold_previous_value=False,
    previous_value_tol=1e-3,
    getall=False,
    verbose=True,
):
    """
    Process recorded data to make an even time vector.

    Parameters
    ----------
    olddata : 2d ndarray or 2-element tuple/list
        If ndarray, it must have 2 columns: ``[time, signal]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(time, signal)``
    sr : scalar, string or None; optional
        If scalar, specifies the sample rate. If 'auto', the algorithm
        chooses a "best" fit. If None, user is prompted after
        displaying some statistics on sample rate.
    negmethod : string; optional
        Specifies how to handle negative time steps:

        ===========   ==========
        `negmethod`   Action
        ===========   ==========
         "stop"       Error out
         "sort"       Sort data
        ===========   ==========

    deldrops : bool; optional
        If True, dropouts are deleted from the data; otherwise, they
        are left in.
    dropval : scalar; optional
        The numerical value of drop-outs. Note that any `np.nan` or
        `np.inf` values in the data are treated as drop-outs in any
        case.
    delouttimes : bool; optional
        If True, outlier times are deleted from the data; otherwise,
        they are left in.
    delspikes : bool or dict; optional
        If False, do not delete spikes. If True, delete spikes by
        calling :func:`despike_diff` with inputs as defined below. If
        a dict, you can take complete control. You can specify one of
        3 methods for despiking:

        ===============  ======================================
            `method`           Action
        ===============  ======================================
         "despike_diff"  Call :func:`despike_diff` (default)
         "despike"       Call :func:`despike`
         "simple"        Detect outliers by standard deviations
                         from a moving average through signal.
        ===============  ======================================

        For example, to set the method to "despike", the number of
        standard deviations to 12, the window size to 25, the maximum
        iterations to 100, and the threshold_value to 0.25::

            delspikes=dict(method='despike', sigma=12, n=25,
                           maxiter=100, threshold_value=0.25)

        Defaults are defined for some parameters (others are accepted
        from the definition of :func:`despike_diff` or :func:`despike`
        ... 'simple' only uses these three). The defaults are::

            method = 'despike_diff'
            n = 15
            sigma = 8
            maxiter = -1   # negative value means no limit

    base : scalar or None; optional
        Scalar value that new time vector would hit exactly if within
        range. If None, new time vector is aligned to longest section
        of "good" data.
    hold_previous_value : bool; optional
        If True, hold previous value instead of finding closest value
        (but see `previous_value_tol`). The default is False; find
        closest value and, in case of a tie, use previous value. For
        example::

            olddata = ([0.0, 4.0], [10.0, 20])
            t, y1 = fixtime(olddata, sr=1.0)
            t, y2 = fixtime(olddata, sr=1.0, hold_previous_value=True)

        Gives::

            t  --> array([  0.,   1.,   2.,   3.,   4.])
            y1 --> array([ 10.,  10.,  10.,  20.,  20.])
            y2 --> array([ 10.,  10.,  10.,  10.,  20.])

    previous_value_tol : float; optional
        If `hold_previous_value` is True, a new time value is
        considered equal to an old time value if it is within
        ``previous_value_tol * dt`` of it, where ``dt`` is the new
        time step. Must be within [0.0, 1.0], inclusive.
    getall : bool; optional
        If True, return `fixinfo`; otherwise only `newdata` is
        returned.
    verbose : bool; optional
        If True, sample rate statistics are printed. Note that if `sr`
        is None, `verbose` is internally set to True.

    Returns
    -------
    newdata : 2d ndarray or tuple
        Cleaned up version of `olddata`. Will be 2d ndarray if
        `olddata` was ndarray; otherwise it is a tuple:
        ``(time, data)``.

    fixinfo : SimpleNamespace; optional
        Only returned if `getall` is True. Members:

        - `sr_stats` : 1d ndarray
           Five-element vector with the sample rate statistics; useful
           to help user select best sample rate or to compare against
           `sr`.  The five elements are::

            [max_sr, min_sr, ave_sr, max_count_sr, max_count_percent]

           The `max_count_sr` is the sample rate that occurred most
           often. This is usually the 'correct' sample rate.
           `max_count_percent` gives the percent occurrence of
           `max_count_sr`.

        - `tp` : 1d ndarray
           Contains indices into old time vector of where time-step
           shifts ("turning points") were done to align the new time
           vector against the old.

        - `alldrops` : SimpleNamespace or None
           Has 1d indexing arrays into `olddata` showing the drops:

               ==========  ==========================================
               `dropouts`  shows infs, nans, and `dropvals` (None if
                           ``not deldrops``)
               `outtimes`  shows where outlier times were found in
                           `olddata` (whether they were deleted or
                           not)
               `spikes`    shows where spikes were found in `olddata`
                           (None if ``not delspikes``)
               `alldrops`  merger of `dropouts` and `spikes` plus
                           possible points in between those
               ==========  ==========================================

        - `despike_info` : SimpleNamespace or None
           If `delspikes` is True or a dict, `despike_info` contains:

               ===========  =======================================
               `delspikes`  Dict of values used for spike removal
                            (input to :func:`despike` for the
                            "despike" methods)
               `niter`      Number of iterations of spike removal
               ===========  =======================================

    Notes
    -----
    This algorithm works as follows:

       1.  Find and delete drop-outs if `deldrops` is True.

       2.  Delete outlier times if `delouttimes` is True. These are
           points with times that are more than 3 standard deviations
           away from the mean. A warning message is printed if any
           such times are found. Note that on a perfect time vector,
           the end points are at 1.73 sigma (eg:
           ``mean + 1.73*sigma = 0.5 + 1.73*0.2887 = 1.0``).

       3.  Check for positive time steps, and if there are none, error
           out.

       4.  Check the time vector for negative steps. Sort or error
           out as specified by `negmethod`. Warnings are printed in any
           case.

       5.  Compute and print sample rates for user review. Perhaps the
           most useful of these printed numbers is the one based on
           the count. :func:`numpy.histogram` is used to count which
           sample rate occurs most often (to the nearest multiple of 5
           in most cases). If there is a high percentage printed with
           that sample rate, it is likely the correct value to use (at
           least within 5 samples/second). If `sr` is not input,
           prompt user for `sr`.

       6.  Call selected despiker if requested to delete data spikes.

       7.  Count number of small time-steps defined as those that are
           less than ``0.93/sr``. If more than 1% of the steps are
           small, print a warning.

       8.  Count number of large time-steps defines as those that are
           greater than ``1.07/sr``. If more than 1% of the steps are
           large, print a warning.

       9.  Make a new, evenly spaced time vector according to the new
           sample rate that spans the range of time in `olddata`.

       10. Find the "turning points" in the old time vector. These are
           where the step differs by more than 1/4 step from the
           ideal. Will issue warning if the number of turning points
           is greater than 50% of total points.

       11. If step 10 did not issue a warning about too many turning
           points, the new time vector is shifted to align with the
           longest section of "good" old time steps.

       12. Loop over the segments defined by the turning points. Each
           segment will shifted left or right to fit with the new time
           vector. The longest section is not shifted due to step 12
           (unless that step was skipped because of too many turning
           points).

       13. If `base` is not None, the new time vector is shifted by up
           to a half time step such that it would hit `base` exactly
           (if it was in range).

       14. Fill in new data vector using best fit times. This means
           that gaps are filled with flat lines using the closest
           value if `hold_previous_value` is False, or the previous
           value if `hold_previous_value` is True. This routine
           does not do any linear interpolation.

    If despiking is not producing good results:

        1. Spikes very near the ends of the signal (in the first or
           last window) can cause trouble for the :func:`despike_diff`
           and :func:`despike` routines. If `exclude_point` is
           'first', spikes in the last window should be avoided (the
           routine works backward); conversely, if `exclude_point` is
           'last', spikes in the first window should be avoided (the
           routine works forward).

        2. Try increasing/decreasing `sigma` to make it more/less
           picky.

        3. If bit toggles or similar small spikes are being considered
           spikes (which can also make the routine take a very long
           time to run), setting `threshold_value` to a suitable value
           for the current data is often a good solution.  Increasing
           `threshold_sigma` can also protect these small spikes. Note
           that the threshold settings are not available for the
           "simple" `delspikes` method.

        4. Try a different window size.

        5. Try a different method. They all have strengths and
           weaknesses, so experiment.

    Examples
    --------
    >>> from pyyeti import dsp
    >>> t = [0., 1., 5., 6.]
    >>> y = [1., 2., 3., 4.]
    >>> tn, yn = dsp.fixtime((t, y), sr=1)
    ==> Info: [min, max, ave, count (% occurrence)] time step:
    ==>           [1, 4, 2, 1 (66.7%)]
    ==>       Corresponding sample rates:
    ==>           [1, 0.25, 0.5, 1 (66.7%)]
    ==>       Note: "count" shows most frequent sample rate to
              nearest 0.2 samples/sec.
    ==> Using sample rate = 1
    >>> tn
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.])
    >>> yn
    array([ 1.,  2.,  2.,  2.,  3.,  3.,  4.])

    Repeat, but with `hold_previous_value` set to True:

    >>> tn, yn = dsp.fixtime(
    ...    (t, y), sr=1, hold_previous_value=True, verbose=False
    ... )
    >>> tn
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.])
    >>> yn
    array([ 1.,  2.,  2.,  2.,  2.,  3.,  4.])
    """
    # begin main routine
    told, olddata, return_ndarray = _get_timedata(olddata)

    # check for negative steps:
    told, olddata, difft, sortvec = _chk_negsteps(told, olddata, negmethod)

    # prep `delspikes` if needed:
    if delspikes:
        delspikes = _prep_delspikes(delspikes)

    # check for drop outs:
    sr_stats = tp = None
    if deldrops:
        keep, dropouts = _del_drops(olddata, dropval, delspikes)
        if len(keep) == 0:
            alldrops = _get_alldrops(
                told, olddata, sortvec, dropouts, None, None, None, delouttimes
            )
            return _return(
                told, olddata, alldrops, sr_stats, tp, getall, return_ndarray, None
            )
    else:
        keep = np.arange(told.size)
        dropouts = None

    # check for outlier times ... outside 3-sigma
    keep, outtimes = _del_outtimes(told, keep, delouttimes)

    # sample rate calculations:
    difft = np.diff(told[keep])
    sr, sr_stats = _sr_calcs(difft, sr, verbose)
    dt = 1 / sr

    # delete spikes if requested:
    if delspikes:
        keep, spikes, despike_info = _del_spikes(olddata, keep, delspikes)
        difft = np.diff(told[keep])
    else:
        spikes = None
        despike_info = None

    # get partition vector for drops, spikes, and remaining "loners":
    told, olddata, alldrops = _get_alldrops(
        told, olddata, sortvec, dropouts, outtimes, spikes, despike_info, delouttimes
    )
    difft = np.diff(told)

    # check for small and large time steps:
    _check_dt_size(difft, dt)

    # make initial new time vector aligned with longest range in
    # told of "good" time steps (tp: turning points):
    tnew, tp = _mk_initial_tnew(told, sr, dt, difft)

    # build a best-fit index by finding closest new time (no
    # interpolation)
    if hold_previous_value:
        if previous_value_tol < 0.0 or previous_value_tol > 1.0:
            raise ValueError(
                "`previous_value_tol` must be in [0.0, 1.0];"
                f" it is {previous_value_tol}"
            )
        index = _find_closest_previous_times(told - dt * previous_value_tol, tnew)
    else:
        index = _find_closest_times(told, tnew)

    # fill in new data vector with closest old data:
    newdata = olddata[index]

    # if want new time to exactly hit base (if base were in range):
    if base is not None:
        t0 = tnew[0]
        t1 = base - t0 - round((base - t0) * sr) / sr
        tnew += t1
    return _return(
        tnew, newdata, alldrops, sr_stats, tp, getall, return_ndarray, despike_info
    )


def aligntime(dct, channels=None, mode="truncate", value=0):
    """
    Aligns the time vectors for specified channels in dct.

    Parameters
    ----------
    dct : dictionary
        Dictionary of channels where each channel is either 2d ndarray
        or 2-element tuple. If ndarray, it must have 2 columns:
        ``[time, signal]``. Otherwise, it must be a 2-element tuple
        or list, eg: ``(time, signal)``. See notes below.
    channels : list or None; optional
        List of names defining which channels to synchronize in time.
        If None, all channels in `dct` will be synchronized.
    mode : string; optional
        Method of aligning:

        ===========  =================================================
          `mode`     Description
        ===========  =================================================
         'truncate'  Keep only data where all channels overlap
          'expand'   Expand all channels to maximum time range of all
                     channels. Channels are expanded by stuffing in
                     `value`'s.
        ===========  =================================================

    value : scalar; optional
        Used for the 'expand' mode.

    Returns
    -------
    dctout : dictionary
        Dictionary containing only those channels specified in
        `channels`. Each channel will be a 1d ndarray. The time vector
        is also a 1d array and is named 't'.

    Notes
    -----
    This routine operates under these assumptions:

        1. The time vector for each channel is perfect (ie, after
           :func:`fixtime`)
        2. All the time vectors have the same step size
        3. They would all hit the same time point if they overlapped

    The first two assumptions are checked. The third is not checked
    and could cause indexing failures if it is not true.
    """
    # check for channels:
    if channels is not None:
        err = 0
        for item in channels:
            if item not in dct:
                err = 1
                print(f"Channel {item} not found in `dct`.")
        if err:
            raise ValueError(
                "`dct` does not contain all requested channels. See above."
            )
        parms = channels
    else:
        parms = list(dct.keys())

    # get time step:
    t, d, isarr = _get_timedata(dct[parms[0]])
    dt = (t[-1] - t[0]) / (len(t) - 1)

    if mode == "truncate":
        # loop to determine maximum overlap:
        tmin = t[0]
        tmax = t[-1]
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            if t[0] > tmax or t[-1] < tmin:
                raise ValueError("not all inputs overlap in time.")
            if not np.allclose(np.diff(t), dt):
                raise ValueError(f"not all time steps in {key} match {dt}")
            tmin = max(tmin, t[0])
            tmax = min(tmax, t[-1])

        n = int(np.ceil((tmax - tmin) / dt))
        if (dt * n + tmin) < (tmax + dt / 2):
            n += 1
        pv = np.arange(n)
        dctout = {}
        dctout["t"] = pv * dt + tmin
        start = tmin + dt / 2  # so index finds closest point
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            i = _get_prev_index(t, start)
            dctout[key] = d[i + pv]
    else:
        # loop to determine maximum range:
        tmin = t[0]
        tmax = t[-1]
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            if not np.allclose(np.diff(t), dt):
                raise ValueError(f"not all time steps in {key} match {dt}")
            tmin = min(tmin, t[0])
            tmax = max(tmax, t[-1])

        n = int(np.ceil((tmax - tmin) / dt))
        if (dt * n + tmin) < (tmax + dt / 2):
            n += 1
        dctout = {}
        t = dctout["t"] = np.arange(n) * dt + tmin
        for key in parms:
            old_t, old_d, isarr = _get_timedata(dct[key])
            i = _get_prev_index(t, old_t[0] + dt / 2)
            new_d = np.empty(n)
            new_d[:] = value
            old_n = len(old_t)
            if i + old_n > n:
                old_n = n - i
            new_d[i : i + old_n] = old_d[:old_n]
            dctout[key] = new_d
    return dctout


def _vector_to_axis(v, ndim, axis):
    """
    Reshape 1d `v` to nd where the only non-unity dimension is `axis`.
    Useful for broadcasting when, for example, multiplying a vector
    along a certain axis of an nd array.

        _vector_to_axis(np.arange(5), 3, 1).shape  --> (1, 5, 1)
    """
    dims = np.ones(ndim, int)
    dims[axis] = -1
    return v.reshape(*dims)


def windowends(sig, portion=0.01, ends="front", axis=-1):
    """
    Apply parts of a cosine window to the ends of a signal.

    This is also called the "Tukey" window. The window values are
    computed from: ``(1 - cos)/2``.

    Parameters
    ----------
    sig : 1d or 2d ndarray
        Vector or matrix; input time signal(s).
    portion : scalar, optional
        If > 1, specifies the number of points to window at each end.
        If in (0, 1], specifies the fraction of signal to window at
        each end: ``npts = int(portion * np.size(sig, axis))``.
    ends : string, optional
        Specify which ends of signal to operate on:

        =======    ====================================
        `ends`     Action
        =======    ====================================
        'none'     no windowing (no change to `signal`)
        'front'    window front end only
        'back'     window back end only
        'both'     window both ends
        =======    ====================================

    axis : int, optional
        Axis along which to operate.

    Returns
    -------
    Returns the windowed signal(s); same size as the input.

    Notes
    -----
    The minimum number of points that can be windowed is 3.
    Therefore, `portion` is internally adjusted to 3 if the input
    value is (or the computed value turns out to be) less than 3.

    Examples
    --------
    >>> from pyyeti import dsp
    >>> import numpy as np
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> dsp.windowends(np.ones(8), 4)
    array([ 0.  ,  0.25,  0.75,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ])
    >>> dsp.windowends(np.ones(8), .7, ends='back')
    array([ 1.  ,  1.  ,  1.  ,  1.  ,  0.85,  0.5 ,  0.15,  0.  ])
    >>> dsp.windowends(np.ones(8), .5, ends='both')
    array([ 0.  ,  0.25,  0.75,  1.  ,  1.  ,  0.75,  0.25,  0.  ])

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.figure('Example', figsize=[8, 3], clear=True,
        ...                layout='constrained')
        >>> sig = np.ones(100)
        >>> wesig = dsp.windowends(sig, 5, ends='both')
        >>> _ = plt.plot(sig, label='Original')
        >>> _ = plt.plot(wesig, label='windowends (ends="both")')
        >>> _ = plt.ylim(0, 1.2)
    """
    if ends == "none":
        return sig
    sig = np.asarray(sig)
    ln = sig.shape[axis]
    if portion <= 1:
        n = int(portion * ln)
    else:
        n = int(portion)
    if n < 3:
        n = 3
    v = np.ones(ln, float)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (2 * n - 2))
    if ends == "front" or ends == "both":
        v[:n] = w
    if ends == "back" or ends == "both":
        v[-n:] = w[::-1]
    v = _vector_to_axis(v, sig.ndim, axis)
    return sig * v


def _proc_timeslice(timeslice, sr, n):
    # work with integers for slicing:
    if isinstance(timeslice, str):
        ntimeslice = int(timeslice)
        timeslice = ntimeslice / sr
    else:
        ntimeslice = int(round(sr * timeslice))

    if ntimeslice > n:
        ntimeslice = n
        timeslice = ntimeslice / sr
    return ntimeslice, timeslice


def waterfall(
    sig,
    sr,
    timeslice,
    tsoverlap,
    func,
    which,
    freq,
    t0=0.0,
    args=None,
    kwargs=None,
    slicefunc=None,
    sliceargs=None,
    slicekwargs=None,
):
    """
    Compute a 'waterfall' map over time and frequency (typically) using
    a user-supplied function.

    Parameters
    ----------
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
    timeslice : scalar or string-integer
        If scalar, it is the length in seconds for each slice. If
        string, it contains the integer number of points for each
        slice. For example, if `sr` is 1000 samples/second,
        ``timeslice=0.75`` is equivalent to ``timeslice="750"``.
    tsoverlap : scalar in [0, 1) or string-integer
        If scalar, is the fraction of each time-slice to overlap. If
        string, it contains the integer number of points to
        overlap. For example, if `sr` is 1000 samples/second,
        ``tsoverlap=0.5`` and ``tsoverlap="500"`` each specify 50%
        overlap.
    func : function
        This function is called for each time slice (denoted as
        "sig_slice" here) and is expected to return amplitude values
        across the frequency range. It can return just the amplitudes,
        or it can have multiple outputs (see also `which` and
        `freq`). The call is: ``func(sig_slice, *args,
        **kwargs)``. Note that the "sig_slice" input is first passed
        through `slicefunc` if one is provided (see below).
    which : None or integer
        Set to None if `func` only returns the amplitudes. Otherwise,
        if `func` returns multiple outputs, set `which` to the index
        of the output corresponding to the amplitudes. For example, if
        `func` returns ``(frequencies, amplitudes)``, `which` would be
        1 (and `freq` would be 0).

        .. note::
           Setting `which` to None is not the same as setting it to
           0. Using None means that the function only returns
           amplitudes, while a 0 indicates that the output of `func`
           must be indexed by 0 to get the amplitudes. For example: if
           the function has ``return amps``, use ``which=None``; if
           the function has ``return (amps,)``, use ``which=0``.

    freq : integer or vector
        If integer, it is the index of the output of `func`
        corresponding to the frequency vector and cannot be equal to
        `which`. Otherwise, if `freq` is a vector, it is the frequency
        vector directly.
    t0 : scalar; optional
        Start time of signal; defaults to 0.0.
    args : tuple or list; optional
        If provided, these are passed to `func`.
    kwargs : dict; optional
        If provided, these are passed to `func`.
    slicefunc : function or None; optional
        If a function, it is called for each time-slice before `func`
        is called. This could be for windowing or detrending, for
        example. The call is:
        ``sig_slice = slicefunc(sig[pv], *sliceargs, **slicekwargs)``.
    sliceargs : tuple or list; optional
        If provided, these are passed to `slicefunc`. Must be None or
        `()` if `slicefunc` is None.
    slicekwargs : dict; optional
        If provided, these are passed to `slicefunc`. Must be None or
        `{}` if `slicefunc` is None.

    Returns
    -------
    mp : 2d ndarray
        The waterfall map; columns span time, rows span frequency. So,
        each column is a vector of frequency amplitudes as returned by
        `func` for a specific time-slice. Time increases going across
        the columns and frequency increases going down the rows.
    t : 1d ndarray
        Time vector of center times; corresponds to columns in `mp`.
        Signal is assumed to start at time = t0.
    f : 1d ndarray
        Frequency vector corresponding to rows in `mp`. Either equal
        to the input `freq` or as returned by `func`.

    Notes
    -----
    Even though the example below shows the use of `args` and
    `sliceargs`, it is recommended to only use `kwargs` and
    `slicekwargs` only to pass arguments to the supplied functions.
    This makes for more readable code.

    Examples
    --------
    Compute a shock response spectrum waterfall for a sine sweep
    signal. The sweep rate 4 octaves/min. Process in 2-second windows
    with 50% overlap; 2% windowends, compute equivalent sine.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import srs, ytools, dsp
        >>> from matplotlib import cm, colors
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> frq = np.arange(1., 50.1)
        >>> Q = 20
        >>> mp, t, f = dsp.waterfall(sig, sr, 2, 0.5, srs.srs,
        ...                          which=None, freq=frq,
        ...                          args=(sr, frq, Q),
        ...                          kwargs=dict(eqsine=1),
        ...                          slicefunc=dsp.windowends,
        ...                          sliceargs=[.02],
        ...                          slicekwargs=dict(ends='front'))
        >>> _ = plt.figure('Example', clear=True, layout='constrained')
        >>> cs = plt.contour(t, f, mp, 40, cmap=cm.plasma_r)
        >>> # This doesn't work in matplotlib 3.5.0:
        >>> #   cbar = plt.colorbar()
        >>> #   cbar.filled = True
        >>> #   cbar.draw_all()
        >>> # But this does:
        >>> norm = colors.Normalize(
        ...            vmin=cs.cvalues.min(), vmax=cs.cvalues.max()
        ...        )
        >>> sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        >>> cb = plt.colorbar(sm, ax=plt.gca())  # , ticks=cs.levels)
        >>> #
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.ylabel('Frequency (Hz)')
        >>> ttl = 'EQSINE Map of Sine-Sweep @ 4 oct/min, Q = 20'
        >>> _ = plt.title(ttl)

    .. plot::
        :context: close-figs

        Also show results on a 3D surface plot:

        >>> fig = plt.figure("Example 2", clear=True,
        ...                  layout='constrained')
        >>> ax = fig.add_subplot(projection="3d")
        >>> x, y = np.meshgrid(t, f)
        >>> surf = ax.plot_surface(x, y, mp, rstride=1, cstride=1,
        ...                        linewidth=0, cmap=cm.plasma_r)
        >>> _ = fig.colorbar(surf, shrink=0.5, aspect=5)
        >>> ax.view_init(azim=-123, elev=48)
        >>> _ = ax.set_xlabel('Time (s)')
        >>> _ = ax.set_ylabel('Frequency (Hz)')
        >>> _ = ax.set_zlabel('Amplitude')
        >>> _ = plt.title(ttl)
    """
    sig = np.atleast_1d(sig)
    if sig.ndim > 1:
        if max(sig.shape) < sig.size:
            raise ValueError("`sig` must be a vector")
        sig = sig.ravel()

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sliceargs is None:
        sliceargs = ()
    if slicekwargs is None:
        slicekwargs = {}

    ntimeslice, timeslice = _proc_timeslice(timeslice, sr, sig.size)

    if isinstance(tsoverlap, str):
        ntsoverlap = int(tsoverlap)
        if not 0 <= ntsoverlap < ntimeslice:
            raise ValueError(f"`tsoverlap` must be in [0, {ntimeslice})")
    else:
        if not 0 <= tsoverlap < 1:
            raise ValueError("`tsoverlap` must be in [0, 1)")
        ntsoverlap = int(round(ntimeslice * tsoverlap))

    # inc = max(1, int(round(ntimeslice * (1.0 - tsoverlap))))
    inc = max(1, ntimeslice - ntsoverlap)
    non_overlap = inc / ntimeslice
    tlen = (sig.size - ntimeslice) // inc + 1
    b = 0

    # make time vector:
    t0_ = timeslice / 2.0
    tf = t0_ + (tlen - 1) * timeslice * non_overlap
    t = np.linspace(t0_, tf, tlen)
    # print('tlen =', tlen, 'inc =', inc, 't[0], t[-1] =', t[0], t[-1])

    if not slicefunc:

        def slicefunc(a):
            return a

    # do first iteration outside loop to get freq and allocate map:
    s = slicefunc(sig[b : b + ntimeslice], *sliceargs, **slicekwargs)
    b += inc
    res = func(s, *args, **kwargs)
    if isinstance(freq, numbers.Integral):
        if which is None:
            raise ValueError("`which` cannot be None when `freq` is an integer")
        freq = res[freq]
        flen = len(freq)
        res_dtype = res[which].dtype
    else:
        flen = len(freq)
        res_dtype = res.dtype if which is None else res[which].dtype

    mp = np.zeros((flen, tlen), res_dtype)
    mp[:, 0] = res[which]

    for j in range(1, tlen):
        s = slicefunc(sig[b : b + ntimeslice], *sliceargs, **slicekwargs)
        b += inc
        res = func(s, *args, **kwargs)
        if which is not None:
            mp[:, j] = res[which]
        else:
            mp[:, j] = res

    return mp, t + t0, freq


def get_turning_pts(y, x=None, getindex=True, tol=1e-6, atol=None):
    """
    Find turning points (where slope changes) in a vector.

    Parameters
    ----------
    y : array_like
        y-axis data vector
    x : array_like or None; optional
        If vector, x-axis data vector; must be monotonically ascending
        and same length as `y`. If None, the index is used.
    getindex : bool, optional
        If True, return the index of turning points; otherwise, return
        the y (and x) data values at the turning points.
    tol : scalar; optional
        A slope is considered different from a neighbor if the
        difference is more than ``tol*max(abs(all differences))``.
    atol : scalar or None; optional
        Alternative to `tol`. If input (and non-zero), `atol`
        specifies the absolute tolerance and `tol` is ignored. In this
        case, slope is considered different from a neighbor if the
        difference is more than `atol`.

    Returns
    -------
    pv : ndarray; if `getindex` is True
        True/False vector with True for the turning points in `y`.
    yn, xn : ndarray, ndarray or None; if `getindex` is False
        The possibly shortened versions of `y` and `x`; if `x` is None,
        `xn` is None.

    Examples
    --------
    >>> from pyyeti import dsp
    >>> dsp.get_turning_pts([1, 2, 3, 3, 3])
    array([ True, False,  True, False,  True], dtype=bool)
    >>> y, x = dsp.get_turning_pts([1, 2, 3, 3, 3],
    ...                            [1, 2, 3, 4, 5],
    ...                            getindex=False)
    >>> y
    array([1, 3, 3])
    >>> x
    array([1, 3, 5])
    """
    if x is not None:
        x, y = np.atleast_1d(x, y)
        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("x must be monotonically ascending")
        if np.size(y) != np.size(x):
            raise ValueError("x and y must be the same length")
        m = np.diff(y) / dx
    else:
        y = np.atleast_1d(y)
        m = np.diff(y)
    m2 = np.diff(m)
    stol = atol if atol else abs(tol * abs(m2).max())
    pv = np.hstack((True, abs(m2) > stol, True))
    if getindex:
        return pv
    if x is not None:
        return y[pv], x[pv]
    return y[pv]


def calcenv(
    x,
    y,
    p=5,
    n=2000,
    method="max",
    base=0.0,
    makeplot="clear",
    polycolor=(1, 0.7, 0.7),
    label="data",
):
    """
    Returns a curve that envelopes the y data that is allowed to shift
    in the x-direction by some percentage. Optionally plots original
    curve with shaded enveloping polygon.

    Parameters
    ----------
    x : array_like
        x-axis data vector; must be monotonically ascending
    y : array_like
        y-axis data vector; must be same length as x
    p : scalar; optional
        Percentage to shift the y data left and right
    n : integer; optional
        Number of points to use for enveloping curve
    method : string; optional
        Specifies how to envelop data:

        ========   =============================================
        `method`   Description
        ========   =============================================
         'max'     compute envelope over the maximum values of y
         'min'     compute envelope over the minimum values of y
        'both'     combine both 'max' and 'min'
        ========   =============================================

    base : scalar or None; optional
        The base y-value (defines one side of the envelope); if None,
        no base y-value is used and `method` is automatically set to
        'both'
    makeplot : string or axes object; optional
        Specifies if and how to plot envelope in current figure:

        ===========   ===============================
        `makeplot`    Description
        ===========   ===============================
            'no'      do not plot
         'clear'      plot after clearing figure
           'add'      plot without clearing figure
           'new'      plot in new figure
        axes object   plot in given axes (like 'add')
        ===========   ===============================

    polycolor : color specification; optional
        Any valid matplotlib color specification for the color of the
        enveloping curve
    label : string; optional
        Label for the x-y data on plot (only used if `makeplot` is
        not 'no')

    Returns
    -------
    xe_max : 1d ndarray
        x-axis data vector for enveloping curve on max side
    ye_max : 1d ndarray
        y-axis data vector for enveloping curve on max side
    xe_min : 1d ndarray
        x-axis data vector for enveloping curve on min side
    ye_min : 1d ndarray
        y-axis data vector for enveloping curve on min side
    h : None or list
        If `makeplot` is not 'no', `h` is a 2-element list of graphic
        handles: [line, patch].

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import dsp
        >>> x = np.arange(1.0, 31.0, 1.0)
        >>> y = np.cos(x)
        >>> fig = plt.figure('Example', figsize=[10, 8], clear=True,
        ...                  layout='constrained')
        >>>
        >>> ax = plt.subplot(411)
        >>> env = dsp.calcenv(x, y, base=None, makeplot='add')
        >>> _ = plt.title('base=None (method="both")')
        >>> _ = ax.legend(handles=env[-1], loc='upper left',
        ...               bbox_to_anchor=(1.02, 1.),
        ...               borderaxespad=0.)
        >>> _ = ax.set_xticklabels([])
        >>>
        >>> ax = plt.subplot(412)
        >>> env = dsp.calcenv(x, y, method='both', makeplot='add')
        >>> _ = plt.title('method="both"')
        >>> ax.legend().set_visible(False)
        >>> _ = ax.set_xticklabels([])
        >>>
        >>> ax = plt.subplot(413)
        >>> env = dsp.calcenv(x, y, method='max', makeplot='add')
        >>> _ = plt.title('method="max"')
        >>> ax.legend().set_visible(False)
        >>> _ = ax.set_xticklabels([])
        >>>
        >>> ax = plt.subplot(414)
        >>> env = dsp.calcenv(x, y, method='min', makeplot='add')
        >>> _ = plt.title('method="min"')
        >>> ax.legend().set_visible(False)
    """
    x, y = np.atleast_1d(x, y)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be monotonically ascending")

    if np.size(y) != np.size(x):
        raise ValueError("x and y must be the same length")

    if method not in ["max", "min", "both"]:
        raise ValueError("`method` must be one of 'max', 'min', or 'both")

    if base is None:
        method = "both"

    up = 1 + p / 100
    dn = 1 - p / 100
    xe = np.linspace(x[0], x[-1], n)
    xe_max = xe_min = xe
    y2 = np.interp(xe, x, y)

    ye_max = np.zeros(n)
    ye_min = np.zeros(n)
    for i in range(n):
        pv = np.logical_and(xe >= xe[i] / up, xe <= xe[i] / dn)
        ye_max[i] = np.max(y2[pv])
        ye_min[i] = np.min(y2[pv])
        pv = np.logical_and(x >= xe[i] / up, x <= xe[i] / dn)
        if np.any(pv):
            ye_max[i] = max(ye_max[i], np.max(y[pv]))
            ye_min[i] = min(ye_min[i], np.min(y[pv]))

    if method == "max":
        ye_max, xe_max = get_turning_pts(ye_max, xe, getindex=0)
    elif method == "min":
        ye_max, xe_max = get_turning_pts(ye_min, xe, getindex=0)
    elif base is not None:
        ye_max[ye_max < base] = base
        ye_min[ye_min > base] = base
        ye_max, xe_max = get_turning_pts(ye_max, xe, getindex=0)
        ye_min, xe_min = get_turning_pts(ye_min, xe, getindex=0)

    ax = _check_makeplot(makeplot)
    if ax:
        envlabel = rf"$\pm${p}% envelope"
        ln = ax.plot(x, y, label=label)[0]
        p = mpatches.Patch(color=polycolor, label=envlabel)
        if base is None:
            ax.fill_between(xe_max, ye_max, ye_min, facecolor=polycolor, lw=0)
        else:
            ax.fill_between(xe_max, ye_max, base, facecolor=polycolor, lw=0)
            if method == "both":
                ax.fill_between(xe_min, ye_min, base, facecolor=polycolor, lw=0)
        ax.grid(True)
        h = [ln, p]
        ax.legend(handles=h, loc="best")
    else:
        h = None
    return xe_max, ye_max, xe_min, ye_min, h


def fdscale(y, sr, scale, axis=-1):
    """
    Scale a time signal in the frequency domain.

    Parameters
    ----------
    y : nd array_like
        Signal(s) to be scaled. Scaling is done along axis `axis`.
    sr : scalar
        Sample rate.
    scale : 2d array_like
        A two column matrix of [freq scale]. It is automatically sized
        to the correct dimensions via linear interpolation (uses
        :func:`numpy.interp`).
    axis : int, optional
        Axis along which to operate.

    Returns
    -------
    y_new : nd ndarray
        The scaled version of `y`.

    Notes
    -----
    This routine uses FFT to convert `y` to the frequency domain,
    applies the scale factors, and does an inverse FFT to get back to
    the time domain. For example, using
    ``scale = [[1. 1.], [100., 1.]]`` would accomplish nothing.

    The function :func:`scipy.signal.firwin2` can accomplish a similar
    scaling via a digital filter.

    Examples
    --------
    Generate a unit amplitude sine sweep signal and scale down the
    middle section:

    .. plot::
        :context: close-figs

        >>> from pyyeti import ytools, dsp
        >>> import matplotlib.pyplot as plt
        >>> sig, t, f = ytools.gensweep(10, 1, 12, 8)
        >>> scale = np.array([[0., 1.0],
        ...                   [4., 1.0],
        ...                   [5., 0.5],
        ...                   [8., 0.5],
        ...                   [9., 1.0],
        ...                   [100., 1.0]])
        >>> sig_scaled = dsp.fdscale(sig, 1/t[1], scale)
        >>> _ = plt.figure('Example', clear=True, layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(f, sig)
        >>> _ = plt.title('Sine Sweep vs Frequency')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.xlim([f[0], f[-1]])
        >>> _ = plt.subplot(212)
        >>> _ = plt.plot(f, sig_scaled)
        >>> _ = plt.plot(*scale.T, 'k-', lw=2, label='Scale')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Scaled Sine Sweep vs Frequency')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.xlim([f[0], f[-1]])
        >>> _ = plt.grid(True)
    """
    y = np.atleast_1d(y)
    n = y.shape[axis]
    even = n % 2 == 0
    m = n // 2 + 1 if even else (n + 1) // 2
    freq = np.arange(m) * (sr / n)  # positive 1/2 frequency scale

    F = np.fft.rfft(y, axis=axis)
    h = np.interp(freq, scale[:, 0], scale[:, 1])
    h = _vector_to_axis(h, y.ndim, axis)
    Ynew = np.fft.irfft(F * h, n=n, axis=axis)
    return Ynew


def nextpow2(x):
    """
    Return next power of two that is >= integer `x`

    Examples
    --------
    >>> nextpow2(4)
    4
    >>> nextpow2(5)
    8
    """
    return 1 << (x - 1).bit_length()


def _get_ramp(df, bw, on):
    # form gaussian ramp from 1.0 to near zero:
    #      exp(-freq ** 2 / den)
    # - to determine 'den', need number of points and how
    #   close to zero to get
    # - number of points:
    npts = int(bw / df) + 1
    # - how close to zero:
    #     nearzero = exp(-(df*(npts-1)) ** 2 / den)
    #     -log(nearzero) = (df*(npts-1)) ** 2 / den
    #     den = (df*(npts-1)) ** 2 / (-log(nearzero))
    nearzero = 1 / npts / 4
    den = -((df * (npts - 1)) ** 2) / np.log(nearzero)
    ramp = np.exp(-((np.arange(npts + 1) * df) ** 2) / den)
    ramp[-1] = 0.0
    if not on:
        ramp = ramp[::-1]
    return ramp


def _make_h(freq, w, bw, pass_zero, mag, nyq):
    df = freq[1] - freq[0]
    if bw is None:
        bw = 0.01 * nyq
    bw = np.atleast_1d(bw)
    if bw.shape[0] != w.shape[0]:
        if bw.shape[0] != 1:
            raise ValueError(
                "`bw` must be either a scalar or compatibly sized with `w`"
            )
        _bw = np.empty(w.shape[0])
        _bw[:] = bw
        bw = _bw

    H = np.empty(freq.shape[0])
    on = pass_zero
    # position ramps; try to have "mag" point closest to each value
    # in w:
    I = 0
    for _w, _bw in zip(w, bw):
        j = np.argmin(abs(freq - _w))
        ramp = _get_ramp(df, _bw, on)
        n = ramp.shape[0]
        i = np.argmin(abs(ramp - mag))
        if i > j or j - i < I or j - i + n > freq.shape[0]:
            # if ramp doesn't fit in range or if it conflicts with
            # previous, error out:
            raise ValueError(
                "filter function could not be formed to satisfy "
                f"requirements as defined; stopped on w={_w}. Using "
                "a narrower bandwidth might help (currently using"
                f" bw={_bw})"
            )
        H[I : j - i] = ramp[0]
        I = j - i + n
        H[j - i : I] = ramp
        on = not on
    H[I:] = ramp[-1]
    return H


def fftfilt(
    sig, w, *, axis=-1, bw=None, pass_zero=None, nyq=1.0, mag=0.5, makeplot="no"
):
    """
    Filter time-domain signals using FFT with Gaussian ramps.

    Parameters
    ----------
    sig : nd array_like
        Signal(s) to filter. Filtering is done along axis `axis`.
    w : scalar or 1d array_like
        Edge (cutoff) frequencies where ``0.0 < w[i] < nyq`` for all
        ``i`` (`w` must not include 0.0 or `nyq`). Can be any length
        and filter will alternate between pass and stop (starting
        according to `pass_zero`). For example, if leaving `pass_zero`
        as default, a low pass filter is created for scalar `w` and a
        band-pass is created for a 2-element `w`. Units are relative
        to the `nyq` input; so, for example, if `nyq` is the Nyquist
        frequency in Hz, `w` would be in Hz.
    axis : int, optional
        Axis along which to operate.
    bw : scalar, 1d array_like, or None; optional
        Width of each transition region (each up or down ramp). If
        None, ``bw = 0.01 * nyq``.
    pass_zero : bool or None; optional
        Specifies whether or not the zero frequency will be in a pass
        band. If None, `pass_zero` is set to True unless `w` has two
        elements. (So the default is low pass and band pass for `w`
        with one or two elements.)
    nyq : scalar; optional
        Specifies the Nyquist frequency: sample_rate/2.0
    mag : scalar; optional
        Specifies the target filter magnitude at each `w`
    makeplot : string or axes object; optional
        Specifies if and how to plot filter function:

        ===========   ===============================
        `makeplot`    Description
        ===========   ===============================
            'no'      do not plot
         'clear'      plot after clearing figure
           'add'      plot without clearing figure
           'new'      plot in new figure
        axes object   plot in given axes (like 'add')
        ===========   ===============================

    Returns
    -------
    fsig : nd ndarray
        Filtered version of `sig`
    freq : 1d ndarray
        Frequency vector from 0.0 to `nyq`
    h : 1d ndarray
        The frequency domain filter function

    Raises
    ------
    ValueError
        When a ramp will not fit in space as required.

    Examples
    --------
    Make a signal composed of 4 sinusoids, then use :func:`fftfilt` to
    try and recover those sinusoids:

    .. plot::
        :context: close-figs

        >>> from pyyeti import dsp
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = 0.001
        >>> t = np.arange(0, 3.0, h)
        >>> y1 = 10 + 3.1*np.sin(2*np.pi*3*t)
        >>> y2 = 5*np.sin(2*np.pi*10*t)
        >>> y3 = 2*np.sin(2*np.pi*30*t)
        >>> y4 = 3*np.sin(2*np.pi*60*t)
        >>> y = y1 + y2 + y3 + y4
        >>> _ = plt.plot(t, y)
        >>> _ = plt.title('Signal of 4 sinusoids')

    .. plot::
        :context: close-figs

        >>> sr = 1/h
        >>> nyq = sr/2
        >>> _ = plt.figure('Ex1', clear=True, layout='constrained')
        >>> _ = plt.figure('Ex2', clear=True, layout='constrained')
        >>> for j, (w, pz, yj) in enumerate(((7, None, y1),
        ...                                 ([7, 18], None, y2),
        ...                                 ([18, 45], None, y3),
        ...                                 (45, False, y4))):
        ...     _ = plt.figure('Ex1')
        ...     _ = plt.subplot(4, 1, j+1)
        ...     yf = dsp.fftfilt(y, w, pass_zero=pz, nyq=nyq,
        ...                      makeplot='add')[0]
        ...     _ = plt.xlim(0, 75)
        ...     _ = plt.figure('Ex2')
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, yj, t, yf)
    """
    # main routine:
    sig, w = np.atleast_1d(sig, w)
    if pass_zero is None:
        pass_zero = True if len(w) != 2 else False

    if np.any(w > nyq):
        raise ValueError("value(s) in `w` exceed `nyq`")

    if not (axis == -1 or axis == sig.ndim - 1):
        # Move the axis containing the data to the end
        sig = np.swapaxes(sig, axis, sig.ndim - 1)

    n = sig.shape[-1]
    n2 = nextpow2(n)
    freq = np.fft.rfftfreq(n2, 0.5 / nyq)
    h = _make_h(freq, w, bw, pass_zero, mag, nyq)
    t = np.arange(n)
    ylines = interp.interp1d(t[[0, -1]], sig[..., [0, -1]], axis=-1)(t)
    y2 = sig - ylines
    Y = np.fft.rfft(y2, n2, axis=-1)
    h_nd = _vector_to_axis(h, sig.ndim, -1)
    y_h = np.fft.irfft(Y * h_nd, n2, axis=-1)[..., :n]

    if pass_zero:
        y_h += ylines

    if not (axis == -1 or axis == sig.ndim - 1):
        # Move the axis back to where it was
        y_h = np.swapaxes(y_h, axis, sig.ndim - 1)

    ax = _check_makeplot(makeplot)
    if ax:
        ax.plot(freq, h)
        style = dict(color="k", lw=2, ls="--")
        for x in w:
            ax.axvline(x, **style)
        ax.axhline(mag, **style)
    return y_h, freq, h


def _fftsize(n, sr, maxdf):
    if maxdf and sr / n > maxdf:
        N = nextpow2(int(sr / maxdf))
    else:
        N = n
    return N


def fftcoef(
    x,
    sr,
    *,
    axis=-1,
    coef="mag",
    window="boxcar",
    dodetrend=False,
    fold=True,
    maxdf=None,
):
    r"""
    FFT sine/cosine or magnitude/phase coefficients of a real signal

    This routine returns the positive frequency coefficients only.

    Parameters
    ----------
    x : nd array_like
        The (real) signal(s) to FFT. The FFT is carried out along axis
        `axis`.
    sr : scalar
        The sample rate (samples/sec)
    axis : int, optional
        Axis along which to operate.
    coef : string; optional
        Specifies how to return the coefficients:

        ==========   ========================
          `coef`     Return
        ==========   ========================
         "mag"       (magnitude, phase, freq)
         "ab"        (a, b, freq)
         "complex"   (a + i b, None, freq)
        ==========   ========================

        See below for more details.
    window : string, tuple, or 1d array_like; optional
        Specifies window function. If a string or tuple, it is passed
        to :func:`scipy.signal.get_window` to get the window. If 1d
        array_like, it must be length ``len(x)`` and is used directly.
    dodetrend : bool; optional
        If True, remove a linear fit from `x`; otherwise, no
        detrending is done.
    fold : bool; optional
        If true, "fold" negative frequencies on top of positive
        frequencies such that the coefficients at frequencies that
        have a negative counterpart are doubled (magnitude is also
        doubled).
    maxdf : scalar or None; optional
        If scalar, this is the maximum allowed frequency step; zero
        padding will be done if necessary to enforce this. Note that
        this is for providing more points between peaks only. If None,
        the delta frequency is simply ``sr/len(x)``.

    Returns
    -------
    3-tuple depending on `coef`:

        ==========   ========================
          `coef`     Return value
        ==========   ========================
         "mag"       (magnitude, phase, freq)
         "ab"        (a, b, freq)
         "complex"   (a + i b, None, freq)
        ==========   ========================

        All values are for the positive side frequencies only. The
        dimensions of the nd arrays `magnitude`, `phase`, `a` and `b`
        are similar to input `x` except that along the axis `axis`;
        the dimension of that axis corresponds to `freq`. `freq` is a
        1d array of the positive side frequencies only.

        Definitions of `magnitude`, `phase`, and the `a` and `b`
        coefficients are shown below. `freq` is in Hz.

    Notes
    -----
    The FFT results are scaled according to the 'coherent gain' of the
    window function. For the "boxcar" window (which is just all
    1.0's), the coherent gain is 1.0. The coherent gain is defined
    by::

        scale = 1/coherent_gain
        coherent_gain = sum(window)/len(window)

    The coefficients are related to the original signal by either of
    these two equivalent summations (if `fold` is True):

    .. math::
        x(t_n) = \sum\limits^{len(x)-1}_{k=0}
                  A_k \cos(k \omega t_n) +
                  B_k \sin(k \omega t_n)

    .. math::
        x(t_n) = \sum\limits^{len(x)-1}_{k=0}
                  M_k\sin(k \omega t_n - \phi_k)

    where :math:`\omega = 2 \pi \Delta freq`, :math:`M` is the
    magnitude, and :math:`\phi` is the phase. The magnitude and phase
    are computed by:

    .. math::
        \begin{aligned}
        M_k &= \sqrt {A_k^2 + B_k^2}

        \phi_k &= \arctan2(-A_k, B_k)
        \end{aligned}

    Normally, the frequency step is defined by:

    .. math::
        \Delta freq = sr / \text{len}(x)

    A finer frequency step can be achieved by specifying the `maxdf`
    parameter. If `maxdf` is specified *and* the frequency step
    computed from the above equation is greater than `maxdf`, the
    frequency step is computed by:

    .. math::
        \begin{aligned}
        N &= \text{nextpow2}(\text{int}(sr / maxdf))

        \Delta freq &= sr / N
        \end{aligned}

    The function :func:`nextpow2` finds the next power of 2
    integer. This approach makes efficient use of the FFT while
    ensuring the final frequency step is less than or equal to
    `maxdf`.

    The example below uses these formulas directly to upsample a
    signal. This is for demonstration only; to truly upsample a signal
    based on FFT in a more efficient manner, see
    :func:`scipy.signal.resample`. (See also :func:`resample`.)

    See also
    --------
    :func:`fftmap`, :func:`transmissibility`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import dsp
        >>> n = 23
        >>> rng = np.random.default_rng()
        >>> x = rng.normal(size=n)
        >>> t = np.arange(n)
        >>> mag, phase, frq = dsp.fftcoef(x, 1.0)
        >>> _ = plt.figure('Example', clear=True, layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(frq, mag)
        >>> _ = plt.ylabel('Magnitude')
        >>> _ = plt.subplot(212)
        >>> _ = plt.plot(frq, np.rad2deg(np.unwrap(phase)))
        >>> _ = plt.ylabel('Phase (deg)')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>>
        >>> w = 2*np.pi*frq[1]
        >>>
        >>> # use a finer time vector for reconstructions:
        >>> t2 = np.arange(0., n, .05)
        >>>
        >>> # reconstruct with magnitude and phase:
        >>> x2 = 0.0
        >>> for k, (m, p, f) in enumerate(zip(mag, phase, frq)):
        ...     x2 = x2 + m*np.sin(k*w*t2 - p)
        >>>
        >>> # reconstruct with A and B:
        >>> A, B, frq = dsp.fftcoef(x, 1.0, coef='ab')
        >>> x3 = 0.0
        >>> for k, (a, b, f) in enumerate(zip(A, B, frq)):
        ...     x3 = x3 + a*np.cos(k*w*t2) + b*np.sin(k*w*t2)
        >>>
        >>> _ = plt.figure('Example 2', clear=True,
        ...                layout='constrained')
        >>> _ = plt.plot(t, x, 'o', label='Original')
        >>> _ = plt.plot(t2, x2, label='FFT fit w/ Mag & Phase')
        >>> _ = plt.plot(t2, x3, '--', label='FFT fit w/ A & B')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Using `fftcoef` for FFT curve fit')
        >>> _ = plt.xlabel('Time (s)')
    """
    if coef not in ("mag", "ab", "complex"):
        raise ValueError(
            f"invalid `coef` ({coef!r}). Must be one of 'mag', 'ab', or 'complex'."
        )
    x = np.atleast_1d(x)
    n = x.shape[axis]
    if isinstance(window, (str, tuple)):
        window = signal.get_window(window, n)
    else:
        window = np.atleast_1d(window)
        if len(window) != n:
            raise ValueError(
                f"window size is {len(window)}; expected {n} to match signal"
            )

    window *= n / window.sum()

    if not (axis == -1 or axis == x.ndim - 1):
        # Move the axis containing the data to the end
        x = np.swapaxes(x, axis, x.ndim - 1)

    window = _vector_to_axis(window, x.ndim, -1)

    if dodetrend:
        x = signal.detrend(x) * window
    else:
        x = x * window

    N = _fftsize(n, sr, maxdf)
    if N > n:
        shape = [*x.shape]
        shape[-1] = N
        X = np.empty(shape)
        X[..., :n] = x
        X[..., n:] = 0.0
    else:
        X = x

    F = np.fft.rfft(X)
    f = np.fft.rfftfreq(N, 1.0 / sr)
    # or, could do this to get same result:
    #     F = np.fft.fft(X)
    #     m = N // 2 + 1
    #     f = np.arange(0.0, m) * (sr / N)
    #     F = F[:m]
    if fold:
        a = 2.0 * F.real / n
        a[..., 0] /= 2.0
        if not N & 1:  # if N is an even number
            a[..., -1] /= 2.0
        b = -2.0 * F.imag / n
    else:
        a = F.real / n
        b = -F.imag / n

    if not (axis == -1 or axis == x.ndim - 1):
        # Move the axis containing the data to the end
        a = np.swapaxes(a, axis, x.ndim - 1)
        b = np.swapaxes(b, axis, x.ndim - 1)

    if coef == "mag":
        return np.sqrt(a**2 + b**2), np.arctan2(-a, b), f
    elif coef == "complex":
        return a + 1j * b, None, f
    return a, b, f


def fftmap(
    timeslice, tsoverlap, sig, sr, window="hann", dodetrend=False, fold=True, maxdf=None
):
    """
    Make an FFT map ('waterfall') over time and frequency.

    Parameters
    ----------
    timeslice : scalar or string-integer
        If scalar, it is the length in seconds for each slice. If
        string, it contains the integer number of points for each
        slice. For example, if `sr` is 1000 samples/second,
        ``timeslice=0.75`` is equivalent to ``timeslice="750"``.
    tsoverlap : scalar in [0, 1) or string-integer
        If scalar, is the fraction of each time-slice to overlap. If
        string, it contains the integer number of points to
        overlap. For example, if `sr` is 1000 samples/second,
        ``tsoverlap=0.5`` and ``tsoverlap="500"`` each specify 50%
        overlap.
    sig : 1d array_like
        Signal to compute FFT map of.
    sr : scalar
        The sample rate (samples/sec)
    window : string, tuple, or 1d array_like; optional
        Specifies window function. If a string or tuple, it is passed
        to :func:`scipy.signal.get_window` to get the window. If 1d
        array_like, it must be length ``len(x)`` and is used directly.
    dodetrend : bool; optional
        If True, remove a linear fit from `x`; otherwise, no
        detrending is done.
    fold : bool; optional
        If true, "fold" negative frequencies on top of positive
        frequencies such that the coefficients at frequencies that
        have a negative counterpart are doubled (magnitude is also
        doubled).
    maxdf : scalar or None; optional
        If scalar, this is the maximum allowed frequency step; zero
        padding will be done if necessary to enforce this. Note that
        this is for providing more points between peaks only. If None,
        the delta frequency is simply ``sr/len(x)``.

    Returns
    -------
    mp : 2d ndarray
        The FFT map; columns span time, rows span frequency (so each
        column is an FFT curve). Time increases going across the
        columns and frequency increases going down the rows.
    t : 1d ndarray
        Time vector of center times; corresponds to columns in map.
        Signal is assumed to start at time = 0.
    f : 1d ndarray
        Frequency vector; corresponds to rows in map.

    Notes
    -----
    This routine calls :func:`fftcoef` for each time slice. `mp` is a
    matrix where each column is the FFT magnitude at all discrete
    frequencies for a certain time-slice.  That is, time increases
    going across the columns and frequency increases going down the
    rows.

    See also
    --------
    :func:`fftcoef`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import cm, colors
        >>> from pyyeti import dsp, ytools
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> mp, t, f = dsp.fftmap(2, .1, sig, sr)
        >>> pv = f <= 50.0
        >>> cs = plt.contour(t, f[pv], mp[pv], 40, cmap=cm.plasma)
        >>> # This doesn't work in matplotlib 3.5.0:
        >>> #   cbar = plt.colorbar()
        >>> #   cbar.filled = True
        >>> #   cbar.draw_all()
        >>> # But this does:
        >>> norm = colors.Normalize(
        ...            vmin=cs.cvalues.min(), vmax=cs.cvalues.max()
        ...        )
        >>> sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        >>> cb = plt.colorbar(sm, ax=plt.gca())  # , ticks=cs.levels)
        >>> #
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.ylabel('Frequency (Hz)')
        >>> ttl = 'FFT Map of Sine-Sweep @ 4 oct/min'
        >>> _ = plt.title(ttl)

    """
    return waterfall(
        sig,
        sr,
        timeslice,
        tsoverlap,
        fftcoef,
        which=0,
        freq=2,
        kwargs=dict(sr=sr, window=window, dodetrend=dodetrend, fold=fold, maxdf=maxdf),
    )


def transmissibility(
    in_data,
    out_data,
    sr,
    timeslice=1.0,
    tsoverlap=0.5,
    window="hann",
    getmap=False,
    **kwargs,
):
    r"""
    Compute transmissibility transfer function using the FFT

    Transmissibility is a common transfer function measurement of
    ``output / input``. It is a type of frequency response function
    where the gain (magnitude) vs frequency is typically of primary
    interest. Note that the phase can be computed from the output of
    this routine as well.

    Parameters
    ----------
    in_data : 1d array_like
        Time series of measurement values for the input data
    out_data : 1d array_like
        Time series of measurement values for the output data
    sr : scalar
        Sample rate.
    timeslice : scalar or string-integer
        If scalar, it is the length in seconds for each slice. If
        string, it contains the integer number of points for each
        slice. For example, if `sr` is 1000 samples/second,
        ``timeslice=0.75`` is equivalent to ``timeslice="750"``.
    tsoverlap : scalar in [0, 1) or string-integer
        If scalar, is the fraction of each time-slice to overlap. If
        string, it contains the integer number of points to
        overlap. For example, if `sr` is 1000 samples/second,
        ``tsoverlap=0.5`` and ``tsoverlap="500"`` each specify 50%
        overlap.
    window : string, tuple, or 1d array_like; optional
        Specifies window function. If a string or tuple, it is passed
        to :func:`scipy.signal.get_window` to get the window. If 1d
        array_like, it must be length ``len(x)`` and is used directly.
    getmap : bool, optional
        If True, get the transfer function map outputs (see below).
    *kwargs : optional
        Named arguments to pass to :func:`fftcoef`. Note that `x`,
        `sr`, `coef` and `window` arguments are passed automatically,
        and that `fold` is irrelevant (due to computing a
        ratio). Therefore, at the time of this writing, only
        `dodetrend`, and `maxdf` are really valid entries in `kwargs`.

    Returns
    -------
    A SimpleNamespace with the members:

    f : 1d ndarray
        Array of sample frequencies.
    mag : 1d ndarray
        Average magnitude of transmissibility transfer function across
        all time slices of ``out_data / in_data``; length is
        ``len(f)``::

             mag = abs(tr_map).mean(axis=1)

    phase : 1d ndarray
        Average phase in degrees of transmissibility transfer function
        across all time slices of ``out_data / in_data``; length is
        ``len(f)``. Computing the average of angles is tricky; for
        example, the average of 15 degrees and 355 degrees is 5
        degrees. To get this result, the approach used here is to
        compute the average of cartesian coordinates of points on a
        unit circle at each angle, and then compute the angle to that
        average location::

             phase = np.angle(
                        (tr_map / abs(tr_map)).mean(axis=1), deg=True
                     )

        This definition of phase follows the negative sign convention
        of phase (as in :func:`fftcoef`): ``sin(theta - phase)``.
    tr_map : complex 2d ndarray; optional
        The complex transmissibility transfer function map. Each
        column is the transmissibility of ``out_data / in_data``
        computed from the FFT ratio (from :func:`fftcoef`) for the
        corresponding time slice. Rows correspond to frequency `f` and
        columns correspond to time `t`. Only output if `getmap` is
        True.
    mag_map : 2d ndarray; optional
        The magnitude of the transmissibility map. Only output if
        `getmap` is True. It is computed by::

            mag_map = abs(tr_map)

    phase_map : 2d ndarray; optional
        The phase in degrees of the transmissibility map. Only output
        if `getmap` is True. It is computed by::

            phase_map = np.angle(tr_map, deg=True)

    t : 1d ndarray; optional
        The time vector for the columns of `tr_map`, `mag_map` and
        `phase_map`. Only output if `getmap` is True.

    Notes
    -----
    This routine calls :func:`waterfall` for handling the timeslices
    and preparing the output and :func:`fftcoef` (with `coef` set to
    "complex") to process each time slice of both `in_data` and
    `out_data`.

    The frequency step size is determined by `timeslice` in seconds::

        freq_step_size = 1 / timeslice

    Examples
    --------
    We'll make up a pseudo-random signal with content from 5 to 60 Hz,
    and use that as a base-excitation (acceleration) input to a spring
    mass system. This is the same system as used in the
    :func:`pyyeti.srs.srs` routine::

                      _____    ^
                     |     |   |
                     |  M  |  ---  SDOF response (x)
                     |_____|
                      /  |
                    K \ |_| C  ^
                      /  |     |
                    [=======] ---  input base acceleration (sig)

    We'll then compute the transmissibility of the acceleration of the
    mass relative to the base-excitation. We should see a nice peak
    (roughly equal to the dynamic amplification factor Q) at the
    natural frequency of the single degree of freedom system. For the
    example, the frequency of the SDOF system is set to 35 Hz and Q is
    set to 25.

    To check the results, the exact magnitude and phase from a
    closed-form frequency-domain solution (via
    :func:`pyyeti.ode.SolveUnc.fsolve`) will be plotted as well.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib import cm, colors
        >>> import matplotlib.gridspec as gridspec
        >>> from pyyeti import dsp, psd, ode
        >>>
        >>> # make up a flat (5-60 Hz) spectrum signal as input:
        >>> psd_ = 1.0
        >>> fstart = 5.0
        >>> fstop = 60.0
        >>> spec = np.array([[0.1, psd_], [100.0, psd_]])
        >>> in_acce, sr, t = psd.psd2time(
        ...     spec, fstart, fstop, ppc=10, df=0.05, gettime=True,
        ...     winends={"ends": "both"}
        ... )
        >>>
        >>> # define a 35 hz system with 2% damping (Q = 25):
        >>> frq = 35.0
        >>> omega = frq * np.pi * 2
        >>> Q = 25
        >>> zeta = 1 / (2 * Q)
        >>> ts = ode.SolveUnc(1, 2 * zeta * omega, omega ** 2, 1 / sr)
        >>>
        >>> # base-drive system like SRS ... in_acce is base input
        >>> # (see pyyeti.srs.srs):
        >>> #    zddot = input
        >>> #    x = absolute displacement of mass
        >>> #    u = x - z
        >>> sol = ts.tsolve(-in_acce)
        >>> out_acce = sol.a.ravel() + in_acce  # xddot
        >>>
        >>> tr = dsp.transmissibility(
        ...           in_acce, out_acce, sr, getmap=True)
        >>>
        >>> fig = plt.figure('Example', figsize=(8, 11), clear=True,
        ...                  layout='tight')
        >>>
        >>> # use GridSpec to make a nice layout with colorbars:
        >>> gs = gridspec.GridSpec(5, 2, width_ratios=[30, 1])
        >>>
        >>> ax = ax_time = plt.subplot(gs[0, 0])
        >>> _ = ax.plot(t, in_acce, label="input")
        >>> _ = ax.plot(t, out_acce, alpha=0.75, label="output")
        >>> _ = ax.set_xlabel("Time (s)")
        >>> _ = ax.set_ylabel("Acceleration")
        >>> _ = ax.legend(loc="upper right")
        >>>
        >>> # plot only frequency range where input has content:
        >>> pv = (tr.f >= fstart) & (tr.f <= fstop)
        >>> fm = tr.f[pv]
        >>>
        >>> # compute exact solution for comparison:
        >>> in_acce_exact = np.ones(len(fm))
        >>> sol = ts.fsolve(-in_acce_exact, fm)
        >>> acce_exact = in_acce_exact + sol.a[0]
        >>> mag_exact = abs(acce_exact)
        >>> phase_exact = -np.angle(acce_exact, deg=True)
        >>>
        >>> # plot magnitude:
        >>> ax = ax_freq = plt.subplot(gs[1, 0])
        >>> _ = ax.plot(fm, tr.mag[pv], label="Estimate")
        >>> _ = ax.plot(fm, mag_exact, label="Exact")
        >>> _ = ax.set_xlabel("Frequency (Hz)")
        >>> _ = ax.set_ylabel("TR Magnitude")
        >>> _ = ax.set_title("Average Transmissibility Magnitude")
        >>> _ = ax.legend(loc="upper right")
        >>>
        >>> ax = plt.subplot(gs[2, 0], sharex=ax_time)
        >>> c = ax.contour(tr.t, fm, tr.mag_map[pv], 40,
        ...                cmap=cm.plasma)
        >>> _ = ax.set_xlabel("Time (s)")
        >>> _ = ax.set_ylabel("Frequency (Hz)")
        >>> _ = ax.set_title("Transmissibility Magnitude Map")
        >>>
        >>> ax = plt.subplot(gs[2, 1])
        >>>
        >>> # This doesn't work in matplotlib 3.5.0:
        >>> #   cb = fig.colorbar(c, cax=ax)
        >>> #   cb.filled = True
        >>> #   cb.draw_all()
        >>> # But this does:
        >>> norm = colors.Normalize(
        ...            vmin=c.cvalues.min(), vmax=c.cvalues.max()
        ...        )
        >>> sm = plt.cm.ScalarMappable(norm=norm, cmap=c.cmap)
        >>> cb = plt.colorbar(sm, cax=ax)  # , ticks=c.levels)
        >>>
        >>> _ = ax.set_title("TR Magnitude")
        >>>
        >>> # plot phase:
        >>> ax = plt.subplot(gs[3, 0], sharex=ax_freq)
        >>> _ = ax.plot(fm, tr.phase[pv], label="Estimate")
        >>> _ = ax.plot(fm, phase_exact, label="Exact")
        >>> _ = ax.set_xlabel("Frequency (Hz)")
        >>> _ = ax.set_ylabel("TR Phase (deg)")
        >>> _ = ax.set_title("Average Transmissibility Phase")
        >>> _ = ax.legend(loc="lower right")
        >>>
        >>> ax = plt.subplot(gs[4, 0], sharex=ax_time)
        >>> c = ax.contour(tr.t, fm, tr.phase_map[pv], 40,
        ...                cmap=cm.plasma)
        >>> _ = ax.set_xlabel("Time (s)")
        >>> _ = ax.set_ylabel("Frequency (Hz)")
        >>> _ = ax.set_title("Transmissibility Phase Map")
        >>>
        >>> ax = plt.subplot(gs[4, 1])
        >>>
        >>> # This doesn't work in matplotlib 3.5.0:
        >>> #   cb = fig.colorbar(c, cax=ax)
        >>> #   cb.filled = True
        >>> #   cb.draw_all()
        >>> # But this does:
        >>> norm = colors.Normalize(
        ...            vmin=c.cvalues.min(), vmax=c.cvalues.max()
        ...        )
        >>> sm = plt.cm.ScalarMappable(norm=norm, cmap=c.cmap)
        >>> cb = plt.colorbar(sm, cax=ax)  # , ticks=c.levels)
        >>>
        >>> _ = ax.set_title("TR Phase (deg)")

    As an aside and for fun, compare the actual root-mean-square
    response to the Miles' equation estimate. Miles' should be a
    little higher on average ... it assumes infinitely wide, flat
    input spectrum.

    We'll also compare the actual peak vs both:

        - a ``3 * sigma`` Miles' peak
        - a ``peak_factor * sigma`` Miles' peak, where ``peak_factor``
          is determined from the Rayleigh distribution

    The Rayleigh peak factor is ``sqrt(2*log(duration*f))``. See
    :func:`pyyeti.fdepsd.fdepsd` for the derivation of this factor. In
    this example, since the number of cycles is quite high, 3 sigma
    will generally be below the peak. The Rayleigh peak factor allows
    for a fairly good estimate of the actual peak.

    >>> actual = np.sqrt((out_acce ** 2).mean())
    >>> miles = np.sqrt(np.pi / 2 * Q * frq * psd_)
    >>> if True:   # doctest: +SKIP
    ...     print("rms comparison:")
    ...     print(f"\tactual        = {actual:.2f}")
    ...     print(f"\tMiles         = {miles:.2f}")
    ...     print(f"\tratio         = {actual/miles:.2f}")
    rms comparison:
            actual        = 36.50
            Miles         = 37.07
            ratio         = 0.98
    >>>
    >>> # Compare actual peak to 3 sigma peak from Miles':
    >>> actual_pk = abs(out_acce).max()
    >>> miles_3s = 3 * miles
    >>> if True:   # doctest: +SKIP
    ...     print("peak comparison #1:")
    ...     print(f"\tactual        = {actual_pk:.2f}")
    ...     print(f"\t3 * Miles     = {miles_3s:.2f}")
    ...     print(f"\tratio         = {actual_pk/miles_3s:.2f}")
    peak comparison #1:
            actual        = 134.00
            3 * Miles     = 111.22
            ratio         = 1.20
    >>>
    >>> rpf = np.sqrt(2 * np.log(frq * t[-1]))  # rayleigh peak factor
    >>> miles_rayleigh = rpf * miles
    >>> if True:   # doctest: +SKIP
    ...     print("peak comparison #2:")
    ...     print(f"\tactual        = {actual_pk:.2f}")
    ...     print(f"\t{rpf:.2f} * Miles  = {miles_rayleigh:.2f}")
    ...     print(f"\tratio         = {actual_pk/miles_rayleigh:.2f}")
    peak comparison #2:
            actual        = 134.00
            3.62 * Miles  = 134.19
            ratio         = 1.00
    """
    in_data, out_data = np.atleast_1d(in_data, out_data)
    if in_data.shape != out_data.shape or in_data.ndim != 1:
        raise ValueError("`in_data` and `out_data` must be 1d arrays of the same size")

    kwargs["sr"] = sr
    kwargs["window"] = window
    kwargs["coef"] = "complex"
    fftmap_in, t, f = waterfall(
        in_data, sr, timeslice, tsoverlap, fftcoef, which=0, freq=2, kwargs=kwargs
    )

    fftmap_out, t, f = waterfall(
        out_data, sr, timeslice, tsoverlap, fftcoef, which=0, freq=2, kwargs=kwargs
    )

    tr_map = fftmap_out / fftmap_in
    mag = abs(tr_map).mean(axis=1)
    phase = np.angle((tr_map / abs(tr_map)).mean(axis=1), deg=True)
    if getmap:
        mag_map = abs(tr_map)
        phase_map = np.angle(tr_map, deg=True)
        return SimpleNamespace(
            f=f,
            mag=mag,
            phase=phase,
            tr_map=tr_map,
            mag_map=mag_map,
            phase_map=phase_map,
            t=t,
        )
    return SimpleNamespace(f=f, mag=mag, phase=phase)
