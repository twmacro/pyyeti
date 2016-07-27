# -*- coding: utf-8 -*-
"""
Digital signal processing tools.
"""

import math
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools
from collections import abc
from warnings import warn
from types import SimpleNamespace


def resample(data, p, q, beta=5, pts=10, t=None, getfir=False):
    """Change sample rate of data by a rational factor using Lanczos
    resampling.

    Parameters
    ----------
    data : 1d or 2d ndarray
        Data to be resampled; if a matrix, every column is resampled.
    p : integer
        The upsample factor.
    q : integer
        The downsample factor.
    beta : scalar
        The beta value for the Kaiser window. See
        :func:`scipy.signal.kaiser`.
    pts : integer
        Number of points in data to average from each side of current
        data point. For example, if ``pts == 10``, a total 21 points of
        original data are used for averaging.
    t : array_like
        If `t` is given, it is assumed to be the sample positions
        associated with the signal data in `data` and the new
        (resampled) positions are returned.
    getfir : bool
        If True, the FIR filter coefficients are returned.

    Returns
    -------
    rdata : 1d or 2d ndarray
        The resampled data. If the signal(s) in `data` have `n`
        samples, the signal(s) in `rdata` have ``ceil(n*p/q)``
        samples.
    tnew : 1d ndarray
        The resampled positions, same length as `rdata`. Only
        returned if `t` is input.
    fir : 1d ndarray
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
        1. Inserts `p` zeros after every point in `mdata`.
        2. Forms an averaging, anti-aliasing FIR filter based on the
           'sinc' function and the Kaiser window to filter `mdata`.
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
    :func:`scipy.signal.resample`.

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
        >>> _ = plt.figure('resample demo 1', figsize=(10, 8))
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(x, data, 'o-', label='Original')
        >>> res = {}
        >>> for pts, m in zip([2, 3, 5],
        ...                   ['^', 'v', '<']):
        ...     res[pts], up2 = dsp.resample(data, p, q, pts=pts, t=x)
        ...     lab = 'Resample, pts={}'.format(pts)
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
        >>> _ = plt.plot(frqup, 2*np.abs(fft.rfft(resfft))/n2,
        ...          label='scipy.signal.resample')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('FFT Mag.')
        >>> _ = plt.xlabel('Frequency - fraction of original sample rate')
        >>> _ = plt.tight_layout()
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
        >>> np.random.seed(1)
        >>> data = np.random.randn(n)
        >>> x = np.arange(n)
        >>> dndata, dnx = dsp.resample(data, p, q, t=x)
        >>> fig = plt.figure('resample demo 2')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(x, data, 'o-', label='Original', alpha=.3)
        >>> _ = plt.plot(dnx, dndata, label='Resample', lw=2)
        >>> resfft = signal.resample(data, np.ceil(n/q))
        >>> _ = plt.plot(dnx, resfft, label='scipy.signal.resample',
        ...              lw=2)
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Signal')
        >>> _ = plt.xlabel('Time')
        >>> _ = plt.subplot(212)
        >>> n2 = len(dnx)
        >>> frq = fft.rfftfreq(n, 1)
        >>> frqdn = fft.rfftfreq(n2, q)
        >>> _ = plt.plot(frq, 2*np.abs(fft.rfft(data))/n, 'o-',
        ...              alpha=.3)
        >>> _ = plt.plot(frqdn, 2*np.abs(fft.rfft(dndata))/n2, lw=2)
        >>> _ = plt.plot(frqdn, 2*np.abs(fft.rfft(resfft))/n2, lw=2)
        >>> _ = plt.title('FFT Mag.')
        >>> xlbl = 'Frequency - fraction of original sample rate'
        >>> _ = plt.xlabel(xlbl)
        >>> _ = plt.tight_layout()
    """
    data = np.atleast_1d(data)
    ndim = data.ndim
    if ndim == 1 or min(data.shape) == 1:
        data = data.reshape(-1, 1)
        cols = 1
    else:
        cols = data.shape[1]
    ln = data.shape[0]

    # setup FIR filter for upsampling given the following parameters:
    gf = math.gcd(p, q)
    p = p//gf
    q = q//gf

    M = 2*pts*max(p, q)
    w = signal.kaiser(M+1, beta)
    # w = signal.hann(M+1)
    n = np.arange(M+1)

    # compute cutoff relative to highest sample rate (P*sr where sr=1)
    #  eg, if Q = 1, cutoff = .5 of old sample rate = .5/P of new
    #      if P = 1, cutoff = .5 of new sample rate = .5/Q of old
    cutoff = min(1/q, 1/p)/2
    # sinc(x) = sin(pi*x)/(pi*x)
    s = 2*cutoff*np.sinc(2*cutoff*(n-M/2))
    fir = p*w*s
    m = np.mean(data, axis=0)

    # insert zeros
    updata1 = np.zeros((ln*p, cols))
    for j in range(cols):
        updata1[::p, j] = data[:, j] - m[j]

    # take care of lag by shifting with zeros:
    nz = M//2
    z = np.zeros((nz, cols), float)
    updata1 = np.vstack((z, updata1, z))
    updata = signal.lfilter(fir, 1, updata1, axis=0)
    updata = updata[M:]

    # downsample:
    n = int(np.ceil(ln*p/q))
    RData = np.zeros((n, cols), float)
    for j in range(cols):
        RData[:, j] = updata[::q, j] + m[j]
    if ndim == 1:
        RData = RData.ravel()
    if t is None:
        if getfir:
            return RData, fir
        return RData
    tnew = np.arange(n) * (t[1]-t[0]) * ln/n + t[0]
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
            raise ValueError('incorrectly sized ndarray for '
                             'time/data input (must be 2d with 2 '
                             'columns)')
        t = data[:, 0]
        d = data[:, 1]
        isndarray = True
    else:
        if len(data) != 2:
            raise ValueError('incorrectly defined time/data input')
        t, d = np.atleast_1d(*data)
        if len(t) != len(d):
            raise ValueError('time and data vectors are incompatibly '
                             'sized')
        isndarray = False
    return t, d, isndarray


def _get_prev_index(vec, val):
    """Finds previous index for scalar `val`"""
    p = np.searchsorted(vec, val) - 1
    if p < 0:
        return 0
    return p


def exclusive_sgfilter(x, n, exclude_midpoint=True, axis=-1):
    """
    0th order 1-d Savitzky-Golay FIR filter that excludes midpoint

    Parameters
    ----------
    x : nd array_like
        Array to filter
    n : odd integer
        Number of points for filter; if even, it is reset to ``n+1``
    exclude_midpoint : bool; optional
        If True, exclude middle point in the filter. That is, the
        moving average is computed without the central point. Can
        be useful for determining if point is an outlier. The average
        is of the n-1 surrounding points.
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
    0th order Savitzky-Golay is a basic "moving average".

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import savgol_filter
    >>> from pyyeti.dsp import exclusive_sgfilter
    >>> x = np.arange(6.)
    >>> x[3] *= 2
    >>> x
    array([ 0.,  1.,  2.,  6.,  4.,  5.])
    >>> savgol_filter(x, 3, polyorder=0)
    array([ 1.,  1.,  3.,  4.,  5.,  5.])
    >>> exclusive_sgfilter(x, 3, exclude_midpoint=True)
    array([ 1.5,  1. ,  3.5,  3. ,  5.5,  5. ])

    If `exclude_midpoint` is False, this is the same as a normal 0th
    order Savitzky-Golay filter:

    >>> exclusive_sgfilter(x, 3, exclude_midpoint=False)
    array([ 1.,  1.,  3.,  4.,  5.,  5.])
    """
    n = n | 1
    n_mid = n // 2
    b = np.empty(n)
    if exclude_midpoint:
        b[:] = 1/(n-1)
        b[n_mid] = 0.0
    else:
        b[:] = 1/n
    x = np.atleast_1d(x)
    if axis != -1 or axis != x.ndim - 1:
        # Move the axis containing the data to the end
        x = np.swapaxes(x, axis, x.ndim - 1)

    # Append pieces of x onto the front and back so the averages on
    # the ends work out properly. For example, if n is 5 and x is:
    #      x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # then x2 is:
    #      x2 = [3, 4,  0, 1, 2, 3, 4, 5, 6, 7, 8,  4, 5]
    # 1st ave:   +  +   -  +  +
    # 2nd ave:      +   +  -  +  +
    # 3rd ave:          +  +  -  +  +
    # ... last ave:                       +  +  -   +  +
    x2 = np.concatenate((x[..., n_mid+1:n],
                         x,
                         x[..., -n:-(n_mid+1)]), axis=-1)

    d = signal.lfilter(b, 1, x2)[..., n-1:]
    if axis != -1 or axis != x.ndim - 1:
        # Move the axis back to where it was
        d = np.swapaxes(d, axis, x.ndim - 1)
    return d


def despike(x, n, sigma=8.0, maxiter=-1, mode='average',
            threshold_sigma=0.1, threshold_value=None, axis=-1):
    """
    Delete outlier data points from signal(s)

    Parameters
    ----------
    x : nd array_like
        Array to filter. If `mode` is 'delete', `x` must be 1d.
    n : odd integer
        Number of points for moving average; if even, it is reset to
        ``n+1``. If greater than the dimension of `x`, it is reset to
        the dimension or 1 less.
    sigma : real scalar; optional
        Number of standard deviations beyond which a point is
        considered an outlier. The default value is quite high; this
        is possible because the point itself if excluded from the
        calculations.
    maxiter : integer; optional
        Maximum number of iterations of outlier removal allowed.
        Multiple iterations are possible because the deletion of an
        outlier may expose other points as outliers. If <= 0, there is
        no set limit and the looping will stop when no more outliers
        are detected.
    mode : string; optional
        Either 'delete' or 'average'.

          =========  ================================================
            mode     description
          =========  ================================================
          'delete'   delete the outliers
          'average'  replace the outliers with an average of the two
                     surrounding points (or nearest point, if at end)
          =========  ================================================

    threshold_sigma : scalar; optional
        Number of standard deviations below which all data is kept.
        This standard deviation is of the entire input signal. This
        value exists to avoid deleting small deviations such as bit
        toggles. `threshold_value` overrides `threshold_sigma` if it
        is not None.
    threshold_value : scalar or None; optional
        Optional method for specifying a minimum threshold. If not
        None, this scalar is used as an absolute minimum deviation
        from the moving average for a value to be considered a spike.
    axis : integer; optional
        Axis along which to delete outliers; each subarray along this
        axis is despiked. For example, to despike each column in a 2d
        array, set `axis` to 0.

    Returns
    -------
    A SimpleNamespace with the members:

    dx : nd ndarray
        Despiked version of `x`. Will be shorter than `x` if `mode`
        is 'delete' and spikes were detected; otherwise, it will
        have the same shape as `x`.
    pv : bool nd ndarray; same shape as `x`
        Has True where an outlier was detected
    hilim : nd ndarray; same shape as `x`
        This is the upper limit: ``mean + sigma*std``
    lolim : nd ndarray; same shape as `x`
        This is the lower limit: ``mean - sigma*std``
    niter : integer
        Number of iterations executed

    Notes
    -----
    Uses :func:`exclusive_sgfilter` to exclude the midpoint in the
    moving average and the moving standard deviation calculations.

    The 'delete' and 'average' methods are different enough that
    they may find a different set of outliers. See example below.

    To not use a threshold, set `threshold_sigma` to 0.0 (or set
    `threshold_value` to 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import dsp
    >>> x = [100, 2, 3, -4, 25, -6, 6, 3, -2, 4, -2, -100]
    >>> np.set_printoptions(precision=2, linewidth=65)
    >>> s = dsp.despike(x, n=9, sigma=2, mode='average')
    >>> s.dx
    array([ 2,  2,  3, -4, -5, -6,  6,  3, -2,  4, -2, -2])
    >>> s.pv
    array([ True, False, False, False,  True, False, False, False,
           False, False, False,  True], dtype=bool)
    >>> s.hilim
    array([ 7.93,  7.93,  7.62,  8.31,  8.12,  8.39,  6.25,  6.56,
            7.66,  6.12,  7.66,  7.66])
    >>> s.niter
    3

    Now, run the same data but with 'delete'. Here, a plot is very
    helpful to visualize each iteration:

    .. plot::
        :context: close-figs

        >>> x = [100, 2, 3, -4, 25, -6, 6, 3, -2, 4, -2, -100]
        >>> _ = plt.figure(figsize=(8, 11))
        >>> plt.clf()
        >>> for i in range(5):
        ...     s = dsp.despike(x, n=9, sigma=2,
        ...                     mode='delete', maxiter=1)
        ...     _ = plt.subplot(5, 1, i+1)
        ...     _ = plt.plot(x)
        ...     _ = plt.plot(s.hilim, 'k--')
        ...     _ = plt.plot(s.lolim, 'k--')
        ...     _ = plt.title('Iteration {}'.format(i+1))
        ...     x = s.dx
        >>> plt.tight_layout()
        >>> s.dx
        array([ 2,  3,  6,  3, -2,  4, -2])

    """
    def _get_min_limit(x, axis, threshold_sigma, threshold_value):
        if threshold_value is not None:
            return threshold_value
        min_limit = threshold_sigma * np.std(x, axis=axis)
        # reshape min_limit so it is broadcast compatible with x:
        shape = list(x.shape)
        shape[axis] = 1
        return min_limit.reshape(shape)

    def _find_outlier_peaks(y, n, sigma, min_limit, axis):
        ave = exclusive_sgfilter(y, n, axis=axis)
        var = exclusive_sgfilter(y**2, n, axis=axis) - ave**2
        # use abs to care of negative numerical zeros:
        std = np.sqrt(abs(var))
        limit = np.fmax(sigma * std, min_limit)
        return abs(y-ave) <= limit, ave+limit, ave-limit

    def _set_ave(x, pv, axis):
        # Set outlier value to be average of two neighbor points
        # (linear interpolation).
        # First, get previous and next indexes:
        pv_prev = list(pv.nonzero())
        pv_next = pv_prev[:]
        pv_next[axis] = pv_next[axis].copy()
        pv_prev[axis] -= 1
        pv_next[axis] += 1
        # If spike is on an edge, just set to equal to the single
        # neighbor:
        j = pv_prev[axis] < 0
        pv_prev[axis][j] += 2
        j = pv_next[axis] >= x.shape[axis]
        pv_next[axis][j] -= 2
        x[pv] = (x[pv_prev] + x[pv_next])/2.0

    x = np.atleast_1d(x).copy()
    min_limit = _get_min_limit(x, axis, threshold_sigma,
                               threshold_value)
    PV = np.ones(x.shape, bool)  # assume no outliers (all are good)
    if mode == 'delete':
        if x.ndim > 1:
            raise ValueError(
                "when `mode` is 'delete', `x` must be 1d")
        hilim = np.empty(x.shape)
        lolim = np.empty(x.shape)
        for i in itertools.count(1):
            y = x[PV]
            if n > y.size:
                n = y.size - 1
            pv, hilim[PV], lolim[PV] = _find_outlier_peaks(
                y, n, sigma, min_limit, 0)
            if pv.all():
                break
            PV[PV] &= pv
            if maxiter > 0 and i >= maxiter:
                break
        x = x[PV]
    elif mode == 'average':
        if n > x.shape[axis]:
            n = x.shape[axis] - 1
        for i in itertools.count(1):
            pv, hilim, lolim = _find_outlier_peaks(
                x, n, sigma, min_limit, axis=axis)
            if pv.all():
                break
            _set_ave(x, ~pv, axis)
            PV &= pv
            if maxiter > 0 and i >= maxiter:
                break
    else:
        raise ValueError(
            "`mode` must be either 'delete' or 'average'")
    return SimpleNamespace(dx=x, pv=~PV,
                           hilim=hilim, lolim=lolim, niter=i)


def fixtime(olddata, sr=None, negmethod='sort', deldrops=True,
            dropval=-1.40130E-45, delouttimes=True, delspikes=False,
            base=None, fixdrift=False, getall=False):
    """
    Process recorded data to make an even time vector.

    Parameters
    ----------
    olddata : 2d ndarray or 2-element tuple
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
        calling :func:`despike`; the window size is set to 15 and the
        other defaults are accepted except for `mode` (see description
        in :func:`despike`). If a dict, it specifies all desired
        inputs to :func:`despike` except for the signal itself. If `n`
        is not included, it is set to 15. For example:
        ``delspikes=dict(n=31, sigma=5, maxiter=4)``. Note that the
        `mode` option is always reset to 'delete', even if you specify
        'average' in the dict.
    base : scalar or None; optional
        Scalar value that new time vector would hit exactly if within
        range. If None, new time vector is aligned to longest section
        of "good" data.
    fixdrift : bool; optional
        If True, shift data
    getall : bool; optional
        If True, return `fixinfo`; otherwise only `newdata` is
        returned.

    Returns
    -------
    newdata : 2d ndarray or tuple
        Cleaned up version of `olddata`. Will be 2d ndarray if
        `olddata` was ndarray; otherwise it is a tuple:
        ``(time, data)``.

    fixinfo : SimpleNamespace; optional
        Only returned if `getall` is True. Members:

        - `dropouts` : 1d ndarray
           If `deldrops` is True (the default), this is a True/False
           vector into `olddata` where drop-outs occurred. Otherwise,
           it is a True/False vector into `newdata`.

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

        - `spike_info` : SimpleNamespace or None
           If `delspikes` is True or a dict, `spike_info` contains:

               ========   ==============================
               `hilim`    As output by :func:`despike`
               `lolim`    As output by :func:`despike`
               `n`        Value input to :func:`despike`
               `niter`    As output by :func:`despike`
               `pv`       As output by :func:`despike`
               `t`        Time vector for `limit`
               ========   ==============================

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
           the count. :func:`np.histogram` is used to count which
           sample rate occurs most often (to the nearest multiple of 5
           in most cases). If there is a high percentage printed with
           that sample rate, it is likely the correct value to use (at
           least within 5 samples/second). If `sr` is not input,
           prompt user for `sr`.

       6.  Call :func:`despike` if requested to delete data spikes.

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

       11. If `fixdrift` is True, and step 10 did not issue a warning
           about too many turning points, search for additional
           turning points to account for drift (when the sample rate
           in the data is slightly off from the ideal).

       12. If step 10 did not issue a warning about too many turning
           points, the new time vector is shifted to align with the
           longest section of "good" old time steps.

       13. Loop over the segments defined by the turning points. Each
           segment will shifted left or right to fit with the new time
           vector. The longest section is not shifted due to step 12
           (unless that step was skipped because of too many turning
           points).

       14. If `base` is not None, the new time vector is shifted by up
           to a half time step such that it would hit `base` exactly
           (if it was in range).

       15. Fill in new data vector using best fit times. This means
           that gaps are filled with previous value (flat line). This
           routine does not do any linear interpolation.

    Examples
    --------
    >>> t = [0, 1, 6, 7]
    >>> y = [1, 2, 3, 4]
    >>> tn, yn = fixtime((t, y), sr=1)
    ==> Info: [min, max, ave, count (% occurrence)] time step:
    ==>           [1, 5, 2.33333, 1 (66.7%)]
    ==>       Corresponding sample rates:
    ==>           [1, 0.2, 0.428571, 1 (66.7%)]
    ==>       Note: "count" shows most frequent sample rate to
              nearest 0.2 samples/sec.
    ==> Using sample rate = 1
    >>> tn
    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> yn
    array([1, 2, 2, 2, 2, 2, 3, 4])
    """
    POOR = ('This may be a poor way to handle the current data, so '
            'please check the results carefully.')

    def _find_drops(d, dropval):
        dropouts = np.logical_or(np.isnan(d), np.isinf(d))
        if np.isfinite(dropval):
            dropouts = np.logical_or(
                dropouts, abs(d-dropval) < abs(dropval)/100)
        return dropouts

    def _del_drops(told, olddata, dropval):
        dropouts = _find_drops(olddata, dropval)
        if np.any(dropouts):
            keep = ~dropouts
            if np.any(keep):
                told = told[keep]
                olddata = olddata[keep]
            else:
                warn('there are only drop-outs!', RuntimeWarning)
                olddata = told = np.zeros(0)
        return told, olddata, dropouts

    def _return(t, data, dropouts, sr_stats, tp, dropval,
                getall, return_ndarray, spike_info):
        if return_ndarray:
            newdata = np.vstack((t, data)).T
        else:
            newdata = (t, data)
        if getall:
            if dropouts is None:
                dropouts = _find_drops(data, dropval)
            fixinfo = SimpleNamespace(
                dropouts=dropouts,
                sr_stats=sr_stats,
                tp=tp,
                spike_info=spike_info)
            return newdata, fixinfo
        return newdata

    def _del_outtimes(told, olddata, delouttimes):
        mn = told.mean()
        sig = 3*told.std(ddof=1)
        pv = np.logical_or(told < mn-sig, told > mn+sig)
        if np.any(pv):
            if delouttimes:
                warn('there are {:d} outlier times being deleted.'
                     ' These are times more than 3-sigma away '
                     'from the mean. {:s}'.format(pv.sum(), POOR),
                     RuntimeWarning)
                told = told[~pv]
                olddata = olddata[~pv]
            else:
                warn('there are {:d} outlier times that are NOT '
                     'being deleted because `delouttimes` is '
                     'False. These are times more than 3-sigma '
                     'away from the mean.'.format(pv.sum()),
                     RuntimeWarning)
        return told, olddata

    def _chk_negsteps(told, olddata, negmethod):
        difft = np.diff(told)
        if np.any(difft < 0):
            npos = (difft > 0).sum()
            if npos == 0:
                raise ValueError('there are no positive steps in the '
                                 'entire time vector. Cannot fix '
                                 'this.')

            nneg = (difft < 0).sum()
            if negmethod == 'stop':
                raise ValueError('There are {:d} negative time steps.'
                                 ' Stopping.'.format(nneg))

            if negmethod == 'sort':
                warn('there are {:d} negative time steps. '
                     'Sorting the data. {:s}'.format(nneg, POOR),
                     RuntimeWarning)
                j = np.argsort(told)
                told = told[j]
                olddata = olddata[j]
                difft = np.diff(told)
        return told, olddata, difft

    def _sr_calcs(difft, sr):
        min_ts = difft.min()
        max_ts = difft.max()
        ave_ts = difft.mean()

        max_sr = 1/min_ts
        min_sr = 1/max_ts
        ave_sr = 1/ave_ts

        # histogram count:
        Ldiff = len(difft)
        difft2 = difft[difft != 0]
        sr_all = 1/difft2
        sr1 = sr_all.min()
        if sr1 > 5:
            dsr = 5
        else:
            dsr = round(10*max(sr1, .1))/10
        bins = np.arange(dsr/2, sr_all.max()+dsr, dsr)
        cnt, bins = np.histogram(sr_all, bins)
        centers = (bins[:-1] + bins[1:])/2
        r = np.argmax(cnt)

        mx = cnt[r]/Ldiff * 100
        cnt_sr = centers[r]
        cnt_ts = 1/cnt_sr

        print('==> Info: [min, max, ave, count (% occurrence)] time step:')
        print('==>           [{:g}, {:g}, {:g}, {:g} ({:.1f}%)]'
              .format(min_ts, max_ts, ave_ts, cnt_ts, mx))
        sr_stats = np.array([max_sr, min_sr, ave_sr, cnt_sr, mx])
        print('==>       Corresponding sample rates:')
        print('==>           [{:g}, {:g}, {:g}, {:g} ({:.1f}%)]'
              .format(*sr_stats))
        print('==>       Note: "count" shows most frequent sample rate to')
        print('          nearest {} samples/sec.'.format(dsr))

        if mx > 90 or abs(cnt_sr-ave_sr) < dsr:
            defsr = round(cnt_sr/dsr)*dsr
        else:
            defsr = round(ave_sr/dsr)*dsr
        if sr == 'auto':
            sr = defsr
        elif not sr:                    # pragma: no cover
            ssr = input('==> Enter desired sample rate [{:g}]: '
                        .format(defsr))
            if not ssr:
                sr = defsr
            else:
                sr = float(ssr)
        print('==> Using sample rate = {:g}'.format(sr))
        return sr, sr_stats

    def _del_spikes(delspikes, told, olddata, difft):
        if not isinstance(delspikes, abc.MutableMapping):
            delspikes = dict()
        else:
            delspikes = dict(**delspikes)  # make a copy
        if 'n' not in delspikes:
            delspikes['n'] = 15
        delspikes['mode'] = 'delete'
        s = despike(olddata, **delspikes)
        olddata = s.dx
        t_limit = told
        if s.pv.any():
            told = told[~s.pv]
            difft = np.diff(told)
        spike_info = SimpleNamespace(
            pv=s.pv, t=t_limit, hilim=s.hilim, lolim=s.lolim,
            n=delspikes['n'], niter=s.niter)
        return told, olddata, spike_info, difft

    def _check_dt_size(difft, dt):
        n = len(difft)
        nsmall = np.sum(difft < .93*dt) / n
        nlarge = np.sum(difft > 1.07*dt) / n
        for n, s1, s2 in zip((nsmall, nlarge),
                             ('smaller', 'larger'),
                             ('low', 'high')):
            if n > .01:
                warn('there are a large ({:.2f}%) number of time '
                     'steps {:s} than {:g} by more than 7%. Double '
                     'check the sample rate; it might be too {:s}.'
                     .format(n*100, s1, dt, s2), RuntimeWarning)

    def _add_drift_turning_pts(tp, told, dt):
        # expand turning points if needed to account for drift
        # (sample rate being slightly off in otherwise good data)
        tp_drift = []
        Lold = len(told)
        for i in range(len(tp)-1):
            m, n = tp[i], tp[i+1]
            while n - m > .1*Lold:
                tdiff = np.arange(n - m)*dt - (told[m:n] - told[m])
                pv = abs(tdiff) > 1.01*dt/2
                if pv[-1]:
                    m += np.nonzero(~pv)[0].max() + 1
                    tp_drift.append(m)
                else:
                    break
        return np.sort(np.hstack((tp, tp_drift)))

    def _get_turning_points(told, dt, difft):
        tp = np.empty(len(told), bool)
        tp[0] = True
        tp[1:] = abs(np.diff(told)-dt) > dt/4
        tp[:-1] |= tp[1:]
        tp[-1] = True
        tp = np.nonzero(tp)[0]
        # tp_old = np.nonzero(get_turning_pts(told, atol=dt/4))[0]
        if len(tp)-2 > len(told) // 2:  # -2 to ignore ends
            align = False
            p = (len(tp)-2)/len(told)*100
            msg = ('there are too many turning points ({:.2f}%) to '
                   'account for drift or align the largest section. '
                   'Skipping steps 11 and 12.')
            warn(msg.format(p), RuntimeWarning)
        else:
            align = True
        return tp, align

    def _mk_initial_tnew(told, sr, dt, difft, fixdrift):
        L = int(round((told[-1] - told[0])*sr)) + 1
        tnew = np.arange(L)/sr + told[0]

        # get turning points and see if we should try to align:
        tp, align = _get_turning_points(told, dt, difft)

        if align:
            if fixdrift:
                tp = _add_drift_turning_pts(tp, told, dt)

            # align with the "good" range:
            j = np.argmax(np.diff(tp))
            t_good = told[tp[j]:tp[j+1]+1]

            p = _get_prev_index(tnew, t_good[0]+dt/2)
            tnew_good = tnew[p:p+len(t_good)]

            delt = np.mean(t_good[:len(tnew_good)] - tnew_good)
            adelt = abs(delt)
            if adelt > dt/2:
                sgn = np.sign(delt)
                factor = int(adelt/delt)
                dt = sgn*(adelt-factor*delt)
            tnew += delt
        return tnew, tp

    def _best_fit_segments(tnew, tp, told, dt):
        L = len(tnew)
        index = np.zeros(L, np.int64) - 1
        lastp = 0
        lastn = 0
        for i in range(len(tp)-1):
            m, n = tp[i], tp[i+1]
            p = _get_prev_index(tnew, told[m]+dt/2)
            index[lastp:p] = lastn
            if p+n-m > L:
                n = L + m - p
            index[p:p+n-m] = np.arange(m, n)
            lastp, lastn = p+n-m, n-1
        if lastp < L:
            # can last point be considered part of a good segment?
            if n - m == 1 and abs(told[n] - told[m] - dt) > dt/4:
                # no, so find index and fill in before moving on
                p = _get_prev_index(tnew, told[n]+dt/2)
                index[lastp:p] = lastn
                lastp = p
            index[lastp:] = n
        return index

    # begin main routine
    told, olddata, return_ndarray = _get_timedata(olddata)

    # check for drop outs:
    dropouts = sr_stats = tp = None
    if deldrops:
        told, olddata, dropouts = _del_drops(told, olddata, dropval)
        if len(told) == 0:
            return _return(told, olddata, dropouts, sr_stats, tp,
                           dropval, getall, return_ndarray,
                           spike_info=None)

    # check for outlier times ... outside 3-sigma
    told, olddata = _del_outtimes(told, olddata, delouttimes)

    # check for negative steps:
    told, olddata, difft = _chk_negsteps(told, olddata, negmethod)

    # sample rate calculations:
    sr, sr_stats = _sr_calcs(difft, sr)
    dt = 1/sr

    # delete spikes if requested:
    if delspikes:
        told, olddata, spike_info, difft = _del_spikes(
            delspikes, told, olddata, difft)
    else:
        spike_info = None

    # check for small and large time steps:
    _check_dt_size(difft, dt)

    # make initial new time vector aligned with longest range in
    # told of "good" time steps (tp: turning points):
    tnew, tp = _mk_initial_tnew(told, sr, dt, difft, fixdrift)

    # build a best-fit index by segments:
    index = _best_fit_segments(tnew, tp, told, dt)

    # if want new time to exactly hit base (if base were in range):
    if base is not None:
        t0 = tnew[0]
        t1 = base - t0 - round((base-t0)*sr)/sr
        tnew += t1

    # fill in new data vector with best-fit times (no interpolation):
    newdata = olddata[index]
    return _return(tnew, newdata, dropouts, sr_stats, tp,
                   dropval, getall, return_ndarray, spike_info)


def aligntime(dct, channels=None, mode='truncate', value=0):
    """Aligns the time vectors for specified channels in dct.

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
                print('Channel {} not found in `dct`.'.format(item))
        if err:
            raise ValueError('`dct` does not contain all requested'
                             'channels. See above.')
        parms = channels
    else:
        parms = list(dct.keys())

    # get time step:
    t, d, isarr = _get_timedata(dct[parms[0]])
    dt = (t[-1] - t[0])/(len(t) - 1)

    if mode == 'truncate':
        # loop to determine maximum overlap:
        tmin = t[0]
        tmax = t[-1]
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            if t[0] > tmax or t[-1] < tmin:
                raise ValueError('not all inputs overlap in time.')
            if not np.allclose(np.diff(t), dt):
                raise ValueError('not all time steps in {} match {}'
                                 .format(key, dt))
            tmin = max(tmin, t[0])
            tmax = min(tmax, t[-1])

        n = int(np.ceil((tmax-tmin)/dt))
        if (dt*n + tmin) < (tmax + dt/2):
            n += 1
        pv = np.arange(n)
        dctout = {}
        dctout['t'] = pv*dt + tmin
        start = tmin + dt/2  # so index finds closest point
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            i = _get_prev_index(t, start)
            dctout[key] = d[i+pv]
    else:
        # loop to determine maximum range:
        tmin = t[0]
        tmax = t[-1]
        for key in parms:
            t, d, isarr = _get_timedata(dct[key])
            if not np.allclose(np.diff(t), dt):
                raise ValueError('not all time steps in {} match {}'
                                 .format(key, dt))
            tmin = min(tmin, t[0])
            tmax = max(tmax, t[-1])

        n = int(np.ceil((tmax-tmin)/dt))
        if (dt*n + tmin) < (tmax + dt/2):
            n += 1
        dctout = {}
        t = dctout['t'] = np.arange(n)*dt + tmin
        for key in parms:
            old_t, old_d, isarr = _get_timedata(dct[key])
            i = _get_prev_index(t, old_t[0]+dt/2)
            new_d = np.empty(n)
            new_d[:] = value
            old_n = len(old_t)
            if i + old_n > n:
                old_n = n - i
            new_d[i:i+old_n] = old_d[:old_n]
            dctout[key] = new_d
    return dctout


def windowends(signal, portion=.01, ends='front', axis=-1):
    """Apply a 1-cos (half-Hann) window to the ends of a signal.

    Parameters
    ----------
    signal : 1d or 2d ndarray
        Vector or matrix; input time signal(s).
    portion : scalar, optional
        If > 1, specifies the number of points to window at each end.
        If in (0, 1], specifies the fraction of signal to window at
        each end: ``npts = int(portion * np.size(signal, axis))``.
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
        >>> _ = plt.figure(figsize=[8, 3])
        >>> sig = np.ones(100)
        >>> wesig = dsp.windowends(sig, 5, ends='both')
        >>> _ = plt.plot(sig, label='Original')
        >>> _ = plt.plot(wesig, label='windowends (ends="both")')
        >>> _ = plt.ylim(0, 1.2)
    """
    if ends == 'none':
        return signal
    signal = np.asarray(signal)
    ln = signal.shape[axis]
    if portion <= 1:
        n = int(portion * ln)
    else:
        n = int(portion)
    if n < 3:
        n = 3
    dims = np.ones(signal.ndim, int)
    dims[axis] = -1
    v = np.ones(ln, float)
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/(2*n-2))
    if ends == 'front' or ends == 'both':
        v[:n] = w
    if ends == 'back' or ends == 'both':
        v[-n:] = w[::-1]
    v = v.reshape(*dims)
    return signal*v


def waterfall(sig, sr, timeslice, tsoverlap, func, which, freq,
              t0=0.0, args=None, kwargs=None, slicefunc=None,
              sliceargs=None, slicekwargs=None):
    """
    Compute a 'waterfall' map over time and frequency (typically) using
    user-supplied function.

    Parameters
    ----------
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
    timeslice : scalar
        The length in seconds of each time slice.
    tsoverlap : scalar in [0, 1)
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
    func : function
        This function is called for each time slice and is expected to
        return amplitude values across the frequency range. Can return
        just the amplitudes, or it can return more values (like the
        frequency vector). The call is:
        ``func(sig_slice, *args, **kwargs)``. Note that the
        `sig_slice` input is first passed through `slicefunc` if one
        is provided (see below).
    which : integer or None
        Specifies which output of `func` is the amplitudes starting at
        zero. If None, `func` only returns the amplitudes. Note that
        `which` cannot be None if `freq` is an integer.
    freq : integer or vector
        If integer, it specifies which output of `func` is the
        frequency vector (starts at zero). If vector, it is the
        frequency vector directly.
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
        >>> from matplotlib import cm
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> frq = np.arange(1., 50.1)
        >>> Q = 20
        >>> mp, t, f = dsp.waterfall(sig, sr, 2, .5, srs.srs,
        ...                          which=None, freq=frq,
        ...                          args=(sr, frq, Q),
        ...                          kwargs=dict(eqsine=1),
        ...                          slicefunc=dsp.windowends,
        ...                          sliceargs=[.02],
        ...                          slicekwargs=dict(ends='front'))
        >>> _ = plt.figure('SRS Map')
        >>> _ = plt.contour(t, f, mp, 40, cmap=cm.cubehelix_r)
        >>> cbar = plt.colorbar()
        >>> cbar.filled = True
        >>> cbar.draw_all()
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.ylabel('Frequency (Hz)')
        >>> ttl = 'EQSINE Map of Sine-Sweep @ 4 oct/min, Q = 20'
        >>> _ = plt.title(ttl)

    .. plot::
        :context: close-figs

        Also show results on a 3D surface plot:

        >>> fig = plt.figure('SRS Map surface')
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> ax = fig.gca(projection='3d')
        >>> x, y = np.meshgrid(t, f)
        >>> surf = ax.plot_surface(x, y, mp, rstride=1, cstride=1,
        ...                        linewidth=0, cmap=cm.cubehelix_r)
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
            raise ValueError('`sig` must be a vector')
        sig = sig.ravel()
    if not (0 <= tsoverlap < 1):
        raise ValueError('`tsoverlap` must be in [0, 1)')

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sliceargs is None:
        sliceargs = ()
    if slicekwargs is None:
        slicekwargs = {}

    # work with integers for slicing:
    ntimeslice = int(sr*timeslice)
    if ntimeslice > sig.size:
        ntimeslice = sig.size
        timeslice = ntimeslice/sr

    inc = max(1, int(round(ntimeslice * (1.0 - tsoverlap))))
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

    try:
        flen = len(freq)
        mp = np.zeros((flen, tlen), float)
        col = 0
    except TypeError:
        if which is None:
            raise ValueError('`which` cannot be None when `freq` is '
                             'an integer')
        # do first iteration outside loop to get freq:
        s = slicefunc(sig[b:b+ntimeslice], *sliceargs, **slicekwargs)
        b += inc
        res = func(s, *args, **kwargs)
        freq = res[freq]
        flen = len(freq)
        mp = np.zeros((flen, tlen), float)
        mp[:, 0] = res[which]
        col = 1

    for j in range(col, tlen):
        s = slicefunc(sig[b:b+ntimeslice], *sliceargs, **slicekwargs)
        b += inc
        res = func(s, *args, **kwargs)
        if which is not None:
            mp[:, j] = res[which]
        else:
            mp[:, j] = res

    return mp, t+t0, freq


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


def calcenv(x, y, p=5, n=2000, method='max', base=0.,
            makeplot='clear', polycolor=(1, .7, .7), label='data'):
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
        percentage to shift the y data left and right
    n : integer; optional
        number of points to use for enveloping curve
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
        the base y-value (defines one side of the envelope); if None,
        no base y-value is used and `method` is automatically set to
        'both'
    makeplot : string; optional
        Specifies if and how to plot envelope in current figure:

        ==========   =============================================
        `makeplot`   Description
        ==========   =============================================
            'no'      do not plot envelope
         'clear'      plot envelope after clearing figure
           'add'      plot envelope without clearing figure
        ==========   =============================================

    polycolor : color specification; optional
        any valid matplotlib color specification for the color of the
        enveloping curve
    label : string; optional
        label for the x-y data on plot (only used if `makeplot` is
        True)

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
        >>> fig = plt.figure('calcenv', figsize=[10, 8])
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
        >>> fig.subplots_adjust(right=0.7)
    """
    # Original Yeti version: Tim Widrick
    # Converted from Yeti to Python: Jason Sloan, 10-6-2015
    # Added 'method' and 'base' options: Tim Widrick, 10-14-2015
    x, y = np.atleast_1d(x, y)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be monotonically ascending")

    if np.size(y) != np.size(x):
        raise ValueError("x and y must be the same length")

    if method not in ['max', 'min', 'both']:
        raise ValueError("`method` must be one of 'max', 'min',"
                         " or 'both")
    if base is None:
        method = 'both'
    up = 1+p/100
    dn = 1-p/100
    xe = np.linspace(x[0], x[-1], n)
    xe_max = xe_min = xe
    y2 = np.interp(xe, x, y)

    ye_max = np.zeros(n)
    ye_min = np.zeros(n)
    for i in range(n):
        pv = np.logical_and(xe >= xe[i]/up, xe <= xe[i]/dn)
        ye_max[i] = np.max(y2[pv])
        ye_min[i] = np.min(y2[pv])
        pv = np.logical_and(x >= xe[i]/up, x <= xe[i]/dn)
        if np.any(pv):
            ye_max[i] = max(ye_max[i], np.max(y[pv]))
            ye_min[i] = min(ye_min[i], np.min(y[pv]))

    if method == 'max':
        ye_max, xe_max = get_turning_pts(ye_max, xe, getindex=0)
    elif method == 'min':
        ye_max, xe_max = get_turning_pts(ye_min, xe, getindex=0)
    elif base is not None:
        ye_max[ye_max < base] = base
        ye_min[ye_min > base] = base
        ye_max, xe_max = get_turning_pts(ye_max, xe, getindex=0)
        ye_min, xe_min = get_turning_pts(ye_min, xe, getindex=0)

    if makeplot != 'no':
        envlabel = '$\pm${}% envelope'.format(p)
        if makeplot == 'clear':
            plt.clf()
        ln = plt.plot(x, y, label=label)[0]
        p = mpatches.Patch(color=polycolor, label=envlabel)
        if base is None:
            plt.fill_between(xe_max, ye_max, ye_min,
                             facecolor=polycolor, lw=0)
        else:
            plt.fill_between(xe_max, ye_max, base,
                             facecolor=polycolor, lw=0)
            if method == 'both':
                plt.fill_between(xe_min, ye_min, base,
                                 facecolor=polycolor, lw=0)
        plt.grid(True)
        h = [ln, p]
        plt.legend(handles=h, loc='best')
    else:
        h = None
    return xe_max, ye_max, xe_min, ye_min, h


def fdscale(y, sr, scale):
    """
    Scale a time signal in the frequency domain.

    Parameters
    ----------
    y : 1d or 2d array_like
        Signal(s) to be scaled. If 2d, each column is scaled.
    sr : scalar
        Sample rate.
    scale : 2d array_like
        A two column matrix of [freq scale]. It is automatically sized
        to the correct dimensions via linear interpolation (uses
        :func:`np.interp`).

    Returns
    -------
    y_new : 1d or 2d ndarray
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
        >>> _ = plt.figure('gensweep2')
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
        >>> _ = plt.tight_layout()
    """
    y = np.atleast_1d(y)
    scale = np.atleast_2d(scale)
    if y.ndim == 1:
        is1d = True
        y = y[:, None]
    else:
        is1d = False

    n = y.shape[0]
    even = n % 2 == 0
    m = n//2 + 1 if even else (n+1)//2
    freq = np.arange(m)*(sr/n)  # positive 1/2 frequency scale

    F = np.fft.rfft(y, axis=0)
    h = np.interp(freq, scale[:, 0], scale[:, 1])
    Ynew = np.fft.irfft(F * h[:, None], n=n, axis=0)

    if is1d:
        return Ynew.ravel()
    return Ynew


def nextpow2(x):
    """
    Return next power of two that is >= `x`

    Examples
    --------
    >>> nextpow2(4)
    4
    >>> nextpow2(5)
    8
    """
    return 1 << (x-1).bit_length()


def fftfilt(sig, w, bw=None, pass_zero=None, nyq=1.0, mag=0.5,
            makeplot='no'):
    """
    Filter time-domain signals using FFT with Gaussian ramps.

    Parameters
    ----------
    sig : 1d or 2d array_like
        Signal(s) to filter. If 2d, each column is filtered
    w : scalar or 1d array_like
        Edge (cutoff) frequencies where ``0.0 < w[i] < nyq`` for all
        ``i`` (`w` must not include 0.0 or `nyq`). Units are relative
        to the `nyq` input; so, for example, if `nyq` is the Nyquist
        frequency in Hz, `w` would be in Hz.
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
    makeplot : string; optional
        Specifies if and how to plot filter function:

        ==========   =============================================
        `makeplot`   Description
        ==========   =============================================
            'no'      do not plot filter function
         'clear'      plot after clearing figure
           'add'      plot without clearing figure
           'new'      plot in new figure
        ==========   =============================================

    Returns
    -------
    fsig : 1d or 2d ndarray
        Filtered version of `sig`
    freq : 1d ndarray
        Frequency vector from 0.0 to `nyq`
    h : 1d ndarray
        The frequency domain filter function

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
        >>> for j, (w, pz, yj) in enumerate(((7, None, y1),
        ...                                 ([7, 18], None, y2),
        ...                                 ([18, 45], None, y3),
        ...                                 (45, False, y4))):
        ...     _ = plt.figure('filters')
        ...     _ = plt.subplot(4, 1, j+1)
        ...     yf = dsp.fftfilt(y, w, pass_zero=pz, nyq=nyq,
        ...                      makeplot='add')[0]
        ...     _ = plt.xlim(0, 75)
        ...     _ = plt.figure('compare')
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, yj, t, yf)
        >>> _ = plt.figure('filters')
        >>> _ = plt.tight_layout()
        >>> _ = plt.figure('compare')
        >>> _ = plt.tight_layout()
    """
    def _make_h(freq, w, bw, pass_zero, mag):
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
            nearzero = 1/npts/4
            den = -(df*(npts-1)) ** 2 / np.log(nearzero)
            ramp = np.exp(-(np.arange(npts+1)*df) ** 2 / den)
            ramp[-1] = 0.0
            if not on:
                ramp = ramp[::-1]
            return ramp

        df = freq[1] - freq[0]
        if bw is None:
            bw = 0.01 * nyq
        bw = np.atleast_1d(bw)
        if bw.shape[0] != w.shape[0]:
            if bw.shape[0] != 1:
                raise ValueError(
                    '`bw` must be either a scalar or'
                    ' compatibly sized with `w`')
            _bw = np.empty(w.shape[0])
            _bw[:] = bw
            bw = _bw

        H = np.empty(freq.shape[0])
        on = pass_zero
        # position ramps; try to have "mag" point closest to each value
        # in w:
        I = 0
        for (_w, _bw) in zip(w, bw):
            j = np.argmin(abs(freq - _w))
            ramp = _get_ramp(df, _bw, on)
            n = ramp.shape[0]
            i = np.argmin(abs(ramp - mag))
            H[I:j-i] = ramp[0]
            I = j-i+n
            H[j-i:I] = ramp
            on = not on
        H[I:] = ramp[-1]
        return H

    # main routine:
    sig, w = np.atleast_1d(sig, w)
    if pass_zero is None:
        pass_zero = True if len(w) != 2 else False
    if sig.ndim == 1:
        is1d = True
        sig = sig[:, None]
    else:
        is1d = False

    if np.any(w > nyq):
        raise ValueError('value(s) in `w` exceed `nyq`')

    n = sig.shape[0]
    freq = np.fft.rfftfreq(n, 0.5/nyq)
    h = _make_h(freq, w, bw, pass_zero, mag)
    t = np.arange(n)
    ylines = interp.interp1d(t[[0, -1]],
                             sig[[0, -1], :],
                             axis=0)(t)
    y2 = sig - ylines
    n2 = nextpow2(n)
    Y = np.fft.rfft(y2, n2, axis=0)
    nf = freq.shape[0]
    Y[:nf] *= h[:, None]
    y_h = np.fft.irfft(Y, n2, axis=0)[:n]
    if pass_zero:
        y_h += ylines
    if is1d:
        y_h = y_h.ravel()
    if makeplot != 'no':
        if makeplot == 'new':
            plt.figure()
            makeplot = 'add'
        if makeplot == 'clear':
            plt.clf()
        plt.plot(freq, h)
        style = dict(color='k', lw=2, ls='--')
        for x in w:
            plt.axvline(x, **style)
        plt.axhline(mag, **style)
    return y_h, freq, h


def fftcoef(x, sr, coef='mag', window='boxcar', dodetrend=False,
            fold=True, maxdf=None):
    r"""
    FFT sine/cosine or magnitude/phase coefficients of a signal

    Parameters
    ----------
    x : 1d array_like
        The (real) signal to FFT
    sr : scalar
        The sample rate (samples/sec)
    coef : string; optional
        If set to 'mag', return magnitude and phase; otherwise,
        return A and B: the cosine and sine coefficients (see
        below)
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
    MAG_or_A : 1d ndarray
        The magnitude or cosine coefficients
    PHASE_or_B : 1d ndarray
        The phase (rad) or sine coefficients
    f : 1d ndarray
        Frequency vector (Hz)

    Notes
    -----
    The FFT results are scaled according to the 'coherent gain' of the
    window function (1.0 "boxcar")::

        scale = 1/coherent_gain
        coherent_gain = sum(window)/len(window)

    The coefficients are related to the original signal by the
    summations (if `fold` is True):

    .. math::
        x(t_n) = \sum\limits^{len(x)-1}_{k=0}
                  M_k\sin(k \omega t_n - \phi_k)

    .. math::
        x(t_n) = \sum\limits^{len(x)-1}_{k=0}
                  A_k \cos(k \omega t_n) +
                  B_k \sin(k \omega t_n)

    where :math:`\omega = 2 \pi \Delta f`, :math:`M` is the magnitude,
    and :math:`\phi` is the phase.

    The example below uses these formulas directly to upsample a
    signal. This is for demonstration only; to truly upsample a
    signal based on FFT, see :func:`scipy.signal.resample`.

    See also
    --------
    :func:`fftmap`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import dsp
        >>> n = 23
        >>> x = np.random.randn(n)
        >>> t = np.arange(n)
        >>> mag, phase, frq = dsp.fftcoef(x, 1.0)
        >>> w = 2*np.pi*frq[1]
        >>> #
        >>> # use a finer time vector for reconstructions:
        >>> t2 = np.arange(0., n, .05)
        >>> #
        >>> # reconstruct with magnitude and phase:
        >>> x2 = 0.0
        >>> for k, (m, p, f) in enumerate(zip(mag, phase, frq)):
        ...     x2 = x2 + m*np.sin(k*w*t2 - p)
        >>> #
        >>> # reconstruct with A and B:
        >>> A, B, frq = dsp.fftcoef(x, 1.0, coef='ab')
        >>> x3 = 0.0
        >>> for k, (a, b, f) in enumerate(zip(A, B, frq)):
        ...     x3 = x3 + a*np.cos(k*w*t2) + b*np.sin(k*w*t2)
        >>> #
        >>> _ = plt.plot(t, x, 'o', label='Original')
        >>> _ = plt.plot(t2, x2, label='FFT fit 1')
        >>> _ = plt.plot(t2, x3, '--', label='FFT fit 2')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.title('Using `fftcoef` for FFT curve fit')
        >>> _ = plt.xlabel('Time (s)')

    """
    n = len(x)
    if isinstance(window, (str, tuple)):
        window = signal.get_window(window, n)
    else:
        window = np.atleast_1d(window)
        if len(window) != n:
            raise ValueError('window size is {}; expected {} to '
                             'match signal'.format(len(window), n))
    scale = n/window.sum()
    window *= scale

    if dodetrend:
        x = signal.detrend(x) * window
    else:
        x = x * window

    def _fftsize(n, sr, maxdf):
        if maxdf and sr/n > maxdf:
            N = nextpow2(int(sr/maxdf))
        else:
            N = n
        return N

    N = _fftsize(n, sr, maxdf)
    if N > n:
        X = np.empty(N)
        X[:n] = x
        X[n:] = 0.0
    else:
        X = x
    even = not (N & 1)
    if even:
        m = N // 2 + 1
    else:
        m = (N+1) // 2

    F = np.fft.fft(X)
    f = np.arange(0., m)*(sr/N)
    if fold:
        a = 2.0 * F[:m].real / n
        a[0] = a[0] / 2.0
        if even:
            a[m-1] = a[m-1] / 2.0
        b = -2.0 * F[:m].imag / n
    else:
        a = F[:m].real / n
        b = -F[:m].imag / n

    if coef == 'mag':
        return np.sqrt(a**2 + b**2), np.arctan2(-a, b), f
    return a, b, f


def fftmap(timeslice, tsoverlap, sig, sr,
           window='hann', dodetrend=False, fold=True, maxdf=None):
    """
    Make an FFT map ('waterfall') over time and frequency.

    Parameters
    ----------
    timeslice : scalar
        The length in seconds of each time slice.
    tsoverlap : scalar
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
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
        >>> from matplotlib import cm
        >>> from pyyeti import dsp, ytools
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> mp, t, f = dsp.fftmap(2, .1, sig, sr)
        >>> pv = f <= 50.0
        >>> _ = plt.contour(t, f[pv], mp[pv], 40, cmap=cm.cubehelix)
        >>> cbar = plt.colorbar()
        >>> cbar.filled = True
        >>> cbar.draw_all()
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.ylabel('Frequency (Hz)')
        >>> ttl = 'FFT Map of Sine-Sweep @ 4 oct/min'
        >>> _ = plt.title(ttl)

    """
    return waterfall(sig, sr, timeslice, tsoverlap, fftcoef,
                     which=0, freq=2,
                     kwargs=dict(sr=sr, window=window,
                                 dodetrend=dodetrend,
                                 fold=fold, maxdf=maxdf))
