# -*- coding: utf-8 -*-
"""Plain Python version of the rainflow algorithm. Numba
(``numba.jit(nopython=True)``) is used if available to achieve speeds
comparable to the compiled-c version.
"""
import numpy as np


def rainflow(peaks, getoffsets=False):
    """
    Rainflow cycle counting in plain Python (slow).

    Parameters
    ----------
    peaks : 1d array-like
        Vector of alternating peaks (as returned by
        :func:`pyyeti.cyclecount.findap`, for example)
    getoffsets : bool; optional
        If True, the tuple ``(rf, os)`` is returned; otherwise, only
        `rf` is returned.

    Returns
    -------
    rf : 2d ndarray
        n x 3 matrix with the rainflow cycle count information
        ``[amp, mean, count]``:

            - amp is the cycle amplitude (half the peak-to-peak range)
            - mean is mean of the cycle
            - count is either 0.5 or 1.0 depending on whether it's
              half or full cycle

    os : 2d ndarray; optional
        n x 2 matrix of cycle offsets ``[start, stop]``. Only returned
        if `getoffsets` is True. The start and stop values are:

            - start is the offset into `peaks` for start of cycle
            - stop is the offset into `peaks` for end of cycle

    Notes
    -----
    This algorithm is derived from reference [#rain2]_. The compiled C
    version is preferred over this one and is very fast (the logic is
    the same).

    References
    ----------
    .. [#rain2] "Standard Practices for Cycle Counting in Fatigue
           Analysis", ASTM E 1049 - 85 (Reapproved 2005).

    Examples
    --------
    Run the example from the ASTM paper:

    >>> from pyyeti.rainflow.py_rain import rainflow
    >>> rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    array([[ 1.5, -0.5,  0.5],
           [ 2. , -1. ,  0.5],
           [ 2. ,  1. ,  1. ],
           [ 4. ,  1. ,  0.5],
           [ 4.5,  0.5,  0.5],
           [ 4. ,  0. ,  0.5],
           [ 3. ,  1. ,  0.5]])

    With offsets:

    >>> rf, os = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2],
    ...                   getoffsets=True)
    >>> rf
    array([[ 1.5, -0.5,  0.5],
           [ 2. , -1. ,  0.5],
           [ 2. ,  1. ,  1. ],
           [ 4. ,  1. ,  0.5],
           [ 4.5,  0.5,  0.5],
           [ 4. ,  0. ,  0.5],
           [ 3. ,  1. ,  0.5]])
    >>> os              # doctest: +ELLIPSIS
    array([[0, 1],
           [1, 2],
           [4, 5],
           [2, 3],
           [3, 6],
           [6, 7],
           [7, 8]]...)
    """
    peaks = np.atleast_1d(peaks)
    L = peaks.size if peaks.ndim == 1 else 0
    if L < 2:
        raise ValueError("`peaks` must be a real vector with length >= 2")
    if getoffsets:
        return _rainflow2(peaks, L)
    return _rainflow1(peaks, L)


def _rainflow1(peaks, L):
    # not getting offsets:
    pts = np.empty(L)
    rf = np.empty((L - 1, 3))
    j = -1
    fullcyclesp1 = 1  # full cycles plus 1
    n = -1
    for k in range(L):
        # /* step 1 from [1]: */
        j += 1
        pts[j] = peaks[k]
        # /* step 2 from [1]: */
        while j > 1:
            # /* step 3 from [1]: */
            Y = abs(pts[j - 2] - pts[j - 1])
            X = abs(pts[j - 1] - pts[j])
            if X < Y:
                break
            if j == 2:
                # /* step 5 from [1]: */
                # /* [count Y as half cycle] */
                n += 1
                rf[n, 0] = Y / 2
                rf[n, 1] = (pts[0] + pts[1]) / 2
                rf[n, 2] = 0.5
                pts[0] = pts[1]  # /* discard j-2 pt */
                pts[1] = pts[2]
                j = 1
            else:
                # /* step 4 from [1]: */
                # /* [count Y as full cycle] */
                fullcyclesp1 += 1
                n += 1
                rf[n, 0] = Y / 2
                rf[n, 1] = (pts[j - 2] + pts[j - 1]) / 2
                rf[n, 2] = 1.0
                pts[j - 2] = pts[j]  # /* discard j-2, j-1 pts */
                j -= 2

    # /* step 6 from [1]: */
    # /* [count all ranges in pts as half cycles] */
    A = pts[0]
    for k in range(j):
        B = pts[k + 1]
        n += 1
        rf[n, 0] = abs(A - B) / 2
        rf[n, 1] = (A + B) / 2
        rf[n, 2] = 0.5
        A = B

    return rf[: L - fullcyclesp1]


def _rainflow2(peaks, L):
    """Utility routine for :func:`rainflow`; returns (rf, os)."""
    pts = np.empty(L)
    rf = np.empty((L - 1, 3))
    j = -1
    fullcyclesp1 = 1  # full cycles plus 1
    n = -1
    cycle_index = np.empty(L, np.int64)
    os = np.empty((L - 1, 2), np.int64)
    for k in range(L):
        # /* step 1 from [1]: */
        j += 1
        pts[j] = peaks[k]
        cycle_index[j] = k
        # /* step 2 from [1]: */
        while j > 1:
            # /* step 3 from [1]: */
            Y = abs(pts[j - 2] - pts[j - 1])
            X = abs(pts[j - 1] - pts[j])
            if X < Y:
                break
            if j == 2:
                # /* step 5 from [1]: */
                # /* [count Y as half cycle] */
                n += 1
                rf[n, 0] = Y / 2
                rf[n, 1] = (pts[0] + pts[1]) / 2
                rf[n, 2] = 0.5
                os[n, 0] = cycle_index[0]
                os[n, 1] = cycle_index[1]
                pts[0] = pts[1]  # /* discard j-2 pt */
                pts[1] = pts[2]
                cycle_index[0] = cycle_index[1]
                cycle_index[1] = cycle_index[2]
                j = 1
            else:
                # /* step 4 from [1]: */
                # /* [count Y as full cycle] */
                fullcyclesp1 += 1
                n += 1
                rf[n, 0] = Y / 2
                rf[n, 1] = (pts[j - 2] + pts[j - 1]) / 2
                rf[n, 2] = 1.0
                os[n, 0] = cycle_index[j - 2]
                os[n, 1] = cycle_index[j - 1]
                pts[j - 2] = pts[j]  # /* discard j-2, j-1 pts */
                cycle_index[j - 2] = cycle_index[j]
                j -= 2

    # /* step 6 from [1]: */
    # /* [count all ranges in pts as half cycles] */
    A = pts[0]
    for k in range(j):
        B = pts[k + 1]
        n += 1
        rf[n, 0] = abs(A - B) / 2
        rf[n, 1] = (A + B) / 2
        rf[n, 2] = 0.5
        os[n, 0] = cycle_index[k]
        os[n, 1] = cycle_index[k + 1]
        A = B

    return rf[: L - fullcyclesp1], os[: L - fullcyclesp1]


try:
    import numba
except ImportError:
    pass
else:
    _rainflow1 = numba.jit(nopython=True)(_rainflow1)
    _rainflow2 = numba.jit(nopython=True)(_rainflow2)
