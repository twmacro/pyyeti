# -*- coding: utf-8 -*-
"""
Some math and I/O tools. The original set of functions provided by
this module were originally translated from Yeti (now a dead language)
to Python.
"""

import pickle
import gzip
import bz2
import sys
import contextlib
import warnings
from types import SimpleNamespace
from functools import wraps
import numpy as np
import scipy.linalg as linalg
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from pyyeti import guitools


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _check_3d(ax, need3d):
    if need3d and not hasattr(ax, "get_zlim"):
        raise ValueError("the axes object does not have a 3d projection")
    return ax


def _check_makeplot(
    makeplot, valid=("no", "new", "clear", "add"), figsize=None, need3d=False
):
    if makeplot not in valid:
        # makeplot must be an axes object if here ... check for 'plot'
        # attribute:
        if hasattr(makeplot, "plot"):
            return _check_3d(makeplot, need3d)
        raise ValueError(
            f"invalid `makeplot` setting; must be in {valid} or be an axes object"
        )

    if makeplot != "no":
        if makeplot == "new" or not plt.get_fignums():
            plt.figure(figsize=figsize)

        if makeplot == "add":
            fig = plt.gcf()
            if not fig.get_axes() and need3d:
                return fig.add_subplot(projection="3d")
            ax = plt.gca()
            return _check_3d(ax, need3d)

        if makeplot == "clear":
            plt.clf()

        if need3d:
            fig = plt.gcf()
            return fig.add_subplot(projection="3d")

        return plt.gca()

    return None


def _norm_vec(vec):
    return vec / np.linalg.norm(vec)


def _initial_circle_fit(basic):
    # See fit_circle_3d for description. Does steps 1-8.
    # - basic is 2d ndarray, 3 x n
    n = basic.shape[1]  # step 1
    if n < 3:
        raise ValueError(f"need at least 3 data points to fit circle, only have {n}")
    p1 = basic[:, 0]
    p2 = basic[:, n // 3]
    p3 = basic[:, 2 * n // 3]

    v1 = p2 - p1  # step 2
    v2 = p3 - p1
    z_l = _norm_vec(np.cross(v1, v2))  # step 3
    x_l = _norm_vec(v1)  # step 4
    y_l = np.cross(z_l, x_l)  # step 5
    basic2local = np.vstack((x_l, y_l, z_l))

    # compute center by using chord bisectors:
    b1 = np.cross(z_l, v1)  # step 6
    b2 = np.cross(z_l, v2)
    mid1 = (p1 + p2) / 2  # step 7
    mid2 = (p1 + p3) / 2
    arr = np.column_stack((b1, -b2))
    ab = np.linalg.lstsq(arr, mid2 - mid1, rcond=None)[0]  # step 8
    center = mid1 + ab[0] * b1

    radius = np.linalg.norm(p1 - center)

    return basic2local, center, radius


def fit_circle_2d(x, y, makeplot="no"):
    """
    Find radius and center point of x-y data points

    Parameters
    ----------
    x, y : 1d array_like
        Vectors x, y data points (in cartesian coordinates) that are
        on a circle: [x, y]
    makeplot : string or axes object; optional
        Specifies if and how to plot data showing the fit.

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
    p : 1d ndarray
        Vector: [xc, yc, R] where ``(xc, yc)`` defines the center of
        the circle and ``R`` is the radius.

    Notes
    -----
    Uses :func:`scipy.optimize.leastsq` to find optimum circle
    parameters.

    Examples
    --------
    For a test, provide precise x, y coordinates, but only for a 1/4
    circle:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti.ytools import fit_circle_2d
        >>> xc, yc, R = 1., 15., 35.
        >>> th = np.linspace(0., np.pi/2, 10)
        >>> x = xc + R*np.cos(th)
        >>> y = yc + R*np.sin(th)
        >>> fit_circle_2d(x, y, makeplot='new')
        array([  1.,  15.,  35.])
    """
    x, y = np.atleast_1d(x, y)
    basic2local, center, radius = _initial_circle_fit(np.vstack((x, y, 0 * x)))
    clx, cly = center[:2]

    # The optimization routine leastsq needs a function that returns
    # the residuals:
    #       y - func(p, x)
    # where "func" is the fit you're trying to match
    def circle_residuals(p, d):
        # p is [xc, yc, R]
        # d is [x;y] coordinates
        xc, yc, R = p
        n = len(d) // 2
        theta = np.arctan2(d[n:] - yc, d[:n] - xc)
        return d - np.hstack((xc + R * np.cos(theta), yc + R * np.sin(theta)))

    p0 = (clx, cly, radius)
    d = np.hstack((x, y))
    res = leastsq(circle_residuals, p0, args=(d,), full_output=1)
    sol = res[0]
    if res[-1] not in (1, 2, 3, 4):
        raise ValueError(":func:`scipy.optimization.leastsq` failed: {}".res[-2])
    ssq = np.sum(res[2]["fvec"] ** 2)
    if ssq > 0.01:
        msg = (
            "data points do not appear to form a good circle, sum "
            f"square of residuals = {ssq}"
        )
        warnings.warn(msg, RuntimeWarning)

    ax = _check_makeplot(makeplot)
    if ax:
        ax.scatter(x, y, c="r", marker="o", s=60, label="Input Points")
        th = np.arange(0, 361) * np.pi / 180.0
        (x, y, radius) = sol
        ax.plot(x + radius * np.cos(th), y + radius * np.sin(th), label="Fit")
        ax.axis("equal")
        ax.legend(loc="best", scatterpoints=1)

    return sol


def axis_equal_3d(ax, buffer_space=10):
    """
    Set equal axes for 3d plot

    Parameters
    ----------
    ax : axes object
        An axes object with a 3d projection
    buffer_space : scalar
        Percent of maximum limit (x, y, or z) to use for buffer room.

    Notes
    -----
    Since matplotlib doesn't have a 3d version of
    ``ax.axis('equal')``, this routine simply checks the current
    limits, and adjusts all axes to be equal. Therefore, for this to
    work properly, you must call this routine after you've plotted all
    your data.
    """
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    max_dimension = max(abs(extents[:, 1] - extents[:, 0]))
    centers = np.mean(extents, axis=1)
    r = max_dimension / 2 * (1 + buffer_space / 100)
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, f"set_{dim}lim")(ctr - r, ctr + r)


def _circle_fit_residuals(p, basic2local, basic, circ_parms):
    # p is [th, ph, xc, yc, zc]
    # - th & ph are angles to change the local z-axis direction:
    #   - th is angle to rotate local coords about x-axis
    #   - ph is angle to rotate result of th rotation about new
    #     local y-axis
    # - xc, yc, zc is center of circle in basic
    # d is [basic2local, basic]
    # - basic2local is original transformation
    # - basic is 3 x n: coordinates of all points in basic
    th, ph, xc, yc, zc = p
    c1, s1 = np.cos(th), np.sin(th)
    c2, s2 = np.cos(ph), np.sin(ph)
    # t1 = np.array([[1, 0, 0], [0, c1, s1], [0, -s1, c1]])
    # t2 = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
    # trans = t2 @ t1
    # or, doing it by hand:
    trans = np.array([[c2, s1 * s2, -s2 * c1], [0, c1, s1], [s2, -c2 * s1, c1 * c2]])
    new_basic2local = trans @ basic2local
    local = new_basic2local @ (basic - [[xc], [yc], [zc]])
    radii = np.linalg.norm(local[:2], axis=0)
    radius = radii.mean()
    if circ_parms is not None:
        circ_parms.basic2local = new_basic2local
        circ_parms.local = local
        circ_parms.radius = radius
        circ_parms.center = np.array([xc, yc, zc])
    return np.hstack((radii / radius - 1, local[2]))


def fit_circle_3d(basic, makeplot="no"):
    """
    Fit a circle through data points in 3D space

    Parameters
    ----------
    basic : 2d array_like, 3 x n
        Coordinates of data points in the basic (rectangular)
        coordinate system; rows `basic` are the x, y, and z
        coordinates
    makeplot : string or axes object; optional
        Specifies if and how to plot data showing the fit.

        ===========   ===============================
        `makeplot`    Description
        ===========   ===============================
            'no'      do not plot
         'clear'      plot after clearing figure
           'add'      plot without clearing figure
           'new'      plot in new figure
        axes object   plot in given axes (like 'add')
        ===========   ===============================

        Note that if `makeplot` is 'add' or an axes object, it must be
        3d; otherwise a ValueError exception is raised.

    Returns
    -------
    A SimpleNamespace with the members:

    local : 2d ndarray, 3 x n
        The coordinates of all points in a local (rectangular)
        coordinate system. The z-axis is perpendicular to the plane
        of the circle so the z-coordinate is 0.0 for all points.
    basic2local : 2d ndarray
        3 x 3 transform from basic to local. The local system is defined
        such that the z-axis is perpendicular to the plane of the
        circle.
    center : 1d ndarray
        Coordinates of circle in basic system (3 elements: x, y, z)
    radius : scalar
        Radius of circle
    ssqerr : scalar
        Sum of the squares of the radius and z-axis errors for each
        point. For a perfect fit, this will be zero.

    Notes
    -----
    At a high level, this routine works by: one, forming a
    (non-unique) transform to a local coordinate system (steps 1-5),
    two, finding the center in basic coordinates from the chord
    bisector approach (steps 6-9), three, finding the radius (step
    10), and four, optimizing the fit (step 11).

      1. Set ``n = basic.shape[1]``.
      2. Create two vectors: ``v1`` is from point 1 to point ``n // 3``,
         and ``v2`` is from point 1 to point ``2 * n // 3``.
      3. Form unit vector from the cross product of ``v1`` and ``v2`` to
         get a perpendicular axis to the circle. This is the local
         z-axis and the 3rd row of the transformation matrix
         `basic2local`.
      4. The local x-axis is defined as the unit vector of
         ``v1``. This is the 1st row of `basic2local`. Note that this
         is just the initial orientation; the final local x-axis will
         be oriented along the vector from the center of the circle to
         the first node.
      5. The local y-axis is the cross product of the local z-axis and
         the local x-axis. This is the 2nd row of `basic2local`.
      6. Noting that ``v1`` and ``v2`` are chords, the bisector of each
         chord is found by crossing the z-axis unit vector with the
         chord. Call these bisector vectors ``b1`` and ``b2``.
      7. Compute the midpoint of each chord: ``mid1`` is center of
         ``v1`` and ``mid2`` is center of ``v2``.
      8. Let `center` denote the center of the circle. Since both
         bisectors must pass through `center`::

             mid1 + alpha * b1 = center
             mid2 + beta * b2 = center

         where ``alpha`` and ``beta`` are unknown scalars. Subtracting
         the second equation from the first gives::

             alpha * b1 - beta * b2 = mid2 - mid1

         That equation is actually three equations with two of them
         being independent. Therefore, we can solve for ``alpha`` and
         ``beta`` using a least-squares approach (
         :func:`np.linalg.lstsq`). Then, we can use either of the two
         equations above to solve for `center`. Note the `center` is
         in basic coordinates.
      9. The coordinates of all points can now be calculated in the
         local coordinate system (note that the local z-coordinate is
         0.0 for all points)::

            local = basic2local @ (basic - center)

     10. The radius for each point in ``local`` is simply the root-sum-
         square of each local x & y coordinate. This routine computes
         the average radius and sum of the squares of the radius errors
         for each point.
     11. For cases where there are more than three data points, this
         routine optimizes the fit by using
         :func:`scipy.optimize.leastsq`. The five optimization
         variables are the direction of local z-axis (two angles) and
         the location of the center point.

    Examples
    --------
    Fit a circle through the three points: [3, 0, 0], [0, 3, 0] and
    [0, 0, 3]. The center should be at [1, 1, 1]:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti.ytools import fit_circle_3d
        >>> params = fit_circle_3d(3*np.eye(3), makeplot='new')
        >>> params.center
        array([ 1.,  1.,  1.])
    """
    basic = np.atleast_2d(basic)
    if basic.shape[0] != 3:
        raise ValueError(f"`basic` must have 3 rows (x, y, z), not {basic.shape[0]}")

    basic2local, center, radius = _initial_circle_fit(basic)

    # steps 9 and 10 are done in _circle_fit_residuals ... which is
    # called during the optimization

    # step 11: optimize solution:
    # - optimization parameters: [th, ph, xc, yc, zc]
    p0 = (0.0, 0.0, *center)
    res = leastsq(
        _circle_fit_residuals, p0, args=(basic2local, basic, None), full_output=True
    )
    sol = res[0]
    if res[-1] not in (1, 2, 3, 4):
        raise ValueError(f":func:`scipy.optimization.leastsq` failed: {res[-2]}")
    ssqerr = np.sum(res[2]["fvec"] ** 2)
    if ssqerr > 0.01:
        msg = (
            "data points do not appear to form a good circle, sum "
            f"square of residuals = {ssqerr}"
        )
        warnings.warn(msg, RuntimeWarning)

    # create output SimpleNamespace:
    circ_parms = SimpleNamespace(ssqerr=ssqerr)

    # get optimized fit ... it will be updated below after updating
    # angle of x axis to point to node 1, but we need the local
    # coordinates of node 1 to get angle:
    _circle_fit_residuals(sol, basic2local, basic, circ_parms)

    # put in pre-optimized parameters in case they're of interest:
    start_parms = SimpleNamespace()
    _circle_fit_residuals(p0, basic2local, basic, start_parms)
    circ_parms.start_parms = start_parms

    # reset the local x-axis to point to 1st node:
    th = np.arctan2(circ_parms.local[1, 0], circ_parms.local[0, 0])
    s = np.sin(th)
    c = np.cos(th)
    trans = np.array([[c, s], [-s, c]])
    basic2local[:2] = trans @ basic2local[:2]

    # get final, optimized fit:
    _circle_fit_residuals(sol, basic2local, basic, circ_parms)

    ax = _check_makeplot(makeplot, need3d=True)
    if ax:
        for item in "xyz":
            get_func = getattr(ax, f"get_{item}label")
            if not get_func():
                set_func = getattr(ax, f"set_{item}label")
                set_func(item.upper())
        ax.plot(*basic, "o", label="Data")

        # compute new points on circle in local coordinates:
        th = np.deg2rad(np.arange(0.0, 360))
        x = circ_parms.radius * np.cos(th)
        y = circ_parms.radius * np.sin(th)
        z = 0 * x

        # transform to basic coordinates and plot:
        circle_basic = (
            circ_parms.center + (np.column_stack((x, y, z)) @ circ_parms.basic2local)
        ).T
        ax.plot(*circle_basic, label="Fit")
        axis_equal_3d(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.get_figure().tight_layout()

    return circ_parms


def histogram(data, binsize):
    """
    Calculate a histogram

    Parameters
    ----------
    data : 1d array_like
        The data to do histogram counting on
    binsize : scalar
        Bin size

    Returns
    -------
    histo : 2d ndarray
        3-column matrix: [bincenter, count, percent]

    Notes
    -----
    Only bins that have count > 0 are included in the output. The
    bin-centers are: ``binsize*[..., -2, -1, 0, 1, 2, ...]``.

    The main difference from :func:`numpy.histogram` is how bins are
    defined and how the data are returned. For
    :func:`numpy.histogram`, you must either define the number of bins
    or the bin edges and the output will include empty bins; for this
    routine, you only define the binsize and only non-empty bins are
    returned.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> from pyyeti import ytools
    >>> data = [1, 2, 345, 2.4, 1.8, 345.1]
    >>> ytools.histogram(data, 1.0)
    array([[   1.    ,    1.    ,   16.6667],
           [   2.    ,    3.    ,   50.    ],
           [ 345.    ,    2.    ,   33.3333]])

    To try to get similar output from :func:`numpy.histogram` you have
    to define the bins:

    >>> binedges = [0.5, 1.5, 2.5, 344.5, 345.5]
    >>> cnt, bins = np.histogram(data, binedges)
    >>> cnt                                # doctest: +ELLIPSIS
    array([1, 3, 0, 2]...)
    >>> bins
    array([   0.5,    1.5,    2.5,  344.5,  345.5])
    """
    # use a generator to simplify the work; only yield a bin
    # if it has data:
    def _get_next_bin(data, binsize):
        data = np.atleast_1d(data)
        data = np.sort(data[np.isfinite(data)])
        if data.size == 0:
            yield [0, 0]
            return
        a = int(np.floor(data[0] / binsize))
        while data.size > 0:
            rgt = (a + 1 / 2) * binsize
            count = np.searchsorted(data, rgt)
            if count > 0:
                yield [a * binsize, count]
                data = data[count:]
                if data.size > 0:
                    a = int(np.floor(data[0] / binsize))
            else:
                a += 1

    bins = []
    for b in _get_next_bin(data, binsize):
        bins.append(b)
    histo = np.zeros((len(bins), 3))
    histo[:, :2] = bins
    s = histo[:, 1].sum()
    if s > 0:
        histo[:, 2] = 100 * histo[:, 1] / s
    return histo


@contextlib.contextmanager
def np_printoptions(*args, **kwargs):
    """
    Defines a context manager for :func:`numpy.set_printoptions`

    Parameters
    ----------
    *args, **kwargs : arguments for :func:`numpy.set_printoptions`
        See that function for a description of all available inputs.

    Notes
    -----
    This is for temporarily (locally) changing how NumPy prints
    matrices.

    Examples
    --------
    Print a matrix with current defaults, re-print it with 2 decimals
    using the "with" statement enabled by this routine, and then
    re-print it one last time again using the current defaults:

    >>> import numpy as np
    >>> from pyyeti import ytools
    >>> a = np.arange(np.pi/20, 1.5, np.pi/17).reshape(2, -1)
    >>> print(a)     # doctest: +SKIP
    [[ 0.15707963  0.3418792   0.52667877  0.71147834]
     [ 0.8962779   1.08107747  1.26587704  1.45067661]]
    >>> with ytools.np_printoptions(precision=2, linewidth=45,
    ...                             suppress=1):
    ...     print(a)
    [[ 0.16  0.34  0.53  0.71]
     [ 0.9   1.08  1.27  1.45]]
    >>> print(a)     # doctest: +SKIP
    [[ 0.15707963  0.3418792   0.52667877  0.71147834]
     [ 0.8962779   1.08107747  1.26587704  1.45067661]]
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def multmd(a, b):
    """
    Multiply a matrix and a diagonal, or two diagonals, in either
    order.

    Parameters
    ----------
    a : ndarray
        Matrix (2d array) or diagonal (1d array).
    b : ndarray
        Matrix (2d array) or diagonal (1d array).

    Returns
    -------
    c : ndarray
        Product of a * b.

    Notes
    -----
    This function should always be faster than numpy.dot() since the
    diagonal is not expanded to full size.

    Examples
    --------
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([10, 100])
    >>> ytools.multmd(a, b)
    array([[ 10, 200],
           [ 30, 400]])
    >>> ytools.multmd(b, a)
    array([[ 10,  20],
           [300, 400]])
    >>> ytools.multmd(b, b)
    array([  100, 10000])
    """
    if np.ndim(a) == 1:
        return (a * b.T).T
    else:
        return a * b


def mkpattvec(start, stop, inc):
    """
    Make a pattern "vector".

    Parameters
    ----------
    start : scalar or array
        Starting value.
    stop : scalar
        Ending value for first element in `start` (exclusive).
    inc : scalar
        Increment for first element in `start`.

    Returns
    -------
    pattvec : array
        Has one higher dimension than `start`. Shape = (-1,
        `start`.shape).

    Notes
    -----
    The first element of `start`, `stop`, and `inc` fully determine the
    number of increments that are generated. The other elements in
    `start` go along for the ride.

    Examples
    --------
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> ytools.mkpattvec([0, 1, 2], 24, 6).ravel()
    array([ 0,  1,  2,  6,  7,  8, 12, 13, 14, 18, 19, 20])
    >>> x = np.array([[10, 20, 30], [40, 50, 60]])
    >>> ytools.mkpattvec(x, 15, 2)
    array([[[10, 20, 30],
            [40, 50, 60]],
    <BLANKLINE>
           [[12, 22, 32],
            [42, 52, 62]],
    <BLANKLINE>
           [[14, 24, 34],
            [44, 54, 64]]])
    """
    start = np.array(start)
    s = start.ravel()
    xn = np.array([s + i for i in range(0, stop - s[0], inc)])
    return xn.reshape((-1,) + start.shape)


def isdiag(A, tol=1e-12):
    """
    Checks contents of square matrix A to see if it is approximately
    diagonal.

    Parameters
    ----------
    A : 2d numpy array
        If not square or if number of dimensions does not equal 2, this
        routine returns False.
    tol : scalar; optional
        The tolerance value.

    Returns
    -------
    True if `A` is a diagonal matrix, False otherwise.

    Notes
    -----
    If all off-diagonal values are less than `tol` times the maximum
    diagonal value (absolute-valuewise), this routine returns
    True. Otherwise, False is returned.

    See also
    --------
    :func:`mattype`

    Examples
    --------
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> A = np.diag(np.random.randn(5))
    >>> ytools.isdiag(A)
    True
    >>> A[0, 2] = .01
    >>> A[2, 0] = .01
    >>> ytools.isdiag(A)  # symmetric but not diagonal
    False
    >>> ytools.isdiag(A[1:, :])  # non-square
    False
    """
    if A.shape[0] != A.shape[1]:
        return False
    d = np.diag(A)
    max_off = abs(np.diag(d) - A).max()
    max_on = abs(d).max()
    return max_off <= tol * max_on


def mattype(A, mtype=None):
    """
    Checks contents of square matrix `A` to see if it is symmetric,
    hermitian, positive-definite, diagonal, and identity.

    Parameters
    ----------
    A : 2d array_like or None
        If not square or if number of dimensions does not equal 2, the
        return type is 0. If None, just return the `mattypes` output
        (not a tuple).
    mtype : string or None
        If string, it must be one of the `mattypes` listed below; in
        this case, True is returned if `A` is of the type specified or
        False otherwise. If None, `Atype` (if `A` is not None) and
        `mattypes` is returned. `mtype` is ignored if `A` is None.

    Returns
    -------
    flag : bool
        True/False flag specifying whether or not `A` is of the type
        specified by `mtype`. Not returned if either `A` or `mtype` is
        None. If `flag` is returned, it is the only returned value.
    Atype : integer
        Integer with bits set according to content. Not returned if
        `A` is None or if `mtype` is specified.
    mattypes : dictionary
        Provided for reference::

            mattypes = {'symmetric': 1,
                        'hermitian': 2,
                        'posdef': 4,
                        'diagonal': 8,
                        'identity': 16}

        Not returned if `mtype` is specified. This is the only return
        if `A` is None.

    Notes
    -----
    Here are some example usages::

        mattype(A)               # returns (Atype, mattypes)
        mattype(A, 'symmetric')  # returns True or False
        mattype(None)            # returns mattypes

    See also
    --------
    :func:`isdiag`

    Examples
    --------
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> A = np.eye(5)
    >>> ytools.mattype(A, 'identity')
    True
    >>> Atype, mattypes = ytools.mattype(A)
    >>>
    >>> Atype == 1 | 4 | 8 | 16
    True
    >>> if Atype & mattypes['identity']:
    ...     print('A is identity')
    A is identity
    >>> for i in sorted(mattypes):
    ...     print(f'{i:10s}: {mattypes[i]:2}')
    diagonal  :  8
    hermitian :  2
    identity  : 16
    posdef    :  4
    symmetric :  1
    >>> mattypes = ytools.mattype(None)
    >>> for i in sorted(mattypes):
    ...     print(f'{i:10s}: {mattypes[i]:2}')
    diagonal  :  8
    hermitian :  2
    identity  : 16
    posdef    :  4
    symmetric :  1
    """
    mattypes = {
        "symmetric": 1,
        "hermitian": 2,
        "posdef": 4,
        "diagonal": 8,
        "identity": 16,
    }
    if A is None:
        return mattypes
    Atype = 0
    A = np.asarray(A)
    if mtype is None:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return Atype, mattypes
        if np.allclose(A, A.T):
            Atype |= mattypes["symmetric"]
            if np.isrealobj(A):
                try:
                    linalg.cholesky(A)
                except linalg.LinAlgError:
                    pass
                else:
                    Atype |= mattypes["posdef"]
        elif np.iscomplexobj(A) and np.allclose(A, A.T.conj()):
            Atype |= mattypes["hermitian"]
            try:
                linalg.cholesky(A)
            except linalg.LinAlgError:
                pass
            else:
                Atype |= mattypes["posdef"]
        if isdiag(A):
            Atype |= mattypes["diagonal"]
            d = np.diag(A)
            if np.allclose(1, d):
                Atype |= mattypes["identity"]
        return Atype, mattypes

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    if mtype == "symmetric":
        return np.allclose(A, A.T)

    if mtype == "hermitian":
        return np.allclose(A, A.T.conj())

    if mtype == "posdef":
        if np.isrealobj(A):
            if not np.allclose(A, A.T):
                return False
        else:
            if not np.allclose(A, A.T.conj()):
                return False
        try:
            linalg.cholesky(A)
            return True
        except linalg.LinAlgError:
            return False

    if mtype in ("diagonal", "identity"):
        if isdiag(A):
            if mtype == "diagonal":
                return True
            d = np.diag(A)
            return np.allclose(1, d)
        else:
            return False

    raise ValueError("invalid `mtype`")


def sturm(A, lam):
    """
    Count number of eigenvalues <= `lam` of symmetric matrix `A`.

    Parameters
    ----------
    A : 2d ndarray
        Symmetric matrix to do Sturm counting on.
    lam : float or array of floats
        Eigenvalue cutoff(s).

    Returns
    -------
    count : 1d ndarray
        Contains number of eigenvalues below the cutoff values in
        `lam`. That is:  count[i] = number of eigenvalues in `A` below
        value `lam[i]`.

    Notes
    -----
    Computes the Hessenberg form of `A` which is tridiagonal if `A` is
    symmetric. Then it does a simple Sturm count on the results (code
    derived from LAPACK routine DLAEBZ).

    Examples
    --------
    Make symmetric matrix, count number of eigenvalues <= 0, and compute
    them:

    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> A = np.array([[  96.,  -67.,   36.,   37.,   93.],
    ...               [ -67.,   28.,   82.,  -66.,  -19.],
    ...               [  36.,   82.,  112.,    0.,  -61.],
    ...               [  37.,  -66.,    0.,  -14.,   47.],
    ...               [  93.,  -19.,  -61.,   47., -134.]])
    >>> w = la.eigh(A, eigvals_only=True)
    >>> w
    array([-195.1278,  -61.9135,  -10.1794,  146.4542,  208.7664])
    >>> ytools.sturm(A, 0)
    array([3])
    >>> ytools.sturm(A, [-200, -100, -20, 200, 1000])
    array([0, 1, 2, 4, 5])
    """
    # assuming A is symmetric, the hessenberg similarity form is
    # tridiagonal:
    h = linalg.hessenberg(A)

    # get diagonal and sub-diagonal:
    d = np.diag(h)
    s = np.diag(h, -1)
    abstol = np.finfo(float).eps
    ssq = s ** 2
    pivmin = max(1.0, np.max(s)) * abstol

    try:
        minp = len(lam)
    except TypeError:
        minp = 1
        lam = [lam]

    # count eigenvalues below lam[i] (adapted from LAPACK routine
    # DLAEBZ)
    count = np.zeros(minp, int)
    n = len(d)
    for i in range(minp):
        val = lam[i]
        tmp = d[0] - val
        if abs(tmp) < pivmin:
            tmp = -pivmin
        if tmp <= 0:
            c = 1
        else:
            c = 0
        for j in range(1, n):
            tmp = d[j] - ssq[j - 1] / tmp - val
            if abs(tmp) < pivmin:
                tmp = -pivmin
            if tmp <= 0:
                c += 1
        count[i] = c
    return count


def eig_si(
    K, M, Xk=None, f=None, p=10, mu=0, tol=1e-6, pmax=None, maxiter=50, verbose=True
):
    r"""
    Perform subspace iteration to calculate eigenvalues and eigenvectors.

    Parameters
    ----------
    K : ndarray
        The stiffness (symmetric).
    M : ndarray
        The mass (positive definite).
    Xk : ndarray or None
        Initial guess @ eigenvectors; # columns > `p`. If None,
        random vectors are used from ``np.random.rand()-.5``.
    f : scalar or None
        Desired cutoff frequency in Hz. `pmax` will override this if
        set. Takes precedence over `p` if both are input.
    p : scalar or None
        Number of desired eigenpairs (eigenvalues and eigenvectors).
        `pmax` will limit this if set. If `f` is input, `p` is
        calculated internally (from :func:`sturm`).
    mu : scalar
        Shift value in (rad/sec)^2 units. See notes.
    tol : scalar
        Eigenvalue convergence tolerance.
    pmax : scalar or None
        Maximum number of eigenpairs; no limit if None.
    maxiter : scalar
        Maximum number of iterations.
    verbose : bool
        If True, print status message for each iteration.

    Returns
    -------
    lam : ndarray
        Ideally, `p` converged eigenvalues.
    phi : ndarray
        Ideally, p converged eigenvectors.
    phiv : ndarray
        First p columns are `phi`, others are leftover iteration
        vectors which may be a good starting point for a second call.

    Notes
    -----
    The routine solves the eigenvalue problem:

    .. math::
       K \Phi = M \Phi \Lambda

    Where :math:`\Phi` is a matrix of right eigenvectors and
    :math:`\Lambda` is a diagonal matrix of eigenvalues.

    This routine works well for relatively small `p`. Trying to
    recover a large portion of modes may fail. Craig-Bampton models
    with residual flexibility modes also cause trouble.

    `mu` must not equal any eigenvalue. For systems with rigid-body
    modes, `mu` must be non-zero. Recommendations:

     - If you have eigenvalue estimates, set `mu` to be average of two
       widely spaced, low frequency eigenvalues. For example,
       ``mu = 5000`` worked well when the actual eigenvalues were:
       [0, 0, 0, 0, .05, 15.8, 27.8, 10745.4, ...]
     - ``mu = -10`` has worked well.
     - ``mu = 1/10`` of the first flexible eigenvalue has worked well.

    It may be temping to set `mu` to a higher value so a few higher
    frequency modes can be calculated. This might work, especially if
    you have good estimates for `Xk`. Otherwise, it is probably
    better to set `mu` to a lower value (as recommended above) and
    recover more modes to span the range of interest.

    In practice, unless you have truly good estimates for the
    eigenvectors (such as the output `phiv` may be), letting `Xk`
    start as random seems to work well.

    Examples
    --------
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> k = np.array([[5, -5, 0], [-5, 10, -5], [0, -5, 5]])
    >>> m = np.eye(3)
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> w, phi, phiv = ytools.eig_si(k, m, mu=-1) # doctest: +ELLIPSIS
    Iteration 1 completed
    Convergence: 3 of 3, tolerance range after 2 iterations is [...
    >>> print(abs(w))
    [  0.   5.  15.]
    >>> import scipy.linalg as linalg
    >>> k = np.random.randn(40, 40)
    >>> m = np.random.randn(40, 40)
    >>> k = np.dot(k.T, k) * 1000
    >>> m = np.dot(m.T, m) * 10
    >>> w1, phi1 = linalg.eigh(k, m, eigvals=(0, 14))
    >>> w2, phi2, phiv2 = ytools.eig_si(k, m, p=15, mu=-1, tol=1e-12,
    ...                                 verbose=False)
    >>> fcut = np.sqrt(w2.max())/2/np.pi * 1.001
    >>> w3, phi3, phiv3 = ytools.eig_si(k, m, f=fcut, tol=1e-12,
    ...                                 verbose=False)
    >>> print(np.allclose(w1, w2))
    True
    >>> print(np.allclose(np.abs(phi1), np.abs(phi2)))
    True
    >>> print(np.allclose(w1, w3))
    True
    >>> print(np.allclose(np.abs(phi1), np.abs(phi3)))
    True
    """
    n = np.size(K, 0)
    if f is not None:
        # use sturm sequence check to determine p:
        lamk = (2 * np.pi * f) ** 2
        p = sturm(K - lamk * M, 0)[0]

    if mu != 0:
        Kmod = K - mu * M
        Kd = linalg.lu_factor(Kmod)
    else:
        Kd = linalg.lu_factor(K)

    if pmax is not None and p > pmax:
        p = pmax
    if p > n:
        p = n
    q = max(2 * p, p + 8)
    if q > n:
        q = n
    if Xk is not None:
        c = np.size(Xk, 1)
    else:
        c = 0
    if c < q:
        if Xk is None:
            Xk = np.random.rand(n, q) - 0.5
        else:
            Xk = np.hstack((Xk, np.random.rand(n, q - c) - 0.5))
    elif c > q:
        Xk = Xk[:, :q]
    lamk = np.ones(q)
    nconv = 0
    loops = 0
    tolc = 1
    posdef = mattype(None)["posdef"]
    eps = np.finfo(float).eps
    while (tolc > tol or nconv < p) and loops < maxiter:
        loops += 1
        lamo = lamk
        MXk = np.dot(M, Xk)
        Xkbar = linalg.lu_solve(Kd, MXk)
        Mk = np.dot(np.dot(Xkbar.T, M), Xkbar)
        Kk = np.dot(Xkbar.T, MXk)

        # solve subspace eigenvalue problem:
        mtp = mattype(Mk)[0]
        if not mtp & posdef:
            factor = 1000 * eps
            pc = 0
            while 1:
                pc += 1
                Mk += np.diag(np.diag(Mk) * factor)
                factor *= 10.0
                mtp = mattype(Mk)[0]
                if mtp & posdef or pc > 5:
                    break

        if mtp & posdef:
            Mkll = linalg.cholesky(Mk, lower=True)
            Kkmod = linalg.solve(Mkll, linalg.solve(Mkll, Kk).T).T
            Kkmod = (Kkmod + Kkmod.T) / 2
            lamk, Qmod = linalg.eigh(Kkmod)
            Q = linalg.solve(Mkll.T, Qmod)
        else:
            raise ValueError(
                "subspace iteration failed, reduced mass"
                " matrix not positive definite"
            )

        dlam = np.abs(lamo - lamk)
        tolc = (dlam / np.abs(lamk))[:p]
        nconv = np.sum(tolc <= tol)
        mntolc = np.min(tolc)
        tolc = np.max(tolc)
        if loops > 1:
            if verbose:
                print(
                    f"Convergence: {nconv} of {p}, tolerance range after {loops} "
                    f"iterations is [{mntolc}, {tolc}]"
                )
        else:
            if verbose:
                print("Iteration 1 completed")
            nconv = 0
        Xk = np.dot(Xkbar, Q)
    return lamk[:p] + mu, Xk[:, :p], Xk


def gensweep(ppc, fstart, fstop, rate):
    r"""
    Generate a unity amplitude sine-sweep time domain signal.

    Parameters
    ----------
    ppc : scalar
        Points per cycle at `fstop` frequency.
    fstart : scalar
        Starting frequency in Hz
    fstop : scalar
        Stopping frequency in Hz
    rate : scalar
        Sweep rate in oct/min

    Returns
    -------
    sig : 1d ndarray
        The sine sweep signal.
    t : 1d ndarray
        Time vector associated with `sig` in seconds.
    f : 1d ndarray
        Frequency vector associated with `sig` in Hz.

    Notes
    -----
    The equation for a sine-sweep that uses a constant rate is:

    .. math::
        sig(f) = \sin \left (
        \frac {2 \pi (f-f_{start})}{\ln(2) \cdot rate}
        \right ) \\
        \text{where:  } f = f_{start} 2^{t \cdot rate}

    This type of sweep is linear in frequency; see plot from example.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ytools
        >>> import matplotlib.pyplot as plt
        >>> sig, t, f = ytools.gensweep(10, 1, 12, 8)
        >>> _ = plt.figure('Example')
        >>> plt.clf()
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(t, sig)
        >>> _ = plt.title('Sine Sweep vs Time')
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.xlim([t[0], t[-1]])
        >>> _ = plt.subplot(212)
        >>> _ = plt.plot(f, sig)
        >>> _ = plt.title('Sine Sweep vs Frequency')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.xlim([f[0], f[-1]])
        >>> _ = plt.tight_layout()
    """
    # make a unity sine sweep
    rate = rate / 60.0
    dt = 1.0 / fstop / ppc
    tstop = (np.log(fstop) - np.log(fstart)) / np.log(2.0) / rate
    t = np.arange(0.0, tstop + dt / 2, dt)
    f = fstart * 2 ** (t * rate)
    sig = np.sin(2 * np.pi / np.log(2) / rate * (f - fstart))
    return sig, t, f


def read_text_file(rdfunc):
    r"""
    Decorator that processes the file argument for reading

    Parameters
    ----------
    rdfunc : function
        Function that reads text from a file. The first argument to
        that function is a file argument. The file argument can be the
        name of a file, or a file_like object as returned by
        :func:`open` or :func:`io.StringIO`. It can also be the name
        of a directory or None; in these cases, a GUI is opened for
        file selection. For example, see
        :func:`pyyeti.nastran.bulk.rdgrids`.

    Returns
    -------
    function
        Function that processes the file argument before calling
        `rdfunc`.

    See also
    --------
    :func:`write_text_file`

    Examples
    --------
    >>> from pyyeti.ytools import read_text_file, write_text_file
    >>> from io import StringIO
    >>> @read_text_file
    ... def doread(f):
    ...     return f.readline()
    >>> @write_text_file
    ... def dowrite(f, string, number):
    ...     f.write(f'{string} = {number:.3f}\n')
    >>> with StringIO() as f:
    ...     dowrite(f, 'param', number=45.3)
    ...     _ = f.seek(0, 0)
    ...     s = doread(f)
    >>> s
    'param = 45.300\n'
    """

    @wraps(rdfunc)
    def mod_func(f, *args, **kwargs):
        f = guitools.get_file_name(f, read=True)
        if isinstance(f, str):
            with open(f, "r") as fin:
                return rdfunc(fin, *args, **kwargs)
        else:
            return rdfunc(f, *args, **kwargs)

    return mod_func


def write_text_file(wtfunc):
    r"""
    Decorator that processes the file argument for writing

    Parameters
    ----------
    wtfunc : function
        Function that writes text to a file. The first argument to
        that function is a file argument. The file argument can be the
        name of a file, or a file_like object as returned by
        :func:`open` or :func:`io.StringIO`. It can also be input as
        the integer 1 to write to stdout (or use
        ``sys.stdout``). Finally, it can also be the name of a
        directory or None; in these cases, a GUI is opened for file
        selection. To write to a string, ``import io`` and set ``f =
        io.StringIO()``; afterwards, retrieve string by
        ``f.getvalue()``. For example, see
        :func:`pyyeti.nastran.bulk.wtgrids`.

    Returns
    -------
    function
        Function that processes the file argument before calling
        `wtfunc`.

    See also
    --------
    :func:`read_text_file`

    Examples
    --------
    >>> from pyyeti.ytools import write_text_file
    >>> @write_text_file
    ... def dowrite(f, string, number):
    ...     f.write(f'{string} = {number:.3f}\n')
    >>> dowrite(1, 'param', number=45.3)
    param = 45.300
    """

    @wraps(wtfunc)
    def mod_func(f, *args, **kwargs):
        f = guitools.get_file_name(f, read=False)
        if isinstance(f, str):
            with open(f, "w") as fout:
                return wtfunc(fout, *args, **kwargs)
        else:
            if f == 1:
                f = sys.stdout
            return wtfunc(f, *args, **kwargs)

    return mod_func


def _get_fopen(name, read=True):
    """Utility for save/load"""
    name = guitools.get_file_name(name, read)
    if name.endswith(".pgz"):
        fopen = gzip.open
    elif name.endswith(".pbz2"):
        fopen = bz2.open
    else:
        fopen = open
    return name, fopen


def save(name, obj):
    """
    Save an object to a file via pickling.

    Parameters
    ----------
    name : string or None
        Name of file or directory or None. If file name, should end in
        either '.p' for an uncompressed pickle file, or in '.pgz' or
        '.pbz2' for a gzip or bz2 compressed pickle file. Note: only
        '.pgz' and 'pbz2' are checked for; anything else is
        uncompressed. If `name` is the name of a directory or None, a
        GUI is opened for file selection.
    obj : any
        Any object to be pickled.

    Notes
    -----
    See :mod:`pickle`
    """
    name, fopen = _get_fopen(name, read=False)
    with fopen(name, "wb") as f:
        pickle.dump(obj, file=f, protocol=-1)


def load(name):
    """
    Load an object from a pickle file.

    Parameters
    ----------
    name : string
        Name of file. Should end in either '.p' for an uncompressed
        pickle file, or in '.pgz' or '.pbz2' for a gzip or bz2
        compressed pickle file. Note: only '.pgz' and 'pbz2' are
        checked for; anything else is uncompressed.

    Returns
    -------
    obj : any
        The pickled object.

    Notes
    -----
    See :mod:`pickle`
    """
    name, fopen = _get_fopen(name, read=True)
    with fopen(name, "rb") as f:
        return pickle.load(f)


def reorder_dict(ordered_dict, keys, where):
    """
    Copy and reorder an ordered dictionary

    .. note::

        This will also work for regular Python ``dict``
        (:class:`dict`) objects for Python 3.7+ where insertion order
        is guaranteed. See also :class:`collections.OrderedDict`.

    Parameters
    ----------
    ordered_dict : instance of `OrderedDict` or other ordered mapping
        The ordered dictionary to copy and put in a new order. Must
        accept tuple of 2-tuples, eg, ``((key1, value1),
        (key2, value2), ...)`` in its ``__init__`` function.
    keys : iterable
        Iterable of keys in the order desired. Note that all keys do
        not need to be included, just those where a new order is
        desired. For example, if you just want to ensure that 'scltm'
        is first::

            new_dict = reorder_dict(ordered_dict, ['scltm'], 'first')

    where : string
        Either 'first' or 'last'. Specifies where to put the
        reordered items in the final order.

    Returns
    -------
    ordered dictionary object (same type as `ordered_dict`)
        A new ordered dictionary, reordered as specified

    Raises
    ------
    ValueError
        If a key is not found
    ValueError
        If `where` is not 'first' or 'last'

    Examples
    --------
    >>> from collections import OrderedDict
    >>> from pyyeti.ytools import reorder_dict
    >>> dct = OrderedDict((('one', 1),
    ...                    ('two', 2),
    ...                    ('three', 3)))
    >>> dct
    OrderedDict([('one', 1), ('two', 2), ('three', 3)])
    >>> reorder_dict(dct, ['three', 'two'], 'first')
    OrderedDict([('three', 3), ('two', 2), ('one', 1)])
    """

    def reorder_keys(all_keys, keys, where):
        all_keys = list(all_keys)
        keys = list(keys)

        # ensure all keys are in all_keys:
        for k in keys:
            if k not in all_keys:
                raise ValueError(f'Key "{k}" not found. Order unchanged.')

        if where == "first":
            new_keys = list(keys)
            new_keys.extend(k for k in all_keys if k not in keys)
        elif where == "last":
            new_keys = [k for k in all_keys if k not in keys]
            new_keys.extend(keys)
        else:
            raise ValueError(f"`where` must be 'first' or 'last', not {where!r}")

        return new_keys

    return type(ordered_dict)(
        (k, ordered_dict[k]) for k in reorder_keys(ordered_dict, keys, where)
    )
