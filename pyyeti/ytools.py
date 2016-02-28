# -*- coding: utf-8 -*-
"""
Some miscellaneous tools translated from Yeti to Python.
"""

import numpy as np
import scipy.linalg as linalg
import scipy.signal as signal
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from warnings import warn
import sys

# to accommodate pre-3.5:
try:
    from math import gcd
except ImportError:
    from fractions import gcd


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
        return (a*b.T).T
    else:
        return a*b


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
    >>> ytools.mkpattvec([0, 1, 2], 24, 6).flatten()
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
    s = start.flatten()
    xn = np.array([s+i for i in range(0, stop-s[0], inc)])
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
    if max_off < tol*max_on:
        return True
    else:
        return False

#    try:
#        if np.allclose(A, np.diag(d)):
#            return True
#        return False
#    except ValueError:
#        return False


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
    ...     print('{:10s}: {:2}'.format(i, mattypes[i]))
    diagonal  :  8
    hermitian :  2
    identity  : 16
    posdef    :  4
    symmetric :  1
    >>> mattypes = ytools.mattype(None)
    >>> for i in sorted(mattypes):
    ...     print('{:10s}: {:2}'.format(i, mattypes[i]))
    diagonal  :  8
    hermitian :  2
    identity  : 16
    posdef    :  4
    symmetric :  1
    """
    mattypes = {'symmetric': 1,
                'hermitian': 2,
                'posdef': 4,
                'diagonal': 8,
                'identity': 16}
    if A is None:
        return mattypes
    Atype = 0
    A = np.asarray(A)
    if mtype is None:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            return Atype, mattypes
        if np.allclose(A, A.T):
            Atype |= mattypes['symmetric']
            if np.isrealobj(A):
                try:
                    linalg.cholesky(A)
                    Atype |= mattypes['posdef']
                except linalg.LinAlgError:
                    pass
        elif np.iscomplexobj(A) and np.allclose(A, A.T.conj()):
            Atype |= mattypes['hermitian']
            try:
                linalg.cholesky(A)
                Atype |= mattypes['posdef']
            except linalg.LinAlgError:
                pass
        if isdiag(A):
            Atype |= mattypes['diagonal']
            d = np.diag(A)
            if np.allclose(1, d):
                Atype |= mattypes['identity']
        return Atype, mattypes

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    if mtype == 'symmetric':
        return np.allclose(A, A.T)

    if mtype == 'hermitian':
        return np.allclose(A, A.T.conj())

    if mtype == 'posdef':
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

    if mtype in ('diagonal', 'identity'):
        if isdiag(A):
            if mtype == 'diagonal':
                return True
            d = np.diag(A)
            if np.allclose(1, d):
                return True
            else:
                return False
        else:
            return False

    raise ValueError('invalid `mtype`')


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
    ssq = s**2
    pivmin = max(1., np.max(s)) * abstol

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
            tmp = d[j] - ssq[j-1] / tmp - val
            if abs(tmp) < pivmin:
                tmp = -pivmin
            if tmp <= 0:
                c += 1
        count[i] = c
    return count


def eig_si(K, M, Xk=None, f=None, p=10, mu=0, tol=1e-6,
           pmax=None, maxiter=50, verbose=True):
    """
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
    This routine works well for relatively small `p`. Trying to
    recover all modes may fail. Craig-Bampton models with residual
    flexibility modes also cause trouble.

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
    >>> print(w+0)
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
        lamk = (2*np.pi*f)**2
        p = sturm(K - lamk*M, 0)[0]

    if mu != 0:
        Kmod = K - mu*M
        Kd = linalg.lu_factor(Kmod)
    else:
        Kd = linalg.lu_factor(K)

    if pmax is not None and p > pmax:
        p = pmax
    if p > n:
        p = n
    q = max(2*p, p+8)
    if q > n:
        q = n
    if Xk is not None:
        c = np.size(Xk, 1)
    else:
        c = 0
    if c < q:
        if Xk is None:
            Xk = np.random.rand(n, q)-.5
        else:
            Xk = np.hstack((Xk, np.random.rand(n, q-c)-.5))
    elif c > q:
        Xk = Xk[:, :q]
    lamk = np.ones(q)
    nconv = 0
    loops = 0
    tolc = 1
    posdef = mattype(None)['posdef']
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
        if not (mtp & posdef):
            factor = 1000*eps
            pc = 0
            while 1:
                pc += 1
                Mk += np.diag(np.diag(Mk)*factor)
                factor *= 10.
                mtp = mattype(Mk)[0]
                if mtp & posdef or pc > 5:
                    break

        if mtp & posdef:
            Mkll = linalg.cholesky(Mk, lower=True)
            Kkmod = linalg.solve(Mkll, linalg.solve(Mkll, Kk).T).T
            Kkmod = (Kkmod+Kkmod.T)/2
            lamk, Qmod = linalg.eigh(Kkmod)
            Q = linalg.solve(Mkll.T, Qmod)
        else:
            raise ValueError('subspace iteration failed, reduced mass'
                             ' matrix not positive definite')

        dlam = np.abs(lamo - lamk)
        tolc = (dlam / np.abs(lamk))[:p]
        nconv = np.sum(tolc <= tol)
        mntolc = np.min(tolc)
        tolc = np.max(tolc)
        if loops > 1:
            if verbose:
                print('Convergence: {} of {}, tolerance range after {} '
                      'iterations is [{}, {}]'.format(nconv, p, loops,
                                                      mntolc, tolc))
        else:
            if verbose:
                print('Iteration 1 completed')
            nconv = 0
        Xk = np.dot(Xkbar, Q)
    return lamk[:p]+mu, Xk[:, :p], Xk


def freq_oct(n, frange=[1., 10000.], exact=False, trim='band',
             anchor=None):
    r"""
    Get frequency vector on an octave scale.

    Parameters
    ----------
    n : scalar
        Specify octave band:  1 for full octave, 3 for 1/3 octave, 6
        for 1/6, etc.
    frange : 1d array_like
        Specifies bounds for the frequency vector; only the first and
        last elements are used. If the first element <= 0.0, 1. is used
        instead. See also the `trim` input.
    trim : string
        Specify how to trim frequency vector to `frange`. If `trim` is
        "band", trimming is such that the first band includes
        `frange[0]` and the last band include `frange[1]`. If `trim`
        is "center", trimming is such that the `frange` values are just
        outside the range of the returned vector.
    exact : bool
        If False, return an approximate octave scale so that it hits
        the power of 10s, achored at 1 Hz. If True, return an exact
        octave scale, anchored at 1000 Hz.
    anchor : scalar or None
        If scalar, it specifies the anchor. If None, the anchor used
        is specified above (1 or 1000).

    Returns
    -------
    F : 1d ndarray
        Contains the center frequencies on an octave scale.
    F_lower : 1d ndarray
        Same size as `F`, band lower frequencies.
    F_upper : 1d ndarray
        Same size as `F`, band upper frequencies.

    Notes
    -----
    If `exact` is False, the center, lower, and upper frequencies are
    computed from (where :math:`i` and :math:`j` are integers
    according to `frange` and `trim`):

    .. math::
        \begin{aligned}
        F &= anchor \cdot 10^{3 [i, i+1, i+2, ..., j] / (10 n)} \\
        F_{lower} &= F/10^{3/(20 n)} \\
        F_{upper} &= F \cdot 10^{3/(20 n)}
        \end{aligned}

    If `exact` is True:

    .. math::
        \begin{aligned}
        F &= anchor \cdot 2^{[i, i+1, i+2, ..., j] / n} \\
        F_{lower} &= F/2^{1/(2 n)} \\
        F_{upper} &= F \cdot 2^{1/(2 n)}
        \end{aligned}

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import ytools
    >>> np.set_printoptions(precision=4)
    >>> np.array(ytools.freq_oct(3, [505, 900]))
    array([[  501.1872,   630.9573,   794.3282,  1000.    ],
           [  446.6836,   562.3413,   707.9458,   891.2509],
           [  562.3413,   707.9458,   891.2509,  1122.0185]])
    >>> np.array(ytools.freq_oct(3, [505, 900], trim='center'))
    array([[ 630.9573,  794.3282],
           [ 562.3413,  707.9458],
           [ 707.9458,  891.2509]])
    >>> np.array(ytools.freq_oct(3, [505, 900], exact=True))
    array([[  500.    ,   629.9605,   793.7005,  1000.    ],
           [  445.4494,   561.231 ,   707.1068,   890.8987],
           [  561.231 ,   707.1068,   890.8987,  1122.462 ]])
    >>> ytools.freq_oct(6, [.8, 2.6])[0]
    array([ 0.7943,  0.8913,  1.    ,  1.122 ,  1.2589,  1.4125,  1.5849,
            1.7783,  1.9953,  2.2387,  2.5119])
    >>> ytools.freq_oct(6, [.8, 2.6], anchor=2)[0]
    array([ 0.7962,  0.8934,  1.0024,  1.1247,  1.2619,  1.4159,  1.5887,
            1.7825,  2.    ,  2.244 ,  2.5179])
    >>> ytools.freq_oct(6, [.8, 2.6], exact=True)[0]
    array([ 0.7751,  0.87  ,  0.9766,  1.0962,  1.2304,  1.3811,  1.5502,
            1.74  ,  1.9531,  2.1923,  2.4608])
    >>> ytools.freq_oct(6, [.8, 2.6], exact=True, anchor=2)[0]
    array([ 0.7937,  0.8909,  1.    ,  1.1225,  1.2599,  1.4142,  1.5874,
            1.7818,  2.    ,  2.2449,  2.5198])
    """
    if frange[0] <= 0.:
        s = 1.
    else:
        s = frange[0]
    e = frange[-1]
    if exact:
        if not anchor:
            anchor = 1000.
        var1 = np.floor(np.log2(s/anchor)*n)
        var2 = np.log2(e/anchor)*n + 1
        bands = np.arange(var1, var2)
        F = anchor * 2**(bands/n)
        factor = 2**(1/(2*n))
    else:
        if not anchor:
            anchor = 1.
        var1 = np.floor(np.log10(s/anchor)*10*n/3)
        var2 = np.log10(e/anchor)*10*n/3 + 1
        bands = np.arange(var1, var2)
        F = anchor * 10**(3*bands/(10*n))
        factor = 10**(3/(20*n))
    FL, FU = F/factor, F*factor
    if trim == 'band':
        Nmax = np.max(np.nonzero(FL <= e)[0]) + 1
        Nmin = np.min(np.nonzero(FU >= s)[0])
    else:
        Nmax = np.max(np.nonzero(F <= e)[0]) + 1
        Nmin = np.min(np.nonzero(F >= s)[0])
    F = F[Nmin:Nmax]
    FL = FL[Nmin:Nmax]
    FU = FU[Nmin:Nmax]
    return F, FL, FU


def psd_rescale(P, F, n_oct=3, freq=None, extendends=True):
    """
    Convert PSD from one frequency scale to another.

    Parameters
    ----------
    P : array_like
        Vector or matrix; PSD(s) to convert. Works columnwise if
        matrix.
    F : array_like
        Vector; input frequency scale. If steps are not linear,
        logarithmic spacing is assumed.
    n_oct : scalar; optional
        Specifies the desired octave scale. 1 means full octave, 3
        means 1/3rd octave, 6 means 1/6th octave, etc. The routine
        :func:`freq_oct` is used to calculate the frequency vector
        with the default options. To change options, call
        :func:`freq_oct` directly and provide that input via `freq`.
    freq : array_like or None; optional
        Alternative to `n_oct` and takes precedence. Specifies
        desired output frequencies directly. If steps are not linear,
        logarithmic spacing is assumed for computing the band
        boundaries.
    extendends : bool; optional
        If True and the first and/or last frequency band extends
        beyond the original data, the area of the new band is adjusted
        up by the ratio of the new bandwidth over the portion that is
        covered by the original data. This will cause the
        corresponding PSD value to be higher than it would otherwise
        be, meaning the overall mean-square value will also be a
        little higher.

    Returns
    -------
    Pout : ndarray
        Vector or matrix; converted PSD(s). Rows correspond to the
        new frequency scale; columns correspond with `P`.
    Fctr : ndarray
        Vector; center frequencies of output. Equal to `freq` if that
        was input.
    msv : scalar or 1d ndarray
        Mean square value estimate for each PSD.
    ms : ndarray
        Vector or matrix; converted mean square values directly
        (instead of density). For constant frequency step df,
        ``ms = df*Pout``. Same size as `Pout`.

    Notes
    -----
    This algorithm works by interpolating on cummulative area such
    that original contributions to total mean-square per band is
    preserved.

    .. note::

        Note that if the area of the first and/or last band is
        extended (see `extendends` above), the overall mean-square
        value will be higher than the original.

    See :func:`freq_oct` for more information on how the octave scales
    are calculated.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> from pyyeti import ytools
        >>>
        >>> g = np.random.randn(10000)
        >>> sr = 400
        >>> f, p = signal.welch(g, sr, nperseg=sr)
        >>> p3, f3, msv3, ms3 = ytools.psd_rescale(p, f)
        >>> p6, f6, msv6, ms6 = ytools.psd_rescale(p, f, n_oct=6)
        >>> p12, f12, msv12, ms12 = ytools.psd_rescale(p, f, n_oct=12)
        >>>
        >>> fig = plt.figure('PSD compare')
        >>> line = plt.semilogx(f, p, label='Linear')
        >>> line = plt.semilogx(f3, p3, label='1/3 Octave')
        >>> line = plt.semilogx(f6, p6, label='1/6 Octave')
        >>> line = plt.semilogx(f12, p12, label='1/12 Octave')
        >>> _ = plt.legend(ncol=2, loc='best')
        >>> _ = plt.xlim([1, 200])
        >>> msv1 = np.sum(p*(f[1]-f[0]))
        >>> abs(msv1/msv3 - 1) < .12
        True
        >>> abs(msv1/msv6 - 1) < .06
        True
        >>> abs(msv1/msv12 - 1) < .03
        True

    Demonstrate ``extendends=False``. Ends will be less than input
    because there is a region of zero area averaged in:

    >>> in_freq = np.arange(0, 10.1, .25)
    >>> out_freq = np.arange(0, 10.1, 5)
    >>> in_p = np.ones_like(in_freq)
    >>> p, f, ms, mvs = ytools.psd_rescale(in_p, in_freq,
    ...                                    freq=out_freq)
    >>> p
    array([ 1.,  1.,  1.])
    >>> p, f, ms, mvs = ytools.psd_rescale(in_p, in_freq,
    ...                                    freq=out_freq,
    ...                                    extendends=False)
    >>> p
    array([ 0.525,  1.   ,  0.525])

    The 0.525 value is from: ``area/width = 1*(2.5-(-.125))/5 = .525``
    """
    F = np.atleast_1d(F)
    P = np.atleast_1d(P)

    def get_fl_fu(fcenter):
        Df = np.diff(fcenter)
        if np.all(Df == Df[0]):
            # linear scale
            Df = Df[0]
            FL = fcenter-Df/2
            FU = fcenter+Df/2
        else:
            # output is not linear, assume log
            mid = np.sqrt(fcenter[:-1] * fcenter[1:])
            fact1 = mid[0] / fcenter[1]
            fact2 = fcenter[-1] / mid[-1]
            FL = np.hstack((fact1*fcenter[0], mid))
            FU = np.hstack((mid, fact2*fcenter[-1]))
        return FL, FU

    if freq is None:
        Wctr, FL, FU = freq_oct(n_oct, F, exact=True)
    else:
        freq = np.atleast_1d(freq)
        FL, FU = get_fl_fu(freq)
        Nmax = np.max(np.nonzero(FL <= F[-1])[0]) + 1
        Nmin = np.min(np.nonzero(FU >= F[0])[0])
        Wctr = freq[Nmin:Nmax]
        FL = FL[Nmin:Nmax]
        FU = FU[Nmin:Nmax]

    oned = False
    if P.ndim == 1 or (P.ndim == 2 and P.shape[0] == 1):
        oned = True
        P = P.reshape(-1, 1)
        cols = 1
    else:
        cols = P.shape[1]

    # calculate cumulative area:
    Df = np.diff(F)
    if np.all(Df == Df[0]):
        # input uses linear frequency scale
        FLin = F - Df[0]/2
        FUin = F + Df[0]/2
    else:
        # not linear, assume log
        FLin, FUin = get_fl_fu(F)

    Df = (FUin - FLin).reshape(-1, 1)
    ca = np.vstack((np.zeros((1, cols)), np.cumsum(Df*P, axis=0)))
    Fa = np.hstack((FLin[0], FUin))

    if extendends:
        fl = FL[0]
        fu = FU[-1]
        if FL[0] < FLin[0]:
            FL[0] = FLin[0]
        if FU[-1] > FUin[-1]:
            FU[-1] = FUin[-1]

    cal = np.zeros((len(FL), cols))
    cau = np.zeros((len(FL), cols))
    for i in range(cols):
        # with np.interp, interpolating cumulative area beyond end
        # points will take the end value -- that's perfect here: 0's
        # on the front, total area on the back
        cal[:, i] = np.interp(FL, Fa, ca[:, i])
        cau[:, i] = np.interp(FU, Fa, ca[:, i])

    # Compute new values
    ms = cau - cal
    psdoct = ms*(1/(FU-FL).reshape(-1, 1))
    if extendends:
        FL[0] = fl
        FU[-1] = fu
        ms = psdoct * (FU-FL).reshape(-1, 1)
    msv = np.sum(ms, axis=0)
    if oned:
        psdoct = psdoct.flatten()
        ms = ms.flatten()
        msv = msv[0]
    return psdoct, Wctr, msv, ms


def splpsd(x, sr, n, overlap=.5, window='hanning',
           fs=3, pref=2.9e-9):
    r"""
    Sound pressure level estimation using PSD.

    Parameters
    ----------
    x : 1d array like
        Vector of pressure values.
    sr : scalar
        Sample rate.
    n : integer
        Window size for PSD.
    overlap : scalar
        Amount of overlap in windows, eg 0.5 would be 50% overlap.
    window : str or tuple or array like
        Passed to :func:`scipy.signal.welch`; see that routine for
        more information.
    fs : integer
        Specifies output frequency scale. Zero means linear scale,
        anything else is passed to :func:`psd_rescale`. Example:

        ===  ======================================================
          0  linear scale as computed by :func:`scipy.signal.welch`
          1  full octave scale
          3  3rd octave scale
          6  6th octave scale
        ===  ======================================================

    pref : scalar
        Reference pressure. 2.9e-9 psi is considered to be the
        threshhold for human hearing. In Pascals, that value is 2e-5.

    Returns
    -------
    f : ndarray
        The output frequency vector (Hz).
    spl : ndarray
        The sound pressure level vector in dB.
    oaspl : scalar
        The overall sound pressure level.

    Notes
    -----
    This routine calls :func:`scipy.signal.welch` to calculate the
    PSD. It then converts that to mean-square values per band (for
    linear frequency steps, this is just PSD * delta_freq). Loosely,
    the math is:

    .. math::
        \begin{aligned}
        V &= \frac{PSD \cdot \Delta f}{P_{ref}^2} \\
        SPL &= 10 \; \log_{10} ( V ) \\
        OASPL &= 10 \; \log_{10} \left ( \sum V \right )
        \end{aligned}

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import ytools
    >>> x = np.random.randn(100000)
    >>> sr = 4000
    >>> f, spl, oaspl = ytools.splpsd(x, sr, sr)
    >>> # oaspl should be around 170.75 (since variance = 1):
    >>> shouldbe = 10*np.log10(1/(2.9e-9)**2)
    >>> abs(oaspl/shouldbe - 1) < .01
    True
    """
    # compute psd
    F, P = signal.welch(x, sr, window=window, nperseg=n,
                        noverlap=int(overlap*n))
    if fs != 0:
        _, F, _, P = psd_rescale(P, F, n_oct=fs)
    else:
        P = P*F[1]
    v = P/pref**2
    return F, 10*np.log10(v), 10*np.log10(np.sum(v))


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

        >>> from pyyeti import ytools
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
        ...     res[pts], up2 = ytools.resample(data, p, q, pts=pts, t=x)
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
        >>> dndata, dnx = ytools.resample(data, p, q, t=x)
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
    gf = gcd(p, q)
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
        RData = RData.flatten()
    if t is None:
        if getfir:
            return RData, fir
        return RData
    tnew = np.arange(n) * (t[1]-t[0]) * ln/n + t[0]
    if getfir:
        return RData, tnew, fir
    return RData, tnew


def fixtime(olddata, sr=None, negmethod='sort', deldrops=True,
            dropval=-1.40130E-45, deloutliers=True, base=None,
            fixdrift=False, getall=False):
    """
    Process recorded data to make an even time vector.

    Parameters
    ----------
    olddata : 2d ndarray or 2-element tuple
        If ndarray, is must have 2 columns: ``[time, signal]``.
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
    deloutliers : bool; optional
        If True, outlier times are deleted from the data; otherwise,
        they are left in.
    base : scalar or None; optional
        Scalar value that new time vector would hit exactly if within
        range. If None, new time vector is aligned to longest section
        of "good" data.
    fixdrift : bool; optional
        If True, shift data
    getall : bool; optional
        If True, return `dropouts` and `sr_stats`; otherwise only
        `newdata` is returned.

    Returns
    -------
    newdata : 2d ndarray or tuple
        Cleaned up version of `olddata`. Will be 2d ndarray if
        `olddata` was ndarray; otherwise it is a tuple:
        ``(time, data)``.
    dropouts : 1d ndarray; optional
        If `deldrops` is True (the default), this is a True/False
        vector into `olddata` where drop-outs occurred. Otherwise,
        it is a True/False vector into `newdata`.
    sr_stats : 1d ndarray or None; optional
        Five-element vector with the sample rate statistics; useful to
        help user select best sample rate or to compare against `sr`.
        The five elements are::

            [max_sr, min_sr, mean_sr, max_count_sr, max_count_percent]

        The `max_count_sr` is the sample rate that occurred most
        often. This is usually the 'correct' sample rate.
        `max_count_percent` gives the percent occurrence of
        `max_count_sr`.

    tp : 1d ndarray or None; optional
        Contains indices into old time vector of where time-step shifts
        ("turning points") were done to align the new time vector
        against the old.

    Notes
    -----
    This algorithm works as follows:

       1.  Find and delete drop-outs if `deldrops` is True.

       2.  Delete outlier times if `deloutliers` is True. These are
           points with times that are more than 3 standard deviations
           away from the mean. A warning message is printed if any
           such times are found. Note that on a perfect time vector,
           the end points are at 1.73 sigma (eg, m+1.73s =
           .5+1.73*.2887 = 1).

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

       6.  Count number of small time-steps defined as those that are
           less than ``0.93/sr``. If more than 1% of the steps are
           small, print a warning.

       7.  Count number of large time-steps defines as those that are
           greater than ``1.07/sr``. If more than 1% of the steps are
           large, print a warning.

       8.  Make a new, evenly spaced time vector according to the new
           sample rate that spans the range of time in `olddata`.

       9.  Find the "turning points" in the old time vector. These are
           where the step differs by more than 1/4 step from the
           ideal. If `fixdrift` is True, each segment is further
           divided if needed to re-align due to drift (when the
           sample rate is slightly off from the ideal). If there are
           too many turning points (more than 50% of total points),
           this routine basically gives up and skips steps 10 and 11.

       10. Shift the new time vector to align with the longest section
           of "good" old time steps.

       11. Loop over the segments defined by the turning points. Each
           segment will shifted left or right to fit with the new time
           vector. The longest section is not shifted due to step 10.

       12. If `base` is not None, the new time vector is shifted by up
           to a half time step such that it would hit `base` exactly
           (if it was in range).

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
                getall, return_ndarray):
        if return_ndarray:
            newdata = np.vstack((t, data)).T
        else:
            newdata = (t, data)
        if getall:
            if dropouts is None:
                dropouts = _find_drops(data, dropval)
            return newdata, dropouts, sr_stats, tp
        return newdata

    def _del_outtimes(told, olddata, deloutliers):
        mn = told.mean()
        sig = 3*told.std()
        pv = np.logical_or(told < mn-sig, told > mn+sig)
        if np.any(pv):
            if deloutliers:
                warn('there are {:d} outlier times being deleted.'
                     ' These are times more than 3-sigma away '
                     'from the mean. {:s}'.format(pv.sum(), POOR),
                     RuntimeWarning)
                told = told[~pv]
                olddata = olddata[~pv]
            else:
                warn('there are {:d} outlier times that are NOT '
                     'being deleted because `leaveoutliers` is '
                     'True. These are times more than 3-sigma '
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
            dsr = np.round(10*max(sr1, .1))/10
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
        elif not sr:
            ssr = input('==> Enter desired sample rate [{:g}]: '
                        .format(defsr))
            if not ssr:
                sr = defsr
            else:
                sr = float(ssr)
        print('==> Using sample rate = {:g}'.format(sr))
        return sr, sr_stats

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

    def _get_prev_index(vec, val):
        p = np.searchsorted(vec, val) - 1
        if p < 0:
            return 0
        return p

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
        tp = np.nonzero(get_turning_pts(told, atol=dt/4))[0]
        msg = ('there are too many turning points ({:.2f}%) to '
               'account for drift. Trying to find only "big" '
               'turning points. Skipping step 10.',
               'there are still too many turning points ({:.2f}%) '
               'to attempt any alignment. Skipping steps 10 and 11.')
        align = True
        for i in range(2):
            if len(tp)-2 > len(told) // 2:  # -2 to ignore ends
                align = False
                p = (len(tp)-2)/len(told)*100
                warn(msg[i].format(p), RuntimeWarning)
                if i == 1:
                    tp = tp[[0, -1]]
                else:
                    meandt = difft[difft > 0].mean()
                    a = 1.1*meandt
                    tp = np.hstack((True, (difft > a)[:-1], True))
                    tp = np.nonzero(tp)[0]
            else:
                break
        return tp, align

    def _mk_initial_tnew(told, sr, dt, difft, fixdrift):
        L = int(np.round((told[-1] - told[0])*sr)) + 1
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

    # main routine
    if isinstance(olddata, np.ndarray):
        told = olddata[:, 0]
        olddata = olddata[:, 1]
        return_ndarray = True
    else:
        if len(olddata) != 2:
            raise ValueError('incorrectly defined `olddata`')
        told, olddata = np.atleast_1d(*olddata)
        if len(told) != len(olddata):
            raise ValueError('time and data vectors are incompatibly '
                             'sized')
        return_ndarray = False

    # check for drop outs:
    dropouts = sr_stats = tp = None
    if deldrops:
        told, olddata, dropouts = _del_drops(told, olddata, dropval)
        if len(told) == 0:
            return _return(told, olddata, dropouts, sr_stats, tp,
                           dropval, getall, return_ndarray)

    # check for outlier times ... outside 3-sigma
    told, olddata = _del_outtimes(told, olddata, deloutliers)

    # check for negative steps:
    told, olddata, difft = _chk_negsteps(told, olddata, negmethod)

    # sample rate calculations:
    sr, sr_stats = _sr_calcs(difft, sr)
    dt = 1/sr

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
        t1 = base - t0 - np.round((base-t0)*sr)/sr
        tnew += t1

    newdata = olddata[index]
    return _return(tnew, newdata, dropouts, sr_stats, tp,
                   dropval, getall, return_ndarray)


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
    >>> from pyyeti import ytools
    >>> import numpy as np
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> ytools.windowends(np.ones(8), 4)
    array([ 0.  ,  0.25,  0.75,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ])
    >>> ytools.windowends(np.ones(8), .7, ends='back')
    array([ 1.  ,  1.  ,  1.  ,  1.  ,  0.85,  0.5 ,  0.15,  0.  ])
    >>> ytools.windowends(np.ones(8), .5, ends='both')
    array([ 0.  ,  0.25,  0.75,  1.  ,  1.  ,  0.75,  0.25,  0.  ])

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.figure(figsize=[8, 3])
        >>> sig = np.ones(100)
        >>> wesig = ytools.windowends(sig, 5, ends='both')
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
        >>> _ = plt.figure('gensweep')
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
    rate = rate/60.
    dt = 1./fstop/ppc
    tstop = (np.log(fstop) - np.log(fstart))/np.log(2.)/rate
    t = np.arange(0., tstop+dt/2, dt)
    f = fstart * 2 ** (t*rate)
    sig = np.sin(2*np.pi/np.log(2)/rate*(f-fstart))
    return sig, t, f


def psdinterp(spec, freq, linear=False):
    """
    Interpolate values on a PSD specification (or analysis curve).

    Parameters
    ----------
    spec : 2d array
        Matrix containing the PSD specification(s) of the base
        excitation. Columns are: [ Freq PSD1 PSD2 ... PSDn ]. The
        frequency vector must be monotonically increasing.
    freq : 1d array
        Vector of frequencies to interpolate the specification to.
    linear : bool
        If True, use linear interpolation to compute the values;
        otherwise, the interpolation is done using the logs. Using logs
        is appropriate if the `spec` is actually a specification that
        uses constant db/octave slopes. Use ``linear=True`` for
        analysis curves (when not assuming constant db/octave slopes).

    Returns
    -------
    spec2 : 2d array
        Matrix of the interpolated PSD values. Has one fewer columns
        than `spec` because the frequency column is not included.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import ytools
    >>> spec = [[20, .0053],
    ...        [150, .04],
    ...        [600, .04],
    ...        [2000, .0036]]
    >>> freq = [100, 200, 600, 1200]
    >>> np.set_printoptions(precision=3)
    >>> ytools.psdinterp(spec, freq).flatten()
    array([ 0.027,  0.04 ,  0.04 ,  0.01 ])
    >>> ytools.psdinterp(spec, freq, linear=True).flatten()
    array([ 0.027,  0.04 ,  0.04 ,  0.024])
    """
    spec = np.atleast_2d(spec)
    freq = np.atleast_1d(freq)
    if linear:
        ifunc = interp.interp1d(spec[:, 0], spec[:, 1:], axis=0,
                                bounds_error=False, fill_value=0,
                                assume_sorted=True)
        psdfull = ifunc(freq)
    else:
        sp = np.log(spec)
        ifunc = interp.interp1d(sp[:, 0], sp[:, 1:], axis=0,
                                bounds_error=False, fill_value=0,
                                assume_sorted=True)
        psdfull = np.exp(ifunc(np.log(freq)))
    return psdfull


def psd2time(fp, ppc, fstart, fstop, df, winends=None, gettime=False):
    """
    Generate a 'random' time domain signal given a PSD specification.

    Parameters
    ----------
    fp : 2d array_like
        Two column matrix of PSD specification:  [freq, spec]. The
        frequency is in Hz and the specification is in units^2/Hz.
    ppc : scalar
        Points per cycle at highest (`fstop`) frequency; if < 2, it is
        internally reset to 2.
    fstart : scalar
        Starting frequency in Hz
    fstop : scalar
        Stopping frequency in Hz
    df : scalar
        Frequency step to be represented in the time signal. If
        routine gives poor results, try refining `df`. If `df` is
        greater than `fstart`, it is reset internally to `fstart`.
    winends : None or dictionary; optional
        If None, :func:`windowends` is not called. Otherwise,
        `winends` must be a dictionary of arguments that will be
        passed to :func:`windowends` (not including `signal`).
    gettime : bool; optional
        If True, a time vector is output.

    Returns
    -------
    sig : 1d ndarray
        The time domain signal with properties set to match input PSD
        spectrum.
    sr : scalar
        The sample rate of the signal.
    time : 1d ndarray; optional
        Time vector for the signal starting at zero with step of
        ``1/sr``: ``time = np.arange(len(sig))/sr``

    Notes
    -----
    This routine uses :func:`psdinterp` to expand `fp` to the desired
    frequencies. That routine assumes a constant db/octave slope for
    all segments and values outside of `fp` frequency range are set to
    zero.

    See also
    --------
    :func:`psdinterp`, :func:`windowends`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ytools
        >>> spec = [[20,  .0768],
        ...         [50,  .48],
        ...         [100, .48]]
        >>> sig, sr = ytools.psd2time(spec, ppc=10, fstart=35,
        ...                           fstop=70, df=.01,
        ...                           winends=dict(portion=.01))
        >>> sr  # the sample rate should be 70*10 = 700
        700.0

        Compare the resulting psd to the spec from 37 to 68:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> pv = np.logical_and(f >= 37, f <= 68)
        >>> fi = f[pv]
        >>> psdi = p[pv]
        >>> speci = ytools.psdinterp(spec, fi).flatten()
        >>> abs(speci - psdi).max() < .05
        True
        >>> abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .25
        True
        >>> fig = plt.figure('psd2time demo')
        >>> a = plt.subplot(211)
        >>> line = plt.plot(np.arange(len(sig))/sr, sig)
        >>> plt.grid()
        >>> a = plt.subplot(212)
        >>> spec = np.array(spec)
        >>> line = plt.loglog(spec[:, 0], spec[:, 1], label='spec')
        >>> line = plt.loglog(f, p, label='PSD of time signal')
        >>> leg = plt.legend(loc='best')
        >>> x = plt.xlim(20, 100)
        >>> y = plt.ylim(.05, 1)
        >>> plt.grid()
        >>> xticks = np.arange(20, 105, 10)
        >>> x = plt.xticks(xticks, xticks)
        >>> yticks = (.05, .1, .2, .5, 1)
        >>> y = plt.yticks(yticks, yticks)
        >>> v = plt.axvline(35, color='black', linestyle='--')
        >>> v = plt.axvline(70, color='black', linestyle='--')
    """
    if df > fstart:
        df = fstart
    if ppc < 2:
        ppc = 2
    # compute parameters
    # 1 cycle of lowest frequency defines length of signal:
    T = 1/df  # number of seconds for lowest frequency cycle
    N = int(np.ceil(fstop*ppc*T))  # total number of pts
    df = fstop*ppc/N
    # adjust df to make sure fstart is an even multiple of df
    df = fstart/np.floor(fstart/df)
    sr = N*df  # sr = N/T = N/(1/df)
    odd = N & 1

    # define constants
    freq = np.arange(fstart, fstop+df/2, df)

    # generate amp(f) vector
    speclevel = psdinterp(fp, freq).flatten()
    amp = np.sqrt(2*speclevel*df)

    m = (N+1)//2 if odd else N//2 + 1

    # build up amp to include zero frequency to fstart and from fstop
    # to fhighest:
    ntop = int(np.floor((fstart-df/2)/df) + 1)
    nbot = m - ntop - len(amp)
    if nbot > 0:
        amp = np.hstack((np.zeros(ntop), amp, np.zeros(nbot)))
    else:
        amp = np.hstack((np.zeros(ntop), amp))

    # generate F(t)
    phi = np.random.rand(m)*np.pi*2  # random phase angle

    # force these terms to be pure cosine
    phi[0] = 0.
    if not odd:
        phi[m-1] = 0

    # coefficients:
    a = amp*np.cos(phi)
    b = -amp*np.sin(phi)

    # form matrix ready for ifft:
    if odd:
        a2 = a[1:m]/2
        b2 = b[1:m]/2
        r = N*np.hstack((a[0], a2, a2[::-1]))  # real part
        i = N*np.hstack((0, b2, -b2[::-1]))    # imag part
    else:
        a2 = a[1:m-1]/2
        b2 = b[1:m-1]/2
        r = N*np.hstack((a[0], a2, a[m-1], a2[::-1]))  # real part
        i = N*np.hstack((0, b2, 0, -b2[::-1]))         # imag part

    F_time = np.fft.ifft(r+1j*i)
    mxi = abs(F_time.imag).max()
    mxr = abs(F_time.real).max()
    if mxi > 1e-7*mxr:
        # bug in the code if this ever happens
        warn('method failed accuracy test; (max imag)/'
             '(max real) = {}'.format(mxi/mxr),
             RuntimeWarning)

    F_time = F_time.real
    if winends is not None:
        F_time = windowends(F_time, **winends)
    if gettime:
        return F_time, sr, np.arange(N)/sr
    return F_time, sr


def waterfall(timeslice, tsoverlap, sig, sr, func, which, freq,
              t0=0.0, args=None, kwargs=None, slicefunc=None,
              sliceargs=None, slicekwargs=None):
    """
    Compute a 'waterfall' map over time and frequency (typically) using
    user-supplied function.

    Parameters
    ----------
    timeslice : scalar
        The length in seconds of each time slice.
    tsoverlap : scalar
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
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
        If provided, these are passed to `slicefunc`. Must be `()` if
        `slicefunc` is None.
    slicekwargs : dict; optional
        If provided, these are passed to `slicefunc`. Must be `{}` if
        `slicefunc` is None.

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
        >>> from pyyeti import srs
        >>> from pyyeti import ytools
        >>> from matplotlib import cm
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> frq = np.arange(1., 50.1)
        >>> Q = 20
        >>> mp, t, f = ytools.waterfall(2, .5, sig, sr, srs.srs,
        ...                             which=None, freq=frq,
        ...                             args=(sr, frq, Q),
        ...                             kwargs=dict(eqsine=1),
        ...                             slicefunc=ytools.windowends,
        ...                             sliceargs=[.02],
        ...                             slicekwargs=dict(ends='front'))
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
        sig = sig.flatten()
    if tsoverlap >= 1:
        raise ValueError('`tsoverlap` must be less than 1')

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
    inc = int(ntimeslice * tsoverlap)
    tlen = (sig.size-inc) // (ntimeslice-inc)
    b = 0

    # make time vector:
    t0_ = timeslice/2
    tf = t0_ + (tlen-1)*(timeslice-timeslice*tsoverlap)
    t = np.linspace(t0_, tf, tlen)

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


def psdmod(timeslice, tsoverlap, sig, sr, nperseg=None, getmap=False,
           **kwargs):
    """Modified method for PSD estimation via FFT.

    Parameters
    ----------
    timeslice : scalar
        The length in seconds of each time slice.
    tsoverlap : scalar
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
    nperseg : int, optional
        Length of each segment for the FFT. Defaults to `sr` for 1 Hz
        frequency step in PSD. Note: frequency step in Hz = sr/nperseg.
    getmap : bool, optional
        If True, get the PSD map output (the `Pmap` and `t` variables
        described below).
    *kwargs : optional
        Named arguments to pass to :func:`scipy.signal.welch`.

    Returns
    -------
    f : 1d ndarray
        Array of sample frequencies.
    Pxx : 1d ndarray
        Power spectral density or power spectrum of `sig`.
    Pmap : 2d ndarray; optional
        The PSD map; each column is an output of
        :func:`scipy.signal.welch`. Rows correspond to frequency `f`
        and columns correspond to time `t`. Only output if `getmap` is
        True.
    t : 1d ndarray; optional
        The time vector for the columns of `Pmap`. Only output if
        `getmap` is True.

    Notes
    -----
    This routine calls :func:`waterfall` for handling the timeslices
    and preparing the output and :func:`scipy.signal.welch` to process
    each time slice. So, the "modified" method is to use the PSD
    averaging (via welch) for each time slice but then take the peaks
    over all these averages.

    For a pure 'maximax' PSD, just set `timeslice` to ``nperseg/sr``
    and `tsoverlap` to 0.5 (assuming 50% overlap is desired).
    Conversely, for a pure Welch periodogram, just set the `timeslice`
    equal to the entire signal (or just use :func:`scipy.signal.welch`
    of course). Usually the desired behavior for :func:`psdmod` is
    somewhere between these two extremes.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import ytools
        >>> from scipy import signal
        >>> TF = 30  # make a 30 second signal
        >>> spec = [[20, 1], [50, 1]]
        >>> sig, sr, t = ytools.psd2time(spec, ppc=10, fstart=20,
        ...                              fstop=50, df=1/TF,
        ...                              winends=dict(portion=10),
        ...                              gettime=True)
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> f2, p2 = ytools.psdmod(4, .5, sig, sr, nperseg=sr)
        >>> f3, p3 = ytools.psdmod(sr/sr, .5, sig, sr, nperseg=sr)
        >>> spec = np.array(spec).T
        >>> fig = plt.figure('psdmod')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(t, sig)
        >>> _ = plt.title(r'Input Signal - Specification Level = '
        ...               '1.0 $g^{2}$/Hz')
        >>> _ = plt.xlabel('Time (sec)')
        >>> _ = plt.ylabel('Acceleration (g)')
        >>> _ = plt.subplot(212)
        >>> _ = plt.plot(*spec, 'k-', lw=1.5, label='Spec')
        >>> _ = plt.plot(f, p, label='Welch PSD')
        >>> _ = plt.plot(f2, p2, label='PSDmod')
        >>> _ = plt.plot(f3, p3, label='Maximax')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.xlim(20, 50)
        >>> _ = plt.title('PSD')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.ylabel(r'PSD ($g^2$/Hz)')
        >>> _ = plt.tight_layout()
    """
    if nperseg is None:
        nperseg = sr
    welch_inputs = dict(fs=sr, nperseg=nperseg, **kwargs)
    pmap, t, f = waterfall(timeslice, tsoverlap, sig, sr,
                           signal.welch, which=1, freq=0,
                           kwargs=welch_inputs)
    p = pmap.max(axis=1)
    if getmap:
        return f, p, pmap, t
    return f, p


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
    >>> from pyyeti import ytools
    >>> ytools.get_turning_pts([1, 2, 3, 3, 3])
    array([ True, False,  True, False,  True], dtype=bool)
    >>> y, x = ytools.get_turning_pts([1, 2, 3, 3, 3],
    ...                               [1, 2, 3, 4, 5],
    ...                               getindex=False)
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
        >>> from pyyeti import ytools
        >>> x = np.arange(1.0, 31.0, 1.0)
        >>> y = np.cos(x)
        >>> fig = plt.figure('calcenv', figsize=[10, 6])
        >>>
        >>> ax = plt.subplot(311)
        >>> env = ytools.calcenv(x, y, base=None, makeplot='add')
        >>> _ = plt.title('base=None (method="both")')
        >>> _ = ax.legend(handles=env[-1], loc='upper left',
        ...               bbox_to_anchor=(1.02, 1.),
        ...               borderaxespad=0.)
        >>> _ = ax.set_xticklabels([])
        >>>
        >>> ax = plt.subplot(312)
        >>> env = ytools.calcenv(x, y, method='max', makeplot='add')
        >>> _ = plt.title('method="max"')
        >>> ax.legend().set_visible(False)
        >>> _ = ax.set_xticklabels([])
        >>>
        >>> ax = plt.subplot(313)
        >>> env = ytools.calcenv(x, y, method='min', makeplot='add')
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


def rdfile(f, rdfunc, *args, **kwargs):
    r"""
    Interface routine for other routines that read from a file.

    Parameters
    ----------
    f : string or file_like
        Either a name of a file or a file handle as returned by
        :func:`open`.
    rdfunc : function
        Function that reads data from file; first argument is the
        input file handle.
    *args, **kwargs : arguments
        Arguments to pass to :func:`rdfunc`.

    Returns
    -------
    res : any
        Returns the output of :func:`rdfunc`

    See also
    --------
    :func:`wtfile`

    Examples
    --------
    >>> from pyyeti.ytools import wtfile, rdfile
    >>> from io import StringIO
    >>> def dowrite(f, string, number):
    ...     f.write('{:s} = {:.3f}\n'.format(string, number))
    >>> def doread(f):
    ...     return f.readline()
    >>> with StringIO() as f:
    ...     wtfile(f, dowrite, 'param', number=45.3)
    ...     _ = f.seek(0, 0)
    ...     s = rdfile(f, doread)
    >>> s
    'param = 45.300\n'
    """
    if isinstance(f, str):
        with open(f, 'r') as fin:
            return rdfunc(fin, *args, **kwargs)
    else:
        return rdfunc(f, *args, **kwargs)


def wtfile(f, wtfunc, *args, **kwargs):
    r"""
    Interface routine for other routines that write to a file.

    Parameters
    ----------
    f : string or file handle or 1
        Either a name of a file or a file handle as returned by
        :func:`open` or :func:`StringIO`. Input as integer 1 to write
        to stdout.
    wtfunc : function
        Function that writes output; first argument is the output
        file handle.
    *args, **kwargs : arguments
        Arguments to pass to :func:`wtfunc`.

    Returns
    -------
    res : any
        Returns the output of :func:`wtfunc`

    See also
    --------
    :func:`rdfile`

    Examples
    --------
    >>> from pyyeti.ytools import wtfile
    >>> def dowrite(f, string, number):
    ...     f.write('{:s} = {:.3f}\n'.format(string, number))
    >>> wtfile(1, dowrite, 'param', number=45.3)
    param = 45.300
    """
    if isinstance(f, str):
        with open(f, 'w') as fout:
            return wtfunc(fout, *args, **kwargs)
    else:
        if f == 1:
            f = sys.stdout
        return wtfunc(f, *args, **kwargs)
