# -*- coding: utf-8 -*-
"""
Some math and I/O tools. Most are translated from Yeti to Python.
"""

import pickle
import gzip
import bz2
import sys
import numpy as np
import scipy.linalg as linalg
from pyyeti import guitools


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
        data = data[np.isfinite(data)]
        mn = data.min()
        mx = data.max()
        a = int(np.floor(mn/binsize))
        b = int(np.ceil(mx/binsize))
        for i in range(a, b+1):
            lft = (i-1/2)*binsize
            rgt = (i+1/2)*binsize
            count = np.count_nonzero((lft <= data) &
                                     (data < rgt))
            if count > 0:
                yield [i*binsize, count]

    bins = []
    for b in _get_next_bin(data, binsize):
        bins.append(b)
    histo = np.empty((len(bins), 3))
    histo[:, :2] = bins
    histo[:, 2] = 100*histo[:, 1]/histo[:, 1].sum()
    return histo


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
    return max_off < tol*max_on


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
            return np.allclose(1, d)
        else:
            return False

    raise ValueError('invalid `mtype`')


def diagmat(*args):
    """
    Assemble matrices on the diagonal

    Parameters
    ----------
    *args : 2d array_like variables
        Any number of matrices (2d arrays) to assemble on the diagonal
        of new, larger matrix. Any ``None`` entries are quietly
        skipped.

    Returns
    -------
    mat : 2d ndarray
        The assembled matrix.

    Examples
    --------
    >>> from pyyeti import ytools
    >>> a = [[1, 1], [2, 2]]
    >>> b = 40
    >>> c = None
    >>> d = [10, 20, 30]
    >>> e = [[10], [11]]
    >>> f = 100
    >>> ytools.diagmat(a, b, c, d, e, f)
    array([[  1,   1,   0,   0,   0,   0,   0,   0],
           [  2,   2,   0,   0,   0,   0,   0,   0],
           [  0,   0,  40,   0,   0,   0,   0,   0],
           [  0,   0,   0,  10,  20,  30,   0,   0],
           [  0,   0,   0,   0,   0,   0,  10,   0],
           [  0,   0,   0,   0,   0,   0,  11,   0],
           [  0,   0,   0,   0,   0,   0,   0, 100]])

    >>> ytools.diagmat(None, None)
    array([], shape=(0, 0), dtype=float64)
    """
    R = C = 0
    dtype = None
    args2d = []
    for arg in args:
        if arg is not None:
            arg = np.atleast_2d(arg)
            args2d.append(arg)
            r, c = arg.shape
            R += r
            C += c
            if dtype is None:
                dtype = arg.dtype
            else:
                dtype = np.promote_types(dtype, arg.dtype)

    if dtype is None:
        return np.zeros((R, C))

    M = np.zeros((R, C), dtype)
    R = C = 0
    for arg in args2d:
        r, c = arg.shape
        M[R:R+r, C:C+c] = arg
        R += r
        C += c
    return M


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
        if not mtp & posdef:
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


def rdfile(f, rdfunc, *args, **kwargs):
    r"""
    Interface routine for other routines that read from a file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Can also be the name of a directory or None; in
        these cases, a GUI is opened for file selection.
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
    f = guitools.get_file_name(f, read=True)
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
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open` or :func:`StringIO`. Input as integer 1 to write
        to stdout. Can also be the name of a directory or None; in
        these cases, a GUI is opened for file selection.
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
    f = guitools.get_file_name(f, read=False)
    if isinstance(f, str):
        with open(f, 'w') as fout:
            return wtfunc(fout, *args, **kwargs)
    else:
        if f == 1:
            f = sys.stdout
        return wtfunc(f, *args, **kwargs)


def _get_fopen(name, read=True):
    """Utility for save/load"""
    name = guitools.get_file_name(name, read)
    if name.endswith('.pgz'):
        fopen = gzip.open
    elif name.endswith('.pbz2'):
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
    """
    name, fopen = _get_fopen(name, read=False)
    with fopen(name, 'wb') as f:
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
    """
    name, fopen = _get_fopen(name, read=True)
    with fopen(name, 'rb') as f:
        return pickle.load(f)
