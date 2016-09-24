# -*- coding: utf-8 -*-
"""
Time and frequency domain ODE solvers for matrix equations. Adapted
and enhanced from the Yeti versions (which were adapted and enhanced
from the original CAM versions). Note that some features depend on the
equations being in modal space (particularly important where there are
distinctions between the rigid-body modes and the elastic modes).
"""

from types import SimpleNamespace
import warnings
import scipy.linalg as la
import numpy as np
from pyyeti import expmint, ytools


def get_su_coef(m, b, k, h, rbmodes=None, rfmodes=None):
    """
    Get uncoupled equations of motion integration coefficients.

    Parameters
    ----------
    m : 1d ndarray or None
        Mass; vector (of diagonal), or None (identity assumed)
    b : 1d ndarray
        Damping; vector (of diagonal)
    k : 1d ndarray
        Stiffness; vector (of diagonal)
    h : scalar or None
        Time step; if None, this return returns None.
    rbmodes : 1d ndarray or None; optional
        Index vector for the rigid-body modes; if None, a mode is
        assumed rigid-body if the `k` value is < .005.
    rfmodes : 1d ndarray or None; optional
        Index vector for the residual flexibility modes; if None,
        there are no residual flexibility modes.

    Returns
    -------
    coefs : SimpleNamespace with the members:
        ``F, G, A, B, Fp, Gp, Ap, Bp, pvrb``.

    Notes
    -----
    All entries in `coefs` are 1d ndarrays. Except for `pvrb`, the
    outputs are the integration coefficients from the algorithm in the
    Nastran Theoretical Manual (sec 11.4). It can handle rigid-body,
    under-damped, critically-damped, and over-damped equations. The
    solver is exact with the assumption that the forces vary linearly
    during a time step (1st order hold). `pvrb` is a boolean vector
    with True specifying where rigid-body modes are).

    The coefficients are used as follows::

        for j in range(1, nt):
            d[:, j+1] = (F * d[:, j] + G * v[:, j] +
                           A * P[:, j] + B * P[:, j+1])
            v[:, j+1] = (Fp * d[:, j] + Gp * v[:, j] +
                           Ap * P[:, j] + Bp * P[:, j+1])

    where `d` is the displacement, `v` is the velocity, and `P` is the
    applied force.

    See also
    --------
    :class:`SolveUnc`
    """
    if h is None:
        return None
    n = len(b)
    if m is None:
        wo2 = k
        C = b/2
    else:
        wo2 = k/m
        C = (b/m)/2
    w2 = wo2 - C**2

    if rbmodes is None:
        pvrb = (wo2 < .005).astype(int)
    else:
        pvrb = np.zeros(n, int)
        pvrb[rbmodes] = 1
    pvel = 1 - pvrb

    # make a zeros/ones rfmodes
    rfmodes2 = np.zeros(n, int)
    if rfmodes is not None:
        pvel[rfmodes] = 0
        rfmodes2[rfmodes] = 1
    pvel = pvel.astype(bool)

    # setup partition vectors for underdamped, critically damped,
    # and overdamped equations
    badrows = None
    if np.any(pvel):
        pvover = np.zeros(n, bool)
        pvcrit = np.zeros(n, bool)
        pvundr = np.zeros(n, bool)
        rat = w2[pvel]/wo2[pvel]
        pvundr[pvel] = rat >= 1.e-8
        pvcrit[pvel] = abs(rat) < 1.e-8
        pvover[pvel] = rat <= -1e-8

        if not np.all(rfmodes2 + pvrb + pvundr +
                      pvover + pvcrit == 1):
            badrows = np.nonzero(rfmodes2 + pvrb + pvundr +
                                 pvover + pvcrit != 1)[0]
    elif not np.all(rfmodes2 + pvrb == 1):
        badrows = np.nonzero(rfmodes2 + pvrb != 1)[0]

    if badrows is not None:
        msg = ('Partitioning problem. Check '
               'settings for mode number(s):')
        print(msg)
        print('badrows = ', badrows)
        raise ValueError('Partitioning problem. See above message.')

    w2 = abs(w2)
    # define the appropriate parameters based on damping
    # ... grab memory and at the same time set the rb equations
    F = pvrb.astype(float)
    G = h*F
    if m is None:
        A = (h*h/3)*F
        Ap = (h/2)*F
    else:
        A = (h*h/3)*F/m
        Ap = (h/2)*F/m
    B = A/2
    Fp = np.zeros(n, float)
    Gp = F.copy()
    Bp = Ap.copy()

    if np.any(pvel):
        if np.any(pvundr):
            w = np.sqrt(w2[pvundr])
            cs = np.cos(w*h)
            sn = np.sin(w*h)
            beta = C[pvundr]
            ex = np.exp(-beta*h)
            _wo2 = wo2[pvundr]
            _w2 = w2[pvundr]
            _k = k[pvundr]

            # for displacement:
            F[pvundr] = ex*(cs + (beta/w)*sn)
            G[pvundr] = (ex*sn)/w
            t0 = 1 / (h*_k*w)
            t1 = (_w2 - beta*beta)/_wo2
            t2 = (2*w*beta)/_wo2
            A[pvundr] = t0 * (ex * ((t1 - h*beta)*sn -
                                    (t2 + h*w)*cs) + t2)
            B[pvundr] = t0 * (ex * (-t1*sn + t2*cs) + w*h - t2)

            # for velocity:
            Fp[pvundr] = -(_wo2/w) * ex * sn
            Gp[pvundr] = ex * (cs - (beta/w) * sn)
            Ap[pvundr] = t0 * (ex * ((beta + h*_wo2)*sn + w*cs)-w)
            Bp[pvundr] = t0 * (-ex * (beta*sn + w*cs) + w)

        if np.any(pvcrit):
            beta = C[pvcrit]
            ex = np.exp(-beta*h)
            _wo2 = wo2[pvcrit]
            _k = k[pvcrit]

            # for displacement:
            hbeta = h*beta
            F[pvcrit] = ex*(1 + hbeta)
            G[pvcrit] = h*ex
            t0 = 1 / (h*_k)
            A[pvcrit] = t0*(2 / beta - (1 / beta)*ex *
                            (2 + 2*hbeta + (hbeta*hbeta)))
            B[pvcrit] = (t0/beta) * (hbeta - 2 + ex*(2+hbeta))

            # for velocity:
            Fp[pvcrit] = -(beta*beta)*(h*ex)
            Gp[pvcrit] = ex*(1 - hbeta)
            Ap[pvcrit] = t0 * (ex * (1 + hbeta + (hbeta*hbeta))-1)
            Bp[pvcrit] = t0 * (1 - ex * (hbeta + 1))

        if np.any(pvover):
            w = np.sqrt(w2[pvover])
            cs = np.cosh(w*h)
            sn = np.sinh(w*h)
            beta = C[pvover]
            ex = np.exp(-beta*h)
            _wo2 = wo2[pvover]
            _w2 = w2[pvover]
            _k = k[pvover]

            # for displacement:
            F[pvover] = ex*(cs + (beta/w)*sn)
            G[pvover] = (ex*sn)/w
            t0 = 1 / (h*_k*w)
            t1 = (_w2 + beta*beta)/_wo2
            t2 = (2*w*beta)/_wo2
            A[pvover] = t0 * (ex * (-(t1 + h*beta)*sn -
                                    (t2 + h*w)*cs) + t2)
            B[pvover] = t0 * (ex * (t1*sn + t2*cs) + w*h - t2)

            # for velocity:
            Fp[pvover] = -(_wo2/w) * ex * sn
            Gp[pvover] = ex * (cs - (beta/w) * sn)
            Ap[pvover] = t0 * (ex * ((beta + h*_wo2)*sn + w*cs)-w)
            Bp[pvover] = t0 * (-ex * (beta*sn + w*cs) + w)

    if rfmodes is not None:
        F[rfmodes] = 0
        G[rfmodes] = 0
        A[rfmodes] = 0
        B[rfmodes] = 1 / k[rfmodes]    # from k q = force
        Fp[rfmodes] = 0
        Gp[rfmodes] = 0
        Ap[rfmodes] = 0
        Bp[rfmodes] = 0

    return SimpleNamespace(F=F, G=G, A=A, B=B,
                           Fp=Fp, Gp=Gp, Ap=Ap, Bp=Bp, pvrb=pvrb)


def finddups(v, tol=0.):
    """
    Find duplicate values in a vector (or within a tolerance).

    Parameters
    ----------
    v : 1d array_like
        Vector to find duplicates in.
    tol : scalar; optional
        Tolerance for checking for duplicates. Values are considered
        duplicates if the absolute value of the difference is <= `tol`.

    Returns
    -------
    dups : 1d ndarray
        Bool partition vector for repeated values. `dups` will have
        True for any value that is repeated anywhere else in the
        vector. It will be all False if there are no repeated values.

    Examples
    --------
    >>> from pyyeti import ode
    >>> ode.finddups([0, 10, 2, 2, 6, 10, 10])
    array([False,  True,  True,  True, False,  True,  True], dtype=bool)
    """
    v = np.atleast_1d(v)
    i = np.argsort(v)
    dif = np.diff(v[i])
    dups = np.zeros(v.size, bool)
    tf = abs(dif) <= tol
    dups[i[1:-1]] = np.logical_or(tf[1:], tf[:-1])
    dups[i[0]] = tf[0]
    dups[i[-1]] = tf[-1]
    return dups


def _eigc_dups(lam, tol=1.e-10):
    """
    Find duplicate complex eigenvalues from state-space formulation.

    Parameters
    ----------
    lam : 1d ndarray
        Vector of complex eigenvalues. Any complex-conjugate pairs must
        be adjacent (this is the normal case for
        :func:`scipy.linalg.eig`).
    tol : scalar; optional
        Tolerance for checking for repeated roots; lam[j] and lam[k]
        are repeated roots if abs(lam[j]-lam[k]) < tol

    Returns
    -------
    lams : 1d ndarray
        Sorted version of lam; real (overdamped) lambdas are first;
        complex lambdas are last. There is no sorting within the real
        lambdas. Conjugates will remain adjacent.
    i : 1d ndarray
        The sorting vector: lams = lam[i]
    dups : 1d ndarray
        Index partition vector for repeated roots; it will be empty
        (`np.array([])`) if there are no repeated roots. For example,
        if only the second and third roots are duplicates of each other,
        `dups` will be `np.array([1, 2])`.

    Notes
    -----
    Input lam is the vector of eigenvalues of A defined by::

        yd - A y = 0

    and the solution of ud - lam u = 0 is of the form::

        u = k exp(lam t)

    This routine is normally called by :func:`eigss`.
    """
    i = np.argsort(abs(lam.imag), kind='mergesort')
    lams = lam[i]  # order: real then complex w/ conjugates adjacent
    # find repeated roots:
    dups = np.nonzero(finddups(lams, tol))[0]
    return lams, i, dups


def eigss(A, delcc):
    r"""
    Solve complex eigen problem for state-space formulation.

    Parameters
    ----------
    A : 2d ndarray
        The state-space matrix (which doesn't have to be defined as
        below)
    delcc : bool
        If True, delete one of each complex-conjugate pair and put the
        appropriate factor of 2. in ur output (see below). This is
        handy for real time-domain solutions, but not for frequency
        domain solutions. See note below.

    Returns
    -------
    lam : 1d ndarray
        The vector of complex eigenvalues
    ur : 2d ndarray
        Normalized complex right eigenvectors
    ur_inv : 2d ndarray
        Inverse of right eigenvectors
    dups : 1d ndarray
        Index partition vector for repeated roots; it will be empty
        (`np.array([])`) if there are no repeated roots. For example,
        if only the second and third roots are duplicates of each other,
        `dups` will be `np.array([1, 2])`.

    Notes
    -----
    The typical 2nd order ODE is:

    .. math::
        M \ddot{q} + B \dot{q} + K q = F

    The 2nd order ODE set of equations are transformed into the
    1st order ODE (see :func:`make_A`):

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} M^{-1} B & M^{-1} K \\ I & 0 \end{array}
        \right] \left\{
            \begin{array}{c} \dot{q} \\ q \end{array}
        \right\} = \left\{
            \begin{array}{c} M^{-1} F \\ 0 \end{array} \right\}

    or:

    .. math::
        \dot{y} - A y = w

    When the `M`, `B` and `K` are assembled into the `A` matrix, they
    must not contain any rigid-body modes since the inverse of `ur`
    may not exist, causing the method to fail. If you seen any warning
    messages about a matrix being singular or near singular, the
    method has likely failed. Duplicate roots can also cause trouble,
    so if there are duplicates, check to see if ``ur_inv @ ur`` and
    ``ur_inv @ A @ ur`` are diagonal matrices (if ``not delcc``, these
    would be identity and the eigenvalues, but if `delcc` is True, the
    factor of 2.0 discussed next has the chance to modify that). If
    method fails, see :class:`SolveExp1` or :class:`SolveExp2`.

    For underdamped modes, the complex eigenvalues and modes come in
    complex conjugate pairs. Each mode of a pair yields the same
    solution for real time-domain problems. This routine takes
    advantage of this fact (if `delcc` is True) by chopping out one of
    the pair and multiplying the other one by 2.0 (in `ur`). Then, if
    all modes are underdamped: ``len(lam) = M.shape[0]`` and if no
    modes are underdamped: ``len(lam) = 2*M.shape[0]``.

    See also
    --------
    :func:`make_A`, :class:`SolveUnc`.
    """
    lam, ur = la.eig(A)
    ur_inv = la.inv(ur)
    lam, i, dups = _eigc_dups(lam)
    ur = ur[:, i]
    ur_inv = ur_inv[i]
    if delcc:
        return delconj(lam, ur, ur_inv, dups)
    return lam, ur, ur_inv, dups


def delconj(lam, ur, ur_inv, dups):
    """
    Delete one eigenvalue/eigenvector from of each pair of complex
    conjugates.

    Parameters
    ----------
    lam : 1d ndarray
        The vector of complex eigenvalues
    ur : 2d ndarray
        Normalized complex right eigenvectors
    ur_inv : 2d ndarray
        Inverse of right eigenvectors
    dups : 1d array_like
        Index partition vector for repeated roots; it will be empty
        (`np.array([])`) if there are no repeated roots. For example,
        if only the second and third roots are duplicates of each other,
        `dups` will be `np.array([1, 2])`.

    Returns
    -------
    lam1 : 1d ndarray
        Trimmed vector of complex eigenvalues
    ur1 : 2d ndarray
        Trimmed normalized complex right eigenvectors; columns may be
        trimmed
    ur_inv1 : 2d ndarray
        Trimmed inverse of right eigenvectors; rows may be trimmed
    dups1 : 1d ndarray
        Version of input `dups` for the trimmed variables.

    Notes
    -----
    This function is typically called via :func:`eigss`. If there are
    any values in `lam` that are not part of a complex-conjugate pair,
    this routine does nothing.

    :func:`delconj` can safely be called even if modes were already
    deleted. In this case, outputs will be the same as the inputs.
    """
    # delete the negatives (imaginary part) of each comp-conj pair:
    neg = lam.imag < 0.
    if np.any(neg):
        # see if lambda's are all comp-conj pairs; if not, do nothing
        posi = np.nonzero(lam.imag > 0.)[0]
        negi = np.nonzero(neg)[0]
        if (posi.size == negi.size and
                np.all(abs(posi-negi) == 1) and
                np.all(lam[posi] == np.conj(lam[negi]))):
            # so, keep reals and positives of each complex conj. pair:
            keep = np.logical_not(neg)
            lam = lam[keep]
            ur = ur[:, keep]
            ur_inv = ur_inv[keep]
            # update dups:
            dups = np.atleast_1d(dups)
            if dups.size > 0:
                ndof = ur.shape[0]
                temp = np.zeros(ndof, bool)
                temp[dups] = True
                temp = temp[keep]
                dups = np.nonzero(temp)[0]
            # put factor of 2 in ur for recovery (because only one of
            # the complex conjugate pairs were solved for)
            pv = lam.imag > 0.
            if np.any(pv):
                ur[:, pv] *= 2.
    return lam, ur, ur_inv, dups


def addconj(lam, ur, ur_inv):
    """
    Add back in the missing complex-conjugate mode

    Parameters
    ----------
    lam : 1d ndarray
        The vector of complex eigenvalues
    ur : 2d ndarray
        Normalized complex right eigenvectors
    ur_inv : 2d ndarray
        Inverse of right eigenvectors

    Returns
    -------
    lam1 : 1d ndarray
        The vector of complex eigenvalues (complete set)
    ur1 : 2d ndarray
        Complete set of normalized complex right eigenvectors; will be
        square
    ur_inv1 : 2d ndarray
        Complete set of inverse of right eigenvectors; will be square

    Notes
    -----
    This routine adds the missing one of each complex-conjugate pair
    from state-space complex eigensolution and divides the other one
    by 2 in ur.

    :func:`addconj` can safely be called even if modes were already
    added back in (or if they were never deleted).

    Though unlikely, :func:`addconj` could be fooled into adding
    inappropriate modes if modes were deleted in a different manner
    than how :func:`eigss` does it. This routine does some checks to
    try to ensure that the inputs have been processed as expected:

    - checks for all positive imaginary parts in lam (:func:`eigss`
      deletes the negative conjugates)
    - check for the factor of 2 (see above) in ur
    - check that `ur1` and `ur_inv1` will be square after adding in
      the modes
    - check that all underdamped modes are sorted last (as
      :func:`eigss` has them)

    See also
    --------
    :func:`eigss`, :class:`SolveUnc`, :class:`SolveExp2`
    """
    conj2 = np.nonzero(lam.imag > 0.)[0]
    if ur.shape[0] > ur.shape[1] and conj2.size > 0:
        if np.any(lam.imag < 0.):
            return lam, ur, ur_inv
        two = ur_inv[conj2[0]] @ ur[:, conj2[0]]
        if abs(two - 2.) > 1e-13:
            raise ValueError('factor of 2.0 seems to be missing: '
                             'error on first underdamped mode = {}'.
                             format(abs(two - 2.)))
        n = len(lam) + len(conj2)
        if n != ur_inv.shape[1] or n != ur.shape[0]:
            raise ValueError('ur and/or ur_inv will not be square '
                             'after adding missing modes')
        reals = np.nonzero(lam.imag == 0.)[0]
        if len(reals) > 0 and np.max(reals) > np.min(conj2):
            raise ValueError('not all underdamped are last')
        conj2_new = conj2 + np.arange(1, len(conj2)+1)
        conj1_new = conj2_new - 1
        lam1 = np.zeros(n, complex)
        ur_inv1 = np.zeros((n, n), complex)
        ur1 = np.zeros((n, n), complex)
        if reals.size > 0:
            lam1[reals] = lam[reals]
            ur1[:, reals] = ur[:, reals]
            ur_inv1[reals] = ur_inv[reals]
        lam1[conj1_new] = lam[conj2]
        lam1[conj2_new] = np.conj(lam[conj2])
        ur1[:, conj1_new] = ur[:, conj2]/2.
        ur1[:, conj2_new] = np.conj(ur[:, conj2])/2.
        ur_inv1[conj1_new] = ur_inv[conj2]
        ur_inv1[conj2_new] = np.conj(ur_inv[conj2])
        return lam1, ur1, ur_inv1
    return lam, ur, ur_inv


def make_A(M, B, K):
    r"""
    Setup the state-space matrix from mass, damping and stiffness.

    Parameters
    ----------
    M : 1d or 2d ndarray or None
        Mass; vector (of diagonal), or full; if None, mass is assumed
        identity
    B : 1d or 2d ndarray
        Damping; vector (of diagonal), or full
    K : 1d or 2d ndarray
        Stiffness; vector (of diagonal), or full

    Returns
    -------
    A : 2d ndarray
        The state-space matrix defined as shown below

    Notes
    -----
    The typical 2nd order ODE is:

    .. math::
        M \ddot{q} + B \dot{q} + K q = F

    The 2nd order ODE set of equations are transformed into the
    1st order ODE:

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} M^{-1} B & M^{-1} K \\ I & 0 \end{array}
        \right] \left\{
            \begin{array}{c} \dot{q} \\ q \end{array}
        \right\} = \left\{
            \begin{array}{c} M^{-1} F \\ 0 \end{array} \right\}

    or:

    .. math::
        \dot{y} - A y = w

    When the `M`, `B` and `K` are assembled into the `A` matrix, they
    must not contain any rigid-body modes. See :func:`eigss`.

    See also
    --------
    :func:`eigss`, :class:`SolveUnc`, :class:`SolveExp2`
    """
    Atype = float
    if M is None:
        B, K = np.atleast_1d(B, K)
        if np.iscomplexobj(B) or np.iscomplexobj(K):
            Atype = complex
    else:
        M, B, K = np.atleast_1d(M, B, K)
        if (np.iscomplexobj(M) or np.iscomplexobj(B) or
                np.iscomplexobj(K)):
            Atype = complex
    n = K.shape[0]
    A = np.zeros((2*n, 2*n), Atype)
    v1 = range(n)
    v2 = range(n, 2*n)
    if B.ndim == 2:
        A[:n, :n] = -B
    else:
        A[v1, v1] = -B
    if K.ndim == 2:
        A[:n, n:] = -K
    else:
        A[v1, v2] = -K
    A[v2, v1] = 1.
    if M is not None:
        if M.ndim == 1:
            A[:n] = (1./M).reshape(-1, 1) * A[:n]
        else:
            A[:n] = la.solve(M, A[:n])
    return A


class SolveExp1(object):
    """
    1st order ODE time domain solver based on the matrix exponential.

    This class is for solving: ``yd - A y = f``

    This solver is exact assuming either piece-wise linear or
    piece-wise constant forces.

    Examples
    --------
    Calculate impulse response of state-space system::

        xd = A @ x + B @ u
        y  = C @ x + D @ u

    where:
        - force = 0's
        - velocity(0) = B

    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> from pyyeti.ssmodel import SSModel
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> f = 5           # 5 hz oscillator
        >>> w = 2*np.pi*f
        >>> w2 = w*w
        >>> zeta = .05
        >>> h = .01
        >>> nt = 500
        >>> A = np.array([[0, 1], [-w2, -2*w*zeta]])
        >>> B = np.array([[0], [3]])
        >>> C = np.array([[8, -5]])
        >>> D = np.array([[0]])
        >>> F = np.zeros((1, nt), float)
        >>> ts = ode.SolveExp1(A, h)
        >>> sol = ts.tsolve(B @ F, B[:, 0])
        >>> y = C @ sol.d
        >>> fig = plt.figure('SolveExp1 demo')
        >>> ax = plt.plot(sol.t, y.T,
        ...               label='SolveExp1')
        >>> ssmodel = SSModel(A, B, C, D)
        >>> z = ssmodel.c2d(h=h, method='zoh')
        >>> x = np.zeros((A.shape[1], nt+1), float)
        >>> y2 = np.zeros((C.shape[0], nt), float)
        >>> x[:, 0:1] = B
        >>> for k in range(nt):
        ...     x[:, k+1] = z.A @ x[:, k] + z.B @ F[:, k]
        ...     y2[:, k]  = z.C @ x[:, k] + z.D @ F[:, k]
        >>> ax = plt.plot(sol.t, y2.T, label='discrete')
        >>> leg = plt.legend(loc='best')
        >>> np.allclose(y, y2)
        True

        Compare against scipy:

        >>> from scipy import signal
        >>> ss = ssmodel.getlti()
        >>> tout, yout = ss.impulse(T=sol.t)
        >>> ax = plt.plot(tout, yout, label='scipy')
        >>> leg = plt.legend(loc='best')
        >>> np.allclose(yout, y.ravel())
        True
    """

    def __init__(self, A, h, order=1):
        """
        Instantiates a :class:`SolveExp1` solver.

        Parameters
        ----------
        A : 2d ndarray
            The state-space matrix: ``yd - A y = f``
        h : scalar or None
            Time step or None; if None, the `E`, `P`, `Q` members will
            not be computed.
        order : scalar, optional
            Specify which solver to use:

            - 0 for the zero order hold (force stays constant across
              time step)
            - 1 for the 1st order hold (force can vary linearly across
              time step)

        Notes
        -----
        The class instance is populated with the following members:

        =======   ================================================
        Member    Description
        =======   ================================================
        A         the input `A`
        h         time step
        n         number of total DOF (``A.shape[0]``)
        order     order of solver (0 or 1; see above)
        E, P, Q   output of :func:`expmint.getEPQ`; they are
                  matrices used to solve the ODE
        pc        True if E, P, and Q member have been calculated;
                  False otherwise
        =======   ================================================

        The E, P, and Q entries are used to solve the ODE::

            for j in range(1, nt):
                d[:, j] = E*d[:, j-1] + P*F[:, j-1] + Q*F[:, j]
        """
        if h:
            E, P, Q = expmint.getEPQ(A, h, order)
            self.E = E
            self.P = P
            self.Q = Q
        self.A = A
        self.h = h
        self.order = order
        self.n = A.shape[0]

    def tsolve(self, force, d0=None):
        """
        Solve time-domain 1st order ODE equations.

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x time
        d0 : 1d ndarray; optional
            Displacement initial conditions; if None, zero ic's are
            used.

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x time
        v : 2d ndarray
            Velocity; ndof x time
        h : scalar
            Time-step
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        """
        force = np.atleast_2d(force)
        if force.shape[0] != self.n:
            raise ValueError('Force matrix has {} rows; {} rows are '
                             'expected'.format(force.shape[0], self.n))
        nt = force.shape[1]
        d = np.zeros((self.n, nt))  # , float, order='F')
        if d0 is not None:
            d[:, 0] = d0
        else:
            d0 = np.zeros(self.n, float)
        if nt > 1:
            if not self.h:
                raise RuntimeError('instantiate the class with a valid '
                                   'time step.')
            # calc force term outside loop:
            if self.order == 1:
                PQF = self.P @ force[:, :-1] + self.Q @ force[:, 1:]
            else:
                PQF = self.P @ force[:, :-1]
            E = self.E
            for j in range(1, nt):
                d0 = d[:, j] = E @ d0 + PQF[:, j-1]
            t = self.h * np.arange(nt)
        else:
            t = 0.
        return SimpleNamespace(d=d, v=force+self.A @ d, h=self.h, t=t)


class _BaseODE(object):
    """
    Base class for time and frequency domain equations of motion
    solvers.
    """

    def tsolve(self):
        """'Abstract' method to solve time-domain equations"""
        raise NotImplementedError

    def fsolve(self):
        """'Abstract' method to solve frequency-domain equations"""
        raise NotImplementedError

    def generator(self):
        """
        'Abstract' method to get Python "generator" version of
        :class:`SolveExp2` solver. This is to interactively solve (or
        re-solve) one step at a time.
        """
        raise NotImplementedError

    def finalize(self, get_force=False):
        """
        Finalize time-domain generator solution.

        Parameters
        ----------
        get_force : bool; optional
            If True, the `force` entry will be included in the
            returned data structure.

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x time
        v : 2d ndarray
            Velocity; ndof x time
        a : 2d ndarray
            Acceleration; ndof x time
        h : scalar
            Time-step or None
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        force : 2d ndarray; optional
            Force; ndof x time. Only included if `get_force` is True.
        """
        d, v, a, f = self._d, self._v, self._a, self._force
        del self._d, self._v, self._a, self._force
        self._calc_acce_kdof(d, v, a, f)
        sol = self._solution(d, v, a)
        if get_force:
            sol.force = f
        return sol

    #
    # Utility routines follow:
    #
    def _solution(self, d, v, a):
        """Returns SimpleNamespace object with d, v, a, h, t"""
        if self.h:
            t = self.h * np.arange(a.shape[1])
        else:
            t = 0.
        if self.pre_eig:
            d = self.phi @ d
            v = self.phi @ v
            a = self.phi @ a
        return SimpleNamespace(d=d, v=v, a=a, h=self.h, t=t)

    def _solution_freq(self, d, v, a, freq):
        """Returns SimpleNamespace object with d, v, a, freq"""
        if self.pre_eig:
            d = self.phi @ d
            v = self.phi @ v
            a = self.phi @ a
        return SimpleNamespace(d=d, v=v, a=a, f=freq)

    def _mk_slice(self, pv):
        """Convert index partition vector to slice object:
        ``start:stop``. Raises ValueError if `pv` cannot be converted
        to this type of slice object."""
        if pv.size == 0:
            return slice(0, 0)
        # if pv.size == pv[-1]+1 - pv[0]:
        if np.all(np.diff(pv) == 1):
            return slice(pv[0], pv[-1]+1)
        raise ValueError('invalid partition vector for conversion '
                         'to slice')

    def _mk_slices(self, dorbel):
        """Convert index partition vectors to slice objects and sets
        ``slices=True`` if successful."""
        try:
            nonrf = self._mk_slice(self.nonrf)
            rf = self._mk_slice(self.rf)
            kdof = self._mk_slice(self.kdof)
            if dorbel:
                rb = self._mk_slice(self.rb)
                el = self._mk_slice(self.el)
                _rb = self._mk_slice(self._rb)
                _el = self._mk_slice(self._el)
            self.nonrf = nonrf
            self.rf = rf
            self.kdof = kdof
            if dorbel:
                self.rb = rb
                self.el = el
                self._rb = _rb
                self._el = _el
            self.slices = True
        except ValueError:
            self.slices = False

    def _chk_diag_part(self, m, b, k):
        """Checks for all-diagonal and partitions based on rf modes"""
        unc = 0
        krf = None
        if (m is None or m.ndim == 1 or
                (m.ndim == 2 and ytools.isdiag(m))):
            unc += 1
        if b.ndim == 1 or (b.ndim == 2 and ytools.isdiag(b)):
            unc += 1
        if k.ndim == 1 or (k.ndim == 2 and ytools.isdiag(k)):
            unc += 1
        if unc == 3:
            unc = True
            if m is not None and m.ndim == 2:
                m = np.diag(m).copy()
            if b.ndim == 2:
                b = np.diag(b).copy()
            if k.ndim == 2:
                k = np.diag(k).copy()
            if self.rfsize:
                if m is not None:
                    m = m[self.nonrf]
                b = b[self.nonrf]
                krf = k[self.rf]
                k = k[self.nonrf]
        else:
            unc = False
            if m is not None and m.ndim == 1:
                m = np.diag(m)
            if b.ndim == 1:
                b = np.diag(b)
            if k.ndim == 1:
                k = np.diag(k)
            if self.rfsize:
                pvrf = np.ix_(self.rf, self.rf)
                pvnonrf = np.ix_(self.nonrf, self.nonrf)
                if m is not None:
                    m = m[pvnonrf]
                b = b[pvnonrf]
                krf = k[pvrf]
                k = k[pvnonrf]
        self.m = m
        self.b = b
        self.k = k
        self.krf = krf
        self.unc = unc

    def _inv_krf(self):
        """Decompose the krf matrix"""
        if self.rfsize:
            krf = self.krf
            if self.unc:
                ikrf = (1./krf).reshape(-1, 1)
                c = abs(krf).max()/abs(krf).min()
            else:
                ikrf = la.lu_factor(krf)
                c = np.linalg.cond(krf)
            if c > 1/np.finfo(float).eps:
                msg = ('the residual-flexibility part of the '
                       'stiffness is poorly conditioned '
                       '(cond={:.3e}). Displacements will likely '
                       'be inaccurate.').format(c)
                warnings.warn(msg, RuntimeWarning)
            self.ikrf = ikrf

    def _get_inv_m(self, m):
        """Decompose the mass matrix"""
        if self.unc:
            invm = (1./m).reshape(-1, 1)
            c = abs(m).max()/abs(m).min()
        else:
            invm = la.lu_factor(m)
            c = np.linalg.cond(m)
        if c > 1/np.finfo(float).eps:
            msg = ('the mass matrix is poorly conditioned '
                   '(cond={:.3e}). Solution will likely be '
                   'inaccurate.').format(c)
            warnings.warn(msg, RuntimeWarning)
        return invm

    def _inv_m(self):
        """Decompose the mass matrix"""
        if self.m is not None and self.ksize:
            self.invm = self._get_inv_m(self.m)

    def _inv_mrb(self):
        """Decompose the rigid-body part of the mass matrix"""
        if self.m is not None and self.rbsize:
            if self.unc:
                mrb = self.m[self.rb]
            else:
                mrb = self.m[np.ix_(self.rb, self.rb)]
            self.imrb = self._get_inv_m(mrb)

    def _assert_square(self, n, m, b, k):
        if m is not None:
            name = ('mass', 'damping', 'stiffness')
            mats = (m, b, k)
        else:
            name = ('damping', 'stiffness')
            mats = (b, k)
        any_2d = False
        for i, mat in enumerate(mats):
            if mat.ndim == 2:
                any_2d = True
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError("{} matrix is non-square!".
                                     format(name[i]))
                if mat.shape[0] != n:
                    raise ValueError("{} matrix has a different "
                                     "number of rows than the "
                                     "stiffness!".format(name[i]))
            elif mat.ndim == 1:
                if mat.shape[0] != n:
                    raise ValueError("length of {} diagonal is "
                                     "not compatible with the "
                                     "stiffness!".format(name[i]))
            else:
                raise ValueError("{} has more than 2 dimensions!".
                                 format(name[i]))
        return any_2d

    def _do_pre_eig(self, m, b, k):
        """Do a "pre" eigensolution to put system in modal space"""
        err = False
        if k.ndim == 1:
            k = np.diag(k)
        if m is None:
            w, u = la.eigh(k)
            kdiag = u.T @ k @ u
            if not ytools.isdiag(kdiag):
                err = True
        else:
            if m.ndim == 1:
                m = np.diag(m)
            w, u = la.eigh(k, m)
            kdiag = u.T @ k @ u
            mdiag = u.T @ m @ u
            if not ytools.isdiag(kdiag) or not ytools.isdiag(mdiag):
                err = True
        if err:
            raise ValueError('`pre_eig` option failed to '
                             'diagonlized the mass and/or '
                             'stiffness. Check '
                             'for symmetric/hermitian stiffness '
                             'and positive-definite mass')
        self.pre_eig = True
        self.phi = u
        if m is not None:
            m = np.diag(mdiag).copy()
        k = np.diag(kdiag).copy()
        b = u.T @ b @ u
        return m, b, k

    def _common_precalcs(self, m, b, k, h, rb, rf, pre_eig=False):
        systype = float
        self.mid = id(m)
        self.bid = id(b)
        self.kid = id(k)
        if m is None:
            b, k = np.atleast_1d(b, k)
            if np.iscomplexobj(b) or np.iscomplexobj(k):
                systype = complex
        else:
            m, b, k = np.atleast_1d(m, b, k)
            if (np.iscomplexobj(m) or np.iscomplexobj(b) or
                    np.iscomplexobj(k)):
                systype = complex
        n = k.shape[0]
        any_2d = self._assert_square(n, m, b, k)
        if pre_eig and any_2d:
            m, b, k = self._do_pre_eig(m, b, k)
        else:
            self.pre_eig = False
        nonrf = np.ones(n, bool)
        if rf is None:
            rf = np.array([], bool)
        else:
            rf = np.ix_(np.atleast_1d(rf))[0]
        nonrf[rf] = False
        nonrf = np.nonzero(nonrf)[0]
        self.n = n
        self.h = h
        self.rf = rf
        self.nonrf = nonrf
        self.rfsize = rf.size
        self.nonrfsz = nonrf.size
        self.kdof = nonrf
        self.ksize = nonrf.size
        self._chk_diag_part(m, b, k)
        self._make_rb_el(rb)
        self._zero_rbpart()
        self._inv_krf()
        self.systype = systype

    def _make_rb_el(self, rb):
        if rb is None:
            if self.ksize:
                tol = .005
                if self.unc:
                    _rb = np.nonzero(abs(self.k) < tol)[0]
                else:
                    _rb = np.nonzero(abs(self.k).max(axis=0) < tol)[0]
                rb = np.zeros(self.n, bool)
                rb[self.nonrf[_rb]] = True
                rb = np.nonzero(rb)[0]
            else:
                rb = _rb = np.array([], bool)
        else:
            rb = np.ix_(np.atleast_1d(rb))[0]
            vec = np.zeros(self.n, bool)
            vec[rb] = True
            _rb = np.nonzero(vec[self.nonrf])[0]
        _el = np.ones(self.ksize, bool)
        _el[_rb] = False
        _el = np.nonzero(_el)[0]
        el = np.zeros(self.n, bool)
        el[self.nonrf[_el]] = True
        el = np.nonzero(el)[0]
        # _rb, _el are relative to non-rf part
        self.rb = rb
        self._rb = _rb
        self.el = el
        self._el = _el
        self.rbsize = rb.size
        self.elsize = el.size

    def _zero_rbpart(self):
        """Zero out rigid-body part of stiffness and damping"""
        if self.rbsize:
            if id(self.b) == self.bid:
                self.b = self.b.copy()
            if id(self.k) == self.kid:
                self.k = self.k.copy()
            if self.unc:
                self.b[self._rb] = 0
                self.k[self._rb] = 0
            else:
                self.b[self._rb, self._rb] = 0
                self.k[self._rb, self._rb] = 0

    def _build_A(self):
        """Builds the A matrix: yd - A y = f"""
        n = self.k.shape[0]
        A = np.zeros((2*n, 2*n), self.systype)
        v1 = range(n)
        v2 = range(n, 2*n)
        if self.unc:
            A[v1, v1] = -self.b
            A[v1, v2] = -self.k
        else:
            A[:n, :n] = -self.b
            A[:n, n:] = -self.k
        A[v2, v1] = 1.
        if self.m is not None:
            if self.unc:
                A[:n] *= self.invm
            else:
                A[:n] = la.lu_solve(self.invm, A[:n],
                                    check_finite=False)
        return A

    def _alloc_dva(self, nt, istime):
        n = self.ksize
        if istime:
            if nt > 1 and n > 0 and not self.pc:
                raise RuntimeError(
                    'rerun `{}` with a valid time step.'
                    .format(type(self).__name__))
            d = np.zeros((self.n, nt), self.systype)
            v = np.zeros((self.n, nt), self.systype)
            a = np.zeros((self.n, nt), self.systype)
        else:
            d = np.zeros((self.n, nt), complex)
            v = np.zeros((self.n, nt), complex)
            a = np.zeros((self.n, nt), complex)
        return d, v, a

    def _init_dv(self, d, v, d0, v0, F0, static_ic):
        if d0 is not None:
            d[:, 0] = d0
        elif static_ic and self.elsize:
            if self.unc:
                d0 = la.lstsq(np.diag(self.k[self._el]),
                              F0[self.el])
                d[self.el, 0] = d0[0]
            else:
                d0 = la.lstsq(self.k, F0[self.kdof])
                d[self.kdof, 0] = d0[0]
        if v0 is not None:
            v[:, 0] = v0

    def _init_dva_part(self, nt, F0, d0, v0, static_ic,
                       istime=True):
        if F0.shape[0] != self.n:
            raise ValueError('Initial force vector has {} elements;'
                             ' {} elements are expected'
                             .format(F0.shape[0], self.n))
        if self.pre_eig:
            raise NotImplementedError(
                ':class:`SolveUnc` generator not yet implemented '
                'using the `pre_eig` option')

        d, v, a = self._alloc_dva(nt, istime)
        f = a.copy()
        f[:, 0] = F0
        self._init_dv(d, v, d0, v0, F0, static_ic)
        if self.rfsize:
            if self.unc:
                d[self.rf, 0] = self.ikrf.ravel() * F0[self.rf]
            else:
                d[self.rf, 0] = la.lu_solve(self.ikrf, F0[self.rf],
                                            check_finite=False)
        return d, v, a, f

    def _init_dva(self, force, d0, v0, static_ic, istime=True):
        if force.shape[0] != self.n:
            raise ValueError('Force matrix has {} rows; {} rows are '
                             'expected'
                             .format(force.shape[0], self.n))

        d, v, a = self._alloc_dva(force.shape[1], istime)

        if self.pre_eig:
            force = self.phi.T @ force

        self._init_dv(d, v, d0, v0, force[:, 0], static_ic)

        if self.rfsize:
            if self.unc:
                d[self.rf] = self.ikrf * force[self.rf]
            else:
                d[self.rf] = la.lu_solve(self.ikrf, force[self.rf],
                                         check_finite=False)
        return d, v, a, force

    def _calc_acce_kdof(self, d, v, a, force):
        """Calculate the `kdof` part of the acceleration"""
        if self.ksize:
            kdof = self.kdof
            F = force[kdof]
            if self.unc:
                B = self.b[:, None] * v[kdof]
                K = self.k[:, None] * d[kdof]
                if self.m is not None:
                    a[kdof] = self.invm * (F - B - K)
                else:
                    a[kdof] = F - B - K
            else:
                B = self.b @ v[kdof]
                K = self.k @ d[kdof]
                if self.m is not None:
                    a[kdof] = la.lu_solve(self.invm, F - B - K,
                                          check_finite=False)
                else:
                    a[kdof] = F - B - K

    def _force_freq_compat_chk(self, force, freq):
        """Check compatibility between force matrix and freq vector"""
        if force.shape[1] != len(freq):
            raise ValueError('Number of columns `force` ({}) does '
                             'not equal length of `freq` ({})'.
                             format(force.shape[1], len(freq)))


class SolveExp2(_BaseODE):
    r"""
    2nd order ODE time domain solver based on the matrix exponential.

    This class is for solving:

    .. math::
        M \ddot{q} + B \dot{q} + K q = F

    The 2nd order ODE set of equations are transformed into the
    1st order ODE:

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} M^{-1} B & M^{-1} K \\ I & 0 \end{array}
        \right] \left\{
            \begin{array}{c} \dot{q} \\ q \end{array}
        \right\} = \left\{
            \begin{array}{c} M^{-1} F \\ 0 \end{array} \right\}

    or:

    .. math::
        \dot{y} - A y = w

    Unlike for the uncoupled solver :class:`SolveUnc`, this solver
    doesn't need or use the `rb` input unless static initial
    conditions are requested when solving equations.

    Like :class:`SolveUnc`, this solver is exact assuming piece-wise
    linear forces (if `order` is 1) or piece-wise constant forces (if
    `order` is 0).

    The above equations are for the non-residual-flexibility
    modes. The 'rf' modes are solved statically and any initial
    conditions are ignored for them.

    For a static solution:

        - rigid-body displacements = zeros
        - elastic displacments = inv(k[elastic]) * F[elastic]
        - velocity = zeros
        - rigid-body accelerations = inv(m[rigid]) * F[rigid]
        - elastic accelerations = zeros

    See also
    --------
    :class:`SolveUnc`.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # percent damp
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> h = .001                              # time step
        >>> t = np.arange(0, .3001, h)            # time vector
        >>> c = 2*np.pi
        >>> f = np.vstack((3*(1-np.cos(c*2*t)),
        ...                4.5*(np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> ts = ode.SolveExp2(m, b, k, h)
        >>> sol = ts.tsolve(f, static_ic=1)

        Solve with scipy.signal.lsim for comparison:

        >>> A = ode.make_A(m, b, k)
        >>> n = len(m)
        >>> Z = np.zeros((n, n), float)
        >>> B = np.vstack((np.eye(n), Z))
        >>> C = np.vstack((A, np.hstack((Z, np.eye(n)))))
        >>> D = np.vstack((B, Z))
        >>> ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
        >>> import scipy.signal
        >>> f2 = (1./m).reshape(-1, 1) * f
        >>> tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T, t,
        ...                                X0=ic)
        >>> print('acce cmp:', np.allclose(yl[:, :n], sol.a.T))
        acce cmp: True
        >>> print('velo cmp:', np.allclose(yl[:, n:2*n], sol.v.T))
        velo cmp: True
        >>> print('disp cmp:', np.allclose(yl[:, 2*n:], sol.d.T))
        disp cmp: True

        Plot the four accelerations:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('SolveExp2 vs. lsim', figsize=[8, 8])
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, sol.a[j], label='SolveExp2')
        ...     _ = plt.plot(tl, yl[:, j], label='scipy lsim')
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Time (s)')
        ...     if j == 0:
        ...         _ = plt.legend(loc='best')
        >>> plt.tight_layout()
    """

    def __init__(self, m, b, k, h, rb=None, rf=None, order=1,
                 pre_eig=False):
        """
        Instantiates a :class:`SolveExp2` solver.

        Parameters
        ----------
        m : 1d or 2d ndarray or None
            Mass; vector (of diagonal), or full; if None, mass is
            assumed identity
        b : 1d or 2d ndarray
            Damping; vector (of diagonal), or full
        k : 1d or 2d ndarray
            Stiffness; vector (of diagonal), or full
        h : scalar or None
            Time step; can be None if only want to solve a static
            problem
        rb : 1d array or None; optional
            Index partition vector for rigid-body modes. If None, the
            rigid-body modes will be automatically detected by this
            logic::

                rb = np.nonzero(abs(k) < .005)[0]  # for diagonal k
                rb = np.nonzero(abs(k).max(0) < .005)[0]  # for full k

            Set to [] to specify no rigid-body modes. Note: the
            detection of rigid-body modes is done after the `pre_eig`
            operation, if that is True.
        rf : 1d array or None; optional
            Index partition vector for res-flex modes; these will be
            solved statically
        order : integer; optional
            Specify which solver to use:

            - 0 for the zero order hold (force stays constant across
              time step)
            - 1 for the 1st order hold (force can vary linearly across
              time step)

        pre_eig : bool; optional
            If True, an eigensolution will be computed with the mass
            and stiffness matrices to convert the system to modal
            space. This will allow the automatic detection of
            rigid-body modes which is necessary for specifying
            "static" initial conditions when calling the solver. No
            modes are truncated. Only works if stiffness is
            symmetric/hermitian and mass is positive definite (see
            :func:`scipy.linalg.eigh`). Just leave it as False if the
            equations are already in modal space or if not using
            "static" initial conditions.

        Notes
        -----
        The instance is populated with the following members:

        =========  ===================================================
        Member     Description
        =========  ===================================================
        m          mass for the non-rf modes
        b          damping for the non-rf modes
        k          stiffness for the non-rf modes
        h          time step
        rb         index vector or slice for the rb modes
        el         index vector or slice for the el modes
        rf         index vector or slice for the rf modes
        _rb        index vector or slice for the rb modes relative to
                   the non-rf modes
        _el        index vector or slice for the el modes relative to
                   the non-rf modes
        nonrf      index vector or slice for the non-rf modes
        kdof       index vector or slice for the non-rf modes
        n          number of total DOF
        rbsize     number of rb modes
        elsize     number of el modes
        rfsize     number of rf modes
        nonrfsz    number of non-rf modes
        ksize      number of non-rf modes
        invm       decomposition of m for the non-rf, non-rb modes
        krf        stiffness for the rf modes
        ikrf       inverse of stiffness for the rf modes
        unc        True if there are no off-diagonal terms in any
                   matrix; False otherwise
        order      order of solver (0 or 1; see above)
        E_vv       partition of "E" which is output of
                   :func:`expmint.getEPQ`
        E_vd       another partition of "E"
        E_dv       another partition of "E"
        E_dd       another partition of "E"
        P, Q       output of :func:`expmint.getEPQ`; they are matrices
                   used to solve the ODE
        pc         True if E*, P, and Q member have been calculated;
                   False otherwise
        pre_eig    True if the "pre" eigensolution was done; False
                   otherwise
        phi        the mode shape matrix from the "pre"
                   eigensolution; only present if `pre_eig` is True
        =========  ===================================================

        The E, P, and Q entries are used to solve the ODE::

            for j in range(1, nt):
                d[:, j] = E*d[:, j-1] + P*F[:, j-1] + Q*F[:, j]
        """
        self._common_precalcs(m, b, k, h, rb, rf, pre_eig)
        self._inv_m()
        self.order = order
        ksize = self.ksize
        if h and ksize > 0:
            A = self._build_A()
            E, P, Q = expmint.getEPQ(A, h, order, half=True)
            self.P = P
            self.Q = Q
            # In state-space, the solution is:
            #   y[n+1] = E @ y[n] + pqf[n, n+1]
            # Put in terms of `d` and `v`:
            #   y = [v; d]
            #   [v[n+1]; d[n+1]] = [E_v, E_d] @ [v[n]; d[n]] +
            #                      pqf[n, n+1]
            #   v[n+1] = [E_vv, E_vd] @ [v[n]; d[n]] +
            #            pqf_v[n, n+1]
            #          = E_vv @ v[n] + E_vd @ d[n] + pqf_v[n, n+1]
            #   d[n+1] = [E_dv, E_dd] @ [v[n]; d[n]] +
            #            pqf_v[n, n+1]
            #          = E_dv @ v[n] + E_dd @ d[n] + pqf_d[n, n+1]

            # copy for faster multiplies:
            self.E_vv = E[:ksize, :ksize].copy()
            self.E_vd = E[:ksize, ksize:].copy()
            self.E_dv = E[ksize:, :ksize].copy()
            self.E_dd = E[ksize:, ksize:].copy()
            self.pc = True
        else:
            self.pc = False
        self._mk_slices(False)

    def tsolve(self, force, d0=None, v0=None, static_ic=False):
        """
        Solve time-domain 2nd order ODE equations

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x time
        d0 : 1d ndarray; optional
            Displacement initial conditions; if None, zero ic's are
            used.
        v0 : 1d ndarray; optional
            Velocity initial conditions; if None, zero ic's are used.
        static_ic : bool; optional
            If True and `d0` is None, then `d0` is calculated such
            that static (steady-state) initial conditions are
            used. Uses the pseudo-inverse in case there are rigid-body
            modes. `static_ic` is ignored if `d0` is not None.

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x time
        v : 2d ndarray
            Velocity; ndof x time
        a : 2d ndarray
            Acceleration; ndof x time
        h : scalar
            Time-step
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        """
        force = np.atleast_2d(force)
        d, v, a, force = self._init_dva(force, d0, v0,
                                        static_ic)
        ksize = self.ksize
        if ksize > 0:
            nt = force.shape[1]
            if nt > 1:
                kdof = self.kdof
                D = d[kdof]
                V = v[kdof]
                if self.m is not None:
                    if self.unc:
                        imf = self.invm * force[kdof]
                    else:
                        imf = la.lu_solve(self.invm, force[kdof],
                                          check_finite=False)
                else:
                    imf = force[kdof]
                if self.order == 1:
                    PQF = self.P @ imf[:, :-1] + self.Q @ imf[:, 1:]
                else:
                    PQF = self.P @ imf[:, :-1]
                E_dd = self.E_dd
                E_dv = self.E_dv
                E_vd = self.E_vd
                E_vv = self.E_vv
                for i in range(nt-1):
                    d0 = D[:, i]
                    v0 = V[:, i]
                    D[:, i+1] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:, i]
                    V[:, i+1] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize, i]
                if not self.slices:
                    d[kdof] = D
                    v[kdof] = V
            self._calc_acce_kdof(d, v, a, force)
        return self._solution(d, v, a)

    def generator(self, nt, F0, d0=None, v0=None, static_ic=False):
        """
        Python "generator" version of :func:`SolveExp2.tsolve`;
        interactively solve (or re-solve) one step at a time.

        Parameters
        ----------
        nt : integer
            Number of time steps
        F0 : 1d ndarray
            Initial force vector
        d0 : 1d ndarray or None; optional
            Displacement initial conditions; if None, zero ic's are
            used.
        v0 : 1d ndarray or None; optional
            Velocity initial conditions; if None, zero ic's are used.
        static_ic : bool; optional
            If True and `d0` is None, then `d0` is calculated such
            that static (steady-state) initial conditions are
            used. Uses the pseudo-inverse in case there are rigid-body
            modes. `static_ic` is ignored if `d0` is not None.

        Returns
        -------
        gen : generator function
            Generator function for solving equations one step at a
            time
        d, v, a : 2d ndarrays
            The displacement, velocity and acceleration arrays. Only
            the first column of `d` and `v` are set; other values are
            all zero.

        Notes
        -----
        To use the generator:

            1. Instantiate a :class:`SolveExp2` instance::

                   ts = SolveExp2(m, b, k, h)

            2. Retrieve a generator and the arrays for holding the
               solution::

                   gen, d, v = ts.generator(len(time), f0)

            3. Use :func:`gen.send` to send a tuple of the next index
               and corresponding force vector in a loop. Re-do
               time-steps as necessary (note that index zero cannot be
               redone)::

                   for i in range(1, len(time)):
                       # Do whatever to get i'th force
                       # - note: d[:, :i] and v[:, :i] are available
                       gen.send((i, fi))

               The class instance will have the attributes `_d`, `_v`
               (same objects as `d` and `v`), `_a`, and `_force`. `d`,
               `v` and `ts._force` are updated on every
               :func:`gen.send`. (`ts._a` is not used until step 4.)

            4. Call :func:`ts.finalize` to get final solution
               in standard form::

                   sol = ts.finalize()

               The internal references `_d`, `_v`, `_a`, and `_force`
               are deleted.

        The generator solver currently has these limitations:

            1. Unlike the normal solver, equations cannot be
               interspersed. That is, each type of equation
               (rigid-body, elastic, residual-flexibility) must be
               contained in a contiguous group (so that `self.slices`
               is True).
            2. Unlike the normal solver, the `pre_eig` option is not
               available.
            3. The first time step cannot be redone.

        Examples
        --------
        Set up some equations and solve both the normal way and via
        the generator:

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> h = .001                              # time step
        >>> t = np.arange(0, .3001, h)            # time vector
        >>> c = 2*np.pi
        >>> f = np.vstack((3*(1-np.cos(c*2*t)),   # ffn
        ...                4.5*(np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> ts = ode.SolveExp2(m, b, k, h)

        Solve the normal way:

        >>> sol = ts.tsolve(f, static_ic=1)

        Solve via the generator:

        >>> nt = f.shape[1]
        >>> gen, d, v = ts.generator(nt, f[:, 0], static_ic=1)
        >>> for i in range(1, nt):
        ...     # Could do stuff here using d[:, :i] & v[:, :i] to
        ...     # get next force
        ...     gen.send((i, f[:, i]))
        >>> sol2 = ts.finalize()

        Confirm the solutions are the same:

        >>> np.allclose(sol2.a, sol.a)
        True
        >>> np.allclose(sol2.v, sol.v)
        True
        >>> np.allclose(sol2.d, sol.d)
        True
        """
        if not self.slices:
            raise NotImplementedError(
                'generator not yet implemented for the case'
                ' when different types of equations are interspersed'
                ' (eg, a res-flex DOF in the middle of the elastic'
                ' DOFs)')
        d, v, a, force = self._init_dva_part(
            nt, F0, d0, v0, static_ic)
        self._d, self._v, self._a, self._force = d, v, a, force
        generator = self._solve_se2_generator(d, v, F0)
        next(generator)
        return generator, d, v

    def _solve_se2_generator(self, d, v, F0):
        """Generator solver for :class:`SolveExp2`"""
        nt = d.shape[1]
        if nt == 1:
            yield
        Force = self._force
        unc = self.unc
        rfsize = self.rfsize
        if self.rfsize:
            rf = self.rf
            ikrf = self.ikrf
            if unc:
                ikrf = ikrf.ravel()
            else:
                ikrf = la.lu_solve(ikrf, np.eye(rfsize),
                                   check_finite=False)
            drf = d[rf]

        ksize = self.ksize
        if not ksize:
            # only rf modes
            i, F1 = yield
            if unc:
                while True:
                    Force[:, i] = F1
                    d[:, i] = ikrf * F1[rf]
                    i, F1 = yield
            else:
                while True:
                    Force[:, i] = F1
                    d[:, i] = ikrf @ F1[rf]
                    i, F1 = yield

        # there are rb/el modes if here
        kdof = self.kdof
        P = self.P
        Q = self.Q
        order = self.order
        if self.m is not None:
            if unc:
                invm = self.invm.ravel()
                P = P * invm
                if order == 1:
                    Q = Q * invm
            else:
                # P @ invm = (invm.T @ P.T).T
                P = la.lu_solve(self.invm, P.T, trans=1,
                                check_finite=False).T
                if order == 1:
                    Q = la.lu_solve(self.invm, Q.T, trans=1,
                                    check_finite=False).T
        E_dd = self.E_dd
        E_dv = self.E_dv
        E_vd = self.E_vd
        E_vv = self.E_vv
        i, F1 = yield
        if rfsize:
            # both rf and non-rf modes present
            D = d[kdof]
            V = v[kdof]
            drf = d[rf]
            while True:
                Force[:, i] = F1
                F0 = Force[:, i-1]
                if self.order == 1:
                    PQF = P @ F0[kdof] + Q @ F1[kdof]
                else:
                    PQF = P @ F0[kdof]
                d0 = D[:, i-1]
                v0 = V[:, i-1]
                D[:, i] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:]
                V[:, i] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize]
                if unc:
                    drf[:, i] = ikrf * F1[rf]
                else:
                    drf[:, i] = ikrf @ F1[rf]
                i, F1 = yield
        else:
            # only non-rf modes present
            while True:
                Force[:, i] = F1
                F0 = Force[:, i-1]
                if self.order == 1:
                    PQF = P @ F0 + Q @ F1
                else:
                    PQF = P @ F0
                d0 = d[:, i-1]
                v0 = v[:, i-1]
                d[:, i] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:]
                v[:, i] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize]
                i, F1 = yield


class SolveUnc(_BaseODE):
    r"""
    2nd order ODE time and frequency domain solvers for "uncoupled"
    equations of motion

    This class is for solving:

    .. math::
        M \ddot{q} + B \dot{q} + K q = F

    Note that the mass, damping and stiffness can be fully populated
    (coupled).

    Like :class:`SolveExp2`, this solver is exact assuming piece-wise
    linear forces (if `order` is 1) or piece-wise constant forces (if
    `order` is 0).

    For uncoupled equations, pre-formulated integration coefficients
    are used (see :func:`get_su_coef`).

    For coupled systems, the elastic modes part of the equation is
    transformed into state-space:

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} M^{-1} B & M^{-1} K \\ I & 0 \end{array}
        \right] \left\{
            \begin{array}{c} \dot{q} \\ q \end{array}
        \right\} = \left\{
            \begin{array}{c} M^{-1} F \\ 0 \end{array} \right\}

    or:

    .. math::
        \dot{y} - A y = w

    The above state-space equations are for the dynamic elastic modes
    only. Then, the complex eigensolution is used to decouple the
    equations. The rigid-body and residual-flexibility modes are
    solved independently. Note that the res-flex modes are solved
    statically and any initial conditions are ignored for them.

    For a static solution:

        - rigid-body displacements = zeros
        - elastic displacments = inv(k[elastic]) * F[elastic]
        - velocity = zeros
        - rigid-body accelerations = inv(m[rigid]) * F[rigid]
        - elastic accelerations = zeros

    See also
    --------
    :class:`SolveExp2`.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> h = .001                              # time step
        >>> t = np.arange(0, .3001, h)            # time vector
        >>> c = 2*np.pi
        >>> f = np.vstack((3*(1-np.cos(c*2*t)),   # ffn
        ...                4.5*(np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> ts = ode.SolveUnc(m, b, k, h)
        >>> sol = ts.tsolve(f, static_ic=1)

        Solve with scipy.signal.lsim for comparison:

        >>> A = ode.make_A(m, b, k)
        >>> n = len(m)
        >>> Z = np.zeros((n, n), float)
        >>> B = np.vstack((np.eye(n), Z))
        >>> C = np.vstack((A, np.hstack((Z, np.eye(n)))))
        >>> D = np.vstack((B, Z))
        >>> ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
        >>> import scipy.signal
        >>> f2 = (1./m).reshape(-1, 1) * f
        >>> tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T, t,
        ...                                X0=ic)
        >>>
        >>> print('acce cmp:', np.allclose(yl[:, :n], sol.a.T))
        acce cmp: True
        >>> print('velo cmp:', np.allclose(yl[:, n:2*n], sol.v.T))
        velo cmp: True
        >>> print('disp cmp:', np.allclose(yl[:, 2*n:], sol.d.T))
        disp cmp: True

        Plot the four accelerations:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('SolveUnc vs. lsim', figsize=[8, 8])
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, sol.a[j], label='SolveUnc')
        ...     _ = plt.plot(tl, yl[:, j], label='scipy lsim')
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Time (s)')
        ...     if j == 0:
        ...         _ = plt.legend(loc='best')
        >>> plt.tight_layout()
    """

    def __init__(self, m, b, k, h=None, rb=None, rf=None, order=1,
                 pre_eig=False):
        """
        Instantiates a :class:`SolveUnc` solver.

        Parameters
        ----------
        m : 1d or 2d ndarray or None
            Mass; vector (of diagonal), or full; if None, mass is
            assumed identity
        b : 1d or 2d ndarray
            Damping; vector (of diagonal), or full
        k : 1d or 2d ndarray
            Stiffness; vector (of diagonal), or full
        h : scalar or None; optional
            Time step; can be None if only want to solve a static
            problem or if only solving frequency domain problems
        rb : 1d array or None; optional
            Index partition vector for rigid-body modes. If None, the
            rigid-body modes will be automatically detected by this
            logic::

                rb = np.nonzero(abs(k) < .005)[0]  # for diagonal k
                rb = np.nonzero(abs(k).max(0) < .005)[0]  # for full k

            Set to [] to specify no rigid-body modes. Note: the
            detection of rigid-body modes is done after the `pre_eig`
            operation, if that is True.
        rf : 1d array or None; optional
            Index partition vector for res-flex modes; these will be
            solved statically
        order : integer; optional
            Specify which solver to use:

            - 0 for the zero order hold (force stays constant across
              time step)
            - 1 for the 1st order hold (force can vary linearly across
              time step)

        pre_eig : bool; optional
            If True, an eigensolution will be computed with the mass
            and stiffness matrices to convert the system to modal
            space. This will allow the automatic detection of
            rigid-body modes which is necessary for the complex
            eigenvalue method to work properly on systems with
            rigid-body modes. No modes are truncated. Only works if
            stiffness is symmetric/hermitian and mass is positive
            definite (see :func:`scipy.linalg.eigh`). Just leave it as
            False if the equations are already in modal space.

        Notes
        -----
        The instance is populated with some or all of the following
        members. Note that in the table, `non-rf/elastic` means
        `non-rf` for uncoupled systems, `elastic` for coupled -- the
        difference is whether or not the rigid-body modes are
        included: they are for uncoupled.

        =========  ===================================================
        Member     Description
        =========  ===================================================
        m          mass for the non-rf/elastic modes
        b          damping for the non-rf/elastic modes
        k          stiffness for the non-rf/elastic modes
        h          time step
        rb         index vector or slice for the rb modes
        el         index vector or slice for the el modes
        rf         index vector or slice for the rf modes
        _rb        index vector or slice for the rb modes relative to
                   the non-rf modes
        _el        index vector or slice for the el modes relative to
                   the non-rf modes
        nonrf      index vector or slice for the non-rf modes
        kdof       index vector or slice for the non-rf/elastic modes
        n          number of total DOF
        rbsize     number of rb modes
        elsize     number of el modes
        rfsize     number of rf modes
        nonrfsz    number of non-rf modes
        ksize      number of non-rf/elastic modes
        invm       decomposition of m for the non-rf/elastic modes
        imrb       decomposition of m for the rb modes
        krf        stiffness for the rf modes
        ikrf       inverse of stiffness for the rf modes
        unc        True if there are no off-diagonal terms in any
                   matrix; False otherwise
        order      order of solver (0 or 1; see above)
        pc         None or record (SimpleNamespace) of integration
                   coefficients; if uncoupled, this is populated by
                   :func:`get_su_coef`; otherwise by
                   :func:`SolveUnc._get_su_eig`
        pre_eig    True if the "pre" eigensolution was done; False
                   otherwise
        phi        the mode shape matrix from the "pre" eigensolution;
                   only present if `pre_eig` is True
        systype    float or complex; determined by `m` `b` `k`
        =========  ===================================================

        Unlike for :class:`SolveExp2`, `order` is not used until the
        solver is called. In other words, this routine prepares the
        integration coefficients for a first order hold no matter what
        the setting of `order` is, but the solver will adjust the use
        of the forces to account for the `order` setting.

        The mass, damping and stiffness may be real or complex.
        """
        self._common_precalcs(m, b, k, h, rb, rf, pre_eig)
        if self.ksize:
            if self.unc and self.systype is float:
                self._inv_m()
                self.pc = get_su_coef(self.m, self.b, self.k,
                                      h, self.rb)
            else:
                self.pc = self._get_su_eig(h is not None)
        else:
            self.pc = None
        self._mk_slices(True)
        self.order = order

    def tsolve(self, force, d0=None, v0=None, static_ic=False):
        """
        Solve time-domain 2nd order ODE equations

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x time
        d0 : 1d ndarray; optional
            Displacement initial conditions; if None, zero ic's are
            used.
        v0 : 1d ndarray; optional
            Velocity initial conditions; if None, zero ic's are used.
        static_ic : bool; optional
            If True and `d0` is None, then `d0` is calculated such
            that static (steady-state) initial conditions are
            used. Uses the pseudo-inverse in case there are rigid-body
            modes. `static_ic` is ignored if `d0` is not None.

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x time
        v : 2d ndarray
            Velocity; ndof x time
        a : 2d ndarray
            Acceleration; ndof x time
        h : scalar
            Time-step
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        """
        force = np.atleast_2d(force)
        d, v, a, force = self._init_dva(force, d0, v0,
                                        static_ic)
        if self.nonrfsz:
            if self.unc and self.systype is float:
                # for uncoupled, m, b, k have rb+el (all nonrf)
                self._solve_real_unc(d, v, force)
            else:
                # for coupled, m, b, k are only el only
                self._solve_complex_unc(d, v, a, force)
        self._calc_acce_kdof(d, v, a, force)
        return self._solution(d, v, a)

    def generator(self, nt, F0, d0=None, v0=None, static_ic=False):
        """
        Python "generator" version of :func:`SolveUnc.tsolve`;
        interactively solve (or re-solve) one step at a time.

        Parameters
        ----------
        nt : integer
            Number of time steps
        F0 : 1d ndarray
            Initial force vector
        d0 : 1d ndarray or None; optional
            Displacement initial conditions; if None, zero ic's are
            used.
        v0 : 1d ndarray or None; optional
            Velocity initial conditions; if None, zero ic's are used.
        static_ic : bool; optional
            If True and `d0` is None, then `d0` is calculated such
            that static (steady-state) initial conditions are
            used. Uses the pseudo-inverse in case there are rigid-body
            modes. `static_ic` is ignored if `d0` is not None.

        Returns
        -------
        gen : generator function
            Generator function for solving equations one step at a
            time
        d, v, a : 2d ndarrays
            The displacement, velocity and acceleration arrays. Only
            the first column of `d` and `v` are set; other values are
            all zero.

        Notes
        -----
        To use the generator:

            1. Instantiate a :class:`SolveUnc` instance::

                   ts = SolveUnc(m, b, k, h)

            2. Retrieve a generator and the arrays for holding the
               solution::

                   gen, d, v = ts.generator(len(time), f0)

            3. Use :func:`gen.send` to send a tuple of the next index
               and corresponding force vector in a loop. Re-do
               time-steps as necessary (note that index zero cannot be
               redone)::

                   for i in range(1, len(time)):
                       # Do whatever to get i'th force
                       # - note: d[:, :i] and v[:, :i] are available
                       gen.send((i, fi))

               The class instance will have the attributes `_d`, `_v`
               (same objects as `d` and `v`), `_a`, and `_force`. `d`,
               `v` and `ts._force` are updated on every
               :func:`gen.send`. (`ts._a` is not used until step 4.)

            4. Call :func:`ts.finalize` to get final solution
               in standard form::

                   sol = ts.finalize()

               The internal references `_d`, `_v`, `_a`, and `_force`
               are deleted.

        The generator solver currently has these limitations:

            1. Unlike the normal solver, equations cannot be
               interspersed. That is, each type of equation
               (rigid-body, elastic, residual-flexibility) must be
               contained in a contiguous group (so that `self.slices`
               is True).
            2. Unlike the normal solver, the `pre_eig` option is not
               available.
            3. The first time step cannot be redone.

        Examples
        --------
        Set up some equations and solve both the normal way and via
        the generator:

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> h = .001                              # time step
        >>> t = np.arange(0, .3001, h)            # time vector
        >>> c = 2*np.pi
        >>> f = np.vstack((3*(1-np.cos(c*2*t)),   # ffn
        ...                4.5*(np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> ts = ode.SolveUnc(m, b, k, h)

        Solve the normal way:

        >>> sol = ts.tsolve(f, static_ic=1)

        Solve via the generator:

        >>> nt = f.shape[1]
        >>> gen, d, v = ts.generator(nt, f[:, 0], static_ic=1)
        >>> for i in range(1, nt):
        ...     # Could do stuff here using d[:, :i] & v[:, :i] to
        ...     # get next force
        ...     gen.send((i, f[:, i]))
        >>> sol2 = ts.finalize()

        Confirm results:

        >>> np.allclose(sol2.a, sol.a)
        True
        >>> np.allclose(sol2.v, sol.v)
        True
        >>> np.allclose(sol2.d, sol.d)
        True
        """
        if not self.slices:
            raise NotImplementedError(
                'generator not yet implemented for the case when'
                ' different types of equations are interspersed (eg,'
                ' a res-flex DOF in the middle of the elastic DOFs)')
        d, v, a, force = self._init_dva_part(
            nt, F0, d0, v0, static_ic)
        self._d, self._v, self._a, self._force = d, v, a, force
        if self.unc and self.systype is float:
            # for uncoupled, m, b, k have rb+el (all nonrf)
            generator = self._solve_real_unc_generator(d, v, F0)
            next(generator)
            return generator, d, v
        else:
            # for coupled, m, b, k are only el only
            generator = self._solve_complex_unc_generator(d, v, a, F0)
            next(generator)
            return generator, d, v

    def fsolve(self, force, freq, incrb=2):
        """
        Solve frequency-domain modal equations of motion using
        uncoupled equations.

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x freq
        freq : 1d ndarray
            Frequency vector in Hz; solution will be computed at all
            frequencies in `freq`
        incrb : 0, 1, or 2; optional
            Specifies how to handle rigid-body responses:

            ======  ==============================================
            incrb   description
            ======  ==============================================
               0    no rigid-body is included
               1    acceleration and velocity rigid-body only
               2    all of rigid-body is included (see note below)
            ======  ==============================================

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x freq
        v : 2d ndarray
            Velocity; ndof x freq
        a : 2d ndarray
            Acceleration; ndof x freq
        f : 1d ndarray
            Frequency vector (same as the input `freq`)

        Notes
        -----
        The rigid-body and residual-flexibility modes are solved
        independently. The res-flex modes are solved statically.

        Rigid-body velocities and displacements are undefined where
        `freq` is zero. So, if `incrb` is 1 or 2, this routine just
        sets these responses to zero.

        See also
        --------
        :class:`FreqDirect`

        Examples
        --------
        .. plot::
            :context: close-figs

            >>> from pyyeti import ode
            >>> import numpy as np
            >>> m = np.array([10., 30., 30., 30.])    # diag mass
            >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
            >>> zeta = np.array([0., .05, 1., 2.])    # % damping
            >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
            >>> freq = np.arange(0, 35, .1)           # frequency
            >>> f = 100*np.ones((4, freq.size))       # constant ffn
            >>> ts = ode.SolveUnc(m, b, k)
            >>> sol = ts.fsolve(f, freq)

            Solve @ 25 Hz by hand for comparison:

            >>> w = 2*np.pi*25
            >>> i = np.argmin(abs(freq-25))
            >>> H = -w**2*m + 1j*w*b + k
            >>> disp = f[:, i] / H
            >>> velo = 1j*w*disp
            >>> acce = -w**2*disp
            >>> np.allclose(disp, sol.d[:, i])
            True
            >>> np.allclose(velo, sol.v[:, i])
            True
            >>> np.allclose(acce, sol.a[:, i])
            True

            Plot the four accelerations:

            >>> import matplotlib.pyplot as plt
            >>> fig = plt.figure(figsize=[8, 8])
            >>> labels = ['Rigid-body', 'Underdamped',
            ...           'Critically Damped', 'Overdamped']
            >>> for j, name in zip(range(4), labels):
            ...     _ = plt.subplot(4, 1, j+1)
            ...     _ = plt.plot(freq, abs(sol.a[j]))
            ...     _ = plt.title(name)
            ...     _ = plt.ylabel('Acceleration')
            ...     _ = plt.xlabel('Frequency (Hz)')
            >>> plt.tight_layout()
        """
        force = np.atleast_2d(force)
        d, v, a, force = self._init_dva(
            force, None, None, False, istime=False)
        freq = np.atleast_1d(freq)
        self._force_freq_compat_chk(force, freq)
        if self.nonrfsz:
            if self.unc:
                # for uncoupled, m, b, k have rb+el (all nonrf)
                self._solve_freq_unc(d, v, a, force, freq, incrb)
            else:
                # for coupled, m, b, k are only el only
                self._solve_freq_coup(d, v, a, force, freq, incrb)
        return self._solution_freq(d, v, a, freq)

    def _solve_real_unc(self, d, v, force):
        """Solve the real uncoupled equations for :class:`SolveUnc`"""
        # solve:
        # for i in range(nt-1):
        #     D[:,i+1] = F *D[:, i] + G *V[:, i] +
        #                A *force[:, i] + B *force[:, i+1]
        #     V[:,i+1] = Fp*D[:, i] + Gp*V[:, i] +
        #                Ap*force[:, i] + Bp*force[:, i+1]
        nt = force.shape[1]
        if nt == 1:
            return
        pc = self.pc
        kdof = self.kdof
        F = pc.F
        G = pc.G
        A = pc.A
        B = pc.B
        Fp = pc.Fp
        Gp = pc.Gp
        Ap = pc.Ap
        Bp = pc.Bp
        D = d[kdof]
        V = v[kdof]
        if self.order == 1:
            ABF = (A[:, None]*force[kdof, :-1] +
                   B[:, None]*force[kdof, 1:])
            ABFp = (Ap[:, None]*force[kdof, :-1] +
                    Bp[:, None]*force[kdof, 1:])
        else:
            ABF = (A+B)[:, None]*force[kdof, :-1]
            ABFp = (Ap+Bp)[:, None]*force[kdof, :-1]
        di = D[:, 0]
        vi = V[:, 0]
        for i in range(nt-1):
            din = F*di + G*vi + ABF[:, i]
            vi = V[:, i+1] = Fp*di + Gp*vi + ABFp[:, i]
            D[:, i+1] = di = din
        if not self.slices:
            d[kdof] = D
            v[kdof] = V

    def _solve_real_unc_generator(self, d, v, F0):
        """Solve the real uncoupled equations for :class:`SolveUnc`"""
        # solve:
        # for i in range(nt-1):
        #     D[:,i+1] = F *D[:, i] + G *V[:, i] +
        #                A *force[:, i] + B *force[:, i+1]
        #     V[:,i+1] = Fp*D[:, i] + Gp*V[:, i] +
        #                Ap*force[:, i] + Bp*force[:, i+1]
        nt = d.shape[1]
        if nt == 1:
            yield
        Force = self._force

        if self.rfsize:
            rf = self.rf
            ikrf = self.ikrf.ravel()

        if not self.ksize:
            i, F1 = yield
            while True:
                Force[:, i] = F1
                d[:, i] = ikrf * F1[rf]
                i, F1 = yield

        # there are rb/el modes if here
        pc = self.pc
        kdof = self.kdof
        F = pc.F
        G = pc.G
        A = pc.A
        B = pc.B
        Fp = pc.Fp
        Gp = pc.Gp
        Ap = pc.Ap
        Bp = pc.Bp

        i, F1 = yield
        if self.order == 1:
            if self.rfsize:
                # rigid-body and elastic equations:
                D = d[kdof]
                V = v[kdof]
                # resflex equations:
                drf = d[rf]
                # for i in range(nt-1):
                while True:
                    Force[:, i] = F1
                    # rb + el:
                    F0k = Force[kdof, i-1]
                    F1k = F1[kdof]
                    di = D[:, i-1]
                    vi = V[:, i-1]
                    D[:, i] = F*di + G*vi + A*F0k + B*F1k
                    V[:, i] = Fp*di + Gp*vi + Ap*F0k + Bp*F1k

                    # rf:
                    drf[:, i] = ikrf * F1[rf]
                    i, F1 = yield
            else:
                # only rigid-body and elastic equations:
                while True:
                    Force[:, i] = F1
                    # rb + el:
                    F0 = Force[:, i-1]
                    di = d[:, i-1]
                    vi = v[:, i-1]
                    d[:, i] = F*di + G*vi + A*F0 + B*F1
                    v[:, i] = Fp*di + Gp*vi + Ap*F0 + Bp*F1
                    i, F1 = yield
        else:
            # order == 0
            AB = A + B
            ABp = Ap + Bp
            if self.rfsize:
                # rigid-body and elastic equations:
                D = d[kdof]
                V = v[kdof]
                # resflex equations:
                drf = d[rf]
                # for i in range(nt-1):
                while True:
                    Force[:, i] = F1
                    # rb + el:
                    F0k = Force[kdof, i-1]
                    di = D[:, i-1]
                    vi = V[:, i-1]
                    D[:, i] = F*di + G*vi + AB*F0k
                    V[:, i] = Fp*di + Gp*vi + ABp*F0k

                    # rf:
                    drf[:, i] = ikrf * F1[rf]
                    i, F1 = yield
            else:
                # only rigid-body and elastic equations:
                while True:
                    Force[:, i] = F1
                    # rb + el:
                    F0 = Force[:, i-1]
                    di = d[:, i-1]
                    vi = v[:, i-1]
                    d[:, i] = F*di + G*vi + AB*F0
                    v[:, i] = Fp*di + Gp*vi + ABp*F0
                    i, F1 = yield

    def _solve_complex_unc(self, d, v, a, force):
        """Solve the complex uncoupled equations for
        :class:`SolveUnc`"""
        nt = force.shape[1]
        pc = self.pc
        if self.rbsize:
            # solve:
            # for i in range(nt-1):
            #     drb[:, i+1] = drb[:, i] + G*vrb[:, i] +
            #                   A*(rbforce[:, i] + rbforce[:, i+1]/2)
            #     vrb[:, i+1] = vrb[:, i] + Ap*(rbforce[:, i] +
            #                                   rbforce[:, i+1])
            rb = self.rb
            if self.m is not None:
                if self.unc:
                    rbforce = self.imrb * force[rb]
                else:
                    rbforce = la.lu_solve(self.imrb, force[rb],
                                          check_finite=False)
            else:
                rbforce = force[rb]
            if nt > 1:
                G = pc.G
                A = pc.A
                Ap = pc.Ap
                if self.order == 1:
                    AF = A*(rbforce[:, :-1] + rbforce[:, 1:]/2)
                    AFp = Ap*(rbforce[:, :-1] + rbforce[:, 1:])
                else:
                    AF = (1.5*A)*rbforce[:, :-1]
                    AFp = (2*Ap)*rbforce[:, :-1]
                drb = d[rb]
                vrb = v[rb]
                di = drb[:, 0]
                vi = vrb[:, 0]
                for i in range(nt-1):
                    di = drb[:, i+1] = di + G*vi + AF[:, i]
                    vi = vrb[:, i+1] = vi + AFp[:, i]
                if not self.slices:
                    d[rb] = drb
                    v[rb] = vrb
            a[rb] = rbforce

        if self.ksize and nt > 1:
            self._delconj()
            # solve:
            # for i in range(nt-1):
            #     u[:, i+1] = Fe*u[:, i] + Ae*w[:, i] + Be*w[:, i+1]
            Fe = pc.Fe
            Ae = pc.Ae
            Be = pc.Be
            ur_d = pc.ur_d
            ur_v = pc.ur_v
            rur_d = pc.rur_d
            iur_d = pc.iur_d
            rur_v = pc.rur_v
            iur_v = pc.iur_v
            ur_inv_d = pc.ur_inv_d
            ur_inv_v = pc.ur_inv_v

            kdof = self.kdof
            if self.m is not None:
                if self.unc:
                    imf = self.invm * force[kdof]
                else:
                    imf = la.lu_solve(self.invm, force[kdof],
                                      check_finite=False)
            else:
                imf = force[kdof]
            w = ur_inv_v @ imf
            if self.order == 1:
                ABF = (Ae[:, None]*w[:, :-1] +
                       Be[:, None]*w[:, 1:])
            else:
                ABF = (Ae+Be)[:, None]*w[:, :-1]

            y = np.empty((ur_inv_v.shape[0], nt), complex)
            di = y[:, 0] = (ur_inv_v @ v[kdof, 0] +
                            ur_inv_d @ d[kdof, 0])
            for i in range(nt-1):
                di = y[:, i+1] = Fe*di + ABF[:, i]
            if self.systype is float:
                # Can do real math for recovery. Note that the
                # imaginary part of 'd' and 'v' would be zero if no
                # modes were deleted of the complex conjugate pairs.
                # The real part is correct however, and that's all we
                # need.
                ry = y[:, 1:].real.copy()
                iy = y[:, 1:].imag.copy()
                d[kdof, 1:] = rur_d @ ry - iur_d @ iy
                v[kdof, 1:] = rur_v @ ry - iur_v @ iy
            else:
                d[kdof, 1:] = ur_d @ y[:, 1:]
                v[kdof, 1:] = ur_v @ y[:, 1:]

    def _solve_complex_unc_generator(self, d, v, a, F0):
        """Solve the complex uncoupled equations for
        :class:`SolveUnc`"""
        nt = d.shape[1]

        # need to handle up to 3 types of equations every loop:
        # - rb, el, rf
        unc = self.unc
        rbsize = self.rbsize
        m = self.m
        if rbsize:
            rb = self.rb
            if m is not None:
                imrb = self.imrb
                if unc:
                    imrb = imrb.ravel()
                    rbforce = imrb * F0[rb]
                else:
                    imrb = la.lu_solve(imrb, np.eye(rbsize),
                                       check_finite=False)
                    rbforce = imrb @ F0[rb]
            else:
                rbforce = F0[rb]
            a[rb, 0] = rbforce

        if nt == 1:
            yield

        pc = self.pc
        if rbsize:
            G = pc.G
            A = pc.A
            Ap = pc.Ap
            drb = d[rb]
            vrb = v[rb]
            arb = a[rb]

        Force = self._force
        ksize = self.ksize
        rfsize = self.rfsize
        systype = self.systype

        if ksize:
            self._delconj()
            Fe = pc.Fe
            Ae = pc.Ae
            Be = pc.Be
            ur_d = pc.ur_d
            ur_v = pc.ur_v
            rur_d = pc.rur_d
            iur_d = pc.iur_d
            rur_v = pc.rur_v
            iur_v = pc.iur_v
            ur_inv_v = pc.ur_inv_v
            ur_inv_d = pc.ur_inv_d

            kdof = self.kdof
            if m is not None:
                invm = self.invm
                if self.unc:
                    invm = invm.ravel()
                else:
                    invm = la.lu_solve(invm, np.eye(ksize),
                                       check_finite=False)
            D = d[kdof]
            V = v[kdof]

        if rfsize:
            rf = self.rf
            ikrf = self.ikrf
            if unc:
                ikrf = ikrf.ravel()
            else:
                ikrf = la.lu_solve(ikrf, np.eye(rfsize),
                                   check_finite=False)
            drf = d[rf]

        order = self.order
        i, F1 = yield
        while True:
            Force[:, i] = F1
            F0 = Force[:, i-1]
            if rbsize:
                if m is not None:
                    if unc:
                        F0rb = imrb * F0[rb]
                        F1rb = imrb * F1[rb]
                    else:
                        F0rb = imrb @ F0[rb]
                        F1rb = imrb @ F1[rb]
                else:
                    F0rb = F0[rb]
                    F1rb = F1[rb]
                if order == 1:
                    AF = A*(F0rb + 0.5*F1rb)
                    AFp = Ap*(F0rb + F1rb)
                else:
                    AF = (1.5*A)*F0rb
                    AFp = (2.0*Ap)*F0rb
                vi = vrb[:, i-1]
                drb[:, i] = drb[:, i-1] + G*vi + AF
                vrb[:, i] = vi + AFp
                arb[:, i] = F1rb

            if ksize:
                F0k = Force[kdof, i-1]
                F1k = F1[kdof]
                if m is not None:
                    if unc:
                        F0k = invm * F0k
                        F1k = invm * F1k
                    else:
                        F0k = invm @ F0k
                        F1k = invm @ F1k
                w0 = ur_inv_v @ F0k
                w1 = ur_inv_v @ F1k
                if self.order == 1:
                    ABF = Ae*w0 + Be*w1
                else:
                    ABF = (Ae+Be)*w0
                # [V; D] = ur @ y
                # y = ur_inv @ [V; D] = [ur_inv_v, ur_inv_d] @ [V; D]
                y = ur_inv_v @ V[:, i-1] + ur_inv_d @ D[:, i-1]
                yn = Fe * y + ABF
                if systype is float:
                    # Can do real math for recovery. Note that the
                    # imaginary part of 'd' and 'v' would be zero if
                    # no modes were deleted of the complex conjugate
                    # pairs. The real part is correct however, and
                    # that's all we need.
                    ry = yn.real
                    iy = yn.imag
                    D[:, i] = rur_d @ ry - iur_d @ iy
                    V[:, i] = rur_v @ ry - iur_v @ iy
                else:
                    # [V; D] = ur @ y
                    D[:, i] = ur_d @ yn
                    V[:, i] = ur_v @ yn
                    # ry = yn.real
                    # iy = yn.imag
                    # D[:, i].real = rur_d @ ry - iur_d @ iy
                    # D[:, i].imag = rur_d @ iy + iur_d @ ry
                    # V[:, i].real = rur_v @ ry - iur_v @ iy
                    # V[:, i].imag = rur_v @ iy + iur_v @ ry

            if rfsize:
                if unc:
                    drf[:, i] = ikrf * F1[rf]
                else:
                    drf[:, i] = ikrf @ F1[rf]
            i, F1 = yield

    def _get_su_eig(self, delcc):
        """
        Does pre-calcs for the `SolveUnc` solver via the complex
        eigenvalue approach.

        Parameters
        ----------
        None, but class is expected to be populated with:

            m : 1d or 2d ndarray or None
                Mass; vector (of diagonal), or full; if None, mass is
                assumed identity. Has only rigid-body b and elastic
                modes.
            b : 1d or 2d ndarray
                Damping; vector (of diagonal), or full. Has only
                rigid-body b and elastic modes.
            k : 1d or 2d ndarray
                Stiffness; vector (of diagonal), or full. Has only
                rigid-body b and elastic modes.
            h : scalar or None
                Time step; can be None if just solving static case.
            rb : 1d array or None
                Index vector for the rigid-body modes; None for no
                rigid-body modes.
            el : 1d array or None
                Index vector for the elastic modes; None for no
                elastic modes.

        Returns
        -------
        A record (SimpleNamespace) containing:

        G, A, Ap, Fe, Ae, Be: 1d ndarrays
            The integration coefficients. ``G, A, Ap`` are for the
            rigid-body equations and ``Fe, Ae, Be`` are for the
            elastic equations. These will only be present if `h` is
            not None.
        lam : 1d ndarray
            The complex eigenvalues.
        ur : 2d ndarray
            The complex right-eigenvectors
        ur_d, ur_v : 2d ndarrays
            Partitions of `ur`
        ur_inv_v, ur_inv_d : 2d ndarrays
            Partitions of ``inv(ur)``
        rur_d, iur_d : 2d ndarrays
            Real and imaginary parts of `ur_d`
        rur_v, iur_v : 2d ndarrays
            Real and imaginary parts of `ur_v`
        invm : 2d ndarray or None
            Decomposition of the elastic part of the mass matrix; None
            if `self.m` is None (identity mass)
        imrb : 2d ndarray or None
            LU decomposition of rigid-body part of mass; or None if
            `m` is None.

        Notes
        -----
        The members `m`, `b`, and `k` are partitioned down to the
        elastic part only.

        See also
        --------
        :class:`SolveUnc`
        """
        msg = ('Repeated roots detected and equations do not appear '
               'to be diagonalized. Generally, this is a failure '
               'condition. For time domain problems, the routine '
               ':class:`SolveExp2` will probably work better. For '
               'frequency domain, see :class:`FreqDirect`. '
               'Proceeding, but check results VERY carefully.\n'
               '\tMax off-diag of ``inv(ur) @ ur = {}``\n'
               '\tMax off-diag of ``inv(ur) @ A @ ur = {}``\n'
               '\tMax ``inv(ur) @ A @ ur = {}``')
        pc = SimpleNamespace()
        h = self.h
        if self.rbsize:
            self._inv_mrb()
            if h:
                pc.G = h
                pc.A = h*h/3
                pc.Ap = h/2
        if self.unc:
            pv = self._el
        else:
            pv = np.ix_(self._el, self._el)
        if self.m is not None:
            self.m = self.m[pv]
        self.k = self.k[pv]
        self.b = self.b[pv]
        self.kdof = self.nonrf[self._el]
        self.ksize = self.kdof.size
        if self.elsize:
            self._inv_m()
            A = self._build_A()
            lam, ur, ur_inv, dups = eigss(A, delcc)
            if dups.size:
                uu = ur_inv[dups] @ ur[:, dups]
                uau = ur_inv[dups] @ A @ ur[:, dups]
                maxuau = abs(uau).max()
                if (not np.allclose(uu, np.eye(uu.shape[0])) or
                        not ytools.isdiag(uau) or
                        maxuau < 5.e-5):
                    od1 = abs(uu - np.diag(np.diag(uu))).max()
                    od2 = abs(uau - np.diag(np.diag(uau))).max()
                    warnings.warn(msg.format(od1, od2, maxuau),
                                  RuntimeWarning)
            if h:
                self._get_complex_su_coefs(pc, lam, h)
            self._add_partition_copies(pc, lam, ur, ur_inv)
        return pc

    def _get_complex_su_coefs(self, pc, lam, h):
        msg = ('Found {} rigid-body modes in elastic solver section.'
               ' If there are no previous warnings about singular '
               'matrices or repeated roots, solution is probably '
               'valid, but check it before trusting it.')
        # form coefficients for piece-wise exact solution:
        Fe = np.exp(lam*h)
        ilam = 1/lam
        ilamh = (ilam*ilam)/h
        Ae = ilamh + Fe*(ilam - ilamh)
        Be = Fe*ilamh - ilam - ilamh
        abslam = abs(lam)
        # check for rb modes (5.e-5 determined by trial and
        #  error comparing against matrix exponential solver)
        pv = np.nonzero(abslam < 5.e-5)[0]
        if pv.size:
            if pv.size > 1:
                warnings.warn(msg.format(len(pv)), RuntimeWarning)
            Fe[pv] = 1.
            Ae[pv] = h/2.
            Be[pv] = h/2.
        pc.Fe = Fe
        pc.Ae = Ae
        pc.Be = Be

    def _add_partition_copies(self, pc, lam, ur, ur_inv):
        ksize = self.ksize
        pc.lam = lam
        pc.ur = ur
        pc.ur_d = pc.ur[ksize:]
        pc.ur_v = pc.ur[:ksize]
        pc.ur_inv_v = ur_inv[:, :ksize].copy()
        pc.ur_inv_d = ur_inv[:, ksize:].copy()
        pc.rur_d = pc.ur_d.real.copy()
        pc.iur_d = pc.ur_d.imag.copy()
        pc.rur_v = pc.ur_v.real.copy()
        pc.iur_v = pc.ur_v.imag.copy()

    def _addconj(self):
        pc = self.pc
        if 2*pc.ur_inv_v.shape[1] > pc.ur_d.shape[1]:
            ur_inv = np.hstack((pc.ur_inv_v, pc.ur_inv_d))
            lam, ur, ur_inv = addconj(pc.lam, pc.ur, ur_inv)
            if self.h:
                self._get_complex_su_coefs(pc, lam, self.h)
            self._add_partition_copies(pc, lam, ur, ur_inv)

    def _delconj(self):
        pc = self.pc
        if 2*pc.ur_inv_v.shape[1] == pc.ur_d.shape[1]:
            ur_inv = np.hstack((pc.ur_inv_v, pc.ur_inv_d))
            lam, ur, ur_inv, _ = delconj(pc.lam, pc.ur, ur_inv, [])
            if self.h:
                self._get_complex_su_coefs(pc, lam, self.h)
            self._add_partition_copies(pc, lam, ur, ur_inv)

    def _solve_freq_rb(self, d, v, a, force, freqw, freqw2, incrb,
                       unc):
        """Solve the rigid-body equations for
        :func:`SolveUnc.fsolve`"""
        if self.rbsize and incrb:
            rb = self.rb
            if self.m is not None:
                if unc:
                    a[rb] = self.invm[self._rb] * force[rb]
                else:
                    a[rb] = la.lu_solve(self.imrb, force[rb],
                                        check_finite=False)
            else:
                a[rb] = force[rb]
            pvnz = freqw != 0
            v[rb, pvnz] = (-1j/freqw[pvnz]) * a[rb, pvnz]
            if incrb == 2:
                d[rb, pvnz] = (-1./freqw2[pvnz]) * a[rb, pvnz]

    def _solve_freq_unc(self, d, v, a, force, freq, incrb):
        """Solve the uncoupled equations for
        :func:`SolveUnc.fsolve`"""
        # convert frequency in Hz to radian/sec:
        freqw = 2*np.pi*freq
        freqw2 = freqw**2

        # solve rigid-body and elastic parts separately
        # - res-flex part was already solved in _init_dva

        # solve rigid-body part:
        self._solve_freq_rb(d, v, a, force, freqw, freqw2, incrb,
                            True)

        # solve elastic part:
        if self.elsize:
            el = self.el
            _el = self._el
            fw = freqw[None, :]
            fw2 = freqw2[None, :]
            if self.m is None:
                d[el] = force[el] / (1j*(self.b[_el][:, None] @ fw) +
                                     self.k[_el][:, None] - fw2)
            else:
                d[el] = force[el] / (1j*(self.b[_el][:, None] @ fw) +
                                     self.k[_el][:, None] -
                                     self.m[_el][:, None] @ fw2)
            a[el] = d[el] * -(freqw2)
            v[el] = d[el] * (1j*freqw)

    def _solve_freq_coup(self, d, v, a, force, freq, incrb):
        """Solve the coupled equations for :func:`SolveUnc.fsolve`"""
        # convert frequency in Hz to radian/sec:
        freqw = 2*np.pi*freq
        freqw2 = freqw**2

        # solve rigid-body and elastic parts separately
        # - res-flex part was already solved in _init_dva

        # solve rigid-body part:
        self._solve_freq_rb(d, v, a, force, freqw, freqw2, incrb,
                            False)

        # solve elastic part:
        if self.ksize:
            self._addconj()
            pc = self.pc
            kdof = self.kdof
            # form complex state-space generalized force:
            if self.m is not None:
                imf = la.lu_solve(self.invm, force[kdof],
                                  check_finite=False)
            else:
                imf = force[kdof]
            w = pc.ur_inv_v @ imf
            n = w.shape[0]
            H = (np.ones((n, 1)) @ (1.0j*freqw[None, :]) -
                 pc.lam[:, None])
            d[kdof] = pc.ur_d @ (w / H)
            a[kdof] = d[kdof] * -(freqw2)
            v[kdof] = d[kdof] * (1j*freqw)


class FreqDirect(_BaseODE):
    """
    2nd order ODE frequency domain solver

    This class is for solving::

        m xdd + b xd + k x = f

    Notes
    -----
    Each frequency is solved via complex matrix inversion. There
    is no partitioning for rigid-body modes or residual-
    flexibility modes. Note that the solution will be fast if all
    matrices are diagonal.

    Unlike :class:`SolveUnc`, since this routine makes no special provisions
    for rigid-body modes, including 0.0 in `freq` can cause a
    divide-by-zero. It is thereforce recommended to ensure that all
    values in `freq` > 0.0, at least when rigid-body modes are
    present.

    See also
    --------
    :func:`SolveUnc.fsolve`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> freq = np.arange(.1, 35, .1)          # frequency
        >>> f = 100*np.ones((4, freq.size))       # constant ffn
        >>> ts = ode.FreqDirect(m, b, k)
        >>> sol = ts.fsolve(f, freq)

        Solve @ 25 Hz by hand for comparison:

        >>> w = 2*np.pi*25
        >>> i = np.argmin(abs(freq-25))
        >>> H = -w**2*m + 1j*w*b + k
        >>> disp = f[:, i] / H
        >>> velo = 1j*w*disp
        >>> acce = -w**2*disp
        >>> np.allclose(disp, sol.d[:, i])
        True
        >>> np.allclose(velo, sol.v[:, i])
        True
        >>> np.allclose(acce, sol.a[:, i])
        True

        Plot the four accelerations:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure(figsize=[8, 8])
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(freq, abs(sol.a[j]))
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Frequency (Hz)')
        >>> plt.tight_layout()
    """

    def __init__(self, m, b, k, rb=None):
        """
        Instantiates a :class:`FreqDirect` solver.

        Parameters
        ----------
        m : 1d or 2d ndarray or None
            Mass; vector (of diagonal), or full; if None, mass is
            assumed identity
        b : 1d or 2d ndarray
            Damping; vector (of diagonal), or full
        k : 1d or 2d ndarray
            Stiffness; vector (of diagonal), or full
        rb : 1d array or None; optional
            Index partition vector for rigid-body modes. If None, the
            rigid-body modes will be automatically detected by this
            logic::

                rb = np.nonzero(abs(k) < .005)[0]  # for diagonal k
                rb = np.nonzero(abs(k).max(0) < .005)[0]  # for full k

            Set to [] to specify no rigid-body modes.

        Notes
        -----
        The instance is populated with some or all of the following
        members.

        ========   ==================================================
         Member    Description
        ========   ==================================================
         m         mass
         b         damping
         k         stiffness
         h         time step (None)
         rb        index vector or slice for the rb modes
         el        index vector or slice for the el modes
         rf        index vector or slice for the rf modes ([])
         _rb       index vector or slice for the rb modes relative to
                   the non-rf modes
         _el       index vector or slice for the el modes relative to
                   the non-rf modes
         nonrf     index vector or slice for the non-rf modes
         kdof      index vector or slice for the non-rf/elastic modes
         n         number of total DOF
         rfsize    number of rf modes
         nonrfsz   number of non-rf modes
         ksize     number of non-rf/elastic modes
         krf       stiffness for the rf modes
         ikrf      inverse of stiffness for the rf modes
         unc       True if there are no off-diagonal terms in any
                   matrix; False otherwise
         systype   float or complex; determined by `m`, `b`, and `k`
        ========   ==================================================

        The mass, damping and stiffness may be real or complex. This
        routine currently does not accept the `rf` input (if there are
        any, they are treated like all other elastic modes).
        """
        self._common_precalcs(m, b, k, h=None, rb=rb, rf=None)

    def fsolve(self, force, freq, incrb=2):
        """
        Solve equations of motion in frequency domain.

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x freq
        freq : 1d ndarray
            Frequency vector in Hz; solution will be computed at all
            frequencies in `freq`
        incrb : 0, 1, or 2; optional
            Specifies how to handle rigid-body responses:

            ======  ==============================================
            incrb   description
            ======  ==============================================
               0    no rigid-body is included
               1    acceleration and velocity rigid-body only
               2    all of rigid-body is included (see note below)
            ======  ==============================================

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x freq
        v : 2d ndarray
            Velocity; ndof x freq
        a : 2d ndarray
            Acceleration; ndof x freq
        f : 1d ndarray
            Frequency vector (same as the input `freq`)
        """
        force = np.atleast_2d(force)
        d, v, a, force = self._init_dva(force, None, None, False,
                                        istime=False)
        freq = np.atleast_1d(freq)
        self._force_freq_compat_chk(force, freq)
        m, b, k = self.m, self.b, self.k
        if self.unc:
            # equations are uncoupled, solve everything in one step:
            Omega = 2*np.pi*freq[None, :]
            if m is None:
                H = (((1j*b)[:, None] @ Omega +
                      k[:, None]) - Omega**2)
            else:
                H = ((1j*b)[:, None] @ Omega +
                     k[:, None] - m[:, None] @ Omega**2)
            d[:] = force / H
        else:
            # equations are coupled, use a loop:
            Omega = 2*np.pi*freq
            if m is None:
                m = np.eye(self.n)
            for i, O in enumerate(Omega):
                Hi = 1j*b*O + k - m*O**2
                d[:, i] = la.solve(Hi, force[:, i])
        a[:] = -Omega**2 * d
        v[:] = 1j*Omega * d
        if incrb < 2:
            d[self.rb] = 0
            if incrb == 0:
                a[self.rb] = 0
                v[self.rb] = 0
        return self._solution_freq(d, v, a, freq)


def solvepsd(fs, forcepsd, t_frc, freq, drmlist, incrb=2,
             forcephi=None, rbduf=1.0, elduf=1.0):
    """
    Solve equations of motion in frequency domain with uncorrelated PSD
    forces.

    Parameters
    ----------
    fs : class
        An instance of :class:`SolveUnc` or :class:`FreqDirect` (or
        similar ... must have `.fsolve` method)
    forcepsd : 2d array_like
        The matrix of force psds; each row is a force PSD
    t_frc : 2d array_like
        Transformation matrix from system modal DOF to forced DOF;
        ``rows(t_frc) = rows(forcepsd)``
    freq : 1d array_like
        Frequency vector at which solution will be computed;
        ``len(freq) = cols(forcepsd)``
    drmlist : list-like
        List of lists (or similar) of any number of pairs of data
        recovery matrices: [[atm1, dtm1], [atm2, dtm2], ...]. To not
        use a particular drm, set it to None. For example, to perform
        these 3 types of data recovery::

                acce = atm*a
                disp = dtm*d
                loads = ltma*a + dtmd*d

        `drmlist` would be::

              [[atm, None], [dtm, None], [ltma, ltmd]]

    incrb : 0, 1, or 2; optional
        Specifies how to handle rigid-body responses:

        ======  ==============================================
        incrb   description
        ======  ==============================================
           0    no rigid-body is included
           1    acceleration and velocity rigid-body only
           2    all of rigid-body is included (see note below)
        ======  ==============================================

    forcephi : 2d array_like or None; optional
        If not None, it is a force transformation data-recovery matrix
        as in::

             resp = atm*a + dtm*d - forcephi*f

    rbduf : scalar; optional
        Rigid-body uncertainty factor
    elduf : scalar; optional
        Dynamic uncertainty factor

    Returns
    -------
    rms : list
        List of vectors (corresponding to `drmlist`) of rms values of
        all data recovery rows; # of rows of each vector = # rows in
        each drm pair
    psd : list
        List of matrices (corresponding to `drmlist`) of PSD responses
        for all data recovery rows::

               # rows in each PSD = # rows in DRM
               # cols in each PSD = len(freq)

    Notes
    -----
    This routine first calls `fs.fsolve` to solve the modal equations
    of motion. Then, it scales the responses by the corresponding PSD
    input force. All PSD responses are summed together for the overall
    response. For example, for a displacement and acceleration
    dependent response::

        resp_psd = 0
        for i in range(forcepsd.shape[0]):
            # solve for unit frequency response function:
            genforce = t_frc[i:i+1].T @ np.ones((1, len(freq)))
            sol = fs.fsolve(genforce, freq, incrb)
            frf = atm @ sol.a + dtm @ sol.d
            resp_psd += forcepsd[i] * abs(frf)**2

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])        # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])      # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])        # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m                # diag damping
        >>> freq = np.arange(.1, 35, .1)              # frequency
        >>> forcepsd = 10000*np.ones((4, freq.size))  # PSD forces
        >>> fs = ode.SolveUnc(m, b, k)
        >>> atm = np.eye(4)    # recover modal accels
        >>> t_frc = np.eye(4)  # assume forces already modal
        >>> drms = [[atm, None]]
        >>> rms, psd = ode.solvepsd(fs, forcepsd, t_frc, freq,
        ...                         drms)

        The rigid-body results should be 100.0 g**2/Hz flat;
        rms = np.sqrt(100*34.8)

        >>> np.allclose(100., psd[0][0])
        True
        >>> np.allclose(np.sqrt(3480.), rms[0][0])
        True

        Plot the four accelerations PSDs:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('solvepsd demo', figsize=[8, 8])
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(freq, psd[0][j])
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel(r'Accel PSD ($g^2$/Hz)')
        ...     _ = plt.xlabel('Frequency (Hz)')
        >>> plt.tight_layout()
    """
    ndrms = len(drmlist)
    forcepsd = np.atleast_2d(forcepsd)
    freq = np.atleast_1d(freq)
    rpsd, cpsd = forcepsd.shape
    unitforce = np.ones((1, cpsd))
    psd = [0.] * ndrms
    rms = [0.] * ndrms

    for i in range(rpsd):
        # solve for unit frequency response function for i'th force:
        genforce = t_frc[i:i+1].T @ unitforce
        sol = fs.fsolve(genforce, freq, incrb)
        if rbduf != 1.0:
            sol.a[fs.rb] *= rbduf
            sol.v[fs.rb] *= rbduf
            sol.d[fs.rb] *= rbduf
        if elduf != 1.0:
            sol.a[fs.el] *= elduf
            sol.v[fs.el] *= elduf
            sol.d[fs.el] *= elduf
        for j, drmpair in enumerate(drmlist):
            atm = drmpair[0]
            dtm = drmpair[1]
            frf = 0.
            if atm is not None:
                frf += atm @ sol.a
            if dtm is not None:
                frf += dtm @ sol.d
            if forcephi is not None:
                frf -= forcephi[:, i:i+1] @ unitforce
            psd[j] += forcepsd[i] * abs(frf)**2

    # compute area under curve:
    freqstep = np.diff(freq)
    for j in range(ndrms):
        sumpsd = psd[j][:, :-1] + psd[j][:, 1:]
        rms[j] = np.sqrt(np.sum((freqstep * sumpsd), axis=1)/2)
    return rms, psd


def getmodepart(h_or_frq, sols, mfreq, factor=2/3, helpmsg=True,
                ylog=False, auto=None, idlabel='', frf_ttl=''):
    """
    Get modal participation from frequency response plots.

    Parameters
    ----------
    h_or_frq : list/tuple or 1d ndarray
        Plot line handles or frequency vector:

        - If list/tuple, it contains the plot line handles to the FRF
          curves; in this case, the analysis frequency is retrieved
          from the plot.
        - If it is a 1d ndarray, it is the frequency vector; in this
          case, a plot of FRFs (from sols) is generated in figure
          'FRF' (or 'FRF - '+idlabel)

    sols : list/tuple of lists/tuples
        Contains info to determine modal particpation::

            sols = [[Trecover1, accel1, Trecover1_row_labels],
                    [Trecover2, accel2, Trecover2_row_labels],
                     ... ]

        - each Trecover matrix is:  any number  x  modes
        - each accel matrix is: modes x frequencies
        - each row_labels entry is a list/tuple: len = # rows in
          corresponding Trecover (if Trecover only has 1 row, then
          row_labels may be just a string)

        The FRFs are recovered by::

                   FRFs1 = Trecover1*accel1
                   FRFs2 = Trecover2*accel2
                   ...

        accel1, accel2, etc are the complex modal acceleration (or
        displacement or velocity) frequency responses; normally output
        by, for example, :func:`SolveUnc.fsolve`
    mfreq : array_like
        Vector of modal frequencies (Hz)
    factor : scalar; optional
        From 0 to 1 for setting the criteria for choosing secondary
        modes: if mode participation of secondary mode(s) >= `factor`
        * max_participation, include it.
    helpmsg : bool; optional
        If True, print brief message explaining the mouse buttons
        before plotting anything
    ylog : bool; optional
        If True, y-axis will be log
    auto : list/tuple or None; optional

        - If None, operate interactively.
        - If a 2 element vector, it specifies an item in `sols` and a
          Trecover row (0-offset) that this routine will automatically
          (non-interactively) use to select modes. It will select the
          peak of the specified response and, based on mode
          participation, return the results. In other words, it acts
          as if you picked the peak of the specified curve and then
          hit 't'. The 1st element of `auto` selects which `sols`
          entry to use and the 2nd selects the matrix row. For
          example, to choose the 12th row of Trecover3, set `auto` to
          [2, 11].

    idlabel : string; optional If not '', it will be
          used in the figure name. This allows multiple
          getmodepart()'s to be run with the same model, each using
          its own FRF and MP windows. The figure names will be::

                 'FRF - '+idlabel   <-- used only if h_or_frq is freq
                 'MP - '+idlabel

    frf_ttl : string; optional
        Title used for FRF plot

    Returns
    -------
    modes : list
        List of selected mode numbers; 0-offset
    freqs : 1d ndarray
        Vector of frequencies in Hz corresponding to `modes`.

    Notes
    -----
    FRF peaks can only be selected in range of the analysis frequency,
    but modes outside this range may be selected based on modal
    participation.

    This routine will echo modal participation factors to the screen
    and plot them in figure 'MP' (or 'MP - '+idlabel).

    If `auto` is None (or some other non-2-element object), this
    routine works as follows:

        1. If `h` is frequency vector, plot FRFs from `sols` in figure
           'FRF' or 'FRF - '+idlabel
        2. Repeat:

           a. Waits for user to select response. Mouse/key commands::

                  Left  - select response point; valid only in the
                          FRF figure
                  Right - erase last selected mode(s)
                  't'   - done

           b. Plots mode participation bar graph in figure 'MP' (or
              'MP - '+idlabel) showing the frequency(s) of modes
              selected.

    If using `auto`, no plots are generated (see `auto` description
    above).

    See also
    --------
    :class:`SolveUnc`, :class:`FreqDirect`, :func:`modeselect`,
    :func:`pyyeti.datacursor`

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyyeti.ode import SolveUnc
    >>> from pyyeti.ode import getmodepart
    >>> import scipy.linalg as la
    >>> K = 40*np.random.randn(5, 5)
    >>> K = K @ K.T  # pos-definite K matrix, M is identity
    >>> M = None
    >>> w2, phi = la.eigh(K)
    >>> mfreq = np.sqrt(w2)/2/np.pi
    >>> zetain = np.array([ .02, .02, .05, .02, .05 ])
    >>> Z = np.diag(2*zetain*np.sqrt(w2))
    >>> freq = np.arange(0.1, 15.05, .1)
    >>> f = np.ones((1, len(freq)))
    >>> Tbot = phi[0:1, :]
    >>> Tmid = phi[2:3, :]
    >>> Ttop = phi[4:5, :]
    >>> fs = SolveUnc(M, Z, w2)
    >>> sol_bot = fs.fsolve(Tbot.T @ f, freq)
    >>> sol_mid = fs.fsolve(Tmid.T @ f, freq)

    Prepare transforms and solutions for :func:`getmodepart`:
    (Note: the top 2 items in sols could be combined since they
    both use the same acceleration)

    >>> sols = [[Tmid, sol_bot.a, 'Bot to Mid'],
    ...         [Ttop, sol_bot.a, 'Bot to Top'],
    ...         [Ttop, sol_mid.a, 'Mid to Top']]

    Approach 1: let :func:`getmodepart` do the FRF plotting:

    >>> lbl = 'getmodepart demo 1'
    >>> mds, frqs = getmodepart(freq, sols,     # doctest: +SKIP
    ...                         mfreq, ylog=1,  # doctest: +SKIP
    ...                         idlabel=lbl)    # doctest: +SKIP
    >>> print('modes =', mds)                   # doctest: +SKIP
    >>> print('freqs =', frqs)                  # doctest: +SKIP

    Approach 2: plot FRFs first, then call :func:`getmodepart`:

    >>> fig = plt.figure('approach 2 FRFs')         # doctest: +SKIP
    >>> for s in sols:                              # doctest: +SKIP
    ...     plt.plot(freq, abs(s[0] @ s[1]).T,      # doctest: +SKIP
    ...              label=s[2])                    # doctest: +SKIP
    >>> _ = plt.xlabel('Frequency (Hz)')            # doctest: +SKIP
    >>> plt.yscale('log')                           # doctest: +SKIP
    >>> _ = plt.legend(loc='best')                  # doctest: +SKIP
    >>> h = plt.gca().lines                         # doctest: +SKIP
    >>> lbl = 'getmodepart demo 2'                  # doctest: +SKIP
    >>> modes, freqs = getmodepart(h, sols,         # doctest: +SKIP
    ...                            mfreq,           # doctest: +SKIP
    ...                            idlabel=lbl)     # doctest: +SKIP
    >>> print('modes =', modes)                     # doctest: +SKIP
    >>> print('freqs =', freqs)                     # doctest: +SKIP
    """
    # check sols:
    if (not isinstance(sols, (list, tuple)) or
            not isinstance(sols[0], (list, tuple))):
        raise ValueError('`sols` must be list/tuple of lists/tuples')

    for j, s in enumerate(sols):
        if len(s) != 3:
            raise ValueError('sols[{}] must have 3 elements: '
                             '[Trecover, accel, labels]'.format(j))
        Trec = np.atleast_2d(s[0])
        acce = np.atleast_2d(s[1])
        labels = s[2]
        if Trec.shape[0] == 1:
            if isinstance(labels, (list, tuple)) and len(labels) != 1:
                raise ValueError('in sols[{}], Trecover has 1 row, '
                                 'but labels is length {}'.
                                 format(j, len(labels)))
        else:
            if not isinstance(labels, (list, tuple)):
                raise ValueError('in sols[{}], labels must be a '
                                 'list/tuple because Trecover has '
                                 'more than 1 row'.format(j))
            if Trec.shape[0] != len(labels):
                raise ValueError('in sols[{}], len(labels) != '
                                 'Trecover.shape[0]'.format(j))
        if Trec.shape[1] != acce.shape[0]:
            raise ValueError('in sols[{}], Trecover is not compatibly '
                             'sized with accel'.format(j))

    def _getmds(modepart):
        # find largest contributor mode:
        mode = np.argmax(modepart)
        mx = modepart[mode]
        # find other import participating modes:
        pv = np.nonzero(modepart >= factor*mx)[0]
        # sort, so most significant contributor is first:
        i = np.argsort(modepart[pv])[::-1]
        mds = pv[i]
        for m in mds:
            print("\tSelected mode index (0-offset) {}, frequency {:.4f}".
                  format(m, mfreq[m]))
        return mds

    mfreq = np.atleast_1d(mfreq)
    if isinstance(h_or_frq, np.ndarray):
        freq = h_or_frq
    else:
        freq = h_or_frq[0].get_xdata()

    if getattr(auto, '__len__', None):
        s = sols[auto[0]]
        r = auto[1]
        Trcv = np.atleast_2d(s[0])[r:r+1]
        resp = abs(Trcv @ s[1])
        # find which frequency index gave peak response:
        i = np.argmax(resp)
        # compute modal participation at this frequency:
        acce = np.atleast_2d(s[1])[:, i]
        modepart = abs(Trcv.ravel() * acce)
        # pv = np.nonzero((mfreq >= freq[0]) & (mfreq <= freq[-1]))[0]
        # mds = _getmds(modepart[pv])
        mds = _getmds(modepart)
        modes = sorted(mds)
        freqs = mfreq[modes]
        return modes, freqs

    if helpmsg:
        print("Mouse buttons:")
        print("\tLeft   - select response point")
        print("\tRight  - erase last selected mode(s)")
        print("To quit, type 't' inside the axes")

    if idlabel:
        frfname = "FRF - {}".format(idlabel)
        mpname = "MP - {}".format(idlabel)
    else:
        frfname = "FRF"
        mpname = "MP"

    import matplotlib.pyplot as plt
    from pyyeti.datacursor import DC
    if isinstance(h_or_frq, np.ndarray):
        freq = h_or_frq
        h = []
        plt.figure(frfname)
        plt.clf()
        for s in sols:
            Trec = np.atleast_2d(s[0])
            acce = np.atleast_2d(s[1])
            curlabels = s[2]
            if isinstance(curlabels, str):
                curlabels = [curlabels]
            for j in range(Trec.shape[0]):
                resp = abs(Trec[j:j+1] @ acce).ravel()
                h += plt.plot(freq, resp,
                              label=curlabels[j])
        if ylog:
            plt.yscale('log')
        if frf_ttl:
            plt.title(frf_ttl)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("FRF")
        plt.legend(loc='best')
    else:
        h = h_or_frq

    plt.figure(mpname)
    plt.clf()

    modes = []    # list to store modes
    primary = []  # flag to help delete plot objects logically

    def _addpoint(axes, x, y, n, i, ln):
        if ln not in h:
            print('invalid curve ... ignoring')
            return

        # find n'th (zero offset) Trec, acce, label:
        j = 0
        for s in sols:
            T = np.atleast_2d(s[0])
            rows = T.shape[0]
            if j+rows > n:
                row = n-j
                T = T[row]
                acce = np.atleast_2d(s[1])[:, i]
                labels = s[2]
                if isinstance(labels, str):
                    labels = [labels]
                label = labels[row]
                break
            j += rows

        # compute modal participation at this frequency
        modepart = abs(T * acce)
        mds = _getmds(modepart)

        # plot bar chart showing modal participation and label top
        # modes:
        fig = plt.figure(mpname)
        plt.clf()
        # pv = np.nonzero((mfreq >= freq[0]) & (mfreq <= freq[-1]))[0]
        # plt.bar(mfreq[pv], modepart[pv], align='center')
        plt.bar(mfreq, modepart, align='center')
        plt.title(label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mode Participation")
        ax = plt.gca()
        for i, m in enumerate(mds):
            barlabel = '{:.4f} Hz'.format(mfreq[m])
            ax.text(mfreq[m], modepart[m], barlabel,
                    ha='center', va='bottom')
            modes.append(m)
            primary.append(i == 0)
        fig.canvas.draw()

    def _deletepoint(axes, x, y, n, i):
        while len(primary) > 0:
            m = modes.pop()
            p = primary.pop()
            print("\tMode {}, {:.4f} Hz erased".format(m, mfreq[m]))
            if p:
                break
    try:
        DC.off()
        DC.addpt_func(_addpoint)
        DC.delpt_func(_deletepoint)
        DC.getdata()
    finally:
        DC.addpt_func(None)
        DC.delpt_func(None)
        DC.off()

    modes = sorted(list(set(modes)))
    freqs = mfreq[modes]
    return modes, freqs


def modeselect(name, fs, force, freq, Trcv, labelrcv, mfreq,
               factor=2/3, helpmsg=True, ylog=False, auto=None,
               idlabel=''):
    """
    Select modes based on mode participation in graphically chosen
    responses.

    Parameters
    ----------
    name : string
        Name of analysis; for example the flight event name; it is
        used for plot title
    fs : class
        An instance of :class:`SolveUnc` or :class:`FreqDirect` (or
        similar ... must have `.fsolve` method)
    force : 2d array_like
        Forcing function in frequency domain; # cols = len(freq)
    freq : 1d array_like
        Frequency vector in Hz where solution is requested
    Trcv : 2d array_like
        Data recovery matrix to the desired DOF
    labelrcv : list/tuple (can be string if `Trcv` has 1 row)
        List/tuple of labels; one for each row in Trcv; used for
        legend. May be a string if `Trcv` has only 1 row.
    mfreq : array_like
        Vector of modal frequencies (Hz)
    factor : scalar; optional
        From 0 to 1 for setting the criteria for choosing secondary
        modes: if mode participation of secondary mode(s) >= `factor`
        * max_participation, include it.
    helpmsg : bool; optional
        If True, print brief message explaining the mouse buttons
        before plotting anything
    ylog : bool; optional
        If True, y-axis will be log
    auto : integer or None; optional

        - If None, operate interactively.
        - If an integer, it specifies a row in `Trcv` row (0-offset)
          that this routine will automatically (non-interactively) use
          to select modes. It will select the peak of the specified
          response and, based on mode participation, return the
          results. In other words, it acts as if you picked the peak
          of the specified curve and then hit 't'. For example, to
          choose the 12th row of `Trcv`, set `auto` to 11.

    idlabel : string; optional If not '', it will be
          used in the figure name. This allows multiple
          getmodepart()'s to be run with the same model, each using
          its own FRF and MP windows. The figure names will be::

                 'FRF - '+idlabel
                 'MP - '+idlabel

    Returns
    -------
    modes : list
        List of selected mode numbers; 0-offset
    freqs : 1d ndarray
        Vector of frequencies in Hz corresponding to `modes`.
    resp : 2d ndarray
        The complex frequency responses of the accelerations recovered
        through `Trcv`; rows(`Trcv`) x len(`freq`)

    Notes
    -----
    This routine is an interface to :func:`getmodepart`. See that
    routine for more information.

    This routine can be very useful in selecting modes for damping or
    just identifying modes. For example, to identify axial modes,
    excite the structure axially and choose axial DOFs for recovery at
    the top, bottom, and somewhere in-between.

    See also
    --------
    :func:`getmodepart`, :class:`SolveUnc`, :class:`FreqDirect`

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyyeti.ode import SolveUnc, modeselect
    >>> import scipy.linalg as la
    >>> K = 40*np.random.randn(5, 5)
    >>> K = K @ K.T       # positive definite K matrix, M is identity
    >>> M = None
    >>> w2, phi = la.eigh(K)
    >>> mfreq = np.sqrt(w2)/2/np.pi
    >>> zetain = np.array([ .02, .02, .05, .02, .05 ])
    >>> Z = np.diag(2*zetain*np.sqrt(w2))
    >>> freq = np.arange(0.1, 15.05, .1)
    >>> f = phi[4:].T @ np.ones((1, len(freq)))      # force of DOF 5
    >>> Trcv = phi[[0, 2, 4]]                        # recover DOF 1, 3, 5
    >>> labels = ['5 to 1', '5 to 3', '5 to 5']
    >>> fs = SolveUnc(M, Z, w2)                     # doctest: +SKIP
    >>> mfr = modeselect('Demo', fs, f, freq,       # doctest: +SKIP
    ...                  Trcv, labels, mfreq)       # doctest: +SKIP
    >>> print('modes =', mfr[0])                    # doctest: +SKIP
    >>> print('freqs =', mfr[1])                    # doctest: +SKIP
    """
    sol = fs.fsolve(force, freq)
    sols = [[Trcv, sol.a, labelrcv]]
    if isinstance(auto, int):
        auto = [0, auto]
    modes, freqs = getmodepart(freq, sols, mfreq, factor, helpmsg,
                               ylog, auto, idlabel, frf_ttl=name)
    return modes, freqs, Trcv @ sol.a
