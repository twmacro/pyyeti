# -*- coding: utf-8 -*-

from types import SimpleNamespace
import warnings
import scipy.linalg as la
import numpy as np
from pyyeti import locate


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


__all__ = [
    "_process_incrb",
    "get_su_coef",
    "get_freq_damping",
    "eigss",
    "delconj",
    "addconj",
    "make_A",
    "solvepsd",
]


def _process_incrb(incrb):
    if not isinstance(incrb, str):
        warnings.warn(
            "the integer form of `incrb` is deprecated and will be "
            "removed in a future version; use the string format "
            "instead; eg: 'dva' instead of 2",
            FutureWarning,
        )
        return {0: "", 1: "va", 2: "dva"}[incrb]
    else:
        if len(set(incrb).difference(set("dva"))) > 0:
            raise ValueError(
                "`incrb` must only contain the letters: 'a', 'v', and/or 'd'"
            )
    return incrb


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
        assumed rigid-body if the `k` value is < 0.005.
    rfmodes : 1d ndarray or None; optional
        Index vector for the residual flexibility modes; if None,
        there are no residual flexibility modes.

    Returns
    -------
    coefs : SimpleNamespace with the members:
        ``F, G, A, B, Fp, Gp, Ap, Bp, pvrb, pvrb_damped``

    Notes
    -----
    All entries in `coefs` are 1d ndarrays. Except for `pvrb`, the
    outputs are the integration coefficients. It can handle
    rigid-body, under-damped, critically-damped, and over-damped
    equations. It can also handle rigid-body with damping. The solver
    is exact with the assumption that the forces vary linearly during
    a time step (1st order hold). `pvrb` is a boolean vector with True
    specifying where rigid-body modes are and `pvrb_damped` is the
    same but for the damped rigid-body modes.

    The coefficients are used as follows::

        for j in range(nt-1):
            d[:, j+1] = (F * d[:, j] + G * v[:, j] +
                           A * P[:, j] + B * P[:, j+1])
            v[:, j+1] = (Fp * d[:, j] + Gp * v[:, j] +
                           Ap * P[:, j] + Bp * P[:, j+1])

    where `d` is the displacement, `v` is the velocity, and `P` is the
    applied force.

    Most of the coefficients can be found in the Nastran Theoretical
    Manual [#nast1]_. For the case where ``k = 0`` but ``b != 0``
    (rigid-body with damping ... which is probably unusual), the
    coefficients were computed by hand and confirmed in Python using
    the "sympy" package.

    References
    ----------
    .. [#nast1] 'The NASTRAN Theoretical Manual', Section 11.5,
           NASA-SP-221(06), Jan 01, 1981.
           https://ntrs.nasa.gov/search.jsp?R=19840010609

    See also
    --------
    :class:`SolveUnc`
    """
    if h is None:
        return None
    n = len(b)
    if m is None:
        wo2 = k
        C = b / 2
    else:
        wo2 = k / m
        C = (b / m) / 2
    w2 = wo2 - C**2

    if rbmodes is None:
        pvrb = (wo2 < 0.005).astype(int)
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

    # check for damped rigid-body equations
    # - from trial and error, it was found that a decent cutoff value
    #   for velocities (to use damping) is when:
    #          1/(b**2*h) < 10**10
    #     or:  b > 10**-5 / sqrt(h)
    # - displacements have a different cutoff. the denominator
    #   is: 1/(b**3*h) < 10**10
    #     or:  b > (1e-10/h)**(1/3)
    #     adjusted to: b > 10*(1e-10/h)**(1/3) by trial and error
    pvrb_damped = (abs(C) > 1e-5 / np.sqrt(h)) & pvrb

    # setup partition vectors for underdamped, critically damped,
    # and overdamped equations
    badrows = None
    if np.any(pvel):
        pvover = np.zeros(n, bool)
        pvcrit = np.zeros(n, bool)
        pvundr = np.zeros(n, bool)
        rat = w2[pvel] / wo2[pvel]
        pvundr[pvel] = rat >= 1.0e-8
        pvcrit[pvel] = abs(rat) < 1.0e-8
        pvover[pvel] = rat <= -1e-8

        if not np.all(rfmodes2 + pvrb + pvundr + pvover + pvcrit == 1):
            badrows = np.nonzero(rfmodes2 + pvrb + pvundr + pvover + pvcrit != 1)[0]
    elif not np.all(rfmodes2 + pvrb == 1):
        badrows = np.nonzero(rfmodes2 + pvrb != 1)[0]

    if badrows is not None:
        msg = f"Partitioning problem. Check settings for mode number(s):\n{badrows}"
        raise ValueError(msg)

    w2 = abs(w2)
    # define the appropriate parameters based on damping
    # ... grab memory and at the same time set the rb equations
    F = pvrb.astype(float)
    G = h * F
    if m is None:
        A = (h * h / 3) * F
        Ap = (h / 2) * F
    else:
        A = (h * h / 3) * F / m
        Ap = (h / 2) * F / m
    B = A / 2
    Fp = np.zeros(n, float)
    Gp = F.copy()
    Bp = Ap.copy()

    if np.any(pvrb_damped):
        pvvelo = pvrb_damped.nonzero()[0]
        pvdisp = (abs(C[pvvelo]) > 10 * (1e-10 / h) ** (1 / 3)).nonzero()[0]
        # F & Fp are correct already
        beta = C[pvvelo] * 2
        ibm = 1 / beta if m is None else 1 / (beta * m[pvvelo])
        ibh = 1 / (beta * h)
        ibbh = 1 / (beta * beta * h)
        ex = np.exp(-beta * h)
        if pvdisp.size:
            # damping coefficients for displacements are only used if
            # damping is bigger ... see note above (otherwise, the rb
            # coefficients are more accurate)
            pv = pvvelo[pvdisp]
            G[pv] = ((1 - ex) / beta)[pvdisp]
            A[pv] = (ibm * ((1 / beta + ibbh) * ex + h / 2 - ibbh))[pvdisp]
            B[pv] = (ibm * (h / 2 - 1 / beta + (1 - ex) * ibbh))[pvdisp]
        Gp[pvvelo] = ex
        Ap[pvvelo] = ibm * (ibh - (1 + ibh) * ex)
        Bp[pvvelo] = ibm * (1 + ibh * (ex - 1))

    if np.any(pvel):
        if np.any(pvundr):
            w = np.sqrt(w2[pvundr])
            cs = np.cos(w * h)
            sn = np.sin(w * h)
            beta = C[pvundr]
            ex = np.exp(-beta * h)
            _wo2 = wo2[pvundr]
            _w2 = w2[pvundr]
            _k = k[pvundr]

            # for displacement:
            F[pvundr] = ex * (cs + (beta / w) * sn)
            G[pvundr] = (ex * sn) / w
            t0 = 1 / (h * _k * w)
            t1 = (_w2 - beta * beta) / _wo2
            t2 = (2 * w * beta) / _wo2
            A[pvundr] = t0 * (ex * ((t1 - h * beta) * sn - (t2 + h * w) * cs) + t2)
            B[pvundr] = t0 * (ex * (-t1 * sn + t2 * cs) + w * h - t2)

            # for velocity:
            Fp[pvundr] = -(_wo2 / w) * ex * sn
            Gp[pvundr] = ex * (cs - (beta / w) * sn)
            Ap[pvundr] = t0 * (ex * ((beta + h * _wo2) * sn + w * cs) - w)
            Bp[pvundr] = t0 * (-ex * (beta * sn + w * cs) + w)

        if np.any(pvcrit):
            beta = C[pvcrit]
            ex = np.exp(-beta * h)
            _wo2 = wo2[pvcrit]
            _k = k[pvcrit]

            # for displacement:
            hbeta = h * beta
            F[pvcrit] = ex * (1 + hbeta)
            G[pvcrit] = h * ex
            t0 = 1 / (h * _k)
            A[pvcrit] = t0 * (
                2 / beta - (1 / beta) * ex * (2 + 2 * hbeta + (hbeta * hbeta))
            )
            B[pvcrit] = (t0 / beta) * (hbeta - 2 + ex * (2 + hbeta))

            # for velocity:
            Fp[pvcrit] = -(beta * beta) * (h * ex)
            Gp[pvcrit] = ex * (1 - hbeta)
            Ap[pvcrit] = t0 * (ex * (1 + hbeta + (hbeta * hbeta)) - 1)
            Bp[pvcrit] = t0 * (1 - ex * (hbeta + 1))

        if np.any(pvover):
            # Original equations in reference use cosh and sinh
            # functions. These can overflow up with high damping
            # values, so the equations have been rearranged by
            # expanding these expressions (noting that beta > w):

            #  exp(-beta h) cosh(w h) -->
            #       = exp(-beta h) (exp(w h) + exp(-w h)) / 2
            #       = (exp(-h (beta - w)) + exp(-h (beta + w))) / 2
            #  exp(-beta h) sinh(w h) -->
            #       = exp(-beta h) (exp(w h) - exp(-w h)) / 2
            #       = (exp(-h (beta - w)) - exp(-h (beta + w))) / 2

            w = np.sqrt(w2[pvover])
            beta = C[pvover]
            ecosh = (np.exp(-h * (beta - w)) + np.exp(-h * (beta + w))) / 2.0
            esinh = (np.exp(-h * (beta - w)) - np.exp(-h * (beta + w))) / 2.0

            # precompute some other terms that are used multiple times:
            _wo2 = wo2[pvover]
            _w2 = w2[pvover]
            _k = k[pvover]
            t0 = 1 / (h * _k * w)
            t1 = (_w2 + beta * beta) / _wo2
            t2 = (2 * w * beta) / _wo2

            # for displacement:
            F[pvover] = ecosh + beta / w * esinh
            G[pvover] = esinh / w
            A[pvover] = t0 * (-(t1 + h * beta) * esinh - (t2 + h * w) * ecosh + t2)
            B[pvover] = t0 * (t1 * esinh + t2 * ecosh + w * h - t2)

            # for velocity:
            Fp[pvover] = -(_wo2 / w) * esinh
            Gp[pvover] = ecosh - beta / w * esinh
            Ap[pvover] = t0 * ((beta + h * _wo2) * esinh + w * ecosh - w)
            Bp[pvover] = t0 * (-beta * esinh - w * ecosh + w)

    if rfmodes is not None:
        F[rfmodes] = 0.0
        G[rfmodes] = 0.0
        A[rfmodes] = 0.0
        B[rfmodes] = 1.0 / k[rfmodes]  # from k q = force
        Fp[rfmodes] = 0.0
        Gp[rfmodes] = 0.0
        Ap[rfmodes] = 0.0
        Bp[rfmodes] = 0.0

    return SimpleNamespace(
        F=F,
        G=G,
        A=A,
        B=B,
        Fp=Fp,
        Gp=Gp,
        Ap=Ap,
        Bp=Bp,
        pvrb=pvrb,
        pvrb_damped=pvrb_damped,
    )


def _eigc_dups(lam, tol=1.0e-10):
    """
    Find duplicate complex eigenvalues from state-space formulation.

    Parameters
    ----------
    lam : 1d ndarray
        Vector of complex eigenvalues. Any complex-conjugate pairs
        must be adjacent (this is the normal case for
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
        if only the second and third roots are duplicates of each
        other, `dups` will be `np.array([1, 2])`.

    Notes
    -----
    Input `lam` is the vector of eigenvalues of `A` defined by::

        yd - A y = 0

    and the solution of ud - lam u = 0 is of the form::

        u = k exp(lam t)

    This routine is normally called by :func:`eigss`.
    """
    i = np.argsort(abs(lam.imag), kind="mergesort")
    lams = lam[i]  # order: real then complex w/ conjugates adjacent
    # find repeated roots:
    dups = np.nonzero(locate.find_duplicates(lams, tol))[0]
    return lams, i, dups


def get_freq_damping(lam, suppress_warning=False):
    r"""
    Get frequency and damping from complex eigenvalues

    Parameters
    ----------
    lam : 1d ndarray; shape = (2n,)
        Vector of potentially complex eigenvalues. Pairs are assumed
        to be adjacent. Furthermore, each pair of values in `lam` is
        assumed to be: :math:`-\omega_n (\zeta \pm \sqrt {\zeta^2 -
        1})`. This is the case when `lam` are the eigenvalues of
        :math:`A`, where :math:`A` is setup as shown in :func:`eigss`
        (which has a :math:`-K` in it as opposed to a :math:`+K`).
    suppress_warning : bool; optional
        If True, do not print a warning if this routine finds that
        there are complex eigenvalues without an adjacent
        conjugate. Suppressing the warning can be useful for cases
        where the state-space matrix is complex. In that case, this
        routine may not be that useful, but there's no use in printing
        a warning. See also the notes below.

    Returns
    -------
    wn : 1d ndarray; shape = (n,)
        Vector of natural frequencies (rad/sec) in the same order as
        `lam`. `wn` is always greater than or equal to zero.
    zeta : 1d ndarray; shape = (n,)
        Vector of critical damping ratios:

        ========  ====================================================
        \|zeta\|   description of eigenvalue pairs
        ========  ====================================================
         < 1.0    underdamped; eigenvalues are complex conjugate pairs
         = 1.0    critically damped; eigenvalues are real duplicates
         > 1.0    overdamped; eigenvalues are real and unequal
        ========  ====================================================

        .. note::
            If `zeta` comes out negative, it could mean that the sign
            on `lam` is the opposite of that expected. Or, you have
            negative damping. If `zeta` should be positive for your
            problem, just change the sign on `lam` (and see
            description of the `lam` input).

    Notes
    -----
    The implicit assumption behind this routine is that the equations
    of motion are real (that the state-space matrix as documented in
    :class:`SolveUnc` is real). If that is not true, this routine may
    or may not be helpful, probably depending on how close to real the
    system is. Note that this routine will issue a warning if it finds
    that there are complex eigenvalues without an adjacent conjugate
    (unless `suppress_warning` is True). The rest of the documentation
    assumes the system is real.

    The complex eigenvalue pairs for all cases (underdamped,
    critically damped, and overdamped) are:

    .. math::
        \begin{aligned}
           \lambda_1 &= -\omega_n (\zeta + \sqrt {\zeta^2 - 1}) \\
           \lambda_2 &= -\omega_n (\zeta - \sqrt {\zeta^2 - 1})
        \end{aligned}

    The eigenvalue pairs will be real if :math:`|\zeta| >= 1.0` and
    complex conjugates otherwise.

    The natural frequency :math:`\omega_n` is computed by taking the
    square-root of the product of the eigenvalue pairs. This works for
    underdamped, critically damped, and overdamped cases:

    .. math::
        \omega_n = \sqrt{\lambda_1 \lambda_2}

    Because:

    .. math::
        \begin{aligned}
           \lambda_1 \lambda_2 &=
                \omega_n (\zeta + \sqrt {\zeta^2 - 1})
                \omega_n (\zeta - \sqrt {\zeta^2 - 1}) \\
             &= \omega_n^2 (\zeta^2 - (\zeta^2 - 1) \\
             &= \omega_n^2
        \end{aligned}

    Once :math:`\omega_n` is known, :math:`\zeta` can be computed by
    dividing the sum of the eigenvalue pairs by :math:`-2 \omega_n`:

    .. math::
        \begin{aligned}
           \zeta &= -\frac{\lambda_1 + \lambda_2}{2 \omega_n} \\
                 &= \frac{\omega_n (\zeta + \sqrt {\zeta^2 - 1})
                        + \omega_n (\zeta - \sqrt {\zeta^2 - 1})}
                        {2 \omega_n} \\
                 &= \frac{2 \omega_n \zeta}{2 \omega_n} \\
                 &= \zeta
        \end{aligned}

    .. note::
        This routine will give incorrect results without warning if
        any real eigenvalue is not adjacent to its pair. (A complex
        eigenvalue that is not adjacent to its pair will trigger a
        warning.)

    Raises
    ------
        ValueError
            When length of `lam` is not even (eigenvalues must be in
            pairs)

    Examples
    --------
    Set up a diagonal system of equations with known natural
    frequencies and damping. Then, to demo the routine, put the
    equations into state-space form (as shown in :class:`SolveUnc`),
    compute eigenvalues, call this routine, and check results:

    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> from pyyeti import ode
    >>>
    >>> m = np.array([10.0, 11.0, 12.0, 13.0])  # mass
    >>> k = np.array([6.0e5, 7.0e5, 8.0e5, 9.0e5])  # stiffness
    >>> zeta = np.array([0.2, 0.05, 1.0, 2.0])  # percent damping
    >>>
    >>> wn = np.sqrt(k / m)
    >>>
    >>> b = 2.0 * zeta * np.sqrt(k / m) * m  # damping
    >>>
    >>> # solve eigenvalue problem on state-space equations:
    >>> A = ode.make_A(m, b, k)
    >>> lam, phi = la.eig(A)
    >>> wn_extracted, zeta_extracted = ode.get_freq_damping(lam)
    >>>
    >>> # check results:
    >>> i = np.argsort(wn_extracted)
    >>> np.allclose(wn_extracted[i], wn)
    True
    >>> np.allclose(zeta_extracted[i], zeta)
    True

    Check sign convention. The first input follows the sign convention
    described above, so should result in a positive 60% damping. The
    second is the opposite, so should give -60% damping:

    >>> ode.get_freq_damping([-3+4j, -3-4j])
    (array([ 5.]), array([ 0.6]))
    >>> ode.get_freq_damping([3+4j, 3-4j])
    (array([ 5.]), array([-0.6]))
    """
    # find complex conjugate pairs:
    lam = np.atleast_1d(lam)

    lam1 = lam[::2]
    lam2 = lam[1::2]
    if lam1.shape[0] != lam2.shape[0]:
        raise ValueError("`lam` must be even length")

    mult = lam1 * lam2
    add = -(lam1 + lam2)
    if not suppress_warning:
        if (abs(mult.imag) > 1e-14 * abs(mult.real)).any() or (
            abs(add.imag).max() > 1e-14
        ):
            warnings.warn(
                "Eigenvalues pairs in `lam` appear not to be adjacent. Multiplying "
                "and adding pairs resulted in a non-zero imaginary parts:\n"
                f"  Multiply: abs((lam1 * lam2).imag).max() = {abs(mult.imag).max()}\n"
                f"  Add:      abs((lam1 + lam2).imag).max() = {abs(add.imag).max()}",
                RuntimeWarning,
            )

    wn = np.sqrt(abs(mult.real))
    zeta = add.real / (2 * wn)
    return wn, zeta


def _eigss_note():
    return (
        "Solution will likely be inaccurate. "
        "Possible causes/solutions:\n"
        "\tThe partition vector for the rigid-body modes is "
        "incorrect or not set\n"
        "\tThe equations are not in modal space, and the "
        "rigid-body modes cannot be detected -- use the "
        "`pre_eig` option\n"
        "\tUse :class:`SolveExp2` instead for time domain, or\n"
        "\tUse :class:`FreqDirect` instead for frequency domain\n\n"
        "\tSetting `eig_success` attribute to False\n"
    )


def eigss(A, delcc):
    r"""Solve complex eigen problem for state-space formulation.

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
    A SimpleNamespace with the members:

    lam : 1d ndarray; complex
        The vector of complex eigenvalues
    ur : 2d ndarray; complex
        Normalized complex right eigenvectors
    ur_inv : 2d ndarray; complex
        Inverse of right eigenvectors
    dups : 1d ndarray
        Index partition vector for repeated roots; it will be empty
        (`np.array([])`) if there are no repeated roots. For example,
        if only the second and third roots are duplicates of each
        other, `dups` will be `np.array([1, 2])`.
    wn : 1d ndarray; real
        Vector of natural frequencies (rad/sec) in the same order as
        `lam` (see :func:`get_freq_damping`)
    zeta : 1d ndarray; real
        Vector of critical damping ratios (see
        :func:`get_freq_damping`)
    eig_success : bool
        True if routine is successful. False if the eigenvectors form
        a singular matrix or they do not diagonalize `A`; in that
        case, ODE solution (if computed) is most likely wrong.

    Notes
    -----
    The typical 2nd order ODE is:

    .. math::
        M \ddot{q} + B \dot{q} + K q = F

    The 2nd order ODE set of equations are transformed into the 1st
    order ODE (see :func:`make_A`):

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} -M^{-1} B & -M^{-1} K \\ I & 0 \end{array}
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
    :func:`make_A`, :class:`SolveUnc`, :func:`get_freq_damping`.

    """
    lam, ur = la.eig(A)
    c = np.linalg.cond(ur)
    if c > 1 / np.finfo(float).eps:
        warn1 = (
            "in :func:`eigss`, the eigenvectors for the state-"
            "space formulation are poorly conditioned (cond={:.3e}).\n"
        )
        warnings.warn(warn1.format(c) + _eigss_note(), RuntimeWarning)
        eig_success = False
    else:
        eig_success = True
    ur_inv = la.inv(ur)
    lam, i, dups = _eigc_dups(lam)
    ur = ur[:, i]
    ur_inv = ur_inv[i]
    if dups.size:
        uau = ur_inv @ A @ ur
        d = np.diag(uau)
        max_off = abs(np.diag(d) - uau).max()
        max_on = abs(d).max()
        if max_off > 1e-8 * max_on:
            warn2 = (
                "Repeated roots detected and equations do not appear "
                "to be diagonalized. Generally, this is a failure "
                "condition.\n"
                "\tMax off-diag / on-diag of `inv(ur) @ A @ ur` = {} / {} = {}\n"
            )
            warnings.warn(
                warn2.format(max_off, max_on, max_off / max_on) + _eigss_note(),
                RuntimeWarning,
            )
            eig_success = False

    wn, zeta = get_freq_damping(lam, np.iscomplexobj(A))
    if delcc:
        lam, ur, ur_inv, dups = delconj(lam, ur, ur_inv, dups)
    return SimpleNamespace(
        lam=lam,
        ur=ur,
        ur_inv=ur_inv,
        dups=dups,
        wn=wn,
        zeta=zeta,
        eig_success=eig_success,
    )


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
        if only the second and third roots are duplicates of each
        other, `dups` will be `np.array([1, 2])`.

    Returns
    -------
    lam1 : 1d ndarray; complex
        Trimmed vector of complex eigenvalues
    ur1 : 2d ndarray; complex
        Trimmed normalized complex right eigenvectors; columns may be
        trimmed
    ur_inv1 : 2d ndarray; complex
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
    neg = lam.imag < 0.0
    if np.any(neg):
        # see if lambda's are all comp-conj pairs; if not, do nothing
        posi = np.nonzero(lam.imag > 0.0)[0]
        negi = np.nonzero(neg)[0]
        if (
            posi.size == negi.size
            and np.all(abs(posi - negi) == 1)
            and np.all(lam[posi] == np.conj(lam[negi]))
        ):
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
            pv = lam.imag > 0.0
            if np.any(pv):
                ur[:, pv] *= 2.0
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
    lam1 : 1d ndarray; complex
        The vector of complex eigenvalues (complete set)
    ur1 : 2d ndarray; complex
        Complete set of normalized complex right eigenvectors; will be
        square
    ur_inv1 : 2d ndarray; complex
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
    conj2 = np.nonzero(lam.imag > 0.0)[0]
    if ur.shape[0] > ur.shape[1] and conj2.size > 0:
        if np.any(lam.imag < 0.0):
            return lam, ur, ur_inv
        two = ur_inv[conj2[0]] @ ur[:, conj2[0]]
        if abs(two - 2.0) > 1e-13:
            raise ValueError(
                "factor of 2.0 seems to be missing: "
                f"error on first underdamped mode = {abs(two - 2.0)}"
            )
        n = len(lam) + len(conj2)
        if n != ur_inv.shape[1] or n != ur.shape[0]:
            raise ValueError(
                "ur and/or ur_inv will not be square after adding missing modes"
            )
        reals = np.nonzero(lam.imag == 0.0)[0]
        if len(reals) > 0 and np.max(reals) > np.min(conj2):
            raise ValueError("not all underdamped are last")
        conj2_new = conj2 + np.arange(1, len(conj2) + 1)
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
        ur1[:, conj1_new] = ur[:, conj2] / 2.0
        ur1[:, conj2_new] = np.conj(ur[:, conj2]) / 2.0
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
            \begin{array}{cc} -M^{-1} B & -M^{-1} K \\ I & 0 \end{array}
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
        if np.iscomplexobj(M) or np.iscomplexobj(B) or np.iscomplexobj(K):
            Atype = complex
    n = K.shape[0]
    A = np.zeros((2 * n, 2 * n), Atype)
    v1 = range(n)
    v2 = range(n, 2 * n)
    if B.ndim == 2:
        A[:n, :n] = -B
    else:
        A[v1, v1] = -B
    if K.ndim == 2:
        A[:n, n:] = -K
    else:
        A[v1, v2] = -K
    A[v2, v1] = 1.0
    if M is not None:
        if M.ndim == 1:
            A[:n] = (1.0 / M).reshape(-1, 1) * A[:n]
        else:
            A[:n] = la.solve(M, A[:n])
    return A


def solvepsd(fs, forcepsd, t_frc, freq, drmlist, rbduf=1.0, elduf=1.0, **kwargs):
    """
    Solve equations of motion in frequency domain with uncorrelated
    PSD forces.

    See also :func:`pyyeti.cla.DR_Results.solvepsd` for a very similar
    routine, but one that is designed for use within the pyYeti
    "coupled loads analysis paradigm" (where the classes defined in
    :mod:`pyyeti.cla` are used).

    Parameters
    ----------
    fs : class instance
        An instance of :class:`SolveUnc` or :class:`FreqDirect` (or
        similar ... must have `.fsolve` method)
    forcepsd : 2d array_like
        The matrix of force psds; each row is a force PSD
    t_frc : 2d array_like
        Transform to put `forcepsd` into the coordinates of the
        equations of motion: ``t_frc @ forcepsd``. Commonly, `t_frc`
        is simply the transpose of a row-partition of the mode shape
        matrix (phi) and the conversion of `forcepsd` is from physical
        space to modal space. In that case, the row-partition is from
        the full set down to just the forced DOF. However, `t_frc` can
        also have force mappings (as from the TLOAD capability in
        Nastran); in that case, ``t_frc = phi.T @
        mapping_vectors``. In any case, the number of columns in
        `t_frc` is the number of rows in `forcepsd`: ``t_frc.shape[1]
        == forcepsd.shape[0]``
    freq : 1d array_like
        Frequency vector at which solution will be computed;
        ``len(freq) = cols(forcepsd)``
    drmlist : list_like
        List of lists (or similar) of any number of data recovery
        matrix quadruples (in the order typically used to write
        equations of motion)::

            [
                [drma1, drmv1, drmd1, drmf1],
                [drma2, drmv2, drmd2, drmf2],
                ...
            ]

        To not use a particular drm, set it to None. For example, to perform
        these 3 types of data recovery::

                acce = atm*a
                disp = dtm*d
                loads = ltma*a + ltmv*v + ltmd*d + ltmf*f

        `drmlist` would be::

              [[atm, None, None, None],
               [None, None, dtm, None],
               [ltma, ltmv, ltmd, ltmf]]

    rbduf : scalar; optional
        Rigid-body uncertainty factor
    elduf : scalar; optional
        Dynamic uncertainty factor
    **kwargs : keyword arguments for ``fs.fsolve``; optional
        Currently, there are two arguments available:

        ============  ============================================
          argument    brief description
        ============  ============================================
        incrb         specifies how to handle rigid-body responses
        rf_disp_only  specifies how to handle residual-flexibility
                      modes
        ============  ============================================

        See, for example, :func:`SolveUnc.fsolve`.

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
    This routine first calls ``fs.fsolve`` to solve the modal
    equations of motion. Then, it scales the responses by the
    corresponding PSD input force. All PSD responses are summed
    together for the overall response. For example::

        resp_psd = 0
        for i in range(forcepsd.shape[0]):
            # solve for unit frequency response function:
            unitforce = np.ones((1, len(freq)))
            genforce = t_frc[:, i:i+1] @ unitforce
            sol = fs.fsolve(genforce, freq, incrb="av")
            frf = (drma @ sol.a
                   + drmv @ sol.v
                   + drmd @ sol.d
                   + drmf[:, [i]] @ unitforce)
            resp_psd += forcepsd[i] * abs(frf)**2

    In that example, the data recovery uses all four drms. Also, the
    looping over the `drmlist` is not included for simplicity.

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
        >>> drms = [[atm, None, None, None]]
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
        >>> fig = plt.figure('Example', figsize=[8, 8], clear=True,
        ...                  layout='constrained')
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(freq, psd[0][j])
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel(r'Accel PSD ($g^2$/Hz)')
        ...     _ = plt.xlabel('Frequency (Hz)')
    """
    ndrms = len(drmlist)
    forcepsd, t_frc = np.atleast_2d(forcepsd, t_frc)
    freq = np.atleast_1d(freq)
    rpsd, cpsd = forcepsd.shape
    unitforce = np.ones((1, cpsd))
    psd = [0.0] * ndrms
    rms = [0.0] * ndrms

    if t_frc.shape[1] != rpsd:
        raise ValueError(
            "`forcepsd` and `t_frc` are incompatibly "
            f"sized: {forcepsd.shape} vs {t_frc.shape}"
        )

    for i in range(rpsd):
        # solve for unit frequency response function for i'th force:
        genforce = t_frc[:, i : i + 1] @ unitforce
        sol = fs.fsolve(genforce, freq, **kwargs)
        if rbduf != 1.0:
            sol.a[fs.rb] *= rbduf
            sol.v[fs.rb] *= rbduf
            sol.d[fs.rb] *= rbduf
        if elduf != 1.0:
            sol.a[fs.el] *= elduf
            sol.v[fs.el] *= elduf
            sol.d[fs.el] *= elduf
        for j, (drma, drmv, drmd, drmf) in enumerate(drmlist):
            frf = 0.0
            if drma is not None:
                frf += drma @ sol.a
            if drmv is not None:
                frf += drmv @ sol.v
            if drmd is not None:
                frf += drmd @ sol.d
            if drmf is not None:
                frf += drmf[:, i : i + 1] @ unitforce
            psd[j] += forcepsd[i] * abs(frf) ** 2

    # compute area under curve:
    freqstep = np.diff(freq)
    for j in range(ndrms):
        sumpsd = psd[j][:, :-1] + psd[j][:, 1:]
        rms[j] = np.sqrt(np.sum((freqstep * sumpsd), axis=1) / 2)
    return rms, psd
