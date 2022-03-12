# -*- coding: utf-8 -*-
"""
Tools for computing the matrix exponential and the integral.
"""

import warnings
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass

mf = spla.matfuncs
try:
    # scipy version 1.7 or older?
    mf._ExpmPadeHelper
except AttributeError:
    # scipy version 1.8 or newer
    mf = spla._matfuncs


class _ExpmIntPadeHelper(mf._ExpmPadeHelper):
    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and
            products will be used. Otherwise, the one-norm of powers
            and products may initially be estimated.
        """
        mf._ExpmPadeHelper.__init__(self, A, structure=None, use_exact_onenorm=False)
        self._A3 = None
        self._A5 = None

    @property
    def A3(self):
        if self._A3 is None:
            self._A3 = mf._smart_matrix_product(
                self.A, self.A2, structure=self.structure
            )
        return self._A3

    @property
    def A5(self):
        if self._A5 is None:
            self._A5 = mf._smart_matrix_product(
                self.A, self.A4, structure=self.structure
            )
        return self._A5

    def pade3_i(self, h):
        b = (120.0, 60.0, 12.0, 1.0)
        U = b[3] * self.A3 + b[1] * self.A
        V = b[2] * self.A2 + b[0] * self.ident
        p = (840.0, 60.0, 20.0, 1.0)
        q = (840.0, -360.0, 60.0, -4.0)
        P = h * (p[3] * self.A3 + p[2] * self.A2 + p[1] * self.A + p[0] * self.ident)
        Q = q[3] * self.A3 + q[2] * self.A2 + q[1] * self.A + q[0] * self.ident
        return U, V, P, Q

    def pade5_i(self, h):
        b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
        U = b[5] * self.A5 + b[3] * self.A3 + b[1] * self.A
        V = b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        p = (332640.0, 15120.0, 10080.0, 420.0, 42.0, 1.0)
        q = (332640.0, -151200.0, 30240.0, -3360.0, 210.0, -6.0)
        P = h * (
            p[5] * self.A5
            + p[4] * self.A4
            + p[3] * self.A3
            + p[2] * self.A2
            + p[1] * self.A
            + p[0] * self.ident
        )
        Q = (
            q[5] * self.A5
            + q[4] * self.A4
            + q[3] * self.A3
            + q[2] * self.A2
            + q[1] * self.A
            + q[0] * self.ident
        )
        return U, V, P, Q

    def pade7_i(self, h):
        U, V = self.pade7()
        p = (259459200.0, 8648640.0, 8648640.0, 277200.0, 55440.0, 1512.0, 72.0, 1.0)
        q = (
            259459200.0,
            -121080960.0,
            25945920.0,
            -3326400.0,
            277200.0,
            -15120.0,
            504.0,
            -8.0,
        )
        P = h * (
            mf._smart_matrix_product(
                self.A,
                p[7] * self.A6 + p[5] * self.A4 + p[3] * self.A2 + p[1] * self.ident,
                structure=self.structure,
            )
            + p[6] * self.A6
            + p[4] * self.A4
            + p[2] * self.A2
            + p[0] * self.ident
        )
        Q = (
            mf._smart_matrix_product(
                self.A,
                q[7] * self.A6 + q[5] * self.A4 + q[3] * self.A2 + q[1] * self.ident,
                structure=self.structure,
            )
            + q[6] * self.A6
            + q[4] * self.A4
            + q[2] * self.A2
            + q[0] * self.ident
        )
        return U, V, P, Q

    def pade9_i(self, h):
        U, V = self.pade9()
        p = (
            335221286400.0,
            8821612800.0,
            11762150400.0,
            302702400.0,
            90810720.0,
            2162160.0,
            205920.0,
            3960.0,
            110.0,
            1.0,
        )
        q = (
            335221286400.0,
            -158789030400.0,
            35286451200.0,
            -4843238400.0,
            454053600.0,
            -30270240.0,
            1441440.0,
            -47520.0,
            990.0,
            -10.0,
        )
        P = h * (
            mf._smart_matrix_product(
                self.A,
                p[9] * self.A8
                + p[7] * self.A6
                + p[5] * self.A4
                + p[3] * self.A2
                + p[1] * self.ident,
                structure=self.structure,
            )
            + p[8] * self.A8
            + p[6] * self.A6
            + p[4] * self.A4
            + p[2] * self.A2
            + p[0] * self.ident
        )
        Q = (
            mf._smart_matrix_product(
                self.A,
                q[9] * self.A8
                + q[7] * self.A6
                + q[5] * self.A4
                + q[3] * self.A2
                + q[1] * self.ident,
                structure=self.structure,
            )
            + q[8] * self.A8
            + q[6] * self.A6
            + q[4] * self.A4
            + q[2] * self.A2
            + q[0] * self.ident
        )
        return U, V, P, Q

    def pade13_scaled_i(self, s, h):
        b = (
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        )
        B = self.A * 2 ** -s
        h = h * 2 ** -s
        B2 = self.A2 * 2 ** (-2 * s)
        B4 = self.A4 * 2 ** (-4 * s)
        B6 = self.A6 * 2 ** (-6 * s)
        U2 = mf._smart_matrix_product(
            B6, b[13] * B6 + b[11] * B4 + b[9] * B2, structure=self.structure
        )
        U = mf._smart_matrix_product(
            B,
            (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * self.ident),
            structure=self.structure,
        )
        V2 = mf._smart_matrix_product(
            B6, b[12] * B6 + b[10] * B4 + b[8] * B2, structure=self.structure
        )
        V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * self.ident
        p = (
            1748648318376960000.0,
            32382376266240000.0,
            64764752532480000.0,
            1187353796428800.0,
            593676898214400.0,
            10559470521600.0,
            2011327718400.0,
            33522128640.0,
            2793510720.0,
            40840800.0,
            1485120.0,
            16380.0,
            210.0,
            1.0,
        )
        q = (
            1748648318376960000.0,
            -841941782922240000.0,
            194294257597440000.0,
            -28496491114291200.0,
            2968384491072000.0,
            -232308351475200.0,
            14079294028800.0,
            -670442572800.0,
            25141596480.0,
            -735134400.0,
            16336320.0,
            -262080.0,
            2730.0,
            -14.0,
        )
        _P2 = mf._smart_matrix_product(
            B6, p[13] * B6 + p[11] * B4 + p[9] * B2, structure=self.structure
        )
        P2 = mf._smart_matrix_product(
            B,
            _P2 + p[7] * B6 + p[5] * B4 + p[3] * B2 + p[1] * self.ident,
            structure=self.structure,
        )
        P1 = mf._smart_matrix_product(
            B6, p[12] * B6 + p[10] * B4 + p[8] * B2, structure=self.structure
        )
        P = h * (P2 + P1 + p[6] * B6 + p[4] * B4 + p[2] * B2 + p[0] * self.ident)
        _Q2 = mf._smart_matrix_product(
            B6, q[13] * B6 + q[11] * B4 + q[9] * B2, structure=self.structure
        )
        Q2 = mf._smart_matrix_product(
            B,
            _Q2 + q[7] * B6 + q[5] * B4 + q[3] * B2 + q[1] * self.ident,
            structure=self.structure,
        )
        Q1 = mf._smart_matrix_product(
            B6, q[12] * B6 + q[10] * B4 + q[8] * B2, structure=self.structure
        )
        Q = Q2 + Q1 + q[6] * B6 + q[4] * B4 + q[2] * B2 + q[0] * self.ident
        return U, V, P, Q


def expmint(A, h, geti2=False):
    """
    Compute the matrix exponential and its integral(s) using Pade
    approximation.

    Parameters
    ----------
    A : (M, M) array_like or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated
    h : scalar
        Time step
    geti2 : bool
        If True, also return the `I2` integral below. Useful for
        first order holds.

    Returns
    -------
    E : (M, M) ndarray
        Matrix exponential of `A*h`: exp(A*h)
    I : (M, M) ndarray
        Integral of exp(A*t) dt from 0 to h
    I2 : (M, M) ndarray; optional
        Integral of exp(A*t)*t dt from 0 to h. Only returned if
        `geti2` is True.

    Notes
    -----
    This routine is modeled after and augments
    :func:`scipy.linalg.expm`. The power series expansions for these
    matrices are (I = identity):

    .. code-block:: none

      E = I + A*h + (A*h)**2/2! + (A*h)**3/3! + ...
      I1 = h*(I + A*h/2 + (A*h)**2/3! + (A*h)**3/4! + ...)
      I2 = h*h*(I/2 + A*h/3 + (A*h)**2/(4*2!) + (A*h)**3/(5*3!) + ...)

    If `A` is non-singular, the exact solutions for `I1` and `I2` are::

        E = exp(A*h)
        I1 = inv(A)*(E-I)
        I2 = inv(A)*(E*h-I1)

    The Pade approximants for those power series are used for `E` and
    `I1`. If necessary, the 'squaring and scaling' method will be used
    such that a 13th order Pade approximation will be accurate. See
    references [#exp1]_, [#exp2]_, and [#exp3]_ for more information.

    For `I2`, a Pade approximation is used if a 3rd, 5th, 7th or 9th
    order is accurate. If not, `A` is checked for singularity. If
    non-singular, the exact solution show above is used. If A is
    singular, a the power series is used directly until it converges
    (and a warning message is printed about using a finer time step.)

    References
    ----------
    .. [#exp1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix
           Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    .. [#exp2] Nicholas J. Higham (2005)
           "The Scaling and Squaring Method for the Matrix Exponential
           Revisited."
           SIAM Journal on Matrix Analysis and Applications.
           Vol 26, No. 4, pp 1179-1193.

    .. [#exp3] David Westreich (1990)
           "A Practical Method for Computing the Exponential of a Matrix
           and its Integral."
           Communications in Applied Numerical Methods, Vol 6, 375-380.

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> e, i, i2 = expmint.expmint(a, .05, True)
    >>> e
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> i
    array([[ 0.052 ,  0.0034,  0.0048],
           [ 0.0067,  0.0583,  0.01  ],
           [ 0.0114,  0.0133,  0.0651]])
    >>> i2
    array([[ 0.0013,  0.0001,  0.0002],
           [ 0.0002,  0.0015,  0.0003],
           [ 0.0004,  0.0005,  0.0018]])
    """
    # Avoid indiscriminate asarray() to allow sparse or other strange
    # arrays.
    if isinstance(A, (list, tuple)):
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected a square matrix")

    # Detect upper triangularity.
    if mf._is_upper_triangular(A):
        structure = mf.UPPER_TRIANGULAR
    else:
        structure = None

    # Hardcode a matrix order threshold for exact vs. estimated
    # one-norms.
    use_exact_onenorm = A.shape[0] < 200

    # Track functions of A to help compute the matrix exponential.
    H = _ExpmIntPadeHelper(
        A * h, structure=structure, use_exact_onenorm=use_exact_onenorm
    )

    def Return(U, V, P, Q, geti2, pade):
        E = mf._solve_P_Q(U, V, structure=structure)
        I = _solve_P_Q_2(P, Q, structure=structure)
        if geti2:
            return E, I, _geti2(H, E, I, h, pade)
        return E, I

    # Try Pade order 3.
    eta_1 = max(H.d4_loose, H.d6_loose)
    if eta_1 < 1.495585217958292e-002 and mf._ell(H.A, 3) == 0:
        U, V, P, Q = H.pade3_i(h)
        return Return(U, V, P, Q, geti2, 3)

    # Try Pade order 5.
    eta_2 = max(H.d4_tight, H.d6_loose)
    if eta_2 < 2.539398330063230e-001 and mf._ell(H.A, 5) == 0:
        U, V, P, Q = H.pade5_i(h)
        return Return(U, V, P, Q, geti2, 5)

    # Try Pade orders 7 and 9.
    eta_3 = max(H.d6_tight, H.d8_loose)
    if eta_3 < 9.504178996162932e-001 and mf._ell(H.A, 7) == 0:
        U, V, P, Q = H.pade7_i(h)
        return Return(U, V, P, Q, geti2, 7)

    if eta_3 < 2.097847961257068e000 and mf._ell(H.A, 9) == 0:
        U, V, P, Q = H.pade9_i(h)
        return Return(U, V, P, Q, geti2, 9)

    # Use Pade order 13.
    eta_4 = max(H.d8_loose, H.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + mf._ell(2 ** -s * H.A, 13)
    U, V, P, Q = H.pade13_scaled_i(s, h)
    E = mf._solve_P_Q(U, V, structure=structure)
    I = _solve_P_Q_2(P, Q, structure=structure)
    # E = r_13(A)^(2^s) by repeated squaring.
    for _ in range(s):
        I += I.dot(E)
        E = E.dot(E)
    if geti2:
        return E, I, _geti2(H, E, I, h, 13)
    return E, I


def _geti2(H, E, I, h, pade):
    """
    A helper routine for expmint; computes I2.  See :func:`expmint`.
    """
    if pade <= 3:
        p = (450240.0, 104860.0, 15480.0, 1001.0)
        q = (900480.0, -390600.0, 66240.0, -4540.0)
        P = (h * h) * (p[3] * H.A3 + p[2] * H.A2 + p[1] * H.A + p[0] * H.ident)
        Q = q[3] * H.A3 + q[2] * H.A2 + q[1] * H.A + q[0] * H.ident
        return _solve_P_Q_2(P, Q, H.structure)

    if pade <= 5:
        p = (
            69363423360.0,
            14569133040.0,
            2595781440.0,
            239983884.0,
            14154000.0,
            398959.0,
        )
        q = (
            138726846720.0,
            -63346298400.0,
            12740716800.0,
            -1425725280.0,
            89937120.0,
            -2602278.0,
        )
        P = (h * h) * (
            mf._smart_matrix_product(
                H.A, p[5] * H.A4 + p[3] * H.A2 + p[1] * H.ident, structure=H.structure
            )
            + p[4] * H.A4
            + p[2] * H.A2
            + p[0] * H.ident
        )
        Q = (
            mf._smart_matrix_product(
                H.A, q[5] * H.A4 + q[3] * H.A2 + q[1] * H.ident, structure=H.structure
            )
            + q[4] * H.A4
            + q[2] * H.A2
            + q[0] * H.ident
        )
        return _solve_P_Q_2(P, Q, H.structure)

    if pade <= 7:
        p = (
            41840477770291200.0,
            8321436219096000.0,
            1617627908856960.0,
            159144893827920.0,
            12344957190720.0,
            614984330664.0,
            19815037200.0,
            312129649.0,
        )
        q = (
            83680955540582400.0,
            -39144431255529600.0,
            8411304436254720.0,
            -1081869058670400.0,
            90503101180800.0,
            -4959510549840.0,
            166265789760.0,
            -2658297528.0,
        )
        P = (h * h) * (
            mf._smart_matrix_product(
                H.A,
                p[7] * H.A6 + p[5] * H.A4 + p[3] * H.A2 + p[1] * H.ident,
                structure=H.structure,
            )
            + p[6] * H.A6
            + p[4] * H.A4
            + p[2] * H.A2
            + p[0] * H.ident
        )
        Q = (
            mf._smart_matrix_product(
                H.A,
                q[7] * H.A6 + q[5] * H.A4 + q[3] * H.A2 + q[1] * H.ident,
                structure=H.structure,
            )
            + q[6] * H.A6
            + q[4] * H.A4
            + q[2] * H.A2
            + q[0] * H.ident
        )
        return _solve_P_Q_2(P, Q, H.structure)

    if pade <= 9:
        p = (
            69491589579005577984000.0,
            13362425784392593708800.0,
            2733673676598876211200.0,
            274218651095712523200.0,
            24016200124299102720.0,
            1393079626219366800.0,
            63177322465033920.0,
            1972643393629480.0,
            40190548856040.0,
            403978495031.0,
        )
        q = (
            138983179158011155968000.0,
            -65930601203222249894400.0,
            14675286699176463360000.0,
            -2017982140321398451200.0,
            189583633133409715200.0,
            -12669377873982429600.0,
            604988181888330240.0,
            -20009929231749600.0,
            418494695659920.0,
            -4247085597370.0,
        )
        P = (h * h) * (
            mf._smart_matrix_product(
                H.A,
                p[9] * H.A8 + p[7] * H.A6 + p[5] * H.A4 + p[3] * H.A2 + p[1] * H.ident,
                structure=H.structure,
            )
            + p[8] * H.A8
            + p[6] * H.A6
            + p[4] * H.A4
            + p[2] * H.A2
            + p[0] * H.ident
        )
        Q = (
            mf._smart_matrix_product(
                H.A,
                q[9] * H.A8 + q[7] * H.A6 + q[5] * H.A4 + q[3] * H.A2 + q[1] * H.ident,
                structure=H.structure,
            )
            + q[8] * H.A8
            + q[6] * H.A6
            + q[4] * H.A4
            + q[2] * H.A2
            + q[0] * H.ident
        )
        return _solve_P_Q_2(P, Q, H.structure)

    # second, try direct solution:
    n = H.A.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            lup = la.lu_factor(H.A)
            # decomp maybe worked ... check it:
            # I = A\(expm(A*h) - Ident)
            I_test = la.lu_solve(lup, h * (E - np.eye(n)))
            if np.allclose(I_test, I):
                # I2 = A\(expm(A*h)*h - A\(expm(A*h) - Ident))
                return la.lu_solve(lup, h * (E * h - I_test))
        except RuntimeWarning:
            pass

    # finally, use taylor series:
    msg = (
        "Using power series expansion directly for `I2` (see"
        " `expmint`).\nRecommendation: use a finer time step."
    )
    warnings.warn(msg, RuntimeWarning)
    I2 = np.eye(n) / 2
    term = H.A
    j = 1.0
    tol = 1e-15
    maxloops = 200
    while abs(term).max() > tol * abs(E).max() and j < maxloops:
        j += 1.0
        I2 += term / (j + 1)
        term = term.dot(H.A) / j
    if j >= maxloops:
        raise RuntimeError(
            f"maximum loops ({maxloops}) exceeded for power series expansion"
        )
    return (h * h) * I2


def _solve_P_Q_2(P, Q, structure=None):
    """
    A helper function for expmint -- modified from the scipy source.

    Parameters
    ----------
    P : ndarray
        Pade numerator.
    Q : ndarray
        Pade denominator.
    structure : str, optional
        A string describing the structure of both matrices `P` and
        `Q`.  Only `upper_triangular` is currently supported.

    Notes
    -----
    The `structure` argument is inspired by similar args
    for theano and cvxopt functions.

    """
    if mf.isspmatrix(P):
        return mf.spsolve(Q, P)
    elif structure is None:
        return mf.solve(Q, P)
    elif structure == mf.UPPER_TRIANGULAR:
        return mf.solve_triangular(Q, P)
    else:
        raise ValueError("unsupported matrix structure: " + str(structure))


def expmint_pow(A, h):
    """
    Compute the matrix exponential and its integrals using the power
    series expansion.

    Parameters
    ----------
    A : (M, M) array_like or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated
    h : scalar
        Time step

    Returns
    -------
    E : (M, M) ndarray
        Matrix exponential of `A*h`: exp(A*h)
    I : (M, M) ndarray
        Integral of exp(A*t) dt from 0 to h
    I2 : (M, M) ndarray
        Integral of exp(A*t)*t dt from 0 to h

    Notes
    -----
    This routine is a simple brute-force alternative to the more
    elegant and optimized :func:`expmint`. The power series expansions
    for these matrices are (I = identity):

    .. code-block:: none

      E = I + A*h + (A*h)**2/2! + (A*h)**3/3! + ...
      I1 = h*(I + A*h/2 + (A*h)**2/3! + (A*h)**3/4! + ...)
      I2 = h*h*(I/2 + A*h/3 + (A*h)**2/(4*2!) + (A*h)**3/(5*3!) + ...)

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> e, i, i2 = expmint.expmint_pow(a, .05)
    >>> e
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])

    >>> i
    array([[ 0.052 ,  0.0034,  0.0048],
           [ 0.0067,  0.0583,  0.01  ],
           [ 0.0114,  0.0133,  0.0651]])

    >>> i2
    array([[ 0.0013,  0.0001,  0.0002],
           [ 0.0002,  0.0015,  0.0003],
           [ 0.0004,  0.0005,  0.0018]])

    """
    Ah = A * h
    E = np.eye(A.shape[0])
    Int1 = E.copy()
    Int2 = E / 2
    term = Ah
    j = 1.0
    tol = 1e-15
    maxloops = 200
    while abs(term).max() > tol * abs(E).max() and j < maxloops:
        j += 1.0
        E += term
        Int1 += term / j
        Int2 += term / (j + 1.0)
        term = term.dot(Ah) / j
    if j >= maxloops:
        raise RuntimeError(
            f"maximum loops ({maxloops}) exceeded for power series expansion"
        )
    return E, h * Int1, h * h * Int2


def _procBhalf(E, P, Q, order, B, half):
    """Helper function for getEPQ1 and getEPQ_pow"""
    if B is not None:
        P = P.dot(B)
        if order == 1:
            Q = Q.dot(B)
    elif half:
        n = P.shape[1]
        if n & 1:
            raise ValueError(
                "`A` must have an even number of rows/cols (or use ``half=False``"
            )
        n = n // 2
        P = P[:, :n]
        if order == 1:
            Q = Q[:, :n]
    return E, P, Q


def getEPQ1(A, h, order=1, B=None, half=False):
    """
    Returns E, P, Q for the exponential solver given the state-space
    matrix `A`.

    Parameters
    ----------
    A : 2d ndarray
        The state-space matrix:  ``xdot = A x + B u``
    h : scalar
        Time step.
    order : integer, optional

        - 0 for the zero order hold (force stays constant across
          time step)
        - 1 for the 1st order hold (force can vary linearly across
          time step)

    B : d2 ndarray or None; optional
        If array, it multiplies the inputs; if None, it is assumed
        identity.
    half : bool; optional
        If `B` is a 2d ndarray, `half` is ignored. Otherwise, if
        `half` is False, a full size identity (same size as `A`) is
        used for `B`. If `half` is True, only the first half of the
        columns are retained (which is handy for converting a 2nd
        order ODE into a 1st order ODE as
        :class:`pyyeti.ode.SolveExp2` does -- where there are force
        inputs only for the first half of the equations).

    Returns
    -------
    E, P, Q : 2d ndarrays, except if ``order == 0``, ``Q = 0.``
        These are the coefficient matrices used to solve the ODE::

            for j in range(nt):
                d[:, j+1] = E*d[:, j] + P*F[:, j] + Q*F[:, j+1]

    Notes
    -----
    Normally, :func:`getEPQ` would be called and that routine will
    call this one or :func:`getEPQ2`.

    `E` is the matrix exponential ``exp(A*h)`` and `P` and `Q` are
    functions of the integral(s) of the matrix exponential::

            if order == 1:
                E, I, I2 = expmint(A, h, 1)
                P = (I2/h) @ B
                Q = (I - I2/h) @ B
            else:
                E, I = expmint(A, h)
                P = I @ B
                Q = 0.

    See also
    --------
    :func:`expmint`, :class:`pyyeti.ode.SolveExp1`,
    :class:`pyyeti.ode.SolveExp2`

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = np.array([[0, 1, 0]]).T
    >>> E, P, Q = expmint.getEPQ1(A, .05, order=1, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0024],
           [ 0.0308],
           [ 0.0091]])
    >>> Q
    array([[ 0.0011],
           [ 0.0276],
           [ 0.0041]])
    >>> E, P, Q = expmint.getEPQ1(A, .05, order=0, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0034],
           [ 0.0583],
           [ 0.0133]])
    >>> Q
    0.0
    """
    if order == 1:
        E, I, I2 = expmint(A, h, 1)
        P = I2 / h
        Q = I - P
    else:
        E, P = expmint(A, h)
        Q = 0.0
    return _procBhalf(E, P, Q, order, B, half)


def getEPQ_pow(A, h, order=1, B=None, half=False):
    """
    Returns E, P, Q for the exponential solver given the state-space
    matrix `A`. Uses the power series expansion directly.

    Parameters
    ----------
    A : 2d ndarray
        The state-space matrix:  ydotD - A y = f
    h : scalar
        Time step.
    order : integer, optional

        - 0 for the zero order hold (force stays constant across time
          step)
        - 1 for the 1st order hold (force can vary linearly across
          time step)

    B : d2 ndarray or None; optional
        If array, it multiplies the inputs; if None, it is assumed
        identity.
    half : bool; optional
        If `B` is a 2d ndarray, `half` is ignored. Otherwise, if
        `half` is False, a full size identity (same size as `A`) is
        used for `B`. If `half` is True, only the first half of the
        columns are retained (which is handy for converting a 2nd
        order ODE into a 1st order ODE as
        :class:`pyyeti.ode.SolveExp2` does -- where there are force
        inputs only for the first half of the equations).

    Returns
    -------
    E, P, Q : 2d ndarrays, except if ``order == 0``, ``Q = 0.``
        These are the coefficient matrices used to solve the ODE::

            for j in range(nt):
                d[:, j+1] = E*d[:, j] + P*F[:, j] + Q*F[:, j+1]

    Notes
    -----
    This routine is primarily for testing purposes; in general, use
    :func:`getEPQ` instead.

    `E` is the matrix exponential ``exp(A*h)`` and `P` and `Q` are
    functions of the integral(s) of the matrix exponential::

            if order == 1:
                E, I, I2 = expmint(A, h, 1)
                P = (I2/h) @ B
                Q = (I - I2/h) @ B
            else:
                E, I = expmint(A, h)
                P = I @ B
                Q = 0.

    See also
    --------
    :func:`getEPQ`, :func:`expmint_pow`,
    :class:`pyyeti.ode.SolveExp1`, :class:`pyyeti.ode.SolveExp2`

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = np.array([[0, 1, 0]]).T
    >>> E, P, Q = expmint.getEPQ_pow(A, .05, order=1, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0024],
           [ 0.0308],
           [ 0.0091]])
    >>> Q
    array([[ 0.0011],
           [ 0.0276],
           [ 0.0041]])
    >>> E, P, Q = expmint.getEPQ_pow(A, .05, order=0, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0034],
           [ 0.0583],
           [ 0.0133]])
    >>> Q
    0.0
    """
    E, I, I2 = expmint_pow(A, h)
    if order == 1:
        P = I2 / h
        Q = I - P
    else:
        P = I
        Q = 0.0
    return _procBhalf(E, P, Q, order, B, half)


class _ExpmPadeHelper_SS(mf._ExpmPadeHelper):
    """
    Class derived from scipy.sparse.linalg.matfuncs._ExpmPadeHelper
    to assist in calculating the matrix exponential and integrals for
    a state-space matrix. See :func:`getEPQ`.
    """

    def __init__(self, A, ssA, order):
        """
        Initialize the object.

        Parameters
        ----------
        A : square ndarray
            The matrix to be exponentiated; formed in such a way to
            get the matrix exponential of `ssA` and its integrals.
            See :func:`getEPQ`.
        ssA : square ndarray
            State-space 'A' matrix; `A` has `ssA` in upper left.
        order : integer
            Specifies how to handle forces: ``order=1`` for first
            order hold; ``order=0`` for zero order hold.
        """
        mf._ExpmPadeHelper.__init__(self, A, structure=None, use_exact_onenorm=True)
        self.ssA = ssA
        self.order = order
        self.nss = ssA.shape[0]
        self.shape = (self.nss, A.shape[1])

    @property
    def A2(self):
        if self._A2 is None:
            n = self.nss
            if self.order == 0:
                self._A2 = np.dot(self.ssA, self.A[:n])
            else:
                i = (self.A.shape[1] - n) // 2
                self._A2 = np.zeros(self.shape, float)
                self._A2[:, : n + i] = np.dot(self.ssA, self.A[:n, : n + i])
                self._A2[:, n + i :] = self.A[:n, n : n + i]
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            n = self.nss
            self._A4 = np.dot(self.A2[:, :n], self.A2)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            n = self.nss
            self._A6 = np.dot(self.A4[:, :n], self.A2)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            n = self.nss
            self._A8 = np.dot(self.A6[:, :n], self.A2)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            n = self.nss
            self._A10 = np.dot(self.A4[:, :n], self.A6)
        return self._A10

    def pade3(self):
        b = (120.0, 60.0, 12.0, 1.0)
        n = self.nss
        U = b[1] * self.A
        V = b[0] * self.ident
        U[:n] += np.dot(self.ssA, b[3] * self.A2)
        V[:n] += b[2] * self.A2
        return U, V

    def pade5(self):
        b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
        n = self.nss
        U = b[1] * self.A
        V = b[0] * self.ident
        U[:n] += np.dot(self.ssA, b[5] * self.A4 + b[3] * self.A2)
        V[:n] += b[4] * self.A4 + b[2] * self.A2
        return U, V

    def pade7(self):
        b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
        n = self.nss
        U = b[1] * self.A
        V = b[0] * self.ident
        U[:n] += np.dot(self.ssA, (b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2))
        V[:n] += b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2
        return U, V

    def pade9(self):
        b = (
            17643225600.0,
            8821612800.0,
            2075673600.0,
            302702400.0,
            30270240.0,
            2162160.0,
            110880.0,
            3960.0,
            90.0,
            1.0,
        )
        n = self.nss
        U = b[1] * self.A
        V = b[0] * self.ident
        U[:n] += np.dot(
            self.ssA,
            (b[9] * self.A8 + b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2),
        )
        V[:n] += b[8] * self.A8 + b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2
        return U, V

    def pade13_scaled(self, s):
        b = (
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        )
        B = self.A * 2 ** -s
        B2 = self.A2 * 2 ** (-2 * s)
        B4 = self.A4 * 2 ** (-4 * s)
        B6 = self.A6 * 2 ** (-6 * s)
        n = self.nss
        U = b[1] * B
        V = b[0] * self.ident
        U2 = np.dot(B6[:, :n], b[13] * B6 + b[11] * B4 + b[9] * B2)
        V2 = np.dot(B6[:, :n], b[12] * B6 + b[10] * B4 + b[8] * B2)
        U[:n] += np.dot(B[:n, :n], U2 + b[7] * B6 + b[5] * B4 + b[3] * B2)
        V[:n] += V2 + b[6] * B6 + b[4] * B4 + b[2] * B2
        return U, V


def _expm_SS(A, ssA, order):  # , use_exact_onenorm='auto'):
    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper_SS(A, ssA, order)
    structure = None

    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and mf._ell(h.A, 3) == 0:
        U, V = h.pade3()
        return mf._solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and mf._ell(h.A, 5) == 0:
        U, V = h.pade5()
        return mf._solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and mf._ell(h.A, 7) == 0:
        U, V = h.pade7()
        return mf._solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e000 and mf._ell(h.A, 9) == 0:
        U, V = h.pade9()
        return mf._solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + mf._ell(2 ** -s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = mf._solve_P_Q(U, V, structure=structure)
    # X = r_13(A)^(2^s) by repeated squaring.
    for _ in range(s):
        X = X.dot(X)
    return X


def getEPQ2(A, h, order=1, B=None, half=False):
    """
    Returns E, P, Q for the exponential solver given the state-space
    matrix `A`.

    Parameters
    ----------
    A : 2d ndarray
        The state-space matrix:  ``xdot = A x + B u``
    h : scalar
        Time step.
    order : integer, optional

        - 0 for the zero order hold (force stays constant across
          time step)
        - 1 for the 1st order hold (force can vary linearly across
          time step)

    B : d2 ndarray or None; optional
        If array, it multiplies the inputs; if None, it is assumed
        identity.
    half : bool; optional
        If `B` is a 2d ndarray, `half` is ignored. Otherwise, if
        `half` is False, a full size identity (same size as `A`) is
        used for `B`. If `half` is True, only the first half of the
        columns are retained (which is handy for converting a 2nd
        order ODE into a 1st order ODE as
        :class:`pyyeti.ode.SolveExp2` does -- where there are force
        inputs only for the first half of the equations).

    Returns
    -------
    E, P, Q : 2d ndarrays, except if ``order == 0``, ``Q = 0.``
        These are the coefficient matrices used to solve the ODE::

            for j in range(nt):
                d[:, j+1] = E*d[:, j] + P*F[:, j] + Q*F[:, j+1]

    Notes
    -----
    Normally, :func:`getEPQ` would be called and that routine will
    call this one or :func:`getEPQ1`.

    This routine is an alternative to :func:`getEPQ1` and is
    generally slower but more robust for large time steps. (If `B` has
    only a few columns, it could also be faster than
    :func:`getEPQ1`.) `E` is the matrix exponential ``exp(A*h)`` and
    `P` and `Q` are functions of the integral(s) of the matrix
    exponential. They are calculated as follows (text from
    :func:`scipy.signal.lsim`).

    If order == 0::

        Zero-order hold
        Algorithm: to integrate from time 0 to time dt, we solve
          xdot = A x + B u,  x(0) = x0
          udot = 0,          u(0) = u0.

        Solution is
          [ x(dt) ]       [ A*dt   B*dt ] [ x0 ]
          [ u(dt) ] = exp [  0     0    ] [ u0 ]

    The `E`, and `P` matrices are partitions of the matrix
    exponential and `Q` is zero.

    If order == 1::

        Linear interpolation between steps
        Algorithm: to integrate from time 0 to time dt, with
        linear interpolation between inputs u(0) = u0 and u(dt) = u1,
        we solve:
          xdot = A x + B u,        x(0) = x0
          udot = (u1 - u0) / dt,   u(0) = u0.

        Solution is
          [ x(dt) ]       [ A*dt  B*dt  0 ] [  x0   ]
          [ u(dt) ] = exp [  0     0    I ] [  u0   ]
          [u1 - u0]       [  0     0    0 ] [u1 - u0]

    The `E`, `P` and `Q` matrices are partitions of the matrix
    exponential.

    See also
    --------
    :func:`getEPQ1`, :func:`getEPQ_pow`,
    :class:`pyyeti.ode.SolveExp1`, :class:`pyyeti.ode.SolveExp2`

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = np.array([[0, 1, 0]]).T
    >>> E, P, Q = expmint.getEPQ2(A, .05, order=1, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0024],
           [ 0.0308],
           [ 0.0091]])
    >>> Q
    array([[ 0.0011],
           [ 0.0276],
           [ 0.0041]])
    >>> E, P, Q = expmint.getEPQ2(A, .05, order=0, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0034],
           [ 0.0583],
           [ 0.0133]])
    >>> Q
    0.0
    """
    n = A.shape[0]
    if B is not None:
        i = B.shape[1]
    else:
        if half:
            if n & 1:
                raise ValueError(
                    "`A` must have an even number "
                    "of rows/cols (or use ``half=False``"
                )
            i = n // 2
        else:
            i = n
        B = np.eye(i)
    r = B.shape[0]
    Ah = A * h
    Bh = B * h
    if order == 1:
        N = n + 2 * i
        M = np.zeros((N, N), float)
        M[:n, :n] = Ah
        M[:r, n : n + i] = Bh
        M[n : n + i, n + i :] = np.eye(i)
        # start = time.time()
        # EM1 = la.expm(M, order)
        # print('1 la.expm et = ', time.time()-start)

        # start = time.time()
        EM = _expm_SS(M, Ah, order)
        # print('1 expm_SS et = ', time.time()-start)
        # print('error :', abs(EM-EM1).max())
        E = EM[:n, :n]
        Q = EM[:n, n + i :]
        P = EM[:n, n : n + i] - Q
    elif order == 0:
        M = np.zeros((n + i, n + i), float)
        M[:n, :n] = Ah
        M[:r, n:] = Bh
        # start = time.time()
        # EM1 = la.expm(M, order)
        # print('0 la.expm et = ', time.time()-start)

        # start = time.time()
        EM = _expm_SS(M, Ah, order)
        # print('0 expm_SS et = ', time.time()-start)
        # print('error :', abs(EM-EM1).max())
        E = EM[:n, :n]
        P = EM[:n, n:]
        Q = 0.0
    else:
        raise ValueError("`order` must be 0 or 1")
    return E, P, Q


def getEPQ(A, h, order=1, B=None, half=False):
    """
    Returns E, P, Q for the exponential solver given the state-space
    matrix `A`.

    Parameters
    ----------
    A : 2d ndarray
        The state-space matrix:  ``xdot = A x + B u``
    h : scalar
        Time step.
    order : integer, optional

        - 0 for the zero order hold (force stays constant across
          time step)
        - 1 for the 1st order hold (force can vary linearly across
          time step)

    B : d2 ndarray or None; optional
        If array, it multiplies the inputs; if None, it is assumed
        identity.
    half : bool; optional
        If `B` is a 2d ndarray, `half` is ignored. Otherwise, if
        `half` is False, a full size identity (same size as `A`) is
        used for `B`. If `half` is True, only the first half of the
        columns are retained (which is handy for converting a 2nd
        order ODE into a 1st order ODE as
        :class:`pyyeti.ode.SolveExp2` does -- where there are force
        inputs only for the first half of the equations).

    Returns
    -------
    E, P, Q : 2d ndarrays, except if ``order == 0``, ``Q = 0.``
        These are the coefficient matrices used to solve the ODE::

            for j in range(nt):
                d[:, j+1] = E*d[:, j] + P*F[:, j] + Q*F[:, j+1]

    Notes
    -----
    This routine calls either :func:`getEPQ1` or :func:`getEPQ2` for
    the bulk of the work. If the 1-norm of `A` is less than
    2.097847961257068 [#exp4]_, :func:`getEPQ1` is called; otherwise,
    :func:`getEPQ2` is called.

    References
    ----------
    .. [#exp4] Nicholas J. Higham (2005)
          "The Scaling and Squaring Method for the Matrix Exponential
          Revisited."
          SIAM Journal on Matrix Analysis and Applications.
          Vol 26, No. 4, pp 1179-1193.

    See also
    --------
    :func:`getEPQ_pow`, :class:`pyyeti.ode.SolveExp1`,
    :class:`pyyeti.ode.SolveExp2`

    Examples
    --------
    >>> from pyyeti import expmint
    >>> import numpy as np
    >>> import scipy.linalg as la
    >>> np.set_printoptions(4)
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> B = np.array([[0, 1, 0]]).T
    >>> E, P, Q = expmint.getEPQ(A, .05, order=1, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0024],
           [ 0.0308],
           [ 0.0091]])
    >>> Q
    array([[ 0.0011],
           [ 0.0276],
           [ 0.0041]])
    >>> E, P, Q = expmint.getEPQ(A, .05, order=0, B=B)
    >>> E
    array([[ 1.0996,  0.1599,  0.2202],
           [ 0.3099,  1.3849,  0.46  ],
           [ 0.5202,  0.61  ,  1.6998]])
    >>> P
    array([[ 0.0034],
           [ 0.0583],
           [ 0.0133]])
    >>> Q
    0.0
    """
    norm1 = h * np.linalg.norm(A, 1)
    if norm1 <= 2.097847961257068:
        return getEPQ1(A, h, order, B, half)
    return getEPQ2(A, h, order, B, half)
