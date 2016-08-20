# -*- coding: utf-8 -*-
"""
Collection of tools for analysis of Craig-Bampton models.
"""

import numpy as np
import math
import scipy.linalg as linalg
import scipy.sparse.linalg as sp_la
from collections import abc
import numbers
from types import SimpleNamespace
from warnings import warn
from pyyeti import n2p, locate, ytools, writer, ode


def cbtf(m, b, k, a, freq, bset, save=None):
    r"""
    Compute frequency domain responses given i/f accel for a CB model.

    Parameters
    ----------
    m : 2d ndarray
        CB mass matrix. This routine does not assume the modal part is
        diagonal, but is assumed to be symmetric. Can be complex.
    b : 2d ndarray
        CB damping matrix. Modal part can be fully populated. Can be
        complex.
    k : 2d ndarray
        CB stiffness matrix. This routine does not assume the modal
        part is diagonal. Can be complex.
    a : 1d or 2d ndarray
        B-set acceleration to enforce; either bset x freq sized or a
        b-set length vector. If a b-set length vector, it is
        internally expanded and used for all frequencies. This is
        handy for constant (such as unity) base inputs.
    freq : 1d ndarray
        Frequency vector in Hz for solution.
    bset : 1d ndarray
        Index partition vector for the b-set; eg, np.arange(6)
    save : None or dict
        When using multiple `a` inputs, set `save` to an empty dict;
        this routine will put items in `save` to avoid unnecessary
        calculations.

    Returns
    -------
    A record (SimpleNamespace class) with the members:

    frc : complex 2d ndarray
        B-set force that was required to accelerate according to input
        `a`.
    a : complex 2d ndarray
        B+Q-set accelerations (B-set part should match input `a`).
    d : complex 2d ndarray
        B+Q-set displacements.
    v : complex 2d ndarray
        B+Q-set velocities.
    freq : 1d ndarray
        The frequency vector (same as `freq`).

    Notes
    -----
    This routine is normally used to compute the transfer function by
    simply setting `a` equal to unity (or 1g) acceleration for the DOF
    of interest and zeros everywhere else.

    This is analogous to doing a seismic mass baseshake, so the
    interface force F should have peaks at fixed-base Craig-Bampton
    frequencies. To exercise a free-free system, see
    :class:`ode.SolveUnc` or :func:`ode.FreqDirect`.

    The Craig-Bampton equations of motion in the frequency domain are:

    .. math::
        \left[
            \begin{array}{cc} M_{bb} & M_{bq} \\ M_{qb} & M_{qq}
            \end{array}
        \right] \left\{
            \begin{array}{c} \ddot{X}_b(\Omega) \\ \ddot{X}_q(\Omega)
            \end{array}
        \right\} + \left[
            \begin{array}{cc} B_{bb} & B_{bq} \\ B_{qb} & B_{qq}
            \end{array}
        \right] \left\{
            \begin{array}{c} \dot{X}_b(\Omega) \\ \dot{X}_q(\Omega)
            \end{array}
        \right\} +  \left[
            \begin{array}{cc} K_{bb} & 0 \\ 0 & K_{qq} \end{array}
        \right] \left\{
            \begin{array}{c} X_b(\Omega) \\ X_q(\Omega)
            \end{array}
        \right\} = \left\{
        \begin{array}{c} F_b(\Omega) \\ 0 \end{array} \right\}

    The input `a` is :math:`\ddot{X}_b(\Omega)`. Rearranging the bottom
    equation and using
    :math:`\dot{X}_b(\Omega)= -i\ddot{X}_b(\Omega)/\Omega` gives:

    .. math::
        \begin{aligned}
        M_{qq} \ddot{X}_q(\Omega) + B_{qq} \dot{X}_q(\Omega) +
        K_{qq} X_q(\Omega) &= -M_{qb} \ddot{X}_b(\Omega) -
        B_{qb} \dot{X}_b(\Omega) \\
        &= \left(-M_{qb} + i B_{qb}/\Omega \right) \ddot{X}_b(\Omega)
        \end{aligned}

    That equation is solved via :func:`ode.SolveUnc.fsolve`. After
    solution, :math:`F_b(\Omega)` is computed from the top equation
    above.

    Examples
    --------
    Use :func:`cbtf` on a very simple 1-D spring mass CB component::

        [m1]--/\/\--[m2]--/\/\--[m3]   <-- m1 is the bset DOF
               k1          k2

        m1 = 4, k1 = 9000, m2 = 5, k2 = 3000, m3 = 7

    .. plot::
        :context: close-figs

        >>> from pyyeti import cb
        >>> import numpy as np
        >>> import scipy.linalg as la
        >>> import matplotlib.pyplot as plt
        >>> np.set_printoptions(precision=4, suppress=True)
        >>> m = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 7]])
        >>> k = np.array([[9000, -9000, 0], [-9000, 12000, -3000],
        ...              [0, -3000, 3000]])

        Compute constraint modes and normal modes:

        >>> psi = [1, 1, 1]
        >>> w, phi = la.eigh(k[1:, 1:], m[1:, 1:])

        Assemble CB transformation matrix, bset first:

        >>> T = np.zeros((3, 3))
        >>> T[:, 0] = psi
        >>> T[1:, 1:] = phi

        Compute CB mass and stiffness:

        >>> mcb = T.T @ m @ T
        >>> kcb = T.T @ k @ T

        Make 3% modal damping matrix:

        >>> zeta = .03
        >>> bcb = np.diag(np.hstack((0, 2*zeta*np.sqrt(w))))

        Set up frequency for solution and call :func:`cbft`:

        >>> outfreq = np.arange(.1, 10, .01)
        >>> tf = cb.cbtf(mcb, bcb, kcb, 1, outfreq, 0)

        Recover physical accelerations and make some plots:

        >>> a = T @ tf.a
        >>> d = T @ tf.d
        >>> fig = plt.figure('cbtf demo')
        >>> ax = plt.subplot(211)
        >>> lines = ax.plot(outfreq, np.abs(tf.frc).T, label='Force')
        >>> lines += ax.plot(outfreq, np.abs(a).T)
        >>> _ = ax.legend(lines,
        ...               ('Force', 'Acce 1', 'Acce 2', 'Acce 3'),
        ...               loc='best')
        >>> _ = ax.set_title('Magnitude')
        >>> _ = ax.set_xlabel('Freq (Hz)')
        >>> ax.set_yscale('log')
        >>> _ = ax.set_ylim([.5, 200])
        >>> ax = plt.subplot(212)
        >>> lines = ax.plot(outfreq, np.angle(tf.frc, deg=True).T)
        >>> lines += ax.plot(outfreq, np.angle(a, deg=True).T)
        >>> _ = ax.legend(lines,
        ...               ('Force', 'Acce 1', 'Acce 2', 'Acce 3'),
        ...               loc='best')
        >>> _ = ax.set_title('Phase')
        >>> _ = ax.set_xlabel('Freq (Hz)')
        >>> plt.tight_layout()
    """
    freq = np.atleast_1d(freq).ravel()
    Omega = 2*math.pi*freq
    lenf = len(Omega)
    a = np.atleast_1d(a)
    bset = np.atleast_1d(bset)
    if a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1):
        a = np.dot(a.reshape(-1, 1), np.ones((1, lenf)))
    r, c = a.shape
    if c != lenf:
        raise ValueError('`a` is not compatibly sized with `freq`.')
    if r != len(bset):
        raise ValueError('number of rows in `a` not compatible '
                         'with `bset`.')
    frc = np.zeros((r, lenf), dtype=complex)
    bset = np.atleast_1d(bset).ravel()
    lt = m.shape[0]
    qset = locate.flippv(bset, lt)

    if qset.size == 0:
        accel = a.copy()
        displ = (-1/Omega**2) * accel
        veloc = 1j*(Omega * displ)
        frc = m @ accel + b @ veloc + k @ displ
    else:
        tf = None
        if isinstance(save, abc.MutableMapping):
            try:
                tf = save['tf']
            except KeyError:
                pass
        if tf is None:
            qq = np.ix_(qset, qset)
            tf = ode.SolveUnc(m[qq], b[qq], k[qq], rb=[])
            if isinstance(save, abc.MutableMapping):
                save['tf'] = tf

        bb = np.ix_(bset, bset)
        qb = np.ix_(qset, bset)
        v = 1j*a/Omega
        f = b[qb] @ v - m[qb] @ a
        sol = tf.fsolve(f, freq)

        displ = np.zeros((lt, lenf), dtype=complex)
        accel = displ.copy()
        displ[bset] = (-1/Omega**2) * a
        displ[qset] = sol.d
        veloc = 1j*(Omega * displ)
        accel[bset] = a
        accel[qset] = sol.a
        frc = (m[bset] @ accel + b[bset] @ veloc +
               k[bb] @ displ[bset])
    return SimpleNamespace(frc=frc, a=accel, d=displ,
                           v=veloc, freq=freq)


def cbreorder(M, b, drm=False, last=False):
    """
    Reorder either a Craig-Bampton mass or stiffness matrix, or a
    Craig-Bampton data recovery matrix.

    Parameters
    ----------
    M : 2d ndarray
        Craig-Bampton mass or stiffness, or Craig-Bampton data recovery
        matrix (DRM). Must be square if `drm` is false.
    b : 1d array
        A vector containing the indices of the b-set DOF in the desired
        order (uses zero offset).
    drm : bool
        If true, `M` is treated as a data recovery matrix. If false,
        `M` is treated as mass or stiffness (and `M` must b.
    last : bool
        If true, reorder such that b-set is last; if false, put b-set
        first.

    Returns
    -------
    M2 : 2d ndarray
        The reordered matrix. If `M` is mass or stiffness, both rows
        and columns are reordered, maintaining symmetry. If `M` is a
        DRM, only columns are reordered.

    Raises
    ------
    ValueError
        If `M` is not square when `drm` is false.

    Notes
    -----
    The size of `b` is checked and if it is not an even multiple of 6,
    a warning message is printed.

    Note that this routine will also reorder DOF within the b-set.
    Just specify `b` in the order you want. See the example below
    where, just for demonstration purposes, the order of the b-set is
    reversed (along with putting b-set in front of the q-set).

    See also
    --------
    :func:`cbcheck`, :func:`cbconvert`

    Examples
    --------
    For a first example, generate 8x8 dummy matrix with the 2 q-set
    first, followed by 6 b-set. Put the b-set first and reverse their
    order:

    >>> from pyyeti import cb
    >>> import numpy as np
    >>> m = np.dot(np.arange(1, 9).reshape(-1, 1),
    ...            np.arange(2, 10).reshape(1, -1))
    >>> m
    array([[ 2,  3,  4,  5,  6,  7,  8,  9],
           [ 4,  6,  8, 10, 12, 14, 16, 18],
           [ 6,  9, 12, 15, 18, 21, 24, 27],
           [ 8, 12, 16, 20, 24, 28, 32, 36],
           [10, 15, 20, 25, 30, 35, 40, 45],
           [12, 18, 24, 30, 36, 42, 48, 54],
           [14, 21, 28, 35, 42, 49, 56, 63],
           [16, 24, 32, 40, 48, 56, 64, 72]])
    >>> cb.cbreorder(m, np.arange(7, 1, -1))
    array([[72, 64, 56, 48, 40, 32, 16, 24],
           [63, 56, 49, 42, 35, 28, 14, 21],
           [54, 48, 42, 36, 30, 24, 12, 18],
           [45, 40, 35, 30, 25, 20, 10, 15],
           [36, 32, 28, 24, 20, 16,  8, 12],
           [27, 24, 21, 18, 15, 12,  6,  9],
           [ 9,  8,  7,  6,  5,  4,  2,  3],
           [18, 16, 14, 12, 10,  8,  4,  6]])

    For another example, generate a 3x5 DRM with 4 b-set followed by
    and 1 q-set. Put b-set last.

    >>> drm = np.arange(1, 16).reshape(3, 5)
    >>> drm
    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15]])
    >>> cb.cbreorder(drm, [0, 1, 2, 3], drm=True, last=True)
    array([[ 5,  1,  2,  3,  4],
           [10,  6,  7,  8,  9],
           [15, 11, 12, 13, 14]])
    """
    lt = np.size(M, 1)
    if not drm and lt != np.size(M, 0):
        raise ValueError('`M` must be square when `drm` is false')

    b = np.atleast_1d(b).ravel()
    lb = len(b)
    lq = lt - lb

    if (lb // 6) * 6 != lb:
        warn('b-set not a multiple of 6.', RuntimeWarning)

    if lq == 0:
        if drm:
            M = M[:, b]
        else:
            M = M[np.ix_(b, b)]
    else:
        q = locate.flippv(b, lt)
        if last:
            pv = np.hstack((q, b))
        else:
            pv = np.hstack((b, q))
        if drm:
            M = M[:, pv]
        else:
            M = M[np.ix_(pv, pv)]
    return M


def _get_conv_factors(conv):
    if conv == 'm2e':
        lengthconv = 1/0.0254
        massconv = 0.005710147154735817
    elif conv == 'e2m':
        lengthconv = 0.0254
        massconv = 175.12683524637913
    else:
        lengthconv, massconv = conv
    return lengthconv, massconv


def _uset_convert(uset, uref, conv):
    lengthconv = _get_conv_factors(conv)[0]
    uset = uset.copy()
    pv = uset[:, 1] == 1
    uset[pv, 3:] *= lengthconv
    pv = uset[:, 1] == 3
    uset[pv, 3:] *= lengthconv
    try:
        if len(uref) == 3:
            uref = np.atleast_1d(uref) * lengthconv
    except TypeError:
        pass
    return uset, uref


def cbconvert(M, b, conv='m2e', drm=False):
    r"""
    Apply unit conversion transform to either a Craig-Bampton mass or
    stiffness matrix, or a Craig-Bampton data recovery matrix.

    Parameters
    ----------
    M : 2d array
        Craig-Bampton mass or stiffness, or Craig-Bampton data recovery
        matrix (DRM). Must be square if `drm` is false.
    b : 1d array
        A vector containing the indices of the b-set DOF in the desired
        order (uses zero offset).
    conv : 2-element array_like or string; optional
        If 2-element array_like, it is::

            (length_conversion, mass_conversion)

        If string, it is one of:

            * 'm2e' (convert from metric to English)
            * 'e2m' (convert from English to metric)

        The string form assumes units of meter & kg, and inch &
        lbf*s**2/inch (slinch).
    drm : bool
        If true, `M` is treated as a data recovery matrix. If false,
        `M` is treated as mass or stiffness (and `M` must be square).

    Returns
    -------
    Mconv : 2d ndarray
        The converted matrix.

    Notes
    -----
    Here is a table showing equivalent settings for `conv`:

        ======    =========================================
        `conv`    array_like equivalent
        ======    =========================================
        'm2e'     (39.37007874015748, 0.005710147154735817)
        'e2m'     (0.0254, 175.12683524637913)
        ======    =========================================

    If `M` is mass or stiffness, symmetry is maintained and units are
    converted using two diagonal matrices: C and D. C converts the
    Craig-Bampton (CB) b-set and q-set units from coupled system units
    to units compatible with `M` as input. D converts force units in
    the opposite direction (from units compatible with `M` as input to
    coupled system units). Conversion of a DRM only uses C.

    The units conversion is most easily understood by example. Assume
    `M` is the CB mass in metric (SI) units and let 'x' be the CB
    b-set and q-set degrees of freedom.

    .. math:: F_{si} = M_{si} \cdot {\ddot x}_{si}

    To work with an English coupled system, the forces must be in
    English while :math:`{\ddot x}` is computed in English and must be
    converted to SI:

    .. math::
        F_{eng} = D \cdot F_{si}

        {\ddot x}_{si} = C \cdot {\ddot x}_{eng}

    Therefore:

    .. math::
        F_{eng} = D \cdot M_{si} \cdot C \cdot {\ddot x}_{eng}

    The mass (and stiffness) conversion is therefore:

    .. math::
        M_{eng} = D \cdot M_{si} \cdot C

    The conversion of units for the DRM is similar. Data recovery is
    done via:

    .. math::
        {data\ recovery\ items}_{si} = DRM_{si} \cdot {\ddot x}_{si}

    Or:

    .. math::
        {data\ recovery\ items}_{si} = DRM_{si} \cdot C \cdot
        {\ddot x}_{eng}

    Therefore, the conversion for the DRM is:

    .. math::
        DRM_{eng} = DRM_{si} \cdot C

    Note that the units of the data recovery items as output by the
    DRM are not changed, ie:

    .. math::
        {data\ recovery\ items}_{si} = DRM_{eng} \cdot {\ddot x}_{eng}

    .. warning::
        It is assumed that the b-set are an even multiple of 6 with
        each grid having the three translations 1st.

    Raises
    ------
    ValueError
        If length of `b` is not an even multiple of six.
    ValueError
        If `M` is not square when `drm` is false.

    See also
    --------
    :func:`cbcheck`, :func:`cbcoordchk`, :func:`cbreorder`,
    :func:`cgmass`

    Examples
    --------
    To show and check the conversion, use identity mass or stiffness
    and a row of ones for a DRM and convert them from metric (kg, m,
    sec) to English (lb, in, sec):

    >>> from pyyeti import cb
    >>> import numpy as np
    >>> b = np.arange(6)
    >>> m_or_k = np.eye(8)
    >>> drm = np.ones((1, 8))
    >>> conv = (39.37007874015748, 0.005710147154735817)
    >>> m_or_k_si = cb.cbconvert(m_or_k, b, conv)
    >>> drm_si = cb.cbconvert(drm, b, conv, drm=True)
    >>> # Or, more simply:
    >>> m_or_k_si2 = cb.cbconvert(m_or_k, b, 'm2e')
    >>> drm_si2 = cb.cbconvert(drm, b, 'm2e', drm=True)
    >>> np.allclose(m_or_k_si, m_or_k_si2)
    True
    >>> np.allclose(drm_si, drm_si2)
    True
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> np.diag(m_or_k_si)
    array([ 0.0057,  0.0057,  0.0057,  8.8507,  8.8507,  8.8507,  1.    ,  1.    ])
    >>> drm_si
    array([[ 0.0254,  0.0254,  0.0254,  1.    ,  1.    ,  1.    ,  0.3361,
             0.3361]])

    Now, convert back to metric and compare to original:

    >>> m_or_k_ = cb.cbconvert(m_or_k_si2, b, 'e2m')
    >>> drm_ = cb.cbconvert(drm_si, b, 'e2m', drm=True)
    >>> np.allclose(m_or_k, m_or_k_)
    True
    >>> np.allclose(drm, drm_)
    True
    """
    # for mass or stiffness, form conversion as:  D*M*C
    #    C converts u from OUT to IN
    #    D converts F (F = K*u or M*u'') from IN to OUT
    lt = np.size(M, 1)
    if not drm and lt != np.size(M, 0):
        raise ValueError('`M` must be square when `drm` is false')

    b = np.array(b).ravel()
    lb = len(b)
    lq = lt - lb

    if (lb // 6) * 6 != lb:
        raise ValueError('b-set not a multiple of 6.')

    lengthconv, massconv = _get_conv_factors(conv)
    C = np.ones(lt)
    D = np.ones(lt)
    trn = ytools.mkpattvec([0, 1, 2], lb, 6).ravel()
    rot = trn+3
    C[b[trn]] = 1/lengthconv
    D[b[trn]] = massconv * lengthconv
    D[b[rot]] = massconv * lengthconv**2
    if lq > 0:
        q = locate.flippv(b, lt)
        c = math.sqrt(massconv)*lengthconv
        C[q] = 1/c
        D[q] = c
    M = ytools.multmd(M, C)
    if not drm:
        M = ytools.multmd(D, M)
    return M


def cgmass(m, all6=False):
    r"""
    Compute 6x6 mass at CG given a 6x6 mass matrix.

    Parameters
    ----------
    m : 2d ndarray
        6x6 symmetric mass matrix at some reference point.
    all6 : bool; optional
        If true, return all 6 values; otherwise, only return `mcg`
        and `dxyz`.

    Returns
    -------
    mcg : 2d ndarray
        6x6 mass matrix at CG.
    dxyz : 1d ndarray
        The [x,y,z] distances from the reference point to the cg
        in the coordinate system of that reference point.
    gyr : 1d ndarray
        A 3-element vector containing the radii of gyration
        relative to the CG about X, Y, and Z axes.
    princ_gyr : 1d ndarray
        Same as `gyr`, but about principal axes.
    I : 2d ndarray
        3x3 inertia matrix at CG in X, Y, Z axes.
    princ_I : 2d ndarray
        Same as I, but in principal axes (smallest to biggest).

    Notes
    -----
    The general 6x6 mass matrix relative to some reference point is:

    .. math::
        M = \left[
        \begin{array}{cccccc}
            m_x &   0 & 0   &        0 &  m_x d_z & -m_z d_y \\
            0   & m_y & 0   & -m_y d_z &      0   &  m_y d_x \\
            0   &   0 & m_z & m_z d_y  & -m_y d_x &        0 \\
            0   & -m_x d_z &  m_z d_y &
                I_{xx} + m_z d_y^2 + m_y d_z^2
                & I_{xy} - m_z d_x d_y
                & I_{xz} - m_y d_x d_z \\
            m_x d_z &      0   & -m_y d_x &
                I_{xy} - m_z d_x d_y
                & I_{yy} + m_z d_x^2 + m_x d_z^2
                & I_{yz} - m_x d_y d_z \\
            -m_z d_y &  m_y d_x &   0 &
                I_{xz} - m_y d_x d_z
                & I_{yz} - m_x d_y d_z
                & I_{zz} + m_x d_y^2 + m_y d_x^2
        \end{array} \right]

    The distances :math:`d_j` are the distances from the reference
    point to the cg in the coordinate system of that reference point.
    The inertias :math:`I_{ij}` are relative to the CG.

    The radius of gyration about the X axis is computed by
    :math:`\sqrt{I_{xx}/m_x}`. The others are computed similarly.
    Principal axis radii of gyration are also computed if requested.

    Raises
    ------
    ValueError
        If `m` is not symmetric.

    See also
    --------
    :func:`cbcheck`, :func:`cbconvert`, :func:`cbcoordchk`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import cb
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> mass = np.array([[3,     0,     0,     0,     0,     0],
    ...                  [0,     3,     0,     0,     0,   120],
    ...                  [0,     0,     3,     0,  -120,     0],
    ...                  [0,     0,     0,  1020,   -60,    22],
    ...                  [0,     0,  -120,   -60,  7808,    23],
    ...                  [0,   120,     0,    22,    23,  7800]])
    >>> mcg1, dcg1 = cb.cgmass(mass)
    >>> mcg1
    array([[    3.,     0.,     0.,     0.,     0.,     0.],
           [    0.,     3.,     0.,     0.,     0.,     0.],
           [    0.,     0.,     3.,     0.,     0.,     0.],
           [    0.,     0.,     0.,  1020.,   -60.,    22.],
           [    0.,     0.,     0.,   -60.,  3008.,    23.],
           [    0.,     0.,     0.,    22.,    23.,  3000.]])
    >>> dcg1
    array([ 40.,   0.,   0.])
    >>> mcg, dcg, gyr, pgyr, I, pI = cb.cgmass(mass, all6=True)
    >>> np.all(mcg == mcg1)
    True
    >>> np.all(dcg == dcg1)
    True
    >>> gyr
    array([ 18.4391,  31.6649,  31.6228])
    >>> pgyr
    array([ 18.4204,  31.5288,  31.7693])
    >>> I
    array([[ 1020.,   -60.,    22.],
           [  -60.,  3008.,    23.],
           [   22.,    23.,  3000.]])
    >>> pI
    array([[ 1017.9312,     0.    ,     0.    ],
           [    0.    ,  2982.2045,     0.    ],
           [    0.    ,     0.    ,  3027.8643]])
    """
    if not ytools.mattype(m, 'symmetric'):
        raise ValueError('mass matrix is not symmetric')

    mx, my, mz = np.diag(m)[:3]
    dx = m[1, 5]/my
    dy = m[2, 3]/mz
    dz = m[0, 4]/mx

    # compute mass terms that will be subtracted off:
    Md = np.array([[      0,     mx*dz,    -mz*dy],
                   [ -my*dz,         0,     my*dx],
                   [  mz*dy,    -my*dx,         0]])

    I = np.array([[mz*dy**2+my*dz**2, -mz*dx*dy, -my*dx*dz],
                  [-mz*dx*dy, mz*dx**2+mx*dz**2, -mx*dy*dz],
                  [-my*dx*dz,         -mx*dy*dz, mx*dy**2+my*dx**2]])

    mcg = m.astype(float, copy=True)
    mcg[:3, 3:] -= Md
    mcg[3:, :3] -= Md.T
    mcg[3:, 3:] -= I

    I = mcg[3:, 3:]
    dxyz = np.array([dx, dy, dz])
    if not all6:
        return mcg, dxyz

    gyr = np.sqrt(np.diag(I)/np.diag(mcg)[:3])
    if np.iscomplexobj(gyr):
        gyr = -np.abs(gyr)

    if np.any(np.isnan(I)):
        princ_gyr = gyr
        princ_I = I
    else:
        w, v = linalg.eigh(I)
        m2 = v.T @ mcg[:3, :3] @ v
        princ_I = np.diag(w)
        princ_gyr = np.sqrt(w/np.diag(m2))
        if np.iscomplexobj(princ_gyr):
            princ_gyr = -np.abs(princ_gyr)

    return mcg, dxyz, gyr, princ_gyr, I, princ_I


def _get_Tlv2sc(sccoord):
    if sccoord is None:
        return np.eye(6)

    sccoord = np.atleast_2d(sccoord)
    if sccoord.shape[0] == 3:
        T = np.zeros((6, 6))
        T[:3, :3] = sccoord
        T[3:, 3:] = sccoord
        return T

    # get transform from l/v basic to s/c:
    uset = n2p.addgrid(None, 1, 'b', sccoord, [0, 0, 0], sccoord)
    return n2p.rbgeom_uset(uset, 1)


def mk_net_drms(Mcb, Kcb, bset, *, bsubset=None, uset=None,
                ref=[0, 0, 0], sccoord=None, conv=None, reorder=True,
                g=9.80665/0.0254):
    """
    Form common data recovery matrices.

    The Craig-Bampton model is referred to as "spacecraft" or "s/c".
    The system is referred to as "launch vehicle" or "l/v". All
    arguments after `bset` must be named.

    Parameters
    ----------
    Mcb : 2d ndarray
        Craig-Bampton mass.
    Kcb : 2d ndarray
        Craig-Bampton stiffness.
    bset : 1d array_like
        Index partition vector specifying location and order of b-set
        (boundary) DOF in Mcb and Kcb. Uses zero offset.
    bsubset : 1d array_like or None; optional
        Index partition vector into `bset` specifying which b-set DOF
        to consider. Note the CG acceleration recovery matrix will
        only consider forces on this subset so if there are other
        boundary DOF that are connected to other superelements, the CG
        recovery transforms are probably not very useful.
    uset : 2d ndarray; optional for single point interface
        A 6-column matrix as output by :func:`op2.rdn2cop2` or
        :func:`n2p.addgrid`. For information on the format of this
        matrix, see :func:`op2.rdn2cop2`. This defines the
        Craig-Bampton interface nodes in s/c coordinates, *not* in l/v
        coordinates. Use `sccoord` to define the transformation from
        l/v to s/c coordinates.

        If `uset` is None, a single grid with id 1 will be
        automatically created at (0, 0, 0). The :func:`n2p.addgrid`
        call for this is::

           uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)

    ref : 1d array_like or integer; optional
        Defines reference location for the recovery transforms; for
        example, the center point of a ring of boundary grids. The
        location is in s/c basic (before any unit conversion). Can
        also be an integer grid id defined in `uset`.
    sccoord : 3x3 or 4x3 array_like or None; optional
        If 3x3, it is transform from l/v basic to s/c. If 4x3, it is
        the CORD2R, CORD2C or CORD2S information specifying the
        coordinate system of the s/c relative to the l/v basic
        (reference_id must be 0)::

            [ cid type reference_id ]
            [ Ax   Ay   Az          ]
            [ Bx   By   Bz          ]
            [ Cx   Cy   Cz          ]

        This is further described in :func:`n2p.addgrid`. The
        transform from l/v basic to s/c is computed from this
        information.

        If None, the transform is assumed to be identity.

    conv : None or 2-element array_like or string; optional
        If None, no unit conversion is done; otherwise, units are
        converted from s/c to l/v. If 2-element array_like, it is::

            (length_conversion, mass_conversion)

        If string, it is one of:

            * 'm2e' (convert from metric to English)
            * 'e2m' (convert from English to metric)

        The string form assumes units of meter & kg, and inch &
        lbf*s**2/inch (slinch). See :func:`cbconvert` for more
        information.

    reorder : bool; optional
        If True, reorder the DOF so the b-set are first (uses
        :func:`cbreorder`).
    g : scalar; optional
        Standard gravity in l/v units.

    Returns
    -------
    A record (SimpleNamespace class) with these 12 members:

    ifltma_sc, ifltma_lv : 2d ndarrays
        The acceleration-dependent portion of the net interface force
        data recovery matrices in s/c and l/v units and coordinates,
        respectively. Along with `ifltmd_*`, recovers the net forces
        on the s/c.
    ifltmd_sc, ifltmd_lv : 2d ndarrays
        The displacement-dependent portion of the net interface force
        data recovery matrices. Should be zero unless `bsubset` is
        used.
    ifatm_sc, ifatm_lv : 2d ndarrays
        The net interface acceleration data recovery matrices in s/c
        and l/v coordinates, respectively. Acceleration-
        dependent. Units are 'g' and 'rad/sec**2'.
    cgatm_sc, cgatm_lv : 2d ndarrays
        The net CG acceleration data recovery matrices in s/c and l/v
        coordinates, respectively. These are based on interface
        forces. Acceleration-dependent. Units are 'g' and
        'rad/sec**2'.
    weight_sc, weight_lv : real scalars
        Weight of the s/c in s/c and l/v units.
    height_sc, height_lv : real scalars
        CG height of the s/c in s/c and l/v units. Height is relative
        to `ref`.
    scaxial_sc, scaxial_lv : integers
        0, 1, or 2 depending on which DOF is axial in s/c coordinates
        and in l/v coordinates, respectively.
    Tsc2lv : 2d ndarray
        The 6x6 transformation from s/c to l/v coordinates. This is
        the transpose of the transform defined by `sccoord`.
    rb : 2d ndarray
        The geometry-based rigid-body modes corresponding to the
        `bsubset` part of the `uset` table. Same as `rb_all` if
        `bsubset` is None.
    rb_all : 2d ndarray
        The geometry-based rigid-body modes corresponding to the
        the `uset` table. Same as `rb` if `bsubset` is None.
    """
    if uset is None:
        uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)

    if reorder:
        Mcb = cbreorder(Mcb, bset)
        Kcb = cbreorder(Kcb, bset)
        i = np.argsort(bset)
        uset = uset[i]
        if bsubset is not None:
            bsubset = locate.index2bool(bsubset, len(bset))
            bsubset = bsubset[i]
        bset = np.arange(len(bset))
    bset_if = bset if bsubset is None else bset[bsubset]
    Tsc2lv = _get_Tlv2sc(sccoord).T

    # only ifltm needs to be computed for both sets of units
    # (ifatm and cgatm both output g's and rad/sec^2)
    # rigid-body modes relative to reference:
    uset_if = uset[bset_if]
    rb = n2p.rbgeom_uset(uset_if, ref)
    bifb = np.ix_(bset_if, bset)
    ifltma = rb.T @ Mcb[bset_if]
    ifltmd = rb.T @ Kcb[bifb]  # should be zero if bsubset is None

    # check grounding:
    rb_all = n2p.rbgeom_uset(uset, ref)
    bb = np.ix_(bset, bset)
    grfrc = Kcb[bb] @ rb_all
    if (abs(grfrc).max() > abs(Kcb[bb]).max()*1e-8):
        warn('Rigid-body grounding forces need to be checked. '
             'Correct `uset`?', RuntimeWarning)
        print('max grounding forces =\n', abs(grfrc).max(axis=0))

    if conv is not None:
        # make s/c version of ifltm compatible with system
        ifltma = cbconvert(ifltma, bset, conv, drm=True)
        ifltmd = cbconvert(ifltmd, bset, conv, drm=True)

        # make new M & K for l/v versions:
        Mcb = cbconvert(Mcb, bset, conv)
        Kcb = cbconvert(Kcb, bset, conv)
        uset, ref = _uset_convert(uset, ref, conv)
        uset_if = uset[bset_if]

        # rigid-body modes relative to reference:
        rb = n2p.rbgeom_uset(uset_if, ref)
        rb_all = n2p.rbgeom_uset(uset, ref)
        ifltma_lv = rb.T @ Mcb[bset_if]
        ifltmd_lv = rb.T @ Kcb[bifb]
    else:
        ifltma_lv = ifltma
        ifltmd_lv = ifltmd

    # use RBE3 for net accelerations
    if len(bset) > 6:
        dof_indep = 123
        xyz = ytools.mkpattvec([0, 1, 2], len(bset_if), 6).ravel()
        xyz = bset_if[xyz]
    else:
        dof_indep = 123456
        xyz = np.arange(6)

    # add center point for RBE3
    if isinstance(ref, numbers.Integral):
        i = np.nonzero(uset_if[:, 0] == ref)[0][0]
        ref = uset[i, 3:]
    grids = uset_if[::6, 0].astype(np.int64)
    new_id = grids.max() + 1
    uset2 = n2p.addgrid(uset_if, new_id, 'b', 0, ref, 0)
    rbe3 = n2p.formrbe3(uset2, new_id, 123456, [dof_indep, grids])
    ifatm = np.zeros((6, Mcb.shape[1]))
    ifatm[:, xyz] = rbe3

    # calculate cg location and mass @ cg (l/v units but s/c coords):
    Mif = rb_all.T @ Mcb[bb] @ rb_all
    Mcg, cg = cgmass(Mif)  # cg is relative to ref

    # form rigid-body modes relative to CG:
    rbcg = n2p.rbgeom_uset(uset_if, cg)

    # for net CG acceleration:
    cgatm = linalg.solve(Mcg, rbcg.T @ Mcb[bset_if])
    ifatm[:3] /= g
    cgatm[:3] /= g
    weight_lv = Mcg[0, 0]*g
    height_lv = abs(cg).max()
    if conv is not None:
        lengthconv, massconv = _get_conv_factors(conv)
        weight_sc = weight_lv / (massconv * lengthconv)
        height_sc = height_lv / lengthconv
    else:
        weight_sc = weight_lv
        height_sc = height_lv
    scaxial_sc = np.argmax(abs(cg))
    cg_lv = Tsc2lv[:3, :3] @ cg[:, None]
    scaxial_lv = np.argmax(abs(cg_lv))
    return SimpleNamespace(
        ifltma_sc=ifltma, ifltma_lv=Tsc2lv @ ifltma_lv,
        ifltmd_sc=ifltmd, ifltmd_lv=Tsc2lv @ ifltmd_lv,
        ifatm_sc=ifatm, ifatm_lv=Tsc2lv @ ifatm,
        cgatm_sc=cgatm, cgatm_lv=Tsc2lv @ cgatm,
        weight_sc=weight_sc, weight_lv=weight_lv,
        height_sc=height_sc, height_lv=height_lv,
        scaxial_sc=scaxial_sc, scaxial_lv=scaxial_lv,
        Tsc2lv=Tsc2lv, rb=rb, rb_all=rb_all)


def _rbmultchk(fout, drm, name, rb, labels, drm2, prtnullrows):
    """
    Routine used by :func:`rbmultchk`. See documentation for
    :func:`rbmultchk`.
    """
    fout.write('----------------------------------------------\n')
    fout.write('Results for {} * RB\n'.format(name))
    fout.write('----------------------------------------------\n')

    n = np.size(drm, 0)
    rbr = np.size(rb, 0)
    cdrm = np.size(drm, 1)
    if cdrm == rbr:
        drmrb = drm @ rb
    elif cdrm < rbr:
        drmrb = drm @ rb[:cdrm]
    else:
        drmrb = drm[:, :rbr] @ rb

    # get rb scale:
    xrss = np.sqrt(rb[:-2, 0]**2 + rb[1:-1, 0]**2 + rb[2:, 0]**2)
    pv = xrss > 1e-6 * abs(xrss).max()  # np.nonzero(xrss)[0]
    xrss = xrss[pv]
    if xrss.size == 0 or not np.allclose(xrss, xrss[0]):
        raise ValueError('failed to get scale of rb modes ... check `rb`')
    rbscale = xrss[0]

    # attempt to find xyz triples
    coords = np.empty((n, 3))
    coords[:] = np.nan
    scales = np.empty(n)
    scales[:] = np.nan
    j = 0
    while j+2 < n:
        pv = j + np.array([0, 1, 2])
        T1 = drmrb[pv, :3]
        # check for a scalar multiplier (like .00259, for example)
        T1tT1 = T1.T @ T1
        csqr = T1tT1[0, 0]
        try:
            T2 = linalg.inv(T1)
            mx = np.max(np.abs(T1))
            good = np.allclose(csqr*T2, T1.T,
                               rtol=0.001, atol=max(0.001*mx, 1.e-5))
        except linalg.LinAlgError:
            good = False
        if good:
            rbrot = T2 @ drmrb[pv, 3:]
            x = rbrot[1, 2]
            y = rbrot[2, 0]
            z = rbrot[0, 1]
            rbrot_ideal = np.array([[0, z, -y],
                                    [-z, 0, x],
                                    [y, -x, 0]])
            mx = abs(rbrot_ideal).max()
            if np.allclose(rbrot, rbrot_ideal,
                           rtol=0.001, atol=max(0.001*mx, 1.e-5)):
                coords[pv, 0] = x
                coords[pv, 1] = y
                coords[pv, 2] = z
                scales[pv] = np.sqrt(csqr)/rbscale
                j += 3
            else:
                j += 1
        else:
            j += 1

    fout.write('\nExtreme Coordinates from {}\n'.format(name))
    headers = ['X', 'Y', 'Z']
    widths = [10, 10, 10]
    formats = ['{:10.4f}']*3
    sep = 2
    hu, f = writer.formheader(headers, widths, formats,
                              sep=sep, just=0)
    hu = ''.join([' '*11+i+'\n' for i in hu.rstrip().split('\n')])
    fout.write(hu)
    if np.all(np.isnan(coords[:, 0])):
        fout.write('             -- no coordinates detected --\n')
    else:
        mn = np.nanmin(coords, axis=0).reshape((1, 3))
        mx = np.nanmax(coords, axis=0).reshape((1, 3))
        writer.vecwrite(fout, '  Minimums:'+f, mn)
        writer.vecwrite(fout, '  Maximums:'+f, mx)

    def pf(s):
        return s.replace(' nan,        nan,        nan         nan',
                         '    ,           ,                       ')

    r = np.arange(1, n+1)
    nonnr = np.any(drm, axis=1)
    nr = ~nonnr
    snonnr = np.sum(nonnr)
    snr = np.sum(nr)
    fout.write('\n{} has {} ({:.1f}%) non-NULL rows and {} ({:.1f}%) '
               'NULL rows.\n'.
               format(name, snonnr, snonnr/n*100, snr, snr/n*100))

    if prtnullrows:
        fout.write('\n{} * RB results:  '
                   '(including NULL rows)\n'.format(name))
    else:
        fout.write('\n{} * RB results:  '
                   '(NOT including NULL rows)\n'.format(name))
    fout.write('  Note: "Unit Scale" is output/input, so is '
               'independent of\n  current rb scaling which is: {}'
               '\n\n'.format(rbscale))
    if labels is None:
        labels = [''] * n
        labform = '{:5s}'
        lablen = 5

    lablen = max(8, len(max(labels, key=len)))
    headers = ['Row', 'Label', 'Coordinates (x, y, z)',
               'Unit Scale',
               name+' * RB Responses (x, y, z, rx, ry, rz)']
    widths = [6, lablen, 10*3+4, 10, 65]
    labform = '{{:{}s}}'.format(lablen)
    formats = ['{:6d}', labform,
               '{:10.4f}, '*2+'{:10.4f}',
               '{:10.6}',
               '{:10.3f} '*5+'{:10.3f}']
    sep = [2, 2, 2, 2, 2]
    hu, f = writer.formheader(headers, widths, formats,
                              sep=sep, just=0)
    fout.write(hu)
    if prtnullrows or np.all(nonnr):
        writer.vecwrite(fout, f, r, labels, coords, scales, drmrb,
                        postfunc=pf)
    else:
        if np.all(nr):
            fout.write('All rows in {} are NULL.\n'.format(name))
        else:
            lbls = np.array(labels)
            writer.vecwrite(fout, f, r[nonnr], lbls[nonnr],
                            coords[nonnr], scales[nonnr],
                            drmrb[nonnr], postfunc=pf)

    fout.write('\nAbsolute Maximums from {} * RB results:\n'.
               format(name))
    j = np.argmax(np.abs(drmrb), axis=0)

    headers = ['Row', 'Label', 'Coordinates (x, y, z)',
               'Unit Scale',
               'Maximum Responses on Diagonal'
               ' (x, y, z, rx, ry, rz)']
    hu, f = writer.formheader(headers, widths, formats,
                              sep=sep, just=0)
    fout.write(hu)
    for k in range(6):
        jk = j[k]
        writer.vecwrite(fout, f, r[jk], labels[jk],
                        coords[jk:jk+1], scales[jk:jk+1],
                        drmrb[jk:jk+1], postfunc=pf)

    # null row check:
    nr = np.nonzero(nr)[0]

    def wrt_null_rows(fout, name, n, nr, labels, labform, lablen):
        if nr.size == 0:
            fout.write('\nThere are no NULL rows in {}.\n'.
                       format(name))
        else:
            fout.write('\nThere are {} ({:.1f}%) NULL rows in {}:\n'.
                       format(nr.size, nr.size/n*100, name))
            hu, f = writer.formheader(['Row', 'Label'],
                                      [6, lablen],
                                      ['{:6d}', labform],
                                      sep=2, just=0)
            fout.write(hu)
            for i in nr:
                fout.write(f.format(i+1, labels[i]))

    wrt_null_rows(fout, name, n, nr, labels, labform, lablen)
    if drm2 is not None:
        if np.size(drm2, 0) != n:
            fout.write('Error: incorrectly sized DRM2 (has {} rows'
                       ' while DRM has {} rows)\n'.
                       format(np.size(drm2, 0), n))
            fout.write('Skipping check. Fix input and rerun.\n')
        else:
            fout.write('\n')
            nr2 = np.nonzero(~np.any(drm2, axis=1))[0]
            err = 0
            if nr2.size != nr.size:
                fout.write(' !! Warning: companion matrix DRM2'
                           ' has different set of NULL rows!!\n')
                err = 1
            elif np.any(nr != nr2):
                fout.write(' !! Warning: companion matrix DRM2'
                           ' has different set of NULL rows!!\n')
                err = 1
            if not err:
                fout.write(' NULL rows in DRM2 match those in {}\n'.
                           format(name))
            else:
                wrt_null_rows(fout, 'DRM2', n, nr2, labels,
                              labform, lablen)
    return drmrb


def rbmultchk(f, drm, name, rb, labels=None, drm2=None,
              prtnullrows=False):
    """
    Rigid-body multiply check on a data recovery matrix.

    Parameters
    ----------
    f : string or file handle or 1
        If string, name of file to write to. If file handle, write to
        that file. Use 1 to write to the screen.
    drm : 2d ndarray
        Data recovery matrix (DRM).
    name : string
        Name of the DRM; used for titling.
    rb : 2d ndarray
        Rigid-body modes; number of rows is either b-set or b+q-set
        sized. Number of columns is 6.
    labels : None or list; optional
        If list, it is a list of strings for DRM labeling. Up to first
        15 characters will be used.
    drm2 : None or 2d ndarray; optional
        Optional second DRM; only used in the null rows check to see if
        `drm` and `drm2` share a common set of null rows. Useful for
        DRMs that are meant to be used together, as in DTMA*a + DTMD*d
        and would be expected to share the same set of null rows (if
        any).
    prtnullrows : bool; optional
        If True, print the null rows in the DRM * rb section; otherwise
        only print the non-null rows. (Note that the null rows are
        still listed below that table.)

    Returns
    -------
    drmrb : 2d ndarray
        Matrix = `drm` * `rb`.

    Notes
    -----
    The printout has these 5 sections:
       1. A header, with `name` in it.
       2. An extreme coordinate table.
       3. A potentially large table showing the complete results of
          the DRM multiplied by the rigid-body modes.
       4. A summary table showing only the 6 rows that yielded the
          maximum response for each of the 6 rigid-body modes.
       5. A list of null rows, includes information about DRM2 if
          input.

    If any triplet of rows matches the pattern shown below, the
    coordinates are included in the printout (sections 2, 3, 4). For
    rows that do not match that pattern, the coordinates are left
    blank.

    The expected pattern for rigid-body displacements for a node is::

      [ 1 0 0    0   Z  -Y
        0 1 0   -Z   0   X
        0 0 1    Y  -X   0 ]

    The pattern shown assumes the node is in the same coordinate
    system as the reference node. If this is not the case, the 3x3
    coordinate transformation matrix (from reference to local) will
    show up in place of the the 3x3 identity matrix shown above. This
    routine will use that 3x3 matrix to convert coordinates to that of
    the reference before checking for the expected pattern. This all
    means is that the use of local coordinate systems is acceptable
    for this routine.

    Raises
    ------
    ValueError
        If `rb` does not have 6 columns.

    See also
    --------
    :func:`cbcheck`, :func:`rbdispchk`, :func:`cbcoordchk`,
    :func:`n2p.rbgeom`, :func:`n2p.rbgeom_uset`.

    Examples
    --------
    >>> from pyyeti import cb, n2p
    >>> import numpy as np
    >>> nodes = [[0., 0., 0.], [10., 20., 30.]]
    >>> ATM = n2p.rbgeom(nodes)
    >>> rb = np.eye(6)
    >>> _ = cb.rbmultchk(1, ATM, 'ATM', rb)   # doctest: +ELLIPSIS
    ----------------------------------------------
    Results for ATM * RB
    ----------------------------------------------
    <BLANKLINE>
    Extreme Coordinates from ATM
                     X           Y           Z
                 ----------  ----------  ----------
      Minimums:      0.0000      0.0000      0.0000
      Maximums:     10.0000     20.0000     30.0000
    <BLANKLINE>
    ATM has 12 (100.0%) non-NULL rows and 0 (0.0%) NULL rows.
    <BLANKLINE>
    ATM * RB results:  (NOT including NULL rows)
      Note: "Unit Scale" is output/input, so is independent of
      current rb scaling which is: 1.0
    <BLANKLINE>
       Row     Label          Coordinates (x, y, z)         ...
      ------  --------  ----------------------------------  ...
           1                0.0000,     0.0000,     0.0000  ...
           2                0.0000,     0.0000,     0.0000  ...
           3                0.0000,     0.0000,     0.0000  ...
           4                      ,           ,             ...
           5                      ,           ,             ...
           6                      ,           ,             ...
           7               10.0000,    20.0000,    30.0000  ...
           8               10.0000,    20.0000,    30.0000  ...
           9               10.0000,    20.0000,    30.0000  ...
          10                      ,           ,             ...
          11                      ,           ,             ...
          12                      ,           ,             ...
    <BLANKLINE>
    Absolute Maximums from ATM * RB results:
       Row     Label          Coordinates (x, y, z)         ...
      ------  --------  ----------------------------------  ...
           1                0.0000,     0.0000,     0.0000  ...
           2                0.0000,     0.0000,     0.0000  ...
           3                0.0000,     0.0000,     0.0000  ...
           8               10.0000,    20.0000,    30.0000  ...
           7               10.0000,    20.0000,    30.0000  ...
           7               10.0000,    20.0000,    30.0000  ...
    <BLANKLINE>
    There are no NULL rows in ATM.
    """
    rb = np.atleast_2d(rb)
    r, c = np.shape(rb)
    if c != 6:
        raise ValueError('`rb` does not have 6 columns')
    return ytools.wtfile(f, _rbmultchk, drm, name, rb, labels, drm2,
                         prtnullrows)


def _rbdispchk(fout, rbdisp, grids, ttl, verbose, tol):
    """
    Routine used by :func:`rbdispchk`. See documentation for
    :func:`rbdispchk`.
    """
    r = rbdisp.shape[0]
    n = r // 3
    coords = np.empty((n, 3))
    errs = np.empty(n)
    maxerr = 0
    for j in range(n):
        row = j*3
        T = rbdisp[row:row+3, :3]
        rb = linalg.solve(T, rbdisp[row:row+3, 3:])
        deltax = rb[1, 2]
        deltay = rb[2, 0]
        deltaz = rb[0, 1]
        deltax2 = -rb[2, 1]
        deltay2 = -rb[0, 2]
        deltaz2 = -rb[1, 0]
        err = max(abs(np.diag(rb)).max(),
                  abs(deltax-deltax2),
                  abs(deltay-deltay2),
                  abs(deltaz-deltaz2))
        maxerr = max(maxerr, err)
        coords[j] = [deltax, deltay, deltaz]
        errs[j] = err
        mc = abs(coords[j]).max()
        if verbose and (err > mc*tol or np.isnan(err)):
            if grids is not None:
                fout.write('Warning: deviation from standard pattern,'
                           ' node ID = {} starting at row {}. '
                           'Max deviation = {:.3g} units.\n'.
                           format(grids[j], row+1, err))
            else:
                fout.write('Warning: deviation from standard pattern,'
                           ' node #{} starting at row {}. '
                           '\tMax deviation = {:.3g} units.\n'.
                           format(j+1, row+1, err))
            fout.write('  Rigid-Body Rotations:\n')
            writer.vecwrite(fout, '{:10.4f} {:10.4f} {:10.4f}\n', rb)
            fout.write('\n')

    if verbose:
        if ttl:
            fout.write('\n{}\n'.format(ttl))
        headers = ['Node']
        widths = [6]
        formats = ['{:6}']
        seps = [8]
        args = [np.arange(1, n+1)]
        if grids is not None:
            headers.append('ID')
            widths.append(8)
            formats.append('{:8}')
            seps.append(2)
            args.append(grids)
        headers.extend(['X', 'Y', 'Z', 'Error'])
        widths.extend([8, 8, 8, 10])
        formats.extend(['{:8.2f}', '{:8.2f}', '{:8.2f}', '{:10.4e}'])
        seps.extend([2, 2, 2, 3])
        args.extend([coords, errs])
        hu, f = writer.formheader(headers, widths, formats,
                                  sep=seps, just=0)
        fout.write(hu)
        writer.vecwrite(fout, f, *args)
        fout.write('\nMaximum absolute coordinate location error:  '
                   '{:3g} units\n\n'.format(maxerr))
    return coords, errs


def rbdispchk(f, rbdisp, grids=None,
              ttl='Coordinates Determined from '
              'Rigid-Body Displacements:',
              verbose=True, tol=1.e-4):
    """
    Rigid-body displacement check.

    Parameters
    ----------
    f : string or file handle or 1
        If string, name of file to write to. If file handle, write to
        that file. Use 1 to write to the screen.
    rbdisp : 2d ndarray
        Rigid-body displacements; size is 3*N x 6 where N is the number
        of nodes. Rows correspond to X, Y, Z triples for each node (in
        any coordinate system).
    grids : 1d array or None; optional
        Length N array of node IDs or None. If array, used only in
        diagnostic message printing.
    ttl : string or None; optional
        String to use for title of coordinates listing, or None for no
        title.
    verbose : bool; optional
        If true, print table of coordinates or any warnings about not
        matching the pattern
    tol : float; optional
        Sets the error tolerance level. If `verbose` is true, a
        warning message is printed for each node that does not fit the
        expected pattern. The criteria for the error message is if the
        maximum deviation for that node is > ``tol*max(node_coords)``.

    Returns
    -------
    coords : 2d ndarray
        A 3-column matrix of [x, y, z] locations of each node,
        relative to refpoint and in the local coordinate system of
        refpoint.
    errs : 1d ndarray
        Vector of maximum absolute deviations from the expected
        pattern for each node (see below).

    Notes
    -----
    The expected pattern for rigid-body displacements for each node is::

      [ 1 0 0    0   Z  -Y
        0 1 0   -Z   0   X
        0 0 1    Y  -X   0 ]

    The pattern shown assumes the node is in the same coordinate
    system as the reference node. If this is not the case, the 3x3
    coordinate transformation matrix (from reference to local) will
    show up in place of the the 3x3 identity matrix shown above. This
    routine will use that 3x3 matrix to convert coordinates to that of
    the reference before checking for the expected pattern. This all
    means is that the use of local coordinate systems is acceptable
    for this routine.

    Raises
    ------
    ValueError
        If `rbdisp` is not n x 6, where n is multiple of 3.

    See also
    --------
    :func:`cbcheck`, :func:`rbmultchk`, :func:`cbcoordchk`,
    :func:`n2p.rbgeom`, :func:`n2p.rbgeom_uset`.

    Examples
    --------
    Define locations for 3 nodes, compute rigid-body modes from them,
    and calculate their locations to test this routine:

    >>> from pyyeti import n2p
    >>> from pyyeti import cb
    >>> import numpy as np
    >>> from pyyeti import ytools
    >>> coords = np.array([[0,  0,  0],
    ...                    [1,  2,  3],
    ...                    [4, -5, 25]])
    >>> rb  = n2p.rbgeom(coords)
    >>> xyz_pv = ytools.mkpattvec([0, 1, 2], 3*6, 6).ravel()
    >>> rbtrimmed = rb[xyz_pv]
    >>> rbtrimmed
    array([[  1.,   0.,   0.,   0.,   0.,   0.],
           [  0.,   1.,   0.,   0.,   0.,   0.],
           [  0.,   0.,   1.,   0.,   0.,   0.],
           [  1.,   0.,   0.,   0.,   3.,  -2.],
           [  0.,   1.,   0.,  -3.,   0.,   1.],
           [  0.,   0.,   1.,   2.,  -1.,   0.],
           [  1.,   0.,   0.,   0.,  25.,   5.],
           [  0.,   1.,   0., -25.,   0.,   4.],
           [  0.,   0.,   1.,  -5.,  -4.,   0.]])
    >>> coords_out, errs = cb.rbdispchk(1, rbtrimmed)
    <BLANKLINE>
    Coordinates Determined from Rigid-Body Displacements:
             Node      X         Y         Z         Error
            ------  --------  --------  --------   ----------
                 1      0.00      0.00      0.00   0.0000e+00
                 2      1.00      2.00      3.00   0.0000e+00
                 3      4.00     -5.00     25.00   0.0000e+00
    <BLANKLINE>
    Maximum absolute coordinate location error:    0 units
    <BLANKLINE>
    >>> np.allclose(coords, coords_out)
    True
    >>> errs.max() < 1e-9
    True
    """
    r, c = np.shape(rbdisp)
    if c != 6:
        raise ValueError('`rbdisp` does not have 6 columns')

    if (r // 3) * 3 != r:
        raise ValueError('number of rows in `rbdisp` must be '
                         'a multiple of 3.')
    return ytools.wtfile(f, _rbdispchk, rbdisp, grids, ttl,
                         verbose, tol)


def cbcoordchk(K, bset, refpoint, grids=None, ttl=None,
               verbose=True, outfile=1, rb_normalizer=None):
    """
    Check interface coordinates of a Craig-Bampton stiffness matrix.

    Parameters
    ----------
    K : 2d numpy array
        Craig-Bampton stiffness matrix (b+q-set size).
    bset : 1d array
        Partition vector to the b-set DOF; length must be multiple of
        6.
    refpoint : 1d array
        6-element subset of `bset` representing a statically-
        determinate set capable of restraining all rigid-body
        modes (similar to a SUPORT card in Nastran). Typically, this
        is just all DOF of a single node; however, if this is not
        possible, you'll also want to define `rb_normalizer`.
    grids : 1d array or None; optional
        Length N array of node IDs or None. If array, used only in
        diagnostic message printing.
    ttl : string or None; optional
        String to use for title of coordinates listing, or None for no
        title.
    verbose : bool; optional
        If true, print table of coordinates and warnings from
        :func:`rbdispchk`.
    outfile : string or file handle or 1; optional
        If string, name of file to write to. If file handle, write to
        that file. Use 1 to write to the screen.
    rb_normalizer : 2d array_like or None
        If not None, the `rbmodes` output will be normalized via::

            rbmodes = rbmodes @ rb_normalizer

        `rb_normalizer` is 6 x 6. This normalization is necessary when
        the DOF in `refpoint` are spread out amongst multiple nodes.
        `rb_normalizer` defines the motion of the `refpoint` DOF
        relative to some reference location. For example, the
        following creates an `rb_normalizer` relative to the origin of
        the basic coordinate system::

            R = [0., 0., 0.]
            rb_normalizer = n2p.rbgeom_uset(uset[bset], R)[refpoint]

        This would cause the returned coordinates (see `coords` below)
        to be relative to the basic origin and in the basic coordinate
        system.

    Returns
    -------
    coords : ndarray
        A 3-column matrix of [x, y, z] locations of each node,
        relative to `refpoint` (or as defined by `rb_normalizer`) and
        in the local coordinate system of `refpoint` (again, or as
        defined by .
    rbmodes : ndarray
        Stiffness-based rigid-body modes (6 columns). Will have
        zeros corresponding to the modal DOF.
    maxerr : float
        Maximum absolute error of any deviation from the expected
        pattern (see :func:`rbdispchk`).

    Notes
    -----
    This routine will fail if the DOF in `refpoint` do not fully
    and minimally restrain all rigid-body motion. It must be a
    statically-determinate set.

    If `coords` doesn't fit the expected pattern shown in
    :func:`rbdispchk`, a warning message is printed.

    Note that :func:`rbdispchk` is used to calculate coords. That
    routine accounts for the possibility of the interface DOF using
    different coordinate systems.

    Example usage::

        from pyyeti import cb
        import numpy as np
        b = np.arange(18)     # 3 boundary grids
        bref = np.arange(6)   # use 1st as reference
        dist = cb.cbcoordchk(kaa, b, bref)

    Raises
    ------
    ValueError
        If length of `b` is not an even multiple of 6.
    ValueError
        If refpoint is not a 6-element subset of `bset`.

    See also
    --------
    :func:`rbdispchk`, :func:`rbmultchk`, :func:`cbcheck`,
    :func:`cbconvert`, :func:`cbreorder`, :func:`cgmass`,
    :func:`n2p.rbgeom`, :func:`n2p.rbgeom_uset`
    """
    lt = np.size(K, 0)
    lb = len(bset)
    lq = lt - lb

    if (lb // 6) * 6 != lb:
        raise ValueError('b-set not a multiple of 6.')

    if 6 != len(refpoint):
        raise ValueError('reference point must have length of 6.')

    kbb = K[np.ix_(bset, bset)]
    o = locate.flippv(refpoint, lb)
    rbmodes = np.zeros((lb, 6))
    rbmodes[refpoint] = np.eye(6)
    if o.size > 0:
        kor = kbb[np.ix_(o, refpoint)]
        koo = kbb[np.ix_(o, o)]
        rbmodes[o] = -linalg.solve(koo, kor)

    if rb_normalizer is not None:
        rbmodes = rbmodes @ rb_normalizer

    xyz = ytools.mkpattvec([0, 1, 2], lb, 6).ravel()
    coords, maxerr = rbdispchk(outfile, rbmodes[xyz],
                               grids, ttl, verbose)
    if lq > 0:
        rbmodes1 = rbmodes
        rbmodes = np.zeros((lt, 6))
        rbmodes[bset] = rbmodes1
    return coords, rbmodes, maxerr


def _cbcheck(fout, Mcb, Kcb, bseto, bref, uset, uref, conv, em_filt,
             rb_norm, reorder):
    """
    Routine used by :func:`cbcheck`. See documentation for
    :func:`cbcheck`.
    """
    n = np.size(Mcb, 0)

    # check matrix properties:
    mtype, mtypes = ytools.mattype(Mcb)
    ktype = ytools.mattype(Kcb)[0]
    if mtype & mtypes['symmetric']:
        fout.write('Mass matrix is symmetric.\n')
    else:
        fout.write('Warning: mass matrix is not symmetric.\n')
    if mtype & mtypes['posdef']:
        fout.write('Mass matrix is positive definite.\n')
    else:
        fout.write('Warning: mass matrix is not positive definite.\n')
    if ktype & mtypes['symmetric']:
        fout.write('Stiffness matrix is symmetric.\n')
    else:
        fout.write('Warning: stiffness matrix is not symmetric.\n')

    # some input checks:
    if uset is None:
        uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)
        uref = 1

    nb = len(bseto)
    if uset.shape[0] != nb:
        raise ValueError('number of rows in `uset` is {}, but must '
                         'equal len(b-set) ({})'.
                         format(uset.shape[0], nb))

    # convert units if necessary:
    if conv is not None:
        m = cbconvert(Mcb, bseto, conv)
        k = cbconvert(Kcb, bseto, conv)
        uset, uref = _uset_convert(uset, uref, conv)
    else:
        m = Mcb
        k = Kcb

    if reorder:
        # reorder mass and stiffness:
        m = cbreorder(m, bseto)
        k = cbreorder(k, bseto)
        i = np.argsort(bseto)
        uset = uset[i]

        # define "new" order of b-set:
        b = locate.index2bool(bref, n)
        bset = np.arange(nb)
        bref = np.nonzero(b[bseto])[0]  # where ref is in new bset
    else:
        bset = np.sort(bseto)
        if not (bset == bseto).all():
            raise ValueError('when `reorder` is False, `bseto` must'
                             ' be in ascending order')

    # check for appropriate zeros:
    qset = locate.flippv(bset, n)
    nq = len(qset)
    B = np.ix_(bset, bset)
    Q = np.ix_(qset, qset)
    QB = np.ix_(qset, bset)
    mbb = m[B]
    mqq = m[Q]
    kbb = k[B]
    kqq = k[Q]
    fout.write('\nMass values check:\n')
    mxmqqerr = np.max(np.abs(mqq - np.eye(nq)))
    fout.write('\tMaximum value of MQQ-I            = {:11g}'
               '  (should be zero)\n'.format(mxmqqerr))
    mnkqq = np.min(np.diag(kqq))
    mxkqqerr = np.max(np.abs(kqq - np.diag(np.diag(kqq))))
    mxkbb = np.max(np.abs(kbb))
    fout.write('\nStiffness values checks:\n')
    fout.write('\tMaximum value of KBB              = {:11g}'
               '  (should be zero only if statically-determinate)'
               '\n'.format(mxkbb))
    fout.write('\tMaximum value of KBQ              = {:11g}'
               '  (should be zero)\n'.
               format(np.max(np.abs(k[np.ix_(bset, qset)]))))
    fout.write('\tMaximum off-diagonal value of KQQ = {:11g}'
               '  (should be zero)\n'.format(mxkqqerr))
    fout.write('\tMinimum diagonal value of KQQ     = {:11g}'
               '  (should be > zero)\n\n'.format(mnkqq))
    if mxkqqerr > 0 or mxmqqerr > 0:
        fout.write('\n\nFATAL: check off-diagonals of MQQ and KQQ'
                   ' -- they should be zero. This needs to be resolved'
                   ' before CLA.\n')

    if nb == 6 and mxkbb > 0:
        fout.write('Echoing KBB for visual inspection since max(KBB)'
                   ' != 0:\n')
        fout.write('KBB =\n')
        form = '\t' + ('{:7.4f}  ')*5 + '{:7.4f}\n'
        writer.vecwrite(fout, form, kbb)
        fout.write('\n\tNote: for comparison KQQ[0, 0] = {:.2f}.\n\n'.
                   format(kqq[0, 0]))

    # Three types of rigid-body modes will be calculated:
    #
    #   1. Stiffness based
    #   2. Geometry based
    #   3. From eigensolution
    #
    #  These will aid in geometry checking, mass properties checks,
    #  and stiffness grounding checks.

    # use geometry to generate a set of rb-modes:
    rbg = n2p.rbgeom_uset(uset, uref)

    if rb_norm is None:
        if np.any(np.diff(bref) != 1):
            rb_norm = True

    if rb_norm:
        rb_normalizer = rbg[bref]
        ttl = ('Stiffness-based coordinates relative to `uref` '
               'because of normalization (`rb_norm`):\n '
               ' (Note: locations are in basic coordinate system.)\n'
               .format(bref[0]+1))
    else:
        rb_normalizer = None
        ttl = ('Stiffness-based coordinates relative to node starting'
               ' at row/col {} (after any reordering):\n '
               ' (Note: locations are in local coordinate system of '
               'the reference node.)\n'.format(bref[0]+1))

    # coordinates of boundary points according to stiffness:
    coords, rbs, _ = cbcoordchk(k, bset, bref,
                                uset[::6, 0].astype(np.int64),
                                ttl, True, fout, rb_normalizer)

    # calculate free-free modes:
    if mtype & mtypes['posdef'] and ktype & mtypes['symmetric']:
        fout.write('Solving hermitian eigen problem.\n')
        w, v = linalg.eigh(k, m)
        w = np.abs(w)
        # due to trouble with low frequency accuracy in DSYEVR, use
        # subspace iteration for lower frequency modes (up to 50 Hz,
        # but limit to 350 modes)
        f = np.sqrt(w)/(2*math.pi)
        p = np.sum(f < 50.)
        # num = max(6, min(10, n/10))
        if 0:
            ws, vs = sp_la.eigsh(k, p, m, sigma=1., mode='normal')
        else:
            ws, vs, _ = ytools.eig_si(k, m, p=p,
                                      mu=-10., Xk=v, pmax=350)
        ws = np.abs(ws)
        if len(w) > len(ws):
            w[:len(ws)] = ws
            v[:, :len(ws)] = vs
        else:
            w = ws
            v = vs
    else:
        fout.write('Solving generalized eigen problem.\n')
        w, v = linalg.eig(k, m, right=True)
        if np.iscomplexobj(w):
            j = np.argsort(np.abs(w))
            w = np.abs(w[j])
            v = np.real(v[:, j])
            normfacs = np.abs(np.diag(v.T @ m @ v))
            v = np.sqrt(1/normfacs) * v

    # assuming 6 rigid-body modes
    rbe = linalg.solve(v[bref, :6].T, v[:, :6].T).T
    if rb_norm:
        rbe = rbe @ rb_normalizer

    # distance of motion check:
    rbs_b = rbs[bset]
    rbg_b = rbg
    rbe_b = rbe[bset]
    nb = len(bset) // 6
    rss_s = np.zeros((nb, 3))
    rss_g = np.zeros((nb, 3))
    rss_e = np.zeros((nb, 3))
    for i in range(nb):
        s = slice(i*6, i*6+3)
        rss_s[i] = np.sqrt((rbs_b[s, :3]**2).sum(axis=0))
        rss_g[i] = np.sqrt((rbg_b[s, :3]**2).sum(axis=0))
        rss_e[i] = np.sqrt((rbe_b[s, :3]**2).sum(axis=0))

    fout.write('RB Translation Movement Check '
               '-- should all be 1.0s:\n\n')
    fout.write('\tNode          Stiffness-based      '
               '  Geometry-based        Eigenvalue-based\n')
    fout.write('\t---------   -------------------    '
               '-------------------    -------------------\n')
    writer.vecwrite(fout, '\t{:8d}    {:5.3f}  {:5.3f}  {:5.3f}'
                    '    {:5.3f}  {:5.3f}  {:5.3f}    {:5.3f}  '
                    '{:5.3f}  {:5.3f}\n',
                    uset[::6, 0].astype(np.int64),
                    rss_s, rss_g, rss_e)
    fout.write('\n\n')

    rss_rot_s = np.zeros((nb, 3))
    rss_rot_g = np.zeros((nb, 3))
    rss_rot_e = np.zeros((nb, 3))
    for i in range(nb):
        s = slice(i*6+3, i*6+6)
        rss_rot_s[i] = np.sqrt((rbs_b[s, 3:]**2).sum(axis=0))
        rss_rot_g[i] = np.sqrt((rbg_b[s, 3:]**2).sum(axis=0))
        rss_rot_e[i] = np.sqrt((rbe_b[s, 3:]**2).sum(axis=0))

    fout.write('RB Rotation Movement Check '
               '-- should all be 1.0s:\n\n')
    fout.write('\tNode          Stiffness-based      '
               '  Geometry-based        Eigenvalue-based\n')
    fout.write('\t---------   -------------------    '
               '-------------------    -------------------\n')
    writer.vecwrite(fout, '\t{:8d}    {:5.3f}  {:5.3f}  {:5.3f}'
                    '    {:5.3f}  {:5.3f}  {:5.3f}    {:5.3f}  '
                    '{:5.3f}  {:5.3f}\n',
                    uset[::6, 0].astype(np.int64),
                    rss_rot_s, rss_rot_g, rss_rot_e)
    fout.write('\n\n')

    # get mass properties:
    ms = rbs.T @ m @ rbs
    mg = rbg.T @ mbb @ rbg
    me = rbe.T @ m @ rbe

    # cg location, radius of gyration:
    (mcgs, ds, gyr_s, pgyr_s, I_s, Ip_s) = cgmass(ms, all6=True)
    (mcgg, dg, gyr_g, pgyr_g, I_g, Ip_g) = cgmass(mg, all6=True)
    (mcge, de, gyr_e, pgyr_e, I_e, Ip_e) = cgmass(me, all6=True)

    fout.write('MASS PROPERTIES CHECKS:\n\n')

    def wrtmass(fout, mass, rbtype):
        fout.write('6x6 mass matrix from {}-based rb modes:\n\n'.
                   format(rbtype))
        writer.vecwrite(fout, '\t{:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}'
                        '  {:12.4f}  {:12.4f}\n', mass)
        fout.write('\n\n')
    wrtmass(fout, ms, 'stiffness')
    wrtmass(fout, mg, 'geometry')
    wrtmass(fout, me, 'eigensolution')

    fout.write('Comparisons from mass properties:\n')

    def wrtdist(fout, ds, dg, de, ttl, use123=False):
        fout.write(ttl)
        if use123:
            fout.write('\t\tRB-Mode from                    1'
                       '               2               3\n')
        else:
            fout.write('\t\tRB-Mode from                    X'
                       '               Y               Z\n')
        fout.write('\t\t------------              -----------------'
                   '----------------------------\n')
        fout.write('\t\t   Stiffness           {:14.6f}  '
                   '{:14.6f}  {:14.6f}\n'.format(*ds))
        fout.write('\t\t   Geometry            {:14.6f}  '
                   '{:14.6f}  {:14.6f}\n'.format(*dg))
        fout.write('\t\t   Eigensolution       {:14.6f}  '
                   '{:14.6f}  {:14.6f}\n'.format(*de))
    ttl = ('\n\tDistance to CG location from relevant reference '
           'point:\n\n')
    wrtdist(fout, ds, dg, de, ttl)
    ttl = ('\n\n\tRadius of gyration about X, Y, Z axes '
           '(from CG):\n\n')
    wrtdist(fout, gyr_s, gyr_g, gyr_e, ttl)
    ttl = ('\n\tRadius of gyration about principal axes '
           '(from CG):\n\n')
    wrtdist(fout, pgyr_s, pgyr_g, pgyr_e, ttl, use123=True)
    fout.write('\n\n')

    def wrtinertia(fout, I, Ip, rbtype):
        fout.write('{}-based Inertia Matrix @ CG about X,Y,Z:\n\n'.
                   format(rbtype.capitalize()))
        writer.vecwrite(fout, '\t\t{:12.4f}  {:12.4f}  {:12.4f}\n',
                        I)
        fout.write('\n')
        fout.write('\tPrincipal Axis Moments of Inertia:\n\n')
        fout.write('\t\t{:12.4f}  {:12.4f}  {:12.4f}\n'.
                   format(*np.sort(np.diag(Ip))))
        fout.write('\n\n')
    wrtinertia(fout, I_s, Ip_s, 'stiffness')
    wrtinertia(fout, I_g, Ip_g, 'geometry')
    wrtinertia(fout, I_e, Ip_e, 'eigensolution')

    fout.write('GROUNDING CHECKS:\n\n')

    def wrtground(fout, uset, rbf, rbfsumm, rbtype):
        fout.write('                            K*RB using {}-'
                   'based rb modes:\n'.format(rbtype))
        fout.write('DOF                    X           Y           Z '
                   '          RX          RY          RZ\n')
        fout.write('-------------     -------------------------------'
                   '-------------------------------------\n')
        nb = uset.shape[0]
        writer.vecwrite(fout, '{:8d} {:3d}    {:10.3f}  {:10.3f}  {:10.3f}'
                        '  {:10.3f}  {:10.3f}  {:10.3f}\n',
                        uset[:, :2].astype(np.int64), rbf[:nb])
        nq = rbf.shape[0] - nb
        if nq > 0:
            writer.vecwrite(fout, '  modal  {:4d}   {:10.3f}  {:10.3f}'
                            '  {:10.3f}  {:10.3f}  {:10.3f}  {:10.3f}\n',
                            np.arange(nq)+1, rbf[nb::])
        fout.write('\nSummation of {}-based rb-forces: '
                   'RB\'*K*RB:\n\n'.format(rbtype))
        writer.vecwrite(fout, '\t{:10.3f}  {:10.3f}  {:10.3f}  '
                        '{:10.3f}  {:10.3f}  {:10.3f}\n', rbfsumm)
        fout.write('\n\n')

    rbfs = k @ rbs
    rbfg = kbb @ rbg
    rbfe = k @ rbe

    wrtground(fout, uset, rbfs, rbs.T @ rbfs, 'stiffness')
    wrtground(fout, uset, rbfg, rbg.T @ rbfg, 'geometry')
    wrtground(fout, uset, rbfe, rbe.T @ rbfe, 'eigensolution')

    # free-free modes:
    fout.write('FREE-FREE MODES:\n\n')
    fout.write('\tMode   Frequency (Hz)\n')
    fout.write('\t----   --------------\n')
    writer.vecwrite(fout, '\t{:4d}  {:15.6f}\n',
                    np.arange(n)+1, np.sqrt(w)/(2*math.pi))

    # compute modal-effective mass:
    if nq > 0:
        # effective mass in percent of total mass:
        effmass = (m[QB] @ rbg)**2 * (100/np.diag(mg))
        num = np.arange(nq)+1
        frq = np.sqrt(np.abs(np.diag(kqq)))/(2*math.pi)
        summ = np.sum(effmass, axis=0)
        fout.write('\n\nFIXED-BASE MODES w/ Percent Modal Effective'
                   ' Mass:\n\n')
        fout.write('Using geometry-based rb modes for effective mass '
                   'calcs.\n')
        if em_filt > 0:
            fout.write('\nPrinting only the modes with at least '
                       '{:.1f}% effective mass.\n'
                       'The sum includes all modes.\n'.
                       format(em_filt))
            pv = np.any(effmass > em_filt, axis=1)
            num = num[pv]
            frq = frq[pv]
            effmass = effmass[pv]
        frm = '{:6d}     {:10.3f}    ' + '  {:6.2f}'*6 + '\n'
        dirstr = '    T1      T2      T3      R1      R2      R3\n'
        linestr = '  ------  ------  ------  ------  ------  ------\n'
        fout.write('\nMode No.  Frequency (Hz) ' + dirstr)
        fout.write('--------  -------------- ' + linestr)
        writer.vecwrite(fout, frm, num, frq, effmass)
        fout.write(('\nTotal Effective Mass:'
                    '    ' + '  {:6.2f}'*6 + '\n').format(*summ))
    else:
        fout.write('\n\nThere are no modes for the modal-effective-'
                   'mass check.\n')
    return SimpleNamespace(m=m, k=k, bset=bset, rbs=rbs, rbg=rbg,
                           rbe=rbe, uset=uset)


def cbcheck(f, Mcb, Kcb, bseto, bref, uset=None,
            uref=[0, 0, 0], conv=None, em_filt=0, rb_norm=None,
            reorder=True):
    """
    Run model checks on Craig-Bampton mass and stiffness matrices.

    Parameters
    ----------
    f : string, file handle, or 1
        If a string, it is a filename that gets created. `f` can
        also be a handle to an open file or just 1 to write to the
        standard output (normally, the screen).
    Mcb : 2d ndarray
        Craig-Bampton mass.
    Kcb : 2d ndarray
        Craig-Bampton stiffness.
    bseto : 1d ndarray
        Index partition vector specifying location and order of b-set
        (boundary) DOF in Mcb and Kcb. Will be used to reorder Mcb and
        Kcb if `reorder` is True; see below. Uses zero offset.
    bref : 1d ndarray
        6-element subset of `bseto` (or equal to `bseto` if `bseto`
        only has 6 elements). Defines reference DOF that must be a
        statically-determinate set capable of restraining all
        rigid-body motion (similar to the SUPORT card in
        Nastran). These DOF are used for the stiffness and eigenvalue
        based rigid-body modes. If `bref` is not all 6-DOF of a single
        node, you'll also want to set `rb_norm` to True.
    uset : 2d ndarray; optional
        A 6-column matrix as output by :func:`op2.rdn2cop2` or
        :func:`n2p.addgrid`. For information on the format of this
        matrix, see :func:`op2.rdn2cop2`. If `uset` is None, a single
        grid with id 1 will be automatically created at (0, 0, 0) and
        `uref` will be set to 1. The :func:`n2p.addgrid` call for this
        is::

           uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)

    uref : integer or array; optional
        Defines reference for geometry-based rigid-body modes in a
        format compatible with :func:`n2p.rbgeom_uset`: either an
        integer grid id defined in `uset`, or a 3-element vector
        specifying a location in the basic coordinate system. If a
        3-element vector, it is in the old units (before `conv` is
        used to convert units).
    conv : None or 2-element array_like or string; optional
        If None, no unit conversion is done. If 2-element array_like,
        it is::

            (length_conversion, mass_conversion)

        If string, it is one of:

            * 'm2e' (convert from metric to English)
            * 'e2m' (convert from English to metric)

        The string form assumes units of meter & kg, and inch &
        lbf*s**2/inch (slinch). See :func:`cbconvert` for more
        information.
    em_filt : scalar; optional
        Effective mass print filter: only modes with percent mass
        above `em_filt` will be filtered. For example, to filter out
        modes below 2% modal effective mass, set `em_filt` to 2.0.
    rb_norm : bool or None; optional
        If None, `rb_norm` will be set to True if and only if `bref`
        represents non-contiguous DOF. If True, the stiffness and
        eigenvalue-based rigid-body modes will be normalized such that
        they are relative to `uref` (as the geometry-based modes
        are). This is needed in cases where a single node cannot
        restrain all rigid-body motion; see `bref`. This would happen
        for example if the drilling DOF were not connected for the
        boundary DOF. For more information, see the related discussion
        in :func:`cbcoordchk` for the "rb_normalizer" input.
    reorder : bool; optional
        If True, reordering is allowed.

    Returns
    -------
    A record (SimpleNamespace class) with the members:

    m : 2d ndarray
        Reordered and converted version of Mcb. Will equal Mcb if
        there is no reordering or unit conversion.
    k : 2d ndarray
        Reordered and converted version of Kcb. Will equal Kcb if
        there is no reordering or unit conversion.
    bset : 1d ndarray
        Vector giving location of reordered b-set. This will equal
        numpy.arange(len(bset)) if `reorder` is True. Will equal
        `bseto` if there is no reordering.
    rbs : 2d ndarray
        The stiffness-based rigid-body modes (b+q x 6). DOF order is
        consistent with returned `m` and `k`.
    rbg : 2d ndarray
        The geometry-based rigid-body modes  (b x 6). DOF order is
        consistent with returned `m` and `k`.
    rbe : 2d ndarray
        The eigenvalue-based rigid-body modes (b+q x 6). DOF order is
        consistent with returned `m` and `k`.
    uset : 2d ndarray
        Converted, reordered version of the input `uset`. Will equal
        `uset` if there is no unit conversion or reordering.

    Notes
    -----
    Reordering, if allowed, will always put the bset first.

    This routine performs these checks:

       - Checks symmetry of Mcb and Kcb.
       - Prints some abs-max values from Mcb and Kcb for visual
         inspection.
       - Converts units and reorders mass, stiffness and uset if
         needed.
       - Calculate the 3 types of rigid-body modes (all are relative
         to reordered DOF).
       - Calculates coordinates of boundary DOF relative to a reference
         (`bref`) from the stiffness matrix.
       - Computes the root-sum-squared distances of motion for each
         boundary grid for each type of rb-mode. These distances
         should be 1.
       - Similarly, computes the root-sum-squared rotations for each
         boundary grid for each type of rb-mode. These distances
         should be 1.
       - Does various mass property checks using 3 types of rigid-body
         modes:  stiffness-, geometry-, and eigensolution-based. The
         following items are printed for checking:

            - the three 6x6 mass matrices
            - the CG location relative to respective reference point
              (`bref` for the stiffness and eigensolution based rb
              modes and Uref for the geometry-based rb modes ... which
              means the geometry-based CG will only match the other two
              if the reference is the same).
            - the radius of gyration from the CG about the coordinate
              axes and about the principal axes.

       - Computes stiffness grounding checks against the three types of
         rb modes.
       - Computes the free-free modes.
       - Computes the fixed-base modes and percent modal effective
         mass.

    `bseto` is used to reorder the matrices via the function
    :func:`cbreorder`. As a simple example, assume there are 3 modal
    DOF followed by 12 b-set DOF (two interface grids). Also assume
    that it is desired to switch the order of these two grids. `bseto`
    should then be defined as::

       bseto = [9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]

    In other words, specify the row/col position as it is within Mcb
    and Kcb, in the order you wish them to be. In this case, rows 10
    through 15 are wanted to be first, 4 through 9 next, and finally,
    1 through 3.

    Pay special attention to any warning messages.

    Example usage::

        import numpy as np
        from pyyeti import op4
        from pyyeti import cb
        from pyyeti import nastran
        o4 = op4.OP4()

        names, mats, *_ = o4.listload('nas2cam_csuper/inboard_mk.op4')
        uset, coords = nastran.bulk2uset('nas2cam_csuper/inboard.asm')
        m, k, *_ = mats[0], mats[1]
        b = np.arange(24)
        cb.cbcheck('inboard.cbcheck', m, k, b, b[:6], uset=uset)

    See also
    --------
    :func:`rbmultchk`, :func:`rbdispchk`, :func:`cbcoordchk`,
    :func:`n2p.addgrid`, :func:`cbconvert`, :func:`cbreorder`,
    :func:`op2.rdn2cop2`

    """
    return ytools.wtfile(f, _cbcheck, Mcb, Kcb, bseto, bref, uset,
                         uref, conv, em_filt, rb_norm, reorder)
