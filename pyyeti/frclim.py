# -*- coding: utf-8 -*-
"""
Tools for force limiting.
"""

from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from pyyeti import cb, ode


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def calcAM(S, freq):
    """
    Calculate apparent mass

    Parameters
    ----------
    S : list/tuple
        Contains: ``[mass, damp, stiff, bdof]`` for structure. These
        are the source mass, damping, and stiffness matrices (see
        :class:`pyyeti.ode.SolveUnc`) and `bdof`, which is described
        below.
    freq : 1d array_like
        Frequency vector (Hz)

    Returns
    -------
    AM : 3d ndarray
        Apparent mass matrix in a 3d array:

        .. code-block:: none

            boundary DOF x Frequencies x boundary DOF
             (response)                   (input)

    Notes
    -----
    The `bdof` input defines boundary DOF in one of two ways as
    follows. Let `N` be total number of DOF in mass, damping, &
    stiffness.

       1.  If `bdof` is a 2d array_like, it is interpreted to be a
           data recovery matrix to the b-set (number b-set =
           ``bdof.shape[0]``). Structure is treated generically (uses
           :class:`pyyeti.ode.SolveUnc` with ``pre_eig=True`` to
           compute apparent mass).
       2.  Otherwise, `bdof` is assumed to be a 1d partition vector
           from full `N` size to b-set and structure is assumed to be
           in Craig-Bampton form (uses :func:`pyyeti.cb.cbtf` to
           compute apparent mass).

    The routine :func:`ntfl` example demonstrates this function.

    See also
    --------
    :func:`ntfl`.
    """
    lf = len(freq)
    m = S[0]
    b = S[1]
    k = S[2]
    bdof = np.atleast_1d(S[3])

    if bdof.ndim == 2:  # bdof is treated as a drm
        r = bdof.shape[0]
        T = bdof
        Frc = np.zeros((r, lf))
        Acc = np.empty((r, lf, r), dtype=complex)
        fs = ode.SolveUnc(m, b, k, pre_eig=True)
        if hasattr(fs.pc, "eig_success") and not fs.pc.eig_success:
            print(
                "Switching from `SolveUnc` to `FreqDirect` because complex"
                " eigensolver failed; see messages above."
            )
            fs = ode.FreqDirect(m, b, k)
        for direc in range(r):
            Frc[direc, :] = 1.0
            sol = fs.fsolve(T.T @ Frc, freq)
            Acc[:, :, direc] = T @ sol.a
            Frc[direc, :] = 0.0
        AM = np.empty((r, lf, r), dtype=complex)
        for j in range(lf):
            AM[:, j, :] = la.inv(Acc[:, j, :])
    else:  # bdof treated as a partition vector for CB model
        r = len(bdof)
        acce = np.eye(r)
        # Perform Baseshake
        # cbtf = craig bampton transfer function; this will genenerate
        # the corresponding interface force required to meet imposed
        # acceleration
        AM = np.empty((r, lf, r), dtype=complex)
        save = {}
        for direc in range(r):
            tf = cb.cbtf(m, b, k, acce[direc, :], freq, bdof, save)
            AM[:, :, direc] = tf.frc
    return AM


def ntfl(Source, Load, As, freq):
    r"""
    Norton Thevenin Force Limit

    Parameters
    ----------
    Source : list/tuple or 3d ndarray
        Can be either:

           1. list/tuple of ``[mass, damp, stiff, bdof]`` for source
              (eg, launch vehicle). These are the source mass,
              damping, and stiffness matrices (see
              :class:`pyyeti.ode.SolveUnc`) and `bdof`, which is
              described below.
           2. SAM, a 3d ndarray of source apparent mass (from a
              previous run). See description of outputs.

    Load :  list/tuple or 3d ndarray
        Same format as `Source` except for the "load" (eg spacecraft)
    As : 2d array_like
        Free acceleration of the source (interface acceleration
        without the Load attached).
    freq : 1d array_like
        Frequency vector in Hz for `Source`, `Load`, `As` and all
        return values.

    Returns
    -------
    A SimpleNamespace with the members:

    R : 2d ndarray
        Norton Thevenin normalized response ratio; complex,
        # bdof x len(freq). `R` is independent of `As`. Each row of
        `R` is the ratio of loaded-response over free-response
        assuming a unit input in that direction (with the other
        directions fixed).
    A : 2d ndarray
        Coupled system interface acceleration, complex, # bdof x
        len(freq).
    F : 2d ndarray
        Coupled system interface force, complex, # bdof x len(freq)
    LAM, SAM, TAM : 3d ndarrays
        Load, Source and Total apparent mass matrices:

        .. code-block:: none

            boundary DOF x Frequencies x boundary DOF
             (response)                   (input)

        ``TAM = SAM + LAM``

    freq : 1d array_like
        Copy of the input `freq`.

    Notes
    -----
    The outputs are computed as follows:

    .. math::

        \begin{aligned}
        R(f) &= diag(T_{AM}(f)^{-1} \cdot L_{AM}(f)) \\
        A(f) &= T_{AM}(f)^{-1} \cdot L_{AM}(f) \cdot A_s(f) \\
        F(f) &= S_{AM}(f) \cdot T_{AM}(f)^{-1} \cdot L_{AM}(f) \cdot A_s(f)
        \end{aligned}

    The `bdof` input defines boundary DOF in one of two ways as
    follows. Let `N` be total number of DOF in mass, damping, &
    stiffness.

       1.  If `bdof` is a 2d array_like, it is interpreted to be a
           data recovery matrix to the b-set (number b-set =
           ``bdof.shape[0]``). Structure is treated generically (uses
           :class:`pyyeti.ode.SolveUnc` with ``pre_eig=True`` to
           compute apparent mass).
       2.  Otherwise, `bdof` is assumed to be a 1d partition vector
           from full `N` size to b-set and structure is assumed to be
           in Craig-Bampton form (uses :func:`pyyeti.cb.cbtf` to
           compute apparent mass).

    Note that the Source and Load `bdof` must define the same number
    of boundary DOF and both sets of boundary DOF must be in the same
    coordinate system.

    Tips:

    - If using a free-free model, include residual vectors defined
      relative to the boundary DOF.
    - The free acceleration of the source is the complex response of
      the boundary DOF without the load attached. Frequency envelopes
      of flight data, or vibration specifications derived from
      envelopes, are not accurate ways to come up with the free
      acceleration.

    Notional example::

        from pyyeti import frclim
        from pyyeti.nastran op2, n2p, op4
        import pickle

        # Load free acceleration of LV
        dct = pickle.load('ifresults_free.p')
        As = dct['As']
        freq = dct['freq']

        # Load source free-free model, with residual vectors included
        nas = op2.rdnas2cam('nas2cam')
        m1 = None
        zeta = 0.01
        k1 = nas['lambda'][0]
        k1[:nas['nrb']] = 0.
        b1 = 2*zeta*np.sqrt(k1)

        # Transformation to i/f node:
        T, dof = n2p.formdrm(nas, seup=0, sedn=0, dof=888888)

        # Load S/C mass and stiffness:
        mk = op4.read('mk.op4')
        kgen = mk['kgen']
        mgen = mk['mgen']
        kgen[:6, :6] = 0.
        zeta = 0.01
        bgen = np.diag(2*zeta*np.sqrt(np.diag(kgen)))

        # Norton Thevenin force limit function:
        r = frclim.ntfl([m1, b1, k1, T], [mgen, bgen, kgen,
                        np.arange(6)], As, freq)

    See also
    --------
    :func:`sefl`, :func:`ctdfs`, :func:`stdfs`.

    Examples
    --------
    This example sets up a simple mass-spring system to demonstrate
    that the Norton-Thevenin equations can be exact.

    Steps:

      1. setup a coupled system of a SOURCE and a LOAD
      2. solve for interface acceleration and force from coupled system
         (frequency domain)
      3. calculate free acceleration from SOURCE alone and setup LOAD
      4. use :func:`ntfl` to couple the system
      5. plot interface acceleration and force to show :func:`ntfl` can
         be an exact coupling method
      6. plot apparent masses
      7. plot free acceleration and coupled acceleration
      8. plot normalized response ratio (can be thought of as force
         limiting factor)

    1. Setup system:

    .. code-block:: none

                |--> x1       |--> x2        |--> x3        |--> x4
                |             |              |              |
             |----|    k1   |----|    k2   |----|    k3   |----|
         Fe  |    |--\/\/\--|    |--\/\/\--|    |--\/\/\--|    |
        ====>| 10 |         | 30 |         |  3 |         |  2 |
             |    |---| |---|    |---| |---|    |---| |---|    |
             |----|    c1   |----|    c2   |----|    c3   |----|

             |<--- SOURCE --->||<------------ LOAD ----------->|

    Define parameters:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti import frclim, ode
        >>> freq = np.arange(0.0, 25.1, 0.1)
        >>> M1 = 10.
        >>> M2 = 30.
        >>> M3 = 3.
        >>> M4 = 2.
        >>> c1 = 15.
        >>> c2 = 15.
        >>> c3 = 15.
        >>> k1 = 45000.
        >>> k2 = 25000.
        >>> k3 = 10000.

        2. Solve coupled system:

        >>> MASS = np.array([[M1, 0, 0, 0],
        ...                  [0, M2, 0, 0],
        ...                  [0, 0, M3, 0],
        ...                  [0, 0, 0, M4]])
        >>> DAMP = np.array([[c1, -c1, 0, 0],
        ...                  [-c1, c1+c2, -c2, 0],
        ...                  [0, -c2, c2+c3, -c3],
        ...                  [0, 0, -c3, c3]])
        >>> STIF = np.array([[k1, -k1, 0, 0],
        ...                  [-k1, k1+k2, -k2, 0],
        ...                  [0, -k2, k2+k3, -k3],
        ...                  [0, 0, -k3, k3]])
        >>> F = np.vstack((np.ones((1, len(freq))),
        ...                np.zeros((3, len(freq)))))
        >>> fs = ode.SolveUnc(MASS, DAMP, STIF, pre_eig=True)
        >>> fullsol = fs.fsolve(F, freq)
        >>> A_coupled = fullsol.a[1]
        >>> F_coupled = (M2/2*A_coupled -
        ...              k2*(fullsol.d[2] - fullsol.d[1]) -
        ...              c2*(fullsol.v[2] - fullsol.v[1]))

        3. Solve for free acceleration; SOURCE setup: [m, b, k, bdof]:

        >>> ms = np.array([[M1, 0], [0, M2/2]])
        >>> cs = np.array([[c1, -c1], [-c1, c1]])
        >>> ks = np.array([[k1, -k1], [-k1, k1]])
        >>> source = [ms, cs, ks, [[0, 1]]]
        >>> fs_source = ode.SolveUnc(ms, cs, ks, pre_eig=True)
        >>> sourcesol = fs_source.fsolve(F[:2], freq)
        >>> As = sourcesol.a[1:2]   # free acceleration

        LOAD setup: [m, b, k, bdof]:

        >>> ml = np.array([[M2/2, 0, 0], [0, M3, 0], [0, 0, M4]])
        >>> cl = np.array([[c2, -c2, 0], [-c2, c2+c3, -c3], [0, -c3, c3]])
        >>> kl = np.array([[k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])
        >>> load = [ml, cl, kl, [[1, 0, 0]]]

        4. Use NT to couple equations. First value (rigid-body motion)
        should equal ``Source Mass / Total Mass = 25/45 = 0.55555...``
        Results should match the coupled method.

        >>> r = frclim.ntfl(source, load, As, freq)
        >>> abs(r.R[0, 0])   # doctest: +ELLIPSIS
        0.55555...
        >>> np.allclose(A_coupled, r.A)
        True
        >>> np.allclose(F_coupled, r.F)
        True

        Calculate the total apparent mass directly using :func:`calcAM`.
        This should match the ``r.TAM`` value.

        >>> TAM = frclim.calcAM((MASS, DAMP, STIF, [[0, 1, 0, 0]]), freq)
        >>> np.allclose(TAM, r.TAM)
        True
        >>> np.allclose(TAM, r.SAM+r.LAM)
        True

        5. Plot comparisons:

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.figure('Example')
        >>> plt.clf()
        >>> _ = plt.subplot(211)
        >>> _ = plt.semilogy(freq, abs(A_coupled),
        ...                  label='Coupled')
        >>> _ = plt.semilogy(freq, abs(r.A).T, '--',
        ...                  label='NT')
        >>> _ = plt.title('Interface Acceleration')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.subplot(212)
        >>> _ = plt.semilogy(freq, abs(F_coupled),
        ...                  label='Coupled')
        >>> _ = plt.semilogy(freq, abs(r.F).T, '--',
        ...                  label='NT')
        >>> _ = plt.title('Interface Force')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.legend(loc='best')
        >>> plt.tight_layout()

    .. plot::
        :context: close-figs

        6. Plot apparent masses:

        >>> _ = plt.figure('Example 2')
        >>> plt.clf()
        >>> _ = plt.semilogy(freq, abs(r.TAM[0, :, 0]),
        ...                  label='Total App. Mass')
        >>> _ = plt.semilogy(freq, abs(r.SAM[0, :, 0]),
        ...                  label='Source App. Mass')
        >>> _ = plt.semilogy(freq, abs(r.LAM[0, :, 0]),
        ...                  label='Load App. Mass')
        >>> _ = plt.title('Apparent Mass Comparison')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.legend(loc='best')

    .. plot::
        :context: close-figs

        7. Plot accelerations and
        8. Plot force limit factor:

        >>> _ = plt.figure('Example 3')
        >>> plt.clf()
        >>> _ = plt.subplot(211)
        >>> _ = plt.semilogy(freq, abs(As).T,
        ...                  label='Free Acce')
        >>> _ = plt.semilogy(freq, abs(r.A).T,
        ...                  label='Coupled Acce')
        >>> _ = plt.title('Interface Acceleration')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.subplot(212)
        >>> _ = plt.semilogy(freq, abs(r.R).T)
        >>> _ = plt.title('NT Response Ratio: '
        ...               'R = Coupled Acce / Free Acce')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> plt.tight_layout()
    """
    # Calculate apparent masses:
    if isinstance(Source, (list, tuple)):
        SAM = calcAM(Source, freq)
    else:
        SAM = Source

    if isinstance(Load, (list, tuple)):
        LAM = calcAM(Load, freq)
    else:
        LAM = Load

    As = np.atleast_2d(As)
    if not len(freq) == As.shape[1] == SAM.shape[1] == LAM.shape[1]:
        raise ValueError(
            "incompatible sizes: ensure that `Source`, "
            "`Load`, and `As` all use the same frequency "
            "vector `freq`"
        )

    TAM = SAM + LAM

    # Application of Norton-Thevenin equations
    r, c, _ = SAM.shape
    R = np.empty((r, c), dtype=complex)
    A = np.empty((r, c), dtype=complex)
    F = np.empty((r, c), dtype=complex)
    for j in range(c):
        Ms = SAM[:, j, :]
        Ml = LAM[:, j, :]
        Mr = la.solve(Ms + Ml, Ms)
        R[:, j] = np.diag(Mr)
        A[:, j] = Mr @ As[:, j]
        F[:, j] = Ml @ A[:, j]
    return SimpleNamespace(R=R, F=F, A=A, LAM=LAM, SAM=SAM, TAM=TAM, freq=freq)


def sefl(c, f, f0):
    """
    Semi-empirical normalized force limit.

    Parameters
    ----------
    c : scalar
        Constant based on experience, typically around 1.5
    f : scalar
        Frequency of interest, typically lower end of band.
    f0 : scalar
        Fundamental frequency in direction of interest.

    Returns
    -------
    nfl : scalar
        The normalized force limit:

        .. code-block:: none

            nfl = c                 f <= f0
            nfl = c / (f/f0)        f > f0

    Notes
    -----
    See reference [#fl1]_ for more information on force limiting.

    References
    ----------
    .. [#fl1] Scharton, T.D. (1997). 'Force Limited Vibration Testing
        Monograph'. NASA. Reference Publication RP-1403, JPL.

    See also
    --------
    :func:`ntfl`, :func:`ctdfs`, :func:`stdfs`.

    Examples
    --------
    Compute force limit for a s/c attached to a launch vehicle, where
    the interface acceleration specification level is 1.75 g, and:

     - frequency of interest starts at 75 Hz
     - fundamental axial mode of s/c is 40 Hz

    >>> from pyyeti import frclim
    >>> m = 6961    # s/c mass
    >>> faf = 40    # fundamental axial frequency of s/c
    >>> spec = 1.75
    >>> m*spec*frclim.sefl(1.5, 75, faf)
    9745.4
    """
    if f <= f0:
        return c
    return f0 * c / f


def stdfs(mr, Q):
    r"""
    Compute the normalized force limit for simple 2-DOF system.

    Parameters
    ----------
    mr : scalar
        Mass ratio m2/m1; 0.0001 to 10 is reasonable.
    Q : scalar to 2 element array_like
        Dynamic amplification factor, 1/2/zeta. If a scalar, the same
        value is used for both dampers. If a 2 element vector, it is
        [Q1, Q2] (see figure below).

    Returns
    -------
    nfl : scalar
        The normalized force limit for m2:

        .. code-block:: none

            nfl = force_limit / (m2 * max(a1))

    Notes
    -----
    The simple 2-DOF system:

    .. code-block:: none

            |---> a0
            |                     |---> a1           |---> a2
        |---------|               |                  |
        |         |     k1     |-----|     k2     |-----|
        |  rigid  |----\/\/\---|     |----\/\/\---|     |
        |  base   |            | m1  |            | m2  |
        |  (M)    |----| |-----|     |----| |-----|     |
        |         |     c1     |-----|     c2     |-----|
        |---------|           (source)            (load)

    Analysis is done in the frequency domain. Methodology:

      1. Define stiffnesses such that ``k1/m1 = k2/m2`` (this is worst
         case, or very near)
      2. Excite M, bounding frequency range of interest
      3. Recover maximum interface acceleration: ``A`` (accel of m1)
      4. Recover maximum interface force on m2: ``F``
      5. Compute normalized force limit: ``F / (A * m2)``

    Note: higher masses result in higher force limits.

    For multi-dof systems, m1 and m2 can be defined as the modal mass
    in the frequency range of interest.  Or, for more conservatism, m1
    and m2 can be defined as the modal mass plus any residual mass.

    The modal masses are defined relative to the interface point. See
    references [#fl2]_ and [#fl3]_ for more information.

    References
    ----------
    .. [#fl2] Scharton, T.D. (1997). 'Force Limited Vibration Testing
        Monograph'. NASA. Reference Publication RP-1403, JPL.
    .. [#fl3] Y. Soucy, A. Cote, 'Reduction of Overtesting during
        Vibration Tests of Space Hardware', Can. Aeronautics and Space
        J., Vol. 48, No. 1, pp. 77-86, 2002.

    See also
    --------
    :func:`ntfl`, :func:`ctdfs`, :func:`sefl`.

    Examples
    --------
    Compute force limit for a s/c attached to a launch vehicle, where
    the interface acceleration specification level is 1.75 g and Q is
    assumed to be 10:

    >>> from pyyeti import frclim
    >>> m1 = 710     # modal mass + residual mass of lv
    >>> m2 = 3060    # modal mass + residual mass of s/c
    >>> Q = 10
    >>> spec = 1.75
    >>> frclim.stdfs(m2/m1, Q) * m2 * 1.75   # doctest: +ELLIPSIS
    6393.1622...

    Generate some curves showing the normalized force limit vs the
    mass ration for a few different damping values:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import frclim
        >>> mr = np.geomspace(0.00001, 100, 100)
        >>> for Q in (10, 25, 50):
        ...     nfl = [frclim.stdfs(i, Q) for i in mr]
        ...     _ = plt.loglog(mr, nfl, label=f'Q={Q}')
        >>> _ = plt.legend()
        >>> _ = plt.xlabel('Mass ratio (Load/Source)')
        >>> _ = plt.ylabel('Normalized Force Limit')
    """
    m2 = 0.1
    M = 1e6
    w1 = 1.0
    w2 = 1.0
    Q = np.atleast_1d(Q)
    if len(Q) == 1:
        zeta1 = zeta2 = 1 / 2 / Q[0]
    else:
        zeta1 = 1 / 2 / Q[0]
        zeta2 = 1 / 2 / Q[1]
    m1 = m2 / mr
    c1 = 2 * zeta1 * w1 * m1
    c2 = 2 * zeta2 * w2 * m2
    k1 = w1 ** 2 * m1
    k2 = w2 ** 2 * m2

    # setup system equations of motion:
    mass = np.array([[M, 0, 0], [0, m1, 0], [0, 0, m2]])
    damp = np.array([[c1, -c1, 0], [-c1, c1 + c2, -c2], [0, -c2, c2]])
    stif = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2]])
    w = la.eigh(stif, mass, eigvals_only=True)

    # lam = np.sqrt(abs(w))/w2
    # freq = w2*np.arange(.2, 5, 0.001)/2/np.pi
    # F = np.zeros((3, len(freq)))
    lam = np.sqrt(abs(w))
    freq = lam[1] / 2 / np.pi
    F = np.zeros((3, 1))
    F[0] = 1e6
    fs = ode.FreqDirect(mass, damp, stif)
    sol = fs.fsolve(F, freq)

    spec_level = np.max(abs(sol.a[1]))
    ifforce = np.max(abs(m2 * sol.a[2]))
    return ifforce / spec_level / m2


def _ctdfs_old(mmr1, mmr2, rmr, Q, wr=(1 / np.sqrt(2), np.sqrt(2))):
    r"""
    Compute the normalized force limit for complex 2-DOF system.

    Parameters
    ----------
    mmr1 : scalar
        Modal to residual mass ratio for source; 0.0001 to 10 is
        reasonable ``m1/M1``
    mmr2 : scalar
        Modal to residual mass ratio for load; 0.0001 to 10 is
        reasonable ``m2/M2``
    rmr : scalar
        Residual mass ratio of source over load; 0.0001 to 10 is
        reasonable ``M2/M1``
    Q : scalar to 2 element array_like
        Dynamic amplification factor, 1/2/zeta. If a scalar, the same
        value is used for both dampers. If a 2 element vector, it is
        [Q1, Q2] (see figure below).
    wr : 2 element array_like

        Two element tuning range for frequency of LOAD. `wr` is a
        ratio of the LOAD frequency to the SOURCE::

            wr = [ w2_min w2_max ] / w1

        Scharton used [1/sqrt(2), sqrt(2)] in [#fl4]_.

    Returns
    -------
    nfl : scalar
        Normalized force limit for M2::

            nfl = force_limit / (M2 * max(a2))

    nw2 : scalar
        Normalized tuned frequency of LOAD::

            nw2 = w2_tuned / w1

    Notes
    -----
    This routine computes the normalize force limit for M2 in the
    complex 2-DOF system (2 flexible body modes)::


               |--> a1         |--> a2        |--> a3         |--> a4
               |               |              |               |
             |----|    k1    |----|    F    |----|    k2    |----|
         F1  |    |---\/\/\--|    |<------->|    |---\/\/\--|    |
        ---->| m1 |          | M1 |         | M2 |          | m2 |
             |    |---| |----|    |  a2=a3  |    |---| |----|    |
             |----|    c1    |----|<------->|----|    c2    |----|
              modal         residual       residual          modal
                    S O U R C E                     L O A D

    Analysis is done in the frequency domain.  Methodology:

      1. Set ``w1 = sqrt(k1/m1) = 1`` and ``M1 = 1``
      2. Tune ``w2 = sqrt(k2/m2)`` such that a worst case force limit
         is achieved within pre-defined frequency limits according to
         input `wr`. Frequency range is limited because modal and
         residual masses as input are only valid in a limited
         frequency range.  For each w2:

         a. Compute natural frequencies and solve equations of motion
            at the two flexible frequencies. (Note: this is not
            guaranteed to be the worst-case frequencies: almost 10%
            "errors" have been seen for ``Q = 5`` systems. Lower
            damping is much closer.)
         b. Recover maximum interface accel: ``A`` (accel of M2)
         c. Recover maximum interface force on M2: ``F``
         d. Compute `nfl`: ``F / (A * M2)``

      3. Keep the maximum `nfl` from 2.

    The modal masses are defined relative to the interface point. See
    references [#fl4]_ and [#fl5]_ for more information.

    References
    ----------
    .. [#fl4] Scharton, T.D. (1997). 'Force Limited Vibration Testing
        Monograph'. NASA. Reference Publication RP-1403, JPL.
    .. [#fl5] Y. Soucy, A. Cote, 'Reduction of Overtesting during
        Vibration Tests of Space Hardware', Can. Aeronautics and Space
        J., Vol. 48, No. 1, pp. 77-86, 2002.

    See also
    --------
    :func:`ntfl`, :func:`stdfs`, :func:`sefl`.

    Examples
    --------
    Compute force limit for a s/c attached to a launch vehicle, where
    the interface acceleration specification level is 1.75 g and Q is
    assumed to be 10. Compare against the :func:`stdfs` and
    :func:`sefl` methods:

    >>> from pyyeti import frclim
    >>> m1 = 30     # lv modal mass 75-90 Hz
    >>> M1 = 622    # lv residual mass above 90 Hz
    >>> m2 = 972    # sc modal mass 75-90 Hz
    >>> M2 = 954    # sc residual mass above 90 Hz
    >>> msc = 6961  # total sc mass
    >>> faf = 40    # fundamental axial frequency of s/c
    >>> Q = 10
    >>> spec = 1.75
    >>> (frclim._ctdfs_old(m1/M1, m2/M2, M2/M1, Q)[0] *
    ...    M2 * spec)                             # doctest: +ELLIPSIS
    8686.1...
    >>> (frclim.stdfs((m2+M2)/(m1+M1), Q) *
    ...    (m2+M2) * spec)                        # doctest: +ELLIPSIS
    4268.2...
    >>> frclim.sefl(1.5, 75, faf) * msc * spec
    9745.4
    """
    M1 = 1.0
    w1 = 1.0
    Q = np.atleast_1d(Q)
    if len(Q) == 1:
        zeta1 = zeta2 = 1 / 2 / Q[0]
    else:
        zeta1 = 1 / 2 / Q[0]
        zeta2 = 1 / 2 / Q[1]

    M2 = M1 * rmr
    m1 = M1 * mmr1
    m2 = M2 * mmr2

    if m2 == 0:
        return 1, 1
    if m1 == 0:
        m1 = 1e-5

    c1 = 2 * zeta1 * w1 * m1
    k1 = w1 ** 2 * m1
    fl = wr[0]  # low bound
    fh = wr[1]  # high bound

    wrange = np.logspace(np.log10(fl), np.log10(fh), 30)
    step = wrange[-1] - wrange[-2]
    maxstep = 0.0001
    tol = 1e-6
    err = 1
    J = 0
    last = 0
    F = np.zeros((3, 2))
    F[0] = 1.0

    mass = np.array([[m1, 0, 0], [0, M1 + M2, 0], [0, 0, m2]])
    while True:
        J += 1
        pknfl = np.zeros_like(wrange)
        for j, w2 in enumerate(wrange * w1):  # tuning loop
            c2 = 2 * zeta2 * w2 * m2
            k2 = w2 ** 2 * m2

            # solve equations of motion at eigenvalues
            damp = np.array([[c1, -c1, 0], [-c1, c1 + c2, -c2], [0, -c2, c2]])
            stif = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2]])
            w = la.eigh(stif, mass, eigvals_only=True)
            lam = np.sqrt(abs(w))
            fq2 = lam[1] / 2 / np.pi
            fq3 = lam[2] / 2 / np.pi
            freq = np.array([fq2, fq3])

            # method 1: solve system only at natural frequencies
            fs = ode.FreqDirect(mass, damp, stif)
            sol = fs.fsolve(F, freq)
            d4 = sol.d[2]
            d2 = sol.d[1]
            a2 = sol.a[1]
            v2 = sol.v[1]
            v4 = sol.v[2]
            ifforce = abs(k2 * (d4 - d2) - M2 * a2 + c2 * (v4 - v2))
            ifaccel = abs(a2)
            pknfl[j] = max(ifforce) / max(ifaccel) / M2

        i = np.argmax(pknfl)
        nfl = pknfl[i]
        if J > 1 and step < maxstep:
            err = abs(last / nfl - 1)
            if err < tol:
                nw2 = wrange[i]
                break
        last = nfl
        fl_new = fl if i == 0 else wrange[i - 1]
        fh_new = fh if i == len(wrange) - 1 else wrange[i + 1]
        wrange = np.logspace(np.log10(fl_new), np.log10(fh_new), 30)
        step = wrange[-1] - wrange[-2]
    return nfl, nw2


def ctdfs(mmr1, mmr2, rmr, Q, wr=(1 / np.sqrt(2), np.sqrt(2))):
    r"""
    Compute the normalized force limit for complex 2-DOF system.

    Parameters
    ----------
    mmr1 : scalar
        Modal to residual mass ratio for source; 0.0001 to 10 is
        reasonable ``m1/M1``
    mmr2 : scalar
        Modal to residual mass ratio for load; 0.0001 to 10 is
        reasonable ``m2/M2``
    rmr : scalar
        Residual mass ratio of source over load; 0.0001 to 10 is
        reasonable ``M2/M1``
    Q : scalar to 2 element array_like
        Dynamic amplification factor, 1/2/zeta. If a scalar, the same
        value is used for both dampers. If a 2 element vector, it is
        [Q1, Q2] (see figure below).
    wr : 2 element array_like

        Two element tuning range for frequency of LOAD. `wr` is a
        ratio of the LOAD frequency to the SOURCE::

            wr = [ w2_min w2_max ] / w1

        Scharton used [1/sqrt(2), sqrt(2)] in [#fl4]_.

    Returns
    -------
    nfl : scalar
        Normalized force limit for M2::

            nfl = force_limit / (M2 * max(a2))

    nw2 : scalar
        Normalized tuned frequency of LOAD::

            nw2 = w2_tuned / w1

    Notes
    -----
    This routine computes the normalize force limit for M2 in the
    complex 2-DOF system (2 flexible body modes)::


               |--> a1         |--> a2        |--> a3         |--> a4
               |               |              |               |
             |----|    k1    |----|    F    |----|    k2    |----|
         F1  |    |---\/\/\--|    |<------->|    |---\/\/\--|    |
        ---->| m1 |          | M1 |         | M2 |          | m2 |
             |    |---| |----|    |  a2=a3  |    |---| |----|    |
             |----|    c1    |----|<------->|----|    c2    |----|
              modal         residual       residual          modal
                    S O U R C E                     L O A D

    Analysis is done in the frequency domain.  Methodology:

      1. Set ``w1 = sqrt(k1/m1) = 1`` and ``M1 = 1``
      2. Tune ``w2 = sqrt(k2/m2)`` such that a worst case force limit
         is achieved within pre-defined frequency limits according to
         input `wr`. Frequency range is limited because modal and
         residual masses as input are only valid in a limited
         frequency range.  For each w2:

         a. Compute natural frequencies and solve equations of motion
            at the two flexible frequencies. (Note: this is not
            guaranteed to be the worst-case frequencies: almost 10%
            "errors" have been seen for ``Q = 5`` systems. Lower
            damping is much closer.)
         b. Recover maximum interface accel: ``A`` (accel of M2)
         c. Recover maximum interface force on M2: ``F``
         d. Compute `nfl`: ``F / (A * M2)``

      3. Keep the maximum `nfl` from 2.

    The optimization is carried out by
    :func:`scipy.optimize.minimize_scalar`.

    The modal masses are defined relative to the interface point. See
    references [#fl4]_ and [#fl5]_ for more information.

    References
    ----------
    .. [#fl4] Scharton, T.D. (1997). 'Force Limited Vibration Testing
        Monograph'. NASA. Reference Publication RP-1403, JPL.
    .. [#fl5] Y. Soucy, A. Cote, 'Reduction of Overtesting during
        Vibration Tests of Space Hardware', Can. Aeronautics and Space
        J., Vol. 48, No. 1, pp. 77-86, 2002.

    See also
    --------
    :func:`ntfl`, :func:`stdfs`, :func:`sefl`.

    Examples
    --------
    Compute force limit for a s/c attached to a launch vehicle, where
    the interface acceleration specification level is 1.75 g and Q is
    assumed to be 10. Compare against the :func:`stdfs` and
    :func:`sefl` methods:

    >>> from pyyeti import frclim
    >>> m1 = 30     # lv modal mass 75-90 Hz
    >>> M1 = 622    # lv residual mass above 90 Hz
    >>> m2 = 972    # sc modal mass 75-90 Hz
    >>> M2 = 954    # sc residual mass above 90 Hz
    >>> msc = 6961  # total sc mass
    >>> faf = 40    # fundamental axial frequency of s/c
    >>> Q = 10
    >>> spec = 1.75
    >>> (frclim.ctdfs(m1/M1, m2/M2, M2/M1, Q)[0] *
    ...    M2 * spec)                             # doctest: +ELLIPSIS
    8686.1...
    >>> (frclim.stdfs((m2+M2)/(m1+M1), Q) *
    ...    (m2+M2) * spec)                        # doctest: +ELLIPSIS
    4268.2...
    >>> frclim.sefl(1.5, 75, faf) * msc * spec
    9745.4
    """
    M1 = 1.0
    w1 = 1.0
    Q = np.atleast_1d(Q)
    if len(Q) == 1:
        zeta1 = zeta2 = 1 / 2 / Q[0]
    else:
        zeta1 = 1 / 2 / Q[0]
        zeta2 = 1 / 2 / Q[1]

    M2 = M1 * rmr
    m1 = M1 * mmr1
    m2 = M2 * mmr2

    if m2 == 0:
        return 1, 1
    if m1 == 0:
        m1 = 1e-5

    c1 = 2 * zeta1 * w1 * m1
    k1 = w1 ** 2 * m1

    def get_neg_pknfl(nw2):
        """
        Computes the negative (for minimization) of peak normalized
        force limit
        """
        w2 = nw2 * w1
        c2 = 2 * zeta2 * w2 * m2
        k2 = w2 ** 2 * m2

        # solve equations of motion at eigenvalues
        damp = np.array([[c1, -c1, 0], [-c1, c1 + c2, -c2], [0, -c2, c2]])
        stif = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2]])
        w = la.eigh(stif, mass, eigvals_only=True)
        lam = np.sqrt(abs(w))
        fq2 = lam[1] / 2 / np.pi
        fq3 = lam[2] / 2 / np.pi
        freq = np.array([fq2, fq3])

        # method 1: solve system only at natural frequencies
        fs = ode.FreqDirect(mass, damp, stif)
        sol = fs.fsolve(F, freq)
        d4 = sol.d[2]
        d2 = sol.d[1]
        a2 = sol.a[1]
        v2 = sol.v[1]
        v4 = sol.v[2]
        ifforce = abs(k2 * (d4 - d2) - M2 * a2 + c2 * (v4 - v2))
        ifaccel = abs(a2)
        return -max(ifforce) / max(ifaccel) / M2

    F = np.zeros((3, 2))
    F[0] = 1.0
    mass = np.array([[m1, 0, 0], [0, M1 + M2, 0], [0, 0, m2]])
    res = minimize_scalar(get_neg_pknfl, bracket=wr)
    if "message" in res:
        raise RuntimeError(
            f"routine :func:`scipy.optimize.minimize_scalar` failed: {res.message}"
        )
    nw2 = res.x
    nfl = -res.fun
    return nfl, nw2
