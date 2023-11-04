# -*- coding: utf-8 -*-
"""
Tools for force limiting.
"""

from types import SimpleNamespace
import warnings
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
from pyyeti import cb, ode


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def calcAM(S, freq, fs=None):
    r"""
    Calculate apparent mass

    Apparent mass is also known as "dynamic mass". The inverse of
    apparent mass is known as "inertance" or "accelerance". See
    :func:`ntfl` for more discussion.

    Parameters
    ----------
    S : list/tuple
        Contains: ``[mass, damp, stiff, bdof]`` for structure. These
        are the Source mass, damping, and stiffness matrices (see
        :class:`pyyeti.ode.SolveUnc`) and `bdof`, which is described
        below.
    freq : 1d array_like
        Frequency vector (Hz)
    fs : class instance or None; optional
        If None, this routine will try :class:`pyyeti.ode.SolveUnc`
        with ``pre_eig=True`` to solve equations of motion in
        frequency domain. If :class:`pyyeti.ode.SolveUnc` fails, this
        routine will then try :class:`pyyeti.ode.FreqDirect`. If `fs`
        is not None, it is excpected to be an instance of
        :class:`pyyeti.ode.SolveUnc` or :class:`pyyeti.ode.FreqDirect`
        (or similar ... must have `.fsolve` method)

        .. note::
            Using the `fs` parameter is the only way to include
            residual vectors statically with this routine.

        .. note::
            The `fs` parameter is ignored if `bdof` is 1d; the
            :func:`pyyeti.cb.cbtf` routine is used to compute apparent
            mass in that case (see Notes below).

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

    .. note::
        In addition to the example shown below, this routine is
        demonstrated in the pyYeti :ref:`tutorial`:
        :doc:`/tutorials/ntfl`. There is also a link to the source
        Jupyter notebook at the top of the tutorial.

    See also
    --------
    :func:`ntfl`.

    Examples
    --------
    Consider the 2 DOF system:

    .. code-block:: none

        |----|    k    |----|
        |    |--\/\/\--|    |
        | m1 |         | m2 |
        |    |---| |---|    |
        |----|    c    |----|

        Where:

            m1 = 10.0 kg
            m2 = 4.0 kg
            k = 5000.0 N/m
            c = 15.0 N/(m/s)

    The following will plot the apparent mass curves relative to each
    of the masses, and annotate the plot with three mass values (4,
    10, & 14 kg) and three frequency values (free-free, with the 4 kg
    mass fixed, and with the 10 kg mass fixed). The percent critical
    damping ratio is also included for each of the three boundary
    conditions. The annotations are meant as an aide to guide
    intuition.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from scipy.linalg import eigh
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import frclim
        >>>
        >>> m1 = 10.0
        >>> m2 = 4.0
        >>> k = 5000.0
        >>> c = 15.0
        >>>
        >>> M = np.diag([m1, m2])
        >>> K = np.array([[k, -k], [-k, k]])
        >>> C = np.array([[c, -c], [-c, c]])
        >>>
        >>> # compute free-free frequencies (1st is 0 Hz rigid-body mode)
        >>> lam_ff, phi_ff = eigh(K, M)
        >>> omega_ff = np.sqrt(abs(lam_ff))
        >>> frq_ff = omega_ff / 2 / np.pi
        >>>
        >>> C_ff = phi_ff.T @ C @ phi_ff
        >>> zeta_ff = C_ff[1, 1] / 2 / omega_ff[1]
        >>>
        >>> freq = np.geomspace(0.5, 100.0, 500)
        >>> fig = plt.figure("apparent mass", clear=True,
        ...                  layout='constrained')
        >>> ax = fig.subplots(1, 1)
        >>>
        >>> fx_frqs = []
        >>> fx_zetas = []
        >>> for fixed, lbl in (
        ...     (np.array([True, False]), "Relative to 10 kg Mass"),
        ...     (np.array([False, True]), "Relative to 4 kg Mass"),
        ... ):
        ...     free = ~fixed
        ...     ff = np.ix_(free, free)
        ...     lam, phi = eigh(K[ff], M[ff])  # compute fixed frequency
        ...
        ...     omega = np.sqrt(lam[0])
        ...     fx_frqs.append(omega / 2 / np.pi)
        ...     fx_zeta = (phi.T @ C[ff] @ phi) / (2 * omega)
        ...     fx_zetas.append(fx_zeta[0, 0])
        ...
        ...     T = np.zeros((1, 2))
        ...     T[0, fixed] = 1.0
        ...     am = frclim.calcAM([M, C, K, T], freq)
        ...     _ = ax.loglog(freq, abs(am[0, :, 0]), label=lbl)
        >>>
        >>> _ = ax.set_title("Apparent Mass")
        >>> _ = ax.set_xlabel("Frequency (Hz)")
        >>> _ = ax.set_ylabel("Apparent Mass (kg)")
        >>> _ = ax.legend(loc="lower right")
        >>>
        >>> opts = dict(lw=1.5, c="gray", zorder=-10)
        >>> _ = ax.axhline(14, ls="--", **opts)
        >>> _ = ax.text(100.0, 14, "14 kg", va="bottom", ha="right")
        >>>
        >>> _ = ax.axhline(10, ls="--", **opts)
        >>> _ = ax.text(100.0, 10, "10 kg", va="top", ha="right")
        >>>
        >>> _ = ax.axhline(4, ls="--", **opts)
        >>> _ = ax.text(100.0, 4, "4 kg", va="bottom", ha="right")
        >>>
        >>> _ = ax.axvline(fx_frqs[0], ls="-.", **opts)
        >>> z = fx_zetas[0] * 100
        >>> _ = label = (
        ...         rf" {fx_frqs[0]:.3f} Hz, $\zeta = {z:.2f}$%"
        ...         "\n (10 kg fixed)"
        ...     )
        >>> _ = ax.text(fx_frqs[0], 50, label, va="bottom", ha="left")
        >>>
        >>> _ = ax.axvline(fx_frqs[1], ls="-.", **opts)
        >>> z = fx_zetas[1] * 100
        >>> _ = label = (
        ...         rf" {fx_frqs[1]:.3f} Hz, $\zeta = {z:.2f}$%"
        ...         "\n (4 kg fixed)"
        ...     )
        >>> _ = ax.text(fx_frqs[1], 115, label, va="bottom", ha="left")
        >>>
        >>> _ = ax.axvline(frq_ff[1], ls="-.", **opts)
        >>> z = zeta_ff * 100
        >>> _ = label = (
        ...         rf" {frq_ff[1]:.3f} Hz, $\zeta = {z:.2f}$%"
        ...         "\n (free-free)"
        ...     )
        >>> _ = ax.text(frq_ff[1], 2.0, label, va="bottom", ha="left")
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

        if fs is None:
            use_freqdirect = False
            try:
                fs = ode.SolveUnc(m, b, k, pre_eig=True)
            except la.LinAlgError:
                use_freqdirect = True
            else:
                if hasattr(fs.pc, "eig_success") and not fs.pc.eig_success:
                    use_freqdirect = True
            if use_freqdirect:
                warnings.warn(
                    "Switching from `SolveUnc` to `FreqDirect` because complex"
                    " eigensolver failed; see messages above. Solution may be slow.",
                    RuntimeWarning,
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
    Norton-Thevenin Force Limit

    Parameters
    ----------
    Source : list/tuple or 3d ndarray
        Can be either:

           1. list/tuple of ``[mass, damp, stiff, bdof]`` for Source
              (eg, launch vehicle). These are the Source mass,
              damping, and stiffness matrices (see
              :class:`pyyeti.ode.SolveUnc`) and `bdof`, which is
              described below.
           2. SAM, a 3d ndarray of Source apparent mass (from a
              previous run). See description of outputs.

    Load :  list/tuple or 3d ndarray
        Same format as `Source` except for the "Load" (eg, spacecraft)
    As : 2d array_like
        Free-acceleration of the Source (interface acceleration
        without the Load attached).
    freq : 1d array_like
        Frequency vector in Hz for `Source`, `Load`, `As` and all
        return values.

    Returns
    -------
    A SimpleNamespace with the members:

    A : 2d ndarray
        Coupled system interface acceleration, complex, # bdof x
        len(freq).
    F : 2d ndarray
        Coupled system interface force, complex, # bdof x len(freq)
    R : 2d ndarray
        Norton-Thevenin normalized response ratio; complex, # bdof x
        len(freq). `R` is independent of `As`. Each row of `R` is the
        diagonal of the ratio of the loaded-response over the
        free-response, which is the same as the diagonal of the Source
        apparent mass over the Total apparent mass.

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
    The Norton-Thevenin (NT) equations couple the Source and Load
    models together in an exact way in the frequency domain. It is a
    very convenient formulation for force-limiting applications simply
    because the Source and Load models are kept separate and coupled
    system responses are computed by using the "apparent mass" of both
    models and the "free-acceleration" of the Source. If those
    quantities are known exactly, the responses computed from the NT
    equations are also exact.

    *Apparent Mass*

    The term "apparent mass" is very apt. Apparent mass is also known
    as "dynamic mass", and its inverse is known as "inertance" or
    "accelerance". If you were to grab some point on a structure and
    push and pull in a sinusoidal fashion at some frequency, the
    amount of mass you would feel resisting you is the apparent
    mass. It varies by frequency. At zero Hz (for rigid-body motion),
    there is no vibration and the mass you feel is the physical mass
    of the structure. At frequencies greater than zero, the structure
    will vibrate and "inertial forces" (mass x acceleration) will come
    into play. Inertial forces can either push against you -- so
    apparent mass > physical mass -- or work with you -- so apparent
    mass < physical mass.

    To compute the apparent mass for the Source (or the Load) relative
    to the boundary "b" DOF that attach to the Load (or the Source),
    consider this frequency domain equation:

    .. math::

        \ddot{X}_{b}(\Omega) = H_{bb}(\Omega) \cdot F_{b}(\Omega)

    The transfer function :math:`H_{bb}(\Omega)` is known as
    "inertance" or "accelerance". By applying unit forces to the "b"
    DOF one at a time (so that :math:`F_{b}(\Omega)` is identity), the
    response :math:`\ddot{X}_{b}(\Omega)` is equal to
    :math:`H_{bb}(\Omega)`. The apparent mass :math:`AM_{bb}(\Omega)`
    is simply the inverse of the accelerance :math:`H_{bb}(\Omega)`:

    .. math::

        \begin{array}{c}
            AM_{bb}(\Omega) = {H_{bb}(\Omega)}^{-1} \\
            F_{b} (\Omega) = AM_{bb}(\Omega) \cdot
                \ddot{X}_{b}(\Omega)
        \end{array}

    This is Newton's second law (``F = ma``) in the frequency
    domain. The routine :func:`calcAM` calculates the apparent mass.

    *Free-Acceleration*

    The term "free-acceleration" is also quite apt. With all Source
    external forces applied as normal, "free-acceleration" is the
    acceleration response of the Source "b" DOF with those DOF in a
    free-free boundary condition (that is, without the Load attached).

    *Deriving the NT Coupling Equations*

    Consider the equations of motion for the Source ("S") or the Load
    ("L"):

    .. math::
        \left\{
            \begin{array}{c}
                \ddot{X}_b(\Omega) \\
                \ddot{X}_o(\Omega)
            \end{array}
        \right\}_{S~or~L} =
        \left[
            \begin{array}{cc}
                H_{bb}(\Omega) & H_{bo}(\Omega) \\
                H_{ob}(\Omega) & H_{oo}(\Omega)
            \end{array}
        \right]_{S~or~L}
        \left\{
            \begin{array}{c}
                F_b(\Omega) + \tilde{F}_b(\Omega) \\
                F_o(\Omega)
            \end{array}
        \right\}_{S~or~L}~~~~~~~(1)

    Here, the "o" DOF are all other DOF (DOF that are not on the
    boundary), :math:`F_b` and :math:`F_o` are externally applied
    forces, and :math:`\tilde{F}_b` are forces from the other
    component.

    For joint compatibility, we need equal accelerations, and equal
    but opposite forces:

    .. math::
        \begin{array}{c}
            \ddot{X}_{b,L}(\Omega) = \ddot{X}_{b,S}(\Omega)
                = \ddot{X}_b(\Omega) \\
            \tilde{F}_{b,L}(\Omega) = - \tilde{F}_{b,S}(\Omega)
                = \tilde{F}_b(\Omega)
        \end{array}

    For the Load without any externally applied forces, :math:`F_{b,L}
    = 0` and :math:`F_{o,L} = 0`, the 1st partition of equation (1)
    becomes:

    .. math::
        \ddot{X}_b(\Omega) = H_{bb,L}(\Omega) \cdot \tilde{F}_b(\Omega)

    Or:

    .. math::
         \tilde{F}_b(\Omega) = AM_{bb,L}(\Omega) \cdot
         \ddot{X}_b(\Omega)~~~~~~~(2)

    To derive an equation for the free-acceleration
    :math:`A_S(\Omega)`, set :math:`\tilde{F}_{b,S} \rightarrow 0`
    (since there is no Load attached) in equation (1):

    .. math::
        A_S(\Omega) = \ddot{X}_{b,S,no-Load}(\Omega) =
            H_{bb,S}(\Omega) \cdot F_{b,S}(\Omega) +
            H_{bo,S}(\Omega) \cdot F_{o,S}(\Omega)

    Therefore, the top of equation (1) for the Source can be written
    as (remembering that :math:`\tilde{F}_{b,S}(\Omega) = -
    \tilde{F}_b(\Omega)`):

    .. math::
        \ddot{X}_b(\Omega) = - H_{bb,S}(\Omega) \cdot
            \tilde{F}_{b}(\Omega) + A_S(\Omega)~~~~~~~(3)

    Substituting equation (2) into (3) and collecting terms:

    .. math::
        \begin{array}{c}
            \ddot{X}_b(\Omega) = - H_{bb,S}(\Omega) \cdot
                AM_{bb,L}(\Omega) \cdot
                \ddot{X}_b(\Omega) + A_S(\Omega) \\
            (I + H_{bb,S}(\Omega) \cdot AM_{bb,L}(\Omega)) \cdot
                \ddot{X}_b(\Omega) = A_S(\Omega)
        \end{array}

    Multiplying that result by the Source apparent mass gives:

    .. math::
        (AM_{bb,S}(\Omega) + AM_{bb,L}(\Omega)) \cdot
            \ddot{X}_b(\Omega) = AM_{bb,S}(\Omega) \cdot A_S(\Omega)

    Solving for :math:`\ddot{X}_b(\Omega)`:

    .. math::
        \ddot{X}_b(\Omega) =
            (AM_{bb,S}(\Omega) + AM_{bb,L}(\Omega))^{-1} \cdot
            AM_{bb,S}(\Omega) \cdot A_S(\Omega)~~~~~~~(4)

    After solving for :math:`\ddot{X}_b(\Omega)` from equation (4), we
    can solve for :math:`\tilde{F}_b(\Omega)` from equation (2). We
    can also use equation (4) to expand equation (2):

    .. math::
         \tilde{F}_b(\Omega) = AM_{bb,L}(\Omega) \cdot
            (AM_{bb,S}(\Omega) + AM_{bb,L}(\Omega))^{-1} \cdot
            AM_{bb,S}(\Omega) \cdot A_S(\Omega)~~~~~~~(5)

    *Outputs of this routine*

    The outputs are computed as follows:

    .. math::
        \begin{aligned}
            A(\Omega) &= \ddot{X}_b(\Omega) = (AM_{bb,S}(\Omega) +
                AM_{bb,L}(\Omega))^{-1} \cdot
                AM_{bb,S}(\Omega) \cdot A_S(\Omega) \\
            F(\Omega) &= \tilde{F}_b(\Omega) = AM_{bb,L}(\Omega) \cdot
                A(\Omega) \\
            R(\Omega) &= diag((AM_{bb,S}(\Omega) +
                AM_{bb,L}(\Omega))^{-1} \cdot AM_{bb,S}(\Omega)) \\
        \end{aligned}

    On output, the 3D apparent mass arrays are named as follows:

    .. math::
        \begin{aligned}
            AM_{bb,L}(\Omega) & \rightarrow LAM \\
            AM_{bb,S}(\Omega) & \rightarrow SAM \\
            AM_{bb,L}(\Omega) + AM_{bb,S}(\Omega) & \rightarrow TAM
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

    *Tips*

    - If using a free-free model, include residual vectors defined
      relative to the boundary DOF. These can be critical for
      retaining the boundary flexibility needed for accurate results
      even in the lowest system frequencies. This is because boundary
      stiffness that only participates in very high frequency modes
      with a free-free boundary condition can participate in the
      lowest frequency modes when connected to another model.

    - The free-acceleration of the Source is the complex response of
      the boundary DOF without the Load attached. Frequency domain
      envelopes of flight data, or vibration specifications derived
      from envelopes, are not accurate ways to come up with the
      free-acceleration because a Load is attached. However, since
      this is sometimes the only viable option, is it conservative?
      Conservatism becomes more and more likely with this approach as
      the number of enveloped curves increases. This is because, for a
      given Load, the coupled system response will likely exceed the
      free-acceleration response in some frequency bands. See, for
      example, the final two subplots in the example below; the
      coupled system response is higher than the free-acceleration
      response at 12.5 Hz. Such enveloping can therefore lead to a
      very conservative free-acceleration curve, which means
      conservative results.

          - As was noted, coupled system responses can exceed
            free-acceleration in some frequency bands (for example, at
            12.5 Hz in the system demonstrated below). This means
            that, in cases where a bounding coupled system
            acceleration level is used for the free-acceleration,
            application of the NT equations will likely result in
            responses that exceed the bound. To avoid excessive
            conservatism, it may be prudent in these cases to accept
            any notching down, but to not allow "notching up". That
            is; do not let :math:`A(\Omega)` exceed
            :math:`As(\Omega)`.

    - The apparent mass of a structure can be used to perform
      frequency domain base-drive analyses instead of using a seismic
      mass. Just compute the force required to meet the acceleration
      requirement via equation (2) above (:math:`\tilde{F}_b(\Omega) =
      AM_{bb}(\Omega) \cdot \ddot{X}_b(\Omega)`). Note that a simple
      matrix multiply will not work for all frequencies at the same
      time since the apparent mass is a 3D array. You could use a
      loop, or you could use the remarkable :func:`numpy.einsum`
      function: ``F = numpy.einsum('ijk,kj->ij', AM, Xdd)``

    *Notional example*::

        from pyyeti import frclim
        from pyyeti.nastran op2, n2p, op4
        import pickle

        # load free-acceleration of LV
        dct = pickle.load('ifresults_free.p')
        As = dct['As']
        freq = dct['freq']

        # load Source free-free model, with residual vectors included
        nas = op2.rdnas2cam('nas2cam')
        m1 = None
        zeta = 0.01
        k1 = nas['lambda'][0]
        k1[:nas['nrb']] = 0.
        b1 = 2*zeta*np.sqrt(k1)

        # Transformation to i/f node:
        T, dof = n2p.formdrm(nas, seup=0, sedn=0, dof=888888)

        # get S/C mass and stiffness of Load:
        mk = op4.read('mk.op4')
        kgen = mk['kgen']
        mgen = mk['mgen']
        kgen[:6, :6] = 0.
        zeta = 0.01
        bgen = np.diag(2*zeta*np.sqrt(np.diag(kgen)))

        # Norton-Thevenin force limit function:
        r = frclim.ntfl([m1, b1, k1, T], [mgen, bgen, kgen,
                        np.arange(6)], As, freq)

    .. note::
        In addition to the example shown below, this routine is
        demonstrated in the pyYeti :ref:`tutorial`:
        :doc:`/tutorials/ntfl`. There is also a link to the source
        Jupyter notebook at the top of the tutorial.

    See also
    --------
    :func:`calcAM`, :func:`sefl`, :func:`ctdfs`, :func:`stdfs`.

    Examples
    --------
    This example sets up a simple mass-spring system to demonstrate
    that the Norton-Thevenin equations can be exact.

    Steps:

      1. Setup a coupled system of a SOURCE and a LOAD
      2. Solve for interface acceleration and force from coupled system
         (frequency domain)
      3. Calculate free-acceleration from SOURCE alone and setup LOAD
      4. Use :func:`ntfl` to couple the system
      5. Plot interface acceleration and force to show :func:`ntfl` can
         be an exact coupling method
      6. Plot apparent masses
      7. Plot free-acceleration and coupled acceleration
      8. Plot normalized response ratio (can be thought of as force
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

        3. Solve for free-acceleration; SOURCE setup: [m, b, k, bdof]:

        >>> ms = np.array([[M1, 0], [0, M2/2]])
        >>> cs = np.array([[c1, -c1], [-c1, c1]])
        >>> ks = np.array([[k1, -k1], [-k1, k1]])
        >>> source = [ms, cs, ks, [[0, 1]]]
        >>> fs_source = ode.SolveUnc(ms, cs, ks, pre_eig=True)
        >>> sourcesol = fs_source.fsolve(F[:2], freq)
        >>> As = sourcesol.a[1:2]   # free-acceleration

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
        >>> _ = plt.figure('Example', clear=True, layout='constrained')
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

    .. plot::
        :context: close-figs

        6. Plot apparent masses:

        >>> _ = plt.figure('Example 2', clear=True,
        ...                layout='constrained')
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

        >>> _ = plt.figure('Example 3', clear=True,
        ...                layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.semilogy(freq, abs(As).T,
        ...                  label='Free-Acce')
        >>> _ = plt.semilogy(freq, abs(r.A).T,
        ...                  label='Coupled Acce')
        >>> _ = plt.title('Interface Acceleration')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.subplot(212)
        >>> _ = plt.semilogy(freq, abs(r.R).T)
        >>> _ = plt.title('NT Response Ratio: '
        ...               'R = Coupled Acce / Free-Acce')
        >>> _ = plt.xlabel('Frequency (Hz)')
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
        A[:, j] = Mr @ As[:, j]
        F[:, j] = Ml @ A[:, j]
        R[:, j] = np.diag(Mr)
    return SimpleNamespace(F=F, A=A, R=R, LAM=LAM, SAM=SAM, TAM=TAM, freq=freq)


def sefl(c, f, f0, n=1):
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
    n : scalar; optional
        The exponent `n` of the rolloff factor ``(f0/f)^n`` is
        included in the equations to reflect the decrease in the
        residual mass of the component with frequency.


    Returns
    -------
    nfl : scalar
        The normalized force limit:

        .. code-block:: none

            nfl = c                   f <= f0
            nfl = c / (f/f0)^n        f > f0

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
    return c / (f / f0) ** n


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
        |---------|           (Source)            (Load)

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
    k1 = w1**2 * m1
    k2 = w2**2 * m2

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
        Modal to residual mass ratio for Source; 0.0001 to 10 is
        reasonable ``m1/M1``
    mmr2 : scalar
        Modal to residual mass ratio for Load; 0.0001 to 10 is
        reasonable ``m2/M2``
    rmr : scalar
        Residual mass ratio of Source over Load; 0.0001 to 10 is
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
    k1 = w1**2 * m1
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
            k2 = w2**2 * m2

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
        Modal to residual mass ratio for Source; 0.0001 to 10 is
        reasonable ``m1/M1``
    mmr2 : scalar
        Modal to residual mass ratio for Load; 0.0001 to 10 is
        reasonable ``m2/M2``
    rmr : scalar
        Residual mass ratio of Source over Load; 0.0001 to 10 is
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
    k1 = w1**2 * m1

    def get_neg_pknfl(nw2):
        """
        Computes the negative (for minimization) of peak normalized
        force limit
        """
        w2 = nw2 * w1
        c2 = 2 * zeta2 * w2 * m2
        k2 = w2**2 * m2

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
    if not res.success:
        msg = res.message if "message" in res else "no info"
        raise RuntimeError(
            f"routine :func:`scipy.optimize.minimize_scalar` failed:\n\t'{msg}'"
        )
    nw2 = res.x
    nfl = -res.fun
    return nfl, nw2
