# -*- coding: utf-8 -*-
import scipy.linalg as la
import numpy as np
from ._base_ode_class import _BaseODE
from ._utilities import _process_incrb


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


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

    Unlike :class:`SolveUnc`, this routine makes no special provisions
    for rigid-body modes when computing the response; therefore,
    including 0.0 in `freq` can cause a divide-by-zero. It is
    therefore recommended to ensure that all values in `freq` > 0.0,
    at least when rigid-body modes are present. After the solution is
    computed, for equations that are in modal space, the rigid-body
    part of the response may be zeroed out according to the `incrb`
    parameter in :func:`fsolve`.

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
        >>> fig = plt.figure('Example', figsize=[8, 8])
        >>> fig.clf()
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(freq, abs(sol.a[j]))
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Frequency (Hz)')
        >>> fig.tight_layout()
    """

    def __init__(self, m, b, k, rb=None, rf=None):
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
            An option for equations in modal space. Index or bool
            partition vector for rigid-body modes. Set to [] to
            specify no rigid-body modes. If None, the rigid-body modes
            will be automatically detected by this logic for uncoupled
            systems::

               rb = np.nonzero(abs(k).max(0) < 0.005)[0]

            And by this logic for coupled systems::

               rb = ((abs(k).max(axis=0) < 0.005) &
                     (abs(k).max(axis=1) < 0.005) &
                     (abs(b).max(axis=0) < 0.005) &
                     (abs(b).max(axis=1) < 0.005)).nonzero()[0]

        rf : 1d array or None; optional
            Index or bool partition vector for residual-flexibility
            modes; these will be solved statically. As for the `rb`
            option, the `rf` option only applies to modal space
            equations.

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
         kdof      index vector or slice for the non-rf modes
         n         number of total DOF
         rfsize    number of rf modes
         nonrfsz   number of non-rf modes
         ksize     number of non-rf modes
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
        self._common_precalcs(m, b, k, h=None, rb=rb, rf=rf)
        self._mk_slices()  # dorbel=False)

    def fsolve(self, force, freq, incrb="dva", rf_disp_only=False):
        """
        Solve equations of motion in frequency domain.

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x freq
        freq : 1d ndarray
            Frequency vector in Hz; solution will be computed at all
            frequencies in `freq`
        incrb : string; optional
            Specifies how to handle rigid-body responses: can include
            "a", "v", and/or "d" to specify that the rigid-body
            response should not be zeroed out for acceleration,
            velocity, and/or displacement. For example, ``incrb=""``
            means that no rigid-body responses are included, and
            ``incrb="va"`` (or ``incrb="av"``) means that the
            rigid-body response will be included for acceleration and
            velocity but not displacement. Letters can be included in
            any order.

            For backward compatibility to versions before 0.99.9,
            `incrb` can also be an integer 0, 1, or 2. This integer
            format is deprecated and will be removed in a future
            version:

            ======  =================================================
            incrb   description
            ======  =================================================
               0    same as "" (no rigid-body is included)
               1    same as "av"
               2    same as "dva" (all rigid-body responses included)
            ======  =================================================

        rf_disp_only : bool; optional
            This option specifies how to handle the velocity and
            acceleration terms for residual-flexibility modes. If
            True, they are set to zero. If False, they are computed
            from the normal frequency-domain relationships::

                velocity = i * omega * displacement
                acceleration = -omega ** 2 * displacement

        Returns
        -------
        A SimpleNamespace with the members:

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
        See :class:`FreqDirect` for more discussion on how rigid-body
        response is handled.
        """
        incrb = _process_incrb(incrb)  # deprecated in v0.99.9
        force = np.atleast_2d(force)
        freq = np.atleast_1d(freq)
        d, v, a, force = self._init_dva(
            force,
            d0=None,
            v0=None,
            static_ic=False,
            istime=False,
            freq=freq,
            rf_disp_only=rf_disp_only,
        )

        if self.ksize == 0:
            return self._solution_freq(d, v, a, freq)

        self._force_freq_compat_chk(force, freq)
        m, b, k = self.m, self.b, self.k
        kdof = self.kdof
        force = force[kdof]
        if self.unc:
            # equations are uncoupled, solve everything in one step:
            Omega = 2 * np.pi * freq[None, :]
            if m is None:
                H = ((1j * b)[:, None] @ Omega + k[:, None]) - Omega ** 2
            else:
                H = (1j * b)[:, None] @ Omega + k[:, None] - m[:, None] @ Omega ** 2
            d[kdof] = force / H
        else:
            # equations are coupled, use a loop:
            Omega = 2 * np.pi * freq
            if m is None:
                m = np.eye(self.ksize)
            for i, O in enumerate(Omega):
                Hi = 1j * b * O + k - m * O ** 2
                d[kdof, i] = la.solve(Hi, force[:, i])
        a[kdof] = -(Omega ** 2) * d[kdof]
        v[kdof] = 1j * Omega * d[kdof]

        if "d" not in incrb:
            d[self.rb] = 0
        if "v" not in incrb:
            v[self.rb] = 0
        if "a" not in incrb:
            a[self.rb] = 0

        return self._solution_freq(d, v, a, freq)
