# -*- coding: utf-8 -*-
from .solveunc import SolveUnc


class SolveCDF(SolveUnc):
    r"""
    2nd order ODE time and frequency domain solvers for "uncoupled"
    equations of motion

    This class is for solving:

    .. math::
        M \ddot{q} + C \dot{q} + K q = P

    This class inherits from :class:`SolveUnc` and adds one additional
    time-domain solver for a special (but quite common) case: when
    damping is coupled but the mass and stiffness are diagonal. For
    all other cases, using :class:`SolveCDF` is the same as using
    :class:`SolveUnc`. In fact, all the coding for this solver is
    actually within the :class:`SolveUnc` class. One can get to this
    solver by instantiating :class:`SolveUnc` directly with the
    `cd_as_force` option set to True. So why does this class exist?
    For two reasons: one, to provide a consistent call signature with
    :class:`SolveUnc` and :class:`SolveExp2`, and two, as a convenient
    way to provide this documentation.

    When the damping is coupled but the mass and stiffness are
    diagonal, this solver (:class:`SolveCDF`) simply treats the
    off-diagonal damping terms as forces and uses the uncoupled
    equations (diagonal) solver of :class:`SolveUnc`. "CDF" in the
    name stands for coupled damping forces. The procedure is outlined
    below.

    Note that the solution is not piece-wise linear exact in this
    case. It is assumed that the off-diagonal damping terms for each
    ODE equation, in total, have a negligible effect on the derivative
    relationships for that equation and can be treated as applied
    forces. In practice, as long as the time-step is fine enough, this
    solver has been found to be fast and accurate.

    :class:`SolveCDF` can be particularly advantageous when the
    generator (:func:`SolveCDF.generator`) feature is used: in one
    test, using :class:`SolveCDF` was approximately 70 times faster
    than using :class:`SolveUnc` (which uses the complex eigensolution
    for coupled damping) and more than 200 times faster than the
    :class:`SolveExp2` solver. In that test, the accuracy was
    acceptable for :class:`SolveCDF` with 25 points per cycle (ppc) at
    the highest frequency (``ppc = 1 / (h * freq_high)``). However,
    accuracy is problem-dependent and needs to be verified before
    the solution can be trusted.

    **Procedure**

    Consider the equations of motion with the off-diagonal damping
    terms moved to the right-hand-side to be treated as an applied
    force:

    .. math::
        M \ddot{q} + C_d \dot{q} + K q = P - C_{od} \dot{q}

    Since the ODE equations are uncoupled, the pre-formulated
    uncoupled integration coefficients are used (see
    :func:`get_su_coef`) as follows:

    .. math::
        \begin{array}{lr}
          \begin{aligned}
            q_{i+1} &= F q_i + G \dot{q}_i +
               A (P_i - C_{od} \dot{q}_i) +
               B (P_{i+1} - C_{od} \dot{q}_{i+1})

            \dot{q}_{i+1} &= F_p q_i + G_p \dot{q}_i +
               A_p (P_i - C_{od} \dot{q}_i) +
               B_p (P_{i+1} - C_{od} \dot{q}_{i+1})
          \end{aligned}
          \begin{aligned}
            \qquad \qquad (1&)

            \qquad \qquad (2&)
          \end{aligned}
        \end{array}

    Equation 2 can be solved for :math:`\dot{q}_{i+1}`:

    .. math::
        \begin{aligned}
        (I + B_p C_{od}) \dot{q}_{i+1} &=
           F_p q_i + G_p \dot{q}_i +
           A_p (P_i - C_{od} \dot{q}_i) +
           B_p P_{i+1}

        \dot{q}_{i+1} &= Z (
           F_p q_i + G_p \dot{q}_i +
           A_p (P_i - C_{od} \dot{q}_i) +
           B_p P_{i+1})

        \dot{q}_{i+1} &= Z Vpart_i
        \end{aligned}

    where :math:`Z = (I + B_p C_{od})^{-1}` and :math:`Vpart_i = F_p
    q_i + G_p \dot{q}_i + A_p (P_i - C_{od} \dot{q}_i) + B_p P_{i+1}`.

    With :math:`\dot{q}_{i+1}`, :math:`q_{i+1}` can be computed from
    Equation 1 above. Using the equations as written, with :math:`Z`
    precomputed, would require two matrix-vector multiplies per loop:
    :math:`Z Vpart_i` and :math:`C_{od} \dot{q}_{i+1}`. We can get rid
    of one matrix-vector multiply per loop by precomputing the product
    of the two matrices outside the loop and reordering some
    calculations. First, precalculate :math:`\alpha`:

    .. math::
        \alpha = C_{od} Z = C_{od} (I + B_p C_{od})^{-1}

    Let the off-diagonal damping force be denoted by :math:`Q`
    (:math:`Q_i \equiv C_{od} \dot{q}_i`). Then, in the loop:

    .. math::
        \begin{aligned}
        Vpart_i &= F_p q_i + G_p \dot{q}_i +
                   A_p (P_i - Q_i) + B_p P_{i+1}

        Q_{i+1} &= \alpha Vpart_i

        q_{i+1} &= F q_i + G \dot{q}_i +
           A (P_i - Q_i) + B (P_{i+1} - Q_{i+1})

        \dot{q}_{i+1} &= Vpart_i - B_p Q_{i+1}
        \end{aligned}

    With that approach, there is only one matrix-vector multiply per
    loop: :math:`\alpha Vpart_i`.

    .. note::

        The above equations are for the non-residual-flexibility
        modes. The 'rf' modes are solved statically and any initial
        conditions are ignored for them.

    For a static solution:

        - rigid-body displacements = zeros
        - elastic displacments = inv(k[elastic]) * P[elastic]
        - velocity = zeros
        - rigid-body accelerations = inv(m[rigid]) * P[rigid]
        - elastic accelerations = zeros

    See also
    --------
    :class:`SolveUnc`, :class:`SolveExp2`.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])    # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])  # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])    # % damping
        >>> b = np.diag(2.*zeta*np.sqrt(k/m)*m)   # diag of damping
        >>> b_off = -np.diag(np.diag(b)[:-1], 1) / 100
        >>> b += b_off + b_off.T                  # coupled damping
        >>> h = 0.001                             # time step
        >>> t = np.arange(0, .3001, h)            # time vector
        >>> f = np.vstack((3*(1-np.cos(2*np.pi*2*t)),   # ffn
        ...                4.5*(np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> ts = ode.SolveCDF(m, b, k, h)
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
        >>> print('acce cmp:',
        ...       np.allclose(yl[:, :n], sol.a.T,
        ...                   atol=1e-5*abs(sol.a).max()))
        acce cmp: True
        >>> print('velo cmp:',
        ...       np.allclose(yl[:, n:2*n], sol.v.T,
        ...                   atol=1e-5*abs(sol.v).max()))
        velo cmp: True
        >>> print('disp cmp:',
        ...       np.allclose(yl[:, 2*n:], sol.d.T,
        ...                   atol=1e-5*abs(sol.d).max()))
        disp cmp: True

        Plot the four accelerations:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('Example', figsize=[8, 8], clear=True,
        ...                  layout='constrained')
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, sol.a[j], label='SolveCDF')
        ...     _ = plt.plot(tl, yl[:, j], label='scipy lsim')
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Time (s)')
        ...     if j == 0:
        ...         _ = plt.legend(loc='best')
    """

    def __init__(self, m, b, k, h=None, rb=None, rf=None, order=1, pre_eig=False):
        """
        Instantiates a :class:`SolveCDF` solver.

        This simply calls :func:`SolveUnc.__init__` with the
        `cd_as_force` option set to True; see that function for more
        information.
        """
        super().__init__(m, b, k, h, rb, rf, order, pre_eig, cd_as_force=True)

    def generator(self, nt, F0, d0=None, v0=None, static_ic=False):
        """
        Python "generator" version of :func:`SolveCDF.tsolve`;
        interactively solve (or re-solve) one step at a time.

        This routine simply calls :func:`SolveUnc.generator` and
        only exists to provide this documentation (this is explained
        in :class:`SolveCDF`). See :func:`SolveUnc.generator` for
        information on using this feature; just replace "SolveUnc"
        with "SolveCDF".
        """
        return super().generator(nt, F0, d0, v0, static_ic)

    def fsolve(self, force, freq, incrb="dva", rf_disp_only=False):
        """
        Solve frequency-domain modal equations of motion using
        uncoupled equations.

        This routine simply calls :func:`SolveUnc.fsolve` and
        only exists to provide this documentation (this is explained
        in :class:`SolveCDF`). See :func:`SolveUnc.fsolve` for
        information on using this feature; just replace "SolveUnc"
        with "SolveCDF".

        .. note::
            Off-diagonal damping as forces is not implemented for the
            frequency domain. In that case (which is the only case
            where :class:`SolveCDF` is different from
            :class:`SolveUnc`), :func:`SolveUnc.fsolve` will raise
            a ``NotImplementedError`` exception.
        """
        return super().fsolve(force, freq, incrb, rf_disp_only)
