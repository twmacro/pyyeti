# -*- coding: utf-8 -*-
import scipy.linalg as la
import numpy as np
from pyyeti import expmint
from ._base_ode_class import _BaseODE


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


class SolveExp2(_BaseODE):
    r"""
    2nd order ODE time domain solver based on the matrix exponential.

    This class is for solving matrix equations of motion:

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

    The general solution is:

    .. math::
        y = e^{A t} \left (
        y(0) + \int_0^t {e^{-A \tau} w d\tau}
        \right )

    By only requiring the solution at every time step and assuming a
    constant step size of `h`:

    .. math::
        y_{n+1} = e^{A h} \left (
        y_{n} + \int_0^h {e^{-A \tau} w(t_n+\tau) d\tau}
        \right )

    By assuming :math:`w(t_n+\tau)` is piece-wise linear (if `order`
    is 1) or piece-wise constant (if `order` is 0) for each step, an
    exact, closed form solution can be calculated. The function
    :func:`pyyeti.expmint.getEPQ` computes the matrix exponential
    :math:`E = e^{A h}`, and solves the integral(s) needed to
    compute `P` and `Q` so that a solution can be computed by:

    .. math::
        y_{n+1} = E y_{n} + P w_{n} + Q w_{n+1}

    Unlike for the uncoupled solver :class:`SolveUnc`, this solver
    doesn't need or use the `rb` input unless static initial
    conditions are requested when solving equations.

    Note that :class:`SolveUnc` is also an exact solver assuming
    piece-wise linear or piece-wise constant forces. :class:`SolveUnc`
    is often faster than :class:`SolveExp2` since it uncouples the
    equations and therefore doesn't need to work with matrices in the
    inner loop. However, it is recommended to experiment with both
    solvers for any particular application.

    .. note::

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
        >>> h = 0.001                             # time step
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
        >>> fig = plt.figure('Example', figsize=[8, 8], clear=True,
        ...                  layout='constrained')
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
    """

    def __init__(self, m, b, k, h, rb=None, rf=None, order=1, pre_eig=False):
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
            Index or bool partition vector for rigid-body modes. Set
            to [] to specify no rigid-body modes. If None, the
            rigid-body modes will be automatically detected by this
            logic for uncoupled systems::

               rb = np.nonzero(abs(k).max(0) < 0.005)[0]

            And by this logic for coupled systems::

               rb = ((abs(k).max(axis=0) < 0.005) &
                     (abs(k).max(axis=1) < 0.005) &
                     (abs(b).max(axis=0) < 0.005) &
                     (abs(b).max(axis=1) < 0.005)).nonzero()[0]

            .. note::
                `rb` applies only to modal-space equations. Use
                `pre_eig` if necessary to convert to modal-space. This
                means that if `rb` is an index vector, it specifies
                the rigid-body modes *after* the `pre_eig`
                operation. See also `pre_eig`.

            .. note::
                Unlike for the :class:`SolveUnc` solver, `rb` for this
                solver is only used if using static initial conditions
                in :func:`SolveExp2.tsolve`.

        rf : 1d array or None; optional
            Index or bool partition vector for residual-flexibility
            modes; these will be solved statically. As for the `rb`
            option, the `rf` option only applies to modal space
            equations (possibly after the `pre_eig` operation).
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
        m_orig     original mass
        b_orig     original damping
        k_orig     original stiffness
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
                   :func:`pyyeti.expmint.getEPQ`
        E_vd       another partition of "E"
        E_dv       another partition of "E"
        E_dd       another partition of "E"
        P, Q       output of :func:`pyyeti.expmint.getEPQ`; they are
                   matrices used to solve the ODE
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
        self._mk_slices()  # dorbel=False)

    def tsolve(self, force, d0=None, v0=None, static_ic=False):
        """
        Solve time-domain 2nd order ODE equations

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x time
        d0 : 1d ndarray; optional
            Displacement initial conditions; if None, zero ic's are
            used unless `static_ic` is True.
        v0 : 1d ndarray; optional
            Velocity initial conditions; if None, zero ic's are used.
        static_ic : bool; optional
            If True and `d0` is None, then `d0` is calculated such
            that static (steady-state) initial conditions are used. Be
            sure to use the "pre_eig" option to put equations in modal
            space if necessary: for static initial conditions, the
            rigid-body part is initialized to 0.0 and the elastic part
            is computed such that the system is in static equilibrium
            (from the elastic part of ``K x = F``).

            .. note::
                `static_ic` is quietly ignored if `d0` is not None.

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
        d, v, a, force = self._init_dva(force, d0, v0, static_ic)
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
                        imf = la.lu_solve(self.invm, force[kdof], check_finite=False)
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
                for i in range(nt - 1):
                    d0 = D[:, i]
                    v0 = V[:, i]
                    D[:, i + 1] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:, i]
                    V[:, i + 1] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize, i]
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
        d, v : 2d ndarrays
            The displacement and velocity arrays. Only the first
            column of `d` and `v` are set; other values are all zero.

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

               There is a second usage of :func:`gen.send`: if the
               index is negative, the force is treated as an addon to
               forces already included for the i'th step. This is for
               efficiency and only does the necessary calculations.
               This feature was originally written for running
               Henkel-Mar simulations, where interface forces are
               computed after computing the solution with all the
               other forces applied. There may be other similar
               situations. To demonstrate this usage::

                   for i in range(1, len(time)):
                       # Do whatever to get i'th force
                       # - note: d[:, :i] and v[:, :i] are available
                       gen.send((i, fi))
                       # Do more calculations to compute an addon
                       # force. Then, update the i'th solution:
                       gen.send((-1, fi_addon))

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
        >>> h = 0.001                             # time step
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
                "generator not yet implemented for the case"
                " when different types of equations are interspersed"
                " (eg, a residual-flexibility DOF in the middle of"
                " the elastic DOFs)"
            )
        d, v, a, force = self._init_dva_part(nt, F0, d0, v0, static_ic)
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
                ikrf = la.lu_solve(ikrf, np.eye(rfsize), check_finite=False)
            drf = d[rf]

        ksize = self.ksize
        if not ksize:
            # only rf modes
            if unc:
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        d[:, i] += ikrf * F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        d[:, i] = ikrf * F1[rf]
            else:
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        d[:, i] += ikrf @ F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        d[:, i] = ikrf @ F1[rf]

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
                P = la.lu_solve(self.invm, P.T, trans=1, check_finite=False).T
                if order == 1:
                    Q = la.lu_solve(self.invm, Q.T, trans=1, check_finite=False).T
        E_dd = self.E_dd
        E_dv = self.E_dv
        E_vd = self.E_vd
        E_vv = self.E_vv
        if rfsize:
            # both rf and non-rf modes present
            D = d[kdof]
            V = v[kdof]
            drf = d[rf]
            while True:
                j, F1 = yield
                if j < 0:
                    # add new force to previous solution
                    Force[:, i] += F1
                    if self.order == 1:
                        PQF = Q @ F1[kdof]
                        D[:, i] += PQF[ksize:]
                        V[:, i] += PQF[:ksize]
                    if unc:
                        drf[:, i] += ikrf * F1[rf]
                    else:
                        drf[:, i] += ikrf @ F1[rf]
                else:
                    i = j
                    Force[:, i] = F1
                    F0 = Force[:, i - 1]
                    if self.order == 1:
                        PQF = P @ F0[kdof] + Q @ F1[kdof]
                    else:
                        PQF = P @ F0[kdof]
                    d0 = D[:, i - 1]
                    v0 = V[:, i - 1]
                    D[:, i] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:]
                    V[:, i] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize]
                    if unc:
                        drf[:, i] = ikrf * F1[rf]
                    else:
                        drf[:, i] = ikrf @ F1[rf]
        else:
            # only non-rf modes present
            while True:
                j, F1 = yield
                if j < 0:
                    # add new force to previous solution
                    Force[:, i] += F1
                    if self.order == 1:
                        PQF = Q @ F1
                        d[:, i] += PQF[ksize:]
                        v[:, i] += PQF[:ksize]
                else:
                    i = j
                    Force[:, i] = F1
                    F0 = Force[:, i - 1]
                    if self.order == 1:
                        PQF = P @ F0 + Q @ F1
                    else:
                        PQF = P @ F0
                    d0 = d[:, i - 1]
                    v0 = v[:, i - 1]
                    d[:, i] = E_dd @ d0 + E_dv @ v0 + PQF[ksize:]
                    v[:, i] = E_vd @ d0 + E_vv @ v0 + PQF[:ksize]

    def get_f2x(self, phi, velo=False):
        """
        Get force-to-displacement or force-to-velocity transform

        Parameters
        ----------
        phi : 2d ndarray
            Transform from ODE coordinates to physical DOF
        velo : bool; optional
            If True, get force to velocity transform instead

        Returns
        -------
        flex : 2d ndarray
            Force to displacement (or velocity) transform

        Notes
        -----
        This routine was written to support Henkel-Mar simulations;
        see [#hm]_. The equations of motion for two separate bodies
        are solved simultaneously while enforcing joint
        compatibility. This is handy for allowing the two bodies to
        separate from each other. The `flex` matrix is part of the
        matrix in the upper right quadrant of equation 27 in ref
        [#hm]_; the remaining part comes from the other body.

        The interface DOF are those DOF that interface with the other
        body. The force is the interface force and the displacement
        (or velocity) is of the interface DOF.

        The reference does not discuss enforcing joint velocity
        compatibility. This routine however lets you choose between
        the two since the velocity method is fundamentally more stable
        than the displacement method.

        Let (see also :func:`__init__`)::

            phik = phi[:, kdof]
            phirf = phi[:, rf]

        If `velo` is False::

            flex = phik @ Q[n:] @ phik.T + phirf @ ikrf @ phirf

        If `velo` is True::

            flex = phik @ Q[:n] @ phik.T

        .. note::

            A zeros matrix is returned if `order` is 0.

        Raises
        ------
        NotImplementedError
            When `systype` is not float.

        References
        ----------
        .. [#hm] E. E. Henkel, and R. Mar "Improved Method for
            Calculating Booster to Launch Pad Interface Transient
            Forces", Journal of Spacecraft and Rockets, Dated Nov-Dec,
            1988, pp 433-438
        """
        if self.systype is not float:
            raise NotImplementedError(
                ":func:`get_f2x` can only handle real equations of motion"
            )

        flex = 0.0
        unc = self.unc
        if self.order == 1:
            if self.ksize:
                # non-rf equations:
                Q = self.Q
                kdof = self.kdof
                phik = phi[:, kdof]
                if self.m is not None:
                    if unc:
                        invm = self.invm.ravel()
                        Q = Q * invm
                    else:
                        Q = la.lu_solve(self.invm, Q.T, trans=1, check_finite=False).T
                n = self.nonrfsz
                if velo:
                    flex = phik @ Q[:n] @ phik.T
                else:
                    flex = phik @ Q[n:] @ phik.T
            flex = self._add_rf_flex(flex, phi, velo, unc)
        return self._flex(flex, phi)
