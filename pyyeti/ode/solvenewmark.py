# -*- coding: utf-8 -*-
import scipy.linalg as la
import numpy as np
from ._base_ode_class import _BaseODE


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


class SolveNewmark(_BaseODE):
    r"""
    2nd order ODE time domain "Newmark-Beta" solver

    This class is for solving:

    .. math::
        M \ddot{u} + B \dot{u} + K u = F

    This routine uses a fixed time step Newmark-Beta method based on
    the Nastran Theoretical Manual, section 11.4 [#newmark]_. The
    algorithm is unconditionally stable but the size of the time step
    is still a concern for accurate results. According to Bathe
    [#bathe]_, a time step that ensures accurate results for
    algorithms like this one is :math:`1/(80 f_{high})`, where
    :math:`f_{high}` is the highest frequency (Hz) in your forcing
    functions.

    In general, the mass, damping and stiffness can be fully populated
    (coupled).

    Unlike :class:`SolveUnc` and :class:`SolveExp2`, this solver can
    handle a singular mass matrix. For most (all?) other cases, the
    other two solvers are preferred since they are exact assuming
    piece-wise linear forces (if `order` is 1) or piece-wise constant
    forces (if `order` is 0). :class:`SolveUnc` is also very likely
    significantly faster for most problems.

    The equations for the non-residual equations are:

    .. math::
        A u_{n+2} = \frac{1}{3}
        \left ( F_{n+2} + F_{n+1} + F_{n} \right ) + N_{n+1} +
        A_1 u_{n+1} + A_0 u_{n}

    where:

    .. math::
        \begin{aligned}
        A &= \left [ \frac{M}{h^2} + \frac{B}{2 h} +
                     \frac{K}{3} \right ] \\
        A_1 &= \left [ \frac{2 M}{h^2} - \frac{K}{3} \right ] \\
        A_0 &= \left [ \frac{-M}{h^2} + \frac{B}{2 h} -
                       \frac{K}{3} \right ]
        \end{aligned}

    :math:`N_{n+1}` is a nonlinear force term which is optional; see
    :func:`def_nonlin`.

    To get the algorithm started, :math:`u_{-1}` and :math:`F_{-1}`
    are needed. They are computed from:

    .. math::
        \begin{aligned}
        u_{-1} &= u_0 - \dot{u}_0 h \\

        F_{-1} &= K u_{-1} + B \dot{u}_0
        \end{aligned}

    Also, :math:`F_0` is replaced by:

    .. math::
        F_{0} &= K u_{0} + B \dot{u}_0

    According to [#newmark]_, that is done to avoid ringing of
    massless degrees of freedom that are subjected to step loads. A
    side-effect of this is that :math:`\ddot{u}_0` would be zero from
    the equation of motion. However, since the acceleration is
    computed from a central finite difference formula (see below), the
    resulting initial acceleration will not be zero in general. The
    acceleration values should be approximately correct by the third
    time step.

    After the displacements have been calculated from the above
    equations, the velocities and accelerations are computed from:

    .. math::
        \begin{aligned}
        \dot{u}_n &= \frac{1}{2 h} \left (
            u_{n+1} - u_{n-1} \right ) \\

        \ddot{u}_n &= \frac{1}{h^2} \left (
            u_{n+1} - 2 u_n + u_{n-1} \right )
        \end{aligned}

    To compute the velocities and accelerations for the final time
    step, one extra integration step is performed by linearly
    extrapolating the force.

    .. note::

        The above equations are for the non-residual-flexibility
        modes. The 'rf' modes are solved statically and any initial
        conditions are ignored for them.

    References
    ----------
    .. [#newmark] 'The NASTRAN Theoretical Manual', Section 11.4,
           NASA-SP-221(06), Jan 01, 1981.
           https://ntrs.nasa.gov/search.jsp?R=19840010609

    .. [#bathe] Klaus-Jurgen Bathe, 'Finite Element Procedures',
           Second Edition, Watertown, MA: Klaus-Jürgen Bathe, 2014.
           http://web.mit.edu/kjb/www/Books/FEP_2nd_Edition_5th_Release.pdf

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
        >>> b = 2.*zeta*np.sqrt(k/m)*m            # diag damping
        >>> h = 0.0005                            # time step
        >>> t = np.arange(0, 0.2, h)              # time vector
        >>> c = 2*np.pi
        >>> f = np.vstack((3*(1-np.cos(c*2*t)),   # ffn
        ...                4.5*(1 - np.cos(np.sqrt(k[1]/m[1])*t)),
        ...                4.5*(1 - np.cos(np.sqrt(k[2]/m[2])*t)),
        ...                4.5*(1 - np.cos(np.sqrt(k[3]/m[3])*t))))
        >>> f *= 1.e4
        >>> t2 = 2 / (np.sqrt(k[1] / m[1]) / 2 / np.pi)
        >>> f[1:, t > t2] = 0.0
        >>> nb = ode.SolveNewmark(m, b, k, h)
        >>> sol = nb.tsolve(f)

        Solve with :class:`SolveUnc` for comparison:

        >>> su = ode.SolveUnc(m, b, k, h)
        >>> solu = su.tsolve(f)

        Check accuracy:

        >>> for r in 'dva':
        ...     unc = getattr(sol, r)
        ...     new = getattr(solu, r)
        ...     atol = 0.01 * abs(unc).max()
        ...     print(r + ' comp:',
        ...           np.allclose(new, unc, atol=atol, rtol=0.001))
        d comp: True
        v comp: True
        a comp: True

        Plot the four accelerations:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('Example', figsize=[8, 8])
        >>> fig.clf()
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(t, solu.a[j], label='SolveUnc')
        ...     _ = plt.plot(t, sol.a[j], '--', label='SolveNewmark')
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel('Acceleration')
        ...     _ = plt.xlabel('Time (s)')
        ...     if j == 0:
        ...         _ = plt.legend(loc='best')
        >>> fig.tight_layout()
    """

    def __init__(self, m, b, k, h=None, rf=None):
        r"""
        Instantiates a :class:`SolveNewmark` solver.

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
        rf : 1d array or None; optional
            Index or bool partition vector for residual-flexibility
            modes; these will be solved statically. The `rf` option
            only applies to modal space equations.

        Notes
        -----
        The instance is populated with some or all of the following
        members.

        ============  ================================================
        Member        Description
        ============  ================================================
        m             mass for the non-rf DOF (or None for identity)
        b             damping for the non-rf DOF
        k             stiffness for the non-rf DOF
        h             time step
        rb            np.array([])
        el            index vector or slice for the non-rf DOF
        rf            index vector or slice for the rf DOF
        nonrf         index vector or slice for the non-rf DOF
        kdof          index vector or slice for the non-rf DOF
        n             number of total DOF
        rbsize        0
        elsize        number of non-rf DOF
        rfsize        number of rf DOF
        nonrfsz       number of non-rf DOF
        ksize         number of non-rf DOF
        krf           stiffness for the rf DOF
        ikrf          inverse of stiffness for the rf DOF
        unc           True if there are no off-diagonal terms in any
                      matrix; False otherwise
        systype       float or complex; determined by `m` `b` `k`
        Ad            decomposed version of matrix :math:`A` (see
                      :class:`SolveNewmark`)
        A0            decomposed version of matrix :math:`A_0`
        A1            decomposed version of matrix :math:`A_1`
        nonlin_terms  number of nonlinear force terms defined by
                      :func:`def_nonlin` (initially set to 0)
        ============  ================================================
        """
        self._common_precalcs(m, b, k, h, rb=[], rf=rf)
        self._newmark_precalcs()
        self._mk_slices()  # dorbel=False)

    def tsolve(self, force, d0=None, v0=None):
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
        z : dictionary of 2d ndarrays; optional
            Only present if there are nonlinear force terms. The
            dictionary keys correspond to the keys used to define the
            nonlinear force terms in :func:`def_nonlin`. The values
            are the output of the functions that defined these force
            terms (see :func:`def_nonlin`).
        """
        force = np.atleast_2d(force)
        d, v, a, F = self._init_dva(force, d0, v0)
        if self.ksize:
            D = d[self.kdof]
            V = v[self.kdof]
            A = a[self.kdof]
            nt = D.shape[1]
            h = self.h
            h2 = 2 * h
            sqh = h * h
            A1 = self.A1
            A0 = self.A0
            # For the last velocity and acceleration, extrapolate
            # force to run 1 more step (so we can calculate the final
            # v and a): Fe = 2*F[:, -1] - F[:, -2] ... when added to
            # other 2 force terms (F[:, -1] + F[:, -2]), we get 3 *
            # F[:, -1].
            if self.nonlin_terms == 0:
                if self.unc:
                    for j in range(2, nt):
                        D[:, j] = (
                            F[:, j]
                            + F[:, j - 1]
                            + F[:, j - 2]
                            + A1 * D[:, j - 1]
                            + A0 * D[:, j - 2]
                        )
                    De = 3 * F[:, -1] + A1 * D[:, -1] + A0 * D[:, -2]
                else:
                    for j in range(2, nt):
                        D[:, j] = (
                            F[:, j]
                            + F[:, j - 1]
                            + F[:, j - 2]
                            + A1 @ D[:, j - 1]
                            + A0 @ D[:, j - 2]
                        )
                    De = 3 * F[:, -1] + A1 @ D[:, -1] + A0 @ D[:, -2]
            else:

                def _get_nonlin(j):
                    N = 0.0
                    for key, (func, T, args) in self.nl_dct.items():
                        z = func(D, j, h, **args)
                        self.z[key][:, j] = z
                        N += T @ z
                    return N

                if self.unc:
                    for j in range(2, nt):
                        D[:, j] = (
                            F[:, j]
                            + F[:, j - 1]
                            + F[:, j - 2]
                            + _get_nonlin(j - 1)
                            + A1 * D[:, j - 1]
                            + A0 * D[:, j - 2]
                        )
                    De = (
                        3 * F[:, -1]
                        + _get_nonlin(nt - 1)
                        + A1 * D[:, -1]
                        + A0 * D[:, -2]
                    )
                else:
                    for j in range(2, nt):
                        D[:, j] = (
                            F[:, j]
                            + F[:, j - 1]
                            + F[:, j - 2]
                            + _get_nonlin(j - 1)
                            + A1 @ D[:, j - 1]
                            + A0 @ D[:, j - 2]
                        )
                    De = (
                        3 * F[:, -1]
                        + _get_nonlin(nt - 1)
                        + A1 @ D[:, -1]
                        + A0 @ D[:, -2]
                    )

            # calculate velocity and acceleration
            V[:, 1:-1] = (D[:, 2:] - D[:, :-2]) / h2
            V[:, -1] = (De - D[:, -2]) / h2
            A[:, 1:-1] = (D[:, 2:] - 2 * D[:, 1:-1] + D[:, :-2]) / sqh
            A[:, -1] = (De - 2 * D[:, -1] + D[:, -2]) / sqh

            if not self.slices:
                d[self.kdof] = D
                v[self.kdof] = V
                a[self.kdof] = A

        sol = self._solution(d, v, a)
        if self.nonlin_terms:
            sol.z = self.z
        return sol

    def def_nonlin(self, dct):
        r"""
        Define nonlinear force terms

        Parameters
        ----------
        dct : dictionary

            Dictionary where each entry defines a nonlinear force
            term. You can define any number of terms. Each value is a
            sequence of 2 or 3 items: a function (or other callable),
            a 2d ndarray, and optionally a dictionary of arguments for
            the function. The dictionary keys are arbitrary but will
            be used as keys in the `z` return value in the output of
            :func:`tsolve`. The form of the dictionary is::

               {
                   key_0: (func_0, T_0 [, optargs_0]),
                   key_1: (func_1, T_1 [, optargs_1]),
                   ...
               }

            =========  ===============================================
            Item       Description
            =========  ===============================================
            func_i     Each function must accept at least 3 arguments:
                       the current solution displacement matrix (`d`),
                       the current step index (`j`), and the time step
                       (`h`). It can accept other arguments which are
                       specified via `optargs`::

                           def func_i(d, j, h [, ...]):

                       The function must return a 1d ndarray. It will
                       be multiplied by the corresponding transform
                       `T` and added to the other force terms (this is
                       the :math:`N_{n+1}` term shown in
                       :class:`SolveNewmark`). See note below if
                       velocity dependent terms are needed.

            T_i        Each `T_i` matrix transforms the corresponding
                       function output to a force. If your function
                       outputs the final force already, just use
                       identity for this transform. The reason this
                       input is here, instead of relying on the
                       user-supplied function `func_i` to apply the
                       transform internally, is for efficiency: the
                       relatively expensive operation of:
                       :math:`A^{-1} T_i` is done now, outside the
                       integration loop. The matrix :math:`A` is
                       defined in :class:`SolveNewmark`.

            optargs_i  If included, each `optargs_i` is a dictionary
                       of arbitrary arguments for `func_i`.
            =========  ===============================================

        Notes
        -----
        The the `j`'th nonlinear force term is computed by::

            N_j = (T_0 @ func_0(d, j, h, **optargs_0) +
                   T_1 @ func_1(d, j, h, **optargs_1) +
                   ...)

        That term is used to compute the ``j+1``'th displacement; see
        equation in :class:`SolveNewmark`. That is appropriate because
        of the nature of the central finite difference formulae used
        in the Newmark-Beta formation.

        The solution namespace returned by :func:`tsolve` will contain
        the outputs of each user defined function in the dictionary
        `z`.

        .. note::

            Nastran estimates velocity by: :math:`v_j = (d_j -
            d_{j-1})/h`. In your function, you can use something like:
            ``vj = (d[:, j] - d[:, j-1])/h``. That will work even for
            the first call, when `j` is 0, because "-1" displacements
            are stored in the last column of `d` for just this
            purpose.

        .. note::

            The nonlinear forces are computed in the integration loop
            for the non-residual flexibility equations only.
            Therefore, it is recommended to not use the `rf` option
            with nonlinear force terms.

        Examples
        --------
        Model a two-mass system with one linear spring and one
        nonlinear spring. The nonlinear spring is only active when
        compressed. There is a gap of 0.01 units before the spring
        starts being compressed.

        Model::

              |--> x1        |--> x2
            |----|    50   |----|
            | 10 |--\/\/\--| 12 |   F(t)
            |    |         |    | =====>
            |----| |-/\/-| |----|
                 K_nonlinear

            F(t) = 5000 * np.cos(2 * np.pi * t + 270 / 180 * np.pi)

        The nonlinear spring force is linearly interpolated according
        to the "lookup" table below. Linear extrapolation is used for
        displacements out of range of the table.

        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> from scipy.interpolate import interp1d
            >>> import matplotlib.pyplot as plt
            >>> from pyyeti import ode
            >>>
            >>> # mass and stiffness:
            >>> m = np.diag([10., 12.])
            >>> k = np.array([[50., -50.],
            ...               [-50.,  50.]])
            >>> c = 0. * k  # no damping
            >>>
            >>> # define time steps and force:
            >>> h = 0.005
            >>> t = np.arange(0, 4 + h / 2, h)
            >>> f = np.zeros((2, t.size))
            >>> f[1] = 5000 * np.cos(2 * np.pi * t + 3 / 2 * np.pi)
            >>>
            >>> # define interpolation table for force (the lookup
            >>> # value is x1 - x2):
            >>> lookup = np.array([[-10,     0.],
            ...                    [0.01,    0.],
            ...                    [5,     200.],
            ...                    [6,    1000.],
            ...                    [10,   1500.]])
            >>>
            >>> # force transforming lookup value to forces on the
            >>> # masses:
            >>> Tfrc = np.array([[-1.], [1.]])
            >>>
            >>> # turn interpolation table into a function for speed
            >>> interp_func = interp1d(*lookup.T,
            ...                        fill_value='extrapolate')
            >>>
            >>> # function needed for ode.SolveNewmark.def_nonlin:
            >>> def nonlin(d, j, h, interp_func):
            ...     return interp_func(d[[0], j] - d[[1], j])
            >>>
            >>> # Solve:
            >>> ts = ode.SolveNewmark(m, c, k, h)
            >>> ts.def_nonlin(
            ...     {'kcomp': (nonlin, Tfrc,
            ...               dict(interp_func=interp_func))})
            >>> sol = ts.tsolve(f)
            >>>
            >>> # for comparison, run in SolveExp2 using the generator
            >>> # feature:
            >>> ts2 = ode.SolveExp2(m, c, k, h)
            >>> gen, d, v = ts2.generator(len(t), f[:, 0])
            >>>
            >>> for i in range(1, len(t)):
            ...     if i == 1:
            ...         dx = d[0, i - 1] - d[1, i - 1]
            ...     else:
            ...         # for improved convergence, use linear
            ...         # interpolation to estimate displacements at
            ...         # current time:
            ...         dx = (2 * d[0, i - 1] - d[0, i - 2] +
            ...               d[1, i - 2] - 2 * d[1, i - 1])
            ...     f_nl = (Tfrc @ interp_func([dx]))
            ...     gen.send((i, f[:, i] + f_nl))
            >>>
            >>> sol2 = ts2.finalize()
            >>>
            >>> # plot results:
            >>> _ = plt.figure('Example', figsize=(8, 8))
            >>> plt.clf()
            >>>
            >>> _ = plt.subplot(311)
            >>> _ = plt.plot(t, sol.d.T)
            >>> _ = plt.plot(t, d.T, '--')
            >>> _ = plt.title('x1 and x2 displacments')
            >>> _ = plt.ylabel('Displacement')
            >>> _ = plt.legend(('SolveNewmark x1',
            ...                 'SolveNewmark x2',
            ...                 'SolveExp2 x1',
            ...                 'SolveExp2 x2'))
            >>>
            >>> _ = plt.subplot(312)
            >>> _ = plt.plot(t, sol.d[0] - sol.d[1],
            ...              label='SolveNewmark')
            >>> _ = plt.plot(t, d[0] - d[1], '--', label='SolveExp2')
            >>> _ = plt.title('Relative displacement: x1 - x2')
            >>> _ = plt.ylabel('Displacement')
            >>> _ = plt.legend()
            >>>
            >>> _ = plt.subplot(313)
            >>> _ = plt.plot(t, sol.z['kcomp'][0],
            ...              label='SolveNewmark')
            >>> _ = plt.plot(t, interp_func(d[0, :] - d[1, :]), '--',
            ...              label='SolveExp2')
            >>> _ = plt.title('Force in nonlinear spring')
            >>> _ = plt.xlabel('Time (s)')
            >>> _ = plt.ylabel('Force')
            >>> _ = plt.legend()
            >>> _ = plt.tight_layout()
        """
        # apply inv(A) to the transforms while making new dict:
        nl_dct = {}
        for k, v in dct.items():
            if len(v) == 2:
                args = {}
            else:
                args = v[2]

            if self.unc:
                T = v[1] / self.Ad[:, None]
            else:
                T = la.lu_solve(self.Ad, v[1])

            nl_dct[k] = (v[0], T, args)

        # if here, everything must be okay:
        self.nonlin_terms = len(nl_dct)
        self.nl_dct = nl_dct

    def _newmark_precalcs(self):
        # setup matrices for newmark - beta solution beta = 1/3
        # A u_(n + 2) = 1 / 3 * (F_(n + 2) + F_(n + 1) + F_(n)) +
        #               NonLin_(n + 1) + A1 u_(n + 1) + A0 u_(n)
        #
        # where:
        #  A = 1 / h ^ 2 M + 1 / (2 h) B + 1 / 3 K
        #  A1 = 2 / h ^ 2 M - 1 / 3 K
        #  A0 = -1 / h ^ 2 M + 1 / (2 h) B - 1 / 3 K
        self.pc = True  # to make _alloc_dva happy
        self.nonlin_terms = 0
        if self.ksize == 0:
            return
        h = self.h
        sqh = h * h
        h2 = 2 * h

        if self.m is None:
            if self.unc:
                # have diagonals:
                mterm = 1.0 / sqh
            else:
                # have matrices:
                mterm = np.diag(np.ones(self.ksize) / sqh)
        else:
            mterm = self.m / sqh

        A = mterm + self.b / h2 + self.k / 3
        A1 = 2 * mterm - self.k / 3
        A0 = self.b / h2 - self.k / 3 - mterm

        if self.unc:
            # have diagonals:
            self.Ad = A

            # define new A1, A0:
            self.A0 = A0 / A
            self.A1 = A1 / A
        else:
            # have square matrices:
            self.Ad = la.lu_factor(A, overwrite_a=True)

            # define new A1, A0:
            self.A0 = la.lu_solve(self.Ad, A0, overwrite_b=True)
            self.A1 = la.lu_solve(self.Ad, A1, overwrite_b=True)

    def _init_dva(self, force, d0, v0):
        """
        Newmark Beta version of _init_dva
        """
        if force.shape[0] != self.n:
            raise ValueError(
                f"Force matrix has {force.shape[0]} rows; {self.n} rows are expected"
            )

        d0, v0 = self._set_initial_cond(d0, v0)
        nt = force.shape[1]
        d, v, a = self._alloc_dva(nt, True)

        # solve resflex part:
        if self.rfsize:
            if self.unc:
                d[self.rf] = self.ikrf * force[self.rf]
            else:
                d[self.rf] = la.lu_solve(self.ikrf, force[self.rf])
            force = force[self.nonrf]

        if self.ksize == 0:
            return d, v, a, force

        self._init_dv(d, v, d0, v0, F0=None, static_ic=False)
        d0 = np.zeros(self.ksize) if d0 is None else d0[self.nonrf]
        v0 = np.zeros(self.ksize) if v0 is None else v0[self.nonrf]

        # to get the algorithm going and stable (see Nastran
        # theoretical manual, section 11.4):
        force = force / 3.0
        h = self.h
        u_1 = d0 - v0 * h

        # compute nonlinear force terms:
        N = 0.0
        if self.nonlin_terms:
            d[self.nonrf, -1] = u_1
            self.z = {}
            for key, (func, T, args) in self.nl_dct.items():
                z0 = func(d, 0, h, **args)
                z = np.empty((z0.shape[0], nt))
                z[:, 0] = z0
                self.z[key] = z
                N += T @ z0

        if self.unc:
            force[:, 0] = (self.k * d0 + self.b * v0) / 3.0
            # method 2 from theory manual:
            # force[:, 0] = (3 * force[:, 0] + self.k * d0 +
            #                self.b * v0) / 6.0
            force /= self.Ad[:, None]
            F_1 = (self.k * u_1 + self.b * v0) / (3 * self.Ad)

            # because of the - 1 subscript, do first step outside of
            # the loop:
            d[self.nonrf, 1] = (
                force[:, 1] + force[:, 0] + F_1 + N + self.A1 * d0 + self.A0 * u_1
            )
        else:
            force[:, 0] = (self.k @ d0 + self.b @ v0) / 3.0
            # force[:, 0] = (3 * force[:, 0] + self.k @ d0 +
            #                self.b @ v0) / 6.0
            force = la.lu_solve(self.Ad, force, overwrite_b=True)
            F_1 = la.lu_solve(
                self.Ad, (self.k @ u_1 + self.b @ v0) / 3, overwrite_b=True
            )

            # because of the - 1 subscript, do first step outside of
            # the loop:
            d[self.nonrf, 1] = (
                force[:, 1] + force[:, 0] + F_1 + N + self.A1 @ d0 + self.A0 @ u_1
            )

        a[self.nonrf, 0] = (d[self.nonrf, 1] - 2 * d0 + u_1) / (h * h)
        return d, v, a, force
