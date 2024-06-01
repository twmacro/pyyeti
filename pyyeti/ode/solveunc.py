# -*- coding: utf-8 -*-
from types import SimpleNamespace
import scipy.linalg as la
import numpy as np
from ._base_ode_class import _BaseODE
from ._utilities import get_su_coef, eigss, addconj, delconj, _process_incrb

try:
    import numba
except ImportError:
    HAVE_NUMBA = False
else:
    HAVE_NUMBA = True

# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _solve_real_unc_inner_loop(order, D, V, F, G, A, B, Fp, Gp, Ap, Bp, fk, nt):
    di = D[:, 0]
    vi = V[:, 0]
    fki = fk[:, 0]
    if order == 1:
        for i in range(1, nt):
            fki1 = fk[:, i]
            ABFi = A * fki + B * fki1
            ABFpi = Ap * fki + Bp * fki1
            din = F * di + G * vi + ABFi
            V[:, i] = vi = Fp * di + Gp * vi + ABFpi
            D[:, i] = di = din
            fki = fki1
    else:
        AB = A + B
        ABp = Ap + Bp
        for i in range(1, nt):
            ABFi = AB * fki
            ABFpi = ABp * fki
            din = F * di + G * vi + ABFi
            V[:, i] = vi = Fp * di + Gp * vi + ABFpi
            D[:, i] = di = din
            fki = fk[:, i]


if HAVE_NUMBA:
    _solve_real_unc_inner_loop = numba.njit(cache=True)(_solve_real_unc_inner_loop)


class SolveUnc(_BaseODE):
    r"""
    2nd order ODE time and frequency domain solvers for "uncoupled"
    equations of motion

    This class is for solving:

    .. math::
        M \ddot{q} + C \dot{q} + K q = P

    Note that the mass, damping and stiffness can be fully populated
    (coupled). If mass and stiffness are diagonal, but damping is
    coupled, see also :class:`SolveCDF`.

    Like :class:`SolveExp2`, this solver is exact assuming piece-wise
    linear forces (if `order` is 1) or piece-wise constant forces (if
    `order` is 0).

    **Uncoupled Equations**

    For uncoupled equations, pre-formulated integration coefficients
    are used as shown in the following displacement and velocity
    recurrence relations (think of the coefficients as diagonal
    matrices):

    .. math::
        \begin{aligned}
        q_{i+1} &= F q_i + G \dot{q}_i + A P_i + B P_{i+1}

        \dot{q}_{i+1} &= F_p q_i + G_p \dot{q}_i +
           A_p P_i + B_p P_{i+1}
        \end{aligned}

    Those coefficients (computed in :func:`get_su_coef` for
    underdamped, critically-damped, and overdamped systems) can be
    derived as follows. Consider the equation of motion for a single
    degree of freedom:

    .. math::
        \ddot{q}(t) + 2 \zeta \omega_n \dot{q}(t) +
            {\omega_n}^2 q(t) = \frac{1}{m} P(t)

    For illustration, we'll assume that we have an underdamped system
    (:math:`\zeta < 1`). The exact solution consists of a free-decay
    part (the homogeneous solution) and a forced-response part (the
    particular solution):

    .. math::
        q(t) = e^{-\zeta \omega_n t}
           \left [ q_0 \cos(\omega_d t) + \frac{v_0 +
                   \zeta \omega_n q_0}{\omega_d} \sin(\omega_d t)
           \right ]
           + \frac{1}{m \omega_d}
             \int_0^t {e^{-\zeta \omega_n (t - \tau)}
                       \sin(\omega_d (t - \tau)) P(\tau) d \tau}

    where the displacement and velocity initial conditions are
    :math:`q_0` and :math:`v_0`, and :math:`\omega_d` is the damped
    natural frequency:

    .. math::
        \omega_d = \omega_n \sqrt{1-\zeta^2}

    Consider a single time step, from :math:`t_n` to
    :math:`t_{n+1}`. If we assume that the force varies linearly
    within that time step, we can solve the integral in the above
    equation (Duhamel’s integral) that will be valid from :math:`t_n`
    to :math:`t_{n+1}` (:math:`\tau` will still range from :math:`0`
    to :math:`t`). Given the initial conditions :math:`q_n` and
    :math:`v_n` for the time step, we would then have the solution
    :math:`q(t)` that would be valid from :math:`t_n` to
    :math:`t_{n+1}`. Define the time step size as
    :math:`h`. Therefore:

        1. In the equation above, :math:`t = 0` now represents
           :math:`t_n` and :math:`t=h` represents :math:`t_{n+1}`
        2. :math:`q_0` becomes :math:`q_n` and :math:`v_0` becomes
           :math:`v_n`
        3. :math:`P(\tau) = P_n + (P_{n+1} - P_n) \tau / h` where
           :math:`0 \le \tau \le h`

    Solving the integral from :math:`\tau = 0` to :math:`\tau = t` and
    then setting :math:`t = h` gives the recurrence relation for
    displacement shown above. For the velocity recurrence relation, we
    take the derivative of the :math:`q(t)` expression just prior to
    setting :math:`t = h` and then set :math:`t = h`.

    **Coupled Equations**

    For coupled systems, the rigid-body modes are solved as described
    above: using the pre-formulated coefficients. The elastic modes
    part of the equations of motion are transformed into state-space:

    .. math::
        \left\{
            \begin{array}{c} \ddot{q} \\ \dot{q} \end{array}
        \right\} - \left[
            \begin{array}{cc} -M^{-1} C & -M^{-1} K \\ I & 0 \end{array}
        \right] \left\{
            \begin{array}{c} \dot{q} \\ q \end{array}
        \right\} = \left\{
            \begin{array}{c} M^{-1} P \\ 0 \end{array} \right\}

    or:

    .. math::
        \dot{y} - A y = w

    The complex eigensolution is used to decouple the equations:

    .. math::
       A \Phi = \Phi \lambda

    Where :math:`\Phi` is a matrix of right eigenvectors and
    :math:`\lambda` is a diagonal matrix of eigenvalues. By defining:

    .. math::
        y = \Phi u

    The equations become:

    .. math::
        \Phi \dot{u} - A \Phi u = w

        \Phi^{-1} \Phi \dot{u} - \Phi^{-1} A \Phi u = \Phi^{-1} w

        \dot{u} - \lambda u = v

    Since those equations are uncoupled, they can be solved very
    efficiently assuming the forces are piece-wise linear or
    piece-wise constant. For the i-th equation:

    .. math::
        u_i = e^{\lambda_i t} \left (
        u_i(0) + \int_0^t {e^{-\lambda_i \tau} v_i d\tau}
        \right )

    By only requiring the solution at every time step and assuming a
    constant step size of `h`:

    .. math::
        u_{i, n+1} = e^{\lambda_i h} \left (
        u_{i, n} + \int_0^h {e^{-\lambda_i \tau} v_i(t_n+\tau) d\tau}
        \right )

    By assuming :math:`v(t_n+\tau)` is piece-wise linear or constant
    for each step, an exact, closed form solution can be
    calculated. The class function
    :func:`SolveUnc._get_complex_su_coefs` computes the integration
    coefficient vectors `Fe` (:math:`e^{\lambda h}`), `Ae`, and `Be`
    so that a solution (for all equations) can be computed by (think
    of `Fe`, `Ae`, and `Be` as diagonal matrices):

    .. math::
        u_{n+1} = F_e u_{n} + A_e v_{n} + B_e v_{n+1}

    .. note::
        The above equations (for both uncoupled and coupled equations)
        are for the non-residual-flexibility modes. The 'rf' modes are
        solved statically and any initial conditions are ignored for
        them.

    For a static solution:

        - rigid-body displacements = zeros
        - elastic displacements = inv(k[elastic]) * P[elastic]
        - velocity = zeros
        - rigid-body accelerations = inv(m[rigid]) * P[rigid]
        - elastic accelerations = zeros

    See also
    --------
    :class:`SolveCDF`, :class:`SolveExp2`.

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
        >>> h = 0.001                             # time step
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
        >>> fig = plt.figure('Example', figsize=[8, 8], clear=True,
        ...                  layout='constrained')
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
    """

    def __init__(
        self,
        m,
        b,
        k,
        h=None,
        rb=None,
        rf=None,
        order=1,
        pre_eig=False,
        cd_as_force=False,
    ):
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
                `pre-eig` if necessary to convert to modal-space. This
                means that if `rb` is a partition vector, it specifies
                the rigid-body modes *after* the `pre_eig`
                operation. See also `pre_eig`.

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
            rigid-body modes which is necessary for the complex
            eigenvalue method to work properly on systems with
            rigid-body modes (unless those modes have damping). No
            modes are truncated. Only works if stiffness is
            symmetric/hermitian and mass is positive definite (see
            :func:`scipy.linalg.eigh`). Just leave it as False if the
            equations are already in modal space.
        cd_as_force : bool; optional
            If damping is the only coupled matrix (after `pre_eig` if
            that is used), then setting this option to True means that:

                1. The ODEs are solved with the diagonal terms only
                   (this uses the standard pre-formulated integration
                   coefficients from :func:`get_su_coef`).
                2. The off-diagonal damping terms are treated as a
                   force by moving them to the right-hand-side.

            Note that the solution is not piece-wise linear exact in
            this case. The assumption is that the damping diagonal is
            dominant enough for the uncoupled solver coefficients to
            be good enough. A finer time-step may also be required.
            This option can be particularly advantageous when the
            generator (:func:`SolveUnc.generator`) feature is used: in
            one test, using this option was approximately 70 times
            faster than the default (which uses the complex eigenvalue
            solution for coupled damping) and more than 200 times
            faster the the :class:`SolveExp2` solver. In that test,
            the accuracy was acceptable with 25 points per cycle (ppc)
            at the highest frequency (``ppc = 1 / (h * freq_high)``).
            However, accuracy is problem dependent and needs to be
            verified before trusting the results.

        Notes
        -----
        The instance is populated with some or all of the following
        members. Note that in the table, `non-rf/elastic` means
        `non-rf` for uncoupled systems, `elastic` for coupled -- the
        difference is whether or not the rigid-body modes are
        included in the final "k" matrix: they are for uncoupled but
        not for coupled.

        =========  ===================================================
        Member     Description
        =========  ===================================================
        m          mass for the non-rf/elastic modes
        b          damping for the non-rf/elastic modes
        k          stiffness for the non-rf/elastic modes
        m_orig     original mass
        b_orig     original damping
        k_orig     original stiffness
        h          time step
        rb         index vector or slice for the rb modes
        el         index vector or slice for the el modes
        rf         index vector or slice for the rf modes
        _rb        index vector or slice for the rb modes relative to
                   the non-rf/elastic modes
        _el        index vector or slice for the el modes relative to
                   the non-rf/elastic modes
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
                   :func:`SolveUnc.get_su_eig`
        pre_eig    True if the "pre" eigensolution was done; False
                   otherwise
        phi        the mode shape matrix from the "pre" eigensolution;
                   only present if `pre_eig` is True
        cdforces   True if `cd_as_force` option is being used (which
                   means damping is coupled, but mass and stiffness
                   are not)
        bo         Off-diagonal damping terms (present when `cdforces`
                   is True
        systype    float or complex; determined by `m` `b` `k`
        =========  ===================================================

        Unlike for :class:`SolveExp2`, `order` is not used until the
        solver is called. In other words, this routine prepares the
        integration coefficients for a first order hold no matter what
        the setting of `order` is, but the solver will adjust the use
        of the forces to account for the `order` setting.

        The mass, damping and stiffness may be real or complex since
        this solver is also used for frequency domain problems.
        """
        self._common_precalcs(m, b, k, h, rb, rf, pre_eig, cd_as_force)
        if self.ksize:
            if self.unc and self.systype is float:
                self._inv_m()
                self.pc = get_su_coef(self.m, self.b, self.k, h, self.rb)
                if self.cdforces and self.pc:
                    # add ``alpha = bo (I + Bp bo)^-1`` to pc:
                    Bp = self.pc.Bp[:, None]
                    tmp = np.eye(self.ksize) + Bp * self.bo
                    self.pc.alpha = la.solve(tmp.T, self.bo.T).T
            else:
                self.pc = self.get_su_eig(h is not None)
        else:
            self.pc = None
        self._mk_slices()  # dorbel=True)
        self.order = order

    def tsolve(self, force, d0=None, v0=None, static_ic=False):
        """Solve time-domain 2nd order ODE equations

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
        if self.nonrfsz:
            if self.unc and self.systype is float:
                # for uncoupled, m, b, k have rb+el (all nonrf)
                if self.cdforces:
                    self._solve_real_unc_cdforces(d, v, force)
                else:
                    self._solve_real_unc(d, v, force)
            else:
                # for coupled, m, b, k have el only
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
        d, v : 2d ndarrays
            The displacement and velocity arrays. Only the first
            column of `d` and `v` are set; other values are all zero.

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
                "generator not yet implemented for the case when"
                " different types of equations are interspersed (eg,"
                " a residual-flexibility DOF in the middle of the"
                " elastic DOFs)"
            )
        d, v, a, force = self._init_dva_part(nt, F0, d0, v0, static_ic)
        self._d, self._v, self._a, self._force = d, v, a, force
        if self.unc and self.systype is float:
            # for uncoupled, m, b, k have rb+el (all nonrf)
            if self.cdforces:
                generator = self._solve_real_unc_generator_cdforces(d, v, F0)
            else:
                generator = self._solve_real_unc_generator(d, v, F0)
            next(generator)
            return generator, d, v
        else:
            # for coupled, m, b, k have el only
            generator = self._solve_complex_unc_generator(d, v, a, F0)
            next(generator)
            return generator, d, v

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

            phir = phi[:, rb]       # for complex
            phik = phi[:, kdof]     # for both; if real, includes rb
            phirf = phi[:, rf]      # for both
            B = pc.B[:, None]       # for real
            Bp = pc.Bp[:, None]     # for real
            A = pc.A                # for complex
            Ap = pc.Ap              # for complex
            Be = pc.Be[:, None]     # for complex
            ur_inv_v = pc.ur_inv_v  # for complex
            rur_d = pc.rur_d        # for complex
            iur_d = pc.iur_d        # for complex
            rur_v = pc.rur_v        # for complex
            iur_v = pc.iur_v        # for complex

        For real, if `velo` is False::

            flex = phik @ (B * phik.T) + phirf @ (ikrf * phirf)

        For real, if `velo` is True::

            flex = phik @ (Bp * phik.T)

        For complex, if `velo` is False::

            flex_rb = phir @ (0.5*A*(imrb @ phir.T))
            temp = Be*(ur_inv_v @ invm @ phik.T)
            flex_el = phik @ (rur_d @ temp.real - iur_d @ temp.imag)
            flex_rf = phirf @ ikrf @ phirf.T
            flex = flex_rb + flex_el + flex_rf

        For complex, if `velo` is True::

            flex_rb = phir @ (Ap*(imrb @ phir.T))
            temp = Be*(ur_inv_v @ invm @ phik.T)  # same as above
            flex_el = phik @ (rur_v @ temp.real - iur_v @ temp.imag)
            flex = flex_rb + flex_el

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
        if self.order == 0:
            flex = 0.0
        else:
            if self.unc:
                flex = self._get_f2x_real_unc(phi, velo)
            else:
                flex = self._get_f2x_complex_unc(phi, velo)
        return self._flex(flex, phi)

    def fsolve(self, force, freq, incrb="dva", rf_disp_only=False):
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
        incrb : string; optional
            Specifies how to handle rigid-body responses: can include
            "a", "v", and/or "d" to specify that the rigid-body
            response should be computed for acceleration, velocity,
            and/or displacement. For example, ``incrb=""`` means that
            no rigid-body responses are included, and ``incrb="va"``
            (or ``incrb="av"``) means that the rigid-body response
            will be included for acceleration and velocity but not
            displacement. Letters can be included in any order.

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
            acceleration terms for residual-flexibility modes. (The
            displacements for these modes is always computed
            statically.) If True, the velocities and accelerations are
            set to zero. If False, the default and generally
            recommended setting, they are computed from the normal
            frequency-domain relationships::

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
        The rigid-body and residual-flexibility modes are solved
        independently. The displacements for the residual-flexibility
        modes are solved statically, and the velocities and
        accelerations are determined according to `rf_disp_only`.

        Rigid-body velocities and displacements are undefined where
        `freq` is zero. So, if those rigid-body responses are
        requested (with `incrb`), this routine just sets these
        responses to zero.

        See also
        --------
        :class:`FreqDirect`

        Raises
        ------
        NotImplementedError
            When attribute `cd_as_force` is True, since off-diagonal
            damping as forces is not implemented for the frequency
            domain.

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
            >>> fig = plt.figure('Example', figsize=[8, 8], clear=True,
            ...                  layout='constrained')
            >>> labels = ['Rigid-body', 'Underdamped',
            ...           'Critically Damped', 'Overdamped']
            >>> for j, name in zip(range(4), labels):
            ...     _ = plt.subplot(4, 1, j+1)
            ...     _ = plt.plot(freq, abs(sol.a[j]))
            ...     _ = plt.title(name)
            ...     _ = plt.ylabel('Acceleration')
            ...     _ = plt.xlabel('Frequency (Hz)')
        """
        incrb = _process_incrb(incrb)  # deprecated in v0.99.9
        force = np.atleast_2d(force)
        freq = np.atleast_1d(freq)
        d, v, a, force = self._init_dva(
            force, None, None, False, istime=False, freq=freq, rf_disp_only=rf_disp_only
        )
        self._force_freq_compat_chk(force, freq)
        if self.nonrfsz:
            if self.unc:
                if self.cdforces:
                    raise NotImplementedError(
                        "off-diagonal damping as forces is not "
                        "implemented for the frequency domain"
                    )
                # for uncoupled, m, b, k have rb+el (all nonrf)
                self._solve_freq_unc(d, v, a, force, freq, incrb)
            else:
                # for coupled, m, b, k have el only
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
        D = d[kdof]
        V = v[kdof]

        _solve_real_unc_inner_loop(
            self.order,
            D,
            V,
            pc.F,
            pc.G,
            pc.A,
            pc.B,
            pc.Fp,
            pc.Gp,
            pc.Ap,
            pc.Bp,
            force[kdof],
            nt,
        )

        if not self.slices:
            d[kdof] = D
            v[kdof] = V

    def _solve_real_unc_cdforces(self, d, v, force):
        """Solve the real uncoupled equations for :class:`SolveUnc`"""
        # solve: ... V[:, i+1] needs to be solved for, but these are
        # the starting equations:
        # for i in range(nt-1):
        #     D[:,i+1] = F *D[:, i] + G *V[:, i] +
        #                A *force[:, i] + B *force[:, i+1]
        #                - A * Co @ V[:, i] - B * Co @ V[:, i+1]
        #     V[:,i+1] = Fp*D[:, i] + Gp*V[:, i] +
        #                Ap*force[:, i] + Bp*force[:, i+1]
        #                - Ap * Co @ V[:, i] - Bp * Co @ V[:, i+1]
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
            ABF = A[:, None] * force[kdof, :-1] + B[:, None] * force[kdof, 1:]
            ABFp = Ap[:, None] * force[kdof, :-1] + Bp[:, None] * force[kdof, 1:]
        else:
            ABF = (A + B)[:, None] * force[kdof, :-1]
            ABFp = (Ap + Bp)[:, None] * force[kdof, :-1]
        di = D[:, 0]
        vi = V[:, 0]
        alpha = self.pc.alpha
        bo = self.bo
        dmpfrc0 = bo @ vi
        for i in range(nt - 1):
            v_part = Fp * di + Gp * vi + ABFp[:, i] - Ap * dmpfrc0
            dmpfrc1 = alpha @ v_part
            D[:, i + 1] = di = F * di + G * vi + ABF[:, i] - A * dmpfrc0 - B * dmpfrc1
            V[:, i + 1] = vi = v_part - Bp * dmpfrc1
            dmpfrc0 = dmpfrc1

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

        if self.order == 1:
            if self.rfsize:
                # rigid-body and elastic equations:
                D = d[kdof]
                V = v[kdof]
                # resflex equations:
                drf = d[rf]
                # for i in range(nt-1):
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        F1k = F1[kdof]
                        D[:, i] += B * F1k
                        V[:, i] += Bp * F1k
                        drf[:, i] += ikrf * F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0k = Force[kdof, i - 1]
                        F1k = F1[kdof]
                        di = D[:, i - 1]
                        vi = V[:, i - 1]
                        D[:, i] = F * di + G * vi + A * F0k + B * F1k
                        V[:, i] = Fp * di + Gp * vi + Ap * F0k + Bp * F1k
                        # rf:
                        drf[:, i] = ikrf * F1[rf]
            else:
                # only rigid-body and elastic equations:
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        d[:, i] += B * F1
                        v[:, i] += Bp * F1
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0 = Force[:, i - 1]
                        di = d[:, i - 1]
                        vi = v[:, i - 1]
                        d[:, i] = F * di + G * vi + A * F0 + B * F1
                        v[:, i] = Fp * di + Gp * vi + Ap * F0 + Bp * F1
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
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        drf[:, i] += ikrf * F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0k = Force[kdof, i - 1]
                        di = D[:, i - 1]
                        vi = V[:, i - 1]
                        D[:, i] = F * di + G * vi + AB * F0k
                        V[:, i] = Fp * di + Gp * vi + ABp * F0k
                        # rf:
                        drf[:, i] = ikrf * F1[rf]
            else:
                # only rigid-body and elastic equations:
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0 = Force[:, i - 1]
                        di = d[:, i - 1]
                        vi = v[:, i - 1]
                        d[:, i] = F * di + G * vi + AB * F0
                        v[:, i] = Fp * di + Gp * vi + ABp * F0

    def _solve_real_unc_generator_cdforces(self, d, v, F0):
        """Solve the real uncoupled equations for :class:`SolveUnc`"""
        nt = d.shape[1]
        if nt == 1:
            yield
        Force = self._force

        if self.rfsize:
            rf = self.rf
            ikrf = self.ikrf.ravel()

        if not self.ksize:
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
        alpha = self.pc.alpha
        bo = self.bo
        i_last = 0

        if self.order == 1:
            if self.rfsize:
                # rigid-body and elastic equations:
                D = d[kdof]
                V = v[kdof]
                dmpfrc1 = bo @ V[:, 0]

                # resflex equations:
                drf = d[rf]
                # for i in range(nt-1):
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        F1k = F1[kdof]

                        v_part = Bp * F1k
                        dmpfrc1_addon = alpha @ v_part
                        dmpfrc1 += dmpfrc1_addon
                        D[:, i] += B * (F1k - dmpfrc1_addon)
                        V[:, i] += v_part - Bp * dmpfrc1_addon

                        drf[:, i] += ikrf * F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0k = Force[kdof, i - 1]
                        F1k = F1[kdof]
                        di = D[:, i - 1]
                        vi = V[:, i - 1]

                        dmpfrc0 = dmpfrc1 if i_last == i - 1 else bo @ vi
                        i_last = i
                        _f0 = F0k - dmpfrc0
                        v_part = Fp * di + Gp * vi + Ap * _f0 + Bp * F1k
                        dmpfrc1 = alpha @ v_part
                        D[:, i] = F * di + G * vi + A * _f0 + B * (F1k - dmpfrc1)
                        V[:, i] = v_part - Bp * dmpfrc1

                        # rf:
                        drf[:, i] = ikrf * F1[rf]
            else:
                dmpfrc1 = bo @ v[:, 0]
                # only rigid-body and elastic equations:
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1

                        v_part = Bp * F1
                        dmpfrc1_addon = alpha @ v_part
                        dmpfrc1 += dmpfrc1_addon
                        d[:, i] += B * (F1 - dmpfrc1_addon)
                        v[:, i] += v_part - Bp * dmpfrc1_addon
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0 = Force[:, i - 1]
                        di = d[:, i - 1]
                        vi = v[:, i - 1]
                        dmpfrc0 = dmpfrc1 if i_last == i - 1 else bo @ vi
                        i_last = i
                        _f0 = F0 - dmpfrc0
                        v_part = Fp * di + Gp * vi + Ap * _f0 + Bp * F1
                        dmpfrc1 = alpha @ v_part
                        d[:, i] = F * di + G * vi + A * _f0 + B * (F1 - dmpfrc1)
                        v[:, i] = v_part - Bp * dmpfrc1
        else:
            # order == 0
            AB = A + B
            ABp = Ap + Bp
            if self.rfsize:
                # rigid-body and elastic equations:
                D = d[kdof]
                V = v[kdof]
                dmpfrc1 = bo @ V[:, 0]

                # resflex equations:
                drf = d[rf]
                # for i in range(nt-1):
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                        drf[:, i] += ikrf * F1[rf]
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0k = Force[kdof, i - 1]
                        di = D[:, i - 1]
                        vi = V[:, i - 1]

                        dmpfrc0 = dmpfrc1 if i_last == i - 1 else bo @ vi
                        i_last = i
                        v_part = Fp * di + Gp * vi + ABp * F0k - Ap * dmpfrc0
                        dmpfrc1 = alpha @ v_part
                        D[:, i] = F * di + G * vi + AB * F0k - A * dmpfrc0 - B * dmpfrc1
                        V[:, i] = v_part - Bp * dmpfrc1

                        # rf:
                        drf[:, i] = ikrf * F1[rf]
            else:
                # only rigid-body and elastic equations:
                dmpfrc1 = bo @ v[:, 0]
                while True:
                    j, F1 = yield
                    if j < 0:
                        # add to previous soln
                        Force[:, i] += F1
                    else:
                        i = j
                        Force[:, i] = F1
                        # rb + el:
                        F0 = Force[:, i - 1]
                        di = d[:, i - 1]
                        vi = v[:, i - 1]

                        dmpfrc0 = dmpfrc1 if i_last == i - 1 else bo @ vi
                        i_last = i
                        v_part = Fp * di + Gp * vi + ABp * F0 - Ap * dmpfrc0
                        dmpfrc1 = alpha @ v_part
                        d[:, i] = F * di + G * vi + AB * F0 - A * dmpfrc0 - B * dmpfrc1
                        v[:, i] = v_part - Bp * dmpfrc1

    def _get_f2x_real_unc(self, phi, velo):
        """
        Get f2x transform for henkel-mar
        """
        if self.ksize:
            # rb and el equations:
            pc = self.pc
            kdof = self.kdof
            phik = phi[:, kdof]
            if velo:
                B = pc.Bp[:, None]
            else:
                B = pc.B[:, None]
            if self.cdforces:
                alpha = pc.alpha
                tmp = B * (np.eye(self.ksize) - alpha * pc.Bp)
                flex = phik @ tmp @ phik.T
            else:
                flex = phik @ (B * phik.T)
        else:
            flex = 0.0
        flex = self._add_rf_flex(flex, phi, velo, True)
        return flex

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
                    rbforce = la.lu_solve(self.imrb, force[rb], check_finite=False)
            else:
                rbforce = force[rb]
            if nt > 1:
                G = pc.G
                A = pc.A
                Ap = pc.Ap
                if self.order == 1:
                    AF = A * (rbforce[:, :-1] + rbforce[:, 1:] / 2)
                    AFp = Ap * (rbforce[:, :-1] + rbforce[:, 1:])
                else:
                    AF = (1.5 * A) * rbforce[:, :-1]
                    AFp = (2 * Ap) * rbforce[:, :-1]
                drb = d[rb]
                vrb = v[rb]
                di = drb[:, 0]
                vi = vrb[:, 0]
                for i in range(nt - 1):
                    di = drb[:, i + 1] = di + G * vi + AF[:, i]
                    vi = vrb[:, i + 1] = vi + AFp[:, i]
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
                    imf = la.lu_solve(self.invm, force[kdof], check_finite=False)
            else:
                imf = force[kdof]
            w = ur_inv_v @ imf
            if self.order == 1:
                ABF = Ae[:, None] * w[:, :-1] + Be[:, None] * w[:, 1:]
            else:
                ABF = (Ae + Be)[:, None] * w[:, :-1]

            y = np.empty((ur_inv_v.shape[0], nt), complex, order="F")
            di = y[:, 0] = ur_inv_v @ v[kdof, 0] + ur_inv_d @ d[kdof, 0]
            for i in range(nt - 1):
                di = y[:, i + 1] = Fe * di + ABF[:, i]
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
        order = self.order

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
                    imrb = la.lu_solve(imrb, np.eye(rbsize), check_finite=False)
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
            if order == 0:
                A = 1.5 * A
                Ap = 2.0 * Ap
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
            if order == 0:
                Ae = Ae + Be
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
                    invm = la.lu_solve(invm, np.eye(ksize), check_finite=False)
            D = d[kdof]
            V = v[kdof]

        if rfsize:
            rf = self.rf
            ikrf = self.ikrf
            if unc:
                ikrf = ikrf.ravel()
            else:
                ikrf = la.lu_solve(ikrf, np.eye(rfsize), check_finite=False)
            drf = d[rf]

        while True:
            j, F1 = yield
            if j < 0:
                # add to previous soln
                Force[:, i] += F1
                if rbsize:
                    if m is not None:
                        if unc:
                            F1rb = imrb * F1[rb]
                        else:
                            F1rb = imrb @ F1[rb]
                    else:
                        F1rb = F1[rb]
                    if order == 1:
                        AF = A * 0.5 * F1rb
                        AFp = Ap * F1rb
                        drb[:, i] += AF
                        vrb[:, i] += AFp
                    arb[:, i] += F1rb

                if order == 1:
                    if ksize:
                        F1k = F1[kdof]
                        if m is not None:
                            if unc:
                                F1k = invm * F1k
                            else:
                                F1k = invm @ F1k
                        w1 = ur_inv_v @ F1k
                        yn = Be * w1
                        if systype is float:
                            ry = yn.real
                            iy = yn.imag
                            D[:, i] += rur_d @ ry - iur_d @ iy
                            V[:, i] += rur_v @ ry - iur_v @ iy
                        else:
                            D[:, i] += ur_d @ yn
                            V[:, i] += ur_v @ yn

                if rfsize:
                    if unc:
                        drf[:, i] += ikrf * F1[rf]
                    else:
                        drf[:, i] += ikrf @ F1[rf]
            else:
                i = j
                Force[:, i] = F1
                F0 = Force[:, i - 1]
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
                        AF = A * (F0rb + 0.5 * F1rb)
                        AFp = Ap * (F0rb + F1rb)
                    else:
                        AF = A * F0rb
                        AFp = Ap * F0rb
                    vi = vrb[:, i - 1]
                    drb[:, i] = drb[:, i - 1] + G * vi + AF
                    vrb[:, i] = vi + AFp
                    arb[:, i] = F1rb

                if ksize:
                    # F0k = Force[kdof, i-1]
                    F0k = F0[kdof]
                    if order == 1:
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
                        ABF = Ae * w0 + Be * w1
                    else:
                        if m is not None:
                            if unc:
                                F0k = invm * F0k
                            else:
                                F0k = invm @ F0k
                        w0 = ur_inv_v @ F0k
                        ABF = Ae * w0
                    # [V; D] = ur @ y
                    # y = ur_inv @ [V; D] =
                    #  [ur_inv_v, ur_inv_d] @ [V; D]
                    y = ur_inv_v @ V[:, i - 1] + ur_inv_d @ D[:, i - 1]
                    yn = Fe * y + ABF
                    if systype is float:
                        # Can do real math for recovery. Note that the
                        # imaginary part of 'd' and 'v' would be zero
                        # if no modes were deleted of the complex
                        # conjugate pairs. The real part is correct
                        # whether or not modes were deleted, and
                        # that's all we need.
                        ry = yn.real
                        iy = yn.imag
                        D[:, i] = rur_d @ ry - iur_d @ iy
                        V[:, i] = rur_v @ ry - iur_v @ iy
                    else:
                        # [V; D] = ur @ y
                        D[:, i] = ur_d @ yn
                        V[:, i] = ur_v @ yn

                if rfsize:
                    if unc:
                        drf[:, i] = ikrf * F1[rf]
                    else:
                        drf[:, i] = ikrf @ F1[rf]

    def _get_f2x_complex_unc(self, phi, velo):
        """
        Get f2x transform for henkel-mar
        """
        unc = self.unc
        m = self.m
        pc = self.pc

        flex = 0.0
        if self.ksize:
            self._delconj()
            Be = pc.Be
            rur_d = pc.rur_d
            iur_d = pc.iur_d
            rur_v = pc.rur_v
            iur_v = pc.iur_v
            ur_inv_v = pc.ur_inv_v

            kdof = self.kdof
            phik = phi[:, kdof]
            flexe = phik.T
            if m is not None:
                invm = self.invm
                if unc:
                    flexe = invm.ravel()[:, None] * flexe
                else:
                    flexe = la.lu_solve(invm, flexe, check_finite=False)
            flexe = Be[:, None] * (ur_inv_v @ flexe)
            if velo:
                flexe = rur_v @ flexe.real - iur_v @ flexe.imag
            else:
                flexe = rur_d @ flexe.real - iur_d @ flexe.imag
            flex = flex + phik @ flexe

        if self.rbsize:
            rb = self.rb
            phir = phi[:, rb]
            flexr = phir.T
            if m is not None:
                imrb = self.imrb
                if unc:
                    flexr = imrb.ravel()[:, None] * flexr
                else:
                    flexr = la.lu_solve(imrb, flexr, check_finite=False)
            if velo:
                flexr = pc.Ap * flexr
            else:
                flexr = (0.5 * pc.A) * flexr
            flex = flex + phir @ flexr

        flex = self._add_rf_flex(flex, phi, velo, unc)
        return flex

    def get_su_eig(self, delcc):
        """Does pre-calcs for the `SolveUnc` solver via the complex
        eigenvalue approach.

        Parameters
        ----------
        delcc : bool
            If True, delete one of each complex-conjugate pair and put
            the appropriate factor of 2.0 in the kept mode
            (see :func:`ode.eigss`). It will be automatically added
            back in if needed later (for example, if
            :func:`SolveUnc.fsolve` is called).

        Class is expected to be populated with:

        m : 1d or 2d ndarray or None
            Mass; vector (of diagonal), or full; if None, mass is
            assumed identity. Has only rigid-body and elastic
            modes.
        b : 1d or 2d ndarray
            Damping; vector (of diagonal), or full. Has only
            rigid-body and elastic modes.
        k : 1d or 2d ndarray
            Stiffness; vector (of diagonal), or full. Has only
            rigid-body and elastic modes.
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
        wn : 1d ndarray
            Real natural frequencies in same order as `lam`. See
            :func:`ode.get_freq_damping`
        zeta : 1d ndarray
            Critical damping ratios. See :func:`ode.get_freq_damping`
        eig_success : bool
            True if routine is successful. False if the eigenvectors
            form a singular matrix or they do not diagonalize `A`; in
            that case, ODE solution (if computed) is most likely
            wrong.

        Notes
        -----
        The members `m`, `b`, and `k` are partitioned down to the
        elastic part only.

        See also
        --------
        :class:`SolveUnc`

        """
        pc = SimpleNamespace()
        h = self.h
        if self.rbsize:
            self._inv_mrb()
            if h:
                pc.G = h
                pc.A = h * h / 3
                pc.Ap = h / 2
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

        self._el = np.arange(self.ksize)  # testing ...
        self._rb = np.arange(0)

        if self.elsize:
            self._inv_m()
            A = self._build_A()
            eig_info = eigss(A, delcc)
            pc.wn = eig_info.wn
            pc.zeta = eig_info.zeta
            pc.eig_success = eig_info.eig_success
            if h:
                self._get_complex_su_coefs(pc, eig_info.lam, h)
            self._add_partition_copies(pc, eig_info.lam, eig_info.ur, eig_info.ur_inv)
        return pc

    def _get_complex_su_coefs(self, pc, lam, h):
        """form coefficients for piece-wise exact solution"""
        # check for rb modes (5.e-5 determined by trial and
        #  error comparing against matrix exponential solver)
        abslam = abs(lam)
        rb = abslam < 5.0e-5
        el = ~rb
        Fe = np.exp(lam * h)
        Ae = np.empty_like(Fe)
        Be = np.empty_like(Fe)
        ilam = 1 / lam[el]
        ilamh = (ilam * ilam) / h
        Ae[el] = ilamh + Fe[el] * (ilam - ilamh)
        Be[el] = Fe[el] * ilamh - ilam - ilamh
        if rb.any():
            Fe[rb] = 1.0
            Ae[rb] = h / 2.0
            Be[rb] = h / 2.0
        pc.Fe = Fe
        pc.Ae = Ae
        pc.Be = Be

    def _add_partition_copies(self, pc, lam, ur, ur_inv):
        ksize = self.ksize
        pc.lam = lam
        pc.ur = ur
        pc.ur_d = pc.ur[ksize:]
        pc.ur_v = pc.ur[:ksize]
        pc.ur_inv = ur_inv
        pc.ur_inv_v = ur_inv[:, :ksize]
        pc.ur_inv_d = ur_inv[:, ksize:]
        # pc.ur_inv_v = ur_inv[:, :ksize].copy()
        # pc.ur_inv_d = ur_inv[:, ksize:].copy()

        # real/imag copies are good for speeding things up:
        pc.rur_d = pc.ur_d.real.copy()
        pc.iur_d = pc.ur_d.imag.copy()
        pc.rur_v = pc.ur_v.real.copy()
        pc.iur_v = pc.ur_v.imag.copy()

    def _addconj(self):
        pc = self.pc
        if 2 * pc.ur_inv_v.shape[1] > pc.ur_d.shape[1]:
            # ur_inv = np.hstack((pc.ur_inv_v, pc.ur_inv_d))
            lam, ur, ur_inv = addconj(pc.lam, pc.ur, pc.ur_inv)
            if self.h:
                self._get_complex_su_coefs(pc, lam, self.h)
            self._add_partition_copies(pc, lam, ur, ur_inv)

    def _delconj(self):
        pc = self.pc
        if 2 * pc.ur_inv_v.shape[1] == pc.ur_d.shape[1]:
            # ur_inv = np.hstack((pc.ur_inv_v, pc.ur_inv_d))
            lam, ur, ur_inv, _ = delconj(pc.lam, pc.ur, pc.ur_inv, [])
            if self.h:
                self._get_complex_su_coefs(pc, lam, self.h)
            self._add_partition_copies(pc, lam, ur, ur_inv)

    def _solve_freq_rb(self, d, v, a, force, freqw, freqw2, incrb, unc):
        """Solve the rigid-body equations for
        :func:`SolveUnc.fsolve`"""
        if self.rbsize and incrb:
            rb = self.rb
            if self.m is not None:
                if unc:
                    a_rb = self.invm[self._rb] * force[rb]
                else:
                    a_rb = la.lu_solve(self.imrb, force[rb], check_finite=False)
            else:
                a_rb = force[rb]
            if "d" in incrb or "v" in incrb:
                pvnz = freqw != 0
                if "v" in incrb:
                    v[rb, pvnz] = (-1j / freqw[pvnz]) * a_rb[:, pvnz]
                if "d" in incrb:
                    d[rb, pvnz] = (-1.0 / freqw2[pvnz]) * a_rb[:, pvnz]
            if "a" in incrb:
                a[rb] = a_rb

    def _solve_freq_unc(self, d, v, a, force, freq, incrb):
        """Solve the uncoupled equations for
        :func:`SolveUnc.fsolve`"""
        # convert frequency in Hz to radian/sec:
        freqw = 2 * np.pi * freq
        freqw2 = freqw**2

        # solve rigid-body and elastic parts separately
        # - residual-flexibility part was already solved in _init_dva

        # solve rigid-body part:
        self._solve_freq_rb(d, v, a, force, freqw, freqw2, incrb, True)

        # solve elastic part:
        if self.elsize:
            el = self.el
            _el = self._el
            fw = freqw[None, :]
            fw2 = freqw2[None, :]
            if self.m is None:
                d[el] = force[el] / (
                    1j * (self.b[_el][:, None] @ fw) + self.k[_el][:, None] - fw2
                )
            else:
                d[el] = force[el] / (
                    1j * (self.b[_el][:, None] @ fw)
                    + self.k[_el][:, None]
                    - self.m[_el][:, None] @ fw2
                )
            a[el] = d[el] * -(freqw2)
            v[el] = d[el] * (1j * freqw)

    def _solve_freq_coup(self, d, v, a, force, freq, incrb):
        """Solve the coupled equations for :func:`SolveUnc.fsolve`"""
        # convert frequency in Hz to radian/sec:
        freqw = 2 * np.pi * freq
        freqw2 = freqw**2

        # solve rigid-body and elastic parts separately
        # - residual-flexibility part was already solved in _init_dva

        # solve rigid-body part:
        self._solve_freq_rb(d, v, a, force, freqw, freqw2, incrb, False)

        # solve elastic part:
        if self.ksize:
            self._addconj()
            pc = self.pc
            kdof = self.kdof
            # form complex state-space generalized force:
            if self.m is not None:
                imf = la.lu_solve(self.invm, force[kdof], check_finite=False)
            else:
                imf = force[kdof]
            w = pc.ur_inv_v @ imf
            n = w.shape[0]
            H = np.ones((n, 1)) @ (1.0j * freqw[None, :]) - pc.lam[:, None]
            d[kdof] = pc.ur_d @ (w / H)
            a[kdof] = d[kdof] * -(freqw2)
            v[kdof] = d[kdof] * (1j * freqw)
