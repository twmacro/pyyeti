# -*- coding: utf-8 -*-
from types import SimpleNamespace
import numpy as np
from pyyeti import expmint


class SolveExp1(object):
    """
    1st order ODE time domain solver based on the matrix exponential.

    This class is for solving: ``yd - A y = f``

    This solver is exact assuming either piece-wise linear or
    piece-wise constant forces. See :class:`SolveExp2` for more
    details on how this algorithm solves the ODE.

    Examples
    --------
    Calculate impulse response of state-space system::

        xd = A @ x + B @ u
        y  = C @ x + D @ u

    where:
        - force = 0's
        - velocity(0) = B

    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> from pyyeti.ssmodel import SSModel
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> f = 5           # 5 hz oscillator
        >>> w = 2*np.pi*f
        >>> w2 = w*w
        >>> zeta = .05
        >>> h = .01
        >>> nt = 500
        >>> A = np.array([[0, 1], [-w2, -2*w*zeta]])
        >>> B = np.array([[0], [3]])
        >>> C = np.array([[8, -5]])
        >>> D = np.array([[0]])
        >>> F = np.zeros((1, nt), float)
        >>> ts = ode.SolveExp1(A, h)
        >>> sol = ts.tsolve(B @ F, B[:, 0])
        >>> y = C @ sol.d
        >>> fig = plt.figure('Example')
        >>> fig.clf()
        >>> ax = plt.plot(sol.t, y.T,
        ...               label='SolveExp1')
        >>> ssmodel = SSModel(A, B, C, D)
        >>> z = ssmodel.c2d(h=h, method='zoh')
        >>> x = np.zeros((A.shape[1], nt+1), float)
        >>> y2 = np.zeros((C.shape[0], nt), float)
        >>> x[:, 0:1] = B
        >>> for k in range(nt):
        ...     x[:, k+1] = z.A @ x[:, k] + z.B @ F[:, k]
        ...     y2[:, k]  = z.C @ x[:, k] + z.D @ F[:, k]
        >>> ax = plt.plot(sol.t, y2.T, label='discrete')
        >>> leg = plt.legend(loc='best')
        >>> np.allclose(y, y2)
        True

        Compare against scipy:

        >>> from scipy import signal
        >>> ss = ssmodel.getlti()
        >>> tout, yout = ss.impulse(T=sol.t)
        >>> ax = plt.plot(tout, yout, label='scipy')
        >>> leg = plt.legend(loc='best')
        >>> np.allclose(yout, y.ravel())
        True
    """

    def __init__(self, A, h, order=1):
        """
        Instantiates a :class:`SolveExp1` solver.

        Parameters
        ----------
        A : 2d ndarray
            The state-space matrix: ``yd - A y = f``
        h : scalar or None
            Time step or None; if None, the `E`, `P`, `Q` members will
            not be computed.
        order : scalar, optional
            Specify which solver to use:

            - 0 for the zero order hold (force stays constant across
              time step)
            - 1 for the 1st order hold (force can vary linearly across
              time step)

        Notes
        -----
        The class instance is populated with the following members:

        =======   =================================================
        Member    Description
        =======   =================================================
        A         the input `A`
        h         time step
        n         number of total DOF (``A.shape[0]``)
        order     order of solver (0 or 1; see above)
        E, P, Q   output of :func:`pyyeti.expmint.getEPQ`; they are
                  matrices used to solve the ODE
        pc        True if E, P, and Q member have been calculated;
                  False otherwise
        =======   =================================================

        The E, P, and Q entries are used to solve the ODE::

            for j in range(1, nt):
                d[:, j] = E*d[:, j-1] + P*F[:, j-1] + Q*F[:, j]
        """
        if h:
            E, P, Q = expmint.getEPQ(A, h, order)
            self.E = E
            self.P = P
            self.Q = Q
        self.A = A
        self.h = h
        self.order = order
        self.n = A.shape[0]

    def tsolve(self, force, d0=None):
        """
        Solve time-domain 1st order ODE equations.

        Parameters
        ----------
        force : 2d ndarray
            The force matrix; ndof x time
        d0 : 1d ndarray; optional
            Displacement initial conditions; if None, zero ic's are
            used.

        Returns
        -------
        A record (SimpleNamespace class) with the members:

        d : 2d ndarray
            Displacement; ndof x time
        v : 2d ndarray
            Velocity; ndof x time
        h : scalar
            Time-step
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        """
        force = np.atleast_2d(force)
        if force.shape[0] != self.n:
            raise ValueError(
                f"Force matrix has {force.shape[0]} rows; {self.n} rows are expected"
            )
        nt = force.shape[1]
        d = np.zeros((self.n, nt))  # , float, order='F')
        if d0 is not None:
            d[:, 0] = d0
        else:
            d0 = np.zeros(self.n, float)
        if nt > 1:
            if not self.h:
                raise RuntimeError("instantiate the class with a valid time step.")
            # calc force term outside loop:
            if self.order == 1:
                PQF = self.P @ force[:, :-1] + self.Q @ force[:, 1:]
            else:
                PQF = self.P @ force[:, :-1]
            E = self.E
            for j in range(1, nt):
                d0 = d[:, j] = E @ d0 + PQF[:, j - 1]
            t = self.h * np.arange(nt)
        else:
            t = np.array([0.0])
        return SimpleNamespace(d=d, v=force + self.A @ d, h=self.h, t=t)
