# -*- coding: utf-8 -*-
"""
Continuous and discrete state-space model conversions.
"""

import scipy.linalg as la
import scipy.signal as signal
import numpy as np
from pyyeti import expmint


class SSModel(object):
    """
    Simple class for storing information about a continuous or
    discrete state-space model with tools for converting from one to
    another.

    Attributes
    ----------
    A, B, C, D : 2d ndarrays
        These are the continuous or discrete state-space matrices (see
        below).
    h : scalar or None
        None for continuous models, the time step for discete models.
    method : string, optional
        For discrete models, this can specify the method used to
        convert from continuous.
    prewarp : scalar or None, optional
        For discrete systems, this can specify the prewarp frequency
        (rad/sec) used in the Tustin transformation.

    Notes
    -----
    The continuous (s-domain) state-space equations are::

           xdot = A*x + B*u
           y    = C*x + D*u

    The discrete (z-domain) state-space equations are::

           x[k+1] = A*x[k] + B*u[k]
           y[k]   = C*x[k] + D*u[k]
    """

    def __init__(self, A, B, C, D, h=None, method=None, prewarp=None):
        """
        Parameters
        ----------
        A, B, C, D : 2d array_like
            These are the continuous or discrete state-space matrices.
        h : scalar or None
            None for continuous models, the time step for discete
            models.
        method : string, optional
            For discrete or continuous models, this can specify the
            method used to convert from the other form.
        prewarp : scalar or None, optional
            For discrete or continuous systems, this can specify the
            prewarp frequency (rad/sec) used in the Tustin
            transformation.
        """
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        self.D = np.atleast_2d(D)
        self.h = h
        self.method = method
        self.prewarp = prewarp

    def __repr__(self):
        """Return representation of the :class:`SSModel` system."""
        return (
            f"{self.__class__.__name__}(\n"
            f"A={self.A!r},\n"
            f"B={self.B!r},\n"
            f"C={self.C!r},\n"
            f"D={self.D!r},\n"
            f"h={self.h!r},\n"
            f"method={self.method!r},\n"
            f"prewarp={self.prewarp!r},\n)"
        )

    def getlti(self):
        """
        Return :class:`scipy.signal.lti` instance of continuous model. If
        model is discrete, :func:`d2c` is called (with defaults) to
        convert to continuous before calling :class:`scipy.signal.lti`.
        """
        if self.h:
            c = self.d2c()
            return signal.lti(c.A, c.B, c.C, c.D)
        return signal.lti(self.A, self.B, self.C, self.D)

    def d2c(self, method="foh", prewarp=0):
        """
        Compute a continuous state-space model (s-plane) from discrete
        state-space model (z-plane).

        Parameters
        ----------
        method : string, optional
            Conversion method: 'zoh', 'zoha', 'foh', or 'tustin':

            ========   ==============================================
            `method`   Transformation method
            ========   ==============================================
            'zoh'      zero order hold, use input at start of time
                       step used for time step
            'zoha'     zero order hold, use average of start and end
                       input values for time step
            'foh'      first order hold, input ramps linearly across
                       time step
            'tustin'   uses the Tustin (or bilinear) transform
            ========   ==============================================

        prewarp : scalar or None, optional
            Prewarp frequency used for the Tustin transform (rad/sec)

        Returns
        -------
        cls : instance of :class:`SSModel`
            Has attributes set for a continuous (s-plane) state-space
            model.

        Notes
        -----
        If `self` is already a continuous model, this routine quietly
        returns itself.

        The conversion is based on the matrix exponential (see
        :func:`pyyeti.expmint.expmint`). Letting `z` and `s` represent
        the discrete and continuous versions of the :class:`SSModel`
        the following equations show the conversion methods. First
        (noting that `E`, `I1`, and `I2` can be calculated by
        :func:`pyyeti.expmint.expmint` once `s.A` has been computed)::

            lambda, phi = eig(z.A)
            E = exp(s.A*h)
            I1 = integral (exp(s.A*t) dt) from 0 to h
            I2 = integral (exp(s.A*t)*t dt) from 0 to h
            I = identity

        For ``method='zoh'``, the conversion is::

            s.A = phi * diag(log(lambda)/h) * inv(phi)
            s.B = inv(z.A - I) * s.A * z.B
            s.C = z.C
            s.D = z.D

        For ``method='zoha'``, the conversion is::

            s.A = phi * diag(log(lambda)/h) * inv(phi)
            P = I1/2
            s.B = inv(P + z.A*P) * z.B
            s.C = z.C
            s.D = z.D - s.C*P*s.B

        For ``method='foh'``, the conversion is::

            s.A = phi * diag(log(lambda)/h) * inv(phi)
            Q = (I1 - I2/h)
            P = I1 - Q
            s.B = inv(P + z.A*Q) * z.B
            s.C = z.C
            s.D = z.D - s.C*Q*s.B

        For ``method='tustin'``, the conversion is::

            if prewarp == 0:
                k = 2/h     # standard Tustin method
            else:
                k = prewarp/tan(prewarp*h/2)
            Q = inv(I+z.A)
            s.A = k*(z.A-I)*Q
            s.B = (k*I-s.A)*Q*z.B
            s.C = z.C
            s.D = z.D - z.C*Q*z.B

        """
        if self.h is None:
            return self
            # raise ValueError('model is already continuous??')

        h = self.h
        if method == "tustin":
            if prewarp is None or prewarp == 0:
                k = 2 / h
            else:
                k = prewarp / np.tan(prewarp * h / 2)
            I = np.eye(self.A.shape[0])
            q = la.lu_factor(I + self.A)
            A = k * la.lu_solve(q, self.A.T - I, 1).T
            QB = la.lu_solve(q, self.B)
            B = (k * I - A).dot(QB)
            C = self.C.copy()
            D = self.D - self.C.dot(QB)
            return SSModel(A, B, C, D, method=method, prewarp=prewarp)

        # the rest all form A the same way:
        lam, phi = la.eig(self.A)
        A = (la.solve(phi.T, (phi * (np.log(lam) / h)).T).T).real

        if method == "foh":
            E, P, Q = expmint.getEPQ(A, h, 1)
            B = la.solve(P + self.A.dot(Q), self.B)
            C = self.C.copy()
            D = self.D - C.dot(Q.dot(B))
            return SSModel(A, B, C, D, method=method)

        if method == "zoh":
            I = np.eye(self.A.shape[0])
            B = la.solve(self.A - I, A.dot(self.B))
            C = self.C.copy()
            D = self.D.copy()
            return SSModel(A, B, C, D, method=method)

        if method == "zoha":
            E, P, Q = expmint.getEPQ(A, h, 0)
            P /= 2.0
            Q = P
            B = la.solve(P + self.A.dot(Q), self.B)
            C = self.C.copy()
            D = self.D - C.dot(Q.dot(B))
            return SSModel(A, B, C, D, method=method)

        raise ValueError("invalid `method` argument")

    def c2d(self, h, method="foh", prewarp=0):
        """
        Compute a discrete state-space model (z-plane) from continuous
        state-space model (s-plane).

        Parameters
        ----------
        h : scalar
            The time step for discretization
        method : string, optional
            Conversion method: 'zoh', 'zoha', 'foh', or 'tustin':

            ========   ==============================================
            `method`   Transformation method
            ========   ==============================================
            'zoh'      zero order hold, use input at start of time
                       step used for time step
            'zoha'     zero order hold, use average of start and end
                       input values for time step
            'foh'      first order hold, input ramps linearly across
                       time step
            'tustin'   uses the Tustin (or bilinear) transform
            ========   ==============================================

        prewarp : scalar or None, optional
            Prewarp frequency used for the Tustin transform (rad/sec)

        Returns
        -------
        cls : instance of :class:`SSModel`
            Has attributes set for a discrete (z-plane) state-space
            model.

        Notes
        -----
        If `self` is already a discrete model, this routine quietly
        returns itself. To convert to a different time-step, call
        :func:`d2c` first.

        The conversion is based on the matrix exponential (see
        :func:`pyyeti.expmint.expmint`). Letting `z` and `s` represent
        the discrete and continuous versions of the :class:`SSModel`,
        the following equations show the conversion methods. First
        (noting that `E`, `I1`, and `I2` can be calculated by
        :func:`pyyeti.expmint.expmint`)::

            lambda, phi = eig(z.A)
            E = exp(s.A*h)
            I1 = integral (exp(s.A*t) dt) from 0 to h
            I2 = integral (exp(s.A*t)*t dt) from 0 to h
            I = identity

        For ``method='zoh'``, the conversion is::

            z.A = E
            z.B = I1*s.B
            z.C = s.C
            z.D = s.D

        For ``method='zoha'``, the conversion is::

            z.A = E
            P = I1/2
            z.B = (P+E*P)*s.B
            z.C = s.C
            z.D = s.C*P*s.B + s.D

        For ``method='foh'``, the conversion is::

            z.A = E
            Q = I1 - I2/h
            P = I1 - Q
            z.B = (P+E*Q)*s.B
            z.C = s.C
            z.D = s.C*Q*s.B + s.D

        For ``method='tustin'``, the conversion is::

            if prewarp == 0:
                k = 2/h     # standard Tustin method
            else:
                k = prewarp/tan(prewarp*h/2)
            Q = inv(k*I-s.A)
            z.A = Q*(k*I+s.A)
            z.B = (I+s.A)*Q*s.B
            z.C = s.C
            z.D = s.C*Q*s.B + s.D

        """
        if self.h:
            return self
            # raise ValueError('model is already discrete??')

        if method == "zoh":
            A, B, Q = expmint.getEPQ(self.A, h, 0, B=self.B)
            C = self.C.copy()
            D = self.D.copy()
            return SSModel(A, B, C, D, h, method)

        if method == "zoha":
            A, P, Q = expmint.getEPQ(self.A, h, 0, B=self.B)
            P /= 2.0
            Q = P
            B = P + A.dot(Q)
            C = self.C.copy()
            D = self.C.dot(Q) + self.D
            return SSModel(A, B, C, D, h, method)

        if method == "foh":
            A, P, Q = expmint.getEPQ(self.A, h, 1, B=self.B)
            B = P + A.dot(Q)
            C = self.C.copy()
            D = self.C.dot(Q) + self.D
            return SSModel(A, B, C, D, h, method)

        if method == "tustin":
            if prewarp is None or prewarp == 0:
                k = 2 / h
            else:
                k = prewarp / np.tan(prewarp * h / 2)
            I = np.eye(self.A.shape[0])
            q = la.lu_factor(k * I - self.A)
            A = la.lu_solve(q, k * I + self.A)
            QB = la.lu_solve(q, self.B)
            B = (I + A).dot(QB)
            C = self.C.copy()
            D = self.C.dot(QB) + self.D
            return SSModel(A, B, C, D, h, method, prewarp)

        raise ValueError("invalid `method` argument")
