# -*- coding: utf-8 -*-
from types import SimpleNamespace
import warnings
import scipy.linalg as la
import numpy as np
from pyyeti import ytools


class _BaseODE:
    """
    Base class for time and frequency domain equations of motion
    solvers.

    This class is abstract-like, but not a subclass of abc.ABC because
    I don't want to require that all subclasses implement all the
    "abstract" methods. Raising NotImplementedError seems appropriate
    here.
    """

    def tsolve(self):
        """'Abstract' method to solve time-domain equations"""
        raise NotImplementedError

    def fsolve(self):
        """'Abstract' method to solve frequency-domain equations"""
        raise NotImplementedError

    def generator(self):
        """
        'Abstract' method to get Python "generator" version of the ODE
        solver. This is to interactively solve (or re-solve) one step
        at a time.
        """
        raise NotImplementedError

    def get_f2x(self):
        """
        'Abstract' method to get the force to displacement transform
        used in Henkel-Mar simulations.
        """
        raise NotImplementedError

    def finalize(self, get_force=False):
        """
        Finalize time-domain generator solution.

        Parameters
        ----------
        get_force : bool; optional
            If True, the `force` entry will be included in the
            returned data structure.

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
            Time-step or None
        t : 1d ndarray
            Time vector: np.arange(d.shape[1])*h
        force : 2d ndarray; optional
            Force; ndof x time. Only included if `get_force` is True.
        """
        d, v, a, f = self._d, self._v, self._a, self._force
        del self._d, self._v, self._a, self._force
        self._calc_acce_kdof(d, v, a, f)
        sol = self._solution(d, v, a)
        if get_force:
            sol.force = f
        return sol

    #
    # Utility routines follow:
    #
    def _solution(self, d, v, a):
        """Returns SimpleNamespace object with d, v, a, h, t"""
        if self.h:
            t = self.h * np.arange(a.shape[1])
        else:
            t = np.array([0.0])
        if self.pre_eig:
            d = self.phi @ d
            v = self.phi @ v
            a = self.phi @ a
        return SimpleNamespace(d=d, v=v, a=a, h=self.h, t=t)

    def _solution_freq(self, d, v, a, freq):
        """Returns SimpleNamespace object with d, v, a, freq"""
        if self.pre_eig:
            d = self.phi @ d
            v = self.phi @ v
            a = self.phi @ a
        return SimpleNamespace(d=d, v=v, a=a, f=freq)

    def _mk_slice(self, pv):
        """Convert index partition vector to slice object:
        ``start:stop``. Raises ValueError if `pv` cannot be converted
        to this type of slice object."""
        if pv.size == 0:
            return slice(0, 0)
        # if pv.size == pv[-1]+1 - pv[0]:
        if np.all(np.diff(pv) == 1):
            return slice(pv[0], pv[-1] + 1)
        raise ValueError("invalid partition vector for conversion to slice")

    def _mk_slices(self):  # , dorbel):
        """Convert index partition vectors to slice objects and sets
        ``slices=True`` if successful."""
        # if not dorbel:
        #     print("Setting dorbel to True for testing...")
        #     dorbel = True
        try:
            nonrf = self._mk_slice(self.nonrf)
            rf = self._mk_slice(self.rf)
            kdof = self._mk_slice(self.kdof)
            # if dorbel:
            rb = self._mk_slice(self.rb)
            el = self._mk_slice(self.el)
            _rb = self._mk_slice(self._rb)
            _el = self._mk_slice(self._el)
        except ValueError:
            self.slices = False
        else:
            self.nonrf = nonrf
            self.rf = rf
            self.kdof = kdof
            # if dorbel:
            self.rb = rb
            self.el = el
            self._rb = _rb
            self._el = _el
            self.slices = True

    def _chk_diag_part(self, m, b, k, cd_as_force=False):
        """Checks for all-diagonal and partitions based on rf modes"""
        krf = None
        cdforces = False

        unc = 0
        if m is None or m.ndim == 1 or (m.ndim == 2 and ytools.isdiag(m)):
            unc += 1

        if b.ndim == 1 or (b.ndim == 2 and ytools.isdiag(b)):
            unc += 1
        elif cd_as_force:
            unc += 1
            cdforces = True

        if k.ndim == 1 or (k.ndim == 2 and ytools.isdiag(k)):
            unc += 1

        if unc == 3:
            unc = True
            if m is not None and m.ndim == 2:
                m = np.diag(m).copy()
            if b.ndim == 2:
                bd = np.diag(b).copy()
                if cdforces:
                    bo = b.copy()
                    i = np.arange(bo.shape[0])
                    bo[i, i] = 0.0  # off diagonal damping
                b = bd
            if k.ndim == 2:
                k = np.diag(k).copy()
            if self.rfsize:
                if m is not None:
                    m = m[self.nonrf]
                b = b[self.nonrf]
                if cdforces:
                    bo = bo[np.ix_(self.nonrf, self.nonrf)]
                krf = k[self.rf]
                k = k[self.nonrf]
        else:
            cdforces = False
            unc = False
            if m is not None and m.ndim == 1:
                m = np.diag(m)
            if b.ndim == 1:
                b = np.diag(b)
            if k.ndim == 1:
                k = np.diag(k)
            if self.rfsize:
                pvrf = np.ix_(self.rf, self.rf)
                pvnonrf = np.ix_(self.nonrf, self.nonrf)
                if m is not None:
                    m = m[pvnonrf]
                b = b[pvnonrf]
                krf = k[pvrf]
                k = k[pvnonrf]
        self.m = m
        self.b = b
        self.cdforces = cdforces
        if cdforces:
            self.bo = bo
        self.k = k
        self.krf = krf
        self.unc = unc

    def _inv_krf(self):
        """Decompose the krf matrix"""
        if self.rfsize:
            krf = self.krf
            if self.unc:
                ikrf = (1.0 / krf).reshape(-1, 1)
                c = abs(krf).max() / abs(krf).min()
            else:
                ikrf = la.lu_factor(krf)
                c = np.linalg.cond(krf)
            if c > 1 / np.finfo(float).eps:
                msg = (
                    "the residual-flexibility part of the stiffness is poorly "
                    f"conditioned (cond={c:.3e}). Displacements will likely "
                    "be inaccurate."
                )
                warnings.warn(msg, RuntimeWarning)
            self.ikrf = ikrf

    def _get_inv_m(self, m):
        """Decompose the mass matrix"""
        if self.unc:
            invm = (1.0 / m).reshape(-1, 1)
            c = abs(m).max() / abs(m).min()
        else:
            invm = la.lu_factor(m)
            c = np.linalg.cond(m)
        if c > 1 / np.finfo(float).eps:
            msg = (
                f"the mass matrix is poorly conditioned (cond={c:.3e}). Solution "
                "will likely be inaccurate."
            )
            warnings.warn(msg, RuntimeWarning)
        return invm

    def _inv_m(self):
        """Decompose the mass matrix"""
        if self.m is not None and self.ksize:
            self.invm = self._get_inv_m(self.m)

    def _inv_mrb(self):
        """Decompose the rigid-body part of the mass matrix"""
        if self.m is not None and self.rbsize:
            if self.unc:
                mrb = self.m[self.rb]
            else:
                mrb = self.m[np.ix_(self.rb, self.rb)]
            self.imrb = self._get_inv_m(mrb)

    def _assert_square(self, n, m, b, k):
        if m is not None:
            name = ("mass", "damping", "stiffness")
            mats = (m, b, k)
        else:
            name = ("damping", "stiffness")
            mats = (b, k)
        any_2d = False
        for i, mat in enumerate(mats):
            if mat.ndim == 2:
                any_2d = True
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError(f"{name[i]} matrix is non-square!")
                if mat.shape[0] != n:
                    raise ValueError(
                        f"{name[i]} matrix has a different number of rows than "
                        "the stiffness!"
                    )
            elif mat.ndim == 1:
                if mat.shape[0] != n:
                    raise ValueError(
                        f"length of {name[i]} diagonal is not compatible with "
                        "the stiffness!"
                    )
            else:
                raise ValueError(f"{name[i]} has more than 2 dimensions!")
        return any_2d

    def _do_pre_eig(self, m, b, k):
        """Do a "pre" eigensolution to put system in modal space"""
        if k.ndim == 1:
            k = np.diag(k)
        else:
            ktype, types = ytools.mattype(k)
            if not ((ktype & types["symmetric"]) or (ktype & types["hermitian"])):
                raise la.LinAlgError(
                    "stiffness matrix must be symmetric or hermitian for the "
                    "`pre_eig` option."
                )
        if m is None:
            w, u = la.eigh(k)
        else:
            if m.ndim == 1:
                m = np.diag(m)
            w, u = la.eigh(k, m)
        self.pre_eig = True
        self.phi = u
        m = None
        k = w
        if b.ndim == 1:
            b = (u.T * b) @ u
        else:
            b = u.T @ b @ u
        return m, b, k

    def _ensure_index_type(self, pv):
        pv = np.atleast_1d(pv)
        if np.issubdtype(pv.dtype, np.bool_):
            return np.nonzero(pv)[0]
        return pv

    def _common_precalcs(self, m, b, k, h, rb, rf, pre_eig=False, cd_as_force=False):
        systype = float
        self.mid = id(m)
        self.bid = id(b)
        self.kid = id(k)
        self.m_orig = m
        self.b_orig = b
        self.k_orig = k

        if m is None:
            b, k = np.atleast_1d(b, k)
            if np.iscomplexobj(b) or np.iscomplexobj(k):
                systype = complex
        else:
            m, b, k = np.atleast_1d(m, b, k)
            if np.iscomplexobj(m) or np.iscomplexobj(b) or np.iscomplexobj(k):
                systype = complex
        n = k.shape[0]
        any_2d = self._assert_square(n, m, b, k)
        if pre_eig and any_2d:
            m, b, k = self._do_pre_eig(m, b, k)
        else:
            self.pre_eig = False
        nonrf = np.ones(n, bool)
        if rf is None or (isinstance(rf, list) and not rf):
            rf = np.array([], bool)
        else:
            # # rf = np.ix_(np.atleast_1d(rf))[0]
            # rf = np.atleast_1d(rf)
            rf = self._ensure_index_type(rf)
        nonrf[rf] = False
        nonrf = np.nonzero(nonrf)[0]
        self.n = n
        self.h = h
        self.rf = rf
        self.nonrf = nonrf
        self.rfsize = rf.size
        self.nonrfsz = nonrf.size
        self.kdof = nonrf
        self.ksize = nonrf.size
        self._chk_diag_part(m, b, k, cd_as_force)
        self._make_rb_el(rb)
        self._inv_krf()
        self.systype = systype

    def _make_rb_el(self, rb):
        """
        - rb and el are relative to full size
        - _rb and _el are relative to the non RF part
        """
        if rb is None:
            if self.ksize:
                tol = 0.005
                if self.unc:
                    _rb = np.nonzero(abs(self.k) < tol)[0]
                else:
                    _rb = (
                        (abs(self.k).max(axis=0) < tol)
                        & (abs(self.k).max(axis=1) < tol)
                        & (abs(self.b).max(axis=0) < tol)
                        & (abs(self.b).max(axis=1) < tol)
                    ).nonzero()[0]
                rb = np.zeros(self.n, bool)
                rb[self.nonrf[_rb]] = True
                rb = np.nonzero(rb)[0]
            else:
                rb = _rb = np.array([], bool)
        elif isinstance(rb, list) and not rb:
            rb = _rb = np.array([], bool)
        else:
            # # rb = np.ix_(np.atleast_1d(rb))[0]
            # rb = np.atleast_1d(rb)
            rb = self._ensure_index_type(rb)
            vec = np.zeros(self.n, bool)
            vec[rb] = True
            _rb = np.nonzero(vec[self.nonrf])[0]
        _el = np.ones(self.ksize, bool)
        _el[_rb] = False
        _el = np.nonzero(_el)[0]
        el = np.zeros(self.n, bool)
        el[self.nonrf[_el]] = True
        el = np.nonzero(el)[0]
        # _rb, _el are relative to non-rf part
        self.rb = rb
        self._rb = _rb
        self.el = el
        self._el = _el
        self.rbsize = rb.size
        self.elsize = el.size

    def _build_A(self):
        """Builds the A matrix: yd - A y = f"""
        n = self.k.shape[0]
        A = np.zeros((2 * n, 2 * n), self.systype)
        v1 = range(n)
        v2 = range(n, 2 * n)
        if self.unc:
            A[v1, v1] = -self.b
            A[v1, v2] = -self.k
        else:
            A[:n, :n] = -self.b
            A[:n, n:] = -self.k
        A[v2, v1] = 1.0
        if self.m is not None:
            if self.unc:
                A[:n] *= self.invm
            else:
                A[:n] = la.lu_solve(self.invm, A[:n], check_finite=False)
        return A

    def _alloc_dva(self, nt, istime):
        ORDER = "F"
        n = self.ksize
        if istime:
            if nt > 1 and n > 0 and not self.pc:
                raise RuntimeError(
                    f"rerun `{type(self).__name__}` with a valid time step."
                )
            d = np.zeros((self.n, nt), self.systype, order=ORDER)
            v = np.zeros((self.n, nt), self.systype, order=ORDER)
            a = np.zeros((self.n, nt), self.systype, order=ORDER)
        else:
            d = np.zeros((self.n, nt), complex, order=ORDER)
            v = np.zeros((self.n, nt), complex, order=ORDER)
            a = np.zeros((self.n, nt), complex, order=ORDER)
        return d, v, a

    def _init_dv(self, d, v, d0, v0, F0, static_ic):
        if d0 is not None:
            d[self.nonrf, 0] = d0[self.nonrf]
        elif static_ic and self.elsize and F0[self.el].any():
            # el is relative to full size
            # _el is relative to "k" size
            if self.unc:
                d[self.el, 0] = F0[self.el] / self.k[self._el]
            else:
                if self.slices:
                    k = self.k[self._el, self._el]
                else:
                    k = self.k[np.ix_(self._el, self._el)]
                d[self.rb, 0] = 0.0
                d[self.el, 0] = np.linalg.solve(k, F0[self.el])
        if v0 is not None:
            v[self.nonrf, 0] = v0[self.nonrf]

    def _set_initial_cond(self, d0, v0):
        d0 = None if d0 is None else np.atleast_1d(d0)
        v0 = None if v0 is None else np.atleast_1d(v0)
        return d0, v0

    def _init_dva_part(self, nt, F0, d0, v0, static_ic, istime=True):
        if F0.shape[0] != self.n:
            raise ValueError(
                f"Initial force vector has {F0.shape[0]} elements;"
                f" {self.n} elements are expected"
            )
        if self.pre_eig:
            raise NotImplementedError(
                f"{type(self).__name__} generator not yet implemented "
                "using the `pre_eig` option"
            )

        d0, v0 = self._set_initial_cond(d0, v0)
        d, v, a = self._alloc_dva(nt, istime)
        f = np.copy(a)  # not a.copy because of `order` default
        f[:, 0] = F0
        self._init_dv(d, v, d0, v0, F0, static_ic)
        if self.rfsize:
            if self.unc:
                d[self.rf, 0] = self.ikrf.ravel() * F0[self.rf]
            else:
                d[self.rf, 0] = la.lu_solve(self.ikrf, F0[self.rf], check_finite=False)
        return d, v, a, f

    def _init_dva(
        self, force, d0, v0, static_ic, istime=True, freq=None, rf_disp_only=False
    ):
        if force.shape[0] != self.n:
            raise ValueError(
                f"Force matrix has {force.shape[0]} rows; {self.n} rows are expected"
            )

        d0, v0 = self._set_initial_cond(d0, v0)
        d, v, a = self._alloc_dva(force.shape[1], istime)

        if self.pre_eig:
            force = self.phi.T @ force

        self._init_dv(d, v, d0, v0, force[:, 0], static_ic)

        if self.rfsize:
            rf = self.rf
            if self.unc:
                d[rf] = self.ikrf * force[rf]
            else:
                d[rf] = la.lu_solve(self.ikrf, force[rf], check_finite=False)
            if not istime and not rf_disp_only:
                freqw = 2 * np.pi * freq
                freqw2 = freqw ** 2
                a[rf] = d[rf] * -(freqw2)
                v[rf] = d[rf] * (1j * freqw)

        return d, v, a, force

    def _calc_acce_kdof(self, d, v, a, force):
        """Calculate the `kdof` part of the acceleration"""
        if self.ksize:
            kdof = self.kdof
            F = force[kdof]
            if self.unc:
                if self.cdforces:
                    b = self.bo.copy()
                    i = np.arange(self.ksize)
                    b[i, i] = self.b  # full damping
                    B = b @ v[kdof]
                else:
                    B = self.b[:, None] * v[kdof]
                K = self.k[:, None] * d[kdof]
                if self.m is not None:
                    a[kdof] = self.invm * (F - B - K)
                else:
                    a[kdof] = F - B - K
            else:
                B = self.b @ v[kdof]
                K = self.k @ d[kdof]
                if self.m is not None:
                    a[kdof] = la.lu_solve(self.invm, F - B - K, check_finite=False)
                else:
                    a[kdof] = F - B - K

    def _add_rf_flex(self, flex, phi, velo, unc):
        if not velo and self.rfsize:
            rf = self.rf
            ikrf = self.ikrf
            phirf = phi[:, rf]
            if unc:
                flexrf = ikrf.ravel()[:, None] * phirf.T
            else:
                flexrf = la.lu_solve(ikrf, phirf.T, check_finite=False)
            flex = flex + phirf @ flexrf
        return flex

    def _flex(self, flex, phi):
        if isinstance(flex, float):
            n = phi.shape[0]
            flex = np.zeros((n, n))
        return flex

    def _force_freq_compat_chk(self, force, freq):
        """Check compatibility between force matrix and freq vector"""
        if force.shape[1] != len(freq):
            raise ValueError(
                f"Number of columns `force` ({force.shape[1]}) does "
                f"not equal length of `freq` ({len(freq)})"
            )
