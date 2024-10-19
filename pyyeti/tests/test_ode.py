import warnings
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
import scipy.signal
from scipy.interpolate import interp1d

from scipy import integrate
from pyyeti import ode, dsp
from pyyeti.ssmodel import SSModel
from pyyeti import expmint
from pyyeti.nastran import n2p, op2

import pytest


def test_expmint():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [0.0001, 0.001, 0.01, 0.05, 0.1]:
        e, i, i2 = expmint.expmint(A, h, geti2=True)
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)

    # these last 2 use the power series expansion for I2 ... check
    # for the warning
    for h in [0.2, 1]:
        with pytest.warns(RuntimeWarning, match="Using power series expansion"):
            e, i, i2 = expmint.expmint(A, h, geti2=True)
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)


def test_expmint2():
    A = np.random.randn(50, 50)
    for h in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1]:
        e, i, i2 = expmint.expmint(A, h, geti2=True)
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)


def test_getEPQ1_2_order0():
    order = 0
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1]:
        e, p, q = expmint.getEPQ1(A, h, order=order)
        et, pt, qt = expmint.getEPQ2(A, h, order=order)
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)


def test_getEPQ1_2_order1():
    order = 1
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [0.0001, 0.001, 0.01, 0.05, 0.1]:
        e, p, q = expmint.getEPQ1(A, h, order=order)
        et, pt, qt = expmint.getEPQ2(A, h, order=order)
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)

    # these last 2 use the power series expansion for I2 ... check
    # for the warning
    for h in [0.2, 1]:
        e, p, q = expmint.getEPQ2(A, h, order=order)
        with pytest.warns(RuntimeWarning, match="Using power series expansion"):
            et, pt, qt = expmint.getEPQ1(A, h, order=order)
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)


def test_getEPQ1():
    for order in (0, 1):
        A = np.random.randn(50, 50)
        for h in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1]:
            e, p, q = expmint.getEPQ1(A, h, order=order)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def test_getEPQ():
    for order in (0, 1):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        for h in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1]:
            e, p, q = expmint.getEPQ(A, h, order=order)
            et, pt, qt = expmint.getEPQ2(A, h, order=order)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def test_getEPQ_half():
    for order in (0, 1):
        A = np.random.randn(50, 50)
        for h in [0.001]:
            e, p, q = expmint.getEPQ(A, h, order=order, half=True)
            e1, p1, q1 = expmint.getEPQ1(A, h, order=order, half=True)
            e2, p2, q2 = expmint.getEPQ2(A, h, order=order, half=True)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order, half=True)
            assert np.allclose(e, e1)
            assert np.allclose(p, p1)
            assert np.allclose(q, q1)
            assert np.allclose(e, e2)
            assert np.allclose(p, p2)
            assert np.allclose(q, q2)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def test_getEPQ_B():
    for order in (0, 1):
        A = np.random.randn(50, 50)
        B = np.random.randn(50, 2)
        for h in [0.001]:
            e, p, q = expmint.getEPQ(A, h, order=order, B=B)
            e1, p1, q1 = expmint.getEPQ1(A, h, order=order, B=B)
            e2, p2, q2 = expmint.getEPQ2(A, h, order=order, B=B)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order, B=B)
            assert np.allclose(e, e1)
            assert np.allclose(p, p1)
            assert np.allclose(q, q1)
            assert np.allclose(e, e2)
            assert np.allclose(p, p2)
            assert np.allclose(q, q2)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def check_true_derivatives(sol, tol=5e-3):
    d = integrate.cumulative_trapezoid(sol.v, sol.t, initial=0, axis=1)
    v = integrate.cumulative_trapezoid(sol.a, sol.t, initial=0, axis=1)
    derr = abs((d + sol.d[:, :1]) - sol.d).max() / abs(sol.d).max()
    verr = abs((v + sol.v[:, :1]) - sol.v).max() / abs(sol.v).max()
    assert derr < tol
    assert verr < tol


def test_ode_ic():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 1.2])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    b[0] = 2 * b[1]  # add damping on rb modes for test

    h = 0.0002  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    d0 = np.random.randn(4)
    v0 = np.random.randn(4)

    se2 = ode.SolveExp2(m, b, k, h)
    su = ode.SolveUnc(m, b, k, h)
    sole = se2.tsolve(f, d0, v0)
    solu = su.tsolve(f, d0, v0)

    assert np.allclose(sole.a, solu.a)
    assert np.allclose(sole.v, solu.v)
    assert np.allclose(sole.d, solu.d)
    assert np.allclose(v0, solu.v[:, 0])
    assert np.allclose(d0, solu.d[:, 0])
    assert np.allclose(v0, sole.v[:, 0])
    assert np.allclose(d0, sole.d[:, 0])

    check_true_derivatives(solu)
    check_true_derivatives(sole)

    fru = m[:, None] * solu.a + b[:, None] * solu.v + k[:, None] * solu.d
    fre = m[:, None] * sole.a + b[:, None] * sole.v + k[:, None] * sole.d

    assert np.allclose(f, fru)
    assert np.allclose(f, fre)

    # plt.clf()
    # for i, r in enumerate('avd'):
    #     plt.subplot(3, 1, i+1)
    #     plt.plot(sole.t, getattr(sole, r).T, '-',
    #              sole.t, getattr(solu, r).T, '--')


def getpdiff(x, y, j):
    den = abs(y[:, j:]).max(axis=1)
    den[den == 0.0] = 1.0
    return 100 * abs(x[:, j:] - y[:, j:]).max(axis=1) / den


def test_newmark_diag():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diag mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diag stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # % damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diag damping
    h = 0.0005  # time step
    t = np.arange(0, 0.2, h)  # time vector
    c = 2 * np.pi
    f = np.vstack(
        (
            3 * (1 - np.cos(c * 2 * t)),  # ffn
            4.5 * (1 - np.cos(np.sqrt(k[1] / m[1]) * t)),
            4.5 * (1 - np.cos(np.sqrt(k[2] / m[2]) * t)),
            4.5 * (1 - np.cos(np.sqrt(k[3] / m[3]) * t)),
        )
    )
    f *= 1.0e4
    t2 = 2 / (np.sqrt(k[1] / m[1]) / 2 / np.pi)
    f[1:, t > t2] = 0.0

    d0 = np.array([1, 2, 3, 4]) / 100
    v0 = np.array([2, 1, 4, 5]) / 10

    su = ode.SolveUnc(m, b, k, h)
    solu = su.tsolve(f, d0=d0, v0=v0)

    nb = ode.SolveNewmark(m, b, k, h)
    soln = nb.tsolve(f, d0=d0, v0=v0)

    assert np.all(getpdiff(soln.d, solu.d, 0) < 5.0)
    assert np.all(getpdiff(soln.v, solu.v, 5) < 5.0)
    assert np.all(getpdiff(soln.a, solu.a, 10) < 5.0)

    assert np.all(soln.d[:, 0] == d0)
    assert np.all(soln.v[:, 0] == v0)

    assert np.all(solu.d[:, 0] == d0)
    assert np.all(solu.v[:, 0] == v0)

    # with rf modes:
    su = ode.SolveUnc(m, b, k, h, rf=3)
    solu = su.tsolve(f, d0=d0, v0=v0)

    nb = ode.SolveNewmark(m, b, k, h, rf=3)
    soln = nb.tsolve(f, d0=d0, v0=v0)

    se = ode.SolveExp2(m, b, k, h, rf=3)
    sole = se.tsolve(f, d0=d0, v0=v0)

    assert np.all(getpdiff(soln.d, solu.d, 0) < 5.0)
    assert np.all(getpdiff(soln.v[:3], solu.v[:3], 5) < 5.0)
    assert np.all(getpdiff(soln.a[:3], solu.a[:3], 10) < 5.0)

    assert np.all(soln.d[:3, 0] == d0[:3])
    assert np.all(soln.v[:3, 0] == v0[:3])

    assert np.all(solu.d[:3, 0] == d0[:3])
    assert np.all(solu.v[:3, 0] == v0[:3])

    assert np.all(soln.a[3] == 0.0)
    assert np.all(soln.v[3] == 0.0)
    assert np.allclose(soln.d[3], f[3] / k[3])

    assert np.all(solu.a[3] == 0.0)
    assert np.all(solu.v[3] == 0.0)
    assert np.allclose(solu.d[3], f[3] / k[3])

    assert np.allclose(sole.d, solu.d)
    assert np.allclose(sole.v, solu.v)
    assert np.allclose(sole.a, solu.a)

    with pytest.raises(ValueError):
        nb.tsolve(np.zeros((m.shape[0] + 1, 1)))


def test_ode_newmark_uncoupled_mNone():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    m1 = np.ones(4)

    for _ in range(2):
        m1 = np.diag(m1)
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            ts = ode.SolveNewmark(m1, b, k, h, rf=rf)
            sol = ts.tsolve(f)

            tsn = ode.SolveNewmark(m, b, k, h, rf=rf)
            soln = tsn.tsolve(f)

            assert np.allclose(sol.a, soln.a)
            assert np.allclose(sol.v, soln.v)
            assert np.allclose(sol.d, soln.d)


def test_ode_newmark_coupled_mNone():
    # coupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    m1 = np.eye(4)

    for rf in (None, 3, 2, np.array([1, 2, 3])):
        ts = ode.SolveNewmark(m1, b, k, h, rf=rf)
        sol = ts.tsolve(f)

        tsn = ode.SolveNewmark(m, b, k, h, rf=rf)
        soln = tsn.tsolve(f)

        assert np.allclose(sol.a, soln.a)
        assert np.allclose(sol.v, soln.v)
        assert np.allclose(sol.d, soln.d)


def test_ode_newmark_coupled_2_mNone():
    # coupled equations
    m = None
    k = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k += np.random.randn(4, 4) * 1000
    b += np.random.randn(4, 4)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    m1 = np.eye(4)

    for rf in (None, 3, 2, np.array([1, 2, 3])):
        ts = ode.SolveNewmark(m1, b, k, h, rf=rf)
        sol = ts.tsolve(f)

        tsn = ode.SolveNewmark(m, b, k, h, rf=rf)
        soln = tsn.tsolve(f)

        assert np.allclose(sol.a, soln.a)
        assert np.allclose(sol.v, soln.v)
        assert np.allclose(sol.d, soln.d)


def test_rbdamped_modes_coupled():
    N = 10
    win = 100
    np.random.seed(1)
    m = np.random.randn(N, N)
    m = m.T @ m
    k = np.zeros(N)
    h = 0.0005
    t = np.arange(int(1 / h)) * h
    f = np.zeros((N, len(t)))
    f[:, :win] = np.ones((N, 1)) * np.hanning(win) * 20

    for i, b in enumerate((np.zeros(N), 10 * np.ones(N))):
        se2 = ode.SolveExp2(m, b, k, h)
        sole = se2.tsolve(f)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            su = ode.SolveUnc(m, b, k, h)  # , rb=[])
            if i == 1:
                assert len(records) == 1
                assert issubclass(records[0].category, RuntimeWarning)
                assert "invalid value" in str(records[0].message)
            else:
                assert len(records) == 0
        solu = su.tsolve(f)

        assert np.allclose(sole.a, solu.a)
        assert np.allclose(sole.v, solu.v)
        assert np.allclose(sole.d, solu.d)
        check_true_derivatives(solu)
        check_true_derivatives(sole)

        fru = m @ solu.a + b[:, None] * solu.v + k[:, None] * solu.d
        fre = m @ sole.a + b[:, None] * sole.v + k[:, None] * sole.d

        assert np.allclose(f, fru)
        assert np.allclose(f, fre)

    with pytest.warns(
        RuntimeWarning,
        match="eigenvectors for the state-space formulation are poorly conditioned",
    ):
        ode.SolveUnc(m, b * 0, k, h, rb=[])


def test_newmark_rbdamp_coupled():
    N = 10
    win = 100
    np.random.seed(1)
    m = np.random.randn(N, N)
    m = m.T @ m
    k = np.zeros(N)
    h = 0.0005
    t = np.arange(int(1 / h)) * h
    f = np.zeros((N, len(t)))
    f[:, :win] = np.ones((N, 1)) * np.hanning(win) * 20

    for d0, v0 in ((None, np.random.randn(N)), (np.random.randn(N), None)):
        for i, b in enumerate((np.zeros(N), 10 * np.ones(N))):
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                su = ode.SolveUnc(m, b, k, h)
                if i == 1:
                    assert len(records) == 1
                    assert issubclass(records[0].category, RuntimeWarning)
                    assert "invalid value" in str(records[0].message)
                else:
                    assert len(records) == 0
            solu = su.tsolve(f, d0=d0, v0=v0)

            nb = ode.SolveNewmark(m, b, k, h)
            soln = nb.tsolve(f, d0=d0, v0=v0)

            assert np.all(getpdiff(soln.d, solu.d, 0) < 5.0)
            assert np.all(getpdiff(soln.v, solu.v, 0) < 5.0)
            assert np.all(getpdiff(soln.a, solu.a, 10) < 5.0)

            d0_ = 0.0 if d0 is None else d0
            v0_ = 0.0 if v0 is None else v0

            assert np.all(soln.d[:, 0] == d0_)
            assert np.all(soln.v[:, 0] == v0_)
            assert np.all(solu.d[:, 0] == d0_)
            assert np.all(solu.v[:, 0] == v0_)


def get_rfsol(k, rf, f):
    rf = np.atleast_1d(rf)
    if k.ndim > 1:
        if np.size(rf) > 1:
            krf = k[np.ix_(rf, rf)]
            rfsol = la.solve(krf, f[rf, :])
        else:
            krf = k[rf, rf]
            rfsol = f[rf, :] / krf
    else:
        krf = k[rf]
        if np.size(krf) > 1:
            rfsol = f[rf, :] / krf[:, None]
        else:
            rfsol = f[rf, :] / krf
    return rfsol.T


def test_ode_uncoupled():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(m)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                if m.ndim > 1:
                    f2 = la.solve(m, f)
                else:
                    f2 = (1 / m)[:, None] * f
                tl, yl, xl = scipy.signal.lsim(
                    (A, B, C, D), f2.T, t, X0=ic, interp=order
                )
                tse1 = ode.SolveExp1(A, h, order=order)
                if abs(ic).max() == 0:
                    sole1 = tse1.tsolve(B.dot(f2))
                else:
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2)
                assert np.allclose(yl, yl2.T)

                tse1 = ode.SolveExp1(A, h=None)
                sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                assert np.allclose(yl[:1, :], yl2.T)

                if rf is not None:
                    rfsol = get_rfsol(k, rf, f)
                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_uncoupled_2():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_ / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip(
            (None, None, 3, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0, 1.0)
        ):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(m)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                if m.ndim > 1:
                    f2 = la.solve(m, f)
                else:
                    f2 = (1 / m)[:, None] * f
                tl, yl, xl = scipy.signal.lsim(
                    (A, B, C, D), f2.T, t, X0=ic, interp=order
                )
                tse1 = ode.SolveExp1(A, h, order=order)
                sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2)
                assert np.allclose(yl, yl2.T)

                tse1 = ode.SolveExp1(A, h=None)
                sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                assert np.allclose(yl[:1, :], yl2.T)

                if rf is not None:
                    rfsol = get_rfsol(k, rf, f)
                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def decouple_rf(args, rf):
    out = []
    for v in args:
        if v.ndim > 1:
            vrf = v[rf, rf]
            v2 = v.copy()
            v2[rf] = 0.0
            v2[:, rf] = 0.0
            v2[rf, rf] = vrf
        else:
            v2 = v.copy()
        out.append(v2)
    return out


def test_ode_coupled():
    # coupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        # for rf in (None, 3, 2, np.array([1, 2, 3])):
        for rf in (
            None,
            [False, False, False, True],
            [False, False, True, False],
            np.array([False, True, True, True]),
        ):
            if (rf == np.array([False, False, True, False])).all() and k.ndim > 1:
                k = np.diag(k)
            if (rf == np.array([False, False, False, True])).all() and m.ndim > 1:
                m = np.diag(m)
                b = np.diag(b)
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                if rf is None:
                    A = ode.make_A(m, b, k)
                    n = len(m)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    if m.ndim == 1:
                        f2 = la.solve(np.diag(m), f)
                    else:
                        f2 = la.solve(m, f)
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)
                else:
                    rfsol = get_rfsol(k, rf, f)
                    m2, b2, k2 = decouple_rf((m, b, k), rf)
                    A = ode.make_A(m2, b2, k2)
                    n = len(m)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    if m2.ndim == 1:
                        f2 = la.solve(np.diag(m2), f)
                    else:
                        f2 = la.solve(m2, f)
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    rf2 = np.nonzero(rf)[0]
                    yl[:, rf2] = 0.0
                    yl[:, rf2 + n] = 0.0
                    yl[:, rf2 + 2 * n] = rfsol

                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_coupled_2():
    # coupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_ / m) * m  # diagonal of damping
    m = np.diag(m)
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    m += np.random.randn(4, 4)
    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf, kmult in zip(
            (None, None, 3, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0, 1.0)
        ):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                if rf is None:
                    A = ode.make_A(m, b, k)
                    n = len(m)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = la.solve(m, f)
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)
                else:
                    rfsol = get_rfsol(k, rf, f)
                    m2, b2, k2 = decouple_rf((m, b, k), rf)
                    A = ode.make_A(m2, b2, k2)
                    n = len(m)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = la.solve(m2, f)
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol

                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_newmark_coupled():
    # coupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_ / m) * m  # diagonal of damping
    m = np.diag(m)
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    m += np.random.randn(4, 4)
    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = 0.0005  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    pv = t < 0.15
    f[:, pv] = dsp.windowends(f[:, pv], 0.1, "both")
    pv = t >= 0.15
    f[:, pv] = 0.0

    for rf in (None, 2, 3, np.array([0, 1, 2, 3])):
        k = k_
        b = b_
        tsu = ode.SolveUnc(m, b, k, h, rf=rf)
        solu = tsu.tsolve(f)

        tsn = ode.SolveNewmark(m, b, k, h, rf=rf)
        soln = tsn.tsolve(f)

        assert np.all(getpdiff(soln.d, solu.d, 0) < 5.0)
        assert np.all(getpdiff(soln.v, solu.v, 0) < 5.0)
        assert np.all(getpdiff(soln.a, solu.a, 0) < 5.0)


def test_ode_uncoupled_mNone():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1, 1, 0):
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(b)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                f2 = f
                tl, yl, xl = scipy.signal.lsim(
                    (A, B, C, D), f2.T, t, X0=ic, interp=order
                )
                tse1 = ode.SolveExp1(A, h, order=order)
                sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2)
                assert np.allclose(yl, yl2.T)

                tse1 = ode.SolveExp1(A, h=None)
                sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                assert np.allclose(yl[:1, :], yl2.T)

                if rf is not None:
                    rfsol = get_rfsol(k, rf, f)
                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol

                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_uncoupled_2_mNone():
    # uncoupled equations
    m = None
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_)  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1, 1, 0):
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip((None, None, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(b)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                f2 = f
                tl, yl, xl = scipy.signal.lsim(
                    (A, B, C, D), f2.T, t, X0=ic, interp=order
                )
                tse1 = ode.SolveExp1(A, h, order=order)
                sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2)
                assert np.allclose(yl, yl2.T)

                tse1 = ode.SolveExp1(A, h=None)
                sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                assert np.allclose(yl[:1, :], yl2.T)

                if rf is not None:
                    rfsol = get_rfsol(k, rf, f)
                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_coupled_mNone():
    # coupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                if rf is None:
                    A = ode.make_A(m, b, k)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)
                else:
                    rfsol = get_rfsol(k, rf, f)
                    b2, k2 = decouple_rf((b, k), rf)
                    A = ode.make_A(m, b2, k2)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol

                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_coupled_2_mNone():
    # coupled equations
    m = None
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_)  # diagonal of damping
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf, kmult in zip((None, None, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                if kmult != 0.0:
                    tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf, rb=[])
                else:
                    tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                if rf is None:
                    A = ode.make_A(m, b, k)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)
                else:
                    rfsol = get_rfsol(k, rf, f)
                    b2, k2 = decouple_rf((b, k), rf)
                    A = ode.make_A(m, b2, k2)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))

                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_coupled_mNone_rblast():
    # coupled equations
    m = None
    k = np.array([6.0e5, 6.0e5, 6.0e5, 0.0])  # diagonal of stiffness
    zeta = np.array([0.05, 1.0, 2.0, 0.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[:-1, :-1] += np.random.randn(3, 3) * 1000
    b[:-1, :-1] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
                3 * (1 - np.cos(c * 2 * t)),
            )
        )
        * 1.0e4
    )

    number_one = 1

    rb = 3
    for order in (0, 1):
        if order == 1:
            rb = [False, False, False, True]
        for rf in ([], 2, number_one, np.array([0, 1, 2])):
            if order == 1 and rf is number_one:
                k = np.diag(k)
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf, rb=rb)
                if tsu.ksize > 0:
                    assert tsu.pc.ur.shape[0] > tsu.pc.ur.shape[1]
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf, rb=rb)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                if rf is None or np.size(rf) == 0:
                    A = ode.make_A(m, b, k)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)
                else:
                    rfsol = get_rfsol(k, rf, f)
                    b2, k2 = decouple_rf((b, k), rf)
                    A = ode.make_A(m, b2, k2)
                    n = len(b)
                    Z = np.zeros((n, n), float)
                    B = np.vstack((np.eye(n), Z))
                    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                    D = np.vstack((B, Z))
                    ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                    f2 = f
                    tl, yl, xl = scipy.signal.lsim(
                        (A, B, C, D), f2.T, t, X0=ic, interp=order
                    )
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.0
                    yl[:, rf + n] = 0.0
                    yl[:, np.atleast_1d(rf + 2 * n)] = rfsol

                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n : 2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n :], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_SSModel_repr():
    s = SSModel(1, 2, 3, 4)
    assert repr(s) == (
        "SSModel(\nA=array([[1]]),\nB=array([[2]]),\n"
        "C=array([[3]]),\nD=array([[4]]),\nh=None,\n"
        "method=None,\nprewarp=None,\n)"
    )


def test_make_A():
    m = None
    b = np.random.randn(3, 3)
    k = np.random.randn(3)
    A1 = ode.make_A(m, b, k)
    A2 = ode.make_A(m, b + 0j, k)
    assert np.all(A1 == A2)

    m = np.eye(3)
    A1 = ode.make_A(m, b, k)
    A2 = ode.make_A(m, b, k + 0j)
    assert np.all(A1 == A2)


def test_se1():
    f = 5  # 5 hz oscillator
    w = 2 * np.pi * f
    w2 = w * w
    zeta = 0.05
    h = 0.01
    nt = 500
    A = np.array([[0, 1], [-w2, -2 * w * zeta]])
    B = np.array([[0], [3]])
    C = np.array([[8, -5]])
    D = np.array([[0]])
    F = np.zeros((1, nt), float)
    ts = ode.SolveExp1(A, h)
    sol = ts.tsolve(B.dot(F), B[:, 0])
    y = C.dot(sol.d)
    ssmodel = SSModel(A, B, C, D)
    z = ssmodel.c2d(h=h, method="zoh")
    x = np.zeros((A.shape[1], nt + 1), float)
    y2 = np.zeros((C.shape[0], nt), float)
    x[:, 0:1] = B
    for k in range(nt):
        x[:, k + 1] = z.A.dot(x[:, k]) + z.B.dot(F[:, k])
        y2[:, k] = z.C.dot(x[:, k]) + z.D.dot(F[:, k])
    assert np.allclose(y, y2)

    # compare against scipy:
    ss = ssmodel.getlti()
    tout, yout = ss.impulse(T=sol.t)
    assert np.allclose(yout, y.flatten())

    # another check of getlti:
    z = ssmodel.c2d(h)
    ss = z.getlti()
    tout, yout = ss.impulse(T=sol.t)
    assert np.allclose(yout, y.flatten())


def test_ode_init():
    def iseq(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, (int, float, np.ndarray)):
            if np.size(a) != np.size(b):
                return False
            return np.all(a == b)
        if isinstance(a, SimpleNamespace):
            return comp_class(a, b)
        return True

    def comp_class(a, b):
        if dir(a) != dir(b):
            return False
        for key in a.__dict__.keys():
            if not iseq(a.__dict__[key], b.__dict__[key]):
                return False
        return True

    m = np.random.randn(4, 4)
    b = np.random.randn(4, 4)
    k = np.random.randn(4, 4)
    h = 0.01
    A = np.random.randn(4, 4)

    ts1 = ode.SolveExp1(A, h, 0)
    ts2 = ode.SolveExp1(A, h, 1)
    assert not comp_class(ts1, ts2)

    ts1 = ode.SolveExp2(m, b, k, h, order=1, rf=3)
    ts2 = ode.SolveExp2(m, b, k, h, order=1)
    assert not comp_class(ts1, ts2)

    ts1 = ode.SolveUnc(m, b, k, h, order=0, rf=1, rb=2)
    ts2 = ode.SolveUnc(m, b, k, h, order=0, rf=1, rb=1)
    assert not comp_class(ts1, ts2)


def runsim(ss_sysz, sysz):
    r, c = ss_sysz.B.shape
    yr = ss_sysz.C.shape[0]
    nt = 301
    u = np.random.randn(c, nt)
    x1 = np.zeros((r, nt + 1), float)
    y1 = np.zeros((yr, nt), float)
    x2 = x1.copy()
    y2 = y1.copy()
    for j in range(nt):
        uj = u[:, j]
        x1[:, j + 1] = ss_sysz.A.dot(x1[:, j]) + ss_sysz.B.dot(uj)
        y1[:, j] = ss_sysz.C.dot(x1[:, j]) + ss_sysz.D.dot(uj)
        x2[:, j + 1] = sysz[0].dot(x2[:, j]) + sysz[1].dot(uj)
        y2[:, j] = sysz[2].dot(x2[:, j]) + sysz[3].dot(uj)
    return np.allclose(y1, y2)


def chk_inverse(s1, old):
    s2 = s1.d2c(method=s1.method, prewarp=s1.prewarp)
    assert np.allclose(old.A, s2.A)
    assert np.allclose(old.B, s2.B)
    assert np.allclose(old.C, s2.C)
    assert np.allclose(old.D, s2.D)


def test_SSModel_c2d_d2c():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m += np.random.randn(4, 4)
    k += np.random.randn(4, 4) * 1000
    b += np.random.randn(4, 4)
    h = 0.001  # time step

    A = ode.make_A(m, b, k)
    n = len(m)
    Z = np.zeros((n, n), float)
    B = np.vstack((np.eye(n), Z))
    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
    D = np.vstack((B, Z))
    sys = (A, B, C, D)
    ss_sys = SSModel(*sys)

    sysz = scipy.signal.cont2discrete(sys, h, method="zoh")
    ss_sysz = ss_sys.c2d(h, method="zoh")
    # assert np.allclose(ss_sysz.C.dot(ss_sysz.A.dot(ss_sysz.B)),
    #                    sysz[2].dot(sysz[0].dot(sysz[1])))
    # assert np.allclose(ss_sysz.D, sysz[3])
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)

    sysz = scipy.signal.cont2discrete(sys, h, method="bilinear")
    ss_sysz = ss_sys.c2d(h, method="tustin")
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)


def test_zoha_c2d_d2c():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m += np.random.randn(4, 4)
    k += np.random.randn(4, 4) * 1000
    b += np.random.randn(4, 4)
    h = 0.001  # time step

    A = ode.make_A(m, b, k)
    n = len(m)
    Z = np.zeros((n, n), float)
    B = np.vstack((np.eye(n), Z))
    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
    D = np.vstack((B, Z))
    sys = (A, B, C, D)
    ss_sys = SSModel(*sys)
    za = ss_sys.c2d(h, method="zoha")
    chk_inverse(za, ss_sys)

    r, c = za.B.shape
    yr = za.C.shape[0]
    nt = 301
    u = np.random.randn(c, nt)
    u[:, 0] = 0  # don't let initial conditions mess us up
    x1 = np.zeros((r, nt + 1), float)
    y1 = np.zeros((yr, nt), float)
    for j in range(nt):
        x1[:, j + 1] = za.A.dot(x1[:, j]) + za.B.dot(u[:, j])
        y1[:, j] = za.C.dot(x1[:, j]) + za.D.dot(u[:, j])

    ts = ode.SolveExp1(A, h, order=1)
    F = B.dot(u)
    PQF = np.copy((ts.P + ts.Q).dot((F[:, :-1] + F[:, 1:]) / 2), order="F")
    E = ts.E
    d = np.zeros((r, nt), float)
    d0 = d[:, 0]
    for j in range(1, nt):
        d0 = d[:, j] = E.dot(d0) + PQF[:, j - 1]
    y2 = C.dot(d) + D.dot(u)
    assert np.allclose(y1, y2)


def test_tustin_c2d_d2c():
    # engine actuator dynamics (5 hz, 70% damping):
    # input:   actuator command
    # output:  [actuator angle, angular rate, angular acceleration]'
    w = 5 * 2 * np.pi
    zeta = 0.7
    w2 = w * w
    damp = 2 * w * zeta
    A = np.array([[0, 1], [-w2, -damp]])
    B = np.array([[0], [w2]])
    C = np.array([[1, 0], [0, 1], [-w2, -damp]])
    D = np.array([[0], [0], [w2]])
    h = 0.01
    prewarp = 50.5

    # answer from Matlab:
    Am = np.array(
        [
            [0.958796353566434, 0.008171399133845],
            [-8.064847685445468, 0.599399448728309],
        ]
    )
    Bm = np.array([[0.041203646433566], [8.064847685445468]])
    Cm = 1.0e2 * np.array(
        [
            [0.009793981767832, 0.000040856995669],
            [-0.040324238427227, 0.007996997243642],
            [-7.892719919134404, -0.392050547506857],
        ]
    )
    Dm = 1.0e2 * np.array(
        [[0.000206018232168], [0.040324238427227], [7.892719919134404]]
    )
    sys = (A, B, C, D)
    sysz = (Am, Bm, Cm, Dm)
    ss_sys = SSModel(*sys)
    ss_sysz = ss_sys.c2d(h, method="tustin", prewarp=prewarp)
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)


def test_foh_c2d_d2c():
    # engine actuator dynamics (5 hz, 70% damping):
    # input:   actuator command
    # output:  [actuator angle, angular rate, angular acceleration]'
    w = 5 * 2 * np.pi
    zeta = 0.7
    w2 = w * w
    damp = 2 * w * zeta
    A = np.array([[0, 1], [-w2, -damp]])
    B = np.array([[0], [w2]])
    C = np.array([[1, 0], [0, 1], [-w2, -damp]])
    D = np.array([[0], [0], [w2]])
    h = 0.005

    # answer from Matlab:
    Am = np.array(
        [
            [0.988542968353218, 0.004469980285414],
            [-4.411693709770504, 0.791942967184347],
        ]
    )
    Bm = np.array([[0.021654992049978], [3.917784066760780]])
    Cm = 1.0e2 * np.array(
        [[0.01, 0.0], [0.0, 0.01], [-9.869604401089358, -0.439822971502571]]
    )
    Dm = 1.0e2 * np.array(
        [[0.000038911226114], [0.022914063293564], [8.823387419541009]]
    )
    sys = (A, B, C, D)
    sysz = (Am, Bm, Cm, Dm)
    ss_sys = SSModel(*sys)
    ss_sysz = ss_sys.c2d(h, method="foh")
    assert runsim(ss_sysz, sysz)

    with pytest.raises(ValueError):
        ss_sys.c2d(h, method="badmethod")
    with pytest.raises(ValueError):
        ss_sysz.d2c(method="badmethod")
    chk_inverse(ss_sysz, ss_sys)


def test_get_freq_damping():
    # uncoupled equations
    m = np.array([10.0, 11.0, 12.0, 13.0])  # diagonal of mass
    k = np.array([6.0e5, 7.0e5, 8.0e5, 9.0e5])  # diagonal of stiffness
    zeta = np.array([0.2, 0.05, 1.0, 2.0])  # percent damping

    wn = np.sqrt(k / m)

    # m = np.array([30.0])  # diagonal of mass
    # k = np.array([7.0e5])  # diagonal of stiffness
    # zeta = np.array([1.000000000010])  # percent damping

    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    A = ode.make_A(m, b, k)
    lam, phi = la.eig(A)
    wn_extracted, zeta_extracted = ode.get_freq_damping(lam)
    i = np.argsort(wn_extracted)
    assert np.allclose(wn_extracted[i], wn)
    assert np.allclose(zeta_extracted[i], zeta)

    with pytest.raises(ValueError):
        ode.get_freq_damping(lam[1:])


def test_eigss():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 0.05, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    A = ode.make_A(m, b, k)
    luud = ode.eigss(A, delcc=0)
    luud = (luud.lam, luud.ur, luud.ur_inv, luud.dups)
    # In [5]: luud[0]
    # Out[5]:
    # array([-527.79168675  +0.j        ,  -37.89373820  +0.j        ,
    #        -7.07106781+141.24446892j,   -7.07106781-141.24446892j,
    #        -7.07106781+141.24446892j,   -7.07106781-141.24446892j,
    #        -17.32050808+172.3368794j ,  -17.32050808-172.3368794j ])
    luud_d = ode.delconj(*luud)
    # In [9]: luud_d[0]
    # Out[9]:
    # array([-527.79168675  +0.j        ,  -37.89373820  +0.j        ,
    #        -7.07106781+141.24446892j,   -7.07106781+141.24446892j,
    #        -17.32050808+172.3368794j ])

    for i in range(3):
        assert sum(luud[i].shape) > sum(luud_d[i].shape)
    assert np.all(luud[3] == [2, 3, 4, 5])
    assert np.all(luud_d[3] == [2, 3])
    luu = ode.addconj(*luud_d[:3])
    for i in range(3):
        assert np.allclose(luu[i], luud[i])

    luu2 = ode.addconj(*luu)
    for i in range(3):
        assert luu[i] is luu2[i]

    # pure fakery for test coverage:
    lam, ur, uri = luud_d[:3]
    with pytest.raises(ValueError):
        ode.addconj(lam, ur / 2, uri)
    with pytest.raises(ValueError):
        ode.addconj(np.hstack((lam, lam[-1])), ur, uri)
    lam = lam[[1, 2, 3, 4, 0]]
    with pytest.raises(ValueError):
        ode.addconj(lam, 2 * ur, uri)

    lam, ur, uri = luu2
    urfake = np.vstack((ur, ur[:1]))
    luu3 = ode.addconj(lam, urfake, uri)
    assert luu3[0] is lam
    assert luu3[1] is urfake
    assert luu3[2] is uri


def test_getfsucoef():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    rf = 3
    s = ode.get_su_coef(m, b, k, h, rfmodes=rf)
    nt = f.shape[1]
    d = np.zeros((4, nt), float)
    v = np.zeros((4, nt), float)
    P = f
    d[rf, 0] = f[rf, 0] / k[rf]
    for j in range(nt - 1):
        d[:, j + 1] = s.F * d[:, j] + s.G * v[:, j] + s.A * P[:, j] + s.B * P[:, j + 1]
        v[:, j + 1] = (
            s.Fp * d[:, j] + s.Gp * v[:, j] + s.Ap * P[:, j] + s.Bp * P[:, j + 1]
        )
    ts = ode.SolveUnc(m, b, k, h, rf=rf)
    sol = ts.tsolve(f, static_ic=0)
    assert np.allclose(sol.v, v)
    assert np.allclose(sol.d, d)

    with pytest.raises(ValueError):
        ode.get_su_coef(m, b, k, h, rfmodes=[0, 1])
    with pytest.raises(ValueError):
        ode.get_su_coef(m, b, k, h, rfmodes=[0, 1, 2, 3])


def test_no_h():
    m = 1
    b = 2 * 35 * 0.05
    k = 35**2
    A = ode.make_A(m, b, k)
    h = None

    ts1 = ode.SolveExp1(A, h)
    ts2 = ode.SolveExp2(m, b, k, h)
    tsu = ode.SolveUnc(m, b, k, h)

    with pytest.raises(ValueError):
        ts1.tsolve([[1], [0], [0]])
    sol1 = ts1.tsolve([[1], [0]])
    sol2 = ts2.tsolve(1)
    solu = tsu.tsolve(1, static_ic=0)
    assert sol1.v[0, 0] == 1.0
    assert sol2.a[0, 0] == 1.0
    assert solu.a[0, 0] == 1.0

    f = np.random.randn(1, 10)
    f1 = np.vstack((f, np.zeros((1, 10))))

    with pytest.raises(RuntimeError):
        ts1.tsolve(f1)
    with pytest.raises(RuntimeError):
        ts2.tsolve(f)
    with pytest.raises(RuntimeError):
        tsu.tsolve(f)

    ts2 = ode.SolveExp2(m, b, k, h, rf=0)
    tsu = ode.SolveUnc(m, b, k, h, rf=0)

    sol2 = ts2.tsolve(f)
    solu = tsu.tsolve(f)

    assert np.allclose(sol2.d, solu.d)
    assert np.allclose(sol2.v, solu.v)
    assert np.allclose(sol2.a, solu.a)


def test_ode_uncoupled_freq():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    rb = 0

    freq = np.arange(0, 35, 0.1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - m[:, None].dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.0
    for rf_disp_only in (True, False):
        for rf in (None, [3], [2], np.array([1, 2, 3])):
            tsu = ode.SolveUnc(m, b, k, rf=rf)
            tfd = ode.FreqDirect(m, b, k, rf=rf)
            for incrb in ["", "av", "dva", "d", "v", "a", "da", "dv"]:
                sol = tsu.fsolve(f, freq, incrb=incrb, rf_disp_only=rf_disp_only)
                sold = tfd.fsolve(
                    f[:, 1:], freq[1:], incrb=incrb, rf_disp_only=rf_disp_only
                )
                d = f / H
                d[rb, 0] = 0
                v = 1j * freqw * d
                a = 1j * freqw * v
                a[rb, 0] = f[rb, 0] / m[rb]
                if rf is not None:
                    d[rf] = f[rf] / (k[rf][:, None])
                    if rf_disp_only:
                        v[rf] = 0
                        a[rf] = 0
                    else:
                        v[rf] = 1j * freqw * d[rf]
                        a[rf] = 1j * freqw * v[rf]

                if "d" not in incrb:
                    d[rb] = 0
                if "v" not in incrb:
                    v[rb] = 0
                if "a" not in incrb:
                    a[rb] = 0

                assert np.allclose(a, sol.a)
                assert np.allclose(v, sol.v)
                assert np.allclose(d, sol.d)
                assert np.all(freq == sol.f)

                assert np.allclose(sol.a[:, 1:], sold.a)
                assert np.allclose(sol.v[:, 1:], sold.v)
                assert np.allclose(sol.d[:, 1:], sold.d)

    rf = np.array([0, 1, 2, 3])
    k[0] = k[1]
    tsu = ode.SolveUnc(m, b, k, rf=rf)
    tfd = ode.FreqDirect(m, b, k, rf=rf)
    sol = tsu.fsolve(f, freq)
    sold = tfd.fsolve(f[:, 1:], freq[1:])

    assert np.allclose(sol.a[:, 1:], sold.a)
    assert np.allclose(sol.v[:, 1:], sold.v)
    assert np.allclose(sol.d[:, 1:], sold.d)

    with pytest.raises(ValueError):
        tsu.fsolve(f, freq, incrb="r")


def test_ode_uncoupled_freq_rblast():
    # uncoupled equations
    m = np.array([30.0, 30.0, 30.0, 10])  # diagonal of mass
    k = np.array([6.0e5, 6.0e5, 6.0e5, 0])  # diagonal of stiffness
    zeta = np.array([0.05, 1.0, 2.0, 0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    rb = 3

    freq = np.arange(0, 35, 0.1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - m[:, None].dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.0
    for rf in (None, [2], [1], np.array([0, 1, 2])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        tfd = ode.FreqDirect(m, b, k, rf=rf)
        for incrb in ["", "va", "vda"]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            sold = tfd.fsolve(f[:, 1:], freq[1:], incrb=incrb)
            d = f / H
            d[rb, 0] = 0
            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0] / m[rb]
            if rf is not None:
                d[rf] = f[rf] / (k[rf][:, None])
                v[rf] = 1j * freqw * d[rf]
                a[rf] = 1j * freqw * v[rf]

            if "d" not in incrb:
                d[rb] = 0
            if "v" not in incrb:
                v[rb] = 0
            if "a" not in incrb:
                a[rb] = 0

            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)

            assert np.allclose(sol.a[:, 1:], sold.a)
            assert np.allclose(sol.v[:, 1:], sold.v)
            assert np.allclose(sol.d[:, 1:], sold.d)


def test_ode_uncoupled_freq_mNone():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    rb = 0

    freq = np.arange(0, 35, 0.1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - np.ones((4, 1)).dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.0
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        tfd = ode.FreqDirect(m, b, k, rf=rf)
        for incrb in ["", "av", "vad"]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            sold = tfd.fsolve(f[:, 1:], freq[1:], incrb=incrb)
            d = f / H
            d[rb, 0] = 0
            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0]
            if rf is not None:
                d[rf] = f[rf] / (k[rf][:, None])
                v[rf] = 1j * freqw * d[rf]
                a[rf] = 1j * freqw * v[rf]

            if "d" not in incrb:
                d[rb] = 0
            if "v" not in incrb:
                v[rb] = 0
            if "a" not in incrb:
                a[rb] = 0

            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)

            assert np.allclose(sol.a[:, 1:], sold.a)
            assert np.allclose(sol.v[:, 1:], sold.v)
            assert np.allclose(sol.d[:, 1:], sold.d)


def test_ode_coupled_freq():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    rb = 0

    freq = np.arange(0, 35, 0.5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    for rf_disp_only in (True, False):
        for rf in (None, [3], [2], np.array([1, 2, 3])):
            tsu = ode.SolveUnc(m, b, k, rf=rf)
            tfd = ode.FreqDirect(m, b, k, rf=rf)
            for incrb in ["", "va", "dva"]:
                sol = tsu.fsolve(f, freq, incrb=incrb, rf_disp_only=rf_disp_only)
                sold = tfd.fsolve(
                    f[:, 1:], freq[1:], incrb=incrb, rf_disp_only=rf_disp_only
                )

                m2, b2, k2 = decouple_rf((m, b, k), rf)
                d = np.zeros((4, freqw.size), complex)
                for i, w in enumerate(freqw):
                    H = (k2 - m2 * w**2) + 1j * (b2 * w)
                    if w == 0.0:
                        H[rb, 0] = 1.0
                    d[:, i] = la.solve(H, f[:, i])
                    if w == 0.0:
                        d[rb, 0] = 0

                v = 1j * freqw * d
                a = 1j * freqw * v
                a[rb, 0] = f[rb, 0] / m[rb, rb]
                if rf is not None:
                    d[rf] = la.solve(k[np.ix_(rf, rf)], f[rf])
                    if rf_disp_only:
                        v[rf] = 0
                        a[rf] = 0
                    else:
                        v[rf] = 1j * freqw * d[rf]
                        a[rf] = 1j * freqw * v[rf]

                if "d" not in incrb:
                    d[rb] = 0
                if "v" not in incrb:
                    v[rb] = 0
                if "a" not in incrb:
                    a[rb] = 0

                assert np.allclose(a, sol.a)
                assert np.allclose(v, sol.v)
                assert np.allclose(d, sol.d)
                assert np.all(freq == sol.f)

                assert np.allclose(sol.a[:, 1:], sold.a)
                assert np.allclose(sol.v[:, 1:], sold.v)
                assert np.allclose(sol.d[:, 1:], sold.d)


def test_ode_coupled_freq_cdf():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    b[1:, 1:] += np.random.randn(3, 3)
    freq = np.arange(0, 35, 0.5)
    f = np.ones((4, freq.size))

    tcdf = ode.SolveCDF(m, b, k)
    with pytest.raises(NotImplementedError):
        tcdf.fsolve(f, freq)


def test_ode_coupled_freq_mNone():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    rb = 0
    freq = np.arange(0, 35, 0.5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        tfd = ode.FreqDirect(m, b, k, rf=rf)
        for incrb in ["", "av", "dva"]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            sold = tfd.fsolve(f[:, 1:], freq[1:], incrb=incrb)

            b2, k2 = decouple_rf((b, k), rf)
            d = np.zeros((4, freqw.size), complex)
            for i, w in enumerate(freqw):
                H = (k2 - np.eye(4) * w**2) + 1j * (b2 * w)
                if w == 0.0:
                    H[rb, 0] = 1.0
                d[:, i] = la.solve(H, f[:, i])
                if w == 0.0:
                    d[rb, 0] = 0

            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0]
            if rf is not None:
                d[rf] = la.solve(k[np.ix_(rf, rf)], f[rf])
                v[rf] = 1j * freqw * d[rf]
                a[rf] = 1j * freqw * v[rf]

            if "d" not in incrb:
                d[rb] = 0
            if "v" not in incrb:
                v[rb] = 0
            if "a" not in incrb:
                a[rb] = 0

            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)

            assert np.allclose(sol.a[:, 1:], sold.a)
            assert np.allclose(sol.v[:, 1:], sold.v)
            assert np.allclose(sol.d[:, 1:], sold.d)


def test_ode_fsd_1():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    freq = np.arange(0.1, 35, 0.1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - m[:, None].dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )

    d = f / H
    v = 1j * freqw * d
    a = 1j * freqw * v

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)

    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(a, solu.a)
    assert np.allclose(v, solu.v)
    assert np.allclose(d, solu.d)
    assert np.allclose(a, sold.a)
    assert np.allclose(v, sold.v)
    assert np.allclose(d, sold.d)
    assert np.all(freq == sold.f)

    solu = tsu.fsolve(f, freq, incrb="va")
    with pytest.warns(FutureWarning, match="the integer form of `incrb` is deprecated"):
        sold = tsd.fsolve(f, freq, incrb=1)
    assert np.allclose(sold.a, solu.a)
    assert np.allclose(sold.v, solu.v)
    assert np.allclose(sold.d, solu.d)
    assert np.all(sold.f == solu.f)
    d[0] = 0
    assert np.allclose(a, solu.a)
    assert np.allclose(v, solu.v)
    assert np.allclose(d, solu.d)

    solu = tsu.fsolve(f, freq, incrb="")
    with pytest.warns(FutureWarning, match="the integer form of `incrb` is deprecated"):
        sold = tsd.fsolve(f, freq, incrb=0)
    assert np.allclose(sold.a, solu.a)
    assert np.allclose(sold.v, solu.v)
    assert np.allclose(sold.d, solu.d)
    assert np.all(sold.f == solu.f)
    v[0] = 0
    a[0] = 0
    assert np.allclose(a, solu.a)
    assert np.allclose(v, solu.v)
    assert np.allclose(d, solu.d)


def test_ode_fsd_2():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    freq = np.arange(0.1, 35, 0.1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (1j * b[:, None].dot(freqw[None, :]) + k[:, None]) - freqw[None, :] ** 2
    d = f / H
    v = 1j * freqw * d
    a = 1j * freqw * v

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)

    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(a, solu.a)
    assert np.allclose(v, solu.v)
    assert np.allclose(d, solu.d)
    assert np.allclose(a, sold.a)
    assert np.allclose(v, sold.v)
    assert np.allclose(d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_3():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_4():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_uncoupled_complex_1():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    m = m + 1j * np.random.randn(4)
    k = k + 1j * np.random.randn(4) * 14
    b = b + 1j * np.random.randn(4)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_uncoupled_complex_2():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    k = k + 1j * np.random.randn(4) * 14
    b = b + 1j * np.random.randn(4)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_coupled_complex_1():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    m = m + 1j * np.random.randn(4, 4)
    k = k + 1j * np.random.randn(4, 4) * 14
    b = b + 1j * np.random.randn(4, 4)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_fsd_coupled_complex_2():
    # uncoupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    k = k + 1j * np.random.randn(4, 4) * 14
    b = b + 1j * np.random.randn(4, 4)

    freq = np.arange(0.5, 35, 2.5)
    f = np.ones((4, freq.size))

    tsu = ode.SolveUnc(m, b, k)
    solu = tsu.fsolve(f, freq)
    tsd = ode.FreqDirect(m, b, k)
    sold = tsd.fsolve(f, freq)
    assert np.allclose(solu.a, sold.a)
    assert np.allclose(solu.v, sold.v)
    assert np.allclose(solu.d, sold.d)
    assert np.all(freq == sold.f)


def test_ode_complex_coefficients():
    aa = np.ones((2, 2)) * (1.0 + 1j)
    m = aa.copy()
    m[0, 0] = 3.0 + 2j
    b = aa.copy()
    b[1, 0] = -2.0
    k = aa.copy()
    k[0, 1] = 2.0
    h = 0.001
    t = np.arange(0, 1, h)
    f = np.array([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 1 * t)])
    for use_diag in [0, 1]:
        if use_diag:
            m = np.diag(m)
            b = np.diag(b)
            k = np.diag(k)
        ts = ode.SolveUnc(m, b, k, h)
        sol = ts.tsolve(f, static_ic=0)
        if use_diag:
            fr = m[:, None] * sol.a + b[:, None] * sol.v + k[:, None] * sol.d
        else:
            fr = m.dot(sol.a) + b.dot(sol.v) + k.dot(sol.d)

        check_true_derivatives(sol, tol=1e-5)
        assert np.allclose(f, fr)

        # test the generator solver:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            gen.send((i, f[:, i]))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)

        # test the generator solver w/ partial update:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            fi = f[:, i] / 2
            gen.send((i, fi))
            gen.send((-1, fi))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)

        if use_diag:
            ts = ode.SolveUnc(m, b, k, h, order=0)
            sol = ts.tsolve(f, static_ic=0)
            nt = len(sol.t)
            gen, d, v = ts.generator(nt, f[:, 0])
            for i in range(1, nt):
                fi = f[:, i] / 2
                gen.send((i, fi))
                gen.send((-1, fi))
            sol2 = ts.finalize(1)
            assert np.allclose(f, sol2.force)
            assert np.allclose(sol.a, sol2.a)
            assert np.allclose(sol.v, sol2.v)
            assert np.allclose(sol.d, sol2.d)


def test_ode_complex_coefficients_with_rf():
    aa = np.ones((2, 2)) * (1.0 + 1j)
    m = aa.copy()
    m[0, 0] = 3.0 + 2j
    b = aa.copy()
    b[1, 0] = -2.0
    k = aa.copy()
    k[0, 1] = 2.0
    h = 0.001
    t = np.arange(0, 0.5, h)

    # add an rf DOF:
    Z = np.zeros((3, 3), complex)
    Z[:2, :2] = m
    m = Z
    m[2, 2] = 1.0

    Z = np.zeros((3, 3), complex)
    Z[:2, :2] = b
    b = Z
    b[2, 2] = 1.0

    Z = np.zeros((3, 3), complex)
    Z[:2, :2] = k
    k = Z
    krf = 10.0
    k[2, 2] = krf

    f = np.array(
        [
            np.sin(2 * np.pi * 3 * t),
            np.cos(2 * np.pi * 1 * t),
            np.sin(2 * np.pi * 2.5 * t),
        ]
    )

    for use_diag in [0, 1]:
        if use_diag:
            m = np.diag(m)
            b = np.diag(b)
            k = np.diag(k)
        ts = ode.SolveUnc(m, b, k, h, rf=2)
        sol = ts.tsolve(f, static_ic=0)
        if use_diag:
            fr = m[:, None] * sol.a + b[:, None] * sol.v + k[:, None] * sol.d
        else:
            fr = m.dot(sol.a) + b.dot(sol.v) + k.dot(sol.d)

        v = integrate.cumulative_trapezoid(sol.a, sol.t, initial=0)
        d = integrate.cumulative_trapezoid(sol.v, sol.t, initial=0)
        d[2] = fr[2] / krf

        assert np.allclose(f, fr)
        assert abs(v - sol.v).max() < 1e-5
        assert abs(d - sol.d).max() < 1e-5

        # test the generator solver:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            gen.send((i, f[:, i]))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)

        # test the generator solver w/ partial update:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            fi = f[:, i] / 2
            gen.send((i, fi))
            gen.send((-1, fi))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)


def test_ode_complex_coefficients_mNone():
    aa = np.ones((2, 2)) * (1.0 + 1j)
    m = None
    b = aa.copy()
    b[1, 0] = -2.0
    k = aa.copy()
    k[0, 1] = 2.0
    h = 0.001
    t = np.arange(0, 1, h)
    f = np.array([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 1 * t)])
    for use_diag in [0, 1]:
        if use_diag:
            b = np.diag(b)
            k = np.diag(k)
        ts = ode.SolveUnc(m, b, k, h)
        sol = ts.tsolve(f, static_ic=0)
        if use_diag:
            fr = sol.a + b[:, None] * sol.v + k[:, None] * sol.d
        else:
            fr = sol.a + b.dot(sol.v) + k.dot(sol.d)

        check_true_derivatives(sol, tol=1e-5)
        assert np.allclose(f, fr)

        # test the generator solver:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            gen.send((i, f[:, i]))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)

        # test the generator solver w/ partial update:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            fi = f[:, i] / 2
            gen.send((i, fi))
            gen.send((-1, fi))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)


# def test_ode_complex_coefficients_dups():
#     aa = np.ones((2, 2)) * (1. + 1j)
#     m = aa.copy()
#     m[0, 0] = 3.+2j
#     b = aa.copy()
#     k = aa.copy()
#     h = .1
#     with pytest.warns(RuntimeWarning) as cm:
#         ode.SolveUnc(m, b, k, h)
#     wrn0 = str(cm.warnings[0].message)
#     assert 0 == wrn0.find('Repeated roots detected')
#     found = False
#     for w in cm.warnings[1:]:
#         if str(w.message).find('found 2 rigid-body modes') > -1:
#             found = True
#     assert found


def test_ode_complex_coefficients_rb():
    aa = np.ones((2, 2)) * (1.0 + 1j)
    m = aa.copy()
    m[0, 0] = 3.0 + 2j
    b = aa.copy()
    b[1, 0] = -2.0
    k = aa.copy()
    k[0, 1] = 2.0
    b[0, 0] = 0.0
    k[0, 0] = 0.0
    h = 0.001
    t = np.arange(0, 1, h)
    f = np.array([np.sin(2 * np.pi * 3 * t), np.cos(2 * np.pi * 1 * t)])
    for use_diag in [0, 1]:
        if use_diag:
            m = np.diag(m)
            b = np.diag(b)
            k = np.diag(k)
        ts = ode.SolveUnc(m, b, k, h)
        sol = ts.tsolve(f, static_ic=0)
        if use_diag:
            fr = m[:, None] * sol.a + b[:, None] * sol.v + k[:, None] * sol.d
        else:
            fr = m.dot(sol.a) + b.dot(sol.v) + k.dot(sol.d)

        check_true_derivatives(sol, tol=1e-5)
        assert np.allclose(f, fr)

        # test the generator solver:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            gen.send((i, f[:, i]))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)

        # test the generator solver w/ partial update:
        nt = len(sol.t)
        gen, d, v = ts.generator(nt, f[:, 0])
        for i in range(1, nt):
            fi = f[:, i] / 2
            gen.send((i, fi))
            gen.send((-1, fi))
        sol2 = ts.finalize(1)
        assert np.allclose(f, sol2.force)
        assert np.allclose(sol.a, sol2.a)
        assert np.allclose(sol.v, sol2.v)
        assert np.allclose(sol.d, sol2.d)


def test_approx_rbmodes():
    from pyyeti.nastran import op2, n2p
    from pyyeti.ode import SolveUnc as su
    from pyyeti.ode import SolveExp2 as se2

    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")

    # setup mass, stiffness, damping:
    m = None  # treated as identity
    k = nas["lambda"][0]  # imperfect zeros
    zeta = 1000  # VERY high damping, for testing
    # we end up with RB modes with damping ... good for testing
    b = 2 * np.sqrt(abs(k)) * zeta
    # k[:6] = 0.0
    # b[:6] = 1e-4

    # step input, 2 second duration
    h = 0.001
    f = np.ones((1, int(2 / h)))

    # form drm for force application: node 8, dof x of se 0:
    drm, dof = n2p.formdrm(nas, 0, [[8, 1]])

    # form drm for data recovery to nodes 35, 36 (x dof) of se 100:
    ATM, dof = n2p.formdrm(nas, 100, [[35, 1], [36, 1]])

    # initialize uncoupled solver:
    su = su(m, b, k, h)
    se = se2(m, b, k, h)

    # solve equations of motion with zero initial conditions:
    g = drm.T @ f
    solu_nosic = su.tsolve(g, static_ic=0)
    solu_sic = su.tsolve(g, static_ic=1)

    check_true_derivatives(solu_nosic)
    check_true_derivatives(solu_sic)

    # check for ValueError when force is incorrectly sized:
    with pytest.raises(ValueError):
        se.tsolve(f)

    # solve:
    sole_nosic = se.tsolve(g, static_ic=0)
    sole_sic = se.tsolve(g, static_ic=1)

    check_true_derivatives(sole_nosic)
    check_true_derivatives(sole_sic)

    fru = solu_nosic.a + b[:, None] * solu_nosic.v + k[:, None] * solu_nosic.d
    fre = sole_nosic.a + b[:, None] * sole_nosic.v + k[:, None] * sole_nosic.d

    assert np.allclose(g, fru)
    assert np.allclose(g, fre)

    fru = solu_sic.a + b[:, None] * solu_sic.v + k[:, None] * solu_sic.d
    fre = sole_sic.a + b[:, None] * sole_sic.v + k[:, None] * sole_sic.d

    assert np.allclose(g, fru)
    assert np.allclose(g, fre)
    assert np.allclose(solu_nosic.a, sole_nosic.a)
    assert np.allclose(solu_nosic.v, sole_nosic.v)
    assert np.allclose(solu_nosic.d, sole_nosic.d, atol=1e-5)
    assert np.allclose(solu_sic.a, sole_sic.a)
    assert np.allclose(solu_sic.v, sole_sic.v)
    assert np.allclose(solu_sic.d, sole_sic.d, atol=1e-5)

    # # recover accels (35 x, 36 x, 8 x):
    # atm = np.vstack((ATM, drm))
    # acceu_nosic = atm @ solu_nosic.a[:, -1]
    # accee_nosic = atm @ sole_nosic.a[:, -1]
    # acceu_sic = atm @ solu_sic.a[:, -1]
    # accee_sic = atm @ sole_sic.a[:, -1]
    #
    # for sole, solu, name in ((sole_sic, solu_sic, 'sic'),
    #                          (sole_nosic, solu_nosic, 'nosic')):
    #     plt.figure(name, figsize=(8, 8))
    #     plt.clf()
    #     for i, r in enumerate('avd'):
    #         plt.subplot(3, 1, i+1)
    #         plt.plot(sole.t, getattr(sole, r).T, '-',
    #                  sole.t, getattr(solu, r).T, '--')


def test_ode_pre_eig():
    # 1. Setup system::
    #
    #             |--> x1       |--> x2        |--> x3        |--> x4
    #             |             |              |              |
    #          |----|    k1   |----|    k2   |----|    k3   |----|
    #      Fe  |    |--\/\/\--|    |--\/\/\--|    |--\/\/\--|    |
    #     ====>| 10 |         | 30 |         |  3 |         |  2 |
    #          |    |---| |---|    |---| |---|    |---| |---|    |
    #          |----|    c1   |----|    c2   |----|    c3   |----|
    #
    #          |<--- SOURCE --->||<------------ LOAD ----------->|
    #
    # Define parameters:

    freq = np.arange(1.0, 25.1, 0.25)
    h = 0.01
    time = np.arange(0.0, 2.0, h)
    M1 = 10.0
    M2 = 30.0
    M3 = 3.0
    M4 = 2.0
    c1 = 15.0
    c2 = 15.0
    c3 = 15.0
    k1 = 45000.0
    k2 = 25000.0
    k3 = 10000.0

    MASS = np.array([[M1, 0, 0, 0], [0, M2, 0, 0], [0, 0, M3, 0], [0, 0, 0, M4]])
    DAMP = np.array(
        [
            [c1, -c1, 0, 0],
            [-c1, c1 + c2, -c2, 0],
            [0, -c2, c2 + c3, -c3],
            [0, 0, -c3, c3],
        ]
    )
    STIF = np.array(
        [
            [k1, -k1, 0, 0],
            [-k1, k1 + k2, -k2, 0],
            [0, -k2, k2 + k3, -k3],
            [0, 0, -k3, k3],
        ]
    )

    # frequency domain first:
    F = np.vstack((np.ones((1, len(freq))), np.zeros((3, len(freq)))))

    sol1 = ode.SolveUnc(MASS, DAMP, STIF, pre_eig=True).fsolve(F, freq)
    sol2 = ode.FreqDirect(MASS, DAMP, STIF).fsolve(F, freq)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)

    # frequency domain with identity M ... as None:
    sol1 = ode.SolveUnc(None, DAMP, STIF, pre_eig=True).fsolve(F, freq)
    fsu = ode.SolveUnc(None, DAMP, STIF, h, pre_eig=True)
    sol1a = fsu.fsolve(F, freq)
    sol2 = ode.FreqDirect(None, DAMP, STIF).fsolve(F, freq)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)
    assert np.allclose(sol1a.d, sol2.d)
    assert np.allclose(sol1a.v, sol2.v)
    assert np.allclose(sol1a.a, sol2.a)

    # time domain:
    fsu = ode.SolveUnc(MASS, DAMP, STIF, h, pre_eig=True)
    sol1a = fsu.fsolve(F, freq)
    F[0] = 0.0
    pv = np.where((time > 0.1) & (time < 0.8))[0]
    F[0][pv] = 10.0
    sol1 = ode.SolveUnc(MASS, DAMP, STIF, h, pre_eig=True).tsolve(F)
    sol1a = fsu.tsolve(F)
    sol2 = ode.SolveExp2(MASS, DAMP, STIF, h).tsolve(F)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)
    assert np.allclose(sol1a.d, sol2.d)
    assert np.allclose(sol1a.v, sol2.v)
    assert np.allclose(sol1a.a, sol2.a)

    STIF2 = np.diag(STIF)
    sol1 = ode.SolveUnc(MASS, DAMP, STIF2, h, pre_eig=True).tsolve(F)
    sol2 = ode.SolveExp2(MASS, DAMP, STIF2, h).tsolve(F)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)

    F[0] = 0.0
    pv = np.where(time < 0.8)[0]
    F[0][pv] = 10.0
    sol1 = ode.SolveUnc(MASS, DAMP, STIF, h, pre_eig=True).tsolve(F, static_ic=1)
    sol2 = ode.SolveExp2(MASS, DAMP, STIF, h, pre_eig=True).tsolve(F, static_ic=1)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)

    STIF2 = np.array(
        [
            [k1, -k1, 0, 0],
            [-k1, k1 + k2, -k2, 0],
            [k1, -k2, k2 + k3, -k3],
            [k1, 0, -k3, k3],
        ]
    )
    with pytest.raises(la.LinAlgError):
        ode.SolveUnc(MASS, DAMP, STIF2, h, pre_eig=True)
    with pytest.raises(la.LinAlgError):
        ode.SolveUnc(None, DAMP, STIF2, h, pre_eig=True)

    MASS[0, 0] = 0.0
    with pytest.raises(la.LinAlgError):
        ode.SolveUnc(MASS, DAMP, STIF, h, pre_eig=True)


def test_ode_badsize():
    m = None
    k = np.random.randn(3, 12)
    b = np.random.randn(3, 12)
    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    f = np.random.randn(3, len(t))
    with pytest.raises(ValueError):
        ode.SolveExp2(m, b, k, h)
    with pytest.raises(ValueError):
        ode.SolveExp2(b[0], b, k, h)
    b1 = np.random.randn(2, 2)
    with pytest.raises(ValueError):
        ode.SolveExp2(m, b1, k, h)
    m1 = np.random.randn(3, 3, 3)
    with pytest.raises(ValueError):
        ode.SolveExp2(m1, b, k, h)


def test_precalc_warnings():
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([50.0, 6.0e5, 6.0e5, 6.0e-15])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    h = 0.001  # time step

    with pytest.warns(RuntimeWarning):
        ode.SolveUnc(m, b, k, h, rf=[2, 3])

    m = np.array([10e3, 30e-18, 30.0, 30.0])  # diagonal of mass
    with pytest.warns(RuntimeWarning):
        ode.SolveUnc(m, b, k, h, rf=3)


def test_ode_solvepsd():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    freq = np.arange(0.1, 35, 0.1)

    forcepsd = 10000 * np.ones((4, freq.size))  # constant PSD forces
    ts = ode.SolveUnc(m, b, k)
    atm = np.random.randn(4, 4)
    vtm = np.random.randn(4, 4)
    dtm = np.random.randn(4, 4)
    t_frc = np.random.randn(4, 4)
    forcephi = np.random.randn(4, 4)
    drms = [[atm, None, None, None], [None, None, dtm, None], [atm, vtm, dtm, forcephi]]

    rbduf = 1.2
    elduf = 1.5

    rms, psd = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    rmsrb1, psdrb1 = ode.solvepsd(ts, forcepsd, t_frc, freq, drms, incrb="va")
    rmsrb0, psdrb0 = ode.solvepsd(ts, forcepsd, t_frc, freq, drms, incrb="")
    rmsduf, psdduf = ode.solvepsd(
        ts, forcepsd, t_frc, freq, drms, rbduf=rbduf, elduf=elduf
    )
    rmsf, psdf = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    rmsphi, psdphi = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    with pytest.raises(ValueError):
        ode.solvepsd(ts, forcepsd, t_frc, freq[:-1], drms)
    with pytest.raises(ValueError):
        ode.solvepsd(ts, forcepsd, t_frc[:, :-1], freq, drms)

    # solve by hand for comparison:
    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - m[:, None].dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )

    dpsd = 0.0
    apsd = 0.0
    adpsd = 0.0
    dpsdduf = 0.0
    apsdduf = 0.0
    adpsdduf = 0.0
    dpsd1 = 0.0
    apsd1 = 0.0
    adpsd1 = 0.0
    dpsd0 = 0.0
    apsd0 = 0.0
    adpsd0 = 0.0
    unitforce = np.ones((1, len(freq)))
    for i in range(forcepsd.shape[0]):
        # solve for unit frequency response function:
        genforce = t_frc[:, i : i + 1] @ unitforce
        # sol = ts.fsolve(genforce, freq)
        d = genforce / H
        v = 1j * freqw * d
        a = 1j * freqw * v
        dpsd = dpsd + abs(dtm @ d) ** 2 * forcepsd[i]
        apsd = apsd + abs(atm @ a) ** 2 * forcepsd[i]
        fterm = forcephi[:, i : i + 1] @ unitforce
        adpsd = adpsd + abs(atm @ a + vtm @ v + dtm @ d + fterm) ** 2 * forcepsd[i]

        dduf = d.copy()
        dduf[0] = dduf[0] * rbduf
        dduf[1:] = dduf[1:] * elduf
        vduf = v.copy()
        vduf[0] = vduf[0] * rbduf
        vduf[1:] = vduf[1:] * elduf
        aduf = a.copy()
        aduf[0] = aduf[0] * rbduf
        aduf[1:] = aduf[1:] * elduf
        dpsdduf = dpsdduf + abs(dtm @ dduf) ** 2 * forcepsd[i]
        apsdduf = apsdduf + abs(atm @ aduf) ** 2 * forcepsd[i]
        adpsdduf = (
            adpsdduf
            + abs(atm @ aduf + vtm @ vduf + dtm @ dduf + fterm) ** 2 * forcepsd[i]
        )

        # incrb = "av"
        d[0] = 0
        dpsd1 = dpsd1 + abs(dtm @ d) ** 2 * forcepsd[i]
        apsd1 = apsd1 + abs(atm @ a) ** 2 * forcepsd[i]
        adpsd1 = adpsd1 + abs(atm @ a + vtm @ v + dtm @ d + fterm) ** 2 * forcepsd[i]

        # incrb = ""
        a[0] = 0
        v[0] = 0
        dpsd0 = dpsd0 + abs(dtm @ d) ** 2 * forcepsd[i]
        apsd0 = apsd0 + abs(atm @ a) ** 2 * forcepsd[i]
        adpsd0 = adpsd0 + abs(atm @ a + vtm @ v + dtm @ d + fterm) ** 2 * forcepsd[i]

    assert np.allclose(psd[0], apsd)
    assert np.allclose(psd[1], dpsd)
    assert np.allclose(psd[2], adpsd)

    # incrb="av"
    assert np.allclose(psdrb1[0], apsd1)
    assert np.allclose(psdrb1[1], dpsd1)
    assert np.allclose(psdrb1[2], adpsd1)

    # incrb=""
    assert np.allclose(psdrb0[0], apsd0)
    assert np.allclose(psdrb0[1], dpsd0)
    assert np.allclose(psdrb0[2], adpsd0)

    # with uncertainty factors
    assert np.allclose(psdduf[0], apsdduf)
    assert np.allclose(psdduf[1], dpsdduf)
    assert np.allclose(psdduf[2], adpsdduf)


def test_ode_solvepsd_rf_disp_only():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    freq = np.arange(0.1, 35, 0.1)

    forcepsd = 10000 * np.ones((4, freq.size))  # constant PSD forces
    rf = 3
    ts = ode.SolveUnc(m, b, k, rf=[rf])
    atm = np.random.randn(4, 4)
    vtm = np.random.randn(4, 4)
    dtm = np.random.randn(4, 4)
    t_frc = np.eye(4)
    drms = [[atm, None, None, None], [None, vtm, None, None], [None, None, dtm, None]]

    rms, psd = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    rms_rfd, psd_rfd = ode.solvepsd(ts, forcepsd, t_frc, freq, drms, rf_disp_only=True)

    # solve by hand for comparison:
    freqw = 2 * np.pi * freq
    H = (
        k[:, None]
        - m[:, None].dot(freqw[None, :] ** 2)
        + (1j * b[:, None].dot(freqw[None, :]))
    )

    apsd = 0.0
    vpsd = 0.0
    dpsd = 0.0

    apsd_rfd = 0.0
    vpsd_rfd = 0.0

    unitforce = np.ones((1, len(freq)))
    for i in range(forcepsd.shape[0]):
        # solve for unit frequency response function:
        genforce = t_frc[:, i : i + 1] @ unitforce

        d = genforce / H
        d[rf] = genforce[rf] / k[rf, None]

        # rf_disp_only=False:
        v = 1j * freqw * d
        a = 1j * freqw * v

        apsd = apsd + abs(atm @ a) ** 2 * forcepsd[i]
        vpsd = vpsd + abs(vtm @ v) ** 2 * forcepsd[i]
        dpsd = dpsd + abs(dtm @ d) ** 2 * forcepsd[i]

        # rf_disp_only=True:
        v[rf] = 0.0
        a[rf] = 0.0
        apsd_rfd = apsd_rfd + abs(atm @ a) ** 2 * forcepsd[i]
        vpsd_rfd = vpsd_rfd + abs(vtm @ v) ** 2 * forcepsd[i]

    # rf_disp_only=False:
    assert np.allclose(psd[0], apsd)
    assert np.allclose(psd[1], vpsd)
    assert np.allclose(psd[2], dpsd)

    # rf_disp_only=True:
    assert np.allclose(psd_rfd[0], apsd_rfd)
    assert np.allclose(psd_rfd[1], vpsd_rfd)
    assert np.allclose(psd_rfd[2], dpsd)

    assert not np.allclose(psd_rfd[0], psd[0])
    assert not np.allclose(psd_rfd[1], psd[1])


def test_getmodepart():
    K = [
        [12312.27, -38.20, 611.56, -4608.26, 2845.92],
        [-38.20, 3072.44, -1487.68, 3206.59, 746.56],
        [611.56, -1487.68, 800.91, -1718.08, -164.51],
        [-4608.26, 3206.59, -1718.08, 9189.42, -1890.31],
        [2845.92, 746.56, -164.51, -1890.31, 10908.62],
    ]
    M = None
    w2, phi = la.eigh(K)
    zetain = np.array([0.02, 0.02, 0.05, 0.02, 0.05])
    Z = np.diag(2 * zetain * np.sqrt(w2))
    mfreq = np.sqrt(w2) / 2 / np.pi

    freq = np.arange(0.1, 15.05, 0.1)
    f = np.ones((1, len(freq)))
    Tbot = phi[0:1, :]
    Tmid = phi[2:3, :]
    Ttop = phi[4:5, :]
    ts = ode.SolveUnc(M, Z, w2)
    sol_bot = ts.fsolve(Tbot.T @ f, freq)
    sol_mid = ts.fsolve(Tmid.T @ f, freq)

    # prepare transforms and solutions for getmodepart: (Note: the top
    # 2 items in sols could be combined since they both use the same
    # acceleration)

    sols = [
        [Tmid, sol_bot.a, "Bot to Mid"],
        [Ttop, sol_bot.a, "Bot to Top"],
        [Ttop, sol_mid.a, "Mid to Top"],
    ]
    #
    # APPROACH 1:  let getmodepart() do the FRF plotting:
    #
    modes, freqs = ode.getmodepart(
        freq, sols, mfreq, ylog=1, idlabel="getmodepart demo 1", factor=0.1, auto=[1, 0]
    )

    mds2, frqs2, r = ode.modeselect(
        "modeselect demo 1",
        ts,
        Tbot.T @ f,
        freq,
        Ttop,
        "Bot to Top",
        mfreq,
        factor=0.1,
        auto=0,
    )
    # from Yeti:
    modes_sbe = [2, 3]
    freqs_sbe = [13.53671044272239, 15.80726801820284]

    assert np.allclose(modes_sbe, modes)
    assert np.allclose(freqs_sbe, freqs)
    assert np.allclose(modes_sbe, mds2)
    assert np.allclose(freqs_sbe, frqs2)

    import matplotlib.pyplot as plt

    plt.close("all")
    from pyyeti.datacursor import DataCursor

    def getdata(self):
        self.off()
        plt.figure("FRF")
        ax = plt.gca()
        # the next line is preferable for normal use ... however,
        # for more complete test coverage, use the 2nd one:

        # self.on(ax, callbacks=False)
        self.on()

        point = self._snap(ax, 200.0, 200.0)
        self._add_point(point)
        self.off()
        return 1

    old_fake_getdata = DataCursor._fake_getdata
    DataCursor._fake_getdata = getdata
    mds3, frqs3, r = ode.modeselect(
        "modeselect demo 2",
        ts,
        Tbot.T @ f,
        freq,
        Ttop,
        "Bot to Top",
        mfreq,
        factor=0.1,
        auto=None,
    )
    plt.close("all")
    DataCursor._fake_getdata = old_fake_getdata

    modes_sbe = [1, 2]
    freqs_sbe = [6.1315401651466273, 13.53671044272239]
    assert np.allclose(modes_sbe, mds3)
    assert np.allclose(freqs_sbe, frqs3)

    # check for some error conditions:
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            4,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )

    sols = [
        [Tmid, sol_bot.a],
        [Ttop, sol_bot.a, "Bot to Top"],
        [Ttop, sol_mid.a, "Mid to Top"],
    ]
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            sols,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )

    T = np.vstack((Tmid, Ttop))
    sols = [[T, sol_bot.a, "Bot to Mid"], [Ttop, sol_mid.a, "Mid to Top"]]
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            sols,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )

    sols = [[T, sol_bot.a, ["Bot to Mid"]], [Ttop, sol_mid.a, "Mid to Top"]]
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            sols,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )

    sols = [
        [Tmid, sol_bot.a, "Bot to Mid"],
        [Ttop, sol_bot.a, "Bot to Top"],
        [Ttop, sol_mid.a[:-1, :], "Mid to Top"],
    ]
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            sols,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )

    sols = [
        [Tmid, sol_bot.a, ["Bot to Mid", "bad label"]],
        [Ttop, sol_bot.a, "Bot to Top"],
        [Ttop, sol_mid.a, "Mid to Top"],
    ]
    with pytest.raises(ValueError):
        ode.getmodepart(
            freq,
            sols,
            mfreq,
            ylog=1,
            idlabel="getmodepart demo 1",
            factor=0.1,
            auto=[1, 0],
        )


def test_ode_ic_generator():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 1.2])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    d0 = np.random.randn(4)
    v0 = np.random.randn(4)

    for get_force in (True, False):
        for order in (0, 1):
            su = ode.SolveUnc(m, b, k, h, order=order)
            solu = su.tsolve(f, d0, v0)

            se = ode.SolveExp2(m, b, k, h, order=order)
            sole = se.tsolve(f, d0, v0)

            nt = len(t)
            gen, d, v = su.generator(nt, f[:, 0], d0, v0)
            for i in range(1, nt):
                gen.send((i, f[:, i]))
            solu2 = su.finalize(get_force)

            nt = len(t)
            gen, d, v = se.generator(nt, f[:, 0], d0, v0)
            for i in range(1, nt):
                gen.send((i, f[:, i]))
            sole2 = se.finalize(get_force)

            assert np.allclose(solu2.a, solu.a)
            assert np.allclose(solu2.v, solu.v)
            assert np.allclose(solu2.d, solu.d)

            assert np.allclose(sole2.a, solu.a)
            assert np.allclose(sole2.v, solu.v)
            assert np.allclose(sole2.d, solu.d)

            if get_force:
                assert np.all(solu2.force == f)
                assert np.all(sole2.force == f)

            assert np.allclose(v0, solu.v[:, 0])
            assert np.allclose(d0, solu.d[:, 0])

            # test the generator solver w/ partial update:
            gen, d, v = su.generator(nt, f[:, 0], d0, v0)
            for i in range(1, nt):
                fi = f[:, i] / 2
                gen.send((i, fi))
                gen.send((-1, fi))
            solu2 = su.finalize(get_force)

            gen, d, v = se.generator(nt, f[:, 0], d0, v0)
            for i in range(1, nt):
                fi = f[:, i] / 2
                gen.send((i, fi))
                gen.send((-1, fi))
            sole2 = se.finalize(get_force)

            assert np.allclose(solu2.a, solu.a)
            assert np.allclose(solu2.v, solu.v)
            assert np.allclose(solu2.d, solu.d)

            assert np.allclose(sole2.a, solu.a)
            assert np.allclose(sole2.v, solu.v)
            assert np.allclose(sole2.d, solu.d)


def test_ode_uncoupled_generator():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )
    get_force = True
    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                nt = 1
                gen, d, v = tsu0.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu20 = tsu0.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                nt = 1
                gen, d, v = tsu0.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu20 = tsu0.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                # se2
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)

                nt = 1
                gen, d, v = tse0.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole20 = tse0.finalize(get_force)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize(get_force)

                nt = 1
                gen, d, v = tse0.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole20 = tse0.finalize(get_force)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                if get_force:
                    assert np.allclose(solu2.force, f)
                    assert np.allclose(solu20.force, f[:, :1])
                    assert np.allclose(sole2.force, f)
                    assert np.allclose(sole20.force, f[:, :1])

    nt = f.shape[1]
    tsu = ode.SolveUnc(m, b, k, h, order=order, rf=2)
    with pytest.raises(NotImplementedError):
        tsu.generator(nt, f[:, 0], static_ic=static_ic)

    tse2 = ode.SolveExp2(m, b, k, h, order=order, rf=2)
    with pytest.raises(NotImplementedError):
        tse2.generator(nt, f[:, 0], static_ic=static_ic)

    tsu = ode.SolveUnc(m, b, k, h, order=order)
    with pytest.raises(ValueError):
        tsu.generator(nt, f[:-1, 0], static_ic=static_ic)

    tsu = ode.SolveUnc(m, b, np.diag(k), h, order=order, pre_eig=True)
    with pytest.raises(NotImplementedError):
        tsu.generator(nt, f[:, 0], static_ic=static_ic)


def test_ode_uncoupled_2_generator():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_ / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    get_force = True
    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip(
            (None, None, 3, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0, 1.0)
        ):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(1, f[:, 0], static_ic=static_ic)
                solu20 = tsu0.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                if get_force:
                    assert np.allclose(solu2.force, f)
                    assert np.allclose(solu20.force, f[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se2
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(1, f[:, 0], static_ic=static_ic)
                sole20 = tse0.finalize(get_force)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                if get_force:
                    assert np.allclose(sole2.force, f)
                    assert np.allclose(sole20.force, f[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize(get_force)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)


def test_ode_coupled_generator():
    # coupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf in (None, 3, np.array([1, 2, 3])):
            if (rf == np.array([3])).all() and m.ndim > 1:
                m = np.diag(m)
                b = np.diag(b)
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(1, f[:, 0], static_ic=static_ic)
                solu20 = tsu0.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(1, f[:, 0], static_ic=static_ic)
                sole20 = tse0.finalize()

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize()

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)


def test_ode_coupled_2_generator():
    # coupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_ / m) * m  # diagonal of damping
    m = np.diag(m)
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    m += np.random.randn(4, 4)
    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    get_force = True
    for order in (0, 1):
        for rf, kmult in zip(
            (None, None, 3, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0, 1.0)
        ):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(1, f[:, 0], static_ic=static_ic)
                solu20 = tsu0.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                if get_force:
                    assert np.allclose(solu2.force, f)
                    assert np.allclose(solu20.force, f[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(1, f[:, 0], static_ic=static_ic)
                sole20 = tse0.finalize(get_force)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                if get_force:
                    assert np.allclose(sole2.force, f)
                    assert np.allclose(sole20.force, f[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize(get_force)

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)


def test_ode_coupled_mNone_generator():
    # coupled equations
    m = None
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 2.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k)  # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf in (None, 3, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(1, f[:, 0], static_ic=static_ic)
                solu20 = tsu0.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(1, f[:, 0], static_ic=static_ic)
                sole20 = tse0.finalize()

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])

                # test the generator solver w/ partial update:
                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize()

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)


def test_ode_coupled_2_mNone_generator():
    # coupled equations
    m = None
    k_ = np.array([3.0e5, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.1, 0.05, 1.0, 2.0])  # percent damping
    b_ = 2.0 * zeta * np.sqrt(k_)  # diagonal of damping
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    for order in (0, 1):
        for rf, kmult in zip((None, 3, np.array([0, 1, 2, 3])), (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))

                # resolve some time steps for test:
                for i in range(nt - 5, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))

                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None, order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(1, f[:, 0], static_ic=static_ic)
                solu20 = tsu0.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])

                assert np.allclose(solu20.a, solu2.a[:, :1])
                assert np.allclose(solu20.v, solu2.v[:, :1])
                assert np.allclose(solu20.d, solu2.d[:, :1])

                # se
                tse = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))

                # resolve some time steps for test:
                for i in range(nt - 5, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))

                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None, order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(1, f[:, 0], static_ic=static_ic)
                sole20 = tse0.finalize()

                assert np.allclose(sole2.a, sole.a)
                assert np.allclose(sole2.v, sole.v)
                assert np.allclose(sole2.d, sole.d)

                assert np.allclose(solu.a, sole.a)
                assert np.allclose(solu.v, sole.v)
                assert np.allclose(solu.d, sole.d)

                assert np.allclose(sole0.a, sole.a[:, :1])
                assert np.allclose(sole0.v, sole.v[:, :1])
                assert np.allclose(sole0.d, sole.d[:, :1])

                assert np.allclose(sole20.a, sole2.a[:, :1])
                assert np.allclose(sole20.v, sole2.v[:, :1])
                assert np.allclose(sole20.d, sole2.d[:, :1])


def test_abstractness():
    from pyyeti.ode._base_ode_class import _BaseODE

    a = _BaseODE()
    with pytest.raises(NotImplementedError):
        a.tsolve()
    with pytest.raises(NotImplementedError):
        a.fsolve()
    with pytest.raises(NotImplementedError):
        a.generator()
    with pytest.raises(NotImplementedError):
        a.get_f2x()


def test_henkel_mar():
    # flag specifying whether to use velocity or displacement for
    # the joint compatibility
    use_velo = 1
    # solver = ode.SolveExp2
    sep_time = 0.32  # if <= 0.0, do not separate
    # sep_time = -1.0

    # system setup:
    #
    #                     |---> x1           |---> x2           |---> x3
    #    /|               |                  |                  |
    #    /|     k1     |-----|     k2     |-----|     k3     |-----|
    #    /|----\/\/\---|     |----\/\/\---|     |----\/\/\---|     |
    #    /|            | 30  |            |  3  |            |  2  |
    #    /|----| |-----|     |----| |-----|     |----| |-----|     |
    #    /|    c1      |-----|    c2      |-----|    c3      |-----|
    #    /|
    #      <---- PAD ---->||<---------------- LV ----------------->|

    for mm in (None, "other"):
        for c2 in (20.0, 30.0):
            M1 = 30.0
            M1_p = 29.0
            M1_l = M1 - M1_p
            M2 = 3.0
            M3 = 2.0
            c1 = 45.0
            c3 = 10.0
            k1 = 45000.0
            k2 = 20000.0
            k3 = 10000.0

            rfcut = 1e6
            # rfcut = 1.0
            for solver in (ode.SolveUnc, ode.SolveExp2):
                # full system:
                full = SimpleNamespace()
                full.M = np.array([[M1, 0, 0], [0, M2, 0], [0, 0, M3]])
                full.K = np.array(
                    [[k1 + k2, -k2, 0], [-k2, k2 + k3, -k3], [0, -k3, k3]]
                )
                full.C = np.array(
                    [[c1 + c2, -c2, 0], [-c2, c2 + c3, -c3], [0, -c3, c3]]
                )
                w, u = la.eigh(full.K, full.M)
                full.freq = np.sqrt(w) / (2 * np.pi)
                full.m = None
                full.w = w
                full.u = u
                full.k = w
                full.c = u.T @ full.C @ u

                # setup time vector:
                sr = 1000.0
                h = 1 / sr
                time = np.arange(0, 0.5, h)
                L = len(time)
                # zeta0 = 0.0

                # solve full system w/o Henkel-Mar:
                physF = np.zeros((1, L))
                physF[0] = np.interp(
                    time,
                    np.array([0.0, 0.05, 0.1, 1.0]),
                    np.array([50000.0, 50000.0, 0.0, 0.0]),
                )

                physF3 = np.array([[0.0], [0.0], [1.0]]) @ physF
                F = full.u.T @ physF3
                rf = (full.k > rfcut).nonzero()[0]
                full.parms = solver(full.m, full.c, full.k, h, rf=rf)

                full.sol = full.parms.tsolve(F, static_ic=True)
                full.x = full.u @ full.sol.d
                # x0 = full.x[:, :1]
                full.xd = full.u @ full.sol.v
                full.xdd = full.u @ full.sol.a

                # define i/f force as force on lv. compute iff by
                # looking at pad:
                #   sum of forces on m1 = m a = -iff - k1*x1 - c1*xd1
                # iff = -m a - k1*x1 - c1*xd1
                full.iff = -(M1_p * full.xdd[0] + k1 * full.x[0] + c1 * full.xd[0])

                # Henkel Mar

                # make 'pad' and lv models - split the 30 mass into
                # two parts:
                # pad:
                pad = SimpleNamespace()
                pad.M = np.array([[M1_p]])
                pad.K = np.array([[k1]])
                pad.C = np.array([[c1]])
                w, u = la.eigh(pad.K, pad.M)
                if mm is None:
                    pad.m = mm
                else:
                    pad.m = np.ones(w.shape[0])
                pad.k = w
                pad.u = u
                pad.c = u.T @ pad.C @ u
                # get transform to i/f dof:
                pad.phi = pad.u[:1]
                rf = (pad.k > rfcut).nonzero()[0]
                pad.parms = solver(pad.m, pad.c, pad.k, h, rf=rf)

                # lv:
                lv = SimpleNamespace()
                lv.M = np.array([[M1_l, 0.0, 0.0], [0.0, M2, 0.0], [0.0, 0.0, M3]])
                lv.K = np.array([[k2, -k2, 0.0], [-k2, k2 + k3, -k3], [0.0, -k3, k3]])
                lv.C = np.array([[c2, -c2, 0.0], [-c2, c2 + c3, -c3], [0.0, -c3, c3]])
                w, u = la.eigh(lv.K, lv.M)
                # print('lv lambda:', w)
                w[0] = 0.0
                if mm is None:
                    lv.m = mm
                else:
                    lv.m = np.ones(w.shape[0])
                lv.k = w
                lv.u = u
                lv.c = u.T @ lv.C @ u
                # get transform to i/f dof:
                lv.phi = lv.u[:1]
                rf = (lv.k > rfcut).nonzero()[0]
                lv.parms = solver(lv.m, lv.c, lv.k, h, rf=rf)

                # setup henkel-mar equations:
                pad.dflex = pad.parms.get_f2x(pad.phi, 0)
                lv.dflex = lv.parms.get_f2x(lv.phi, 0)
                pad.flex = pad.parms.get_f2x(pad.phi, use_velo)
                lv.flex = lv.parms.get_f2x(lv.phi, use_velo)
                assert 10 * abs(pad.dflex) < abs(pad.flex)
                assert 10 * abs(lv.dflex) < abs(lv.flex)

                # setup some variables for working with Henkel-Mar:
                r = 1  # number of i/f dof
                pvd = np.array([0])
                pvf = r + pvd

                # "Work" will contain components of matrix in eq 27
                # and variables for working through solution:
                Work = SimpleNamespace()
                # all constrained to begin:
                Work.constraints = np.ones(r)
                Work.Flex = pad.flex + lv.flex
                Work.rhs = np.zeros(r * 2)

                # initialize coupled solution solution and interface
                #  disps and forces:
                Sol = SimpleNamespace()
                Sol.reldisp = np.zeros((r, L))
                Sol.fpad = np.zeros((r, L))

                # static initial conditions:
                nrb = 1
                flr = lv.u[:, :nrb].T @ physF3[:, :1]
                fpad = -la.inv(lv.phi[:, :nrb].T) @ flr
                qel = (
                    1
                    / lv.k[nrb:][:, None]
                    * (lv.u[:, nrb:].T @ physF3[:, :1] + lv.phi[:, nrb:].T @ fpad)
                )
                qep = 1 / pad.k[:, None] * (-pad.phi.T @ fpad)
                qrl = la.inv(lv.phi[:, :nrb]) @ (pad.phi @ qep - lv.phi[:, nrb:] @ qel)
                # x0_lv = lv.u[:, :nrb] @ qrl + lv.u[:, nrb:] @ qel
                # print('disp err @ t=0:', x0_lv, x0, x0_lv-x0)
                Sol.fpad[:, :1] = fpad
                lv.d0 = np.vstack((qrl, qel))
                pad.d0 = qep

                # forces (include the pad forces for first time step):
                lv.force = lv.u.T @ physF3
                lv.force[:, 0] += lv.phi.T @ Sol.fpad[:, 0]
                pad.force = np.zeros((pad.k.shape[0], L))
                pad.force[:, 0] -= pad.phi.T @ Sol.fpad[:, 0]

                lv.gen, lv.d, lv.v = lv.parms.generator(
                    L, lv.force[:, 0], d0=lv.d0.ravel()
                )
                pad.gen, pad.d, pad.v = pad.parms.generator(
                    L, pad.force[:, 0], d0=pad.d0.ravel()
                )

                # while bolted to the pad:
                Work.K = np.zeros((2 * r, 2 * r))
                Work.K[:r, :r] = np.eye(r)
                Work.K[:r, r:] = -Work.Flex
                Work.K[r:, :r] = np.diag(Work.constraints)
                Work.K[r:, r:] = np.diag(1.0 - Work.constraints)
                Work.Kd = la.inv(Work.K)

                if sep_time <= 0:
                    conn = L
                else:
                    conn = abs(time - sep_time).argmin()

                def SolveStep(lv, pad, i, Work, pvd, pvf):
                    # update lv & pad solutions ... only missing
                    # updated pad force:
                    lv.gen.send((i, lv.force[:, i]))
                    pad.gen.send((i, pad.force[:, i]))
                    if use_velo:
                        Work.rhs[pvd] = lv.phi @ lv.v[:, i] - pad.phi @ pad.v[:, i]
                    else:
                        Work.rhs[pvd] = lv.phi @ lv.d[:, i] - pad.phi @ pad.d[:, i]
                    Work.lhs = Work.Kd @ Work.rhs
                    Work.padf = Work.lhs[pvf]

                def StoreStep(Sol, lv, pad, i, Work, pvd, pvf):
                    Sol.reldisp[:, i] = Work.lhs[pvd]
                    Sol.fpad[:, i] = Work.padf
                    lvf = lv.phi.T @ Work.padf
                    padf = -pad.phi.T @ Work.padf
                    # final lv & pad solutions for i'th time step:
                    lv.gen.send((-1, lvf))
                    pad.gen.send((-1, padf))

                def StoreStep2(Sol, lv, pad, i, Work, pvd, pvf):
                    Sol.reldisp[:, i] = Work.lhs[pvd]

                # while bolted together:
                for i in range(1, conn):
                    SolveStep(lv, pad, i, Work, pvd, pvf)
                    StoreStep(Sol, lv, pad, i, Work, pvd, pvf)

                if conn < L:
                    # separate from pad when for goes into tension
                    # (negative):
                    if Work.padf > 0:
                        # in compression, run until it goes into
                        # tension:
                        for i in range(conn, L):
                            SolveStep(lv, pad, i, Work, pvd, pvf)
                            if Work.padf <= 0.0:
                                break  # i'th time step will be redone
                            StoreStep(Sol, lv, pad, i, Work, pvd, pvf)

                    # separate:
                    Work.constraints = np.zeros(r)
                    Work.K[r:, :r] = np.diag(Work.constraints)
                    Work.K[r:, r:] = np.diag(1.0 - Work.constraints)
                    Work.Kd = la.inv(Work.K)
                    for i in range(i, L):
                        SolveStep(lv, pad, i, Work, pvd, pvf)
                        StoreStep2(Sol, lv, pad, i, Work, pvd, pvf)

                # finalize solution:
                lv.sol = lv.parms.finalize(get_force=True)
                pad.sol = pad.parms.finalize(get_force=True)

                db = lv.phi @ lv.d  # or lv.phi @ lv.sol.d
                dp = pad.phi @ pad.d

                vb = lv.phi @ lv.v
                vp = pad.phi @ pad.v

                ab = lv.phi @ lv.sol.a
                ap = pad.phi @ pad.sol.a

                # check solution:
                # separation occurs after 0.38 seconds
                pv = time < 0.3805
                assert abs(full.iff[pv] - Sol.fpad[0, pv]).max() < 20.0
                assert abs(Sol.fpad[0, ~pv]).max() == 0.0
                assert abs(ap[0, pv] - full.xdd[0, pv]).max() < 20.0
                assert abs(ap[0, ~pv] - full.xdd[0, ~pv]).max() > 200.0
                assert abs(ab[0, pv] - full.xdd[0, pv]).max() < 20.0
                assert abs(ab[0, ~pv] - full.xdd[0, ~pv]).max() > 2000.0

                # velocity is what's being enforced ... :
                assert abs(vp[0, pv] - vb[0, pv]).max() < 1e-10
                assert abs(vp[0, pv] - full.xd[0, pv]).max() < 1e-4
                assert abs(vp[0, ~pv] - full.xd[0, ~pv]).max() > 10.0
                assert abs(vb[0, pv] - full.xd[0, pv]).max() < 1e-4
                assert abs(vb[0, ~pv] - full.xd[0, ~pv]).max() > 100.0

                assert abs(dp[0, pv] - full.x[0, pv]).max() < 1e-5
                assert abs(dp[0, ~pv] - full.x[0, ~pv]).max() > 0.1
                assert abs(db[0, pv] - full.x[0, pv]).max() < 1e-5
                assert abs(db[0, ~pv] - full.x[0, ~pv]).max() > 1.0


def test_newmark_nonlinear():
    r"""
    Model a two-mass system with one linear spring and one nonlinear
    spring. The nonlinear spring is only active when compressed. There is
    a gap of 0.01 units before the spring starts being compressed.

          |--> x1        |--> x2
        |----|    50   |----|
        | 10 |--\/\/\--| 12 |   F(t)
        |    |         |    | =====>
        |----| |-/\/-| |----|
             Kcompression

        F(t) = 5000 * np.cos(2 * np.pi * t + 270 / 180 * np.pi)

    The nonlinear spring force is linearly interpolated according to the
    "lookup" table below. Linear extrapolation is used for displacements
    out of range of the table.
    """

    h = 0.005
    t = np.arange(0, 4 + h / 2, h)
    f = np.zeros((2, t.size))
    f[1] = 5000 * np.cos(2 * np.pi * t + 270 / 180 * np.pi)

    # define interpolation table for force (the lookup value is x1 - x2):
    lookup = np.array([[-10, 0.0], [0.01, 0.0], [5, 200.0], [6, 1000.0], [10, 1500.0]])

    # force transforming lookup value to forces on the masses:
    Tfrc = np.array([[-1.0], [1.0]])

    # turn interpolation table into a function for speed
    ifunc = interp1d(*lookup.T, fill_value="extrapolate")

    def nonlin(d, j, h, ifunc):
        return ifunc(d[[0], j] - d[[1], j])

    # mass and stiffness:
    m = np.diag([10.0, 12.0])
    for k in (np.array([[50.0, -50.0], [-50.0, 50.0]]), np.array([50.0, 50.0])):
        for c_factor in (0.0, 0.1):
            c = c_factor * k

            ts = ode.SolveNewmark(m, c, k, h)
            dct = {"disp": (nonlin, Tfrc, dict(ifunc=ifunc))}
            ts.def_nonlin(dct)
            sol = ts.tsolve(f)

            # run in SolveExp2 via the generator feature:
            ts2 = ode.SolveExp2(m, c, k, h)
            gen, d, v = ts2.generator(len(t), f[:, 0])

            for i in range(1, len(t)):
                if i == 1:
                    dx = d[0, i - 1] - d[1, i - 1]
                else:
                    dx = 2 * d[0, i - 1] - d[0, i - 2] + d[1, i - 2] - 2 * d[1, i - 1]
                f_nl = Tfrc @ ifunc([dx])
                gen.send((i, f[:, i] + f_nl))

            sol2 = ts2.finalize()

            diffs = (
                abs(sol.d - sol2.d).max(),
                abs(sol.v - sol2.v).max(),
                abs(sol.a - sol2.a).max(),
            )
            assert np.all(diffs < (0.1, 0.1, 5.0))


# nastran data for next test:
def get_nas1():
    """
    TABLED1 20
            -10.    340.0   .01     55.0    10.     500.    ENDT
    """
    return {
        "A": {
            2: np.array(
                [
                    -4.661479,
                    -2.443206,
                    2.512486,
                    10.88762,
                    23.10986,
                    38.49422,
                    55.35503,
                    71.37072,
                    84.11267,
                    91.60832,
                    92.80007,
                    87.78738,
                    77.79242,
                    64.85818,
                    51.35362,
                    39.40809,
                    30.41431,
                    24.72234,
                    21.59903,
                    19.46215,
                    16.32969,
                    10.37197,
                    0.428369,
                    -30.37465,
                    -109.0221,
                    -186.2092,
                    -243.2773,
                    -263.7413,
                    -237.6241,
                    -164.4552,
                    -54.13656,
                    13.22423,
                    41.3247,
                    66.89066,
                    87.63401,
                    102.1712,
                    110.2735,
                    112.8402,
                    111.6054,
                    108.6502,
                    105.8443,
                    104.3557,
                    104.3498,
                    104.9534,
                    104.4927,
                    100.9445,
                    92.48921,
                    78.02777,
                    57.53166,
                    32.13886,
                    3.969033,
                ]
            ),
            12: np.array(
                [
                    70.84612,
                    186.2409,
                    320.6714,
                    372.3601,
                    326.4095,
                    192.2124,
                    1.177288,
                    -200.998,
                    -365.5927,
                    -452.8844,
                    -441.9509,
                    -335.83,
                    -160.7594,
                    40.31347,
                    218.3474,
                    330.3221,
                    349.8557,
                    273.6989,
                    122.4999,
                    -64.34845,
                    -238.5024,
                    -354.6795,
                    -381.9147,
                    -297.1463,
                    -92.12619,
                    157.2241,
                    389.587,
                    545.4805,
                    582.1889,
                    484.7785,
                    270.4245,
                    36.75019,
                    -175.6288,
                    -351.0514,
                    -449.5287,
                    -449.8634,
                    -354.816,
                    -190.3519,
                    0.8427056,
                    169.9694,
                    274.2271,
                    287.4238,
                    206.4662,
                    52.12039,
                    -136.1432,
                    -309.9461,
                    -424.0145,
                    -447.4351,
                    -371.3693,
                    -211.3144,
                    -3.351435,
                ]
            ),
        },
        "D": {
            2: np.array(
                [
                    0.00000000e00,
                    -2.98334600e-02,
                    -7.53034500e-02,
                    -1.04693500e-01,
                    -6.44028500e-02,
                    1.23790900e-01,
                    5.58347600e-01,
                    1.34717700e00,
                    2.59277800e00,
                    4.37670100e00,
                    6.74691700e00,
                    9.71105300e00,
                    1.32370300e01,
                    1.72608800e01,
                    2.16998200e01,
                    2.64674200e01,
                    3.14872300e01,
                    3.67017000e01,
                    4.20743900e01,
                    4.75853100e01,
                    5.32207900e01,
                    5.89607800e01,
                    6.47671500e01,
                    7.05762600e01,
                    7.61909800e01,
                    8.11079500e01,
                    8.48331800e01,
                    8.70014500e01,
                    8.74817600e01,
                    8.64412800e01,
                    8.43482900e01,
                    8.19088300e01,
                    7.95539900e01,
                    7.74636500e01,
                    7.58013900e01,
                    7.47000000e01,
                    7.42525000e01,
                    7.45107500e01,
                    7.54911800e01,
                    7.71858800e01,
                    7.95759500e01,
                    8.26434200e01,
                    8.63787700e01,
                    9.07819500e01,
                    9.58568300e01,
                    1.01600500e02,
                    1.07990200e02,
                    1.14971800e02,
                    1.22452800e02,
                    1.30302000e02,
                    1.38356800e02,
                ]
            ),
            12: np.array(
                [
                    0.0,
                    0.4534151,
                    2.098772,
                    5.796426,
                    11.87718,
                    20.04696,
                    29.4469,
                    38.85437,
                    46.97546,
                    52.75676,
                    55.63959,
                    55.69393,
                    53.59897,
                    50.47514,
                    47.60932,
                    46.14093,
                    46.78659,
                    49.67133,
                    54.30775,
                    59.72816,
                    64.73674,
                    68.21891,
                    69.43113,
                    68.19909,
                    65.06532,
                    61.34194,
                    58.62479,
                    58.401,
                    61.66829,
                    68.66158,
                    78.75746,
                    90.58406,
                    102.6459,
                    113.5836,
                    122.2747,
                    128.0887,
                    131.0237,
                    131.6878,
                    131.1336,
                    130.5849,
                    131.1239,
                    133.418,
                    137.5517,
                    143.0067,
                    148.7953,
                    153.7125,
                    156.6461,
                    156.866,
                    154.2224,
                    149.2019,
                    142.8291,
                ]
            ),
        },
        "N": {
            122: np.array(
                [
                    -55.28471,
                    -69.04354,
                    -117.184,
                    -223.2986,
                    -395.28,
                    -622.5279,
                    -877.786,
                    -1123.172,
                    -1318.927,
                    -1432.739,
                    -1447.334,
                    -1364.488,
                    -1204.451,
                    -1000.946,
                    -792.968,
                    -615.4196,
                    -490.8809,
                    -424.55,
                    -403.5872,
                    -401.0102,
                    -383.1614,
                    -318.8778,
                    -188.0752,
                    -160.4447,
                    -550.142,
                    -935.0226,
                    -1221.995,
                    -1328.548,
                    -1204.404,
                    -846.5431,
                    -303.5955,
                    -302.2818,
                    -712.7452,
                    -1083.676,
                    -1378.45,
                    -1575.344,
                    -1671.646,
                    -1683.202,
                    -1639.51,
                    -1575.636,
                    -1522.935,
                    -1500.916,
                    -1512.255,
                    -1542.203,
                    -1562.522,
                    -1538.994,
                    -1440.594,
                    -1248.078,
                    -959.8135,
                    -593.3954,
                    -182.6158,
                ]
            )
        },
        "V": {
            2: np.array(
                [
                    0.00000000e00,
                    -4.70646600e-01,
                    -4.67875400e-01,
                    6.81287300e-02,
                    1.42802800e00,
                    3.89219100e00,
                    7.64616100e00,
                    1.27151900e01,
                    1.89345300e01,
                    2.59633700e01,
                    3.33397000e01,
                    4.05632000e01,
                    4.71863900e01,
                    5.28924200e01,
                    5.75408900e01,
                    6.11713600e01,
                    6.39642500e01,
                    6.61697200e01,
                    6.80225800e01,
                    6.96650200e01,
                    7.10966900e01,
                    7.21647600e01,
                    7.25967700e01,
                    7.13989300e01,
                    6.58230500e01,
                    5.40138000e01,
                    3.68343400e01,
                    1.65536000e01,
                    -3.50101600e00,
                    -1.95841800e01,
                    -2.83278500e01,
                    -2.99643500e01,
                    -2.77823900e01,
                    -2.34537800e01,
                    -1.72727900e01,
                    -9.68058000e00,
                    -1.18279100e00,
                    7.74175800e00,
                    1.67195900e01,
                    2.55298100e01,
                    3.41095900e01,
                    4.25176000e01,
                    5.08658100e01,
                    5.92379400e01,
                    6.76157800e01,
                    7.58332700e01,
                    8.35706200e01,
                    9.03913000e01,
                    9.58136700e01,
                    9.94005000e01,
                    1.00844800e02,
                ]
            ),
            12: np.array(
                [
                    0.00000000e00,
                    1.31173200e01,
                    3.33938200e01,
                    6.11150800e01,
                    8.90658600e01,
                    1.09810700e02,
                    1.17546300e02,
                    1.09553500e02,
                    8.68898700e01,
                    5.41507800e01,
                    1.83573700e01,
                    -1.27538700e01,
                    -3.26174400e01,
                    -3.74352800e01,
                    -2.70888500e01,
                    -5.14207100e00,
                    2.20650300e01,
                    4.70072100e01,
                    6.28551600e01,
                    6.51812200e01,
                    5.30671800e01,
                    2.93399100e01,
                    -1.23861700e-01,
                    -2.72863000e01,
                    -4.28572000e01,
                    -4.02532800e01,
                    -1.83808400e01,
                    1.90218600e01,
                    6.41286400e01,
                    1.06807300e02,
                    1.37015500e02,
                    1.49302400e02,
                    1.43747300e02,
                    1.22680100e02,
                    9.06568900e01,
                    5.46812000e01,
                    2.24940200e01,
                    6.87308500e-01,
                    -6.89305800e00,
                    -6.05730600e-02,
                    1.77072900e01,
                    4.01733200e01,
                    5.99289200e01,
                    7.02723900e01,
                    6.69114800e01,
                    4.90679100e01,
                    1.97094800e01,
                    -1.51485000e01,
                    -4.79006800e01,
                    -7.12080300e01,
                    -7.97946600e01,
                ]
            ),
        },
    }


def test_newmark_nonlinear2():
    # mass and stiffness:
    m = np.diag([10.1321, 12, 0])
    k = np.array([[50, -50, 0], [-50, 50, 0], [-1, 1, 1]])

    h = 0.08
    t = np.arange(0, 4 + h / 2, h)
    f = np.zeros((3, t.size))
    f[1] = 5000 * np.cos(2 * np.pi * t + 270 / 180 * np.pi)

    # epoint 22 (dof 3) is the dof to be quantized:
    Tquant = np.array([[0, 0, 1]])

    # output of quantization is turned into a pos force on 2, neg force on
    # 12:
    Tfrc = np.array([[1], [-1], [0]])

    # get lookup table for disp-to-force:
    nas = get_nas1()
    lookup = np.array([[-10, -340.0], [0.01, -55.0], [10, -500.0]])

    # test pyyeti's version:
    ifunc = interp1d(*lookup.T, fill_value="extrapolate")

    def nonlin(d, j, h, ifunc):
        return ifunc(d[[2], j])

    ts = ode.SolveNewmark(m, 0 * m, k, h)
    dct = {"disp": (nonlin, Tfrc, dict(ifunc=ifunc))}
    ts.def_nonlin(dct)
    sol = ts.tsolve(f)

    # compare to nastran:
    def _get_max_err(dct, x):
        return abs(np.vstack((dct[2], dct[12])) - x[:2]).max()

    for x in "dva":
        assert _get_max_err(nas[x.upper()], getattr(sol, x)) < 0.001

    assert abs(nas["N"][122] - sol.z["disp"][0]).max() < 0.001

    # test no optargs:
    def nonlin2(d, j, h):
        return ifunc(d[[2], j])

    ts = ode.SolveNewmark(m, 0 * m, k, h)
    ts.def_nonlin({"disp": (nonlin2, Tfrc)})
    sol = ts.tsolve(f)

    # compare to nastran:
    def _get_max_err(dct, x):
        return abs(np.vstack((dct[2], dct[12])) - x[:2]).max()

    for x in "dva":
        assert _get_max_err(nas[x.upper()], getattr(sol, x)) < 0.001

    assert abs(nas["N"][122] - sol.z["disp"][0]).max() < 0.001


def get_nas2():
    return {
        "A": {
            2: np.array(
                [
                    9.199976,
                    -83.1228,
                    -190.6367,
                    -253.2929,
                    -377.5045,
                    -324.6692,
                    -178.5038,
                    -119.7888,
                    -81.57825,
                    -44.53752,
                    -10.45112,
                    19.9711,
                    47.07952,
                    72.00427,
                    96.1829,
                    120.8176,
                    146.4012,
                    172.433,
                    197.3994,
                    219.027,
                    234.7492,
                    288.1437,
                    449.605,
                    510.952,
                    476.4268,
                    428.1939,
                    369.4263,
                    303.5561,
                    233.6989,
                    162.207,
                    -84.9483,
                    -374.0407,
                    -620.0351,
                    -806.9226,
                    -923.1299,
                    -1007.706,
                    -1169.304,
                    -1037.413,
                    -767.9774,
                    -410.2387,
                    -111.9286,
                    -9.549417,
                    97.11478,
                    204.5666,
                    308.5033,
                    404.3333,
                    487.792,
                    555.519,
                    605.4638,
                    637.0312,
                    678.0464,
                ]
            ),
            12: np.array(
                [
                    59.14231,
                    254.362,
                    483.7553,
                    595.4188,
                    664.6649,
                    498.8464,
                    198.6341,
                    -39.59401,
                    -225.693,
                    -337.9308,
                    -354.7717,
                    -278.5699,
                    -134.8272,
                    34.27972,
                    180.4962,
                    261.5846,
                    251.9231,
                    148.9806,
                    -25.93571,
                    -232.8494,
                    -422.9231,
                    -589.2138,
                    -761.1733,
                    -754.2109,
                    -586.4451,
                    -361.5421,
                    -127.7441,
                    66.48768,
                    184.2313,
                    208.9641,
                    296.4402,
                    363.7341,
                    382.7849,
                    386.7456,
                    403.9014,
                    487.252,
                    725.5848,
                    780.8548,
                    743.5112,
                    608.0889,
                    458.1019,
                    383.5987,
                    212.575,
                    -31.98717,
                    -308.3979,
                    -566.1099,
                    -757.7851,
                    -850.6009,
                    -834.0112,
                    -722.0502,
                    -572.5031,
                ]
            ),
        },
        "D": {
            2: np.array(
                [
                    10.0,
                    14.05888,
                    17.58577,
                    19.89259,
                    20.57834,
                    18.84805,
                    15.03989,
                    10.0893,
                    4.372057,
                    -1.867283,
                    -8.391664,
                    -14.98293,
                    -21.44638,
                    -27.60853,
                    -33.30984,
                    -38.39559,
                    -42.7081,
                    -46.08364,
                    -48.35562,
                    -49.36424,
                    -48.97108,
                    -47.07553,
                    -43.33587,
                    -36.71872,
                    -26.83149,
                    -13.89512,
                    1.781684,
                    19.82282,
                    39.80671,
                    61.28628,
                    83.80397,
                    105.778,
                    125.3582,
                    140.9701,
                    151.4177,
                    155.9573,
                    154.0476,
                    144.6543,
                    128.6216,
                    107.6739,
                    84.1006,
                    59.81097,
                    35.46022,
                    11.73101,
                    -10.68897,
                    -31.13454,
                    -48.99237,
                    -63.72833,
                    -74.90897,
                    -82.21465,
                    -85.44332,
                ]
            ),
            12: np.array(
                [
                    -10.0,
                    -13.62149,
                    -15.61506,
                    -14.5126,
                    -9.599458,
                    -0.4324614,
                    11.92715,
                    25.55802,
                    38.93549,
                    50.86853,
                    60.63881,
                    68.13854,
                    73.85544,
                    78.70943,
                    83.78282,
                    90.01138,
                    97.91409,
                    107.4291,
                    117.8976,
                    128.2001,
                    137.0123,
                    143.1179,
                    145.4525,
                    142.9156,
                    135.5517,
                    124.4346,
                    111.0036,
                    96.75504,
                    82.93201,
                    70.28806,
                    58.98148,
                    49.57211,
                    42.49065,
                    37.859,
                    35.70253,
                    36.13103,
                    39.67794,
                    47.8686,
                    61.05672,
                    79.00331,
                    100.8417,
                    125.6119,
                    152.8371,
                    181.4229,
                    209.8039,
                    236.2111,
                    258.9953,
                    276.9296,
                    289.4201,
                    296.5729,
                    299.1046,
                ]
            ),
        },
        "N": {
            122: np.array(
                [
                    554.5546,
                    172.4782,
                    -343.4434,
                    -936.6584,
                    -2427.188,
                    -2413.396,
                    -1693.21,
                    -1995.703,
                    -2539.359,
                    -3056.754,
                    -3520.688,
                    -3921.88,
                    -4268.673,
                    -4582.319,
                    -4889.092,
                    -5211.228,
                    -5559.013,
                    -5926.027,
                    -6288.768,
                    -6610.812,
                    -6850.517,
                    -6496.586,
                    -4754.825,
                    -3669.748,
                    -3178.582,
                    -2493.743,
                    -1664.999,
                    -745.6625,
                    216.8723,
                    1188.421,
                    339.7387,
                    -1058.22,
                    -2245.85,
                    -3147.591,
                    -3709.035,
                    -4378.323,
                    -6331.146,
                    -5865.832,
                    -4564.204,
                    -2831.672,
                    -2031.929,
                    -3428.738,
                    -4897.185,
                    -6386.671,
                    -7833.053,
                    -9167.024,
                    -10324.16,
                    -11254.34,
                    -11928.29,
                    -12339.95,
                    -12223.97,
                ]
            )
        },
        "V": {
            2: np.array(
                [
                    50.0,
                    47.41109,
                    36.4607,
                    18.70352,
                    -6.528369,
                    -34.61531,
                    -54.74223,
                    -66.67394,
                    -74.72862,
                    -79.77325,
                    -81.9728,
                    -81.592,
                    -78.90997,
                    -74.14662,
                    -67.41914,
                    -58.73912,
                    -48.05037,
                    -35.297,
                    -20.5037,
                    -3.846644,
                    14.3044,
                    35.22012,
                    64.73007,
                    103.1524,
                    142.6475,
                    178.8323,
                    210.7371,
                    237.6564,
                    259.1466,
                    274.9829,
                    278.0732,
                    259.7137,
                    219.9506,
                    162.8723,
                    93.67021,
                    16.43679,
                    -70.64361,
                    -158.9123,
                    -231.1279,
                    -278.2565,
                    -299.1432,
                    -304.0023,
                    -300.4998,
                    -288.4325,
                    -267.9097,
                    -239.3962,
                    -203.7112,
                    -161.9788,
                    -115.5395,
                    -65.83967,
                    -13.23656,
                ]
            ),
            12: np.array(
                [
                    -50.0,
                    -35.09414,
                    -5.56944,
                    37.59752,
                    88.00086,
                    134.5413,
                    162.4405,
                    168.8021,
                    158.1907,
                    135.6457,
                    107.9376,
                    82.60394,
                    66.06806,
                    62.04616,
                    70.63719,
                    88.32042,
                    108.8607,
                    124.8969,
                    129.8187,
                    119.4673,
                    93.23637,
                    52.75089,
                    -1.264595,
                    -61.87996,
                    -115.5062,
                    -153.4257,
                    -172.9971,
                    -175.4474,
                    -165.4186,
                    -149.6908,
                    -129.4747,
                    -103.0677,
                    -73.20692,
                    -42.4257,
                    -10.79983,
                    24.8463,
                    73.35977,
                    133.6174,
                    194.592,
                    248.656,
                    291.3036,
                    324.9716,
                    348.8186,
                    356.0421,
                    342.4267,
                    307.4464,
                    254.4906,
                    190.1552,
                    122.7707,
                    60.52821,
                    8.74608,
                ]
            ),
        },
    }


def test_newmark_nonlinear3():
    # mass and stiffness:
    m = np.diag([10.1321, 12, 0])
    k = np.array([[50, -50, 0], [-50, 50, 0], [-1, 1, 1]])  # x3 = x1 - x2

    h = 0.08
    t = np.arange(0, 4 + h / 2, h)
    f = np.zeros((3, t.size))
    f[1] = 5000 * np.cos(2 * np.pi * t + 270 / 180 * np.pi)

    # output of quantization is turned into a pos force on 2, neg
    # force on 12:
    Tfrc = np.array([[1], [-1], [0]])

    # get lookup table for disp-to-force:
    dlookup = np.array([[-10, -340.0], [0.01, -55.0], [10, -500.0]])
    vlookup = np.array(
        [
            [-200.0, -1500.0],
            [-100.0, -1500.0],
            [-0.01, -500.0],
            [0.01, 500.0],
            [100.0, 1500.0],
            [200.0, 1500.0],
        ]
    )

    # test pyyeti's version:
    dfunc = interp1d(*dlookup.T, fill_value="extrapolate")
    vfunc = interp1d(*vlookup.T, fill_value="extrapolate")

    def dnonlin(d, j, h, ifunc):
        return ifunc(d[[2], j])

    def vnonlin(d, j, h, ifunc):
        vj = (d[2, j] - d[2, j - 1]) / h
        return ifunc([vj])

    ts = ode.SolveNewmark(m, 0 * m, k, h)
    nl_dct = {
        "Disp": (dnonlin, Tfrc, dict(ifunc=dfunc)),
        "Velo": (vnonlin, Tfrc, dict(ifunc=vfunc)),
    }
    ts.def_nonlin(nl_dct)
    sol = ts.tsolve(f, d0=[10.0, -10.0, 20.0], v0=[50.0, -50.0, 100.0])

    # compare to nastran:
    nas = get_nas2()

    def _get_max_err(dct, x):
        return abs(np.vstack((dct[2], dct[12])) - x[:2]).max()

    for x in "dva":
        assert _get_max_err(nas[x.upper()], getattr(sol, x)) < 0.001

    z = sol.z["Disp"][0] + sol.z["Velo"][0]
    assert abs(nas["N"][122] - z).max() < 0.01


def run_solvers_cd_as_forces(m, b, k, h, order, rf, f):
    n = k.shape[0]
    d0 = np.random.randn(n)
    v0 = np.random.randn(n)

    se2 = ode.SolveExp2(m, b, k, h, order=order, rf=rf)
    su = ode.SolveUnc(m, b, k, h, order=order, rf=rf)
    sudf = ode.SolveCDF(m, b, k, h, order=order, rf=rf)

    assert sudf.cdforces

    nt = f.shape[1]
    f2 = f / 2
    f4 = f / 4
    sol = {}

    solvers = (("exp2", "unc ", "cdf "), (se2, su, sudf))

    # solve like normal and via the generator:
    for name, slvr in zip(*solvers):
        sol[name] = slvr.tsolve(f, d0, v0)

    for j in range(3):
        solg = {}
        for name, slvr in zip(*solvers):
            gen, d, v = slvr.generator(nt, f[:, 0], d0, v0)
            if j == 0:
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solg[name] = slvr.finalize()
            elif j == 1:
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                    # redo time step with only half of the forces:
                    gen.send((i, f2[:, i]))
                    # send half of the rest:
                    gen.send((-1, f4[:, i]))
                    # send the other half:
                    gen.send((-1, f4[:, i]))
                solg[name] = slvr.finalize()
            else:
                for i in range(1, nt):
                    gen.send((i, f2[:, i]))
                    # send half of the rest:
                    gen.send((-1, f4[:, i]))
                    # redo step:
                    gen.send((i, f[:, i]))
                solg[name] = slvr.finalize()

        # the two answers better be the same:
        for name in solvers[0]:
            for r in "avd":
                resp = getattr(sol[name], r)
                respg = getattr(solg[name], r)
                assert np.allclose(resp, respg, atol=1e-6)

    for r, atol in zip("avd", (0.5, 0.005, 0.00005)):
        resp = getattr(sol["exp2"], r)
        resp2 = getattr(sol["cdf "], r)
        assert np.allclose(resp, resp2, atol=atol)
        resp = getattr(sol["unc "], r)
        assert np.allclose(resp, resp2, atol=atol)

    check_true_derivatives(sol["cdf "], tol=0.05)

    nonrf = np.arange(n)
    mask = np.ones(n, bool)
    mask[rf] = False
    nonrf = nonrf[mask]

    if m is None:
        fr = sol["cdf "].a + b @ sol["cdf "].v + k[:, None] * sol["cdf "].d
    else:
        fr = m[:, None] * sol["cdf "].a + b @ sol["cdf "].v + k[:, None] * sol["cdf "].d

    assert np.allclose(f[nonrf], fr[nonrf])
    assert np.allclose(f[rf], k[rf, None] * sol["cdf "].d[rf])


def test_solveunc_cd_as_force():
    def _ensure_minmax(a, mn, mx):
        sc = (mx - mn) / np.ptp(a)
        return sc * (a - a.min()) + mn

    np.random.seed(0)
    h = 0.0002  # time step
    t = np.arange(0, 0.04, h)  # time vector
    sr = 1 / h

    n = 45

    # forcing function
    minm = 0.8
    maxk = 3.0e5
    frq = np.sqrt(maxk / minm) / (2.0 * np.pi)
    f, filtfreq, filt = dsp.fftfilt(
        dsp.windowends(np.random.randn(len(t), n)), frq.max() * 0.8, nyq=sr / 2, axis=0
    )
    f = f.T

    # things to test:
    # - w/ and w/o rb modes
    # - w/ and w/o rf modes
    # - w/ and w/o damping on rb modes
    # - order 0 and order 1
    # - generator:
    #    - redoing a time step
    #    - send an update force
    #    - sending more than one update force

    for order in (0, 1):
        for el in (0, 8):  # 0 or 8 rb modes
            for rf in ([], np.arange(13)):
                for rbdamp in (False, True):
                    m = _ensure_minmax(np.random.randn(n), minm, 12.0)
                    zeta = _ensure_minmax(np.random.randn(n), 0.01, 1.2)
                    k = _ensure_minmax(np.random.randn(n), 3.0e4, maxk)
                    w = np.sqrt(k / m)
                    b = 2.0 * zeta * w * m

                    # make damping coupled:
                    b = np.diag(b)
                    if rbdamp:
                        b_addon = 0.003 * np.random.randn(n, n)
                        # b_addon = b_addon @ b_addon.T
                        b += b_addon
                    else:
                        b_addon = 0.003 * np.random.randn(n - el, n - el)
                        # b_addon = b_addon @ b_addon.T
                        b[el:, el:] += b_addon

                    run_solvers_cd_as_forces(m, b, k, h, order, rf, f)
                    # this works, but adds no more coverage:
                    # run_solvers_cd_as_forces(
                    #     None, b, k, h, order, rf, f)


def test_ode_pre_eig_diagdamp():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    uset = nas["uset"][se]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    rb = n2p.rbgeom_uset(uset.iloc[b], 3)

    pv = np.any(maa, axis=0)
    q = q[pv]
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]
    baa1 = np.zeros(maa.shape[0])
    baa1[q] = 2 * 0.05 * np.sqrt(kaa[q, q])
    baa2 = 1e-5 * kaa

    h = 0.001
    t = np.arange(0.0, 0.20, h)
    f = np.zeros((maa.shape[0], len(t)))
    f[b] = 10.0 * rb[:, :1] * np.ones((1, len(t)))
    cnst_steps = 20
    f[b, cnst_steps:] = 10.0 * rb[:, :1] * np.cos(2 * np.pi * 12.0 * t[:-cnst_steps])

    for baa in (baa1, baa2):
        ts = ode.SolveExp2(maa, baa, kaa, h, pre_eig=True)
        sol = ts.tsolve(f, static_ic=True)

        tsu = ode.SolveUnc(maa, baa, kaa, h, pre_eig=True)
        solu = tsu.tsolve(f, static_ic=True)

        assert abs(np.diff(sol.a[:, :cnst_steps], axis=1)).max() < 1e-7
        assert abs(np.diff(solu.a[:, :cnst_steps], axis=1)).max() < 1e-7
        assert abs(maa @ sol.a[:, 0] + kaa @ sol.d[:, 0] - f[:, 0]).max() < 1e-8

        assert np.allclose(sol.a, solu.a, atol=1e-6)
        assert np.allclose(sol.v, solu.v)
        assert np.allclose(sol.d, solu.d)


def test_ode_uncoupled_high_damping():
    # uncoupled equations
    m = np.array([10.0, 30.0, 30.0, 30.0])  # diagonal of mass
    k = np.array([0.0, 6.0e5, 6.0e5, 6.0e5])  # diagonal of stiffness
    zeta = np.array([0.0, 0.05, 1.0, 20000.0])  # percent damping
    b = 2.0 * zeta * np.sqrt(k / m) * m  # diagonal of damping

    h = 0.001  # time step
    t = np.arange(0, 0.3001, h)  # time vector
    c = 2 * np.pi
    f = (
        np.vstack(
            (
                3 * (1 - np.cos(c * 2 * t)),  # forcing function
                4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                6 * (np.cos(np.sqrt(6e5 / 30) * t)),
            )
        )
        * 1.0e4
    )

    sol = ode.SolveUnc(m, b, k, h).tsolve(f)
    new_f = np.diag(m) @ sol.a + np.diag(b) @ sol.v + np.diag(k) @ sol.d
    assert np.allclose(new_f, f)
