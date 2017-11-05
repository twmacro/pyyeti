import scipy.linalg as la
import scipy.signal
import numpy as np
from types import SimpleNamespace
from nose.tools import *
from scipy import integrate
from pyyeti import ode
from pyyeti.ssmodel import SSModel
from pyyeti import expmint


def test_expmint():
    # unittest bug avoidance:
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [.0001, .001, .01, .05, .1]:
        e, i, i2 = expmint.expmint(A, h, geti2=True)
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)

    # these last 2 use the power series expansion for I2 ... check
    # for the warning
    for h in [.2, 1]:
        with assert_warns(RuntimeWarning) as cm:
            e, i, i2 = expmint.expmint(A, h, geti2=True)
        the_warning = str(cm.warning)
        assert 0 == the_warning.find('Using power series expansion')
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)


def test_expmint2():
    A = np.random.randn(50, 50)
    for h in [.0001, .001, .01, .05, .1, .2, 1]:
        e, i, i2 = expmint.expmint(A, h, geti2=True)
        et, it, i2t = expmint.expmint_pow(A, h)
        assert np.allclose(et, e)
        assert np.allclose(it, i)
        assert np.allclose(i2t, i2)


def test_getEPQ1_2_order0():
    order = 0
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [.0001, .001, .01, .05, .1, .2, 1]:
        e, p, q = expmint.getEPQ1(A, h, order=order)
        et, pt, qt = expmint.getEPQ2(A, h, order=order)
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)


def test_getEPQ1_2_order1():
    order = 1
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for h in [.0001, .001, .01, .05, .1]:
        e, p, q = expmint.getEPQ1(A, h, order=order)
        et, pt, qt = expmint.getEPQ2(A, h, order=order)
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)

    # these last 2 use the power series expansion for I2 ... check
    # for the warning
    for h in [.2, 1]:
        e, p, q = expmint.getEPQ2(A, h, order=order)
        with assert_warns(RuntimeWarning) as cm:
            et, pt, qt = expmint.getEPQ1(A, h, order=order)
        the_warning = str(cm.warning)
        assert 0 == the_warning.find('Using power series expansion')
        assert np.allclose(e, et)
        assert np.allclose(p, pt)
        assert np.allclose(q, qt)


def test_getEPQ1():
    for order in (0, 1):
        A = np.random.randn(50, 50)
        for h in [.0001, .001, .01, .05, .1, .2, 1]:
            e, p, q = expmint.getEPQ1(A, h, order=order)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def test_getEPQ():
    for order in (0, 1):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        for h in [.0001, .001, .01, .05, .1, .2, 1]:
            e, p, q = expmint.getEPQ(A, h, order=order)
            et, pt, qt = expmint.getEPQ2(A, h, order=order)
            assert np.allclose(e, et)
            assert np.allclose(p, pt)
            assert np.allclose(q, qt)


def test_getEPQ_half():
    for order in (0, 1):
        A = np.random.randn(50, 50)
        for h in [.001]:
            e, p, q = expmint.getEPQ(A, h, order=order, half=True)
            e1, p1, q1 = expmint.getEPQ1(A, h, order=order, half=True)
            e2, p2, q2 = expmint.getEPQ2(A, h, order=order, half=True)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order,
                                            half=True)
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
        for h in [.001]:
            e, p, q = expmint.getEPQ(A, h, order=order, B=B)
            e1, p1, q1 = expmint.getEPQ1(A, h, order=order, B=B)
            e2, p2, q2 = expmint.getEPQ2(A, h, order=order, B=B)
            et, pt, qt = expmint.getEPQ_pow(A, h, order=order,
                                            B=B)
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
    d = integrate.cumtrapz(sol.v, sol.t, initial=0, axis=1)
    v = integrate.cumtrapz(sol.a, sol.t, initial=0, axis=1)
    derr = abs((d + sol.d[:, :1]) - sol.d).max() / abs(sol.d).max()
    verr = abs((v + sol.v[:, :1]) - sol.v).max() / abs(sol.v).max()
    assert derr < 5.e-3
    assert verr < 5.e-3


def test_ode_ic():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 1.2])    # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    b[0] = 2 * b[1]  # add damping on rb modes for test

    h = 0.0002                             # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

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


def test_rbdamped_modes_coupled():
    N = 10
    win = 100
    m = np.random.randn(N, N)
    m = m.T @ m
    k = np.zeros(N)
    h = 0.001
    t = np.arange(int(1 / h)) * h
    f = np.zeros((N, len(t)))
    f[:, :win] = np.ones((N, 1)) * np.hanning(win) * 20

    for b in (np.zeros(N), 10 * np.ones(N)):
        se2 = ode.SolveExp2(m, b, k, h)
        su = ode.SolveUnc(m, b, k, h)  # , rb=[])
        sole = se2.tsolve(f)
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

    # unittest bug avoidance:
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    with assert_warns(RuntimeWarning) as cm:
        ode.SolveUnc(m, b * 0, k, h, rb=[])
    found = False
    for w in cm.warnings:
        if str(w.message).find(
                'eigenvectors for the state-space '
                'formulation are poorly conditioned') > -1:
            found = True
    assert found


def get_rfsol(k, rf, f):
    if k.ndim > 1:
        if np.size(rf) > 1:
            krf = k[np.ix_(rf, rf)]
            rfsol = la.solve(krf, f[rf, :]).T
        else:
            krf = k[rf, rf]
            rfsol = f[rf, :] / krf
    else:
        krf = k[rf]
        if np.size(krf) > 1:
            rfsol = (f[rf, :] / krf[:, None]).T
        else:
            rfsol = f[rf, :] / krf
    return rfsol


def test_ode_uncoupled():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                               t, X0=ic,
                                               interp=order)
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
                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_ / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip((None, None, 3, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                               t, X0=ic,
                                               interp=order)
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
                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
            v2[rf] = 0.
            v2[:, rf] = 0.
            v2[rf, rf] = vrf
        else:
            v2 = v.copy()
        out.append(v2)
    return out


def test_ode_coupled():
    # coupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            if rf is 2 and k.ndim > 1:
                k = np.diag(k)
            if rf is 3 and m.ndim > 1:
                m = np.diag(m)
                b = np.diag(b)
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_ / m) * m             # diagonal of damping
    m = np.diag(m)
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    m += np.random.randn(4, 4)
    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf, kmult in zip((None, None, 3, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

                assert np.allclose(sol.a, solu.a)
                assert np.allclose(sol.v, solu.v)
                assert np.allclose(sol.d, solu.d)

                assert np.allclose(sol0.a, solu.a[:, :1])
                assert np.allclose(sol0.v, solu.v[:, :1])
                assert np.allclose(sol0.d, solu.d[:, :1])

                assert np.allclose(solu0.a, solu.a[:, :1])
                assert np.allclose(solu0.v, solu.v[:, :1])
                assert np.allclose(solu0.d, solu.d[:, :1])


def test_ode_uncoupled_mNone():
    # uncoupled equations
    m = None
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)                 # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1, 1, 0):
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(b)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                f2 = f
                tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                               t, X0=ic,
                                               interp=order)
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
                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_)                # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1, 1, 0):
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip((None, None, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                A = ode.make_A(m, b, k)
                n = len(b)
                Z = np.zeros((n, n), float)
                B = np.vstack((np.eye(n), Z))
                C = np.vstack((A, np.hstack((Z, np.eye(n)))))
                D = np.vstack((B, Z))
                ic = np.hstack((sol.v[:, 0], sol.d[:, 0]))
                f2 = f
                tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                               t, X0=ic,
                                               interp=order)
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
                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)                 # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf in (None, 3, 2, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_)                # diagonal of damping
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf, kmult in zip((None, None, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                if kmult != 0.0:
                    tsu = ode.SolveUnc(m, b, k, h,
                                       order=order, rf=rf, rb=[])
                else:
                    tsu = ode.SolveUnc(m, b, k, h,
                                       order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    k = np.array([6.e5, 6.e5, 6.e5, 0.])   # diagonal of stiffness
    zeta = np.array([.05, 1., 2., 0.])     # percent damping
    b = 2. * zeta * np.sqrt(k)                 # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[:-1, :-1] += np.random.randn(3, 3) * 1000
    b[:-1, :-1] += np.random.randn(3, 3)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   3 * (1 - np.cos(c * 2 * t)))) * 1.e4

    rb = 3
    for order in (0, 1):
        for rf in ([], 2, 1, np.array([0, 1, 2])):
            if order == 1 and rf is 1:
                k = np.diag(k)
            for static_ic in (0, 1):
                ts = ode.SolveExp2(m, b, k, h,
                                   order=order, rf=rf)
                sol = ts.tsolve(f, static_ic=static_ic)
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf, rb=rb)
                if tsu.ksize > 0:
                    assert tsu.pc.ur.shape[0] > tsu.pc.ur.shape[1]
                solu = tsu.tsolve(f, static_ic=static_ic)

                ts0 = ode.SolveExp2(m, b, k, h=None,
                                    order=order, rf=rf)
                sol0 = ts0.tsolve(f[:, :1], static_ic=static_ic)
                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf, rb=rb)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
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
                    tl, yl, xl = scipy.signal.lsim((A, B, C, D), f2.T,
                                                   t, X0=ic,
                                                   interp=order)
                    tse1 = ode.SolveExp1(A, h, order=order)
                    sole1 = tse1.tsolve(B.dot(f2), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2)
                    assert np.allclose(yl, yl2.T)

                    tse1 = ode.SolveExp1(A, h=None)
                    sole1 = tse1.tsolve(B.dot(f2[:, :1]), d0=ic)
                    yl2 = C.dot(sole1.d) + D.dot(f2[:, :1])
                    assert np.allclose(yl[:1, :], yl2.T)

                    yl[:, rf] = 0.
                    yl[:, rf + n] = 0.
                    yl[:, rf + 2 * n] = rfsol
                assert np.allclose(yl[:, :n], sol.a.T)
                assert np.allclose(yl[:, n:2 * n], sol.v.T)
                assert np.allclose(yl[:, 2 * n:], sol.d.T)

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
    assert repr(s) == ('SSModel(\nA=array([[1]]),\nB=array([[2]]),\n'
                       'C=array([[3]]),\nD=array([[4]]),\nh=None,\n'
                       'method=None,\nprewarp=None\n)')


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
    f = 5           # 5 hz oscillator
    w = 2 * np.pi * f
    w2 = w * w
    zeta = .05
    h = .01
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
    z = ssmodel.c2d(h=h, method='zoh')
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
    h = .01
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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m += np.random.randn(4, 4)
    k += np.random.randn(4, 4) * 1000
    b += np.random.randn(4, 4)
    h = .001                               # time step

    A = ode.make_A(m, b, k)
    n = len(m)
    Z = np.zeros((n, n), float)
    B = np.vstack((np.eye(n), Z))
    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
    D = np.vstack((B, Z))
    sys = (A, B, C, D)
    ss_sys = SSModel(*sys)

    sysz = scipy.signal.cont2discrete(sys, h, method='zoh')
    ss_sysz = ss_sys.c2d(h, method='zoh')
    # assert np.allclose(ss_sysz.C.dot(ss_sysz.A.dot(ss_sysz.B)),
    #                    sysz[2].dot(sysz[0].dot(sysz[1])))
    # assert np.allclose(ss_sysz.D, sysz[3])
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)

    sysz = scipy.signal.cont2discrete(sys, h, method='bilinear')
    ss_sysz = ss_sys.c2d(h, method='tustin')
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)


def test_zoha_c2d_d2c():
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m += np.random.randn(4, 4)
    k += np.random.randn(4, 4) * 1000
    b += np.random.randn(4, 4)
    h = .001                               # time step

    A = ode.make_A(m, b, k)
    n = len(m)
    Z = np.zeros((n, n), float)
    B = np.vstack((np.eye(n), Z))
    C = np.vstack((A, np.hstack((Z, np.eye(n)))))
    D = np.vstack((B, Z))
    sys = (A, B, C, D)
    ss_sys = SSModel(*sys)
    za = ss_sys.c2d(h, method='zoha')
    chk_inverse(za, ss_sys)

    r, c = za.B.shape
    yr = za.C.shape[0]
    nt = 301
    u = np.random.randn(c, nt)
    u[:, 0] = 0   # don't let initial conditions mess us up
    x1 = np.zeros((r, nt + 1), float)
    y1 = np.zeros((yr, nt), float)
    for j in range(nt):
        x1[:, j + 1] = za.A.dot(x1[:, j]) + za.B.dot(u[:, j])
        y1[:, j] = za.C.dot(x1[:, j]) + za.D.dot(u[:, j])

    ts = ode.SolveExp1(A, h, order=1)
    F = B.dot(u)
    PQF = np.copy((ts.P + ts.Q).dot((F[:, :-1] +
                                     F[:, 1:]) / 2), order='F')
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
    zeta = .7
    w2 = w * w
    damp = 2 * w * zeta
    A = np.array([[0, 1], [-w2, -damp]])
    B = np.array([[0], [w2]])
    C = np.array([[1, 0], [0, 1], [-w2, -damp]])
    D = np.array([[0], [0], [w2]])
    h = .01
    prewarp = 50.5

    # answer from Matlab:
    Am = np.array([[0.958796353566434, 0.008171399133845],
                   [-8.064847685445468, 0.599399448728309]])
    Bm = np.array([[0.041203646433566], [8.064847685445468]])
    Cm = 1.e2 * np.array([[0.009793981767832,  0.000040856995669],
                          [-0.040324238427227,  0.007996997243642],
                          [-7.892719919134404, -0.392050547506857]])
    Dm = 1.e2 * np.array([[0.000206018232168],
                          [0.040324238427227],
                          [7.892719919134404]])
    sys = (A, B, C, D)
    sysz = (Am, Bm, Cm, Dm)
    ss_sys = SSModel(*sys)
    ss_sysz = ss_sys.c2d(h, method='tustin', prewarp=prewarp)
    assert runsim(ss_sysz, sysz)
    chk_inverse(ss_sysz, ss_sys)


def test_foh_c2d_d2c():
    # engine actuator dynamics (5 hz, 70% damping):
    # input:   actuator command
    # output:  [actuator angle, angular rate, angular acceleration]'
    w = 5 * 2 * np.pi
    zeta = .7
    w2 = w * w
    damp = 2 * w * zeta
    A = np.array([[0, 1], [-w2, -damp]])
    B = np.array([[0], [w2]])
    C = np.array([[1, 0], [0, 1], [-w2, -damp]])
    D = np.array([[0], [0], [w2]])
    h = .005

    # answer from Matlab:
    Am = np.array([[0.988542968353218, 0.004469980285414],
                   [-4.411693709770504, 0.791942967184347]])
    Bm = np.array([[0.021654992049978], [3.917784066760780]])
    Cm = 1.e2 * np.array([[0.01, 0.0],
                          [0.0, 0.01],
                          [-9.869604401089358, -0.439822971502571]])
    Dm = 1.e2 * np.array([[0.000038911226114],
                          [0.022914063293564],
                          [8.823387419541009]])
    sys = (A, B, C, D)
    sysz = (Am, Bm, Cm, Dm)
    ss_sys = SSModel(*sys)
    ss_sysz = ss_sys.c2d(h, method='foh')
    assert runsim(ss_sysz, sysz)

    assert_raises(ValueError, ss_sys.d2c)
    assert_raises(ValueError, ss_sysz.c2d, h)
    assert_raises(ValueError, ss_sys.c2d, h, method='badmethod')
    assert_raises(ValueError, ss_sysz.d2c, method='badmethod')
    chk_inverse(ss_sysz, ss_sys)


def test_eigss():
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, .05, 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    A = ode.make_A(m, b, k)
    luud = ode.eigss(A, delcc=0)
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
    assert_raises(ValueError, ode.addconj, lam, ur / 2, uri)
    assert_raises(ValueError, ode.addconj,
                  np.hstack((lam, lam[-1])), ur, uri)
    lam = lam[[1, 2, 3, 4, 0]]
    assert_raises(ValueError, ode.addconj, lam, 2 * ur, uri)

    lam, ur, uri = luu2
    urfake = np.vstack((ur, ur[:1]))
    luu3 = ode.addconj(lam, urfake, uri)
    assert luu3[0] is lam
    assert luu3[1] is urfake
    assert luu3[2] is uri


def test_getfsucoef():
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    rf = 3
    s = ode.get_su_coef(m, b, k, h, rfmodes=rf)
    nt = f.shape[1]
    d = np.zeros((4, nt), float)
    v = np.zeros((4, nt), float)
    P = f
    d[rf, 0] = f[rf, 0] / k[rf]
    for j in range(nt - 1):
        d[:, j + 1] = (s.F * d[:, j] + s.G * v[:, j] +
                       s.A * P[:, j] + s.B * P[:, j + 1])
        v[:, j + 1] = (s.Fp * d[:, j] + s.Gp * v[:, j] +
                       s.Ap * P[:, j] + s.Bp * P[:, j + 1])
    ts = ode.SolveUnc(m, b, k, h, rf=rf)
    sol = ts.tsolve(f, static_ic=0)
    assert np.allclose(sol.v, v)
    assert np.allclose(sol.d, d)

    assert_raises(ValueError, ode.get_su_coef, m, b, k, h,
                  rfmodes=[0, 1])
    assert_raises(ValueError, ode.get_su_coef, m, b, k, h,
                  rfmodes=[0, 1, 2, 3])


def test_no_h():
    m = 1
    b = 2 * 35 * .05
    k = 35**2
    A = ode.make_A(m, b, k)
    h = None

    ts1 = ode.SolveExp1(A, h)
    ts2 = ode.SolveExp2(m, b, k, h)
    tsu = ode.SolveUnc(m, b, k, h)

    assert_raises(ValueError, ts1.tsolve, [[1], [0], [0]])
    sol1 = ts1.tsolve([[1], [0]])
    sol2 = ts2.tsolve(1)
    solu = tsu.tsolve(1, static_ic=0)
    assert sol1.v[0, 0] == 1.
    assert sol2.a[0, 0] == 1.
    assert solu.a[0, 0] == 1.

    f = np.random.randn(1, 10)
    f1 = np.vstack((f, np.zeros((1, 10))))

    assert_raises(RuntimeError, ts1.tsolve, f1)
    assert_raises(RuntimeError, ts2.tsolve, f)
    assert_raises(RuntimeError, tsu.tsolve, f)

    ts2 = ode.SolveExp2(m, b, k, h, rf=0)
    tsu = ode.SolveUnc(m, b, k, h, rf=0)

    sol2 = ts2.tsolve(f)
    solu = tsu.tsolve(f)

    assert np.allclose(sol2.d, solu.d)
    assert np.allclose(sol2.v, solu.v)
    assert np.allclose(sol2.a, solu.a)


def test_ode_uncoupled_freq():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    rb = 0

    freq = np.arange(0, 35, .1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (k[:, None] - m[:, None].dot(freqw[None, :]**2) +
         (1j * b[:, None].dot(freqw[None, :])))
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        for incrb in [0, 1, 2]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            d = f / H
            d[rb, 0] = 0
            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0] / m[rb]
            if rf is not None:
                d[rf] = f[rf] / (k[rf][:, None])
                v[rf] = 0
                a[rf] = 0
            if incrb < 2:
                d[rb] = 0
                if incrb < 1:
                    a[rb] = 0
                    v[rb] = 0
            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)


def test_ode_uncoupled_freq_rblast():
    # uncoupled equations
    m = np.array([30., 30., 30., 10])     # diagonal of mass
    k = np.array([6.e5, 6.e5, 6.e5, 0])   # diagonal of stiffness
    zeta = np.array([.05, 1., 2., 0])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    rb = 3

    freq = np.arange(0, 35, .1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (k[:, None] - m[:, None].dot(freqw[None, :]**2) +
         (1j * b[:, None].dot(freqw[None, :])))
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.
    for rf in (None, [2], [1], np.array([0, 1, 2])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        for incrb in [0, 1, 2]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            d = f / H
            d[rb, 0] = 0
            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0] / m[rb]
            if rf is not None:
                d[rf] = f[rf] / (k[rf][:, None])
                v[rf] = 0
                a[rf] = 0
            if incrb < 2:
                d[rb] = 0
                if incrb < 1:
                    a[rb] = 0
                    v[rb] = 0
            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)


def test_ode_uncoupled_freq_mNone():
    # uncoupled equations
    m = None
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)             # diagonal of damping

    rb = 0

    freq = np.arange(0, 35, .1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (k[:, None] - np.ones((4, 1)).dot(freqw[None, :]**2) +
         (1j * b[:, None].dot(freqw[None, :])))
    # to accommodate a freqw of zero:
    H[rb, 0] = 1.
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        for incrb in [0, 1, 2]:
            sol = tsu.fsolve(f, freq, incrb=incrb)
            d = f / H
            d[rb, 0] = 0
            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0]
            if rf is not None:
                d[rf] = f[rf] / (k[rf][:, None])
                v[rf] = 0
                a[rf] = 0
            if incrb < 2:
                d[rb] = 0
                if incrb < 1:
                    a[rb] = 0
                    v[rb] = 0
            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)


def test_ode_coupled_freq():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    rb = 0

    freq = np.arange(0, 35, .5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        for incrb in [0, 1, 2]:
            sol = tsu.fsolve(f, freq, incrb=incrb)

            m2, b2, k2 = decouple_rf((m, b, k), rf)
            d = np.zeros((4, freqw.size), complex)
            for i, w in enumerate(freqw):
                H = (k2 - m2 * w**2) + 1j * (b2 * w)
                if w == 0.:
                    H[rb, 0] = 1.
                d[:, i] = la.solve(H, f[:, i])
                if w == 0.:
                    d[rb, 0] = 0

            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0] / m[rb, rb]
            if rf is not None:
                d[rf] = la.solve(k[np.ix_(rf, rf)], f[rf])
                v[rf] = 0
                a[rf] = 0
            if incrb < 2:
                d[rb] = 0
                if incrb < 1:
                    a[rb] = 0
                    v[rb] = 0
            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)


def test_ode_coupled_freq_mNone():
    # uncoupled equations
    m = None
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)             # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    rb = 0
    freq = np.arange(0, 35, .5)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    for rf in (None, [3], [2], np.array([1, 2, 3])):
        tsu = ode.SolveUnc(m, b, k, rf=rf)
        for incrb in [0, 1, 2]:
            sol = tsu.fsolve(f, freq, incrb=incrb)

            b2, k2 = decouple_rf((b, k), rf)
            d = np.zeros((4, freqw.size), complex)
            for i, w in enumerate(freqw):
                H = (k2 - np.eye(4) * w**2) + 1j * (b2 * w)
                if w == 0.:
                    H[rb, 0] = 1.
                d[:, i] = la.solve(H, f[:, i])
                if w == 0.:
                    d[rb, 0] = 0

            v = 1j * freqw * d
            a = 1j * freqw * v
            a[rb, 0] = f[rb, 0]
            if rf is not None:
                d[rf] = la.solve(k[np.ix_(rf, rf)], f[rf])
                v[rf] = 0
                a[rf] = 0
            if incrb < 2:
                d[rb] = 0
                if incrb < 1:
                    a[rb] = 0
                    v[rb] = 0
            assert np.allclose(a, sol.a)
            assert np.allclose(v, sol.v)
            assert np.allclose(d, sol.d)
            assert np.all(freq == sol.f)


def test_ode_fsd_1():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    freq = np.arange(.1, 35, .1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = (k[:, None] - m[:, None].dot(freqw[None, :]**2) +
         (1j * b[:, None].dot(freqw[None, :])))

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

    solu = tsu.fsolve(f, freq, incrb=1)
    sold = tsd.fsolve(f, freq, incrb=1)
    assert np.allclose(sold.a, solu.a)
    assert np.allclose(sold.v, solu.v)
    assert np.allclose(sold.d, solu.d)
    assert np.all(sold.f == solu.f)
    d[0] = 0
    assert np.allclose(a, solu.a)
    assert np.allclose(v, solu.v)
    assert np.allclose(d, solu.d)

    solu = tsu.fsolve(f, freq, incrb=0)
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
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)             # diagonal of damping

    freq = np.arange(.1, 35, .1)
    f = np.ones((4, freq.size))

    freqw = 2 * np.pi * freq
    H = ((1j * b[:, None].dot(freqw[None, :]) + k[:, None]) -
         freqw[None, :]**2)
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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    freq = np.arange(.5, 35, 2.5)
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
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)            # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    freq = np.arange(.5, 35, 2.5)
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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m            # diagonal of damping

    m = m + 1j * np.random.randn(4)
    k = k + 1j * np.random.randn(4) * 14
    b = b + 1j * np.random.randn(4)

    freq = np.arange(.5, 35, 2.5)
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


def test_ode_fsd_uncoupled_complex_2():
    # uncoupled equations
    m = None
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)            # diagonal of damping

    k = k + 1j * np.random.randn(4) * 14
    b = b + 1j * np.random.randn(4)

    freq = np.arange(.5, 35, 2.5)
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


def test_ode_fsd_coupled_complex_1():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m            # diagonal of damping

    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    m = m + 1j * np.random.randn(4, 4)
    k = k + 1j * np.random.randn(4, 4) * 14
    b = b + 1j * np.random.randn(4, 4)

    freq = np.arange(.5, 35, 2.5)
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


def test_ode_fsd_coupled_complex_2():
    # uncoupled equations
    m = None
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)            # diagonal of damping

    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    k = k + 1j * np.random.randn(4, 4) * 14
    b = b + 1j * np.random.randn(4, 4)

    freq = np.arange(.5, 35, 2.5)
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


def test_ode_complex_coefficients():
    aa = np.ones((2, 2)) * (1. + 1j)
    m = aa.copy()
    m[0, 0] = 3. + 2j
    b = aa.copy()
    b[1, 0] = -2.
    k = aa.copy()
    k[0, 1] = 2.
    h = .001
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
    aa = np.ones((2, 2)) * (1. + 1j)
    m = aa.copy()
    m[0, 0] = 3. + 2j
    b = aa.copy()
    b[1, 0] = -2.
    k = aa.copy()
    k[0, 1] = 2.
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
    krf = 10.
    k[2, 2] = krf

    f = np.array([np.sin(2 * np.pi * 3 * t),
                  np.cos(2 * np.pi * 1 * t),
                  np.sin(2 * np.pi * 2.5 * t)])

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

        v = integrate.cumtrapz(sol.a, sol.t, initial=0)
        d = integrate.cumtrapz(sol.v, sol.t, initial=0)
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
    aa = np.ones((2, 2)) * (1. + 1j)
    m = None
    b = aa.copy()
    b[1, 0] = -2.
    k = aa.copy()
    k[0, 1] = 2.
    h = .001
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
#     with assert_warns(RuntimeWarning) as cm:
#         ode.SolveUnc(m, b, k, h)
#     wrn0 = str(cm.warnings[0].message)
#     assert 0 == wrn0.find('Repeated roots detected')
#     found = False
#     for w in cm.warnings[1:]:
#         if str(w.message).find('found 2 rigid-body modes') > -1:
#             found = True
#     assert found


def test_ode_complex_coefficients_rb():
    aa = np.ones((2, 2)) * (1. + 1j)
    m = aa.copy()
    m[0, 0] = 3. + 2j
    b = aa.copy()
    b[1, 0] = -2.
    k = aa.copy()
    k[0, 1] = 2.
    b[0, 0] = 0.
    k[0, 0] = 0.
    h = .001
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
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')

    # setup mass, stiffness, damping:
    m = None  # treated as identity
    k = nas['lambda'][0]   # imperfect zeros
    zeta = 1000  # VERY high damping, for testing
    # we end up with RB modes with damping ... good for testing
    b = 2 * np.sqrt(abs(k)) * zeta
    # k[:6] = 0.0
    # b[:6] = 1e-4

    # step input, 2 second duration
    h = .001
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
    assert_raises(ValueError, se.tsolve, f)

    # solve:
    sole_nosic = se.tsolve(g, static_ic=0)
    sole_sic = se.tsolve(g, static_ic=1)

    check_true_derivatives(sole_nosic)
    check_true_derivatives(sole_sic)

    fru = (solu_nosic.a + b[:, None] * solu_nosic.v +
           k[:, None] * solu_nosic.d)
    fre = (sole_nosic.a + b[:, None] * sole_nosic.v +
           k[:, None] * sole_nosic.d)

    assert np.allclose(g, fru)
    assert np.allclose(g, fre)

    fru = (solu_sic.a + b[:, None] * solu_sic.v +
           k[:, None] * solu_sic.d)
    fre = (sole_sic.a + b[:, None] * sole_sic.v +
           k[:, None] * sole_sic.d)

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

    freq = np.arange(1., 25.1, .25)
    h = .01
    time = np.arange(0., 2., h)
    M1 = 10.
    M2 = 30.
    M3 = 3.
    M4 = 2.
    c1 = 15.
    c2 = 15.
    c3 = 15.
    k1 = 45000.
    k2 = 25000.
    k3 = 10000.

    MASS = np.array([[M1, 0, 0, 0],
                     [0, M2, 0, 0],
                     [0, 0, M3, 0],
                     [0, 0, 0, M4]])
    DAMP = np.array([[c1, -c1, 0, 0],
                     [-c1, c1 + c2, -c2, 0],
                     [0, -c2, c2 + c3, -c3],
                     [0, 0, -c3, c3]])
    STIF = np.array([[k1, -k1, 0, 0],
                     [-k1, k1 + k2, -k2, 0],
                     [0, -k2, k2 + k3, -k3],
                     [0, 0, -k3, k3]])

    # frequency domain first:
    F = np.vstack((np.ones((1, len(freq))),
                   np.zeros((3, len(freq)))))

    sol1 = (ode.SolveUnc(MASS, DAMP, STIF, pre_eig=True)
            .fsolve(F, freq))
    sol2 = ode.FreqDirect(MASS, DAMP, STIF).fsolve(F, freq)

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)

    # frequency domain with identity M ... as None:
    sol1 = (ode.SolveUnc(None, DAMP, STIF, pre_eig=True)
            .fsolve(F, freq))
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
    F[0] = 0.
    pv = np.where((time > .1) & (time < .8))[0]
    F[0][pv] = 10.
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

    F[0] = 0.
    pv = np.where(time < .8)[0]
    F[0][pv] = 10.
    sol1 = ode.SolveUnc(MASS, DAMP, STIF, h,
                        pre_eig=True).tsolve(F, static_ic=1)
    sol2 = (ode.SolveExp2(MASS, DAMP, STIF, h, pre_eig=True)
            .tsolve(F, static_ic=1))

    assert np.allclose(sol1.d, sol2.d)
    assert np.allclose(sol1.v, sol2.v)
    assert np.allclose(sol1.a, sol2.a)

    STIF2 = np.array([[k1, -k1, 0, 0],
                      [-k1, k1 + k2, -k2, 0],
                      [k1, -k2, k2 + k3, -k3],
                      [k1, 0, -k3, k3]])
    assert_raises(ValueError, ode.SolveUnc, MASS, DAMP, STIF2, h,
                  pre_eig=True)
    assert_raises(ValueError, ode.SolveUnc, None, DAMP, STIF2, h,
                  pre_eig=True)


def test_ode_badsize():
    m = None
    k = np.random.randn(3, 12)
    b = np.random.randn(3, 12)
    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    f = np.random.randn(3, len(t))
    assert_raises(ValueError, ode.SolveExp2, m, b, k, h)
    assert_raises(ValueError, ode.SolveExp2, b[0], b, k, h)
    b1 = np.random.randn(2, 2)
    assert_raises(ValueError, ode.SolveExp2, m, b1, k, h)
    m1 = np.random.randn(3, 3, 3)
    assert_raises(ValueError, ode.SolveExp2, m1, b, k, h)


def test_precalc_warnings():
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([50., 6.e5, 6.e5, 6.e-15])  # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    h = .001                               # time step

    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    assert_warns(RuntimeWarning, ode.SolveUnc,
                 m, b, k, h, rf=[2, 3])

    m = np.array([10e3, 30e-18, 30., 30.])     # diagonal of mass
    assert_warns(RuntimeWarning, ode.SolveUnc,
                 m, b, k, h, rf=3)


def test_ode_solvepsd():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    freq = np.arange(.1, 35, .1)

    forcepsd = 10000 * np.ones((4, freq.size))  # constant PSD forces
    ts = ode.SolveUnc(m, b, k)
    atm = np.random.randn(4, 4)
    dtm = np.random.randn(4, 4)
    t_frc = np.random.randn(4, 4)
    drms = [[atm, None], [None, dtm], [atm, dtm]]
    forcephi = np.random.randn(4, 4)

    rbduf = 1.2
    elduf = 1.5

    rms, psd = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    rmsrb1, psdrb1 = ode.solvepsd(ts, forcepsd, t_frc, freq, drms,
                                  incrb=1)
    rmsrb0, psdrb0 = ode.solvepsd(ts, forcepsd, t_frc, freq, drms,
                                  incrb=0)
    rmsduf, psdduf = ode.solvepsd(ts, forcepsd, t_frc, freq, drms,
                                  rbduf=rbduf, elduf=elduf)
    rmsf, psdf = ode.solvepsd(ts, forcepsd, t_frc, freq, drms)
    rmsphi, psdphi = ode.solvepsd(ts, forcepsd, t_frc, freq, drms,
                                  forcephi=forcephi)
    assert_raises(ValueError, ode.solvepsd, ts, forcepsd, t_frc,
                  freq[:-1], drms)

    # solve by hand for comparison:
    freqw = 2 * np.pi * freq
    H = (k[:, None] - m[:, None].dot(freqw[None, :]**2) +
         (1j * b[:, None].dot(freqw[None, :])))

    dpsd = 0.
    apsd = 0.
    adpsd = 0.
    dpsdduf = 0.
    apsdduf = 0.
    adpsdduf = 0.
    dpsd1 = 0.
    apsd1 = 0.
    adpsd1 = 0.
    dpsd0 = 0.
    apsd0 = 0.
    adpsd0 = 0.
    dpsdphi = 0.
    apsdphi = 0.
    adpsdphi = 0.
    unitforce = np.ones((1, len(freq)))
    for i in range(forcepsd.shape[0]):
        # solve for unit frequency response function:
        genforce = t_frc[i:i + 1].T @ unitforce
        # sol = ts.fsolve(genforce, freq)
        d = genforce / H
        v = 1j * freqw * d
        a = 1j * freqw * v
        dpsd = dpsd + abs(dtm @ d)**2 * forcepsd[i]
        apsd = apsd + abs(atm @ a)**2 * forcepsd[i]
        adpsd = adpsd + abs(atm @ a + dtm @ d)**2 * forcepsd[i]

        F = forcephi[:, i:i + 1] @ unitforce
        dpsdphi = (dpsdphi + abs(dtm @ d - F)**2 * forcepsd[i])
        apsdphi = (apsdphi + abs(atm @ a - F)**2 * forcepsd[i])
        adpsdphi = (adpsdphi + abs(atm @ a + dtm @ d - F)**2 *
                    forcepsd[i])

        dduf = d.copy()
        dduf[0] = dduf[0] * rbduf
        dduf[1:] = dduf[1:] * elduf
        aduf = a.copy()
        aduf[0] = aduf[0] * rbduf
        aduf[1:] = aduf[1:] * elduf
        dpsdduf = dpsdduf + abs(dtm @ dduf)**2 * forcepsd[i]
        apsdduf = apsdduf + abs(atm @ aduf)**2 * forcepsd[i]
        adpsdduf = adpsdduf + abs(atm @ aduf +
                                  dtm @ dduf)**2 * forcepsd[i]

        # incrb = 1
        d[0] = 0
        dpsd1 = dpsd1 + abs(dtm @ d)**2 * forcepsd[i]
        apsd1 = apsd1 + abs(atm @ a)**2 * forcepsd[i]
        adpsd1 = adpsd1 + abs(atm @ a + dtm @ d)**2 * forcepsd[i]

        # incrb = 0
        a[0] = 0
        dpsd0 = dpsd0 + abs(dtm @ d)**2 * forcepsd[i]
        apsd0 = apsd0 + abs(atm @ a)**2 * forcepsd[i]
        adpsd0 = adpsd0 + abs(atm @ a + dtm @ d)**2 * forcepsd[i]

    assert np.allclose(psd[0], apsd)
    assert np.allclose(psd[1], dpsd)
    assert np.allclose(psd[2], adpsd)

    # incrb=1
    assert np.allclose(psdrb1[0], apsd1)
    assert np.allclose(psdrb1[1], dpsd1)
    assert np.allclose(psdrb1[2], adpsd1)

    # incrb=0
    assert np.allclose(psdrb0[0], apsd0)
    assert np.allclose(psdrb0[1], dpsd0)
    assert np.allclose(psdrb0[2], adpsd0)

    # with uncertainty factors
    assert np.allclose(psdduf[0], apsdduf)
    assert np.allclose(psdduf[1], dpsdduf)
    assert np.allclose(psdduf[2], adpsdduf)

    # with the forcephi matrix
    assert np.allclose(psdphi[0], apsdphi)
    assert np.allclose(psdphi[1], dpsdphi)
    assert np.allclose(psdphi[2], adpsdphi)


def test_getmodepart():
    K = [[12312.27,   -38.20,   611.56, -4608.26,  2845.92],
         [-38.20,  3072.44, -1487.68,  3206.59,   746.56],
         [611.56, -1487.68,   800.91, -1718.08,  -164.51],
         [-4608.26,  3206.59, -1718.08,  9189.42, -1890.31],
         [2845.92,   746.56,  -164.51, -1890.31, 10908.62]]
    M = None
    w2, phi = la.eigh(K)
    zetain = np.array([.02, .02, .05, .02, .05])
    Z = np.diag(2 * zetain * np.sqrt(w2))
    mfreq = np.sqrt(w2) / 2 / np.pi

    freq = np.arange(0.1, 15.05, .1)
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

    sols = [[Tmid, sol_bot.a, 'Bot to Mid'],
            [Ttop, sol_bot.a, 'Bot to Top'],
            [Ttop, sol_mid.a, 'Mid to Top']]
    #
    # APPROACH 1:  let getmodepart() do the FRF plotting:
    #
    modes, freqs = ode.getmodepart(freq, sols, mfreq,
                                   ylog=1, idlabel='getmodepart demo 1',
                                   factor=.1, auto=[1, 0])

    mds2, frqs2, r = ode.modeselect('modeselect demo 1', ts,
                                    Tbot.T @ f, freq, Ttop,
                                    'Bot to Top', mfreq,
                                    factor=.1, auto=0)
    # from Yeti:
    modes_sbe = [2, 3]
    freqs_sbe = [13.53671044272239, 15.80726801820284]

    assert np.allclose(modes_sbe, modes)
    assert np.allclose(freqs_sbe, freqs)
    assert np.allclose(modes_sbe, mds2)
    assert np.allclose(freqs_sbe, frqs2)

    import matplotlib.pyplot as plt
    plt.close('all')
    from pyyeti.datacursor import DataCursor

    def getdata(self):
        self.off()
        plt.figure('FRF')
        ax = plt.gca()
        # the next line is preferable for normal use ... however,
        # for more complete test coverage, use the 2nd one:
        # self.on(ax, callbacks=False)
        self.on()
        x, y, n, ind, lineh = self._snap(ax, 7.0, 0.8)
        self._add_point(ax, x, y, n, ind, lineh)
        self.off()
        return 1

    old_fake_getdata = DataCursor._fake_getdata
    DataCursor._fake_getdata = getdata
    mds3, frqs3, r = ode.modeselect('modeselect demo 2', ts,
                                    Tbot.T @ f, freq, Ttop,
                                    'Bot to Top', mfreq,
                                    factor=.1, auto=None)
    plt.close('all')
    DataCursor._fake_getdata = old_fake_getdata

    modes_sbe = [1, 2]
    freqs_sbe = [6.1315401651466273, 13.53671044272239]
    assert np.allclose(modes_sbe, mds3)
    assert np.allclose(freqs_sbe, frqs3)

    # check for some error conditions:
    assert_raises(ValueError, ode.getmodepart, freq, 4, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])

    sols = [[Tmid, sol_bot.a],
            [Ttop, sol_bot.a, 'Bot to Top'],
            [Ttop, sol_mid.a, 'Mid to Top']]
    assert_raises(ValueError, ode.getmodepart, freq, sols, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])

    T = np.vstack((Tmid, Ttop))
    sols = [[T, sol_bot.a, 'Bot to Mid'],
            [Ttop, sol_mid.a, 'Mid to Top']]
    assert_raises(ValueError, ode.getmodepart, freq, sols, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])

    sols = [[T, sol_bot.a, ['Bot to Mid']],
            [Ttop, sol_mid.a, 'Mid to Top']]
    assert_raises(ValueError, ode.getmodepart, freq, sols, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])

    sols = [[Tmid, sol_bot.a, 'Bot to Mid'],
            [Ttop, sol_bot.a, 'Bot to Top'],
            [Ttop, sol_mid.a[:-1, :], 'Mid to Top']]
    assert_raises(ValueError, ode.getmodepart, freq, sols, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])

    sols = [[Tmid, sol_bot.a, ['Bot to Mid', 'bad label']],
            [Ttop, sol_bot.a, 'Bot to Top'],
            [Ttop, sol_mid.a, 'Mid to Top']]
    assert_raises(ValueError, ode.getmodepart, freq, sols, mfreq,
                  ylog=1, idlabel='getmodepart demo 1',
                  factor=.1, auto=[1, 0])


def test_ode_ic_generator():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 1.2])    # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4
    get_force = True
    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k = np.diag(k)
        b = np.diag(b)
        for rf in (None, 3, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)

                nt = 1
                gen, d, v = tsu0.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                nt = 1
                gen, d, v = tsu0.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)

                nt = 1
                gen, d, v = tse0.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                sole2 = tse.finalize(get_force)

                nt = 1
                gen, d, v = tse0.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
    assert_raises(NotImplementedError, tsu.generator,
                  nt, f[:, 0], static_ic=static_ic)

    tse2 = ode.SolveExp2(m, b, k, h, order=order, rf=2)
    assert_raises(NotImplementedError, tse2.generator,
                  nt, f[:, 0], static_ic=static_ic)

    tsu = ode.SolveUnc(m, b, k, h, order=order)
    assert_raises(ValueError, tsu.generator,
                  nt, f[:-1, 0], static_ic=static_ic)

    tsu = ode.SolveUnc(m, b, np.diag(k), h, order=order, pre_eig=True)
    assert_raises(NotImplementedError, tsu.generator,
                  nt, f[:, 0], static_ic=static_ic)


def test_ode_uncoupled_2_generator():
    # uncoupled equations
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_ / m) * m             # diagonal of damping

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    get_force = True
    for order in (0, 1, 1, 0):
        m = np.diag(m)
        k_ = np.diag(k_)
        b_ = np.diag(b_)
        for rf, kmult in zip((None, None, 3, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se2
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k / m) * m             # diagonal of damping
    m = np.diag(m)
    k = np.diag(k)
    b = np.diag(b)

    m[1:, 1:] += np.random.randn(3, 3)
    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf in (None, 3, np.array([1, 2, 3])):
            if rf is 3 and m.ndim > 1:
                m = np.diag(m)
                b = np.diag(b)
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
    m = np.array([10., 30., 30., 30.])     # diagonal of mass
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_ / m) * m             # diagonal of damping
    m = np.diag(m)
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    m += np.random.randn(4, 4)
    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    get_force = True
    for order in (0, 1):
        for rf, kmult in zip((None, None, 3, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                get_force = not get_force
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize(get_force)

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize(get_force)

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize(get_force)

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
    k = np.array([0., 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0., .05, 1., 2.])     # percent damping
    b = 2. * zeta * np.sqrt(k)                 # diagonal of damping
    k = np.diag(k)
    b = np.diag(b)

    k[1:, 1:] += np.random.randn(3, 3) * 1000
    b[1:, 1:] += np.random.randn(3, 3)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf in (None, 3, np.array([1, 2, 3])):
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))
                solu2 = tsu.finalize()

                assert np.allclose(solu2.a, solu.a)
                assert np.allclose(solu2.v, solu.v)
                assert np.allclose(solu2.d, solu.d)

                # se
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))
                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
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
    k_ = np.array([3.e5, 6.e5, 6.e5, 6.e5])   # diagonal of stiffness
    zeta = np.array([0.1, .05, 1., 2.])     # percent damping
    b_ = 2. * zeta * np.sqrt(k_)                # diagonal of damping
    k_ = np.diag(k_)
    b_ = np.diag(b_)

    k_ += np.random.randn(4, 4) * 1000
    b_ += np.random.randn(4, 4)

    h = .001                               # time step
    t = np.arange(0, .3001, h)             # time vector
    c = 2 * np.pi
    f = np.vstack((3 * (1 - np.cos(c * 2 * t)),    # forcing function
                   4 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   5 * (np.cos(np.sqrt(6e5 / 30) * t)),
                   6 * (np.cos(np.sqrt(6e5 / 30) * t)))) * 1.e4

    for order in (0, 1):
        for rf, kmult in zip((None, 3, np.array([0, 1, 2, 3])),
                             (0.0, 1.0, 1.0)):
            k = k_ * kmult
            b = b_ * kmult
            for static_ic in (0, 1):
                # su
                tsu = ode.SolveUnc(m, b, k, h,
                                   order=order, rf=rf)
                solu = tsu.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tsu.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))

                # resolve some time steps for test:
                for i in range(nt - 5, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))

                solu2 = tsu.finalize()

                tsu0 = ode.SolveUnc(m, b, k, h=None,
                                    order=order, rf=rf)
                solu0 = tsu0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tsu0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
                tse = ode.SolveExp2(m, b, k, h,
                                    order=order, rf=rf)
                sole = tse.tsolve(f, static_ic=static_ic)

                nt = f.shape[1]
                gen, d, v = tse.generator(
                    nt, f[:, 0], static_ic=static_ic)
                for i in range(1, nt):
                    gen.send((i, f[:, i]))

                # resolve some time steps for test:
                for i in range(nt - 5, nt):
                    fi = f[:, i] / 2
                    gen.send((i, fi))
                    gen.send((-1, fi))

                sole2 = tse.finalize()

                tse0 = ode.SolveExp2(m, b, k, h=None,
                                     order=order, rf=rf)
                sole0 = tse0.tsolve(f[:, :1], static_ic=static_ic)
                gen, d, v = tse0.generator(
                    1, f[:, 0], static_ic=static_ic)
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
    a = ode._BaseODE()
    assert_raises(NotImplementedError, a.tsolve)
    assert_raises(NotImplementedError, a.fsolve)
    assert_raises(NotImplementedError, a.generator)
    assert_raises(NotImplementedError, a.get_f2x)


def test_henkel_mar():
    # flag specifying whether to use velocity or displacement for
    # the joint compatibility
    use_velo = 1
    # solver = ode.SolveExp2
    sep_time = 0.32   # if <= 0.0, do not separate
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

    for mm in (None, 'other'):
        for c2 in (20.0, 30.0):
            M1 = 30.
            M1_p = 29.
            M1_l = M1 - M1_p
            M2 = 3.
            M3 = 2.
            c1 = 45.
            c3 = 10.
            k1 = 45000.
            k2 = 20000.
            k3 = 10000.

            rfcut = 1e6
            # rfcut = 1.0
            for solver in (ode.SolveUnc, ode.SolveExp2):

                # full system:
                full = SimpleNamespace()
                full.M = np.array([[M1, 0, 0],
                                   [0, M2, 0],
                                   [0, 0, M3]])
                full.K = np.array([[k1 + k2, -k2, 0],
                                   [-k2, k2 + k3, -k3],
                                   [0, -k3, k3]])
                full.C = np.array([[c1 + c2, -c2, 0],
                                   [-c2, c2 + c3, -c3],
                                   [0, -c3, c3]])
                w, u = la.eigh(full.K, full.M)
                full.freq = np.sqrt(w) / (2 * np.pi)
                full.m = None
                full.w = w
                full.u = u
                full.k = w
                full.c = u.T @ full.C @ u

                # setup time vector:
                sr = 1000.
                h = 1 / sr
                time = np.arange(0, 0.5, h)
                L = len(time)
                zeta0 = 0.0

                # solve full system w/o Henkel-Mar:
                physF = np.zeros((1, L))
                physF[0] = np.interp(
                    time,
                    np.array([0.0, 0.05, 0.1, 1.0]),
                    np.array([50000.0, 50000.0, 0.0, 0.0]))

                physF3 = np.array([[0.], [0.], [1.]]) @ physF
                F = full.u.T @ physF3
                rf = (full.k > rfcut).nonzero()[0]
                full.parms = solver(full.m, full.c, full.k, h, rf=rf)

                full.sol = full.parms.tsolve(F, static_ic=True)
                full.x = full.u @ full.sol.d
                x0 = full.x[:, :1]
                full.xd = full.u @ full.sol.v
                full.xdd = full.u @ full.sol.a

                # define i/f force as force on lv. compute iff by
                # looking at pad:
                #   sum of forces on m1 = m a = -iff - k1*x1 - c1*xd1
                # iff = -m a - k1*x1 - c1*xd1
                full.iff = -(M1_p * full.xdd[0] + k1 * full.x[0] +
                             c1 * full.xd[0])

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
                lv.M = np.array([[M1_l, 0., 0.],
                                 [0., M2, 0.],
                                 [0., 0., M3]])
                lv.K = np.array([[k2, -k2, 0.],
                                 [-k2, k2 + k3, -k3],
                                 [0., -k3, k3]])
                lv.C = np.array([[c2, -c2, 0.],
                                 [-c2, c2 + c3, -c3],
                                 [0., -c3, c3]])
                w, u = la.eigh(lv.K, lv.M)
                #print('lv lambda:', w)
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
                qel = (1 / lv.k[nrb:][:, None] *
                       (lv.u[:, nrb:].T @ physF3[:, :1] +
                        lv.phi[:, nrb:].T @ fpad))
                qep = 1 / pad.k[:, None] * (-pad.phi.T @ fpad)
                qrl = la.inv(lv.phi[:, :nrb]) @ (
                    pad.phi @ qep - lv.phi[:, nrb:] @ qel)
                x0_lv = lv.u[:, :nrb] @ qrl + lv.u[:, nrb:] @ qel
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
                    L, lv.force[:, 0], d0=lv.d0.ravel())
                pad.gen, pad.d, pad.v = pad.parms.generator(
                    L, pad.force[:, 0], d0=pad.d0.ravel())

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
                        Work.rhs[pvd] = (lv.phi @ lv.v[:, i] -
                                         pad.phi @ pad.v[:, i])
                    else:
                        Work.rhs[pvd] = (lv.phi @ lv.d[:, i] -
                                         pad.phi @ pad.d[:, i])
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

                db = lv.phi @ lv.d   # or lv.phi @ lv.sol.d
                dp = pad.phi@ pad.d

                vb = lv.phi @ lv.v
                vp = pad.phi@ pad.v

                ab = lv.phi @ lv.sol.a
                ap = pad.phi@ pad.sol.a

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
