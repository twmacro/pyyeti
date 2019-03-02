import numpy as np
import tempfile
import os
from pyyeti import ytools
from nose.tools import *
import scipy.linalg as linalg


# def test_fit_circle_2d():
#     x = np.arange(100)
#     y = np.arange(100)
#
#     import sys
#     for v in list(sys.modules.values()):
#         if getattr(v, '__warningregistry__', None):
#             v.__warningregistry__ = {}
#
#     assert_warns(RuntimeWarning, ytools.fit_circle_2d, x, y)


def test_fit_circle_3d():
    parms = ytools.fit_circle_3d(3 * np.eye(3), makeplot="new")
    center = np.array([1.0, 1.0, 1.0])
    radius = np.linalg.norm(np.array([3.0, 0, 0]) - center)
    z_direction = ytools._norm_vec(center)

    assert np.allclose(parms.center, center)
    assert np.allclose(parms.radius, radius)
    assert np.allclose(parms.basic2local[2], z_direction)


def test_histogram():
    assert np.allclose(
        ytools.histogram([1, 1, 3, 3, 4], 1),
        np.array([[1.0, 2.0, 40.0], [3.0, 2.0, 40.0], [4.0, 1.0, 20.0]]),
    )
    assert np.allclose(ytools.histogram([np.inf], 1), np.array([[0.0, 0.0, 0.0]]))


def test_sturm():
    a = np.array([0.0, 0.16, 1.55, 2.78, 9.0, 14.0])
    A = np.diag(a)
    assert np.all(1 == ytools.sturm(A, 0.0))
    assert np.all(3 == ytools.sturm(A, 1.55))


def test_eig_si():
    k = np.random.randn(40, 40)
    m = np.random.randn(40, 40)
    k = np.dot(k.T, k) * 1000
    m = np.dot(m.T, m) * 10
    w1, phi1 = linalg.eigh(k, m, eigvals=(0, 14))
    w2, phi2, phiv2 = ytools.eig_si(k, m, p=15, mu=-1, tol=1e-12, verbose=False)
    fcut = np.sqrt(w2.max()) / 2 / np.pi * 1.001
    w3, phi3, phiv3 = ytools.eig_si(k, m, f=fcut, mu=-1, tol=1e-12)
    assert np.allclose(w1, w2)
    assert np.allclose(np.abs(phi1), np.abs(phi2))
    assert np.allclose(w1, w3)
    assert np.allclose(np.abs(phi1), np.abs(phi3))
    w4, phi4, phiv4 = ytools.eig_si(k, m, f=fcut, Xk=phiv3, tol=1e-12, pmax=10)
    assert np.allclose(w1[:10], w4)
    assert np.allclose(np.abs(phi1[:, :10]), np.abs(phi4))

    w5, phi5, phiv5 = ytools.eig_si(k, m, p=15, Xk=phi4, tol=1e-12)
    assert np.allclose(w1, w5)
    assert np.allclose(np.abs(phi1), np.abs(phi5))

    mmod = m.copy()
    mmod[1, 0] = 2 * mmod[1, 0]
    assert_raises(ValueError, ytools.eig_si, k, mmod, p=15, Xk=phi4, tol=1e-12)


def test_gensweep():
    sig, t, f = ytools.gensweep(10, 1, 12, 8)
    assert np.allclose(np.max(sig), 1.0)
    assert np.allclose(np.min(sig), -1.0)
    # 12 = 1*2**n_oct
    n_oct = np.log(12.0) / np.log(2.0)
    t_elapsed = n_oct / 8 * 60
    assert np.abs(t[-1] - t_elapsed) < 0.01
    assert np.allclose(f[0], 1)
    assert np.abs(f[-1] - 12) < 0.01
    assert np.allclose(t[1] - t[0], 1 / 10 / 12)
    # at 8 oct/min or 8/60 = 4/30 = 2/15 oct/sec, time for 3 octaves
    # is: 3/(2/15) = 45/2 = 22.5 s
    i = np.argmin(np.abs(t - 22.5))
    assert np.allclose(f[i], 8.0)


def test_mattype():
    t, m = ytools.mattype([1, 2, 3])
    assert t == 0
    assert not ytools.mattype([1, 2, 3], "symmetric")

    a = np.random.randn(4, 4)
    a = a.dot(a.T)
    t, m = ytools.mattype(a)
    assert t & m["posdef"] and t & m["symmetric"]
    assert ytools.mattype(a, "posdef")
    assert ytools.mattype(a, "symmetric")

    a[1, 1] = 0
    t, m = ytools.mattype(a)
    assert (not (t & m["posdef"])) and t & m["symmetric"]
    assert not ytools.mattype(a, "posdef")
    assert ytools.mattype(a, "symmetric")

    c = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    c = c.dot(np.conj(c.T))
    t, m = ytools.mattype(c)
    assert t & m["posdef"] and t & m["hermitian"]
    assert ytools.mattype(c, "posdef")
    assert ytools.mattype(c, "hermitian")

    c[1, 1] = 0
    t, m = ytools.mattype(c)
    assert (not (t & m["posdef"])) and t & m["hermitian"]
    assert ytools.mattype(c, "hermitian")
    assert not ytools.mattype(c, "posdef")

    assert ytools.mattype(np.eye(5), "diagonal")
    assert ytools.mattype(np.eye(5), "identity")
    assert not ytools.mattype(np.eye(5) * 2.0, "identity")
    assert not ytools.mattype(np.random.randn(5, 5), "diagonal")
    assert not ytools.mattype(np.random.randn(5, 5), "identity")
    assert not ytools.mattype(np.random.randn(5, 5), "posdef")
    assert not ytools.mattype(np.random.randn(5, 5) * (2 + 1j), "posdef")

    assert_raises(ValueError, ytools.mattype, c, "badtype")


def test_save_load():
    a = np.arange(18).reshape(2, 3, 3)
    b = np.arange(3)
    d = dict(A="test var", B=[1, 2, 3])

    names = []
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        name = f.name
        f.close()
        obj = dict(a=a, b=b, c=a, d=d, e=d)
        for ext in ("", ".pgz", ".pbz2"):
            fname = name + ext
            names.append(fname)
            ytools.save(fname, obj)
            obj_in = ytools.load(fname)
            assert np.all(obj["a"] == obj_in["a"])
            assert np.all(obj["b"] == obj_in["b"])
            assert obj_in["c"] is obj_in["a"]
            assert obj["d"] == obj_in["d"]
            assert obj_in["e"] is obj_in["d"]
    finally:
        for name in names:
            os.remove(name)
