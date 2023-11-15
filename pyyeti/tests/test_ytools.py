import numpy as np
from numpy.random import Generator, MT19937
import tempfile
import os
from pyyeti import ytools
from scipy import linalg
from scipy import interpolate as sint
import pytest


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
    w1, phi1 = linalg.eigh(k, m, subset_by_index=(0, 14))
    w2, phi2, phiv2 = ytools.eig_si(
        k, m, p=15, mu=-1, tol=1e-12, verbose=False, rng=Generator(MT19937(1))
    )
    w2b, phi2b, phiv2b = ytools.eig_si(
        k, m, p=15, mu=-1, tol=1e-12, verbose=False, rng=Generator(MT19937(1))
    )

    assert (w2b == w2).all()
    assert (phi2b == phi2).all()
    assert (phiv2b == phiv2).all()

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
    with pytest.raises(ValueError):
        ytools.eig_si(k, mmod, p=15, Xk=phi4, tol=1e-12)


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

    with pytest.raises(ValueError):
        ytools.mattype(c, "badtype")


def test_mattype2():
    # real a
    a = np.array([[346500.0, 1e-7], [2.1e-7, 1000000.1]])
    assert 13 == ytools.mattype(a)[0]
    t, mattypes, ch = ytools.mattype(a, return_cholesky=True)
    assert t == 13
    assert np.allclose(ch, linalg.cholesky(a))

    # complex a
    a = np.array([[346500.0, 1e-7 * (1 + 1j)], [2.1e-7 * (1 - 1j), 1000000.1]])
    assert 15 == ytools.mattype(a)[0]
    t, mattypes, ch = ytools.mattype(a, return_cholesky=True)
    assert t == 15
    assert np.allclose(ch, linalg.cholesky(a))

    # real a
    a = np.array([[346500.0, 1e-7], [2000.1e-7, 1000000.1]])
    assert 0 == ytools.mattype(a)[0]
    t, mattypes, ch = ytools.mattype(a, return_cholesky=True)
    assert t == 0
    assert ch is None

    # complex a
    a = np.array([[346500.0, 1e-7 * (1 + 1j)], [2000.1e-7 * (1 - 1j), 1000000.1]])
    assert 0 == ytools.mattype(a)[0]
    t, mattypes, ch = ytools.mattype(a, return_cholesky=True)
    assert t == 0
    assert ch is None


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


def test_compmat():
    A = np.array([[1, 4, 5, 40], [15, 16, 17, 80]])
    B = np.array([[2, 4.2, 5, 43], [20, 14, 17, 82]])
    a, s = ytools.compmat(A, B, 0.1, method="row", pdiff_tol=5, verbose=0)
    assert np.allclose(a, 33.333333333333)

    A = A + B * 1j
    B = B + A.real * 1j
    a, s = ytools.compmat(A, B, 0.04, method="row", pdiff_tol=5, verbose=0)

    # A = array([[ 1. +2.j ,  4. +4.2j,  5. +5.j , 40.+43.j ],
    #            [15.+20.j , 16.+14.j , 17.+17.j , 80.+82.j ]])
    #
    # B = array([[ 2.  +1.j,  4.2 +4.j,  5.  +5.j, 43. +40.j],
    #            [20. +15.j, 14. +16.j, 17. +17.j, 82. +80.j]])

    real_pdiff = (B.real / A.real - 1) * 100
    # array([[100.        ,   5.        ,   0.        ,   7.5       ],
    #        [ 33.33333333, -12.5       ,   0.        ,   2.5       ]])

    imag_pdiff = (B.imag / A.imag - 1) * 100
    # array([[-50.        ,  -4.76190476,   0.        ,  -6.97674419],
    #        [-25.        ,  14.28571429,   0.        ,  -2.43902439]])

    assert a == 100
    real_sbe = np.array(
        [
            real_pdiff.min(),
            real_pdiff.max(),
            real_pdiff.mean(),
            real_pdiff.std(ddof=1),
        ]
    )
    imag_sbe = np.array(
        [
            imag_pdiff.min(),
            imag_pdiff.max(),
            imag_pdiff.mean(),
            imag_pdiff.std(ddof=1),
        ]
    )

    assert np.allclose(s, np.vstack((real_sbe, imag_sbe)))

    a, s = ytools.compmat(A, B.real, 0.04, method="row", pdiff_tol=5, verbose=0)
    assert np.allclose(s, np.vstack((real_sbe, [-100.0, -100.0, -100.0, 0.0])))

    a, s = ytools.compmat(
        A.real, B.real + 0j, 0.04, method="row", pdiff_tol=5, verbose=0
    )
    assert a == 100.0

    a, s = ytools.compmat(A, B, 0.2, method="col", pdiff_tol=5, verbose=0)
    assert np.allclose(a, 33.333333333333)

    a, s = ytools.compmat(A, B, 0.5, method="max", pdiff_tol=5, verbose=0)
    assert np.allclose(a, 7.5)

    a, s = ytools.compmat(A, B, 40.0, method="abs", pdiff_tol=5, verbose=0)
    assert np.allclose(a, 7.5)

    with pytest.raises(ValueError):
        ytools.compmat(A, A[:2, :2])

    with pytest.raises(ValueError):
        ytools.compmat(A, B, method="badmethod")

    a = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        ytools.compmat(a, a)


def test_max_complex_vector_sum():
    x = [3.0, 1.0 + 2.0j]
    y = [4.0, 3.0 + 4.0j]
    h, th, c, s = ytools.max_complex_vector_sum(x, y)

    # check for consistent output:
    for xi, yi, hi, thi, ci, si in zip(x, y, h, th, c, s):
        assert np.allclose(xi * ci + yi * si, hi)
        assert np.allclose(np.cos(thi), ci)
        assert np.allclose(np.sin(thi), si)

    # check for correct solution:
    mx = np.zeros(len(x))
    for i, (xi, yi) in enumerate(zip(x, y)):
        for a in np.arange(0.0, np.pi, 0.001):
            mx[i] = max(mx[i], np.abs(np.cos(a) * xi + np.sin(a) * yi))
    assert (np.abs(h) >= mx).all()

try:
    import numba
except ImportError:
    pass
else:

    def test_numba_interp_32():
        fp = np.array([[2, 1, 4]])
        fp = np.tile(fp, (100, 1))
        xp = np.array([1., 2., 3.])
        x = np.arange(0, 4, 0.1)

        f = ytools.numba_interp(x, xp, fp.astype(np.float32))

        assert f.dtype == np.float32

        # Compare against SciPy Interpolate, interp1d - with "hold value" extrapolation instead of linear extrapolation
        interp_func = sint.interp1d(xp, fp, bounds_error=False, fill_value=(fp[:, 0], fp[:, -1]))
        f_scipy = interp_func(x)

        np.testing.assert_array_almost_equal(f, f_scipy, decimal=7)


    def test_numba_interp_64():
        fp = np.array([[2, 1, 4]])
        fp = np.tile(fp, (100, 1))
        xp = np.array([1., 2., 3.])
        x = np.arange(0, 4, 0.1)

        f = ytools.numba_interp(x, xp, fp.astype(np.int32))
        assert f.dtype == np.float64

        f = ytools.numba_interp(x, xp, fp.astype(np.int64))
        assert f.dtype == np.float64

        f = ytools.numba_interp(x, xp, fp.astype(np.float64))
        assert f.dtype == np.float64

        # Compare against SciPy Interpolate, interp1d - with "hold value" extrapolation instead of linear extrapolation
        interp_func = sint.interp1d(xp, fp, bounds_error=False, fill_value=(fp[:, 0], fp[:, -1]))
        f_scipy = interp_func(x)

        np.testing.assert_array_equal(f, f_scipy)
