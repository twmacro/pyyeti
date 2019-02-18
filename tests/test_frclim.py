import numpy as np
from pyyeti import cb, ytools, frclim, ode
from pyyeti.nastran import op2, n2p
from nose.tools import *


def test_calcAM():
    nas = op2.rdnas2cam('tests/nas2cam_csuper/nas2cam')
    maa = nas['maa'][101]
    kaa = nas['kaa'][101]
    uset = nas['uset'][101]
    b = n2p.mksetpv(uset, 'a', 'b')
    q = ~b
    b = np.nonzero(b)[0]

    freq = np.arange(1., 80., 1.)

    pv = np.any(maa, axis=0)
    q = q[pv]
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]
    baa = np.zeros_like(maa)
    baa[q, q] = 2 * .05 * np.sqrt(kaa[q, q])

    nb = len(b)
    bdrm = np.zeros((nb, maa.shape[0]))
    bdrm[:nb, :nb] = np.eye(nb)
    AM1 = frclim.calcAM((maa, baa, kaa, b), freq)
    AM2 = frclim.calcAM((maa, baa, kaa, bdrm), freq)
    assert np.allclose(AM1, AM2)


def test_ntfl():
    freq = np.arange(0., 25.1, .1)
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

    # 2. Solve coupled system:

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
    F = np.vstack((np.ones((1, len(freq))),
                   np.zeros((3, len(freq)))))
    fs = ode.SolveUnc(MASS, DAMP, STIF, pre_eig=True)
    fullsol = fs.fsolve(F, freq)
    A_coupled = fullsol.a[1]
    F_coupled = (M2 / 2 * A_coupled - k2 * (fullsol.d[2] - fullsol.d[1])
                 - c2 * (fullsol.v[2] - fullsol.v[1]))

    # 3. Solve for free acceleration; SOURCE setup: [m, b, k, bdof]:

    ms = np.array([[M1, 0], [0, M2 / 2]])
    cs = np.array([[c1, -c1], [-c1, c1]])
    ks = np.array([[k1, -k1], [-k1, k1]])
    source = [ms, cs, ks, [[0, 1]]]
    fs_source = ode.SolveUnc(ms, cs, ks, pre_eig=True)
    sourcesol = fs_source.fsolve(F[:2], freq)
    As = sourcesol.a[1:2]   # free acceleration

    # LOAD setup: [m, b, k, bdof]:

    ml = np.array([[M2 / 2, 0, 0], [0, M3, 0], [0, 0, M4]])
    cl = np.array([[c2, -c2, 0], [-c2, c2 + c3, -c3], [0, -c3, c3]])
    kl = np.array([[k2, -k2, 0], [-k2, k2 + k3, -k3], [0, -k3, k3]])
    load = [ml, cl, kl, [[1, 0, 0]]]

    # 4. Use NT to couple equations. First value (rigid-body motion)
    # should equal ``Source Mass / Total Mass = 25/45 = 0.55555...``
    # Results should match the coupled method.

    r = frclim.ntfl(source, load, As, freq)
    assert np.allclose(25 / 45, abs(r.R[0, 0]))
    assert np.allclose(A_coupled, r.A)
    assert np.allclose(F_coupled, r.F)
    assert_raises(ValueError, frclim.ntfl, source, load, As,
                  freq[:-1])

    r2 = frclim.ntfl(r.SAM, r.LAM, As, freq)
    assert r.SAM is r2.SAM
    assert r.LAM is r2.LAM
    assert np.all(r.TAM == r2.TAM)
    assert np.allclose(r.R, r2.R)
    assert np.allclose(r.A, r2.A)
    assert np.allclose(r.F, r2.F)


def test_sefl():
    assert np.allclose(1.5, frclim.sefl(1.5, 40, 80))
    assert np.allclose(1.5 / 2, frclim.sefl(1.5, 80, 40))


def test_stdfs():
    m1 = 710     # modal mass + residual mass of lv
    m2 = 3060    # modal mass + residual mass of s/c
    Q = 10
    spec = 1.75
    fl = frclim.stdfs(m2 / m1, Q) * m2 * 1.75
    assert abs(6393.1622 - fl) < 1e-3
    fl = frclim.stdfs(m2 / m1, [Q, Q]) * m2 * 1.75
    assert abs(6393.1622 - fl) < 1e-3


def test_ctdfs():
    m1 = 30     # lv modal mass 75-90 Hz
    M1 = 622    # lv residual mass above 90 Hz
    m2 = 972    # sc modal mass 75-90 Hz
    M2 = 954    # sc residual mass above 90 Hz
    msc = 6961  # total sc mass
    faf = 40    # fundamental axial frequency of s/c
    Q = 10
    spec = 1.75
    fl = (frclim._ctdfs_old(m1 / M1, m2 / M2, M2 / M1, Q)[0] *
          M2 * spec)
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = (frclim._ctdfs_old(m1 / M1, m2 / M2, M2 / M1, [Q, Q])[0] *
          M2 * spec)
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = (frclim.ctdfs(m1 / M1, m2 / M2, M2 / M1, Q)[0] *
          M2 * spec)
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = (frclim.ctdfs(m1 / M1, m2 / M2, M2 / M1, [Q, Q])[0] *
          M2 * spec)
    assert abs(8686.1 / fl - 1) < 1e-4

    fl1 = frclim._ctdfs_old(1e-5, m2 / M2, M2 / M1, Q)
    fl2 = frclim._ctdfs_old(0, m2 / M2, M2 / M1, [Q, Q])
    assert np.allclose(fl1, fl2)

    fl1 = frclim.ctdfs(1e-5, m2 / M2, M2 / M1, Q)
    fl2 = frclim.ctdfs(0, m2 / M2, M2 / M1, [Q, Q])
    assert np.allclose(fl1, fl2)

    assert np.all((1, 1) == frclim._ctdfs_old(m1 / M1, 0, M2 / M1, Q))
    assert np.all((1, 1) == frclim.ctdfs(m1 / M1, 0, M2 / M1, Q))
