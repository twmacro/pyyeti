import numpy as np
from io import StringIO
import tempfile
import os
import inspect
from pyyeti import cb, op2, n2p, ytools, locate, op4, nastran
from pyyeti.tfsolve import TFSolve
from nose.tools import *


def test_cbreorder():
    drm = np.arange(1, 16).reshape(3, 5)
    assert_raises(ValueError, cb.cbreorder, drm, [0, 1, 2, 3],
                  drm=False, last=True)


def test_cbconvert():
    b = np.arange(6)
    drm = np.ones((1, 8))
    assert_raises(ValueError, cb.cbconvert, drm, b, 'm2e')
    b = np.arange(5)
    assert_raises(ValueError, cb.cbconvert, drm, b, 'm2e', drm=True)


def test_cbconvert_2():
    n = 19
    nb = 12
    mass = np.random.randn(n, n)
    b = np.arange(n-nb, n)
    m1_1 = cb.cbconvert(mass, b, 'm2e')
    m1_2 = cb.cbreorder(m1_1, b)

    m2_1 = cb.cbreorder(mass, b)
    bnew = np.arange(nb)
    m2_2 = cb.cbconvert(m2_1, bnew, 'm2e')
    assert np.allclose(m1_2, m2_2)
    assert np.allclose(mass[n-nb, n-nb]*0.005710147154735817, m1_2[0, 0])


def test_cbtf():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    maa = nas['maa'][102]
    kaa = nas['kaa'][102]
    uset = nas['uset'][102]
    b = n2p.mksetpv(uset, 'a', 'b')
    q = ~b
    b = np.nonzero(b)[0]

    rb = n2p.rbgeom_uset(uset[b], 3)
    freq = np.arange(1., 80., 1.)
    a = rb[:, :1]
    a2 = a.dot(np.ones((1, len(freq))))
    a3 = rb[:, 0]

    pv = np.any(maa, axis=0)
    q = q[pv]
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]
    baa1 = np.zeros_like(maa)
    baa1[q, q] = 2*.05*np.sqrt(kaa[q, q])
    baa2 = .1*np.random.randn(*maa.shape)
    baa2 = baa2.dot(baa2.T)

    bb = np.ix_(b, b)

    for baa in [baa1, baa2]:
        for delq in [False, True]:
            if delq:
                m = maa[bb]
                c = baa[bb]
                k = kaa[bb]
            else:
                m = maa
                c = baa
                k = kaa

            tf = cb.cbtf(m, c, k, a, freq, b)
            tf2 = cb.cbtf(m, c, k, a2, freq, b)
            save = {}
            tf3 = cb.cbtf(m, c, k, a3, freq, b, save)
            tf4 = cb.cbtf(m, c, k, a2, freq, b, save)

            assert np.allclose(tf.frc, tf2.frc)
            assert np.allclose(tf.a, tf2.a)
            assert np.allclose(tf.d, tf2.d)
            assert np.allclose(tf.v, tf2.v)

            assert np.allclose(tf.frc, tf3.frc)
            assert np.allclose(tf.a, tf3.a)
            assert np.allclose(tf.d, tf3.d)
            assert np.allclose(tf.v, tf3.v)

            assert np.allclose(tf.frc, tf4.frc)
            assert np.allclose(tf.a, tf4.a)
            assert np.allclose(tf.d, tf4.d)
            assert np.allclose(tf.v, tf4.v)

            # confirm proper solution:
            O = 2*np.pi*freq
            velo = 1j*O*tf.d
            acce = 1j*O*velo
            f = m.dot(acce) + c.dot(velo) + k.dot(tf.d)
            assert np.allclose(acce, tf.a)
            assert np.allclose(velo, tf.v)
            assert np.allclose(f[b], tf.frc)
            if not delq:
                assert np.allclose(f[q], 0)

    assert_raises(ValueError, cb.cbtf, maa, baa1, kaa,
                  a2[:, :3], freq, b)

    assert_raises(ValueError, cb.cbtf, maa, baa1, kaa,
                  a2[:3, :], freq, b)


def test_cbreorder_m():
    m = np.dot(np.arange(1, 9).reshape(-1, 1),
               np.arange(2, 10).reshape(1, -1))
    # array([[ 2,  3,  4,  5,  6,  7,  8,  9],
    #        [ 4,  6,  8, 10, 12, 14, 16, 18],
    #        [ 6,  9, 12, 15, 18, 21, 24, 27],
    #        [ 8, 12, 16, 20, 24, 28, 32, 36],
    #        [10, 15, 20, 25, 30, 35, 40, 45],
    #        [12, 18, 24, 30, 36, 42, 48, 54],
    #        [14, 21, 28, 35, 42, 49, 56, 63],
    #        [16, 24, 32, 40, 48, 56, 64, 72]])

    mnew = cb.cbreorder(m, np.arange(7, 1, -1))
    sbe = np.array([[72, 64, 56, 48, 40, 32, 16, 24],
                    [63, 56, 49, 42, 35, 28, 14, 21],
                    [54, 48, 42, 36, 30, 24, 12, 18],
                    [45, 40, 35, 30, 25, 20, 10, 15],
                    [36, 32, 28, 24, 20, 16,  8, 12],
                    [27, 24, 21, 18, 15, 12,  6,  9],
                    [ 9,  8,  7,  6,  5,  4,  2,  3],
                    [18, 16, 14, 12, 10,  8,  4,  6]])
    assert np.all(mnew == sbe)

    # without the following, when pandas is imported, we get:
    # ERROR: test_n2p_nose.test_badrbe3_error
    # ----------------------------------------------------------------------
    # Traceback (most recent call last):
    #   File "/home/macro/anaconda3/lib/python3.5/site-packages/nose/case.py", line 198, in runTest
    #     self.test(*self.arg)
    #   File "/home/macro/code/pyyeti/pyyeti/tests/test_n2p_nose.py", line 1380, in test_badrbe3_error
    #     with assert_warns(RuntimeWarning) as cm:
    #   File "/home/macro/anaconda3/lib/python3.5/unittest/case.py", line 225, in __enter__
    #     for v in sys.modules.values():
    # RuntimeError: dictionary changed size during iteration
    #
    # Reported here: https://github.com/pytest-dev/pytest/issues/1288
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    with assert_warns(RuntimeWarning):
        mnew = cb.cbreorder(m, np.arange(7, -1, -1))
    sbe = np.array([[72, 64, 56, 48, 40, 32, 24, 16],
                    [63, 56, 49, 42, 35, 28, 21, 14],
                    [54, 48, 42, 36, 30, 24, 18, 12],
                    [45, 40, 35, 30, 25, 20, 15, 10],
                    [36, 32, 28, 24, 20, 16, 12,  8],
                    [27, 24, 21, 18, 15, 12,  9,  6],
                    [18, 16, 14, 12, 10,  8,  6,  4],
                    [ 9,  8,  7,  6,  5,  4,  3,  2]])
    assert np.all(mnew == sbe)


def test_cbreorder_drm():
    drm = np.dot(np.arange(1, 7).reshape(-1, 1),
                 np.arange(2, 10).reshape(1, -1))
    # array([[ 2,  3,  4,  5,  6,  7,  8,  9],
    #        [ 4,  6,  8, 10, 12, 14, 16, 18],
    #        [ 6,  9, 12, 15, 18, 21, 24, 27],
    #        [ 8, 12, 16, 20, 24, 28, 32, 36],
    #        [10, 15, 20, 25, 30, 35, 40, 45],
    #        [12, 18, 24, 30, 36, 42, 48, 54]])

    dnew = cb.cbreorder(drm, np.arange(7, 1, -1), drm=True)
    sbe = np.array([[ 9,  8,  7,  6,  5,  4,  2,  3],
                    [18, 16, 14, 12, 10,  8,  4,  6],
                    [27, 24, 21, 18, 15, 12,  6,  9],
                    [36, 32, 28, 24, 20, 16,  8, 12],
                    [45, 40, 35, 30, 25, 20, 10, 15],
                    [54, 48, 42, 36, 30, 24, 12, 18]])
    assert np.all(dnew == sbe)

    assert_raises(ValueError, cb.cbreorder, drm,
                  np.arange(7, 1, -1))

    with assert_warns(RuntimeWarning):
        dnew = cb.cbreorder(drm, np.arange(7, -1, -1), drm=True)
    assert np.all(drm[:, ::-1] == dnew)

    drm = np.arange(1, 16).reshape(3, 5)
    # array([[ 1,  2,  3,  4,  5],
    #        [ 6,  7,  8,  9, 10],
    #        [11, 12, 13, 14, 15]])
    with assert_warns(RuntimeWarning):
        dnew = cb.cbreorder(drm, [0, 1, 2, 3], drm=True, last=True)
    sbe = np.array([[ 5,  1,  2,  3,  4],
                    [10,  6,  7,  8,  9],
                    [15, 11, 12, 13, 14]])
    assert np.all(dnew == sbe)


def test_cgmass():
    mass = np.array([[3,     0,     0,     0,     0,     0],
                     [0,     3,     0,     0,     0,   120],
                     [0,     0,     3,     0,  -120,     0],
                     [0,     0,     0,  1020,   -60,    22],
                     [0,     0,  -120,   -60,  7808,    23],
                     [0,   120,     0,    22,    23,  7800]])
    mcg1, dcg1 = cb.cgmass(mass)
    sbe = np.array([[    3.,     0.,     0.,     0.,     0.,     0.],
                    [    0.,     3.,     0.,     0.,     0.,     0.],
                    [    0.,     0.,     3.,     0.,     0.,     0.],
                    [    0.,     0.,     0.,  1020.,   -60.,    22.],
                    [    0.,     0.,     0.,   -60.,  3008.,    23.],
                    [    0.,     0.,     0.,    22.,    23.,  3000.]])
    assert np.allclose(mcg1, sbe)
    assert np.allclose(dcg1, [40., 0., 0.])

    mcg, dcg, gyr, pgyr, I, pI = cb.cgmass(mass, all6=True)
    assert np.all(mcg == mcg1)
    assert np.all(dcg == dcg1)
    assert np.all(abs(gyr - [18.4391, 31.6649, 31.6228]) < 1e-3)
    assert np.all(abs(pgyr - [18.4204, 31.7693, 31.5288]) < 1e-3)

    sbe = np.array([[ 1020.,   -60.,    22.],
                    [  -60.,  3008.,    23.],
                    [   22.,    23.,  3000.]])
    assert np.allclose(I, sbe)
    sbe = np.array([[ 1017.9312,     0.    ,     0.    ],
                    [    0.    ,  2982.2045,     0.    ],
                    [    0.    ,     0.    ,  3027.8643]])
    assert np.all(abs(pI - sbe) < 1e-3)

    mass[1, 0] = 4.
    assert_raises(ValueError, cb.cgmass, mass)


def gettable(lines, j, col=0, label=None, skip=0):
    table = []
    if label:
        while j < len(lines) and -1 == lines[j].find(label):
            j += 1
    j += skip
    for line in lines[j:]:
        if len(line) == 0:
            break
        line = line.replace(',',' ')
        table.append([float(item) for item in line[col:].split()])
    return np.array(table), j+len(table)


def comptable(s1, s2, j, col=0, label=None, skip=0):
    table1, nj1 = gettable(s1, j[0], col, label, skip)
    table2, nj2 = gettable(s2, j[0], col, label, skip)
    j[0] = min(nj1, nj2)
    return np.allclose(table1, table2, atol=1e-4)


def compare_cbcheck_output(s, sy):
    j = [15]
    assert comptable(s, sy, j, label=' ID ', skip=2)
    assert comptable(s, sy, j, label='RB ', skip=4)
    assert comptable(s, sy, j, label='RB ', skip=4)
    assert comptable(s, sy, j, label='6x6 ', skip=2)
    assert comptable(s, sy, j, label='6x6 ', skip=2)
    assert comptable(s, sy, j, label='6x6 ', skip=2)

    assert comptable(s, sy, j, label='Distance to CG', skip=4, col=23)
    assert comptable(s, sy, j, label=' gyration', skip=4, col=23)
    assert comptable(s, sy, j, label=' gyration', skip=4, col=23)

    assert comptable(s, sy, j, label='Inertia ', skip=2)
    assert comptable(s, sy, j, label='Principal ', skip=2)

    assert comptable(s, sy, j, label='Inertia ', skip=2)
    assert comptable(s, sy, j, label='Principal ', skip=2)

    assert comptable(s, sy, j, label='Inertia ', skip=2)
    assert comptable(s, sy, j, label='Principal ', skip=2)

    assert comptable(s, sy, j, label='-----------', skip=1, col=8)
    assert comptable(s, sy, j, label='Summation', skip=2, col=8)

    assert comptable(s, sy, j, label='-----------', skip=1, col=8)
    assert comptable(s, sy, j, label='Summation', skip=2, col=8)

    assert comptable(s, sy, j, label='-----------', skip=1, col=8)
    assert comptable(s, sy, j, label='Summation', skip=2, col=8)

    assert comptable(s, sy, j, label='Mode', skip=2, col=8)

    mef1 = gettable(s, j[0], label='Mode No.', skip=2)[0]
    mef2 = gettable(sy, j[0], label='Mode No.', skip=2)[0]
    # trim mef2 down:
    pv = np.any(mef2[:, 2:] >= 2.0, axis=1)
    mef2 = mef2[pv]
    assert np.allclose(mef1, mef2)

    assert comptable(s, sy, j, label='Total Eff', skip=0, col=21)


def test_cbcheck_indeterminate():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    se = 101
    maa = nas['maa'][se]
    kaa = nas['kaa'][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas['uset'][se]
    bset = n2p.mksetpv(uset, 'p', 'b')
    usetb = nas['uset'][se][bset]
    b = n2p.mksetpv(uset, 'a', 'b')
    q = ~b
    b = np.nonzero(b)[0]

    # write and read a file:
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
        name, maa, kaa, b, b[:6], usetb, em_filt=2)
    with open(name) as f:
        sfile = f.read()
    os.remove(name)

    # write to a string:
    with StringIO() as f:
        m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
            f, maa, kaa, b, b[:6], usetb, em_filt=2)
        s = f.getvalue()

    assert sfile == s
    s = s.splitlines()

    with open('pyyeti/tests/nas2cam_csuper/yeti_outputs/'
              'cbcheck_yeti_101.out') as f:
        sy = f.read().splitlines()

    assert s[0] == 'Mass matrix is symmetric.'
    assert s[1] == 'Mass matrix is positive definite.'
    assert s[2] == 'Stiffness matrix is symmetric.'

    compare_cbcheck_output(s, sy)

    with StringIO() as f:
        m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
            f, maa, kaa, b, b[:6], usetb, em_filt=2, conv='e2m')
        m, k, bset, rbs, rbg, rbe, usetconv2 = cb.cbcheck(
            f, m, k, b, b[:6], usetconv, em_filt=2, conv='m2e')
        assert np.allclose(usetconv2, usetb)
        assert np.allclose(maa, m)
        assert np.allclose(kaa, k)

    # check for error catches:
    with StringIO() as f:
        assert_raises(ValueError, cb.cbcheck, f, maa, kaa, b, b[:6],
                      usetb[:-6], em_filt=2)


def test_cbcheck_determinate():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    se = 101
    maa = nas['maa'][se]
    kaa = nas['kaa'][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas['uset'][se]
    bset = n2p.mksetpv(uset, 'p', 'b')
    usetb = nas['uset'][se][bset]
    b = n2p.mksetpv(uset, 'a', 'b')

    q = ~b
    b = np.nonzero(b)[0]
    q = np.nonzero(q)[0]

    center = np.mean(usetb[::6, 3:], axis=0)
    rb = n2p.rbgeom_uset(usetb, center)

    # transform to single pt on centerline:
    # [b, q]_old = T*[b, q]_new
    #            = [[rb, 0], [0, I]] * [b, q]_new
    T = np.zeros((len(b)+len(q), 6+len(q)))
    T[:len(b), :6] = rb
    T[len(b):, 6:] = np.eye(len(q))

    kaa = T.T @ kaa @ T
    maa = T.T @ maa @ T
    b = np.arange(6)

    # write to a string:
    with StringIO() as f:
        m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
            f, maa, kaa, b, b[:6], em_filt=2)
        s = f.getvalue()

    s = s.splitlines()
    with open('pyyeti/tests/nas2cam_csuper/yeti_outputs/'
              'cbcheck_yeti_101_single.out') as f:
        sy = f.read().splitlines()

    assert s[0] == 'Mass matrix is symmetric.'
    assert s[1] == 'Mass matrix is positive definite.'
    assert s[2] == 'Warning: stiffness matrix is not symmetric.'

    j = [10]
    assert comptable(s, sy, j, label='KBB =', skip=1)
    compare_cbcheck_output(s, sy)


def test_cbcheck_unit_convert():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    se = 101
    maa = nas['maa'][se]
    kaa = nas['kaa'][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas['uset'][se]
    bset = n2p.mksetpv(uset, 'p', 'b')
    usetb = nas['uset'][se][bset]
    b = n2p.mksetpv(uset, 'a', 'b')

    q = ~b
    b = np.nonzero(b)[0]
    q = np.nonzero(q)[0]

    # write to a string:
    with StringIO() as f:
        m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
            f, maa, kaa, b, b[:6], uset=usetb, em_filt=2,
            conv=[1/25.4, 0.005710147154735817],
            uref=[600, 150, 150])
        s = f.getvalue()

    s = s.splitlines()
    with open('pyyeti/tests/nas2cam_csuper/yeti_outputs/'
              'cbcheck_yeti_101_unitconv.out') as f:
        sy = f.read().splitlines()

    assert s[0] == 'Mass matrix is symmetric.'
    assert s[1] == 'Mass matrix is positive definite.'
    assert s[2] == 'Stiffness matrix is symmetric.'
    compare_cbcheck_output(s, sy)


def test_cbcheck_reorder():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    se = 101
    maa = nas['maa'][se]
    kaa = nas['kaa'][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas['uset'][se]
    bset = n2p.mksetpv(uset, 'p', 'b')
    usetb = nas['uset'][se][bset]
    b = np.nonzero(n2p.mksetpv(uset, 'a', 'b'))[0]

    maa = cb.cbreorder(maa, b, last=True)
    kaa = cb.cbreorder(kaa, b, last=True)
    b += maa.shape[0] - len(b)

    # write to a string:
    with StringIO() as f:
        m, k, bset, rbs, rbg, rbe, usetconv = cb.cbcheck(
            f, maa, kaa, b, b[:6], usetb, em_filt=2)
        s = f.getvalue()
    s = s.splitlines()
    with open('pyyeti/tests/nas2cam_csuper/yeti_outputs/'
              'cbcheck_yeti_101.out') as f:
        sy = f.read().splitlines()

    assert s[0] == 'Mass matrix is symmetric.'
    assert s[1] == 'Mass matrix is positive definite.'
    assert s[2] == 'Stiffness matrix is symmetric.'
    compare_cbcheck_output(s, sy)


def test_rbmultchk():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    se = 101
    maa = nas['maa'][se]
    kaa = nas['kaa'][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas['uset'][se]
    bset = n2p.mksetpv(uset, 'p', 'b')
    usetb = nas['uset'][se][bset]
    b = n2p.mksetpv(uset, 'a', 'b')
    q = ~b
    b = np.nonzero(b)[0]

    grids = [[11, 123456],
             [45, 123456],
             [60, 123456]]
    drm101, dof101 = n2p.formtran(nas, 101, grids)

    # write and read a file:
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()

    rb = n2p.rbgeom_uset(usetb)
    cb.rbmultchk(name, drm101, 'DRM101', rb)
    with open(name) as f:
        sfile = f.read()
    os.remove(name)

    # write to a string:
    with StringIO() as f:
        cb.rbmultchk(f, drm101, 'DRM101', rb)
        s = f.getvalue()
    assert sfile == s

    # add q-set rows to rb:
    rb2 = np.vstack((rb, np.zeros((len(q), 6))))
    with StringIO() as f:
        cb.rbmultchk(f, drm101, 'DRM101', rb2)
        s2 = f.getvalue()
    assert s2 == s

    # trim q-set columns out of drm:
    labels = [str(i[0])+'  '+str(i[1]) for i in dof101]
    with StringIO() as f:
        cb.rbmultchk(f, drm101[:, b], 'DRM101', rb,
                     drm2=drm101[:, b], prtnullrows=True,
                     labels=labels)
        s2 = f.getvalue()
    # row 16 is now all zeros ... not comparable
    drm2 = drm101.copy()
    drm2[15] = 0
    with StringIO() as f:
        cb.rbmultchk(f, drm2, 'DRM101', rb,
                     drm2=drm2, prtnullrows=True,
                     labels=labels)
        s3 = f.getvalue()
    assert s2 == s3

    s = s.splitlines()
    with open('pyyeti/tests/nas2cam_csuper/yeti_outputs/'
              'rbmultchk_yeti_101.out') as f:
        sy = f.read().splitlines()

    j = [2]
    assert comptable(s, sy, j, label='Extreme ', skip=3, col=11)
    assert comptable(s, sy, j, label=' Row ', skip=2, col=54)
    assert comptable(s, sy, j, label=' Row ', skip=2)
    assert comptable(s, sy, j, label=' Row ', skip=2)


def test_rbmultchk2():
    # write to a string:
    with StringIO() as f:
        cb.rbmultchk(f, np.zeros((15, 6)), 'DRM', np.eye(6),
                     drm2=np.random.randn(15, 6))
        s = f.getvalue()
    assert s.find(' -- no coordinates detected --') > -1
    assert s.find('All rows in DRM are NULL') > -1
    assert s.find('There are no NULL rows in DRM2.') > -1

    with StringIO() as f:
        cb.rbmultchk(f, np.zeros((15, 6)), 'DRM', np.eye(6),
                     drm2=np.random.randn(14, 6))
        s = f.getvalue()
    assert s.find('Error: incorrectly sized DRM2') > -1

    drm = np.random.randn(14, 6)
    drm2 = np.random.randn(14, 6)
    drm[::3] = 0.
    drm2[1::3] = 0.
    with StringIO() as f:
        cb.rbmultchk(f, drm, 'DRM', np.eye(6), drm2=drm2)
        s = f.getvalue()
    assert s.find('different set of NULL rows') > -1

    drm = np.random.randn(14, 6)
    drm2 = np.random.randn(14, 6)
    drm[::3] = 0.
    drm2[2::3] = 0.
    with StringIO() as f:
        cb.rbmultchk(f, drm, 'DRM', np.eye(6), drm2=drm2)
        s = f.getvalue()
    assert s.find('different set of NULL rows') > -1

    assert_raises(ValueError, cb.rbmultchk, f, drm, 'drm', 1)


def test_rbdispchk():
    coords = np.array([[0,  0,  0],
                       [1,  2,  3],
                       [4, -5, 25]])
    rb  = n2p.rbgeom(coords)
    xyz_pv = ytools.mkpattvec([0, 1, 2], 3*6, 6).ravel()
    rbtrimmed = rb[xyz_pv]
    # array([[  1.,   0.,   0.,   0.,   0.,   0.],
    #        [  0.,   1.,   0.,   0.,   0.,   0.],
    #        [  0.,   0.,   1.,   0.,   0.,   0.],
    #        [  1.,   0.,   0.,   0.,   3.,  -2.],
    #        [  0.,   1.,   0.,  -3.,   0.,   1.],
    #        [  0.,   0.,   1.,   2.,  -1.,   0.],
    #        [  1.,   0.,   0.,   0.,  25.,   5.],
    #        [  0.,   1.,   0., -25.,   0.,   4.],
    #        [  0.,   0.,   1.,  -5.,  -4.,   0.]])

    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed)
        s = f.getvalue()

    sbe = ['',
           'Coordinates Determined from Rigid-Body Displacements:',
           '         Node      X         Y         Z',
           '        ------  --------  --------  --------',
           '             1      0.00      0.00      0.00',
           '             2      1.00      2.00      3.00',
           '             3      4.00     -5.00     25.00',
           '',
           'Maximum absolute coordinate location error:    0 units',
           '',
           '']

    assert s == '\n'.join(sbe)

    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed,
                                          grids=[100, 200, 300])
        s = f.getvalue()

    sbe = ['',
           'Coordinates Determined from Rigid-Body Displacements:',
           '         Node      ID        X         Y         Z',
           '        ------  --------  --------  --------  --------',
           '             1       100      0.00      0.00      0.00',
           '             2       200      1.00      2.00      3.00',
           '             3       300      4.00     -5.00     25.00',
           '',
           'Maximum absolute coordinate location error:    0 units',
           '',
           '']

    assert s == '\n'.join(sbe)

    # add a little error:
    # array([[  1.,   0.,   0.,   0.,   0.,   0.],
    #        [  0.,   1.,   0.,   0.,   0.,   0.],
    #        [  0.,   0.,   1.,   0.,   0.,   0.],
    #        [  1.,   0.,   0.,   0.,   3.,  -2.],
    #        [  0.,   1.,   0.,  -3.,   0.,   1.],
    #        [  0.,   0.,   1.,   2.,  -1.,   0.],
    #        [  1.,   0.,   0.,   0.,  25.,   5.],
    #        [  0.,   1.,   0., -25.,   0.,   4.],
    #        [  0.,   0.,   1.,  -5.,  -4.,   0.]])
    rbtrimmed[7, 3] = -25.006
    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed)
        s = f.getvalue()

    sbe = ['Warning: deviation from standard pattern, node #3 '
           'starting at row 7. \tMax deviation = 0.006 units.',
           '  Rigid-Body Rotations:',
           '    0.0000    25.0000     5.0000',
           '  -25.0060     0.0000     4.0000',
           '   -5.0000    -4.0000     0.0000',
           '',
           '',
           'Coordinates Determined from Rigid-Body Displacements:',
           '         Node      X         Y         Z',
           '        ------  --------  --------  --------',
           '             1      0.00      0.00      0.00',
           '             2      1.00      2.00      3.00',
           '             3      4.00     -5.00     25.00',
           '',
           'Maximum absolute coordinate location error:  0.006 units',
           '',
           '']

    assert s == '\n'.join(sbe)

    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed,
                                          grids=[100, 200, 300])
        s = f.getvalue()

    sbe = ['Warning: deviation from standard pattern, node ID = 300 '
           'starting at row 7. Max deviation = 0.006 units.',
           '  Rigid-Body Rotations:',
           '    0.0000    25.0000     5.0000',
           '  -25.0060     0.0000     4.0000',
           '   -5.0000    -4.0000     0.0000',
           '',
           '',
           'Coordinates Determined from Rigid-Body Displacements:',
           '         Node      ID        X         Y         Z',
           '        ------  --------  --------  --------  --------',
           '             1       100      0.00      0.00      0.00',
           '             2       200      1.00      2.00      3.00',
           '             3       300      4.00     -5.00     25.00',
           '',
           'Maximum absolute coordinate location error:  0.006 units',
           '',
           '']

    assert s == '\n'.join(sbe)

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()

    coords_out, maxerr = cb.rbdispchk(name, rbtrimmed,
                                      grids=[100, 200, 300])
    with open(name) as f:
        s2 = f.read()
    os.remove(name)
    assert s == s2

    with StringIO() as f:
        assert_raises(ValueError, cb.rbdispchk, f, rbtrimmed[:, :4])
        assert_raises(ValueError, cb.rbdispchk, f, rbtrimmed[:5, :])


def test_cbcoordchk():
    k = np.random.randn(14, 14)
    k = k.dot(k.T)
    b = np.arange(4)
    assert_raises(ValueError, cb.cbcoordchk, k, b, b)

    b2 = np.arange(6)
    assert_raises(ValueError, cb.cbcoordchk, k, b2, b)


def test_mk_net_drms():
    pth = os.path.dirname(inspect.getfile(cb))
    pth = os.path.join(pth, 'tests')
    pth = os.path.join(pth, 'nas2cam_csuper')

    # Load the mass and stiffness from the .op4 file
    # This loads the data into a dict:
    mk = op4.load(os.path.join(pth, 'inboard.op4'))
    maa = mk['mxx'][0]
    kaa = mk['kxx'][0]

    # Get the USET table The USET table has the boundary DOF
    # information (id, location, coordinate system). This is needed
    # for superelements with an indeterminate interface. The nastran
    # module has the function bulk2uset which is handy for forming the
    # USET table from bulk data.

    uset, coords = nastran.bulk2uset(os.path.join(pth, 'inboard.asm'))

    # uset[::6, [0, 3, 4, 5]]
    # array([[   3.,  600.,    0.,  300.],
    #        [  11.,  600.,  300.,  300.],
    #        [  19.,  600.,  300.,    0.],
    #        [  27.,  600.,    0.,    0.]])

    # Form b-set partition vector into a-set
    # In this case, we already know the b-set are first:

    n = uset.shape[0]
    b = np.arange(n)

    # array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    #        16, 17, 18, 19, 20, 21, 22, 23])

    # y s/c  z l/v
    #    \   |   / x s/c
    #       \|/
    #        ----- y l/v
    sccoord = [[900, 1, 0],
               [0, 0, 0],
               [1, 0, 0],   # z is l/v x
               [0, 1, 1]]   # x is 45 deg ... between l/v y, z
    # y ends up 45 deg ... between l/v -y, z

    # convert s/c from mm, kg --> m, kg
    ref = [600, 150, 150]
    conv = (0.001, 1.)
    g = 9.80665

    u = n2p.addgrid(None, 1, 'b', sccoord, [0, 0, 0], sccoord)
    Tsc2lv = np.zeros((6, 6))
    Tsc2lv[:3, :3] = u[3:, 3:]
    Tsc2lv[3:, 3:] = u[3:, 3:]
    c = np.cos(45/180*np.pi)
    Tl2s = np.array([[0, c, c], [0, -c, c], [1, 0, 0]])
    assert np.allclose(Tl2s.T, Tsc2lv[:3, :3])

    net = cb.mk_net_drms(maa, kaa, b, uset=uset, ref=ref,
                         sccoord=sccoord, conv=conv, g=g)
    net2 = cb.mk_net_drms(maa, kaa, b, uset=uset, ref=ref,
                          sccoord=Tl2s, conv=conv, g=g)

    # rb modes in system units:
    uset2, ref2 = cb._uset_convert(uset, ref, conv)
    rb = n2p.rbgeom_uset(uset2, ref2)
    l_sc = net.ifltma_sc[:, :n] @ rb
    l_lv = net.ifltma_lv[:, :n] @ rb
    l_scd = net.ifltmd_sc[:, :n] @ rb
    l_lvd = net.ifltmd_lv[:, :n] @ rb
    a_sc = net.ifatm_sc[:, :n] @ rb
    a_lv = net.ifatm_lv[:, :n] @ rb
    c_sc = net.cgatm_sc[:, :n] @ rb
    c_lv = net.cgatm_lv[:, :n] @ rb


    assert np.allclose(Tsc2lv @ a_sc, a_lv)
    assert np.allclose(Tsc2lv @ c_sc, c_lv)
    assert abs(l_scd).max() < 1e-6*abs(l_sc).max()
    assert abs(l_lvd).max() < 1e-6*abs(l_lv).max()
    scale = np.array([[1000],
                      [1000],
                      [1000],
                      [1000000],
                      [1000000],
                      [1000000]])
    assert np.allclose((1/scale) * (Tsc2lv @ l_sc), l_lv)
    # height and weight values from cbcheck tutorial:
    assert abs(net.height - (1.039998351-.6)) < .000001
    assert abs(net.weight - 1.7551*g) < .001
    assert net.scaxial_sc == 2
    assert net.scaxial_lv == 0

    for name in net.__dict__:
        assert np.allclose(net.__dict__[name], net2.__dict__[name])
