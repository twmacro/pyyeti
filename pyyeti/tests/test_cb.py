import math
import numpy as np
import scipy.linalg as la
from io import StringIO
import tempfile
import os
import inspect
from pyyeti import cb, ytools, nastran
from pyyeti.nastran import op2, n2p, op4
import pytest


def test_cbreorder():
    drm = np.arange(1, 16).reshape(3, 5)
    with pytest.raises(ValueError, match="`M` must be square when `drm` is false"):
        cb.cbreorder(drm, [0, 1, 2, 3], drm=False, last=True)


def test_uset_convert():
    import numpy as np
    from pyyeti import cb
    from pyyeti.nastran import n2p

    # node 100 in basic is @ [50, 100, 150] inches
    uset = n2p.addgrid(None, 100, "b", 0, [50, 100, 150], 0)
    ref = (100, 100, 100)
    uset_conv, ref_conv = cb.uset_convert(uset, ref, "e2m")
    assert np.allclose(uset_conv.loc[(100, 1), "x":"z"], [1.27, 2.54, 3.81])
    assert np.allclose(ref_conv, [2.54, 2.54, 2.54])

    uset_back, ref_back = cb.uset_convert(uset_conv, ref_conv)
    assert np.allclose(uset_back, uset)
    assert np.allclose(ref_back, ref)

    uset_conv2, ref_conv = cb.uset_convert(uset_conv)
    assert np.allclose(uset, uset_conv2)
    assert ref_conv is None


def test_cbconvert():
    b = np.arange(6)
    drm = np.ones((1, 8))
    with pytest.raises(ValueError, match="`M` must be square when `drm` is false"):
        cb.cbconvert(drm, b, "m2e")
    b = np.arange(5)
    with pytest.raises(ValueError, match="b-set not a multiple of 6"):
        cb.cbconvert(drm, b, "m2e", drm=True)


def test_cbconvert_2():
    n = 19
    nb = 12
    mass = np.random.randn(n, n)
    b = np.arange(n - nb, n)
    m1_1 = cb.cbconvert(mass, b, "m2e")
    m1_2 = cb.cbreorder(m1_1, b)

    m2_1 = cb.cbreorder(mass, b)
    bnew = np.arange(nb)
    m2_2 = cb.cbconvert(m2_1, bnew, "m2e")
    assert np.allclose(m1_2, m2_2)
    assert np.allclose(mass[n - nb, n - nb] * 0.005710147154735817, m1_2[0, 0])


def test_cbtf():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    maa = nas["maa"][102]
    kaa = nas["kaa"][102]
    uset = nas["uset"][102]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    rb = n2p.rbgeom_uset(uset.iloc[b], 3)
    freq = np.arange(1.0, 80.0, 1.0)
    a = rb[:, :1]
    a2 = a.dot(np.ones((1, len(freq))))
    a3 = rb[:, 0]

    pv = np.any(maa, axis=0)
    q = q[pv]
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]
    baa1 = np.zeros_like(maa)
    baa1[q, q] = 2 * 0.05 * np.sqrt(kaa[q, q])
    baa2 = 0.1 * np.random.randn(*maa.shape)
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

            assert np.all(freq == tf.freq)
            assert np.all(freq == tf2.freq)
            assert np.all(freq == tf3.freq)
            assert np.all(freq == tf4.freq)

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
            O = 2 * np.pi * freq
            velo = 1j * O * tf.d
            acce = 1j * O * velo
            f = m.dot(acce) + c.dot(velo) + k.dot(tf.d)
            assert np.allclose(acce, tf.a)
            assert np.allclose(velo, tf.v)
            assert np.allclose(f[b], tf.frc)
            if not delq:
                assert np.allclose(f[q], 0)

    with pytest.raises(ValueError, match="`a` is not compatibly sized with `freq`"):
        cb.cbtf(maa, baa1, kaa, a2[:, :3], freq, b)

    with pytest.raises(
        ValueError, match="number of rows in `a` not compatible with `bset`"
    ):
        cb.cbtf(maa, baa1, kaa, a2[:3, :], freq, b)


def test_cbreorder_m():
    m = np.dot(np.arange(1, 9).reshape(-1, 1), np.arange(2, 10).reshape(1, -1))
    # array([[ 2,  3,  4,  5,  6,  7,  8,  9],
    #        [ 4,  6,  8, 10, 12, 14, 16, 18],
    #        [ 6,  9, 12, 15, 18, 21, 24, 27],
    #        [ 8, 12, 16, 20, 24, 28, 32, 36],
    #        [10, 15, 20, 25, 30, 35, 40, 45],
    #        [12, 18, 24, 30, 36, 42, 48, 54],
    #        [14, 21, 28, 35, 42, 49, 56, 63],
    #        [16, 24, 32, 40, 48, 56, 64, 72]])

    mnew = cb.cbreorder(m, np.arange(7, 1, -1))
    sbe = np.array(
        [
            [72, 64, 56, 48, 40, 32, 16, 24],
            [63, 56, 49, 42, 35, 28, 14, 21],
            [54, 48, 42, 36, 30, 24, 12, 18],
            [45, 40, 35, 30, 25, 20, 10, 15],
            [36, 32, 28, 24, 20, 16, 8, 12],
            [27, 24, 21, 18, 15, 12, 6, 9],
            [9, 8, 7, 6, 5, 4, 2, 3],
            [18, 16, 14, 12, 10, 8, 4, 6],
        ]
    )
    assert np.all(mnew == sbe)

    with pytest.warns(RuntimeWarning, match="b-set not a multiple of 6"):
        mnew = cb.cbreorder(m, np.arange(7, -1, -1))

    sbe = np.array(
        [
            [72, 64, 56, 48, 40, 32, 24, 16],
            [63, 56, 49, 42, 35, 28, 21, 14],
            [54, 48, 42, 36, 30, 24, 18, 12],
            [45, 40, 35, 30, 25, 20, 15, 10],
            [36, 32, 28, 24, 20, 16, 12, 8],
            [27, 24, 21, 18, 15, 12, 9, 6],
            [18, 16, 14, 12, 10, 8, 6, 4],
            [9, 8, 7, 6, 5, 4, 3, 2],
        ]
    )
    assert np.all(mnew == sbe)


def test_cbreorder_drm():
    drm = np.dot(np.arange(1, 7).reshape(-1, 1), np.arange(2, 10).reshape(1, -1))
    # array([[ 2,  3,  4,  5,  6,  7,  8,  9],
    #        [ 4,  6,  8, 10, 12, 14, 16, 18],
    #        [ 6,  9, 12, 15, 18, 21, 24, 27],
    #        [ 8, 12, 16, 20, 24, 28, 32, 36],
    #        [10, 15, 20, 25, 30, 35, 40, 45],
    #        [12, 18, 24, 30, 36, 42, 48, 54]])

    dnew = cb.cbreorder(drm, np.arange(7, 1, -1), drm=True)
    sbe = np.array(
        [
            [9, 8, 7, 6, 5, 4, 2, 3],
            [18, 16, 14, 12, 10, 8, 4, 6],
            [27, 24, 21, 18, 15, 12, 6, 9],
            [36, 32, 28, 24, 20, 16, 8, 12],
            [45, 40, 35, 30, 25, 20, 10, 15],
            [54, 48, 42, 36, 30, 24, 12, 18],
        ]
    )
    assert np.all(dnew == sbe)

    with pytest.raises(ValueError, match="`M` must be square when `drm` is false"):
        cb.cbreorder(drm, np.arange(7, 1, -1))

    with pytest.warns(RuntimeWarning, match="b-set not a multiple of 6"):
        dnew = cb.cbreorder(drm, np.arange(7, -1, -1), drm=True)
    assert np.all(drm[:, ::-1] == dnew)

    drm = np.arange(1, 16).reshape(3, 5)
    # array([[ 1,  2,  3,  4,  5],
    #        [ 6,  7,  8,  9, 10],
    #        [11, 12, 13, 14, 15]])
    with pytest.warns(RuntimeWarning, match="b-set not a multiple of 6"):
        dnew = cb.cbreorder(drm, [0, 1, 2, 3], drm=True, last=True)

    sbe = np.array([[5, 1, 2, 3, 4], [10, 6, 7, 8, 9], [15, 11, 12, 13, 14]])
    assert np.all(dnew == sbe)


def test_cgmass():
    mass = np.array(
        [
            [3, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 120],
            [0, 0, 3, 0, -120, 0],
            [0, 0, 0, 1020, -60, 22],
            [0, 0, -120, -60, 7808, 23],
            [0, 120, 0, 22, 23, 7800],
        ]
    )
    mcg1, dcg1 = cb.cgmass(mass)
    sbe = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1020.0, -60.0, 22.0],
            [0.0, 0.0, 0.0, -60.0, 3008.0, 23.0],
            [0.0, 0.0, 0.0, 22.0, 23.0, 3000.0],
        ]
    )
    assert np.allclose(mcg1, sbe)
    assert np.allclose(dcg1, [40.0, 0.0, 0.0])

    mcg, dcg, gyr, pgyr, I, pI = cb.cgmass(mass, all6=True)
    assert np.all(mcg == mcg1)
    assert np.all(dcg == dcg1)
    assert np.all(abs(gyr - [18.4391, 31.6649, 31.6228]) < 1e-3)
    assert np.all(abs(pgyr - [18.4204, 31.5288, 31.7693]) < 1e-3)

    sbe = np.array([[1020.0, -60.0, 22.0], [-60.0, 3008.0, 23.0], [22.0, 23.0, 3000.0]])
    assert np.allclose(I, sbe)
    sbe = np.array(
        [[1017.9312, 0.0, 0.0], [0.0, 2982.2045, 0.0], [0.0, 0.0, 3027.8643]]
    )
    assert np.all(abs(pI - sbe) < 1e-3)

    mass[1, 0] = 4.0
    with pytest.raises(ValueError, match="mass matrix is not symmetric"):
        cb.cgmass(mass)


def gettable(lines, j, col=0, label=None, skip=0):
    table = []
    if label:
        while j < len(lines) and -1 == lines[j].find(label):
            j += 1
    j += skip
    for line in lines[j:]:
        line = line.rstrip()
        if len(line) == 0:
            break
        line = line.replace(",", " ")
        table.append([float(item) for item in line[col:].split()])
    return np.array(table), j + len(table)


def comptable(s1, s2, j1, j2, col=0, label=None, skip=0, sort2=0, truncate_rows=False):
    table1, nj1 = gettable(s1, j1[0], col, label, skip)
    table2, nj2 = gettable(s2, j2[0], col, label, skip)
    if sort2:
        table2 = np.sort(table2, axis=1)
    j1[0] = nj1
    j2[0] = nj2
    if truncate_rows:
        n = min(table1.shape[0], table2.shape[0])
        table1 = table1[:n]
        table2 = table2[:n]
    return np.allclose(table1, table2, atol=1e-1)


def compare_cbcheck_output(s, sy):
    j = [15]
    jy = [15]
    assert comptable(s, sy, j, jy, label=" ID ", skip=2)
    assert comptable(s, sy, j, jy, label="----------", skip=1)
    assert comptable(s, sy, j, jy, label="----------", skip=1)
    assert comptable(s, sy, j, jy, label="6x6 ", skip=2)
    assert comptable(s, sy, j, jy, label="6x6 ", skip=2)
    assert comptable(s, sy, j, jy, label="6x6 ", skip=2)

    assert comptable(s, sy, j, jy, label="Distance to CG", skip=4, col=23)
    assert comptable(s, sy, j, jy, label=" gyration", skip=4, col=23)
    assert comptable(s, sy, j, jy, label=" gyration", skip=4, col=23, sort2=True)

    assert comptable(s, sy, j, jy, label="Inertia ", skip=2)
    assert comptable(s, sy, j, jy, label="Principal ", skip=2)

    assert comptable(s, sy, j, jy, label="Inertia ", skip=2)
    assert comptable(s, sy, j, jy, label="Principal ", skip=2)

    assert comptable(s, sy, j, jy, label="Inertia ", skip=2)
    assert comptable(s, sy, j, jy, label="Principal ", skip=2)

    assert comptable(s, sy, j, jy, label="-----------", skip=1, col=8)
    assert comptable(s, sy, j, jy, label="Summation", skip=2, col=8)

    assert comptable(s, sy, j, jy, label="-----------", skip=1, col=8)
    assert comptable(s, sy, j, jy, label="Summation", skip=2, col=8)

    assert comptable(s, sy, j, jy, label="-----------", skip=1, col=8)
    assert comptable(s, sy, j, jy, label="Summation", skip=2, col=8)

    assert comptable(s, sy, j, jy, label="Mode", skip=2, col=8, truncate_rows=True)

    mef1 = gettable(s, j[0], label="Mode No.", skip=2)[0]
    mef2 = gettable(sy, jy[0], label="Mode No.", skip=2)[0]
    # trim mef2 down:
    pv = np.any(mef2[:, 2:] >= 2.0, axis=1)
    mef2 = mef2[pv]
    assert np.allclose(mef1, mef2)

    assert comptable(s, sy, j, jy, label="Total Eff", skip=0, col=21)


def test_cbcheck_indeterminate():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    # write and read a file:
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    out = cb.cbcheck(name, maa, kaa, b, b[:6], uset=uset, em_filt=2)
    with open(name) as f:
        sfile = f.read()
    os.remove(name)
    assert (out.m == maa).all()
    assert (out.k == kaa).all()
    assert out.uset.equals(usetb)

    rbg = n2p.rbgeom_uset(out.uset)
    assert np.allclose(rbg, out.rbg)
    rbg_s = np.vstack((la.solve(rbg[:6].T, rbg.T).T, np.zeros((q[q].size, 6))))
    assert abs(out.rbs - rbg_s).max() < 1e-5
    assert abs(out.rbe - rbg_s).max() < 1e-5

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, b[:6], usetb, em_filt=2)
        s = f.getvalue()

    assert sfile[:15] == s[:15]
    s = s.splitlines()
    sfile = sfile.splitlines()
    compare_cbcheck_output(s, sfile)

    with open("pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_101.out") as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Mass matrix is positive-definite."
    assert s[2] == "Stiffness matrix is symmetric."

    compare_cbcheck_output(s, sy)

    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, b[:6], usetb, em_filt=2, conv="e2m")
        out2 = cb.cbcheck(f, out.m, out.k, b, b[:6], out.uset, em_filt=2, conv="m2e")
        assert np.allclose(out2.uset, usetb)
        assert np.allclose(maa, out2.m)
        assert np.allclose(kaa, out2.k)

    # check for error catches:
    with StringIO() as f:
        with pytest.raises(ValueError, match="number of rows in `uset`"):
            cb.cbcheck(f, maa, kaa, b, b[:6], usetb.iloc[:-6], em_filt=2)


def test_cbcheck_determinate():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")

    q = ~b
    b = np.nonzero(b)[0]
    q = np.nonzero(q)[0]

    center = np.mean(usetb.iloc[::6, 1:], axis=0)
    rb = n2p.rbgeom_uset(usetb, center.values)

    # transform to single pt on centerline:
    # [b, q]_old = T*[b, q]_new
    #            = [[rb, 0], [0, I]] * [b, q]_new
    T = np.zeros((len(b) + len(q), 6 + len(q)))
    T[: len(b), :6] = rb
    T[len(b) :, 6:] = np.eye(len(q))

    kaa = T.T @ kaa @ T
    maa = T.T @ maa @ T
    b = np.arange(6)

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, b[:6], em_filt=2)
        s = f.getvalue()

    s = s.splitlines()
    with open(
        "pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_101_single.out"
    ) as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Mass matrix is positive-definite."
    assert s[2].startswith("Warning: stiffness matrix is not symmetric:")

    j = [10]
    jy = [10]
    assert comptable(s, sy, j, jy, label="KBB =", skip=1)
    compare_cbcheck_output(s, sy)

    # check with no em filter:
    with StringIO() as f:
        out2 = cb.cbcheck(f, maa, kaa, b, b[:6])
        s2 = f.getvalue()
    s2 = s2.splitlines()

    s_unique = [i for i in s if i not in s2]
    # ['Printing only the modes with at least 2.0% effective mass.',
    #  'The sum includes all modes.']

    s2_unique = [i for i in s2 if i not in s]
    # ['     5          7.025        0.00    0.00    0.00    0.00    0.88    0.00',
    #  '     6          7.025        0.00    0.00    0.00    0.00    0.00    0.00',
    #  '     7         10.913        0.00    0.00    0.00    0.00    0.00    1.52',
    #  '    11         25.135        0.00    0.00    0.00    0.00    0.00    0.42',
    #  '    12         25.140        1.03    0.00    0.00    0.00    0.00    0.00',
    #  '    13         42.173        0.00    0.00    0.00    0.51    0.00    0.00',
    #  '    14         42.193        0.00    0.00    1.04    0.00    1.02    0.00',
    #  '    16         46.895        0.00    0.00    0.00    0.00    0.00    0.00',
    #  '    17         69.173        0.00    0.13    0.00    0.00    0.00    0.99']

    assert len(s2) > len(s)
    assert "Printing only the modes with at least 2.0% effective mass." in s_unique
    assert "The sum includes all modes." in s_unique
    assert len(s2_unique) == len(s_unique) + 7

    # make k perfect:
    kaa[b] = 0.0
    kaa[:, b] = 0.0

    # write to a string:
    with StringIO() as f:
        out2 = cb.cbcheck(f, maa, kaa, b, b[:6], em_filt=2)

    for key, val1 in out.__dict__.items():
        val2 = getattr(out2, key)
        if key == "k":
            assert np.allclose(val1[6:, 6:], val2[6:, 6:], atol=1e-6)
        else:
            assert np.allclose(val1, val2, atol=1e-6)


def test_cbcheck_unit_convert():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")

    q = ~b
    b = np.nonzero(b)[0]
    q = np.nonzero(q)[0]

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(
            f,
            maa,
            kaa,
            b,
            b[:6],
            uset=usetb,
            em_filt=2,
            conv=[1 / 25.4, 0.005710147154735817],
            uref=[600, 150, 150],
            rb_norm=False,
        )
        s = f.getvalue()

    s = s.splitlines()
    with open(
        "pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_101_unitconv.out"
    ) as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Mass matrix is positive-definite."
    assert s[2] == "Stiffness matrix is symmetric."
    compare_cbcheck_output(s, sy)


def test_cbcheck_reorder():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = np.nonzero(n2p.mksetpv(uset, "a", "b"))[0]

    maa = cb.cbreorder(maa, b, last=True)
    kaa = cb.cbreorder(kaa, b, last=True)
    b += maa.shape[0] - len(b)

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, b[:6], usetb, em_filt=2)
        s = f.getvalue()
    s = s.splitlines()
    with open("pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_101.out") as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Mass matrix is positive-definite."
    assert s[2] == "Stiffness matrix is symmetric."
    compare_cbcheck_output(s, sy)


def test_cbcheck_indeterminate_rb_norm():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    bref = n2p.mkdofpv(usetb, "b", [[3, 12356], [19, 3]])[0]

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, bref, usetb, em_filt=2)
        s = f.getvalue()
    s = s.splitlines()

    with open(
        "pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_101_rbnorm.out"
    ) as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Mass matrix is positive-definite."
    assert s[2] == "Stiffness matrix is symmetric."

    compare_cbcheck_output(s, sy)


def test_cbcheck_indeterminate_rb_norm2():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 102
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    bref = n2p.mkdofpv(usetb, "b", [[3, 12356], [19, 3]])[0]

    # write to a string:
    with StringIO() as f:
        out = cb.cbcheck(f, maa, kaa, b, bref, usetb, em_filt=2, rb_norm=True)
        s = f.getvalue()
    s = s.splitlines()

    with open(
        "pyyeti/tests/nas2cam_csuper/yeti_outputs/cbcheck_yeti_102_rbnorm.out"
    ) as f:
        sy = f.read().splitlines()

    assert s[0] == "Mass matrix is symmetric."
    assert s[1] == "Warning: mass matrix is not positive-definite."
    msg = "Warning: stiffness matrix is not symmetric:"
    assert s[2].startswith(msg) or s[4].startswith(msg)

    compare_cbcheck_output(s, sy)


def test_cbcoordchk():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    kaa = kaa[pv]

    uset = nas["uset"][se]
    b = n2p.mksetpv(uset, "a", "b")
    b = np.nonzero(b)[0]

    chk0 = cb.cbcoordchk(kaa, b, b[-6:], verbose=False)
    rbmodes0 = chk0.rbmodes[b]

    # maa = cb.cbreorder(maa, b, last=True)
    kaa = cb.cbreorder(kaa, b, last=True)
    b += kaa.shape[0] - len(b)
    bref = b[-6:]

    chk1 = cb.cbcoordchk(kaa, b, bref, verbose=False)
    rbmodes1 = chk1.rbmodes[b]

    assert np.allclose(chk1.coords, chk0.coords)
    assert np.allclose(rbmodes1, rbmodes0)
    assert np.allclose(chk1.maxerr, chk0.maxerr)
    assert chk0.refpoint_chk == chk1.refpoint_chk == "pass"
    assert abs(chk0.maxerr).max() < 1e-5

    # a case where the refpoint_chk should be 'fail':
    with pytest.warns(la.LinAlgWarning, match=r"Ill\-conditioned matrix"):
        chk2 = cb.cbcoordchk(kaa, b, [25, 26, 27, 31, 32, 33], verbose=False)
    assert chk2.refpoint_chk == "fail"


def test_rbmultchk():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    grids = [[11, 123456], [45, 123456], [60, 123456]]
    drm101, dof101 = n2p.formtran(nas, 101, grids)

    # write and read a file:
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()

    rb = n2p.rbgeom_uset(usetb)
    cb.rbmultchk(name, drm101, "DRM101", rb)
    with open(name) as f:
        sfile = f.read()
    os.remove(name)

    # test rbscale and unit scale:
    with StringIO() as f:
        cb.rbmultchk(f, 0.00259 * drm101, "DRM101", 100 * rb)
        s = f.getvalue()

    pos = s.find(" which is: ")
    pos2 = s[pos:].find("\n")
    assert math.isclose(float(s[pos + 10 : pos + pos2]), 100)
    s = s.splitlines()
    table, nj1 = gettable(s, 15, 0, "Absolute Maximums", 3)
    sbe = np.array(
        [
            [600, 300, 300, 0.00259],
            [600, 300, 300, 0.00259],
            [600, 300, 300, 0.00259],
            [150, -930, 150, 0.00259],
            [600, 300, 300, 0.00259],
            [150, -930, 150, 0.00259],
        ]
    )

    assert np.allclose(table[:, 1:5], sbe)

    # write to a string:
    with StringIO() as f:
        cb.rbmultchk(f, drm101, "DRM101", rb)
        s = f.getvalue()
    assert sfile == s

    # add q-set rows to rb:
    nq = np.count_nonzero(q)
    rb2 = np.vstack((rb, np.zeros((nq, 6))))
    with StringIO() as f:
        cb.rbmultchk(f, drm101, "DRM101", rb2)
        s2 = f.getvalue()
    assert s2 == s

    # check results when b-set are last:
    drm101_last = np.hstack((drm101[:, q], drm101[:, b]))
    with StringIO() as f:
        cb.rbmultchk(f, drm101_last, "DRM101", rb, bset="last")
        s2 = f.getvalue()
    assert s2 == s

    # check results when b-set are last ... using pv:
    with StringIO() as f:
        bsetpv = np.zeros((len(b) + nq), bool)
        bsetpv[-len(b) :] = True
        cb.rbmultchk(f, drm101_last, "DRM101", rb, bset=bsetpv)
        s2 = f.getvalue()
    assert s2 == s

    with StringIO() as f:
        with pytest.raises(ValueError, match="invalid `bset` string"):
            cb.rbmultchk(f, drm101, "asdf", rb, bset="bad string")

    # trim q-set columns out of drm:
    labels = [str(i[0]) + "  " + str(i[1]) for i in dof101]
    with StringIO() as f:
        cb.rbmultchk(
            f,
            drm101[:, b],
            "DRM101",
            rb,
            drm2=drm101[:, b],
            prtnullrows=True,
            labels=labels,
        )
        s2 = f.getvalue()

    # row 16 is now all zeros ... not comparable
    drm2 = drm101.copy()
    drm2[15] = 0
    with StringIO() as f:
        cb.rbmultchk(f, drm2, "DRM101", rb, drm2=drm2, prtnullrows=True, labels=labels)
        s3 = f.getvalue()
    assert s2 == s3

    s = s.splitlines()
    with open("pyyeti/tests/nas2cam_csuper/yeti_outputs/rbmultchk_yeti_101.out") as f:
        sy = f.read().splitlines()

    j = [2]
    jy = [2]
    assert comptable(s, sy, j, jy, label="Extreme ", skip=3, col=11)
    assert comptable(s, sy, j, jy, label=" Row ", skip=2, col=68)
    assert comptable(s, sy, j, jy, label=" Row ", skip=2)
    assert comptable(s, sy, j, jy, label=" Row ", skip=2)


def test_rbmultchk2():
    # write to a string:
    with StringIO() as f:
        cb.rbmultchk(
            f, np.zeros((15, 6)), "DRM", np.eye(6), drm2=np.random.randn(15, 6)
        )
        s = f.getvalue()
    assert s.find(" -- no coordinates detected --") > -1
    assert s.find("All rows in DRM are NULL") > -1
    assert s.find("There are no NULL rows in DRM2.") > -1

    with StringIO() as f:
        cb.rbmultchk(
            f, np.zeros((15, 6)), "DRM", np.eye(6), drm2=np.random.randn(14, 6)
        )
        s = f.getvalue()
    assert s.find("Error: incorrectly sized DRM2") > -1

    drm = np.random.randn(14, 6)
    drm2 = np.random.randn(14, 6)
    drm[::3] = 0.0
    drm2[1::3] = 0.0
    with StringIO() as f:
        cb.rbmultchk(f, drm, "DRM", np.eye(6), drm2=drm2)
        s = f.getvalue()
    assert s.find("different set of NULL rows") > -1

    drm = np.random.randn(14, 6)
    drm2 = np.random.randn(14, 6)
    drm[::3] = 0.0
    drm2[2::3] = 0.0
    with StringIO() as f:
        cb.rbmultchk(f, drm, "DRM", np.eye(6), drm2=drm2)
        s = f.getvalue()
    assert s.find("different set of NULL rows") > -1

    with StringIO() as f:
        with pytest.raises(ValueError, match="`rb` does not have 6 columns"):
            cb.rbmultchk(f, drm, "drm", 1)

    with StringIO() as f:
        with pytest.raises(ValueError, match="failed to get scale of rb modes"):
            cb.rbmultchk(f, drm, "drm", np.zeros((6, 6)))


def test_rbdispchk():
    coords = np.array([[0, 0, 0], [1, 2, 3], [4, -5, 25]])
    rb = n2p.rbgeom(coords)
    xyz_pv = ytools.mkpattvec([0, 1, 2], 3 * 6, 6).ravel()
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

    sbe = [
        "",
        "Coordinates Determined from Rigid-Body Displacements:",
        "         Node      X         Y         Z         Error",
        "        ------  --------  --------  --------   ----------",
        "             1      0.00      0.00      0.00   0.0000e+00",
        "             2      1.00      2.00      3.00   0.0000e+00",
        "             3      4.00     -5.00     25.00   0.0000e+00",
        "",
        "Maximum absolute coordinate location error:    0 units",
        "",
        "",
    ]

    assert s == "\n".join(sbe)

    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed, grids=[100, 200, 300])
        s = f.getvalue()

    sbe = [
        "",
        "Coordinates Determined from Rigid-Body Displacements:",
        "         Node      ID        X         Y         Z         Error",
        "        ------  --------  --------  --------  --------   ----------",
        "             1       100      0.00      0.00      0.00   0.0000e+00",
        "             2       200      1.00      2.00      3.00   0.0000e+00",
        "             3       300      4.00     -5.00     25.00   0.0000e+00",
        "",
        "Maximum absolute coordinate location error:    0 units",
        "",
        "",
    ]

    assert s == "\n".join(sbe)

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

    sbe = [
        "Warning: deviation from standard pattern, node #3 "
        "starting at row 7. \tMax deviation = 0.006 units.",
        "  Rigid-Body Rotations:",
        "    0.0000    25.0000     5.0000",
        "  -25.0060     0.0000     4.0000",
        "   -5.0000    -4.0000     0.0000",
        "",
        "",
        "Coordinates Determined from Rigid-Body Displacements:",
        "         Node      X         Y         Z         Error",
        "        ------  --------  --------  --------   ----------",
        "             1      0.00      0.00      0.00   0.0000e+00",
        "             2      1.00      2.00      3.00   0.0000e+00",
        "             3      4.00     -5.00     25.00   6.0000e-03",
        "",
        "Maximum absolute coordinate location error:  0.006 units",
        "",
        "",
    ]

    assert s == "\n".join(sbe)

    with StringIO() as f:
        coords_out, maxerr = cb.rbdispchk(f, rbtrimmed, grids=[100, 200, 300])
        s = f.getvalue()

    sbe = [
        "Warning: deviation from standard pattern, node ID = 300 "
        "starting at row 7. Max deviation = 0.006 units.",
        "  Rigid-Body Rotations:",
        "    0.0000    25.0000     5.0000",
        "  -25.0060     0.0000     4.0000",
        "   -5.0000    -4.0000     0.0000",
        "",
        "",
        "Coordinates Determined from Rigid-Body Displacements:",
        "         Node      ID        X         Y         Z         Error",
        "        ------  --------  --------  --------  --------   ----------",
        "             1       100      0.00      0.00      0.00   0.0000e+00",
        "             2       200      1.00      2.00      3.00   0.0000e+00",
        "             3       300      4.00     -5.00     25.00   6.0000e-03",
        "",
        "Maximum absolute coordinate location error:  0.006 units",
        "",
        "",
    ]

    assert s == "\n".join(sbe)

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()

    coords_out, maxerr = cb.rbdispchk(name, rbtrimmed, grids=[100, 200, 300])
    with open(name) as f:
        s2 = f.read()
    os.remove(name)
    assert s == s2

    with StringIO() as f:
        with pytest.raises(ValueError, match="`rbdisp` does not have 6 columns"):
            cb.rbdispchk(f, rbtrimmed[:, :4])
        with pytest.raises(ValueError, match="number of rows in `rbdisp`"):
            cb.rbdispchk(f, rbtrimmed[:5, :])


def test_cbcoordchk2():
    k = np.random.randn(14, 14)
    k = k.dot(k.T)
    b = np.arange(4)
    with pytest.raises(ValueError, match="b-set not a multiple of 6"):
        cb.cbcoordchk(k, b, b)

    b2 = np.arange(6)
    with pytest.raises(ValueError, match="reference point must have length of 6"):
        cb.cbcoordchk(k, b2, b)


def test_cbcoordchk3():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    kaa = kaa[pv]

    uset = nas["uset"][se]
    b = n2p.mksetpv(uset, "a", "b")
    b = np.nonzero(b)[0]

    chk0 = cb.cbcoordchk(kaa, b, b[-6:], verbose=False)
    rbmodes0 = chk0.rbmodes[b]

    # maa = cb.cbreorder(maa, b, last=True)
    kaa = cb.cbreorder(kaa, b, last=True)
    b += kaa.shape[0] - len(b)
    bref = b[-6:]

    chk1 = cb.cbcoordchk(kaa, b, bref, verbose=False)
    rbmodes1 = chk1.rbmodes[b]

    assert np.allclose(chk1.coords, chk0.coords)
    assert np.allclose(rbmodes1, rbmodes0)
    assert np.allclose(chk1.maxerr, chk0.maxerr)
    assert chk0.refpoint_chk == chk1.refpoint_chk == "pass"
    assert abs(chk0.maxerr).max() < 1e-5

    # a case where the refpoint_chk should be 'fail':
    with pytest.warns(la.LinAlgWarning, match=r"Ill\-conditioned matrix"):
        chk2 = cb.cbcoordchk(kaa, b, [25, 26, 27, 31, 32, 33], verbose=False)
    assert chk2.refpoint_chk == "fail"


def compare_nets(net, net2):
    for name in net.__dict__:
        if isinstance(net.__dict__[name], list):
            assert net.__dict__[name] == net2.__dict__[name]
        else:
            assert np.allclose(net.__dict__[name], net2.__dict__[name])


def test_mk_net_drms():
    pth = os.path.dirname(inspect.getfile(cb))
    pth = os.path.join(pth, "tests", "nas2cam_csuper")

    # Load the mass and stiffness from the .op4 file
    # This loads the data into a dict:
    mk = op4.load(os.path.join(pth, "inboard.op4"))
    maa = mk["mxx"][0]
    kaa = mk["kxx"][0]

    # Get the USET table The USET table has the boundary DOF
    # information (id, location, coordinate system). This is needed
    # for superelements with an indeterminate interface. The nastran
    # module has the function bulk2uset which is handy for forming the
    # USET table from bulk data.

    uset, coords = nastran.bulk2uset(os.path.join(pth, "inboard.asm"))

    # uset[::6, [0, 3, 4, 5]]
    # array([[   3.,  600.,    0.,  300.],
    #        [  11.,  600.,  300.,  300.],
    #        [  19.,  600.,  300.,    0.],
    #        [  27.,  600.,    0.,    0.]])

    # x s/c is axial (see figure in cbtf or cbcheck tutorial)
    # - make z l/v axial, but pointing down

    # z s/c  ^ y l/v
    #    \   |   / y s/c
    #       \|/
    #  <------
    # x l/v

    sccoord = [
        [900, 1, 0],
        [0, 0, 0],
        [1, 1, 0],  # z is 45 deg between x & y of l/v
        [0, 0, -1],
    ]  # x is -z l/v
    c = np.cos(45 / 180 * np.pi)
    Tl2s = np.array([[0, 0, -1.0], [-c, c, 0], [c, c, 0]])

    # Form b-set partition vector into a-set
    # In this case, we already know the b-set are first:

    n = uset.shape[0]
    b = np.arange(n)

    # array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    #        16, 17, 18, 19, 20, 21, 22, 23])

    # convert s/c from mm, kg --> m, kg
    ref = [600, 150, 150]
    conv = (0.001, 1.0)
    g = 9.80665

    u = n2p.addgrid(None, 1, "b", sccoord, [0, 0, 0], sccoord)
    Tsc2lv = np.zeros((6, 6))
    T = u.iloc[3:, 1:]
    Tsc2lv[:3, :3] = T
    Tsc2lv[3:, 3:] = T
    assert np.allclose(Tl2s.T, Tsc2lv[:3, :3])

    net = cb.mk_net_drms(
        maa, kaa, b, uset=uset, ref=ref, sccoord=sccoord, conv=conv, g=g
    )

    usetbq, c, bset = nastran.asm2uset(os.path.join(pth, "inboard.asm"))
    with pytest.raises(ValueError, match="number of rows in `uset`"):
        cb.mk_net_drms(
            maa, kaa, b[:3], uset=usetbq, ref=ref, sccoord=Tl2s, conv=conv, g=g
        )

    net2 = cb.mk_net_drms(
        maa, kaa, b, uset=usetbq, ref=ref, sccoord=Tl2s, conv=conv, g=g
    )

    # rb modes in system units:
    uset2, ref2 = cb.uset_convert(uset, ref, conv)
    rb = n2p.rbgeom_uset(uset2, ref2)
    l_sc = net.ifltma_sc[:, :n] @ rb
    l_lv = net.ifltma_lv[:, :n] @ rb
    l_scd = net.ifltmd_sc[:, :n] @ rb
    l_lvd = net.ifltmd_lv[:, :n] @ rb
    a_sc = net.ifatm_sc[:, :n] @ rb
    a_lv = net.ifatm_lv[:, :n] @ rb
    c_sc = net.cgatm_sc[:, :n] @ rb
    c_lv = net.cgatm_lv[:, :n] @ rb

    sbe = np.eye(6)
    sbe[:3] *= 1 / g
    assert np.allclose(a_sc, sbe)

    # calc what the interface forces should be:
    # - acceleration = 1 m/s**2 = 1000 mm/s**2
    # - the forces should be very similar to the 6x6 mass
    #   matrix at the reference point ... which, conveniently,
    #   is provided in the cbtf tutorial:
    mass = np.array(
        [
            [1.755, 0.0, -0.0, 0.0, 0.0, 0.0],
            [0.0, 1.755, -0.0, -0.0, 0.0, 772.22],
            [-0.0, -0.0, 1.755, 0.0, -772.22, -0.0],
            [0.0, -0.0, 0.0, 35905.202, -0.0, -0.0],
            [0.0, 0.0, -772.22, -0.0, 707976.725, 109.558],
            [0.0, 772.22, -0.0, -0.0, 109.558, 707976.725],
        ]
    )
    sbe = mass
    sbe[:, :3] *= 1000  # scale up translations
    assert abs(sbe - l_sc).max() < 0.5

    assert np.allclose(Tsc2lv @ a_sc, a_lv)
    assert np.allclose(Tsc2lv @ c_sc, c_lv)
    assert abs(l_scd).max() < 1e-6 * abs(l_sc).max()
    assert abs(l_lvd).max() < 1e-6 * abs(l_lv).max()
    scale = np.array([[1000], [1000], [1000], [1000000], [1000000], [1000000]])
    assert np.allclose((1 / scale) * (Tsc2lv @ l_sc), l_lv)

    # height and mass values from cbcheck tutorial (and then refined):
    m_kg = 1.75505183
    h_m = 1.039998351 - 0.6
    assert abs(net.height_lv - h_m) < 0.000001
    assert abs(net.weight_lv - m_kg * g) < 0.000001
    assert abs(net.height_sc - 1000 * h_m) < 0.000001 * 1000
    assert abs(net.weight_sc - 1000 * m_kg * g) < 0.000001 * 1000
    assert net.scaxial_sc == 0
    assert net.scaxial_lv == 2

    compare_nets(net, net2)

    # check the natural unit output:
    net3 = cb.mk_net_drms(
        maa, kaa, b, uset=uset, ref=ref, sccoord=Tl2s, conv=conv, g=g, tau=("mm", "m")
    )
    for drm in ("ifatm", "cgatm"):
        if drm == "ifatm":
            # only ifatm has 12 rows
            drm1 = getattr(net, drm)
            drm3 = getattr(net3, drm)
            assert np.allclose(drm3[:3], drm1[:3] * g * 1000)
            assert np.allclose(drm3[3:6], drm1[3:6])
            assert np.allclose(drm3[6:9], drm1[6:9] * g)
            assert np.allclose(drm3[9:], drm1[9:])
        for ext, factor in (("_sc", 1000), ("_lv", 1)):
            drm1 = getattr(net, drm + ext)
            drm3 = getattr(net3, drm + ext)
            assert np.allclose(drm3[:3], drm1[:3] * g * factor)
            assert np.allclose(drm3[3:], drm1[3:])

    labels3 = [i.replace("g", "mm/s^2") for i in net.ifatm_labels[:6]] + [
        i.replace("g", "m/s^2") for i in net.ifatm_labels[6:]
    ]
    print(labels3)
    print()
    print(net3.ifatm_labels)
    assert labels3 == net3.ifatm_labels

    net4 = cb.mk_net_drms(
        maa, kaa, b, uset=uset, ref=ref, sccoord=Tl2s, conv=conv, g=g, tau=("g", "m")
    )
    for drm in ("ifatm", "cgatm"):
        if drm == "ifatm":
            # only ifatm has 12 rows
            drm1 = getattr(net, drm)
            drm3 = getattr(net3, drm)
            drm4 = getattr(net4, drm)
            assert np.allclose(drm4[:3], drm1[:3])
            assert np.allclose(drm4[3:6], drm1[3:6])
            assert np.allclose(drm4[6:9], drm3[6:9])
            assert np.allclose(drm4[9:], drm3[9:])
        for ext, net_ in (("_sc", net), ("_lv", net3)):
            drm1 = getattr(net_, drm + ext)
            drm4 = getattr(net4, drm + ext)
            assert np.allclose(drm4[:3], drm1[:3])
            assert np.allclose(drm4[3:], drm1[3:])

    labels4 = net.ifatm_labels[:6] + net3.ifatm_labels[6:]
    assert labels4 == net4.ifatm_labels

    # test mixed b-set/q-set:
    na = maa.shape[0]
    q = np.r_[n:na]
    newb = np.r_[0:12, na - 12 : na]
    newq = np.r_[12 : na - 12]
    kaa_newb = np.empty((na, na))
    maa_newb = np.empty((na, na))
    for newr, oldr in [(newb, b), (newq, q)]:
        for newc, oldc in [(newb, b), (newq, q)]:
            maa_newb[np.ix_(newr, newc)] = maa[np.ix_(oldr, oldc)]
            kaa_newb[np.ix_(newr, newc)] = kaa[np.ix_(oldr, oldc)]

    net5 = cb.mk_net_drms(
        maa_newb,
        kaa_newb,
        newb,
        uset=uset,
        ref=ref,
        sccoord=sccoord,
        conv=conv,
        g=g,
        reorder=False,
    )

    assert np.allclose(net5.ifltma[:, newb], net.ifltma[:, b])
    assert np.allclose(net5.ifltma[:, newq], net.ifltma[:, q])

    net6 = cb.mk_net_drms(
        maa_newb,
        kaa_newb,
        newb,
        uset=uset,
        ref=ref,
        sccoord=sccoord,
        conv=conv,
        g=g,
        reorder=False,
        rbe3_indep_dof=123456,
    )

    assert np.allclose(net6.ifltma, net5.ifltma)
    # translations match:
    assert np.allclose(net6.ifatm_sc[:3], net5.ifatm_sc[:3])
    # rotations do not:
    assert not np.allclose(net6.ifatm_sc[3:], net5.ifatm_sc[3:])
    vec = ytools.mkpattvec([3, 4, 5], len(newb), 6).ravel()
    assert (net5.ifatm_sc[:, newb[vec]] == 0.0).all()
    assert not (net6.ifatm_sc[:, newb[vec]] == 0.0).all()


def test_mk_net_drms_6dof():
    # same as above, but reduced to single point interface
    pth = os.path.dirname(inspect.getfile(cb))
    pth = os.path.join(pth, "tests", "nas2cam_csuper")
    mk = op4.load(os.path.join(pth, "inboard.op4"))
    maa = mk["mxx"][0]
    kaa = mk["kxx"][0]
    uset, coords = nastran.bulk2uset(os.path.join(pth, "inboard.asm"))
    n = uset.shape[0]
    b = np.arange(n)
    q = np.arange(n, maa.shape[0])
    ttl = maa.shape[0]

    # reduce to single point interface:
    rb = n2p.rbgeom_uset(uset, [600, 150, 150])

    # old = {b_24} = {rb.T @ b_6}  = [rb.T  0_6?] {b_6}
    #       {q_?}    {q_?}           [0_?6   I??] {q_?}
    trans = np.zeros((len(q) + 6, ttl))
    trans[:6, :n] = rb.T
    trans[6:, n:] = np.eye(len(q))
    maa = trans @ maa @ trans.T
    kaa = trans @ kaa @ trans.T

    assert abs(kaa[:6, :6]).max() < 0.02
    kaa[:6, :] = 0.0
    kaa[:, :6] = 0.0

    # no conversion, no coordinate change:
    g = 9.80665
    n = 6
    b = np.arange(n)
    net = cb.mk_net_drms(maa, kaa, b, g=g)
    net2 = cb.mk_net_drms(maa, kaa, b, g=g, bsubset=b)
    l_sc = net.ifltma_sc[:, :n]
    l_lv = net.ifltma_lv[:, :n]
    l_scd = net.ifltmd_sc[:, :n]
    l_lvd = net.ifltmd_lv[:, :n]
    a_sc = net.ifatm_sc[:, :n]
    a_lv = net.ifatm_lv[:, :n]
    c_sc = net.cgatm_sc[:, :n]
    c_lv = net.cgatm_lv[:, :n]

    sbe = np.eye(6)
    sbe[:3] *= 1 / g
    assert np.allclose(a_sc, sbe)
    mass = np.array(
        [
            [1.755, 0.0, -0.0, 0.0, 0.0, 0.0],
            [0.0, 1.755, -0.0, -0.0, 0.0, 772.22],
            [-0.0, -0.0, 1.755, 0.0, -772.22, -0.0],
            [0.0, -0.0, 0.0, 35905.202, -0.0, -0.0],
            [0.0, 0.0, -772.22, -0.0, 707976.725, 109.558],
            [0.0, 772.22, -0.0, -0.0, 109.558, 707976.725],
        ]
    )
    sbe = mass
    assert abs(sbe - l_sc).max() < 0.0005

    Tsc2lv = np.eye(6)
    assert np.allclose(Tsc2lv @ a_sc, a_lv)
    assert np.allclose(Tsc2lv @ c_sc, c_lv)
    assert abs(l_scd).max() < 1e-6 * abs(l_sc).max()
    assert abs(l_lvd).max() < 1e-6 * abs(l_lv).max()
    assert np.allclose((Tsc2lv @ l_sc), l_lv)

    # height and mass values from cbcheck tutorial (and then refined):
    m_kg = 1.75505183
    h_m = 1039.998351 - 600
    assert abs(net.height_lv - h_m) < 0.0001
    assert abs(net.weight_lv - m_kg * g) < 0.0001
    assert abs(net.height_sc - h_m) < 0.0001
    assert abs(net.weight_sc - m_kg * g) < 0.0001
    assert net.scaxial_sc == 0
    assert net.scaxial_lv == 0

    compare_nets(net, net2)


def test_cglf_moment_signs():
    pth = "pyyeti/tests/cla_test_data"

    se = 101
    uset, coords = nastran.bulk2uset(os.path.join(pth, "outboard.asm"))
    dct = op4.read(os.path.join(pth, "outboard.op4"))
    maa = dct["mxx"]
    kaa = dct["kxx"]
    atm = dct["mug1"]
    ltm = dct["mef1"]
    pch = os.path.join(pth, "outboard.pch")
    atm_labels = [
        "Grid {:4d}-{:1d}".format(grid, dof) for grid, dof in nastran.rddtipch(pch)
    ]
    ltm_labels = [
        "CBAR {:4d}-{:1d}".format(cbar, arg)
        for cbar, arg in nastran.rddtipch(pch, "tef1")
    ]

    nb = uset.shape[0]
    nq = maa.shape[0] - nb
    bset = np.arange(nb)
    qset = np.arange(nq) + nb
    ref = [600.0, 150.0, 150.0]
    g = 9806.65

    # use addgrid to get coordinate transformations from lv to sc:
    cid = [1, 0, 0]
    A = [0, 0, 0]
    # define sc in terms of lv coords:
    # (all drawn out by hand)
    BC = [
        [[0, 0, 1.0], [1.0, 0, 0]],  # lv x is up
        [[0, 0, -1.0], [0, 1.0, 0]],  # lv y is up
        [[-1.0, 0, 0.0], [0, 0, 1.0]],  # lv z is up
        [[0, 0, -1.0], [-1.0, 0, 0]],  # lv x is down
        [[0, 0, 1.0], [0, -1.0, 0]],  # lv y is down
        [[0, -1.0, 0], [0, 0, -1.0]],  # lv z is down
    ]
    Ts = []
    nets = []
    rb = n2p.rbgeom_uset(uset, ref)
    rbcglfa = []
    for bc in BC:
        CI = n2p.mkusetcoordinfo([cid, A, *bc], None, {})
        T = CI[2:]
        Ts.append(T)
        net = cb.mk_net_drms(maa, kaa, bset, uset=uset, ref=ref, g=g, sccoord=T)
        nets.append(net)
        rba = net.cglfa[:, :24] @ rb
        rbcglfa.append(rba)
        # sc rows:
        assert np.all(np.sign(rba[1, [1, 5]]) == np.sign(rba[3, [1, 5]]))
        assert np.all(np.sign(rba[2, [2, 4]]) == np.sign(rba[4, [2, 4]]))
        # lv rows:
        assert np.all(np.sign(rba[6, [1, 5]]) == np.sign(rba[8, [1, 5]]))
        assert np.all(np.sign(rba[7, [2, 4]]) == np.sign(rba[9, [2, 4]]))

    wh_sc = nets[0].weight_sc * nets[0].height_sc
    wh_lv = nets[0].weight_lv * nets[0].height_lv
    n = nets[0].cgatm_sc.shape[1]
    # x is down:
    cgdrm = np.vstack(
        (
            # 5 s/c rows
            nets[0].cgatm_sc[:3],
            -nets[0].ifltma_sc[5] / wh_sc,
            nets[0].ifltma_sc[4] / wh_sc,
            # 5 l/v rows
            nets[0].cgatm_lv[:3],
            -nets[0].ifltma_lv[5] / wh_lv,
            nets[0].ifltma_lv[4] / wh_lv,
            # 4 RSS rows ... filled in during data recovery
            np.zeros((4, n)),
        )
    )
    assert np.allclose(cgdrm, nets[0].cglfa)
