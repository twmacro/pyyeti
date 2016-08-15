import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from pyyeti import dsp, locate, ytools
from nose.tools import *


def test_resample():
    p = 5
    q = 5
    n = 530
    d = np.random.randn(n) - 55
    x = np.arange(n) + 100

    rtol = atol = 1e-15
    d0, x0, fir0 = dsp.resample(d, p, q, t=x, getfir=1)
    assert np.allclose(d0, d, rtol, atol)
    assert np.allclose(x0, x, rtol, atol)
    n = 2*10 + 1
    mid = (n-1)//2
    assert np.allclose(fir0[mid], 1, rtol, atol)
    fir0[mid] = 0
    assert np.allclose(0, fir0, rtol, atol)

    pts = 5
    d0, x0, fir0 = dsp.resample(d, p, q, t=x, getfir=1, pts=pts)
    assert np.allclose(d0, d, rtol, atol)
    assert np.allclose(x0, x, rtol, atol)
    n = 2*pts + 1
    mid = (n-1)//2
    assert np.allclose(fir0[mid], 1, rtol, atol)
    fir0_ = fir0.copy()
    fir0[mid] = 0
    assert np.allclose(0, fir0, rtol, atol)
    d1, fir1 = dsp.resample(d, p, q, getfir=1, pts=pts)
    d2, x2 = dsp.resample(d, p, q, t=x, pts=pts)
    assert np.all(d1 == d0)
    assert np.all(d1 == d2)
    assert np.allclose(fir0_, fir1, rtol, atol)
    assert np.allclose(x2, x, rtol, atol)

    D = np.vstack((d, d)).T
    D2, X2 = dsp.resample(D, p, q, t=x, pts=pts)
    assert np.all(d1 == D[:, 0])
    assert np.all(d1 == D[:, 1])
    assert np.allclose(x2, X2)

    lcc = {}
    for pts in [10, 50]:
        dup4, xup4 = dsp.resample(d, 4, 1, t=x, pts=pts)
        dup2, xup2 = dsp.resample(dup4, 10, 20, t=xup4, pts=pts)
        d0, x0 = dsp.resample(dup2, 1, 2, t=xup2, pts=pts)
        assert np.allclose(x0, x, rtol, atol)
        lcc[pts] = stats.pearsonr(d0, d)[0]
        assert 1-lcc[pts] < .015
        assert np.abs(np.mean(d)-np.mean(d0)) < .005
    assert lcc[50] > lcc[10]


def test_exclusive_sgfilter():
    assert_raises(ValueError, dsp.exclusive_sgfilter,
                  [1, 2, 3, 4, 5, 6, 7, 8], 5, 'badoption')
    assert_raises(ValueError, dsp.exclusive_sgfilter,
                  [1, 2, 3, 4, 5, 6, 7, 8], 5, 6)
    x = np.arange(4*2*150*3.0).reshape(4, 2, 150, 3)
    x1 = signal.savgol_filter(x, 11, 0, axis=2)
    x2 = dsp.exclusive_sgfilter(x, 11, exclude_point=None, axis=2)
    assert np.allclose(x1, x2)


def test_windowends():
    s = dsp.windowends(np.ones(8), 4)
    assert np.allclose([0., 0.25, 0.75, 1., 1., 1., 1., 1.], s)
    s = dsp.windowends(np.ones(8), .7, ends='back')
    assert np.allclose([1., 1., 1., 1., 0.85355339059327373,
                        0.5, 0.14644660940672621, 0.], s)
    s = dsp.windowends(np.ones(8), .5, ends='both')
    assert np.allclose([0., 0.25, 0.75, 1., 1., 0.75, 0.25, 0.], s)
    s = dsp.windowends(np.ones(8), .5, ends='none')
    assert np.allclose([1., 1., 1., 1., 1., 1., 1., 1.], s)
    s = dsp.windowends(np.ones(8), 3)
    assert np.allclose([0., 0.5, 1., 1., 1., 1., 1., 1.], s)
    s = dsp.windowends(np.ones(8), 2)
    assert np.allclose([0., 0.5, 1., 1., 1., 1., 1., 1.], s)


def test_waterfall():
    from pyyeti import srs
    sig, t, f = ytools.gensweep(10, 1, 50, 4)
    sr = 1/t[1]
    frq = np.arange(1., 5.1)
    Q = 20
    sig2 = sig[:int(sr*1.5), None]
    def func(s):
        return srs.srs(s, sr, frq, Q), frq
    mp, t, f = dsp.waterfall(sig, sr, 2, .5, func,
                             which=0, freq=1)
    mp, t, f = dsp.waterfall(sig2, sr, 2, .5, func,
                             which=0, freq=1)
    assert_raises(ValueError, dsp.waterfall, sig, sr, 2, 0.5, func,
                  which=None, freq=1)
    assert_raises(ValueError, dsp.waterfall, sig, sr, 2, 1.5, func,
                  which=None, freq=frq)
    assert_raises(ValueError, dsp.waterfall, sig, sr, 2, 1.0, func,
                  which=None, freq=frq)
    assert_raises(ValueError, dsp.waterfall, sig, sr, 2, -0.5, func,
                  which=None, freq=frq)
    sig = np.hstack((sig2, sig2))
    assert_raises(ValueError, dsp.waterfall, sig, sr, 2, .5, func,
                  which=None, freq=frq)


def test_waterfall2():
    N = 5000
    sig = np.ones(N)
    _t = np.arange(N)/1000.
    sr = 1/_t[1]
    frq = np.arange(1., 5.1)
    def func(s):
        return np.ones(len(frq)), frq
    # test different overlaps
    S = 1.0
    O = 0.5
    mp, t, f = dsp.waterfall(sig, sr, S, O, func,
                             which=0, freq=1)
    step = S*(1-O)
    tcmp = np.arange(S/2., _t[-1]-S/2+step/2, step)
    assert np.allclose(t, tcmp)

    O = 0.9
    mp, t, f = dsp.waterfall(sig, sr, S, O, func,
                             which=0, freq=1)
    step = S*(1-O)
    tcmp = np.arange(S/2., _t[-1]-S/2+step/2, step)
    assert np.allclose(t, tcmp)

    O = 0.1
    mp, t, f = dsp.waterfall(sig, sr, S, O, func,
                             which=0, freq=1)
    step = S*(1-O)
    tcmp = np.arange(S/2., _t[-1]-S/2+step/2, step)
    assert np.allclose(t, tcmp)


def test_get_turning_pts():
    y, x = dsp.get_turning_pts([1, 2, 3, 3, 3], [1, 2, 3, 4, 5],
                                  getindex=False)
    assert np.all([1, 3, 3] == y)
    assert np.all([1, 3, 5] == x)
    y2 = dsp.get_turning_pts([1, 2, 3, 3, 3], getindex=False)
    assert np.all(y == y2)
    assert_raises(ValueError, dsp.get_turning_pts,
                  [1, 2, 3, 3, 3], [1, 2, 3, 3, 5])
    assert_raises(ValueError, dsp.get_turning_pts,
                  [1, 2, 3, 3, 3, 5], [1, 2, 3, 4, 5])


def test_calcenv():
    p = 5
    x = [1, 2, 3, 2, 5]
    y = [1, 4, 9, 4, 2]
    assert_raises(ValueError, dsp.calcenv, x, y, p)

    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16]
    assert_raises(ValueError, dsp.calcenv, x, y, p)

    assert_raises(ValueError, dsp.calcenv, [0, 1, 2],
                  [0, 5, 0], method='default')

    xe, ye, *_ = dsp.calcenv([0, 1, 2], [0, 5, 0], makeplot='no')
    x_sb = [0, .95, 1.05, 2*1.05]
    y_sb = [0, 5, 5, 0]
    yi = np.interp(xe, x_sb, y_sb)
    err = abs(yi-ye).max()
    assert err < .006

    xen, yen, *_ = dsp.calcenv([0, 1, 2], [0, -5, 0],
                                  makeplot='no', method='min')
    assert np.allclose(xe, xen)
    assert np.allclose(ye, -yen)

    xe, ye, *_ = dsp.calcenv([0, 1, 2], [0, 5, 0],
                                p=3, n=4000, makeplot='no')
    x_sb = [0, .97, 1.03, 2*1.03]
    y_sb = [0, 5, 5, 0]
    yi = np.interp(xe, x_sb, y_sb)
    assert abs(yi-ye).max() < err/2

    xe, ye, *_ = dsp.calcenv([0, 1, 2], [0, -5, 0],
                                p=20, makeplot='no', method='max')
    # find where the 2 line segments meet
    # first, get the line segments:
    #  y1 => (0, 0) to (1.20*1, -5): y1 = -5/1.2 x
    #  y2 => (.8, -5) to (.8*2, 0):  y2 = 5/.8 x + b
    #     get b:  0 = 5/.8 * 1.6 + b; b = -10
    #  y1 = -5/1.2 x
    #  y2 = 6.25 x - 10
    #  --> intersect @ x = 10 / (6.25+5/1.2)
    x_int = 10 / (6.25+5/1.2)
    y_int = -5/1.2 * x_int
    x_sb = [0, x_int, 1.6]
    y_sb = [0, y_int, 0]
    yi = np.interp(xe, x_sb, y_sb)
    assert abs(yi-ye).max() < .006

    xen, yen, *_ = dsp.calcenv([0, 1, 2], [0, 5, 0],
                                  p=20, makeplot='no', method='min')
    assert np.allclose(xe, xen)
    assert np.allclose(ye, -yen)

    (xmax, ymax,
     xmin, ymin, h) = dsp.calcenv([0, 1, 2], [0, -5, 0], n=200,
                                     p=20, makeplot='no', base=None)
    yi = np.interp(xmax, x_sb, y_sb)
    assert abs(yi-ymax).max() < .06
    x_sb = [0, .8, 1.2, 2*1.2]
    y_sb = [0, -5, -5, 0]
    yi = np.interp(xmin, x_sb, y_sb)
    assert abs(yi-ymin).max() < .06

    (xmax, ymax,
     xmin, ymin, h) = dsp.calcenv([0, 1, 2], [0, -5, 0], n=200,
                                     p=20, makeplot='no',
                                     method='both')
    assert np.all(xmax == [0, 2])
    assert np.all(ymax == [0, 0])
    assert h is None
    x_sb = [0, .8, 1.2, 2*1.2]
    y_sb = [0, -5, -5, 0]
    yi = np.interp(xmin, x_sb, y_sb)
    assert abs(yi-ymin).max() < .06

    # test base:
    base = 2
    xe, ye, *_ = dsp.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                 makeplot='no', n=100)
    xeb, yeb, *_ = dsp.calcenv([0, 1, 2], [0, 5, 0], makeplot='no',
                                  base=base, n=100)
    assert np.allclose(xe, xeb)
    assert np.allclose(ye+base, yeb)

    base = 2
    xe, ye, *_ = dsp.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                makeplot='no', n=100, method='min')
    xeb, yeb, *_ = dsp.calcenv([0, 1, 2], [0, 5, 0], makeplot='no',
                                  base=base, n=100, method='min')
    assert np.allclose(xe, xeb)
    assert np.allclose(ye+base, yeb)

    base = 2
    (xex, yex,
     xen, yen, h) = dsp.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                   makeplot='no', n=100, method='both')
    (xexb, yexb,
     xenb, yenb, h) = dsp.calcenv([0, 1, 2], [0, 5, 0],
                                     makeplot='no',
                                     base=base, n=100, method='both')
    assert np.allclose(xex, xexb)
    assert np.allclose(yex+base, yexb)
    assert np.allclose(xen, xenb)
    assert np.allclose(yen+base, yenb)


def test_fixtime():
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    assert_raises(ValueError, dsp.fixtime,
                  ([1, 2, 3, 4], [1, 2, 3]))

    t = [0, 1, 6, 7]
    y = [1, 2, 3, 4]
    with assert_warns(RuntimeWarning):
        tn, yn = dsp.fixtime((t, y), sr=1)
        ty2 = dsp.fixtime(np.vstack([t, y]).T, sr=1)

    t2, y2 = ty2.T
    assert np.all(tn == t2)
    assert np.all(yn == y2)
    assert np.all(tn == np.arange(8))
    assert np.all(yn == [1, 2, 2, 2, 2, 2, 3, 4])

    with assert_warns(RuntimeWarning):
        (t2, y2), info = dsp.fixtime((t, y), getall=True, sr=1)
        #((t2, y2), pv,
        # sr_stats, tp) = dsp.fixtime((t, y), getall=True,
        #                                sr=1)
    assert np.all(tn == t2)
    assert np.all(yn == y2)
    assert np.all(info.alldrops.dropouts == [])
    assert np.allclose(info.sr_stats, [1, .2, 3/7, 1, 200/3])
    assert np.all(info.tp == np.arange(4))
    assert info.despike_info is None
    assert_raises(ValueError, dsp.fixtime, (1, 1, 1))

    drop = -1.40130E-45
    t = [0, .5, 1, 6, 7]
    y = [1, drop, 2, 3, 4]

    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime([t, y], sr=1)
    assert np.all(tn == t2)
    assert np.all(yn == y2)

    t = [1, 2, 3]
    y = [drop, drop, drop]
    assert_warns(RuntimeWarning, dsp.fixtime, (t, y), 'auto')


    (t2, y2), info = dsp.fixtime((t, y), 'auto', deldrops=False,
                                 getall=True)
    assert np.all(info.tp == [0, 2])
    assert np.all(t2 == t)
    assert np.all(y2 == y)
    assert info.alldrops.dropouts is None

    t = np.arange(100.)
    t = np.hstack((t, 200.))
    y = np.arange(101)
    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime((t, y), 'auto')
    assert np.all(t2 == np.arange(100.))
    assert np.all(y2 == np.arange(100))

    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime((t, y), 'auto', delouttimes=0)
    assert np.all(t2 == np.arange(200.1))
    sbe = np.ones(201, int)*99
    sbe[:100] = np.arange(100)
    sbe[-1] = 100
    assert np.all(y2 == sbe)

    t = [0, 1, 2, 3, 4, 5, 7, 6]
    y = [0, 1, 2, 3, 4, 5, 7, 6]
    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime((t, y), 1)
    assert np.all(t2 == np.arange(8))
    assert np.all(y2 == np.arange(8))

    assert_raises(ValueError, dsp.fixtime, (t, y), 1,
                  negmethod='stop')

    t = np.arange(8)[::-1]
    assert_raises(ValueError, dsp.fixtime, (t, y), 1)

    t = np.arange(0, 1, .001)
    noise = np.random.randn(t.size)
    noise = noise / noise.max() / 100000
    y = np.random.randn(t.size)
    t2, y2 = dsp.fixtime((t+noise, y), 'auto', base=0)
    assert np.allclose(t, t2)
    assert np.all(y == y2)

    dt = np.ones(100)
    dt[80:] = .91*dt[80:]
    t = dt.cumsum()
    y = np.arange(100)
    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime((t, y), 1)

    dt = np.ones(100)
    dt[80:] = 1.08*dt[80:]
    t = dt.cumsum()
    y = np.arange(100)
    with assert_warns(RuntimeWarning):
        t2, y2 = dsp.fixtime((t, y), 1)


def test_fixtime2():
    dt = .001
    sr = 1000
    t = np.arange(0, 1, dt)
    t[1:] += dt/2
    # t[400:] -= dt/2
    noise = np.random.randn(t.size)
    noise = noise / noise.max() / 100000
    y = np.random.randn(t.size)
    told = t+noise

    t2, y2 = dsp.fixtime((t+noise, y), 'auto', base=None)
    pv = locate.find_subseq(y2, y[1:-1])
    assert pv.size > 0

    t3, y3 = dsp.fixtime((t+noise, y), 'auto', base=0.0)
    pv = locate.find_subseq(y2, y[1:-1])
    assert pv.size > 0
    assert not np.allclose(t2[0], t3[0])
    assert np.allclose(t2-t2[0], t3-t3[0])
    assert abs(t3 - .003).min() < 1e-14


def test_fixtime_drift():
    L = 10000
    for factor in (.9995, 1.0005):
        t = np.arange(L)*(.001*factor)
        y = np.random.randn(L)
        t2, y2 = dsp.fixtime((t, y), 1000, base=0)
        t3, y3 = dsp.fixtime((t, y), 1000, fixdrift=1, base=0)

        i = np.argmax(y[3:100])
        i2 = np.nonzero(y2[:103] == y[i])[0][0]
        i3 = np.nonzero(y3[:103] == y[i])[0][0]
        d2 = abs(t2[i2] - t[i])
        d3 = abs(t3[i3] - t[i])

        L = len(y)
        i = L-100 + np.argmax(y[-100:-10])
        L = len(y2)
        i2 = L-110 + np.nonzero(y2[-110:] == y[i])[0][0]
        L = len(y3)
        i3 = L-110 + np.nonzero(y3[-110:] == y[i])[0][0]
        d2 += abs(t2[i2] - t[i])
        d3 += abs(t3[i3] - t[i])
        assert d3 < d2


def test_fixtime_too_many_tp():
    # actual data:
    fd = np.array([[ 0.91878125,  1.17070317],
                   [ 0.9226875 ,  1.17070317],
                   [ 0.9226875 ,  1.17070317],
                   [ 0.92659375,  1.17070317],
                   [ 0.92659375,  1.17070317],
                   [ 0.9305    ,  1.17070317],
                   [ 0.9305    ,  1.17070317],
                   [ 0.93440625,  1.17070317],
                   [ 0.93440625,  1.17070317],
                   [ 0.9383125 ,  1.17070317],
                   [ 0.9383125 ,  1.36582029],
                   [ 0.94221875,  1.29265141],
                   [ 0.94221875,  0.97558594],
                   [ 0.946125  ,  1.63410652],
                   [ 0.946125  ,  0.92680663],
                   [ 0.95003125,  1.5365479 ],
                   [ 0.95003125,  1.07314456],
                   [ 0.9539375 ,  1.26826179],
                   [ 0.9539375 ,  1.31704104],
                   [ 0.95784375,  1.1950928 ],
                   [ 0.95784375,  1.29265141],
                   [ 0.96175   ,  1.04875493],
                   [ 0.96175   ,  1.34143066],
                   [ 0.96565625,  0.92680663],
                   [ 0.96565625,  1.36582029],
                   [ 0.9695625 ,  0.87802738],
                   [ 0.9695625 ,  1.24387205],
                   [ 0.97346875,  0.97558594],
                   [ 0.97346875,  1.14631355],
                   [ 0.977375  ,  0.99997562],
                   [ 0.977375  ,  1.14631355],
                   [ 0.98128125,  1.07314456],
                   [ 0.98128125,  1.07314456],
                   [ 0.9851875 ,  1.1950928 ],
                   [ 0.9851875 ,  1.09753418],
                   [ 0.98909375,  1.1950928 ],
                   [ 0.98909375,  1.1219238 ],
                   [ 0.993     ,  1.1950928 ],
                   [ 0.993     ,  1.17070317],
                   [ 0.99690625,  1.1950928 ],
                   [ 0.99690625,  1.14631355],
                   [ 1.0008125 ,  1.17070317]])

    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    assert_raises(ValueError, dsp.fixtime,
                  ([1, 2, 3, 4], [1, 2, 3]))
    assert_raises(ValueError, dsp.fixtime,
                  np.random.randn(3, 3, 3))

    with assert_warns(RuntimeWarning):
        t, d = dsp.fixtime((*fd.T,), sr='auto')
    # assert np.all(fd[:, 1] == d)
    assert np.allclose(np.diff(t), .002)
    assert np.allclose(t[0], fd[0, 0])


def test_fixtime_naninf():
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    t = np.arange(15)
    y = [1, 2, 3, 4, np.nan, 6, 7, np.inf, 9, 10,
         -np.inf, 12, -1.40130E-45, 14, 15]

    with assert_warns(RuntimeWarning):
        (tn, yn), info = dsp.fixtime((t, y), sr=1, getall=1)

    d = info.alldrops.dropouts
    assert np.all(d == [4, 7, 10, 12])
    assert np.all(t == tn)
    assert np.all(yn == [1, 2, 3, 4, 4, 6, 7, 7, 9,
                         10, 10, 12, 12, 14, 15])


def test_fixtime_despike():
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    t = np.arange(16)
    y = [1, 2, 3, 4, np.nan, 6, 7, np.inf, 9, 10,
         -np.inf, 12, -1.40130E-45, 14, 134, 15]

    with assert_warns(RuntimeWarning):
        (tn, yn), info = dsp.fixtime((t, y), sr=1, getall=1,
                                     delspikes=dict(n=9, sigma=2))
        (tn2, yn2), info2 = dsp.fixtime((t, y), sr=1, getall=1,
                                        delspikes=dict(sigma=2))

    d = info.alldrops.dropouts
    assert np.all(d == range(4, 13))
    assert np.all(t == tn)
    assert np.all(t == tn2)
    assert np.allclose(yn, [1, 2, 3, 4, 4, 4, 4, 4, 4,
                            4, 4, 4, 4, 4, 4, 15])
    assert np.allclose(yn2, yn)
    assert info.despike_info.delspikes['n'] == 9
    assert info2.despike_info.delspikes['n'] == 15
    # spike is at pos 14
    assert np.all(info.alldrops.spikes == 14)
    assert np.all(info2.alldrops.spikes == 14)
    assert np.all(info.alldrops.alldrops == range(4, 15))


def test_fixtime_perfect():
    dt = .001
    t = np.arange(1000)*dt
    y = np.random.randn(len(t))
    t2, y2 = dsp.fixtime((t, y), 'auto')
    assert np.allclose(t2, t)
    assert np.all(y2 == y)


def test_fixtime_perfect_delspike():
    dt = .001
    t = np.arange(1000)*dt
    y = np.sin(2*np.pi*5*t)
    (t2, y2), info = dsp.fixtime((t, y), 'auto',
                                 delspikes=True, getall=1)
    assert np.allclose(t2, t)
    assert np.all(y2 == y)

    assert_raises(ValueError, dsp.fixtime, (t, y), 'auto',
                  delspikes=dict(method='badmethod'), getall=1)


def test_fixtime_exclude_point_change():
    dt = .001
    t = np.arange(1000)*dt
    y = np.empty(1000)
    y[::2] = 0.1
    y[1::2] = -0.1
    ycopy = y.copy()
    ycopy[100:105] = y[99]
    y[100:105] = 10.0

    t2, y2 = dsp.fixtime((t, y), 'auto', delspikes=True,
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    t2, y2 = dsp.fixtime(
        (t, y), 'auto', delspikes=dict(exclude_point='last'),
        verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    assert_raises(ValueError, dsp.fixtime, (t, y), 'auto',
                  delspikes=dict(exclude_point='middle'),
                  verbose=False)

    # some of a ramp up should be deleted:
    y[100:105] = np.arange(0.0, 5.0)*3
    ycopy = y.copy()
    ycopy[102:105] = y[101]
    t2, y2 = dsp.fixtime((t, y), 'auto', delspikes=dict(sigma=18),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    # none of a ramp down should be deleted:
    y[100:105] = np.arange(5.0, 0.0, -1.0)*3
    t2, y2 = dsp.fixtime((t, y), 'auto', delspikes=dict(sigma=18),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == y)


def test_fixtime_exclude_point_change2():
    dt = .001
    t = np.arange(1000)*dt
    y = np.empty(1000)
    y[::2] = 0.1
    y[1::2] = -0.1
    ycopy = y.copy()
    ycopy[100:105] = y[99]
    y[100:105] = 10.0

    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(method='despike'),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    t2, y2 = dsp.fixtime(
        (t, y), 'auto', delspikes=dict(exclude_point='last',
                                       method='despike'),
        verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    t2, y2 = dsp.fixtime(
        (t, y), 'auto', delspikes=dict(exclude_point='middle',
                                       method='despike'),
        verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == y)

    # some of a ramp up should be deleted:
    y[100:105] = np.arange(0.0, 5.0)*3
    ycopy = y.copy()
    ycopy[102:105] = y[101]
    t2, y2 = dsp.fixtime((t, y), 'auto', delspikes=dict(sigma=35,
                                                        method='despike'),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    # none of a ramp down should be deleted:
    y[100:105] = np.arange(5.0, 0.0, -1.0)*3
    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(sigma=35, method='despike'),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == y)


def test_fixtime_simple_despike():
    dt = .001
    t = np.arange(1000)*dt
    y = np.empty(1000)
    y[::2] = 0.1
    y[1::2] = -0.1
    ycopy = y.copy()
    ycopy[100] = y[99]
    y[100] = 10.0

    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(method='simple', sigma=4),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)

    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(method='simple', sigma=4,
                                        maxiter=1),
                         verbose=False)
    assert np.allclose(t2, t)
    assert np.all(y2 == ycopy)
    

def test_fixtime_despike_edge_conditions():
    dt = .001
    n = 45
    t = np.arange(n)*dt
    y = np.empty(n)
    for i in range(1, n-5):
        y[::2] = 0.1
        y[1::2] = -0.1
        ycopy = y.copy()
        s = slice(i, i+5)
        ycopy[s] = y[i-1]
        y[s] = 10.0

        # with despike_diff:
        t2, y2 = dsp.fixtime((t, y), 'auto', delspikes=True,
                             verbose=False)
        assert np.allclose(t2, t)
        if i < n-15:
            assert np.allclose(y2, ycopy)
        else:
            assert np.allclose(y2, y)
        
        t2, y2 = dsp.fixtime((t, y), 'auto',
                             delspikes=dict(exclude_point='last'),
                             verbose=False)
        assert np.allclose(t2, t)
        if i > 10:
            assert np.allclose(y2, ycopy)
        else:
            assert np.allclose(y2, y)

        # now with despike:
        t2, y2 = dsp.fixtime((t, y), 'auto',
                             delspikes=dict(method='despike'),
                             verbose=False)
        assert np.allclose(t2, t)
        if i < n-15-3:
            assert np.allclose(y2, ycopy)
        else:
            assert np.allclose(y2, y)

        t2, y2 = dsp.fixtime((t, y), 'auto',
                             delspikes=dict(method='despike',
                                            exclude_point='last'),
                             verbose=False)
        assert np.allclose(t2, t)
        if i > 13:
            assert np.allclose(y2, ycopy)
        else:
            assert np.allclose(y2, y)


def test_fix_edges2():
    dt = .01
    t = np.arange(50)*dt

    y = np.zeros(50)
    n = 44
    y[n:] = 1
    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(exclude_point='last'),
                         verbose=False)
    assert np.allclose(t2, t[:n])
    assert np.all(y2 == y[:n])

    y = np.zeros(50)
    n = 5
    y[:n] = 1
    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(exclude_point='first'),
                         verbose=False)
    assert np.allclose(t2, t[n:])
    assert np.all(y2 == y[n:])

    y = np.zeros(50)
    n = 44
    y[n:] = 1
    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(exclude_point='last',
                                        method='despike'),
                         verbose=False)
    assert np.allclose(t2, t[:n])
    assert np.all(y2 == y[:n])

    y = np.zeros(50)
    n = 5
    y[:n] = 1
    t2, y2 = dsp.fixtime((t, y), 'auto',
                         delspikes=dict(exclude_point='first',
                                        method='despike'),
                         verbose=False)
    assert np.allclose(t2, t[n:])
    assert np.all(y2 == y[n:])


def test_aligntime():
    dt = .001
    t = np.arange(1000)*dt
    y = np.random.randn(len(t))

    t2 = np.arange(100, 800)*dt
    y2 = np.random.randn(len(t2))

    t3 = np.arange(200, 850)*dt
    y3 = np.random.randn(len(t3))

    dct = dict(ax1=(t, y),
               ax2=(t2, y2),
               ax3=(t3, y3))
    newdct = dsp.aligntime(dct)
    assert abs(newdct['t'][0] - t3[0]) < 1e-14
    print(newdct['t'][-1], t2[-1], abs(newdct['t'][-1] - t2[-1]))
    assert abs(newdct['t'][-1] - t2[-1]) < 1e-14

    newdct = dsp.aligntime(dct, ['ax1', 'ax3'])
    assert abs(newdct['t'][0] - t3[0]) < 1e-14
    assert abs(newdct['t'][-1] - t3[-1]) < 1e-14

    assert_raises(ValueError, dsp.aligntime, dct, ['ax1', 'ax4'])

    newdct = dsp.aligntime(dct, mode='expand')
    assert abs(newdct['t'][0] - t[0]) < 1e-14
    assert abs(newdct['t'][-1] - t[-1]) < 1e-14

    t4 = np.arange(1200, 1350)*dt
    y4 = np.random.randn(len(t4))
    dct['ax4'] = t4, y4
    assert_raises(ValueError, dsp.aligntime, dct, ['ax1', 'ax4'])


def test_aligntime2():
    dt = .001
    t = np.arange(1000)*dt
    y = np.random.randn(len(t))

    dt = .0011
    t2 = np.arange(100, 800)*dt
    y2 = np.random.randn(len(t2))

    dct = dict(ax1=(t, y),
               ax2=(t2, y2))
    assert_raises(ValueError, dsp.aligntime, dct)
    assert_raises(ValueError, dsp.aligntime, dct, mode='expand')


def test_fdscale():
    sig, t, f = ytools.gensweep(10, 1, 12, 8)
    scale = np.array([[0., 1.0],
                      [4., 1.0],
                      [5., 0.5],
                      [8., 0.5],
                      [9., 1.0],
                      [100., 1.0]])
    sig_scaled = dsp.fdscale(sig, 1/t[1], scale)
    x = np.arange(0, 10, .001)
    y = 1/np.interp(x, scale[:, 0], scale[:, 1])
    unscale = np.vstack((x, y)).T
    sig_unscaled = dsp.fdscale(sig_scaled, 1/t[1], unscale)
    assert np.allclose(sig, sig_unscaled, atol=1e-6)

    sig_scaled = dsp.fdscale(sig[:-1], 1/t[1], scale)
    sig_unscaled = dsp.fdscale(sig_scaled, 1/t[1], unscale)
    assert np.allclose(sig[:-1], sig_unscaled, atol=1e-6)

    sig2 = np.vstack((sig, sig)).T
    sig_scaled = dsp.fdscale(sig2, 1/t[1], scale)
    sig_unscaled = dsp.fdscale(sig_scaled, 1/t[1], unscale)
    assert np.allclose(sig2, sig_unscaled, atol=1e-6)


def test_fftfilt():
    # make a signal of sinusoids for testing:
    h = 0.001
    t = np.arange(0, 3.0 + h/2, h)
    y1 = 10 + 3.1*np.sin(2*np.pi*3*t)
    y2 = 5*np.sin(2*np.pi*10*t)
    y3 = 2*np.sin(2*np.pi*30*t)
    y4 = 3*np.sin(2*np.pi*60*t)
    y = y1 + y2 + y3 + y4

    sr = 1/h
    nyq = sr/2
    for j, (w, pz, yj) in enumerate(((7, None, y1),
                                     ([7, 18], None, y2),
                                     ([18, 45], None, y3),
                                     (45, False, y4))):
        yf = dsp.fftfilt(y, w, bw=None, pass_zero=pz, nyq=nyq)[0]
        assert 1-stats.pearsonr(yf, yj)[0] < 0.01

    yf = dsp.fftfilt(y, [7, 45], pass_zero=True, nyq=nyq)[0]
    assert 1-stats.pearsonr(yf, y1+y4)[0] < 0.01

    yf = dsp.fftfilt(y, [7, 18, 45], nyq=nyq)[0]
    assert 1-stats.pearsonr(yf, y1+y3)[0] < 0.01
    assert yf.ndim == 1

    assert_raises(ValueError, dsp.fftfilt, y, 2)
    assert_raises(ValueError, dsp.fftfilt, y, [.1, .3], bw=[.2, .2, .2])

    y2 = np.column_stack((y, y))
    yf2 = dsp.fftfilt(y2, np.array([7, 18, 45])/nyq)[0]
    # yf2 = dsp.fftfilt(y2, np.array([7, 18, 45]), nyq=nyq)[0]
    assert np.allclose(yf2[:, 0], yf2[:, 1])
    assert np.allclose(yf2[:, 0], yf)
    assert yf2.shape[0] == y1.shape[0]

    yf, freq, H = dsp.fftfilt(y, [7, 18, 45], nyq=nyq)
    assert freq.shape == H.shape
    t_end = t[-1] + h
    assert freq[0] == 0.0
    assert np.allclose(freq[1], 1/t_end)
    assert nyq - freq[-1] < freq[1]


def test_despike1():
    x = np.arange(100.)
    x[45] = 4.5
    x[0] = -20
    x[-1] = 110
    s = dsp.despike(x, 9, sigma=4)
    pv = np.zeros(x.shape, bool)
    pv[0] = True
    pv[45] = True
    pv[-1] = True
    assert (s.pv == pv).all()
    assert (s.x == x[~pv]).all()
    assert_raises(ValueError, dsp.despike, np.ones((5, 5)), 3)


def test_despike2():
    dt = .001
    t = np.arange(1000)*dt
    y = np.sin(2*np.pi*5*t)
    y[400:500] = 0.0
    y[450] = 0.3
    y1 = np.concatenate((y[:450], y[451:]))
    pv = np.zeros(1000, bool)
    pv[450] = True
    desp = dsp.despike(y, 15)
    assert np.all(desp.x == y1)
    assert np.all(desp.pv == pv)

    desp = dsp.despike(y, 15, threshold_value=0.29)
    assert np.all(desp.x == y1)
    assert np.all(desp.pv == pv)

    desp = dsp.despike(y, 15, threshold_value=0.31)
    assert np.all(desp.x == y)
    pv = np.zeros(1000, bool)
    assert np.all(desp.pv == pv)

    y[450] = 0.05
    desp = dsp.despike(y, 15, threshold_value=0.051)
    assert np.all(desp.x == y)
    assert np.all(desp.pv == pv)


def test_despike3():
    # tests the sweep_out_priors feature:
    a = np.ones(1500)
    a[75:100] = 5.0
    a[115:125] = 5.0
    a[575:600] = 5.0
    desp = dsp.despike(a, 35)
    assert desp.niter < 5
    pv = np.zeros(1500)
    pv[75:100] = True
    pv[115:125] = True
    pv[575:600] = True
    assert (desp.pv == pv).all()

    
def test_despike4():
    # tests the sweep_out_nexts feature:
    a = np.ones(1500)
    a[75:100] = 5.0
    a[115:125] = 5.0
    a[575:600] = 5.0
    desp = dsp.despike(a, 35, exclude_point='last')
    assert desp.niter < 5

    pv = np.zeros(1500)
    pv[75:100] = True
    pv[115:125] = True
    pv[575:600] = True
    assert (desp.pv == pv).all()
    assert desp.niter < 5


def test_despike_diff():
    x = np.arange(100.)
    x[45] = 4.5
    x[0] = -20
    x[-13] = 115
    s = dsp.despike_diff(x, 9, sigma=4)
    pv = np.zeros(x.shape, bool)
    pv[0] = True
    pv[45] = True
    pv[-13] = True
    assert (s.pv == pv).all()
    assert (s.x == x[~pv]).all()

    x[45] = 4.5
    x[0] = -20
    x[-13] = 115
    s = dsp.despike_diff(x, 9, sigma=4, maxiter=1)
    pv = np.zeros(x.shape, bool)
    pv[-13] = True
    assert (s.pv == pv).all()
    assert (s.x == x[~pv]).all()
    assert s.niter == 1

    assert_raises(ValueError, dsp.despike_diff, np.ones((5, 5)), 3)


def test_fftcoef():
    for n in (50, 51):
        t = np.arange(n)/n

        x = 4.0 * np.sin(2*np.pi*4*t)
        mag, phase, frq = dsp.fftcoef(x, 1/t[1])
        # lowest frequency = 1/T = 1.0
        # highest frequency = sr/2 = 500.0
        nfrq = (n // 2) + 1
        assert np.allclose(frq, 0.0 + np.arange(nfrq))
        x2 = np.zeros(len(frq))
        x2[4] = 4.0
        assert np.allclose(mag, x2)
        assert np.allclose(phase[4], 0.0)

        x = 4.0 * np.cos(2*np.pi*4*t)
        mag, phase, frq = dsp.fftcoef(x, 1/t[1])
        # lowest frequency = 1/T = 1.0
        # highest frequency = sr/2 = 500.0
        nfrq = (n // 2) + 1
        assert np.allclose(frq, 0.0 + np.arange(nfrq))
        x2 = np.zeros(len(frq))
        x2[4] = 4.0
        assert np.allclose(mag, x2)
        assert np.allclose(phase[4], -np.pi/2)


def test_fftcoef_recon():
    for n in (50, 51):
        t = np.arange(n)/n
        x = np.random.randn(n)
        mag, phase, frq = dsp.fftcoef(x, 1/t[1])
        A, B, frq = dsp.fftcoef(x, 1/t[1], coef='ab',
                                window=np.ones(x.shape))

        w = 2*np.pi*frq[1]
        # reconstruct with magnitude and phase:
        x2 = 0.0
        for k, (m, p, f) in enumerate(zip(mag, phase, frq)):
            x2 = x2 + m*np.sin(k*w*t - p)
        assert np.allclose(x, x2)

        # reconstruct with A and B:
        x3 = 0.0
        for k, (a, b, f) in enumerate(zip(A, B, frq)):
            x3 = x3 + a*np.cos(k*w*t) + b*np.sin(k*w*t)
        assert np.allclose(x, x3)


def test_fftcoef_fold():
    for n in (10, 11):
        t = np.arange(n)/n
        x = np.random.randn(n)
        mag, phase, frq = dsp.fftcoef(x, 1/t[1], fold=True)
        mag1, phase1, frq = dsp.fftcoef(x, 1/t[1], fold=False)
        if not (n & 1):
            mag1[1:-1] *= 2.0
        else:
            mag1[1:] *= 2.0
        assert np.allclose(mag, mag1)
        assert np.allclose(phase, phase1)


def test_fftcoef_detrend():
    for n in (50, 51):
        t = np.arange(n)/n
        x = np.random.randn(n)
        xd = signal.detrend(x)
        mag, phase, frq = dsp.fftcoef(x, 1/t[1], dodetrend=True)
        A, B, frq = dsp.fftcoef(x, 1/t[1], coef='ab',
                                window=np.ones(x.shape),
                                dodetrend=True)

        w = 2*np.pi*frq[1]
        # reconstruct with magnitude and phase:
        x2 = 0.0
        for k, (m, p, f) in enumerate(zip(mag, phase, frq)):
            x2 = x2 + m*np.sin(k*w*t - p)
        assert np.allclose(xd, x2)

        # reconstruct with A and B:
        x3 = 0.0
        for k, (a, b, f) in enumerate(zip(A, B, frq)):
            x3 = x3 + a*np.cos(k*w*t) + b*np.sin(k*w*t)
        assert np.allclose(xd, x3)

    assert_raises(ValueError, dsp.fftcoef, x, 1/t[1],
                            window=np.ones(x.size-1))


def test_fftcoef_maxdf():
    n = 16
    t = np.arange(n)/n
    x = np.random.randn(n)
    mag, phase, frq = dsp.fftcoef(x, 1/t[1])
    mag1, phase1, frq1 = dsp.fftcoef(x, 1/t[1], maxdf=0.6)
    assert frq1[1] <= 0.6
    assert np.allclose(mag1[::2], mag)
