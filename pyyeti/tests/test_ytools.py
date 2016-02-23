import numpy as np
import scipy.stats as stats
from pyyeti import ytools, locate
from nose.tools import *
import scipy.linalg as linalg
import scipy.signal as signal


def test_sturm():
    a = np.array([0., .16, 1.55, 2.78, 9., 14.])
    A = np.diag(a)
    assert np.all(1 == ytools.sturm(A, 0.))
    assert np.all(3 == ytools.sturm(A, 1.55))


def test_eig_si():
    k = np.random.randn(40, 40)
    m = np.random.randn(40, 40)
    k = np.dot(k.T, k) * 1000
    m = np.dot(m.T, m) * 10
    w1, phi1 = linalg.eigh(k, m, eigvals=(0, 14))
    w2, phi2, phiv2 = ytools.eig_si(k, m, p=15, mu=-1, tol=1e-12,
                                    verbose=False)
    fcut = np.sqrt(w2.max())/2/np.pi * 1.001
    w3, phi3, phiv3 = ytools.eig_si(k, m, f=fcut,
                                    mu=-1, tol=1e-12)
    assert np.allclose(w1, w2)
    assert np.allclose(np.abs(phi1), np.abs(phi2))
    assert np.allclose(w1, w3)
    assert np.allclose(np.abs(phi1), np.abs(phi3))
    w4, phi4, phiv4 = ytools.eig_si(k, m, f=fcut, Xk=phiv3, tol=1e-12,
                                    pmax=10)
    assert np.allclose(w1[:10], w4)
    assert np.allclose(np.abs(phi1[:, :10]), np.abs(phi4))

    w5, phi5, phiv5 = ytools.eig_si(k, m, p=15, Xk=phi4, tol=1e-12)
    assert np.allclose(w1, w5)
    assert np.allclose(np.abs(phi1), np.abs(phi5))

    mmod = m.copy()
    mmod[1, 0] = 2*mmod[1, 0]
    assert_raises(ValueError, ytools.eig_si, k, mmod, p=15, Xk=phi4, tol=1e-12)


def test_psd_rescale():
    g = np.random.randn(10000)
    sr = 400
    f, p = signal.welch(g, sr, nperseg=sr)
    p3, f3, msv3, ms3 = ytools.psd_rescale(p, f)
    p6, f6, msv6, ms6 = ytools.psd_rescale(p, f, n_oct=6)
    p6_2, f6_2, msv6_2, ms6_2 = ytools.psd_rescale(p, f, freq=f6)
    p12, f12, msv12, ms12 = ytools.psd_rescale(p6, f6, n_oct=12)
    msv1 = np.sum(p*(f[1]-f[0]))
    assert abs(msv1/msv3 - 1) < .12
    assert abs(msv1/msv6 - 1) < .06
    assert abs(msv1/msv12 - 1) < .03
    assert np.allclose(p6, p6_2)
    assert np.allclose(f6, f6_2)
    assert np.allclose(msv6, msv6_2)
    assert np.allclose(ms6, ms6_2)
    p_2, f_2, msv_2, ms_2 = ytools.psd_rescale(p, f, freq=f)
    assert np.allclose(p, p_2)
    assert np.allclose(f, f_2)
    assert np.allclose(msv1, msv_2)

    P = np.vstack((p, p)).T
    P3, F3, MSV3, MS3 = ytools.psd_rescale(P, f)
    assert np.allclose(p3, P3[:, 0])
    assert np.allclose(f3, F3)
    assert np.allclose(msv3, MSV3[0])
    assert np.allclose(ms3, MS3[:, 0])
    assert np.allclose(p3, P3[:, 1])
    assert np.allclose(msv3, MSV3[1])
    assert np.allclose(ms3, MS3[:, 1])

    in_freq = np.arange(0, 10.1, .25)
    out_freq = np.arange(0, 10.1, 5)
    in_p = np.ones_like(in_freq)
    p, f, ms, mvs = ytools.psd_rescale(in_p, in_freq, freq=out_freq)
    assert np.allclose(p, [ 1.,  1.,  1.])
    p, f, ms, mvs = ytools.psd_rescale(in_p, in_freq, freq=out_freq,
                                       extendends=False)
    assert np.allclose(p, [ 0.525,  1.   ,  0.525])


def test_splpsd():
    x = np.random.randn(100000)
    sr = 4000
    f, spl, oaspl = ytools.splpsd(x, sr, sr)
    # oaspl should be around 170.75 (since variance = 1):
    shouldbe = 10*np.log10(1/(2.9e-9)**2)
    abs(oaspl/shouldbe - 1) < .01

    f, spl, oaspl = ytools.splpsd(x, sr, sr, fs=0)
    # oaspl should be around 170.75 (since variance = 1):
    abs(oaspl/shouldbe - 1) < .01


def test_resample():
    p = 5
    q = 5
    n = 530
    d = np.random.randn(n) - 55
    x = np.arange(n) + 100

    rtol = atol = 1e-15
    d0, x0, fir0 = ytools.resample(d, p, q, t=x, getfir=1)
    assert np.allclose(d0, d, rtol, atol)
    assert np.allclose(x0, x, rtol, atol)
    n = 2*10 + 1
    mid = (n-1)//2
    assert np.allclose(fir0[mid], 1, rtol, atol)
    fir0[mid] = 0
    assert np.allclose(0, fir0, rtol, atol)

    pts = 5
    d0, x0, fir0 = ytools.resample(d, p, q, t=x, getfir=1, pts=pts)
    assert np.allclose(d0, d, rtol, atol)
    assert np.allclose(x0, x, rtol, atol)
    n = 2*pts + 1
    mid = (n-1)//2
    assert np.allclose(fir0[mid], 1, rtol, atol)
    fir0_ = fir0.copy()
    fir0[mid] = 0
    assert np.allclose(0, fir0, rtol, atol)
    d1, fir1 = ytools.resample(d, p, q, getfir=1, pts=pts)
    d2, x2 = ytools.resample(d, p, q, t=x, pts=pts)
    assert np.all(d1 == d0)
    assert np.all(d1 == d2)
    assert np.allclose(fir0_, fir1, rtol, atol)
    assert np.allclose(x2, x, rtol, atol)

    D = np.vstack((d, d)).T
    D2, X2 = ytools.resample(D, p, q, t=x, pts=pts)
    assert np.all(d1 == D[:, 0])
    assert np.all(d1 == D[:, 1])
    assert np.allclose(x2, X2)

    lcc = {}
    for pts in [10, 50]:
        dup4, xup4 = ytools.resample(d, 4, 1, t=x, pts=pts)
        dup2, xup2 = ytools.resample(dup4, 10, 20, t=xup4, pts=pts)
        d0, x0 = ytools.resample(dup2, 1, 2, t=xup2, pts=pts)
        assert np.allclose(x0, x, rtol, atol)
        lcc[pts] = stats.pearsonr(d0, d)[0]
        assert 1-lcc[pts] < .015
        assert np.abs(np.mean(d)-np.mean(d0)) < .005
    assert lcc[50] > lcc[10]


def test_windowends():
    s = ytools.windowends(np.ones(8), 4)
    assert np.allclose([0., 0.25, 0.75, 1., 1., 1., 1., 1.], s)
    s = ytools.windowends(np.ones(8), .7, ends='back')
    assert np.allclose([1., 1., 1., 1., 0.85355339059327373,
                        0.5, 0.14644660940672621, 0.], s)
    s = ytools.windowends(np.ones(8), .5, ends='both')
    assert np.allclose([0., 0.25, 0.75, 1., 1., 0.75, 0.25, 0.], s)
    s = ytools.windowends(np.ones(8), .5, ends='none')
    assert np.allclose([1., 1., 1., 1., 1., 1., 1., 1.], s)
    s = ytools.windowends(np.ones(8), 3)
    assert np.allclose([0., 0.5, 1., 1., 1., 1., 1., 1.], s)
    s = ytools.windowends(np.ones(8), 2)
    assert np.allclose([0., 0.5, 1., 1., 1., 1., 1., 1.], s)


def test_psd2time():
    spec = [[20,  .0768],
            [50,  .48],
            [100, .48]]
    sig, sr = ytools.psd2time(spec, ppc=10, fstart=35, fstop=70,
                              df=.01, winends=dict(portion=.01))
    assert np.allclose(700., sr)  # 70*10
    assert sig.size == 700*100
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= 37, f <= 68)
    fi = f[pv]
    psdi = p[pv]
    speci = ytools.psdinterp(spec, fi).flatten()
    assert abs(speci - psdi).max() < .05
    assert abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .25

    spec = [[.1,  .1],
            [5,  .1]]
    sig, sr = ytools.psd2time(spec, ppc=10, fstart=.1, fstop=5,
                              df=.2, winends=dict(portion=.01))
    # df gets internally reset to fstart
    assert np.allclose(5*10., sr)
    assert sig.size == 50*10
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= .5, f <= 3.)
    fi = f[pv]
    psdi = p[pv]
    speci = ytools.psdinterp(spec, fi).flatten()
    assert abs(speci - psdi).max() < .05
    assert abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .065

    # ppc gets reset to 2 case:
    spec = [[.1,  2.],
            [5,  2.]]
    sig, sr = ytools.psd2time(spec, ppc=1, fstart=.1, fstop=5,
                              df=.2, winends=dict(portion=.01))
    assert (np.sum(np.mean(sig**2)) - 2.*4.9)/(2.*4.9) < .1

    # odd length FFT case:
    spec = [[.1,  2.],
            [5,  2.]]
    sig, sr, t = ytools.psd2time(spec, ppc=3, fstart=.2, fstop=5,
                                 df=.2, winends=dict(portion=.01),
                                 gettime=1)
    assert np.allclose(5*3, sr)
    assert sig.size == 15*5
    assert (np.sum(np.mean(sig**2)) - 2.*4.8)/(2.*4.8) < .1
    assert np.allclose(t, np.arange(15*5)/sr)


def test_waterfall():
    from pyyeti import srs
    sig, t, f = ytools.gensweep(10, 1, 50, 4)
    sr = 1/t[1]
    frq = np.arange(1., 5.1)
    Q = 20
    sig2 = sig[:int(sr*1.5), None]
    def func(s):
        return srs.srs(s, sr, frq, Q), frq
    mp, t, f = ytools.waterfall(2, .5, sig, sr, func,
                                which=0, freq=1)
    mp, t, f = ytools.waterfall(2, .5, sig2, sr, func,
                                which=0, freq=1)
    assert_raises(ValueError, ytools.waterfall, 2, .5, sig, sr, func,
                  which=None, freq=1)
    assert_raises(ValueError, ytools.waterfall, 2, 1.5, sig, sr, func,
                  which=None, freq=frq)
    sig = np.hstack((sig2, sig2))
    assert_raises(ValueError, ytools.waterfall, 2, .5, sig, sr, func,
                  which=None, freq=frq)


def test_psdmod():
    TF = 30  # make a 30 second signal
    spec = [[20, 1], [50, 1]]
    sig, sr, t = ytools.psd2time(spec, ppc=10, fstart=20, fstop=50,
                                 df=1/TF, winends=dict(portion=10),
                                 gettime=True)
    # sr = 500
    freq = np.arange(20., 50.1)
    f, p = signal.welch(sig, sr, nperseg=sr)  # 1 second windows, df=1
    f2, p2 = ytools.psdmod(4, .5, sig, sr, nperseg=sr)
    pv = np.logical_and(f2 > 24, f2 < 47)
    assert np.all(p2[pv] > p[pv])

    # mimic standard welch:
    f3, p3 = ytools.psdmod(30, .5, sig, sr, nperseg=sr)
    assert np.allclose(p3, p)

    # mimic maximax:
    f4, p4 = ytools.psdmod(1, .5, sig, sr, nperseg=sr)
    assert np.all(p4[pv] > p2[pv])

    # test the map output:
    f5, p5, pmap, t = ytools.psdmod(1, .5, sig, sr, getmap=1)
    assert np.allclose(p5, np.max(pmap, axis=1))
    tshouldbe = np.arange(.5, 30.-.25, .5)
    assert np.allclose(t, tshouldbe)


def test_get_turning_pts():
    y, x = ytools.get_turning_pts([1, 2, 3, 3, 3], [1, 2, 3, 4, 5],
                                  getindex=False)
    assert np.all([1, 3, 3] == y)
    assert np.all([1, 3, 5] == x)
    y2 = ytools.get_turning_pts([1, 2, 3, 3, 3], getindex=False)
    assert np.all(y == y2)
    assert_raises(ValueError, ytools.get_turning_pts,
                  [1, 2, 3, 3, 3], [1, 2, 3, 3, 5])
    assert_raises(ValueError, ytools.get_turning_pts,
                  [1, 2, 3, 3, 3, 5], [1, 2, 3, 4, 5])


def test_gensweep():
    sig, t, f = ytools.gensweep(10, 1, 12, 8)
    assert np.allclose(np.max(sig), 1.)
    assert np.allclose(np.min(sig), -1.)
    # 12 = 1*2**n_oct
    n_oct = np.log(12.)/np.log(2.)
    t_elapsed = n_oct/8 * 60
    assert np.abs(t[-1] - t_elapsed) < .01
    assert np.allclose(f[0], 1)
    assert np.abs(f[-1]-12) < .01
    assert np.allclose(t[1]-t[0], 1/10/12)
    # at 8 oct/min or 8/60 = 4/30 = 2/15 oct/sec, time for 3 octaves
    # is: 3/(2/15) = 45/2 = 22.5 s
    i = np.argmin(np.abs(t - 22.5))
    assert np.allclose(f[i], 8.)


def test_calcenv():
    p = 5
    x = [1, 2, 3, 2, 5]
    y = [1, 4, 9, 4, 2]
    assert_raises(ValueError, ytools.calcenv, x, y, p)

    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16]
    assert_raises(ValueError, ytools.calcenv, x, y, p)

    assert_raises(ValueError, ytools.calcenv, [0, 1, 2],
                  [0, 5, 0], method='default')

    xe, ye, *_ = ytools.calcenv([0, 1, 2], [0, 5, 0], makeplot='no')
    x_sb = [0, .95, 1.05, 2*1.05]
    y_sb = [0, 5, 5, 0]
    yi = np.interp(xe, x_sb, y_sb)
    err = abs(yi-ye).max()
    assert err < .006

    xen, yen, *_ = ytools.calcenv([0, 1, 2], [0, -5, 0],
                                  makeplot='no', method='min')
    assert np.allclose(xe, xen)
    assert np.allclose(ye, -yen)

    xe, ye, *_ = ytools.calcenv([0, 1, 2], [0, 5, 0],
                                p=3, n=4000, makeplot='no')
    x_sb = [0, .97, 1.03, 2*1.03]
    y_sb = [0, 5, 5, 0]
    yi = np.interp(xe, x_sb, y_sb)
    assert abs(yi-ye).max() < err/2

    xe, ye, *_ = ytools.calcenv([0, 1, 2], [0, -5, 0],
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

    xen, yen, *_ = ytools.calcenv([0, 1, 2], [0, 5, 0],
                                  p=20, makeplot='no', method='min')
    assert np.allclose(xe, xen)
    assert np.allclose(ye, -yen)

    (xmax, ymax,
     xmin, ymin, h) = ytools.calcenv([0, 1, 2], [0, -5, 0], n=200,
                                     p=20, makeplot='no', base=None)
    yi = np.interp(xmax, x_sb, y_sb)
    assert abs(yi-ymax).max() < .06
    x_sb = [0, .8, 1.2, 2*1.2]
    y_sb = [0, -5, -5, 0]
    yi = np.interp(xmin, x_sb, y_sb)
    assert abs(yi-ymin).max() < .06

    (xmax, ymax,
     xmin, ymin, h) = ytools.calcenv([0, 1, 2], [0, -5, 0], n=200,
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
    xe, ye, *_ = ytools.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                 makeplot='no', n=100)
    xeb, yeb, *_ = ytools.calcenv([0, 1, 2], [0, 5, 0], makeplot='no',
                                  base=base, n=100)
    assert np.allclose(xe, xeb)
    assert np.allclose(ye+base, yeb)

    base = 2
    xe, ye, *_ = ytools.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                makeplot='no', n=100, method='min')
    xeb, yeb, *_ = ytools.calcenv([0, 1, 2], [0, 5, 0], makeplot='no',
                                  base=base, n=100, method='min')
    assert np.allclose(xe, xeb)
    assert np.allclose(ye+base, yeb)

    base = 2
    (xex, yex,
     xen, yen, h) = ytools.calcenv([0, 1, 2], [0-base, 5-base, 0-base],
                                   makeplot='no', n=100, method='both')
    (xexb, yexb,
     xenb, yenb, h) = ytools.calcenv([0, 1, 2], [0, 5, 0],
                                     makeplot='no',
                                     base=base, n=100, method='both')
    assert np.allclose(xex, xexb)
    assert np.allclose(yex+base, yexb)
    assert np.allclose(xen, xenb)
    assert np.allclose(yen+base, yenb)


def test_mattype():
    t, m = ytools.mattype([1, 2, 3])
    assert t == 0
    assert not ytools.mattype([1, 2, 3], 'symmetric')

    a = np.random.randn(4, 4)
    a = a.dot(a.T)
    t, m = ytools.mattype(a)
    assert t & m['posdef'] and t & m['symmetric']
    assert ytools.mattype(a, 'posdef')
    assert ytools.mattype(a, 'symmetric')

    a[1, 1] = 0
    t, m = ytools.mattype(a)
    assert (not (t & m['posdef'])) and t & m['symmetric']
    assert not ytools.mattype(a, 'posdef')
    assert ytools.mattype(a, 'symmetric')

    c = np.random.randn(4, 4) + 1j*np.random.randn(4, 4)
    c = c.dot(np.conj(c.T))
    t, m = ytools.mattype(c)
    assert t & m['posdef'] and t & m['hermitian']
    assert ytools.mattype(c, 'posdef')
    assert ytools.mattype(c, 'hermitian')

    c[1, 1] = 0
    t, m = ytools.mattype(c)
    assert (not (t & m['posdef'])) and t & m['hermitian']
    assert ytools.mattype(c, 'hermitian')
    assert not ytools.mattype(c, 'posdef')

    assert ytools.mattype(np.eye(5), 'diagonal')
    assert ytools.mattype(np.eye(5), 'identity')
    assert not ytools.mattype(np.eye(5)*2., 'identity')
    assert not ytools.mattype(np.random.randn(5, 5), 'diagonal')
    assert not ytools.mattype(np.random.randn(5, 5), 'identity')
    assert not ytools.mattype(np.random.randn(5, 5), 'posdef')
    assert not ytools.mattype(np.random.randn(5, 5)*(2+1j), 'posdef')

    assert_raises(ValueError, ytools.mattype, c, 'badtype')


def test_fixtime():
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    assert_raises(ValueError, ytools.fixtime,
                  ([1, 2, 3, 4], [1, 2, 3]))

    t = [0, 1, 6, 7]
    y = [1, 2, 3, 4]
    with assert_warns(RuntimeWarning):
        tn, yn = ytools.fixtime((t, y), sr=1)
        ty2 = ytools.fixtime(np.vstack([t, y]).T, sr=1)

    t2, y2 = ty2.T
    assert np.all(tn == t2)
    assert np.all(yn == y2)
    assert np.all(tn == np.arange(8))
    assert np.all(yn == [1, 2, 2, 2, 2, 2, 3, 4])

    with assert_warns(RuntimeWarning):
        ((t2, y2), pv,
         sr_stats, tp) = ytools.fixtime((t, y), getall=True,
                                        sr=1)
    assert np.all(tn == t2)
    assert np.all(yn == y2)
    assert np.all(pv == [False, False, False, False])
    assert np.allclose(sr_stats, [1, .2, 3/7, 1, 200/3])
    assert np.all(tp == np.arange(4))
    assert_raises(ValueError, ytools.fixtime, (1, 1, 1))

    drop = -1.40130E-45
    t = [0, .5, 1, 6, 7]
    y = [1, drop, 2, 3, 4]

    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime([t, y], sr=1)
    assert np.all(tn == t2)
    assert np.all(yn == y2)

    t = [1, 2, 3]
    y = [drop, drop, drop]
    assert_warns(RuntimeWarning, ytools.fixtime, (t, y), 'auto')

    ((t2, y2), drops,
     sr_stats, tp) = ytools.fixtime((t, y), 'auto', leavedrops=True,
                                    getall=True)
    assert np.all(tp == [0, 2])
    assert np.all(t2 == t)
    assert np.all(y2 == y)
    assert np.all(drops == [True, True, True])

    t = np.arange(100.)
    t = np.hstack((t, 200.))
    y = np.arange(101)
    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime((t, y), 'auto')
    assert np.all(t2 == np.arange(100.))
    assert np.all(y2 == np.arange(100))

    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime((t, y), 'auto', leaveoutliers=1)
    assert np.all(t2 == np.arange(200.1))
    sbe = np.ones(201, int)*99
    sbe[:100] = np.arange(100)
    sbe[-1] = 100
    assert np.all(y2 == sbe)

    t = [0, 1, 2, 3, 4, 5, 7, 6]
    y = [0, 1, 2, 3, 4, 5, 7, 6]
    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime((t, y), 1)
    assert np.all(t2 == np.arange(8))
    assert np.all(y2 == np.arange(8))

    assert_raises(ValueError, ytools.fixtime, (t, y), 1,
                  negmethod='stop')

    t = np.arange(8)[::-1]
    assert_raises(ValueError, ytools.fixtime, (t, y), 1)

    t = np.arange(0, 1, .001)
    noise = np.random.randn(t.size)
    noise = noise / noise.max() / 100000
    y = np.random.randn(t.size)
    t2, y2 = ytools.fixtime((t+noise, y), 'auto', base=0)
    assert np.allclose(t, t2)
    assert np.all(y == y2)

    dt = np.ones(100)
    dt[80:] = .91*dt[80:]
    t = dt.cumsum()
    y = np.arange(100)
    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime((t, y), 1)

    dt = np.ones(100)
    dt[80:] = 1.08*dt[80:]
    t = dt.cumsum()
    y = np.arange(100)
    with assert_warns(RuntimeWarning):
        t2, y2 = ytools.fixtime((t, y), 1)


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

    t2, y2 = ytools.fixtime((t+noise, y), 'auto', base=None)
    pv = locate.find_subsequence(y2, y[1:-1])
    assert pv.size > 0

    t3, y3 = ytools.fixtime((t+noise, y), 'auto', base=0.0)
    pv = locate.find_subsequence(y2, y[1:-1])
    assert pv.size > 0
    assert not np.allclose(t2[0], t3[0])
    assert np.allclose(t2-t2[0], t3-t3[0])
    assert abs(t3 - .003).min() < 1e-14


def test_fixtime_drift():
    L = 10000
    for factor in (.9995, 1.0005):
        t = np.arange(L)*(.001*factor)
        y = np.random.randn(L)
        t2, y2 = ytools.fixtime((t, y), 1000, base=0)
        t3, y3 = ytools.fixtime((t, y), 1000, fixdrift=1, base=0)

        i = np.argmax(y[3:100])
        i2 = np.nonzero(y2[:103] == y[i])[0][0]
        i3 = np.nonzero(y3[:103] == y[i])[0][0]
        d2 = abs(t2[i2] - t[i])
        d3 = abs(t3[i3] - t[i])

        L = len(y)
        i = L-100 + np.argmax(y[-100:-3])
        L = len(y2)
        i2 = L-103 + np.nonzero(y2[-103:] == y[i])[0][0]
        L = len(y3)
        i3 = L-103 + np.nonzero(y3[-103:] == y[i])[0][0]
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

    assert_raises(ValueError, ytools.fixtime,
                  ([1, 2, 3, 4], [1, 2, 3]))

    with assert_warns(RuntimeWarning):
        t, d = ytools.fixtime((*fd.T,), sr='auto')
    assert np.all(fd[:, 1] == d)
    assert np.allclose(np.diff(t), .002)
    assert np.allclose(t[0], fd[0, 0])
